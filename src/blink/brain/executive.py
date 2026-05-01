"""Explicit agenda and durable commitment-dispatch loop for Blink."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

from blink.brain._executive import (
    BrainCommitmentWakeRouter,
    BrainCommitmentWakeRouterPolicy,
    BrainCommitmentWakeRouterResult,
    BrainExecutivePolicyFrame,
    BrainPlanningCallback,
    BrainPlanningCoordinator,
    BrainPlanningCoordinatorResult,
    BrainPlanningOutcome,
    BrainPlanningRequest,
    BrainPlanningRequestKind,
    BrainPresenceDirector,
    BrainPresenceDirectorPolicy,
    BrainPresenceDirectorResult,
    apply_executive_policy_to_planning_result,
    build_executive_scope_ids,
    compile_executive_policy,
    default_commitment_scope_type,
    neutral_executive_policy_frame,
    should_auto_promote_goal,
    should_promote_goal_on_block,
)
from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import (
    CapabilityDispatchMode,
    CapabilityExecutionContext,
    CapabilityExecutionResult,
    CapabilityFamily,
    CapabilityRegistry,
    CapabilitySideEffectSink,
)
from blink.brain.context import BrainContextSelector, BrainContextTask
from blink.brain.context_surfaces import BrainContextMode
from blink.brain.embodied_executive import BrainHierarchicalEmbodiedCoordinator
from blink.brain.events import BrainEventType
from blink.brain.procedural_planning import (
    BrainPlanningSkillCandidate,
    match_planning_skills,
    planning_completed_prefix,
)
from blink.brain.projections import (
    BrainAgendaProjection,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentRecord,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainEmbodiedDispatchDisposition,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainRehearsalDecisionRecommendation,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _append_reason_code(target: list[str], value: str | None) -> None:
    """Append one normalized reason code exactly once."""
    text = _optional_text(value)
    if text is not None and text not in target:
        target.append(text)


def _policy_local_reason_code(executive_policy: BrainExecutivePolicyFrame | None) -> str | None:
    """Return the coarse local reason code implied by one policy posture."""
    if executive_policy is None:
        return None
    if executive_policy.action_posture == "suppress":
        return "policy_blocked_action"
    if executive_policy.approval_requirement == "user_confirmation":
        return "policy_requires_confirmation"
    if executive_policy.action_posture == "defer":
        return "policy_conservative_deferral"
    return None


def _policy_reason_codes(
    *,
    base_reason: str | None,
    executive_policy: BrainExecutivePolicyFrame | None,
    include_local_reason: bool,
) -> list[str]:
    """Return one stable merged reason-code list."""
    codes: list[str] = []
    _append_reason_code(codes, base_reason)
    if include_local_reason:
        _append_reason_code(codes, _policy_local_reason_code(executive_policy))
    for reason_code in (executive_policy.reason_codes if executive_policy is not None else ()):
        _append_reason_code(codes, reason_code)
    return codes


_NEUTRAL_SCENE_DEGRADED_REASON_CODES = {
    "camera_disconnected",
    "perception_disabled",
}
_MACHINE_CLEARABLE_BLOCKERS = {
    "robot_head_busy",
    "robot_head_unavailable",
    "robot_head_unarmed",
    "robot_head_status_unavailable",
}
_PLAN_DETAIL_KEEP = object()


def _default_goal_family(intent: str) -> str:
    """Return the default goal family for one intent."""
    if intent == "robot_head.sequence":
        return BrainGoalFamily.ENVIRONMENT.value
    return BrainGoalFamily.CONVERSATION.value


def _goal_status_for_activation(goal: BrainGoal) -> str:
    """Return the correct goal status when a commitment becomes active."""
    if goal.steps or goal.details.get("capabilities"):
        return BrainGoalStatus.OPEN.value
    return BrainGoalStatus.WAITING.value


def _goal_initiative_class(goal: BrainGoal) -> str | None:
    """Return initiative metadata attached to one goal, if present."""
    autonomy = goal.details.get("autonomy")
    if not isinstance(autonomy, dict):
        return None
    value = str(autonomy.get("initiative_class", "")).strip()
    return value or None


_ROBOT_HEAD_EXECUTION_SOURCES = frozenset(
    {
        "interpreter",
        "operator",
        "policy",
        "policy_idle",
        "policy_shutdown",
        "policy_startup",
        "tool",
    }
)
_POLICY_INITIATIVE_CLASSES = frozenset(
    {
        BrainInitiativeClass.SILENT_POSTURE_ADJUSTMENT.value,
        BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        BrainInitiativeClass.INSPECT_ONLY.value,
        BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
        BrainInitiativeClass.DEFER_UNTIL_USER_TURN.value,
        BrainInitiativeClass.MAINTENANCE_INTERNAL.value,
    }
)


def _goal_dispatch_source(*, goal: BrainGoal, capability_family: str) -> str:
    """Return the normalized capability execution source for one goal-backed step."""
    source = _optional_text(goal.source) or "executive"
    if capability_family != CapabilityFamily.ROBOT_HEAD.value:
        return source
    if source in _ROBOT_HEAD_EXECUTION_SOURCES:
        return source
    if _goal_initiative_class(goal) in _POLICY_INITIATIVE_CLASSES:
        return "policy"
    return "operator"


@dataclass(frozen=True)
class BrainExecutiveCycleResult:
    """Result from one executive cycle."""

    progressed: bool
    goal_id: str | None = None
    goal_status: str | None = None
    paused: bool = False


@dataclass(frozen=True)
class BrainExecutiveRecoveryDecision:
    """Critic or recovery decision for one failed capability execution."""

    decision: str
    summary: str
    blocked_reason: BrainBlockedReason | None = None
    wake_conditions: tuple[BrainWakeCondition, ...] = ()


class BrainExecutivePlanner:
    """Deterministic planner for bounded capability-backed goals."""

    def plan(self, *, goal: BrainGoal, registry: CapabilityRegistry) -> list[BrainGoalStep] | None:
        """Return planned steps for one supported goal intent."""
        requested_steps = list(goal.details.get("capabilities", []))
        if not requested_steps:
            requested_steps = _autonomy_template_capabilities(goal)
        if goal.intent != "robot_head.sequence" and not requested_steps:
            return None
        planned: list[BrainGoalStep] = []
        for item in requested_steps:
            if isinstance(item, str):
                capability_id = item.strip()
                arguments: dict[str, Any] = {}
            elif isinstance(item, dict):
                capability_id = str(item.get("capability_id", "")).strip()
                arguments = dict(item.get("arguments", {}))
            else:
                continue
            if not capability_id:
                continue
            registry.get(capability_id)
            planned.append(BrainGoalStep(capability_id=capability_id, arguments=arguments))
        return planned or None


def _autonomy_template_capabilities(goal: BrainGoal) -> list[dict[str, Any]] | None:
    """Return deterministic capability templates for supported autonomy intents."""
    scene_candidate = (
        dict(goal.details.get("scene_candidate", {}))
        if isinstance(goal.details.get("scene_candidate"), dict)
        else {}
    )
    commitment_wake = (
        dict(goal.details.get("commitment_wake", {}))
        if isinstance(goal.details.get("commitment_wake"), dict)
        else {}
    )
    maintenance = (
        dict(goal.details.get("maintenance", {}))
        if isinstance(goal.details.get("maintenance"), dict)
        else {}
    )
    presence_scope_key = str(scene_candidate.get("presence_scope_key", "")).strip()
    templates: dict[str, list[dict[str, Any]]] = {
        "autonomy.presence_user_reentered": [
            {
                "capability_id": "observation.inspect_presence_state",
                "arguments": {
                    "presence_scope_key": presence_scope_key,
                    "scene_candidate": scene_candidate,
                },
            },
            {
                "capability_id": "reporting.record_presence_event",
                "arguments": {"scene_candidate": scene_candidate},
            },
        ],
        "autonomy.presence_attention_returned": [
            {
                "capability_id": "observation.inspect_presence_state",
                "arguments": {
                    "presence_scope_key": presence_scope_key,
                    "scene_candidate": scene_candidate,
                },
            },
            {
                "capability_id": "reporting.record_presence_event",
                "arguments": {"scene_candidate": scene_candidate},
            },
        ],
        "autonomy.camera_degraded": [
            {
                "capability_id": "observation.inspect_camera_health",
                "arguments": {
                    "presence_scope_key": presence_scope_key,
                    "scene_candidate": scene_candidate,
                },
            },
            {
                "capability_id": "reporting.record_presence_event",
                "arguments": {"scene_candidate": scene_candidate},
            },
        ],
        "autonomy.presence_brief_reengagement_speech": [
            {
                "capability_id": "observation.inspect_presence_state",
                "arguments": {
                    "presence_scope_key": presence_scope_key,
                    "scene_candidate": scene_candidate,
                },
            },
            {
                "capability_id": "dialogue.emit_brief_reengagement",
                "arguments": {
                    "presence_scope_key": presence_scope_key,
                    "scene_candidate": scene_candidate,
                },
            },
            {
                "capability_id": "reporting.record_presence_event",
                "arguments": {"scene_candidate": scene_candidate},
            },
        ],
        "autonomy.commitment_wake_thread_idle": [
            {
                "capability_id": "reporting.record_commitment_wake",
                "arguments": {"commitment_wake": commitment_wake},
            }
        ],
        "autonomy.commitment_wake_user_response": [
            {
                "capability_id": "reporting.record_commitment_wake",
                "arguments": {"commitment_wake": commitment_wake},
            }
        ],
        "autonomy.maintenance_review_findings": [
            {
                "capability_id": "maintenance.review_memory_health",
                "arguments": {"maintenance": maintenance},
            },
            {
                "capability_id": "reporting.record_maintenance_note",
                "arguments": {"maintenance": maintenance},
            },
        ],
        "autonomy.maintenance_thread_active_backpressure": [
            {
                "capability_id": "maintenance.review_scheduler_backpressure",
                "arguments": {"maintenance": maintenance},
            },
            {
                "capability_id": "reporting.record_maintenance_note",
                "arguments": {"maintenance": maintenance},
            },
        ],
    }
    selected = templates.get(goal.intent)
    if selected is None:
        return None
    return [item for item in selected if item.get("capability_id")]


class BrainExecutiveCritic:
    """Small explicit critic for retry, blocked, or failed execution."""

    def __init__(self, *, max_retries: int = 1):
        """Initialize the critic."""
        self._max_retries = max_retries

    def evaluate(
        self,
        *,
        goal: BrainGoal,
        step: BrainGoalStep,
        result: CapabilityExecutionResult,
    ) -> BrainExecutiveRecoveryDecision:
        """Choose retry, block, or fail for one rejected capability result."""
        if result.retryable and step.attempts <= self._max_retries:
            return BrainExecutiveRecoveryDecision(
                decision=BrainGoalStatus.RETRY.value,
                summary=result.summary,
            )
        blocked_reason = BrainBlockedReason(
            kind=(
                BrainBlockedReasonKind.CAPABILITY_BLOCKED.value
                if result.outcome == "blocked"
                else BrainBlockedReasonKind.CAPABILITY_FAILED.value
            ),
            summary=result.summary,
            details={
                "capability_id": step.capability_id,
                "retryable": result.retryable,
                "error_code": result.error_code,
                "outcome": result.outcome,
            },
        )
        if (
            result.outcome == "blocked"
            and result.retryable
            and (result.error_code or "") in _MACHINE_CLEARABLE_BLOCKERS
        ):
            wake_conditions = (
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.CONDITION_CLEARED.value,
                    summary="Resume when the capability blocker clears.",
                    details={
                        "capability_id": step.capability_id,
                        "error_code": result.error_code,
                    },
                ),
            )
        else:
            wake_conditions = (
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume only after the blocker is cleared.",
                    details={"capability_id": step.capability_id},
                ),
            )
        if result.outcome == "blocked":
            return BrainExecutiveRecoveryDecision(
                decision=BrainGoalStatus.BLOCKED.value,
                summary=result.summary,
                blocked_reason=blocked_reason,
                wake_conditions=wake_conditions,
            )
        return BrainExecutiveRecoveryDecision(
            decision=BrainGoalStatus.FAILED.value,
            summary=result.summary,
            blocked_reason=blocked_reason,
            wake_conditions=wake_conditions,
        )


class BrainExecutive:
    """Own an explicit agenda loop plus durable commitment state."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        capability_registry: CapabilityRegistry,
        planner: BrainExecutivePlanner | None = None,
        critic: BrainExecutiveCritic | None = None,
        presence_scope_key: str = "local:presence",
        context_surface_builder: Any = None,
        context_selector: BrainContextSelector | None = None,
        capability_side_effect_sink: CapabilitySideEffectSink | None = None,
        counterfactual_rehearsal_engine: Any | None = None,
        embodied_coordinator: Any | None = None,
        autonomy_state_changed_callback: Callable[[], None] | None = None,
        planning_callback: BrainPlanningCallback | None = None,
    ):
        """Initialize the executive."""
        self._store = store
        self._session_resolver = session_resolver
        self._capability_registry = capability_registry
        self._planner = planner or BrainExecutivePlanner()
        self._critic = critic or BrainExecutiveCritic()
        self._presence_scope_key = presence_scope_key
        self._context_surface_builder = context_surface_builder
        self._context_selector = context_selector or BrainContextSelector()
        self._capability_side_effect_sink = capability_side_effect_sink
        self._counterfactual_rehearsal_engine = counterfactual_rehearsal_engine
        self._embodied_coordinator = embodied_coordinator
        if (
            self._embodied_coordinator is None
            and self._context_surface_builder is not None
            and self._counterfactual_rehearsal_engine is not None
        ):
            action_engine = getattr(self._counterfactual_rehearsal_engine, "action_engine", None)
            if action_engine is not None:
                self._embodied_coordinator = BrainHierarchicalEmbodiedCoordinator(
                    store=self._store,
                    session_resolver=self._session_resolver,
                    presence_scope_key=self._presence_scope_key,
                    context_surface_builder=self._context_surface_builder,
                    action_engine=action_engine,
                )
        self._autonomy_state_changed_callback = autonomy_state_changed_callback
        self._presence_director = BrainPresenceDirector(
            store=self._store,
            session_resolver=self._session_resolver,
            goal_creator=self.create_goal,
            policy=BrainPresenceDirectorPolicy(),
        )
        self._wake_router = BrainCommitmentWakeRouter(
            store=self._store,
            session_resolver=self._session_resolver,
            capability_registry=self._capability_registry,
            presence_scope_key=self._presence_scope_key,
            policy=BrainCommitmentWakeRouterPolicy(),
        )
        self._planning_coordinator = BrainPlanningCoordinator(
            registry=self._capability_registry,
            planning_callback=planning_callback,
        )

    def set_capability_side_effect_sink(self, sink: CapabilitySideEffectSink | None):
        """Set the narrow side-effect sink used by bounded internal capabilities."""
        self._capability_side_effect_sink = sink

    def create_goal(
        self,
        *,
        title: str,
        intent: str,
        source: str,
        details: dict[str, Any] | None = None,
        goal_id: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        goal_family: str | None = None,
        commitment_id: str | None = None,
    ) -> str:
        """Append a structured goal-created event and return the goal id."""
        session_ids = self._session_resolver()
        goal = BrainGoal(
            goal_id=goal_id or uuid4().hex,
            title=" ".join((title or "").split()).strip(),
            intent=intent,
            source=source,
            goal_family=goal_family or _default_goal_family(intent),
            commitment_id=commitment_id,
            details=dict(details or {}),
        )
        self._append_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_CREATED,
            source=source,
            payload={"goal": goal.as_dict()},
            correlation_id=correlation_id or goal.goal_id,
            causal_parent_id=causal_parent_id,
        )
        if goal.commitment_id is None:
            decision = should_auto_promote_goal(goal)
            if decision.durable and decision.scope_type is not None:
                self.promote_goal_to_commitment(
                    goal_id=goal.goal_id,
                    scope_type=decision.scope_type,
                    correlation_id=correlation_id or goal.goal_id,
                    causal_parent_id=causal_parent_id,
                )
        return goal.goal_id

    def create_commitment_goal(
        self,
        *,
        title: str,
        intent: str,
        source: str,
        details: dict[str, Any] | None = None,
        goal_family: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        goal_status: str | None = None,
        scope_type: str | None = None,
    ) -> str:
        """Create or refresh one durable commitment-backed goal."""
        session_ids = self._session_resolver()
        resolved_goal_family = goal_family or _default_goal_family(intent)
        resolved_scope_type = scope_type or default_commitment_scope_type(resolved_goal_family)
        scope_ids = build_executive_scope_ids(session_ids)
        resolved_scope_id = self._commitment_scope_id(
            session_ids=session_ids,
            scope_type=resolved_scope_type,
            scope_ids=scope_ids,
        )
        existing_commitment = self._store.find_executive_commitment(
            scope_type=resolved_scope_type,
            scope_id=resolved_scope_id,
            goal_family=resolved_goal_family,
            intent=intent,
            title=title,
        )
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        existing_goal = (
            agenda.goal(existing_commitment.current_goal_id)
            if existing_commitment is not None and existing_commitment.current_goal_id
            else None
        )
        if existing_goal is not None and existing_goal.status not in {
            BrainGoalStatus.COMPLETED.value,
            BrainGoalStatus.CANCELLED.value,
        }:
            updated_goal = self._copy_goal(existing_goal)
            updated_goal.title = " ".join((title or "").split()).strip()
            updated_goal.details = {
                **updated_goal.details,
                **dict(details or {}),
                "commitment_status": existing_commitment.status,
            }
            updated_goal.goal_family = resolved_goal_family
            updated_goal.updated_at = _utc_now()
            self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_UPDATED,
                goal=updated_goal,
                source=source,
                commitment=existing_commitment,
                correlation_id=correlation_id or existing_commitment.commitment_id,
                causal_parent_id=causal_parent_id,
            )
            return updated_goal.goal_id

        goal = BrainGoal(
            goal_id=uuid4().hex,
            title=" ".join((title or "").split()).strip(),
            intent=intent,
            source=source,
            goal_family=resolved_goal_family,
            status=goal_status or BrainGoalStatus.WAITING.value,
            details={
                **dict(details or {}),
                "commitment_status": BrainCommitmentStatus.ACTIVE.value,
            },
        )
        commitment = self._store.upsert_executive_commitment(
            scope_type=resolved_scope_type,
            scope_id=resolved_scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=goal.title,
            goal_family=resolved_goal_family,
            intent=intent,
            status=BrainCommitmentStatus.ACTIVE.value,
            details={"summary": goal.title, **goal.details},
            current_goal_id=goal.goal_id,
            source_event_id=causal_parent_id,
        )
        goal.commitment_id = commitment.commitment_id
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_CREATED,
            goal=goal,
            source=source,
            commitment=commitment,
            correlation_id=correlation_id or commitment.commitment_id,
            causal_parent_id=causal_parent_id,
        )
        return goal.goal_id

    def promote_goal_to_commitment(
        self,
        *,
        goal_id: str,
        scope_type: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
    ) -> BrainCommitmentRecord:
        """Promote one already-created goal into a durable commitment explicitly."""
        session_ids = self._session_resolver()
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = agenda.goal(goal_id)
        if goal is None:
            raise KeyError(f"Missing goal '{goal_id}'.")
        if goal.commitment_id:
            existing_commitment = self._store.get_executive_commitment(commitment_id=goal.commitment_id)
            if existing_commitment is not None:
                return existing_commitment

        resolved_scope_type = scope_type or default_commitment_scope_type(goal.goal_family)
        scope_ids = build_executive_scope_ids(session_ids)
        resolved_scope_id = self._commitment_scope_id(
            session_ids=session_ids,
            scope_type=resolved_scope_type,
            scope_ids=scope_ids,
        )
        commitment_status = str(goal.details.get("commitment_status", "")).strip() or (
            BrainCommitmentStatus.BLOCKED.value
            if goal.status in {BrainGoalStatus.BLOCKED.value, BrainGoalStatus.FAILED.value}
            else (
                BrainCommitmentStatus.DEFERRED.value
                if goal.status == BrainGoalStatus.WAITING.value
                else BrainCommitmentStatus.ACTIVE.value
            )
        )
        commitment = self._store.upsert_executive_commitment(
            scope_type=resolved_scope_type,
            scope_id=resolved_scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=goal.title,
            goal_family=goal.goal_family,
            intent=goal.intent,
            status=commitment_status,
            details={
                **goal.details,
                "summary": goal.title,
            },
            current_goal_id=goal.goal_id,
            blocked_reason=goal.blocked_reason,
            wake_conditions=goal.wake_conditions,
            plan_revision=goal.plan_revision,
            resume_count=goal.resume_count,
            source_event_id=causal_parent_id,
        )
        promoted_goal = self._copy_goal(goal)
        promoted_goal.commitment_id = commitment.commitment_id
        promoted_goal.details["commitment_status"] = commitment.status
        promoted_goal.plan_revision = commitment.plan_revision
        promoted_goal.resume_count = commitment.resume_count
        promoted_goal.updated_at = _utc_now()
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_UPDATED,
            goal=promoted_goal,
            source="executive",
            commitment=commitment,
            correlation_id=correlation_id or commitment.commitment_id,
            causal_parent_id=causal_parent_id,
        )
        return commitment

    async def run_startup_pass(self, *, max_iterations: int = 4) -> BrainExecutiveCycleResult:
        """Continue active runnable commitments on runtime startup."""
        commitment_result = await self._run_commitment_pass(max_iterations=max_iterations)
        wake_router_result = await self.run_commitment_wake_router(boundary_kind="startup_recovery")
        expiry_cleanup_result = self.run_presence_director_expiry_cleanup(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.STARTUP_RECOVERY.value,
                summary="Expire stale candidates during startup recovery.",
                source_event_type="runtime.startup",
            )
        )
        reevaluation_result = self.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.STARTUP_RECOVERY.value,
                summary="Reevaluate held candidates on startup recovery.",
                source_event_type="runtime.startup",
            )
        )
        return self._merge_cycle_and_director_results(
            commitment_result=commitment_result,
            router_results=(wake_router_result,),
            director_results=(expiry_cleanup_result, reevaluation_result),
        )

    async def run_turn_end_pass(self, *, max_iterations: int = 4) -> BrainExecutiveCycleResult:
        """Continue active runnable commitments at turn end."""
        commitment_result = await self._run_commitment_pass(max_iterations=max_iterations)
        latest_assistant_turn_end = self._store.latest_brain_event(
            user_id=self._session_resolver().user_id,
            thread_id=self._session_resolver().thread_id,
            event_types=(BrainEventType.ASSISTANT_TURN_ENDED,),
        )
        wake_router_result = await self.run_commitment_wake_router(
            boundary_kind="assistant_turn_end",
            source_event=latest_assistant_turn_end,
        )
        reevaluation_result = self.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value,
                summary="Reevaluate held candidates after the assistant turn closes.",
                source_event_type=BrainEventType.ASSISTANT_TURN_ENDED,
                source_event_id=(
                    latest_assistant_turn_end.event_id if latest_assistant_turn_end is not None else None
                ),
                ts=latest_assistant_turn_end.ts if latest_assistant_turn_end is not None else _utc_now(),
                details={"turn": "assistant"},
            )
        )
        return self._merge_cycle_and_director_results(
            commitment_result=commitment_result,
            router_results=(wake_router_result,),
            director_results=(reevaluation_result,),
        )

    def run_presence_director_pass(self) -> BrainPresenceDirectorResult:
        """Run one explicit bounded PresenceDirector decision pass."""
        result = self._presence_director.run_once(
            executive_policy=self._compile_presence_director_policy()
        )
        self._notify_autonomy_state_changed()
        return result

    def run_presence_director_reevaluation(
        self,
        trigger: BrainReevaluationTrigger,
    ) -> BrainPresenceDirectorResult:
        """Run one explicit bounded reevaluation pass over held candidates."""
        result = self._presence_director.reevaluate_once(
            trigger,
            executive_policy=self._compile_presence_director_policy(trigger=trigger),
        )
        self._notify_autonomy_state_changed()
        return result

    def run_presence_director_expiry_cleanup(
        self,
        trigger: BrainReevaluationTrigger,
    ) -> BrainPresenceDirectorResult:
        """Run one explicit bounded expiry-cleanup pass over current candidates."""
        result = self._presence_director.expire_once(trigger)
        self._notify_autonomy_state_changed()
        return result

    def _compile_presence_director_policy(
        self,
        *,
        trigger: BrainReevaluationTrigger | None = None,
    ) -> BrainExecutivePolicyFrame:
        """Compile one executive-policy frame for Presence Director work."""
        reference_ts = trigger.ts if trigger is not None else _utc_now()
        leading_candidate = self._leading_presence_candidate()
        if (
            leading_candidate is not None
            and (
                leading_candidate.source == BrainCandidateGoalSource.COMMITMENT.value
                or leading_candidate.initiative_class
                in {
                    BrainInitiativeClass.MAINTENANCE_INTERNAL.value,
                    BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
                }
            )
        ):
            return neutral_executive_policy_frame(
                task=BrainContextTask.REEVALUATION,
                updated_at=reference_ts,
            )
        return self._compile_executive_policy_for_task(
            task=BrainContextTask.REEVALUATION,
            query_text=(
                trigger.summary
                if trigger is not None
                else _optional_text(getattr(leading_candidate, "summary", None)) or ""
            ),
            reference_ts=reference_ts,
        )

    def _compile_executive_policy_for_task(
        self,
        *,
        task: BrainContextTask,
        query_text: str,
        reference_ts: str | None = None,
    ) -> BrainExecutivePolicyFrame:
        """Compile one executive-policy frame from the canonical context surface."""
        resolved_reference_ts = reference_ts or _utc_now()
        if self._context_surface_builder is None:
            return neutral_executive_policy_frame(
                task=task,
                updated_at=resolved_reference_ts,
            )
        surface = self._context_surface_builder.build(
            latest_user_text=query_text,
            task=task,
        )
        policy = compile_executive_policy(
            surface,
            task=task,
            reference_ts=resolved_reference_ts,
        )
        if task in {BrainContextTask.PLANNING, BrainContextTask.WAKE}:
            scene_reason_codes = {
                str(code).strip()
                for code in surface.scene_world_state.degraded_reason_codes
                if str(code).strip()
            }
            if (
                surface.scene_world_state.degraded_mode == "unavailable"
                and scene_reason_codes
                and scene_reason_codes <= _NEUTRAL_SCENE_DEGRADED_REASON_CODES
            ):
                adjusted_reason_codes = [
                    code
                    for code in policy.reason_codes
                    if code not in {"scene_unavailable", "procedural_reuse_advisory_only", "procedural_reuse_blocked"}
                ]
                adjusted_suppression_codes = [
                    code for code in policy.suppression_codes if code != "scene_unavailable"
                ]
                if policy.pending_operator_review_count > 0:
                    adjusted_action_posture = "suppress"
                    adjusted_conservatism = "high"
                    adjusted_procedural_reuse = "blocked"
                elif any(
                    (
                        policy.review_debt_count > 0,
                        policy.held_claim_count > 0,
                        policy.pending_user_review_count > 0,
                        policy.stale_claim_count > 0,
                        policy.unresolved_active_state_count > 0,
                        policy.blocked_commitment_count > 0,
                        policy.deferred_commitment_count > 0,
                    )
                ):
                    adjusted_action_posture = "defer"
                    adjusted_conservatism = "elevated"
                    adjusted_procedural_reuse = "advisory_only"
                else:
                    adjusted_action_posture = "allow"
                    adjusted_conservatism = "normal"
                    adjusted_procedural_reuse = "allowed"
                if adjusted_procedural_reuse == "blocked":
                    adjusted_reason_codes.append("procedural_reuse_blocked")
                elif adjusted_procedural_reuse == "advisory_only":
                    adjusted_reason_codes.append("procedural_reuse_advisory_only")
                policy = replace(
                    policy,
                    action_posture=adjusted_action_posture,
                    conservatism=adjusted_conservatism,
                    procedural_reuse_eligibility=adjusted_procedural_reuse,
                    scene_degraded_mode="healthy",
                    reason_codes=sorted(set(adjusted_reason_codes)),
                    suppression_codes=sorted(set(adjusted_suppression_codes)),
                )
            normalized_reason_codes = {
                code
                for code in policy.reason_codes
                if code not in {"procedural_reuse_advisory_only", "procedural_reuse_blocked"}
            }
            unresolved_records = [
                record
                for record in surface.active_situation_model.records
                if record.state == "unresolved"
            ]
            unresolved_scene_only = bool(unresolved_records) and all(
                record.record_kind in {"scene_state", "world_state", "uncertainty_state"}
                and record.goal_id is None
                and record.commitment_id is None
                and record.plan_proposal_id is None
                for record in unresolved_records
            )
            if (
                normalized_reason_codes in (
                    {"active_state_unresolved"},
                    {"scene_unavailable"},
                    {"scene_unavailable", "active_state_unresolved"},
                )
                and unresolved_scene_only
            ):
                return neutral_executive_policy_frame(
                    task=task,
                    updated_at=policy.updated_at,
                )
        return policy

    def _leading_presence_candidate(self):
        """Return the canonical leading candidate for the current director pass."""
        session_ids = self._session_resolver()
        ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        sorted_candidates = sorted(
            ledger.current_candidates,
            key=self._presence_director._policy.candidate_sort_key,
        )
        return sorted_candidates[0] if sorted_candidates else None

    def _wake_policy_query_text(self, *, decision) -> str:
        """Return one deterministic policy-compilation query for wake routing."""
        commitment_summary = _optional_text(decision.commitment.details.get("summary"))
        return (
            _optional_text(decision.commitment.title)
            or commitment_summary
            or _optional_text(decision.trigger.summary)
            or _optional_text(decision.wake_condition.summary)
            or ""
        )

    def _wake_policy_changes_route(
        self,
        *,
        decision,
        executive_policy: BrainExecutivePolicyFrame,
    ) -> bool:
        """Return whether policy should actively downgrade this wake route."""
        if executive_policy.action_posture == "allow":
            return False
        normalized_reason_codes = {
            code
            for code in executive_policy.reason_codes
            if code not in {"procedural_reuse_advisory_only", "procedural_reuse_blocked"}
        }
        if (
            normalized_reason_codes
            <= {
                "blocked_commitment_present",
                "deferred_commitment_present",
                "active_state_unresolved",
                "pending_user_review",
                "pending_operator_review",
            }
            and set(executive_policy.source_commitment_ids or [])
            == {decision.commitment.commitment_id}
        ):
            return False
        return True

    async def run_commitment_wake_router(
        self,
        *,
        boundary_kind: str,
        source_event=None,
    ) -> BrainCommitmentWakeRouterResult:
        """Run one bounded durable-commitment wake-router pass."""
        decision = await self._wake_router.route_once(
            boundary_kind=boundary_kind,
            source_event=source_event,
        )
        if decision is None:
            return BrainCommitmentWakeRouterResult(progressed=False)
        wake_policy = self._compile_executive_policy_for_task(
            task=BrainContextTask.WAKE,
            query_text=self._wake_policy_query_text(decision=decision),
            reference_ts=decision.trigger.ts or _utc_now(),
        )
        route_was_changed_by_policy = (
            decision.routing.route_kind != BrainCommitmentWakeRouteKind.KEEP_WAITING.value
            and self._wake_policy_changes_route(
                decision=decision,
                executive_policy=wake_policy,
            )
        )
        effective_reason = _optional_text(decision.routing.details.get("reason"))
        effective_routing_details = dict(decision.routing.details)
        effective_route_kind = decision.routing.route_kind
        if route_was_changed_by_policy:
            effective_route_kind = BrainCommitmentWakeRouteKind.KEEP_WAITING.value
            effective_reason = _policy_local_reason_code(wake_policy) or effective_reason
            effective_routing_details.update(
                {
                    "reason": effective_reason,
                    "policy_changed_route": True,
                    "original_route_kind": decision.routing.route_kind,
                    "policy_action_posture": wake_policy.action_posture,
                    "policy_approval_requirement": wake_policy.approval_requirement,
                }
            )
        effective_routing = replace(
            decision.routing,
            route_kind=effective_route_kind,
            details=effective_routing_details,
            reason_codes=_policy_reason_codes(
                base_reason=effective_reason,
                executive_policy=wake_policy,
                include_local_reason=route_was_changed_by_policy,
            ),
            executive_policy=wake_policy.as_dict(),
        )
        session_ids = self._session_resolver()
        wake_event = self._store.append_commitment_wake_triggered(
            commitment=decision.commitment,
            wake_condition=decision.wake_condition,
            trigger=decision.trigger,
            routing=effective_routing,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="wake_router",
            correlation_id=decision.commitment.commitment_id,
            causal_parent_id=decision.trigger.source_event_id,
        )
        if effective_routing.route_kind == BrainCommitmentWakeRouteKind.RESUME_DIRECT.value:
            resumed = self.resume_commitment(
                commitment_id=decision.commitment.commitment_id,
                correlation_id=decision.commitment.commitment_id,
                causal_parent_id=wake_event.event_id,
                source="wake_router",
            )
            return BrainCommitmentWakeRouterResult(
                progressed=True,
                matched_commitment_id=decision.commitment.commitment_id,
                trigger_event_id=wake_event.event_id,
                route_kind=effective_routing.route_kind,
                resumed_commitment_id=resumed.commitment_id,
                reason=effective_reason,
                reason_codes=tuple(effective_routing.reason_codes),
                executive_policy=effective_routing.executive_policy,
            )
        if (
            effective_routing.route_kind == BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value
            and decision.candidate_goal is not None
        ):
            self.propose_candidate_goal(
                candidate_goal=decision.candidate_goal,
                source="wake_router",
                correlation_id=decision.commitment.commitment_id,
                causal_parent_id=wake_event.event_id,
            )
            return BrainCommitmentWakeRouterResult(
                progressed=True,
                matched_commitment_id=decision.commitment.commitment_id,
                trigger_event_id=wake_event.event_id,
                route_kind=effective_routing.route_kind,
                proposed_candidate_goal_id=decision.candidate_goal.candidate_goal_id,
                reason=effective_reason,
                reason_codes=tuple(effective_routing.reason_codes),
                executive_policy=effective_routing.executive_policy,
            )
        return BrainCommitmentWakeRouterResult(
            progressed=True,
            matched_commitment_id=decision.commitment.commitment_id,
            trigger_event_id=wake_event.event_id,
            route_kind=effective_routing.route_kind,
            reason=effective_reason,
            reason_codes=tuple(effective_routing.reason_codes),
            executive_policy=effective_routing.executive_policy,
        )

    def propose_candidate_goal(
        self,
        *,
        candidate_goal: BrainCandidateGoal,
        source: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> BrainPresenceDirectorResult:
        """Append one candidate goal and immediately run a bounded director pass."""
        session_ids = self._session_resolver()
        self._store.append_candidate_goal_created(
            candidate_goal=candidate_goal,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=source or candidate_goal.source,
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
            confidence=float(candidate_goal.confidence if confidence is None else confidence),
            tags=tags,
        )
        return self.run_presence_director_pass()

    async def run_until_quiescent(self, *, max_iterations: int = 8) -> BrainExecutiveCycleResult:
        """Drive the executive until it pauses or no more progress is possible."""
        any_progress = False
        last_result = BrainExecutiveCycleResult(progressed=False)
        for _ in range(max_iterations):
            last_result = await self.run_once()
            any_progress = any_progress or last_result.progressed
            if not last_result.progressed or last_result.paused:
                return BrainExecutiveCycleResult(
                    progressed=any_progress,
                    goal_id=last_result.goal_id,
                    goal_status=last_result.goal_status,
                    paused=last_result.paused,
                )
        return BrainExecutiveCycleResult(
            progressed=any_progress,
            goal_id=last_result.goal_id,
            goal_status=last_result.goal_status,
            paused=last_result.paused,
        )

    def _recent_plan_proposal_summary(self, *, proposal_ids: list[str]) -> str | None:
        """Return the latest known summary for one linked proposal id."""
        wanted_ids = {proposal_id for proposal_id in proposal_ids if proposal_id}
        if not wanted_ids:
            return None
        session_ids = self._session_resolver()
        for event in self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=64,
        ):
            if event.event_type not in {
                BrainEventType.PLANNING_PROPOSED,
                BrainEventType.PLANNING_ADOPTED,
                BrainEventType.PLANNING_REJECTED,
            }:
                continue
            proposal = dict((event.payload or {}).get("proposal") or {})
            proposal_id = _optional_text(proposal.get("plan_proposal_id"))
            if proposal_id not in wanted_ids:
                continue
            summary = _optional_text(proposal.get("summary"))
            if summary is not None:
                return summary
        return None

    def _planning_policy_query_text(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
    ) -> str:
        """Return one deterministic planning-policy query seed."""
        proposal_ids: list[str] = []
        for container in (goal.details, commitment.details if commitment is not None else {}):
            if not isinstance(container, dict):
                continue
            for key in ("pending_plan_proposal_id", "current_plan_proposal_id"):
                proposal_id = _optional_text(container.get(key))
                if proposal_id is not None and proposal_id not in proposal_ids:
                    proposal_ids.append(proposal_id)
        return (
            _optional_text(goal.title)
            or _optional_text(goal.details.get("summary"))
            or self._recent_plan_proposal_summary(proposal_ids=proposal_ids)
            or _optional_text(commitment.title if commitment is not None else None)
            or ""
        )

    def _planning_policy(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
    ) -> BrainExecutivePolicyFrame:
        """Compile the shared executive policy frame for planning work."""
        initiative_class = _goal_initiative_class(goal)
        if initiative_class in {
            BrainInitiativeClass.MAINTENANCE_INTERNAL.value,
            BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        }:
            return neutral_executive_policy_frame(
                task=BrainContextTask.PLANNING,
                updated_at=_utc_now(),
            )
        return self._compile_executive_policy_for_task(
            task=BrainContextTask.PLANNING,
            query_text=self._planning_policy_query_text(
                goal=goal,
                commitment=commitment,
            ),
        )

    @staticmethod
    def _planning_rehearsal_details(rehearsal_result: Any) -> dict[str, Any]:
        """Return one compact replay-safe rehearsal summary."""
        selected_evaluation = next(
            (
                evaluation
                for evaluation in rehearsal_result.evaluations
                if evaluation.evaluation_id == rehearsal_result.selected_evaluation_id
            ),
            None,
        )
        return {
            "rehearsal_id": rehearsal_result.rehearsal_id,
            "step_index": rehearsal_result.step_index,
            "candidate_action_id": rehearsal_result.candidate_action_id,
            "decision_recommendation": rehearsal_result.decision_recommendation,
            "predicted_success_probability": rehearsal_result.predicted_success_probability,
            "confidence_band": rehearsal_result.confidence_band,
            "risk_codes": list(rehearsal_result.risk_codes),
            "skipped": bool(rehearsal_result.skipped),
            "summary": rehearsal_result.summary,
            "rehearsal_kind": rehearsal_result.rehearsal_kind,
            "selected_rehearsal_kind": (
                selected_evaluation.rehearsal_kind if selected_evaluation is not None else None
            ),
            "supporting_prediction_ids": list(rehearsal_result.supporting_prediction_ids),
            "supporting_event_ids": list(rehearsal_result.supporting_event_ids),
        }

    @staticmethod
    def _merge_proposal_rehearsal_details(
        proposal: BrainPlanProposal,
        *,
        rehearsal_result: Any,
        operator_review_floor: bool = False,
        rejected_by_rehearsal: bool = False,
    ) -> BrainPlanProposal:
        """Attach compact rehearsal provenance to one plan proposal."""
        rehearsal_details = BrainExecutive._planning_rehearsal_details(rehearsal_result)
        details = dict(proposal.details)
        rehearsal_by_step = (
            dict(details.get("rehearsal_by_step"))
            if isinstance(details.get("rehearsal_by_step"), dict)
            else {}
        )
        rehearsal_by_step[str(rehearsal_result.step_index)] = dict(rehearsal_details)
        details["rehearsal_by_step"] = rehearsal_by_step
        details["counterfactual_rehearsal"] = dict(rehearsal_details)
        if operator_review_floor:
            details["rehearsal_operator_review_floor"] = True
        if rejected_by_rehearsal:
            details["rehearsal_rejected"] = True
        return replace(proposal, details=details)

    @staticmethod
    def _plan_artifact_details(proposal: BrainPlanProposal | None) -> dict[str, Any]:
        """Return portable proposal-owned runtime details for goal and commitment state."""
        if proposal is None:
            return {}
        details = dict(proposal.details)
        artifact_details: dict[str, Any] = {}
        for key in (
            "counterfactual_rehearsal",
            "rehearsal_by_step",
            "rehearsal_operator_review_floor",
            "rehearsal_rejected",
        ):
            value = details.get(key)
            if isinstance(value, dict):
                artifact_details[key] = deepcopy(value)
            elif isinstance(value, bool):
                artifact_details[key] = value
        return artifact_details

    async def _apply_counterfactual_rehearsal_to_planning_result(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        result: BrainPlanningCoordinatorResult,
        request_kind: str,
    ) -> BrainPlanningCoordinatorResult:
        """Conservatively tighten one planning result with counterfactual rehearsal."""
        if self._counterfactual_rehearsal_engine is None or result.proposal is None:
            return result
        rehearsal_result = await self._counterfactual_rehearsal_engine.rehearse_plan_proposal(
            goal=goal,
            commitment=commitment,
            proposal=result.proposal,
            request_kind=request_kind,
        )
        if rehearsal_result is None:
            return result
        proposal = self._merge_proposal_rehearsal_details(
            result.proposal,
            rehearsal_result=rehearsal_result,
        )
        original_outcome = result.outcome
        original_review_policy = proposal.review_policy
        decision_details = dict(result.decision.details)
        decision_details["counterfactual_rehearsal"] = self._planning_rehearsal_details(
            rehearsal_result
        )
        explicit_abort_risk_codes = {
            "unsafe_action_sensitivity",
            "simulation_plan_invalid",
            "robot_head_preview_not_allowed",
        }
        if (
            rehearsal_result.decision_recommendation
            == BrainRehearsalDecisionRecommendation.ABORT.value
            and explicit_abort_risk_codes.intersection(rehearsal_result.risk_codes)
        ):
            proposal = replace(
                self._merge_proposal_rehearsal_details(
                    proposal,
                    rehearsal_result=rehearsal_result,
                    rejected_by_rehearsal=True,
                ),
                review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
            )
            return BrainPlanningCoordinatorResult(
                progressed=result.progressed,
                outcome=BrainPlanningOutcome.REJECTED.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary="Counterfactual rehearsal rejected the first embodied step.",
                    reason="rehearsal_rejected_embodied_step",
                    details={
                        **decision_details,
                        "original_outcome": original_outcome,
                        "original_review_policy": original_review_policy,
                    },
                ),
            )
        if rehearsal_result.decision_recommendation in {
            BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
            BrainRehearsalDecisionRecommendation.WAIT.value,
            BrainRehearsalDecisionRecommendation.REPAIR.value,
        }:
            proposal = replace(
                self._merge_proposal_rehearsal_details(
                    proposal,
                    rehearsal_result=rehearsal_result,
                    operator_review_floor=True,
                ),
                review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
            )
            return BrainPlanningCoordinatorResult(
                progressed=result.progressed,
                outcome=BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary="Counterfactual rehearsal requires operator review before dispatch.",
                    reason="rehearsal_requires_operator_review",
                    details={
                        **decision_details,
                        "original_outcome": original_outcome,
                        "original_review_policy": original_review_policy,
                    },
                ),
            )
        return BrainPlanningCoordinatorResult(
            progressed=result.progressed,
            outcome=result.outcome,
            proposal=proposal,
            decision=BrainPlanProposalDecision(
                summary=result.decision.summary,
                reason=result.decision.reason,
                details=decision_details,
                reason_codes=list(result.decision.reason_codes),
                executive_policy=(
                    dict(result.decision.executive_policy)
                    if result.decision.executive_policy is not None
                    else None
                ),
            ),
        )

    async def request_plan_proposal(
        self,
        *,
        goal_id: str,
    ) -> BrainPlanningCoordinatorResult:
        """Request one initial bounded plan proposal for a goal without steps."""
        session_ids = self._session_resolver()
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = agenda.goal(goal_id)
        if goal is None:
            raise KeyError(f"Missing goal '{goal_id}'.")
        if goal.steps:
            raise ValueError("Initial plan proposal only applies to goals without steps.")
        commitment = (
            self._store.get_executive_commitment(commitment_id=goal.commitment_id)
            if goal.commitment_id
            else None
        )
        executive_policy = self._planning_policy(
            goal=goal,
            commitment=commitment,
        )
        planning_requested_event = self._append_event(
            session_ids=session_ids,
            event_type=BrainEventType.PLANNING_REQUESTED,
            source="executive",
            payload={"goal_id": goal.goal_id, "intent": goal.intent},
            correlation_id=goal.goal_id,
        )
        planned_steps = self._planner.plan(goal=goal, registry=self._capability_registry)
        if planned_steps:
            proposal = self._build_plan_proposal(
                goal=goal,
                commitment=commitment,
                source=BrainPlanProposalSource.DETERMINISTIC_PLANNER.value,
                summary=f"Deterministic planner expanded {goal.intent or goal.title}.",
                current_plan_revision=goal.plan_revision,
                plan_revision=goal.plan_revision,
                review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
                steps=planned_steps,
                preserved_prefix_count=0,
                assumptions=[],
                missing_inputs=[],
                details={"intent": goal.intent},
            )
            planning_result = await self._apply_counterfactual_rehearsal_to_planning_result(
                goal=goal,
                commitment=commitment,
                result=BrainPlanningCoordinatorResult(
                    progressed=True,
                    outcome=BrainPlanningOutcome.AUTO_ADOPTED.value,
                    proposal=proposal,
                    decision=BrainPlanProposalDecision(
                        summary="Adopted deterministic plan.",
                        reason="deterministic_plan_available",
                        details={"intent": goal.intent},
                    ),
                ),
                request_kind=BrainPlanningRequestKind.INITIAL_PLAN.value,
            )
            planning_result = apply_executive_policy_to_planning_result(
                result=planning_result,
                executive_policy=executive_policy,
            )
            return await self._apply_initial_planning_result(
                goal=goal,
                commitment=commitment,
                result=planning_result,
                planning_requested_event_id=planning_requested_event.event_id,
            )

        planning_result = await self._planning_coordinator.request_proposal(
            BrainPlanningRequest(
                request_kind=BrainPlanningRequestKind.INITIAL_PLAN.value,
                goal=goal,
                commitment=commitment,
                skill_candidates=self._planning_skill_candidates(
                    goal=goal,
                    commitment=commitment,
                    completed_prefix=[],
                ),
                supersedes_plan_proposal_id=_optional_text(
                    goal.details.get("pending_plan_proposal_id")
                    or goal.details.get("current_plan_proposal_id")
                ),
                executive_policy=executive_policy,
            )
        )
        planning_result = await self._apply_counterfactual_rehearsal_to_planning_result(
            goal=goal,
            commitment=commitment,
            result=planning_result,
            request_kind=BrainPlanningRequestKind.INITIAL_PLAN.value,
        )
        return await self._apply_initial_planning_result(
            goal=goal,
            commitment=commitment,
            result=planning_result,
            planning_requested_event_id=planning_requested_event.event_id,
        )

    async def request_commitment_plan_revision(
        self,
        *,
        commitment_id: str,
    ) -> BrainPlanningCoordinatorResult:
        """Request one bounded remaining-tail revision for durable work."""
        session_ids = self._session_resolver()
        commitment = self._require_commitment(commitment_id)
        goal = self._require_goal(commitment, session_ids=session_ids)
        completed_prefix = planning_completed_prefix(goal)
        executive_policy = self._planning_policy(
            goal=goal,
            commitment=commitment,
        )
        planning_result = await self._planning_coordinator.request_proposal(
            BrainPlanningRequest(
                request_kind=BrainPlanningRequestKind.REVISE_TAIL.value,
                goal=goal,
                commitment=commitment,
                completed_prefix=completed_prefix,
                skill_candidates=self._planning_skill_candidates(
                    goal=goal,
                    commitment=commitment,
                    completed_prefix=completed_prefix,
                ),
                supersedes_plan_proposal_id=_optional_text(
                    goal.details.get("pending_plan_proposal_id")
                    or goal.details.get("current_plan_proposal_id")
                    or commitment.details.get("pending_plan_proposal_id")
                    or commitment.details.get("current_plan_proposal_id")
                ),
                executive_policy=executive_policy,
            )
        )
        planning_result = await self._apply_counterfactual_rehearsal_to_planning_result(
            goal=goal,
            commitment=commitment,
            result=planning_result,
            request_kind=BrainPlanningRequestKind.REVISE_TAIL.value,
        )
        return await self._apply_revision_planning_result(
            goal=goal,
            commitment=commitment,
            result=planning_result,
        )

    def _planning_skill_candidates(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        completed_prefix: list[BrainGoalStep],
    ) -> list[BrainPlanningSkillCandidate]:
        session_ids = self._session_resolver()
        procedural_skills = self._store.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=session_ids.thread_id,
        )
        return match_planning_skills(
            procedural_skills=procedural_skills,
            goal=goal,
            commitment=commitment,
            completed_prefix=completed_prefix,
            query_text=goal.title,
        )

    async def run_once(
        self,
        *,
        allowed_commitment_ids: set[str] | None = None,
    ) -> BrainExecutiveCycleResult:
        """Advance the next runnable goal by one explicit executive step."""
        session_ids = self._session_resolver()
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = self._select_next_goal(agenda, allowed_commitment_ids=allowed_commitment_ids)
        if goal is None:
            return BrainExecutiveCycleResult(progressed=False)

        commitment = (
            self._store.get_executive_commitment(commitment_id=goal.commitment_id)
            if goal.commitment_id
            else None
        )
        if not goal.steps:
            planning_result = await self.request_plan_proposal(goal_id=goal.goal_id)
            if planning_result.outcome != BrainPlanningOutcome.AUTO_ADOPTED.value:
                held_goal = self._store.get_agenda_projection(
                    scope_key=session_ids.thread_id,
                    user_id=session_ids.user_id,
                ).goal(goal.goal_id)
                return BrainExecutiveCycleResult(
                    progressed=planning_result.progressed,
                    goal_id=goal.goal_id,
                    goal_status=held_goal.status if held_goal is not None else goal.status,
                    paused=True,
                )
            goal = self._store.get_agenda_projection(
                scope_key=session_ids.thread_id,
                user_id=session_ids.user_id,
            ).goal(goal.goal_id) or goal
            commitment = (
                self._store.get_executive_commitment(commitment_id=goal.commitment_id)
                if goal.commitment_id
                else None
            )

        step_index = goal.next_runnable_step_index()
        if step_index is None:
            return BrainExecutiveCycleResult(progressed=False)

        current_step = deepcopy(goal.steps[step_index])
        plan_proposal_id = _optional_text(
            goal.details.get("current_plan_proposal_id")
            or goal.details.get("pending_plan_proposal_id")
            or (commitment.details.get("current_plan_proposal_id") if commitment is not None else None)
            or (commitment.details.get("pending_plan_proposal_id") if commitment is not None else None)
        )
        rehearsal_by_step = (
            dict(goal.details.get("rehearsal_by_step"))
            if isinstance(goal.details.get("rehearsal_by_step"), dict)
            else {}
        )
        rehearsal_details = (
            dict(rehearsal_by_step.get(str(step_index)))
            if isinstance(rehearsal_by_step.get(str(step_index)), dict)
            else {}
        )
        execution_metadata = {
            "goal_id": goal.goal_id,
            "step_index": step_index,
            **({"commitment_id": commitment.commitment_id} if commitment is not None else {}),
            **({"plan_proposal_id": plan_proposal_id} if plan_proposal_id is not None else {}),
            **(
                {"rehearsal_id": _optional_text(rehearsal_details.get("rehearsal_id"))}
                if _optional_text(rehearsal_details.get("rehearsal_id")) is not None
                else {}
            ),
        }
        embodied_decision = None
        if goal.intent == "robot_head.sequence" and self._embodied_coordinator is not None:
            embodied_decision = await self._embodied_coordinator.prepare_dispatch(
                goal=goal,
                commitment=commitment,
                step=current_step,
                step_index=step_index,
                plan_proposal_id=plan_proposal_id,
                rehearsal_details=rehearsal_details,
                executive_policy=self._compile_executive_policy_for_task(
                    task=BrainContextTask.PLANNING,
                    query_text=goal.title,
                ),
            )
            if not embodied_decision.should_dispatch:
                return self._apply_embodied_pre_dispatch_hold(
                    session_ids=session_ids,
                    goal=goal,
                    commitment=commitment,
                    decision=embodied_decision,
                    step_index=step_index,
                )
        request_event = self._append_event(
            session_ids=session_ids,
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            source="executive",
            payload={
                "goal_id": goal.goal_id,
                "capability_id": current_step.capability_id,
                "arguments": current_step.arguments,
                "step_index": step_index,
                "metadata": execution_metadata,
            },
            correlation_id=goal.goal_id,
        )
        capability_family = self._capability_registry.get(current_step.capability_id).family
        execution = await self._capability_registry.execute(
            current_step.capability_id,
            current_step.arguments,
            context=CapabilityExecutionContext(
                source=_goal_dispatch_source(goal=goal, capability_family=capability_family),
                session_ids=session_ids,
                store=self._store,
                presence_scope_key=self._presence_scope_key,
                dispatch_mode=CapabilityDispatchMode.GOAL.value,
                goal_family=goal.goal_family,
                goal_intent=goal.intent,
                initiative_class=_goal_initiative_class(goal),
                side_effect_sink=self._capability_side_effect_sink,
                metadata=execution_metadata,
            ),
        )

        updated_goal = self._copy_goal(goal)
        updated_goal.active_step_index = step_index
        updated_goal.status = BrainGoalStatus.IN_PROGRESS.value
        updated_goal.steps[step_index].attempts += 1
        updated_goal.steps[step_index].summary = execution.summary
        updated_goal.steps[step_index].error_code = execution.error_code
        updated_goal.steps[step_index].warnings = list(execution.warnings)
        updated_goal.steps[step_index].output = dict(execution.output)
        updated_goal.steps[step_index].updated_at = _utc_now()
        updated_goal.last_summary = execution.summary
        updated_goal.details["commitment_status"] = (
            commitment.status if commitment is not None else BrainCommitmentStatus.ACTIVE.value
        )
        if embodied_decision is not None:
            updated_goal.details["embodied_coordinator"] = {
                "intent_id": embodied_decision.intent.intent_id,
                "intent_kind": embodied_decision.intent.intent_kind,
                "disposition": embodied_decision.disposition,
                "action_id": embodied_decision.envelope.action_id,
                "trace_id": embodied_decision.trace.trace_id,
                "envelope_id": embodied_decision.envelope.envelope_id,
            }

        if execution.accepted:
            updated_goal.steps[step_index].status = "completed"
            robot_action_event_id = (
                self._latest_robot_action_event_id(
                    session_ids=session_ids,
                    goal_id=goal.goal_id,
                    step_index=step_index,
                    action_id=(
                        embodied_decision.envelope.action_id
                        if embodied_decision is not None
                        else None
                    ),
                )
                if embodied_decision is not None
                else None
            )
            capability_terminal_event = self._append_event(
                session_ids=session_ids,
                event_type=BrainEventType.CAPABILITY_COMPLETED,
                source="executive",
                payload={
                    "goal_id": goal.goal_id,
                    "capability_id": current_step.capability_id,
                    "step_index": step_index,
                    "result": execution.model_dump(),
                    **(
                        {"robot_action_event_id": robot_action_event_id}
                        if robot_action_event_id is not None
                        else {}
                    ),
                },
                correlation_id=goal.goal_id,
                causal_parent_id=request_event.event_id,
            )
            if embodied_decision is not None:
                trace = self._embodied_coordinator.record_dispatch_completion(
                    intent=embodied_decision.intent,
                    envelope=embodied_decision.envelope,
                    prepared_trace=embodied_decision.trace,
                    request_event=request_event,
                    terminal_event=capability_terminal_event,
                    execution=execution,
                )
                updated_goal.details["embodied_execution"] = {
                    "trace_id": trace.trace_id,
                    "status": trace.status,
                    "outcome_summary": trace.outcome_summary,
                    "robot_action_event_id": trace.robot_action_event_id,
                }
            if updated_goal.next_runnable_step_index() is None:
                updated_goal.status = BrainGoalStatus.COMPLETED.value
                updated_goal.details["commitment_status"] = BrainCommitmentStatus.COMPLETED.value
                updated_goal.updated_at = _utc_now()
                if commitment is not None:
                    self._store.upsert_executive_commitment(
                        commitment_id=commitment.commitment_id,
                        scope_type=commitment.scope_type,
                        scope_id=commitment.scope_id,
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        title=commitment.title,
                        goal_family=commitment.goal_family,
                        intent=commitment.intent,
                        status=BrainCommitmentStatus.COMPLETED.value,
                        details={
                            **commitment.details,
                            "summary": commitment.details.get("summary") or commitment.title,
                        },
                        current_goal_id=commitment.current_goal_id,
                        blocked_reason=None,
                        wake_conditions=[],
                        plan_revision=commitment.plan_revision,
                        resume_count=commitment.resume_count,
                        source_event_id=request_event.event_id,
                    )
                completed_event = self._append_goal_lifecycle_event(
                    session_ids=session_ids,
                    event_type=BrainEventType.GOAL_COMPLETED,
                    goal=updated_goal,
                    source="executive",
                    commitment=commitment,
                    correlation_id=goal.goal_id,
                    causal_parent_id=request_event.event_id,
                )
                await self._run_post_goal_terminal_hooks(
                    goal_family=updated_goal.goal_family,
                    released_goal_id=updated_goal.goal_id,
                    source_event=completed_event,
                )
                return BrainExecutiveCycleResult(
                    progressed=True,
                    goal_id=goal.goal_id,
                    goal_status=updated_goal.status,
                    paused=False,
                )

            self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_UPDATED,
                goal=updated_goal,
                source="executive",
                commitment=commitment,
                correlation_id=goal.goal_id,
            )
            return BrainExecutiveCycleResult(
                progressed=True,
                goal_id=goal.goal_id,
                goal_status=updated_goal.status,
                paused=False,
            )

        decision = self._critic.evaluate(goal=updated_goal, step=updated_goal.steps[step_index], result=execution)
        updated_goal.recovery_count += 1
        updated_goal.last_error = execution.error_code or execution.outcome
        robot_action_event_id = (
            self._latest_robot_action_event_id(
                session_ids=session_ids,
                goal_id=goal.goal_id,
                step_index=step_index,
                action_id=(
                    embodied_decision.envelope.action_id if embodied_decision is not None else None
                ),
            )
            if embodied_decision is not None
            else None
        )
        capability_terminal_event = self._append_event(
            session_ids=session_ids,
            event_type=BrainEventType.CAPABILITY_FAILED,
            source="executive",
            payload={
                "goal_id": goal.goal_id,
                "capability_id": current_step.capability_id,
                "step_index": step_index,
                "result": execution.model_dump(),
                **(
                    {"robot_action_event_id": robot_action_event_id}
                    if robot_action_event_id is not None
                    else {}
                ),
            },
            correlation_id=goal.goal_id,
            causal_parent_id=request_event.event_id,
        )
        if embodied_decision is not None:
            trace = self._embodied_coordinator.record_dispatch_completion(
                intent=embodied_decision.intent,
                envelope=embodied_decision.envelope,
                prepared_trace=embodied_decision.trace,
                request_event=request_event,
                terminal_event=capability_terminal_event,
                execution=execution,
            )
            recovery = self._embodied_coordinator.record_recovery(
                intent=embodied_decision.intent,
                trace=trace,
                execution=execution,
            )
            updated_goal.details["embodied_execution"] = {
                "trace_id": trace.trace_id,
                "status": trace.status,
                "outcome_summary": trace.outcome_summary,
                "mismatch_codes": list(trace.mismatch_codes),
                "recovery_action_id": trace.recovery_action_id,
            }
            if recovery is not None:
                updated_goal.details["embodied_recovery"] = {
                    "recovery_id": recovery.recovery_id,
                    "action_id": recovery.action_id,
                    "status": recovery.status,
                    "summary": recovery.summary,
                }
        self._append_event(
            session_ids=session_ids,
            event_type=BrainEventType.CRITIC_FEEDBACK,
            source="executive",
            payload={
                "goal_id": goal.goal_id,
                "capability_id": current_step.capability_id,
                "step_index": step_index,
                "result": execution.model_dump(),
                "recovery": {
                    "decision": decision.decision,
                    "summary": decision.summary,
                },
            },
            correlation_id=goal.goal_id,
            causal_parent_id=capability_terminal_event.event_id,
        )
        if decision.decision == BrainGoalStatus.RETRY.value:
            updated_goal.status = BrainGoalStatus.RETRY.value
            updated_goal.steps[step_index].status = "retry"
            updated_goal.updated_at = _utc_now()
            self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_UPDATED,
                goal=updated_goal,
                source="executive",
                commitment=commitment,
                correlation_id=goal.goal_id,
            )
            return BrainExecutiveCycleResult(
                progressed=True,
                goal_id=goal.goal_id,
                goal_status=updated_goal.status,
                paused=True,
            )

        updated_goal.steps[step_index].status = (
            "blocked" if decision.decision == BrainGoalStatus.BLOCKED.value else "failed"
        )
        updated_goal.status = decision.decision
        updated_goal.blocked_reason = decision.blocked_reason
        updated_goal.wake_conditions = list(decision.wake_conditions)
        if commitment is None and decision.decision == BrainGoalStatus.BLOCKED.value:
            promotion_decision = should_promote_goal_on_block(updated_goal)
            if promotion_decision.durable and promotion_decision.scope_type is not None:
                commitment = self.promote_goal_to_commitment(
                    goal_id=updated_goal.goal_id,
                    scope_type=promotion_decision.scope_type,
                    correlation_id=goal.goal_id,
                    causal_parent_id=request_event.event_id,
                )
                updated_goal.commitment_id = commitment.commitment_id
        updated_goal.details["commitment_status"] = (
            BrainCommitmentStatus.BLOCKED.value if commitment is not None else updated_goal.status
        )
        updated_goal.updated_at = _utc_now()
        if commitment is not None:
            self._store.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=BrainCommitmentStatus.BLOCKED.value,
                details={
                    **commitment.details,
                    "summary": commitment.details.get("summary") or commitment.title,
                },
                current_goal_id=commitment.current_goal_id,
                blocked_reason=decision.blocked_reason,
                wake_conditions=list(decision.wake_conditions),
                plan_revision=commitment.plan_revision,
                resume_count=commitment.resume_count,
                source_event_id=request_event.event_id,
            )
        terminal_event = self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=(
                BrainEventType.GOAL_FAILED
                if decision.decision == BrainGoalStatus.FAILED.value
                else BrainEventType.GOAL_UPDATED
            ),
            goal=updated_goal,
            source="executive",
            commitment=commitment,
            correlation_id=goal.goal_id,
            causal_parent_id=request_event.event_id,
        )
        await self._run_post_goal_terminal_hooks(
            goal_family=updated_goal.goal_family,
            released_goal_id=updated_goal.goal_id,
            source_event=terminal_event,
        )
        return BrainExecutiveCycleResult(
            progressed=True,
            goal_id=goal.goal_id,
            goal_status=updated_goal.status,
            paused=True,
        )

    def _apply_embodied_pre_dispatch_hold(
        self,
        *,
        session_ids: BrainSessionIds,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        decision,
        step_index: int,
    ) -> BrainExecutiveCycleResult:
        """Persist one pre-dispatch embodied defer or abort outcome."""
        updated_goal = self._copy_goal(goal)
        updated_goal.active_step_index = step_index
        updated_goal.last_summary = decision.trace.outcome_summary
        updated_goal.blocked_reason = decision.blocked_reason
        updated_goal.wake_conditions = list(decision.wake_conditions)
        updated_goal.details["embodied_coordinator"] = {
            "intent_id": decision.intent.intent_id,
            "intent_kind": decision.intent.intent_kind,
            "disposition": decision.disposition,
            "action_id": decision.envelope.action_id,
            "trace_id": decision.trace.trace_id,
            "envelope_id": decision.envelope.envelope_id,
        }
        if decision.disposition == BrainEmbodiedDispatchDisposition.ABORT.value:
            updated_goal.status = BrainGoalStatus.BLOCKED.value
            updated_goal.steps[step_index].status = "blocked"
            commitment_status = BrainCommitmentStatus.BLOCKED.value
            event_type = BrainEventType.GOAL_UPDATED
        else:
            updated_goal.status = BrainGoalStatus.WAITING.value
            updated_goal.steps[step_index].status = "waiting"
            commitment_status = (
                BrainCommitmentStatus.DEFERRED.value
                if commitment is not None
                else BrainGoalStatus.WAITING.value
            )
            event_type = BrainEventType.GOAL_DEFERRED if commitment is not None else BrainEventType.GOAL_UPDATED
        updated_goal.details["commitment_status"] = commitment_status
        updated_goal.updated_at = _utc_now()
        updated_commitment = None
        if commitment is not None:
            updated_commitment = self._store.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=commitment_status,
                details={
                    **commitment.details,
                    "summary": commitment.details.get("summary") or commitment.title,
                },
                current_goal_id=commitment.current_goal_id,
                blocked_reason=decision.blocked_reason,
                wake_conditions=list(decision.wake_conditions),
                plan_revision=commitment.plan_revision,
                resume_count=commitment.resume_count,
            )
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=event_type,
            goal=updated_goal,
            source="executive",
            commitment=updated_commitment,
            correlation_id=goal.goal_id,
            event_details={"embodied_coordinator": dict(updated_goal.details["embodied_coordinator"])},
        )
        return BrainExecutiveCycleResult(
            progressed=True,
            goal_id=goal.goal_id,
            goal_status=updated_goal.status,
            paused=True,
        )

    def _latest_robot_action_event_id(
        self,
        *,
        session_ids: BrainSessionIds,
        goal_id: str,
        step_index: int,
        action_id: str | None,
    ) -> str | None:
        """Return the latest matching robot action outcome event id for one goal step."""
        for event in self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=24,
            event_types=(BrainEventType.ROBOT_ACTION_OUTCOME,),
        ):
            payload = event.payload if isinstance(event.payload, dict) else {}
            if _optional_text(payload.get("goal_id")) != goal_id:
                continue
            if int(payload.get("step_index", -1)) != int(step_index):
                continue
            if action_id is not None and _optional_text(payload.get("action_id")) != action_id:
                continue
            return event.event_id
        return None

    def defer_commitment(
        self,
        *,
        commitment_id: str,
        reason: BrainBlockedReason | None = None,
        wake_conditions: list[BrainWakeCondition] | None = None,
        source: str = "executive",
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        event_details: dict[str, Any] | None = None,
    ) -> BrainCommitmentRecord:
        """Defer one durable commitment without auto-resuming it."""
        session_ids = self._session_resolver()
        commitment = self._require_commitment(commitment_id)
        goal = self._require_goal(commitment, session_ids=session_ids)
        resolved_reason = reason or BrainBlockedReason(
            kind=BrainBlockedReasonKind.EXPLICIT_DEFER.value,
            summary="Deferred explicitly.",
            details={},
        )
        updated_goal = self._copy_goal(goal)
        updated_goal.status = BrainGoalStatus.WAITING.value
        updated_goal.blocked_reason = resolved_reason
        updated_goal.wake_conditions = list(
            wake_conditions
            or [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume explicitly when ready.",
                    details={},
                )
            ]
        )
        updated_goal.details["commitment_status"] = BrainCommitmentStatus.DEFERRED.value
        updated_goal.updated_at = _utc_now()
        updated_commitment = self._store.upsert_executive_commitment(
            commitment_id=commitment.commitment_id,
            scope_type=commitment.scope_type,
            scope_id=commitment.scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=commitment.title,
            goal_family=commitment.goal_family,
            intent=commitment.intent,
            status=BrainCommitmentStatus.DEFERRED.value,
            details={**commitment.details, "summary": commitment.details.get("summary") or commitment.title},
            current_goal_id=commitment.current_goal_id,
            blocked_reason=resolved_reason,
            wake_conditions=list(updated_goal.wake_conditions),
            plan_revision=commitment.plan_revision,
            resume_count=commitment.resume_count,
        )
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_DEFERRED,
            goal=updated_goal,
            source=source,
            commitment=updated_commitment,
            correlation_id=correlation_id or commitment.commitment_id,
            causal_parent_id=causal_parent_id,
            event_details=event_details,
        )
        return updated_commitment

    def resume_commitment(
        self,
        *,
        commitment_id: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        source: str = "executive",
        event_details: dict[str, Any] | None = None,
    ) -> BrainCommitmentRecord:
        """Resume one durable commitment explicitly."""
        session_ids = self._session_resolver()
        commitment = self._require_commitment(commitment_id)
        goal = self._require_goal(commitment, session_ids=session_ids)
        updated_goal = self._copy_goal(goal)
        updated_goal.status = _goal_status_for_activation(updated_goal)
        updated_goal.blocked_reason = None
        updated_goal.wake_conditions = []
        updated_goal.resume_count = commitment.resume_count + 1
        for step in updated_goal.steps:
            if step.status in {"failed", "blocked"}:
                step.status = "retry"
        updated_goal.details["commitment_status"] = BrainCommitmentStatus.ACTIVE.value
        updated_goal.updated_at = _utc_now()
        updated_commitment = self._store.upsert_executive_commitment(
            commitment_id=commitment.commitment_id,
            scope_type=commitment.scope_type,
            scope_id=commitment.scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=commitment.title,
            goal_family=commitment.goal_family,
            intent=commitment.intent,
            status=BrainCommitmentStatus.ACTIVE.value,
            details={**commitment.details, "summary": commitment.details.get("summary") or commitment.title},
            current_goal_id=commitment.current_goal_id,
            blocked_reason=None,
            wake_conditions=[],
            plan_revision=commitment.plan_revision,
            resume_count=commitment.resume_count + 1,
        )
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_RESUMED,
            goal=updated_goal,
            source=source,
            commitment=updated_commitment,
            correlation_id=correlation_id or commitment.commitment_id,
            causal_parent_id=causal_parent_id,
            event_details=event_details,
        )
        return updated_commitment

    def cancel_commitment(self, *, commitment_id: str) -> BrainCommitmentRecord:
        """Cancel one durable commitment explicitly."""
        session_ids = self._session_resolver()
        commitment = self._require_commitment(commitment_id)
        goal = self._require_goal(commitment, session_ids=session_ids)
        updated_goal = self._copy_goal(goal)
        updated_goal.status = BrainGoalStatus.CANCELLED.value
        updated_goal.details["commitment_status"] = BrainCommitmentStatus.CANCELLED.value
        updated_goal.updated_at = _utc_now()
        updated_commitment = self._store.upsert_executive_commitment(
            commitment_id=commitment.commitment_id,
            scope_type=commitment.scope_type,
            scope_id=commitment.scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=commitment.title,
            goal_family=commitment.goal_family,
            intent=commitment.intent,
            status=BrainCommitmentStatus.CANCELLED.value,
            details={**commitment.details, "summary": commitment.details.get("summary") or commitment.title},
            current_goal_id=commitment.current_goal_id,
            blocked_reason=None,
            wake_conditions=[],
            plan_revision=commitment.plan_revision,
            resume_count=commitment.resume_count,
        )
        cancelled_event = self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_CANCELLED,
            goal=updated_goal,
            source="executive",
            commitment=updated_commitment,
            correlation_id=commitment.commitment_id,
        )
        self._maybe_run_sync_goal_terminal_wake_router(
            goal_family=updated_goal.goal_family,
            released_goal_id=updated_goal.goal_id,
            source_event=cancelled_event,
        )
        return updated_commitment

    def repair_commitment(
        self,
        *,
        commitment_id: str,
        capabilities: list[dict[str, Any] | str],
    ) -> BrainCommitmentRecord:
        """Repair a commitment plan while preserving the completed prefix."""
        session_ids = self._session_resolver()
        commitment = self._require_commitment(commitment_id)
        goal = self._require_goal(commitment, session_ids=session_ids)
        updated_goal = self._copy_goal(goal)
        completed_prefix: list[BrainGoalStep] = []
        for step in updated_goal.steps:
            if step.status != "completed":
                break
            completed_prefix.append(step)
        repaired_tail = [
            BrainGoalStep(
                capability_id=(
                    item.strip() if isinstance(item, str) else str(item.get("capability_id", "")).strip()
                ),
                arguments={} if isinstance(item, str) else dict(item.get("arguments", {})),
            )
            for item in capabilities
            if (item.strip() if isinstance(item, str) else str(item.get("capability_id", "")).strip())
        ]
        updated_goal.steps = completed_prefix + repaired_tail
        updated_goal.status = BrainGoalStatus.WAITING.value
        updated_goal.plan_revision = commitment.plan_revision + 1
        updated_goal.blocked_reason = None
        updated_goal.wake_conditions = []
        commitment_status = BrainCommitmentStatus.ACTIVE.value
        if commitment.status in {
            BrainCommitmentStatus.BLOCKED.value,
            BrainCommitmentStatus.DEFERRED.value,
        }:
            commitment_status = BrainCommitmentStatus.DEFERRED.value
            updated_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume explicitly after the repaired plan is ready.",
                    details={"commitment_id": commitment.commitment_id},
                )
            ]
        updated_goal.details = {
            **self._plan_linked_details(
                updated_goal.details,
                current_plan_proposal_id=None,
                pending_plan_proposal_id=None,
                plan_review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            ),
            "capabilities": capabilities,
            "commitment_status": commitment_status,
        }
        updated_goal.updated_at = _utc_now()
        proposal = self._build_plan_proposal(
            goal=updated_goal,
            commitment=commitment,
            source=BrainPlanProposalSource.REPAIR.value,
            summary=f"Repair plan for {updated_goal.title}.",
            current_plan_revision=commitment.plan_revision,
            plan_revision=commitment.plan_revision + 1,
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            steps=updated_goal.steps,
            preserved_prefix_count=len(completed_prefix),
            assumptions=[],
            missing_inputs=[],
            supersedes_plan_proposal_id=_optional_text(
                goal.details.get("pending_plan_proposal_id")
                or goal.details.get("current_plan_proposal_id")
                or commitment.details.get("pending_plan_proposal_id")
                or commitment.details.get("current_plan_proposal_id")
            ),
            details={
                "intent": updated_goal.intent,
                "capabilities": capabilities,
            },
        )
        repair_decision = BrainPlanProposalDecision(
            summary="Adopted repaired plan tail.",
            reason="repair_applied",
            details={
                "commitment_id": commitment.commitment_id,
                "preserved_prefix_count": len(completed_prefix),
            },
        )
        proposed_event = self._store.append_planning_proposed(
            proposal=proposal,
            decision=repair_decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=commitment.commitment_id,
        )
        adopted_event = self._store.append_planning_adopted(
            proposal=proposal,
            decision=repair_decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=commitment.commitment_id,
            causal_parent_id=proposed_event.event_id,
        )
        updated_commitment = self._store.upsert_executive_commitment(
            commitment_id=commitment.commitment_id,
            scope_type=commitment.scope_type,
            scope_id=commitment.scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=commitment.title,
            goal_family=commitment.goal_family,
            intent=commitment.intent,
            status=commitment_status,
            details=self._plan_linked_details(
                {
                    **commitment.details,
                    "summary": commitment.details.get("summary") or commitment.title,
                },
                current_plan_proposal_id=proposal.plan_proposal_id,
                pending_plan_proposal_id=None,
                plan_review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            ),
            current_goal_id=commitment.current_goal_id,
            blocked_reason=None,
            wake_conditions=list(updated_goal.wake_conditions),
            plan_revision=commitment.plan_revision + 1,
            resume_count=commitment.resume_count,
        )
        updated_goal.details = self._plan_linked_details(
            updated_goal.details,
            current_plan_proposal_id=proposal.plan_proposal_id,
            pending_plan_proposal_id=None,
            plan_review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        )
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_REPAIRED,
            goal=updated_goal,
            source="executive",
            commitment=updated_commitment,
            correlation_id=commitment.commitment_id,
            causal_parent_id=adopted_event.event_id,
        )
        return updated_commitment

    def explain_agenda_state(self) -> str:
        """Return a concise planning-facing agenda explanation."""
        session_ids = self._session_resolver()
        if self._context_surface_builder is not None:
            snapshot = self._context_surface_builder.build(
                latest_user_text="",
                mode=BrainContextMode.PLANNING,
                include_historical_claims=False,
            )
            selected = self._context_selector.select(
                snapshot=snapshot,
                task=BrainContextTask.PLANNING,
                language=self._context_surface_builder.language,
            )
            commitment_projection = snapshot.commitment_projection
            agenda = snapshot.agenda
        else:
            commitment_projection = self._store.get_session_commitment_projection(
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
            )
            agenda = self._store.get_agenda_projection(
                scope_key=session_ids.thread_id,
                user_id=session_ids.user_id,
            )
            selected = None
        active = [record.title for record in commitment_projection.active_commitments[:4]]
        blocked = [
            f"{record.title}: {record.blocked_reason.summary}"
            if record.blocked_reason is not None
            else record.title
            for record in commitment_projection.blocked_commitments[:4]
        ]
        deferred = [record.title for record in commitment_projection.deferred_commitments[:4]]
        resumable = [
            record.title
            for record in commitment_projection.blocked_commitments[:2]
            + commitment_projection.deferred_commitments[:2]
        ]
        lines = [
            f"Trying now: {agenda.active_goal_summary or 'None'}",
            f"Owes the user: {'; '.join(active) if active else 'None'}",
            f"Blocked: {'; '.join(blocked) if blocked else 'None'}",
            f"Deferred: {'; '.join(deferred) if deferred else 'None'}",
            f"Can be resumed: {'; '.join(resumable) if resumable else 'None'}",
        ]
        if selected is not None:
            trace = selected.selection_trace
            lines.append(
                f"Planning context: {trace.total_selected_tokens}/{trace.budget_profile.max_tokens} tokens"
            )
        return "\n".join(
            lines
        )

    async def _run_commitment_pass(self, *, max_iterations: int) -> BrainExecutiveCycleResult:
        """Drive only active runnable durable commitments."""
        session_ids = self._session_resolver()
        active_commitments = self._store.list_session_commitments(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            statuses=(BrainCommitmentStatus.ACTIVE.value,),
            limit=24,
        )
        allowed_commitment_ids = {
            record.commitment_id for record in active_commitments if record.current_goal_id
        }
        if not allowed_commitment_ids:
            return BrainExecutiveCycleResult(progressed=False)
        any_progress = False
        last_result = BrainExecutiveCycleResult(progressed=False)
        for _ in range(max_iterations):
            last_result = await self.run_once(allowed_commitment_ids=allowed_commitment_ids)
            any_progress = any_progress or last_result.progressed
            if not last_result.progressed or last_result.paused:
                break
        return BrainExecutiveCycleResult(
            progressed=any_progress,
            goal_id=last_result.goal_id,
            goal_status=last_result.goal_status,
            paused=last_result.paused,
        )

    def _merge_cycle_and_director_results(
        self,
        *,
        commitment_result: BrainExecutiveCycleResult,
        router_results: tuple[BrainCommitmentWakeRouterResult | None, ...],
        director_results: tuple[BrainPresenceDirectorResult | None, ...],
    ) -> BrainExecutiveCycleResult:
        """Merge one commitment cycle result with router/director side results."""
        progressed_router_results = [
            result for result in router_results if result is not None and result.progressed
        ]
        progressed_results = [
            result for result in director_results if result is not None and result.progressed
        ]
        if not progressed_results and not progressed_router_results:
            return commitment_result
        if not progressed_results:
            primary_router_result = progressed_router_results[-1]
            goal_id = commitment_result.goal_id
            goal_status = commitment_result.goal_status
            if primary_router_result.resumed_commitment_id is not None:
                session_ids = self._session_resolver()
                resumed_commitment = self._store.get_executive_commitment(
                    commitment_id=primary_router_result.resumed_commitment_id
                )
                resumed_goal = (
                    self._store.get_agenda_projection(
                        scope_key=session_ids.thread_id,
                        user_id=session_ids.user_id,
                    ).goal(resumed_commitment.current_goal_id)
                    if resumed_commitment is not None and resumed_commitment.current_goal_id
                    else None
                )
                if resumed_goal is not None:
                    goal_id = resumed_goal.goal_id
                    goal_status = resumed_goal.status
            return BrainExecutiveCycleResult(
                progressed=True,
                goal_id=goal_id,
                goal_status=goal_status,
                paused=commitment_result.paused,
            )
        primary_result = next(
            (
                result
                for result in reversed(progressed_results)
                if result.accepted_goal_id is not None
            ),
            progressed_results[-1],
        )
        session_ids = self._session_resolver()
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        accepted_goal = (
            agenda.goal(primary_result.accepted_goal_id)
            if primary_result.accepted_goal_id is not None
            else None
        )
        return BrainExecutiveCycleResult(
            progressed=True,
            goal_id=primary_result.accepted_goal_id or commitment_result.goal_id,
            goal_status=accepted_goal.status if accepted_goal is not None else commitment_result.goal_status,
            paused=commitment_result.paused,
        )

    async def _run_post_goal_terminal_hooks(
        self,
        *,
        goal_family: str,
        released_goal_id: str | None,
        source_event: Any,
    ):
        """Fire bounded candidate reevaluation and wake-router checks after goal release."""
        self._maybe_trigger_goal_family_available_reevaluation(
            goal_family=goal_family,
            released_goal_id=released_goal_id,
            source_event=source_event,
        )
        await self.run_commitment_wake_router(
            boundary_kind="goal_terminal",
            source_event=source_event,
        )

    def _maybe_run_sync_goal_terminal_wake_router(
        self,
        *,
        goal_family: str,
        released_goal_id: str | None,
        source_event: Any,
    ):
        """Best-effort sync hook for explicit terminal transitions outside the async loop."""
        self._maybe_trigger_goal_family_available_reevaluation(
            goal_family=goal_family,
            released_goal_id=released_goal_id,
            source_event=source_event,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(
                self.run_commitment_wake_router(
                    boundary_kind="goal_terminal",
                    source_event=source_event,
                )
            )

    def _maybe_trigger_goal_family_available_reevaluation(
        self,
        *,
        goal_family: str,
        released_goal_id: str | None,
        source_event: Any,
    ) -> BrainPresenceDirectorResult | None:
        """Reevaluate held same-family candidates once a busy family becomes available."""
        session_ids = self._session_resolver()
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        if self._family_busy_excluding_goal(
            goal_family=goal_family,
            agenda=agenda,
            excluded_goal_id=released_goal_id,
        ):
            return None
        return self.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.GOAL_FAMILY_AVAILABLE.value,
                summary=f"Reevaluate held {goal_family} candidates after the active family cleared.",
                details={"goal_family": goal_family},
                source_event_type=source_event.event_type if source_event is not None else None,
                source_event_id=source_event.event_id if source_event is not None else None,
                ts=source_event.ts if source_event is not None else _utc_now(),
            )
        )

    @staticmethod
    def _family_busy_excluding_goal(
        *,
        goal_family: str,
        agenda: BrainAgendaProjection,
        excluded_goal_id: str | None,
    ) -> bool:
        for goal in agenda.goals:
            if goal.goal_id == excluded_goal_id or goal.goal_family != goal_family:
                continue
            if goal.status in {
                BrainGoalStatus.OPEN.value,
                BrainGoalStatus.PLANNING.value,
                BrainGoalStatus.IN_PROGRESS.value,
                BrainGoalStatus.RETRY.value,
            }:
                return True
        return False

    def _select_next_goal(
        self,
        agenda: BrainAgendaProjection,
        *,
        allowed_commitment_ids: set[str] | None = None,
    ) -> BrainGoal | None:
        """Choose the next explicit runnable goal."""
        goals = [
            goal
            for goal in agenda.goals
            if goal.status
            in {
                BrainGoalStatus.OPEN.value,
                BrainGoalStatus.PLANNING.value,
                BrainGoalStatus.IN_PROGRESS.value,
                BrainGoalStatus.RETRY.value,
            }
            and (
                allowed_commitment_ids is None
                or (goal.commitment_id is not None and goal.commitment_id in allowed_commitment_ids)
            )
        ]
        if not goals:
            return None
        return sorted(
            goals,
            key=lambda goal: (
                self._family_priority(goal.goal_family),
                goal.updated_at,
                goal.created_at,
                goal.title,
            ),
        )[0]

    def _append_goal_lifecycle_event(
        self,
        *,
        session_ids: BrainSessionIds,
        event_type: str,
        goal: BrainGoal,
        source: str,
        commitment: BrainCommitmentRecord | None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        event_details: dict[str, Any] | None = None,
    ):
        """Append one goal lifecycle event, including commitment metadata when present."""
        payload: dict[str, Any] = {"goal": goal.as_dict()}
        if commitment is not None:
            payload["commitment"] = {
                "commitment_id": commitment.commitment_id,
                "scope_type": commitment.scope_type,
                "scope_id": commitment.scope_id,
                "status": commitment.status,
            }
        if event_details:
            payload.update({str(key): value for key, value in event_details.items()})
        return self._append_event(
            session_ids=session_ids,
            event_type=event_type,
            source=source,
            payload=payload,
            correlation_id=correlation_id or goal.goal_id,
            causal_parent_id=causal_parent_id,
        )

    async def _apply_initial_planning_result(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        result: BrainPlanningCoordinatorResult,
        planning_requested_event_id: str,
    ) -> BrainPlanningCoordinatorResult:
        """Persist one initial-plan proposal decision."""
        if result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value:
            return await self._adopt_initial_plan(
                goal=goal,
                commitment=commitment,
                proposal=result.proposal,
                decision=result.decision,
                proposed_causal_parent_id=planning_requested_event_id,
            )

        session_ids = self._session_resolver()
        proposal = result.proposal
        if result.outcome == BrainPlanningOutcome.REJECTED.value:
            decision_event = self._store.append_planning_rejected(
                proposal=proposal,
                decision=result.decision,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="executive",
                correlation_id=goal.goal_id,
                causal_parent_id=planning_requested_event_id,
            )
            rejected_goal = self._copy_goal(goal)
            rejected_goal.status = BrainGoalStatus.BLOCKED.value
            rejected_goal.planning_requested = True
            rejected_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.OPERATOR_REVIEW.value,
                summary="A bounded operator review is required before planning can continue.",
                details={
                    "plan_proposal_id": proposal.plan_proposal_id,
                    "reason": result.decision.reason,
                },
            )
            rejected_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.OPERATOR_REVIEW.value,
                    summary="Resume after bounded operator review.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            rejected_goal.last_summary = result.decision.summary
            rejected_goal.last_error = result.decision.reason
            rejected_goal.details = self._plan_linked_details(
                {
                    **rejected_goal.details,
                    "commitment_status": BrainCommitmentStatus.BLOCKED.value,
                    **self._plan_artifact_details(proposal),
                },
                pending_plan_proposal_id=None,
                plan_review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
            )
            rejected_goal.updated_at = _utc_now()
            persisted_commitment = self._persist_planning_commitment_state(
                session_ids=session_ids,
                goal=rejected_goal,
                commitment=commitment,
                status=BrainCommitmentStatus.BLOCKED.value,
                source_event_id=decision_event.event_id,
                use_block_policy=True,
            )
            if persisted_commitment is not None:
                rejected_goal.commitment_id = persisted_commitment.commitment_id
            blocked_event = self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_UPDATED,
                goal=rejected_goal,
                source="executive",
                commitment=persisted_commitment,
                correlation_id=goal.goal_id,
                causal_parent_id=decision_event.event_id,
            )
            await self._run_post_goal_terminal_hooks(
                goal_family=rejected_goal.goal_family,
                released_goal_id=rejected_goal.goal_id,
                source_event=blocked_event,
            )
            return result

        proposed_event = self._store.append_planning_proposed(
            proposal=proposal,
            decision=result.decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=goal.goal_id,
            causal_parent_id=planning_requested_event_id,
        )
        review_goal = self._copy_goal(goal)
        review_goal.planning_requested = True
        if result.outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value:
            review_goal.status = BrainGoalStatus.WAITING.value
            review_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.WAITING_USER.value,
                summary="Waiting for user-owned inputs before adopting the bounded plan.",
                details={
                    "plan_proposal_id": proposal.plan_proposal_id,
                    "missing_inputs": list(proposal.missing_inputs),
                },
            )
            review_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.USER_RESPONSE.value,
                    summary="Resume after the user supplies the missing planning inputs.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            review_goal.details = self._plan_linked_details(
                {
                    **review_goal.details,
                    "commitment_status": BrainCommitmentStatus.DEFERRED.value,
                    **self._plan_artifact_details(proposal),
                },
                pending_plan_proposal_id=proposal.plan_proposal_id,
                plan_review_policy=proposal.review_policy,
            )
            commitment_status = BrainCommitmentStatus.DEFERRED.value
            use_block_policy = False
        else:
            review_goal.status = BrainGoalStatus.BLOCKED.value
            review_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.OPERATOR_REVIEW.value,
                summary="The bounded plan requires operator review before adoption.",
                details={
                    "plan_proposal_id": proposal.plan_proposal_id,
                    "capability_families": list(
                        proposal.details.get("capability_families", [])
                    ),
                },
            )
            review_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.OPERATOR_REVIEW.value,
                    summary="Resume after bounded operator review.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            review_goal.details = self._plan_linked_details(
                {
                    **review_goal.details,
                    "commitment_status": BrainCommitmentStatus.BLOCKED.value,
                    **self._plan_artifact_details(proposal),
                },
                pending_plan_proposal_id=proposal.plan_proposal_id,
                plan_review_policy=proposal.review_policy,
            )
            commitment_status = BrainCommitmentStatus.BLOCKED.value
            use_block_policy = True
        review_goal.last_summary = result.decision.summary
        review_goal.last_error = result.decision.reason
        review_goal.updated_at = _utc_now()
        persisted_commitment = self._persist_planning_commitment_state(
            session_ids=session_ids,
            goal=review_goal,
            commitment=commitment,
            status=commitment_status,
            source_event_id=proposed_event.event_id,
            use_block_policy=use_block_policy,
        )
        if persisted_commitment is not None:
            review_goal.commitment_id = persisted_commitment.commitment_id
        goal_event = self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_UPDATED,
            goal=review_goal,
            source="executive",
            commitment=persisted_commitment,
            correlation_id=goal.goal_id,
            causal_parent_id=proposed_event.event_id,
        )
        if result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value:
            await self._run_post_goal_terminal_hooks(
                goal_family=review_goal.goal_family,
                released_goal_id=review_goal.goal_id,
                source_event=goal_event,
            )
        return result

    async def _adopt_initial_plan(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        proposal: BrainPlanProposal,
        decision: BrainPlanProposalDecision,
        proposed_causal_parent_id: str,
    ) -> BrainPlanningCoordinatorResult:
        """Persist one adopted initial plan proposal."""
        session_ids = self._session_resolver()
        proposed_event = self._store.append_planning_proposed(
            proposal=proposal,
            decision=decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=goal.goal_id,
            causal_parent_id=proposed_causal_parent_id,
        )
        adopted_event = self._store.append_planning_adopted(
            proposal=proposal,
            decision=decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=goal.goal_id,
            causal_parent_id=proposed_event.event_id,
        )
        planned_goal = self._copy_goal(goal)
        planned_goal.status = BrainGoalStatus.OPEN.value
        planned_goal.steps = [BrainGoalStep.from_dict(step.as_dict()) for step in proposal.steps]
        planned_goal.planning_requested = True
        planned_goal.blocked_reason = None
        planned_goal.wake_conditions = []
        planned_goal.plan_revision = proposal.plan_revision
        planned_goal.last_summary = decision.summary
        planned_goal.last_error = None
        planned_goal.details = self._plan_linked_details(
            {
                **planned_goal.details,
                "commitment_status": BrainCommitmentStatus.ACTIVE.value,
                **self._plan_artifact_details(proposal),
            },
            current_plan_proposal_id=proposal.plan_proposal_id,
            pending_plan_proposal_id=None,
            plan_review_policy=proposal.review_policy,
        )
        planned_goal.updated_at = _utc_now()
        persisted_commitment = None
        if commitment is not None:
            persisted_commitment = self._store.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=BrainCommitmentStatus.ACTIVE.value,
                details=self._plan_linked_details(
                    {
                        **commitment.details,
                        "summary": commitment.details.get("summary") or commitment.title,
                        **self._plan_artifact_details(proposal),
                    },
                    current_plan_proposal_id=proposal.plan_proposal_id,
                    pending_plan_proposal_id=None,
                    plan_review_policy=proposal.review_policy,
                ),
                current_goal_id=commitment.current_goal_id,
                blocked_reason=None,
                wake_conditions=[],
                plan_revision=proposal.plan_revision,
                resume_count=commitment.resume_count,
                source_event_id=adopted_event.event_id,
            )
            planned_goal.commitment_id = persisted_commitment.commitment_id
        self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_UPDATED,
            goal=planned_goal,
            source="executive",
            commitment=persisted_commitment or commitment,
            correlation_id=goal.goal_id,
            causal_parent_id=adopted_event.event_id,
        )
        return BrainPlanningCoordinatorResult(
            progressed=True,
            outcome=BrainPlanningOutcome.AUTO_ADOPTED.value,
            proposal=proposal,
            decision=decision,
        )

    async def _apply_revision_planning_result(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord,
        result: BrainPlanningCoordinatorResult,
    ) -> BrainPlanningCoordinatorResult:
        """Persist one remaining-tail revision proposal decision."""
        session_ids = self._session_resolver()
        proposal = result.proposal
        if result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value:
            updated_goal = self._copy_goal(goal)
            updated_goal.steps = [BrainGoalStep.from_dict(step.as_dict()) for step in proposal.steps]
            updated_goal.status = BrainGoalStatus.WAITING.value
            updated_goal.plan_revision = proposal.plan_revision
            updated_goal.blocked_reason = None
            updated_goal.wake_conditions = []
            commitment_status = BrainCommitmentStatus.ACTIVE.value
            if commitment.status in {
                BrainCommitmentStatus.BLOCKED.value,
                BrainCommitmentStatus.DEFERRED.value,
            }:
                commitment_status = BrainCommitmentStatus.DEFERRED.value
                updated_goal.wake_conditions = [
                    BrainWakeCondition(
                        kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                        summary="Resume explicitly after the revised plan is ready.",
                        details={"commitment_id": commitment.commitment_id},
                    )
                ]
            updated_goal.last_summary = result.decision.summary
            updated_goal.last_error = None
            updated_goal.details = self._plan_linked_details(
                {
                    **updated_goal.details,
                    "commitment_status": commitment_status,
                    **self._plan_artifact_details(proposal),
                },
                current_plan_proposal_id=proposal.plan_proposal_id,
                pending_plan_proposal_id=None,
                plan_review_policy=proposal.review_policy,
            )
            updated_goal.updated_at = _utc_now()
            proposed_event = self._store.append_planning_proposed(
                proposal=proposal,
                decision=result.decision,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="executive",
                correlation_id=commitment.commitment_id,
            )
            adopted_event = self._store.append_planning_adopted(
                proposal=proposal,
                decision=result.decision,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="executive",
                correlation_id=commitment.commitment_id,
                causal_parent_id=proposed_event.event_id,
            )
            updated_commitment = self._store.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=commitment_status,
                details=self._plan_linked_details(
                    {
                        **commitment.details,
                        "summary": commitment.details.get("summary") or commitment.title,
                        **self._plan_artifact_details(proposal),
                    },
                    current_plan_proposal_id=proposal.plan_proposal_id,
                    pending_plan_proposal_id=None,
                    plan_review_policy=proposal.review_policy,
                ),
                current_goal_id=commitment.current_goal_id,
                blocked_reason=None,
                wake_conditions=list(updated_goal.wake_conditions),
                plan_revision=proposal.plan_revision,
                resume_count=commitment.resume_count,
                source_event_id=adopted_event.event_id,
            )
            self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_REPAIRED,
                goal=updated_goal,
                source="executive",
                commitment=updated_commitment,
                correlation_id=commitment.commitment_id,
                causal_parent_id=adopted_event.event_id,
            )
            return BrainPlanningCoordinatorResult(
                progressed=True,
                outcome=BrainPlanningOutcome.AUTO_ADOPTED.value,
                proposal=proposal,
                decision=result.decision,
            )

        if result.outcome == BrainPlanningOutcome.REJECTED.value:
            decision_event = self._store.append_planning_rejected(
                proposal=proposal,
                decision=result.decision,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="executive",
                correlation_id=commitment.commitment_id,
            )
            review_goal = self._copy_goal(goal)
            review_goal.status = BrainGoalStatus.BLOCKED.value
            review_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.OPERATOR_REVIEW.value,
                summary="Operator review is required before revising the remaining plan tail.",
                details={
                    "plan_proposal_id": proposal.plan_proposal_id,
                    "reason": result.decision.reason,
                },
            )
            review_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.OPERATOR_REVIEW.value,
                    summary="Resume after bounded operator review.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            review_goal.last_summary = result.decision.summary
            review_goal.last_error = result.decision.reason
            review_goal.details = self._plan_linked_details(
                {
                    **review_goal.details,
                    "commitment_status": BrainCommitmentStatus.BLOCKED.value,
                    **self._plan_artifact_details(proposal),
                },
                pending_plan_proposal_id=None,
                plan_review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
            )
            review_goal.updated_at = _utc_now()
            updated_commitment = self._store.upsert_executive_commitment(
                commitment_id=commitment.commitment_id,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                title=commitment.title,
                goal_family=commitment.goal_family,
                intent=commitment.intent,
                status=BrainCommitmentStatus.BLOCKED.value,
                details=self._plan_linked_details(
                    {
                        **commitment.details,
                        "summary": commitment.details.get("summary") or commitment.title,
                        **self._plan_artifact_details(proposal),
                    },
                    pending_plan_proposal_id=None,
                    plan_review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
                ),
                current_goal_id=commitment.current_goal_id,
                blocked_reason=review_goal.blocked_reason,
                wake_conditions=list(review_goal.wake_conditions),
                plan_revision=commitment.plan_revision,
                resume_count=commitment.resume_count,
                source_event_id=decision_event.event_id,
            )
            blocked_event = self._append_goal_lifecycle_event(
                session_ids=session_ids,
                event_type=BrainEventType.GOAL_UPDATED,
                goal=review_goal,
                source="executive",
                commitment=updated_commitment,
                correlation_id=commitment.commitment_id,
                causal_parent_id=decision_event.event_id,
            )
            await self._run_post_goal_terminal_hooks(
                goal_family=review_goal.goal_family,
                released_goal_id=review_goal.goal_id,
                source_event=blocked_event,
            )
            return result

        proposed_event = self._store.append_planning_proposed(
            proposal=proposal,
            decision=result.decision,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="executive",
            correlation_id=commitment.commitment_id,
        )
        review_goal = self._copy_goal(goal)
        if result.outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value:
            review_goal.status = BrainGoalStatus.WAITING.value
            review_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.WAITING_USER.value,
                summary="Waiting for user-owned inputs before revising the remaining plan tail.",
                details={
                    "plan_proposal_id": proposal.plan_proposal_id,
                    "missing_inputs": list(proposal.missing_inputs),
                },
            )
            review_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.USER_RESPONSE.value,
                    summary="Resume after the user supplies the missing planning inputs.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            commitment_status = BrainCommitmentStatus.DEFERRED.value
        else:
            review_goal.status = BrainGoalStatus.BLOCKED.value
            review_goal.blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.OPERATOR_REVIEW.value,
                summary="The revised remaining tail requires operator review before adoption.",
                details={"plan_proposal_id": proposal.plan_proposal_id},
            )
            review_goal.wake_conditions = [
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.OPERATOR_REVIEW.value,
                    summary="Resume after bounded operator review.",
                    details={"plan_proposal_id": proposal.plan_proposal_id},
                )
            ]
            commitment_status = BrainCommitmentStatus.BLOCKED.value
        review_goal.last_summary = result.decision.summary
        review_goal.last_error = result.decision.reason
        review_goal.details = self._plan_linked_details(
            {
                **review_goal.details,
                "commitment_status": commitment_status,
                **self._plan_artifact_details(proposal),
            },
            pending_plan_proposal_id=proposal.plan_proposal_id,
            plan_review_policy=proposal.review_policy,
        )
        review_goal.updated_at = _utc_now()
        updated_commitment = self._store.upsert_executive_commitment(
            commitment_id=commitment.commitment_id,
            scope_type=commitment.scope_type,
            scope_id=commitment.scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=commitment.title,
            goal_family=commitment.goal_family,
            intent=commitment.intent,
            status=commitment_status,
            details=self._plan_linked_details(
                {
                    **commitment.details,
                    "summary": commitment.details.get("summary") or commitment.title,
                    **self._plan_artifact_details(proposal),
                },
                pending_plan_proposal_id=proposal.plan_proposal_id,
                plan_review_policy=proposal.review_policy,
            ),
            current_goal_id=commitment.current_goal_id,
            blocked_reason=review_goal.blocked_reason,
            wake_conditions=list(review_goal.wake_conditions),
            plan_revision=commitment.plan_revision,
            resume_count=commitment.resume_count,
            source_event_id=proposed_event.event_id,
        )
        review_event = self._append_goal_lifecycle_event(
            session_ids=session_ids,
            event_type=BrainEventType.GOAL_UPDATED,
            goal=review_goal,
            source="executive",
            commitment=updated_commitment,
            correlation_id=commitment.commitment_id,
            causal_parent_id=proposed_event.event_id,
        )
        if result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value:
            await self._run_post_goal_terminal_hooks(
                goal_family=review_goal.goal_family,
                released_goal_id=review_goal.goal_id,
                source_event=review_event,
            )
        return result

    def _persist_planning_commitment_state(
        self,
        *,
        session_ids: BrainSessionIds,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        status: str,
        source_event_id: str,
        use_block_policy: bool,
    ) -> BrainCommitmentRecord | None:
        """Persist one planning-related commitment state when durability is needed."""
        resolved_commitment = commitment
        resolved_scope_type = commitment.scope_type if commitment is not None else None
        resolved_scope_id = commitment.scope_id if commitment is not None else None
        if resolved_commitment is None:
            promotion = (
                should_promote_goal_on_block(goal)
                if use_block_policy
                else should_auto_promote_goal(goal)
            )
            if not promotion.durable or promotion.scope_type is None:
                return None
            scope_ids = build_executive_scope_ids(session_ids)
            resolved_scope_type = promotion.scope_type
            resolved_scope_id = self._commitment_scope_id(
                session_ids=session_ids,
                scope_type=promotion.scope_type,
                scope_ids=scope_ids,
            )
        return self._store.upsert_executive_commitment(
            commitment_id=(
                resolved_commitment.commitment_id if resolved_commitment is not None else None
            ),
            scope_type=resolved_scope_type,
            scope_id=resolved_scope_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            title=goal.title,
            goal_family=goal.goal_family,
            intent=goal.intent,
            status=status,
            details={
                **goal.details,
                "summary": goal.details.get("summary") or goal.title,
            },
            current_goal_id=goal.goal_id,
            blocked_reason=goal.blocked_reason,
            wake_conditions=list(goal.wake_conditions),
            plan_revision=goal.plan_revision,
            resume_count=goal.resume_count,
            source_event_id=source_event_id,
        )

    @staticmethod
    def _plan_linked_details(
        details: dict[str, Any],
        *,
        current_plan_proposal_id: str | None | object = _PLAN_DETAIL_KEEP,
        pending_plan_proposal_id: str | None | object = _PLAN_DETAIL_KEEP,
        plan_review_policy: str | None | object = _PLAN_DETAIL_KEEP,
    ) -> dict[str, Any]:
        """Update replayable plan-linkage details deterministically."""
        updated = dict(details)
        if current_plan_proposal_id is not _PLAN_DETAIL_KEEP:
            if current_plan_proposal_id is None:
                updated.pop("current_plan_proposal_id", None)
            else:
                updated["current_plan_proposal_id"] = current_plan_proposal_id
        if pending_plan_proposal_id is not _PLAN_DETAIL_KEEP:
            if pending_plan_proposal_id is None:
                updated.pop("pending_plan_proposal_id", None)
            else:
                updated["pending_plan_proposal_id"] = pending_plan_proposal_id
        if plan_review_policy is not _PLAN_DETAIL_KEEP:
            if plan_review_policy is None:
                updated.pop("plan_review_policy", None)
            else:
                updated["plan_review_policy"] = plan_review_policy
        return updated

    @staticmethod
    def _build_plan_proposal(
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        source: str,
        summary: str,
        current_plan_revision: int,
        plan_revision: int,
        review_policy: str = BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps: list[BrainGoalStep],
        preserved_prefix_count: int,
        assumptions: list[str] | None = None,
        missing_inputs: list[str] | None = None,
        supersedes_plan_proposal_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> BrainPlanProposal:
        """Build one typed, replayable plan proposal."""
        return BrainPlanProposal(
            plan_proposal_id=f"plan-proposal-{uuid4()}",
            goal_id=goal.goal_id,
            commitment_id=commitment.commitment_id if commitment is not None else goal.commitment_id,
            source=source,
            summary=summary,
            current_plan_revision=current_plan_revision,
            plan_revision=plan_revision,
            review_policy=review_policy,
            steps=[BrainGoalStep.from_dict(step.as_dict()) for step in steps],
            preserved_prefix_count=preserved_prefix_count,
            assumptions=list(assumptions or []),
            missing_inputs=list(missing_inputs or []),
            supersedes_plan_proposal_id=supersedes_plan_proposal_id,
            details=dict(details or {}),
        )

    def _append_event(
        self,
        *,
        session_ids: BrainSessionIds,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
    ):
        """Append one executive event to the canonical event spine."""
        return self._store.append_brain_event(
            event_type=event_type,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=source,
            payload=payload,
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
        )

    def _notify_autonomy_state_changed(self):
        """Notify the runtime that candidate wake scheduling may need a refresh."""
        if self._autonomy_state_changed_callback is None:
            return
        self._autonomy_state_changed_callback()

    def _commitment_scope_id(
        self,
        *,
        session_ids: BrainSessionIds,
        scope_type: str,
        scope_ids=None,
    ) -> str:
        """Resolve one canonical durable commitment scope id."""
        resolved_scope_ids = scope_ids or build_executive_scope_ids(session_ids)
        if scope_type == BrainCommitmentScopeType.AGENT.value:
            return resolved_scope_ids.agent_scope_id
        if scope_type == BrainCommitmentScopeType.THREAD.value:
            return resolved_scope_ids.thread_scope_id
        return resolved_scope_ids.relationship_scope_id

    def _require_commitment(self, commitment_id: str) -> BrainCommitmentRecord:
        """Return one durable commitment or raise a helpful error."""
        commitment = self._store.get_executive_commitment(commitment_id=commitment_id)
        if commitment is None:
            raise KeyError(f"Missing commitment '{commitment_id}'.")
        return commitment

    def _require_goal(
        self,
        commitment: BrainCommitmentRecord,
        *,
        session_ids: BrainSessionIds,
    ) -> BrainGoal:
        """Return the current projected goal for one durable commitment."""
        if not commitment.current_goal_id:
            raise KeyError(f"Commitment '{commitment.commitment_id}' has no current goal.")
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = agenda.goal(commitment.current_goal_id)
        if goal is None:
            raise KeyError(
                f"Missing goal '{commitment.current_goal_id}' for commitment '{commitment.commitment_id}'."
            )
        return goal

    @staticmethod
    def _family_priority(goal_family: str) -> int:
        """Return the family-aware scheduling priority."""
        return {
            BrainGoalFamily.CONVERSATION.value: 0,
            BrainGoalFamily.ENVIRONMENT.value: 1,
            BrainGoalFamily.MEMORY_MAINTENANCE.value: 2,
        }.get(goal_family, 99)

    @staticmethod
    def _copy_goal(goal: BrainGoal) -> BrainGoal:
        """Copy one goal through the JSON-compatible projection shape."""
        return BrainGoal.from_dict(goal.as_dict())


__all__ = [
    "BrainExecutive",
    "BrainExecutiveCritic",
    "BrainExecutiveCycleResult",
    "BrainExecutivePlanner",
    "BrainPresenceDirector",
    "BrainPresenceDirectorPolicy",
    "BrainPresenceDirectorResult",
    "BrainExecutiveRecoveryDecision",
]

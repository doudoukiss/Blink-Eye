"""Bounded durable-commitment wake router for Blink."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
)
from blink.brain.capabilities import (
    CapabilityDispatchMode,
    CapabilityExecutionContext,
    CapabilityExecutionResult,
    CapabilityRegistry,
)
from blink.brain.core.events import BrainEventRecord
from blink.brain.core.projections import (
    BrainAgendaProjection,
    BrainCommitmentRecord,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainGoal,
    BrainGoalStatus,
    BrainWakeCondition,
    BrainWakeConditionKind,
    BrainWorkingContextProjection,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(value: str | None) -> datetime | None:
    """Parse one stored ISO timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _goal_initiative_class(goal: BrainGoal) -> str | None:
    """Return initiative metadata attached to one goal, if present."""
    autonomy = goal.details.get("autonomy")
    if not isinstance(autonomy, dict):
        return None
    value = str(autonomy.get("initiative_class", "")).strip()
    return value or None


def _keep_waiting_cooldowns() -> dict[str, int]:
    return {
        BrainWakeConditionKind.THREAD_IDLE.value: 15,
        BrainWakeConditionKind.CONDITION_CLEARED.value: 15,
    }


@dataclass(frozen=True)
class BrainCommitmentWakeRouterPolicy:
    """Deterministic routing policy for one bounded wake-router pass."""

    max_commitments_per_pass: int = 24
    recent_event_limit: int = 128
    keep_waiting_cooldown_seconds_by_wake_kind: dict[str, int] = field(
        default_factory=_keep_waiting_cooldowns
    )

    def keep_waiting_cooldown_seconds(self, wake_kind: str) -> int:
        """Return the short cooldown window for repeated keep-waiting wakes."""
        return int(self.keep_waiting_cooldown_seconds_by_wake_kind.get(wake_kind, 0))


@dataclass(frozen=True)
class BrainCommitmentWakeRouterResult:
    """Inspectable result from one bounded commitment wake-router pass."""

    progressed: bool = False
    matched_commitment_id: str | None = None
    trigger_event_id: str | None = None
    route_kind: str | None = None
    resumed_commitment_id: str | None = None
    proposed_candidate_goal_id: str | None = None
    reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    executive_policy: dict[str, Any] | None = None


@dataclass(frozen=True)
class BrainCommitmentWakeDecision:
    """One matched wake plus the selected bounded route."""

    commitment: BrainCommitmentRecord
    wake_condition: BrainWakeCondition
    trigger: BrainCommitmentWakeTrigger
    routing: BrainCommitmentWakeRoutingDecision
    candidate_goal: BrainCandidateGoal | None = None


class BrainCommitmentWakeRouter:
    """Match durable wake conditions into bounded resume/propose/wait decisions."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver: Callable[[], BrainSessionIds],
        capability_registry: CapabilityRegistry,
        presence_scope_key: str,
        policy: BrainCommitmentWakeRouterPolicy | None = None,
    ):
        """Initialize the wake router."""
        self._store = store
        self._session_resolver = session_resolver
        self._capability_registry = capability_registry
        self._presence_scope_key = presence_scope_key
        self._policy = policy or BrainCommitmentWakeRouterPolicy()

    async def route_once(
        self,
        *,
        boundary_kind: str,
        source_event: BrainEventRecord | None = None,
    ) -> BrainCommitmentWakeDecision | None:
        """Return one bounded wake decision or ``None`` when nothing matches."""
        session_ids = self._session_resolver()
        working_context = self._store.get_working_context_projection(scope_key=session_ids.thread_id)
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        recent_events = self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=self._policy.recent_event_limit,
        )
        wake_events = [
            event
            for event in recent_events
            if event.event_type == "commitment.wake.triggered"
        ]
        latest_user_turn_event = self._latest_user_turn_event(
            recent_events=recent_events,
            source_event=source_event,
        )
        now = _parse_ts(source_event.ts if source_event is not None else None) or _utc_now()
        commitments = self._store.list_session_commitments(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            statuses=(
                BrainCommitmentStatus.DEFERRED.value,
                BrainCommitmentStatus.BLOCKED.value,
            ),
            limit=self._policy.max_commitments_per_pass,
        )
        decisions: list[BrainCommitmentWakeDecision] = []
        for commitment in commitments:
            decision = await self._match_commitment(
                commitment=commitment,
                boundary_kind=boundary_kind,
                source_event=source_event,
                latest_user_turn_event=latest_user_turn_event,
                working_context=working_context,
                agenda=agenda,
                current_candidates=ledger.current_candidates,
                wake_events=wake_events,
                session_ids=session_ids,
                now=now,
            )
            if decision is not None:
                decisions.append(decision)
        if not decisions:
            return None

        family_leaders: dict[str, BrainCommitmentWakeDecision] = {}
        for decision in decisions:
            current = family_leaders.get(decision.commitment.goal_family)
            if current is None or self._family_leader_key(decision) < self._family_leader_key(current):
                family_leaders[decision.commitment.goal_family] = decision
        return min(
            family_leaders.values(),
            key=lambda item: self._family_attention_key(item, wake_events=wake_events),
        )

    async def _match_commitment(
        self,
        *,
        commitment: BrainCommitmentRecord,
        boundary_kind: str,
        source_event: BrainEventRecord | None,
        latest_user_turn_event: BrainEventRecord | None,
        working_context: BrainWorkingContextProjection,
        agenda: BrainAgendaProjection,
        current_candidates: list[BrainCandidateGoal],
        wake_events: list[BrainEventRecord],
        session_ids: BrainSessionIds,
        now: datetime,
    ) -> BrainCommitmentWakeDecision | None:
        current_goal = agenda.goal(commitment.current_goal_id) if commitment.current_goal_id else None
        base_goal_details = (
            dict(current_goal.details)
            if current_goal is not None and isinstance(current_goal.details, dict)
            else {}
        )
        blocked_reason_summary = (
            commitment.blocked_reason.summary if commitment.blocked_reason is not None else None
        )
        for wake in commitment.wake_conditions:
            if wake.kind == BrainWakeConditionKind.EXPLICIT_RESUME.value:
                continue
            if wake.kind == BrainWakeConditionKind.OPERATOR_REVIEW.value:
                continue
            if wake.kind == BrainWakeConditionKind.THREAD_IDLE.value:
                if working_context.user_turn_open or working_context.assistant_turn_open:
                    continue
                if self._family_busy_excluding_goal(
                    goal_family=commitment.goal_family,
                    agenda=agenda,
                    excluded_goal_id=commitment.current_goal_id,
                ):
                    continue
                dedupe_key = f"{commitment.commitment_id}:{wake.kind}"
                current_candidate = self._current_candidate_for_dedupe_key(
                    current_candidates=current_candidates,
                    dedupe_key=dedupe_key,
                )
                if current_candidate is not None:
                    if self._skip_keep_waiting_due_to_cooldown(
                        wake_events=wake_events,
                        commitment_id=commitment.commitment_id,
                        wake_kind=wake.kind,
                        now=now,
                    ):
                        return None
                    return self._build_keep_waiting_decision(
                        commitment=commitment,
                        wake_condition=wake,
                        summary="Keep waiting because a current wake candidate already exists.",
                        reason="candidate_already_current",
                        details={
                            "boundary_kind": boundary_kind,
                            "existing_candidate_goal_id": current_candidate.candidate_goal_id,
                            "dedupe_key": dedupe_key,
                        },
                        source_event=source_event,
                        now=now,
                    )
                return self._build_candidate_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    boundary_kind=boundary_kind,
                    candidate_type="commitment_wake_thread_idle",
                    initiative_class=(
                        BrainInitiativeClass.MAINTENANCE_INTERNAL.value
                        if commitment.goal_family == "memory_maintenance"
                        else BrainInitiativeClass.INSPECT_ONLY.value
                    ),
                    source_event=source_event,
                    source_event_for_trigger=source_event,
                    blocked_reason_summary=blocked_reason_summary,
                    base_goal_details=base_goal_details,
                    session_ids=session_ids,
                    now=now,
                )
            if wake.kind == BrainWakeConditionKind.USER_RESPONSE.value:
                if latest_user_turn_event is None:
                    continue
                if latest_user_turn_event.ts <= commitment.updated_at:
                    continue
                if self._wake_already_seen_for_source_event(
                    wake_events=wake_events,
                    commitment_id=commitment.commitment_id,
                    wake_kind=wake.kind,
                    source_event_id=latest_user_turn_event.event_id,
                ):
                    return None
                dedupe_key = f"{commitment.commitment_id}:{wake.kind}"
                current_candidate = self._current_candidate_for_dedupe_key(
                    current_candidates=current_candidates,
                    dedupe_key=dedupe_key,
                )
                if current_candidate is not None:
                    return self._build_keep_waiting_decision(
                        commitment=commitment,
                        wake_condition=wake,
                        summary="Keep waiting because a current wake candidate already exists.",
                        reason="candidate_already_current",
                        details={
                            "boundary_kind": boundary_kind,
                            "existing_candidate_goal_id": current_candidate.candidate_goal_id,
                            "source_event_id": latest_user_turn_event.event_id,
                            "dedupe_key": dedupe_key,
                        },
                        source_event=latest_user_turn_event,
                        now=now,
                    )
                return self._build_candidate_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    boundary_kind=boundary_kind,
                    candidate_type="commitment_wake_user_response",
                    initiative_class=BrainInitiativeClass.DEFER_UNTIL_USER_TURN.value,
                    source_event=latest_user_turn_event,
                    source_event_for_trigger=latest_user_turn_event,
                    blocked_reason_summary=blocked_reason_summary,
                    base_goal_details=base_goal_details,
                    session_ids=session_ids,
                    now=now,
                )
            if wake.kind != BrainWakeConditionKind.CONDITION_CLEARED.value:
                continue
            if commitment.status != BrainCommitmentStatus.BLOCKED.value:
                continue
            capability_id = self._condition_cleared_capability_id(
                wake_condition=wake,
                commitment=commitment,
                goal=current_goal,
            )
            if not capability_id:
                if self._skip_keep_waiting_due_to_cooldown(
                    wake_events=wake_events,
                    commitment_id=commitment.commitment_id,
                    wake_kind=wake.kind,
                    now=now,
                ):
                    return None
                return self._build_keep_waiting_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    summary="Keep waiting because the blocked capability is not machine-identifiable yet.",
                    reason="capability_id_missing",
                    details={"boundary_kind": boundary_kind},
                    source_event=source_event,
                    now=now,
                )
            if current_goal is None:
                if self._skip_keep_waiting_due_to_cooldown(
                    wake_events=wake_events,
                    commitment_id=commitment.commitment_id,
                    wake_kind=wake.kind,
                    now=now,
                ):
                    return None
                return self._build_keep_waiting_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    summary="Keep waiting because the blocked goal is not currently available.",
                    reason="goal_context_missing",
                    details={
                        "boundary_kind": boundary_kind,
                        "capability_id": capability_id,
                    },
                    source_event=source_event,
                    now=now,
                )
            if self._family_busy_excluding_goal(
                goal_family=commitment.goal_family,
                agenda=agenda,
                excluded_goal_id=commitment.current_goal_id,
            ):
                if self._skip_keep_waiting_due_to_cooldown(
                    wake_events=wake_events,
                    commitment_id=commitment.commitment_id,
                    wake_kind=wake.kind,
                    now=now,
                ):
                    return None
                return self._build_keep_waiting_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    summary="Keep waiting until the current goal family clears.",
                    reason="goal_family_busy",
                    details={
                        "boundary_kind": boundary_kind,
                        "goal_family": commitment.goal_family,
                        "capability_id": capability_id,
                    },
                    source_event=source_event,
                    now=now,
                )
            preflight = await self._evaluate_condition_cleared_preflight(
                goal=current_goal,
                capability_id=capability_id,
                session_ids=session_ids,
            )
            if preflight.accepted:
                return self._build_resume_direct_decision(
                    commitment=commitment,
                    wake_condition=wake,
                    boundary_kind=boundary_kind,
                    capability_id=capability_id,
                    source_event=source_event,
                    now=now,
                )
            if self._skip_keep_waiting_due_to_cooldown(
                wake_events=wake_events,
                commitment_id=commitment.commitment_id,
                wake_kind=wake.kind,
                now=now,
            ):
                return None
            return self._build_keep_waiting_decision(
                commitment=commitment,
                wake_condition=wake,
                summary=preflight.summary or "Keep waiting because the blocker is still present.",
                reason=preflight.error_code or "blocker_still_present",
                details={
                    "boundary_kind": boundary_kind,
                    "capability_id": capability_id,
                    "error_code": preflight.error_code,
                    "outcome": preflight.outcome,
                },
                source_event=source_event,
                now=now,
            )
        return None

    def _build_candidate_decision(
        self,
        *,
        commitment: BrainCommitmentRecord,
        wake_condition: BrainWakeCondition,
        boundary_kind: str,
        candidate_type: str,
        initiative_class: str,
        source_event: BrainEventRecord | None,
        source_event_for_trigger: BrainEventRecord | None,
        blocked_reason_summary: str | None,
        base_goal_details: dict[str, Any],
        session_ids: BrainSessionIds,
        now: datetime,
    ) -> BrainCommitmentWakeDecision:
        candidate_goal_id = uuid4().hex
        candidate_payload = {
            "goal_intent": f"autonomy.{candidate_type}",
            "goal_details": {
                "transient_only": True,
                "commitment_wake": {
                    "commitment_id": commitment.commitment_id,
                    "wake_kind": wake_condition.kind,
                    "plan_revision": commitment.plan_revision,
                    "resume_count": commitment.resume_count,
                    "blocked_reason_summary": blocked_reason_summary,
                    "target_goal_intent": commitment.intent,
                    "target_goal_details": base_goal_details,
                    "boundary_kind": boundary_kind,
                    "wake_trigger_kind": wake_condition.kind,
                },
            },
            "commitment_id": commitment.commitment_id,
            "wake_kind": wake_condition.kind,
            "plan_revision": commitment.plan_revision,
            "resume_count": commitment.resume_count,
            "blocked_reason_summary": blocked_reason_summary,
            "target_goal_intent": commitment.intent,
            "target_goal_details": base_goal_details,
            "boundary_kind": boundary_kind,
        }
        candidate_goal = BrainCandidateGoal(
            candidate_goal_id=candidate_goal_id,
            candidate_type=candidate_type,
            source=BrainCandidateGoalSource.COMMITMENT.value,
            summary=f"Revisit deferred commitment: {commitment.title}",
            goal_family=commitment.goal_family,
            urgency=0.7,
            confidence=1.0,
            initiative_class=initiative_class,
            cooldown_key=f"{session_ids.thread_id}:commitment:{commitment.commitment_id}:{wake_condition.kind}",
            dedupe_key=f"{commitment.commitment_id}:{wake_condition.kind}",
            policy_tags=["phase8b", "commitment_wake", wake_condition.kind, boundary_kind],
            requires_user_turn_gap=True,
            expires_at=(now + timedelta(seconds=60)).isoformat(),
            payload=candidate_payload,
            created_at=now.isoformat(),
        )
        trigger = BrainCommitmentWakeTrigger(
            commitment_id=commitment.commitment_id,
            wake_kind=wake_condition.kind,
            summary=f"Matched durable commitment wake: {wake_condition.kind}.",
            details={
                "boundary_kind": boundary_kind,
                "candidate_goal_id": candidate_goal_id,
                "candidate_type": candidate_type,
                "commitment_status": commitment.status,
            },
            source_event_type=(
                source_event_for_trigger.event_type if source_event_for_trigger is not None else None
            ),
            source_event_id=(
                source_event_for_trigger.event_id if source_event_for_trigger is not None else None
            ),
            ts=(source_event_for_trigger.ts if source_event_for_trigger is not None else now.isoformat()),
        )
        routing = BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value,
            summary="Route this wake through bounded candidate policy.",
            details={
                "reason": "wake_matched",
                "boundary_kind": boundary_kind,
                "candidate_goal_id": candidate_goal_id,
                "candidate_type": candidate_type,
                "goal_family": commitment.goal_family,
            },
        )
        return BrainCommitmentWakeDecision(
            commitment=commitment,
            wake_condition=wake_condition,
            trigger=trigger,
            routing=routing,
            candidate_goal=candidate_goal,
        )

    def _build_resume_direct_decision(
        self,
        *,
        commitment: BrainCommitmentRecord,
        wake_condition: BrainWakeCondition,
        boundary_kind: str,
        capability_id: str,
        source_event: BrainEventRecord | None,
        now: datetime,
    ) -> BrainCommitmentWakeDecision:
        trigger = BrainCommitmentWakeTrigger(
            commitment_id=commitment.commitment_id,
            wake_kind=wake_condition.kind,
            summary=f"Matched durable commitment wake: {wake_condition.kind}.",
            details={
                "boundary_kind": boundary_kind,
                "capability_id": capability_id,
                "commitment_status": commitment.status,
            },
            source_event_type=source_event.event_type if source_event is not None else None,
            source_event_id=source_event.event_id if source_event is not None else None,
            ts=source_event.ts if source_event is not None else now.isoformat(),
        )
        routing = BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.RESUME_DIRECT.value,
            summary="Resume this commitment directly because the blocker cleared.",
            details={
                "reason": "blocker_cleared",
                "boundary_kind": boundary_kind,
                "capability_id": capability_id,
                "goal_family": commitment.goal_family,
            },
        )
        return BrainCommitmentWakeDecision(
            commitment=commitment,
            wake_condition=wake_condition,
            trigger=trigger,
            routing=routing,
        )

    def _build_keep_waiting_decision(
        self,
        *,
        commitment: BrainCommitmentRecord,
        wake_condition: BrainWakeCondition,
        summary: str,
        reason: str,
        details: dict[str, Any],
        source_event: BrainEventRecord | None,
        now: datetime,
    ) -> BrainCommitmentWakeDecision:
        trigger = BrainCommitmentWakeTrigger(
            commitment_id=commitment.commitment_id,
            wake_kind=wake_condition.kind,
            summary=f"Matched durable commitment wake: {wake_condition.kind}.",
            details={
                "boundary_kind": details.get("boundary_kind"),
                "commitment_status": commitment.status,
            },
            source_event_type=source_event.event_type if source_event is not None else None,
            source_event_id=source_event.event_id if source_event is not None else None,
            ts=source_event.ts if source_event is not None else now.isoformat(),
        )
        routing = BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.KEEP_WAITING.value,
            summary=summary,
            details={"reason": reason, **details},
        )
        return BrainCommitmentWakeDecision(
            commitment=commitment,
            wake_condition=wake_condition,
            trigger=trigger,
            routing=routing,
        )

    async def _evaluate_condition_cleared_preflight(
        self,
        *,
        goal: BrainGoal,
        capability_id: str,
        session_ids: BrainSessionIds,
    ):
        blocked_step = self._blocked_step_for_capability(goal=goal, capability_id=capability_id)
        try:
            if blocked_step is None:
                self._capability_registry.get(capability_id)
                return await self._capability_registry.evaluate_preconditions(
                    capability_id,
                    {},
                    context=CapabilityExecutionContext(
                        source=goal.source,
                        session_ids=session_ids,
                        store=self._store,
                        presence_scope_key=self._presence_scope_key,
                        dispatch_mode=CapabilityDispatchMode.GOAL.value,
                        goal_family=goal.goal_family,
                        goal_intent=goal.intent,
                        initiative_class=_goal_initiative_class(goal),
                        metadata={"goal_id": goal.goal_id, "wake_router_preflight": True},
                    ),
                )
            return await self._capability_registry.evaluate_preconditions(
                capability_id,
                blocked_step.arguments,
                context=CapabilityExecutionContext(
                    source=goal.source,
                    session_ids=session_ids,
                    store=self._store,
                    presence_scope_key=self._presence_scope_key,
                    dispatch_mode=CapabilityDispatchMode.GOAL.value,
                    goal_family=goal.goal_family,
                    goal_intent=goal.intent,
                    initiative_class=_goal_initiative_class(goal),
                    metadata={
                        "goal_id": goal.goal_id,
                        "wake_router_preflight": True,
                        "step_capability_id": capability_id,
                    },
                ),
            )
        except KeyError:
            return CapabilityExecutionResult.blocked(
                capability_id=capability_id,
                summary=f"Keep waiting because capability '{capability_id}' is unsupported.",
                error_code="unsupported_capability",
            )

    @staticmethod
    def _blocked_step_for_capability(goal: BrainGoal, *, capability_id: str):
        for step in goal.steps:
            if step.capability_id != capability_id:
                continue
            if step.status in {"blocked", "failed", "retry", "pending"}:
                return step
        return None

    @staticmethod
    def _condition_cleared_capability_id(
        *,
        wake_condition: BrainWakeCondition,
        commitment: BrainCommitmentRecord,
        goal: BrainGoal | None,
    ) -> str | None:
        from_wake = _optional_text(wake_condition.details.get("capability_id"))
        if from_wake:
            return from_wake
        if commitment.blocked_reason is not None:
            from_blocked_reason = _optional_text(commitment.blocked_reason.details.get("capability_id"))
            if from_blocked_reason:
                return from_blocked_reason
        if goal is None:
            return None
        blocked_steps = [step.capability_id for step in goal.steps if step.status in {"blocked", "failed"}]
        if len(blocked_steps) == 1:
            return blocked_steps[0]
        return None

    @staticmethod
    def _current_candidate_for_dedupe_key(
        *,
        current_candidates: list[BrainCandidateGoal],
        dedupe_key: str,
    ) -> BrainCandidateGoal | None:
        for candidate in current_candidates:
            if candidate.dedupe_key == dedupe_key:
                return candidate
        return None

    def _skip_keep_waiting_due_to_cooldown(
        self,
        *,
        wake_events: list[BrainEventRecord],
        commitment_id: str,
        wake_kind: str,
        now: datetime,
    ) -> bool:
        cooldown_seconds = self._policy.keep_waiting_cooldown_seconds(wake_kind)
        if cooldown_seconds <= 0:
            return False
        for event in wake_events:
            payload = event.payload
            commitment_payload = payload.get("commitment", {})
            routing_payload = payload.get("routing", {})
            if commitment_payload.get("commitment_id") != commitment_id:
                continue
            if (payload.get("trigger") or {}).get("wake_kind") != wake_kind:
                continue
            if routing_payload.get("route_kind") != BrainCommitmentWakeRouteKind.KEEP_WAITING.value:
                continue
            event_ts = _parse_ts(event.ts)
            if event_ts is None:
                return False
            return (now - event_ts) < timedelta(seconds=cooldown_seconds)
        return False

    @staticmethod
    def _wake_already_seen_for_source_event(
        *,
        wake_events: list[BrainEventRecord],
        commitment_id: str,
        wake_kind: str,
        source_event_id: str,
    ) -> bool:
        for event in wake_events:
            payload = event.payload
            commitment_payload = payload.get("commitment", {})
            trigger_payload = payload.get("trigger", {})
            if commitment_payload.get("commitment_id") != commitment_id:
                continue
            if trigger_payload.get("wake_kind") != wake_kind:
                continue
            if trigger_payload.get("source_event_id") == source_event_id:
                return True
        return False

    @staticmethod
    def _latest_user_turn_event(
        *,
        recent_events: list[BrainEventRecord],
        source_event: BrainEventRecord | None,
    ) -> BrainEventRecord | None:
        if source_event is not None and source_event.event_type in {
            "user.turn.ended",
            "user.turn.transcribed",
            "user.turn.started",
        }:
            return source_event
        for event in recent_events:
            if event.event_type in {"user.turn.ended", "user.turn.transcribed", "user.turn.started"}:
                return event
        return None

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

    @staticmethod
    def _family_leader_key(decision: BrainCommitmentWakeDecision) -> tuple[datetime, str]:
        updated_at = _parse_ts(decision.commitment.updated_at) or datetime.min.replace(tzinfo=UTC)
        return (updated_at, decision.commitment.commitment_id)

    def _family_attention_key(
        self,
        decision: BrainCommitmentWakeDecision,
        *,
        wake_events: list[BrainEventRecord],
    ) -> tuple[int, datetime, datetime, str, str]:
        last_attention = self._last_family_attention(
            goal_family=decision.commitment.goal_family,
            wake_events=wake_events,
        )
        updated_at = _parse_ts(decision.commitment.updated_at) or datetime.min.replace(tzinfo=UTC)
        return (
            1 if last_attention is not None else 0,
            last_attention or datetime.min.replace(tzinfo=UTC),
            updated_at,
            decision.commitment.commitment_id,
            decision.commitment.goal_family,
        )

    @staticmethod
    def _last_family_attention(
        *,
        goal_family: str,
        wake_events: list[BrainEventRecord],
    ) -> datetime | None:
        for event in wake_events:
            commitment_payload = event.payload.get("commitment", {})
            if commitment_payload.get("goal_family") != goal_family:
                continue
            return _parse_ts(event.ts)
        return None


__all__ = [
    "BrainCommitmentWakeRouter",
    "BrainCommitmentWakeRouterPolicy",
    "BrainCommitmentWakeRouterResult",
]

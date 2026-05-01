"""Typed hierarchical embodied coordinator above low-level robot-head execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.context import BrainContextTask
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainEmbodiedActionEnvelope,
    BrainEmbodiedDispatchDisposition,
    BrainEmbodiedExecutionTrace,
    BrainEmbodiedExecutiveProjection,
    BrainEmbodiedExecutorKind,
    BrainEmbodiedIntent,
    BrainEmbodiedIntentKind,
    BrainEmbodiedRecoveryRecord,
    BrainEmbodiedTraceStatus,
    BrainWakeCondition,
    BrainWakeConditionKind,
)

if TYPE_CHECKING:
    from blink.brain._executive import BrainExecutivePolicyFrame
    from blink.brain.actions import EmbodiedActionEngine
    from blink.brain.context_surfaces import BrainContextSurfaceBuilder
    from blink.brain.projections import BrainCommitmentRecord, BrainGoal, BrainGoalStep

_EMBODIED_EVENT_TYPES = frozenset(
    {
        BrainEventType.EMBODIED_INTENT_SELECTED,
        BrainEventType.EMBODIED_DISPATCH_PREPARED,
        BrainEventType.EMBODIED_DISPATCH_COMPLETED,
        BrainEventType.EMBODIED_DISPATCH_DEFERRED,
        BrainEventType.EMBODIED_RECOVERY_RECORDED,
    }
)
_RECOVERY_ACTION_ID = "cmd_return_neutral"
_MACHINE_CLEARABLE_BLOCKERS = frozenset(
    {
        "robot_head_busy",
        "robot_head_unavailable",
        "robot_head_unarmed",
        "robot_head_status_unavailable",
    }
)
_EMBODIED_TRACE_STATUS_RANK = {
    BrainEmbodiedTraceStatus.PREPARED.value: 0,
    BrainEmbodiedTraceStatus.DISPATCHED.value: 1,
    BrainEmbodiedTraceStatus.SUCCEEDED.value: 2,
    BrainEmbodiedTraceStatus.FAILED.value: 2,
    BrainEmbodiedTraceStatus.DEFERRED.value: 2,
    BrainEmbodiedTraceStatus.ABORTED.value: 2,
    BrainEmbodiedTraceStatus.REPAIRED.value: 3,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _stable_id(prefix: str, payload: str) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, payload).hex}"


def _trace_status_rank(status: str) -> int:
    return _EMBODIED_TRACE_STATUS_RANK.get(str(status).strip(), 0)


def _merge_embodied_execution_traces(
    existing: BrainEmbodiedExecutionTrace,
    incoming: BrainEmbodiedExecutionTrace,
) -> BrainEmbodiedExecutionTrace:
    """Merge duplicate trace updates without regressing lifecycle state."""
    existing_rank = _trace_status_rank(existing.status)
    incoming_rank = _trace_status_rank(incoming.status)
    if incoming_rank > existing_rank:
        primary, secondary = incoming, existing
    elif existing_rank > incoming_rank:
        primary, secondary = existing, incoming
    elif incoming.updated_at >= existing.updated_at:
        primary, secondary = incoming, existing
    else:
        primary, secondary = existing, incoming
    return BrainEmbodiedExecutionTrace.from_dict(
        {
            **secondary.as_dict(),
            **primary.as_dict(),
            "capability_request_event_id": primary.capability_request_event_id
            or secondary.capability_request_event_id,
            "capability_result_event_id": primary.capability_result_event_id
            or secondary.capability_result_event_id,
            "robot_action_event_id": primary.robot_action_event_id or secondary.robot_action_event_id,
            "disposition": primary.disposition or secondary.disposition,
            "status": primary.status or secondary.status,
            "outcome_summary": primary.outcome_summary or secondary.outcome_summary,
            "mismatch_codes": _sorted_unique(
                [*secondary.mismatch_codes, *primary.mismatch_codes]
            ),
            "repair_codes": _sorted_unique(
                [*secondary.repair_codes, *primary.repair_codes]
            ),
            "recovery_action_id": primary.recovery_action_id or secondary.recovery_action_id,
            "prepared_at": primary.prepared_at or secondary.prepared_at,
            "completed_at": primary.completed_at or secondary.completed_at,
            "updated_at": max(existing.updated_at, incoming.updated_at),
            "details": {**secondary.details, **primary.details},
        }
    ) or primary


def embodied_event_types() -> frozenset[str]:
    """Return the explicit embodied lifecycle event set."""
    return _EMBODIED_EVENT_TYPES


def is_embodied_event_type(event_type: str) -> bool:
    """Return whether one event type belongs to the embodied lifecycle."""
    return event_type in _EMBODIED_EVENT_TYPES


def append_embodied_intent(
    projection: BrainEmbodiedExecutiveProjection,
    intent: BrainEmbodiedIntent,
) -> None:
    """Replace the current embodied intent."""
    projection.current_intent = intent
    projection.updated_at = intent.updated_at
    projection.sync_lists()


def append_embodied_action_envelope(
    projection: BrainEmbodiedExecutiveProjection,
    envelope: BrainEmbodiedActionEnvelope,
    *,
    max_recent_envelopes: int = 24,
) -> None:
    """Append one embodied action envelope."""
    projection.recent_action_envelopes = [
        record
        for record in projection.recent_action_envelopes
        if record.envelope_id != envelope.envelope_id
    ]
    projection.recent_action_envelopes.append(envelope)
    projection.recent_action_envelopes = sorted(
        projection.recent_action_envelopes,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_envelopes]
    projection.updated_at = envelope.updated_at
    projection.sync_lists()


def append_embodied_execution_trace(
    projection: BrainEmbodiedExecutiveProjection,
    trace: BrainEmbodiedExecutionTrace,
    *,
    max_recent_traces: int = 24,
) -> None:
    """Append one embodied execution trace."""
    existing = next(
        (record for record in projection.recent_execution_traces if record.trace_id == trace.trace_id),
        None,
    )
    if existing is not None:
        trace = _merge_embodied_execution_traces(existing, trace)
    projection.recent_execution_traces = [
        record for record in projection.recent_execution_traces if record.trace_id != trace.trace_id
    ]
    projection.recent_execution_traces.append(trace)
    projection.recent_execution_traces = sorted(
        projection.recent_execution_traces,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_traces]
    projection.updated_at = trace.updated_at
    projection.sync_lists()


def append_embodied_recovery(
    projection: BrainEmbodiedExecutiveProjection,
    recovery: BrainEmbodiedRecoveryRecord,
    *,
    max_recent_recoveries: int = 24,
) -> None:
    """Append one embodied recovery record."""
    projection.recent_recoveries = [
        record for record in projection.recent_recoveries if record.recovery_id != recovery.recovery_id
    ]
    projection.recent_recoveries.append(recovery)
    projection.recent_recoveries = sorted(
        projection.recent_recoveries,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_recoveries]
    projection.updated_at = recovery.updated_at
    projection.sync_lists()


@dataclass(frozen=True)
class BrainEmbodiedDispatchDecision:
    """Prepared embodied intent, envelope, and trace for one goal step."""

    intent: BrainEmbodiedIntent
    envelope: BrainEmbodiedActionEnvelope
    trace: BrainEmbodiedExecutionTrace
    disposition: str
    blocked_reason: BrainBlockedReason | None = None
    wake_conditions: tuple[BrainWakeCondition, ...] = ()

    @property
    def should_dispatch(self) -> bool:
        """Return whether the coordinator approved low-level dispatch."""
        return self.disposition == BrainEmbodiedDispatchDisposition.DISPATCH.value


class BrainHierarchicalEmbodiedCoordinator:
    """Select typed embodied intent and trace goal-backed robot-head execution."""

    def __init__(
        self,
        *,
        store: Any,
        session_resolver: Any,
        presence_scope_key: str,
        context_surface_builder: BrainContextSurfaceBuilder,
        action_engine: EmbodiedActionEngine,
    ):
        """Initialize the embodied coordinator."""
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._context_surface_builder = context_surface_builder
        self._action_engine = action_engine

    @property
    def action_engine(self) -> EmbodiedActionEngine:
        """Expose the bounded action engine used by the coordinator."""
        return self._action_engine

    async def prepare_dispatch(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        step: BrainGoalStep,
        step_index: int,
        plan_proposal_id: str | None,
        rehearsal_details: dict[str, Any],
        executive_policy: BrainExecutivePolicyFrame,
    ) -> BrainEmbodiedDispatchDecision:
        """Select one embodied intent and bind it to a low-level action envelope."""
        session_ids = self._session_resolver()
        from blink.brain.actions import action_id_for_capability

        action_id = action_id_for_capability(step.capability_id)
        action_definition = self._action_engine.library.get(action_id)
        surface = self._context_surface_builder.build(
            latest_user_text=goal.title,
            task=BrainContextTask.PLANNING,
            include_historical_claims=True,
        )
        status_result = await self._action_engine.status()
        status = status_result.status
        rehearsal_risk_codes = _sorted_unique(rehearsal_details.get("risk_codes", []) or [])
        supporting_prediction_ids = _sorted_unique(
            [
                record.prediction_id
                for record in surface.predictive_world_model.active_predictions[:6]
                if record.action_id == action_id
                or record.subject_id in {
                    action_id,
                    goal.goal_id,
                    plan_proposal_id or "",
                    commitment.commitment_id if commitment is not None else "",
                }
            ]
        )
        scene_uncertainty_codes = _sorted_unique(
            [
                code
                for code in rehearsal_risk_codes
                if code.startswith("scene_") or code.startswith("affordance_")
            ]
        )
        scene_uncertain = bool(scene_uncertainty_codes)
        reason_codes = _sorted_unique(
            [
                *executive_policy.reason_codes,
                *scene_uncertainty_codes,
                (
                    "scene_uncertain"
                    if scene_uncertain
                    else None
                ),
                str(rehearsal_details.get("decision_recommendation")).strip()
                if rehearsal_details
                else None,
            ]
        )

        disposition = BrainEmbodiedDispatchDisposition.DISPATCH.value
        intent_kind = BrainEmbodiedIntentKind.EXECUTE_ACTION.value
        blocked_reason: BrainBlockedReason | None = None
        wake_conditions: tuple[BrainWakeCondition, ...] = ()
        if executive_policy.action_posture == "suppress":
            disposition = BrainEmbodiedDispatchDisposition.ABORT.value
            intent_kind = BrainEmbodiedIntentKind.PREPARE_ACTION.value
            blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.OPERATOR_REVIEW.value,
                summary="Embodied coordinator aborted dispatch under the current executive policy.",
                details={"reason_codes": reason_codes},
            )
            wake_conditions = (
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume explicitly once the embodied policy block clears.",
                    details={"reason_codes": reason_codes},
                ),
            )
        elif scene_uncertain:
            disposition = BrainEmbodiedDispatchDisposition.DEFER.value
            intent_kind = BrainEmbodiedIntentKind.INSPECT_SCENE.value
            blocked_reason = BrainBlockedReason(
                kind=BrainBlockedReasonKind.EXPLICIT_DEFER.value,
                summary="Embodied coordinator deferred dispatch until the scene becomes clearer.",
                details={"reason_codes": reason_codes},
            )
            wake_conditions = (
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.CONDITION_CLEARED.value,
                    summary="Resume once scene uncertainty clears.",
                    details={"presence_scope_key": self._presence_scope_key},
                ),
            )

        selected_at = _utc_now()
        intent = BrainEmbodiedIntent(
            intent_id=_stable_id(
                "embodied_intent",
                f"blink:embodied:intent:{goal.goal_id}:{step_index}:{intent_kind}:{action_id}",
            ),
            intent_kind=intent_kind,
            goal_id=goal.goal_id,
            commitment_id=commitment.commitment_id if commitment is not None else None,
            plan_proposal_id=plan_proposal_id,
            step_index=step_index,
            selected_action_id=action_id,
            executor_kind=BrainEmbodiedExecutorKind.ROBOT_HEAD_CAPABILITY.value,
            policy_posture=executive_policy.action_posture,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_rehearsal_id=_optional_text(rehearsal_details.get("rehearsal_id")),
            reason_codes=reason_codes,
            status=(
                "selected"
                if disposition == BrainEmbodiedDispatchDisposition.DISPATCH.value
                else disposition
            ),
            summary=(
                f"{intent_kind} for {action_id}."
                if disposition == BrainEmbodiedDispatchDisposition.DISPATCH.value
                else blocked_reason.summary
                if blocked_reason is not None
                else f"{intent_kind} for {action_id}."
            ),
            selected_at=selected_at,
            updated_at=selected_at,
            details={
                "goal_title": goal.title,
                "commitment_title": commitment.title if commitment is not None else None,
                "scene_degraded_mode": surface.scene_world_state.degraded_mode,
                "scene_degraded_reason_codes": list(surface.scene_world_state.degraded_reason_codes),
                "rehearsal_risk_codes": rehearsal_risk_codes,
                "procedural_active_skill_ids": list(
                    (surface.procedural_skills.active_skill_ids if surface.procedural_skills else [])
                )[:4],
            },
        )
        envelope = BrainEmbodiedActionEnvelope(
            envelope_id=_stable_id(
                "embodied_envelope",
                f"blink:embodied:envelope:{intent.intent_id}:{step.capability_id}:{action_id}",
            ),
            intent_id=intent.intent_id,
            goal_id=goal.goal_id,
            plan_proposal_id=plan_proposal_id,
            step_index=step_index,
            capability_id=step.capability_id,
            action_id=action_id,
            arguments=dict(step.arguments),
            dispatch_source=str(goal.source or "executive"),
            executor_backend=str(
                status.mode if status is not None else self._action_engine.execution_backend
            ),
            preview_only=bool(status_result.preview_only),
            rehearsal_id=_optional_text(rehearsal_details.get("rehearsal_id")),
            policy_snapshot=executive_policy.as_dict(),
            reason_codes=reason_codes,
            summary=f"Prepared {action_id} via {step.capability_id}.",
            prepared_at=selected_at,
            updated_at=selected_at,
            details={
                "allowed_sources": list(action_definition.allowed_sources),
                "action_description": action_definition.description,
                "recent_low_level_action_ids": [
                    record.action_id
                    for record in self._store.recent_action_events(
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        limit=6,
                    )
                ],
            },
        )
        trace_status = (
            BrainEmbodiedTraceStatus.PREPARED.value
            if disposition == BrainEmbodiedDispatchDisposition.DISPATCH.value
            else BrainEmbodiedTraceStatus.DEFERRED.value
            if disposition == BrainEmbodiedDispatchDisposition.DEFER.value
            else BrainEmbodiedTraceStatus.ABORTED.value
        )
        trace = BrainEmbodiedExecutionTrace(
            trace_id=_stable_id(
                "embodied_trace",
                f"blink:embodied:trace:{intent.intent_id}:{envelope.envelope_id}",
            ),
            intent_id=intent.intent_id,
            envelope_id=envelope.envelope_id,
            goal_id=goal.goal_id,
            step_index=step_index,
            disposition=disposition,
            status=trace_status,
            outcome_summary=(
                "Embodied coordinator prepared the action for low-level dispatch."
                if disposition == BrainEmbodiedDispatchDisposition.DISPATCH.value
                else blocked_reason.summary
                if blocked_reason is not None
                else "Embodied coordinator deferred the action."
            ),
            mismatch_codes=reason_codes if disposition != BrainEmbodiedDispatchDisposition.DISPATCH.value else [],
            prepared_at=selected_at,
            updated_at=selected_at,
            details={
                "goal_title": goal.title,
                "presence_scope_key": self._presence_scope_key,
                "executor_backend": envelope.executor_backend,
                "supporting_prediction_ids": supporting_prediction_ids,
                "supporting_rehearsal_id": _optional_text(rehearsal_details.get("rehearsal_id")),
            },
        )

        intent_event = self._append_event(
            event_type=BrainEventType.EMBODIED_INTENT_SELECTED,
            payload={
                "intent": intent.as_dict(),
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=goal.goal_id,
        )
        self._append_event(
            event_type=BrainEventType.EMBODIED_DISPATCH_PREPARED,
            payload={
                "intent": intent.as_dict(),
                "envelope": envelope.as_dict(),
                "execution_trace": trace.as_dict(),
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=goal.goal_id,
            causal_parent_id=intent_event.event_id,
        )
        if disposition != BrainEmbodiedDispatchDisposition.DISPATCH.value:
            self._append_event(
                event_type=BrainEventType.EMBODIED_DISPATCH_DEFERRED,
                payload={
                    "intent": intent.as_dict(),
                    "envelope": envelope.as_dict(),
                    "execution_trace": trace.as_dict(),
                    "presence_scope_key": self._presence_scope_key,
                },
                correlation_id=goal.goal_id,
                causal_parent_id=intent_event.event_id,
            )
        return BrainEmbodiedDispatchDecision(
            intent=intent,
            envelope=envelope,
            trace=trace,
            disposition=disposition,
            blocked_reason=blocked_reason,
            wake_conditions=wake_conditions,
        )

    def record_dispatch_completion(
        self,
        *,
        intent: BrainEmbodiedIntent,
        envelope: BrainEmbodiedActionEnvelope,
        prepared_trace: BrainEmbodiedExecutionTrace,
        request_event: BrainEventRecord,
        terminal_event: BrainEventRecord,
        execution: Any,
    ) -> BrainEmbodiedExecutionTrace:
        """Append one completed embodied execution trace."""
        payload = terminal_event.payload if isinstance(terminal_event.payload, dict) else {}
        updated_at = terminal_event.ts
        trace = BrainEmbodiedExecutionTrace(
            trace_id=prepared_trace.trace_id,
            intent_id=intent.intent_id,
            envelope_id=envelope.envelope_id,
            goal_id=intent.goal_id,
            step_index=intent.step_index,
            capability_request_event_id=request_event.event_id,
            capability_result_event_id=terminal_event.event_id,
            robot_action_event_id=_optional_text(payload.get("robot_action_event_id")),
            disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
            status=(
                BrainEmbodiedTraceStatus.SUCCEEDED.value
                if execution.accepted
                else BrainEmbodiedTraceStatus.FAILED.value
            ),
            outcome_summary=str(execution.summary or ""),
            mismatch_codes=_sorted_unique(
                [execution.error_code, *(execution.warnings or ())]
            ),
            repair_codes=list(prepared_trace.repair_codes),
            recovery_action_id=prepared_trace.recovery_action_id,
            prepared_at=prepared_trace.prepared_at,
            completed_at=updated_at,
            updated_at=updated_at,
            details={
                **dict(prepared_trace.details),
                "execution_output": dict(execution.output),
                "execution_outcome": execution.outcome,
                "execution_error_code": execution.error_code,
            },
        )
        updated_intent = BrainEmbodiedIntent.from_dict(
            {
                **intent.as_dict(),
                "status": trace.status,
                "summary": trace.outcome_summary,
                "updated_at": updated_at,
            }
        ) or intent
        self._append_event(
            event_type=BrainEventType.EMBODIED_DISPATCH_COMPLETED,
            payload={
                "intent": updated_intent.as_dict(),
                "envelope": envelope.as_dict(),
                "execution_trace": trace.as_dict(),
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=intent.goal_id,
            causal_parent_id=terminal_event.event_id,
        )
        return trace

    def record_recovery(
        self,
        *,
        intent: BrainEmbodiedIntent,
        trace: BrainEmbodiedExecutionTrace,
        execution: Any,
    ) -> BrainEmbodiedRecoveryRecord | None:
        """Record one bounded safe recovery recommendation after a failed trace."""
        if trace.status != BrainEmbodiedTraceStatus.FAILED.value:
            return None
        if intent.selected_action_id == _RECOVERY_ACTION_ID:
            return None
        recovery_reason_codes = _sorted_unique(
            [
                execution.error_code,
                "safe_recovery_recommended",
                *trace.mismatch_codes,
            ]
        )
        if not recovery_reason_codes:
            return None
        recovery_at = _utc_now()
        recovery = BrainEmbodiedRecoveryRecord(
            recovery_id=_stable_id(
                "embodied_recovery",
                f"blink:embodied:recovery:{trace.trace_id}:{_RECOVERY_ACTION_ID}",
            ),
            trace_id=trace.trace_id,
            intent_id=intent.intent_id,
            action_id=_RECOVERY_ACTION_ID,
            reason_codes=recovery_reason_codes,
            status=(
                "recommended"
                if execution.error_code in _MACHINE_CLEARABLE_BLOCKERS
                else "recorded"
            ),
            summary=(
                f"Recommend {_RECOVERY_ACTION_ID} after failed {intent.selected_action_id} dispatch."
            ),
            recorded_at=recovery_at,
            updated_at=recovery_at,
            details={"failed_action_id": intent.selected_action_id},
        )
        updated_trace = BrainEmbodiedExecutionTrace.from_dict(
            {
                **trace.as_dict(),
                "repair_codes": recovery_reason_codes,
                "recovery_action_id": _RECOVERY_ACTION_ID,
                "updated_at": recovery_at,
            }
        ) or trace
        recovery_intent = BrainEmbodiedIntent.from_dict(
            {
                **intent.as_dict(),
                "intent_kind": BrainEmbodiedIntentKind.RECOVER_SAFE_STATE.value,
                "status": BrainEmbodiedTraceStatus.REPAIRED.value,
                "summary": recovery.summary,
                "selected_action_id": _RECOVERY_ACTION_ID,
                "updated_at": recovery_at,
            }
        ) or intent
        self._append_event(
            event_type=BrainEventType.EMBODIED_RECOVERY_RECORDED,
            payload={
                "intent": recovery_intent.as_dict(),
                "execution_trace": updated_trace.as_dict(),
                "recovery": recovery.as_dict(),
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=intent.goal_id,
            causal_parent_id=trace.capability_result_event_id,
        )
        return recovery

    def _append_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        correlation_id: str | None,
        causal_parent_id: str | None = None,
    ) -> BrainEventRecord:
        session_ids = self._session_resolver()
        return self._store.append_brain_event(
            event_type=event_type,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="embodied_executive",
            payload=payload,
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
        )


__all__ = [
    "BrainEmbodiedDispatchDecision",
    "BrainHierarchicalEmbodiedCoordinator",
    "append_embodied_action_envelope",
    "append_embodied_execution_trace",
    "append_embodied_intent",
    "append_embodied_recovery",
    "embodied_event_types",
    "is_embodied_event_type",
]

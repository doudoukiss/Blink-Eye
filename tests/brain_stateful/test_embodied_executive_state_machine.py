from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from blink.brain.embodied_executive import (
    append_embodied_action_envelope,
    append_embodied_execution_trace,
    append_embodied_intent,
    append_embodied_recovery,
)
from blink.brain.projections import (
    BrainEmbodiedActionEnvelope,
    BrainEmbodiedDispatchDisposition,
    BrainEmbodiedExecutionTrace,
    BrainEmbodiedExecutiveProjection,
    BrainEmbodiedExecutorKind,
    BrainEmbodiedIntent,
    BrainEmbodiedIntentKind,
    BrainEmbodiedRecoveryRecord,
    BrainEmbodiedTraceStatus,
)

pytestmark = pytest.mark.brain_stateful


class EmbodiedExecutiveStateMachine(RuleBasedStateMachine):
    """Exercise bounded embodied intent, trace, and recovery lifecycles."""

    intents = Bundle("intents")
    prepared = Bundle("prepared")
    failed = Bundle("failed")

    def __init__(self):
        super().__init__()
        self.projection = BrainEmbodiedExecutiveProjection(
            scope_key="thread-1",
            presence_scope_key="browser:presence",
        )
        self._counter = 0

    @rule(target=intents)
    def select_intent(self):
        index = self._counter
        self._counter += 1
        intent_id = f"intent-{index}"
        append_embodied_intent(
            self.projection,
            BrainEmbodiedIntent(
                intent_id=intent_id,
                intent_kind=BrainEmbodiedIntentKind.EXECUTE_ACTION.value,
                goal_id=f"goal-{index}",
                commitment_id=f"commitment-{index}",
                plan_proposal_id=f"proposal-{index}",
                step_index=index,
                selected_action_id="cmd_look_left",
                executor_kind=BrainEmbodiedExecutorKind.ROBOT_HEAD_CAPABILITY.value,
                policy_posture="allow",
                selected_at=f"2026-01-01T00:00:{index:02d}+00:00",
                updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
            ),
        )
        return intent_id

    @rule(target=prepared, intent_id=intents)
    def prepare_trace(self, intent_id: str):
        suffix = intent_id.split("-", maxsplit=1)[-1]
        append_embodied_action_envelope(
            self.projection,
            BrainEmbodiedActionEnvelope(
                envelope_id=f"envelope-{suffix}",
                intent_id=intent_id,
                goal_id=f"goal-{suffix}",
                plan_proposal_id=f"proposal-{suffix}",
                step_index=int(suffix),
                capability_id="robot_head.look_left",
                action_id="cmd_look_left",
                dispatch_source="operator",
                executor_backend="simulation",
                prepared_at=f"2026-01-01T00:00:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:00:{int(suffix):02d}+00:00",
            ),
        )
        append_embodied_execution_trace(
            self.projection,
            BrainEmbodiedExecutionTrace(
                trace_id=f"trace-{suffix}",
                intent_id=intent_id,
                envelope_id=f"envelope-{suffix}",
                goal_id=f"goal-{suffix}",
                step_index=int(suffix),
                disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
                status=BrainEmbodiedTraceStatus.PREPARED.value,
                prepared_at=f"2026-01-01T00:00:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:00:{int(suffix):02d}+00:00",
            ),
        )
        return f"trace-{suffix}"

    @rule(trace_id=prepared)
    def complete_success(self, trace_id: str):
        suffix = trace_id.split("-", maxsplit=1)[-1]
        current = next(
            (record for record in self.projection.recent_execution_traces if record.trace_id == trace_id),
            None,
        )
        if current is not None and current.status in {
            BrainEmbodiedTraceStatus.FAILED.value,
            BrainEmbodiedTraceStatus.SUCCEEDED.value,
            BrainEmbodiedTraceStatus.ABORTED.value,
        }:
            return
        append_embodied_execution_trace(
            self.projection,
            BrainEmbodiedExecutionTrace(
                trace_id=trace_id,
                intent_id=f"intent-{suffix}",
                envelope_id=f"envelope-{suffix}",
                goal_id=f"goal-{suffix}",
                step_index=int(suffix),
                capability_request_event_id=f"evt-request-{suffix}",
                capability_result_event_id=f"evt-result-{suffix}",
                robot_action_event_id=f"evt-robot-{suffix}",
                disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
                status=BrainEmbodiedTraceStatus.SUCCEEDED.value,
                completed_at=f"2026-01-01T00:01:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:01:{int(suffix):02d}+00:00",
            ),
        )

    @rule(target=failed, trace_id=prepared)
    def complete_failure(self, trace_id: str):
        suffix = trace_id.split("-", maxsplit=1)[-1]
        current = next(
            (record for record in self.projection.recent_execution_traces if record.trace_id == trace_id),
            None,
        )
        if current is not None and current.status in {
            BrainEmbodiedTraceStatus.FAILED.value,
            BrainEmbodiedTraceStatus.SUCCEEDED.value,
            BrainEmbodiedTraceStatus.ABORTED.value,
        }:
            return trace_id
        append_embodied_execution_trace(
            self.projection,
            BrainEmbodiedExecutionTrace(
                trace_id=trace_id,
                intent_id=f"intent-{suffix}",
                envelope_id=f"envelope-{suffix}",
                goal_id=f"goal-{suffix}",
                step_index=int(suffix),
                capability_request_event_id=f"evt-request-{suffix}",
                capability_result_event_id=f"evt-result-{suffix}",
                disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
                status=BrainEmbodiedTraceStatus.FAILED.value,
                mismatch_codes=["robot_head_busy"],
                completed_at=f"2026-01-01T00:01:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:01:{int(suffix):02d}+00:00",
            ),
        )
        return trace_id

    @rule(trace_id=failed)
    def record_recovery(self, trace_id: str):
        suffix = trace_id.split("-", maxsplit=1)[-1]
        current = next(
            (record for record in self.projection.recent_execution_traces if record.trace_id == trace_id),
            None,
        )
        if current is None or current.status != BrainEmbodiedTraceStatus.FAILED.value:
            return
        append_embodied_execution_trace(
            self.projection,
            BrainEmbodiedExecutionTrace(
                trace_id=trace_id,
                intent_id=f"intent-{suffix}",
                envelope_id=f"envelope-{suffix}",
                goal_id=f"goal-{suffix}",
                step_index=int(suffix),
                disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
                status=BrainEmbodiedTraceStatus.FAILED.value,
                repair_codes=["safe_recovery_recommended"],
                recovery_action_id="cmd_return_neutral",
                completed_at=f"2026-01-01T00:02:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:02:{int(suffix):02d}+00:00",
            ),
        )
        append_embodied_recovery(
            self.projection,
            BrainEmbodiedRecoveryRecord(
                recovery_id=f"recovery-{suffix}",
                trace_id=trace_id,
                intent_id=f"intent-{suffix}",
                action_id="cmd_return_neutral",
                reason_codes=["safe_recovery_recommended"],
                recorded_at=f"2026-01-01T00:02:{int(suffix):02d}+00:00",
                updated_at=f"2026-01-01T00:02:{int(suffix):02d}+00:00",
            ),
        )

    @invariant()
    def recent_ids_stay_unique(self):
        assert len(set(self.projection.recent_trace_ids)) == len(self.projection.recent_trace_ids)
        assert len(set(self.projection.recent_envelope_ids)) == len(
            self.projection.recent_envelope_ids
        )
        assert len(set(self.projection.recent_recovery_ids)) == len(
            self.projection.recent_recovery_ids
        )

    @invariant()
    def recoveries_point_back_to_traces(self):
        traces_by_id = {
            record.trace_id: record for record in self.projection.recent_execution_traces
        }
        for recovery in self.projection.recent_recoveries:
            assert recovery.trace_id in traces_by_id
            assert traces_by_id[recovery.trace_id].recovery_action_id == recovery.action_id


TestEmbodiedExecutiveStateMachine = EmbodiedExecutiveStateMachine.TestCase
TestEmbodiedExecutiveStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)

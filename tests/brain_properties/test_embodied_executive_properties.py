from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.embodied_executive import (
    append_embodied_action_envelope,
    append_embodied_execution_trace,
    append_embodied_intent,
    append_embodied_recovery,
)
from blink.brain.embodied_executive_digest import build_embodied_executive_digest
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

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
_TRACE_STATUS_RANK = {
    BrainEmbodiedTraceStatus.PREPARED.value: 0,
    BrainEmbodiedTraceStatus.DISPATCHED.value: 1,
    BrainEmbodiedTraceStatus.SUCCEEDED.value: 2,
    BrainEmbodiedTraceStatus.FAILED.value: 2,
    BrainEmbodiedTraceStatus.DEFERRED.value: 2,
    BrainEmbodiedTraceStatus.ABORTED.value: 2,
    BrainEmbodiedTraceStatus.REPAIRED.value: 3,
}


def _intent(index: int) -> BrainEmbodiedIntent:
    return BrainEmbodiedIntent(
        intent_id=f"intent-{index}",
        intent_kind=BrainEmbodiedIntentKind.EXECUTE_ACTION.value,
        goal_id=f"goal-{index}",
        commitment_id=f"commitment-{index}",
        plan_proposal_id=f"proposal-{index}",
        step_index=index,
        selected_action_id="cmd_look_left",
        executor_kind=BrainEmbodiedExecutorKind.ROBOT_HEAD_CAPABILITY.value,
        policy_posture="allow",
        reason_codes=["bounded_dispatch"],
        selected_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _envelope(index: int) -> BrainEmbodiedActionEnvelope:
    return BrainEmbodiedActionEnvelope(
        envelope_id=f"envelope-{index}",
        intent_id=f"intent-{index}",
        goal_id=f"goal-{index}",
        plan_proposal_id=f"proposal-{index}",
        step_index=index,
        capability_id="robot_head.look_left",
        action_id="cmd_look_left",
        dispatch_source="operator",
        executor_backend="simulation",
        policy_snapshot={"action_posture": "allow"},
        prepared_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _trace(index: int, *, status: str) -> BrainEmbodiedExecutionTrace:
    return BrainEmbodiedExecutionTrace(
        trace_id=f"trace-{index}",
        intent_id=f"intent-{index}",
        envelope_id=f"envelope-{index}",
        goal_id=f"goal-{index}",
        step_index=index,
        disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
        status=status,
        outcome_summary=f"Trace {index} -> {status}",
        prepared_at=f"2026-01-01T00:00:{index:02d}+00:00",
        completed_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _recovery(index: int) -> BrainEmbodiedRecoveryRecord:
    return BrainEmbodiedRecoveryRecord(
        recovery_id=f"recovery-{index}",
        trace_id=f"trace-{index}",
        intent_id=f"intent-{index}",
        action_id="cmd_return_neutral",
        reason_codes=["safe_recovery_recommended"],
        recorded_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


@given(
    statuses=st.lists(
        st.sampled_from(
            [
                BrainEmbodiedTraceStatus.PREPARED.value,
                BrainEmbodiedTraceStatus.DISPATCHED.value,
                BrainEmbodiedTraceStatus.SUCCEEDED.value,
                BrainEmbodiedTraceStatus.FAILED.value,
                BrainEmbodiedTraceStatus.DEFERRED.value,
                BrainEmbodiedTraceStatus.ABORTED.value,
            ]
        ),
        min_size=1,
        max_size=6,
    ),
    low_level_sources=st.lists(
        st.sampled_from(["policy", "operator", "tool"]),
        min_size=1,
        max_size=6,
    ),
)
@_SETTINGS
def test_embodied_digest_counts_match_projection_rows(statuses, low_level_sources):
    projection = BrainEmbodiedExecutiveProjection(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
    )

    for index, status in enumerate(statuses):
        append_embodied_intent(projection, _intent(index))
        append_embodied_action_envelope(projection, _envelope(index))
        append_embodied_execution_trace(projection, _trace(index, status=status))
        if status == BrainEmbodiedTraceStatus.FAILED.value:
            append_embodied_recovery(projection, _recovery(index))

    low_level_actions = [
        {
            "action_id": "cmd_look_left",
            "source": source,
            "accepted": True,
            "preview_only": False,
            "summary": f"{source} action",
            "created_at": f"2026-01-01T00:00:{index:02d}+00:00",
        }
        for index, source in enumerate(low_level_sources)
    ]
    digest = build_embodied_executive_digest(
        embodied_executive=projection.as_dict(),
        recent_action_events=low_level_actions,
    )

    assert projection.current_intent is not None
    assert projection.current_intent.intent_id == f"intent-{len(statuses) - 1}"
    assert len(set(projection.recent_trace_ids)) == len(projection.recent_trace_ids)
    assert len(set(projection.recent_envelope_ids)) == len(projection.recent_envelope_ids)
    assert sum(digest["trace_status_counts"].values()) == len(projection.recent_execution_traces)
    assert sum(digest["low_level_source_counts"].values()) == len(low_level_actions)


@given(
    statuses=st.lists(
        st.sampled_from(
            [
                BrainEmbodiedTraceStatus.PREPARED.value,
                BrainEmbodiedTraceStatus.DISPATCHED.value,
                BrainEmbodiedTraceStatus.SUCCEEDED.value,
                BrainEmbodiedTraceStatus.FAILED.value,
            ]
        ),
        min_size=1,
        max_size=6,
    )
)
@_SETTINGS
def test_embodied_projection_replaces_duplicate_trace_ids_monotonically(statuses):
    projection = BrainEmbodiedExecutiveProjection(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
    )
    append_embodied_intent(projection, _intent(0))
    append_embodied_action_envelope(projection, _envelope(0))

    for index, status in enumerate(statuses):
        append_embodied_execution_trace(
            projection,
            BrainEmbodiedExecutionTrace(
                trace_id="trace-stable",
                intent_id="intent-0",
                envelope_id="envelope-0",
                goal_id="goal-0",
                step_index=0,
                disposition=BrainEmbodiedDispatchDisposition.DISPATCH.value,
                status=status,
                outcome_summary=f"update-{index}",
                prepared_at="2026-01-01T00:00:00+00:00",
                completed_at=f"2026-01-01T00:00:{index:02d}+00:00",
                updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
            ),
        )

    expected_status = statuses[0]
    expected_rank = _TRACE_STATUS_RANK[expected_status]
    for status in statuses[1:]:
        rank = _TRACE_STATUS_RANK[status]
        if rank >= expected_rank:
            expected_status = status
            expected_rank = rank

    assert projection.recent_trace_ids == ["trace-stable"]
    assert len(projection.recent_execution_traces) == 1
    assert projection.recent_execution_traces[0].status == expected_status

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from blink.brain.projections import (
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryRecordState,
)
from tests.brain_properties._active_state_property_helpers import (
    ActiveStateSeedSpec,
    active_state_seed_strategy,
    backing_id_is_reachable,
    build_active_state_bundle,
    collect_active_state_reachable_ids,
    normalize_private_working_memory,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
_BUFFER_CAPS = {
    BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value: 4,
    BrainPrivateWorkingMemoryBufferKind.SELF_POLICY.value: 4,
    BrainPrivateWorkingMemoryBufferKind.GOAL_COMMITMENT.value: 6,
    BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value: 6,
    BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value: 4,
    BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value: 6,
    BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value: 6,
}


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_equivalent_recent_event_streams_rebuild_identical_private_working_memory(spec):
    # Normalized-equivalent recent-event streams must rebuild the same private working-memory state.
    deterministic_spec = ActiveStateSeedSpec(
        zone_count=min(spec.zone_count, 3),
        entity_count=min(spec.entity_count, 4),
        fresh_for_secs=spec.fresh_for_secs,
        expire_after_secs=spec.expire_after_secs,
        plan_resolution=spec.plan_resolution,
        extra_tool_outcomes=spec.extra_tool_outcomes,
        degraded_mode=spec.degraded_mode,
        include_skill=spec.include_skill,
    )
    left = build_active_state_bundle(
        deterministic_spec,
        padded_text=False,
        reverse_entities=False,
        client_suffix="equiv",
    )
    right = build_active_state_bundle(
        deterministic_spec,
        padded_text=True,
        reverse_entities=True,
        client_suffix="equiv",
    )
    try:
        assert normalize_private_working_memory(
            left.build_private_working_memory()
        ) == normalize_private_working_memory(right.build_private_working_memory())
    finally:
        left.close()
        right.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_private_working_memory_stays_capped_resolves_closed_records_and_keeps_provenance_reachable(
    spec,
):
    # Caps, derived counts, closure state, and provenance links must remain bounded and traceable.
    bundle = build_active_state_bundle(spec, client_suffix="pwm")
    try:
        projection = bundle.build_private_working_memory()
        reachable = collect_active_state_reachable_ids(bundle)
        record_ids = {record.record_id for record in projection.records}

        expected_buffer_counts: dict[str, int] = {}
        expected_state_counts: dict[str, int] = {}
        expected_evidence_kind_counts: dict[str, int] = {}
        for record in projection.records:
            expected_buffer_counts[record.buffer_kind] = (
                expected_buffer_counts.get(record.buffer_kind, 0) + 1
            )
            expected_state_counts[record.state] = expected_state_counts.get(record.state, 0) + 1
            expected_evidence_kind_counts[record.evidence_kind] = (
                expected_evidence_kind_counts.get(record.evidence_kind, 0) + 1
            )

        assert projection.buffer_counts == dict(sorted(expected_buffer_counts.items()))
        assert projection.state_counts == dict(sorted(expected_state_counts.items()))
        assert projection.evidence_kind_counts == dict(sorted(expected_evidence_kind_counts.items()))
        assert set(projection.active_record_ids) <= record_ids
        assert set(projection.stale_record_ids) <= record_ids
        assert set(projection.resolved_record_ids) <= record_ids

        for buffer_kind, limit in _BUFFER_CAPS.items():
            assert projection.buffer_counts.get(buffer_kind, 0) <= limit

        tool_outcomes = [
            record
            for record in projection.records
            if record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value
        ]
        assert sum(
            record.state == BrainPrivateWorkingMemoryRecordState.ACTIVE.value
            for record in tool_outcomes
        ) <= 3
        if spec.extra_tool_outcomes > 3:
            assert any(
                record.state != BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                for record in tool_outcomes
            )

        if spec.plan_resolution != "pending":
            assert all(
                record.state != BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                for record in projection.records
                if record.plan_proposal_id == "proposal-primary"
                and record.buffer_kind
                in {
                    BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                    BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                }
            )

        unresolved_uncertainties = [
            record
            for record in projection.records
            if record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value
        ]
        for record in unresolved_uncertainties:
            assert record.record_id
            assert record.summary.strip()
            assert set(record.source_event_ids) <= reachable.event_ids
            for backing_id in record.backing_ids:
                assert backing_id_is_reachable(backing_id, reachable=reachable)

        for record in projection.records:
            assert set(record.source_event_ids) <= reachable.event_ids
            if record.goal_id is not None:
                assert record.goal_id in reachable.goal_ids
            if record.commitment_id is not None:
                assert record.commitment_id in reachable.commitment_ids
            if record.plan_proposal_id is not None:
                assert record.plan_proposal_id in reachable.plan_proposal_ids
            if record.skill_id is not None:
                assert record.skill_id in reachable.skill_ids
            for backing_id in record.backing_ids:
                assert backing_id_is_reachable(backing_id, reachable=reachable)
    finally:
        bundle.close()

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings

from blink.brain.projections import BrainActiveSituationRecordState
from tests.brain_properties._active_state_property_helpers import (
    ActiveStateSeedSpec,
    active_state_seed_strategy,
    backing_id_is_reachable,
    build_active_state_bundle,
    collect_active_state_reachable_ids,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
_KIND_CAPS = {
    "scene_state": 4,
    "world_state": 6,
    "goal_state": 4,
    "commitment_state": 6,
    "plan_state": 6,
    "procedural_state": 4,
    "uncertainty_state": 6,
}


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_equivalent_inputs_rebuild_stable_active_situation_output(spec):
    # Normalized-equivalent inputs must rebuild the same active-situation ordering and linkage.
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
        client_suffix="situation-equiv",
    )
    right = build_active_state_bundle(
        deterministic_spec,
        padded_text=True,
        reverse_entities=True,
        client_suffix="situation-equiv",
    )
    try:
        left_projection = left.build_active_situation_model()
        right_projection = right.build_active_situation_model()

        def _normalize_backing_id(value: str) -> str:
            if value.startswith("commitment_"):
                return "commitment"
            if value.startswith("scene_entity_"):
                return "scene_entity"
            if value.startswith("scene_affordance_"):
                return "scene_affordance"
            if value == "proposal-primary":
                return value
            if value.startswith("zone:") or value.startswith("scene:") or value.startswith("entity:"):
                return value
            return "dynamic"

        def _semantic_signature(projection) -> list[tuple[object, ...]]:
            return [
                (
                    record.record_kind,
                    record.state,
                    record.evidence_kind,
                    record.freshness,
                    tuple(record.uncertainty_codes),
                    tuple(sorted(_normalize_backing_id(item) for item in record.backing_ids)),
                    bool(record.goal_id),
                    bool(record.commitment_id),
                    bool(record.plan_proposal_id),
                    bool(record.skill_id),
                    len(record.private_record_ids),
                )
                for record in projection.records
            ]

        assert _semantic_signature(left_projection) == _semantic_signature(right_projection)
        assert left_projection.kind_counts == right_projection.kind_counts
        assert left_projection.state_counts == right_projection.state_counts
        assert left_projection.uncertainty_code_counts == right_projection.uncertainty_code_counts
        assert left_projection.linked_commitment_ids == right_projection.linked_commitment_ids
        assert left_projection.linked_plan_proposal_ids == right_projection.linked_plan_proposal_ids
        assert left_projection.linked_skill_ids == right_projection.linked_skill_ids
    finally:
        left.close()
        right.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_active_situation_links_stay_reachable_and_per_kind_caps_hold(spec):
    # Active-situation records must stay capped and every linked source id must be reachable.
    bundle = build_active_state_bundle(spec, client_suffix="situation-links")
    try:
        projection = bundle.build_active_situation_model()
        reachable = collect_active_state_reachable_ids(bundle)

        for record_kind, limit in _KIND_CAPS.items():
            assert projection.kind_counts.get(record_kind, 0) <= limit

        for record in projection.records:
            assert set(record.private_record_ids) <= reachable.private_record_ids
            assert set(record.source_event_ids) <= reachable.event_ids
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


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_degraded_scene_or_pending_plan_keeps_uncertainty_visible(spec):
    # Degraded scene inputs or pending plans must still surface visible uncertainty state.
    assume(spec.degraded_mode != "healthy" or spec.plan_resolution == "pending")
    bundle = build_active_state_bundle(spec, client_suffix="situation-uncertainty")
    try:
        projection = bundle.build_active_situation_model()
        unresolved = [
            record
            for record in projection.records
            if record.state == BrainActiveSituationRecordState.UNRESOLVED.value
        ]
        assert unresolved
        if spec.degraded_mode != "healthy":
            assert any(
                "scene_state" in record.backing_ids
                or "scene_world_state" in record.backing_ids
                for record in unresolved
            )
        if spec.plan_resolution == "pending":
            assert any(
                "proposal-primary" in record.backing_ids
                or "missing_input" in record.uncertainty_codes
                for record in projection.records
                if record.record_kind in {"uncertainty_state", "plan_state"}
            )
    finally:
        bundle.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_stale_or_unresolved_active_situation_records_do_not_silently_reactivate(spec):
    # Without matching source changes, stale or unresolved situation records must not reactivate.
    bundle = build_active_state_bundle(spec, client_suffix="situation-decay")
    try:
        current = bundle.build_active_situation_model(
            reference_second=bundle.scene_observed_second + spec.fresh_for_secs + 1
        )
        later = bundle.build_active_situation_model(
            reference_second=bundle.scene_observed_second + spec.expire_after_secs + 4
        )
        later_by_id = {record.record_id: record for record in later.records}

        for record in current.records:
            if record.state not in {
                BrainActiveSituationRecordState.STALE.value,
                BrainActiveSituationRecordState.UNRESOLVED.value,
            }:
                continue
            later_record = later_by_id.get(record.record_id)
            if later_record is not None:
                assert later_record.state != BrainActiveSituationRecordState.ACTIVE.value
    finally:
        bundle.close()

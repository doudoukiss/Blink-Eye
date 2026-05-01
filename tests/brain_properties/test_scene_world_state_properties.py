from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings

from blink.brain.projections import BrainSceneWorldEntityKind, BrainSceneWorldRecordState
from tests.brain_properties._active_state_property_helpers import (
    ActiveStateSeedSpec,
    active_state_seed_strategy,
    build_active_state_bundle,
    normalize_scene_world_state,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_equivalent_observations_rebuild_identical_scene_world_state(spec):
    # Normalized-equivalent scene observations must rebuild the same scene-world state.
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
        client_suffix="scene-equiv",
    )
    right = build_active_state_bundle(
        deterministic_spec,
        padded_text=True,
        reverse_entities=True,
        client_suffix="scene-equiv",
    )
    try:
        assert normalize_scene_world_state(
            left.build_scene_world_state()
        ) == normalize_scene_world_state(right.build_scene_world_state())
    finally:
        left.close()
        right.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_scene_world_state_stays_bounded_and_degraded_reasons_remain_coherent(spec):
    # Scene-world entities and affordances must stay bounded and keep reachable degraded-state semantics.
    bundle = build_active_state_bundle(spec, client_suffix="scene-bounds")
    try:
        projection = bundle.build_scene_world_state()
        entity_ids = {entity.entity_id for entity in projection.entities}
        zone_entities = [
            entity
            for entity in projection.entities
            if entity.entity_kind == BrainSceneWorldEntityKind.ZONE.value
        ]
        non_zone_entities = [
            entity
            for entity in projection.entities
            if entity.entity_kind != BrainSceneWorldEntityKind.ZONE.value
        ]

        assert len(zone_entities) <= 4
        assert len(non_zone_entities) <= 5
        assert len(projection.entities) <= 12
        assert len(projection.affordances) <= 12
        assert all(len(entity.affordance_ids) <= 2 for entity in non_zone_entities)
        assert all(affordance.entity_id in entity_ids for affordance in projection.affordances)

        if spec.degraded_mode == "healthy":
            assert projection.degraded_mode == "healthy"
            assert projection.degraded_reason_codes == []
        elif spec.degraded_mode == "limited":
            assert projection.degraded_mode == "limited"
            assert projection.degraded_reason_codes
        else:
            assert projection.degraded_mode == "unavailable"
            assert "camera_disconnected" in projection.degraded_reason_codes
    finally:
        bundle.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_scene_world_freshness_only_decays_without_newer_evidence(spec):
    # With no newer scene evidence, increasing the reference time may only decay freshness.
    bundle = build_active_state_bundle(spec, client_suffix="scene-decay")
    try:
        active = bundle.build_scene_world_state(reference_second=bundle.scene_observed_second + 1)
        stale = bundle.build_scene_world_state(
            reference_second=bundle.scene_observed_second + spec.fresh_for_secs + 1
        )
        expired = bundle.build_scene_world_state(
            reference_second=bundle.scene_observed_second + spec.expire_after_secs + 1
        )

        def _first_non_zone_state(projection) -> tuple[str, str]:
            for entity in projection.entities:
                if entity.entity_kind != BrainSceneWorldEntityKind.ZONE.value:
                    return entity.canonical_label, entity.state
            raise AssertionError("Missing non-zone entity.")

        active_label, active_state = _first_non_zone_state(active)
        stale_label, stale_state = _first_non_zone_state(stale)
        expired_label, expired_state = _first_non_zone_state(expired)
        assume(active_label == stale_label == expired_label)

        order = {
            BrainSceneWorldRecordState.ACTIVE.value: 0,
            BrainSceneWorldRecordState.STALE.value: 1,
            BrainSceneWorldRecordState.EXPIRED.value: 2,
        }
        if spec.degraded_mode == "healthy":
            assert active_state == BrainSceneWorldRecordState.ACTIVE.value
        assert order[active_state] <= order[stale_state] <= order[expired_state]
        assert expired_state == BrainSceneWorldRecordState.EXPIRED.value
    finally:
        bundle.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_expired_scene_world_records_do_not_reactivate_without_newer_evidence(spec):
    # Once scene records have expired, later reference times must not reactivate them without new evidence.
    bundle = build_active_state_bundle(spec, client_suffix="scene-expired")
    try:
        expired = bundle.build_scene_world_state(
            reference_second=bundle.scene_observed_second + spec.expire_after_secs + 1
        )
        later = bundle.build_scene_world_state(
            reference_second=bundle.scene_observed_second + spec.expire_after_secs + 8
        )
        later_by_entity_id = {entity.entity_id: entity for entity in later.entities}

        for entity in expired.entities:
            if entity.entity_kind == BrainSceneWorldEntityKind.ZONE.value:
                continue
            if entity.state != BrainSceneWorldRecordState.EXPIRED.value:
                continue
            assert entity.entity_id not in later.active_entity_ids
            later_entity = later_by_entity_id.get(entity.entity_id)
            if later_entity is not None:
                assert later_entity.state != BrainSceneWorldRecordState.ACTIVE.value
    finally:
        bundle.close()

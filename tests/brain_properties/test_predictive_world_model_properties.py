from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)
from blink.brain.world_model import build_baseline_predictions

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@st.composite
def _scene_world_projection_strategy(draw) -> BrainSceneWorldProjection:
    entity_count = draw(st.integers(min_value=1, max_value=5))
    affordance_count = draw(st.integers(min_value=0, max_value=4))
    degraded_mode = draw(st.sampled_from(("healthy", "limited", "unavailable")))
    entities = []
    for index in range(entity_count):
        entity_kind = draw(
            st.sampled_from(
                (
                    BrainSceneWorldEntityKind.PERSON.value,
                    BrainSceneWorldEntityKind.OBJECT.value,
                )
            )
        )
        entities.append(
            BrainSceneWorldEntityRecord(
                entity_id=f"entity-{index}",
                entity_kind=entity_kind,
                canonical_label=f"Entity {index}",
                summary=f"Entity {index} is visible.",
                state=draw(
                    st.sampled_from(
                        (
                            BrainSceneWorldRecordState.ACTIVE.value,
                            BrainSceneWorldRecordState.STALE.value,
                        )
                    )
                ),
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                confidence=draw(st.floats(min_value=0.35, max_value=0.99)),
                source_event_ids=[f"evt-entity-{index}"],
                observed_at="2026-01-01T00:00:02+00:00",
                updated_at="2026-01-01T00:00:02+00:00",
                details={"stable_key": f"entity-{index}"},
            )
        )
    affordances = []
    for index in range(affordance_count):
        affordances.append(
            BrainSceneWorldAffordanceRecord(
                affordance_id=f"affordance-{index}",
                entity_id=f"entity-{index % entity_count}",
                capability_family=f"capability-{index}",
                summary=f"Affordance {index} is visible.",
                availability=draw(
                    st.sampled_from(
                        (
                            BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                            BrainSceneWorldAffordanceAvailability.BLOCKED.value,
                        )
                    )
                ),
                confidence=draw(st.floats(min_value=0.35, max_value=0.99)),
                source_event_ids=[f"evt-affordance-{index}"],
                observed_at="2026-01-01T00:00:02+00:00",
                updated_at="2026-01-01T00:00:02+00:00",
                details={"stable_key": f"affordance-{index}"},
            )
        )
    projection = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=entities,
        affordances=affordances,
        degraded_mode=degraded_mode,
        degraded_reason_codes=([] if degraded_mode == "healthy" else [f"{degraded_mode}_scene"]),
        updated_at="2026-01-01T00:00:02+00:00",
    )
    projection.sync_lists()
    return projection


def _empty_active_situation() -> BrainActiveSituationProjection:
    projection = BrainActiveSituationProjection(
        scope_type="thread",
        scope_id="thread-1",
        updated_at="2026-01-01T00:00:02+00:00",
    )
    projection.sync_lists()
    return projection


@given(scene_world_state=_scene_world_projection_strategy())
@_SETTINGS
def test_baseline_predictions_are_deterministic_and_bounded(scene_world_state):
    first = build_baseline_predictions(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        reference_ts="2026-01-01T00:00:02+00:00",
        scene_world_state=scene_world_state,
        active_situation_model=_empty_active_situation(),
        private_working_memory=None,
        procedural_skills=None,
        scene_episodes=(),
        commitment_projection=None,
    )
    second = build_baseline_predictions(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        reference_ts="2026-01-01T00:00:02+00:00",
        scene_world_state=scene_world_state,
        active_situation_model=_empty_active_situation(),
        private_working_memory=None,
        procedural_skills=None,
        scene_episodes=(),
        commitment_projection=None,
    )

    assert [
        (
            record.prediction_id,
            record.prediction_kind,
            record.subject_id,
            record.predicted_state,
            record.confidence_band,
        )
        for record in first
    ] == [
        (
            record.prediction_id,
            record.prediction_kind,
            record.subject_id,
            record.predicted_state,
            record.confidence_band,
        )
        for record in second
    ]

    counts: dict[str, int] = {}
    for record in first:
        counts[record.prediction_kind] = counts.get(record.prediction_kind, 0) + 1
    assert len(first) <= 8
    assert counts.get("entity_persistence", 0) <= 2
    assert counts.get("affordance_persistence", 0) <= 2
    assert counts.get("engagement_drift", 0) <= 1
    assert counts.get("scene_transition", 0) <= 1
    assert counts.get("action_outcome", 0) <= 1
    assert counts.get("wake_readiness", 0) <= 1
    assert len({record.prediction_id for record in first}) == len(first)


@given(scene_world_state=_scene_world_projection_strategy())
@_SETTINGS
def test_building_predictions_does_not_mutate_observed_inputs(scene_world_state):
    active_situation_model = _empty_active_situation()
    scene_before = deepcopy(scene_world_state.as_dict())
    active_before = deepcopy(active_situation_model.as_dict())

    _ = build_baseline_predictions(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        reference_ts="2026-01-01T00:00:02+00:00",
        scene_world_state=scene_world_state,
        active_situation_model=active_situation_model,
        private_working_memory=None,
        procedural_skills=None,
        scene_episodes=(),
        commitment_projection=None,
    )

    assert scene_world_state.as_dict() == scene_before
    assert active_situation_model.as_dict() == active_before

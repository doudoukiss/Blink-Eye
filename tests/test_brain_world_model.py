from __future__ import annotations

from pathlib import Path

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.world_model import (
    WorldModelAdapterRequest,
    WorldModelAdapterResponse,
    WorldModelPredictionProposal,
)
from blink.brain.events import BrainEventType
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationRecordKind,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStep,
    BrainPredictionKind,
    BrainPredictionResolutionKind,
    BrainPredictionSubjectKind,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldEvidenceKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.brain.world_model_digest import build_world_model_digest


class RecordingWorldModelAdapter:
    def __init__(self):
        self.descriptor = BrainAdapterDescriptor(
            backend_id="recording_world_model",
            backend_version="v1",
            capabilities=("prediction_proposal",),
            degraded_mode_id="empty_proposals",
            default_timeout_ms=10,
        )
        self.requests: list[WorldModelAdapterRequest] = []

    def propose_predictions(self, request: WorldModelAdapterRequest) -> WorldModelAdapterResponse:
        self.requests.append(request)
        return WorldModelAdapterResponse(
            proposals=(
                WorldModelPredictionProposal(
                    prediction_kind=BrainPredictionKind.ACTION_OUTCOME.value,
                    subject_kind=BrainPredictionSubjectKind.ACTION.value,
                    subject_id="adapter-action",
                    summary="Adapter-backed action should remain safe.",
                    predicted_state={"accepted": True},
                    confidence=0.8,
                    risk_codes=("adapter_risk",),
                    supporting_event_ids=("adapter-event",),
                    backing_ids=("adapter-action",),
                    plan_proposal_id="adapter-plan",
                    details={"prediction_role": "opportunity"},
                ),
            ),
        )


class UnavailableWorldModelAdapter:
    def __init__(self):
        self.descriptor = BrainAdapterDescriptor(
            backend_id="unavailable_world_model",
            backend_version="v1",
            capabilities=("prediction_proposal",),
            degraded_mode_id="empty_proposals",
            default_timeout_ms=10,
        )

    def propose_predictions(self, request: WorldModelAdapterRequest) -> WorldModelAdapterResponse:
        return WorldModelAdapterResponse(
            proposals=(),
            available=False,
            reason="backend_unavailable",
            warnings=("backend_unavailable",),
        )


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _ensure_blocks(store: BrainStore) -> None:
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )


def _append_body_state(
    store: BrainStore,
    session_ids,
    *,
    second: int,
    presence_scope_key: str = "browser:presence",
):
    return store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "presence_scope_key": presence_scope_key,
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                vision_enabled=True,
                vision_connected=True,
                camera_track_state="tracking",
                sensor_health="healthy",
                camera_disconnected=False,
                perception_unreliable=False,
                last_fresh_frame_at=_ts(second),
                frame_age_ms=250,
                updated_at=_ts(second),
            ).as_dict(),
        },
        ts=_ts(second),
    )


def _append_scene_changed(
    store: BrainStore,
    session_ids,
    *,
    second: int,
    presence_scope_key: str = "browser:presence",
    include_person: bool = True,
    affordance_availability: str = BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
    engagement_state: str = "engaged",
    camera_fresh: bool = True,
):
    scene_entities = [
        *(
            [
                {
                    "entity_key": "person-1",
                    "kind": "person",
                    "label": "Ada",
                    "summary": "Ada is at the desk.",
                }
            ]
            if include_person
            else []
        ),
        {
            "entity_key": "desk-1",
            "kind": "object",
            "label": "Desk",
            "summary": "A work desk is visible.",
            "affordances": [
                {
                    "capability_family": "inspect",
                    "summary": "Inspect the desk.",
                    "availability": affordance_availability,
                }
            ],
        },
    ]
    return store.append_brain_event(
        event_type=BrainEventType.SCENE_CHANGED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "presence_scope_key": presence_scope_key,
            "camera_connected": True,
            "camera_fresh": camera_fresh,
            "camera_track_state": "tracking" if camera_fresh else "stalled",
            "person_present": "present" if include_person else "absent",
            "engagement_state": engagement_state if include_person else "idle",
            "attention_to_camera": "focused" if include_person else "unknown",
            "summary": "Ada and a desk are visible." if include_person else "Only the desk is visible.",
            "observed_at": _ts(second),
            "scene_entities": scene_entities,
        },
        ts=_ts(second),
    )


def _append_goal_created(
    store: BrainStore,
    session_ids,
    *,
    second: int,
    goal_id: str = "goal-predictive-1",
    title: str = "Inspect the desk",
):
    goal = BrainGoal(
        goal_id=goal_id,
        title=title,
        intent="environment.inspect",
        source="test",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        status="open",
        steps=[BrainGoalStep(capability_id="inspect")],
        created_at=_ts(second),
        updated_at=_ts(second),
    )
    event = store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal.as_dict()},
        correlation_id=goal_id,
        ts=_ts(second),
    )
    return goal, event


def _append_goal_updated(
    store: BrainStore,
    session_ids,
    *,
    goal: BrainGoal,
    second: int,
):
    updated_goal = BrainGoal(
        goal_id=goal.goal_id,
        title=goal.title,
        intent=goal.intent,
        source=goal.source,
        goal_family=goal.goal_family,
        status="open",
        steps=list(goal.steps),
        created_at=goal.created_at,
        updated_at=_ts(second),
    )
    event = store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": updated_goal.as_dict()},
        correlation_id=goal.goal_id,
        ts=_ts(second),
    )
    return updated_goal, event


def _append_robot_action_outcome(
    store: BrainStore,
    session_ids,
    *,
    second: int,
    accepted: bool,
):
    return store.append_brain_event(
        event_type=BrainEventType.ROBOT_ACTION_OUTCOME,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"accepted": accepted, "result": "accepted" if accepted else "blocked"},
        ts=_ts(second),
    )


def _predictive_projection(store: BrainStore, session_ids):
    return store.build_predictive_world_model_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        reference_ts=_ts(120),
        agent_id=session_ids.agent_id,
        presence_scope_key="browser:presence",
    )


def test_predictive_world_model_generation_is_deterministic_and_bounded(tmp_path):
    first_store = BrainStore(path=tmp_path / "first.db")
    second_store = BrainStore(path=tmp_path / "second.db")
    try:
        first_session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="predictive-det")
        second_session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="predictive-det",
        )

        for store, session_ids in (
            (first_store, first_session_ids),
            (second_store, second_session_ids),
        ):
            _ensure_blocks(store)
            _append_body_state(store, session_ids, second=1)
            _append_scene_changed(store, session_ids, second=2)
            _append_goal_created(store, session_ids, second=3)
            _append_scene_changed(store, session_ids, second=4)

        first_projection = _predictive_projection(first_store, first_session_ids)
        second_projection = _predictive_projection(second_store, second_session_ids)

        assert [
            (
                record.prediction_id,
                record.prediction_kind,
                record.subject_kind,
                record.subject_id,
                record.predicted_state,
                record.confidence_band,
            )
            for record in first_projection.active_predictions
        ] == [
            (
                record.prediction_id,
                record.prediction_kind,
                record.subject_kind,
                record.subject_id,
                record.predicted_state,
                record.confidence_band,
            )
            for record in second_projection.active_predictions
        ]
        assert [
            (
                record.prediction_id,
                record.resolution_kind,
                record.subject_id,
            )
            for record in first_projection.recent_resolutions
        ] == [
            (
                record.prediction_id,
                record.resolution_kind,
                record.subject_id,
            )
            for record in second_projection.recent_resolutions
        ]
        assert first_projection.presence_scope_key == "browser:presence"
        assert len(first_projection.active_predictions) <= 8
        assert first_projection.active_kind_counts.get(
            BrainPredictionKind.ENTITY_PERSISTENCE.value,
            0,
        ) <= 2
        assert first_projection.active_kind_counts.get(
            BrainPredictionKind.AFFORDANCE_PERSISTENCE.value,
            0,
        ) <= 2
        assert first_projection.active_kind_counts.get(
            BrainPredictionKind.ENGAGEMENT_DRIFT.value,
            0,
        ) <= 1
        assert first_projection.active_kind_counts.get(
            BrainPredictionKind.SCENE_TRANSITION.value,
            0,
        ) <= 1
        assert first_projection.active_kind_counts.get(
            BrainPredictionKind.ACTION_OUTCOME.value,
            0,
        ) <= 1

        digest = build_world_model_digest(predictive_world_model=first_projection.as_dict())
        assert digest["active_kind_counts"] == first_projection.active_kind_counts
        assert digest["active_confidence_band_counts"] == first_projection.active_confidence_band_counts
        assert digest["highest_risk_prediction_ids"] == sorted(digest["highest_risk_prediction_ids"])
    finally:
        first_store.close()
        second_store.close()


def test_predictive_world_model_lifecycle_confirms_invalidates_and_expires(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    try:
        _ensure_blocks(store)
        session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="predictive-lifecycle",
        )
        _append_body_state(store, session_ids, second=1)
        _append_scene_changed(store, session_ids, second=2)
        goal, _ = _append_goal_created(store, session_ids, second=3)
        refresh_event = _append_scene_changed(store, session_ids, second=4)

        initial_projection = _predictive_projection(store, session_ids)
        initial_action_prediction = next(
            record
            for record in initial_projection.active_predictions
            if record.prediction_kind == BrainPredictionKind.ACTION_OUTCOME.value
        )

        accepted_event = _append_robot_action_outcome(
            store,
            session_ids,
            second=5,
            accepted=True,
        )
        after_confirm = _predictive_projection(store, session_ids)
        assert any(
            record.prediction_id == initial_action_prediction.prediction_id
            and record.resolution_kind == BrainPredictionResolutionKind.CONFIRMED.value
            for record in after_confirm.recent_resolutions
        )

        _append_goal_updated(store, session_ids, goal=goal, second=6)
        refreshed_event = _append_scene_changed(store, session_ids, second=7)
        refreshed_projection = _predictive_projection(store, session_ids)
        refreshed_action_prediction = next(
            record
            for record in refreshed_projection.active_predictions
            if record.prediction_kind == BrainPredictionKind.ACTION_OUTCOME.value
        )

        blocked_event = _append_robot_action_outcome(
            store,
            session_ids,
            second=8,
            accepted=False,
        )
        after_invalidation = _predictive_projection(store, session_ids)
        assert any(
            record.prediction_id == refreshed_action_prediction.prediction_id
            and record.resolution_kind == BrainPredictionResolutionKind.INVALIDATED.value
            for record in after_invalidation.recent_resolutions
        )

        expiry_trigger = _append_scene_changed(
            store,
            session_ids,
            second=40,
            include_person=False,
            affordance_availability=BrainSceneWorldAffordanceAvailability.BLOCKED.value,
            engagement_state="idle",
            camera_fresh=False,
        )
        after_expiry = _predictive_projection(store, session_ids)
        assert any(
            record.resolution_kind == BrainPredictionResolutionKind.EXPIRED.value
            for record in after_expiry.recent_resolutions
        )

        all_events = store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=128,
        )
        assert any(
            event.event_type == BrainEventType.ACTION_OUTCOME_PREDICTED
            and event.causal_parent_id == refresh_event.event_id
            for event in all_events
        )
        assert any(
            event.event_type == BrainEventType.PREDICTION_CONFIRMED
            and event.causal_parent_id == accepted_event.event_id
            for event in all_events
        )
        assert any(
            event.event_type == BrainEventType.PREDICTION_INVALIDATED
            and event.causal_parent_id in {blocked_event.event_id, refreshed_event.event_id}
            for event in all_events
        )
        assert any(
            event.event_type == BrainEventType.PREDICTION_EXPIRED
            and event.causal_parent_id == expiry_trigger.event_id
            for event in all_events
        )
    finally:
        store.close()


def test_predictive_state_stays_separate_from_observed_scene_and_active_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    try:
        _ensure_blocks(store)
        session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="predictive-separation")
        _append_body_state(store, session_ids, second=1)
        _append_scene_changed(store, session_ids, second=2)
        _append_goal_created(store, session_ids, second=3)
        _append_scene_changed(store, session_ids, second=4)
        _append_robot_action_outcome(store, session_ids, second=5, accepted=False)

        scene_world_state = store.build_scene_world_state_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            reference_ts=_ts(5),
            agent_id=session_ids.agent_id,
            presence_scope_key="browser:presence",
        )
        active_situation_model = store.build_active_situation_model_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            reference_ts=_ts(5),
            agent_id=session_ids.agent_id,
            presence_scope_key="browser:presence",
        )

        assert all(
            record.evidence_kind != BrainSceneWorldEvidenceKind.HYPOTHESIZED.value
            for record in scene_world_state.entities
        )
        assert all(
            record.availability in {"available", "blocked", "uncertain", "stale"}
            for record in scene_world_state.affordances
        )

        prediction_records = [
            record
            for record in active_situation_model.records
            if record.record_kind == BrainActiveSituationRecordKind.PREDICTION_STATE.value
        ]
        assert prediction_records
        assert all(
            record.evidence_kind == BrainActiveSituationEvidenceKind.HYPOTHESIZED.value
            for record in prediction_records
        )
        assert all(record.details.get("prediction_id") for record in prediction_records)
        assert all(
            record.record_kind != BrainActiveSituationRecordKind.PREDICTION_STATE.value
            for record in active_situation_model.records
            if record.evidence_kind != BrainActiveSituationEvidenceKind.HYPOTHESIZED.value
        )
    finally:
        store.close()


def test_brain_store_consumes_world_model_adapter_proposals_and_keeps_stable_prediction_ids(tmp_path):
    first_adapter = RecordingWorldModelAdapter()
    second_adapter = RecordingWorldModelAdapter()
    first_store = BrainStore(path=tmp_path / "first.db", world_model_adapter=first_adapter)
    second_store = BrainStore(path=tmp_path / "second.db", world_model_adapter=second_adapter)
    try:
        first_session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="adapter-det")
        second_session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="adapter-det")

        for store, session_ids in (
            (first_store, first_session_ids),
            (second_store, second_session_ids),
        ):
            _ensure_blocks(store)
            _append_body_state(store, session_ids, second=1)
            _append_scene_changed(store, session_ids, second=2)
            _append_goal_created(store, session_ids, second=3)
            _append_scene_changed(store, session_ids, second=4)

        first_projection = _predictive_projection(first_store, first_session_ids)
        second_projection = _predictive_projection(second_store, second_session_ids)

        first_adapter_prediction = next(
            record
            for record in first_projection.active_predictions
            if record.subject_id == "adapter-action"
        )
        second_adapter_prediction = next(
            record
            for record in second_projection.active_predictions
            if record.subject_id == "adapter-action"
        )

        assert first_adapter.requests
        assert second_adapter.requests
        assert first_adapter_prediction.prediction_id == second_adapter_prediction.prediction_id
        assert first_adapter_prediction.supporting_event_ids == ["adapter-event"]
        assert first_adapter_prediction.details["prediction_role"] == "opportunity"
    finally:
        first_store.close()
        second_store.close()


def test_brain_store_handles_unavailable_world_model_adapter_without_generating_predictions(tmp_path):
    adapter = UnavailableWorldModelAdapter()
    store = BrainStore(path=tmp_path / "brain.db", world_model_adapter=adapter)
    try:
        _ensure_blocks(store)
        session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="adapter-unavailable")
        _append_body_state(store, session_ids, second=1)
        _append_scene_changed(store, session_ids, second=2)
        _append_goal_created(store, session_ids, second=3)
        _append_scene_changed(store, session_ids, second=4)

        projection = _predictive_projection(store, session_ids)
        events = store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=32,
        )

        assert projection.active_predictions == []
        assert not any(event.event_type.endswith(".predicted") for event in events)
        assert not any(event.event_type.startswith("prediction.") for event in events)
    finally:
        store.close()


__all__ = [
    "_append_body_state",
    "_append_goal_created",
    "_append_goal_updated",
    "_append_robot_action_outcome",
    "_append_scene_changed",
    "_ensure_blocks",
    "_predictive_projection",
    "_ts",
]

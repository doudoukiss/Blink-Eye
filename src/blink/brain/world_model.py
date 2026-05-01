"""Deterministic short-horizon predictive world-model helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2 import BrainAutobiographicalEntryRecord, BrainProceduralSkillProjection
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainActiveSituationRecordKind,
    BrainCommitmentProjection,
    BrainPredictionConfidenceBand,
    BrainPredictionKind,
    BrainPredictionRecord,
    BrainPredictionResolutionKind,
    BrainPredictionSubjectKind,
    BrainPredictiveWorldModelProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)

_PREDICTION_WINDOW_SECONDS = {
    BrainPredictionKind.ENTITY_PERSISTENCE.value: 30,
    BrainPredictionKind.AFFORDANCE_PERSISTENCE.value: 20,
    BrainPredictionKind.ENGAGEMENT_DRIFT.value: 15,
    BrainPredictionKind.SCENE_TRANSITION.value: 20,
    BrainPredictionKind.ACTION_OUTCOME.value: 20,
    BrainPredictionKind.WAKE_READINESS.value: 25,
}

_OBSERVATION_EVENT_TYPES = {
    BrainEventType.BODY_STATE_UPDATED,
    BrainEventType.PERCEPTION_OBSERVED,
    BrainEventType.ENGAGEMENT_CHANGED,
    BrainEventType.ATTENTION_CHANGED,
    BrainEventType.SCENE_CHANGED,
}

_PREDICTIVE_GENERATION_EVENT_TYPES = {
    BrainPredictionKind.ENTITY_PERSISTENCE.value: BrainEventType.ENTITY_PREDICTION_GENERATED,
    BrainPredictionKind.AFFORDANCE_PERSISTENCE.value: (
        BrainEventType.AFFORDANCE_PREDICTION_GENERATED
    ),
    BrainPredictionKind.ENGAGEMENT_DRIFT.value: BrainEventType.ENGAGEMENT_PREDICTION_GENERATED,
    BrainPredictionKind.SCENE_TRANSITION.value: BrainEventType.SCENE_PREDICTION_GENERATED,
    BrainPredictionKind.ACTION_OUTCOME.value: BrainEventType.ACTION_OUTCOME_PREDICTED,
    BrainPredictionKind.WAKE_READINESS.value: BrainEventType.WAKE_PREDICTION_GENERATED,
}

_PREDICTIVE_EVENT_TYPES = frozenset(
    {
        *tuple(_PREDICTIVE_GENERATION_EVENT_TYPES.values()),
        BrainEventType.PREDICTION_CONFIRMED,
        BrainEventType.PREDICTION_INVALIDATED,
        BrainEventType.PREDICTION_EXPIRED,
    }
)

_PREDICTION_TRIGGER_EVENT_TYPES = frozenset(
    {
        BrainEventType.PERCEPTION_OBSERVED,
        BrainEventType.ENGAGEMENT_CHANGED,
        BrainEventType.ATTENTION_CHANGED,
        BrainEventType.SCENE_CHANGED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
    }
)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _ts_after(reference_ts: str, *, seconds: int) -> str:
    base = _parse_ts(reference_ts) or datetime.now(UTC)
    return (base + timedelta(seconds=seconds)).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _stable_prediction_id(
    *,
    prediction_kind: str,
    subject_kind: str,
    subject_id: str,
    predicted_state: dict[str, Any],
    valid_from: str,
    valid_to: str | None,
) -> str:
    payload = json.dumps(
        {
            "prediction_kind": prediction_kind,
            "subject_kind": subject_kind,
            "subject_id": subject_id,
            "predicted_state": predicted_state,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"prediction_{uuid5(NAMESPACE_URL, f'blink:prediction:{payload}').hex}"


def _prediction_confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return BrainPredictionConfidenceBand.HIGH.value
    if confidence >= 0.55:
        return BrainPredictionConfidenceBand.MEDIUM.value
    return BrainPredictionConfidenceBand.LOW.value


def prediction_event_types() -> frozenset[str]:
    """Return all predictive lifecycle event types."""
    return _PREDICTIVE_EVENT_TYPES


def prediction_trigger_event_types() -> frozenset[str]:
    """Return source event types that may refresh predictive state."""
    return _PREDICTION_TRIGGER_EVENT_TYPES


def prediction_generation_event_type(prediction_kind: str) -> str:
    """Return the canonical generation event type for one prediction kind."""
    return _PREDICTIVE_GENERATION_EVENT_TYPES[prediction_kind]


def should_refresh_predictive_world_model(event_type: str) -> bool:
    """Return whether one source event should refresh predictive state."""
    return event_type in _PREDICTION_TRIGGER_EVENT_TYPES


def is_predictive_event_type(event_type: str) -> bool:
    """Return whether one event belongs to the predictive lifecycle."""
    return event_type in _PREDICTIVE_EVENT_TYPES


def _latest_scene_episode(
    scene_episodes: Iterable[BrainAutobiographicalEntryRecord],
) -> BrainAutobiographicalEntryRecord | None:
    return next(iter(scene_episodes), None)


def _active_scene_state_details(
    active_situation_model: BrainActiveSituationProjection,
) -> dict[str, Any]:
    for record in active_situation_model.records:
        if record.record_kind == BrainActiveSituationRecordKind.SCENE_STATE.value:
            return dict(record.details)
    return {}


def _first_action_subject(
    active_situation_model: BrainActiveSituationProjection,
    procedural_skills: BrainProceduralSkillProjection | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            if skill.status != "active":
                continue
            commitment_id = next(iter(skill.supporting_commitment_ids), None)
            plan_proposal_id = next(iter(skill.supporting_plan_proposal_ids), None)
            subject_id = skill.skill_id
            if subject_id:
                return subject_id, skill.skill_id, commitment_id, plan_proposal_id
    for record in active_situation_model.records:
        if record.record_kind not in {
            BrainActiveSituationRecordKind.PLAN_STATE.value,
            BrainActiveSituationRecordKind.PROCEDURAL_STATE.value,
            BrainActiveSituationRecordKind.COMMITMENT_STATE.value,
            BrainActiveSituationRecordKind.GOAL_STATE.value,
        }:
            continue
        subject_id = record.skill_id or record.plan_proposal_id or record.commitment_id or record.goal_id
        if subject_id is not None:
            return subject_id, record.skill_id, record.commitment_id, record.plan_proposal_id
    return None, None, None, None


def _person_present(scene_world_state: BrainSceneWorldProjection) -> bool:
    return any(
        record.entity_kind == "person" and record.state == BrainSceneWorldRecordState.ACTIVE.value
        for record in scene_world_state.entities
    )


def _wake_ready(scene_world_state: BrainSceneWorldProjection) -> bool:
    return scene_world_state.degraded_mode == "healthy" and _person_present(scene_world_state)


def _make_prediction(
    *,
    prediction_kind: str,
    subject_kind: str,
    subject_id: str,
    scope_key: str,
    presence_scope_key: str,
    summary: str,
    predicted_state: dict[str, Any],
    confidence: float,
    risk_codes: Iterable[str] = (),
    supporting_event_ids: Iterable[str] = (),
    backing_ids: Iterable[str] = (),
    reference_ts: str,
    action_id: str | None = None,
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainPredictionRecord:
    valid_to = _ts_after(
        reference_ts,
        seconds=_PREDICTION_WINDOW_SECONDS.get(prediction_kind, 15),
    )
    prediction_id = _stable_prediction_id(
        prediction_kind=prediction_kind,
        subject_kind=subject_kind,
        subject_id=subject_id,
        predicted_state=predicted_state,
        valid_from=reference_ts,
        valid_to=valid_to,
    )
    normalized_confidence = max(0.0, min(float(confidence), 1.0))
    return BrainPredictionRecord(
        prediction_id=prediction_id,
        prediction_kind=prediction_kind,
        subject_kind=subject_kind,
        subject_id=subject_id,
        scope_key=scope_key,
        presence_scope_key=presence_scope_key,
        summary=summary,
        predicted_state=dict(predicted_state),
        confidence=normalized_confidence,
        confidence_band=_prediction_confidence_band(normalized_confidence),
        risk_codes=_sorted_unique(risk_codes),
        supporting_event_ids=_sorted_unique(supporting_event_ids),
        backing_ids=_sorted_unique(backing_ids),
        action_id=action_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_id=skill_id,
        predicted_at=reference_ts,
        valid_from=reference_ts,
        valid_to=valid_to,
        details=dict(details or {}),
        updated_at=reference_ts,
    )


def build_baseline_predictions(
    *,
    scope_key: str,
    presence_scope_key: str,
    reference_ts: str,
    scene_world_state: BrainSceneWorldProjection,
    active_situation_model: BrainActiveSituationProjection,
    private_working_memory: Any,
    procedural_skills: BrainProceduralSkillProjection | None,
    scene_episodes: Iterable[BrainAutobiographicalEntryRecord],
    commitment_projection: BrainCommitmentProjection | None,
) -> list[BrainPredictionRecord]:
    """Build deterministic short-horizon predictions from the local adapter proposals."""
    from blink.brain.adapters.world_model import (
        LocalDeterministicWorldModelAdapter,
        WorldModelAdapterRequest,
    )

    response = LocalDeterministicWorldModelAdapter().propose_predictions(
        WorldModelAdapterRequest(
            scope_key=scope_key,
            presence_scope_key=presence_scope_key,
            reference_ts=reference_ts,
            scene_world_state=scene_world_state,
            active_situation_model=active_situation_model,
            private_working_memory=private_working_memory,
            procedural_skills=procedural_skills,
            scene_episodes=tuple(scene_episodes),
            commitment_projection=commitment_projection,
        )
    )
    return build_prediction_records_from_proposals(
        scope_key=scope_key,
        presence_scope_key=presence_scope_key,
        reference_ts=reference_ts,
        proposals=response.proposals,
    )


def build_prediction_records_from_proposals(
    *,
    scope_key: str,
    presence_scope_key: str,
    reference_ts: str,
    proposals: Iterable[Any],
) -> list[BrainPredictionRecord]:
    """Convert proposal-only adapter rows into stable replay-owned prediction records."""
    records = [
        _make_prediction(
            prediction_kind=str(proposal.prediction_kind),
            subject_kind=str(proposal.subject_kind),
            subject_id=str(proposal.subject_id),
            scope_key=scope_key,
            presence_scope_key=presence_scope_key,
            summary=str(proposal.summary),
            predicted_state=dict(proposal.predicted_state),
            confidence=float(proposal.confidence),
            risk_codes=list(getattr(proposal, "risk_codes", ()) or ()),
            supporting_event_ids=list(getattr(proposal, "supporting_event_ids", ()) or ()),
            backing_ids=list(getattr(proposal, "backing_ids", ()) or ()),
            reference_ts=reference_ts,
            action_id=_optional_text(getattr(proposal, "action_id", None)),
            goal_id=_optional_text(getattr(proposal, "goal_id", None)),
            commitment_id=_optional_text(getattr(proposal, "commitment_id", None)),
            plan_proposal_id=_optional_text(getattr(proposal, "plan_proposal_id", None)),
            skill_id=_optional_text(getattr(proposal, "skill_id", None)),
            details=dict(getattr(proposal, "details", {}) or {}),
        )
        for proposal in proposals
    ]
    return sorted(
        records,
        key=lambda record: (
            record.prediction_kind,
            record.subject_kind,
            record.subject_id,
            record.prediction_id,
        ),
    )


def resolve_prediction_against_state(
    *,
    prediction: BrainPredictionRecord,
    trigger_event: BrainEventRecord,
    scene_world_state: BrainSceneWorldProjection,
    active_situation_model: BrainActiveSituationProjection,
) -> bool | None:
    """Return whether one active prediction is confirmed or invalidated, if decidable."""
    if prediction.prediction_kind == BrainPredictionKind.ACTION_OUTCOME.value:
        if trigger_event.event_type != BrainEventType.ROBOT_ACTION_OUTCOME:
            return None
        accepted = bool((trigger_event.payload or {}).get("accepted"))
        return accepted == bool(prediction.predicted_state.get("accepted"))

    if trigger_event.event_type not in _OBSERVATION_EVENT_TYPES:
        return None

    if prediction.prediction_kind == BrainPredictionKind.ENTITY_PERSISTENCE.value:
        entity = next(
            (record for record in scene_world_state.entities if record.entity_id == prediction.subject_id),
            None,
        )
        if entity is None:
            return False
        return entity.state == prediction.predicted_state.get("expected_state")

    if prediction.prediction_kind == BrainPredictionKind.AFFORDANCE_PERSISTENCE.value:
        affordance = next(
            (
                record
                for record in scene_world_state.affordances
                if record.affordance_id == prediction.subject_id
            ),
            None,
        )
        if affordance is None:
            return False
        return affordance.availability == prediction.predicted_state.get("expected_availability")

    scene_details = _active_scene_state_details(active_situation_model)
    if prediction.prediction_kind == BrainPredictionKind.ENGAGEMENT_DRIFT.value:
        current_engagement = str(scene_details.get("engagement_state", "")).strip()
        current_person_present = str(scene_details.get("person_present", "")).strip() or (
            "present" if _person_present(scene_world_state) else "absent"
        )
        expected_engagement = str(prediction.predicted_state.get("engagement_state", "")).strip()
        expected_person_present = str(
            prediction.predicted_state.get("person_present", current_person_present)
        ).strip()
        return (
            current_engagement == expected_engagement
            and current_person_present == expected_person_present
        )

    if prediction.prediction_kind == BrainPredictionKind.SCENE_TRANSITION.value:
        current_scene_change_state = str(scene_details.get("scene_change_state", "")).strip() or "stable"
        expected_scene_change_state = str(
            prediction.predicted_state.get("scene_change_state", current_scene_change_state)
        ).strip()
        expected_degraded_mode = str(
            prediction.predicted_state.get("degraded_mode", scene_world_state.degraded_mode)
        ).strip()
        return (
            current_scene_change_state == expected_scene_change_state
            and scene_world_state.degraded_mode == expected_degraded_mode
        )

    if prediction.prediction_kind == BrainPredictionKind.WAKE_READINESS.value:
        return _wake_ready(scene_world_state) == bool(prediction.predicted_state.get("ready"))

    return None


def build_prediction_resolution(
    *,
    prediction: BrainPredictionRecord,
    trigger_event: BrainEventRecord,
    resolution_kind: str,
    resolution_summary: str,
) -> BrainPredictionRecord:
    """Return a terminal resolution record derived from one active prediction."""
    return BrainPredictionRecord(
        prediction_id=prediction.prediction_id,
        prediction_kind=prediction.prediction_kind,
        subject_kind=prediction.subject_kind,
        subject_id=prediction.subject_id,
        scope_key=prediction.scope_key,
        presence_scope_key=prediction.presence_scope_key,
        summary=prediction.summary,
        predicted_state=dict(prediction.predicted_state),
        confidence=prediction.confidence,
        confidence_band=prediction.confidence_band,
        risk_codes=list(prediction.risk_codes),
        supporting_event_ids=list(prediction.supporting_event_ids),
        backing_ids=list(prediction.backing_ids),
        action_id=prediction.action_id,
        goal_id=prediction.goal_id,
        commitment_id=prediction.commitment_id,
        plan_proposal_id=prediction.plan_proposal_id,
        skill_id=prediction.skill_id,
        predicted_at=prediction.predicted_at,
        valid_from=prediction.valid_from,
        valid_to=prediction.valid_to,
        resolved_at=trigger_event.ts,
        resolution_kind=resolution_kind,
        resolution_event_ids=_sorted_unique(
            [*prediction.resolution_event_ids, trigger_event.event_id]
        ),
        resolution_summary=resolution_summary,
        details=dict(prediction.details),
        updated_at=trigger_event.ts,
    )


def append_prediction_generation(
    projection: BrainPredictiveWorldModelProjection,
    prediction: BrainPredictionRecord,
) -> None:
    """Append one generated prediction into the projection."""
    projection.active_predictions = [
        record
        for record in projection.active_predictions
        if record.prediction_id != prediction.prediction_id
    ]
    projection.active_predictions.append(prediction)
    calibration_summary = projection.calibration_summary
    generated_kind_counts = dict(calibration_summary.generated_kind_counts)
    generated_kind_counts[prediction.prediction_kind] = (
        generated_kind_counts.get(prediction.prediction_kind, 0) + 1
    )
    projection.calibration_summary = type(projection.calibration_summary)(
        generated_count=calibration_summary.generated_count + 1,
        active_count=calibration_summary.active_count,
        confirmed_count=calibration_summary.confirmed_count,
        invalidated_count=calibration_summary.invalidated_count,
        expired_count=calibration_summary.expired_count,
        generated_kind_counts=dict(sorted(generated_kind_counts.items())),
        resolution_kind_counts=dict(calibration_summary.resolution_kind_counts),
        updated_at=prediction.updated_at,
    )
    projection.updated_at = prediction.updated_at
    projection.sync_lists()


def append_prediction_resolution(
    projection: BrainPredictiveWorldModelProjection,
    prediction: BrainPredictionRecord,
    *,
    max_recent_resolutions: int = 24,
) -> None:
    """Apply one terminal prediction resolution to the projection."""
    projection.active_predictions = [
        record
        for record in projection.active_predictions
        if record.prediction_id != prediction.prediction_id
    ]
    projection.recent_resolutions = [
        record
        for record in projection.recent_resolutions
        if record.prediction_id != prediction.prediction_id
    ]
    projection.recent_resolutions.append(prediction)
    projection.recent_resolutions = sorted(
        projection.recent_resolutions,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_resolutions]
    resolution_kind_counts = dict(projection.calibration_summary.resolution_kind_counts)
    resolution_kind = prediction.resolution_kind or BrainPredictionResolutionKind.INVALIDATED.value
    resolution_kind_counts[resolution_kind] = resolution_kind_counts.get(resolution_kind, 0) + 1
    projection.calibration_summary = type(projection.calibration_summary)(
        generated_count=projection.calibration_summary.generated_count,
        active_count=projection.calibration_summary.active_count,
        confirmed_count=projection.calibration_summary.confirmed_count
        + int(resolution_kind == BrainPredictionResolutionKind.CONFIRMED.value),
        invalidated_count=projection.calibration_summary.invalidated_count
        + int(resolution_kind == BrainPredictionResolutionKind.INVALIDATED.value),
        expired_count=projection.calibration_summary.expired_count
        + int(resolution_kind == BrainPredictionResolutionKind.EXPIRED.value),
        generated_kind_counts=dict(projection.calibration_summary.generated_kind_counts),
        resolution_kind_counts=dict(sorted(resolution_kind_counts.items())),
        updated_at=prediction.updated_at,
    )
    projection.updated_at = prediction.updated_at
    projection.sync_lists()


__all__ = [
    "append_prediction_generation",
    "append_prediction_resolution",
    "build_baseline_predictions",
    "build_prediction_records_from_proposals",
    "build_prediction_resolution",
    "is_predictive_event_type",
    "prediction_event_types",
    "prediction_generation_event_type",
    "prediction_trigger_event_types",
    "resolve_prediction_against_state",
    "should_refresh_predictive_world_model",
]

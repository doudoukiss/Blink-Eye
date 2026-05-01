"""Brain-side world-model adapter contracts and local proposal backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from blink.brain.adapters import LOCAL_WORLD_MODEL_DESCRIPTOR, BrainAdapterDescriptor
from blink.brain.events import BrainEventType
from blink.brain.memory_v2 import BrainAutobiographicalEntryRecord, BrainProceduralSkillProjection
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainActiveSituationRecordKind,
    BrainCommitmentProjection,
    BrainPredictionKind,
    BrainPredictionSubjectKind,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)

_PREDICTION_CAPS = {
    BrainPredictionKind.ENTITY_PERSISTENCE.value: 2,
    BrainPredictionKind.AFFORDANCE_PERSISTENCE.value: 2,
    BrainPredictionKind.ENGAGEMENT_DRIFT.value: 1,
    BrainPredictionKind.SCENE_TRANSITION.value: 1,
    BrainPredictionKind.ACTION_OUTCOME.value: 1,
    BrainPredictionKind.WAKE_READINESS.value: 1,
}


@dataclass(frozen=True)
class WorldModelAdapterRequest:
    """Bounded proposal request for predictive world-model backends."""

    scope_key: str
    presence_scope_key: str
    reference_ts: str
    scene_world_state: BrainSceneWorldProjection
    active_situation_model: BrainActiveSituationProjection
    private_working_memory: Any
    procedural_skills: BrainProceduralSkillProjection | None = None
    scene_episodes: tuple[BrainAutobiographicalEntryRecord, ...] = ()
    commitment_projection: BrainCommitmentProjection | None = None


@dataclass(frozen=True)
class WorldModelPredictionProposal:
    """One bounded prediction proposal before core lifecycle ownership."""

    prediction_kind: str
    subject_kind: str
    subject_id: str
    summary: str
    predicted_state: dict[str, Any]
    confidence: float
    risk_codes: tuple[str, ...] = ()
    supporting_event_ids: tuple[str, ...] = ()
    backing_ids: tuple[str, ...] = ()
    action_id: str | None = None
    goal_id: str | None = None
    commitment_id: str | None = None
    plan_proposal_id: str | None = None
    skill_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldModelAdapterResponse:
    """Bounded world-model proposal response."""

    proposals: tuple[WorldModelPredictionProposal, ...] = ()
    available: bool = True
    reason: str | None = None
    warnings: tuple[str, ...] = ()


class WorldModelAdapter(Protocol):
    """Proposal-only predictive backend seam."""

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""

    def propose_predictions(self, request: WorldModelAdapterRequest) -> WorldModelAdapterResponse:
        """Return bounded prediction proposals for the current state."""


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: list[str | None] | tuple[str | None, ...]) -> tuple[str, ...]:
    return tuple(sorted({text for value in values if (text := _optional_text(value)) is not None}))


def _latest_scene_episode(
    scene_episodes: tuple[BrainAutobiographicalEntryRecord, ...],
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


def _proposal(
    *,
    prediction_kind: str,
    subject_kind: str,
    subject_id: str,
    summary: str,
    predicted_state: dict[str, Any],
    confidence: float,
    risk_codes: list[str] | tuple[str, ...] = (),
    supporting_event_ids: list[str] | tuple[str, ...] = (),
    backing_ids: list[str] | tuple[str, ...] = (),
    action_id: str | None = None,
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> WorldModelPredictionProposal:
    return WorldModelPredictionProposal(
        prediction_kind=prediction_kind,
        subject_kind=subject_kind,
        subject_id=subject_id,
        summary=summary,
        predicted_state=dict(predicted_state),
        confidence=max(0.0, min(float(confidence), 1.0)),
        risk_codes=_sorted_unique(list(risk_codes)),
        supporting_event_ids=_sorted_unique(list(supporting_event_ids)),
        backing_ids=_sorted_unique(list(backing_ids)),
        action_id=action_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_id=skill_id,
        details=dict(details or {}),
    )


class LocalDeterministicWorldModelAdapter:
    """Local deterministic proposal backend over the landed scene and active-state surfaces."""

    def __init__(self):
        """Initialize the local world-model adapter."""
        self._descriptor = LOCAL_WORLD_MODEL_DESCRIPTOR

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""
        return self._descriptor

    def propose_predictions(self, request: WorldModelAdapterRequest) -> WorldModelAdapterResponse:
        """Return deterministic short-horizon proposals from landed surfaces."""
        _ = request.private_working_memory
        latest_scene_episode = _latest_scene_episode(request.scene_episodes)
        latest_episode_event_ids = (
            latest_scene_episode.source_event_ids if latest_scene_episode is not None else []
        )
        latest_episode_backing = (
            [latest_scene_episode.entry_id] if latest_scene_episode is not None else []
        )
        scene_details = _active_scene_state_details(request.active_situation_model)
        candidates: list[WorldModelPredictionProposal] = []

        entities = sorted(
            (
                record
                for record in request.scene_world_state.entities
                if record.state in {
                    BrainSceneWorldRecordState.ACTIVE.value,
                    BrainSceneWorldRecordState.STALE.value,
                }
            ),
            key=lambda record: (
                0 if record.state == BrainSceneWorldRecordState.ACTIVE.value else 1,
                -(record.confidence or 0.0),
                record.entity_id,
            ),
        )
        for record in entities[: _PREDICTION_CAPS[BrainPredictionKind.ENTITY_PERSISTENCE.value]]:
            degraded = request.scene_world_state.degraded_mode != "healthy"
            confidence = record.confidence or (0.82 if record.state == "active" else 0.58)
            if degraded:
                confidence = min(confidence, 0.62)
            candidates.append(
                _proposal(
                    prediction_kind=BrainPredictionKind.ENTITY_PERSISTENCE.value,
                    subject_kind=BrainPredictionSubjectKind.ENTITY.value,
                    subject_id=record.entity_id,
                    summary=f"{record.canonical_label} likely remains present in the next window.",
                    predicted_state={
                        "entity_id": record.entity_id,
                        "expected_state": BrainSceneWorldRecordState.ACTIVE.value,
                    },
                    confidence=confidence,
                    risk_codes=[
                        *record.contradiction_codes[:2],
                        *(request.scene_world_state.degraded_reason_codes[:1] if degraded else []),
                    ],
                    supporting_event_ids=[*record.source_event_ids, *latest_episode_event_ids],
                    backing_ids=[record.entity_id, *record.backing_ids, *latest_episode_backing],
                    details={
                        "canonical_label": record.canonical_label,
                        "entity_kind": record.entity_kind,
                        "prediction_role": "opportunity",
                    },
                )
            )

        affordances = sorted(
            (
                record
                for record in request.scene_world_state.affordances
                if record.availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value
            ),
            key=lambda record: (-(record.confidence or 0.0), record.affordance_id),
        )
        for record in affordances[
            : _PREDICTION_CAPS[BrainPredictionKind.AFFORDANCE_PERSISTENCE.value]
        ]:
            degraded = request.scene_world_state.degraded_mode != "healthy"
            confidence = record.confidence or 0.74
            if degraded:
                confidence = min(confidence, 0.59)
            candidates.append(
                _proposal(
                    prediction_kind=BrainPredictionKind.AFFORDANCE_PERSISTENCE.value,
                    subject_kind=BrainPredictionSubjectKind.AFFORDANCE.value,
                    subject_id=record.affordance_id,
                    summary=f"{record.capability_family} likely remains available in the next window.",
                    predicted_state={
                        "affordance_id": record.affordance_id,
                        "expected_availability": BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                    },
                    confidence=confidence,
                    risk_codes=[
                        *record.reason_codes[:2],
                        *(request.scene_world_state.degraded_reason_codes[:1] if degraded else []),
                    ],
                    supporting_event_ids=[*record.source_event_ids, *latest_episode_event_ids],
                    backing_ids=[record.affordance_id, record.entity_id, *record.backing_ids],
                    details={
                        "capability_family": record.capability_family,
                        "prediction_role": "opportunity",
                    },
                )
            )

        engagement_state = str(scene_details.get("engagement_state", "")).strip()
        person_present = str(scene_details.get("person_present", "")).strip() or (
            "present" if _person_present(request.scene_world_state) else "absent"
        )
        if engagement_state or person_present:
            likely_state = engagement_state or ("engaged" if person_present == "present" else "away")
            degraded = request.scene_world_state.degraded_mode != "healthy"
            confidence = 0.78 if likely_state in {"engaged", "listening", "speaking"} else 0.64
            if degraded:
                confidence = min(confidence, 0.56)
            candidates.append(
                _proposal(
                    prediction_kind=BrainPredictionKind.ENGAGEMENT_DRIFT.value,
                    subject_kind=BrainPredictionSubjectKind.ENGAGEMENT.value,
                    subject_id="engagement",
                    summary=(
                        "User engagement likely stays attentive in the next window."
                        if likely_state in {"engaged", "listening", "speaking"}
                        else "User attention likely stays away in the next window."
                    ),
                    predicted_state={
                        "engagement_state": likely_state,
                        "person_present": person_present,
                    },
                    confidence=confidence,
                    risk_codes=request.scene_world_state.degraded_reason_codes[:1],
                    supporting_event_ids=latest_episode_event_ids,
                    backing_ids=[*latest_episode_backing, "engagement_state", "scene_state"],
                    details={
                        "prediction_role": (
                            "opportunity"
                            if likely_state in {"engaged", "listening", "speaking"}
                            else "blocker"
                        ),
                    },
                )
            )

        scene_change_state = str(scene_details.get("scene_change_state", "")).strip() or (
            latest_scene_episode.content.get("degraded_mode", "stable")
            if latest_scene_episode is not None
            else "stable"
        )
        scene_summary = (
            latest_scene_episode.rendered_summary
            if latest_scene_episode is not None
            else "Current scene conditions likely persist in the next window."
        )
        candidates.append(
            _proposal(
                prediction_kind=BrainPredictionKind.SCENE_TRANSITION.value,
                subject_kind=BrainPredictionSubjectKind.SCENE.value,
                subject_id=request.presence_scope_key,
                summary=f"Scene likely remains bounded: {scene_summary}",
                predicted_state={
                    "scene_change_state": scene_change_state,
                    "degraded_mode": request.scene_world_state.degraded_mode,
                },
                confidence=0.71 if request.scene_world_state.degraded_mode == "healthy" else 0.58,
                risk_codes=request.scene_world_state.degraded_reason_codes[:2],
                supporting_event_ids=latest_episode_event_ids,
                backing_ids=[request.presence_scope_key, *latest_episode_backing],
                details={
                    "prediction_role": (
                        "blocker"
                        if request.scene_world_state.degraded_mode != "healthy"
                        else "opportunity"
                    ),
                },
            )
        )

        action_subject_id, action_skill_id, action_commitment_id, action_plan_proposal_id = (
            _first_action_subject(request.active_situation_model, request.procedural_skills)
        )
        if action_subject_id is not None:
            action_safe = (
                request.scene_world_state.degraded_mode == "healthy"
                and bool(
                    request.scene_world_state.active_affordance_ids
                    or not request.scene_world_state.affordances
                )
            )
            action_risk_codes = list(request.scene_world_state.degraded_reason_codes[:2])
            if not action_safe and not action_risk_codes:
                action_risk_codes.append("no_available_affordance")
            candidates.append(
                _proposal(
                    prediction_kind=BrainPredictionKind.ACTION_OUTCOME.value,
                    subject_kind=BrainPredictionSubjectKind.ACTION.value,
                    subject_id=action_subject_id,
                    summary=(
                        "Next bounded action is likely safe to accept."
                        if action_safe
                        else "Next bounded action is likely to be blocked."
                    ),
                    predicted_state={"accepted": action_safe},
                    confidence=0.72 if action_safe else 0.66,
                    risk_codes=action_risk_codes,
                    supporting_event_ids=latest_episode_event_ids,
                    backing_ids=[
                        action_subject_id,
                        *request.scene_world_state.active_affordance_ids[:1],
                    ],
                    commitment_id=action_commitment_id,
                    plan_proposal_id=action_plan_proposal_id,
                    skill_id=action_skill_id,
                    details={
                        "prediction_role": "opportunity" if action_safe else "blocker",
                    },
                )
            )

        if request.commitment_projection is not None:
            waiting_commitments = [
                *request.commitment_projection.blocked_commitments,
                *request.commitment_projection.deferred_commitments,
            ]
            if waiting_commitments:
                record = waiting_commitments[0]
                ready = _wake_ready(request.scene_world_state)
                candidates.append(
                    _proposal(
                        prediction_kind=BrainPredictionKind.WAKE_READINESS.value,
                        subject_kind=BrainPredictionSubjectKind.WAKE_CONDITION.value,
                        subject_id=record.commitment_id,
                        summary=(
                            f"{record.title} likely remains ready to resume soon."
                            if ready
                            else f"{record.title} likely remains blocked by current scene conditions."
                        ),
                        predicted_state={"ready": ready},
                        confidence=0.68 if ready else 0.74,
                        risk_codes=(
                            []
                            if ready
                            else request.scene_world_state.degraded_reason_codes[:1]
                            or ["scene_not_ready"]
                        ),
                        supporting_event_ids=latest_episode_event_ids,
                        backing_ids=[record.commitment_id, *latest_episode_backing],
                        commitment_id=record.commitment_id,
                        details={
                            "prediction_role": "opportunity" if ready else "blocker",
                            "commitment_status": record.status,
                        },
                    )
                )

        proposals_by_kind: dict[str, list[WorldModelPredictionProposal]] = {}
        for record in candidates:
            proposals_by_kind.setdefault(record.prediction_kind, []).append(record)
        bounded: list[WorldModelPredictionProposal] = []
        for kind, records in proposals_by_kind.items():
            bounded.extend(records[: _PREDICTION_CAPS.get(kind, len(records))])
        bounded = sorted(
            bounded,
            key=lambda record: (
                record.prediction_kind,
                record.subject_kind,
                record.subject_id,
                record.summary,
            ),
        )
        if bounded:
            return WorldModelAdapterResponse(proposals=tuple(bounded))
        return WorldModelAdapterResponse(
            proposals=(),
            available=True,
            reason="no_prediction_proposals",
            warnings=("no_prediction_proposals",),
        )


__all__ = [
    "LocalDeterministicWorldModelAdapter",
    "WorldModelAdapter",
    "WorldModelAdapterRequest",
    "WorldModelAdapterResponse",
    "WorldModelPredictionProposal",
]

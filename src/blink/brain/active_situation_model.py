"""Provider-light active situation-model projection helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2 import (
    BrainContinuityDossierProjection,
    BrainContinuityGraphProjection,
    BrainProceduralSkillProjection,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainAgendaProjection,
    BrainCommitmentProjection,
    BrainEngagementStateProjection,
    BrainPlanProposal,
    BrainPredictiveWorldModelProjection,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneStateProjection,
    BrainSceneWorldProjection,
)

_SITUATION_CAPS = {
    BrainActiveSituationRecordKind.SCENE_STATE.value: 4,
    BrainActiveSituationRecordKind.WORLD_STATE.value: 6,
    BrainActiveSituationRecordKind.GOAL_STATE.value: 4,
    BrainActiveSituationRecordKind.COMMITMENT_STATE.value: 6,
    BrainActiveSituationRecordKind.PLAN_STATE.value: 6,
    BrainActiveSituationRecordKind.PROCEDURAL_STATE.value: 4,
    BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value: 6,
    BrainActiveSituationRecordKind.PREDICTION_STATE.value: 8,
}
_ACTIVE_GOAL_STATUSES = {"open", "planning", "in_progress", "retry", "waiting"}


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _max_ts(values: Iterable[str | None]) -> str | None:
    parsed = [item for item in (_parse_ts(value) for value in values) if item is not None]
    if not parsed:
        return None
    return max(parsed).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted(
        {text for value in values if (text := _optional_text(value)) is not None}
    )


def _stable_record_id(
    *,
    record_kind: str,
    summary: str,
    private_record_ids: Iterable[str],
    backing_ids: Iterable[str],
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
) -> str:
    payload = {
        "record_kind": record_kind,
        "summary": summary.strip(),
        "private_record_ids": list(_sorted_unique(private_record_ids)),
        "backing_ids": list(_sorted_unique(backing_ids)),
        "goal_id": goal_id,
        "commitment_id": commitment_id,
        "plan_proposal_id": plan_proposal_id,
        "skill_id": skill_id,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return f"situation_{digest.hexdigest()[:20]}"


def _record(
    *,
    record_kind: str,
    summary: str,
    state: str,
    evidence_kind: str,
    reference_ts: str,
    confidence: float | None = None,
    freshness: str | None = None,
    uncertainty_codes: Iterable[str] = (),
    private_record_ids: Iterable[str] = (),
    backing_ids: Iterable[str] = (),
    source_event_ids: Iterable[str] = (),
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
    observed_at: str | None = None,
    updated_at: str | None = None,
    expires_at: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainActiveSituationRecord:
    normalized_private_record_ids = _sorted_unique(private_record_ids)
    normalized_backing_ids = _sorted_unique(backing_ids)
    return BrainActiveSituationRecord(
        record_id=_stable_record_id(
            record_kind=record_kind,
            summary=summary,
            private_record_ids=normalized_private_record_ids,
            backing_ids=normalized_backing_ids,
            goal_id=goal_id,
            commitment_id=commitment_id,
            plan_proposal_id=plan_proposal_id,
            skill_id=skill_id,
        ),
        record_kind=record_kind,
        summary=summary.strip(),
        state=state,
        evidence_kind=evidence_kind,
        confidence=confidence,
        freshness=freshness,
        uncertainty_codes=_sorted_unique(uncertainty_codes),
        private_record_ids=normalized_private_record_ids,
        backing_ids=normalized_backing_ids,
        source_event_ids=_sorted_unique(source_event_ids),
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_id=skill_id,
        observed_at=observed_at,
        updated_at=updated_at or reference_ts,
        expires_at=expires_at,
        details=dict(details or {}),
    )


def _cap_records(
    *,
    records: list[tuple[float, BrainActiveSituationRecord]],
    limit: int,
) -> list[BrainActiveSituationRecord]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item[0]),
            str(item[1].updated_at or ""),
            item[1].record_kind,
            item[1].record_id,
        ),
        reverse=False,
    )
    return [record for _, record in ordered[:limit]]


def _cap_prediction_records(
    *,
    records: list[tuple[float, BrainActiveSituationRecord]],
    limit: int,
) -> list[BrainActiveSituationRecord]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item[0]),
            str(item[1].details.get("prediction_kind", "")),
            str(item[1].details.get("subject_id", "")),
            str(item[1].details.get("resolution_kind", "")),
            str(item[1].details.get("prediction_id", "")),
        ),
    )
    return [record for _, record in ordered[:limit]]


def _prediction_backing_ids(prediction: BrainPredictionRecord) -> list[str]:
    return [
        backing_id
        for backing_id in prediction.backing_ids
        if not str(backing_id).startswith("autobio_")
        and not str(backing_id).endswith(":presence")
    ]


def _resolve_reference_ts(
    *,
    reference_ts: str | None,
    private_working_memory: BrainPrivateWorkingMemoryProjection,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    body: BrainPresenceSnapshot,
    recent_events: list[BrainEventRecord],
) -> str:
    return (
        reference_ts
        or _max_ts(
            [
                recent_events[0].ts if recent_events else None,
                private_working_memory.updated_at,
                agenda.updated_at,
                commitment_projection.updated_at,
                scene.updated_at,
                engagement.updated_at,
                body.updated_at,
            ]
        )
        or _utc_now()
    )


def _planning_event_index(
    recent_events: list[BrainEventRecord],
) -> tuple[
    dict[str, tuple[BrainEventRecord, BrainPlanProposal]],
    dict[str, BrainEventRecord],
    dict[str, BrainEventRecord],
]:
    proposed_by_id: dict[str, tuple[BrainEventRecord, BrainPlanProposal]] = {}
    adopted_by_id: dict[str, BrainEventRecord] = {}
    rejected_by_id: dict[str, BrainEventRecord] = {}
    for event in sorted(recent_events, key=lambda item: (int(getattr(item, "id", 0)), item.ts, item.event_id)):
        if event.event_type not in {
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.PLANNING_REJECTED,
        }:
            continue
        proposal = BrainPlanProposal.from_dict((event.payload or {}).get("proposal"))
        if proposal is None:
            continue
        if event.event_type == BrainEventType.PLANNING_PROPOSED:
            proposed_by_id[proposal.plan_proposal_id] = (event, proposal)
        elif event.event_type == BrainEventType.PLANNING_ADOPTED:
            adopted_by_id[proposal.plan_proposal_id] = event
        elif event.event_type == BrainEventType.PLANNING_REJECTED:
            rejected_by_id[proposal.plan_proposal_id] = event
    return proposed_by_id, adopted_by_id, rejected_by_id


def _scene_is_fresh(scene: BrainSceneStateProjection, *, reference_dt: datetime | None) -> bool:
    if reference_dt is None:
        reference_dt = _parse_ts(scene.updated_at)
    fresh_dt = _parse_ts(scene.last_fresh_frame_at) or _parse_ts(scene.updated_at)
    if fresh_dt is None or reference_dt is None:
        return bool(scene.camera_connected and scene.person_present != "uncertain")
    if scene.frame_age_ms is not None:
        return int(scene.frame_age_ms) <= 15_000
    return abs((reference_dt - fresh_dt).total_seconds()) <= 15


def _private_records_by_buffer(
    private_working_memory: BrainPrivateWorkingMemoryProjection,
) -> dict[str, list[BrainPrivateWorkingMemoryRecord]]:
    grouped: dict[str, list[BrainPrivateWorkingMemoryRecord]] = {}
    for record in private_working_memory.records:
        grouped.setdefault(record.buffer_kind, []).append(record)
    return grouped


def _build_scene_state_records(
    *,
    private_records_by_buffer: dict[str, list[BrainPrivateWorkingMemoryRecord]],
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    body: BrainPresenceSnapshot,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    reference_dt = _parse_ts(reference_ts)
    scene_records = [
        record
        for record in private_records_by_buffer.get("scene_world_state", [])
        if record.state != BrainPrivateWorkingMemoryRecordState.RESOLVED.value
    ]
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    if scene_records:
        for record in scene_records:
            uncertainty_codes: list[str] = []
            state = BrainActiveSituationRecordState.ACTIVE.value
            freshness = "current"
            if record.state == BrainPrivateWorkingMemoryRecordState.STALE.value:
                state = BrainActiveSituationRecordState.STALE.value
                freshness = "stale"
                uncertainty_codes.append("scene_stale")
            if scene.person_present == "uncertain":
                uncertainty_codes.append("person_presence_uncertain")
            candidates.append(
                (
                    7.0,
                    _record(
                        record_kind=BrainActiveSituationRecordKind.SCENE_STATE.value,
                        summary=record.summary,
                        state=state,
                        evidence_kind=record.evidence_kind,
                        reference_ts=reference_ts,
                        confidence=scene.confidence,
                        freshness=freshness,
                        uncertainty_codes=uncertainty_codes,
                        private_record_ids=[record.record_id],
                        backing_ids=record.backing_ids,
                        source_event_ids=record.source_event_ids,
                        observed_at=record.observed_at,
                        updated_at=record.updated_at,
                        expires_at=record.expires_at,
                        details={
                            "camera_connected": scene.camera_connected,
                            "camera_track_state": scene.camera_track_state,
                            "engagement_state": engagement.engagement_state,
                            "vision_connected": body.vision_connected,
                            **dict(record.details),
                        },
                    ),
                )
            )
    if not candidates:
        scene_fresh = _scene_is_fresh(scene, reference_dt=reference_dt)
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.ACTIVE.value
        freshness = "current"
        if not scene_fresh:
            state = BrainActiveSituationRecordState.STALE.value
            freshness = "stale"
            uncertainty_codes.append("scene_stale")
        if scene.person_present == "uncertain":
            uncertainty_codes.append("person_presence_uncertain")
        summary = (
            scene.last_visual_summary
            or f"Scene {scene.scene_change_state}, person={scene.person_present}"
        )
        candidates.append(
            (
                6.0,
                _record(
                    record_kind=BrainActiveSituationRecordKind.SCENE_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.OBSERVED.value,
                    reference_ts=reference_ts,
                    confidence=scene.confidence,
                    freshness=freshness,
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=["scene_state", "engagement_state", "body_state"],
                    source_event_ids=[],
                    observed_at=scene.last_observed_at or scene.updated_at,
                    updated_at=scene.updated_at,
                    details={
                        "camera_connected": scene.camera_connected,
                        "camera_track_state": scene.camera_track_state,
                        "person_present": scene.person_present,
                        "scene_change_state": scene.scene_change_state,
                        "sensor_health_reason": scene.sensor_health_reason,
                        "engagement_state": engagement.engagement_state,
                    },
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.SCENE_STATE.value],
    )


def _build_goal_state_records(
    *,
    agenda: BrainAgendaProjection,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for goal in agenda.goals:
        if goal.status not in (_ACTIVE_GOAL_STATUSES | {"blocked", "failed"}):
            continue
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.ACTIVE.value
        freshness = "current"
        score = 8.0
        if goal.status in {"blocked", "failed"}:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            freshness = goal.status
            score = 9.0
            if goal.blocked_reason is not None:
                uncertainty_codes.append(
                    _optional_text(goal.blocked_reason.kind) or "blocked_reason"
                )
        elif goal.status == "waiting":
            state = BrainActiveSituationRecordState.STALE.value
            freshness = "waiting"
            score = 6.0
        summary = f"{goal.title} [{goal.status}]"
        if goal.blocked_reason is not None:
            summary = f"{summary}: {goal.blocked_reason.summary}"
        candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.GOAL_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness=freshness,
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[goal.goal_id, *( [goal.commitment_id] if goal.commitment_id else [] )],
                    source_event_ids=[],
                    goal_id=goal.goal_id,
                    commitment_id=goal.commitment_id,
                    observed_at=goal.created_at,
                    updated_at=goal.updated_at,
                    details={
                        "goal_status": goal.status,
                        "goal_family": goal.goal_family,
                        "plan_revision": goal.plan_revision,
                        "planning_requested": goal.planning_requested,
                        "active_step_index": goal.active_step_index,
                    },
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.GOAL_STATE.value],
    )


def _build_world_state_records(
    *,
    scene_world_state: BrainSceneWorldProjection | None,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    if scene_world_state is None:
        return []
    affordances_by_entity: dict[str, list[Any]] = {}
    for affordance in scene_world_state.affordances:
        affordances_by_entity.setdefault(affordance.entity_id, []).append(affordance)
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for entity in scene_world_state.entities:
        if entity.entity_kind == "zone" and entity.state not in {"active", "stale"}:
            continue
        uncertainty_codes = list(entity.contradiction_codes)
        state = BrainActiveSituationRecordState.ACTIVE.value
        freshness = entity.freshness or "current"
        score = 8.0
        if entity.state == "stale":
            state = BrainActiveSituationRecordState.STALE.value
            freshness = "stale"
            score = 6.5
        elif entity.state in {"contradicted", "expired"}:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            freshness = entity.state
            score = 8.5
            uncertainty_codes.append(f"world_{entity.state}")
        affordances = affordances_by_entity.get(entity.entity_id, [])
        affordance_bits = [
            f"{record.capability_family}:{record.availability}"
            for record in affordances[:2]
        ]
        summary = entity.summary
        if affordance_bits:
            summary = f"{summary} [affordances: {', '.join(affordance_bits)}]"
        candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.WORLD_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=entity.evidence_kind,
                    reference_ts=reference_ts,
                    confidence=entity.confidence,
                    freshness=freshness,
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[
                        entity.entity_id,
                        *entity.affordance_ids[:2],
                    ],
                    source_event_ids=entity.source_event_ids,
                    observed_at=entity.observed_at,
                    updated_at=entity.updated_at,
                    expires_at=entity.expires_at,
                    details={
                        "entity_kind": entity.entity_kind,
                        "zone_id": entity.zone_id,
                        "degraded_mode": scene_world_state.degraded_mode,
                        "degraded_reason_codes": list(scene_world_state.degraded_reason_codes),
                        "affordance_ids": list(entity.affordance_ids),
                    },
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.WORLD_STATE.value],
    )


def _build_commitment_state_records(
    *,
    commitment_projection: BrainCommitmentProjection,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for record in [
        *commitment_projection.active_commitments,
        *commitment_projection.deferred_commitments,
        *commitment_projection.blocked_commitments,
    ]:
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.ACTIVE.value
        freshness = "current"
        score = 8.0
        if record.status == "deferred":
            state = BrainActiveSituationRecordState.STALE.value
            freshness = "deferred"
            score = 6.5
            uncertainty_codes.append("deferred_commitment")
        elif record.status == "blocked" or record.blocked_reason is not None:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            freshness = "blocked"
            score = 9.0
            if record.blocked_reason is not None:
                uncertainty_codes.append(
                    _optional_text(record.blocked_reason.kind) or "blocked_reason"
                )
        details = dict(record.details)
        review_policy = _optional_text(details.get("plan_review_policy"))
        if review_policy in {"needs_user_review", "needs_operator_review"}:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            uncertainty_codes.append(review_policy)
            score = max(score, 8.5)
        summary = f"{record.title} [{record.status}]"
        if record.blocked_reason is not None:
            summary = f"{summary}: {record.blocked_reason.summary}"
        candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.COMMITMENT_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness=freshness,
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[
                        record.commitment_id,
                        *(
                            [details.get("pending_plan_proposal_id")]
                            if _optional_text(details.get("pending_plan_proposal_id")) is not None
                            else []
                        ),
                        *(
                            [details.get("current_plan_proposal_id")]
                            if _optional_text(details.get("current_plan_proposal_id")) is not None
                            else []
                        ),
                    ],
                    source_event_ids=[],
                    goal_id=record.current_goal_id,
                    commitment_id=record.commitment_id,
                    plan_proposal_id=(
                        _optional_text(details.get("pending_plan_proposal_id"))
                        or _optional_text(details.get("current_plan_proposal_id"))
                    ),
                    observed_at=record.created_at,
                    updated_at=record.updated_at,
                    details={
                        "goal_family": record.goal_family,
                        "intent": record.intent,
                        "resume_count": record.resume_count,
                        "plan_revision": record.plan_revision,
                        "review_policy": review_policy,
                    },
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.COMMITMENT_STATE.value],
    )


def _build_plan_state_records(
    *,
    planning_digest: dict[str, Any],
    recent_events: list[BrainEventRecord],
    active_goal_ids: set[str],
    active_commitment_ids: set[str],
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    proposed_by_id, adopted_by_id, rejected_by_id = _planning_event_index(recent_events)
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []

    for proposal_state in planning_digest.get("current_pending_proposals", []):
        proposal_id = _optional_text(proposal_state.get("plan_proposal_id"))
        if proposal_id is None:
            continue
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.ACTIVE.value
        score = 9.0
        missing_inputs = [
            str(item).strip()
            for item in proposal_state.get("missing_inputs", [])
            if str(item).strip()
        ]
        if missing_inputs:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            uncertainty_codes.append("missing_input")
        review_policy = _optional_text(proposal_state.get("review_policy"))
        if review_policy in {"needs_user_review", "needs_operator_review"}:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            uncertainty_codes.append(review_policy)
        for reason in proposal_state.get("skill_rejection_reasons", []):
            reason_text = _optional_text(reason)
            if reason_text is not None:
                state = BrainActiveSituationRecordState.UNRESOLVED.value
                uncertainty_codes.append(reason_text)
        proposed_event, proposal = proposed_by_id.get(proposal_id, (None, None))
        summary = f"{proposal_state.get('title') or proposal_state.get('summary') or proposal_id}"
        if missing_inputs:
            summary = f"{summary}: Missing input {missing_inputs[0]}"
        candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness="pending",
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[proposal_id],
                    source_event_ids=(
                        [proposed_event.event_id] if proposed_event is not None else []
                    )
                    + ([rejected_by_id[proposal_id].event_id] if proposal_id in rejected_by_id else []),
                    goal_id=_optional_text(proposal_state.get("goal_id")),
                    commitment_id=_optional_text(proposal_state.get("commitment_id")),
                    plan_proposal_id=proposal_id,
                    skill_id=_optional_text(proposal_state.get("selected_skill_id")),
                    observed_at=(proposal.created_at if proposal is not None else None),
                    updated_at=(
                        proposed_event.ts
                        if proposed_event is not None
                        else reference_ts
                    ),
                    details={
                        "review_policy": review_policy,
                        "missing_inputs": missing_inputs,
                        "assumptions": list(proposal_state.get("assumptions", [])),
                        "procedural_origin": _optional_text(
                            proposal_state.get("procedural_origin")
                        ),
                    },
                ),
            )
        )

    for state in planning_digest.get("current_plan_states", []):
        proposal_id = _optional_text(state.get("current_plan_proposal_id"))
        pending_id = _optional_text(state.get("pending_plan_proposal_id"))
        if proposal_id is None or pending_id is not None:
            continue
        if _optional_text(state.get("goal_id")) not in active_goal_ids and _optional_text(
            state.get("commitment_id")
        ) not in active_commitment_ids:
            continue
        candidates.append(
            (
                7.0,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary=f"{state.get('title') or proposal_id}: current plan revision {state.get('plan_revision')}",
                    state=BrainActiveSituationRecordState.ACTIVE.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness="current",
                    backing_ids=[proposal_id],
                    source_event_ids=(
                        [proposed_by_id[proposal_id][0].event_id]
                        if proposal_id in proposed_by_id
                        else []
                    )
                    + ([adopted_by_id[proposal_id].event_id] if proposal_id in adopted_by_id else []),
                    goal_id=_optional_text(state.get("goal_id")),
                    commitment_id=_optional_text(state.get("commitment_id")),
                    plan_proposal_id=proposal_id,
                    observed_at=(
                        proposed_by_id[proposal_id][1].created_at
                        if proposal_id in proposed_by_id
                        else None
                    ),
                    updated_at=(
                        adopted_by_id[proposal_id].ts
                        if proposal_id in adopted_by_id
                        else reference_ts
                    ),
                    details={
                        "plan_revision": state.get("plan_revision"),
                        "goal_status": state.get("goal_status"),
                        "commitment_status": state.get("commitment_status"),
                    },
                ),
            )
        )

    for rejected in list(planning_digest.get("recent_rejections", []))[-3:]:
        goal_id = _optional_text(rejected.get("goal_id"))
        commitment_id = _optional_text(rejected.get("commitment_id"))
        if goal_id not in active_goal_ids and commitment_id not in active_commitment_ids:
            continue
        proposal_id = _optional_text(rejected.get("plan_proposal_id"))
        if proposal_id is None:
            continue
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.STALE.value
        decision_reason = _optional_text(rejected.get("decision_reason"))
        if decision_reason in {"missing_required_input", "needs_user_review", "needs_operator_review"}:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            uncertainty_codes.append(decision_reason)
        for reason in rejected.get("skill_rejection_reasons", []):
            reason_text = _optional_text(reason)
            if reason_text is not None:
                state = BrainActiveSituationRecordState.UNRESOLVED.value
                uncertainty_codes.append(reason_text)
        candidates.append(
            (
                6.5 if state == BrainActiveSituationRecordState.STALE.value else 8.0,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary=f"Rejected plan: {rejected.get('summary') or rejected.get('title') or proposal_id}",
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness="rejected",
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[proposal_id],
                    source_event_ids=(
                        [rejected_by_id[proposal_id].event_id] if proposal_id in rejected_by_id else []
                    ),
                    goal_id=goal_id,
                    commitment_id=commitment_id,
                    plan_proposal_id=proposal_id,
                    skill_id=_optional_text(rejected.get("selected_skill_id")),
                    observed_at=_optional_text(rejected.get("created_at")),
                    updated_at=_optional_text(rejected.get("ts")) or reference_ts,
                    details={
                        "decision_reason": decision_reason,
                        "review_policy": _optional_text(rejected.get("review_policy")),
                        "procedural_origin": _optional_text(rejected.get("procedural_origin")),
                    },
                ),
            )
        )

    for revision in list(planning_digest.get("recent_revision_flows", []))[-3:]:
        proposal_id = _optional_text(revision.get("plan_proposal_id"))
        if proposal_id is None:
            continue
        goal_id = _optional_text(revision.get("goal_id"))
        commitment_id = _optional_text(revision.get("commitment_id"))
        if goal_id not in active_goal_ids and commitment_id not in active_commitment_ids:
            continue
        if _optional_text(revision.get("supersedes_plan_proposal_id")) is None:
            continue
        candidates.append(
            (
                5.5,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary=f"Superseded plan revision for {revision.get('title') or proposal_id}",
                    state=BrainActiveSituationRecordState.STALE.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness="superseded",
                    uncertainty_codes=["superseded_proposal"],
                    backing_ids=[
                        proposal_id,
                        _optional_text(revision.get("supersedes_plan_proposal_id")),
                    ],
                    source_event_ids=(
                        [proposed_by_id[proposal_id][0].event_id]
                        if proposal_id in proposed_by_id
                        else []
                    ),
                    goal_id=goal_id,
                    commitment_id=commitment_id,
                    plan_proposal_id=proposal_id,
                    skill_id=_optional_text(revision.get("selected_skill_id")),
                    observed_at=_optional_text(revision.get("created_at")),
                    updated_at=_optional_text(revision.get("ts")) or reference_ts,
                    details={
                        "preserved_prefix_count": int(revision.get("preserved_prefix_count") or 0),
                        "procedural_origin": _optional_text(revision.get("procedural_origin")),
                    },
                ),
            )
        )

    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.PLAN_STATE.value],
    )


def _build_procedural_state_records(
    *,
    procedural_skills: BrainProceduralSkillProjection | None,
    planning_digest: dict[str, Any],
    active_commitment_ids: set[str],
    active_plan_proposal_ids: set[str],
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    if procedural_skills is None:
        return []
    selected_skill_ids = {
        _optional_text(item)
        for item in planning_digest.get("recent_selected_skill_ids", [])
    }
    for item in planning_digest.get("current_pending_proposals", []):
        selected_skill_ids.add(_optional_text(item.get("selected_skill_id")))
    relevant_skill_ids = {item for item in selected_skill_ids if item is not None}
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for skill in procedural_skills.skills:
        if (
            skill.skill_id not in relevant_skill_ids
            and not set(skill.supporting_plan_proposal_ids).intersection(active_plan_proposal_ids)
            and not set(skill.supporting_commitment_ids).intersection(active_commitment_ids)
        ):
            continue
        uncertainty_codes: list[str] = []
        state = BrainActiveSituationRecordState.ACTIVE.value
        freshness = "current"
        score = 7.5
        if skill.status in {"retired", "superseded"}:
            state = BrainActiveSituationRecordState.STALE.value
            freshness = skill.status
            score = 5.5
            uncertainty_codes.append(skill.status)
        if float(skill.confidence) < 0.5:
            state = BrainActiveSituationRecordState.UNRESOLVED.value
            uncertainty_codes.append("skill_low_confidence")
            score = 8.0
        if skill.skill_id in {
            _optional_text(item)
            for item in planning_digest.get("recent_selected_skill_ids", [])
        }:
            score += 0.5
        summary = f"{skill.title or skill.skill_id} [{skill.status}]"
        candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PROCEDURAL_STATE.value,
                    summary=summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    confidence=float(skill.confidence),
                    freshness=freshness,
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=[
                        skill.skill_id,
                        *skill.supporting_plan_proposal_ids[:2],
                        *skill.supporting_trace_ids[:2],
                    ],
                    source_event_ids=[],
                    commitment_id=next(iter(skill.supporting_commitment_ids), None),
                    plan_proposal_id=next(iter(skill.supporting_plan_proposal_ids), None),
                    skill_id=skill.skill_id,
                    observed_at=skill.created_at,
                    updated_at=skill.updated_at,
                    details={
                        "status": skill.status,
                        "goal_family": skill.goal_family,
                        "support_trace_count": skill.stats.support_trace_count,
                        "support_plan_proposal_ids": list(skill.supporting_plan_proposal_ids[:3]),
                    },
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.PROCEDURAL_STATE.value],
    )


def _build_uncertainty_state_records(
    *,
    private_records_by_buffer: dict[str, list[BrainPrivateWorkingMemoryRecord]],
    continuity_dossiers: BrainContinuityDossierProjection | None,
    planning_digest: dict[str, Any],
    procedural_skills: BrainProceduralSkillProjection | None,
    scene_world_state: BrainSceneWorldProjection | None,
    scene: BrainSceneStateProjection,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for record in private_records_by_buffer.get("unresolved_uncertainty", []):
        if record.state == BrainPrivateWorkingMemoryRecordState.RESOLVED.value:
            continue
        candidates.append(
            (
                9.0 if record.state == BrainPrivateWorkingMemoryRecordState.ACTIVE.value else 6.5,
                _record(
                    record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                    summary=record.summary,
                    state=(
                        BrainActiveSituationRecordState.UNRESOLVED.value
                        if record.state == BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                        else BrainActiveSituationRecordState.STALE.value
                    ),
                    evidence_kind=record.evidence_kind,
                    reference_ts=reference_ts,
                    freshness=record.state,
                    uncertainty_codes=[
                        _optional_text(record.details.get("kind")) or record.buffer_kind
                    ],
                    private_record_ids=[record.record_id],
                    backing_ids=record.backing_ids,
                    source_event_ids=record.source_event_ids,
                    goal_id=record.goal_id,
                    commitment_id=record.commitment_id,
                    plan_proposal_id=record.plan_proposal_id,
                    skill_id=record.skill_id,
                    observed_at=record.observed_at,
                    updated_at=record.updated_at,
                    expires_at=record.expires_at,
                    details=dict(record.details),
                ),
            )
        )
    if continuity_dossiers is not None:
        for dossier in continuity_dossiers.dossiers:
            uncertainty_codes: list[str] = []
            state: str | None = None
            freshness: str | None = None
            if dossier.contradiction in {"uncertain", "contradicted"}:
                state = BrainActiveSituationRecordState.UNRESOLVED.value
                uncertainty_codes.append(f"dossier_{dossier.contradiction}")
                freshness = dossier.contradiction
            elif dossier.freshness in {"stale", "needs_refresh"}:
                state = (
                    BrainActiveSituationRecordState.UNRESOLVED.value
                    if dossier.freshness == "needs_refresh"
                    else BrainActiveSituationRecordState.STALE.value
                )
                uncertainty_codes.append(f"dossier_{dossier.freshness}")
                freshness = dossier.freshness
            if state is None:
                continue
            candidates.append(
                (
                    8.0 if state == BrainActiveSituationRecordState.UNRESOLVED.value else 6.0,
                    _record(
                        record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                        summary=f"{dossier.title}: {dossier.summary}",
                        state=state,
                        evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                        reference_ts=reference_ts,
                        confidence=float(dossier.support_strength),
                        freshness=freshness,
                        uncertainty_codes=uncertainty_codes,
                        backing_ids=[
                            dossier.dossier_id,
                            *[
                                claim_id
                                for claim_id in dossier.source_claim_ids[:2]
                                if not str(claim_id).startswith("autobio_")
                            ],
                            *[
                                entry_id
                                for entry_id in dossier.source_entry_ids[:2]
                                if not str(entry_id).startswith("autobio_")
                            ],
                        ],
                        source_event_ids=list(dossier.source_event_ids[:3]),
                        observed_at=None,
                        updated_at=reference_ts,
                        details={
                            "dossier_id": dossier.dossier_id,
                            "kind": dossier.kind,
                            "freshness": dossier.freshness,
                            "contradiction": dossier.contradiction,
                        },
                    ),
                )
            )
    scene_stale = not _scene_is_fresh(scene, reference_dt=_parse_ts(reference_ts))
    if scene_stale or scene.person_present == "uncertain":
        uncertainty_codes = []
        if scene_stale:
            uncertainty_codes.append("scene_stale")
        if scene.person_present == "uncertain":
            uncertainty_codes.append("person_presence_uncertain")
        candidates.append(
            (
                7.5,
                _record(
                    record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                    summary=(
                        "Scene freshness degraded"
                        if scene.person_present != "uncertain"
                        else "Scene freshness degraded; person presence remains uncertain"
                    ),
                    state=BrainActiveSituationRecordState.UNRESOLVED.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.HYPOTHESIZED.value,
                    reference_ts=reference_ts,
                    freshness="stale" if scene_stale else "uncertain",
                    uncertainty_codes=uncertainty_codes,
                    backing_ids=["scene_state"],
                    source_event_ids=[],
                    observed_at=scene.last_observed_at or scene.updated_at,
                    updated_at=scene.updated_at,
                    details={
                        "camera_track_state": scene.camera_track_state,
                        "frame_age_ms": scene.frame_age_ms,
                        "sensor_health_reason": scene.sensor_health_reason,
                    },
                ),
            )
        )
    if scene_world_state is not None:
        if scene_world_state.degraded_mode != "healthy":
            candidates.append(
                (
                    8.25,
                    _record(
                        record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                        summary=f"Scene world state is running in {scene_world_state.degraded_mode} mode",
                        state=BrainActiveSituationRecordState.UNRESOLVED.value,
                        evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                        reference_ts=reference_ts,
                        freshness=scene_world_state.degraded_mode,
                        uncertainty_codes=list(scene_world_state.degraded_reason_codes),
                        backing_ids=["scene_world_state"],
                        source_event_ids=[],
                        updated_at=scene_world_state.updated_at,
                        details={"degraded_mode": scene_world_state.degraded_mode},
                    ),
                )
            )
        for entity in scene_world_state.entities:
            if entity.state != "contradicted":
                continue
            candidates.append(
                (
                    8.0,
                    _record(
                        record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                        summary=f"Scene contradiction remains unresolved: {entity.summary}",
                        state=BrainActiveSituationRecordState.UNRESOLVED.value,
                        evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                        reference_ts=reference_ts,
                        confidence=entity.confidence,
                        freshness="contradicted",
                        uncertainty_codes=list(entity.contradiction_codes) or ["world_contradicted"],
                        backing_ids=[entity.entity_id, *(entity.affordance_ids[:2])],
                        source_event_ids=entity.source_event_ids,
                        observed_at=entity.observed_at,
                        updated_at=entity.updated_at,
                        expires_at=entity.expires_at,
                        details={"entity_kind": entity.entity_kind, "zone_id": entity.zone_id},
                    ),
                )
            )
    for pending in planning_digest.get("current_pending_proposals", []):
        missing_inputs = [
            str(item).strip()
            for item in pending.get("missing_inputs", [])
            if str(item).strip()
        ]
        if not missing_inputs and not pending.get("skill_rejection_reasons"):
            continue
        proposal_id = _optional_text(pending.get("plan_proposal_id"))
        if proposal_id is None:
            continue
        codes = ["missing_input"] * bool(missing_inputs)
        codes.extend(
            reason
            for reason in (
                _optional_text(item) for item in pending.get("skill_rejection_reasons", [])
            )
            if reason is not None
        )
        candidates.append(
            (
                8.5,
                _record(
                    record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                    summary=f"Pending plan still needs resolution: {pending.get('title') or proposal_id}",
                    state=BrainActiveSituationRecordState.UNRESOLVED.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    reference_ts=reference_ts,
                    freshness="pending",
                    uncertainty_codes=codes,
                    backing_ids=[proposal_id],
                    source_event_ids=[],
                    goal_id=_optional_text(pending.get("goal_id")),
                    commitment_id=_optional_text(pending.get("commitment_id")),
                    plan_proposal_id=proposal_id,
                    skill_id=_optional_text(pending.get("selected_skill_id")),
                    observed_at=_optional_text(pending.get("created_at")),
                    updated_at=reference_ts,
                    details={
                        "missing_inputs": missing_inputs,
                        "skill_rejection_reasons": list(pending.get("skill_rejection_reasons", [])),
                    },
                ),
            )
        )
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            if float(skill.confidence) >= 0.5 or skill.status not in {"candidate", "active"}:
                continue
            candidates.append(
                (
                    7.0,
                    _record(
                        record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                        summary=f"Procedural confidence remains low: {skill.title or skill.skill_id}",
                        state=BrainActiveSituationRecordState.UNRESOLVED.value,
                        evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                        reference_ts=reference_ts,
                        confidence=float(skill.confidence),
                        freshness="current",
                        uncertainty_codes=["skill_low_confidence"],
                        backing_ids=[skill.skill_id, *skill.supporting_trace_ids[:2]],
                        source_event_ids=[],
                        plan_proposal_id=next(iter(skill.supporting_plan_proposal_ids), None),
                        skill_id=skill.skill_id,
                        observed_at=skill.created_at,
                        updated_at=skill.updated_at,
                        details={"status": skill.status},
                    ),
                )
            )
    return _cap_records(
        records=candidates,
        limit=_SITUATION_CAPS[BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value],
    )


def _build_prediction_state_records(
    *,
    predictive_world_model: BrainPredictiveWorldModelProjection | None,
    reference_ts: str,
) -> list[BrainActiveSituationRecord]:
    if predictive_world_model is None:
        return []
    active_candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    recent_candidates: list[tuple[float, BrainActiveSituationRecord]] = []
    for prediction in predictive_world_model.active_predictions:
        prediction_role = _optional_text(prediction.details.get("prediction_role")) or (
            "blocker" if prediction.risk_codes else "opportunity"
        )
        unresolved = prediction_role == "blocker"
        state = (
            BrainActiveSituationRecordState.UNRESOLVED.value
            if unresolved
            else BrainActiveSituationRecordState.ACTIVE.value
        )
        score = 9.25 if unresolved else 8.25
        active_candidates.append(
            (
                score,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PREDICTION_STATE.value,
                    summary=prediction.summary,
                    state=state,
                    evidence_kind=BrainActiveSituationEvidenceKind.HYPOTHESIZED.value,
                    reference_ts=reference_ts,
                    confidence=prediction.confidence,
                    freshness="predicted",
                    uncertainty_codes=prediction.risk_codes[:3],
                    backing_ids=_prediction_backing_ids(prediction),
                    source_event_ids=prediction.supporting_event_ids,
                    goal_id=prediction.goal_id,
                    commitment_id=prediction.commitment_id,
                    plan_proposal_id=prediction.plan_proposal_id,
                    skill_id=prediction.skill_id,
                    observed_at=prediction.predicted_at,
                    updated_at=prediction.updated_at,
                    expires_at=prediction.valid_to,
                    details={
                        "prediction_id": prediction.prediction_id,
                        "prediction_kind": prediction.prediction_kind,
                        "prediction_role": prediction_role,
                        "confidence_band": prediction.confidence_band,
                        "subject_kind": prediction.subject_kind,
                        "subject_id": prediction.subject_id,
                        "resolution_kind": prediction.resolution_kind,
                        "predicted_state": dict(prediction.predicted_state),
                        "risk_codes": list(prediction.risk_codes),
                        **dict(prediction.details),
                    },
                ),
            )
        )
    for prediction in predictive_world_model.recent_resolutions:
        resolution_kind = _optional_text(prediction.resolution_kind) or "invalidated"
        prediction_role = _optional_text(prediction.details.get("prediction_role")) or (
            "blocker" if resolution_kind in {"invalidated", "expired"} else "opportunity"
        )
        recent_candidates.append(
            (
                9.1 if resolution_kind in {"invalidated", "expired"} else 7.0,
                _record(
                    record_kind=BrainActiveSituationRecordKind.PREDICTION_STATE.value,
                    summary=prediction.resolution_summary or prediction.summary,
                    state=(
                        BrainActiveSituationRecordState.UNRESOLVED.value
                        if resolution_kind in {"invalidated", "expired"}
                        else BrainActiveSituationRecordState.STALE.value
                    ),
                    evidence_kind=BrainActiveSituationEvidenceKind.HYPOTHESIZED.value,
                    reference_ts=reference_ts,
                    confidence=prediction.confidence,
                    freshness="predicted",
                    uncertainty_codes=prediction.risk_codes[:3],
                    backing_ids=_prediction_backing_ids(prediction),
                    source_event_ids=prediction.supporting_event_ids,
                    goal_id=prediction.goal_id,
                    commitment_id=prediction.commitment_id,
                    plan_proposal_id=prediction.plan_proposal_id,
                    skill_id=prediction.skill_id,
                    observed_at=prediction.predicted_at,
                    updated_at=prediction.updated_at,
                    expires_at=prediction.valid_to,
                    details={
                        "prediction_id": prediction.prediction_id,
                        "prediction_kind": prediction.prediction_kind,
                        "prediction_role": prediction_role,
                        "confidence_band": prediction.confidence_band,
                        "subject_kind": prediction.subject_kind,
                        "subject_id": prediction.subject_id,
                        "resolution_kind": prediction.resolution_kind,
                        "resolution_event_ids": list(prediction.resolution_event_ids),
                        "resolution_summary": prediction.resolution_summary,
                        "predicted_state": dict(prediction.predicted_state),
                        "risk_codes": list(prediction.risk_codes),
                        **dict(prediction.details),
                    },
                ),
            )
        )
    return [
        *_cap_prediction_records(records=active_candidates, limit=4),
        *_cap_prediction_records(records=recent_candidates, limit=4),
    ][: _SITUATION_CAPS[BrainActiveSituationRecordKind.PREDICTION_STATE.value]]


def build_active_situation_model_projection(
    *,
    scope_type: str,
    scope_id: str,
    private_working_memory: BrainPrivateWorkingMemoryProjection,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    scene: BrainSceneStateProjection,
    scene_world_state: BrainSceneWorldProjection | None = None,
    engagement: BrainEngagementStateProjection,
    body: BrainPresenceSnapshot,
    continuity_graph: BrainContinuityGraphProjection | None,
    continuity_dossiers: BrainContinuityDossierProjection | None,
    procedural_skills: BrainProceduralSkillProjection | None,
    recent_events: list[BrainEventRecord],
    planning_digest: dict[str, Any],
    predictive_world_model: BrainPredictiveWorldModelProjection | None = None,
    reference_ts: str | None = None,
) -> BrainActiveSituationProjection:
    """Build a bounded active situation-model projection from landed state."""
    del continuity_graph
    resolved_reference_ts = _resolve_reference_ts(
        reference_ts=reference_ts,
        private_working_memory=private_working_memory,
        agenda=agenda,
        commitment_projection=commitment_projection,
        scene=scene,
        engagement=engagement,
        body=body,
        recent_events=recent_events,
    )
    private_records_by_buffer = _private_records_by_buffer(private_working_memory)
    active_goal_ids = {
        goal.goal_id
        for goal in agenda.goals
        if goal.goal_id and goal.status in (_ACTIVE_GOAL_STATUSES | {"blocked", "failed"})
    }
    active_commitment_ids = {
        record.commitment_id
        for record in [
            *commitment_projection.active_commitments,
            *commitment_projection.deferred_commitments,
            *commitment_projection.blocked_commitments,
        ]
        if record.commitment_id
    }
    active_plan_proposal_ids = {
        item
        for item in (
            _optional_text(entry.get("plan_proposal_id"))
            for entry in planning_digest.get("current_pending_proposals", [])
        )
        if item is not None
    } | {
        item
        for item in (
            _optional_text(entry.get("current_plan_proposal_id"))
            for entry in planning_digest.get("current_plan_states", [])
        )
        if item is not None
    }

    projection = BrainActiveSituationProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        records=[
            *_build_scene_state_records(
                private_records_by_buffer=private_records_by_buffer,
                scene=scene,
                engagement=engagement,
                body=body,
                reference_ts=resolved_reference_ts,
            ),
            *_build_goal_state_records(
                agenda=agenda,
                reference_ts=resolved_reference_ts,
            ),
            *_build_world_state_records(
                scene_world_state=scene_world_state,
                reference_ts=resolved_reference_ts,
            ),
            *_build_commitment_state_records(
                commitment_projection=commitment_projection,
                reference_ts=resolved_reference_ts,
            ),
            *_build_plan_state_records(
                planning_digest=planning_digest,
                recent_events=recent_events,
                active_goal_ids=active_goal_ids,
                active_commitment_ids=active_commitment_ids,
                reference_ts=resolved_reference_ts,
            ),
            *_build_procedural_state_records(
                procedural_skills=procedural_skills,
                planning_digest=planning_digest,
                active_commitment_ids=active_commitment_ids,
                active_plan_proposal_ids=active_plan_proposal_ids,
                reference_ts=resolved_reference_ts,
            ),
            *_build_prediction_state_records(
                predictive_world_model=predictive_world_model,
                reference_ts=resolved_reference_ts,
            ),
            *_build_uncertainty_state_records(
                private_records_by_buffer=private_records_by_buffer,
                continuity_dossiers=continuity_dossiers,
                planning_digest=planning_digest,
                procedural_skills=procedural_skills,
                scene_world_state=scene_world_state,
                scene=scene,
                reference_ts=resolved_reference_ts,
            ),
        ],
        updated_at=resolved_reference_ts,
    )
    projection.sync_lists()
    return projection


__all__ = ["build_active_situation_model_projection"]

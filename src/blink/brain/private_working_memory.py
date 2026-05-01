"""Provider-light private working-memory projection helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2 import (
    BrainContinuityDossierProjection,
    BrainContinuityGraphNodeKind,
    BrainContinuityGraphProjection,
    BrainCoreMemoryBlockRecord,
    BrainProceduralSkillProjection,
)
from blink.brain.planning_digest import build_planning_digest
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainAgendaProjection,
    BrainCommitmentProjection,
    BrainEngagementStateProjection,
    BrainPlanProposal,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneStateProjection,
    BrainSceneWorldProjection,
    BrainWorkingContextProjection,
)

_SCENE_FRESH_MAX_AGE_MS = 15_000
_RECENT_TOOL_OUTCOME_ACTIVE_COUNT = 3
_BUFFER_CAPS = {
    BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value: 4,
    BrainPrivateWorkingMemoryBufferKind.SELF_POLICY.value: 4,
    BrainPrivateWorkingMemoryBufferKind.GOAL_COMMITMENT.value: 6,
    BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value: 6,
    BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value: 4,
    BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value: 6,
    BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value: 6,
}


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


def _event_payload(event: BrainEventRecord) -> dict[str, Any]:
    payload = event.payload
    return payload if isinstance(payload, dict) else {}


def _query_tokens(*parts: str | None) -> tuple[str, ...]:
    joined = " ".join(part for part in parts if part).lower()
    return tuple(sorted({token for token in re.findall(r"[\w-]+", joined) if token}))


def _text_score(*parts: str | None, tokens: tuple[str, ...]) -> float:
    haystack = " ".join(part for part in parts if part).lower()
    score = 0.0
    for token in tokens:
        if token in haystack:
            score += 1.0
    return score


def _stable_record_id(
    *,
    buffer_kind: str,
    summary: str,
    backing_ids: Iterable[str],
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
) -> str:
    payload = {
        "buffer_kind": buffer_kind,
        "summary": summary.strip(),
        "backing_ids": list(_sorted_unique(backing_ids)),
        "goal_id": goal_id,
        "commitment_id": commitment_id,
        "plan_proposal_id": plan_proposal_id,
        "skill_id": skill_id,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return f"pwm_{digest.hexdigest()[:20]}"


def _record(
    *,
    buffer_kind: str,
    summary: str,
    state: str,
    evidence_kind: str,
    backing_ids: Iterable[str],
    source_event_ids: Iterable[str],
    reference_ts: str,
    observed_at: str | None = None,
    updated_at: str | None = None,
    expires_at: str | None = None,
    goal_id: str | None = None,
    commitment_id: str | None = None,
    plan_proposal_id: str | None = None,
    skill_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainPrivateWorkingMemoryRecord:
    normalized_backing_ids = _sorted_unique(backing_ids)
    return BrainPrivateWorkingMemoryRecord(
        record_id=_stable_record_id(
            buffer_kind=buffer_kind,
            summary=summary,
            backing_ids=normalized_backing_ids,
            goal_id=goal_id,
            commitment_id=commitment_id,
            plan_proposal_id=plan_proposal_id,
            skill_id=skill_id,
        ),
        buffer_kind=buffer_kind,
        summary=summary.strip(),
        state=state,
        evidence_kind=evidence_kind,
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
    records: list[tuple[float, BrainPrivateWorkingMemoryRecord]],
    limit: int,
) -> list[BrainPrivateWorkingMemoryRecord]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item[0]),
            item[1].buffer_kind,
            item[1].state,
            item[1].updated_at,
            item[1].record_id,
        ),
    )
    return [record for _, record in ordered[:limit]]


def _resolve_reference_ts(
    *,
    reference_ts: str | None,
    working_context: BrainWorkingContextProjection,
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
                working_context.updated_at,
                agenda.updated_at,
                commitment_projection.updated_at,
                scene.updated_at,
                engagement.updated_at,
                body.updated_at,
            ]
        )
        or _utc_now()
    )


def _dossier_updated_at(dossier: Any, fallback: str) -> str:
    candidates: list[str | None] = []
    for fact in list(getattr(dossier, "recent_changes", []) or []) + list(
        getattr(dossier, "key_current_facts", []) or []
    ):
        candidates.extend(
            [
                getattr(fact, "valid_from", None),
                getattr(fact, "valid_to", None),
                getattr(getattr(fact, "details", {}), "get", lambda *_: None)("updated_at"),
                getattr(getattr(fact, "details", {}), "get", lambda *_: None)("observed_at"),
            ]
        )
    details = getattr(dossier, "details", {}) or {}
    if isinstance(details, dict):
        candidates.extend(
            [
                details.get("updated_at"),
                details.get("observed_at"),
                details.get("compiled_at"),
            ]
        )
    return _max_ts(candidates) or fallback


def _graph_node_updated_at(node: Any, fallback: str) -> str:
    details = getattr(node, "details", {}) or {}
    candidates = [
        getattr(node, "valid_from", None),
        getattr(node, "valid_to", None),
    ]
    if isinstance(details, dict):
        candidates.extend(
            [
                details.get("updated_at"),
                details.get("observed_at"),
                details.get("valid_from"),
                details.get("valid_to"),
            ]
        )
    return _max_ts(candidates) or fallback


def _proposal_event_index(
    recent_events: list[BrainEventRecord],
) -> tuple[
    dict[str, tuple[BrainEventRecord, BrainPlanProposal]],
    dict[str, BrainEventRecord],
    dict[str, BrainEventRecord],
    dict[str, str],
]:
    proposed_by_id: dict[str, tuple[BrainEventRecord, BrainPlanProposal]] = {}
    adopted_by_id: dict[str, BrainEventRecord] = {}
    rejected_by_id: dict[str, BrainEventRecord] = {}
    superseded_by_id: dict[str, str] = {}
    for event in sorted(recent_events, key=lambda item: (int(item.id or 0), item.event_id)):
        payload = _event_payload(event)
        if event.event_type == BrainEventType.PLANNING_PROPOSED:
            proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
            if proposal is None:
                continue
            proposed_by_id[proposal.plan_proposal_id] = (event, proposal)
            if proposal.supersedes_plan_proposal_id:
                superseded_by_id[str(proposal.supersedes_plan_proposal_id)] = proposal.plan_proposal_id
        elif event.event_type == BrainEventType.PLANNING_ADOPTED:
            proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
            if proposal is not None:
                adopted_by_id[proposal.plan_proposal_id] = event
        elif event.event_type == BrainEventType.PLANNING_REJECTED:
            proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
            if proposal is not None:
                rejected_by_id[proposal.plan_proposal_id] = event
    return proposed_by_id, adopted_by_id, rejected_by_id, superseded_by_id


def _goal_event_ids(recent_events: list[BrainEventRecord]) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    event_ids_by_goal_id: dict[str, list[str]] = {}
    goal_payloads_by_goal_id: dict[str, list[dict[str, Any]]] = {}
    for event in sorted(recent_events, key=lambda item: (int(item.id or 0), item.event_id)):
        payload = _event_payload(event)
        goal_payload = payload.get("goal")
        if not isinstance(goal_payload, dict):
            continue
        goal_id = str(goal_payload.get("goal_id", "")).strip()
        if not goal_id:
            continue
        event_ids_by_goal_id.setdefault(goal_id, []).append(event.event_id)
        goal_payloads_by_goal_id.setdefault(goal_id, []).append(
            {
                "event_type": event.event_type,
                "event_id": event.event_id,
                "ts": event.ts,
                "goal": goal_payload,
            }
        )
    return event_ids_by_goal_id, goal_payloads_by_goal_id


def _summarize_block_content(content: dict[str, Any], *, max_items: int = 3) -> str:
    parts: list[str] = []
    for key, value in list(content.items())[:max_items]:
        rendered = value
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value[:3])
        elif isinstance(value, dict):
            rendered = ", ".join(
                f"{item_key}={item_value}" for item_key, item_value in list(value.items())[:3]
            )
        parts.append(f"{key}={rendered}")
    return "; ".join(part for part in parts if part)


def _scene_is_fresh(scene: BrainSceneStateProjection, *, reference_dt: datetime | None) -> bool:
    if reference_dt is None:
        return scene.frame_age_ms is None or int(scene.frame_age_ms) <= _SCENE_FRESH_MAX_AGE_MS
    if scene.last_fresh_frame_at:
        fresh_dt = _parse_ts(scene.last_fresh_frame_at)
        if fresh_dt is not None:
            return (reference_dt - fresh_dt) <= timedelta(milliseconds=_SCENE_FRESH_MAX_AGE_MS)
    if scene.frame_age_ms is not None:
        return int(scene.frame_age_ms) <= _SCENE_FRESH_MAX_AGE_MS
    return bool(scene.camera_connected and scene.person_present != "uncertain")


def _build_user_model_records(
    *,
    continuity_graph: BrainContinuityGraphProjection | None,
    continuity_dossiers: BrainContinuityDossierProjection | None,
    relevance_tokens: tuple[str, ...],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    candidates: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    if continuity_dossiers is not None:
        for dossier in continuity_dossiers.dossiers:
            if dossier.kind not in {"relationship", "project"}:
                continue
            dossier_updated_at = _dossier_updated_at(dossier, reference_ts)
            state = (
                BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                if dossier.freshness == "fresh" and dossier.contradiction == "clear"
                else BrainPrivateWorkingMemoryRecordState.STALE.value
            )
            score = 6.0 if dossier.kind == "relationship" else 5.0
            score += _text_score(
                dossier.title,
                dossier.summary,
                dossier.project_key,
                tokens=relevance_tokens,
            )
            summary = f"{dossier.title}: {dossier.summary}"
            candidates.append(
                (
                    score,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value,
                        summary=summary,
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                        backing_ids=[
                            dossier.dossier_id,
                            *list(dossier.summary_evidence.claim_ids),
                            *list(dossier.summary_evidence.entry_ids),
                        ],
                        source_event_ids=dossier.summary_evidence.source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=dossier_updated_at,
                        updated_at=dossier_updated_at,
                        details={
                            "dossier_id": dossier.dossier_id,
                            "freshness": dossier.freshness,
                            "contradiction": dossier.contradiction,
                            "kind": dossier.kind,
                            "project_key": dossier.project_key,
                        },
                    ),
                )
            )
    if continuity_graph is not None:
        current_node_ids = set(continuity_graph.current_node_ids)
        stale_node_ids = set(continuity_graph.stale_node_ids)
        for node in continuity_graph.nodes:
            if node.node_id not in current_node_ids:
                continue
            if node.kind != BrainContinuityGraphNodeKind.CLAIM.value:
                continue
            if node.status == "uncertain":
                continue
            node_updated_at = _graph_node_updated_at(node, reference_ts)
            score = 4.0 + _text_score(node.summary, tokens=relevance_tokens)
            candidates.append(
                (
                    score,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value,
                        summary=node.summary,
                        state=(
                            BrainPrivateWorkingMemoryRecordState.STALE.value
                            if node.node_id in stale_node_ids
                            else BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                        ),
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                        backing_ids=[node.backing_record_id],
                        source_event_ids=node.source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=node_updated_at,
                        updated_at=node_updated_at,
                        details={
                            "node_id": node.node_id,
                            "status": node.status,
                            "predicate": node.details.get("predicate"),
                        },
                    ),
                )
            )
    return _cap_records(
        records=candidates,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value],
    )


def _build_self_policy_records(
    *,
    self_core_blocks: dict[str, BrainCoreMemoryBlockRecord],
    planning_digest: dict[str, Any],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    candidates: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    for key in ("self_core", "self_current_arc"):
        block = self_core_blocks.get(key)
        if block is None or not block.content:
            continue
        summary = f"{key}: {_summarize_block_content(block.content)}"
        candidates.append(
            (
                6.0 if key == "self_core" else 5.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.SELF_POLICY.value,
                    summary=summary,
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[block.block_id],
                    source_event_ids=[block.source_event_id],
                    reference_ts=reference_ts,
                    observed_at=block.updated_at,
                    updated_at=block.updated_at,
                    details={"block_kind": block.block_kind},
                ),
            )
        )
    for entry in planning_digest.get("current_pending_proposals", []):
        review_policy = str(entry.get("review_policy", "")).strip()
        proposal_id = str(entry.get("plan_proposal_id", "")).strip()
        if not review_policy or not proposal_id:
            continue
        summary = f"Pending plan {proposal_id} requires {review_policy}"
        candidates.append(
            (
                7.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.SELF_POLICY.value,
                    summary=summary,
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[proposal_id],
                    source_event_ids=[],
                    reference_ts=reference_ts,
                    plan_proposal_id=proposal_id,
                    updated_at=reference_ts,
                    details={"review_policy": review_policy},
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.SELF_POLICY.value],
    )


def _build_goal_commitment_records(
    *,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    goal_event_ids: dict[str, list[str]],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    candidates: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    for record in (
        list(commitment_projection.active_commitments)
        + list(commitment_projection.blocked_commitments)
        + list(commitment_projection.deferred_commitments)
    ):
        summary_parts = [f"{record.title} [{record.status}]", f"rev={record.plan_revision}"]
        if record.blocked_reason is not None:
            summary_parts.append(f"blocked={record.blocked_reason.summary}")
        if record.wake_conditions:
            summary_parts.append(
                "wake=" + ", ".join(item.summary for item in record.wake_conditions[:2] if item.summary)
            )
        score = {
            "active": 9.0,
            "blocked": 8.0,
            "deferred": 7.0,
        }.get(record.status, 6.0)
        candidates.append(
            (
                score,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.GOAL_COMMITMENT.value,
                    summary=", ".join(summary_parts),
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[
                        record.commitment_id,
                        *( [record.current_goal_id] if record.current_goal_id else [] ),
                    ],
                    source_event_ids=goal_event_ids.get(record.current_goal_id or "", []),
                    reference_ts=reference_ts,
                    observed_at=record.created_at,
                    updated_at=record.updated_at,
                    goal_id=record.current_goal_id,
                    commitment_id=record.commitment_id,
                    details={"status": record.status, "goal_family": record.goal_family},
                ),
            )
        )
    for goal in agenda.goals:
        if goal.commitment_id:
            continue
        if goal.status not in {"open", "planning", "in_progress", "retry", "waiting", "blocked", "failed"}:
            continue
        candidates.append(
            (
                6.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.GOAL_COMMITMENT.value,
                    summary=f"{goal.title} [{goal.status}]",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[goal.goal_id],
                    source_event_ids=goal_event_ids.get(goal.goal_id, []),
                    reference_ts=reference_ts,
                    observed_at=goal.created_at,
                    updated_at=goal.updated_at,
                    goal_id=goal.goal_id,
                    details={"goal_family": goal.goal_family},
                ),
            )
        )
    return _cap_records(
        records=candidates,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.GOAL_COMMITMENT.value],
    )


def _proposal_state(
    *,
    proposal_id: str,
    pending_proposal_ids: set[str],
    adopted_by_id: dict[str, BrainEventRecord],
    rejected_by_id: dict[str, BrainEventRecord],
    superseded_by_id: dict[str, str],
) -> tuple[str, str | None]:
    if proposal_id in pending_proposal_ids:
        return BrainPrivateWorkingMemoryRecordState.ACTIVE.value, None
    if proposal_id in adopted_by_id:
        return BrainPrivateWorkingMemoryRecordState.RESOLVED.value, adopted_by_id[proposal_id].ts
    if proposal_id in rejected_by_id:
        return BrainPrivateWorkingMemoryRecordState.RESOLVED.value, rejected_by_id[proposal_id].ts
    if proposal_id in superseded_by_id:
        return BrainPrivateWorkingMemoryRecordState.RESOLVED.value, None
    return BrainPrivateWorkingMemoryRecordState.STALE.value, None


def _build_plan_assumption_records(
    *,
    recent_events: list[BrainEventRecord],
    planning_digest: dict[str, Any],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    proposed_by_id, adopted_by_id, rejected_by_id, superseded_by_id = _proposal_event_index(recent_events)
    pending_proposal_ids = {
        str(item.get("plan_proposal_id", "")).strip()
        for item in planning_digest.get("current_pending_proposals", [])
        if str(item.get("plan_proposal_id", "")).strip()
    }
    candidates: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    for proposal_id, (event, proposal) in proposed_by_id.items():
        state, resolved_ts = _proposal_state(
            proposal_id=proposal_id,
            pending_proposal_ids=pending_proposal_ids,
            adopted_by_id=adopted_by_id,
            rejected_by_id=rejected_by_id,
            superseded_by_id=superseded_by_id,
        )
        updated_at = resolved_ts or event.ts
        source_event_ids = [event.event_id]
        if proposal_id in adopted_by_id:
            source_event_ids.append(adopted_by_id[proposal_id].event_id)
        if proposal_id in rejected_by_id:
            source_event_ids.append(rejected_by_id[proposal_id].event_id)
        for assumption in proposal.assumptions:
            candidates.append(
                (
                    9.0 if state == "active" else 5.0,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                        summary=f"Proposal {proposal_id}: assumption {assumption}",
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                        backing_ids=[proposal_id, proposal.goal_id, *( [proposal.commitment_id] if proposal.commitment_id else [] )],
                        source_event_ids=source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=proposal.created_at,
                        updated_at=updated_at,
                        expires_at=resolved_ts,
                        goal_id=proposal.goal_id,
                        commitment_id=proposal.commitment_id,
                        plan_proposal_id=proposal_id,
                        details={"assumption": assumption, "request_kind": proposal.details.get("request_kind")},
                    ),
                )
            )
        for missing_input in proposal.missing_inputs:
            candidates.append(
                (
                    8.5 if state == "active" else 4.5,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                        summary=f"Proposal {proposal_id}: missing input {missing_input}",
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                        backing_ids=[proposal_id, proposal.goal_id],
                        source_event_ids=source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=proposal.created_at,
                        updated_at=updated_at,
                        expires_at=resolved_ts,
                        goal_id=proposal.goal_id,
                        commitment_id=proposal.commitment_id,
                        plan_proposal_id=proposal_id,
                        details={"missing_input": missing_input},
                    ),
                )
            )
        if proposal.preserved_prefix_count > 0:
            candidates.append(
                (
                    8.0 if state == "active" else 4.0,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                        summary=(
                            f"Proposal {proposal_id}: preserve completed prefix "
                            f"{proposal.preserved_prefix_count}"
                        ),
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                        backing_ids=[proposal_id, proposal.goal_id],
                        source_event_ids=source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=proposal.created_at,
                        updated_at=updated_at,
                        expires_at=resolved_ts,
                        goal_id=proposal.goal_id,
                        commitment_id=proposal.commitment_id,
                        plan_proposal_id=proposal_id,
                        details={"preserved_prefix_count": proposal.preserved_prefix_count},
                    ),
                )
            )
        procedural = proposal.details.get("procedural", {})
        if isinstance(procedural, dict):
            selected_skill_id = str(procedural.get("selected_skill_id", "")).strip()
            origin = str(procedural.get("origin", "")).strip()
            if origin or selected_skill_id:
                candidates.append(
                    (
                        8.0 if state == "active" else 4.0,
                        _record(
                            buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                            summary=(
                                f"Proposal {proposal_id}: {origin or 'procedural'}"
                                + (f" via skill {selected_skill_id}" if selected_skill_id else "")
                            ),
                            state=state,
                            evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                            backing_ids=[proposal_id, proposal.goal_id, selected_skill_id or None],
                            source_event_ids=source_event_ids,
                            reference_ts=reference_ts,
                            observed_at=proposal.created_at,
                            updated_at=updated_at,
                            expires_at=resolved_ts,
                            goal_id=proposal.goal_id,
                            commitment_id=proposal.commitment_id,
                            plan_proposal_id=proposal_id,
                            skill_id=selected_skill_id or None,
                            details={
                                "procedural_origin": origin or None,
                                "delta": dict(procedural.get("delta", {}))
                                if isinstance(procedural.get("delta"), dict)
                                else {},
                            },
                        ),
                    )
                )
    return _cap_records(
        records=candidates,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value],
    )


def _build_scene_world_state_records(
    *,
    scene_world_state: BrainSceneWorldProjection | None,
    body: BrainPresenceSnapshot,
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    recent_events: list[BrainEventRecord],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    recent_scene_event_ids = [
        event.event_id
        for event in recent_events
        if event.event_type
        in {
            BrainEventType.BODY_STATE_UPDATED,
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }
    ][:6]
    records: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    if scene_world_state is not None:
        for entity in scene_world_state.entities:
            if entity.state == "expired" or entity.entity_kind == "zone":
                continue
            state = (
                BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                if entity.state == "active"
                else BrainPrivateWorkingMemoryRecordState.STALE.value
            )
            tags = [entity.entity_kind]
            if entity.zone_id:
                tags.append(f"zone={entity.zone_id}")
            if entity.contradiction_codes:
                tags.append("contradiction=" + ",".join(entity.contradiction_codes[:2]))
            records.append(
                (
                    9.0 if entity.state == "active" else 7.0,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                        summary=f"{entity.summary} [{' | '.join(tags)}]",
                        state=state,
                        evidence_kind=entity.evidence_kind,
                        backing_ids=entity.backing_ids or [entity.entity_id],
                        source_event_ids=entity.source_event_ids or recent_scene_event_ids,
                        reference_ts=reference_ts,
                        observed_at=entity.observed_at,
                        updated_at=entity.updated_at,
                        expires_at=entity.expires_at,
                        details={
                            "entity_id": entity.entity_id,
                            "entity_kind": entity.entity_kind,
                            "zone_id": entity.zone_id,
                            "confidence": entity.confidence,
                            "contradiction_codes": list(entity.contradiction_codes),
                            "affordance_ids": list(entity.affordance_ids),
                        },
                    ),
                )
            )
        for affordance in scene_world_state.affordances:
            if affordance.availability == "stale":
                state = BrainPrivateWorkingMemoryRecordState.STALE.value
                score = 6.5
            elif affordance.availability == "uncertain":
                state = BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                score = 7.5
            elif affordance.availability == "blocked":
                state = BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                score = 7.0
            else:
                state = BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                score = 8.5
            records.append(
                (
                    score,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                        summary=f"{affordance.summary} [affordance={affordance.capability_family}][{affordance.availability}]",
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                        backing_ids=affordance.backing_ids or [affordance.affordance_id],
                        source_event_ids=affordance.source_event_ids or recent_scene_event_ids,
                        reference_ts=reference_ts,
                        observed_at=affordance.observed_at,
                        updated_at=affordance.updated_at,
                        expires_at=affordance.expires_at,
                        details={
                            "affordance_id": affordance.affordance_id,
                            "entity_id": affordance.entity_id,
                            "capability_family": affordance.capability_family,
                            "availability": affordance.availability,
                            "reason_codes": list(affordance.reason_codes),
                        },
                    ),
                )
            )
        if records:
            return _cap_records(
                records=records,
                limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value],
            )

    reference_dt = _parse_ts(reference_ts)
    is_fresh = _scene_is_fresh(scene, reference_dt=reference_dt)
    state = (
        BrainPrivateWorkingMemoryRecordState.ACTIVE.value
        if is_fresh
        else BrainPrivateWorkingMemoryRecordState.STALE.value
    )
    if scene.person_present != "unknown":
        records.append(
            (
                8.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                    summary=f"Person presence is {scene.person_present}",
                    state=state,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
                    backing_ids=["scene:person_present"],
                    source_event_ids=recent_scene_event_ids,
                    reference_ts=reference_ts,
                    observed_at=scene.last_observed_at or scene.last_fresh_frame_at,
                    updated_at=scene.updated_at,
                    expires_at=(
                        (_parse_ts(scene.last_fresh_frame_at) + timedelta(milliseconds=_SCENE_FRESH_MAX_AGE_MS)).isoformat()
                        if scene.last_fresh_frame_at and _parse_ts(scene.last_fresh_frame_at) is not None
                        else None
                    ),
                    details={"confidence": scene.confidence, "frame_age_ms": scene.frame_age_ms},
                ),
            )
        )
    if scene.last_visual_summary:
        records.append(
            (
                7.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                    summary=f"Visual summary: {scene.last_visual_summary}",
                    state=state,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
                    backing_ids=["scene:visual_summary"],
                    source_event_ids=recent_scene_event_ids,
                    reference_ts=reference_ts,
                    observed_at=scene.last_observed_at or scene.last_fresh_frame_at,
                    updated_at=scene.updated_at,
                    details={"detection_backend": scene.detection_backend},
                ),
            )
        )
    records.append(
        (
            6.5,
            _record(
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                summary=f"Camera track is {scene.camera_track_state}",
                state=state if scene.camera_connected else BrainPrivateWorkingMemoryRecordState.STALE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
                backing_ids=["scene:camera_track"],
                source_event_ids=recent_scene_event_ids,
                reference_ts=reference_ts,
                observed_at=scene.last_observed_at or scene.updated_at,
                updated_at=scene.updated_at,
                details={"camera_connected": scene.camera_connected},
            ),
        )
    )
    records.append(
        (
            6.0,
            _record(
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
                summary=f"Engagement is {engagement.engagement_state}; attention {engagement.attention_to_camera}",
                state=state,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
                backing_ids=["scene:engagement"],
                source_event_ids=recent_scene_event_ids,
                reference_ts=reference_ts,
                observed_at=engagement.last_engaged_at or engagement.updated_at,
                updated_at=engagement.updated_at,
                details={"user_present": engagement.user_present, "attention_target": body.attention_target},
            ),
        )
    )
    return _cap_records(
        records=records,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value],
    )


def _build_uncertainty_records(
    *,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    continuity_graph: BrainContinuityGraphProjection | None,
    procedural_skills: BrainProceduralSkillProjection | None,
    recent_events: list[BrainEventRecord],
    planning_digest: dict[str, Any],
    scene: BrainSceneStateProjection,
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    candidates: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    proposed_by_id, adopted_by_id, rejected_by_id, superseded_by_id = _proposal_event_index(recent_events)
    pending_proposal_ids = {
        str(item.get("plan_proposal_id", "")).strip()
        for item in planning_digest.get("current_pending_proposals", [])
        if str(item.get("plan_proposal_id", "")).strip()
    }
    for proposal_id, (event, proposal) in proposed_by_id.items():
        state, resolved_ts = _proposal_state(
            proposal_id=proposal_id,
            pending_proposal_ids=pending_proposal_ids,
            adopted_by_id=adopted_by_id,
            rejected_by_id=rejected_by_id,
            superseded_by_id=superseded_by_id,
        )
        updated_at = resolved_ts or event.ts
        for missing_input in proposal.missing_inputs:
            candidates.append(
                (
                    9.0 if state == "active" else 4.0,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                        summary=f"Missing input: {missing_input}",
                        state=state,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                        backing_ids=[proposal_id, proposal.goal_id],
                        source_event_ids=[
                            event.event_id,
                            adopted_by_id[proposal_id].event_id if proposal_id in adopted_by_id else None,
                            rejected_by_id[proposal_id].event_id if proposal_id in rejected_by_id else None,
                        ],
                        reference_ts=reference_ts,
                        observed_at=proposal.created_at,
                        updated_at=updated_at,
                        expires_at=resolved_ts,
                        goal_id=proposal.goal_id,
                        commitment_id=proposal.commitment_id,
                        plan_proposal_id=proposal_id,
                        details={"kind": "missing_input"},
                    ),
                )
            )
    blocked_commitments = (
        list(commitment_projection.blocked_commitments) + list(commitment_projection.deferred_commitments)
    )
    for record in blocked_commitments:
        reason = record.blocked_reason.summary if record.blocked_reason else record.status
        candidates.append(
            (
                8.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                    summary=f"{record.title}: {reason}",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[record.commitment_id, record.current_goal_id],
                    source_event_ids=[],
                    reference_ts=reference_ts,
                    observed_at=record.updated_at,
                    updated_at=record.updated_at,
                    goal_id=record.current_goal_id,
                    commitment_id=record.commitment_id,
                    details={
                        "kind": "blocked_reason",
                        "status": record.status,
                        "wake_conditions": [item.summary for item in record.wake_conditions],
                    },
                ),
            )
        )
    goal_payloads_by_goal_id = _goal_event_ids(recent_events)[1]
    for goal_id, entries in goal_payloads_by_goal_id.items():
        blocked_entries = [
            entry
            for entry in entries
            if isinstance(entry.get("goal"), dict)
            and (
                str((entry["goal"]).get("status", "")).strip() in {"blocked", "failed"}
                or isinstance((entry["goal"]).get("blocked_reason"), dict)
            )
        ]
        if not blocked_entries:
            continue
        latest_entry = entries[-1]
        latest_goal = dict(latest_entry.get("goal") or {})
        latest_blocked = bool(
            str(latest_goal.get("status", "")).strip() in {"blocked", "failed"}
            or isinstance(latest_goal.get("blocked_reason"), dict)
        )
        if latest_blocked:
            continue
        blocked_entry = blocked_entries[-1]
        blocked_goal = dict(blocked_entry.get("goal") or {})
        blocked_reason = dict(blocked_goal.get("blocked_reason") or {})
        summary = str(blocked_reason.get("summary") or blocked_goal.get("title") or goal_id).strip()
        candidates.append(
            (
                3.5,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                    summary=f"Resolved blocker: {summary}",
                    state=BrainPrivateWorkingMemoryRecordState.RESOLVED.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=[goal_id],
                    source_event_ids=[blocked_entry.get("event_id"), latest_entry.get("event_id")],
                    reference_ts=reference_ts,
                    observed_at=str(blocked_entry.get("ts") or ""),
                    updated_at=str(latest_entry.get("ts") or reference_ts),
                    expires_at=str(latest_entry.get("ts") or reference_ts),
                    goal_id=goal_id,
                    details={"kind": "blocked_reason_cleared"},
                ),
            )
        )
    if continuity_graph is not None:
        current_node_ids = set(continuity_graph.current_node_ids)
        stale_node_ids = set(continuity_graph.stale_node_ids)
        for node in continuity_graph.nodes:
            if node.kind != BrainContinuityGraphNodeKind.CLAIM.value or node.node_id not in current_node_ids:
                continue
            if node.status != "uncertain":
                continue
            node_updated_at = _graph_node_updated_at(node, reference_ts)
            candidates.append(
                (
                    7.0,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                        summary=f"Uncertain claim: {node.summary}",
                        state=(
                            BrainPrivateWorkingMemoryRecordState.STALE.value
                            if node.node_id in stale_node_ids
                            else BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                        ),
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                        backing_ids=[node.backing_record_id],
                        source_event_ids=node.source_event_ids,
                        reference_ts=reference_ts,
                        observed_at=node_updated_at,
                        updated_at=node_updated_at,
                        details={"kind": "uncertain_claim", "node_id": node.node_id},
                    ),
                )
            )
    if not _scene_is_fresh(scene, reference_dt=_parse_ts(reference_ts)) or scene.person_present == "uncertain":
        summary = "Scene freshness degraded"
        if scene.person_present == "uncertain":
            summary = f"{summary}; person presence remains uncertain"
        candidates.append(
            (
                7.0,
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                    summary=summary,
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                    backing_ids=["scene:uncertainty"],
                    source_event_ids=[],
                    reference_ts=reference_ts,
                    observed_at=scene.last_observed_at or scene.updated_at,
                    updated_at=scene.updated_at,
                    details={
                        "kind": "scene_uncertainty",
                        "person_present": scene.person_present,
                        "frame_age_ms": scene.frame_age_ms,
                        "sensor_health_reason": scene.sensor_health_reason,
                    },
                ),
            )
        )
    selected_skill_ids = {
        str(item.get("selected_skill_id", "")).strip()
        for item in planning_digest.get("current_pending_proposals", [])
        if str(item.get("selected_skill_id", "")).strip()
    } | {
        str(item).strip()
        for item in planning_digest.get("recent_selected_skill_ids", [])
        if str(item).strip()
    }
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            if skill.skill_id not in selected_skill_ids:
                continue
            if float(skill.confidence) >= 0.5:
                continue
            candidates.append(
                (
                    6.5,
                    _record(
                        buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                        summary=f"Low-confidence procedural skill: {skill.title or skill.skill_id}",
                        state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                        backing_ids=[skill.skill_id, *skill.supporting_trace_ids[:2]],
                        source_event_ids=[],
                        reference_ts=reference_ts,
                        observed_at=skill.created_at,
                        updated_at=skill.updated_at,
                        skill_id=skill.skill_id,
                        details={"kind": "low_confidence_skill", "confidence": skill.confidence},
                    ),
                )
            )
    return _cap_records(
        records=candidates,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value],
    )


def _tool_outcome_summary(event: BrainEventRecord) -> tuple[str | None, list[str]]:
    payload = _event_payload(event)
    if event.event_type == BrainEventType.TOOL_COMPLETED:
        function_name = str(payload.get("function_name", "")).strip()
        result = payload.get("result")
        summary = f"Tool {function_name} completed"
        if isinstance(result, dict):
            if str(result.get("summary", "")).strip():
                summary = f"{summary}: {str(result.get('summary')).strip()}"
            elif str(result.get("answer", "")).strip():
                summary = f"{summary}: {str(result.get('answer')).strip()}"
        return summary, [payload.get("tool_call_id"), function_name]
    if event.event_type == BrainEventType.CAPABILITY_COMPLETED:
        capability_id = str(payload.get("capability_id", "")).strip()
        result = dict(payload.get("result", {}))
        summary = str(result.get("summary", "")).strip() or f"Capability {capability_id} completed"
        return summary, [capability_id, payload.get("goal_id")]
    if event.event_type == BrainEventType.CAPABILITY_FAILED:
        capability_id = str(payload.get("capability_id", "")).strip()
        error = payload.get("error") or {}
        reason = str(dict(error).get("code", "")).strip() if isinstance(error, dict) else ""
        summary = f"Capability {capability_id} failed"
        if reason:
            summary = f"{summary}: {reason}"
        return summary, [capability_id, payload.get("goal_id")]
    return None, []


def _build_recent_tool_outcome_records(
    *,
    recent_events: list[BrainEventRecord],
    reference_ts: str,
) -> list[BrainPrivateWorkingMemoryRecord]:
    outcome_events = [
        event
        for event in recent_events
        if event.event_type
        in {
            BrainEventType.TOOL_COMPLETED,
            BrainEventType.CAPABILITY_COMPLETED,
            BrainEventType.CAPABILITY_FAILED,
        }
    ]
    records: list[tuple[float, BrainPrivateWorkingMemoryRecord]] = []
    for index, event in enumerate(outcome_events[: _BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value]]):
        summary, backing_ids = _tool_outcome_summary(event)
        if not summary:
            continue
        records.append(
            (
                float(_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value] - index),
                _record(
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value,
                    summary=summary,
                    state=(
                        BrainPrivateWorkingMemoryRecordState.ACTIVE.value
                        if index < _RECENT_TOOL_OUTCOME_ACTIVE_COUNT
                        else BrainPrivateWorkingMemoryRecordState.STALE.value
                    ),
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
                    backing_ids=backing_ids,
                    source_event_ids=[event.event_id, event.causal_parent_id],
                    reference_ts=reference_ts,
                    observed_at=event.ts,
                    updated_at=event.ts,
                    details={"event_type": event.event_type},
                ),
            )
        )
    return _cap_records(
        records=records,
        limit=_BUFFER_CAPS[BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value],
    )


def build_private_working_memory_projection(
    *,
    scope_type: str,
    scope_id: str,
    working_context: BrainWorkingContextProjection,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    scene_world_state: BrainSceneWorldProjection | None,
    body: BrainPresenceSnapshot,
    continuity_graph: BrainContinuityGraphProjection | None,
    continuity_dossiers: BrainContinuityDossierProjection | None,
    procedural_skills: BrainProceduralSkillProjection | None,
    recent_events: list[BrainEventRecord],
    planning_digest: dict[str, Any] | None = None,
    self_core_blocks: dict[str, BrainCoreMemoryBlockRecord] | None = None,
    reference_ts: str | None = None,
) -> BrainPrivateWorkingMemoryProjection:
    """Build one bounded private working-memory projection from landed state."""
    resolved_reference_ts = _resolve_reference_ts(
        reference_ts=reference_ts,
        working_context=working_context,
        agenda=agenda,
        commitment_projection=commitment_projection,
        scene=scene,
        engagement=engagement,
        body=body,
        recent_events=recent_events,
    )
    planning_digest = planning_digest or build_planning_digest(
        agenda=agenda,
        commitment_projection=commitment_projection,
        recent_events=recent_events,
    )
    self_core_blocks = dict(self_core_blocks or {})
    relevance_tokens = _query_tokens(
        working_context.last_user_text,
        agenda.active_goal_summary,
        commitment_projection.current_active_summary,
        *[
            str(item.get("title", "")).strip()
            for item in planning_digest.get("current_pending_proposals", [])
        ],
    )
    goal_event_ids, _goal_payloads = _goal_event_ids(recent_events)
    records = [
        *_build_user_model_records(
            continuity_graph=continuity_graph,
            continuity_dossiers=continuity_dossiers,
            relevance_tokens=relevance_tokens,
            reference_ts=resolved_reference_ts,
        ),
        *_build_self_policy_records(
            self_core_blocks=self_core_blocks,
            planning_digest=planning_digest,
            reference_ts=resolved_reference_ts,
        ),
        *_build_goal_commitment_records(
            agenda=agenda,
            commitment_projection=commitment_projection,
            goal_event_ids=goal_event_ids,
            reference_ts=resolved_reference_ts,
        ),
        *_build_plan_assumption_records(
            recent_events=recent_events,
            planning_digest=planning_digest,
            reference_ts=resolved_reference_ts,
        ),
        *_build_scene_world_state_records(
            body=body,
            scene=scene,
            engagement=engagement,
            scene_world_state=scene_world_state,
            recent_events=recent_events,
            reference_ts=resolved_reference_ts,
        ),
        *_build_uncertainty_records(
            agenda=agenda,
            commitment_projection=commitment_projection,
            continuity_graph=continuity_graph,
            procedural_skills=procedural_skills,
            recent_events=recent_events,
            planning_digest=planning_digest,
            scene=scene,
            reference_ts=resolved_reference_ts,
        ),
        *_build_recent_tool_outcome_records(
            recent_events=recent_events,
            reference_ts=resolved_reference_ts,
        ),
    ]
    projection = BrainPrivateWorkingMemoryProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        records=records,
        updated_at=resolved_reference_ts,
    )
    projection.sync_lists()
    return projection


__all__ = ["build_private_working_memory_projection"]

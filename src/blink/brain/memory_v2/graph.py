"""Derived temporal continuity graph projection for Blink memory v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2.autobiography import BrainAutobiographicalEntryRecord
from blink.brain.memory_v2.claims import (
    BrainClaimEvidenceRecord,
    BrainClaimRecord,
    BrainClaimSupersessionRecord,
    render_claim_summary,
)
from blink.brain.memory_v2.core_blocks import BrainCoreMemoryBlockRecord
from blink.brain.memory_v2.entities import BrainEntityRecord
from blink.brain.memory_v2.multimodal_autobiography import parse_multimodal_autobiography_record
from blink.brain.memory_v2.skills import BrainProceduralSkillProjection, BrainProceduralSkillRecord
from blink.brain.projections import (
    BrainClaimCurrentnessStatus,
    BrainCommitmentProjection,
    BrainCommitmentRecord,
    BrainPlanProposal,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _sorted_unique_texts(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _sorted_unique_episode_ids(values: Iterable[int | str | None]) -> list[int]:
    episode_ids: set[int] = set()
    for value in values:
        if value in (None, ""):
            continue
        try:
            episode_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(episode_ids)


def _event_sort_key(event: BrainEventRecord) -> tuple[int, datetime, str]:
    return (
        int(getattr(event, "id", 0)),
        _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
        event.event_id,
    )


def _node_sort_key(record: "BrainContinuityGraphNodeRecord") -> tuple[str, str, str]:
    return (record.kind, record.backing_record_id, record.node_id)


def _edge_sort_key(record: "BrainContinuityGraphEdgeRecord") -> tuple[str, str, str, str]:
    return (record.kind, record.from_node_id, record.to_node_id, record.edge_id)


class BrainContinuityGraphNodeKind(str, Enum):
    """Typed node kinds for the derived continuity graph."""

    ENTITY = "entity"
    CLAIM = "claim"
    AUTOBIOGRAPHY_ENTRY = "autobiography_entry"
    COMMITMENT = "commitment"
    PLAN_PROPOSAL = "plan_proposal"
    CORE_MEMORY_BLOCK = "core_memory_block"
    PROCEDURAL_SKILL = "procedural_skill"
    SCENE_WORLD_ENTITY = "scene_world_entity"
    SCENE_WORLD_AFFORDANCE = "scene_world_affordance"
    EVENT_ANCHOR = "event_anchor"
    EPISODE_ANCHOR = "episode_anchor"


class BrainContinuityGraphEdgeKind(str, Enum):
    """Typed edge kinds for the derived continuity graph."""

    CLAIM_SUBJECT = "claim_subject"
    CLAIM_OBJECT = "claim_object"
    SUPPORTED_BY_EVENT = "supported_by_event"
    SUPPORTED_BY_EPISODE = "supported_by_episode"
    SUPERSEDES = "supersedes"
    AUTOBIOGRAPHY_SUPPORTS_CLAIM = "autobiography_supports_claim"
    AUTOBIOGRAPHY_SUPPORTS_EVENT = "autobiography_supports_event"
    COMMITMENT_HAS_PLAN_PROPOSAL = "commitment_has_plan_proposal"
    PLAN_PROPOSAL_SUPERSEDES = "plan_proposal_supersedes"
    PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT = "plan_proposal_adopted_into_commitment"
    CORE_BLOCK_SUPERSEDES = "core_block_supersedes"
    PROCEDURAL_SKILL_SUPPORTS_COMMITMENT = "procedural_skill_supports_commitment"
    PROCEDURAL_SKILL_SUPPORTS_PLAN_PROPOSAL = "procedural_skill_supports_plan_proposal"
    PROCEDURAL_SKILL_SUPERSEDES = "procedural_skill_supersedes"
    SCENE_WORLD_ENTITY_HAS_AFFORDANCE = "scene_world_entity_has_affordance"
    AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_ENTITY = "autobiography.references.scene_world_entity"
    AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_AFFORDANCE = (
        "autobiography.references.scene_world_affordance"
    )


@dataclass(frozen=True)
class BrainContinuityGraphNodeRecord:
    """One typed node in the derived continuity graph."""

    node_id: str
    kind: str
    backing_record_id: str
    summary: str
    status: str
    scope_type: str | None = None
    scope_id: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[int] = field(default_factory=list)
    supporting_claim_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the node record into a JSON-safe mapping."""
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "backing_record_id": self.backing_record_id,
            "summary": self.summary,
            "status": self.status,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_event_ids": _sorted_unique_texts(self.source_event_ids),
            "source_episode_ids": _sorted_unique_episode_ids(self.source_episode_ids),
            "supporting_claim_ids": _sorted_unique_texts(self.supporting_claim_ids),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityGraphNodeRecord | None":
        """Hydrate one node record from stored JSON."""
        if not isinstance(data, dict):
            return None
        node_id = str(data.get("node_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        backing_record_id = str(data.get("backing_record_id", "")).strip()
        summary = str(data.get("summary", "")).strip()
        status = str(data.get("status", "")).strip()
        if not node_id or not kind or not backing_record_id or not status:
            return None
        return cls(
            node_id=node_id,
            kind=kind,
            backing_record_id=backing_record_id,
            summary=summary,
            status=status,
            scope_type=_optional_text(data.get("scope_type")),
            scope_id=_optional_text(data.get("scope_id")),
            valid_from=_optional_text(data.get("valid_from")),
            valid_to=_optional_text(data.get("valid_to")),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            source_episode_ids=_sorted_unique_episode_ids(data.get("source_episode_ids", [])),
            supporting_claim_ids=_sorted_unique_texts(data.get("supporting_claim_ids", [])),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainContinuityGraphEdgeRecord:
    """One typed edge in the derived continuity graph."""

    edge_id: str
    kind: str
    from_node_id: str
    to_node_id: str
    status: str
    valid_from: str | None = None
    valid_to: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[int] = field(default_factory=list)
    supporting_claim_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the edge record into a JSON-safe mapping."""
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_event_ids": _sorted_unique_texts(self.source_event_ids),
            "source_episode_ids": _sorted_unique_episode_ids(self.source_episode_ids),
            "supporting_claim_ids": _sorted_unique_texts(self.supporting_claim_ids),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityGraphEdgeRecord | None":
        """Hydrate one edge record from stored JSON."""
        if not isinstance(data, dict):
            return None
        edge_id = str(data.get("edge_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        from_node_id = str(data.get("from_node_id", "")).strip()
        to_node_id = str(data.get("to_node_id", "")).strip()
        status = str(data.get("status", "")).strip()
        if not edge_id or not kind or not from_node_id or not to_node_id or not status:
            return None
        return cls(
            edge_id=edge_id,
            kind=kind,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            status=status,
            valid_from=_optional_text(data.get("valid_from")),
            valid_to=_optional_text(data.get("valid_to")),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            source_episode_ids=_sorted_unique_episode_ids(data.get("source_episode_ids", [])),
            supporting_claim_ids=_sorted_unique_texts(data.get("supporting_claim_ids", [])),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainContinuityGraphProjection:
    """Connected continuity graph derived from existing memory and executive state."""

    scope_type: str
    scope_id: str
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    nodes: list[BrainContinuityGraphNodeRecord] = field(default_factory=list)
    edges: list[BrainContinuityGraphEdgeRecord] = field(default_factory=list)
    current_node_ids: list[str] = field(default_factory=list)
    historical_node_ids: list[str] = field(default_factory=list)
    stale_node_ids: list[str] = field(default_factory=list)
    superseded_node_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the continuity graph projection."""
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "node_counts": dict(self.node_counts),
            "edge_counts": dict(self.edge_counts),
            "nodes": [record.as_dict() for record in self.nodes],
            "edges": [record.as_dict() for record in self.edges],
            "current_node_ids": _sorted_unique_texts(self.current_node_ids),
            "historical_node_ids": _sorted_unique_texts(self.historical_node_ids),
            "stale_node_ids": _sorted_unique_texts(self.stale_node_ids),
            "superseded_node_ids": _sorted_unique_texts(self.superseded_node_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityGraphProjection | None":
        """Hydrate one continuity graph projection from stored JSON."""
        if not isinstance(data, dict):
            return None
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        if not scope_type or not scope_id:
            return None
        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            node_counts={
                str(key): int(value) for key, value in dict(data.get("node_counts", {})).items()
            },
            edge_counts={
                str(key): int(value) for key, value in dict(data.get("edge_counts", {})).items()
            },
            nodes=[
                record
                for item in data.get("nodes", [])
                if (record := BrainContinuityGraphNodeRecord.from_dict(item)) is not None
            ],
            edges=[
                record
                for item in data.get("edges", [])
                if (record := BrainContinuityGraphEdgeRecord.from_dict(item)) is not None
            ],
            current_node_ids=_sorted_unique_texts(data.get("current_node_ids", [])),
            historical_node_ids=_sorted_unique_texts(data.get("historical_node_ids", [])),
            stale_node_ids=_sorted_unique_texts(data.get("stale_node_ids", [])),
            superseded_node_ids=_sorted_unique_texts(data.get("superseded_node_ids", [])),
        )


def _merge_node(
    existing: BrainContinuityGraphNodeRecord,
    incoming: BrainContinuityGraphNodeRecord,
) -> BrainContinuityGraphNodeRecord:
    summary = (
        existing.summary
        if existing.summary and not existing.summary.startswith("Unknown ")
        else incoming.summary
    )
    details = dict(existing.details)
    for key, value in incoming.details.items():
        if key not in details or details[key] in ("", None, [], {}):
            details[key] = value
    return BrainContinuityGraphNodeRecord(
        node_id=existing.node_id,
        kind=existing.kind,
        backing_record_id=existing.backing_record_id,
        summary=summary or incoming.summary,
        status=existing.status if existing.status != "referenced" else incoming.status,
        scope_type=existing.scope_type or incoming.scope_type,
        scope_id=existing.scope_id or incoming.scope_id,
        valid_from=existing.valid_from or incoming.valid_from,
        valid_to=existing.valid_to or incoming.valid_to,
        source_event_ids=_sorted_unique_texts(
            [*existing.source_event_ids, *incoming.source_event_ids]
        ),
        source_episode_ids=_sorted_unique_episode_ids(
            [*existing.source_episode_ids, *incoming.source_episode_ids]
        ),
        supporting_claim_ids=_sorted_unique_texts(
            [*existing.supporting_claim_ids, *incoming.supporting_claim_ids]
        ),
        details=details,
    )


def _merge_edge(
    existing: BrainContinuityGraphEdgeRecord,
    incoming: BrainContinuityGraphEdgeRecord,
) -> BrainContinuityGraphEdgeRecord:
    details = dict(existing.details)
    for key, value in incoming.details.items():
        if key not in details or details[key] in ("", None, [], {}):
            details[key] = value
    return BrainContinuityGraphEdgeRecord(
        edge_id=existing.edge_id,
        kind=existing.kind,
        from_node_id=existing.from_node_id,
        to_node_id=existing.to_node_id,
        status=existing.status if existing.status != "linked" else incoming.status,
        valid_from=existing.valid_from or incoming.valid_from,
        valid_to=existing.valid_to or incoming.valid_to,
        source_event_ids=_sorted_unique_texts(
            [*existing.source_event_ids, *incoming.source_event_ids]
        ),
        source_episode_ids=_sorted_unique_episode_ids(
            [*existing.source_episode_ids, *incoming.source_episode_ids]
        ),
        supporting_claim_ids=_sorted_unique_texts(
            [*existing.supporting_claim_ids, *incoming.supporting_claim_ids]
        ),
        details=details,
    )


def _entity_summary(record: BrainEntityRecord | None, entity_id: str) -> str:
    if record is None:
        return f"Unknown entity {entity_id}"
    return record.canonical_name or entity_id


def _build_entity_node(
    *,
    entity_id: str,
    scope_type: str,
    scope_id: str,
    entity_lookup: Callable[[str], BrainEntityRecord | None] | None,
) -> BrainContinuityGraphNodeRecord:
    record = entity_lookup(entity_id) if entity_lookup is not None else None
    details = (
        {
            "entity_type": record.entity_type,
            "canonical_name": record.canonical_name,
            "aliases": record.aliases,
            "attributes": record.attributes,
        }
        if record is not None
        else {}
    )
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id("graph_node", BrainContinuityGraphNodeKind.ENTITY.value, entity_id),
        kind=BrainContinuityGraphNodeKind.ENTITY.value,
        backing_record_id=entity_id,
        summary=_entity_summary(record, entity_id),
        status="referenced",
        scope_type=scope_type,
        scope_id=scope_id,
        valid_from=record.created_at if record is not None else None,
        valid_to=None,
        details=details,
    )


def _event_anchor_node(
    *,
    event_id: str,
    scope_type: str,
    scope_id: str,
    event_map: dict[str, BrainEventRecord],
) -> BrainContinuityGraphNodeRecord:
    event = event_map.get(event_id)
    summary = event.event_type if event is not None else f"Event {event_id}"
    details = {"event_type": event.event_type, "ts": event.ts} if event is not None else {}
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id("graph_node", BrainContinuityGraphNodeKind.EVENT_ANCHOR.value, event_id),
        kind=BrainContinuityGraphNodeKind.EVENT_ANCHOR.value,
        backing_record_id=event_id,
        summary=summary,
        status="referenced",
        scope_type=scope_type,
        scope_id=scope_id,
        valid_from=event.ts if event is not None else None,
        valid_to=None,
        source_event_ids=[event_id],
        details=details,
    )


def _episode_anchor_node(
    *,
    episode_id: int,
    scope_type: str,
    scope_id: str,
) -> BrainContinuityGraphNodeRecord:
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id(
            "graph_node", BrainContinuityGraphNodeKind.EPISODE_ANCHOR.value, episode_id
        ),
        kind=BrainContinuityGraphNodeKind.EPISODE_ANCHOR.value,
        backing_record_id=str(episode_id),
        summary=f"Episode {episode_id}",
        status="referenced",
        scope_type=scope_type,
        scope_id=scope_id,
        source_episode_ids=[episode_id],
        details={"episode_id": episode_id},
    )


def _resolve_now_ts(
    *,
    now: str | None,
    recent_events: Iterable[BrainEventRecord],
    claims: Iterable[BrainClaimRecord],
    autobiography: Iterable[BrainAutobiographicalEntryRecord],
    commitment_records: Iterable[BrainCommitmentRecord],
    core_blocks: Iterable[BrainCoreMemoryBlockRecord],
    procedural_skills: BrainProceduralSkillProjection | None,
    scene_world_state: BrainSceneWorldProjection | None,
) -> datetime:
    candidates = [_parse_ts(now)]
    candidates.extend(_parse_ts(event.ts) for event in recent_events)
    for claim in claims:
        candidates.extend(
            (
                _parse_ts(claim.updated_at),
                _parse_ts(claim.valid_from),
                _parse_ts(claim.valid_to),
            )
        )
    for entry in autobiography:
        candidates.extend(
            (
                _parse_ts(entry.updated_at),
                _parse_ts(entry.valid_from),
                _parse_ts(entry.valid_to),
            )
        )
    for record in commitment_records:
        candidates.extend(
            (
                _parse_ts(record.updated_at),
                _parse_ts(record.created_at),
                _parse_ts(record.completed_at),
            )
        )
    for block in core_blocks:
        candidates.extend((_parse_ts(block.updated_at), _parse_ts(block.created_at)))
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            candidates.extend(
                (
                    _parse_ts(skill.updated_at),
                    _parse_ts(skill.created_at),
                    _parse_ts(skill.retired_at),
                )
            )
    if scene_world_state is not None:
        candidates.append(_parse_ts(scene_world_state.updated_at))
        for entity in scene_world_state.entities:
            candidates.extend(
                (
                    _parse_ts(entity.updated_at),
                    _parse_ts(entity.observed_at),
                    _parse_ts(entity.expires_at),
                )
            )
        for affordance in scene_world_state.affordances:
            candidates.extend(
                (
                    _parse_ts(affordance.updated_at),
                    _parse_ts(affordance.observed_at),
                    _parse_ts(affordance.expires_at),
                )
            )
    resolved = [candidate for candidate in candidates if candidate is not None]
    return max(resolved) if resolved else datetime.min.replace(tzinfo=UTC)


def _block_summary(record: BrainCoreMemoryBlockRecord) -> str:
    content = record.content
    if record.block_kind == "user_core":
        parts = [
            _optional_text(content.get("name")),
            _optional_text(content.get("role")),
            _optional_text(content.get("origin")),
        ]
        rendered = ", ".join(part for part in parts if part)
        if rendered:
            return rendered
    if record.block_kind == "active_commitments":
        commitments = content.get("commitments", [])
        if isinstance(commitments, list) and commitments:
            title = _optional_text(dict(commitments[0]).get("title"))
            if title is not None:
                return title
    for key in ("summary", "last_session_summary", "name", "title"):
        value = _optional_text(content.get(key))
        if value is not None:
            return value
    content_keys = sorted(str(key) for key in content.keys())
    if content_keys:
        return f"{record.block_kind} ({', '.join(content_keys[:3])})"
    return record.block_kind


def _core_block_node(
    record: BrainCoreMemoryBlockRecord,
    *,
    source_event_ts: str | None = None,
) -> BrainContinuityGraphNodeRecord:
    content = record.content
    has_source_event = bool(_optional_text(record.source_event_id))
    anchored_to_replayed_event = bool(_optional_text(source_event_ts))
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
            record.block_id,
        ),
        kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
        backing_record_id=record.block_id,
        summary=_block_summary(record),
        status=record.status,
        scope_type=record.scope_type,
        scope_id=record.scope_id,
        valid_from=(
            source_event_ts
            if anchored_to_replayed_event
            else (record.created_at if has_source_event else None)
        ),
        valid_to=(
            None
            if anchored_to_replayed_event
            else (record.updated_at if has_source_event and record.status != "current" else None)
        ),
        source_event_ids=_sorted_unique_texts([record.source_event_id]),
        details={
            "block_kind": record.block_kind,
            "version": record.version,
            "source_event_id": record.source_event_id,
            "supersedes_block_id": record.supersedes_block_id,
            "content_keys": sorted(str(key) for key in content.keys()),
        },
    )


def _procedural_skill_node(record: BrainProceduralSkillRecord) -> BrainContinuityGraphNodeRecord:
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.PROCEDURAL_SKILL.value,
            record.skill_id,
        ),
        kind=BrainContinuityGraphNodeKind.PROCEDURAL_SKILL.value,
        backing_record_id=record.skill_id,
        summary=record.title or record.purpose or record.skill_family_key or record.skill_id,
        status=record.status,
        scope_type=record.scope_type,
        scope_id=record.scope_id,
        valid_from=record.created_at,
        valid_to=record.retired_at,
        details={
            "title": record.title,
            "purpose": record.purpose,
            "skill_family_key": record.skill_family_key,
            "goal_family": record.goal_family,
            "confidence": record.confidence,
            "supporting_trace_ids": list(record.supporting_trace_ids),
            "supporting_plan_proposal_ids": list(record.supporting_plan_proposal_ids),
            "supporting_commitment_ids": list(record.supporting_commitment_ids),
            "supersedes_skill_id": record.supersedes_skill_id,
            "superseded_by_skill_id": record.superseded_by_skill_id,
            "retirement_reason": record.retirement_reason,
            "review_policy": record.details.get("review_policy"),
        },
    )


def _scene_entity_node(
    record: Any,
    *,
    scope_type: str,
    scope_id: str,
) -> BrainContinuityGraphNodeRecord:
    source_event_ids = _sorted_unique_texts(record.source_event_ids)
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
            record.entity_id,
        ),
        kind=BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
        backing_record_id=record.entity_id,
        summary=record.summary,
        status=record.state,
        scope_type=scope_type,
        scope_id=scope_id,
        valid_from=record.observed_at if source_event_ids else None,
        valid_to=record.expires_at if source_event_ids else None,
        source_event_ids=source_event_ids,
        details={
            "entity_kind": record.entity_kind,
            "canonical_label": record.canonical_label,
            "zone_id": record.zone_id,
            "confidence": record.confidence,
            "freshness": record.freshness,
            "contradiction_codes": list(record.contradiction_codes),
            "affordance_ids": list(record.affordance_ids),
            "backing_ids": list(record.backing_ids),
        },
    )


def _scene_affordance_node(
    record: Any,
    *,
    scope_type: str,
    scope_id: str,
) -> BrainContinuityGraphNodeRecord:
    source_event_ids = _sorted_unique_texts(record.source_event_ids)
    return BrainContinuityGraphNodeRecord(
        node_id=_stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
            record.affordance_id,
        ),
        kind=BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
        backing_record_id=record.affordance_id,
        summary=record.summary,
        status=record.availability,
        scope_type=scope_type,
        scope_id=scope_id,
        valid_from=record.observed_at if source_event_ids else None,
        valid_to=record.expires_at if source_event_ids else None,
        source_event_ids=source_event_ids,
        details={
            "entity_id": record.entity_id,
            "capability_family": record.capability_family,
            "confidence": record.confidence,
            "freshness": record.freshness,
            "reason_codes": list(record.reason_codes),
            "backing_ids": list(record.backing_ids),
        },
    )


def build_continuity_graph_projection(
    *,
    scope_type: str,
    scope_id: str,
    current_claims: Iterable[BrainClaimRecord],
    historical_claims: Iterable[BrainClaimRecord],
    claim_supersessions: Iterable[BrainClaimSupersessionRecord],
    autobiography: Iterable[BrainAutobiographicalEntryRecord],
    commitment_projection: BrainCommitmentProjection,
    core_blocks: Iterable[BrainCoreMemoryBlockRecord] = (),
    procedural_skills: BrainProceduralSkillProjection | None = None,
    scene_world_state: BrainSceneWorldProjection | None = None,
    recent_events: Iterable[BrainEventRecord] = (),
    claim_evidence_by_id: dict[str, list[BrainClaimEvidenceRecord]] | None = None,
    entity_lookup: Callable[[str], BrainEntityRecord | None] | None = None,
    now: str | None = None,
) -> BrainContinuityGraphProjection:
    """Build one deterministic continuity graph projection from replayable state."""
    commitment_records = [
        *commitment_projection.active_commitments,
        *commitment_projection.deferred_commitments,
        *commitment_projection.blocked_commitments,
        *commitment_projection.recent_terminal_commitments,
    ]
    claim_list = [*historical_claims, *current_claims]
    autobiography_list = list(autobiography)
    core_block_list = list(core_blocks)
    recent_event_list = list(recent_events)
    now_ts = _resolve_now_ts(
        now=now,
        recent_events=recent_event_list,
        claims=claim_list,
        autobiography=autobiography_list,
        commitment_records=commitment_records,
        core_blocks=core_block_list,
        procedural_skills=procedural_skills,
        scene_world_state=scene_world_state,
    )
    event_map = {
        event.event_id: event
        for event in sorted(recent_event_list, key=_event_sort_key)
        if event.event_id
    }
    evidence_by_id = claim_evidence_by_id or {}

    nodes: dict[str, BrainContinuityGraphNodeRecord] = {}
    edges: dict[str, BrainContinuityGraphEdgeRecord] = {}
    node_status_hints: dict[str, set[str]] = {}

    def add_node(record: BrainContinuityGraphNodeRecord, *, status_hint: str | None = None):
        existing = nodes.get(record.node_id)
        nodes[record.node_id] = record if existing is None else _merge_node(existing, record)
        if status_hint:
            node_status_hints.setdefault(record.node_id, set()).add(status_hint)

    def add_edge(record: BrainContinuityGraphEdgeRecord):
        existing = edges.get(record.edge_id)
        edges[record.edge_id] = record if existing is None else _merge_edge(existing, record)

    def ensure_event_anchor(event_id: str | None):
        normalized = _optional_text(event_id)
        if normalized is None:
            return None
        node = _event_anchor_node(
            event_id=normalized,
            scope_type=scope_type,
            scope_id=scope_id,
            event_map=event_map,
        )
        add_node(node)
        return node

    def ensure_episode_anchor(episode_id: int | str | None):
        if episode_id in (None, ""):
            return None
        try:
            normalized = int(episode_id)
        except (TypeError, ValueError):
            return None
        node = _episode_anchor_node(
            episode_id=normalized,
            scope_type=scope_type,
            scope_id=scope_id,
        )
        add_node(node)
        return node

    all_claims: dict[str, BrainClaimRecord] = {}
    for claim in historical_claims:
        all_claims.setdefault(claim.claim_id, claim)
    for claim in current_claims:
        all_claims[claim.claim_id] = claim

    plan_proposal_events = [
        event
        for event in sorted(event_map.values(), key=_event_sort_key)
        if event.event_type
        in {
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.PLANNING_REJECTED,
        }
    ]
    proposal_timelines: dict[str, dict[str, Any]] = {}
    plan_superseded_ids: set[str] = set()

    for event in plan_proposal_events:
        proposal = BrainPlanProposal.from_dict((event.payload or {}).get("proposal"))
        if proposal is None:
            continue
        timeline = proposal_timelines.setdefault(
            proposal.plan_proposal_id,
            {
                "proposal": proposal,
                "proposed_event": None,
                "adopted_event": None,
                "rejected_event": None,
            },
        )
        timeline["proposal"] = proposal
        if event.event_type == BrainEventType.PLANNING_PROPOSED:
            timeline["proposed_event"] = event
        elif event.event_type == BrainEventType.PLANNING_ADOPTED:
            timeline["adopted_event"] = event
        elif event.event_type == BrainEventType.PLANNING_REJECTED:
            timeline["rejected_event"] = event
        if proposal.supersedes_plan_proposal_id:
            plan_superseded_ids.add(proposal.supersedes_plan_proposal_id)

    for claim in all_claims.values():
        subject_entity = (
            entity_lookup(claim.subject_entity_id) if entity_lookup is not None else None
        )
        object_entity = (
            entity_lookup(claim.object_entity_id)
            if claim.object_entity_id is not None and entity_lookup is not None
            else None
        )
        claim_node = BrainContinuityGraphNodeRecord(
            node_id=_stable_id(
                "graph_node", BrainContinuityGraphNodeKind.CLAIM.value, claim.claim_id
            ),
            kind=BrainContinuityGraphNodeKind.CLAIM.value,
            backing_record_id=claim.claim_id,
            summary=render_claim_summary(
                claim,
                subject_name=subject_entity.canonical_name if subject_entity is not None else None,
                object_name=object_entity.canonical_name if object_entity is not None else None,
            ),
            status=claim.status,
            scope_type=claim.scope_type or scope_type,
            scope_id=claim.scope_id or scope_id,
            valid_from=claim.valid_from,
            valid_to=claim.valid_to,
            source_event_ids=_sorted_unique_texts([claim.source_event_id]),
            supporting_claim_ids=[claim.claim_id],
            details={
                "predicate": claim.predicate,
                "object": claim.object,
                "confidence": claim.confidence,
                "truth_status": claim.status,
                "currentness_status": claim.effective_currentness_status,
                "review_state": claim.effective_review_state,
                "retention_class": claim.effective_retention_class,
                "reason_codes": list(claim.governance_reason_codes),
                "subject_entity_id": claim.subject_entity_id,
                "object_entity_id": claim.object_entity_id,
                "claim_key": claim.claim_key,
                "stale_after_seconds": claim.stale_after_seconds,
            },
        )
        add_node(claim_node, status_hint="historical" if claim.is_historical else "current")
        if claim.is_stale:
            node_status_hints.setdefault(claim_node.node_id, set()).add("stale")
        if claim.is_held:
            node_status_hints.setdefault(claim_node.node_id, set()).add("held")
        if claim.status == "superseded" or "superseded" in claim.governance_reason_codes:
            node_status_hints.setdefault(claim_node.node_id, set()).add("superseded")

        subject_node = _build_entity_node(
            entity_id=claim.subject_entity_id,
            scope_type=claim.scope_type or scope_type,
            scope_id=claim.scope_id or scope_id,
            entity_lookup=entity_lookup,
        )
        add_node(subject_node)
        add_edge(
            BrainContinuityGraphEdgeRecord(
                edge_id=_stable_id(
                    "graph_edge",
                    BrainContinuityGraphEdgeKind.CLAIM_SUBJECT.value,
                    claim_node.node_id,
                    subject_node.node_id,
                ),
                kind=BrainContinuityGraphEdgeKind.CLAIM_SUBJECT.value,
                from_node_id=claim_node.node_id,
                to_node_id=subject_node.node_id,
                status="linked",
                valid_from=claim.valid_from,
                valid_to=claim.valid_to,
                source_event_ids=_sorted_unique_texts([claim.source_event_id]),
                supporting_claim_ids=[claim.claim_id],
                details={"predicate": claim.predicate},
            )
        )
        if claim.object_entity_id:
            object_node = _build_entity_node(
                entity_id=claim.object_entity_id,
                scope_type=claim.scope_type or scope_type,
                scope_id=claim.scope_id or scope_id,
                entity_lookup=entity_lookup,
            )
            add_node(object_node)
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.CLAIM_OBJECT.value,
                        claim_node.node_id,
                        object_node.node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.CLAIM_OBJECT.value,
                    from_node_id=claim_node.node_id,
                    to_node_id=object_node.node_id,
                    status="linked",
                    valid_from=claim.valid_from,
                    valid_to=claim.valid_to,
                    source_event_ids=_sorted_unique_texts([claim.source_event_id]),
                    supporting_claim_ids=[claim.claim_id],
                    details={"predicate": claim.predicate},
                )
            )

        if claim.source_event_id:
            event_node = ensure_event_anchor(claim.source_event_id)
            if event_node is not None:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            claim_node.node_id,
                            event_node.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=claim_node.node_id,
                        to_node_id=event_node.node_id,
                        status="supported",
                        valid_from=claim.valid_from,
                        valid_to=claim.valid_to,
                        source_event_ids=[claim.source_event_id],
                        supporting_claim_ids=[claim.claim_id],
                        details={"source": "claim_record"},
                    )
                )

        for evidence in evidence_by_id.get(claim.claim_id, []):
            if evidence.source_event_id:
                event_node = ensure_event_anchor(evidence.source_event_id)
                if event_node is not None:
                    add_edge(
                        BrainContinuityGraphEdgeRecord(
                            edge_id=_stable_id(
                                "graph_edge",
                                BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                                claim_node.node_id,
                                event_node.node_id,
                            ),
                            kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            from_node_id=claim_node.node_id,
                            to_node_id=event_node.node_id,
                            status="supported",
                            valid_from=claim.valid_from,
                            valid_to=claim.valid_to,
                            source_event_ids=[evidence.source_event_id],
                            supporting_claim_ids=[claim.claim_id],
                            details={
                                "evidence_id": evidence.evidence_id,
                                "evidence_summary": evidence.evidence_summary,
                            },
                        )
                    )
            if evidence.source_episode_id is not None:
                episode_node = ensure_episode_anchor(evidence.source_episode_id)
                if episode_node is not None:
                    add_edge(
                        BrainContinuityGraphEdgeRecord(
                            edge_id=_stable_id(
                                "graph_edge",
                                BrainContinuityGraphEdgeKind.SUPPORTED_BY_EPISODE.value,
                                claim_node.node_id,
                                episode_node.node_id,
                            ),
                            kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EPISODE.value,
                            from_node_id=claim_node.node_id,
                            to_node_id=episode_node.node_id,
                            status="supported",
                            valid_from=claim.valid_from,
                            valid_to=claim.valid_to,
                            source_episode_ids=[evidence.source_episode_id],
                            supporting_claim_ids=[claim.claim_id],
                            details={
                                "evidence_id": evidence.evidence_id,
                                "evidence_summary": evidence.evidence_summary,
                            },
                        )
                    )

    for supersession in claim_supersessions:
        prior_node_id = _stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.CLAIM.value,
            supersession.prior_claim_id,
        )
        new_node_id = _stable_id(
            "graph_node",
            BrainContinuityGraphNodeKind.CLAIM.value,
            supersession.new_claim_id,
        )
        if prior_node_id not in nodes or new_node_id not in nodes:
            continue
        node_status_hints.setdefault(prior_node_id, set()).add("superseded")
        add_edge(
            BrainContinuityGraphEdgeRecord(
                edge_id=_stable_id(
                    "graph_edge",
                    BrainContinuityGraphEdgeKind.SUPERSEDES.value,
                    prior_node_id,
                    new_node_id,
                ),
                kind=BrainContinuityGraphEdgeKind.SUPERSEDES.value,
                from_node_id=prior_node_id,
                to_node_id=new_node_id,
                status="superseded",
                valid_from=supersession.created_at,
                valid_to=None,
                source_event_ids=_sorted_unique_texts([supersession.source_event_id]),
                supporting_claim_ids=[
                    supersession.prior_claim_id,
                    supersession.new_claim_id,
                ],
                details={"reason": supersession.reason},
            )
        )

    scene_episode_entry_nodes: list[tuple[BrainAutobiographicalEntryRecord, BrainContinuityGraphNodeRecord]] = []
    for entry in autobiography_list:
        multimodal = parse_multimodal_autobiography_record(entry)
        entry_node = BrainContinuityGraphNodeRecord(
            node_id=_stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value,
                entry.entry_id,
            ),
            kind=BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value,
            backing_record_id=entry.entry_id,
            summary=entry.rendered_summary,
            status=entry.status,
            scope_type=entry.scope_type,
            scope_id=entry.scope_id,
            valid_from=entry.valid_from,
            valid_to=entry.valid_to,
            source_event_ids=_sorted_unique_texts(entry.source_event_ids),
            source_episode_ids=_sorted_unique_episode_ids(entry.source_episode_ids),
            supporting_claim_ids=_sorted_unique_texts(entry.source_claim_ids),
            details={
                "entry_kind": entry.entry_kind,
                "salience": entry.salience,
                "supersedes_entry_id": entry.supersedes_entry_id,
                "modality": entry.modality,
                "review_state": entry.review_state,
                "retention_class": entry.retention_class,
                "privacy_class": entry.privacy_class,
                "governance_reason_codes": list(entry.governance_reason_codes),
                "source_presence_scope_key": entry.source_presence_scope_key,
                "source_scene_entity_ids": list(entry.source_scene_entity_ids),
                "source_scene_affordance_ids": list(entry.source_scene_affordance_ids),
                "redacted_at": entry.redacted_at,
                "content": entry.content,
            },
        )
        add_node(entry_node, status_hint="current" if entry.status == "current" else "historical")
        if entry.status == "superseded":
            node_status_hints.setdefault(entry_node.node_id, set()).add("superseded")
        if multimodal is not None:
            scene_episode_entry_nodes.append((entry, entry_node))

        for claim_id in entry.source_claim_ids:
            claim_node_id = _stable_id(
                "graph_node", BrainContinuityGraphNodeKind.CLAIM.value, claim_id
            )
            if claim_node_id not in nodes:
                continue
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_CLAIM.value,
                        entry_node.node_id,
                        claim_node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_CLAIM.value,
                    from_node_id=entry_node.node_id,
                    to_node_id=claim_node_id,
                    status="supported",
                    valid_from=entry.valid_from,
                    valid_to=entry.valid_to,
                    source_event_ids=_sorted_unique_texts(entry.source_event_ids),
                    source_episode_ids=_sorted_unique_episode_ids(entry.source_episode_ids),
                    supporting_claim_ids=[claim_id],
                    details={"entry_kind": entry.entry_kind},
                )
            )

        for event_id in entry.source_event_ids:
            event_node = ensure_event_anchor(event_id)
            if event_node is None:
                continue
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_EVENT.value,
                        entry_node.node_id,
                        event_node.node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_EVENT.value,
                    from_node_id=entry_node.node_id,
                    to_node_id=event_node.node_id,
                    status="supported",
                    valid_from=entry.valid_from,
                    valid_to=entry.valid_to,
                    source_event_ids=[event_id],
                    source_episode_ids=_sorted_unique_episode_ids(entry.source_episode_ids),
                    supporting_claim_ids=_sorted_unique_texts(entry.source_claim_ids),
                    details={"entry_kind": entry.entry_kind},
                )
            )

        for episode_id in entry.source_episode_ids:
            ensure_episode_anchor(episode_id)

    commitment_map = {
        record.commitment_id: record for record in commitment_records if record.commitment_id
    }
    for record in commitment_map.values():
        commitment_node = BrainContinuityGraphNodeRecord(
            node_id=_stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.COMMITMENT.value,
                record.commitment_id,
            ),
            kind=BrainContinuityGraphNodeKind.COMMITMENT.value,
            backing_record_id=record.commitment_id,
            summary=record.title or record.commitment_id,
            status=record.status,
            scope_type=record.scope_type,
            scope_id=record.scope_id,
            valid_from=None,
            valid_to=record.completed_at,
            supporting_claim_ids=[],
            details={
                "goal_family": record.goal_family,
                "intent": record.intent,
                "current_goal_id": record.current_goal_id,
                "plan_revision": record.plan_revision,
                "resume_count": record.resume_count,
                "blocked_reason": (
                    record.blocked_reason.as_dict() if record.blocked_reason is not None else None
                ),
                "wake_conditions": [item.as_dict() for item in record.wake_conditions],
                "details": dict(record.details),
            },
        )
        add_node(
            commitment_node,
            status_hint=(
                "current"
                if record.status not in {"completed", "cancelled", "failed"}
                else "historical"
            ),
        )

    for proposal_id in sorted(proposal_timelines):
        timeline = proposal_timelines[proposal_id]
        proposal = timeline["proposal"]
        proposed_event = timeline.get("proposed_event")
        adopted_event = timeline.get("adopted_event")
        rejected_event = timeline.get("rejected_event")
        status = "proposed"
        valid_to = None
        source_event_ids: list[str] = []
        if proposed_event is not None:
            source_event_ids.append(proposed_event.event_id)
        if adopted_event is not None:
            status = "adopted"
            source_event_ids.append(adopted_event.event_id)
        elif rejected_event is not None:
            status = "rejected"
            valid_to = rejected_event.ts
            source_event_ids.append(rejected_event.event_id)
        if proposal.plan_proposal_id in plan_superseded_ids:
            valid_to = valid_to or proposal.created_at
        proposal_node = BrainContinuityGraphNodeRecord(
            node_id=_stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value,
                proposal.plan_proposal_id,
            ),
            kind=BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value,
            backing_record_id=proposal.plan_proposal_id,
            summary=proposal.summary,
            status=status,
            scope_type=scope_type,
            scope_id=scope_id,
            valid_from=proposal.created_at
            or (proposed_event.ts if proposed_event is not None else None)
            or (adopted_event.ts if adopted_event is not None else None)
            or (rejected_event.ts if rejected_event is not None else None),
            valid_to=valid_to,
            source_event_ids=_sorted_unique_texts(source_event_ids),
            supporting_claim_ids=[],
            details={
                "goal_id": proposal.goal_id,
                "commitment_id": proposal.commitment_id,
                "source": proposal.source,
                "review_policy": proposal.review_policy,
                "current_plan_revision": proposal.current_plan_revision,
                "plan_revision": proposal.plan_revision,
                "preserved_prefix_count": proposal.preserved_prefix_count,
                "assumptions": list(proposal.assumptions),
                "missing_inputs": list(proposal.missing_inputs),
                "supersedes_plan_proposal_id": proposal.supersedes_plan_proposal_id,
                "procedural": dict(proposal.details.get("procedural", {})),
            },
        )
        add_node(
            proposal_node,
            status_hint="historical"
            if status == "rejected" or proposal.plan_proposal_id in plan_superseded_ids
            else "current",
        )
        if proposal.plan_proposal_id in plan_superseded_ids:
            node_status_hints.setdefault(proposal_node.node_id, set()).add("superseded")

        proposed_event_node = (
            ensure_event_anchor(proposed_event.event_id) if proposed_event is not None else None
        )
        if proposed_event_node is not None:
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        proposal_node.node_id,
                        proposed_event_node.node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                    from_node_id=proposal_node.node_id,
                    to_node_id=proposed_event_node.node_id,
                    status="supported",
                    valid_from=proposal.created_at or proposed_event.ts,
                    valid_to=None,
                    source_event_ids=[proposed_event.event_id],
                    details={"event_type": BrainEventType.PLANNING_PROPOSED},
                )
            )
        if adopted_event is not None:
            adopted_anchor = ensure_event_anchor(adopted_event.event_id)
            if adopted_anchor is not None:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            proposal_node.node_id,
                            adopted_anchor.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=proposal_node.node_id,
                        to_node_id=adopted_anchor.node_id,
                        status="supported",
                        valid_from=proposal.created_at or adopted_event.ts,
                        valid_to=None,
                        source_event_ids=[adopted_event.event_id],
                        details={"event_type": BrainEventType.PLANNING_ADOPTED},
                    )
                )
        if rejected_event is not None:
            rejected_anchor = ensure_event_anchor(rejected_event.event_id)
            if rejected_anchor is not None:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            proposal_node.node_id,
                            rejected_anchor.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=proposal_node.node_id,
                        to_node_id=rejected_anchor.node_id,
                        status="supported",
                        valid_from=proposal.created_at or rejected_event.ts,
                        valid_to=rejected_event.ts,
                        source_event_ids=[rejected_event.event_id],
                        details={"event_type": BrainEventType.PLANNING_REJECTED},
                    )
                )

        if proposal.commitment_id and proposal.commitment_id in commitment_map:
            commitment_node_id = _stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.COMMITMENT.value,
                proposal.commitment_id,
            )
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                        commitment_node_id,
                        proposal_node.node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                    from_node_id=commitment_node_id,
                    to_node_id=proposal_node.node_id,
                    status="linked",
                    valid_from=proposal.created_at
                    or (proposed_event.ts if proposed_event is not None else None)
                    or (adopted_event.ts if adopted_event is not None else None)
                    or (rejected_event.ts if rejected_event is not None else None),
                    valid_to=None,
                    source_event_ids=_sorted_unique_texts(source_event_ids),
                    details={"goal_id": proposal.goal_id},
                )
            )
            if adopted_event is not None:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                            proposal_node.node_id,
                            commitment_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                        from_node_id=proposal_node.node_id,
                        to_node_id=commitment_node_id,
                        status="adopted",
                        valid_from=adopted_event.ts,
                        valid_to=None,
                        source_event_ids=[adopted_event.event_id],
                        details={"goal_id": proposal.goal_id},
                    )
                )

        if proposal.supersedes_plan_proposal_id:
            prior_node_id = _stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value,
                proposal.supersedes_plan_proposal_id,
            )
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_SUPERSEDES.value,
                        proposal_node.node_id,
                        prior_node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_SUPERSEDES.value,
                    from_node_id=proposal_node.node_id,
                    to_node_id=prior_node_id,
                    status="superseded",
                    valid_from=proposal.created_at
                    or (proposed_event.ts if proposed_event is not None else None)
                    or (adopted_event.ts if adopted_event is not None else None)
                    or (rejected_event.ts if rejected_event is not None else None),
                    valid_to=None,
                    source_event_ids=_sorted_unique_texts(source_event_ids),
                    details={"goal_id": proposal.goal_id},
                )
            )

    for block in core_block_list:
        source_event_ts = (
            event_map[block.source_event_id].ts
            if block.source_event_id and block.source_event_id in event_map
            else None
        )
        block_node = _core_block_node(block, source_event_ts=source_event_ts)
        add_node(
            block_node,
            status_hint="current" if block.status == "current" else "historical",
        )
        if block.status == "superseded":
            node_status_hints.setdefault(block_node.node_id, set()).add("superseded")
        if block.source_event_id:
            event_node = ensure_event_anchor(block.source_event_id)
            if event_node is not None:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            block_node.node_id,
                            event_node.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=block_node.node_id,
                        to_node_id=event_node.node_id,
                        status="supported",
                        valid_from=source_event_ts or block.created_at,
                        valid_to=(
                            None
                            if source_event_ts is not None
                            else (block.updated_at if block.status != "current" else None)
                        ),
                        source_event_ids=[block.source_event_id],
                        details={"source": "core_memory_block"},
                    )
                )
        if block.supersedes_block_id:
            prior_node_id = _stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                block.supersedes_block_id,
            )
            add_edge(
                BrainContinuityGraphEdgeRecord(
                    edge_id=_stable_id(
                        "graph_edge",
                        BrainContinuityGraphEdgeKind.CORE_BLOCK_SUPERSEDES.value,
                        block_node.node_id,
                        prior_node_id,
                    ),
                    kind=BrainContinuityGraphEdgeKind.CORE_BLOCK_SUPERSEDES.value,
                    from_node_id=block_node.node_id,
                    to_node_id=prior_node_id,
                    status="superseded",
                    valid_from=source_event_ts or block.created_at,
                    valid_to=None,
                    source_event_ids=_sorted_unique_texts([block.source_event_id]),
                    details={"block_kind": block.block_kind},
                )
            )

    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            skill_node = _procedural_skill_node(skill)
            add_node(
                skill_node,
                status_hint="historical"
                if skill.status in {"superseded", "retired"}
                else "current",
            )
            if skill.status in {"superseded", "retired"}:
                node_status_hints.setdefault(skill_node.node_id, set()).add("superseded")
            for commitment_id in skill.supporting_commitment_ids:
                commitment_node_id = _stable_id(
                    "graph_node",
                    BrainContinuityGraphNodeKind.COMMITMENT.value,
                    commitment_id,
                )
                if commitment_node_id not in nodes:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPPORTS_COMMITMENT.value,
                            skill_node.node_id,
                            commitment_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPPORTS_COMMITMENT.value,
                        from_node_id=skill_node.node_id,
                        to_node_id=commitment_node_id,
                        status="supported",
                        valid_from=skill.created_at,
                        valid_to=skill.retired_at,
                        details={"skill_family_key": skill.skill_family_key},
                    )
                )
            for proposal_id in skill.supporting_plan_proposal_ids:
                proposal_node_id = _stable_id(
                    "graph_node",
                    BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value,
                    proposal_id,
                )
                if proposal_node_id not in nodes:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPPORTS_PLAN_PROPOSAL.value,
                            skill_node.node_id,
                            proposal_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPPORTS_PLAN_PROPOSAL.value,
                        from_node_id=skill_node.node_id,
                        to_node_id=proposal_node_id,
                        status="supported",
                        valid_from=skill.created_at,
                        valid_to=skill.retired_at,
                        details={"skill_family_key": skill.skill_family_key},
                    )
                )
            if skill.supersedes_skill_id:
                prior_skill_node_id = _stable_id(
                    "graph_node",
                    BrainContinuityGraphNodeKind.PROCEDURAL_SKILL.value,
                    skill.supersedes_skill_id,
                )
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPERSEDES.value,
                            skill_node.node_id,
                            prior_skill_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.PROCEDURAL_SKILL_SUPERSEDES.value,
                        from_node_id=skill_node.node_id,
                        to_node_id=prior_skill_node_id,
                        status="superseded",
                        valid_from=skill.created_at,
                        valid_to=None,
                        details={"skill_family_key": skill.skill_family_key},
                    )
                )

    if scene_world_state is not None:
        for entity in scene_world_state.entities:
            entity_node = _scene_entity_node(
                entity,
                scope_type=scene_world_state.scope_type,
                scope_id=scene_world_state.scope_id,
            )
            add_node(
                entity_node,
                status_hint="historical"
                if entity.state in {
                    BrainSceneWorldRecordState.CONTRADICTED.value,
                    BrainSceneWorldRecordState.EXPIRED.value,
                }
                else "current",
            )
            if entity.state == BrainSceneWorldRecordState.STALE.value:
                node_status_hints.setdefault(entity_node.node_id, set()).add("stale")
            for event_id in entity.source_event_ids:
                event_node = ensure_event_anchor(event_id)
                if event_node is None:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            entity_node.node_id,
                            event_node.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=entity_node.node_id,
                        to_node_id=event_node.node_id,
                        status="supported",
                        valid_from=entity.observed_at,
                        valid_to=entity.expires_at,
                        source_event_ids=[event_id],
                        details={"source": "scene_world_entity"},
                    )
                )
        for affordance in scene_world_state.affordances:
            affordance_node = _scene_affordance_node(
                affordance,
                scope_type=scene_world_state.scope_type,
                scope_id=scene_world_state.scope_id,
            )
            add_node(
                affordance_node,
                status_hint="current",
            )
            if affordance.availability in {
                BrainSceneWorldAffordanceAvailability.STALE.value,
                BrainSceneWorldAffordanceAvailability.UNCERTAIN.value,
            }:
                node_status_hints.setdefault(affordance_node.node_id, set()).add("stale")
            for event_id in affordance.source_event_ids:
                event_node = ensure_event_anchor(event_id)
                if event_node is None:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                            affordance_node.node_id,
                            event_node.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
                        from_node_id=affordance_node.node_id,
                        to_node_id=event_node.node_id,
                        status="supported",
                        valid_from=affordance.observed_at,
                        valid_to=affordance.expires_at,
                        source_event_ids=[event_id],
                        details={"source": "scene_world_affordance"},
                    )
                )
            entity_node_id = _stable_id(
                "graph_node",
                BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
                affordance.entity_id,
            )
            if entity_node_id in nodes:
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.SCENE_WORLD_ENTITY_HAS_AFFORDANCE.value,
                            entity_node_id,
                            affordance_node.node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.SCENE_WORLD_ENTITY_HAS_AFFORDANCE.value,
                        from_node_id=entity_node_id,
                        to_node_id=affordance_node.node_id,
                        status="linked",
                        valid_from=affordance.observed_at,
                        valid_to=affordance.expires_at,
                        source_event_ids=_sorted_unique_texts(affordance.source_event_ids),
                        details={"capability_family": affordance.capability_family},
                    )
                )
        for entry, entry_node in scene_episode_entry_nodes:
            for entity_id in entry.source_scene_entity_ids:
                entity_node_id = _stable_id(
                    "graph_node",
                    BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
                    entity_id,
                )
                if entity_node_id not in nodes:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_ENTITY.value,
                            entry_node.node_id,
                            entity_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_ENTITY.value,
                        from_node_id=entry_node.node_id,
                        to_node_id=entity_node_id,
                        status="supported",
                        valid_from=entry.valid_from,
                        valid_to=entry.valid_to,
                        source_event_ids=_sorted_unique_texts(entry.source_event_ids),
                        source_episode_ids=_sorted_unique_episode_ids(entry.source_episode_ids),
                        supporting_claim_ids=_sorted_unique_texts(entry.source_claim_ids),
                        details={"entry_kind": entry.entry_kind},
                    )
                )
            for affordance_id in entry.source_scene_affordance_ids:
                affordance_node_id = _stable_id(
                    "graph_node",
                    BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
                    affordance_id,
                )
                if affordance_node_id not in nodes:
                    continue
                add_edge(
                    BrainContinuityGraphEdgeRecord(
                        edge_id=_stable_id(
                            "graph_edge",
                            BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_AFFORDANCE.value,
                            entry_node.node_id,
                            affordance_node_id,
                        ),
                        kind=BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_AFFORDANCE.value,
                        from_node_id=entry_node.node_id,
                        to_node_id=affordance_node_id,
                        status="supported",
                        valid_from=entry.valid_from,
                        valid_to=entry.valid_to,
                        source_event_ids=_sorted_unique_texts(entry.source_event_ids),
                        source_episode_ids=_sorted_unique_episode_ids(entry.source_episode_ids),
                        supporting_claim_ids=_sorted_unique_texts(entry.source_claim_ids),
                        details={"entry_kind": entry.entry_kind},
                    )
                )

    adjacency: dict[str, set[str]] = {}
    for edge in edges.values():
        adjacency.setdefault(edge.from_node_id, set()).add(edge.to_node_id)
        adjacency.setdefault(edge.to_node_id, set()).add(edge.from_node_id)

    current_node_ids: list[str] = []
    historical_node_ids: list[str] = []
    stale_node_ids: list[str] = []
    superseded_node_ids: list[str] = []
    for node in sorted(nodes.values(), key=_node_sort_key):
        hints = node_status_hints.get(node.node_id, set())
        related_hints = {
            related_hint
            for related_id in adjacency.get(node.node_id, set())
            for related_hint in node_status_hints.get(related_id, set())
        }
        valid_to = _parse_ts(node.valid_to)
        is_expired = valid_to is not None and valid_to <= now_ts
        currentness_status = str(node.details.get("currentness_status", "")).strip()
        is_superseded = "superseded" in hints or node.status == "superseded"
        is_stale = "stale" in hints or currentness_status == BrainClaimCurrentnessStatus.STALE.value
        if is_stale:
            stale_node_ids.append(node.node_id)
        if is_superseded:
            superseded_node_ids.append(node.node_id)
        if node.kind in {
            BrainContinuityGraphNodeKind.ENTITY.value,
            BrainContinuityGraphNodeKind.EVENT_ANCHOR.value,
            BrainContinuityGraphNodeKind.EPISODE_ANCHOR.value,
        }:
            if "current" in related_hints:
                current_node_ids.append(node.node_id)
            elif related_hints or is_expired:
                historical_node_ids.append(node.node_id)
            continue
        if (
            "historical" in hints
            or node.status
            in {
                "rejected",
                "revoked",
                "completed",
                "cancelled",
                "failed",
                "retired",
                "expired",
                "contradicted",
            }
            or is_expired
            or is_superseded
        ):
            historical_node_ids.append(node.node_id)
        else:
            current_node_ids.append(node.node_id)

    sorted_nodes = sorted(nodes.values(), key=_node_sort_key)
    sorted_edges = sorted(edges.values(), key=_edge_sort_key)
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for node in sorted_nodes:
        node_counts[node.kind] = node_counts.get(node.kind, 0) + 1
    for edge in sorted_edges:
        edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1

    return BrainContinuityGraphProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        node_counts=node_counts,
        edge_counts=edge_counts,
        nodes=sorted_nodes,
        edges=sorted_edges,
        current_node_ids=_sorted_unique_texts(current_node_ids),
        historical_node_ids=_sorted_unique_texts(historical_node_ids),
        stale_node_ids=_sorted_unique_texts(stale_node_ids),
        superseded_node_ids=_sorted_unique_texts(superseded_node_ids),
    )


__all__ = [
    "BrainContinuityGraphEdgeKind",
    "BrainContinuityGraphEdgeRecord",
    "BrainContinuityGraphNodeKind",
    "BrainContinuityGraphNodeRecord",
    "BrainContinuityGraphProjection",
    "build_continuity_graph_projection",
]

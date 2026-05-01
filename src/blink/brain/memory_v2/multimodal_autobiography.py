"""Typed multimodal autobiography helpers for bounded scene-linked memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2.autobiography import (
    BrainAutobiographicalEntryRecord,
    BrainAutobiographyEntryKind,
)
from blink.brain.projections import (
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainGovernanceReasonCode,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldEntityKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)

_SCENE_EVENT_TYPES = {
    BrainEventType.PERCEPTION_OBSERVED,
    BrainEventType.SCENE_CHANGED,
    BrainEventType.ENGAGEMENT_CHANGED,
    BrainEventType.ATTENTION_CHANGED,
}
_SENSITIVE_EVENT_TYPES = {
    BrainEventType.ENGAGEMENT_CHANGED,
    BrainEventType.ATTENTION_CHANGED,
}


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


def _fingerprint(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class BrainMultimodalAutobiographyModality(str, Enum):
    """Supported multimodal autobiography modalities for the first bounded slice."""

    SCENE_WORLD = "scene_world"


class BrainMultimodalAutobiographyPrivacyClass(str, Enum):
    """Explicit privacy classes for bounded multimodal autobiography."""

    STANDARD = "standard"
    SENSITIVE = "sensitive"
    REDACTED = "redacted"


@dataclass(frozen=True)
class BrainMultimodalAutobiographyRecord:
    """Typed multimodal autobiography view over one canonical autobiography entry."""

    entry_id: str
    scope_type: str
    scope_id: str
    entry_kind: str
    modality: str
    review_state: str
    retention_class: str
    privacy_class: str
    governance_reason_codes: tuple[str, ...]
    last_governance_event_id: str | None
    source_presence_scope_key: str | None
    source_scene_entity_ids: tuple[str, ...]
    source_scene_affordance_ids: tuple[str, ...]
    redacted_at: str | None
    rendered_summary: str
    content: dict[str, Any]
    status: str
    salience: float
    source_episode_ids: tuple[int, ...]
    source_claim_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    supersedes_entry_id: str | None
    valid_from: str
    valid_to: str | None
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize the typed multimodal record."""
        return {
            "entry_id": self.entry_id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "entry_kind": self.entry_kind,
            "modality": self.modality,
            "review_state": self.review_state,
            "retention_class": self.retention_class,
            "privacy_class": self.privacy_class,
            "governance_reason_codes": list(self.governance_reason_codes),
            "last_governance_event_id": self.last_governance_event_id,
            "source_presence_scope_key": self.source_presence_scope_key,
            "source_scene_entity_ids": list(self.source_scene_entity_ids),
            "source_scene_affordance_ids": list(self.source_scene_affordance_ids),
            "redacted_at": self.redacted_at,
            "rendered_summary": self.rendered_summary,
            "content": dict(self.content),
            "status": self.status,
            "salience": self.salience,
            "source_episode_ids": list(self.source_episode_ids),
            "source_claim_ids": list(self.source_claim_ids),
            "source_event_ids": list(self.source_event_ids),
            "supersedes_entry_id": self.supersedes_entry_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None
    ) -> "BrainMultimodalAutobiographyRecord | None":
        """Hydrate one typed multimodal autobiography record from JSON."""
        if not isinstance(data, dict):
            return None
        entry_id = str(data.get("entry_id", "")).strip()
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        entry_kind = str(data.get("entry_kind", "")).strip()
        modality = str(data.get("modality", "")).strip()
        if not entry_id or not scope_type or not scope_id or not entry_kind or not modality:
            return None
        return cls(
            entry_id=entry_id,
            scope_type=scope_type,
            scope_id=scope_id,
            entry_kind=entry_kind,
            modality=modality,
            review_state=str(data.get("review_state", BrainClaimReviewState.NONE.value)).strip()
            or BrainClaimReviewState.NONE.value,
            retention_class=str(
                data.get("retention_class", BrainClaimRetentionClass.SESSION.value)
            ).strip()
            or BrainClaimRetentionClass.SESSION.value,
            privacy_class=str(
                data.get(
                    "privacy_class",
                    BrainMultimodalAutobiographyPrivacyClass.STANDARD.value,
                )
            ).strip()
            or BrainMultimodalAutobiographyPrivacyClass.STANDARD.value,
            governance_reason_codes=tuple(
                _sorted_unique_texts(data.get("governance_reason_codes", []))
            ),
            last_governance_event_id=_optional_text(data.get("last_governance_event_id")),
            source_presence_scope_key=_optional_text(data.get("source_presence_scope_key")),
            source_scene_entity_ids=tuple(
                _sorted_unique_texts(data.get("source_scene_entity_ids", []))
            ),
            source_scene_affordance_ids=tuple(
                _sorted_unique_texts(data.get("source_scene_affordance_ids", []))
            ),
            redacted_at=_optional_text(data.get("redacted_at")),
            rendered_summary=str(data.get("rendered_summary", "")).strip(),
            content=dict(data.get("content", {})),
            status=str(data.get("status", "current")).strip() or "current",
            salience=float(data.get("salience", 0.0)),
            source_episode_ids=tuple(
                _sorted_unique_episode_ids(data.get("source_episode_ids", []))
            ),
            source_claim_ids=tuple(_sorted_unique_texts(data.get("source_claim_ids", []))),
            source_event_ids=tuple(_sorted_unique_texts(data.get("source_event_ids", []))),
            supersedes_entry_id=_optional_text(data.get("supersedes_entry_id")),
            valid_from=str(data.get("valid_from", "")).strip(),
            valid_to=_optional_text(data.get("valid_to")),
            created_at=str(data.get("created_at", "")).strip(),
            updated_at=str(data.get("updated_at", "")).strip(),
        )


@dataclass(frozen=True)
class BrainSceneEpisodeDistillation:
    """Pure bounded distillation of scene state into a derived autobiographical episode."""

    rendered_summary: str
    content: dict[str, Any]
    salience: float
    review_state: str
    retention_class: str
    privacy_class: str
    governance_reason_codes: tuple[str, ...]
    source_presence_scope_key: str
    source_scene_entity_ids: tuple[str, ...]
    source_scene_affordance_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    semantic_fingerprint: str
    valid_from: str


def parse_multimodal_autobiography_record(
    entry: BrainAutobiographicalEntryRecord | None,
) -> BrainMultimodalAutobiographyRecord | None:
    """Return the typed multimodal view for one scene-episode autobiography entry."""
    if entry is None or entry.entry_kind != BrainAutobiographyEntryKind.SCENE_EPISODE.value:
        return None
    modality = _optional_text(entry.modality)
    if modality != BrainMultimodalAutobiographyModality.SCENE_WORLD.value:
        return None
    return BrainMultimodalAutobiographyRecord(
        entry_id=entry.entry_id,
        scope_type=entry.scope_type,
        scope_id=entry.scope_id,
        entry_kind=entry.entry_kind,
        modality=modality,
        review_state=entry.review_state or BrainClaimReviewState.NONE.value,
        retention_class=entry.retention_class or BrainClaimRetentionClass.SESSION.value,
        privacy_class=entry.privacy_class
        or BrainMultimodalAutobiographyPrivacyClass.STANDARD.value,
        governance_reason_codes=tuple(entry.governance_reason_codes),
        last_governance_event_id=entry.last_governance_event_id,
        source_presence_scope_key=entry.source_presence_scope_key,
        source_scene_entity_ids=tuple(entry.source_scene_entity_ids),
        source_scene_affordance_ids=tuple(entry.source_scene_affordance_ids),
        redacted_at=entry.redacted_at,
        rendered_summary=entry.rendered_summary,
        content=entry.content,
        status=entry.status,
        salience=entry.salience,
        source_episode_ids=tuple(entry.source_episode_ids),
        source_claim_ids=tuple(entry.source_claim_ids),
        source_event_ids=tuple(entry.source_event_ids),
        supersedes_entry_id=entry.supersedes_entry_id,
        valid_from=entry.valid_from,
        valid_to=entry.valid_to,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


def _scene_anchor_entities(scene_world_state: BrainSceneWorldProjection) -> list[dict[str, Any]]:
    active = [
        record
        for record in scene_world_state.entities
        if record.state == BrainSceneWorldRecordState.ACTIVE.value
    ]
    fallback = list(scene_world_state.entities)
    selected = (active or fallback)[:3]
    return [
        {
            "entity_id": record.entity_id,
            "entity_kind": record.entity_kind,
            "summary": record.summary,
            "confidence": record.confidence,
        }
        for record in selected
    ]


def _scene_anchor_affordances(scene_world_state: BrainSceneWorldProjection) -> list[dict[str, Any]]:
    available = [
        record
        for record in scene_world_state.affordances
        if record.availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value
    ]
    fallback = list(scene_world_state.affordances)
    selected = (available or fallback)[:2]
    return [
        {
            "affordance_id": record.affordance_id,
            "entity_id": record.entity_id,
            "capability_family": record.capability_family,
            "summary": record.summary,
            "confidence": record.confidence,
        }
        for record in selected
    ]


def _scene_supporting_events(
    *,
    scene_world_state: BrainSceneWorldProjection,
    recent_events: Iterable[BrainEventRecord],
) -> tuple[list[str], set[str]]:
    relevant_events = [
        event
        for event in recent_events
        if event.event_type in _SCENE_EVENT_TYPES
    ]
    relevant_events = sorted(
        relevant_events,
        key=lambda item: (_parse_ts(item.ts) or datetime.min.replace(tzinfo=UTC), item.event_id),
        reverse=True,
    )
    event_ids = {
        *(
            event.event_id
            for event in relevant_events[:8]
            if _optional_text(event.event_id) is not None
        ),
        *(
            event_id
            for record in scene_world_state.entities
            for event_id in record.source_event_ids
        ),
        *(
            event_id
            for record in scene_world_state.affordances
            for event_id in record.source_event_ids
        ),
    }
    event_types = {event.event_type for event in relevant_events[:8]}
    return _sorted_unique_texts(event_ids), event_types


def _scene_summary_text(
    *,
    scene_world_state: BrainSceneWorldProjection,
    anchor_entities: list[dict[str, Any]],
    anchor_affordances: list[dict[str, Any]],
) -> str:
    bits = [scene_world_state.degraded_mode]
    bits.extend(
        str(item.get("summary", "")).strip()
        for item in anchor_entities[:2]
        if str(item.get("summary", "")).strip()
    )
    affordance_bits = [
        str(item.get("capability_family", "")).strip()
        for item in anchor_affordances
        if str(item.get("capability_family", "")).strip()
    ]
    if affordance_bits:
        bits.append(f"affordances: {', '.join(affordance_bits[:2])}")
    return "; ".join(bit for bit in bits if bit).strip() or "Scene episode observed."


def distill_scene_episode(
    *,
    scene_world_state: BrainSceneWorldProjection,
    recent_events: Iterable[BrainEventRecord],
    presence_scope_key: str,
    reference_ts: str | None,
) -> BrainSceneEpisodeDistillation | None:
    """Distill bounded scene/perception state into one current scene episode."""
    if not presence_scope_key.strip():
        raise ValueError("presence_scope_key must not be empty")
    if (
        not scene_world_state.entities
        and not scene_world_state.affordances
        and scene_world_state.degraded_mode == "healthy"
    ):
        return None

    anchor_entities = _scene_anchor_entities(scene_world_state)
    anchor_affordances = _scene_anchor_affordances(scene_world_state)
    supporting_event_ids, supporting_event_types = _scene_supporting_events(
        scene_world_state=scene_world_state,
        recent_events=recent_events,
    )
    summary_text = _scene_summary_text(
        scene_world_state=scene_world_state,
        anchor_entities=anchor_entities,
        anchor_affordances=anchor_affordances,
    )
    semantic_fingerprint = _fingerprint(
        {
            "presence_scope_key": presence_scope_key,
            "summary": summary_text,
            "degraded_mode": scene_world_state.degraded_mode,
            "active_entity_ids": list(scene_world_state.active_entity_ids),
            "stale_entity_ids": list(scene_world_state.stale_entity_ids),
            "contradicted_entity_ids": list(scene_world_state.contradicted_entity_ids),
            "active_affordance_ids": list(scene_world_state.active_affordance_ids),
            "blocked_affordance_ids": list(scene_world_state.blocked_affordance_ids),
            "uncertain_affordance_ids": list(scene_world_state.uncertain_affordance_ids),
        }
    )
    confidence_values = [
        float(value)
        for value in [
            *(item.get("confidence") for item in anchor_entities),
            *(item.get("confidence") for item in anchor_affordances),
        ]
        if value is not None
    ]
    average_confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
    )
    confidence_band = "high" if average_confidence >= 0.75 else (
        "medium" if average_confidence >= 0.45 else "low"
    )
    sensitive = any(
        str(item.get("entity_kind", "")).strip() == BrainSceneWorldEntityKind.PERSON.value
        for item in anchor_entities
    ) or bool(_SENSITIVE_EVENT_TYPES & supporting_event_types)
    reason_codes: list[str] = []
    if sensitive:
        reason_codes.append(BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value)
    if scene_world_state.degraded_mode != "healthy":
        reason_codes.append(BrainGovernanceReasonCode.DEGRADED_SCENE_EVIDENCE.value)
    review_state = (
        BrainClaimReviewState.REQUESTED.value
        if sensitive or scene_world_state.degraded_mode != "healthy"
        else BrainClaimReviewState.NONE.value
    )
    privacy_class = (
        BrainMultimodalAutobiographyPrivacyClass.SENSITIVE.value
        if sensitive
        else BrainMultimodalAutobiographyPrivacyClass.STANDARD.value
    )
    observed_candidates = [
        *(record.observed_at for record in scene_world_state.entities[:3]),
        *(record.observed_at for record in scene_world_state.affordances[:2]),
        *(
            event.ts
            for event in recent_events
            if event.event_type in _SCENE_EVENT_TYPES and event.event_id in supporting_event_ids
        ),
    ]
    observed_at = max(
        (parsed for value in observed_candidates if (parsed := _parse_ts(value)) is not None),
        default=_parse_ts(scene_world_state.updated_at),
    )
    updated_candidates = [
        reference_ts,
        scene_world_state.updated_at,
        *(record.updated_at for record in scene_world_state.entities[:3]),
        *(record.updated_at for record in scene_world_state.affordances[:2]),
        *(record.observed_at for record in scene_world_state.entities[:3]),
        *(record.observed_at for record in scene_world_state.affordances[:2]),
        *(event.ts for event in recent_events if event.event_type in _SCENE_EVENT_TYPES),
    ]
    updated_at_dt = max(
        (parsed for value in updated_candidates if (parsed := _parse_ts(value)) is not None),
        default=datetime(1970, 1, 1, tzinfo=UTC),
    )
    updated_at = updated_at_dt.isoformat()
    salience = min(
        2.5,
        1.0
        + (0.4 if scene_world_state.degraded_mode != "healthy" else 0.0)
        + (0.25 if sensitive else 0.0)
        + (0.15 if anchor_affordances else 0.0),
    )
    content = {
        "summary": summary_text,
        "semantic_fingerprint": semantic_fingerprint,
        "degraded_mode": scene_world_state.degraded_mode,
        "degraded_reason_codes": list(scene_world_state.degraded_reason_codes),
        "anchor_entity_ids": [item["entity_id"] for item in anchor_entities],
        "anchor_affordance_ids": [item["affordance_id"] for item in anchor_affordances],
        "supporting_event_ids": list(supporting_event_ids),
        "confidence_band": confidence_band,
        "salience": round(float(salience), 3),
        "observed_at": observed_at.isoformat() if observed_at is not None else updated_at,
        "updated_at": updated_at,
    }
    return BrainSceneEpisodeDistillation(
        rendered_summary=summary_text,
        content=content,
        salience=round(float(salience), 3),
        review_state=review_state,
        retention_class=BrainClaimRetentionClass.SESSION.value,
        privacy_class=privacy_class,
        governance_reason_codes=tuple(_sorted_unique_texts(reason_codes)),
        source_presence_scope_key=presence_scope_key,
        source_scene_entity_ids=tuple(
            _sorted_unique_texts(item["entity_id"] for item in anchor_entities)
        ),
        source_scene_affordance_ids=tuple(
            _sorted_unique_texts(item["affordance_id"] for item in anchor_affordances)
        ),
        source_event_ids=tuple(_sorted_unique_texts(supporting_event_ids)),
        semantic_fingerprint=semantic_fingerprint,
        valid_from=updated_at,
    )


def build_multimodal_autobiography_digest(
    entries: Iterable[BrainAutobiographicalEntryRecord | BrainMultimodalAutobiographyRecord],
) -> dict[str, Any]:
    """Build compact operator-facing multimodal autobiography counts and recent rows."""
    records: list[BrainMultimodalAutobiographyRecord] = []
    for entry in entries:
        if isinstance(entry, BrainMultimodalAutobiographyRecord):
            records.append(entry)
            continue
        parsed = parse_multimodal_autobiography_record(entry)
        if parsed is not None:
            records.append(parsed)

    privacy_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    retention_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    current_ids: list[str] = []
    historical_ids: list[str] = []
    redacted_rows: list[dict[str, Any]] = []

    records = sorted(
        records,
        key=lambda item: (
            _parse_ts(item.updated_at) or datetime.min.replace(tzinfo=UTC),
            item.entry_id,
        ),
        reverse=True,
    )
    for record in records:
        privacy_counts[record.privacy_class] = privacy_counts.get(record.privacy_class, 0) + 1
        review_counts[record.review_state] = review_counts.get(record.review_state, 0) + 1
        retention_counts[record.retention_class] = retention_counts.get(record.retention_class, 0) + 1
        modality_counts[record.modality] = modality_counts.get(record.modality, 0) + 1
        if record.status == "current":
            current_ids.append(record.entry_id)
        else:
            historical_ids.append(record.entry_id)
        if record.privacy_class == BrainMultimodalAutobiographyPrivacyClass.REDACTED.value:
            redacted_rows.append(
                {
                    "entry_id": record.entry_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "modality": record.modality,
                    "review_state": record.review_state,
                    "retention_class": record.retention_class,
                    "redacted_at": record.redacted_at,
                    "rendered_summary": record.rendered_summary,
                    "source_event_ids": list(record.source_event_ids),
                    "source_scene_entity_ids": list(record.source_scene_entity_ids),
                    "source_scene_affordance_ids": list(record.source_scene_affordance_ids),
                }
            )
    return {
        "entry_counts": {
            "privacy": dict(sorted(privacy_counts.items())),
            "review": dict(sorted(review_counts.items())),
            "retention": dict(sorted(retention_counts.items())),
            "modality": dict(sorted(modality_counts.items())),
        },
        "current_entry_ids": current_ids,
        "historical_entry_ids": historical_ids,
        "recent_redacted_rows": redacted_rows[:6],
    }


__all__ = [
    "BrainMultimodalAutobiographyModality",
    "BrainMultimodalAutobiographyPrivacyClass",
    "BrainMultimodalAutobiographyRecord",
    "BrainSceneEpisodeDistillation",
    "build_multimodal_autobiography_digest",
    "distill_scene_episode",
    "parse_multimodal_autobiography_record",
]

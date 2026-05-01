"""Provider-light symbolic scene world-state projection helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainEngagementStateProjection,
    BrainSceneStateProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)

_SCENE_FRESH_SECS = 15
_DEFAULT_EXPIRE_MULTIPLIER = 4
_MAX_SCENE_ZONES = 3
_MAX_SCENE_ENTITIES = 4
_MAX_ENTITY_AFFORDANCES = 2
_ENTITY_RECORD_CAP = 12
_AFFORDANCE_RECORD_CAP = 12
_DEGRADED_UNAVAILABLE_REASONS = {
    "camera_disconnected",
    "presence_detector_unavailable",
    "presence_detector_invalid_frame",
}
_DEGRADED_LIMITED_REASONS = {
    "camera_frame_stale",
    "stale_frame",
    "vision_enrichment_unavailable",
    "vision_enrichment_parse_error",
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


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _slug(text: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return normalized or "unknown"


def _scene_is_fresh(scene: BrainSceneStateProjection, *, reference_dt: datetime | None) -> bool:
    fresh_dt = _parse_ts(scene.last_fresh_frame_at) or _parse_ts(scene.updated_at)
    if reference_dt is None:
        reference_dt = fresh_dt
    if fresh_dt is None or reference_dt is None:
        return bool(scene.camera_connected and scene.person_present != "uncertain")
    if scene.frame_age_ms is not None:
        return int(scene.frame_age_ms) <= (_SCENE_FRESH_SECS * 1000)
    return abs((reference_dt - fresh_dt).total_seconds()) <= _SCENE_FRESH_SECS


def _resolve_reference_ts(
    *,
    reference_ts: str | None,
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
                scene.updated_at,
                engagement.updated_at,
                body.updated_at,
            ]
        )
        or _utc_now()
    )


def _int_value(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _freshness_windows(payload: dict[str, Any] | None) -> tuple[int, int]:
    fresh_for_secs = _int_value((payload or {}).get("fresh_for_secs"), _SCENE_FRESH_SECS)
    expire_after_secs = _int_value(
        (payload or {}).get("expire_after_secs"),
        max(fresh_for_secs * _DEFAULT_EXPIRE_MULTIPLIER, fresh_for_secs + 15),
    )
    return fresh_for_secs, max(expire_after_secs, fresh_for_secs)


def _zone_key(payload: dict[str, Any]) -> str:
    key = (
        _optional_text(payload.get("zone_key"))
        or _optional_text(payload.get("zone_id"))
        or _optional_text(payload.get("id"))
        or _slug(
            _optional_text(payload.get("canonical_label"))
            or _optional_text(payload.get("label"))
            or _optional_text(payload.get("name"))
            or "camera_view"
        )
    )
    return f"zone:{key}"


def _entity_key(payload: dict[str, Any], *, zone_key: str | None) -> str:
    explicit = _optional_text(payload.get("entity_key")) or _optional_text(payload.get("entity_id"))
    if explicit is not None:
        return f"entity:{explicit}"
    entity_kind = (
        _optional_text(payload.get("entity_kind"))
        or _optional_text(payload.get("kind"))
        or BrainSceneWorldEntityKind.UNKNOWN.value
    )
    canonical_label = (
        _optional_text(payload.get("canonical_label"))
        or _optional_text(payload.get("label"))
        or _optional_text(payload.get("name"))
        or _optional_text(payload.get("summary"))
        or "unknown"
    )
    return _stable_id("entity", entity_kind, _slug(canonical_label), zone_key or "no-zone")


def _record_state_from_age(
    *,
    observed_at: str | None,
    reference_dt: datetime | None,
    fresh_for_secs: int,
    expire_after_secs: int,
) -> str:
    observed_dt = _parse_ts(observed_at)
    if observed_dt is None or reference_dt is None:
        return BrainSceneWorldRecordState.ACTIVE.value
    age_secs = max(0.0, (reference_dt - observed_dt).total_seconds())
    if age_secs <= fresh_for_secs:
        return BrainSceneWorldRecordState.ACTIVE.value
    if age_secs <= expire_after_secs:
        return BrainSceneWorldRecordState.STALE.value
    return BrainSceneWorldRecordState.EXPIRED.value


def _degraded_mode(
    *,
    scene: BrainSceneStateProjection,
    body: BrainPresenceSnapshot,
    reference_dt: datetime | None,
) -> tuple[str, list[str]]:
    reason_codes: set[str] = set()
    if not scene.camera_connected or body.camera_disconnected:
        reason_codes.add("camera_disconnected")
    if scene.sensor_health_reason:
        reason_codes.add(str(scene.sensor_health_reason))
    if body.sensor_health_reason:
        reason_codes.add(str(body.sensor_health_reason))
    if scene.enrichment_available is False:
        reason_codes.add("vision_enrichment_unavailable")
    if body.perception_unreliable or not _scene_is_fresh(scene, reference_dt=reference_dt):
        reason_codes.add("scene_stale")
    if scene.camera_track_state in {"stalled", "recovering"}:
        reason_codes.add(f"track_{scene.camera_track_state}")
    if any(code in _DEGRADED_UNAVAILABLE_REASONS for code in reason_codes):
        return "unavailable", sorted(reason_codes)
    if reason_codes:
        return "limited", sorted(reason_codes)
    return "healthy", []


def _zone_record(
    *,
    zone_key: str,
    label: str,
    summary: str,
    state: str,
    evidence_kind: str,
    confidence: float | None,
    freshness: str,
    source_event_ids: Iterable[str],
    backing_ids: Iterable[str],
    observed_at: str | None,
    updated_at: str,
    expires_at: str | None,
    details: dict[str, Any] | None = None,
) -> BrainSceneWorldEntityRecord:
    return BrainSceneWorldEntityRecord(
        entity_id=zone_key,
        entity_kind=BrainSceneWorldEntityKind.ZONE.value,
        canonical_label=label,
        summary=summary,
        state=state,
        evidence_kind=evidence_kind,
        confidence=confidence,
        freshness=freshness,
        backing_ids=_sorted_unique(backing_ids),
        source_event_ids=_sorted_unique(source_event_ids),
        observed_at=observed_at,
        updated_at=updated_at,
        expires_at=expires_at,
        details=dict(details or {}),
    )


def _entity_record(observation: dict[str, Any]) -> BrainSceneWorldEntityRecord:
    return BrainSceneWorldEntityRecord(
        entity_id=str(observation["record_id"]),
        entity_kind=str(observation["entity_kind"]),
        canonical_label=str(observation["canonical_label"]),
        summary=str(observation["summary"]),
        state=str(observation["state"]),
        evidence_kind=str(observation["evidence_kind"]),
        zone_id=_optional_text(observation.get("zone_id")),
        confidence=observation.get("confidence"),
        freshness=_optional_text(observation.get("freshness")),
        contradiction_codes=_sorted_unique(observation.get("contradiction_codes", [])),
        affordance_ids=_sorted_unique(observation.get("affordance_ids", [])),
        backing_ids=_sorted_unique(observation.get("backing_ids", [])),
        source_event_ids=_sorted_unique(observation.get("source_event_ids", [])),
        observed_at=_optional_text(observation.get("observed_at")),
        updated_at=str(observation.get("updated_at") or _utc_now()),
        expires_at=_optional_text(observation.get("expires_at")),
        details=dict(observation.get("details", {})),
    )


def _affordance_record(observation: dict[str, Any]) -> BrainSceneWorldAffordanceRecord:
    return BrainSceneWorldAffordanceRecord(
        affordance_id=str(observation["affordance_id"]),
        entity_id=str(observation["entity_id"]),
        capability_family=str(observation["capability_family"]),
        summary=str(observation["summary"]),
        availability=str(observation["availability"]),
        confidence=observation.get("confidence"),
        freshness=_optional_text(observation.get("freshness")),
        reason_codes=_sorted_unique(observation.get("reason_codes", [])),
        backing_ids=_sorted_unique(observation.get("backing_ids", [])),
        source_event_ids=_sorted_unique(observation.get("source_event_ids", [])),
        observed_at=_optional_text(observation.get("observed_at")),
        updated_at=str(observation.get("updated_at") or _utc_now()),
        expires_at=_optional_text(observation.get("expires_at")),
        details=dict(observation.get("details", {})),
    )


def _normalize_zone_payloads(payload: dict[str, Any], *, source_event_id: str, observed_at: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in list(payload.get("scene_zones", []) or [])[:_MAX_SCENE_ZONES]:
        if not isinstance(item, dict):
            continue
        label = (
            _optional_text(item.get("canonical_label"))
            or _optional_text(item.get("label"))
            or _optional_text(item.get("name"))
            or "camera_view"
        )
        zone_key = _zone_key(item)
        fresh_for_secs, expire_after_secs = _freshness_windows(item)
        normalized.append(
            {
                "zone_key": zone_key,
                "label": label,
                "summary": _optional_text(item.get("summary")) or f"Zone {label}",
                "confidence": float(item["confidence"]) if item.get("confidence") is not None else None,
                "observed_at": _optional_text(item.get("observed_at")) or observed_at,
                "fresh_for_secs": fresh_for_secs,
                "expire_after_secs": expire_after_secs,
                "source_event_ids": [source_event_id],
                "backing_ids": [zone_key],
                "details": dict(item),
            }
        )
    return normalized


def _normalize_entity_payloads(
    payload: dict[str, Any],
    *,
    source_event_id: str,
    observed_at: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in list(payload.get("scene_entities", []) or [])[:_MAX_SCENE_ENTITIES]:
        if not isinstance(item, dict):
            continue
        zone_key = (
            _optional_text(item.get("zone_key"))
            or _optional_text(item.get("zone_id"))
            or _optional_text(item.get("location_zone"))
        )
        zone_key = f"zone:{zone_key}" if zone_key and not zone_key.startswith("zone:") else zone_key
        entity_kind = (
            _optional_text(item.get("entity_kind"))
            or _optional_text(item.get("kind"))
            or BrainSceneWorldEntityKind.UNKNOWN.value
        )
        canonical_label = (
            _optional_text(item.get("canonical_label"))
            or _optional_text(item.get("label"))
            or _optional_text(item.get("name"))
            or _optional_text(item.get("summary"))
            or "unknown"
        )
        visibility = item.get("present")
        if visibility is None:
            visibility = item.get("visible")
        if visibility is None and _optional_text(item.get("state")) in {"absent", "removed", "gone"}:
            visibility = False
        fresh_for_secs, expire_after_secs = _freshness_windows(item)
        normalized.append(
            {
                "stable_key": _entity_key(item, zone_key=zone_key),
                "entity_kind": entity_kind,
                "canonical_label": canonical_label,
                "summary": _optional_text(item.get("summary")) or canonical_label,
                "zone_key": zone_key,
                "visible": visibility,
                "confidence": float(item["confidence"]) if item.get("confidence") is not None else None,
                "fresh_for_secs": fresh_for_secs,
                "expire_after_secs": expire_after_secs,
                "observed_at": _optional_text(item.get("observed_at")) or observed_at,
                "source_event_ids": [source_event_id],
                "backing_ids": [_entity_key(item, zone_key=zone_key)],
                "details": dict(item),
                "affordances": [
                    affordance
                    for affordance in list(item.get("affordances", []) or [])[:_MAX_ENTITY_AFFORDANCES]
                    if isinstance(affordance, dict)
                ],
            }
        )
    return normalized


def _fallback_entities_from_flat_scene(
    *,
    scene: BrainSceneStateProjection,
    source_event_id: str | None,
    observed_at: str,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    if scene.person_present == "present":
        entities.append(
            {
                "stable_key": "entity:current_user",
                "entity_kind": BrainSceneWorldEntityKind.PERSON.value,
                "canonical_label": "current_user",
                "summary": _optional_text(scene.last_visual_summary) or "User is present in the current camera view.",
                "zone_key": "zone:camera_view",
                "visible": True,
                "confidence": scene.confidence,
                "fresh_for_secs": _SCENE_FRESH_SECS,
                "expire_after_secs": _SCENE_FRESH_SECS * _DEFAULT_EXPIRE_MULTIPLIER,
                "observed_at": observed_at,
                "source_event_ids": [source_event_id] if source_event_id else [],
                "backing_ids": ["scene:current_user"],
                "details": {"derived_from_flat_scene": True},
                "affordances": [],
            }
        )
    if scene.person_present == "absent":
        entities.append(
            {
                "stable_key": "entity:current_user",
                "entity_kind": BrainSceneWorldEntityKind.PERSON.value,
                "canonical_label": "current_user",
                "summary": "User is absent from the current camera view.",
                "zone_key": "zone:camera_view",
                "visible": False,
                "confidence": scene.confidence,
                "fresh_for_secs": _SCENE_FRESH_SECS,
                "expire_after_secs": _SCENE_FRESH_SECS * _DEFAULT_EXPIRE_MULTIPLIER,
                "observed_at": observed_at,
                "source_event_ids": [source_event_id] if source_event_id else [],
                "backing_ids": ["scene:current_user"],
                "details": {"derived_from_flat_scene": True},
                "affordances": [],
            }
        )
    return entities


def _affordance_signature(affordances: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    for affordance in affordances:
        family = _optional_text(affordance.get("capability_family")) or _optional_text(
            affordance.get("capability_id")
        )
        availability = _optional_text(affordance.get("availability")) or BrainSceneWorldAffordanceAvailability.AVAILABLE.value
        if family is None:
            continue
        signature.append((family, availability))
    return tuple(sorted(signature))


def _observation_signature(observation: dict[str, Any]) -> tuple[str | None, str | None, tuple[tuple[str, str], ...]]:
    visible = observation.get("visible")
    visible_text = None if visible is None else ("visible" if bool(visible) else "hidden")
    return (
        _optional_text(observation.get("zone_key")),
        visible_text,
        _affordance_signature(list(observation.get("affordances", []))),
    )


def _contradiction_codes(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    previous_zone = _optional_text(previous.get("zone_key"))
    current_zone = _optional_text(current.get("zone_key"))
    if previous_zone != current_zone:
        codes.append("zone_changed")
    previous_visible = previous.get("visible")
    current_visible = current.get("visible")
    if previous_visible is not None and current_visible is not None and bool(previous_visible) != bool(current_visible):
        codes.append("presence_changed")
    if _affordance_signature(previous.get("affordances", [])) != _affordance_signature(
        current.get("affordances", [])
    ):
        codes.append("affordance_changed")
    return sorted(set(codes or ["scene_world_changed"]))


def _archive_state(previous: dict[str, Any], *, state: str, codes: Iterable[str], updated_at: str) -> dict[str, Any]:
    archived = deepcopy(previous)
    archived["state"] = state
    archived["contradiction_codes"] = _sorted_unique(
        list(archived.get("contradiction_codes", [])) + list(codes)
    )
    archived["updated_at"] = updated_at
    return archived


def _refresh_observation(previous: dict[str, Any], current: dict[str, Any], *, updated_at: str) -> dict[str, Any]:
    refreshed = deepcopy(previous)
    refreshed["summary"] = current["summary"]
    refreshed["zone_key"] = current.get("zone_key") or refreshed.get("zone_key")
    refreshed["confidence"] = current.get("confidence", refreshed.get("confidence"))
    refreshed["fresh_for_secs"] = current.get("fresh_for_secs", refreshed.get("fresh_for_secs"))
    refreshed["expire_after_secs"] = current.get("expire_after_secs", refreshed.get("expire_after_secs"))
    refreshed["observed_at"] = current.get("observed_at", refreshed.get("observed_at"))
    refreshed["updated_at"] = updated_at
    refreshed["source_event_ids"] = _sorted_unique(
        list(refreshed.get("source_event_ids", [])) + list(current.get("source_event_ids", []))
    )
    refreshed["backing_ids"] = _sorted_unique(
        list(refreshed.get("backing_ids", [])) + list(current.get("backing_ids", []))
    )
    refreshed["details"] = {**dict(refreshed.get("details", {})), **dict(current.get("details", {}))}
    refreshed["affordances"] = list(current.get("affordances", []))
    return refreshed


def _current_zone_records(
    *,
    scene: BrainSceneStateProjection,
    reference_dt: datetime | None,
    recent_events: list[BrainEventRecord],
) -> list[BrainSceneWorldEntityRecord]:
    zone_state = (
        BrainSceneWorldRecordState.ACTIVE.value
        if _scene_is_fresh(scene, reference_dt=reference_dt) and scene.camera_connected
        else BrainSceneWorldRecordState.STALE.value
    )
    fresh_for_secs, expire_after_secs = _freshness_windows({})
    fresh_dt = _parse_ts(scene.last_fresh_frame_at) or _parse_ts(scene.updated_at)
    expires_at = (
        (fresh_dt + timedelta(seconds=expire_after_secs)).isoformat()
        if fresh_dt is not None
        else None
    )
    recent_scene_event_ids = [
        event.event_id
        for event in recent_events
        if event.event_type
        in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.SCENE_CHANGED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.BODY_STATE_UPDATED,
        }
    ][:6]
    camera_view = _zone_record(
        zone_key="zone:camera_view",
        label="camera_view",
        summary="Current camera view",
        state=zone_state,
        evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
        confidence=scene.confidence,
        freshness="current" if zone_state == BrainSceneWorldRecordState.ACTIVE.value else "stale",
        source_event_ids=recent_scene_event_ids,
        backing_ids=["zone:camera_view"],
        observed_at=scene.last_observed_at or scene.updated_at,
        updated_at=scene.updated_at,
        expires_at=expires_at,
        details={
            "camera_connected": scene.camera_connected,
            "camera_track_state": scene.camera_track_state,
            "person_present": scene.person_present,
        },
    )
    return [camera_view]


def _materialize_affordances(
    *,
    entity: BrainSceneWorldEntityRecord,
    affordances: list[dict[str, Any]],
    degraded_mode: str,
    degraded_reason_codes: list[str],
    entity_state: str,
) -> list[BrainSceneWorldAffordanceRecord]:
    records: list[BrainSceneWorldAffordanceRecord] = []
    for affordance in affordances[:_MAX_ENTITY_AFFORDANCES]:
        capability_family = _optional_text(affordance.get("capability_family")) or _optional_text(
            affordance.get("capability_id")
        )
        if capability_family is None:
            continue
        availability = _optional_text(affordance.get("availability")) or BrainSceneWorldAffordanceAvailability.AVAILABLE.value
        reason_codes = _sorted_unique(affordance.get("reason_codes", []))
        if degraded_mode in {"limited", "unavailable"} and availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value:
            availability = BrainSceneWorldAffordanceAvailability.UNCERTAIN.value
            reason_codes = _sorted_unique(reason_codes + degraded_reason_codes)
        if entity_state in {
            BrainSceneWorldRecordState.STALE.value,
            BrainSceneWorldRecordState.EXPIRED.value,
        } and availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value:
            availability = BrainSceneWorldAffordanceAvailability.STALE.value
        if entity_state == BrainSceneWorldRecordState.CONTRADICTED.value and availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value:
            availability = BrainSceneWorldAffordanceAvailability.UNCERTAIN.value
            reason_codes = _sorted_unique(reason_codes + ["entity_contradicted"])
        observed_at = _optional_text(affordance.get("observed_at")) or entity.observed_at
        fresh_for_secs, expire_after_secs = _freshness_windows(affordance)
        observed_dt = _parse_ts(observed_at)
        expires_at = (
            (observed_dt + timedelta(seconds=expire_after_secs)).isoformat()
            if observed_dt is not None
            else entity.expires_at
        )
        records.append(
            _affordance_record(
                {
                    "affordance_id": _stable_id(
                        "scene_affordance",
                        entity.entity_id,
                        capability_family,
                        observed_at or entity.updated_at,
                    ),
                    "entity_id": entity.entity_id,
                    "capability_family": capability_family,
                    "summary": _optional_text(affordance.get("summary"))
                    or f"{entity.canonical_label} supports {capability_family}",
                    "availability": availability,
                    "confidence": (
                        float(affordance["confidence"])
                        if affordance.get("confidence") is not None
                        else entity.confidence
                    ),
                    "freshness": (
                        "current"
                        if availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value
                        else availability
                    ),
                    "reason_codes": reason_codes,
                    "backing_ids": [entity.entity_id, capability_family],
                    "source_event_ids": entity.source_event_ids,
                    "observed_at": observed_at,
                    "updated_at": entity.updated_at,
                    "expires_at": expires_at,
                    "details": {**dict(affordance), "fresh_for_secs": fresh_for_secs},
                }
            )
        )
    return records


def build_scene_world_state_projection(
    *,
    scope_type: str,
    scope_id: str,
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    body: BrainPresenceSnapshot,
    recent_events: list[BrainEventRecord],
    reference_ts: str | None = None,
) -> BrainSceneWorldProjection:
    """Build a replay-safe symbolic scene world-state projection."""
    resolved_reference_ts = _resolve_reference_ts(
        reference_ts=reference_ts,
        scene=scene,
        engagement=engagement,
        body=body,
        recent_events=recent_events,
    )
    reference_dt = _parse_ts(resolved_reference_ts)
    degraded_mode, degraded_reason_codes = _degraded_mode(
        scene=scene,
        body=body,
        reference_dt=reference_dt,
    )
    current_entities: dict[str, dict[str, Any]] = {}
    archived_entities: list[dict[str, Any]] = []
    current_zones: dict[str, dict[str, Any]] = {}

    relevant_events = sorted(
        [
            event
            for event in recent_events
            if event.event_type
            in {
                BrainEventType.PERCEPTION_OBSERVED,
                BrainEventType.SCENE_CHANGED,
                BrainEventType.ENGAGEMENT_CHANGED,
                BrainEventType.BODY_STATE_UPDATED,
            }
        ],
        key=lambda item: (int(getattr(item, "id", 0)), item.ts, item.event_id),
    )

    for event in relevant_events:
        payload = dict(event.payload or {})
        observed_at = _optional_text(payload.get("observed_at")) or event.ts
        for zone in _normalize_zone_payloads(payload, source_event_id=event.event_id, observed_at=observed_at):
            current_zones[zone["zone_key"]] = zone

        entity_payloads = _normalize_entity_payloads(
            payload,
            source_event_id=event.event_id,
            observed_at=observed_at,
        )
        if not entity_payloads:
            scene_fallback = BrainSceneStateProjection.from_dict(
                {
                    **scene.as_dict(),
                    "camera_connected": payload.get("camera_connected", scene.camera_connected),
                    "camera_track_state": payload.get("camera_track_state", scene.camera_track_state),
                    "person_present": payload.get("person_present", scene.person_present),
                    "last_visual_summary": payload.get("summary", scene.last_visual_summary),
                    "last_observed_at": payload.get("observed_at", scene.last_observed_at),
                    "last_fresh_frame_at": payload.get("last_fresh_frame_at", scene.last_fresh_frame_at),
                    "frame_age_ms": payload.get("frame_age_ms", scene.frame_age_ms),
                    "sensor_health_reason": payload.get("sensor_health_reason", scene.sensor_health_reason),
                    "confidence": payload.get("confidence", scene.confidence),
                    "updated_at": event.ts,
                }
            )
            entity_payloads = _fallback_entities_from_flat_scene(
                scene=scene_fallback,
                source_event_id=event.event_id,
                observed_at=observed_at,
            )

        camera_disconnected = not bool(payload.get("camera_connected", scene.camera_connected))
        if camera_disconnected:
            for key, previous in list(current_entities.items()):
                archived_entities.append(
                    _archive_state(
                        previous,
                        state=BrainSceneWorldRecordState.EXPIRED.value,
                        codes=["camera_disconnected"],
                        updated_at=event.ts,
                    )
                )
                current_entities.pop(key, None)

        for candidate in entity_payloads:
            if candidate.get("zone_key") is None:
                candidate["zone_key"] = "zone:camera_view"
            stable_key = str(candidate["stable_key"])
            previous = current_entities.get(stable_key)
            visible = candidate.get("visible")
            if previous is None:
                if visible is False:
                    continue
                candidate["updated_at"] = event.ts
                current_entities[stable_key] = candidate
                continue
            if _observation_signature(previous) == _observation_signature(candidate):
                current_entities[stable_key] = _refresh_observation(previous, candidate, updated_at=event.ts)
                continue
            archived_entities.append(
                _archive_state(
                    previous,
                    state=BrainSceneWorldRecordState.CONTRADICTED.value,
                    codes=_contradiction_codes(previous, candidate),
                    updated_at=event.ts,
                )
            )
            if visible is False:
                current_entities.pop(stable_key, None)
                continue
            candidate["updated_at"] = event.ts
            current_entities[stable_key] = candidate

    zone_records = _current_zone_records(
        scene=scene,
        reference_dt=reference_dt,
        recent_events=recent_events,
    )
    for zone in list(current_zones.values())[:_MAX_SCENE_ZONES]:
        zone_state = _record_state_from_age(
            observed_at=zone.get("observed_at"),
            reference_dt=reference_dt,
            fresh_for_secs=int(zone.get("fresh_for_secs", _SCENE_FRESH_SECS)),
            expire_after_secs=int(zone.get("expire_after_secs", _SCENE_FRESH_SECS * _DEFAULT_EXPIRE_MULTIPLIER)),
        )
        if degraded_mode == "unavailable" and zone_state == BrainSceneWorldRecordState.ACTIVE.value:
            zone_state = BrainSceneWorldRecordState.STALE.value
        zone_records.append(
            _zone_record(
                zone_key=str(zone["zone_key"]),
                label=str(zone["label"]),
                summary=str(zone["summary"]),
                state=zone_state,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                confidence=zone.get("confidence"),
                freshness="current" if zone_state == BrainSceneWorldRecordState.ACTIVE.value else zone_state,
                source_event_ids=zone.get("source_event_ids", []),
                backing_ids=zone.get("backing_ids", []),
                observed_at=_optional_text(zone.get("observed_at")),
                updated_at=str(zone.get("observed_at") or resolved_reference_ts),
                expires_at=(
                    (_parse_ts(zone.get("observed_at")) + timedelta(seconds=int(zone.get("expire_after_secs", 60)))).isoformat()
                    if _parse_ts(zone.get("observed_at")) is not None
                    else None
                ),
                details=dict(zone.get("details", {})),
            )
        )

    entity_records: list[BrainSceneWorldEntityRecord] = []
    for candidate in archived_entities:
        state = str(candidate.get("state") or BrainSceneWorldRecordState.CONTRADICTED.value)
        observed_at = _optional_text(candidate.get("observed_at"))
        entity_records.append(
            _entity_record(
                {
                    "record_id": _stable_id(
                        "scene_entity",
                        candidate["stable_key"],
                        observed_at or candidate.get("updated_at") or resolved_reference_ts,
                    ),
                    "entity_kind": candidate["entity_kind"],
                    "canonical_label": candidate["canonical_label"],
                    "summary": candidate["summary"],
                    "state": state,
                    "evidence_kind": BrainSceneWorldEvidenceKind.OBSERVED.value,
                    "zone_id": candidate.get("zone_key"),
                    "confidence": candidate.get("confidence"),
                    "freshness": state,
                    "contradiction_codes": candidate.get("contradiction_codes", []),
                    "affordance_ids": [],
                    "backing_ids": candidate.get("backing_ids", []),
                    "source_event_ids": candidate.get("source_event_ids", []),
                    "observed_at": observed_at,
                    "updated_at": candidate.get("updated_at", resolved_reference_ts),
                    "expires_at": (
                        (_parse_ts(observed_at) + timedelta(seconds=int(candidate.get("expire_after_secs", 60)))).isoformat()
                        if _parse_ts(observed_at) is not None
                        else None
                    ),
                    "details": {**dict(candidate.get("details", {})), "stable_key": candidate["stable_key"]},
                }
            )
        )

    for candidate in current_entities.values():
        observed_at = _optional_text(candidate.get("observed_at"))
        fresh_for_secs = int(candidate.get("fresh_for_secs", _SCENE_FRESH_SECS))
        expire_after_secs = int(
            candidate.get("expire_after_secs", fresh_for_secs * _DEFAULT_EXPIRE_MULTIPLIER)
        )
        state = _record_state_from_age(
            observed_at=observed_at,
            reference_dt=reference_dt,
            fresh_for_secs=fresh_for_secs,
            expire_after_secs=expire_after_secs,
        )
        if degraded_mode == "limited" and state == BrainSceneWorldRecordState.ACTIVE.value:
            state = BrainSceneWorldRecordState.STALE.value
        if degraded_mode == "unavailable" and state == BrainSceneWorldRecordState.ACTIVE.value:
            state = BrainSceneWorldRecordState.STALE.value
        entity_records.append(
            _entity_record(
                {
                    "record_id": _stable_id(
                        "scene_entity",
                        candidate["stable_key"],
                        observed_at or candidate.get("updated_at") or resolved_reference_ts,
                    ),
                    "entity_kind": candidate["entity_kind"],
                    "canonical_label": candidate["canonical_label"],
                    "summary": candidate["summary"],
                    "state": state,
                    "evidence_kind": BrainSceneWorldEvidenceKind.OBSERVED.value,
                    "zone_id": candidate.get("zone_key"),
                    "confidence": candidate.get("confidence"),
                    "freshness": "current" if state == BrainSceneWorldRecordState.ACTIVE.value else state,
                    "contradiction_codes": (
                        degraded_reason_codes if state != BrainSceneWorldRecordState.ACTIVE.value and degraded_mode != "healthy" else []
                    ),
                    "affordance_ids": [],
                    "backing_ids": candidate.get("backing_ids", []),
                    "source_event_ids": candidate.get("source_event_ids", []),
                    "observed_at": observed_at,
                    "updated_at": candidate.get("updated_at", resolved_reference_ts),
                    "expires_at": (
                        (_parse_ts(observed_at) + timedelta(seconds=expire_after_secs)).isoformat()
                        if _parse_ts(observed_at) is not None
                        else None
                    ),
                    "details": {**dict(candidate.get("details", {})), "stable_key": candidate["stable_key"]},
                }
            )
        )

    all_entities = list(zone_records) + entity_records
    all_entities = sorted(
        all_entities,
        key=lambda item: (
            item.entity_kind != BrainSceneWorldEntityKind.ZONE.value,
            item.entity_kind,
            item.state,
            str(item.updated_at or ""),
            item.entity_id,
        ),
    )[:_ENTITY_RECORD_CAP]

    affordance_records: list[BrainSceneWorldAffordanceRecord] = []
    affordance_ids_by_entity: dict[str, list[str]] = {}
    for entity in all_entities:
        if entity.entity_kind == BrainSceneWorldEntityKind.ZONE.value:
            continue
        affordances = [
            item
            for item in list(entity.details.get("affordances", []) or [])
            if isinstance(item, dict)
        ]
        records = _materialize_affordances(
            entity=entity,
            affordances=affordances,
            degraded_mode=degraded_mode,
            degraded_reason_codes=degraded_reason_codes,
            entity_state=entity.state,
        )
        affordance_records.extend(records)
        affordance_ids_by_entity[entity.entity_id] = [record.affordance_id for record in records]
    affordance_records = sorted(
        affordance_records,
        key=lambda item: (item.capability_family, item.availability, str(item.updated_at or ""), item.affordance_id),
    )[:_AFFORDANCE_RECORD_CAP]

    all_entities = [
        BrainSceneWorldEntityRecord(
            entity_id=entity.entity_id,
            entity_kind=entity.entity_kind,
            canonical_label=entity.canonical_label,
            summary=entity.summary,
            state=entity.state,
            evidence_kind=entity.evidence_kind,
            zone_id=entity.zone_id,
            confidence=entity.confidence,
            freshness=entity.freshness,
            contradiction_codes=list(entity.contradiction_codes),
            affordance_ids=list(affordance_ids_by_entity.get(entity.entity_id, [])),
            backing_ids=list(entity.backing_ids),
            source_event_ids=list(entity.source_event_ids),
            observed_at=entity.observed_at,
            updated_at=entity.updated_at,
            expires_at=entity.expires_at,
            details={
                **dict(entity.details),
                **(
                    {
                        "affordance_capability_families": [
                            record.capability_family
                            for record in affordance_records
                            if record.entity_id == entity.entity_id
                        ]
                    }
                    if affordance_ids_by_entity.get(entity.entity_id)
                    else {}
                ),
            },
        )
        for entity in all_entities
    ]

    projection = BrainSceneWorldProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        entities=all_entities,
        affordances=affordance_records,
        degraded_mode=degraded_mode,
        degraded_reason_codes=degraded_reason_codes,
        updated_at=resolved_reference_ts,
    )
    projection.sync_lists()
    return projection


__all__ = ["build_scene_world_state_projection"]

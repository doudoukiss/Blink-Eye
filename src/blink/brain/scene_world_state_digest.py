"""Compact operator digest for symbolic scene world-state projections."""

from __future__ import annotations

from typing import Any


def build_scene_world_state_digest(*, scene_world_state: dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact digest for scene world-state operator surfaces."""
    projection = dict(scene_world_state or {})
    entities = [dict(item) for item in projection.get("entities", []) if isinstance(item, dict)]
    affordances = [dict(item) for item in projection.get("affordances", []) if isinstance(item, dict)]
    active_entities_by_zone: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        if entity.get("state") != "active":
            continue
        zone_id = str(entity.get("zone_id") or "unscoped")
        active_entities_by_zone.setdefault(zone_id, []).append(
            {
                "entity_id": entity.get("entity_id"),
                "entity_kind": entity.get("entity_kind"),
                "canonical_label": entity.get("canonical_label"),
                "summary": entity.get("summary"),
                "confidence": entity.get("confidence"),
                "affordance_ids": list(entity.get("affordance_ids", []))[:3],
                "source_event_ids": list(entity.get("source_event_ids", []))[:3],
            }
        )
    for items in active_entities_by_zone.values():
        items.sort(key=lambda item: (str(item.get("canonical_label", "")), str(item.get("entity_id", ""))))
    uncertain_affordance_ids = [
        str(item.get("affordance_id", "")).strip()
        for item in affordances
        if str(item.get("availability", "")).strip() in {"uncertain", "stale"}
        and str(item.get("affordance_id", "")).strip()
    ]
    return {
        "entity_counts": dict(projection.get("entity_counts", {})),
        "affordance_counts": dict(projection.get("affordance_counts", {})),
        "state_counts": dict(projection.get("state_counts", {})),
        "contradiction_counts": dict(projection.get("contradiction_counts", {})),
        "active_entities_by_zone": dict(sorted(active_entities_by_zone.items())),
        "stale_entity_ids": list(projection.get("stale_entity_ids", [])),
        "contradicted_entity_ids": list(projection.get("contradicted_entity_ids", [])),
        "expired_entity_ids": list(projection.get("expired_entity_ids", [])),
        "uncertain_affordance_ids": uncertain_affordance_ids,
        "degraded_mode": projection.get("degraded_mode", "healthy"),
        "degraded_reason_codes": list(projection.get("degraded_reason_codes", [])),
    }


__all__ = ["build_scene_world_state_digest"]

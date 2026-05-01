"""Compact operator digest for private working-memory projections."""

from __future__ import annotations

from typing import Any


def build_private_working_memory_digest(
    *,
    private_working_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact digest for private working-memory operator surfaces."""
    projection = dict(private_working_memory or {})
    records = [dict(item) for item in projection.get("records", []) if isinstance(item, dict)]
    active_records = [item for item in records if item.get("state") == "active"]
    compact_active_by_buffer: dict[str, list[dict[str, Any]]] = {}
    unresolved_record_ids: list[str] = []
    for record in active_records:
        buffer_kind = str(record.get("buffer_kind", "")).strip()
        if not buffer_kind:
            continue
        compact_active_by_buffer.setdefault(buffer_kind, []).append(
            {
                "record_id": record.get("record_id"),
                "summary": record.get("summary"),
                "evidence_kind": record.get("evidence_kind"),
                "backing_ids": list(record.get("backing_ids", []))[:3],
                "source_event_ids": list(record.get("source_event_ids", []))[:3],
            }
        )
        if buffer_kind == "unresolved_uncertainty":
            record_id = str(record.get("record_id", "")).strip()
            if record_id:
                unresolved_record_ids.append(record_id)
    for items in compact_active_by_buffer.values():
        items.sort(key=lambda item: (str(item.get("summary", "")), str(item.get("record_id", ""))))
    return {
        "buffer_counts": dict(projection.get("buffer_counts", {})),
        "state_counts": dict(projection.get("state_counts", {})),
        "evidence_kind_counts": dict(projection.get("evidence_kind_counts", {})),
        "active_records_by_buffer": dict(sorted(compact_active_by_buffer.items())),
        "stale_record_ids": list(projection.get("stale_record_ids", [])),
        "resolved_record_ids": list(projection.get("resolved_record_ids", [])),
        "unresolved_record_ids": sorted(unresolved_record_ids),
    }


__all__ = ["build_private_working_memory_digest"]

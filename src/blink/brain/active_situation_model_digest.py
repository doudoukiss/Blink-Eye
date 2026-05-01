"""Compact operator digest for active situation-model projections."""

from __future__ import annotations

from typing import Any


def build_active_situation_model_digest(
    *,
    active_situation_model: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact digest for active situation-model operator surfaces."""
    projection = dict(active_situation_model or {})
    records = [dict(item) for item in projection.get("records", []) if isinstance(item, dict)]
    active_records = [item for item in records if item.get("state") == "active"]
    compact_active_by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in active_records:
        record_kind = str(record.get("record_kind", "")).strip()
        if not record_kind:
            continue
        compact_active_by_kind.setdefault(record_kind, []).append(
            {
                "record_id": record.get("record_id"),
                "summary": record.get("summary"),
                "evidence_kind": record.get("evidence_kind"),
                "uncertainty_codes": list(record.get("uncertainty_codes", []))[:3],
                "backing_ids": list(record.get("backing_ids", []))[:3],
                "source_event_ids": list(record.get("source_event_ids", []))[:3],
                "goal_id": record.get("goal_id"),
                "commitment_id": record.get("commitment_id"),
                "plan_proposal_id": record.get("plan_proposal_id"),
                "skill_id": record.get("skill_id"),
                "prediction_id": dict(record.get("details", {})).get("prediction_id"),
                "prediction_kind": dict(record.get("details", {})).get("prediction_kind"),
                "confidence_band": dict(record.get("details", {})).get("confidence_band"),
                "prediction_role": dict(record.get("details", {})).get("prediction_role"),
            }
        )
    for items in compact_active_by_kind.values():
        items.sort(key=lambda item: (str(item.get("summary", "")), str(item.get("record_id", ""))))
    return {
        "kind_counts": dict(projection.get("kind_counts", {})),
        "state_counts": dict(projection.get("state_counts", {})),
        "uncertainty_code_counts": dict(projection.get("uncertainty_code_counts", {})),
        "active_records_by_kind": dict(sorted(compact_active_by_kind.items())),
        "stale_record_ids": list(projection.get("stale_record_ids", [])),
        "unresolved_record_ids": list(projection.get("unresolved_record_ids", [])),
        "linked_commitment_ids": list(projection.get("linked_commitment_ids", [])),
        "linked_plan_proposal_ids": list(projection.get("linked_plan_proposal_ids", [])),
        "linked_skill_ids": list(projection.get("linked_skill_ids", [])),
    }


__all__ = ["build_active_situation_model_digest"]

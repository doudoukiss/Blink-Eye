"""Compact operator digest for predictive world-model state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


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


def build_world_model_digest(
    *,
    predictive_world_model: dict[str, Any] | None,
    soon_expiring_within_seconds: int = 30,
) -> dict[str, Any]:
    """Build one compact digest for predictive world-model inspection."""
    projection = dict(predictive_world_model or {})
    active_predictions = [
        dict(item)
        for item in projection.get("active_predictions", [])
        if isinstance(item, dict)
    ]
    recent_resolutions = [
        dict(item)
        for item in projection.get("recent_resolutions", [])
        if isinstance(item, dict)
    ]
    now = max(
        (
            ts
            for ts in (
                _parse_ts(projection.get("updated_at")),
                *(_parse_ts(item.get("updated_at")) for item in active_predictions),
                *(_parse_ts(item.get("updated_at")) for item in recent_resolutions),
            )
            if ts is not None
        ),
        default=datetime.now(UTC),
    )
    deadline = now + timedelta(seconds=soon_expiring_within_seconds)
    soon_expiring_prediction_ids = sorted(
        {
            str(item.get("prediction_id", "")).strip()
            for item in active_predictions
            if (valid_to := _parse_ts(item.get("valid_to"))) is not None and valid_to <= deadline
        }
    )
    highest_risk_prediction_ids = sorted(
        {
            str(item.get("prediction_id", "")).strip()
            for item in active_predictions
            if list(item.get("risk_codes", []))
        }
    )
    active_predictions_by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in active_predictions:
        prediction_kind = str(item.get("prediction_kind", "")).strip()
        if not prediction_kind:
            continue
        active_predictions_by_kind.setdefault(prediction_kind, []).append(
            {
                "prediction_id": item.get("prediction_id"),
                "summary": item.get("summary"),
                "subject_id": item.get("subject_id"),
                "confidence_band": item.get("confidence_band"),
                "risk_codes": list(item.get("risk_codes", []))[:3],
                "valid_to": item.get("valid_to"),
            }
        )
    for rows in active_predictions_by_kind.values():
        rows.sort(
            key=lambda row: (
                str(row.get("confidence_band", "")),
                str(row.get("prediction_id", "")),
            )
        )
    return {
        "active_kind_counts": dict(projection.get("active_kind_counts", {})),
        "active_confidence_band_counts": dict(projection.get("active_confidence_band_counts", {})),
        "resolution_kind_counts": dict(projection.get("resolution_kind_counts", {})),
        "soon_expiring_prediction_ids": soon_expiring_prediction_ids,
        "highest_risk_prediction_ids": highest_risk_prediction_ids,
        "active_predictions_by_kind": dict(sorted(active_predictions_by_kind.items())),
        "recent_resolutions": [
            {
                "prediction_id": item.get("prediction_id"),
                "prediction_kind": item.get("prediction_kind"),
                "subject_id": item.get("subject_id"),
                "resolution_kind": item.get("resolution_kind"),
                "resolution_summary": item.get("resolution_summary"),
                "resolved_at": item.get("resolved_at"),
            }
            for item in recent_resolutions[:8]
        ],
        "calibration_summary": dict(projection.get("calibration_summary", {})),
    }


__all__ = ["build_world_model_digest"]

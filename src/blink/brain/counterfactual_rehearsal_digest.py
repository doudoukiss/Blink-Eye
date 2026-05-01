"""Compact operator digest for counterfactual rehearsal and calibration state."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_counterfactual_rehearsal_digest(
    *,
    counterfactual_rehearsal: dict[str, Any] | None,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build one compact digest for rehearsal requests, outcomes, and calibration."""
    projection = _mapping(counterfactual_rehearsal)
    open_requests = [
        dict(record)
        for record in projection.get("open_requests", [])
        if isinstance(record, dict)
    ]
    recent_rehearsals = [
        dict(record)
        for record in projection.get("recent_rehearsals", [])
        if isinstance(record, dict)
    ]
    recent_comparisons = [
        dict(record)
        for record in projection.get("recent_comparisons", [])
        if isinstance(record, dict)
    ]
    recommendation_counts: Counter[str] = Counter()
    risk_code_counts: Counter[str] = Counter()
    calibration_bucket_counts: Counter[str] = Counter()
    observed_outcome_counts: Counter[str] = Counter()
    recurrent_mismatch_patterns: Counter[str] = Counter()

    for record in recent_rehearsals:
        recommendation = str(record.get("decision_recommendation", "")).strip()
        if recommendation:
            recommendation_counts[recommendation] += 1
        for code in record.get("risk_codes", []) or []:
            text = str(code).strip()
            if text:
                risk_code_counts[text] += 1

    for record in recent_comparisons:
        calibration_bucket = str(record.get("calibration_bucket", "")).strip()
        observed_outcome_kind = str(record.get("observed_outcome_kind", "")).strip()
        recommendation = str(record.get("decision_recommendation", "")).strip()
        if calibration_bucket:
            calibration_bucket_counts[calibration_bucket] += 1
        if observed_outcome_kind:
            observed_outcome_counts[observed_outcome_kind] += 1
        mismatch_key = "|".join(
            item
            for item in (recommendation, calibration_bucket, observed_outcome_kind)
            if item
        )
        if mismatch_key and calibration_bucket not in {"", "aligned", "not_calibrated"}:
            recurrent_mismatch_patterns[mismatch_key] += 1
        for code in record.get("risk_codes", []) or []:
            text = str(code).strip()
            if text:
                risk_code_counts[text] += 1

    return {
        "scope_key": str(projection.get("scope_key", "")).strip(),
        "presence_scope_key": str(projection.get("presence_scope_key", "")).strip(),
        "open_request_count": len(open_requests),
        "recent_rehearsal_count": len(recent_rehearsals),
        "recent_comparison_count": len(recent_comparisons),
        "recommendation_counts": _sorted_counter(recommendation_counts),
        "risk_code_counts": _sorted_counter(risk_code_counts),
        "calibration_bucket_counts": _sorted_counter(calibration_bucket_counts),
        "observed_outcome_counts": _sorted_counter(observed_outcome_counts),
        "recurrent_mismatch_patterns": _sorted_counter(recurrent_mismatch_patterns),
        "recent_rehearsals": recent_rehearsals[:recent_limit],
        "recent_comparisons": recent_comparisons[:recent_limit],
        "calibration_summary": _mapping(projection.get("calibration_summary")),
        "open_rehearsal_ids": [
            str(item).strip()
            for item in projection.get("open_rehearsal_ids", [])
            if str(item).strip()
        ],
    }


__all__ = ["build_counterfactual_rehearsal_digest"]

"""Compact operator digest for active context packet traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from blink.brain.context.policy import all_brain_context_tasks

_TASK_NAMES = tuple(task.value for task in all_brain_context_tasks())


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _extract_identifiers(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    backing_ids: list[str] = []
    provenance_ids: list[str] = []
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            if key.endswith("_id") or key in {"backing_record_id"}:
                provenance_ids.append(value)
                if key in {
                    "claim_id",
                    "entry_id",
                    "commitment_id",
                    "goal_id",
                    "plan_proposal_id",
                    "skill_id",
                    "dossier_id",
                    "backing_record_id",
                    "record_id",
                    "entity_id",
                    "affordance_id",
                }:
                    backing_ids.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    provenance_ids.append(item)
                    if key in {
                        "backing_ids",
                        "claim_ids",
                        "entry_ids",
                        "graph_node_ids",
                        "support_plan_proposal_ids",
                        "supporting_claim_ids",
                    }:
                        backing_ids.append(item)
    return backing_ids, provenance_ids


def _summarize_scene_episode_trace(
    *,
    selected_items: list[dict[str, Any]],
    dropped_items: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_entry_ids: list[str] = []
    suppressed_entry_ids: list[str] = []
    selected_presence_scope_keys: list[str] = []
    privacy_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    retention_counts: Counter[str] = Counter()
    governance_reason_code_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()

    for record in selected_items + dropped_items:
        if str(record.get("item_type", "")).strip() != "scene_episode":
            continue
        provenance = dict(record.get("provenance") or {})
        entry_id = str(provenance.get("entry_id") or provenance.get("backing_record_id") or "").strip()
        privacy_class = str(provenance.get("privacy_class", "")).strip()
        review_state = str(provenance.get("review_state", "")).strip()
        retention_class = str(provenance.get("retention_class", "")).strip()
        source_presence_scope_key = str(provenance.get("source_presence_scope_key", "")).strip()
        if entry_id:
            if record in selected_items:
                selected_entry_ids.append(entry_id)
            else:
                suppressed_entry_ids.append(entry_id)
        if source_presence_scope_key and record in selected_items:
            selected_presence_scope_keys.append(source_presence_scope_key)
        if privacy_class:
            privacy_counts[privacy_class] += 1
        if review_state:
            review_counts[review_state] += 1
        if retention_class:
            retention_counts[retention_class] += 1
        if record in dropped_items:
            reason = str(record.get("reason", "")).strip()
            if reason:
                drop_reason_counts[reason] += 1
        for reason_code in record.get("governance_reason_codes", []) or []:
            code = str(reason_code).strip()
            if code:
                governance_reason_code_counts[code] += 1

    return {
        "selected_entry_ids": _sorted_unique(selected_entry_ids),
        "suppressed_entry_ids": _sorted_unique(suppressed_entry_ids),
        "selected_presence_scope_keys": _sorted_unique(selected_presence_scope_keys),
        "privacy_counts": dict(sorted(privacy_counts.items())),
        "review_counts": dict(sorted(review_counts.items())),
        "retention_counts": dict(sorted(retention_counts.items())),
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
        "governance_reason_code_counts": dict(sorted(governance_reason_code_counts.items())),
    }


def _summarize_prediction_trace(
    *,
    selected_items: list[dict[str, Any]],
    dropped_items: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_prediction_ids: list[str] = []
    suppressed_prediction_ids: list[str] = []
    selected_prediction_kinds: list[str] = []
    confidence_band_counts: Counter[str] = Counter()
    resolution_kind_counts: Counter[str] = Counter()
    risk_code_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()

    for record in selected_items + dropped_items:
        if str(record.get("item_type", "")).strip() != "prediction":
            continue
        provenance = dict(record.get("provenance") or {})
        prediction_id = str(provenance.get("prediction_id", "")).strip()
        prediction_kind = str(provenance.get("prediction_kind", "")).strip()
        confidence_band = str(provenance.get("confidence_band", "")).strip()
        resolution_kind = str(provenance.get("resolution_kind", "")).strip()
        if prediction_id:
            if record in selected_items:
                selected_prediction_ids.append(prediction_id)
            else:
                suppressed_prediction_ids.append(prediction_id)
        if prediction_kind and record in selected_items:
            selected_prediction_kinds.append(prediction_kind)
        if confidence_band:
            confidence_band_counts[confidence_band] += 1
        if resolution_kind:
            resolution_kind_counts[resolution_kind] += 1
        if record in dropped_items:
            reason = str(record.get("reason", "")).strip()
            if reason:
                drop_reason_counts[reason] += 1
        for risk_code in provenance.get("risk_codes", []) or []:
            code = str(risk_code).strip()
            if code:
                risk_code_counts[code] += 1

    return {
        "selected_prediction_ids": _sorted_unique(selected_prediction_ids),
        "suppressed_prediction_ids": _sorted_unique(suppressed_prediction_ids),
        "selected_prediction_kinds": _sorted_unique(selected_prediction_kinds),
        "confidence_band_counts": dict(sorted(confidence_band_counts.items())),
        "resolution_kind_counts": dict(sorted(resolution_kind_counts.items())),
        "risk_code_counts": dict(sorted(risk_code_counts.items())),
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
    }


def _summarize_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {
            "query_text": "",
            "temporal_mode": "current_first",
            "trace_verbosity": "standard",
            "static_token_usage": 0,
            "dynamic_budget": 0,
            "selected_anchor_counts": {},
            "selected_anchor_types": [],
            "selected_item_counts": {},
            "selected_temporal_counts": {},
            "selected_temporal_kinds": [],
            "drop_reason_counts": {},
            "selected_backing_ids": [],
            "selected_provenance_ids": [],
            "scene_episode_trace": {
                "selected_entry_ids": [],
                "suppressed_entry_ids": [],
                "selected_presence_scope_keys": [],
                "privacy_counts": {},
                "review_counts": {},
                "retention_counts": {},
                "drop_reason_counts": {},
                "governance_reason_code_counts": {},
            },
            "prediction_trace": {
                "selected_prediction_ids": [],
                "suppressed_prediction_ids": [],
                "selected_prediction_kinds": [],
                "confidence_band_counts": {},
                "resolution_kind_counts": {},
                "risk_code_counts": {},
                "drop_reason_counts": {},
            },
        }

    selected_anchors = list(trace.get("selected_anchors") or [])
    selected_items = list(trace.get("selected_items") or [])
    dropped_items = list(trace.get("dropped_items") or [])
    anchor_candidates = list(trace.get("anchor_candidates") or [])
    selected_anchor_counts = Counter(
        str(record.get("anchor_type"))
        for record in selected_anchors
        if str(record.get("anchor_type", "")).strip()
    )
    selected_item_counts = Counter(
        str(record.get("item_type"))
        for record in selected_items
        if str(record.get("item_type", "")).strip()
    )
    selected_availability_counts = Counter(
        str(record.get("availability_state"))
        for record in selected_items
        if str(record.get("availability_state", "")).strip()
    )
    selected_temporal_counts = Counter(
        str(record.get("temporal_kind"))
        for record in selected_items
        if str(record.get("temporal_kind", "")).strip()
    )
    drop_reason_counts = Counter(
        str(record.get("reason"))
        for record in dropped_items
        + [item for item in anchor_candidates if not item.get("selected")]
        if str(record.get("reason", "")).strip()
    )
    selected_backing_ids: list[str] = []
    selected_provenance_ids: list[str] = []
    annotated_backing_ids: list[str] = []
    suppressed_backing_ids: list[str] = []
    governance_reason_code_counts = Counter()
    governance_drop_reason_counts = Counter()
    for record in selected_anchors + selected_items:
        item_backing_ids, item_provenance_ids = _extract_identifiers(
            dict(record.get("provenance") or {})
        )
        selected_backing_ids.extend(item_backing_ids)
        selected_provenance_ids.extend(item_provenance_ids)
        if (
            str(record.get("availability_state", "")).strip() == "annotated"
            and record in selected_items
        ):
            annotated_backing_ids.extend(item_backing_ids)
        for reason_code in record.get("governance_reason_codes", []) or []:
            code = str(reason_code).strip()
            if code:
                governance_reason_code_counts[code] += 1
    for record in dropped_items:
        item_backing_ids, _item_provenance_ids = _extract_identifiers(
            dict(record.get("provenance") or {})
        )
        if str(record.get("reason", "")).strip() == "governance_suppressed":
            suppressed_backing_ids.extend(item_backing_ids)
            for reason_code in record.get("governance_reason_codes", []) or []:
                code = str(reason_code).strip()
                if code:
                    governance_drop_reason_counts[code] += 1
        for reason_code in record.get("governance_reason_codes", []) or []:
            code = str(reason_code).strip()
            if code:
                governance_reason_code_counts[code] += 1
    return {
        "query_text": str(trace.get("query_text") or ""),
        "temporal_mode": str(trace.get("temporal_mode") or "current_first"),
        "trace_verbosity": str(
            ((trace.get("mode_policy") or {}).get("trace_verbosity") or "standard")
        ),
        "static_token_usage": int(trace.get("static_token_usage") or 0),
        "dynamic_budget": int(trace.get("dynamic_token_budget") or 0),
        "selected_anchor_counts": dict(sorted(selected_anchor_counts.items())),
        "selected_anchor_types": sorted(selected_anchor_counts),
        "selected_item_counts": dict(sorted(selected_item_counts.items())),
        "selected_availability_counts": dict(sorted(selected_availability_counts.items())),
        "selected_temporal_counts": dict(sorted(selected_temporal_counts.items())),
        "selected_temporal_kinds": sorted(selected_temporal_counts),
        "drop_reason_counts": dict(sorted(drop_reason_counts.items())),
        "governance_drop_reason_counts": dict(sorted(governance_drop_reason_counts.items())),
        "governance_reason_code_counts": dict(sorted(governance_reason_code_counts.items())),
        "selected_backing_ids": _sorted_unique(selected_backing_ids),
        "selected_provenance_ids": _sorted_unique(selected_provenance_ids),
        "annotated_backing_ids": _sorted_unique(annotated_backing_ids),
        "suppressed_backing_ids": _sorted_unique(suppressed_backing_ids),
        "scene_episode_trace": _summarize_scene_episode_trace(
            selected_items=selected_items,
            dropped_items=dropped_items,
        ),
        "prediction_trace": _summarize_prediction_trace(
            selected_items=selected_items,
            dropped_items=dropped_items,
        ),
    }


def build_context_packet_digest(*, packet_traces: dict[str, Any] | None) -> dict[str, Any]:
    """Build a compact operator digest for task-aware packet traces."""
    traces = dict(packet_traces or {})
    task_names = tuple(dict.fromkeys([*_TASK_NAMES, *sorted(traces)]))
    return {task: _summarize_trace(traces.get(task)) for task in task_names}


__all__ = ["build_context_packet_digest"]

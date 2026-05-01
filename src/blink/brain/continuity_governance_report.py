"""Governance summary above continuity dossiers and graph state."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, dict):
            return payload
    return {}


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


def _build_embodied_execution_rows(
    *,
    embodied_executive: Any,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    projection = _as_mapping(embodied_executive)
    current_intent = _as_mapping(projection.get("current_intent"))
    if current_intent and str(current_intent.get("intent_id") or "").strip():
        intents_by_id = {str(current_intent.get("intent_id")): current_intent}
    else:
        intents_by_id = {}
    envelopes_by_id = {
        str(record.get("envelope_id")): dict(record)
        for record in projection.get("recent_action_envelopes", []) or []
        if str(record.get("envelope_id") or "").strip()
    }
    recoveries_by_trace = {
        str(record.get("trace_id")): dict(record)
        for record in projection.get("recent_recoveries", []) or []
        if str(record.get("trace_id") or "").strip()
    }
    decision_counts: Counter[str] = Counter()
    policy_posture_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for record in projection.get("recent_execution_traces", []) or []:
        trace = dict(record)
        trace_id = str(trace.get("trace_id") or "").strip()
        if not trace_id:
            continue
        intent = intents_by_id.get(str(trace.get("intent_id") or "").strip(), {})
        envelope = envelopes_by_id.get(str(trace.get("envelope_id") or "").strip(), {})
        recovery = recoveries_by_trace.get(trace_id, {})
        disposition = str(trace.get("disposition") or "").strip()
        status = str(trace.get("status") or "").strip()
        policy_snapshot = dict(envelope.get("policy_snapshot") or {})
        decision = (
            "repaired"
            if recovery
            else "deferred"
            if disposition == "defer" or status == "deferred"
            else "aborted"
            if disposition == "abort" or status == "aborted"
            else "selected"
        )
        decision_counts[decision] += 1
        policy_posture = str(
            intent.get("policy_posture") or policy_snapshot.get("action_posture") or ""
        ).strip()
        if policy_posture:
            policy_posture_counts[policy_posture] += 1
        if len(rows) >= max_rows:
            continue
        rows.append(
            {
                "decision": decision,
                "trace_id": trace_id,
                "intent_id": trace.get("intent_id"),
                "envelope_id": trace.get("envelope_id"),
                "goal_id": trace.get("goal_id"),
                "plan_proposal_id": envelope.get("plan_proposal_id") or intent.get("plan_proposal_id"),
                "step_index": trace.get("step_index"),
                "selected_action_id": intent.get("selected_action_id") or envelope.get("action_id"),
                "executor_kind": intent.get("executor_kind"),
                "executor_backend": envelope.get("executor_backend"),
                "policy_posture": policy_posture or None,
                "status": status,
                "disposition": disposition,
                "outcome_summary": trace.get("outcome_summary"),
                "reason_codes": _sorted_unique(
                    [
                        *[str(code).strip() for code in intent.get("reason_codes", []) or []],
                        *[str(code).strip() for code in envelope.get("reason_codes", []) or []],
                        *[str(code).strip() for code in trace.get("mismatch_codes", []) or []],
                        *[str(code).strip() for code in trace.get("repair_codes", []) or []],
                        *[str(code).strip() for code in recovery.get("reason_codes", []) or []],
                    ]
                ),
                "mismatch_codes": _sorted_unique(trace.get("mismatch_codes", []) or []),
                "repair_codes": _sorted_unique(
                    [
                        *[str(code).strip() for code in trace.get("repair_codes", []) or []],
                        *[str(code).strip() for code in recovery.get("reason_codes", []) or []],
                    ]
                ),
                "recovery_action_id": trace.get("recovery_action_id") or recovery.get("action_id"),
                "trace_ref": {
                    "trace_id": trace_id,
                    "intent_id": trace.get("intent_id"),
                    "envelope_id": trace.get("envelope_id"),
                },
            }
        )
    return rows, dict(sorted(decision_counts.items())), dict(sorted(policy_posture_counts.items()))


def build_continuity_governance_report(
    *,
    continuity_dossiers: Any,
    continuity_graph: Any,
    claim_governance: Any = None,
    packet_traces: dict[str, Any] | None = None,
    embodied_executive: Any = None,
    max_issue_rows: int = 12,
    max_graph_follow_up_ids: int = 12,
    max_suppressed_packet_rows: int = 12,
    max_multimodal_packet_rows: int = 16,
    max_embodied_execution_rows: int = 16,
) -> dict[str, Any]:
    """Build a compact governance report from dossiers, graph, and packet traces."""
    dossiers = _as_mapping(continuity_dossiers)
    graph = _as_mapping(continuity_graph)
    claim_governance_payload = _as_mapping(claim_governance)
    dossier_records = list(dossiers.get("dossiers") or [])
    open_issue_rows: list[dict[str, Any]] = []
    open_issue_counts = Counter()
    dossier_availability_counts_by_task: dict[str, Counter] = {}
    review_debt_dossier_ids: list[str] = []
    for dossier in dossier_records:
        dossier_id = str(dossier.get("dossier_id") or "")
        dossier_kind = str(dossier.get("kind") or "")
        governance = dict(dossier.get("governance") or {})
        if int(governance.get("review_debt_count") or 0) > 0:
            review_debt_dossier_ids.append(dossier_id)
        for task_availability in governance.get("task_availability", []) or []:
            task = str(task_availability.get("task") or "").strip()
            availability = str(task_availability.get("availability") or "").strip()
            if not task or not availability:
                continue
            dossier_availability_counts_by_task.setdefault(task, Counter())[availability] += 1
        for issue in dossier.get("open_issues", []) or []:
            kind = str(issue.get("kind") or "").strip()
            if not kind:
                continue
            open_issue_counts[kind] += 1
            if len(open_issue_rows) >= max_issue_rows:
                continue
            evidence = dict(issue.get("evidence") or {})
            open_issue_rows.append(
                {
                    "dossier_id": dossier_id,
                    "dossier_kind": dossier_kind,
                    "kind": kind,
                    "summary": issue.get("summary"),
                    "status": issue.get("status"),
                    "evidence": {
                        "claim_ids": list(evidence.get("claim_ids") or []),
                        "entry_ids": list(evidence.get("entry_ids") or []),
                        "source_event_ids": list(evidence.get("source_event_ids") or []),
                        "source_episode_ids": list(evidence.get("source_episode_ids") or []),
                        "graph_node_ids": list(evidence.get("graph_node_ids") or []),
                        "graph_edge_ids": list(evidence.get("graph_edge_ids") or []),
                    },
                }
            )

    node_by_id = {
        str(record.get("node_id")): record
        for record in graph.get("nodes", []) or []
        if str(record.get("node_id", "")).strip()
    }
    stale_graph_backing_ids = _sorted_unique(
        [
            str(node_by_id[node_id].get("backing_record_id") or "")
            for node_id in graph.get("stale_node_ids", []) or []
            if node_id in node_by_id
        ]
    )[:max_graph_follow_up_ids]
    superseded_graph_backing_ids = _sorted_unique(
        [
            str(node_by_id[node_id].get("backing_record_id") or "")
            for node_id in graph.get("superseded_node_ids", []) or []
            if node_id in node_by_id
        ]
    )[:max_graph_follow_up_ids]
    packet_temporal_modes = {
        task: str((trace or {}).get("temporal_mode") or "current_first")
        for task, trace in dict(packet_traces or {}).items()
    }
    packet_suppression_counts_by_task: dict[str, dict[str, int]] = {}
    suppressed_packet_rows: list[dict[str, Any]] = []
    multimodal_packet_rows: list[dict[str, Any]] = []
    (
        embodied_execution_rows,
        embodied_decision_counts,
        embodied_policy_posture_counts,
    ) = _build_embodied_execution_rows(
        embodied_executive=embodied_executive,
        max_rows=max_embodied_execution_rows,
    )
    for task, trace in dict(packet_traces or {}).items():
        selected_items = list((trace or {}).get("selected_items") or [])
        dropped_items = list((trace or {}).get("dropped_items") or [])
        task_counts = Counter()
        for decision, items in (("selected", selected_items), ("dropped", dropped_items)):
            for item in items:
                if str(item.get("item_type", "")).strip() != "scene_episode":
                    continue
                if len(multimodal_packet_rows) >= max_multimodal_packet_rows:
                    continue
                provenance = dict(item.get("provenance") or {})
                backing_ids, provenance_ids = _extract_identifiers(provenance)
                multimodal_packet_rows.append(
                    {
                        "task": task,
                        "decision": decision,
                        "item_id": item.get("item_id"),
                        "item_type": item.get("item_type"),
                        "section_key": item.get("section_key"),
                        "entry_id": provenance.get("entry_id") or provenance.get("backing_record_id"),
                        "backing_ids": _sorted_unique(backing_ids),
                        "provenance_ids": _sorted_unique(provenance_ids),
                        "reason": item.get("reason"),
                        "reason_codes": _sorted_unique(
                            [
                                *(
                                    str(code).strip()
                                    for code in item.get("governance_reason_codes", []) or []
                                ),
                                *(
                                    str(code).strip()
                                    for code in item.get("decision_reason_codes", []) or []
                                ),
                            ]
                        ),
                        "privacy_class": provenance.get("privacy_class"),
                        "review_state": provenance.get("review_state"),
                        "retention_class": provenance.get("retention_class"),
                        "source_presence_scope_key": provenance.get("source_presence_scope_key"),
                        "trace_ref": {
                            "item_id": item.get("item_id"),
                            "section_key": item.get("section_key"),
                            "reason": item.get("reason"),
                        },
                    }
                )
        for item in dropped_items:
            if str(item.get("reason") or "").strip() != "governance_suppressed":
                continue
            reason_codes = _sorted_unique(
                [str(code).strip() for code in item.get("governance_reason_codes", []) or []]
            )
            for reason_code in reason_codes:
                task_counts[reason_code] += 1
            if len(suppressed_packet_rows) >= max_suppressed_packet_rows:
                continue
            provenance = dict(item.get("provenance") or {})
            backing_ids, provenance_ids = _extract_identifiers(provenance)
            suppressed_packet_rows.append(
                {
                    "task": task,
                    "item_id": item.get("item_id"),
                    "item_type": item.get("item_type"),
                    "section_key": item.get("section_key"),
                    "dossier_id": provenance.get("dossier_id"),
                    "backing_ids": _sorted_unique(backing_ids),
                    "provenance_ids": _sorted_unique(provenance_ids),
                    "reason_codes": reason_codes,
                    "trace_ref": {
                        "item_id": item.get("item_id"),
                        "section_key": item.get("section_key"),
                        "reason": item.get("reason"),
                    },
                }
            )
        if task_counts:
            packet_suppression_counts_by_task[task] = dict(sorted(task_counts.items()))
    return {
        "claim_currentness_counts": dict(
            claim_governance_payload.get("currentness_counts") or {}
        ),
        "claim_review_state_counts": dict(
            claim_governance_payload.get("review_state_counts") or {}
        ),
        "claim_retention_class_counts": dict(
            claim_governance_payload.get("retention_class_counts") or {}
        ),
        "freshness_counts": dict(dossiers.get("freshness_counts") or {}),
        "contradiction_counts": dict(dossiers.get("contradiction_counts") or {}),
        "dossier_availability_counts_by_task": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(dossier_availability_counts_by_task.items())
        },
        "review_debt_dossier_ids": _sorted_unique(review_debt_dossier_ids),
        "open_issue_counts": dict(sorted(open_issue_counts.items())),
        "stale_dossier_ids": list(dossiers.get("stale_dossier_ids") or []),
        "needs_refresh_dossier_ids": list(dossiers.get("needs_refresh_dossier_ids") or []),
        "uncertain_dossier_ids": list(dossiers.get("uncertain_dossier_ids") or []),
        "contradicted_dossier_ids": list(dossiers.get("contradicted_dossier_ids") or []),
        "open_issue_rows": open_issue_rows,
        "stale_graph_backing_ids": stale_graph_backing_ids,
        "superseded_graph_backing_ids": superseded_graph_backing_ids,
        "packet_temporal_modes": packet_temporal_modes,
        "packet_suppression_counts_by_task": packet_suppression_counts_by_task,
        "suppressed_packet_rows": suppressed_packet_rows,
        "multimodal_packet_rows": multimodal_packet_rows,
        "embodied_decision_counts": embodied_decision_counts,
        "embodied_policy_posture_counts": embodied_policy_posture_counts,
        "embodied_execution_rows": embodied_execution_rows,
    }


__all__ = ["build_continuity_governance_report"]

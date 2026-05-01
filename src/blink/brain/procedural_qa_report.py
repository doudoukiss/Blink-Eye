"""Compact QA and proof summary for procedural-memory artifacts."""

from __future__ import annotations

import json
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


def _sorted_dict_counter(values: Counter[str]) -> dict[str, int]:
    return dict(sorted(values.items()))


def _filter_flow(record: Any) -> dict[str, Any] | None:
    payload = _as_mapping(record)
    if not payload:
        return None
    return {
        "plan_proposal_id": payload.get("plan_proposal_id"),
        "goal_id": payload.get("goal_id"),
        "commitment_id": payload.get("commitment_id"),
        "title": payload.get("title"),
        "procedural_origin": payload.get("procedural_origin"),
        "selected_skill_id": payload.get("selected_skill_id"),
        "selected_skill_support_trace_ids": [
            str(item).strip()
            for item in payload.get("selected_skill_support_trace_ids", [])
            if str(item).strip()
        ],
        "selected_skill_support_plan_proposal_ids": [
            str(item).strip()
            for item in payload.get("selected_skill_support_plan_proposal_ids", [])
            if str(item).strip()
        ],
        "rejected_skill_ids": [
            str(item).strip()
            for item in payload.get("rejected_skill_ids", [])
            if str(item).strip()
        ],
        "rejection_reasons": [
            str(item).strip()
            for item in payload.get("rejection_reasons", payload.get("skill_rejection_reasons", []))
            if str(item).strip()
        ],
        "decision_reason": str(payload.get("decision_reason", "")).strip() or None,
        "procedural_policy_effect": str(payload.get("procedural_policy_effect", "")).strip()
        or None,
        "procedural_policy_action_posture": str(
            payload.get("procedural_policy_action_posture", "")
        ).strip()
        or None,
        "procedural_policy_approval_requirement": str(
            payload.get("procedural_policy_approval_requirement", "")
        ).strip()
        or None,
        "procedural_policy_reason_codes": [
            str(item).strip()
            for item in payload.get("procedural_policy_reason_codes", [])
            if str(item).strip()
        ],
        "delta_operation_count": int(payload.get("delta_operation_count") or 0),
    }


def _filter_failure_signature(record: Any) -> dict[str, Any] | None:
    payload = _as_mapping(record)
    if not payload:
        return None
    reason_code = str(payload.get("reason_code", "")).strip()
    if not reason_code:
        return None
    return {
        "skill_id": payload.get("skill_id"),
        "kind": payload.get("kind"),
        "reason_code": reason_code,
        "summary": payload.get("summary"),
        "support_trace_ids": [
            str(item).strip()
            for item in payload.get("support_trace_ids", [])
            if str(item).strip()
        ],
        "support_outcome_ids": [
            str(item).strip()
            for item in payload.get("support_outcome_ids", [])
            if str(item).strip()
        ],
    }


def build_procedural_qa_state_excerpt(*, actual_state: Any) -> dict[str, Any]:
    """Return one bounded replay-safe procedural QA excerpt from continuity state."""
    state = _as_mapping(actual_state)
    planning = _as_mapping(state.get("planning_digest"))
    governance = _as_mapping(state.get("procedural_skill_governance_report"))
    skill_digest = _as_mapping(state.get("procedural_skill_digest"))
    packet_digest = _as_mapping(state.get("context_packet_digest"))
    recent_skill_flows = [
        flow
        for flow in (
            _filter_flow(record)
            for record in (
                list(planning.get("recent_skill_linked_proposals", []))
                + list(planning.get("recent_skill_linked_adoptions", []))
                + list(planning.get("recent_skill_linked_revision_flows", []))
            )
        )
        if flow is not None
    ]
    recent_negative_transfer_flows = [
        flow
        for flow in (
            _filter_flow(record)
            for record in governance.get("recent_negative_transfer_flows", [])
        )
        if flow is not None
    ]
    high_risk_failure_signatures = [
        signature
        for signature in (
            _filter_failure_signature(record)
            for record in governance.get("high_risk_failure_signatures", [])
        )
        if signature is not None
    ]
    return {
        "planning_digest": {
            "procedural_origin_counts": dict(planning.get("procedural_origin_counts", {})),
            "skill_rejection_reason_counts": dict(
                planning.get("skill_rejection_reason_counts", {})
            ),
            "delta_operation_counts": dict(planning.get("delta_operation_counts", {})),
            "recent_selected_skill_ids": list(planning.get("recent_selected_skill_ids", [])),
            "recent_skill_flows": recent_skill_flows,
        },
        "procedural_skill_governance_report": {
            "low_confidence_skill_ids": list(governance.get("low_confidence_skill_ids", [])),
            "retired_skill_ids": list(governance.get("retired_skill_ids", [])),
            "superseded_skill_ids": list(governance.get("superseded_skill_ids", [])),
            "follow_up_trace_ids": list(governance.get("follow_up_trace_ids", [])),
            "high_risk_failure_signatures": high_risk_failure_signatures,
            "policy_limited_reuse_counts": dict(
                governance.get("policy_limited_reuse_counts", {})
            ),
            "policy_limited_reason_code_counts": dict(
                governance.get("policy_limited_reason_code_counts", {})
            ),
            "recent_policy_limited_reuse_flows": [
                flow
                for flow in (
                    _filter_flow(record)
                    for record in governance.get("recent_policy_limited_reuse_flows", [])
                )
                if flow is not None
            ],
            "negative_transfer_reason_counts": dict(
                governance.get("negative_transfer_reason_counts", {})
            ),
            "rejected_reusable_skill_ids": list(
                governance.get("rejected_reusable_skill_ids", [])
            ),
            "recent_negative_transfer_flows": recent_negative_transfer_flows,
        },
        "procedural_skill_digest": {
            "active_skill_ids": list(skill_digest.get("active_skill_ids", [])),
            "candidate_skill_ids": list(skill_digest.get("candidate_skill_ids", [])),
        },
        "context_packet_digest": {
            "planning": {
                "selected_anchor_types": list(
                    _as_mapping(packet_digest.get("planning")).get("selected_anchor_types", [])
                ),
                "selected_backing_ids": list(
                    _as_mapping(packet_digest.get("planning")).get("selected_backing_ids", [])
                ),
            }
        },
    }


def _qa_sources(
    *,
    procedural_skill_digest: dict[str, Any],
    procedural_skill_governance_report: dict[str, Any],
    planning_digest: dict[str, Any],
    replay_regressions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = []
    for record in replay_regressions:
        excerpt = _as_mapping(
            record.get("procedural_qa_state_excerpt")
            or record.get("qa_state_excerpt")
            or record.get("state_excerpt")
        )
        if excerpt:
            sources.append(excerpt)
    if sources:
        return sources
    return [
        build_procedural_qa_state_excerpt(
            actual_state={
                "procedural_skill_digest": procedural_skill_digest,
                "procedural_skill_governance_report": procedural_skill_governance_report,
                "planning_digest": planning_digest,
            }
        )
    ]


def build_procedural_qa_report(
    *,
    procedural_skill_digest: Any,
    procedural_skill_governance_report: Any,
    planning_digest: Any,
    replay_regressions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact proof-oriented QA report for procedural memory."""
    skill_digest = _as_mapping(procedural_skill_digest)
    governance_report = _as_mapping(procedural_skill_governance_report)
    planning = _as_mapping(planning_digest)
    replay_results = list(replay_regressions or [])

    category_counts: Counter[str] = Counter()
    passed_case_names: list[str] = []
    failed_case_names: list[str] = []
    for record in replay_results:
        name = str(record.get("name", "")).strip()
        matched = bool(record.get("matched"))
        if name:
            if matched:
                passed_case_names.append(name)
            else:
                failed_case_names.append(name)
        for category in record.get("qa_categories", []) or []:
            category_text = str(category).strip()
            if category_text:
                category_counts[category_text] += 1

    coverage_flags = {
        category: category_counts.get(category, 0) > 0
        for category in (
            "skill_learning",
            "skill_reuse",
            "negative_transfer",
            "retirement",
            "supersession",
        )
    }

    procedural_origin_counts: Counter[str] = Counter()
    skill_rejection_reason_counts: Counter[str] = Counter()
    delta_operation_counts: Counter[str] = Counter()
    negative_transfer_reason_counts: Counter[str] = Counter()
    policy_limited_reuse_counts: Counter[str] = Counter()
    policy_limited_reason_code_counts: Counter[str] = Counter()
    low_confidence_skill_ids: list[str] = []
    retired_skill_ids: list[str] = []
    superseded_skill_ids: list[str] = []
    follow_up_trace_ids: list[str] = []
    active_skill_ids: list[str] = []
    candidate_skill_ids: list[str] = []
    selected_skill_ids: list[str] = []
    rejected_reusable_skill_ids: list[str] = []
    high_risk_failure_signatures: list[dict[str, Any]] = []
    recent_negative_transfer_flows: list[dict[str, Any]] = []
    recent_policy_limited_reuse_flows: list[dict[str, Any]] = []
    recent_skill_flows: list[dict[str, Any]] = []

    for source in _qa_sources(
        procedural_skill_digest=skill_digest,
        procedural_skill_governance_report=governance_report,
        planning_digest=planning,
        replay_regressions=replay_results,
    ):
        source_planning = _as_mapping(source.get("planning_digest"))
        source_governance = _as_mapping(source.get("procedural_skill_governance_report"))
        source_skill_digest = _as_mapping(source.get("procedural_skill_digest"))

        for key, value in source_planning.get("procedural_origin_counts", {}).items():
            procedural_origin_counts[str(key)] += int(value)
        for key, value in source_planning.get("skill_rejection_reason_counts", {}).items():
            skill_rejection_reason_counts[str(key)] += int(value)
        for key, value in source_planning.get("delta_operation_counts", {}).items():
            delta_operation_counts[str(key)] += int(value)
        for key, value in source_governance.get("negative_transfer_reason_counts", {}).items():
            negative_transfer_reason_counts[str(key)] += int(value)
        for key, value in source_governance.get("policy_limited_reuse_counts", {}).items():
            policy_limited_reuse_counts[str(key)] += int(value)
        for key, value in source_governance.get("policy_limited_reason_code_counts", {}).items():
            policy_limited_reason_code_counts[str(key)] += int(value)

        low_confidence_skill_ids.extend(
            str(item).strip()
            for item in source_governance.get("low_confidence_skill_ids", [])
            if str(item).strip()
        )
        retired_skill_ids.extend(
            str(item).strip()
            for item in source_governance.get("retired_skill_ids", [])
            if str(item).strip()
        )
        superseded_skill_ids.extend(
            str(item).strip()
            for item in source_governance.get("superseded_skill_ids", [])
            if str(item).strip()
        )
        follow_up_trace_ids.extend(
            str(item).strip()
            for item in source_governance.get("follow_up_trace_ids", [])
            if str(item).strip()
        )
        active_skill_ids.extend(
            str(item).strip()
            for item in source_skill_digest.get("active_skill_ids", [])
            if str(item).strip()
        )
        candidate_skill_ids.extend(
            str(item).strip()
            for item in source_skill_digest.get("candidate_skill_ids", [])
            if str(item).strip()
        )
        selected_skill_ids.extend(
            str(item).strip()
            for item in source_planning.get("recent_selected_skill_ids", [])
            if str(item).strip()
        )
        rejected_reusable_skill_ids.extend(
            str(item).strip()
            for item in source_governance.get("rejected_reusable_skill_ids", [])
            if str(item).strip()
        )
        recent_negative_transfer_flows.extend(
            flow
            for flow in (
                _filter_flow(record)
                for record in source_governance.get("recent_negative_transfer_flows", [])
            )
            if flow is not None
        )
        recent_policy_limited_reuse_flows.extend(
            flow
            for flow in (
                _filter_flow(record)
                for record in source_governance.get("recent_policy_limited_reuse_flows", [])
            )
            if flow is not None
        )
        recent_skill_flows.extend(
            flow
            for flow in (
                _filter_flow(record)
                for record in source_planning.get("recent_skill_flows", [])
            )
            if flow is not None
        )
        high_risk_failure_signatures.extend(
            signature
            for signature in (
                _filter_failure_signature(record)
                for record in source_governance.get("high_risk_failure_signatures", [])
            )
            if signature is not None
        )

    deduped_recent_negative_transfer_flows = [
        json.loads(item)
        for item in sorted(
            {
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in recent_negative_transfer_flows
            }
        )
    ]
    deduped_recent_skill_flows = [
        json.loads(item)
        for item in sorted(
            {
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in recent_skill_flows
            }
        )
    ]
    deduped_recent_policy_limited_reuse_flows = [
        json.loads(item)
        for item in sorted(
            {
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in recent_policy_limited_reuse_flows
            }
        )
    ]
    deduped_high_risk_failure_signatures = [
        json.loads(item)
        for item in sorted(
            {
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in high_risk_failure_signatures
            }
        )
    ]

    return {
        "case_counts": {
            "total": len(replay_results),
            "passed": len(passed_case_names),
            "failed": len(failed_case_names),
        },
        "category_counts": _sorted_dict_counter(category_counts),
        "coverage_flags": coverage_flags,
        "passed_case_names": sorted(passed_case_names),
        "failed_case_names": sorted(failed_case_names),
        "procedural_origin_counts": _sorted_dict_counter(procedural_origin_counts),
        "skill_rejection_reason_counts": _sorted_dict_counter(skill_rejection_reason_counts),
        "delta_operation_counts": _sorted_dict_counter(delta_operation_counts),
        "negative_transfer_reason_counts": _sorted_dict_counter(
            negative_transfer_reason_counts
        ),
        "policy_limited_reuse_counts": _sorted_dict_counter(policy_limited_reuse_counts),
        "policy_limited_reason_code_counts": _sorted_dict_counter(
            policy_limited_reason_code_counts
        ),
        "low_confidence_skill_ids": _sorted_unique(low_confidence_skill_ids),
        "retired_skill_ids": _sorted_unique(retired_skill_ids),
        "superseded_skill_ids": _sorted_unique(superseded_skill_ids),
        "follow_up_trace_ids": _sorted_unique(follow_up_trace_ids),
        "active_skill_ids": _sorted_unique(active_skill_ids),
        "candidate_skill_ids": _sorted_unique(candidate_skill_ids),
        "selected_skill_ids": _sorted_unique(selected_skill_ids),
        "rejected_reusable_skill_ids": _sorted_unique(rejected_reusable_skill_ids),
        "high_risk_failure_signature_codes": _sorted_unique(
            [
                str(record.get("reason_code", "")).strip()
                for record in deduped_high_risk_failure_signatures
            ]
        ),
        "high_risk_failure_signatures": deduped_high_risk_failure_signatures,
        "recent_policy_limited_reuse_flows": deduped_recent_policy_limited_reuse_flows,
        "recent_negative_transfer_flows": deduped_recent_negative_transfer_flows,
        "recent_skill_flows": deduped_recent_skill_flows,
    }


__all__ = [
    "build_procedural_qa_report",
    "build_procedural_qa_state_excerpt",
]

"""Governance summary above procedural skills and traces."""

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


def build_procedural_skill_governance_report(
    *,
    procedural_skills: Any,
    procedural_traces: Any,
    planning_digest: Any | None = None,
    max_failure_signatures: int = 10,
    max_follow_up_trace_ids: int = 16,
    max_negative_transfer_flows: int = 10,
) -> dict[str, Any]:
    """Build a compact governance report from procedural skills and traces."""
    skills_projection = _as_mapping(procedural_skills)
    traces_projection = _as_mapping(procedural_traces)
    planning = _as_mapping(planning_digest)
    skills = list(skills_projection.get("skills") or [])
    retirement_reason_counts = Counter(
        str(record.get("retirement_reason"))
        for record in skills
        if str(record.get("retirement_reason", "")).strip()
    )
    low_confidence_skill_ids = sorted(
        str(record.get("skill_id"))
        for record in skills
        if str(record.get("skill_id", "")).strip() and float(record.get("confidence", 0.0)) < 0.5
    )

    failure_signatures: list[dict[str, Any]] = []
    follow_up_trace_ids: list[str] = []
    trace_status_by_id = {
        str(record.get("trace_id")): str(record.get("status"))
        for record in traces_projection.get("traces", []) or []
        if str(record.get("trace_id", "")).strip()
    }
    for skill in skills:
        if float(skill.get("confidence", 0.0)) < 0.5:
            follow_up_trace_ids.extend(str(item) for item in skill.get("supporting_trace_ids", []) or [])
        if skill.get("status") in {"retired", "superseded"}:
            follow_up_trace_ids.extend(str(item) for item in skill.get("supporting_trace_ids", []) or [])
        for signature in skill.get("failure_signatures", []) or []:
            support_trace_ids = [
                str(item) for item in signature.get("support_trace_ids", []) or [] if str(item).strip()
            ]
            follow_up_trace_ids.extend(support_trace_ids)
            failure_signatures.append(
                {
                    "skill_id": skill.get("skill_id"),
                    "kind": signature.get("kind"),
                    "reason_code": signature.get("reason_code"),
                    "summary": signature.get("summary"),
                    "support_count": int((signature.get("details") or {}).get("support_count", 0)),
                    "support_trace_ids": support_trace_ids,
                    "support_outcome_ids": [
                        str(item)
                        for item in signature.get("support_outcome_ids", []) or []
                        if str(item).strip()
                    ],
                    "trace_statuses": dict(
                        sorted(
                            Counter(
                                trace_status_by_id[trace_id]
                                for trace_id in support_trace_ids
                                if trace_id in trace_status_by_id
                            ).items()
                        )
                    ),
                }
            )
    high_risk_failure_signatures = sorted(
        failure_signatures,
        key=lambda item: (
            -int(item.get("support_count", 0)),
            str(item.get("kind") or ""),
            str(item.get("reason_code") or ""),
            str(item.get("skill_id") or ""),
        ),
    )[:max_failure_signatures]
    negative_transfer_reason_counts: Counter[str] = Counter()
    policy_limited_reuse_counts: Counter[str] = Counter()
    policy_limited_reason_code_counts: Counter[str] = Counter()
    recent_negative_transfer_flows: list[dict[str, Any]] = []
    recent_policy_limited_reuse_flows: list[dict[str, Any]] = []
    for record in (
        list(planning.get("recent_skill_linked_proposals", []))
        + list(planning.get("recent_skill_linked_adoptions", []))
        + list(planning.get("recent_skill_linked_rejections", []))
        + list(planning.get("recent_skill_linked_revision_flows", []))
    ):
        if not isinstance(record, dict):
            continue
        rejected_skill_ids = [
            str(item).strip()
            for item in record.get("rejected_skill_ids", []) or []
            if str(item).strip()
        ]
        rejection_reasons = [
            str(item).strip()
            for item in record.get("skill_rejection_reasons", []) or []
            if str(item).strip()
        ]
        decision_reason = str(record.get("decision_reason", "")).strip()
        selected_skill_id = str(record.get("selected_skill_id", "")).strip()
        procedural_policy_effect = str(record.get("procedural_policy_effect", "")).strip()
        procedural_policy_reason_codes = [
            str(item).strip()
            for item in record.get("procedural_policy_reason_codes", []) or []
            if str(item).strip()
        ]
        if procedural_policy_effect in {"advisory_only", "blocked"}:
            policy_limited_reuse_counts[procedural_policy_effect] += 1
            for reason_code in procedural_policy_reason_codes:
                policy_limited_reason_code_counts[reason_code] += 1
            recent_policy_limited_reuse_flows.append(
                {
                    "plan_proposal_id": record.get("plan_proposal_id"),
                    "goal_id": record.get("goal_id"),
                    "commitment_id": record.get("commitment_id"),
                    "title": record.get("title"),
                    "procedural_origin": record.get("procedural_origin"),
                    "selected_skill_id": selected_skill_id or None,
                    "procedural_policy_effect": procedural_policy_effect,
                    "procedural_policy_action_posture": record.get(
                        "procedural_policy_action_posture"
                    ),
                    "procedural_policy_approval_requirement": record.get(
                        "procedural_policy_approval_requirement"
                    ),
                    "procedural_policy_reason_codes": procedural_policy_reason_codes,
                    "decision_reason": decision_reason or None,
                    "delta_operation_count": int(record.get("delta_operation_count") or 0),
                }
            )
        if not rejected_skill_ids and not rejection_reasons and decision_reason not in {
            "unknown_skill_candidate",
            "skill_not_reusable",
            "skill_reuse_mismatch",
            "skill_delta_out_of_bounds",
            "completed_prefix_mismatch",
        }:
            continue
        recent_negative_transfer_flows.append(
            {
                "plan_proposal_id": record.get("plan_proposal_id"),
                "goal_id": record.get("goal_id"),
                "commitment_id": record.get("commitment_id"),
                "title": record.get("title"),
                "procedural_origin": record.get("procedural_origin"),
                "selected_skill_id": selected_skill_id or None,
                "rejected_skill_ids": rejected_skill_ids,
                "rejection_reasons": rejection_reasons,
                "decision_reason": decision_reason or None,
                "selected_skill_support_trace_ids": [
                    str(item).strip()
                    for item in record.get("selected_skill_support_trace_ids", []) or []
                    if str(item).strip()
                ],
                "selected_skill_support_plan_proposal_ids": [
                    str(item).strip()
                    for item in record.get("selected_skill_support_plan_proposal_ids", []) or []
                    if str(item).strip()
                ],
                "delta_operation_count": int(record.get("delta_operation_count") or 0),
            }
        )
        follow_up_trace_ids.extend(
            str(item).strip()
            for item in record.get("selected_skill_support_trace_ids", []) or []
            if str(item).strip()
        )
        for reason in rejection_reasons:
            negative_transfer_reason_counts[reason] += 1
        if decision_reason:
            negative_transfer_reason_counts[decision_reason] += 1
    rejected_reusable_skill_ids = _sorted_unique(
        [
            str(record.get("selected_skill_id", "")).strip()
            for record in recent_negative_transfer_flows
            if str(record.get("selected_skill_id", "")).strip()
        ]
        + [
            skill_id
            for record in recent_negative_transfer_flows
            for skill_id in record.get("rejected_skill_ids", [])
        ]
    )
    return {
        "retirement_reason_counts": dict(sorted(retirement_reason_counts.items())),
        "low_confidence_skill_ids": low_confidence_skill_ids,
        "retired_skill_ids": list(skills_projection.get("retired_skill_ids") or []),
        "superseded_skill_ids": list(skills_projection.get("superseded_skill_ids") or []),
        "high_risk_failure_signatures": high_risk_failure_signatures,
        "policy_limited_reuse_counts": dict(sorted(policy_limited_reuse_counts.items())),
        "policy_limited_reason_code_counts": dict(sorted(policy_limited_reason_code_counts.items())),
        "negative_transfer_reason_counts": dict(sorted(negative_transfer_reason_counts.items())),
        "rejected_reusable_skill_ids": rejected_reusable_skill_ids,
        "recent_policy_limited_reuse_flows": recent_policy_limited_reuse_flows[
            -max_negative_transfer_flows:
        ],
        "recent_negative_transfer_flows": recent_negative_transfer_flows[
            -max_negative_transfer_flows:
        ],
        "follow_up_trace_ids": _sorted_unique(follow_up_trace_ids)[:max_follow_up_trace_ids],
    }


__all__ = ["build_procedural_skill_governance_report"]

"""Derived executive-policy audit summaries above existing loop digests."""

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


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _aggregate_counter_fields(
    target: Counter[str],
    *mappings: dict[str, Any],
) -> None:
    for mapping in mappings:
        for key, value in mapping.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            target[key_text] += int(value or 0)


def _tail(records: list[dict[str, Any]], *, recent_limit: int) -> list[dict[str, Any]]:
    if len(records) <= recent_limit:
        return list(records)
    return list(records[-recent_limit:])


def build_executive_policy_audit(
    *,
    autonomy_digest: Any,
    reevaluation_digest: Any,
    wake_digest: Any,
    planning_digest: Any,
    procedural_skill_governance_report: Any,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build one replay-safe executive-policy audit above existing digest surfaces."""
    presence = _as_mapping(autonomy_digest)
    reevaluation = _as_mapping(reevaluation_digest)
    wake = _as_mapping(wake_digest)
    planning = _as_mapping(planning_digest)
    procedural = _as_mapping(procedural_skill_governance_report)

    aggregate_policy_posture_counts: Counter[str] = Counter()
    aggregate_approval_requirement_counts: Counter[str] = Counter()
    aggregate_why_now_reason_code_counts: Counter[str] = Counter()
    aggregate_why_not_reason_code_counts: Counter[str] = Counter()

    for digest in (presence, reevaluation, wake, planning):
        _aggregate_counter_fields(
            aggregate_policy_posture_counts,
            dict(digest.get("policy_posture_counts", {})),
        )
        _aggregate_counter_fields(
            aggregate_approval_requirement_counts,
            dict(digest.get("approval_requirement_counts", {})),
        )
        _aggregate_counter_fields(
            aggregate_why_now_reason_code_counts,
            dict(digest.get("why_now_reason_code_counts", {})),
        )
        _aggregate_counter_fields(
            aggregate_why_not_reason_code_counts,
            dict(digest.get("why_not_reason_code_counts", {})),
        )

    return {
        "policy_posture_counts": _sorted_counter(aggregate_policy_posture_counts),
        "approval_requirement_counts": _sorted_counter(
            aggregate_approval_requirement_counts
        ),
        "why_now_reason_code_counts": _sorted_counter(
            aggregate_why_now_reason_code_counts
        ),
        "why_not_reason_code_counts": _sorted_counter(
            aggregate_why_not_reason_code_counts
        ),
        "loop_summaries": {
            "presence": {
                "policy_posture_counts": dict(presence.get("policy_posture_counts", {})),
                "approval_requirement_counts": dict(
                    presence.get("approval_requirement_counts", {})
                ),
                "why_now_reason_code_counts": dict(
                    presence.get("why_now_reason_code_counts", {})
                ),
                "why_not_reason_code_counts": dict(
                    presence.get("why_not_reason_code_counts", {})
                ),
                "recent_actions": _tail(
                    list(presence.get("recent_actions", [])),
                    recent_limit=recent_limit,
                ),
                "recent_non_actions": _tail(
                    list(presence.get("recent_non_actions", [])),
                    recent_limit=recent_limit,
                ),
            },
            "reevaluation": {
                "policy_posture_counts": dict(reevaluation.get("policy_posture_counts", {})),
                "approval_requirement_counts": dict(
                    reevaluation.get("approval_requirement_counts", {})
                ),
                "why_now_reason_code_counts": dict(
                    reevaluation.get("why_now_reason_code_counts", {})
                ),
                "why_not_reason_code_counts": dict(
                    reevaluation.get("why_not_reason_code_counts", {})
                ),
                "recent_transitions": _tail(
                    list(reevaluation.get("recent_transitions", [])),
                    recent_limit=recent_limit,
                ),
            },
            "wake": {
                "policy_posture_counts": dict(wake.get("policy_posture_counts", {})),
                "approval_requirement_counts": dict(
                    wake.get("approval_requirement_counts", {})
                ),
                "why_now_reason_code_counts": dict(
                    wake.get("why_now_reason_code_counts", {})
                ),
                "why_not_reason_code_counts": dict(
                    wake.get("why_not_reason_code_counts", {})
                ),
                "recent_direct_resumes": _tail(
                    list(wake.get("recent_direct_resumes", [])),
                    recent_limit=recent_limit,
                ),
                "recent_candidate_proposals": _tail(
                    list(wake.get("recent_candidate_proposals", [])),
                    recent_limit=recent_limit,
                ),
                "recent_keep_waiting": _tail(
                    list(wake.get("recent_keep_waiting", [])),
                    recent_limit=recent_limit,
                ),
            },
            "planning": {
                "policy_posture_counts": dict(planning.get("policy_posture_counts", {})),
                "approval_requirement_counts": dict(
                    planning.get("approval_requirement_counts", {})
                ),
                "why_now_reason_code_counts": dict(
                    planning.get("why_now_reason_code_counts", {})
                ),
                "why_not_reason_code_counts": dict(
                    planning.get("why_not_reason_code_counts", {})
                ),
                "recent_adoptions": _tail(
                    list(planning.get("recent_adoptions", [])),
                    recent_limit=recent_limit,
                ),
                "current_pending_proposals": _tail(
                    list(planning.get("current_pending_proposals", [])),
                    recent_limit=recent_limit,
                ),
                "recent_rejections": _tail(
                    list(planning.get("recent_rejections", [])),
                    recent_limit=recent_limit,
                ),
            },
            "procedural": {
                "policy_limited_reuse_counts": dict(
                    procedural.get("policy_limited_reuse_counts", {})
                ),
                "policy_limited_reason_code_counts": dict(
                    procedural.get("policy_limited_reason_code_counts", {})
                ),
                "recent_policy_limited_reuse_flows": _tail(
                    list(procedural.get("recent_policy_limited_reuse_flows", [])),
                    recent_limit=recent_limit,
                ),
            },
        },
    }


__all__ = ["build_executive_policy_audit"]

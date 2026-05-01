"""Operator-facing autonomy summaries derived from the raw autonomy ledger."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from blink.brain.autonomy import BrainAutonomyDecisionKind, BrainAutonomyLedgerProjection
from blink.brain.projections import BrainAgendaProjection


def _parse_ts(value: str | None) -> datetime | None:
    """Parse one stored ISO timestamp into UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _candidate_priority_key(candidate) -> tuple[float, float, datetime, str]:
    """Return the canonical candidate priority key used for family leaders."""
    created_at = _parse_ts(candidate.created_at) or datetime.min.replace(tzinfo=UTC)
    return (-float(candidate.urgency), -float(candidate.confidence), created_at, candidate.candidate_goal_id)


def build_autonomy_digest(
    *,
    autonomy_ledger: BrainAutonomyLedgerProjection,
    agenda: BrainAgendaProjection,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact operator-facing summary from the raw autonomy ledger."""
    goal_titles_by_id = {goal.goal_id: goal.title for goal in agenda.goals if goal.goal_id}
    reference_ts = _parse_ts(autonomy_ledger.updated_at)
    current_candidates = sorted(
        autonomy_ledger.current_candidates,
        key=lambda candidate: (_parse_ts(candidate.created_at) or datetime.min.replace(tzinfo=UTC), candidate.candidate_goal_id),
    )
    family_counts: Counter[str] = Counter(candidate.goal_family for candidate in current_candidates)
    family_leaders: dict[str, Any] = {}
    next_expiry_at: str | None = None
    current_candidate_payloads = [
        {
            "candidate_goal_id": candidate.candidate_goal_id,
            "candidate_type": candidate.candidate_type,
            "summary": candidate.summary,
            "source": candidate.source,
            "goal_family": candidate.goal_family,
            "initiative_class": candidate.initiative_class,
            "confidence": candidate.confidence,
            "pending_age_secs": (
                max(
                    0.0,
                    round(
                        (
                            reference_ts
                            - (_parse_ts(candidate.created_at) or reference_ts)
                        ).total_seconds(),
                        3,
                    ),
                )
                if reference_ts is not None
                else None
            ),
            "expires_at": candidate.expires_at,
            "expected_reevaluation_condition_kind": candidate.expected_reevaluation_condition_kind,
            "family_pending_count": family_counts.get(candidate.goal_family, 0),
        }
        for candidate in current_candidates
    ]
    for candidate in sorted(current_candidates, key=_candidate_priority_key):
        family_leaders.setdefault(candidate.goal_family, candidate)
        candidate_expiry = _parse_ts(candidate.expires_at)
        if candidate_expiry is None:
            continue
        if next_expiry_at is None or candidate_expiry < (_parse_ts(next_expiry_at) or candidate_expiry):
            next_expiry_at = candidate.expires_at
    pending_families = [
        {
            "goal_family": goal_family,
            "pending_count": family_counts[goal_family],
            "leader_candidate_goal_id": leader.candidate_goal_id,
            "leader_summary": leader.summary,
            "leader_initiative_class": leader.initiative_class,
        }
        for goal_family, leader in sorted(family_leaders.items())
    ]

    reason_counts: Counter[str] = Counter()
    reason_code_counts: Counter[str] = Counter()
    policy_posture_counts: Counter[str] = Counter()
    approval_requirement_counts: Counter[str] = Counter()
    why_now_reason_code_counts: Counter[str] = Counter()
    why_not_reason_code_counts: Counter[str] = Counter()
    decision_counts = {
        BrainAutonomyDecisionKind.ACCEPTED.value: 0,
        BrainAutonomyDecisionKind.SUPPRESSED.value: 0,
        BrainAutonomyDecisionKind.MERGED.value: 0,
        BrainAutonomyDecisionKind.EXPIRED.value: 0,
        BrainAutonomyDecisionKind.NON_ACTION.value: 0,
    }
    recent_decisions: list[dict[str, Any]] = []
    recent_actions: list[dict[str, Any]] = []
    recent_non_actions: list[dict[str, Any]] = []
    recent_suppressions: list[dict[str, Any]] = []
    recent_merges: list[dict[str, Any]] = []

    for entry in autonomy_ledger.recent_entries:
        decision_kind = str(entry.decision_kind or "").strip()
        if decision_kind == BrainAutonomyDecisionKind.CREATED.value:
            continue
        if decision_kind in decision_counts:
            decision_counts[decision_kind] += 1
        if entry.reason:
            reason_counts[entry.reason] += 1
        for reason_code in entry.reason_codes:
            reason_code_counts[reason_code] += 1
        if entry.executive_policy is not None:
            action_posture = str(entry.executive_policy.get("action_posture", "")).strip()
            approval_requirement = str(
                entry.executive_policy.get("approval_requirement", "")
            ).strip()
            if action_posture:
                policy_posture_counts[action_posture] += 1
            if approval_requirement:
                approval_requirement_counts[approval_requirement] += 1
        if decision_kind == BrainAutonomyDecisionKind.ACCEPTED.value:
            for reason_code in entry.reason_codes:
                why_now_reason_code_counts[reason_code] += 1
        elif decision_kind in {
            BrainAutonomyDecisionKind.NON_ACTION.value,
            BrainAutonomyDecisionKind.SUPPRESSED.value,
            BrainAutonomyDecisionKind.EXPIRED.value,
        }:
            for reason_code in entry.reason_codes:
                why_not_reason_code_counts[reason_code] += 1

        compact_entry = {
            "candidate_goal_id": entry.candidate_goal_id,
            "summary": entry.summary,
            "decision_kind": decision_kind,
            "reason": entry.reason,
            "reason_codes": list(entry.reason_codes),
            "executive_policy": (
                dict(entry.executive_policy) if entry.executive_policy is not None else None
            ),
            "goal_family": entry.reason_details.get("goal_family"),
            "accepted_goal_id": entry.accepted_goal_id,
            "accepted_goal_title": goal_titles_by_id.get(entry.accepted_goal_id or ""),
            "merged_into_candidate_goal_id": entry.merged_into_candidate_goal_id,
            "expected_reevaluation_condition": entry.expected_reevaluation_condition,
            "expected_reevaluation_condition_kind": entry.expected_reevaluation_condition_kind,
            "selected_by": entry.reason_details.get("selected_by"),
            "family_pending_count": entry.reason_details.get("family_pending_count"),
            "ts": entry.ts,
        }
        recent_decisions.append(compact_entry)

        if decision_kind == BrainAutonomyDecisionKind.ACCEPTED.value:
            recent_actions.append(
                {
                    "candidate_goal_id": entry.candidate_goal_id,
                    "summary": entry.summary,
                    "accepted_goal_id": entry.accepted_goal_id,
                    "accepted_goal_title": goal_titles_by_id.get(entry.accepted_goal_id or ""),
                    "goal_family": entry.reason_details.get("goal_family"),
                    "reason_codes": list(entry.reason_codes),
                    "executive_policy": (
                        dict(entry.executive_policy)
                        if entry.executive_policy is not None
                        else None
                    ),
                    "selected_by": entry.reason_details.get("selected_by"),
                    "family_pending_count": entry.reason_details.get("family_pending_count"),
                    "ts": entry.ts,
                }
            )
        elif decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value:
            recent_non_actions.append(
                {
                    "candidate_goal_id": entry.candidate_goal_id,
                    "summary": entry.summary,
                    "reason": entry.reason,
                    "reason_codes": list(entry.reason_codes),
                    "executive_policy": (
                        dict(entry.executive_policy)
                        if entry.executive_policy is not None
                        else None
                    ),
                    "expected_reevaluation_condition": entry.expected_reevaluation_condition,
                    "expected_reevaluation_condition_kind": entry.expected_reevaluation_condition_kind,
                    "goal_family": entry.reason_details.get("goal_family"),
                    "selected_by": entry.reason_details.get("selected_by"),
                    "family_pending_count": entry.reason_details.get("family_pending_count"),
                    "ts": entry.ts,
                }
            )
        elif decision_kind == BrainAutonomyDecisionKind.SUPPRESSED.value:
            recent_suppressions.append(
                {
                    "candidate_goal_id": entry.candidate_goal_id,
                    "summary": entry.summary,
                    "reason": entry.reason,
                    "ts": entry.ts,
                }
            )
        elif decision_kind == BrainAutonomyDecisionKind.MERGED.value:
            recent_merges.append(
                {
                    "candidate_goal_id": entry.candidate_goal_id,
                    "summary": entry.summary,
                    "reason": entry.reason,
                    "merged_into_candidate_goal_id": entry.merged_into_candidate_goal_id,
                    "ts": entry.ts,
                }
            )

    def _tail(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(records) <= recent_limit:
            return list(records)
        return list(records[-recent_limit:])

    return {
        "current_candidate_count": len(current_candidate_payloads),
        "current_candidates": current_candidate_payloads,
        "pending_family_counts": dict(sorted(family_counts.items())),
        "current_family_leaders": pending_families,
        "next_expiry_at": next_expiry_at,
        "decision_counts": decision_counts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "policy_posture_counts": dict(sorted(policy_posture_counts.items())),
        "approval_requirement_counts": dict(sorted(approval_requirement_counts.items())),
        "why_now_reason_code_counts": dict(sorted(why_now_reason_code_counts.items())),
        "why_not_reason_code_counts": dict(sorted(why_not_reason_code_counts.items())),
        "recent_decisions": _tail(recent_decisions),
        "recent_actions": _tail(recent_actions),
        "recent_non_actions": _tail(recent_non_actions),
        "recent_suppressions": _tail(recent_suppressions),
        "recent_merges": _tail(recent_merges),
    }


__all__ = ["build_autonomy_digest"]

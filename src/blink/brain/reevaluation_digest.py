"""Operator-facing reevaluation summaries derived from recent brain events."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainAutonomyLedgerProjection,
    BrainCandidateGoal,
    BrainReevaluationTrigger,
)
from blink.brain.events import BrainEventRecord, BrainEventType

_REEVALUATION_OUTCOME_BY_EVENT_TYPE = {
    BrainEventType.GOAL_CANDIDATE_ACCEPTED: BrainAutonomyDecisionKind.ACCEPTED.value,
    BrainEventType.GOAL_CANDIDATE_SUPPRESSED: BrainAutonomyDecisionKind.SUPPRESSED.value,
    BrainEventType.GOAL_CANDIDATE_EXPIRED: BrainAutonomyDecisionKind.EXPIRED.value,
    BrainEventType.DIRECTOR_NON_ACTION_RECORDED: BrainAutonomyDecisionKind.NON_ACTION.value,
}
_RELEVANT_EVENT_TYPES = {
    BrainEventType.GOAL_CANDIDATE_CREATED,
    BrainEventType.GOAL_CANDIDATE_ACCEPTED,
    BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
    BrainEventType.GOAL_CANDIDATE_EXPIRED,
    BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
    BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED,
}


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


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _event_sort_key(event: BrainEventRecord) -> tuple[int, datetime, str]:
    """Return one deterministic sort key for stored events."""
    return (
        int(getattr(event, "id", 0)),
        _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
        event.event_id,
    )


def _candidate_sort_key(candidate: BrainCandidateGoal) -> tuple[datetime, str]:
    """Return one stable display ordering for held candidates."""
    return (
        _parse_ts(candidate.created_at) or datetime.min.replace(tzinfo=UTC),
        candidate.candidate_goal_id,
    )


def _candidate_payload(
    candidate: BrainCandidateGoal,
    *,
    reference_ts: datetime | None,
    hold_reason: str | None = None,
    hold_reason_codes: list[str] | None = None,
    hold_executive_policy: dict[str, Any] | None = None,
    hold_reason_details: dict[str, Any] | None = None,
    held_at: str | None = None,
) -> dict[str, Any]:
    """Return one compact held-candidate payload."""
    created_at = _parse_ts(candidate.created_at)
    return {
        "candidate_goal_id": candidate.candidate_goal_id,
        "candidate_type": candidate.candidate_type,
        "summary": candidate.summary,
        "goal_family": candidate.goal_family,
        "initiative_class": candidate.initiative_class,
        "expires_at": candidate.expires_at,
        "expected_reevaluation_condition": candidate.expected_reevaluation_condition,
        "expected_reevaluation_condition_kind": candidate.expected_reevaluation_condition_kind,
        "pending_age_secs": (
            max(0.0, round((reference_ts - created_at).total_seconds(), 3))
            if reference_ts is not None and created_at is not None
            else None
        ),
        "hold_reason": hold_reason,
        "hold_reason_codes": list(hold_reason_codes or []),
        "hold_policy_action_posture": _optional_text(
            (hold_executive_policy or {}).get("action_posture")
        ),
        "hold_policy_approval_requirement": _optional_text(
            (hold_executive_policy or {}).get("approval_requirement")
        ),
        "hold_reason_details": dict(hold_reason_details or {}),
        "held_at": held_at,
    }


def build_reevaluation_digest(
    *,
    autonomy_ledger: BrainAutonomyLedgerProjection,
    recent_events: list[BrainEventRecord],
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact operator-facing summary for reevaluation flows."""
    reference_ts = _parse_ts(autonomy_ledger.updated_at)
    relevant_events = sorted(
        (event for event in recent_events if event.event_type in _RELEVANT_EVENT_TYPES),
        key=_event_sort_key,
    )
    candidates_by_id: dict[str, BrainCandidateGoal] = {
        candidate.candidate_goal_id: candidate for candidate in autonomy_ledger.current_candidates
    }
    for event in relevant_events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.event_type != BrainEventType.GOAL_CANDIDATE_CREATED:
            continue
        candidate = BrainCandidateGoal.from_dict(payload.get("candidate_goal"))
        if candidate.candidate_goal_id:
            candidates_by_id.setdefault(candidate.candidate_goal_id, candidate)

    trigger_events: dict[str, dict[str, Any]] = {}
    trigger_counts: Counter[str] = Counter()
    reason_code_counts: Counter[str] = Counter()
    policy_posture_counts: Counter[str] = Counter()
    approval_requirement_counts: Counter[str] = Counter()
    why_now_reason_code_counts: Counter[str] = Counter()
    why_not_reason_code_counts: Counter[str] = Counter()
    outcome_counts = {
        BrainAutonomyDecisionKind.ACCEPTED.value: 0,
        BrainAutonomyDecisionKind.SUPPRESSED.value: 0,
        BrainAutonomyDecisionKind.EXPIRED.value: 0,
        BrainAutonomyDecisionKind.NON_ACTION.value: 0,
    }
    hold_state: dict[str, dict[str, Any]] = {}
    current_holds: list[dict[str, Any]] = []
    recent_triggers: list[dict[str, Any]] = []
    recent_transitions: list[dict[str, Any]] = []

    for event in relevant_events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED:
            trigger = BrainReevaluationTrigger.from_dict(payload.get("trigger"))
            candidate_goal_ids = [
                str(candidate_goal_id)
                for candidate_goal_id in payload.get("candidate_goal_ids", [])
                if str(candidate_goal_id).strip()
            ]
            trigger_events[event.event_id] = {
                "event_id": event.event_id,
                "trigger": trigger,
                "candidate_goal_ids": candidate_goal_ids,
                "ts": event.ts,
            }
            if trigger.kind:
                trigger_counts[trigger.kind] += 1
            recent_triggers.append(
                {
                    "event_id": event.event_id,
                    "kind": trigger.kind,
                    "summary": trigger.summary,
                    "candidate_goal_ids": candidate_goal_ids,
                    "source_event_type": trigger.source_event_type,
                    "source_event_id": trigger.source_event_id,
                    "ts": event.ts,
                }
            )
            continue

        candidate_goal_id = str(payload.get("candidate_goal_id", "")).strip()
        if not candidate_goal_id:
            continue
        candidate = candidates_by_id.get(candidate_goal_id)
        prior_hold = hold_state.get(candidate_goal_id)
        trigger_record = trigger_events.get(event.causal_parent_id or "")
        outcome_kind = _REEVALUATION_OUTCOME_BY_EVENT_TYPE.get(event.event_type)
        reason_codes = sorted(
            {
                str(item).strip()
                for item in payload.get("reason_codes", [])
                if str(item).strip()
            }
        )
        executive_policy = (
            dict(payload.get("executive_policy", {}))
            if isinstance(payload.get("executive_policy"), dict)
            else None
        )
        for reason_code in reason_codes:
            reason_code_counts[reason_code] += 1
        if executive_policy is not None:
            action_posture = _optional_text(executive_policy.get("action_posture"))
            approval_requirement = _optional_text(
                executive_policy.get("approval_requirement")
            )
            if action_posture is not None:
                policy_posture_counts[action_posture] += 1
            if approval_requirement is not None:
                approval_requirement_counts[approval_requirement] += 1
        if outcome_kind == BrainAutonomyDecisionKind.ACCEPTED.value:
            for reason_code in reason_codes:
                why_now_reason_code_counts[reason_code] += 1
        elif outcome_kind in {
            BrainAutonomyDecisionKind.NON_ACTION.value,
            BrainAutonomyDecisionKind.SUPPRESSED.value,
            BrainAutonomyDecisionKind.EXPIRED.value,
        }:
            for reason_code in reason_codes:
                why_not_reason_code_counts[reason_code] += 1

        if trigger_record is not None and outcome_kind:
            outcome_counts[outcome_kind] += 1
            recent_transitions.append(
                {
                    "candidate_goal_id": candidate_goal_id,
                    "summary": (
                        candidate.summary
                        if candidate is not None
                        else (prior_hold or {}).get("summary")
                    ),
                    "goal_family": (
                        candidate.goal_family
                        if candidate is not None
                        else (prior_hold or {}).get("goal_family")
                    ),
                    "hold_reason": (prior_hold or {}).get("hold_reason"),
                    "hold_reason_codes": list((prior_hold or {}).get("hold_reason_codes") or []),
                    "hold_condition": (prior_hold or {}).get("hold_condition"),
                    "hold_condition_kind": (prior_hold or {}).get("hold_condition_kind"),
                    "trigger_kind": trigger_record["trigger"].kind,
                    "trigger_summary": trigger_record["trigger"].summary,
                    "outcome_decision_kind": outcome_kind,
                    "outcome_reason": payload.get("reason"),
                    "outcome_reason_codes": reason_codes,
                    "outcome_executive_policy": executive_policy,
                    "accepted_goal_id": payload.get("goal_id"),
                    "ts": event.ts,
                }
            )

        if event.event_type == BrainEventType.DIRECTOR_NON_ACTION_RECORDED:
            hold_state[candidate_goal_id] = {
                "summary": (
                    candidate.summary if candidate is not None else (prior_hold or {}).get("summary")
                ),
                "goal_family": (
                    candidate.goal_family
                    if candidate is not None
                    else (prior_hold or {}).get("goal_family")
                ),
                "hold_reason": payload.get("reason"),
                "hold_reason_codes": reason_codes,
                "hold_executive_policy": executive_policy,
                "hold_reason_details": dict(payload.get("reason_details") or {}),
                "hold_condition": payload.get("expected_reevaluation_condition"),
                "hold_condition_kind": payload.get("expected_reevaluation_condition_kind"),
                "held_at": event.ts,
            }
            continue

        if event.event_type in {
            BrainEventType.GOAL_CANDIDATE_ACCEPTED,
            BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
            BrainEventType.GOAL_CANDIDATE_EXPIRED,
        }:
            hold_state.pop(candidate_goal_id, None)

    held_candidates = sorted(
        (
            candidate
            for candidate in autonomy_ledger.current_candidates
            if candidate.expected_reevaluation_condition_kind
        ),
        key=_candidate_sort_key,
    )
    hold_kind_counts: Counter[str] = Counter(
        candidate.expected_reevaluation_condition_kind for candidate in held_candidates
    )
    for candidate in held_candidates:
        hold_record = hold_state.get(candidate.candidate_goal_id, {})
        current_holds.append(
            _candidate_payload(
                candidate,
                reference_ts=reference_ts,
                hold_reason=hold_record.get("hold_reason"),
                hold_reason_codes=hold_record.get("hold_reason_codes"),
                hold_executive_policy=hold_record.get("hold_executive_policy"),
                hold_reason_details=hold_record.get("hold_reason_details"),
                held_at=hold_record.get("held_at"),
            )
        )

    def _tail(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(records) <= recent_limit:
            return list(records)
        return list(records[-recent_limit:])

    return {
        "current_hold_count": len(current_holds),
        "current_hold_kinds": dict(sorted(hold_kind_counts.items())),
        "current_holds": current_holds,
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "policy_posture_counts": dict(sorted(policy_posture_counts.items())),
        "approval_requirement_counts": dict(sorted(approval_requirement_counts.items())),
        "why_now_reason_code_counts": dict(sorted(why_now_reason_code_counts.items())),
        "why_not_reason_code_counts": dict(sorted(why_not_reason_code_counts.items())),
        "outcome_counts": outcome_counts,
        "recent_triggers": _tail(recent_triggers),
        "recent_transitions": _tail(recent_transitions),
    }


__all__ = ["build_reevaluation_digest"]

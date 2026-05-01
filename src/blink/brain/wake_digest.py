"""Operator-facing wake-router summaries derived from recent brain events."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from blink.brain.core.projections import BrainCommitmentProjection, BrainCommitmentRecord
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.projections import BrainCommitmentWakeRouteKind


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


def _event_sort_key(event: BrainEventRecord) -> tuple[int, datetime, str]:
    """Return one deterministic sort key for stored events."""
    return (
        int(getattr(event, "id", 0)),
        _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
        event.event_id,
    )


def _commitment_sort_key(record: BrainCommitmentRecord) -> tuple[datetime, datetime, str]:
    """Return one stable display ordering for waiting commitments."""
    return (
        _parse_ts(record.updated_at) or datetime.min.replace(tzinfo=UTC),
        _parse_ts(record.created_at) or datetime.min.replace(tzinfo=UTC),
        record.commitment_id,
    )


def _compact_waiting_commitment(record: BrainCommitmentRecord) -> dict[str, Any]:
    """Return one compact waiting-commitment payload."""
    return {
        "commitment_id": record.commitment_id,
        "title": record.title,
        "status": record.status,
        "goal_family": record.goal_family,
        "blocked_reason_summary": (
            record.blocked_reason.summary if record.blocked_reason is not None else None
        ),
        "wake_kinds": [item.kind for item in record.wake_conditions if item.kind],
        "plan_revision": record.plan_revision,
        "resume_count": record.resume_count,
    }


def _policy_excerpt(value: Any) -> dict[str, Any] | None:
    """Return one compact executive-policy excerpt."""
    if not isinstance(value, dict):
        return None
    action_posture = str(value.get("action_posture", "")).strip()
    approval_requirement = str(value.get("approval_requirement", "")).strip()
    if not action_posture and not approval_requirement:
        return None
    return {
        "action_posture": action_posture or None,
        "approval_requirement": approval_requirement or None,
    }


def build_wake_digest(
    *,
    commitment_projection: BrainCommitmentProjection,
    recent_events: list[BrainEventRecord],
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact operator-facing summary for wake-router flows."""
    waiting_commitments = sorted(
        [
            *commitment_projection.deferred_commitments,
            *commitment_projection.blocked_commitments,
        ],
        key=_commitment_sort_key,
    )
    current_wait_kind_counts: Counter[str] = Counter(
        wake_kind
        for record in waiting_commitments
        for wake_kind in (item.kind for item in record.wake_conditions if item.kind)
    )
    current_waiting_commitments = [
        _compact_waiting_commitment(record) for record in waiting_commitments
    ]

    wake_events = sorted(
        [
            event
            for event in recent_events
            if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
        ],
        key=_event_sort_key,
    )
    resumed_events_by_parent = {
        (event.causal_parent_id or ""): event
        for event in sorted(
            [
                event
                for event in recent_events
                if event.event_type == BrainEventType.GOAL_RESUMED
            ],
            key=_event_sort_key,
        )
        if event.causal_parent_id
    }
    candidate_events_by_parent = {
        (event.causal_parent_id or ""): event
        for event in sorted(
            [
                event
                for event in recent_events
                if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED
            ],
            key=_event_sort_key,
        )
        if event.causal_parent_id
    }

    trigger_counts: Counter[str] = Counter()
    route_counts = {
        BrainCommitmentWakeRouteKind.RESUME_DIRECT.value: 0,
        BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value: 0,
        BrainCommitmentWakeRouteKind.KEEP_WAITING.value: 0,
    }
    reason_counts: Counter[str] = Counter()
    policy_posture_counts: Counter[str] = Counter()
    approval_requirement_counts: Counter[str] = Counter()
    why_now_reason_code_counts: Counter[str] = Counter()
    why_not_reason_code_counts: Counter[str] = Counter()
    recent_triggers: list[dict[str, Any]] = []
    recent_direct_resumes: list[dict[str, Any]] = []
    recent_candidate_proposals: list[dict[str, Any]] = []
    recent_keep_waiting: list[dict[str, Any]] = []

    for event in wake_events:
        payload = event.payload
        commitment_payload = dict(payload.get("commitment") or {})
        trigger_payload = dict(payload.get("trigger") or {})
        routing_payload = dict(payload.get("routing") or {})
        routing_details = dict(routing_payload.get("details") or {})
        commitment_id = str(commitment_payload.get("commitment_id", "")).strip()
        title = str(commitment_payload.get("title", "")).strip()
        wake_kind = str(trigger_payload.get("wake_kind", "")).strip()
        route_kind = str(routing_payload.get("route_kind", "")).strip()
        reason = str(routing_details.get("reason", "")).strip()
        reason_codes = sorted(
            {
                str(item).strip()
                for item in routing_payload.get("reason_codes", [])
                if str(item).strip()
            }
        )
        executive_policy = _policy_excerpt(routing_payload.get("executive_policy"))
        boundary_kind = (
            str(routing_details.get("boundary_kind", "")).strip()
            or str((trigger_payload.get("details") or {}).get("boundary_kind", "")).strip()
            or None
        )

        if wake_kind:
            trigger_counts[wake_kind] += 1
        if route_kind in route_counts:
            route_counts[route_kind] += 1
        if reason:
            reason_counts[reason] += 1
        if executive_policy is not None:
            action_posture = str(executive_policy.get("action_posture", "")).strip()
            approval_requirement = str(executive_policy.get("approval_requirement", "")).strip()
            if action_posture:
                policy_posture_counts[action_posture] += 1
            if approval_requirement:
                approval_requirement_counts[approval_requirement] += 1
        target_reason_codes = (
            why_not_reason_code_counts
            if route_kind == BrainCommitmentWakeRouteKind.KEEP_WAITING.value
            else why_now_reason_code_counts
        )
        for reason_code in reason_codes:
            target_reason_codes[reason_code] += 1

        recent_triggers.append(
            {
                "commitment_id": commitment_id,
                "title": title,
                "wake_kind": wake_kind,
                "route_kind": route_kind,
                "reason_codes": reason_codes,
                "executive_policy": executive_policy,
                "source_event_type": trigger_payload.get("source_event_type"),
                "source_event_id": trigger_payload.get("source_event_id"),
                "ts": event.ts,
            }
        )

        if route_kind == BrainCommitmentWakeRouteKind.RESUME_DIRECT.value:
            resume_event = resumed_events_by_parent.get(event.event_id)
            resume_goal = (
                dict(resume_event.payload.get("goal") or {})
                if resume_event is not None
                else {}
            )
            recent_direct_resumes.append(
                {
                    "commitment_id": commitment_id,
                    "title": title,
                    "wake_kind": wake_kind,
                    "route_kind": route_kind,
                    "reason_codes": reason_codes,
                    "executive_policy": executive_policy,
                    "resumed_goal_id": resume_goal.get("goal_id"),
                    "resumed_goal_title": resume_goal.get("title"),
                    "source_event_type": trigger_payload.get("source_event_type"),
                    "source_event_id": trigger_payload.get("source_event_id"),
                    "ts": event.ts,
                }
            )
        elif route_kind == BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value:
            candidate_event = candidate_events_by_parent.get(event.event_id)
            candidate_goal = (
                dict(candidate_event.payload.get("candidate_goal") or {})
                if candidate_event is not None
                else {}
            )
            recent_candidate_proposals.append(
                {
                    "commitment_id": commitment_id,
                    "title": title,
                    "wake_kind": wake_kind,
                    "route_kind": route_kind,
                    "candidate_goal_id": candidate_goal.get("candidate_goal_id")
                    or routing_details.get("candidate_goal_id")
                    or (trigger_payload.get("details") or {}).get("candidate_goal_id"),
                    "candidate_type": candidate_goal.get("candidate_type")
                    or routing_details.get("candidate_type"),
                    "candidate_summary": candidate_goal.get("summary"),
                    "reason_codes": reason_codes,
                    "executive_policy": executive_policy,
                    "source_event_type": trigger_payload.get("source_event_type"),
                    "source_event_id": trigger_payload.get("source_event_id"),
                    "ts": event.ts,
                }
            )
        elif route_kind == BrainCommitmentWakeRouteKind.KEEP_WAITING.value:
            recent_keep_waiting.append(
                {
                    "commitment_id": commitment_id,
                    "title": title,
                    "wake_kind": wake_kind,
                    "route_kind": route_kind,
                    "reason": reason or None,
                    "reason_codes": reason_codes,
                    "executive_policy": executive_policy,
                    "boundary_kind": boundary_kind,
                    "details": routing_details,
                    "source_event_type": trigger_payload.get("source_event_type"),
                    "source_event_id": trigger_payload.get("source_event_id"),
                    "ts": event.ts,
                }
            )

    def _tail(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(records) <= recent_limit:
            return list(records)
        return list(records[-recent_limit:])

    return {
        "current_wait_count": len(current_waiting_commitments),
        "current_waiting_commitments": current_waiting_commitments,
        "current_wait_kind_counts": dict(sorted(current_wait_kind_counts.items())),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "route_counts": route_counts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "policy_posture_counts": dict(sorted(policy_posture_counts.items())),
        "approval_requirement_counts": dict(sorted(approval_requirement_counts.items())),
        "why_now_reason_code_counts": dict(sorted(why_now_reason_code_counts.items())),
        "why_not_reason_code_counts": dict(sorted(why_not_reason_code_counts.items())),
        "recent_triggers": _tail(recent_triggers),
        "recent_direct_resumes": _tail(recent_direct_resumes),
        "recent_candidate_proposals": _tail(recent_candidate_proposals),
        "recent_keep_waiting": _tail(recent_keep_waiting),
    }


__all__ = ["build_wake_digest"]

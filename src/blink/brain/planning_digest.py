"""Operator-facing planning summaries derived from recent brain events."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from blink.brain.core.projections import (
    BrainAgendaProjection,
    BrainCommitmentProjection,
    BrainCommitmentRecord,
    BrainGoal,
    BrainPlanProposal,
    BrainPlanReviewPolicy,
)
from blink.brain.events import BrainEventRecord, BrainEventType


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


def _goal_sort_key(goal: BrainGoal) -> tuple[datetime, datetime, str]:
    """Return one stable display ordering for goal-backed plan state."""
    return (
        _parse_ts(goal.updated_at) or datetime.min.replace(tzinfo=UTC),
        _parse_ts(goal.created_at) or datetime.min.replace(tzinfo=UTC),
        goal.goal_id,
    )


def _commitment_sort_key(record: BrainCommitmentRecord) -> tuple[datetime, datetime, str]:
    """Return one stable display ordering for commitment-backed plan state."""
    return (
        _parse_ts(record.updated_at) or datetime.min.replace(tzinfo=UTC),
        _parse_ts(record.created_at) or datetime.min.replace(tzinfo=UTC),
        record.commitment_id,
    )


def _plan_linked_state(goal: BrainGoal, commitment: BrainCommitmentRecord | None) -> dict[str, Any]:
    """Return one compact current planning-state payload for a goal."""
    details = dict(goal.details)
    return {
        "goal_id": goal.goal_id,
        "commitment_id": goal.commitment_id,
        "title": goal.title,
        "goal_status": goal.status,
        "commitment_status": (
            commitment.status
            if commitment is not None
            else details.get("commitment_status")
        ),
        "plan_revision": goal.plan_revision,
        "current_plan_proposal_id": details.get("current_plan_proposal_id"),
        "pending_plan_proposal_id": details.get("pending_plan_proposal_id"),
        "plan_review_policy": details.get("plan_review_policy"),
    }


def _compact_pending_proposal(
    *,
    proposal_id: str,
    title: str,
    state: dict[str, Any],
    proposal: BrainPlanProposal | None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one compact pending-proposal payload."""
    return {
        "plan_proposal_id": proposal_id,
        "goal_id": proposal.goal_id if proposal is not None else state.get("goal_id"),
        "commitment_id": (
            proposal.commitment_id if proposal is not None else state.get("commitment_id")
        ),
        "title": title,
        "source": proposal.source if proposal is not None else None,
        "review_policy": (
            proposal.review_policy if proposal is not None else state.get("plan_review_policy")
        ),
        "current_plan_revision": (
            proposal.current_plan_revision if proposal is not None else state.get("plan_revision")
        ),
        "plan_revision": proposal.plan_revision if proposal is not None else state.get("plan_revision"),
        "preserved_prefix_count": (
            proposal.preserved_prefix_count if proposal is not None else 0
        ),
        "missing_inputs": list(proposal.missing_inputs) if proposal is not None else [],
        "assumptions": list(proposal.assumptions) if proposal is not None else [],
        "supersedes_plan_proposal_id": (
            proposal.supersedes_plan_proposal_id if proposal is not None else None
        ),
        "created_at": proposal.created_at if proposal is not None else None,
        "decision_reason": _optional_text((decision or {}).get("reason")),
        "decision_reason_codes": [
            str(item).strip()
            for item in (decision or {}).get("reason_codes", [])
            if str(item).strip()
        ],
        "decision_executive_policy": _policy_excerpt((decision or {}).get("executive_policy")),
        **_procedural_summary(proposal),
    }


def _compact_recent_plan_event(
    *,
    event: BrainEventRecord,
    proposal: BrainPlanProposal,
    title: str | None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one compact recent proposal payload."""
    return {
        "plan_proposal_id": proposal.plan_proposal_id,
        "goal_id": proposal.goal_id,
        "commitment_id": proposal.commitment_id,
        "title": title,
        "source": proposal.source,
        "summary": proposal.summary,
        "review_policy": proposal.review_policy,
        "current_plan_revision": proposal.current_plan_revision,
        "plan_revision": proposal.plan_revision,
        "preserved_prefix_count": proposal.preserved_prefix_count,
        "missing_inputs": list(proposal.missing_inputs),
        "assumptions": list(proposal.assumptions),
        "supersedes_plan_proposal_id": proposal.supersedes_plan_proposal_id,
        "created_at": proposal.created_at,
        "ts": event.ts,
        "decision_reason": _optional_text((decision or {}).get("reason")),
        "decision_reason_codes": [
            str(item).strip()
            for item in (decision or {}).get("reason_codes", [])
            if str(item).strip()
        ],
        "decision_executive_policy": _policy_excerpt((decision or {}).get("executive_policy")),
        **_procedural_summary(proposal),
    }


def _goal_title_map(agenda: BrainAgendaProjection) -> dict[str, str]:
    """Return a stable goal-id to title mapping."""
    return {
        goal.goal_id: goal.title
        for goal in agenda.goals
        if goal.goal_id and goal.title
    }


def _commitment_maps(
    commitment_projection: BrainCommitmentProjection,
) -> tuple[dict[str, BrainCommitmentRecord], dict[str, str]]:
    """Return commitment lookup maps."""
    records = [
        *commitment_projection.active_commitments,
        *commitment_projection.deferred_commitments,
        *commitment_projection.blocked_commitments,
        *commitment_projection.recent_terminal_commitments,
    ]
    record_map = {
        record.commitment_id: record
        for record in sorted(records, key=_commitment_sort_key)
        if record.commitment_id
    }
    title_map = {
        record.commitment_id: record.title
        for record in records
        if record.commitment_id and record.title
    }
    return record_map, title_map


def _planning_touched(goal: BrainGoal) -> bool:
    """Return whether the goal carries planning-specific state."""
    details = dict(goal.details)
    return bool(
        goal.planning_requested
        or goal.status == "planning"
        or goal.plan_revision > 1
        or details.get("current_plan_proposal_id")
        or details.get("pending_plan_proposal_id")
        or details.get("plan_review_policy")
    )


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _procedural_metadata(proposal: BrainPlanProposal | None) -> dict[str, Any]:
    """Return normalized procedural metadata for one proposal."""
    if proposal is None:
        return {}
    procedural = proposal.details.get("procedural", {})
    return dict(procedural) if isinstance(procedural, dict) else {}


def _policy_excerpt(value: Any) -> dict[str, Any] | None:
    """Return one compact executive-policy excerpt."""
    if not isinstance(value, dict):
        return None
    action_posture = _optional_text(value.get("action_posture"))
    approval_requirement = _optional_text(value.get("approval_requirement"))
    if action_posture is None and approval_requirement is None:
        return None
    return {
        "action_posture": action_posture,
        "approval_requirement": approval_requirement,
    }


def _procedural_summary(proposal: BrainPlanProposal | None) -> dict[str, Any]:
    """Return compact procedural planning metadata for digest surfaces."""
    procedural = _procedural_metadata(proposal)
    delta = procedural.get("delta", {})
    if not isinstance(delta, dict):
        delta = {}
    rejected = [
        dict(item)
        for item in procedural.get("rejected_skills", [])
        if isinstance(item, dict)
    ]
    policy = dict(procedural.get("policy") or {}) if isinstance(procedural.get("policy"), dict) else {}
    return {
        "procedural_origin": _optional_text(procedural.get("origin")),
        "selected_skill_id": _optional_text(procedural.get("selected_skill_id")),
        "selected_skill_support_trace_ids": [
            str(item).strip()
            for item in procedural.get("selected_skill_support_trace_ids", [])
            if str(item).strip()
        ],
        "selected_skill_support_plan_proposal_ids": [
            str(item).strip()
            for item in procedural.get("selected_skill_support_plan_proposal_ids", [])
            if str(item).strip()
        ],
        "skill_rejection_reasons": [
            str(item.get("reason", "")).strip()
            for item in rejected
            if str(item.get("reason", "")).strip()
        ],
        "rejected_skill_ids": [
            str(item.get("skill_id", "")).strip()
            for item in rejected
            if str(item.get("skill_id", "")).strip()
        ],
        "delta_operation_count": int(delta.get("operation_count") or 0),
        "procedural_policy_effect": _optional_text(policy.get("effect")),
        "procedural_policy_action_posture": _optional_text(policy.get("action_posture")),
        "procedural_policy_approval_requirement": _optional_text(
            policy.get("approval_requirement")
        ),
        "procedural_policy_reason_codes": [
            str(item).strip()
            for item in policy.get("reason_codes", [])
            if str(item).strip()
        ],
    }


def build_planning_digest(
    *,
    agenda: BrainAgendaProjection,
    commitment_projection: BrainCommitmentProjection,
    recent_events: list[BrainEventRecord],
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build a compact operator-facing summary for planning and revision flows."""
    commitment_map, commitment_titles = _commitment_maps(commitment_projection)
    current_plan_states = [
        _plan_linked_state(goal, commitment_map.get(goal.commitment_id or ""))
        for goal in sorted(
            [goal for goal in agenda.goals if _planning_touched(goal)],
            key=_goal_sort_key,
        )
    ]

    goal_ids_in_state = {state["goal_id"] for state in current_plan_states if state.get("goal_id")}
    for record in sorted(commitment_map.values(), key=_commitment_sort_key):
        details = dict(record.details)
        if not (
            details.get("current_plan_proposal_id")
            or details.get("pending_plan_proposal_id")
            or details.get("plan_review_policy")
        ):
            continue
        if record.current_goal_id and record.current_goal_id in goal_ids_in_state:
            continue
        current_plan_states.append(
            {
                "goal_id": record.current_goal_id,
                "commitment_id": record.commitment_id,
                "title": record.title,
                "goal_status": None,
                "commitment_status": record.status,
                "plan_revision": record.plan_revision,
                "current_plan_proposal_id": details.get("current_plan_proposal_id"),
                "pending_plan_proposal_id": details.get("pending_plan_proposal_id"),
                "plan_review_policy": details.get("plan_review_policy"),
            }
        )

    current_review_policy_counts: Counter[str] = Counter(
        str(state.get("plan_review_policy", "")).strip()
        for state in current_plan_states
        if str(state.get("plan_review_policy", "")).strip()
    )

    planning_events = sorted(
        [
            event
            for event in recent_events
            if event.event_type
            in {
                BrainEventType.PLANNING_PROPOSED,
                BrainEventType.PLANNING_ADOPTED,
                BrainEventType.PLANNING_REJECTED,
                BrainEventType.GOAL_UPDATED,
                BrainEventType.GOAL_REPAIRED,
            }
        ],
        key=_event_sort_key,
    )
    proposal_events = [
        event for event in planning_events if event.event_type == BrainEventType.PLANNING_PROPOSED
    ]
    adopted_events = [
        event for event in planning_events if event.event_type == BrainEventType.PLANNING_ADOPTED
    ]
    rejected_events = [
        event for event in planning_events if event.event_type == BrainEventType.PLANNING_REJECTED
    ]
    downstream_events_by_parent = {
        event.causal_parent_id or "": event
        for event in planning_events
        if event.event_type in {BrainEventType.GOAL_UPDATED, BrainEventType.GOAL_REPAIRED}
        and event.causal_parent_id
    }
    adopted_events_by_parent = {
        event.causal_parent_id or "": event
        for event in adopted_events
        if event.causal_parent_id
    }
    rejected_events_by_parent = {
        event.causal_parent_id or "": event
        for event in rejected_events
        if event.causal_parent_id
    }

    goal_titles = _goal_title_map(agenda)
    recent_proposals: list[dict[str, Any]] = []
    recent_adoptions: list[dict[str, Any]] = []
    recent_rejections: list[dict[str, Any]] = []
    recent_revision_flows: list[dict[str, Any]] = []
    recent_rehearsals: list[dict[str, Any]] = []
    recent_rehearsal_comparisons: list[dict[str, Any]] = []
    embodied_intents_by_id: dict[str, dict[str, Any]] = {}
    embodied_execution_traces_by_id: dict[str, dict[str, Any]] = {}
    embodied_recoveries_by_id: dict[str, dict[str, Any]] = {}
    proposal_source_counts: Counter[str] = Counter()
    procedural_origin_counts: Counter[str] = Counter()
    policy_posture_counts: Counter[str] = Counter()
    approval_requirement_counts: Counter[str] = Counter()
    why_now_reason_code_counts: Counter[str] = Counter()
    why_not_reason_code_counts: Counter[str] = Counter()
    rehearsal_recommendation_counts: Counter[str] = Counter()
    rehearsal_calibration_bucket_counts: Counter[str] = Counter()
    rehearsal_risk_code_counts: Counter[str] = Counter()
    embodied_disposition_counts: Counter[str] = Counter()
    embodied_trace_status_counts: Counter[str] = Counter()
    embodied_policy_posture_counts: Counter[str] = Counter()
    embodied_reason_code_counts: Counter[str] = Counter()
    outcome_counts = {
        "adopted": 0,
        "rejected": 0,
        "pending_user_review": 0,
        "pending_operator_review": 0,
    }
    reason_counts: Counter[str] = Counter()
    skill_rejection_reason_counts: Counter[str] = Counter()
    delta_operation_counts: Counter[str] = Counter()
    rehearsal_operator_review_floor_count = 0
    embodied_operator_review_floor_count = 0
    proposals_by_id: dict[str, BrainPlanProposal] = {}
    proposal_decisions_by_id: dict[str, dict[str, Any]] = {}
    recent_selected_skill_ids: list[str] = []
    recent_skill_linked_proposals: list[dict[str, Any]] = []
    recent_skill_linked_adoptions: list[dict[str, Any]] = []
    recent_skill_linked_rejections: list[dict[str, Any]] = []
    recent_skill_linked_revision_flows: list[dict[str, Any]] = []
    counted_procedural_proposal_ids: set[str] = set()

    def register_procedural(proposal: BrainPlanProposal | None) -> dict[str, Any]:
        procedural = _procedural_summary(proposal)
        proposal_id = proposal.plan_proposal_id if proposal is not None else None
        if proposal_id and proposal_id not in counted_procedural_proposal_ids:
            counted_procedural_proposal_ids.add(proposal_id)
            if procedural["procedural_origin"] is not None:
                procedural_origin_counts[procedural["procedural_origin"]] += 1
            if procedural["selected_skill_id"] is not None:
                recent_selected_skill_ids.append(procedural["selected_skill_id"])
            for reason in procedural["skill_rejection_reasons"]:
                skill_rejection_reason_counts[reason] += 1
            delta_operation_counts[str(procedural["delta_operation_count"])] += 1
        return procedural

    def register_policy_counts(
        *,
        decision: dict[str, Any] | None,
        why_now: bool,
    ) -> None:
        if not isinstance(decision, dict):
            return
        executive_policy = _policy_excerpt(decision.get("executive_policy"))
        if executive_policy is not None:
            action_posture = _optional_text(executive_policy.get("action_posture"))
            approval_requirement = _optional_text(executive_policy.get("approval_requirement"))
            if action_posture is not None:
                policy_posture_counts[action_posture] += 1
            if approval_requirement is not None:
                approval_requirement_counts[approval_requirement] += 1
        for reason_code in [
            str(item).strip()
            for item in decision.get("reason_codes", [])
            if str(item).strip()
        ]:
            if why_now:
                why_now_reason_code_counts[reason_code] += 1
            else:
                why_not_reason_code_counts[reason_code] += 1

    for event in proposal_events:
        payload = event.payload or {}
        proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
        if proposal is None:
            continue
        decision = dict(payload.get("decision") or {})
        proposals_by_id[proposal.plan_proposal_id] = proposal
        proposal_decisions_by_id[proposal.plan_proposal_id] = decision
        proposal_source_counts[proposal.source] += 1
        register_procedural(proposal)
        decision_reason = _optional_text(decision.get("reason"))
        if decision_reason is not None:
            reason_counts[decision_reason] += 1
        if proposal.review_policy != BrainPlanReviewPolicy.AUTO_ADOPT_OK.value:
            register_policy_counts(decision=decision, why_now=False)
        if proposal.details.get("rehearsal_operator_review_floor") or decision_reason == "rehearsal_requires_operator_review":
            rehearsal_operator_review_floor_count += 1
        compact = _compact_recent_plan_event(
            event=event,
            proposal=proposal,
            title=goal_titles.get(proposal.goal_id) or commitment_titles.get(proposal.commitment_id or ""),
            decision=decision,
        )
        recent_proposals.append(compact)
        if compact.get("procedural_origin") in {"skill_reuse", "skill_delta"}:
            recent_skill_linked_proposals.append(compact)

    for event in adopted_events:
        payload = event.payload
        proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
        if proposal is None:
            continue
        register_procedural(proposal)
        decision = dict(payload.get("decision") or {})
        reason = _optional_text(decision.get("reason"))
        if reason is not None:
            reason_counts[reason] += 1
        if proposal.details.get("rehearsal_operator_review_floor") or reason == "rehearsal_requires_operator_review":
            rehearsal_operator_review_floor_count += 1
        register_policy_counts(decision=decision, why_now=True)
        outcome_counts["adopted"] += 1
        downstream_event = downstream_events_by_parent.get(event.event_id)
        downstream_goal = (
            dict((downstream_event.payload or {}).get("goal") or {})
            if downstream_event is not None
            else {}
        )
        compact = {
            **_compact_recent_plan_event(
                event=event,
                proposal=proposal,
                title=goal_titles.get(proposal.goal_id)
                or commitment_titles.get(proposal.commitment_id or ""),
                decision=decision,
            ),
            "decision_summary": decision.get("summary"),
            "downstream_event_type": downstream_event.event_type if downstream_event else None,
            "downstream_goal_id": downstream_goal.get("goal_id"),
            "downstream_goal_title": downstream_goal.get("title"),
        }
        recent_adoptions.append(compact)
        if compact.get("procedural_origin") in {"skill_reuse", "skill_delta"}:
            recent_skill_linked_adoptions.append(compact)

    for event in rejected_events:
        payload = event.payload
        proposal = BrainPlanProposal.from_dict(payload.get("proposal"))
        if proposal is None:
            continue
        register_procedural(proposal)
        decision = dict(payload.get("decision") or {})
        reason = _optional_text(decision.get("reason"))
        if reason is not None:
            reason_counts[reason] += 1
        if proposal.details.get("rehearsal_operator_review_floor") or reason in {
            "rehearsal_requires_operator_review",
            "rehearsal_rejected_embodied_step",
        }:
            rehearsal_operator_review_floor_count += 1
        register_policy_counts(decision=decision, why_now=False)
        outcome_counts["rejected"] += 1
        compact = {
            **_compact_recent_plan_event(
                event=event,
                proposal=proposal,
                title=goal_titles.get(proposal.goal_id)
                or commitment_titles.get(proposal.commitment_id or ""),
                decision=decision,
            ),
            "decision_summary": decision.get("summary"),
        }
        recent_rejections.append(compact)
        if compact.get("procedural_origin") in {"skill_reuse", "skill_delta"} or compact.get(
            "skill_rejection_reasons"
        ):
            if reason is not None:
                skill_rejection_reason_counts[reason] += 1
            recent_skill_linked_rejections.append(compact)

    pending_states = [
        state
        for state in current_plan_states
        if _optional_text(state.get("pending_plan_proposal_id"))
    ]
    current_pending_proposals = []
    seen_pending_ids: set[str] = set()
    for state in pending_states:
        proposal_id = _optional_text(state.get("pending_plan_proposal_id"))
        if proposal_id is None or proposal_id in seen_pending_ids:
            continue
        seen_pending_ids.add(proposal_id)
        proposal = proposals_by_id.get(proposal_id)
        review_policy = (
            proposal.review_policy
            if proposal is not None
            else str(state.get("plan_review_policy", "")).strip()
        )
        if review_policy == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
            outcome_counts["pending_user_review"] += 1
        elif review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value:
            outcome_counts["pending_operator_review"] += 1
        current_pending_proposals.append(
            _compact_pending_proposal(
                proposal_id=proposal_id,
                title=str(state.get("title") or ""),
                state=state,
                proposal=proposal,
                decision=proposal_decisions_by_id.get(proposal_id),
            )
        )

    pending_outcome_by_proposal = {
        entry["plan_proposal_id"]: (
            "pending_user_review"
            if entry.get("review_policy") == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value
            else "pending_operator_review"
            if entry.get("review_policy") == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
            else "proposed"
        )
        for entry in current_pending_proposals
        if entry.get("plan_proposal_id")
    }

    for event in proposal_events:
        proposal = BrainPlanProposal.from_dict((event.payload or {}).get("proposal"))
        if proposal is None:
            continue
        request_kind = str(proposal.details.get("request_kind", "")).strip()
        if not (
            request_kind == "revise_tail"
            or proposal.preserved_prefix_count > 0
            or proposal.plan_revision > proposal.current_plan_revision
        ):
            continue
        adopted_event = adopted_events_by_parent.get(event.event_id)
        rejected_event = rejected_events_by_parent.get(event.event_id)
        outcome_kind = pending_outcome_by_proposal.get(proposal.plan_proposal_id, "proposed")
        proposal_decision = proposal_decisions_by_id.get(proposal.plan_proposal_id, {})
        decision_reason = _optional_text(proposal_decision.get("reason"))
        decision_reason_codes = [
            str(item).strip()
            for item in proposal_decision.get("reason_codes", [])
            if str(item).strip()
        ]
        decision_executive_policy = _policy_excerpt(proposal_decision.get("executive_policy"))
        downstream_event = None
        if adopted_event is not None:
            outcome_kind = "adopted"
            outcome_decision = dict(adopted_event.payload.get("decision") or {})
            decision_reason = _optional_text(outcome_decision.get("reason"))
            decision_reason_codes = [
                str(item).strip()
                for item in outcome_decision.get("reason_codes", [])
                if str(item).strip()
            ]
            decision_executive_policy = _policy_excerpt(
                outcome_decision.get("executive_policy")
            )
            downstream_event = downstream_events_by_parent.get(adopted_event.event_id)
        elif rejected_event is not None:
            outcome_kind = "rejected"
            outcome_decision = dict(rejected_event.payload.get("decision") or {})
            decision_reason = _optional_text(outcome_decision.get("reason"))
            decision_reason_codes = [
                str(item).strip()
                for item in outcome_decision.get("reason_codes", [])
                if str(item).strip()
            ]
            decision_executive_policy = _policy_excerpt(
                outcome_decision.get("executive_policy")
            )
        downstream_goal = (
            dict((downstream_event.payload or {}).get("goal") or {})
            if downstream_event is not None
            else {}
        )
        compact = {
            "plan_proposal_id": proposal.plan_proposal_id,
            "goal_id": proposal.goal_id,
            "commitment_id": proposal.commitment_id,
            "title": goal_titles.get(proposal.goal_id)
            or commitment_titles.get(proposal.commitment_id or ""),
            "current_plan_revision": proposal.current_plan_revision,
            "plan_revision": proposal.plan_revision,
            "preserved_prefix_count": proposal.preserved_prefix_count,
            "supersedes_plan_proposal_id": proposal.supersedes_plan_proposal_id,
            "review_policy": proposal.review_policy,
            "outcome_kind": outcome_kind,
            "decision_reason": decision_reason,
            "decision_reason_codes": decision_reason_codes,
            "decision_executive_policy": decision_executive_policy,
            "downstream_event_type": downstream_event.event_type if downstream_event else None,
            "downstream_goal_id": downstream_goal.get("goal_id"),
            "downstream_goal_title": downstream_goal.get("title"),
            "created_at": proposal.created_at,
            "ts": event.ts,
            **_procedural_summary(proposal),
        }
        recent_revision_flows.append(compact)
        if compact.get("procedural_origin") in {"skill_reuse", "skill_delta"} or compact.get(
            "skill_rejection_reasons"
        ):
            if decision_reason:
                skill_rejection_reason_counts[str(decision_reason)] += 1
            recent_skill_linked_revision_flows.append(compact)

    for event in sorted(recent_events, key=_event_sort_key):
        payload = event.payload or {}
        if event.event_type in {
            BrainEventType.ACTION_REHEARSAL_COMPLETED,
            BrainEventType.ACTION_REHEARSAL_SKIPPED,
        }:
            result = dict(payload.get("rehearsal_result") or {})
            recommendation = _optional_text(result.get("decision_recommendation"))
            if recommendation is not None:
                rehearsal_recommendation_counts[recommendation] += 1
            for code in result.get("risk_codes", []) or []:
                text = _optional_text(code)
                if text is not None:
                    rehearsal_risk_code_counts[text] += 1
            recent_rehearsals.append(
                {
                    "rehearsal_id": result.get("rehearsal_id"),
                    "goal_id": result.get("goal_id"),
                    "plan_proposal_id": result.get("plan_proposal_id"),
                    "step_index": result.get("step_index"),
                    "candidate_action_id": result.get("candidate_action_id"),
                    "decision_recommendation": recommendation,
                    "confidence_band": result.get("confidence_band"),
                    "predicted_success_probability": result.get("predicted_success_probability"),
                    "risk_codes": list(result.get("risk_codes") or []),
                    "summary": result.get("summary"),
                    "skipped": bool(result.get("skipped")),
                    "completed_at": result.get("completed_at"),
                }
            )
        elif event.event_type == BrainEventType.ACTION_OUTCOME_COMPARED:
            comparison = dict(payload.get("comparison") or {})
            calibration_bucket = _optional_text(comparison.get("calibration_bucket"))
            if calibration_bucket is not None:
                rehearsal_calibration_bucket_counts[calibration_bucket] += 1
            for code in comparison.get("risk_codes", []) or []:
                text = _optional_text(code)
                if text is not None:
                    rehearsal_risk_code_counts[text] += 1
            recent_rehearsal_comparisons.append(
                {
                    "comparison_id": comparison.get("comparison_id"),
                    "rehearsal_id": comparison.get("rehearsal_id"),
                    "goal_id": comparison.get("goal_id"),
                    "plan_proposal_id": comparison.get("plan_proposal_id"),
                    "candidate_action_id": comparison.get("candidate_action_id"),
                    "observed_outcome_kind": comparison.get("observed_outcome_kind"),
                    "calibration_bucket": calibration_bucket,
                    "comparison_summary": comparison.get("comparison_summary"),
                    "compared_at": comparison.get("compared_at"),
                }
            )
        elif event.event_type == BrainEventType.EMBODIED_INTENT_SELECTED:
            intent = dict(payload.get("intent") or {})
            intent_id = _optional_text(intent.get("intent_id"))
            if intent_id is None:
                continue
            embodied_intents_by_id[intent_id] = {
                "intent_id": intent_id,
                "goal_id": intent.get("goal_id"),
                "plan_proposal_id": intent.get("plan_proposal_id"),
                "step_index": intent.get("step_index"),
                "intent_kind": intent.get("intent_kind"),
                "selected_action_id": intent.get("selected_action_id"),
                "executor_kind": intent.get("executor_kind"),
                "policy_posture": intent.get("policy_posture"),
                "reason_codes": list(intent.get("reason_codes") or []),
                "status": intent.get("status"),
                "summary": intent.get("summary"),
                "ts": event.ts,
            }
        elif event.event_type in {
            BrainEventType.EMBODIED_DISPATCH_PREPARED,
            BrainEventType.EMBODIED_DISPATCH_COMPLETED,
            BrainEventType.EMBODIED_DISPATCH_DEFERRED,
            BrainEventType.EMBODIED_RECOVERY_RECORDED,
        }:
            intent = dict(payload.get("intent") or {})
            envelope = dict(payload.get("envelope") or {})
            trace = dict(payload.get("execution_trace") or {})
            recovery = dict(payload.get("recovery") or {})
            trace_id = _optional_text(trace.get("trace_id"))
            if trace_id is not None:
                embodied_execution_traces_by_id[trace_id] = {
                    "trace_id": trace_id,
                    "intent_id": trace.get("intent_id"),
                    "envelope_id": trace.get("envelope_id"),
                    "goal_id": trace.get("goal_id"),
                    "plan_proposal_id": envelope.get("plan_proposal_id")
                    or intent.get("plan_proposal_id"),
                    "step_index": trace.get("step_index"),
                    "intent_kind": intent.get("intent_kind"),
                    "selected_action_id": intent.get("selected_action_id")
                    or envelope.get("action_id"),
                    "disposition": trace.get("disposition"),
                    "status": trace.get("status"),
                    "executor_backend": envelope.get("executor_backend"),
                    "policy_posture": intent.get("policy_posture"),
                    "reason_codes": [
                        *[str(code).strip() for code in intent.get("reason_codes", []) or []],
                        *[str(code).strip() for code in envelope.get("reason_codes", []) or []],
                    ],
                    "mismatch_codes": [
                        str(code).strip() for code in trace.get("mismatch_codes", []) or []
                    ],
                    "repair_codes": [
                        str(code).strip() for code in trace.get("repair_codes", []) or []
                    ],
                    "recovery_action_id": trace.get("recovery_action_id"),
                    "outcome_summary": trace.get("outcome_summary"),
                    "ts": event.ts,
                }
            recovery_id = _optional_text(recovery.get("recovery_id"))
            if recovery_id is not None:
                embodied_recoveries_by_id[recovery_id] = {
                    "recovery_id": recovery_id,
                    "trace_id": recovery.get("trace_id"),
                    "intent_id": recovery.get("intent_id"),
                    "action_id": recovery.get("action_id"),
                    "reason_codes": [
                        str(code).strip() for code in recovery.get("reason_codes", []) or []
                    ],
                    "status": recovery.get("status"),
                    "summary": recovery.get("summary"),
                    "ts": event.ts,
                }

    recent_embodied_intents = sorted(
        embodied_intents_by_id.values(),
        key=lambda row: (
            _parse_ts(row.get("ts")) or datetime.min.replace(tzinfo=UTC),
            str(row.get("intent_id") or ""),
        ),
    )
    recent_embodied_execution_traces = sorted(
        embodied_execution_traces_by_id.values(),
        key=lambda row: (
            _parse_ts(row.get("ts")) or datetime.min.replace(tzinfo=UTC),
            str(row.get("trace_id") or ""),
        ),
    )
    recent_embodied_recoveries = sorted(
        embodied_recoveries_by_id.values(),
        key=lambda row: (
            _parse_ts(row.get("ts")) or datetime.min.replace(tzinfo=UTC),
            str(row.get("recovery_id") or ""),
        ),
    )
    for row in recent_embodied_execution_traces:
        disposition = _optional_text(row.get("disposition"))
        status = _optional_text(row.get("status"))
        policy_posture = _optional_text(row.get("policy_posture"))
        if disposition is not None:
            embodied_disposition_counts[disposition] += 1
        if status is not None:
            embodied_trace_status_counts[status] += 1
        if policy_posture is not None:
            embodied_policy_posture_counts[policy_posture] += 1
        if disposition in {"defer", "abort"} or row.get("recovery_action_id") or row.get(
            "repair_codes"
        ):
            embodied_operator_review_floor_count += 1
        for code in row.get("reason_codes", []):
            text = _optional_text(code)
            if text is not None:
                embodied_reason_code_counts[text] += 1
        for code in row.get("mismatch_codes", []):
            text = _optional_text(code)
            if text is not None:
                embodied_reason_code_counts[text] += 1
        for code in row.get("repair_codes", []):
            text = _optional_text(code)
            if text is not None:
                embodied_reason_code_counts[text] += 1
    for row in recent_embodied_recoveries:
        for code in row.get("reason_codes", []):
            text = _optional_text(code)
            if text is not None:
                embodied_reason_code_counts[text] += 1

    current_embodied_step = (
        dict(recent_embodied_execution_traces[-1])
        if recent_embodied_execution_traces
        else dict(recent_embodied_intents[-1])
        if recent_embodied_intents
        else {}
    )

    return {
        "current_plan_state_count": len(current_plan_states),
        "current_plan_states": current_plan_states,
        "current_pending_proposal_count": len(current_pending_proposals),
        "current_pending_proposals": current_pending_proposals,
        "current_review_policy_counts": dict(current_review_policy_counts),
        "proposal_source_counts": dict(proposal_source_counts),
        "procedural_origin_counts": dict(procedural_origin_counts),
        "policy_posture_counts": dict(sorted(policy_posture_counts.items())),
        "approval_requirement_counts": dict(sorted(approval_requirement_counts.items())),
        "why_now_reason_code_counts": dict(sorted(why_now_reason_code_counts.items())),
        "why_not_reason_code_counts": dict(sorted(why_not_reason_code_counts.items())),
        "outcome_counts": outcome_counts,
        "reason_counts": dict(reason_counts),
        "rehearsal_count": len(recent_rehearsals),
        "rehearsal_comparison_count": len(recent_rehearsal_comparisons),
        "rehearsal_recommendation_counts": dict(sorted(rehearsal_recommendation_counts.items())),
        "rehearsal_calibration_bucket_counts": dict(
            sorted(rehearsal_calibration_bucket_counts.items())
        ),
        "rehearsal_risk_code_counts": dict(sorted(rehearsal_risk_code_counts.items())),
        "rehearsal_operator_review_floor_count": rehearsal_operator_review_floor_count,
        "embodied_trace_count": len(recent_embodied_execution_traces),
        "embodied_recovery_count": len(recent_embodied_recoveries),
        "embodied_disposition_counts": dict(sorted(embodied_disposition_counts.items())),
        "embodied_trace_status_counts": dict(sorted(embodied_trace_status_counts.items())),
        "embodied_policy_posture_counts": dict(
            sorted(embodied_policy_posture_counts.items())
        ),
        "embodied_reason_code_counts": dict(sorted(embodied_reason_code_counts.items())),
        "embodied_operator_review_floor_count": embodied_operator_review_floor_count,
        "current_embodied_step": current_embodied_step,
        "skill_rejection_reason_counts": dict(skill_rejection_reason_counts),
        "delta_operation_counts": dict(delta_operation_counts),
        "recent_selected_skill_ids": sorted({item for item in recent_selected_skill_ids if item}),
        "recent_proposals": recent_proposals[-recent_limit:],
        "recent_adoptions": recent_adoptions[-recent_limit:],
        "recent_rejections": recent_rejections[-recent_limit:],
        "recent_revision_flows": recent_revision_flows[-recent_limit:],
        "recent_skill_linked_proposals": recent_skill_linked_proposals[-recent_limit:],
        "recent_skill_linked_adoptions": recent_skill_linked_adoptions[-recent_limit:],
        "recent_skill_linked_rejections": recent_skill_linked_rejections[-recent_limit:],
        "recent_skill_linked_revision_flows": recent_skill_linked_revision_flows[-recent_limit:],
        "recent_rehearsals": recent_rehearsals[-recent_limit:],
        "recent_rehearsal_comparisons": recent_rehearsal_comparisons[-recent_limit:],
        "recent_embodied_intents": recent_embodied_intents[-recent_limit:],
        "recent_embodied_execution_traces": recent_embodied_execution_traces[-recent_limit:],
        "recent_embodied_recoveries": recent_embodied_recoveries[-recent_limit:],
    }


__all__ = ["build_planning_digest"]

"""Derived procedural trace projection for Blink memory v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.projections import (
    BrainBlockedReason,
    BrainGoal,
    BrainGoalStatus,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sorted_unique_texts(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _trace_sort_key(record: "BrainProceduralExecutionTraceRecord") -> tuple[str, str, int, str]:
    return (
        record.started_at or "",
        record.goal_id,
        int(record.plan_revision),
        record.trace_id,
    )


def _outcome_sort_key(record: "BrainProceduralOutcomeRecord") -> tuple[str, str, str]:
    return (record.created_at, record.outcome_kind, record.outcome_id)


def _step_sort_key(record: "BrainProceduralStepTraceRecord") -> tuple[int, str, str]:
    return (int(record.step_index), record.started_at or "", record.step_trace_id)


class BrainProceduralTraceStatus(str, Enum):
    """Canonical statuses for one harvested procedural execution trace."""

    OPEN = "open"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class BrainProceduralOutcomeKind(str, Enum):
    """Canonical normalized outcome kinds used for procedural harvesting."""

    PLANNING_ADOPTED = "planning_adopted"
    PLANNING_REJECTED = "planning_rejected"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED_BY_REVISION = "superseded_by_revision"


@dataclass(frozen=True)
class BrainProceduralStepTraceRecord:
    """One normalized capability invocation inside a procedural execution trace."""

    step_trace_id: str
    trace_id: str
    step_index: int
    capability_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "requested"
    attempts: int = 0
    summary: str | None = None
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    request_event_id: str | None = None
    result_event_id: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the step trace."""
        return {
            "step_trace_id": self.step_trace_id,
            "trace_id": self.trace_id,
            "step_index": self.step_index,
            "capability_id": self.capability_id,
            "arguments": dict(self.arguments),
            "status": self.status,
            "attempts": self.attempts,
            "summary": self.summary,
            "error_code": self.error_code,
            "warnings": list(self.warnings),
            "output": dict(self.output),
            "request_event_id": self.request_event_id,
            "result_event_id": self.result_event_id,
            "source_event_ids": _sorted_unique_texts(self.source_event_ids),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralStepTraceRecord | None":
        """Hydrate one step trace from stored JSON."""
        if not isinstance(data, dict):
            return None
        step_trace_id = str(data.get("step_trace_id", "")).strip()
        trace_id = str(data.get("trace_id", "")).strip()
        capability_id = str(data.get("capability_id", "")).strip()
        if not step_trace_id or not trace_id or not capability_id:
            return None
        return cls(
            step_trace_id=step_trace_id,
            trace_id=trace_id,
            step_index=int(data.get("step_index", 0)),
            capability_id=capability_id,
            arguments=dict(data.get("arguments", {})),
            status=str(data.get("status", "requested")).strip() or "requested",
            attempts=int(data.get("attempts", 0)),
            summary=_optional_text(data.get("summary")),
            error_code=_optional_text(data.get("error_code")),
            warnings=[str(item) for item in data.get("warnings", []) if str(item).strip()],
            output=dict(data.get("output", {})),
            request_event_id=_optional_text(data.get("request_event_id")),
            result_event_id=_optional_text(data.get("result_event_id")),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            started_at=str(data.get("started_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralExecutionTraceRecord:
    """One normalized bounded execution trace ready for later skill extraction."""

    trace_id: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str
    goal_title: str
    goal_intent: str
    goal_family: str
    proposal_source: str
    review_policy: str
    current_plan_revision: int
    plan_revision: int
    preserved_prefix_count: int = 0
    supersedes_plan_proposal_id: str | None = None
    status: str = BrainProceduralTraceStatus.OPEN.value
    planned_steps: list[BrainGoalStep] = field(default_factory=list)
    step_executions: list[BrainProceduralStepTraceRecord] = field(default_factory=list)
    outcome_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the execution trace."""
        return {
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "goal_title": self.goal_title,
            "goal_intent": self.goal_intent,
            "goal_family": self.goal_family,
            "proposal_source": self.proposal_source,
            "review_policy": self.review_policy,
            "current_plan_revision": self.current_plan_revision,
            "plan_revision": self.plan_revision,
            "preserved_prefix_count": self.preserved_prefix_count,
            "supersedes_plan_proposal_id": self.supersedes_plan_proposal_id,
            "status": self.status,
            "planned_steps": [step.as_dict() for step in self.planned_steps],
            "step_executions": [record.as_dict() for record in sorted(self.step_executions, key=_step_sort_key)],
            "outcome_ids": _sorted_unique_texts(self.outcome_ids),
            "source_event_ids": _sorted_unique_texts(self.source_event_ids),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralExecutionTraceRecord | None":
        """Hydrate one execution trace from stored JSON."""
        if not isinstance(data, dict):
            return None
        trace_id = str(data.get("trace_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        plan_proposal_id = str(data.get("plan_proposal_id", "")).strip()
        if not trace_id or not goal_id or not plan_proposal_id:
            return None
        planned_steps = [
            step
            for item in data.get("planned_steps", [])
            if (step := BrainGoalStep.from_dict(item)).capability_id
        ]
        step_executions = [
            record
            for item in data.get("step_executions", [])
            if (record := BrainProceduralStepTraceRecord.from_dict(item)) is not None
        ]
        return cls(
            trace_id=trace_id,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=plan_proposal_id,
            goal_title=str(data.get("goal_title", "")).strip(),
            goal_intent=str(data.get("goal_intent", "")).strip(),
            goal_family=str(data.get("goal_family", "")).strip(),
            proposal_source=str(data.get("proposal_source", "")).strip(),
            review_policy=str(data.get("review_policy", "")).strip(),
            current_plan_revision=int(data.get("current_plan_revision", 1)),
            plan_revision=int(data.get("plan_revision", 1)),
            preserved_prefix_count=int(data.get("preserved_prefix_count", 0)),
            supersedes_plan_proposal_id=_optional_text(data.get("supersedes_plan_proposal_id")),
            status=str(data.get("status", BrainProceduralTraceStatus.OPEN.value)).strip()
            or BrainProceduralTraceStatus.OPEN.value,
            planned_steps=planned_steps,
            step_executions=step_executions,
            outcome_ids=_sorted_unique_texts(data.get("outcome_ids", [])),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            started_at=str(data.get("started_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            ended_at=_optional_text(data.get("ended_at")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralOutcomeRecord:
    """One normalized outcome record linked back to its supporting execution trace."""

    outcome_id: str
    outcome_kind: str
    trace_id: str | None
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str | None
    plan_revision: int | None = None
    summary: str = ""
    reason_code: str | None = None
    source_event_id: str | None = None
    causal_parent_event_id: str | None = None
    source_trace_ids: list[str] = field(default_factory=list)
    source_step_trace_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the outcome record."""
        return {
            "outcome_id": self.outcome_id,
            "outcome_kind": self.outcome_kind,
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "plan_revision": self.plan_revision,
            "summary": self.summary,
            "reason_code": self.reason_code,
            "source_event_id": self.source_event_id,
            "causal_parent_event_id": self.causal_parent_event_id,
            "source_trace_ids": _sorted_unique_texts(self.source_trace_ids),
            "source_step_trace_ids": _sorted_unique_texts(self.source_step_trace_ids),
            "source_event_ids": _sorted_unique_texts(self.source_event_ids),
            "created_at": self.created_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralOutcomeRecord | None":
        """Hydrate one outcome record from stored JSON."""
        if not isinstance(data, dict):
            return None
        outcome_id = str(data.get("outcome_id", "")).strip()
        outcome_kind = str(data.get("outcome_kind", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        if not outcome_id or not outcome_kind or not goal_id:
            return None
        raw_plan_revision = data.get("plan_revision")
        plan_revision = int(raw_plan_revision) if raw_plan_revision is not None else None
        return cls(
            outcome_id=outcome_id,
            outcome_kind=outcome_kind,
            trace_id=_optional_text(data.get("trace_id")),
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            plan_revision=plan_revision,
            summary=str(data.get("summary", "")).strip(),
            reason_code=_optional_text(data.get("reason_code")),
            source_event_id=_optional_text(data.get("source_event_id")),
            causal_parent_event_id=_optional_text(data.get("causal_parent_event_id")),
            source_trace_ids=_sorted_unique_texts(data.get("source_trace_ids", [])),
            source_step_trace_ids=_sorted_unique_texts(data.get("source_step_trace_ids", [])),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            created_at=str(data.get("created_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralTraceProjection:
    """Procedural trace substrate derived from Blink's append-only event spine."""

    scope_type: str
    scope_id: str
    trace_counts: dict[str, int] = field(default_factory=dict)
    outcome_counts: dict[str, int] = field(default_factory=dict)
    traces: list[BrainProceduralExecutionTraceRecord] = field(default_factory=list)
    outcomes: list[BrainProceduralOutcomeRecord] = field(default_factory=list)
    open_trace_ids: list[str] = field(default_factory=list)
    paused_trace_ids: list[str] = field(default_factory=list)
    completed_trace_ids: list[str] = field(default_factory=list)
    failed_trace_ids: list[str] = field(default_factory=list)
    cancelled_trace_ids: list[str] = field(default_factory=list)
    superseded_trace_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "trace_counts": dict(self.trace_counts),
            "outcome_counts": dict(self.outcome_counts),
            "traces": [record.as_dict() for record in sorted(self.traces, key=_trace_sort_key)],
            "outcomes": [record.as_dict() for record in sorted(self.outcomes, key=_outcome_sort_key)],
            "open_trace_ids": _sorted_unique_texts(self.open_trace_ids),
            "paused_trace_ids": _sorted_unique_texts(self.paused_trace_ids),
            "completed_trace_ids": _sorted_unique_texts(self.completed_trace_ids),
            "failed_trace_ids": _sorted_unique_texts(self.failed_trace_ids),
            "cancelled_trace_ids": _sorted_unique_texts(self.cancelled_trace_ids),
            "superseded_trace_ids": _sorted_unique_texts(self.superseded_trace_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralTraceProjection | None":
        """Hydrate one procedural trace projection from stored JSON."""
        if not isinstance(data, dict):
            return None
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        if not scope_type or not scope_id:
            return None
        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            trace_counts={
                str(key): int(value) for key, value in dict(data.get("trace_counts", {})).items()
            },
            outcome_counts={
                str(key): int(value) for key, value in dict(data.get("outcome_counts", {})).items()
            },
            traces=[
                record
                for item in data.get("traces", [])
                if (record := BrainProceduralExecutionTraceRecord.from_dict(item)) is not None
            ],
            outcomes=[
                record
                for item in data.get("outcomes", [])
                if (record := BrainProceduralOutcomeRecord.from_dict(item)) is not None
            ],
            open_trace_ids=_sorted_unique_texts(data.get("open_trace_ids", [])),
            paused_trace_ids=_sorted_unique_texts(data.get("paused_trace_ids", [])),
            completed_trace_ids=_sorted_unique_texts(data.get("completed_trace_ids", [])),
            failed_trace_ids=_sorted_unique_texts(data.get("failed_trace_ids", [])),
            cancelled_trace_ids=_sorted_unique_texts(data.get("cancelled_trace_ids", [])),
            superseded_trace_ids=_sorted_unique_texts(data.get("superseded_trace_ids", [])),
        )


@dataclass
class _MutableStepTrace:
    step_trace_id: str
    trace_id: str
    step_index: int
    capability_id: str
    arguments: dict[str, Any]
    status: str = "requested"
    attempts: int = 0
    summary: str | None = None
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    request_event_id: str | None = None
    result_event_id: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> BrainProceduralStepTraceRecord:
        return BrainProceduralStepTraceRecord(
            step_trace_id=self.step_trace_id,
            trace_id=self.trace_id,
            step_index=self.step_index,
            capability_id=self.capability_id,
            arguments=dict(self.arguments),
            status=self.status,
            attempts=self.attempts,
            summary=self.summary,
            error_code=self.error_code,
            warnings=list(self.warnings),
            output=dict(self.output),
            request_event_id=self.request_event_id,
            result_event_id=self.result_event_id,
            source_event_ids=_sorted_unique_texts(self.source_event_ids),
            started_at=self.started_at,
            updated_at=self.updated_at,
            details=dict(self.details),
        )


@dataclass
class _MutableExecutionTrace:
    trace_id: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str
    goal_title: str
    goal_intent: str
    goal_family: str
    proposal_source: str
    review_policy: str
    current_plan_revision: int
    plan_revision: int
    preserved_prefix_count: int = 0
    supersedes_plan_proposal_id: str | None = None
    status: str = BrainProceduralTraceStatus.OPEN.value
    planned_steps: list[BrainGoalStep] = field(default_factory=list)
    step_by_request_event_id: dict[str, _MutableStepTrace] = field(default_factory=dict)
    outcome_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> BrainProceduralExecutionTraceRecord:
        return BrainProceduralExecutionTraceRecord(
            trace_id=self.trace_id,
            goal_id=self.goal_id,
            commitment_id=self.commitment_id,
            plan_proposal_id=self.plan_proposal_id,
            goal_title=self.goal_title,
            goal_intent=self.goal_intent,
            goal_family=self.goal_family,
            proposal_source=self.proposal_source,
            review_policy=self.review_policy,
            current_plan_revision=self.current_plan_revision,
            plan_revision=self.plan_revision,
            preserved_prefix_count=self.preserved_prefix_count,
            supersedes_plan_proposal_id=self.supersedes_plan_proposal_id,
            status=self.status,
            planned_steps=[BrainGoalStep.from_dict(step.as_dict()) for step in self.planned_steps],
            step_executions=[
                step.to_record()
                for step in sorted(
                    self.step_by_request_event_id.values(),
                    key=lambda item: (item.step_index, item.started_at),
                )
            ],
            outcome_ids=_sorted_unique_texts(self.outcome_ids),
            source_event_ids=_sorted_unique_texts(self.source_event_ids),
            started_at=self.started_at,
            updated_at=self.updated_at,
            ended_at=self.ended_at,
            details=dict(self.details),
        )


def _goal_from_payload(event: BrainEventRecord) -> BrainGoal | None:
    return BrainGoal.from_dict(_event_payload(event).get("goal"))


def _proposal_from_event(event: BrainEventRecord) -> BrainPlanProposal | None:
    return BrainPlanProposal.from_dict(_event_payload(event).get("proposal"))


def _decision_from_event(event: BrainEventRecord) -> BrainPlanProposalDecision | None:
    return BrainPlanProposalDecision.from_dict(_event_payload(event).get("decision"))


def _event_payload(event: BrainEventRecord) -> dict[str, Any]:
    payload = event.payload
    return payload if isinstance(payload, dict) else {}


def _goal_commitment_details(event: BrainEventRecord) -> dict[str, Any]:
    payload = _event_payload(event)
    commitment = payload.get("commitment")
    return dict(commitment) if isinstance(commitment, dict) else {}


def _blocked_reason(goal: BrainGoal) -> BrainBlockedReason | None:
    return goal.blocked_reason if isinstance(goal.blocked_reason, BrainBlockedReason) else None


def _append_unique(values: list[str], *candidates: str | None):
    existing = set(values)
    for candidate in candidates:
        normalized = _optional_text(candidate)
        if normalized is None or normalized in existing:
            continue
        values.append(normalized)
        existing.add(normalized)


def _trace_status_ids(traces: Iterable[BrainProceduralExecutionTraceRecord], status: str) -> list[str]:
    return sorted(record.trace_id for record in traces if record.status == status)


def build_procedural_trace_projection(
    *,
    scope_type: str,
    scope_id: str,
    events: Iterable[BrainEventRecord],
) -> BrainProceduralTraceProjection:
    """Build one deterministic procedural trace projection from append-only events."""
    sorted_events = sorted(
        (event for event in events),
        key=lambda event: (
            int(getattr(event, "id", 0)),
            _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
            event.event_id,
        ),
    )
    goal_state_by_id: dict[str, BrainGoal] = {}
    trace_by_id: dict[str, _MutableExecutionTrace] = {}
    trace_id_by_goal: dict[str, str] = {}
    trace_id_by_proposal: dict[str, str] = {}
    outcome_by_id: dict[str, BrainProceduralOutcomeRecord] = {}

    def add_outcome(
        *,
        outcome_kind: str,
        event: BrainEventRecord,
        goal_id: str,
        commitment_id: str | None,
        plan_proposal_id: str | None,
        plan_revision: int | None,
        summary: str,
        reason_code: str | None = None,
        trace_id: str | None = None,
        step_trace_ids: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> BrainProceduralOutcomeRecord:
        outcome_id = _stable_id(
            "procedural_outcome",
            outcome_kind,
            event.event_id,
            goal_id,
            plan_proposal_id or "",
            trace_id or "",
        )
        source_trace_ids = [trace_id] if trace_id else []
        record = BrainProceduralOutcomeRecord(
            outcome_id=outcome_id,
            outcome_kind=outcome_kind,
            trace_id=trace_id,
            goal_id=goal_id,
            commitment_id=commitment_id,
            plan_proposal_id=plan_proposal_id,
            plan_revision=plan_revision,
            summary=summary,
            reason_code=reason_code,
            source_event_id=event.event_id,
            causal_parent_event_id=event.causal_parent_id,
            source_trace_ids=source_trace_ids,
            source_step_trace_ids=list(step_trace_ids or []),
            source_event_ids=_sorted_unique_texts([event.event_id, event.causal_parent_id]),
            created_at=event.ts,
            details=dict(details or {}),
        )
        outcome_by_id[record.outcome_id] = record
        if trace_id is not None and trace_id in trace_by_id:
            _append_unique(trace_by_id[trace_id].outcome_ids, record.outcome_id)
            trace_by_id[trace_id].updated_at = event.ts
        return record

    for event in sorted_events:
        goal = _goal_from_payload(event)
        if goal is not None and goal.goal_id:
            goal_state_by_id[goal.goal_id] = goal
            current_trace_id = trace_id_by_goal.get(goal.goal_id)
            if current_trace_id is not None and current_trace_id in trace_by_id:
                trace = trace_by_id[current_trace_id]
                trace.goal_title = goal.title or trace.goal_title
                trace.goal_intent = goal.intent or trace.goal_intent
                trace.goal_family = goal.goal_family or trace.goal_family
                trace.commitment_id = goal.commitment_id or trace.commitment_id
                trace.updated_at = event.ts
                commitment_details = _goal_commitment_details(event)
                if commitment_details:
                    trace.details.setdefault("commitment", {}).update(commitment_details)
                if event.event_type == BrainEventType.GOAL_REPAIRED:
                    trace.details["repair_event_id"] = event.event_id
                    trace.details["repair_summary"] = goal.last_summary

        if event.event_type == BrainEventType.PLANNING_ADOPTED:
            proposal = _proposal_from_event(event)
            decision = _decision_from_event(event)
            if proposal is None:
                continue
            goal_state = goal_state_by_id.get(proposal.goal_id)
            resolved_commitment_id = proposal.commitment_id or (
                goal_state.commitment_id if goal_state is not None else None
            )
            trace_id = _stable_id(
                "procedural_trace",
                proposal.goal_id,
                resolved_commitment_id or "",
                proposal.plan_proposal_id,
                proposal.plan_revision,
            )
            prior_trace_id = None
            if proposal.supersedes_plan_proposal_id:
                prior_trace_id = trace_id_by_proposal.get(proposal.supersedes_plan_proposal_id)
                if prior_trace_id is not None and prior_trace_id in trace_by_id:
                    prior_trace = trace_by_id[prior_trace_id]
                    prior_trace.status = BrainProceduralTraceStatus.SUPERSEDED.value
                    prior_trace.ended_at = event.ts
                    prior_trace.updated_at = event.ts
                    prior_trace.details["superseded_by_plan_proposal_id"] = proposal.plan_proposal_id
                    add_outcome(
                        outcome_kind=BrainProceduralOutcomeKind.SUPERSEDED_BY_REVISION.value,
                        event=event,
                        goal_id=prior_trace.goal_id,
                        commitment_id=prior_trace.commitment_id,
                        plan_proposal_id=prior_trace.plan_proposal_id,
                        plan_revision=prior_trace.plan_revision,
                        summary=f"Plan {prior_trace.plan_proposal_id} was superseded by a revised tail.",
                        reason_code="superseded_by_revision",
                        trace_id=prior_trace_id,
                        details={
                            "superseded_by_plan_proposal_id": proposal.plan_proposal_id,
                            "superseded_by_trace_id": trace_id,
                        },
                    )
            trace = _MutableExecutionTrace(
                trace_id=trace_id,
                goal_id=proposal.goal_id,
                commitment_id=resolved_commitment_id,
                plan_proposal_id=proposal.plan_proposal_id,
                goal_title=goal_state.title if goal_state is not None else proposal.summary,
                goal_intent=goal_state.intent if goal_state is not None else "",
                goal_family=goal_state.goal_family if goal_state is not None else "",
                proposal_source=proposal.source,
                review_policy=proposal.review_policy,
                current_plan_revision=proposal.current_plan_revision,
                plan_revision=proposal.plan_revision,
                preserved_prefix_count=proposal.preserved_prefix_count,
                supersedes_plan_proposal_id=proposal.supersedes_plan_proposal_id,
                status=BrainProceduralTraceStatus.OPEN.value,
                planned_steps=[BrainGoalStep.from_dict(step.as_dict()) for step in proposal.steps],
                started_at=event.ts,
                updated_at=event.ts,
                details={
                    "planning_adopted_event_id": event.event_id,
                    "planning_proposed_event_id": event.causal_parent_id,
                    "assumptions": list(proposal.assumptions),
                    "missing_inputs": list(proposal.missing_inputs),
                    "proposal_summary": proposal.summary,
                    "proposal_details": dict(proposal.details),
                    "procedural": dict(proposal.details.get("procedural", {})),
                },
            )
            _append_unique(trace.source_event_ids, event.event_id, event.causal_parent_id)
            trace_by_id[trace_id] = trace
            trace_id_by_goal[proposal.goal_id] = trace_id
            trace_id_by_proposal[proposal.plan_proposal_id] = trace_id
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.PLANNING_ADOPTED.value,
                event=event,
                goal_id=proposal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=proposal.plan_proposal_id,
                plan_revision=proposal.plan_revision,
                summary=decision.summary if decision is not None else proposal.summary,
                reason_code=decision.reason if decision is not None else None,
                trace_id=trace_id,
                details={
                    "proposal_source": proposal.source,
                    "procedural": dict(proposal.details.get("procedural", {})),
                },
            )
            continue

        if event.event_type == BrainEventType.PLANNING_REJECTED:
            proposal = _proposal_from_event(event)
            decision = _decision_from_event(event)
            if proposal is None:
                continue
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.PLANNING_REJECTED.value,
                event=event,
                goal_id=proposal.goal_id,
                commitment_id=proposal.commitment_id,
                plan_proposal_id=proposal.plan_proposal_id,
                plan_revision=proposal.plan_revision,
                summary=decision.summary if decision is not None else proposal.summary,
                reason_code=decision.reason if decision is not None else None,
                trace_id=trace_id_by_proposal.get(proposal.plan_proposal_id),
                details={
                    "proposal_source": proposal.source,
                    "review_policy": proposal.review_policy,
                    "procedural": dict(proposal.details.get("procedural", {})),
                },
            )
            continue

        if event.event_type == BrainEventType.CAPABILITY_REQUESTED:
            payload = _event_payload(event)
            goal_id = str(payload.get("goal_id", "")).strip()
            trace_id = trace_id_by_goal.get(goal_id)
            if not goal_id or trace_id is None or trace_id not in trace_by_id:
                continue
            trace = trace_by_id[trace_id]
            step_index = int(payload.get("step_index", 0))
            capability_id = str(payload.get("capability_id", "")).strip()
            if not capability_id:
                continue
            step_trace_id = _stable_id(
                "procedural_step",
                trace_id,
                event.event_id,
                capability_id,
                step_index,
            )
            trace.step_by_request_event_id[event.event_id] = _MutableStepTrace(
                step_trace_id=step_trace_id,
                trace_id=trace_id,
                step_index=step_index,
                capability_id=capability_id,
                arguments=dict(payload.get("arguments", {})),
                status="requested",
                attempts=1,
                request_event_id=event.event_id,
                source_event_ids=[event.event_id],
                started_at=event.ts,
                updated_at=event.ts,
            )
            trace.updated_at = event.ts
            _append_unique(trace.source_event_ids, event.event_id)
            continue

        if event.event_type in {
            BrainEventType.CAPABILITY_COMPLETED,
            BrainEventType.CAPABILITY_FAILED,
            BrainEventType.CRITIC_FEEDBACK,
        }:
            request_event_id = _optional_text(event.causal_parent_id)
            if request_event_id is None:
                continue
            trace = next(
                (
                    candidate
                    for candidate in trace_by_id.values()
                    if request_event_id in candidate.step_by_request_event_id
                ),
                None,
            )
            if trace is None:
                continue
            step = trace.step_by_request_event_id[request_event_id]
            payload = _event_payload(event)
            result = dict(payload.get("result", {}))
            step.result_event_id = event.event_id
            step.summary = _optional_text(result.get("summary")) or step.summary
            step.error_code = _optional_text(result.get("error_code")) or step.error_code
            step.warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
            step.output = dict(result.get("output", {}))
            if event.event_type == BrainEventType.CAPABILITY_COMPLETED:
                step.status = str(result.get("outcome", "completed")).strip() or "completed"
            elif event.event_type == BrainEventType.CAPABILITY_FAILED:
                step.status = str(result.get("outcome", "failed")).strip() or "failed"
            else:
                recovery = dict(payload.get("recovery", {}))
                recovery_decision = _optional_text(recovery.get("decision"))
                step.status = recovery_decision or str(result.get("outcome", "failed")).strip() or "failed"
                if recovery:
                    step.details["recovery"] = recovery
            step.details["result"] = result
            step.updated_at = event.ts
            _append_unique(step.source_event_ids, event.event_id)
            _append_unique(trace.source_event_ids, event.event_id, request_event_id)
            trace.updated_at = event.ts
            continue

        if goal is None or not goal.goal_id:
            continue

        current_trace_id = trace_id_by_goal.get(goal.goal_id)
        if current_trace_id is None or current_trace_id not in trace_by_id:
            continue
        trace = trace_by_id[current_trace_id]
        trace.commitment_id = goal.commitment_id or trace.commitment_id
        step_trace_ids = [
            step.step_trace_id
            for step in trace.step_by_request_event_id.values()
            if step.request_event_id == event.causal_parent_id
        ]

        if event.event_type == BrainEventType.GOAL_UPDATED and goal.status == BrainGoalStatus.BLOCKED.value:
            trace.status = BrainProceduralTraceStatus.PAUSED.value
            trace.updated_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.BLOCKED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=(
                    _blocked_reason(goal).summary
                    if _blocked_reason(goal) is not None
                    else goal.last_summary or goal.title
                ),
                reason_code=(
                    _blocked_reason(goal).kind
                    if _blocked_reason(goal) is not None
                    else _optional_text(goal.last_error)
                ),
                trace_id=current_trace_id,
                step_trace_ids=step_trace_ids,
                details={
                    "wake_condition_kinds": [item.kind for item in goal.wake_conditions],
                },
            )
            continue

        if event.event_type == BrainEventType.GOAL_DEFERRED:
            trace.status = BrainProceduralTraceStatus.PAUSED.value
            trace.updated_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.DEFERRED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=(
                    _blocked_reason(goal).summary
                    if _blocked_reason(goal) is not None
                    else goal.last_summary or goal.title
                ),
                reason_code=(
                    _blocked_reason(goal).kind
                    if _blocked_reason(goal) is not None
                    else _optional_text(goal.last_error)
                ),
                trace_id=current_trace_id,
                details={"resume_count": goal.resume_count},
            )
            continue

        if event.event_type == BrainEventType.GOAL_RESUMED:
            trace.status = BrainProceduralTraceStatus.OPEN.value
            trace.updated_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.RESUMED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=goal.last_summary or f"{goal.title} resumed.",
                reason_code=None,
                trace_id=current_trace_id,
                details={"resume_count": goal.resume_count},
            )
            continue

        if event.event_type == BrainEventType.GOAL_COMPLETED:
            trace.status = BrainProceduralTraceStatus.COMPLETED.value
            trace.updated_at = event.ts
            trace.ended_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.COMPLETED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=goal.last_summary or f"{goal.title} completed.",
                reason_code=None,
                trace_id=current_trace_id,
                step_trace_ids=step_trace_ids,
                details={"resume_count": goal.resume_count},
            )
            trace_id_by_goal.pop(goal.goal_id, None)
            continue

        if event.event_type == BrainEventType.GOAL_FAILED:
            trace.status = BrainProceduralTraceStatus.FAILED.value
            trace.updated_at = event.ts
            trace.ended_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.FAILED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=goal.last_summary or f"{goal.title} failed.",
                reason_code=_optional_text(goal.last_error),
                trace_id=current_trace_id,
                step_trace_ids=step_trace_ids,
                details={
                    "blocked_reason": _blocked_reason(goal).as_dict()
                    if _blocked_reason(goal) is not None
                    else None,
                },
            )
            trace_id_by_goal.pop(goal.goal_id, None)
            continue

        if event.event_type == BrainEventType.GOAL_CANCELLED:
            trace.status = BrainProceduralTraceStatus.CANCELLED.value
            trace.updated_at = event.ts
            trace.ended_at = event.ts
            add_outcome(
                outcome_kind=BrainProceduralOutcomeKind.CANCELLED.value,
                event=event,
                goal_id=goal.goal_id,
                commitment_id=trace.commitment_id,
                plan_proposal_id=trace.plan_proposal_id,
                plan_revision=trace.plan_revision,
                summary=goal.last_summary or f"{goal.title} cancelled.",
                reason_code=_optional_text(goal.last_error),
                trace_id=current_trace_id,
                details={"resume_count": goal.resume_count},
            )
            trace_id_by_goal.pop(goal.goal_id, None)
            continue

    trace_records = [trace.to_record() for trace in trace_by_id.values()]
    trace_counts: dict[str, int] = {}
    for record in trace_records:
        trace_counts[record.status] = trace_counts.get(record.status, 0) + 1
    outcome_counts: dict[str, int] = {}
    for record in outcome_by_id.values():
        outcome_counts[record.outcome_kind] = outcome_counts.get(record.outcome_kind, 0) + 1
    return BrainProceduralTraceProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        trace_counts=trace_counts,
        outcome_counts=outcome_counts,
        traces=sorted(trace_records, key=_trace_sort_key),
        outcomes=sorted(outcome_by_id.values(), key=_outcome_sort_key),
        open_trace_ids=_trace_status_ids(trace_records, BrainProceduralTraceStatus.OPEN.value),
        paused_trace_ids=_trace_status_ids(trace_records, BrainProceduralTraceStatus.PAUSED.value),
        completed_trace_ids=_trace_status_ids(trace_records, BrainProceduralTraceStatus.COMPLETED.value),
        failed_trace_ids=_trace_status_ids(trace_records, BrainProceduralTraceStatus.FAILED.value),
        cancelled_trace_ids=_trace_status_ids(trace_records, BrainProceduralTraceStatus.CANCELLED.value),
        superseded_trace_ids=_trace_status_ids(
            trace_records,
            BrainProceduralTraceStatus.SUPERSEDED.value,
        ),
    )


__all__ = [
    "BrainProceduralExecutionTraceRecord",
    "BrainProceduralOutcomeKind",
    "BrainProceduralOutcomeRecord",
    "BrainProceduralStepTraceRecord",
    "BrainProceduralTraceProjection",
    "BrainProceduralTraceStatus",
    "build_procedural_trace_projection",
]

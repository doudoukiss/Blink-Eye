"""Derived procedural skill projection for Blink memory v2."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.memory_v2.procedural import (
    BrainProceduralExecutionTraceRecord,
    BrainProceduralOutcomeKind,
    BrainProceduralOutcomeRecord,
    BrainProceduralTraceProjection,
    BrainProceduralTraceStatus,
)
from blink.brain.projections import BrainGoalStep


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


def _condition_sort_key(record: "BrainProceduralActivationConditionRecord") -> tuple[str, str, str]:
    return (record.kind, record.match_value, record.condition_id)


def _invariant_sort_key(record: "BrainProceduralInvariantRecord") -> tuple[str, str, str]:
    return (record.kind, record.summary, record.invariant_id)


def _effect_sort_key(record: "BrainProceduralEffectRecord") -> tuple[str, str, str]:
    return (record.kind, record.summary, record.effect_id)


def _failure_signature_sort_key(
    record: "BrainProceduralFailureSignatureRecord",
) -> tuple[int, str, str, str]:
    return (
        -int(record.details.get("support_count", 0)),
        record.kind,
        record.reason_code or "",
        record.failure_signature_id,
    )


_STATUS_ORDER = {
    "candidate": 0,
    "active": 1,
    "superseded": 2,
    "retired": 3,
}


def _skill_sort_key(record: "BrainProceduralSkillRecord") -> tuple[int, str, str, str]:
    return (
        _STATUS_ORDER.get(record.status, 99),
        record.goal_family,
        record.template_fingerprint,
        record.skill_id,
    )


def _trace_sort_key(record: BrainProceduralExecutionTraceRecord) -> tuple[str, str]:
    return (
        record.ended_at or record.updated_at or record.started_at,
        record.trace_id,
    )


def _trace_last_ts(record: BrainProceduralExecutionTraceRecord) -> str:
    return record.ended_at or record.updated_at or record.started_at


def _copy_step(step: BrainGoalStep) -> BrainGoalStep:
    return BrainGoalStep.from_dict(step.as_dict())


def _trace_template_steps(record: BrainProceduralExecutionTraceRecord) -> list[BrainGoalStep]:
    if record.planned_steps:
        source_steps = record.planned_steps
    else:
        source_steps = [
            BrainGoalStep(
                capability_id=step.capability_id,
                arguments=dict(step.arguments),
                status=step.status,
                attempts=step.attempts,
                summary=step.summary,
                error_code=step.error_code,
                warnings=list(step.warnings),
                output=dict(step.output),
                updated_at=step.updated_at,
            )
            for step in sorted(
                record.step_executions,
                key=lambda item: (int(item.step_index), item.started_at or "", item.step_trace_id),
            )
        ]
    normalized: list[BrainGoalStep] = []
    for step in source_steps:
        capability_id = _optional_text(step.capability_id)
        if capability_id is None:
            continue
        normalized.append(
            BrainGoalStep(
                capability_id=capability_id,
                arguments=dict(step.arguments),
                status="pending",
                attempts=0,
                summary=None,
                error_code=None,
                warnings=[],
                output={},
                updated_at=str(step.updated_at or _utc_now()),
            )
        )
    return normalized


def _capability_sequence(record: BrainProceduralExecutionTraceRecord) -> tuple[str, ...]:
    return tuple(
        step.capability_id
        for step in _trace_template_steps(record)
        if _optional_text(step.capability_id) is not None
    )


def _is_prefix(prefix: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not prefix or len(prefix) >= len(sequence):
        return False
    return sequence[: len(prefix)] == prefix


def _confidence_band(confidence: float) -> str:
    if confidence < 0.5:
        return "low"
    if confidence < 0.75:
        return "medium"
    return "high"


class BrainProceduralSkillStatus(str, Enum):
    """Lifecycle statuses for one consolidated procedural skill."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


@dataclass(frozen=True)
class BrainProceduralActivationConditionRecord:
    """One activation condition for a reusable procedural skill."""

    condition_id: str
    kind: str
    summary: str
    match_value: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the activation condition."""
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "summary": self.summary,
            "match_value": self.match_value,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainProceduralActivationConditionRecord | None":
        """Build an activation condition record from serialized data."""
        if not isinstance(data, dict):
            return None
        condition_id = str(data.get("condition_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        match_value = str(data.get("match_value", "")).strip()
        if not condition_id or not kind or not summary or not match_value:
            return None
        return cls(
            condition_id=condition_id,
            kind=kind,
            summary=summary,
            match_value=match_value,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralInvariantRecord:
    """One invariant that must stay true for the skill template."""

    invariant_id: str
    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the invariant."""
        return {
            "invariant_id": self.invariant_id,
            "kind": self.kind,
            "summary": self.summary,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralInvariantRecord | None":
        """Build an invariant record from serialized data."""
        if not isinstance(data, dict):
            return None
        invariant_id = str(data.get("invariant_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not invariant_id or not kind or not summary:
            return None
        return cls(
            invariant_id=invariant_id,
            kind=kind,
            summary=summary,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralEffectRecord:
    """One expected effect from reusing the procedural skill."""

    effect_id: str
    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the effect."""
        return {
            "effect_id": self.effect_id,
            "kind": self.kind,
            "summary": self.summary,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralEffectRecord | None":
        """Build an effect record from serialized data."""
        if not isinstance(data, dict):
            return None
        effect_id = str(data.get("effect_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not effect_id or not kind or not summary:
            return None
        return cls(
            effect_id=effect_id,
            kind=kind,
            summary=summary,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralFailureSignatureRecord:
    """One aggregated failure signature for a procedural skill."""

    failure_signature_id: str
    kind: str
    reason_code: str | None
    summary: str
    support_trace_ids: list[str] = field(default_factory=list)
    support_outcome_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the failure signature."""
        return {
            "failure_signature_id": self.failure_signature_id,
            "kind": self.kind,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "support_trace_ids": _sorted_unique_texts(self.support_trace_ids),
            "support_outcome_ids": _sorted_unique_texts(self.support_outcome_ids),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainProceduralFailureSignatureRecord | None":
        """Build a failure signature record from serialized data."""
        if not isinstance(data, dict):
            return None
        failure_signature_id = str(data.get("failure_signature_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not failure_signature_id or not kind or not summary:
            return None
        return cls(
            failure_signature_id=failure_signature_id,
            kind=kind,
            reason_code=_optional_text(data.get("reason_code")),
            summary=summary,
            support_trace_ids=_sorted_unique_texts(data.get("support_trace_ids", [])),
            support_outcome_ids=_sorted_unique_texts(data.get("support_outcome_ids", [])),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralSkillStatsRecord:
    """Deterministic support statistics for one procedural skill."""

    support_trace_count: int = 0
    success_trace_count: int = 0
    failure_trace_count: int = 0
    blocked_or_deferred_count: int = 0
    independent_plan_count: int = 0
    last_supported_at: str | None = None
    last_failure_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the skill stats."""
        return {
            "support_trace_count": self.support_trace_count,
            "success_trace_count": self.success_trace_count,
            "failure_trace_count": self.failure_trace_count,
            "blocked_or_deferred_count": self.blocked_or_deferred_count,
            "independent_plan_count": self.independent_plan_count,
            "last_supported_at": self.last_supported_at,
            "last_failure_at": self.last_failure_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralSkillStatsRecord":
        """Build skill stats from serialized data."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            support_trace_count=int(data.get("support_trace_count", 0)),
            success_trace_count=int(data.get("success_trace_count", 0)),
            failure_trace_count=int(data.get("failure_trace_count", 0)),
            blocked_or_deferred_count=int(data.get("blocked_or_deferred_count", 0)),
            independent_plan_count=int(data.get("independent_plan_count", 0)),
            last_supported_at=_optional_text(data.get("last_supported_at")),
            last_failure_at=_optional_text(data.get("last_failure_at")),
        )


@dataclass(frozen=True)
class BrainProceduralSkillRecord:
    """One persistent procedural skill compiled from repeated bounded traces."""

    skill_id: str
    skill_family_key: str
    template_fingerprint: str
    scope_type: str
    scope_id: str
    title: str
    purpose: str
    goal_family: str
    status: str
    confidence: float
    activation_conditions: list[BrainProceduralActivationConditionRecord] = field(default_factory=list)
    invariants: list[BrainProceduralInvariantRecord] = field(default_factory=list)
    step_template: list[BrainGoalStep] = field(default_factory=list)
    required_capability_ids: list[str] = field(default_factory=list)
    effects: list[BrainProceduralEffectRecord] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)
    failure_signatures: list[BrainProceduralFailureSignatureRecord] = field(default_factory=list)
    stats: BrainProceduralSkillStatsRecord = field(default_factory=BrainProceduralSkillStatsRecord)
    supporting_trace_ids: list[str] = field(default_factory=list)
    supporting_outcome_ids: list[str] = field(default_factory=list)
    supporting_plan_proposal_ids: list[str] = field(default_factory=list)
    supporting_commitment_ids: list[str] = field(default_factory=list)
    supersedes_skill_id: str | None = None
    superseded_by_skill_id: str | None = None
    retirement_reason: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    retired_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the procedural skill."""
        return {
            "skill_id": self.skill_id,
            "skill_family_key": self.skill_family_key,
            "template_fingerprint": self.template_fingerprint,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "title": self.title,
            "purpose": self.purpose,
            "goal_family": self.goal_family,
            "status": self.status,
            "confidence": self.confidence,
            "activation_conditions": [
                record.as_dict()
                for record in sorted(self.activation_conditions, key=_condition_sort_key)
            ],
            "invariants": [record.as_dict() for record in sorted(self.invariants, key=_invariant_sort_key)],
            "step_template": [_copy_step(step).as_dict() for step in self.step_template],
            "required_capability_ids": list(self.required_capability_ids),
            "effects": [record.as_dict() for record in sorted(self.effects, key=_effect_sort_key)],
            "termination_conditions": list(self.termination_conditions),
            "failure_signatures": [
                record.as_dict()
                for record in sorted(self.failure_signatures, key=_failure_signature_sort_key)
            ],
            "stats": self.stats.as_dict(),
            "supporting_trace_ids": _sorted_unique_texts(self.supporting_trace_ids),
            "supporting_outcome_ids": _sorted_unique_texts(self.supporting_outcome_ids),
            "supporting_plan_proposal_ids": _sorted_unique_texts(self.supporting_plan_proposal_ids),
            "supporting_commitment_ids": _sorted_unique_texts(self.supporting_commitment_ids),
            "supersedes_skill_id": self.supersedes_skill_id,
            "superseded_by_skill_id": self.superseded_by_skill_id,
            "retirement_reason": self.retirement_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retired_at": self.retired_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralSkillRecord | None":
        """Build a procedural skill record from serialized data."""
        if not isinstance(data, dict):
            return None
        skill_id = str(data.get("skill_id", "")).strip()
        skill_family_key = str(data.get("skill_family_key", "")).strip()
        template_fingerprint = str(data.get("template_fingerprint", "")).strip()
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        goal_family = str(data.get("goal_family", "")).strip()
        status = str(data.get("status", "")).strip()
        if not skill_id or not skill_family_key or not template_fingerprint or not scope_type or not scope_id:
            return None
        if not goal_family or not status:
            return None
        return cls(
            skill_id=skill_id,
            skill_family_key=skill_family_key,
            template_fingerprint=template_fingerprint,
            scope_type=scope_type,
            scope_id=scope_id,
            title=str(data.get("title", "")).strip(),
            purpose=str(data.get("purpose", "")).strip(),
            goal_family=goal_family,
            status=status,
            confidence=float(data.get("confidence", 0.0)),
            activation_conditions=[
                record
                for item in data.get("activation_conditions", [])
                if (record := BrainProceduralActivationConditionRecord.from_dict(item)) is not None
            ],
            invariants=[
                record
                for item in data.get("invariants", [])
                if (record := BrainProceduralInvariantRecord.from_dict(item)) is not None
            ],
            step_template=[BrainGoalStep.from_dict(item) for item in data.get("step_template", [])],
            required_capability_ids=_sorted_unique_texts(data.get("required_capability_ids", [])),
            effects=[
                record
                for item in data.get("effects", [])
                if (record := BrainProceduralEffectRecord.from_dict(item)) is not None
            ],
            termination_conditions=_sorted_unique_texts(data.get("termination_conditions", [])),
            failure_signatures=[
                record
                for item in data.get("failure_signatures", [])
                if (record := BrainProceduralFailureSignatureRecord.from_dict(item)) is not None
            ],
            stats=BrainProceduralSkillStatsRecord.from_dict(data.get("stats")),
            supporting_trace_ids=_sorted_unique_texts(data.get("supporting_trace_ids", [])),
            supporting_outcome_ids=_sorted_unique_texts(data.get("supporting_outcome_ids", [])),
            supporting_plan_proposal_ids=_sorted_unique_texts(
                data.get("supporting_plan_proposal_ids", [])
            ),
            supporting_commitment_ids=_sorted_unique_texts(data.get("supporting_commitment_ids", [])),
            supersedes_skill_id=_optional_text(data.get("supersedes_skill_id")),
            superseded_by_skill_id=_optional_text(data.get("superseded_by_skill_id")),
            retirement_reason=_optional_text(data.get("retirement_reason")),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            retired_at=_optional_text(data.get("retired_at")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainProceduralSkillProjection:
    """Persistent procedural skills for one scope."""

    scope_type: str
    scope_id: str
    skill_counts: dict[str, int] = field(default_factory=dict)
    confidence_band_counts: dict[str, int] = field(default_factory=dict)
    skills: list[BrainProceduralSkillRecord] = field(default_factory=list)
    candidate_skill_ids: list[str] = field(default_factory=list)
    active_skill_ids: list[str] = field(default_factory=list)
    superseded_skill_ids: list[str] = field(default_factory=list)
    retired_skill_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the skill projection."""
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "skill_counts": dict(self.skill_counts),
            "confidence_band_counts": dict(self.confidence_band_counts),
            "skills": [record.as_dict() for record in sorted(self.skills, key=_skill_sort_key)],
            "candidate_skill_ids": _sorted_unique_texts(self.candidate_skill_ids),
            "active_skill_ids": _sorted_unique_texts(self.active_skill_ids),
            "superseded_skill_ids": _sorted_unique_texts(self.superseded_skill_ids),
            "retired_skill_ids": _sorted_unique_texts(self.retired_skill_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainProceduralSkillProjection | None":
        """Build a procedural skill projection from serialized data."""
        if not isinstance(data, dict):
            return None
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        if not scope_type or not scope_id:
            return None
        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            skill_counts={str(key): int(value) for key, value in dict(data.get("skill_counts", {})).items()},
            confidence_band_counts={
                str(key): int(value) for key, value in dict(data.get("confidence_band_counts", {})).items()
            },
            skills=[
                record
                for item in data.get("skills", [])
                if (record := BrainProceduralSkillRecord.from_dict(item)) is not None
            ],
            candidate_skill_ids=_sorted_unique_texts(data.get("candidate_skill_ids", [])),
            active_skill_ids=_sorted_unique_texts(data.get("active_skill_ids", [])),
            superseded_skill_ids=_sorted_unique_texts(data.get("superseded_skill_ids", [])),
            retired_skill_ids=_sorted_unique_texts(data.get("retired_skill_ids", [])),
        )


@dataclass
class _SignatureAccumulator:
    kind: str
    reason_code: str | None
    summary: str
    support_trace_ids: list[str] = field(default_factory=list)
    support_outcome_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _projection_from_records(
    *,
    scope_type: str,
    scope_id: str,
    skills: Iterable[BrainProceduralSkillRecord],
) -> BrainProceduralSkillProjection:
    records = sorted(list(skills), key=_skill_sort_key)
    skill_counts = dict(sorted(Counter(record.status for record in records).items()))
    confidence_band_counts = dict(
        sorted(Counter(_confidence_band(float(record.confidence)) for record in records).items())
    )
    return BrainProceduralSkillProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        skill_counts=skill_counts,
        confidence_band_counts=confidence_band_counts,
        skills=records,
        candidate_skill_ids=sorted(
            record.skill_id
            for record in records
            if record.status == BrainProceduralSkillStatus.CANDIDATE.value
        ),
        active_skill_ids=sorted(
            record.skill_id
            for record in records
            if record.status == BrainProceduralSkillStatus.ACTIVE.value
        ),
        superseded_skill_ids=sorted(
            record.skill_id
            for record in records
            if record.status == BrainProceduralSkillStatus.SUPERSEDED.value
        ),
        retired_skill_ids=sorted(
            record.skill_id
            for record in records
            if record.status == BrainProceduralSkillStatus.RETIRED.value
        ),
    )


def build_procedural_skill_projection(
    *,
    scope_type: str,
    scope_id: str,
    procedural_traces: BrainProceduralTraceProjection,
) -> BrainProceduralSkillProjection:
    """Build one deterministic procedural skill projection from harvested traces."""
    completed_traces = [
        trace
        for trace in procedural_traces.traces
        if trace.status == BrainProceduralTraceStatus.COMPLETED.value
    ]
    outcomes_by_trace: dict[str, list[BrainProceduralOutcomeRecord]] = defaultdict(list)
    for outcome in procedural_traces.outcomes:
        if _optional_text(outcome.trace_id) is not None:
            outcomes_by_trace[str(outcome.trace_id)].append(outcome)

    class _Group(dict):
        traces: list[BrainProceduralExecutionTraceRecord]

    groups: dict[str, dict[str, Any]] = {}
    trace_sequences: dict[str, tuple[str, ...]] = {}
    for trace in procedural_traces.traces:
        trace_sequences[trace.trace_id] = _capability_sequence(trace)
    for trace in completed_traces:
        sequence = trace_sequences.get(trace.trace_id, ())
        if not sequence:
            continue
        skill_family_key = _stable_id(
            "procedural_skill_family",
            trace.goal_family,
            sequence[0],
            sequence[-1],
        )
        template_fingerprint = _stable_id(
            "procedural_skill_template",
            trace.goal_family,
            *sequence,
        )
        group = groups.setdefault(
            template_fingerprint,
            {
                "goal_family": trace.goal_family,
                "sequence": sequence,
                "skill_family_key": skill_family_key,
                "traces": [],
            },
        )
        group["traces"].append(trace)

    skill_records: list[BrainProceduralSkillRecord] = []
    traces_by_skill_id: dict[str, list[BrainProceduralExecutionTraceRecord]] = {}
    for template_fingerprint, group in groups.items():
        support_traces = sorted(group["traces"], key=_trace_sort_key)
        if not support_traces:
            continue
        sequence = tuple(group["sequence"])
        support_outcomes = [
            outcome
            for trace in support_traces
            for outcome in outcomes_by_trace.get(trace.trace_id, [])
            if outcome.outcome_kind == BrainProceduralOutcomeKind.COMPLETED.value
        ]
        latest_support_trace = max(support_traces, key=_trace_sort_key)
        latest_support_ts = _trace_last_ts(latest_support_trace)
        latest_summary = (
            _optional_text(latest_support_trace.details.get("proposal_summary"))
            or next(
                (
                    _optional_text(outcome.summary)
                    for outcome in sorted(
                        outcomes_by_trace.get(latest_support_trace.trace_id, []),
                        key=lambda item: (item.created_at, item.outcome_id),
                        reverse=True,
                    )
                    if _optional_text(outcome.summary) is not None
                ),
                None,
            )
            or _optional_text(latest_support_trace.goal_title)
            or latest_support_trace.goal_family
        )
        latest_steps = _trace_template_steps(latest_support_trace)
        activation_conditions = [
            BrainProceduralActivationConditionRecord(
                condition_id=_stable_id(
                    "procedural_skill_condition",
                    template_fingerprint,
                    "goal_family",
                ),
                kind="goal_family_equals",
                summary=f"Goal family must equal {latest_support_trace.goal_family}.",
                match_value=latest_support_trace.goal_family,
                details={"goal_family": latest_support_trace.goal_family},
            ),
            BrainProceduralActivationConditionRecord(
                condition_id=_stable_id(
                    "procedural_skill_condition",
                    template_fingerprint,
                    "capability_sequence",
                ),
                kind="capability_sequence_exact",
                summary="Capability sequence must match the learned bounded procedure.",
                match_value=" -> ".join(sequence),
                details={"capability_sequence": list(sequence)},
            ),
        ]
        if all(_optional_text(trace.commitment_id) is not None for trace in support_traces):
            activation_conditions.append(
                BrainProceduralActivationConditionRecord(
                    condition_id=_stable_id(
                        "procedural_skill_condition",
                        template_fingerprint,
                        "requires_commitment",
                    ),
                    kind="requires_commitment",
                    summary="The procedure expects an existing commitment boundary.",
                    match_value="true",
                    details={},
                )
            )
        proposal_sources = {
            _optional_text(trace.proposal_source)
            for trace in support_traces
            if _optional_text(trace.proposal_source) is not None
        }
        if len(proposal_sources) == 1:
            proposal_source = next(iter(proposal_sources))
            activation_conditions.append(
                BrainProceduralActivationConditionRecord(
                    condition_id=_stable_id(
                        "procedural_skill_condition",
                        template_fingerprint,
                        "proposal_source",
                    ),
                    kind="proposal_source_equals",
                    summary=f"Proposal source must remain {proposal_source}.",
                    match_value=proposal_source,
                    details={"proposal_source": proposal_source},
                )
            )

        invariants = [
            BrainProceduralInvariantRecord(
                invariant_id=_stable_id(
                    "procedural_skill_invariant",
                    template_fingerprint,
                    "required_capabilities",
                ),
                kind="required_capability_ids",
                summary="Ordered capability ids are fixed for the learned template.",
                details={"required_capability_ids": list(sequence)},
            ),
            BrainProceduralInvariantRecord(
                invariant_id=_stable_id(
                    "procedural_skill_invariant",
                    template_fingerprint,
                    "step_count",
                ),
                kind="step_count_exact",
                summary=f"The template contains exactly {len(sequence)} steps.",
                details={"step_count": len(sequence)},
            ),
        ]
        review_policies = {
            _optional_text(trace.review_policy)
            for trace in support_traces
            if _optional_text(trace.review_policy) is not None
        }
        if len(review_policies) == 1:
            review_policy = next(iter(review_policies))
            invariants.append(
                BrainProceduralInvariantRecord(
                    invariant_id=_stable_id(
                        "procedural_skill_invariant",
                        template_fingerprint,
                        "review_policy",
                    ),
                    kind="review_policy_equals",
                    summary=f"Review policy is consistently {review_policy}.",
                    details={"review_policy": review_policy},
                )
            )

        effects = [
            BrainProceduralEffectRecord(
                effect_id=_stable_id("procedural_skill_effect", template_fingerprint, "goal_family"),
                kind="completed_goal_family",
                summary=f"Completes one {latest_support_trace.goal_family} goal.",
                details={"goal_family": latest_support_trace.goal_family},
            ),
            BrainProceduralEffectRecord(
                effect_id=_stable_id("procedural_skill_effect", template_fingerprint, "terminal_outcome"),
                kind="terminal_outcome",
                summary="Terminates with a completed outcome.",
                details={"outcome_kind": BrainProceduralOutcomeKind.COMPLETED.value},
            ),
            BrainProceduralEffectRecord(
                effect_id=_stable_id("procedural_skill_effect", template_fingerprint, "result_summary"),
                kind="result_summary",
                summary=latest_summary,
                details={"result_summary": latest_summary},
            ),
        ]

        relevant_failure_trace_ids: set[str] = set()
        blocked_or_deferred_trace_ids: set[str] = set()
        relevant_failure_ts: list[str] = []
        relevant_signature_accumulators: dict[tuple[str, str | None], _SignatureAccumulator] = {}
        post_support_relevant_failures: set[str] = set()
        for trace in procedural_traces.traces:
            if trace.trace_id in {item.trace_id for item in support_traces}:
                continue
            trace_sequence = trace_sequences.get(trace.trace_id, ())
            if trace.goal_family != latest_support_trace.goal_family or not trace_sequence:
                continue
            if not _is_prefix(sequence, trace_sequence) and trace_sequence != sequence:
                continue
            trace_last_ts = _trace_last_ts(trace)
            trace_outcomes = outcomes_by_trace.get(trace.trace_id, [])
            trace_has_failure = trace.status in {
                BrainProceduralTraceStatus.FAILED.value,
                BrainProceduralTraceStatus.CANCELLED.value,
            } or any(
                outcome.outcome_kind
                in {
                    BrainProceduralOutcomeKind.FAILED.value,
                    BrainProceduralOutcomeKind.CANCELLED.value,
                }
                for outcome in trace_outcomes
            )
            trace_has_blocked_or_deferred = trace.status == BrainProceduralTraceStatus.PAUSED.value or any(
                outcome.outcome_kind
                in {
                    BrainProceduralOutcomeKind.BLOCKED.value,
                    BrainProceduralOutcomeKind.DEFERRED.value,
                }
                for outcome in trace_outcomes
            )
            if trace_has_failure:
                relevant_failure_trace_ids.add(trace.trace_id)
                relevant_failure_ts.append(trace_last_ts)
            if trace_has_blocked_or_deferred:
                blocked_or_deferred_trace_ids.add(trace.trace_id)
                relevant_failure_ts.append(trace_last_ts)
            if trace_last_ts > latest_support_ts and (trace_has_failure or trace_has_blocked_or_deferred):
                post_support_relevant_failures.add(trace.trace_id)

            for outcome in trace_outcomes:
                if outcome.outcome_kind not in {
                    BrainProceduralOutcomeKind.FAILED.value,
                    BrainProceduralOutcomeKind.CANCELLED.value,
                    BrainProceduralOutcomeKind.BLOCKED.value,
                    BrainProceduralOutcomeKind.DEFERRED.value,
                }:
                    continue
                reason_code = _optional_text(outcome.reason_code) or outcome.outcome_kind
                summary = (
                    _optional_text(outcome.summary)
                    or f"Outcome {outcome.outcome_kind} with reason {reason_code}."
                )
                key = ("outcome_reason", reason_code)
                accumulator = relevant_signature_accumulators.setdefault(
                    key,
                    _SignatureAccumulator(
                        kind="outcome_reason",
                        reason_code=reason_code,
                        summary=summary,
                    ),
                )
                accumulator.support_trace_ids.append(trace.trace_id)
                accumulator.support_outcome_ids.append(outcome.outcome_id)
                accumulator.details.setdefault("outcome_kinds", [])
                accumulator.details["outcome_kinds"].append(outcome.outcome_kind)
            for step in trace.step_executions:
                if _optional_text(step.error_code) is not None:
                    reason_code = _optional_text(step.error_code)
                    key = ("step_error_code", reason_code)
                    accumulator = relevant_signature_accumulators.setdefault(
                        key,
                        _SignatureAccumulator(
                            kind="step_error_code",
                            reason_code=reason_code,
                            summary=f"Step {step.capability_id} failed with error {reason_code}.",
                        ),
                    )
                    accumulator.support_trace_ids.append(trace.trace_id)
                    accumulator.details.setdefault("capability_ids", [])
                    accumulator.details["capability_ids"].append(step.capability_id)
                if step.status not in {"completed", "requested", "pending"}:
                    reason_code = _optional_text(step.status)
                    key = ("step_status", reason_code)
                    accumulator = relevant_signature_accumulators.setdefault(
                        key,
                        _SignatureAccumulator(
                            kind="step_status",
                            reason_code=reason_code,
                            summary=f"Step {step.capability_id} ended with status {step.status}.",
                        ),
                    )
                    accumulator.support_trace_ids.append(trace.trace_id)
                    accumulator.details.setdefault("capability_ids", [])
                    accumulator.details["capability_ids"].append(step.capability_id)

        success_trace_count = len(support_traces)
        independent_plan_count = len(
            {
                trace.plan_proposal_id
                for trace in support_traces
                if _optional_text(trace.plan_proposal_id) is not None
            }
        )
        failure_trace_count = len(relevant_failure_trace_ids)
        blocked_or_deferred_count = len(blocked_or_deferred_trace_ids)
        status = (
            BrainProceduralSkillStatus.ACTIVE.value
            if success_trace_count >= 2 and independent_plan_count >= 2
            else BrainProceduralSkillStatus.CANDIDATE.value
        )
        confidence = 0.20
        confidence += 0.18 * min(success_trace_count, 3)
        confidence += 0.08 * min(independent_plan_count, 3)
        confidence -= 0.12 * min(failure_trace_count, 3)
        confidence -= 0.06 * min(blocked_or_deferred_count, 2)
        confidence = max(0.0, min(0.95, confidence))
        if status == BrainProceduralSkillStatus.CANDIDATE.value:
            confidence = min(confidence, 0.49)

        failure_signatures = []
        for (kind, reason_code), accumulator in relevant_signature_accumulators.items():
            support_trace_ids = _sorted_unique_texts(accumulator.support_trace_ids)
            support_outcome_ids = _sorted_unique_texts(accumulator.support_outcome_ids)
            details = dict(accumulator.details)
            details["support_count"] = len(support_trace_ids)
            for key, value in list(details.items()):
                if isinstance(value, list):
                    details[key] = _sorted_unique_texts(value)
            failure_signatures.append(
                BrainProceduralFailureSignatureRecord(
                    failure_signature_id=_stable_id(
                        "procedural_failure_signature",
                        template_fingerprint,
                        kind,
                        reason_code or "",
                    ),
                    kind=kind,
                    reason_code=reason_code,
                    summary=accumulator.summary,
                    support_trace_ids=support_trace_ids,
                    support_outcome_ids=support_outcome_ids,
                    details=details,
                )
            )
        failure_signatures = sorted(
            failure_signatures,
            key=_failure_signature_sort_key,
        )[:6]

        created_candidates = [
            parsed
            for trace in support_traces
            if (parsed := _parse_ts(trace.started_at)) is not None
        ]
        created_at = (
            min(created_candidates).isoformat()
            if created_candidates
            else min(
                (
                    _optional_text(trace.started_at)
                    or _optional_text(trace.updated_at)
                    or _optional_text(trace.ended_at)
                    or latest_support_ts
                )
                for trace in support_traces
            )
        )
        updated_candidates = [parsed for parsed in [_parse_ts(latest_support_ts)] if parsed is not None]
        if relevant_failure_ts:
            updated_candidates.extend(_parse_ts(value) for value in relevant_failure_ts if _parse_ts(value) is not None)
        updated_at = (
            max(updated_candidates).isoformat()
            if updated_candidates
            else latest_support_ts
        )
        skill_id = _stable_id("procedural_skill", scope_type, scope_id, template_fingerprint)
        supporting_trace_ids = [trace.trace_id for trace in support_traces]
        supporting_outcome_ids = [outcome.outcome_id for outcome in support_outcomes]
        supporting_plan_proposal_ids = [trace.plan_proposal_id for trace in support_traces]
        supporting_commitment_ids = [
            trace.commitment_id for trace in support_traces if _optional_text(trace.commitment_id) is not None
        ]
        stats = BrainProceduralSkillStatsRecord(
            support_trace_count=len(supporting_trace_ids),
            success_trace_count=success_trace_count,
            failure_trace_count=failure_trace_count,
            blocked_or_deferred_count=blocked_or_deferred_count,
            independent_plan_count=independent_plan_count,
            last_supported_at=latest_support_ts,
            last_failure_at=max(relevant_failure_ts) if relevant_failure_ts else None,
        )
        skill_record = BrainProceduralSkillRecord(
            skill_id=skill_id,
            skill_family_key=str(group["skill_family_key"]),
            template_fingerprint=template_fingerprint,
            scope_type=scope_type,
            scope_id=scope_id,
            title=_optional_text(latest_support_trace.goal_title) or latest_summary,
            purpose=latest_summary,
            goal_family=latest_support_trace.goal_family,
            status=status,
            confidence=confidence,
            activation_conditions=activation_conditions,
            invariants=invariants,
            step_template=latest_steps,
            required_capability_ids=list(sequence),
            effects=effects,
            termination_conditions=["goal_completed"],
            failure_signatures=failure_signatures,
            stats=stats,
            supporting_trace_ids=_sorted_unique_texts(supporting_trace_ids),
            supporting_outcome_ids=_sorted_unique_texts(supporting_outcome_ids),
            supporting_plan_proposal_ids=_sorted_unique_texts(supporting_plan_proposal_ids),
            supporting_commitment_ids=_sorted_unique_texts(supporting_commitment_ids),
            created_at=created_at,
            updated_at=updated_at,
            details={
                "confidence_band": _confidence_band(confidence),
                "goal_intents": _sorted_unique_texts(trace.goal_intent for trace in support_traces),
                "proposal_sources": _sorted_unique_texts(
                    trace.proposal_source for trace in support_traces
                ),
                "support_sequences": [list(sequence)],
                "post_support_relevant_failure_trace_ids": sorted(post_support_relevant_failures),
            },
        )
        skill_records.append(skill_record)
        traces_by_skill_id[skill_id] = support_traces

    if skill_records:
        records_by_id = {record.skill_id: record for record in skill_records}
        mutable: dict[str, dict[str, Any]] = {record.skill_id: record.as_dict() for record in skill_records}
        sequences_by_id = {
            record.skill_id: tuple(record.required_capability_ids)
            for record in skill_records
        }
        ordered_candidates = sorted(
            skill_records,
            key=lambda record: (
                len(record.required_capability_ids),
                record.created_at,
                record.skill_id,
            ),
        )
        for newer in ordered_candidates:
            if int(newer.stats.success_trace_count) < 2:
                continue
            newer_sequence = sequences_by_id[newer.skill_id]
            best_prefix: BrainProceduralSkillRecord | None = None
            for older in ordered_candidates:
                if older.skill_id == newer.skill_id:
                    continue
                if older.skill_family_key != newer.skill_family_key:
                    continue
                if newer.confidence <= older.confidence:
                    continue
                older_sequence = sequences_by_id[older.skill_id]
                if not _is_prefix(older_sequence, newer_sequence):
                    continue
                if best_prefix is None or len(older_sequence) > len(sequences_by_id[best_prefix.skill_id]):
                    best_prefix = older
            if best_prefix is None:
                continue
            mutable[best_prefix.skill_id]["status"] = BrainProceduralSkillStatus.SUPERSEDED.value
            mutable[best_prefix.skill_id]["superseded_by_skill_id"] = newer.skill_id
            mutable[best_prefix.skill_id]["updated_at"] = max(
                str(mutable[best_prefix.skill_id]["updated_at"]),
                str(mutable[newer.skill_id]["updated_at"]),
            )
            mutable[newer.skill_id]["supersedes_skill_id"] = best_prefix.skill_id

        for record in ordered_candidates:
            current = mutable[record.skill_id]
            if current.get("status") == BrainProceduralSkillStatus.SUPERSEDED.value:
                continue
            last_supported_at = _optional_text(record.stats.last_supported_at) or ""
            post_support_failures = list(
                current.get("details", {}).get("post_support_relevant_failure_trace_ids", [])
            )
            if len(post_support_failures) < 2:
                continue
            current["status"] = BrainProceduralSkillStatus.RETIRED.value
            current["retirement_reason"] = "repeated_relevant_failures"
            retired_at = max(
                (
                    _trace_last_ts(trace)
                    for trace in procedural_traces.traces
                    if trace.trace_id in post_support_failures
                ),
                default=last_supported_at,
            )
            current["retired_at"] = retired_at
            current["updated_at"] = max(str(current["updated_at"]), retired_at)

        skill_records = [
            BrainProceduralSkillRecord.from_dict(payload)
            for payload in mutable.values()
            if BrainProceduralSkillRecord.from_dict(payload) is not None
        ]
        skill_records = [record for record in skill_records if record is not None]

    return _projection_from_records(scope_type=scope_type, scope_id=scope_id, skills=skill_records)


__all__ = [
    "BrainProceduralActivationConditionRecord",
    "BrainProceduralEffectRecord",
    "BrainProceduralFailureSignatureRecord",
    "BrainProceduralInvariantRecord",
    "BrainProceduralSkillProjection",
    "BrainProceduralSkillRecord",
    "BrainProceduralSkillStatsRecord",
    "BrainProceduralSkillStatus",
    "build_procedural_skill_projection",
]

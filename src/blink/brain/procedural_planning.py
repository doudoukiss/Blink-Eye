"""Shared skill-aware planning helpers for Blink's bounded planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable

from blink.brain.memory_v2.skills import (
    BrainProceduralSkillProjection,
    BrainProceduralSkillRecord,
)
from blink.brain.projections import (
    BrainCommitmentRecord,
    BrainGoal,
    BrainGoalStep,
    BrainPlanReviewPolicy,
)


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


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _normalized_sequence(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(text for value in values if (text := _optional_text(value)) is not None)


def _query_tokens(text: str) -> tuple[str, ...]:
    normalized = (text or "").lower().replace("_", " ").replace(".", " ")
    tokens = {
        token
        for token in normalized.split()
        if token
    }
    return tuple(sorted(tokens))


_REVIEW_POLICY_RANK = {
    BrainPlanReviewPolicy.AUTO_ADOPT_OK.value: 0,
    BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value: 1,
    BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value: 2,
}


class BrainPlanningProceduralOrigin(str, Enum):
    """Explicit procedural origin for one bounded planning draft."""

    FRESH_DRAFT = "fresh_draft"
    SKILL_REUSE = "skill_reuse"
    SKILL_DELTA = "skill_delta"


class BrainPlanningSkillEligibility(str, Enum):
    """Whether a retrieved procedural skill may drive the current plan."""

    REUSABLE = "reusable"
    ADVISORY = "advisory"
    REJECTED = "rejected"


@dataclass(frozen=True)
class BrainPlanningSkillRejection:
    """Explicit reason one procedural skill was not reused."""

    skill_id: str
    reason: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the rejection."""
        return {
            "skill_id": self.skill_id,
            "reason": self.reason,
            "summary": self.summary,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanningSkillRejection | None":
        """Build a rejection record from serialized data."""
        if not isinstance(data, dict):
            return None
        skill_id = str(data.get("skill_id", "")).strip()
        reason = str(data.get("reason", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not skill_id or not reason or not summary:
            return None
        return cls(
            skill_id=skill_id,
            reason=reason,
            summary=summary,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainPlanningSkillDelta:
    """Bounded delta between a retrieved skill template and the planned result."""

    operation_count: int
    operations: tuple[dict[str, Any], ...] = ()
    selected_skill_capability_ids: tuple[str, ...] = ()
    planned_capability_ids: tuple[str, ...] = ()
    preserved_prefix_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the delta."""
        return {
            "operation_count": self.operation_count,
            "operations": [dict(item) for item in self.operations],
            "selected_skill_capability_ids": list(self.selected_skill_capability_ids),
            "planned_capability_ids": list(self.planned_capability_ids),
            "preserved_prefix_count": self.preserved_prefix_count,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanningSkillDelta | None":
        """Build a bounded delta record from serialized data."""
        if not isinstance(data, dict):
            return None
        return cls(
            operation_count=int(data.get("operation_count", 0)),
            operations=tuple(
                dict(item)
                for item in data.get("operations", [])
                if isinstance(item, dict)
            ),
            selected_skill_capability_ids=_normalized_sequence(
                data.get("selected_skill_capability_ids", [])
            ),
            planned_capability_ids=_normalized_sequence(data.get("planned_capability_ids", [])),
            preserved_prefix_count=int(data.get("preserved_prefix_count", 0)),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainPlanningSkillCandidate:
    """One retrieved procedural skill candidate for planning."""

    skill_id: str
    title: str
    purpose: str
    goal_family: str
    skill_status: str
    eligibility: str
    confidence: float
    score: float
    required_capability_ids: tuple[str, ...] = ()
    support_trace_ids: tuple[str, ...] = ()
    support_plan_proposal_ids: tuple[str, ...] = ()
    review_policy: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the candidate."""
        return {
            "skill_id": self.skill_id,
            "title": self.title,
            "purpose": self.purpose,
            "goal_family": self.goal_family,
            "skill_status": self.skill_status,
            "eligibility": self.eligibility,
            "confidence": self.confidence,
            "score": self.score,
            "required_capability_ids": list(self.required_capability_ids),
            "support_trace_ids": list(self.support_trace_ids),
            "support_plan_proposal_ids": list(self.support_plan_proposal_ids),
            "review_policy": self.review_policy,
            "reason": self.reason,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanningSkillCandidate | None":
        """Build a planning skill candidate from serialized data."""
        if not isinstance(data, dict):
            return None
        skill_id = str(data.get("skill_id", "")).strip()
        goal_family = str(data.get("goal_family", "")).strip()
        skill_status = str(data.get("skill_status", "")).strip()
        eligibility = str(data.get("eligibility", "")).strip()
        if not skill_id or not goal_family or not skill_status or not eligibility:
            return None
        return cls(
            skill_id=skill_id,
            title=str(data.get("title", "")).strip(),
            purpose=str(data.get("purpose", "")).strip(),
            goal_family=goal_family,
            skill_status=skill_status,
            eligibility=eligibility,
            confidence=float(data.get("confidence", 0.0)),
            score=float(data.get("score", 0.0)),
            required_capability_ids=_normalized_sequence(data.get("required_capability_ids", [])),
            support_trace_ids=tuple(_sorted_unique(data.get("support_trace_ids", []))),
            support_plan_proposal_ids=tuple(
                _sorted_unique(data.get("support_plan_proposal_ids", []))
            ),
            review_policy=_optional_text(data.get("review_policy")),
            reason=_optional_text(data.get("reason")),
            details=dict(data.get("details", {})),
        )


def planning_skill_review_policy(skill: BrainProceduralSkillRecord) -> str | None:
    """Return the strongest serialized review policy attached to one skill."""
    for invariant in skill.invariants:
        if invariant.kind != "review_policy_equals":
            continue
        policy = _optional_text(invariant.details.get("review_policy"))
        if policy is not None:
            return policy
    return _optional_text(skill.details.get("review_policy"))


def planning_completed_prefix(goal: BrainGoal) -> list[BrainGoalStep]:
    """Return the completed step prefix for one goal."""
    prefix: list[BrainGoalStep] = []
    for step in goal.steps:
        if step.status != "completed":
            break
        prefix.append(BrainGoalStep.from_dict(step.as_dict()))
    return prefix


def build_planning_skill_delta(
    *,
    selected_skill_capability_ids: Iterable[str],
    planned_capability_ids: Iterable[str],
    preserved_prefix_count: int = 0,
) -> BrainPlanningSkillDelta:
    """Build a deterministic edit delta between a retrieved skill and one planned tail."""
    reference = _normalized_sequence(selected_skill_capability_ids)
    planned = _normalized_sequence(planned_capability_ids)
    ref_seq = reference
    plan_seq = planned
    ref_len = len(ref_seq)
    plan_len = len(plan_seq)
    costs = [[0] * (plan_len + 1) for _ in range(ref_len + 1)]
    moves = [[None] * (plan_len + 1) for _ in range(ref_len + 1)]
    for i in range(1, ref_len + 1):
        costs[i][0] = i
        moves[i][0] = "delete"
    for j in range(1, plan_len + 1):
        costs[0][j] = j
        moves[0][j] = "insert"
    for i in range(1, ref_len + 1):
        for j in range(1, plan_len + 1):
            if ref_seq[i - 1] == plan_seq[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                moves[i][j] = "match"
                continue
            replace_cost = costs[i - 1][j - 1] + 1
            delete_cost = costs[i - 1][j] + 1
            insert_cost = costs[i][j - 1] + 1
            best_cost = min(replace_cost, delete_cost, insert_cost)
            costs[i][j] = best_cost
            if best_cost == replace_cost:
                moves[i][j] = "replace"
            elif best_cost == delete_cost:
                moves[i][j] = "delete"
            else:
                moves[i][j] = "insert"
    operations: list[dict[str, Any]] = []
    i = ref_len
    j = plan_len
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "match":
            i -= 1
            j -= 1
            continue
        if move == "replace":
            operations.append(
                {
                    "kind": "replace",
                    "index": i - 1,
                    "from_capability_id": ref_seq[i - 1],
                    "to_capability_id": plan_seq[j - 1],
                }
            )
            i -= 1
            j -= 1
            continue
        if move == "delete":
            operations.append(
                {
                    "kind": "delete",
                    "index": i - 1,
                    "from_capability_id": ref_seq[i - 1],
                }
            )
            i -= 1
            continue
        if move == "insert":
            operations.append(
                {
                    "kind": "insert",
                    "index": i,
                    "to_capability_id": plan_seq[j - 1],
                }
            )
            j -= 1
            continue
        break
    operations.reverse()
    return BrainPlanningSkillDelta(
        operation_count=len(operations),
        operations=tuple(operations),
        selected_skill_capability_ids=tuple(reference),
        planned_capability_ids=tuple(planned),
        preserved_prefix_count=preserved_prefix_count,
        details={},
    )


def is_review_policy_weaker(requested: str | None, minimum: str | None) -> bool:
    """Return whether one requested review policy is weaker than the required minimum."""
    requested_value = _optional_text(requested) or BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
    minimum_value = _optional_text(minimum)
    if minimum_value is None:
        return False
    return _REVIEW_POLICY_RANK.get(requested_value, 0) < _REVIEW_POLICY_RANK.get(minimum_value, 0)


def match_planning_skills(
    *,
    procedural_skills: BrainProceduralSkillProjection | None,
    goal: BrainGoal,
    commitment: BrainCommitmentRecord | None,
    completed_prefix: Iterable[BrainGoalStep] = (),
    query_text: str = "",
    max_candidates: int = 6,
) -> list[BrainPlanningSkillCandidate]:
    """Match persisted procedural skills to one planning request deterministically."""
    if procedural_skills is None:
        return []
    goal_family = _optional_text(goal.goal_family)
    if goal_family is None:
        return []
    goal_intent = _optional_text(goal.intent)
    has_commitment = commitment is not None or _optional_text(goal.commitment_id) is not None
    prefix_sequence = tuple(
        step.capability_id
        for step in completed_prefix
        if _optional_text(step.capability_id) is not None
    )
    query_tokens = _query_tokens(query_text or goal.title or "")
    matches: list[BrainPlanningSkillCandidate] = []
    for skill in procedural_skills.skills:
        if skill.goal_family != goal_family:
            continue
        required_capability_ids = _normalized_sequence(skill.required_capability_ids)
        if not required_capability_ids:
            continue
        score = float(skill.confidence)
        reason = None
        eligibility = BrainPlanningSkillEligibility.REUSABLE.value
        details = {
            "template_fingerprint": skill.template_fingerprint,
            "skill_family_key": skill.skill_family_key,
        }
        if skill.status == "candidate":
            eligibility = BrainPlanningSkillEligibility.ADVISORY.value
            reason = "candidate_skill_advisory_only"
            score -= 0.75
        elif skill.status != "active":
            eligibility = BrainPlanningSkillEligibility.REJECTED.value
            reason = "skill_not_reusable"
            score -= 2.0

        if reason is None and prefix_sequence:
            if required_capability_ids[: len(prefix_sequence)] != prefix_sequence:
                eligibility = BrainPlanningSkillEligibility.REJECTED.value
                reason = "completed_prefix_mismatch"
                score -= 1.5
            else:
                score += 0.5
        if reason is None and not has_commitment and any(
            condition.kind == "requires_commitment" for condition in skill.activation_conditions
        ):
            eligibility = BrainPlanningSkillEligibility.REJECTED.value
            reason = "requires_commitment_mismatch"
            score -= 1.0

        goal_intents = tuple(
            _sorted_unique((skill.details.get("goal_intents") or []))
        )
        if goal_intent is not None and goal_intents:
            if goal_intent in goal_intents:
                score += 0.6
            elif reason is None:
                eligibility = BrainPlanningSkillEligibility.REJECTED.value
                reason = "goal_intent_mismatch"
                score -= 1.25

        review_policy = planning_skill_review_policy(skill)
        if review_policy is not None:
            details["review_policy"] = review_policy

        haystack = " ".join(
            part
            for part in (
                skill.title,
                skill.purpose,
                skill.goal_family,
                " ".join(required_capability_ids),
            )
            if part
        ).lower()
        token_matches = sum(1 for token in query_tokens if token in haystack)
        score += token_matches * 0.35

        last_supported_at = _parse_ts(skill.stats.last_supported_at or skill.updated_at)
        if last_supported_at is not None:
            age_days = max((datetime.now(UTC) - last_supported_at).days, 0)
            score += max(0.0, 0.5 - min(age_days, 30) * 0.02)

        if skill.failure_signatures:
            score -= min(len(skill.failure_signatures), 3) * 0.1

        matches.append(
            BrainPlanningSkillCandidate(
                skill_id=skill.skill_id,
                title=skill.title,
                purpose=skill.purpose,
                goal_family=skill.goal_family,
                skill_status=skill.status,
                eligibility=eligibility,
                confidence=float(skill.confidence),
                score=score,
                required_capability_ids=tuple(required_capability_ids),
                support_trace_ids=tuple(_sorted_unique(skill.supporting_trace_ids)),
                support_plan_proposal_ids=tuple(
                    _sorted_unique(skill.supporting_plan_proposal_ids)
                ),
                review_policy=review_policy,
                reason=reason,
                details=details,
            )
        )
    eligibility_order = {
        BrainPlanningSkillEligibility.REUSABLE.value: 0,
        BrainPlanningSkillEligibility.ADVISORY.value: 1,
        BrainPlanningSkillEligibility.REJECTED.value: 2,
    }
    return sorted(
        matches,
        key=lambda record: (
            eligibility_order.get(record.eligibility, 99),
            -record.score,
            record.title,
            record.skill_id,
        ),
    )[:max_candidates]


def split_planning_skill_candidates(
    candidates: Iterable[BrainPlanningSkillCandidate],
) -> dict[str, list[dict[str, Any]]]:
    """Split skill candidates by eligibility for JSON payloads."""
    split = {
        BrainPlanningSkillEligibility.REUSABLE.value: [],
        BrainPlanningSkillEligibility.ADVISORY.value: [],
        BrainPlanningSkillEligibility.REJECTED.value: [],
    }
    for candidate in candidates:
        split.setdefault(candidate.eligibility, []).append(candidate.as_dict())
    return split


__all__ = [
    "BrainPlanningProceduralOrigin",
    "BrainPlanningSkillCandidate",
    "BrainPlanningSkillDelta",
    "BrainPlanningSkillEligibility",
    "BrainPlanningSkillRejection",
    "build_planning_skill_delta",
    "is_review_policy_weaker",
    "match_planning_skills",
    "planning_completed_prefix",
    "planning_skill_review_policy",
    "split_planning_skill_candidates",
]

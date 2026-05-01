"""Typed executive-policy helpers for Blink's executive."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from blink.brain.context.policy import BrainContextTask
from blink.brain.core.projections import BrainCommitmentScopeType, BrainGoal, BrainGoalFamily
from blink.brain.core.session import BrainSessionIds


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sorted_unique_texts(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return sorted({text for value in values or () if (text := _optional_text(value)) is not None})


class BrainExecutivePolicyActionPosture(str, Enum):
    """Compiled action posture for one executive pass."""

    ALLOW = "allow"
    DEFER = "defer"
    SUPPRESS = "suppress"


class BrainExecutivePolicyConservatism(str, Enum):
    """Compiled conservatism level for one executive pass."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"


class BrainExecutivePolicyApprovalRequirement(str, Enum):
    """Approval requirement implied by one executive policy frame."""

    NONE = "none"
    USER_CONFIRMATION = "user_confirmation"
    OPERATOR_REVIEW = "operator_review"


class BrainExecutiveProceduralReuseEligibility(str, Enum):
    """Whether procedural reuse is allowed for one executive pass."""

    ALLOWED = "allowed"
    ADVISORY_ONLY = "advisory_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BrainExecutiveScopeIds:
    """Resolved durable commitment scope ids for one session."""

    relationship_scope_id: str
    thread_scope_id: str
    agent_scope_id: str


@dataclass(frozen=True)
class BrainCommitmentPromotionDecision:
    """Deterministic commitment-promotion decision for one goal."""

    durable: bool
    scope_type: str | None
    reason: str


@dataclass(frozen=True)
class BrainExecutivePolicyFrame:
    """Derived executive-policy surface for one bounded decision pass."""

    task: str
    action_posture: str = BrainExecutivePolicyActionPosture.ALLOW.value
    conservatism: str = BrainExecutivePolicyConservatism.NORMAL.value
    approval_requirement: str = BrainExecutivePolicyApprovalRequirement.NONE.value
    procedural_reuse_eligibility: str = BrainExecutiveProceduralReuseEligibility.ALLOWED.value
    reason_codes: list[str] = field(default_factory=list)
    suppression_codes: list[str] = field(default_factory=list)
    review_debt_count: int = 0
    held_claim_count: int = 0
    stale_claim_count: int = 0
    unresolved_active_state_count: int = 0
    blocked_commitment_count: int = 0
    deferred_commitment_count: int = 0
    pending_user_review_count: int = 0
    pending_operator_review_count: int = 0
    scene_degraded_mode: str = "healthy"
    source_dossier_ids: list[str] = field(default_factory=list)
    source_active_record_ids: list[str] = field(default_factory=list)
    source_claim_ids: list[str] = field(default_factory=list)
    source_commitment_ids: list[str] = field(default_factory=list)
    source_plan_proposal_ids: list[str] = field(default_factory=list)
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the executive-policy frame."""
        return {
            "task": self.task,
            "action_posture": self.action_posture,
            "conservatism": self.conservatism,
            "approval_requirement": self.approval_requirement,
            "procedural_reuse_eligibility": self.procedural_reuse_eligibility,
            "reason_codes": list(self.reason_codes),
            "suppression_codes": list(self.suppression_codes),
            "review_debt_count": int(self.review_debt_count),
            "held_claim_count": int(self.held_claim_count),
            "stale_claim_count": int(self.stale_claim_count),
            "unresolved_active_state_count": int(self.unresolved_active_state_count),
            "blocked_commitment_count": int(self.blocked_commitment_count),
            "deferred_commitment_count": int(self.deferred_commitment_count),
            "pending_user_review_count": int(self.pending_user_review_count),
            "pending_operator_review_count": int(self.pending_operator_review_count),
            "scene_degraded_mode": self.scene_degraded_mode,
            "source_dossier_ids": list(self.source_dossier_ids),
            "source_active_record_ids": list(self.source_active_record_ids),
            "source_claim_ids": list(self.source_claim_ids),
            "source_commitment_ids": list(self.source_commitment_ids),
            "source_plan_proposal_ids": list(self.source_plan_proposal_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainExecutivePolicyFrame | None":
        """Hydrate one executive-policy frame from stored JSON."""
        if not isinstance(data, dict):
            return None
        task = _optional_text(data.get("task"))
        if task is None:
            return None
        return cls(
            task=task,
            action_posture=(
                _optional_text(data.get("action_posture"))
                or BrainExecutivePolicyActionPosture.ALLOW.value
            ),
            conservatism=(
                _optional_text(data.get("conservatism"))
                or BrainExecutivePolicyConservatism.NORMAL.value
            ),
            approval_requirement=(
                _optional_text(data.get("approval_requirement"))
                or BrainExecutivePolicyApprovalRequirement.NONE.value
            ),
            procedural_reuse_eligibility=(
                _optional_text(data.get("procedural_reuse_eligibility"))
                or BrainExecutiveProceduralReuseEligibility.ALLOWED.value
            ),
            reason_codes=_sorted_unique_texts(data.get("reason_codes")),
            suppression_codes=_sorted_unique_texts(data.get("suppression_codes")),
            review_debt_count=int(data.get("review_debt_count") or 0),
            held_claim_count=int(data.get("held_claim_count") or 0),
            stale_claim_count=int(data.get("stale_claim_count") or 0),
            unresolved_active_state_count=int(data.get("unresolved_active_state_count") or 0),
            blocked_commitment_count=int(data.get("blocked_commitment_count") or 0),
            deferred_commitment_count=int(data.get("deferred_commitment_count") or 0),
            pending_user_review_count=int(data.get("pending_user_review_count") or 0),
            pending_operator_review_count=int(data.get("pending_operator_review_count") or 0),
            scene_degraded_mode=_optional_text(data.get("scene_degraded_mode")) or "healthy",
            source_dossier_ids=_sorted_unique_texts(data.get("source_dossier_ids")),
            source_active_record_ids=_sorted_unique_texts(data.get("source_active_record_ids")),
            source_claim_ids=_sorted_unique_texts(data.get("source_claim_ids")),
            source_commitment_ids=_sorted_unique_texts(data.get("source_commitment_ids")),
            source_plan_proposal_ids=_sorted_unique_texts(data.get("source_plan_proposal_ids")),
            updated_at=str(data.get("updated_at") or ""),
        )


def neutral_executive_policy_frame(
    *,
    task: BrainContextTask | str,
    updated_at: str = "",
) -> BrainExecutivePolicyFrame:
    """Return a neutral allow-policy frame for compatibility paths."""
    return BrainExecutivePolicyFrame(
        task=getattr(task, "value", str(task)),
        updated_at=updated_at,
    )


def build_executive_scope_ids(session_ids: BrainSessionIds) -> BrainExecutiveScopeIds:
    """Resolve the canonical commitment scope ids for one session."""
    return BrainExecutiveScopeIds(
        relationship_scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
        thread_scope_id=session_ids.thread_id,
        agent_scope_id=session_ids.agent_id,
    )


def default_commitment_scope_type(goal_family: str) -> str:
    """Return the default durable scope type for one goal family."""
    if goal_family == BrainGoalFamily.MEMORY_MAINTENANCE.value:
        return BrainCommitmentScopeType.AGENT.value
    if goal_family == BrainGoalFamily.ENVIRONMENT.value:
        return BrainCommitmentScopeType.THREAD.value
    return BrainCommitmentScopeType.RELATIONSHIP.value


def should_auto_promote_goal(goal: BrainGoal) -> BrainCommitmentPromotionDecision:
    """Return whether a newly created transient goal should be promoted immediately."""
    if goal.details.get("transient_only"):
        return BrainCommitmentPromotionDecision(
            durable=False,
            scope_type=None,
            reason="explicit_transient_override",
        )
    if goal.details.get("durable_commitment"):
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=default_commitment_scope_type(goal.goal_family),
            reason="explicit_durable_override",
        )
    if goal.goal_family == BrainGoalFamily.CONVERSATION.value:
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.RELATIONSHIP.value,
            reason="conversation_goal_default",
        )
    if goal.goal_family == BrainGoalFamily.MEMORY_MAINTENANCE.value and _goal_needs_durability(goal):
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.AGENT.value,
            reason="maintenance_goal_requires_restart_survival",
        )
    if goal.goal_family == BrainGoalFamily.ENVIRONMENT.value and _environment_goal_needs_durability(goal):
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.THREAD.value,
            reason="environment_goal_marked_durable",
        )
    return BrainCommitmentPromotionDecision(
        durable=False,
        scope_type=None,
        reason="transient_by_default",
    )


def should_promote_goal_on_block(goal: BrainGoal) -> BrainCommitmentPromotionDecision:
    """Return whether a transient blocked goal should be promoted durably."""
    if goal.details.get("transient_only"):
        return BrainCommitmentPromotionDecision(
            durable=False,
            scope_type=None,
            reason="explicit_transient_override",
        )
    if goal.goal_family == BrainGoalFamily.CONVERSATION.value:
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.RELATIONSHIP.value,
            reason="conversation_goal_blocked",
        )
    if goal.goal_family == BrainGoalFamily.MEMORY_MAINTENANCE.value and _goal_can_be_resumed(goal):
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.AGENT.value,
            reason="maintenance_goal_blocked",
        )
    if goal.goal_family == BrainGoalFamily.ENVIRONMENT.value and (
        _goal_can_be_resumed(goal) or _environment_goal_needs_durability(goal)
    ):
        return BrainCommitmentPromotionDecision(
            durable=True,
            scope_type=BrainCommitmentScopeType.THREAD.value,
            reason="environment_goal_blocked",
        )
    return BrainCommitmentPromotionDecision(
        durable=False,
        scope_type=None,
        reason="blocked_goal_remains_transient",
    )


def _goal_can_be_resumed(goal: BrainGoal) -> bool:
    return bool(
        goal.details.get("allow_defer")
        or goal.details.get("survive_restart")
        or goal.details.get("user_visible")
        or goal.details.get("resumable")
    )


def _goal_needs_durability(goal: BrainGoal) -> bool:
    return bool(goal.details.get("spans_turns") or _goal_can_be_resumed(goal))


def _environment_goal_needs_durability(goal: BrainGoal) -> bool:
    capabilities = list(goal.details.get("capabilities", []))
    return bool(
        _goal_can_be_resumed(goal)
        or goal.details.get("durable_environment")
        or goal.details.get("survive_restart")
        or (goal.details.get("user_visible") and capabilities)
    )


__all__ = [
    "BrainCommitmentPromotionDecision",
    "BrainExecutivePolicyActionPosture",
    "BrainExecutivePolicyApprovalRequirement",
    "BrainExecutivePolicyConservatism",
    "BrainExecutivePolicyFrame",
    "BrainExecutiveProceduralReuseEligibility",
    "BrainExecutiveScopeIds",
    "build_executive_scope_ids",
    "default_commitment_scope_type",
    "neutral_executive_policy_frame",
    "should_auto_promote_goal",
    "should_promote_goal_on_block",
]

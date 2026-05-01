"""Bounded planning coordinator for replayable durable work proposals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from blink.brain._executive.policy import (
    BrainExecutivePolicyActionPosture,
    BrainExecutivePolicyApprovalRequirement,
    BrainExecutivePolicyFrame,
    BrainExecutiveProceduralReuseEligibility,
)
from blink.brain.capabilities import CapabilityFamily, CapabilityRegistry
from blink.brain.procedural_planning import (
    BrainPlanningProceduralOrigin,
    BrainPlanningSkillCandidate,
    BrainPlanningSkillDelta,
    BrainPlanningSkillEligibility,
    BrainPlanningSkillRejection,
    build_planning_skill_delta,
    is_review_policy_weaker,
)
from blink.brain.projections import (
    BrainCommitmentRecord,
    BrainGoal,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _append_reason_code(target: list[str], value: str | None) -> None:
    text = _optional_text(value)
    if text is not None and text not in target:
        target.append(text)


def _policy_local_reason_code(executive_policy: BrainExecutivePolicyFrame | None) -> str | None:
    if executive_policy is None:
        return None
    if executive_policy.action_posture == BrainExecutivePolicyActionPosture.SUPPRESS.value:
        return "policy_blocked_action"
    if (
        executive_policy.approval_requirement
        == BrainExecutivePolicyApprovalRequirement.USER_CONFIRMATION.value
    ):
        return "policy_requires_confirmation"
    if executive_policy.action_posture == BrainExecutivePolicyActionPosture.DEFER.value:
        return "policy_conservative_deferral"
    return None


def _decision_reason_codes(
    *,
    reason: str,
    executive_policy: BrainExecutivePolicyFrame | None,
    policy_changed: bool,
) -> list[str]:
    codes: list[str] = []
    _append_reason_code(codes, reason)
    if policy_changed:
        _append_reason_code(codes, _policy_local_reason_code(executive_policy))
        for value in (executive_policy.reason_codes if executive_policy is not None else ()):
            _append_reason_code(codes, value)
    return codes


def _review_policy_rank(value: str | None) -> int:
    normalized = _optional_text(value) or BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
    return {
        BrainPlanReviewPolicy.AUTO_ADOPT_OK.value: 0,
        BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value: 1,
        BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value: 2,
    }.get(normalized, 0)


def _review_policy_for_outcome(outcome: str) -> str:
    if outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value:
        return BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value
    if outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value:
        return BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
    return BrainPlanReviewPolicy.AUTO_ADOPT_OK.value


def _outcome_for_review_policy(review_policy: str) -> str:
    if review_policy == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
        return BrainPlanningOutcome.NEEDS_USER_REVIEW.value
    if review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value:
        return BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    return BrainPlanningOutcome.AUTO_ADOPTED.value


def _policy_review_floor(
    *,
    current_review_policy: str,
    executive_policy: BrainExecutivePolicyFrame | None,
) -> str:
    if executive_policy is None:
        return current_review_policy
    if (
        executive_policy.action_posture == BrainExecutivePolicyActionPosture.SUPPRESS.value
        or executive_policy.approval_requirement
        == BrainExecutivePolicyApprovalRequirement.OPERATOR_REVIEW.value
        or (
            executive_policy.action_posture == BrainExecutivePolicyActionPosture.DEFER.value
            and executive_policy.approval_requirement
            == BrainExecutivePolicyApprovalRequirement.NONE.value
        )
    ):
        return BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
    if (
        executive_policy.approval_requirement
        == BrainExecutivePolicyApprovalRequirement.USER_CONFIRMATION.value
    ):
        return BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value
    return current_review_policy


def _policy_hold_summary(review_policy: str) -> str:
    if review_policy == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
        return "The plan is available but held for user confirmation under the current executive policy."
    return "The plan is available but held for operator review under the current executive policy."


def _policy_allows_skill_linked_hold(executive_policy: BrainExecutivePolicyFrame | None) -> bool:
    if executive_policy is None:
        return False
    return executive_policy.procedural_reuse_eligibility in {
        BrainExecutiveProceduralReuseEligibility.ADVISORY_ONLY.value,
        BrainExecutiveProceduralReuseEligibility.BLOCKED.value,
    }


def apply_executive_policy_to_planning_result(
    *,
    result: "BrainPlanningCoordinatorResult",
    executive_policy: BrainExecutivePolicyFrame | None,
) -> "BrainPlanningCoordinatorResult":
    """Attach policy metadata and conservatively floor planning outcomes when needed."""
    proposal = result.proposal
    decision = result.decision
    updated_decision_details = dict(decision.details)
    current_review_policy = proposal.review_policy or _review_policy_for_outcome(result.outcome)
    floored_review_policy = current_review_policy
    policy_changed = False
    new_outcome = result.outcome
    new_summary = decision.summary

    if result.outcome != BrainPlanningOutcome.REJECTED.value:
        floored_review_policy = _policy_review_floor(
            current_review_policy=current_review_policy,
            executive_policy=executive_policy,
        )
        if _review_policy_rank(floored_review_policy) > _review_policy_rank(current_review_policy):
            policy_changed = True
            new_outcome = _outcome_for_review_policy(floored_review_policy)
            new_summary = _policy_hold_summary(floored_review_policy)
            updated_decision_details.update(
                {
                    "policy_changed_outcome": True,
                    "original_outcome": result.outcome,
                    "original_review_policy": current_review_policy,
                    "policy_review_policy_floor": floored_review_policy,
                    "policy_reason_code": _policy_local_reason_code(executive_policy),
                }
            )
            proposal = replace(
                proposal,
                review_policy=floored_review_policy,
                details={
                    **proposal.details,
                    "policy_changed_outcome": True,
                    "policy_review_policy_floor": floored_review_policy,
                    "policy_reason_code": _policy_local_reason_code(executive_policy),
                },
            )

    updated_decision = replace(
        decision,
        summary=new_summary,
        details=updated_decision_details,
        reason_codes=_decision_reason_codes(
            reason=decision.reason,
            executive_policy=executive_policy,
            policy_changed=policy_changed,
        ),
        executive_policy=(
            executive_policy.as_dict() if executive_policy is not None else None
        ),
    )
    return BrainPlanningCoordinatorResult(
        progressed=result.progressed,
        outcome=new_outcome,
        proposal=proposal,
        decision=updated_decision,
    )


class BrainPlanningRequestKind(str, Enum):
    """Supported bounded-planning request kinds."""

    INITIAL_PLAN = "initial_plan"
    REVISE_TAIL = "revise_tail"


class BrainPlanningOutcome(str, Enum):
    """Terminal coordinator outcomes for one bounded-planning request."""

    AUTO_ADOPTED = "auto_adopted"
    NEEDS_USER_REVIEW = "needs_user_review"
    NEEDS_OPERATOR_REVIEW = "needs_operator_review"
    REJECTED = "rejected"


@dataclass(frozen=True)
class BrainPlanningDraft:
    """One provider-produced bounded planning draft."""

    summary: str
    remaining_steps: list[BrainGoalStep] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    review_policy: str | None = None
    procedural_origin: str = BrainPlanningProceduralOrigin.FRESH_DRAFT.value
    selected_skill_id: str | None = None
    selected_skill_support_trace_ids: list[str] = field(default_factory=list)
    selected_skill_support_plan_proposal_ids: list[str] = field(default_factory=list)
    rejected_skills: list[BrainPlanningSkillRejection] = field(default_factory=list)
    delta: BrainPlanningSkillDelta | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the draft."""
        return {
            "summary": self.summary,
            "remaining_steps": [step.as_dict() for step in self.remaining_steps],
            "assumptions": list(self.assumptions),
            "missing_inputs": list(self.missing_inputs),
            "review_policy": self.review_policy,
            "procedural_origin": self.procedural_origin,
            "selected_skill_id": self.selected_skill_id,
            "selected_skill_support_trace_ids": list(self.selected_skill_support_trace_ids),
            "selected_skill_support_plan_proposal_ids": list(
                self.selected_skill_support_plan_proposal_ids
            ),
            "rejected_skills": [item.as_dict() for item in self.rejected_skills],
            "delta": self.delta.as_dict() if self.delta is not None else None,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanningDraft | None":
        """Hydrate one draft from strict JSON."""
        if not isinstance(data, dict):
            return None
        summary = str(data.get("summary", "")).strip()
        if not summary:
            return None
        remaining_steps: list[BrainGoalStep] = []
        for item in data.get("remaining_steps", []):
            if not isinstance(item, dict):
                continue
            step = BrainGoalStep.from_dict(item)
            if step.capability_id:
                remaining_steps.append(step)
        return cls(
            summary=summary,
            remaining_steps=remaining_steps,
            assumptions=[str(item).strip() for item in data.get("assumptions", []) if str(item).strip()],
            missing_inputs=[
                str(item).strip() for item in data.get("missing_inputs", []) if str(item).strip()
            ],
            review_policy=(str(data.get("review_policy", "")).strip() or None),
            procedural_origin=(
                str(
                    data.get(
                        "procedural_origin",
                        BrainPlanningProceduralOrigin.FRESH_DRAFT.value,
                    )
                ).strip()
                or BrainPlanningProceduralOrigin.FRESH_DRAFT.value
            ),
            selected_skill_id=_optional_text(data.get("selected_skill_id")),
            selected_skill_support_trace_ids=[
                str(item).strip()
                for item in data.get("selected_skill_support_trace_ids", [])
                if str(item).strip()
            ],
            selected_skill_support_plan_proposal_ids=[
                str(item).strip()
                for item in data.get("selected_skill_support_plan_proposal_ids", [])
                if str(item).strip()
            ],
            rejected_skills=[
                record
                for item in data.get("rejected_skills", [])
                if (record := BrainPlanningSkillRejection.from_dict(item)) is not None
            ],
            delta=BrainPlanningSkillDelta.from_dict(data.get("delta")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainPlanningRequest:
    """Typed bounded-planning request assembled by the executive."""

    request_kind: str
    goal: BrainGoal
    commitment: BrainCommitmentRecord | None = None
    completed_prefix: list[BrainGoalStep] = field(default_factory=list)
    supersedes_plan_proposal_id: str | None = None
    skill_candidates: list[BrainPlanningSkillCandidate] = field(default_factory=list)
    executive_policy: BrainExecutivePolicyFrame | None = None


@dataclass(frozen=True)
class BrainPlanningCoordinatorResult:
    """Decision returned by the bounded planning coordinator."""

    progressed: bool
    outcome: str
    proposal: BrainPlanProposal
    decision: BrainPlanProposalDecision


BrainPlanningCallback = Callable[[BrainPlanningRequest], Awaitable[BrainPlanningDraft | None]]


class BrainPlanningCoordinator:
    """Validate bounded plan proposals against the registered capability surface."""

    _ALLOWED_FAMILIES = {
        CapabilityFamily.OBSERVATION.value,
        CapabilityFamily.REPORTING.value,
        CapabilityFamily.MAINTENANCE.value,
        CapabilityFamily.DIALOGUE.value,
        CapabilityFamily.ROBOT_HEAD.value,
    }
    _AUTO_ADOPT_FAMILIES = {
        CapabilityFamily.OBSERVATION.value,
        CapabilityFamily.REPORTING.value,
        CapabilityFamily.MAINTENANCE.value,
    }
    _OPERATOR_REVIEW_FAMILIES = {
        CapabilityFamily.DIALOGUE.value,
        CapabilityFamily.ROBOT_HEAD.value,
    }

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        planning_callback: BrainPlanningCallback | None = None,
    ):
        """Initialize the provider-light coordinator."""
        self._registry = registry
        self._planning_callback = planning_callback

    async def request_proposal(
        self,
        request: BrainPlanningRequest,
    ) -> BrainPlanningCoordinatorResult:
        """Build one bounded planning result from the injected callback draft."""
        draft = await self._invoke_callback(request)
        if draft is None:
            return self._rejected_result(
                request=request,
                summary="No bounded plan proposal was available.",
                reason="no_bounded_plan_available",
                details={"request_kind": request.request_kind},
            )

        if not draft.remaining_steps:
            return self._rejected_result(
                request=request,
                summary="The bounded planner returned no valid remaining steps.",
                reason="no_bounded_plan_available",
                details={"request_kind": request.request_kind},
            )

        procedural_details, procedural_rejection = self._procedural_details_for_draft(
            request=request,
            draft=draft,
        )
        if procedural_rejection is not None:
            return procedural_rejection

        invalid_capabilities: list[str] = []
        unsupported_families: list[str] = []
        referenced_families: list[str] = []
        for step in draft.remaining_steps:
            try:
                definition = self._registry.get(step.capability_id)
            except KeyError:
                invalid_capabilities.append(step.capability_id)
                continue
            family = str(definition.family).strip()
            if family not in referenced_families:
                referenced_families.append(family)
            if family not in self._ALLOWED_FAMILIES and family not in unsupported_families:
                unsupported_families.append(family)

        if invalid_capabilities:
            return self._rejected_result(
                request=request,
                summary="The bounded planner referenced unsupported capabilities.",
                reason="unsupported_capability",
                details={
                    "invalid_capability_ids": invalid_capabilities,
                    "request_kind": request.request_kind,
                    "procedural": procedural_details,
                },
            )

        if unsupported_families:
            return self._rejected_result(
                request=request,
                summary="The bounded planner referenced out-of-policy capability families.",
                reason="unsupported_capability_family",
                details={
                    "unsupported_capability_families": unsupported_families,
                    "request_kind": request.request_kind,
                    "procedural": procedural_details,
                },
            )

        effective_review_policy = self._resolve_review_policy(
            requested_review_policy=draft.review_policy,
            missing_inputs=draft.missing_inputs,
            families=referenced_families,
        )
        current_plan_revision = (
            request.commitment.plan_revision
            if request.commitment is not None
            else request.goal.plan_revision
        )
        plan_revision = (
            current_plan_revision
            if request.request_kind == BrainPlanningRequestKind.INITIAL_PLAN.value
            else current_plan_revision + 1
        )
        full_steps = [BrainGoalStep.from_dict(step.as_dict()) for step in request.completed_prefix]
        full_steps.extend(BrainGoalStep.from_dict(step.as_dict()) for step in draft.remaining_steps)
        proposal = BrainPlanProposal(
            plan_proposal_id=f"plan-proposal-{uuid4()}",
            goal_id=request.goal.goal_id,
            commitment_id=(
                request.commitment.commitment_id
                if request.commitment is not None
                else request.goal.commitment_id
            ),
            source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
            summary=draft.summary,
            current_plan_revision=current_plan_revision,
            plan_revision=plan_revision,
            review_policy=effective_review_policy,
            steps=full_steps,
            preserved_prefix_count=len(request.completed_prefix),
            assumptions=list(draft.assumptions),
            missing_inputs=list(draft.missing_inputs),
            supersedes_plan_proposal_id=request.supersedes_plan_proposal_id,
            details={
                **dict(draft.details),
                "request_kind": request.request_kind,
                "capability_families": referenced_families,
                "remaining_step_count": len(draft.remaining_steps),
                "requested_review_policy": draft.review_policy,
                "procedural": procedural_details,
            },
            created_at=_utc_now(),
        )

        if effective_review_policy == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
            return apply_executive_policy_to_planning_result(
                result=BrainPlanningCoordinatorResult(
                progressed=True,
                outcome=BrainPlanningOutcome.NEEDS_USER_REVIEW.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary="The bounded plan is waiting for user-owned inputs.",
                    reason="waiting_user_input",
                    details={
                        "missing_inputs": list(draft.missing_inputs),
                        "request_kind": request.request_kind,
                        "procedural_origin": procedural_details["origin"],
                        "selected_skill_id": procedural_details.get("selected_skill_id"),
                        "delta_operation_count": int(
                            ((procedural_details.get("delta") or {}).get("operation_count") or 0)
                        ),
                    },
                ),
                ),
                executive_policy=request.executive_policy,
            )
        if effective_review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value:
            return apply_executive_policy_to_planning_result(
                result=BrainPlanningCoordinatorResult(
                progressed=True,
                outcome=BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary="The bounded plan requires operator review before adoption.",
                    reason="operator_review_required",
                    details={
                        "capability_families": referenced_families,
                        "request_kind": request.request_kind,
                        "procedural_origin": procedural_details["origin"],
                        "selected_skill_id": procedural_details.get("selected_skill_id"),
                        "delta_operation_count": int(
                            ((procedural_details.get("delta") or {}).get("operation_count") or 0)
                        ),
                    },
                ),
                ),
                executive_policy=request.executive_policy,
            )
        return apply_executive_policy_to_planning_result(
            result=BrainPlanningCoordinatorResult(
                progressed=True,
                outcome=BrainPlanningOutcome.AUTO_ADOPTED.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary="The bounded plan is safe to adopt automatically.",
                    reason="bounded_plan_available",
                    details={
                        "capability_families": referenced_families,
                        "request_kind": request.request_kind,
                        "procedural_origin": procedural_details["origin"],
                        "selected_skill_id": procedural_details.get("selected_skill_id"),
                        "delta_operation_count": int(
                            ((procedural_details.get("delta") or {}).get("operation_count") or 0)
                        ),
                    },
                ),
            )
            ,
            executive_policy=request.executive_policy,
        )

    async def _invoke_callback(
        self,
        request: BrainPlanningRequest,
    ) -> BrainPlanningDraft | None:
        if self._planning_callback is None:
            return None
        return await self._planning_callback(request)

    def _procedural_details_for_draft(
        self,
        *,
        request: BrainPlanningRequest,
        draft: BrainPlanningDraft,
    ) -> tuple[dict[str, Any], BrainPlanningCoordinatorResult | None]:
        candidate_map = {
            candidate.skill_id: candidate
            for candidate in request.skill_candidates
            if _optional_text(candidate.skill_id) is not None
        }
        procedural_origin = str(
            draft.procedural_origin or BrainPlanningProceduralOrigin.FRESH_DRAFT.value
        ).strip() or BrainPlanningProceduralOrigin.FRESH_DRAFT.value
        if procedural_origin not in {
            BrainPlanningProceduralOrigin.FRESH_DRAFT.value,
            BrainPlanningProceduralOrigin.SKILL_REUSE.value,
            BrainPlanningProceduralOrigin.SKILL_DELTA.value,
        }:
            procedural_origin = BrainPlanningProceduralOrigin.FRESH_DRAFT.value
        selected_candidate = candidate_map.get(draft.selected_skill_id or "")
        procedural_details = {
            "origin": procedural_origin,
            "selected_skill_id": selected_candidate.skill_id if selected_candidate is not None else draft.selected_skill_id,
            "selected_skill_status": (
                selected_candidate.skill_status if selected_candidate is not None else None
            ),
            "selected_skill_confidence": (
                selected_candidate.confidence if selected_candidate is not None else None
            ),
            "selected_skill_support_trace_ids": list(
                selected_candidate.support_trace_ids
                if selected_candidate is not None
                else draft.selected_skill_support_trace_ids
            ),
            "selected_skill_support_plan_proposal_ids": list(
                selected_candidate.support_plan_proposal_ids
                if selected_candidate is not None
                else draft.selected_skill_support_plan_proposal_ids
            ),
            "rejected_skills": [item.as_dict() for item in draft.rejected_skills],
            "delta": draft.delta.as_dict() if draft.delta is not None else None,
            "retrieved_skill_candidates": [candidate.as_dict() for candidate in request.skill_candidates],
            "policy": (
                {
                    "action_posture": request.executive_policy.action_posture,
                    "approval_requirement": request.executive_policy.approval_requirement,
                    "procedural_reuse_eligibility": (
                        request.executive_policy.procedural_reuse_eligibility
                    ),
                    "reason_codes": list(request.executive_policy.reason_codes),
                    "effect": request.executive_policy.procedural_reuse_eligibility,
                }
                if request.executive_policy is not None
                else None
            ),
        }
        if procedural_origin == BrainPlanningProceduralOrigin.FRESH_DRAFT.value:
            return procedural_details, None
        if selected_candidate is None:
            return procedural_details, self._procedural_rejected_result(
                request=request,
                summary="The bounded planner selected an unknown procedural skill.",
                reason="unknown_skill_candidate",
                procedural_details=procedural_details,
            )
        if (
            selected_candidate.eligibility == BrainPlanningSkillEligibility.ADVISORY.value
            and _policy_allows_skill_linked_hold(request.executive_policy)
        ):
            procedural_details["policy_selected_skill_hold"] = True
            procedural_details["policy_selected_skill_reason"] = selected_candidate.reason
        elif selected_candidate.eligibility != BrainPlanningSkillEligibility.REUSABLE.value:
            return procedural_details, self._procedural_rejected_result(
                request=request,
                summary="The bounded planner selected a procedural skill that is not reusable.",
                reason="skill_not_reusable",
                procedural_details=procedural_details,
            )
        if is_review_policy_weaker(draft.review_policy, selected_candidate.review_policy):
            return procedural_details, self._procedural_rejected_result(
                request=request,
                summary="The bounded planner weakened the review policy required by the selected skill.",
                reason="skill_not_reusable",
                procedural_details=procedural_details,
            )
        selected_skill_sequence = tuple(selected_candidate.required_capability_ids)
        completed_prefix = tuple(
            step.capability_id
            for step in request.completed_prefix
            if _optional_text(step.capability_id) is not None
        )
        if (
            request.request_kind == BrainPlanningRequestKind.REVISE_TAIL.value
            and selected_skill_sequence[: len(completed_prefix)] != completed_prefix
        ):
            return procedural_details, self._procedural_rejected_result(
                request=request,
                summary="The selected procedural skill does not preserve the completed prefix.",
                reason="completed_prefix_mismatch",
                procedural_details=procedural_details,
            )
        reference_tail = (
            selected_skill_sequence[len(completed_prefix) :]
            if request.request_kind == BrainPlanningRequestKind.REVISE_TAIL.value
            else selected_skill_sequence
        )
        planned_tail = tuple(
            step.capability_id
            for step in draft.remaining_steps
            if _optional_text(step.capability_id) is not None
        )
        if procedural_origin == BrainPlanningProceduralOrigin.SKILL_REUSE.value:
            if planned_tail != reference_tail:
                return procedural_details, self._procedural_rejected_result(
                    request=request,
                    summary="The bounded planner claimed exact skill reuse, but the returned steps do not match the selected skill.",
                    reason="skill_reuse_mismatch",
                    procedural_details=procedural_details,
                )
            procedural_details["delta"] = None
            return procedural_details, None
        delta = build_planning_skill_delta(
            selected_skill_capability_ids=reference_tail,
            planned_capability_ids=planned_tail,
            preserved_prefix_count=len(completed_prefix),
        )
        procedural_details["delta"] = delta.as_dict()
        if delta.operation_count < 1 or delta.operation_count > 2:
            return procedural_details, self._procedural_rejected_result(
                request=request,
                summary="The bounded planner adapted the selected skill beyond the allowed bounded delta.",
                reason="skill_delta_out_of_bounds",
                procedural_details=procedural_details,
            )
        return procedural_details, None

    def _procedural_rejected_result(
        self,
        *,
        request: BrainPlanningRequest,
        summary: str,
        reason: str,
        procedural_details: dict[str, Any],
    ) -> BrainPlanningCoordinatorResult:
        return self._rejected_result(
            request=request,
            summary=summary,
            reason=reason,
            details={
                "request_kind": request.request_kind,
                "procedural": procedural_details,
            },
        )

    def _resolve_review_policy(
        self,
        *,
        requested_review_policy: str | None,
        missing_inputs: list[str],
        families: list[str],
    ) -> str:
        requested = (requested_review_policy or "").strip()
        if missing_inputs or requested == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
            return BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value
        if (
            requested == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
            or any(family in self._OPERATOR_REVIEW_FAMILIES for family in families)
        ):
            return BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
        return BrainPlanReviewPolicy.AUTO_ADOPT_OK.value

    def _rejected_result(
        self,
        *,
        request: BrainPlanningRequest,
        summary: str,
        reason: str,
        details: dict[str, Any],
    ) -> BrainPlanningCoordinatorResult:
        current_plan_revision = (
            request.commitment.plan_revision
            if request.commitment is not None
            else request.goal.plan_revision
        )
        proposal = BrainPlanProposal(
            plan_proposal_id=f"plan-proposal-{uuid4()}",
            goal_id=request.goal.goal_id,
            commitment_id=(
                request.commitment.commitment_id
                if request.commitment is not None
                else request.goal.commitment_id
            ),
            source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
            summary=summary,
            current_plan_revision=current_plan_revision,
            plan_revision=(
                current_plan_revision
                if request.request_kind == BrainPlanningRequestKind.INITIAL_PLAN.value
                else current_plan_revision + 1
            ),
            review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
            steps=[BrainGoalStep.from_dict(step.as_dict()) for step in request.completed_prefix],
            preserved_prefix_count=len(request.completed_prefix),
            assumptions=[],
            missing_inputs=[],
            supersedes_plan_proposal_id=request.supersedes_plan_proposal_id,
            details=dict(details),
            created_at=_utc_now(),
        )
        return apply_executive_policy_to_planning_result(
            result=BrainPlanningCoordinatorResult(
                progressed=True,
                outcome=BrainPlanningOutcome.REJECTED.value,
                proposal=proposal,
                decision=BrainPlanProposalDecision(
                    summary=summary,
                    reason=reason,
                    details=dict(details),
                ),
            ),
            executive_policy=request.executive_policy,
        )

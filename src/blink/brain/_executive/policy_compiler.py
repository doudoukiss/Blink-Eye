"""Compile one bounded executive-policy surface from the current situation state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blink.brain._executive.policy import (
    BrainExecutivePolicyActionPosture,
    BrainExecutivePolicyApprovalRequirement,
    BrainExecutivePolicyConservatism,
    BrainExecutivePolicyFrame,
    BrainExecutiveProceduralReuseEligibility,
)
from blink.brain.context.policy import BrainContextTask
from blink.brain.core.projections import (
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainClaimCurrentnessStatus,
    BrainPlanReviewPolicy,
)

if TYPE_CHECKING:
    from blink.brain.context_surfaces import BrainContextSurfaceSnapshot

_POLICY_REASON_CODES = {
    "scene_limited",
    "scene_unavailable",
    "dossier_review_debt",
    "dossier_held_support",
    "dossier_contradicted",
    "held_claim_present",
    "stale_claim_present",
    "active_state_unresolved",
    "blocked_commitment_present",
    "deferred_commitment_present",
    "pending_user_review",
    "pending_operator_review",
    "procedural_reuse_advisory_only",
    "procedural_reuse_blocked",
}
_SUPPRESSION_CODES = {
    "scene_unavailable",
    "operator_review_required",
    "user_confirmation_required",
    "governance_hold",
    "uncertainty_conservative",
}


def _append_code(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def compile_executive_policy(
    surface: "BrainContextSurfaceSnapshot",
    task: BrainContextTask,
    reference_ts: str | None = None,
) -> BrainExecutivePolicyFrame:
    """Compile one typed executive-policy frame from an existing context surface."""
    dossier_ids: set[str] = set()
    claim_ids: set[str] = set()
    commitment_ids: set[str] = set()
    plan_proposal_ids: set[str] = set()

    review_debt_count = 0
    held_support_present = False
    dossier_contradicted = False
    if surface.continuity_dossiers is not None:
        for dossier in surface.continuity_dossiers.dossiers:
            governance = dossier.governance
            if int(governance.review_debt_count or 0) > 0:
                review_debt_count += int(governance.review_debt_count or 0)
                dossier_ids.add(dossier.dossier_id)
            if dossier.contradiction == "contradicted":
                dossier_contradicted = True
                dossier_ids.add(dossier.dossier_id)
            if any(issue.kind == "held_support" for issue in dossier.open_issues):
                held_support_present = True
                dossier_ids.add(dossier.dossier_id)

    held_claim_count = 0
    stale_claim_count = 0
    if surface.claim_governance is not None:
        for record in surface.claim_governance.records:
            if record.currentness_status == BrainClaimCurrentnessStatus.HELD.value:
                held_claim_count += 1
                claim_ids.add(record.claim_id)
            elif record.currentness_status == BrainClaimCurrentnessStatus.STALE.value:
                stale_claim_count += 1
                claim_ids.add(record.claim_id)

    unresolved_active_record_ids = [
        record.record_id
        for record in surface.active_situation_model.records
        if record.state == BrainActiveSituationRecordState.UNRESOLVED.value
        and record.record_kind != BrainActiveSituationRecordKind.PREDICTION_STATE.value
    ]
    unresolved_active_state_count = len(unresolved_active_record_ids)

    pending_user_review_keys: set[tuple[str, str]] = set()
    pending_operator_review_keys: set[tuple[str, str]] = set()
    for record in surface.active_situation_model.records:
        if record.state == BrainActiveSituationRecordState.STALE.value:
            continue
        if record.record_kind not in {
            BrainActiveSituationRecordKind.COMMITMENT_STATE.value,
            BrainActiveSituationRecordKind.PLAN_STATE.value,
        }:
            continue
        review_policy = str(record.details.get("review_policy", "")).strip()
        record_key = (
            record.record_kind,
            record.plan_proposal_id or record.commitment_id or record.record_id,
        )
        if review_policy == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value:
            pending_user_review_keys.add(record_key)
            if record.commitment_id:
                commitment_ids.add(record.commitment_id)
            if record.plan_proposal_id:
                plan_proposal_ids.add(record.plan_proposal_id)
        elif review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value:
            pending_operator_review_keys.add(record_key)
            if record.commitment_id:
                commitment_ids.add(record.commitment_id)
            if record.plan_proposal_id:
                plan_proposal_ids.add(record.plan_proposal_id)

    blocked_commitment_ids = {
        record.commitment_id for record in surface.commitment_projection.blocked_commitments
    }
    deferred_commitment_ids = {
        record.commitment_id for record in surface.commitment_projection.deferred_commitments
    }
    commitment_ids.update(blocked_commitment_ids)
    commitment_ids.update(deferred_commitment_ids)

    scene_degraded_mode = str(surface.scene_world_state.degraded_mode or "healthy").strip() or "healthy"

    reason_codes: list[str] = []
    if scene_degraded_mode == "limited":
        _append_code(reason_codes, "scene_limited")
    elif scene_degraded_mode == "unavailable":
        _append_code(reason_codes, "scene_unavailable")
    if review_debt_count > 0:
        _append_code(reason_codes, "dossier_review_debt")
    if held_support_present:
        _append_code(reason_codes, "dossier_held_support")
    if dossier_contradicted:
        _append_code(reason_codes, "dossier_contradicted")
    if held_claim_count > 0:
        _append_code(reason_codes, "held_claim_present")
    if stale_claim_count > 0:
        _append_code(reason_codes, "stale_claim_present")
    if unresolved_active_state_count > 0:
        _append_code(reason_codes, "active_state_unresolved")
    if blocked_commitment_ids:
        _append_code(reason_codes, "blocked_commitment_present")
    if deferred_commitment_ids:
        _append_code(reason_codes, "deferred_commitment_present")
    if pending_user_review_keys:
        _append_code(reason_codes, "pending_user_review")
    if pending_operator_review_keys:
        _append_code(reason_codes, "pending_operator_review")

    if scene_degraded_mode == "unavailable" or pending_operator_review_keys:
        action_posture = BrainExecutivePolicyActionPosture.SUPPRESS.value
    elif any(
        (
            review_debt_count > 0,
            held_claim_count > 0,
            pending_user_review_keys,
            scene_degraded_mode == "limited",
            stale_claim_count > 0,
            unresolved_active_state_count > 0,
            blocked_commitment_ids,
            deferred_commitment_ids,
        )
    ):
        action_posture = BrainExecutivePolicyActionPosture.DEFER.value
    else:
        action_posture = BrainExecutivePolicyActionPosture.ALLOW.value

    if pending_operator_review_keys:
        approval_requirement = BrainExecutivePolicyApprovalRequirement.OPERATOR_REVIEW.value
    elif pending_user_review_keys or review_debt_count > 0 or held_claim_count > 0:
        approval_requirement = BrainExecutivePolicyApprovalRequirement.USER_CONFIRMATION.value
    else:
        approval_requirement = BrainExecutivePolicyApprovalRequirement.NONE.value

    if action_posture == BrainExecutivePolicyActionPosture.SUPPRESS.value:
        conservatism = BrainExecutivePolicyConservatism.HIGH.value
        procedural_reuse_eligibility = BrainExecutiveProceduralReuseEligibility.BLOCKED.value
        _append_code(reason_codes, "procedural_reuse_blocked")
    elif action_posture == BrainExecutivePolicyActionPosture.DEFER.value:
        conservatism = BrainExecutivePolicyConservatism.ELEVATED.value
        procedural_reuse_eligibility = BrainExecutiveProceduralReuseEligibility.ADVISORY_ONLY.value
        _append_code(reason_codes, "procedural_reuse_advisory_only")
    else:
        conservatism = BrainExecutivePolicyConservatism.NORMAL.value
        procedural_reuse_eligibility = BrainExecutiveProceduralReuseEligibility.ALLOWED.value

    suppression_codes: list[str] = []
    if scene_degraded_mode == "unavailable":
        _append_code(suppression_codes, "scene_unavailable")
    if pending_operator_review_keys:
        _append_code(suppression_codes, "operator_review_required")
    if approval_requirement == BrainExecutivePolicyApprovalRequirement.USER_CONFIRMATION.value:
        _append_code(suppression_codes, "user_confirmation_required")
    if review_debt_count > 0 or held_claim_count > 0:
        _append_code(suppression_codes, "governance_hold")
    if any(
        (
            scene_degraded_mode == "limited",
            stale_claim_count > 0,
            unresolved_active_state_count > 0,
            blocked_commitment_ids,
            deferred_commitment_ids,
        )
    ):
        _append_code(suppression_codes, "uncertainty_conservative")

    frame = BrainExecutivePolicyFrame(
        task=task.value,
        action_posture=action_posture,
        conservatism=conservatism,
        approval_requirement=approval_requirement,
        procedural_reuse_eligibility=procedural_reuse_eligibility,
        reason_codes=_sorted_unique(reason_codes),
        suppression_codes=_sorted_unique(suppression_codes),
        review_debt_count=review_debt_count,
        held_claim_count=held_claim_count,
        stale_claim_count=stale_claim_count,
        unresolved_active_state_count=unresolved_active_state_count,
        blocked_commitment_count=len(blocked_commitment_ids),
        deferred_commitment_count=len(deferred_commitment_ids),
        pending_user_review_count=len(pending_user_review_keys),
        pending_operator_review_count=len(pending_operator_review_keys),
        scene_degraded_mode=scene_degraded_mode,
        source_dossier_ids=sorted(dossier_ids),
        source_active_record_ids=sorted(set(unresolved_active_record_ids)),
        source_claim_ids=sorted(claim_ids),
        source_commitment_ids=sorted(commitment_ids),
        source_plan_proposal_ids=sorted(plan_proposal_ids),
        updated_at=reference_ts or surface.generated_at,
    )
    for code in frame.reason_codes:
        if code not in _POLICY_REASON_CODES:
            raise ValueError(f"Unsupported executive policy reason code: {code}")
    for code in frame.suppression_codes:
        if code not in _SUPPRESSION_CODES:
            raise ValueError(f"Unsupported executive policy suppression code: {code}")
    return frame


__all__ = ["compile_executive_policy"]

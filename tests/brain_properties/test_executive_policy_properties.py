from __future__ import annotations

from dataclasses import replace
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain._executive import (
    BrainExecutivePolicyActionPosture,
    BrainExecutivePolicyApprovalRequirement,
    BrainExecutiveProceduralReuseEligibility,
    BrainPlanningCoordinatorResult,
    BrainPlanningOutcome,
    apply_executive_policy_to_planning_result,
    compile_executive_policy,
)
from blink.brain.context import BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.memory_v2 import (
    BrainContinuityDossierGovernanceRecord,
    BrainContinuityDossierProjection,
    BrainContinuityDossierRecord,
)
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainClaimGovernanceProjection,
    BrainClaimGovernanceRecord,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _surface(
    *,
    scene_degraded_mode: str = "healthy",
    review_debt_count: int = 0,
    held_claim_count: int = 0,
    stale_claim_count: int = 0,
    unresolved_count: int = 0,
    blocked_commitment_count: int = 0,
    deferred_commitment_count: int = 0,
    pending_user_review_count: int = 0,
    pending_operator_review_count: int = 0,
):
    tmpdir = TemporaryDirectory()
    store = BrainStore(path=f"{tmpdir.name}/brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="policy-prop")
    builder = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        capability_registry=None,
    )
    base = builder.build(
        latest_user_text="policy properties",
        task=BrainContextTask.REEVALUATION,
    )
    dossiers = []
    if review_debt_count > 0:
        dossiers.append(
            BrainContinuityDossierRecord(
                dossier_id="dossier-review",
                kind="relationship",
                scope_type="relationship",
                scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
                title="Relationship",
                summary="Review debt is present.",
                status="current",
                freshness="fresh",
                contradiction="clear",
                support_strength=0.8,
                governance=BrainContinuityDossierGovernanceRecord(
                    review_debt_count=review_debt_count,
                    last_refresh_cause="fresh_current_support",
                ),
            )
        )
    dossier_projection = BrainContinuityDossierProjection(
        scope_type="user",
        scope_id=session_ids.user_id,
        dossiers=dossiers,
        dossier_counts={"relationship": len(dossiers)} if dossiers else {},
        freshness_counts={"fresh": len(dossiers)} if dossiers else {},
        contradiction_counts={"clear": len(dossiers)} if dossiers else {},
        current_dossier_ids=[record.dossier_id for record in dossiers],
    )
    claim_records = [
        BrainClaimGovernanceRecord(
            claim_id=f"claim-held-{index}",
            scope_type="user",
            scope_id=session_ids.user_id,
            truth_status="active",
            currentness_status="held",
            review_state="requested",
            retention_class="durable",
            updated_at=base.generated_at,
        )
        for index in range(held_claim_count)
    ] + [
        BrainClaimGovernanceRecord(
            claim_id=f"claim-stale-{index}",
            scope_type="user",
            scope_id=session_ids.user_id,
            truth_status="active",
            currentness_status="stale",
            review_state="none",
            retention_class="durable",
            updated_at=base.generated_at,
        )
        for index in range(stale_claim_count)
    ]
    claim_governance = BrainClaimGovernanceProjection(
        scope_type="user",
        scope_id=session_ids.user_id,
        records=claim_records,
        currentness_counts={
            key: value
            for key, value in {
                "held": held_claim_count,
                "stale": stale_claim_count,
            }.items()
            if value
        },
        review_state_counts={
            key: value
            for key, value in {
                "requested": held_claim_count,
                "none": stale_claim_count,
            }.items()
            if value
        },
        retention_class_counts={"durable": len(claim_records)} if claim_records else {},
        held_claim_ids=[record.claim_id for record in claim_records if record.currentness_status == "held"],
        stale_claim_ids=[record.claim_id for record in claim_records if record.currentness_status == "stale"],
        updated_at=base.generated_at,
    )
    active_records = [
        BrainActiveSituationRecord(
            record_id=f"unresolved-{index}",
            record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
            summary=f"Unresolved state {index}",
            state=BrainActiveSituationRecordState.UNRESOLVED.value,
            evidence_kind="derived",
            updated_at=base.generated_at,
        )
        for index in range(unresolved_count)
    ]
    active_records.extend(
        BrainActiveSituationRecord(
            record_id=f"plan-user-review-{index}",
            record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
            summary=f"Plan needs user review {index}",
            state=BrainActiveSituationRecordState.UNRESOLVED.value,
            evidence_kind="derived",
            plan_proposal_id=f"plan-user-{index}",
            updated_at=base.generated_at,
            details={"review_policy": BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value},
        )
        for index in range(pending_user_review_count)
    )
    active_records.extend(
        BrainActiveSituationRecord(
            record_id=f"plan-operator-review-{index}",
            record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
            summary=f"Plan needs operator review {index}",
            state=BrainActiveSituationRecordState.UNRESOLVED.value,
            evidence_kind="derived",
            plan_proposal_id=f"plan-operator-{index}",
            updated_at=base.generated_at,
            details={"review_policy": BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value},
        )
        for index in range(pending_operator_review_count)
    )
    active_state = BrainActiveSituationProjection(
        scope_type=base.active_situation_model.scope_type,
        scope_id=base.active_situation_model.scope_id,
        records=active_records,
        unresolved_record_ids=[record.record_id for record in active_records],
        linked_plan_proposal_ids=[
            record.plan_proposal_id for record in active_records if record.plan_proposal_id
        ],
        updated_at=base.generated_at,
    )
    commitment_projection = replace(
        base.commitment_projection,
        blocked_commitments=list(base.commitment_projection.blocked_commitments)
        + [
            replace(
                base.commitment_projection.active_commitments[0],
                commitment_id=f"blocked-{index}",
                status="blocked",
            )
            for index in range(blocked_commitment_count)
        ]
        if base.commitment_projection.active_commitments
        else list(base.commitment_projection.blocked_commitments),
        deferred_commitments=list(base.commitment_projection.deferred_commitments)
        + [
            replace(
                base.commitment_projection.active_commitments[0],
                commitment_id=f"deferred-{index}",
                status="deferred",
            )
            for index in range(deferred_commitment_count)
        ]
        if base.commitment_projection.active_commitments
        else list(base.commitment_projection.deferred_commitments),
    )
    scene_world_state = replace(
        base.scene_world_state,
        degraded_mode=scene_degraded_mode,
        degraded_reason_codes=(
            ["camera_disconnected"] if scene_degraded_mode == "unavailable" else ["scene_stale"]
            if scene_degraded_mode == "limited"
            else []
        ),
    )
    snapshot = replace(
        base,
        continuity_dossiers=dossier_projection,
        claim_governance=claim_governance,
        active_situation_model=active_state,
        commitment_projection=commitment_projection,
        scene_world_state=scene_world_state,
    )
    store.close()
    tmpdir.cleanup()
    return snapshot


def _posture_rank(value: str) -> int:
    return {
        BrainExecutivePolicyActionPosture.ALLOW.value: 0,
        BrainExecutivePolicyActionPosture.DEFER.value: 1,
        BrainExecutivePolicyActionPosture.SUPPRESS.value: 2,
    }[value]


def _planning_outcome_rank(value: str) -> int:
    return {
        BrainPlanningOutcome.AUTO_ADOPTED.value: 0,
        BrainPlanningOutcome.NEEDS_USER_REVIEW.value: 1,
        BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value: 2,
        BrainPlanningOutcome.REJECTED.value: 3,
    }[value]


def _planning_result() -> BrainPlanningCoordinatorResult:
    proposal = BrainPlanProposal(
        plan_proposal_id="plan-proposal-policy-prop",
        goal_id="goal-policy-prop",
        commitment_id="commitment-policy-prop",
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary="Safe bounded plan.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id="maintenance.review_memory_health")],
        created_at="2026-04-21T10:00:00+00:00",
    )
    return BrainPlanningCoordinatorResult(
        progressed=True,
        outcome=BrainPlanningOutcome.AUTO_ADOPTED.value,
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary="The bounded plan is safe to adopt automatically.",
            reason="bounded_plan_available",
        ),
    )


@given(
    review_debt_count=st.integers(min_value=0, max_value=2),
    held_claim_count=st.integers(min_value=0, max_value=2),
    stale_claim_count=st.integers(min_value=0, max_value=2),
    unresolved_count=st.integers(min_value=0, max_value=2),
    pending_user_review_count=st.integers(min_value=0, max_value=1),
    pending_operator_review_count=st.integers(min_value=0, max_value=1),
    scene_mode=st.sampled_from(["healthy", "limited", "unavailable"]),
)
@_SETTINGS
def test_executive_policy_compiler_is_deterministic(
    review_debt_count,
    held_claim_count,
    stale_claim_count,
    unresolved_count,
    pending_user_review_count,
    pending_operator_review_count,
    scene_mode,
):
    snapshot = _surface(
        scene_degraded_mode=scene_mode,
        review_debt_count=review_debt_count,
        held_claim_count=held_claim_count,
        stale_claim_count=stale_claim_count,
        unresolved_count=unresolved_count,
        pending_user_review_count=pending_user_review_count,
        pending_operator_review_count=pending_operator_review_count,
    )

    first = compile_executive_policy(
        snapshot,
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    second = compile_executive_policy(
        snapshot,
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    assert first == second


@given(
    review_debt_count=st.integers(min_value=0, max_value=1),
    held_claim_count=st.integers(min_value=0, max_value=1),
    stale_claim_count=st.integers(min_value=0, max_value=1),
    unresolved_count=st.integers(min_value=0, max_value=1),
    blocked_commitment_count=st.integers(min_value=0, max_value=1),
    deferred_commitment_count=st.integers(min_value=0, max_value=1),
    pending_user_review_count=st.integers(min_value=0, max_value=1),
    scene_mode=st.sampled_from(["healthy", "limited"]),
)
@_SETTINGS
def test_stronger_policy_inputs_never_reduce_conservatism(
    review_debt_count,
    held_claim_count,
    stale_claim_count,
    unresolved_count,
    blocked_commitment_count,
    deferred_commitment_count,
    pending_user_review_count,
    scene_mode,
):
    base = compile_executive_policy(
        _surface(
            scene_degraded_mode=scene_mode,
            review_debt_count=review_debt_count,
            held_claim_count=held_claim_count,
            stale_claim_count=stale_claim_count,
            unresolved_count=unresolved_count,
            blocked_commitment_count=blocked_commitment_count,
            deferred_commitment_count=deferred_commitment_count,
            pending_user_review_count=pending_user_review_count,
        ),
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    stronger = compile_executive_policy(
        _surface(
            scene_degraded_mode="unavailable" if scene_mode != "unavailable" else scene_mode,
            review_debt_count=max(1, review_debt_count),
            held_claim_count=max(1, held_claim_count),
            stale_claim_count=max(1, stale_claim_count),
            unresolved_count=max(1, unresolved_count),
            blocked_commitment_count=max(1, blocked_commitment_count),
            deferred_commitment_count=max(1, deferred_commitment_count),
            pending_user_review_count=max(1, pending_user_review_count),
            pending_operator_review_count=1,
        ),
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    assert _posture_rank(stronger.action_posture) >= _posture_rank(base.action_posture)


@given(pending_operator_review_count=st.integers(min_value=1, max_value=2))
@_SETTINGS
def test_operator_review_never_resolves_to_none(pending_operator_review_count):
    policy = compile_executive_policy(
        _surface(pending_operator_review_count=pending_operator_review_count),
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    assert policy.approval_requirement != BrainExecutivePolicyApprovalRequirement.NONE.value


@given(
    scene_mode=st.sampled_from(["unavailable"]),
    pending_operator_review_count=st.integers(min_value=0, max_value=1),
)
@_SETTINGS
def test_suppress_posture_blocks_procedural_reuse(scene_mode, pending_operator_review_count):
    policy = compile_executive_policy(
        _surface(
            scene_degraded_mode=scene_mode,
            pending_operator_review_count=pending_operator_review_count,
        ),
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    assert policy.action_posture == BrainExecutivePolicyActionPosture.SUPPRESS.value
    assert (
        policy.procedural_reuse_eligibility
        == BrainExecutiveProceduralReuseEligibility.BLOCKED.value
    )


def test_neutral_policy_keeps_planning_auto_adopt_outcome():
    policy = compile_executive_policy(
        _surface(),
        task=BrainContextTask.PLANNING,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    result = apply_executive_policy_to_planning_result(
        result=_planning_result(),
        executive_policy=policy,
    )

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert result.proposal.review_policy == BrainPlanReviewPolicy.AUTO_ADOPT_OK.value


def test_stronger_policy_never_upgrades_planning_outcome():
    neutral = compile_executive_policy(
        _surface(),
        task=BrainContextTask.PLANNING,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    stronger = compile_executive_policy(
        _surface(
            scene_degraded_mode="limited",
            pending_user_review_count=1,
        ),
        task=BrainContextTask.PLANNING,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    neutral_result = apply_executive_policy_to_planning_result(
        result=_planning_result(),
        executive_policy=neutral,
    )
    stronger_result = apply_executive_policy_to_planning_result(
        result=_planning_result(),
        executive_policy=stronger,
    )

    assert _planning_outcome_rank(stronger_result.outcome) >= _planning_outcome_rank(
        neutral_result.outcome
    )


def test_suppress_posture_never_allows_planning_auto_adopt():
    policy = compile_executive_policy(
        _surface(scene_degraded_mode="unavailable"),
        task=BrainContextTask.PLANNING,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    result = apply_executive_policy_to_planning_result(
        result=_planning_result(),
        executive_policy=policy,
    )

    assert result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    assert result.proposal.review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value

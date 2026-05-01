import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from blink.brain._executive import (
    BrainExecutivePolicyActionPosture,
    BrainExecutivePolicyApprovalRequirement,
    BrainExecutiveProceduralReuseEligibility,
    compile_executive_policy,
)
from blink.brain.actions import build_brain_capability_registry
from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainAutonomyLedgerEntry,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context import BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
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
    BrainGoalFamily,
    BrainPlanReviewPolicy,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


def _iso_now(*, offset_seconds: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_seconds)).isoformat()


def _candidate(
    candidate_goal_id: str,
    summary: str,
    **overrides,
) -> BrainCandidateGoal:
    return BrainCandidateGoal(
        candidate_goal_id=candidate_goal_id,
        candidate_type=overrides.pop("candidate_type", "presence_acknowledgement"),
        source=overrides.pop("source", BrainCandidateGoalSource.PERCEPTION.value),
        summary=summary,
        goal_family=overrides.pop("goal_family", BrainGoalFamily.ENVIRONMENT.value),
        urgency=overrides.pop("urgency", 0.7),
        confidence=overrides.pop("confidence", 0.9),
        initiative_class=overrides.pop(
            "initiative_class",
            BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        cooldown_key=overrides.pop("cooldown_key", f"cooldown:{candidate_goal_id}"),
        dedupe_key=overrides.pop("dedupe_key", f"dedupe:{candidate_goal_id}"),
        policy_tags=overrides.pop("policy_tags", ["phase1"]),
        requires_user_turn_gap=overrides.pop("requires_user_turn_gap", True),
        expires_at=overrides.pop("expires_at", _iso_now(offset_seconds=300)),
        payload=overrides.pop("payload", {"kind": "presence"}),
        created_at=overrides.pop("created_at", _iso_now(offset_seconds=-30)),
        **overrides,
    )


def _executive(tmp_path, session_ids):
    store = BrainStore(path=tmp_path / "brain.db")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    return store, executive


def _append_candidate(store: BrainStore, session_ids, candidate: BrainCandidateGoal):
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )


class _StaticSurfaceBuilder:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def build(self, **_kwargs):
        return self._snapshot


def _policy_surface_for_store(
    *,
    store: BrainStore,
    session_ids,
    scene_degraded_mode: str = "healthy",
    review_debt_count: int = 0,
    held_claim_ids: tuple[str, ...] = (),
    unresolved_count: int = 0,
    pending_user_review_count: int = 0,
    pending_operator_review_count: int = 0,
):
    builder = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        capability_registry=CapabilityRegistry(),
    )
    base = builder.build(
        latest_user_text="policy surface",
        task=BrainContextTask.REEVALUATION,
    )
    dossiers = []
    if review_debt_count > 0:
        dossiers.append(
            BrainContinuityDossierRecord(
                dossier_id="dossier-review-debt",
                kind="relationship",
                scope_type="relationship",
                scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
                title="Relationship",
                summary="Relationship continuity needs review.",
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
        scope_type=base.continuity_dossiers.scope_type if base.continuity_dossiers is not None else "user",
        scope_id=base.continuity_dossiers.scope_id if base.continuity_dossiers is not None else session_ids.user_id,
        dossiers=dossiers,
        dossier_counts={"relationship": len(dossiers)} if dossiers else {},
        freshness_counts={"fresh": len(dossiers)} if dossiers else {},
        contradiction_counts={"clear": len(dossiers)} if dossiers else {},
        current_dossier_ids=[record.dossier_id for record in dossiers],
    )
    claim_records = [
        BrainClaimGovernanceRecord(
            claim_id=claim_id,
            scope_type="user",
            scope_id=session_ids.user_id,
            truth_status="active",
            currentness_status="held",
            review_state="requested",
            retention_class="durable",
            updated_at=base.generated_at,
        )
        for claim_id in held_claim_ids
    ]
    claim_governance = BrainClaimGovernanceProjection(
        scope_type="user",
        scope_id=session_ids.user_id,
        records=claim_records,
        currentness_counts={"held": len(claim_records)} if claim_records else {},
        review_state_counts={"requested": len(claim_records)} if claim_records else {},
        retention_class_counts={"durable": len(claim_records)} if claim_records else {},
        held_claim_ids=[record.claim_id for record in claim_records],
        updated_at=base.generated_at,
    )
    active_records = [
        BrainActiveSituationRecord(
            record_id=f"active-unresolved-{index}",
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
            summary=f"Plan requires user review {index}",
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
            summary=f"Plan requires operator review {index}",
            state=BrainActiveSituationRecordState.UNRESOLVED.value,
            evidence_kind="derived",
            plan_proposal_id=f"plan-operator-{index}",
            updated_at=base.generated_at,
            details={"review_policy": BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value},
        )
        for index in range(pending_operator_review_count)
    )
    active_situation_model = BrainActiveSituationProjection(
        scope_type=base.active_situation_model.scope_type,
        scope_id=base.active_situation_model.scope_id,
        records=active_records,
        unresolved_record_ids=[record.record_id for record in active_records],
        linked_plan_proposal_ids=[
            record.plan_proposal_id for record in active_records if record.plan_proposal_id
        ],
        updated_at=base.generated_at,
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
    return replace(
        base,
        continuity_dossiers=dossier_projection,
        claim_governance=claim_governance,
        active_situation_model=active_situation_model,
        scene_world_state=scene_world_state,
    )


def test_candidate_goal_roundtrip():
    candidate = _candidate(
        "candidate-1",
        "Person re-entered frame.",
        expected_reevaluation_condition="after user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"turn": "user"},
    )

    restored = BrainCandidateGoal.from_dict(json.loads(json.dumps(candidate.as_dict())))

    assert restored == candidate


def test_autonomy_ledger_entry_roundtrip():
    entry = BrainAutonomyLedgerEntry(
        event_id="event-1",
        event_type=BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
        decision_kind=BrainAutonomyDecisionKind.NON_ACTION.value,
        candidate_goal_id="candidate-1",
        summary="Wait until the user turn ends.",
        reason="user_turn_open",
        reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
        expected_reevaluation_condition="after user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"turn": "user"},
        ts="2026-04-19T10:00:00+00:00",
    )

    restored = BrainAutonomyLedgerEntry.from_dict(json.loads(json.dumps(entry.as_dict())))

    assert restored == entry


def test_autonomy_ledger_tracks_candidate_lifecycle_without_touching_agenda(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")

    suppressed = _candidate("candidate-suppressed", "Suppressed candidate")
    merge_target = _candidate("candidate-target", "Merge target")
    merge_source = _candidate("candidate-source", "Merge source")
    accepted = _candidate("candidate-accepted", "Accepted candidate")
    expired = _candidate("candidate-expired", "Expired candidate")
    non_action = _candidate("candidate-non-action", "Candidate kept after non-action")

    for candidate in (suppressed, merge_target, merge_source, accepted, expired, non_action):
        store.append_candidate_goal_created(
            candidate_goal=candidate,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
        )

    store.append_candidate_goal_suppressed(
        candidate_goal_id=suppressed.candidate_goal_id,
        reason="duplicate_candidate",
        reason_details={"dedupe_key": suppressed.dedupe_key},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_merged(
        candidate_goal_id=merge_source.candidate_goal_id,
        merged_into_candidate_goal_id=merge_target.candidate_goal_id,
        reason="same_social_slot",
        reason_details={"slot": "attention"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_accepted(
        candidate_goal_id=accepted.candidate_goal_id,
        goal_id="goal-accepted",
        reason="accepted_for_dispatch",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_expired(
        candidate_goal_id=expired.candidate_goal_id,
        reason="stale_window_elapsed",
        reason_details={"cooldown_key": expired.cooldown_key},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id=non_action.candidate_goal_id,
        reason="user_currently_speaking",
        reason_details={"gate": "user_turn"},
        expected_reevaluation_condition="after user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"turn": "user"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )

    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    agenda_before_goal_create = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    )

    assert agenda_before_goal_create.goal("goal-accepted") is None
    assert [candidate.candidate_goal_id for candidate in ledger.current_candidates] == [
        merge_target.candidate_goal_id,
        non_action.candidate_goal_id,
    ]
    merged_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.MERGED.value
    )
    assert merged_entry.merged_into_candidate_goal_id == merge_target.candidate_goal_id
    accepted_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.ACCEPTED.value
    )
    assert accepted_entry.accepted_goal_id == "goal-accepted"
    non_action_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )
    assert non_action_entry.expected_reevaluation_condition == "after user turn ends"
    assert (
        non_action_entry.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )
    assert non_action_entry.expected_reevaluation_condition_details == {"turn": "user"}
    current_candidate = ledger.candidate(non_action.candidate_goal_id)
    assert current_candidate is not None
    assert current_candidate.expected_reevaluation_condition == "after user turn ends"
    assert (
        current_candidate.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )
    assert current_candidate.expected_reevaluation_condition_details == {"turn": "user"}

    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": {
                "goal_id": "goal-accepted",
                "title": "Shift attention silently",
                "intent": "presence_acknowledgement",
                "source": "test",
                "goal_family": BrainGoalFamily.ENVIRONMENT.value,
                "status": "open",
                "details": {},
                "steps": [],
                "active_step_index": None,
                "recovery_count": 0,
                "planning_requested": False,
                "blocked_reason": None,
                "wake_conditions": [],
                "plan_revision": 1,
                "resume_count": 0,
                "last_summary": None,
                "last_error": None,
                "created_at": "2026-04-18T12:10:00+00:00",
                "updated_at": "2026-04-18T12:10:00+00:00",
            }
        },
    )

    agenda_after_goal_create = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    )
    assert agenda_after_goal_create.goal("goal-accepted") is not None


def test_rebuild_brain_projections_preserves_autonomy_ledger(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    candidate = _candidate("candidate-rebuild", "Candidate survives rebuild")

    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id=candidate.candidate_goal_id,
        reason="reply_in_progress",
        reason_details={"gate": "assistant_turn"},
        expected_reevaluation_condition="after assistant turn ends",
        expected_reevaluation_condition_kind=(
            BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value
        ),
        expected_reevaluation_condition_details={"turn": "assistant"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    before = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()

    store.rebuild_brain_projections()

    after = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()
    assert after == before


def test_presence_director_merges_duplicates_and_records_non_action(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    canonical = _candidate(
        "candidate-canonical",
        "Keep the strongest presence candidate.",
        dedupe_key="dedupe:presence-slot",
        cooldown_key="cooldown:presence-slot",
        urgency=0.9,
        created_at=_iso_now(offset_seconds=-20),
    )
    merged = _candidate(
        "candidate-merged",
        "Merge this weaker duplicate.",
        dedupe_key="dedupe:presence-slot",
        cooldown_key="cooldown:presence-slot",
        urgency=0.4,
        created_at=_iso_now(offset_seconds=-10),
    )
    _append_candidate(store, session_ids, canonical)
    _append_candidate(store, session_ids, merged)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)

    assert result.merged_candidate_ids == (merged.candidate_goal_id,)
    assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert result.terminal_candidate_goal_id == canonical.candidate_goal_id
    assert [candidate.candidate_goal_id for candidate in ledger.current_candidates] == [
        canonical.candidate_goal_id
    ]
    merged_entry = next(
        entry for entry in ledger.recent_entries if entry.decision_kind == BrainAutonomyDecisionKind.MERGED.value
    )
    assert merged_entry.merged_into_candidate_goal_id == canonical.candidate_goal_id
    non_action_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )
    assert non_action_entry.reason == "user_turn_open"
    assert (
        non_action_entry.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )
    assert non_action_entry.expected_reevaluation_condition_details == {"turn": "user"}
    current_candidate = ledger.candidate(canonical.candidate_goal_id)
    assert current_candidate is not None
    assert current_candidate.expected_reevaluation_condition == "after user turn ends"
    assert (
        current_candidate.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )
    assert current_candidate.expected_reevaluation_condition_details == {"turn": "user"}
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    assert agenda.goals == []


def test_presence_director_suppresses_low_confidence_candidate(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-low-confidence",
        "Weak visual signal.",
        confidence=0.59,
        source=BrainCandidateGoalSource.PERCEPTION.value,
        requires_user_turn_gap=False,
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)

    assert result.suppressed_candidate_ids == (candidate.candidate_goal_id,)
    assert result.terminal_decision is None
    assert ledger.current_candidates == []
    suppressed_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.SUPPRESSED.value
    )
    assert suppressed_entry.reason == "low_confidence"


def test_presence_director_suppresses_candidate_on_cooldown(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    previous = _candidate(
        "candidate-previous",
        "Already acted on this recently.",
        cooldown_key="cooldown:presence",
        dedupe_key=None,
        requires_user_turn_gap=False,
        created_at=_iso_now(offset_seconds=-10),
    )
    current = _candidate(
        "candidate-current",
        "Same slot should cool down.",
        cooldown_key="cooldown:presence",
        dedupe_key=None,
        requires_user_turn_gap=False,
        created_at=_iso_now(offset_seconds=-1),
    )
    _append_candidate(store, session_ids, previous)
    store.append_candidate_goal_accepted(
        candidate_goal_id=previous.candidate_goal_id,
        goal_id="goal-previous",
        reason="accepted_for_goal_creation",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        ts=_iso_now(offset_seconds=-5),
    )
    _append_candidate(store, session_ids, current)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)

    assert result.suppressed_candidate_ids == (current.candidate_goal_id,)
    suppressed_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.SUPPRESSED.value
    )
    assert suppressed_entry.reason == "cooldown_active"
    assert ledger.current_candidates == []


def test_presence_director_records_non_action_for_assistant_turn(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-assistant-turn",
        "Wait until the assistant finishes.",
        requires_user_turn_gap=True,
    )
    _append_candidate(store, session_ids, candidate)
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)

    assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert result.reason == "assistant_turn_open"
    assert [candidate_item.candidate_goal_id for candidate_item in ledger.current_candidates] == [
        candidate.candidate_goal_id
    ]
    non_action_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )
    assert (
        non_action_entry.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value
    )
    assert non_action_entry.expected_reevaluation_condition_details == {"turn": "assistant"}
    current_candidate = ledger.candidate(candidate.candidate_goal_id)
    assert current_candidate is not None
    assert current_candidate.expected_reevaluation_condition == "after assistant turn ends"
    assert (
        current_candidate.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value
    )
    assert current_candidate.expected_reevaluation_condition_details == {"turn": "assistant"}


@pytest.mark.asyncio
async def test_run_turn_end_pass_reevaluates_held_assistant_turn_close_candidate(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-turn-end-reeval",
        "Resume after the assistant turn closes.",
    )
    _append_candidate(store, session_ids, candidate)
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"text": "好的。"},
    )

    result = await executive.run_turn_end_pass()
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)

    assert result.progressed is True
    assert any(goal.details.get("autonomy", {}).get("candidate_goal_id") == candidate.candidate_goal_id for goal in agenda.goals)


def test_presence_director_reevaluates_held_candidate_after_user_turn_close_without_duplicate_proposal(
    tmp_path,
):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-user-turn-recheck",
        "Resume this held candidate after the user turn closes.",
    )
    _append_candidate(store, session_ids, candidate)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    first = executive.run_presence_director_pass()
    turn_end_event = store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    second = executive.run_presence_director_reevaluation(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            summary="User turn closed.",
            details={"turn": "user"},
            source_event_type=turn_end_event.event_type,
            source_event_id=turn_end_event.event_id,
            ts=turn_end_event.ts,
        )
    )

    events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=16,
            )
        )
    )
    created_events = [
        event for event in events if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED
    ]
    reevaluation_events = [
        event
        for event in events
        if event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
    ]
    accepted_event = next(
        event for event in events if event.event_type == BrainEventType.GOAL_CANDIDATE_ACCEPTED
    )

    assert first.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert second.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert len(created_events) == 1
    assert reevaluation_events
    assert accepted_event.causal_parent_id == reevaluation_events[-1].event_id


@pytest.mark.asyncio
async def test_presence_director_startup_recovery_reevaluates_pending_held_candidate(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-startup-recovery",
        "Resume this held candidate on startup recovery.",
    )
    _append_candidate(store, session_ids, candidate)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    result = await executive.run_startup_pass()
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=16,
    )

    assert result.progressed is True
    assert any(goal.details.get("autonomy", {}).get("candidate_goal_id") == candidate.candidate_goal_id for goal in agenda.goals)
    assert len([event for event in events if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED]) == 1
    assert any(
        event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
        and event.payload["trigger"]["kind"] == BrainReevaluationConditionKind.STARTUP_RECOVERY.value
        for event in events
    )


def test_presence_director_holds_maintenance_candidate_until_idle_window_opens(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
        ts=_iso_now(offset_seconds=-5),
    )
    candidate = _candidate(
        "candidate-maintenance-window",
        "Wait for the maintenance idle window.",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        requires_user_turn_gap=False,
        dedupe_key=None,
        cooldown_key=None,
    )
    _append_candidate(store, session_ids, candidate)

    held = executive.run_presence_director_pass()
    held_candidate = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).candidate(
        candidate.candidate_goal_id
    )
    assert held.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert held.reason == "maintenance_window_closed"
    assert held_candidate is not None
    assert (
        held_candidate.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value
    )

    not_before = held_candidate.expected_reevaluation_condition_details["not_before"]
    resumed = executive.run_presence_director_reevaluation(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
            summary="Maintenance window opened.",
            details={"not_before": not_before},
            ts=not_before,
        )
    )

    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    assert resumed.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert any(goal.details.get("autonomy", {}).get("candidate_goal_id") == candidate.candidate_goal_id for goal in agenda.goals)


def test_presence_director_time_reached_expires_held_candidate_without_new_proposal(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-expire-on-reevaluation",
        "Expire this held candidate when time is reached.",
        expires_at=_iso_now(offset_seconds=5),
    )
    _append_candidate(store, session_ids, candidate)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()

    result = executive.run_presence_director_expiry_cleanup(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.TIME_REACHED.value,
            summary="Time reached.",
            ts=_iso_now(offset_seconds=6),
        )
    )
    events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=16,
    )
    chronological_events = list(reversed(events))
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    trigger_index = next(
        index
        for index, event in enumerate(chronological_events)
        if event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
    )
    expired_event_index = next(
        index
        for index, event in enumerate(chronological_events)
        if event.event_type == BrainEventType.GOAL_CANDIDATE_EXPIRED
    )
    expired_event = next(
        event for event in chronological_events if event.event_type == BrainEventType.GOAL_CANDIDATE_EXPIRED
    )

    assert result.expired_candidate_ids == (candidate.candidate_goal_id,)
    assert ledger.current_candidates == []
    assert len([event for event in events if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED]) == 1
    assert trigger_index < expired_event_index
    assert expired_event.payload["reason_details"]["cleanup_owner"] == "explicit_expiry_cleanup"
    assert (
        expired_event.payload["reason_details"]["trigger_kind"]
        == BrainReevaluationConditionKind.TIME_REACHED.value
    )


def test_presence_director_expiry_cleanup_does_not_auto_accept_fresh_candidates(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-fresh-expiry-cleanup",
        "Do not run this untouched fresh candidate during expiry cleanup.",
        requires_user_turn_gap=False,
        expires_at=_iso_now(offset_seconds=120),
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_expiry_cleanup(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.TIME_REACHED.value,
            summary="Cleanup only.",
            ts=_iso_now(offset_seconds=5),
        )
    )
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)

    assert result.progressed is False
    assert result.terminal_decision is None
    assert result.expired_candidate_ids == ()
    assert agenda.goals == []
    assert [item.candidate_goal_id for item in ledger.current_candidates] == [candidate.candidate_goal_id]


@pytest.mark.asyncio
async def test_presence_director_startup_cleanup_expires_stale_candidates_before_recovery(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    stale = _candidate(
        "candidate-stale-startup",
        "Expire this stale candidate on startup cleanup.",
        expires_at=_iso_now(offset_seconds=-5),
        requires_user_turn_gap=False,
    )
    held = _candidate(
        "candidate-held-startup",
        "Resume this held candidate after startup recovery.",
    )
    _append_candidate(store, session_ids, stale)
    _append_candidate(store, session_ids, held)
    store.append_director_non_action(
        candidate_goal_id=held.candidate_goal_id,
        reason="user_turn_open",
        reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
        expected_reevaluation_condition="after user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"turn": "user"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    result = await executive.run_startup_pass()
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=24,
            )
        )
    )
    expired_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == BrainEventType.GOAL_CANDIDATE_EXPIRED
        and event.payload["candidate_goal_id"] == stale.candidate_goal_id
    )
    accepted_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type == BrainEventType.GOAL_CANDIDATE_ACCEPTED
        and event.payload["candidate_goal_id"] == held.candidate_goal_id
    )

    assert result.progressed is True
    assert expired_index < accepted_index
    assert ledger.candidate(stale.candidate_goal_id) is None
    assert ledger.candidate(held.candidate_goal_id) is None
    assert any(goal.details.get("autonomy", {}).get("candidate_goal_id") == held.candidate_goal_id for goal in agenda.goals)


def test_presence_director_rotates_terminal_attention_across_families(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    environment_leader = _candidate(
        "candidate-environment-leader",
        "Environment family leader.",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        created_at=_iso_now(offset_seconds=-90),
    )
    environment_backlog = _candidate(
        "candidate-environment-backlog",
        "Environment family backlog.",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        created_at=_iso_now(offset_seconds=-60),
    )
    maintenance_candidate = _candidate(
        "candidate-maintenance-family",
        "Maintenance family candidate.",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        created_at=_iso_now(offset_seconds=-30),
    )
    for candidate in (environment_leader, environment_backlog, maintenance_candidate):
        _append_candidate(store, session_ids, candidate)

    first = executive.run_presence_director_pass()
    second = executive.run_presence_director_pass()
    third = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    non_action_entries = [
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    ]

    assert first.terminal_candidate_goal_id == environment_leader.candidate_goal_id
    assert second.terminal_candidate_goal_id == maintenance_candidate.candidate_goal_id
    assert third.terminal_candidate_goal_id == environment_leader.candidate_goal_id
    assert len([candidate for candidate in ledger.current_candidates if candidate.goal_family == BrainGoalFamily.ENVIRONMENT.value]) == 2
    assert [entry.reason_details.get("goal_family") for entry in non_action_entries[-3:]] == [
        BrainGoalFamily.ENVIRONMENT.value,
        BrainGoalFamily.MEMORY_MAINTENANCE.value,
        BrainGoalFamily.ENVIRONMENT.value,
    ]
    assert all(entry.reason_details.get("selected_by") == "family_rotation" for entry in non_action_entries[-3:])


def test_presence_director_projection_change_only_rechecks_opted_in_held_perception_candidates(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    opted_in = _candidate(
        "candidate-projection-opted-in",
        "Projection changes may recheck this held candidate.",
        source=BrainCandidateGoalSource.PERCEPTION.value,
        requires_user_turn_gap=False,
    )
    ignored = _candidate(
        "candidate-projection-ignored",
        "Do not reconsider this candidate on projection changes.",
        source=BrainCandidateGoalSource.PERCEPTION.value,
        requires_user_turn_gap=False,
        candidate_type="presence_attention_returned",
    )
    _append_candidate(store, session_ids, opted_in)
    _append_candidate(store, session_ids, ignored)
    store.append_director_non_action(
        candidate_goal_id=opted_in.candidate_goal_id,
        reason="goal_family_busy",
        reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
        expected_reevaluation_condition="after a meaningful scene change",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"allow_projection_recheck": True},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id=ignored.candidate_goal_id,
        reason="goal_family_busy",
        reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
        expected_reevaluation_condition="wait for explicit family availability",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    result = executive.run_presence_director_reevaluation(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
            summary="Scene projection changed.",
            details={"person_present": "present"},
        )
    )
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)

    assert result.terminal_candidate_goal_id == opted_in.candidate_goal_id
    assert ledger.candidate(ignored.candidate_goal_id) is not None


def test_presence_director_accepts_candidate_and_embeds_autonomy_metadata(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    candidate = _candidate(
        "candidate-accept",
        "Acknowledge presence quietly.",
        source=BrainCandidateGoalSource.RUNTIME.value,
        requires_user_turn_gap=False,
        dedupe_key="dedupe:accept",
        cooldown_key="cooldown:accept",
        payload={
            "goal_intent": "autonomy.presence_acknowledgement",
            "goal_details": {"channel": "presence"},
        },
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    accepted_goal = agenda.goal(result.accepted_goal_id)

    assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert accepted_goal is not None
    assert accepted_goal.intent == "autonomy.presence_acknowledgement"
    assert accepted_goal.details["channel"] == "presence"
    assert accepted_goal.details["autonomy"]["candidate_goal_id"] == candidate.candidate_goal_id
    assert accepted_goal.details["autonomy"]["dedupe_key"] == candidate.dedupe_key
    assert accepted_goal.details["autonomy"]["cooldown_key"] == candidate.cooldown_key
    assert accepted_goal.details["autonomy"]["initiative_class"] == candidate.initiative_class
    assert ledger.current_candidates == []
    accepted_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.ACCEPTED.value
    )
    assert accepted_entry.accepted_goal_id == result.accepted_goal_id


def test_presence_director_suppresses_duplicate_active_goal(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store, executive = _executive(tmp_path, session_ids)
    existing_goal_id = executive.create_goal(
        title="Existing autonomy goal",
        intent="autonomy.presence_acknowledgement",
        source="test",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        details={
            "autonomy": {
                "dedupe_key": "dedupe:duplicate",
                "cooldown_key": "cooldown:duplicate",
            }
        },
    )
    candidate = _candidate(
        "candidate-duplicate",
        "Do not create a duplicate goal.",
        source=BrainCandidateGoalSource.RUNTIME.value,
        requires_user_turn_gap=False,
        dedupe_key="dedupe:duplicate",
        cooldown_key="cooldown:duplicate",
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)

    assert result.suppressed_candidate_ids == (candidate.candidate_goal_id,)
    assert len(agenda.goals) == 1
    assert agenda.goal(existing_goal_id) is not None
    assert ledger.current_candidates == []
    suppressed_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.SUPPRESSED.value
    )
    assert suppressed_entry.reason == "duplicate_active_goal"


def test_executive_policy_compiler_is_deterministic_for_seeded_surface(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="policy")
    store = BrainStore(path=tmp_path / "brain.db")
    surface = _policy_surface_for_store(
        store=store,
        session_ids=session_ids,
        scene_degraded_mode="limited",
        review_debt_count=1,
        held_claim_ids=("claim-held",),
        unresolved_count=1,
    )

    first = compile_executive_policy(
        surface,
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )
    second = compile_executive_policy(
        surface,
        task=BrainContextTask.REEVALUATION,
        reference_ts="2026-04-21T10:00:00+00:00",
    )

    assert first == second
    assert first.action_posture == BrainExecutivePolicyActionPosture.DEFER.value
    assert first.approval_requirement == BrainExecutivePolicyApprovalRequirement.USER_CONFIRMATION.value
    assert (
        first.procedural_reuse_eligibility
        == BrainExecutiveProceduralReuseEligibility.ADVISORY_ONLY.value
    )
    assert "dossier_review_debt" in first.reason_codes
    assert "held_claim_present" in first.reason_codes


def test_presence_director_accepts_with_reason_codes_and_policy_frame(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="policy-accept")
    store = BrainStore(path=tmp_path / "brain.db")
    surface = _policy_surface_for_store(store=store, session_ids=session_ids)
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=_StaticSurfaceBuilder(surface),
    )
    candidate = _candidate(
        "candidate-policy-accept",
        "Safe presence acknowledgement.",
        source=BrainCandidateGoalSource.RUNTIME.value,
        requires_user_turn_gap=False,
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    accepted_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.ACCEPTED.value
    )

    assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert result.reason_codes == ("accepted_for_goal_creation",)
    assert accepted_entry.reason_codes == ["accepted_for_goal_creation"]
    assert accepted_entry.executive_policy is not None
    assert accepted_entry.executive_policy["action_posture"] == "allow"


def test_presence_director_policy_defers_on_review_debt(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="policy-review-debt")
    store = BrainStore(path=tmp_path / "brain.db")
    surface = _policy_surface_for_store(
        store=store,
        session_ids=session_ids,
        review_debt_count=2,
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=_StaticSurfaceBuilder(surface),
    )
    candidate = _candidate(
        "candidate-policy-review-debt",
        "Wait until governance debt clears.",
        source=BrainCandidateGoalSource.RUNTIME.value,
        requires_user_turn_gap=False,
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    non_action_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )

    assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert result.reason == "policy_conservative_deferral"
    assert "dossier_review_debt" in result.reason_codes
    assert non_action_entry.expected_reevaluation_condition_kind == "projection_changed"
    assert "policy_conservative_deferral" in non_action_entry.reason_codes
    assert "dossier_review_debt" in non_action_entry.reason_codes
    assert non_action_entry.executive_policy is not None
    assert non_action_entry.executive_policy["approval_requirement"] == "user_confirmation"


def test_presence_director_policy_blocks_on_unavailable_scene(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="policy-scene")
    store = BrainStore(path=tmp_path / "brain.db")
    surface = _policy_surface_for_store(
        store=store,
        session_ids=session_ids,
        scene_degraded_mode="unavailable",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=_StaticSurfaceBuilder(surface),
    )
    candidate = _candidate(
        "candidate-policy-scene",
        "Do not act while the scene is unavailable.",
        source=BrainCandidateGoalSource.RUNTIME.value,
        requires_user_turn_gap=False,
    )
    _append_candidate(store, session_ids, candidate)

    result = executive.run_presence_director_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    non_action_entry = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )

    assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
    assert result.reason == "policy_blocked_action"
    assert "scene_unavailable" in result.reason_codes
    assert "policy_blocked_action" in non_action_entry.reason_codes
    assert "scene_unavailable" in non_action_entry.reason_codes
    assert non_action_entry.executive_policy is not None
    assert non_action_entry.executive_policy["action_posture"] == "suppress"


@pytest.mark.asyncio
async def test_presence_director_accepted_scene_goal_runs_internal_capability_template(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store = BrainStore(path=tmp_path / "brain.db")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
    )
    candidate = _candidate(
        "candidate-routed-scene",
        "Acknowledge returned attention silently.",
        candidate_type="presence_user_reentered",
        source=BrainCandidateGoalSource.PERCEPTION.value,
        requires_user_turn_gap=False,
        payload={
            "goal_intent": "autonomy.presence_user_reentered",
            "goal_details": {
                "scene_candidate": {
                    "presence_scope_key": "browser:presence",
                    "person_present": "present",
                    "attention_to_camera": "toward_camera",
                    "engagement_state": "engaged",
                }
            },
        },
    )
    _append_candidate(store, session_ids, candidate)

    decision = executive.run_presence_director_pass()
    result = await executive.run_until_quiescent(max_iterations=4)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(decision.accepted_goal_id)

    assert decision.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert result.progressed is True
    assert goal is not None
    assert goal.status == "completed"
    assert [step.capability_id for step in goal.steps] == [
        "observation.inspect_presence_state",
        "reporting.record_presence_event",
    ]
    assert goal.steps[0].output["person_present"] == "uncertain"
    assert goal.steps[1].output["candidate_type"] == "presence_user_reentered"

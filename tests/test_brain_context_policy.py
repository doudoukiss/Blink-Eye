from dataclasses import replace
from pathlib import Path

from blink.brain.capabilities import CapabilityExecutionResult, CapabilityRegistry
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.context import (
    BrainContextBudgetProfile,
    BrainContextCompiler,
    BrainContextSelector,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt
from blink.brain.memory_v2 import BrainGovernanceReasonCode, ClaimLedger
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentScopeType,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


def _build_compiler(*, store: BrainStore, session_ids):
    return BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.EN,
        base_prompt=base_brain_system_prompt(Language.EN),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
        ),
    )


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _record_review_held_role_claim(store: BrainStore, session_ids) -> tuple[str, str]:
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=store)
    event_context = store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="context-governance-held",
    )
    prior_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-role-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role",
        event_context=event_context,
    )
    current_claim = ledger.supersede_claim(
        prior_claim.claim_id,
        replacement_claim={
            "subject_entity_id": user.entity_id,
            "predicate": "profile.role",
            "object_data": {"value": "product manager"},
            "scope_type": "user",
            "scope_id": session_ids.user_id,
            "claim_key": "profile.role",
        },
        reason="updated the recorded role",
        source_event_id="evt-role-2",
        event_context=event_context,
    )
    held_claim = ledger.request_claim_review(
        current_claim.claim_id,
        source_event_id="evt-role-review",
        reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
        event_context=event_context,
    )
    return prior_claim.claim_id, held_claim.claim_id


def _record_stale_role_claim(store: BrainStore, session_ids) -> str:
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=store)
    event_context = store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="context-governance-stale",
    )
    claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-role-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role",
        event_context=event_context,
    )
    stale_claim = ledger.expire_claim(
        claim.claim_id,
        source_event_id="evt-role-expire",
        reason_codes=[BrainGovernanceReasonCode.STALE_WITHOUT_REFRESH.value],
        event_context=event_context,
    )
    return stale_claim.claim_id


def _append_completed_procedural_goal(
    store: BrainStore,
    session_ids,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
):
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=goal_title,
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    proposal = BrainPlanProposal(
        plan_proposal_id=proposal_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=f"Execute {goal_title}.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        created_at=_ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    adopted_event = store.append_planning_adopted(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary=f"Adopt {goal_title}.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=_ts(start_second + 1),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
                commitment_id=commitment_id,
                status="open",
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=1,
                last_summary=f"Adopt {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=_ts(start_second + 2),
    )
    completed_steps: list[BrainGoalStep] = []
    current_second = start_second + 3
    for step_index, capability_id in enumerate(sequence):
        request_event = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=goal_id,
            ts=_ts(current_second),
        )
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_COMPLETED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "step_index": step_index,
                "result": CapabilityExecutionResult.success(
                    capability_id=capability_id,
                    summary=f"Completed {capability_id}.",
                    output={"slot": step_index},
                ).model_dump(),
            },
            correlation_id=goal_id,
            causal_parent_id=request_event.event_id,
            ts=_ts(current_second + 1),
        )
        completed_steps.append(
            BrainGoalStep(
                capability_id=capability_id,
                status="completed",
                attempts=1,
                summary=f"Completed {capability_id}.",
                output={"slot": step_index},
            )
        )
        current_second += 2
    store.append_brain_event(
        event_type=BrainEventType.GOAL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
                commitment_id=commitment_id,
                status="completed",
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=1,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "completed"},
        },
        correlation_id=goal_id,
        ts=_ts(current_second),
    )


def _surface_with_task_state(surface):
    return replace(
        surface,
        private_working_memory=BrainPrivateWorkingMemoryProjection(
            scope_type="thread",
            scope_id=surface.private_working_memory.scope_id,
            records=[
                BrainPrivateWorkingMemoryRecord(
                    record_id="pwm-plan",
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                    summary="Need confirmation before the current plan can proceed.",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=["proposal-1"],
                    source_event_ids=["evt-plan-1"],
                    goal_id="goal-1",
                    commitment_id="commitment-1",
                    plan_proposal_id="proposal-1",
                    observed_at=_ts(10),
                    updated_at=_ts(10),
                ),
                BrainPrivateWorkingMemoryRecord(
                    record_id="pwm-uncertainty",
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                    summary="The scene feed is degraded and needs refresh.",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                    backing_ids=["scene-world-1"],
                    source_event_ids=["evt-scene-1"],
                    commitment_id="commitment-1",
                    observed_at=_ts(11),
                    updated_at=_ts(11),
                ),
            ],
            updated_at=_ts(12),
        ),
        scene_world_state=BrainSceneWorldProjection(
            scope_type="presence",
            scope_id="browser:presence",
            entities=[
                BrainSceneWorldEntityRecord(
                    entity_id="entity-1",
                    entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                    canonical_label="dock_pad",
                    summary="A dock pad is visible on the desk.",
                    state=BrainSceneWorldRecordState.ACTIVE.value,
                    evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                    zone_id="zone:desk",
                    confidence=0.8,
                    freshness="current",
                    affordance_ids=["aff-1"],
                    backing_ids=["entity:dock_pad"],
                    source_event_ids=["evt-scene-1"],
                    observed_at=_ts(12),
                    updated_at=_ts(12),
                    expires_at=_ts(40),
                )
            ],
            affordances=[
                BrainSceneWorldAffordanceRecord(
                    affordance_id="aff-1",
                    entity_id="entity-1",
                    capability_family="vision.inspect",
                    summary="The dock pad can be visually inspected.",
                    availability=BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                    confidence=0.75,
                    freshness="current",
                    backing_ids=["entity:dock_pad", "vision.inspect"],
                    source_event_ids=["evt-scene-1"],
                    observed_at=_ts(12),
                    updated_at=_ts(12),
                    expires_at=_ts(40),
                )
            ],
            degraded_mode="limited",
            degraded_reason_codes=["camera_frame_stale"],
            updated_at=_ts(12),
        ),
        active_situation_model=BrainActiveSituationProjection(
            scope_type="thread",
            scope_id=surface.active_situation_model.scope_id,
            records=[
                BrainActiveSituationRecord(
                    record_id="situation-plan",
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary="The current dock review plan is active.",
                    state=BrainActiveSituationRecordState.ACTIVE.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    confidence=0.77,
                    freshness="current",
                    backing_ids=["proposal-1"],
                    source_event_ids=["evt-plan-1"],
                    goal_id="goal-1",
                    commitment_id="commitment-1",
                    plan_proposal_id="proposal-1",
                    observed_at=_ts(10),
                    updated_at=_ts(12),
                ),
                BrainActiveSituationRecord(
                    record_id="situation-uncertainty",
                    record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                    summary="Perception is degraded; treat world-state cautiously.",
                    state=BrainActiveSituationRecordState.UNRESOLVED.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    confidence=0.5,
                    freshness="limited",
                    uncertainty_codes=["scene_stale", "camera_frame_stale"],
                    private_record_ids=["pwm-uncertainty"],
                    backing_ids=["scene-world-1"],
                    source_event_ids=["evt-scene-1"],
                    commitment_id="commitment-1",
                    observed_at=_ts(11),
                    updated_at=_ts(12),
                ),
            ],
            updated_at=_ts(12),
        ),
    )


def test_context_policy_is_task_aware_and_budgeted(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "designer"},
        rendered_text="user role is designer",
        confidence=0.8,
        singleton=True,
        source_event_id="evt-role-1",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "product manager"},
        rendered_text="user role is product manager",
        confidence=0.95,
        singleton=True,
        source_event_id="evt-role-2",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.upsert_task(
        user_id=session_ids.user_id,
        title="Follow up on Alpha",
        details={"summary": "Need user confirmation"},
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
        ),
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for user confirmation.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                summary="Resume after user confirmation.",
            ),
        ],
    )

    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(
        latest_user_text="What do you remember about me and Alpha?",
        include_historical_claims=True,
    )
    selector = BrainContextSelector()
    reply = selector.select(snapshot=surface, task=BrainContextTask.REPLY, language=Language.EN)
    recall = selector.select(snapshot=surface, task=BrainContextTask.RECALL, language=Language.EN)
    planning = selector.select(
        snapshot=surface, task=BrainContextTask.PLANNING, language=Language.EN
    )
    constrained = selector.select(
        snapshot=surface,
        task=BrainContextTask.REPLY,
        language=Language.EN,
        static_sections={
            "policy": "policy " * 40,
            "identity": "identity " * 20,
            "style": "style",
            "capabilities": "capabilities",
        },
        budget_profile=BrainContextBudgetProfile(
            task="reply", max_tokens=80, section_caps={"recent_memory": 1}
        ),
    )

    assert reply.section("historical_claims") is None
    assert recall.section("historical_claims") is not None
    assert recall.section("claim_provenance") is not None
    assert planning.section("commitment_projection") is not None
    assert surface.active_situation_model.scope_id == session_ids.thread_id
    assert surface.active_situation_model.kind_counts
    assert "wake" in planning.section("commitment_projection").content.lower()
    assert (
        BrainCommitmentScopeType.RELATIONSHIP.value
        in planning.section("commitment_projection").content
    )
    assert "rev=" in planning.section("commitment_projection").content
    assert constrained.estimated_tokens <= 80
    assert any(decision.reason == "budget_exceeded" for decision in constrained.selection_trace.decisions)


def test_context_compiler_compile_packet_exposes_selection_trace(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": "goal-private-1",
            "capability_id": "capability.preview",
            "step_index": 0,
            "result": {"summary": "Preview completed."},
        },
        ts=_ts(1),
    )
    compiler = _build_compiler(store=store, session_ids=session_ids)

    packet = compiler.compile_packet(
        latest_user_text="Remember my current role.",
        task=BrainContextTask.REPLY,
    )

    assert "## Policy" in packet.prompt
    assert "## Private Working Memory" in packet.prompt
    assert "[active][observed][recent_tool_outcome]" in packet.prompt
    assert packet.task == BrainContextTask.REPLY
    assert packet.selected_context.selection_trace.task == BrainContextTask.REPLY.value
    assert packet.packet_trace is not None
    assert packet.packet_trace.task == BrainContextTask.REPLY.value
    assert packet.packet_trace.temporal_mode == "current_first"
    assert packet.packet_trace.mode_policy.task == BrainContextTask.REPLY
    assert packet.packet_trace.static_token_usage > 0
    assert all(
        decision.decision_reason_codes
        for decision in packet.selected_context.selection_trace.decisions
    )


def test_context_compiler_keeps_internal_capabilities_out_of_reply_but_in_planning(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    registry = build_brain_capability_registry(language=Language.EN)

    compiler = BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.EN,
        base_prompt=base_brain_system_prompt(Language.EN),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=registry,
        ),
    )

    reply_packet = compiler.compile_packet(
        latest_user_text="What can you do?",
        task=BrainContextTask.REPLY,
    )
    planning_packet = compiler.compile_packet(
        latest_user_text="Plan the next maintenance step.",
        task=BrainContextTask.PLANNING,
    )

    assert "## Internal Capabilities" not in reply_packet.prompt
    assert "## Internal Capabilities" in planning_packet.prompt
    assert "maintenance.review_memory_health" in planning_packet.prompt
    assert planning_packet.packet_trace is not None
    assert planning_packet.packet_trace.task == BrainContextTask.PLANNING.value


def test_planning_packet_surfaces_procedural_skill_anchors_and_provenance(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="skill-packet")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed skill one",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=0,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-2",
        commitment_id="seed-commitment-2",
        goal_title="Seed skill two",
        proposal_id="seed-proposal-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=40,
    )
    store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
        ),
    )
    executive.create_commitment_goal(
        title="Plan maintenance with prior skill",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        details={"survive_restart": True},
    )
    compiler = _build_compiler(store=store, session_ids=session_ids)

    packet = compiler.compile_packet(
        latest_user_text="Plan the next maintenance step.",
        task=BrainContextTask.PLANNING,
    )

    assert packet.packet_trace is not None
    assert any(
        anchor.anchor_type == "procedural_skill" for anchor in packet.packet_trace.selected_anchors
    )
    skill_items = [
        item for item in packet.packet_trace.selected_items if item.item_type == "procedural_skill"
    ]
    assert skill_items
    assert any(item.provenance.get("skill_id") for item in skill_items)
    assert any(item.provenance.get("support_trace_ids") for item in skill_items)
    assert "Seed skill" in packet.prompt


def test_active_context_reply_distinguishes_current_and_historical_truth(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "designer"},
        rendered_text="user role is designer",
        confidence=0.8,
        singleton=True,
        source_event_id="evt-role-1",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "product manager"},
        rendered_text="user role is product manager",
        confidence=0.95,
        singleton=True,
        source_event_id="evt-role-2",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    compiler = _build_compiler(store=store, session_ids=session_ids)

    current_packet = compiler.compile_packet(
        latest_user_text="What is my role right now?",
        task=BrainContextTask.REPLY,
    )
    historical_packet = compiler.compile_packet(
        latest_user_text="How has my role changed before?",
        task=BrainContextTask.REPLY,
    )

    assert "## Active Continuity" in current_packet.prompt
    assert "product manager" in current_packet.prompt
    assert "designer" not in current_packet.prompt
    assert current_packet.packet_trace is not None
    assert current_packet.packet_trace.temporal_mode == "current_first"

    assert "## Recent Changes" in historical_packet.prompt
    assert "product manager" in historical_packet.prompt
    assert "designer" in historical_packet.prompt
    assert historical_packet.packet_trace is not None
    assert historical_packet.packet_trace.temporal_mode == "historical_focus"
    assert any(
        item.temporal_kind == "historical" for item in historical_packet.packet_trace.selected_items
    )


def test_active_context_historical_focus_prefers_dossier_recent_changes(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "designer"},
        rendered_text="user role is designer",
        confidence=0.8,
        singleton=True,
        source_event_id="evt-role-1",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "product manager"},
        rendered_text="user role is product manager",
        confidence=0.95,
        singleton=True,
        source_event_id="evt-role-2",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    current_claim = next(
        claim
        for claim in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            predicate="profile.role",
            limit=4,
        )
        if claim.object == {"value": "product manager"}
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="We knew your role as designer.",
        content={"summary": "We knew your role as designer."},
        salience=1.0,
        source_event_ids=["evt-role-1"],
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="Your role is now product manager.",
        content={"summary": "Your role is now product manager."},
        salience=1.0,
        source_claim_ids=[current_claim.claim_id],
        source_event_ids=["evt-role-2"],
    )

    compiler = _build_compiler(store=store, session_ids=session_ids)

    historical_packet = compiler.compile_packet(
        latest_user_text="How has our relationship changed before?",
        task=BrainContextTask.REPLY,
    )

    assert "## Recent Changes" in historical_packet.prompt
    assert "designer" in historical_packet.prompt
    assert historical_packet.packet_trace is not None
    assert historical_packet.packet_trace.temporal_mode == "historical_focus"
    assert any(
        item.item_type == "dossier_change" and item.temporal_kind == "historical"
        for item in historical_packet.packet_trace.selected_items
    )


def test_active_context_reply_suppresses_held_continuity_with_traceability(tmp_path):
    store = BrainStore(path=tmp_path / "reply-governance.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="reply-governance")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    _prior_claim_id, held_claim_id = _record_review_held_role_claim(store, session_ids)
    compiler = _build_compiler(store=store, session_ids=session_ids)

    packet = compiler.compile_packet(
        latest_user_text="What is my role right now?",
        task=BrainContextTask.REPLY,
    )
    digest = build_context_packet_digest(
        packet_traces={"reply": packet.packet_trace.as_dict() if packet.packet_trace else None}
    )

    assert packet.packet_trace is not None
    assert not any(
        item.item_type == "dossier"
        and item.section_key == "active_continuity"
        and item.selected
        for item in packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "dossier_change"
        and item.section_key == "unresolved_state"
        and item.availability_state == "suppressed"
        and {"held_support", "review_debt"} <= set(item.governance_reason_codes)
        for item in packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "dossier"
        and item.reason == "governance_suppressed"
        and {"held_support", "review_debt"} <= set(item.governance_reason_codes)
        for item in packet.packet_trace.dropped_items
    )
    assert digest["reply"]["governance_drop_reason_counts"]["held_support"] >= 1
    assert digest["reply"]["governance_drop_reason_counts"]["review_debt"] >= 1
    assert held_claim_id in digest["reply"]["suppressed_backing_ids"]


def test_active_context_planning_and_recall_preserve_governance_annotations(tmp_path):
    store = BrainStore(path=tmp_path / "annotation-governance.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="annotation-governance",
    )
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    stale_claim_id = _record_stale_role_claim(store, session_ids)
    surface_builder = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    )
    planning_surface = surface_builder.build(
        latest_user_text="Plan the next step around my current role.",
        include_historical_claims=True,
    )
    recall_surface = surface_builder.build(
        latest_user_text="Recall my role changes.",
        include_historical_claims=True,
    )
    reflection_surface = surface_builder.build(
        latest_user_text="Reflect on my role history.",
        include_historical_claims=True,
    )
    planning_packet = compile_context_packet_from_surface(
        snapshot=planning_surface,
        latest_user_text="Plan the next step around my current role.",
        task=BrainContextTask.PLANNING,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="planning", max_tokens=1200),
    )
    recall_packet = compile_context_packet_from_surface(
        snapshot=recall_surface,
        latest_user_text="Recall my role changes.",
        task=BrainContextTask.RECALL,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="recall", max_tokens=1200),
    )
    reflection_packet = compile_context_packet_from_surface(
        snapshot=reflection_surface,
        latest_user_text="Reflect on my role history.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="reflection", max_tokens=1200),
    )

    planning_claim = next(
        item
        for item in (
            [*planning_packet.packet_trace.selected_items, *planning_packet.packet_trace.dropped_items]
        )
        if item.item_type == "claim"
    )
    recall_claim = next(
        item for item in recall_packet.packet_trace.selected_items if item.item_type == "claim"
    )
    reflection_claim = next(
        item
        for item in reflection_packet.packet_trace.selected_items
        if item.item_type == "claim"
    )

    assert planning_claim.section_key == "unresolved_state"
    assert planning_claim.availability_state == "annotated"
    assert {"stale_support", "stale_without_refresh"} <= set(
        planning_claim.governance_reason_codes
    )
    assert planning_claim.provenance["backing_record_id"] == stale_claim_id
    assert recall_claim.section_key == "relevant_continuity"
    assert recall_claim.availability_state == "annotated"
    assert recall_claim.provenance["backing_record_id"] == stale_claim_id
    assert "[annotated]" in recall_claim.content
    assert reflection_claim.section_key == "relevant_continuity"
    assert reflection_claim.availability_state == "annotated"
    assert reflection_claim.provenance["backing_record_id"] == stale_claim_id


def test_active_context_reply_uses_multi_hop_expansion_and_dossier_match(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="project.alpha.launch_status",
        subject="project-alpha",
        value={"value": "launch delayed"},
        rendered_text="Project Alpha launch is delayed.",
        confidence=0.92,
        singleton=True,
        source_event_id="evt-alpha-launch",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="project.alpha.owner",
        subject="project-alpha",
        value={"value": "Ada"},
        rendered_text="Owner is Ada.",
        confidence=0.88,
        singleton=True,
        source_event_id="evt-alpha-owner",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    project_claim_ids = [
        record.claim_id
        for record in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=8,
        )
    ]
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="project_arc",
        rendered_summary="Alpha project is in launch recovery.",
        content={"project_key": "alpha", "summary": "Alpha project is in launch recovery."},
        salience=0.9,
        source_claim_ids=project_claim_ids,
        source_event_ids=["evt-alpha-arc"],
    )
    compiler = _build_compiler(store=store, session_ids=session_ids)

    packet = compiler.compile_packet(
        latest_user_text="What is happening with alpha launch?",
        task=BrainContextTask.REPLY,
    )

    assert packet.packet_trace is not None
    assert "## Active Continuity" in packet.prompt
    assert "Alpha project is in launch recovery." in packet.prompt
    assert any(anchor.anchor_type == "dossier" for anchor in packet.packet_trace.selected_anchors)
    assert any(expansion.accepted for expansion in packet.packet_trace.expansions)
    assert any(item.item_type == "dossier" for item in packet.packet_trace.selected_items)


def test_active_context_reply_respects_dynamic_budget_and_records_drops(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "",
            "policy": "",
            "style": "",
            "action_library": "",
        }
    )
    for index in range(6):
        store.remember_fact(
            user_id=session_ids.user_id,
            namespace=f"profile.topic_{index}",
            subject="user",
            value={"value": f"topic-{index}"},
            rendered_text=f"user topic {index} is still active for the ongoing operating context",
            confidence=0.9,
            singleton=False,
            source_event_id=f"evt-topic-{index}",
            source_episode_id=None,
            provenance={"source": "test"},
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(
        latest_user_text="What is still active?",
        include_historical_claims=True,
    )

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="What is still active?",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        base_prompt="",
        budget_profile=BrainContextBudgetProfile(task="reply", max_tokens=35),
    )

    assert packet.selected_context.estimated_tokens <= 35
    assert packet.packet_trace is not None
    assert any(
        item.reason == "dynamic_budget_exceeded" for item in packet.packet_trace.dropped_items
    )


def test_context_compiler_generates_rich_packet_traces_for_multi_mode_context_packets(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="rich-tasks")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(
        latest_user_text="Review the dock state.",
        include_historical_claims=True,
    )
    rich_surface = _surface_with_task_state(surface)

    recall_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Recall the active dock state.",
        task=BrainContextTask.RECALL,
        language=Language.EN,
    )
    reflection_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Reflect on what is unresolved in the dock state.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )
    critique_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Critique the current dock plan.",
        task=BrainContextTask.CRITIQUE,
        language=Language.EN,
    )
    wake_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Route the wake decision for the dock plan.",
        task=BrainContextTask.WAKE,
        language=Language.EN,
    )
    reevaluation_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Reevaluate the unresolved dock plan.",
        task=BrainContextTask.REEVALUATION,
        language=Language.EN,
    )
    operator_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Audit the current dock situation.",
        task=BrainContextTask.OPERATOR_AUDIT,
        language=Language.EN,
    )
    governance_packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text="Review dock governance state.",
        task=BrainContextTask.GOVERNANCE_REVIEW,
        language=Language.EN,
    )

    assert recall_packet.packet_trace is not None
    assert reflection_packet.packet_trace is not None
    assert critique_packet.packet_trace is not None
    assert wake_packet.packet_trace is not None
    assert reevaluation_packet.packet_trace is not None
    assert operator_packet.packet_trace is not None
    assert governance_packet.packet_trace is not None
    assert recall_packet.packet_trace.task == BrainContextTask.RECALL.value
    assert reflection_packet.packet_trace.task == BrainContextTask.REFLECTION.value
    assert critique_packet.packet_trace.task == BrainContextTask.CRITIQUE.value
    assert wake_packet.packet_trace.task == BrainContextTask.WAKE.value
    assert reevaluation_packet.packet_trace.task == BrainContextTask.REEVALUATION.value
    assert operator_packet.packet_trace.mode_policy.trace_verbosity.value == "verbose"
    assert governance_packet.packet_trace.mode_policy.trace_verbosity.value == "verbose"
    assert "planning_anchors" in wake_packet.packet_trace.mode_policy.dynamic_section_keys
    assert reevaluation_packet.selected_context.section("unresolved_state") is not None
    assert "relevant_continuity" in operator_packet.packet_trace.mode_policy.dynamic_section_keys
    assert "planning_anchors" not in governance_packet.packet_trace.mode_policy.dynamic_section_keys
    assert any(
        item.item_type == "active_situation_record"
        and item.provenance.get("record_id") == "situation-plan"
        for item in recall_packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "scene_world_entity" and item.provenance.get("entity_id") == "entity-1"
        for item in recall_packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "scene_world_affordance"
        and item.provenance.get("affordance_id") == "aff-1"
        for item in recall_packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "uncertainty_record"
        and item.provenance.get("record_id") == "situation-uncertainty"
        for item in reflection_packet.packet_trace.selected_items
    )
    assert any(
        item.item_type == "private_working_memory_record"
        and item.provenance.get("record_id") == "pwm-plan"
        for item in critique_packet.packet_trace.selected_items
    )
    assert all(
        item.decision_reason_codes
        for packet in (
            recall_packet,
            reflection_packet,
            critique_packet,
            wake_packet,
            reevaluation_packet,
            operator_packet,
            governance_packet,
        )
        for item in packet.packet_trace.selected_items + packet.packet_trace.dropped_items
    )


def test_scene_episode_packet_policy_is_explicit_and_traceable_by_task(tmp_path):
    from tests.test_brain_memory_v2 import (
        _multimodal_event_context,
        _scene_world_projection_for_multimodal,
        _seed_scene_episode,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="scene-policy")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    requested_entry = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-scene-requested"],
            updated_at=_ts(10),
            include_person=True,
        ),
        start_second=10,
        include_attention=True,
    )
    assert requested_entry is not None
    _ = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-scene-current"],
            updated_at=_ts(40),
            include_person=False,
        ),
        start_second=40,
        isolated_recent_events=True,
    )
    redacted_source = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-scene-redacted"],
            updated_at=_ts(70),
            include_person=True,
            degraded_mode="limited",
            affordance_availability=BrainSceneWorldAffordanceAvailability.BLOCKED.value,
        ),
        start_second=70,
        include_attention=True,
    )
    assert redacted_source is not None
    store.redact_autobiographical_entry(
        redacted_source.entry_id,
        redacted_summary="Redacted scene episode.",
        source_event_id="evt-scene-redact",
        reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
        event_context=_multimodal_event_context(
            store,
            session_ids,
            correlation_id="scene-policy-redact",
        ),
    )
    current_entry = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-scene-final"],
            updated_at=_ts(100),
            include_person=False,
        ),
        start_second=100,
        isolated_recent_events=True,
    )
    assert current_entry is not None

    compiler = _build_compiler(store=store, session_ids=session_ids)
    surface = compiler.context_surface_builder.build(
        latest_user_text="Audit the current scene episodes.",
        include_historical_claims=True,
    )
    reply_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="What is active in the scene right now?",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="reply", max_tokens=420),
    )
    planning_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Plan the next scene-aware step.",
        task=BrainContextTask.PLANNING,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="planning", max_tokens=520),
    )
    reflection_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Reflect on recent scene episodes.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )
    operator_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Audit the multimodal scene episode policy.",
        task=BrainContextTask.OPERATOR_AUDIT,
        language=Language.EN,
    )
    governance_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Review scene episode governance state.",
        task=BrainContextTask.GOVERNANCE_REVIEW,
        language=Language.EN,
    )

    reply_scene_selected = [
        item for item in reply_packet.packet_trace.selected_items if item.item_type == "scene_episode"
    ]
    reply_scene_dropped = [
        item for item in reply_packet.packet_trace.dropped_items if item.item_type == "scene_episode"
    ]
    planning_scene_selected = [
        item for item in planning_packet.packet_trace.selected_items if item.item_type == "scene_episode"
    ]
    planning_scene_dropped = [
        item for item in planning_packet.packet_trace.dropped_items if item.item_type == "scene_episode"
    ]
    allowed_scene_selected = [
        item
        for packet in (reflection_packet, operator_packet, governance_packet)
        for item in packet.packet_trace.selected_items
        if item.item_type == "scene_episode"
    ]

    assert surface.scene_episodes
    assert not reply_scene_selected
    assert reply_scene_dropped
    assert all(item.reason == "scene_episode_task_ineligible" for item in reply_scene_dropped)

    assert len(planning_scene_selected) <= 1
    assert planning_scene_selected
    assert planning_scene_selected[0].provenance.get("entry_id") == current_entry.entry_id
    assert planning_scene_selected[0].provenance.get("privacy_class") == "standard"
    assert planning_scene_selected[0].provenance.get("review_state") == "none"
    assert planning_scene_dropped
    assert {
        item.reason for item in planning_scene_dropped
    } >= {"scene_episode_not_current", "scene_episode_redacted"}

    assert allowed_scene_selected
    assert any(item.provenance.get("entry_id") == requested_entry.entry_id for item in allowed_scene_selected)
    assert all(
        item.provenance.get("entry_kind") == "scene_episode"
        and item.provenance.get("source_presence_scope_key") == "browser:presence"
        and item.provenance.get("source_event_ids")
        for item in allowed_scene_selected
    )

    digests = build_context_packet_digest(
        packet_traces={
            "reply": reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None,
            "planning": planning_packet.packet_trace.as_dict() if planning_packet.packet_trace else None,
            "reflection": (
                reflection_packet.packet_trace.as_dict() if reflection_packet.packet_trace else None
            ),
            "operator_audit": (
                operator_packet.packet_trace.as_dict() if operator_packet.packet_trace else None
            ),
            "governance_review": (
                governance_packet.packet_trace.as_dict()
                if governance_packet.packet_trace
                else None
            ),
        }
    )

    assert digests["reply"]["scene_episode_trace"]["suppressed_entry_ids"]
    assert digests["planning"]["scene_episode_trace"]["selected_entry_ids"] == [current_entry.entry_id]
    assert digests["planning"]["scene_episode_trace"]["drop_reason_counts"]["scene_episode_not_current"] >= 1
    assert digests["planning"]["scene_episode_trace"]["drop_reason_counts"]["scene_episode_redacted"] >= 1
    assert digests["operator_audit"]["scene_episode_trace"]["selected_entry_ids"]
    assert digests["governance_review"]["scene_episode_trace"]["selected_entry_ids"]


def test_prediction_packet_policy_is_explicit_and_traceable_by_task(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_goal_updated,
        _append_robot_action_outcome,
        _append_scene_changed,
        _ensure_blocks,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="prediction-policy")
    _ensure_blocks(store)
    _append_body_state(store, session_ids, second=1)
    _append_scene_changed(store, session_ids, second=2)
    _append_goal_created(store, session_ids, second=3)
    _append_scene_changed(store, session_ids, second=4)
    _append_robot_action_outcome(store, session_ids, second=5, accepted=False)
    _append_scene_changed(
        store,
        session_ids,
        second=40,
        include_person=False,
        affordance_availability="blocked",
        camera_fresh=False,
    )

    compiler = _build_compiler(store=store, session_ids=session_ids)
    surface = compiler.context_surface_builder.build(
        latest_user_text="Audit the current predictions.",
        include_historical_claims=True,
    )
    reply_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Reply without predictive leakage.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="reply", max_tokens=420),
    )
    planning_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Plan the next step conservatively.",
        task=BrainContextTask.PLANNING,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="planning", max_tokens=560),
    )
    reflection_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Reflect on predictive drift.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )
    operator_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Audit the predictive packet policy.",
        task=BrainContextTask.OPERATOR_AUDIT,
        language=Language.EN,
    )
    governance_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Review predictive governance traces.",
        task=BrainContextTask.GOVERNANCE_REVIEW,
        language=Language.EN,
    )

    reply_prediction_selected = [
        item for item in reply_packet.packet_trace.selected_items if item.item_type == "prediction"
    ]
    reply_prediction_dropped = [
        item for item in reply_packet.packet_trace.dropped_items if item.item_type == "prediction"
    ]
    planning_prediction_selected = [
        item
        for item in planning_packet.packet_trace.selected_items
        if item.item_type == "prediction"
    ]
    planning_prediction_dropped = [
        item
        for item in planning_packet.packet_trace.dropped_items
        if item.item_type == "prediction"
    ]
    allowed_prediction_selected = [
        item
        for packet in (reflection_packet, operator_packet, governance_packet)
        for item in packet.packet_trace.selected_items
        if item.item_type == "prediction"
    ]

    assert surface.predictive_world_model.active_predictions
    assert surface.predictive_world_model.recent_resolutions

    assert not reply_prediction_selected
    assert reply_prediction_dropped
    assert all(item.reason == "prediction_task_ineligible" for item in reply_prediction_dropped)

    assert planning_prediction_selected
    assert len(planning_prediction_selected) <= 2
    assert {item.section_key for item in planning_prediction_selected} <= {
        "active_state",
        "unresolved_state",
    }
    assert len({item.section_key for item in planning_prediction_selected}) == len(
        planning_prediction_selected
    )
    assert all(item.provenance.get("confidence_band") in {"medium", "high"} for item in planning_prediction_selected)
    assert all(item.provenance.get("resolution_kind") is None for item in planning_prediction_selected)
    assert {
        item.reason for item in planning_prediction_dropped
    } >= {"prediction_resolution_ineligible", "prediction_role_cap_exceeded"}

    assert allowed_prediction_selected
    assert any(item.provenance.get("resolution_kind") == "invalidated" for item in allowed_prediction_selected)
    assert all(
        item.provenance.get("prediction_id")
        and item.provenance.get("prediction_kind")
        and item.provenance.get("subject_kind")
        and item.provenance.get("supporting_event_ids")
        for item in allowed_prediction_selected
    )

    digests = build_context_packet_digest(
        packet_traces={
            "reply": reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None,
            "planning": planning_packet.packet_trace.as_dict() if planning_packet.packet_trace else None,
            "reflection": (
                reflection_packet.packet_trace.as_dict() if reflection_packet.packet_trace else None
            ),
            "operator_audit": (
                operator_packet.packet_trace.as_dict() if operator_packet.packet_trace else None
            ),
            "governance_review": (
                governance_packet.packet_trace.as_dict()
                if governance_packet.packet_trace
                else None
            ),
        }
    )

    assert digests["reply"]["prediction_trace"]["suppressed_prediction_ids"]
    assert len(digests["planning"]["prediction_trace"]["selected_prediction_ids"]) <= 2
    assert digests["planning"]["prediction_trace"]["drop_reason_counts"][
        "prediction_resolution_ineligible"
    ] >= 1
    assert digests["operator_audit"]["prediction_trace"]["selected_prediction_ids"]
    assert digests["operator_audit"]["prediction_trace"]["resolution_kind_counts"]["invalidated"] >= 1
    assert digests["governance_review"]["prediction_trace"]["selected_prediction_ids"]

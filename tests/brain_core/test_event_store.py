import json

from blink.brain.core import (
    BrainCandidateGoal,
    BrainCommitmentRecord,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainCoreStore,
    BrainEventType,
    BrainGoalStep,
    BrainInitiativeClass,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainPresenceSnapshot,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.core.session import resolve_brain_session_ids


def test_brain_core_store_appends_events_and_replays_projections(tmp_path):
    store = BrainCoreStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")

    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="runtime",
        payload={
            "scope_key": "browser:presence",
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                robot_head_enabled=True,
                robot_head_mode="simulation",
                robot_head_available=True,
                vision_enabled=True,
            ).as_dict(),
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "提醒我给妈妈打电话。"},
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="memory",
        payload={"title": "给妈妈打电话", "status": "open"},
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_CALLED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        correlation_id="call_1",
        payload={
            "tool_call_id": "call_1",
            "function_name": "fetch_user_image",
            "arguments": {"question": "我手里拿着什么？"},
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        correlation_id="call_1",
        payload={
            "tool_call_id": "call_1",
            "function_name": "fetch_user_image",
            "result": {"answer": "你手里拿着一个杯子。"},
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        payload={"text": "好的，我会记住，也看到你拿着一个杯子。"},
    )
    store.append_brain_event(
        event_type=BrainEventType.ROBOT_ACTION_OUTCOME,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="tool",
        payload={
            "presence_scope_key": "browser:presence",
            "action_id": "cmd_blink",
            "accepted": True,
            "preview_only": False,
            "summary": "Blink executed cmd_blink.",
            "status": {
                "mode": "simulation",
                "armed": False,
                "available": True,
                "warnings": [],
                "details": {"driver": "simulation"},
            },
        },
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
    assert [event.event_type for event in events] == [
        BrainEventType.BODY_STATE_UPDATED,
        BrainEventType.USER_TURN_TRANSCRIBED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.TOOL_CALLED,
        BrainEventType.TOOL_COMPLETED,
        BrainEventType.ASSISTANT_TURN_ENDED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
    ]
    assert json.loads(events[3].payload_json)["function_name"] == "fetch_user_image"

    body_state = store.get_body_state_projection(scope_key="browser:presence")
    scene = store.get_scene_state_projection(scope_key="browser:presence")
    engagement = store.get_engagement_state_projection(scope_key="browser:presence")
    relationship_state = store.get_relationship_state_projection(scope_key="browser:presence")
    working_context = store.get_working_context_projection(scope_key=session_ids.thread_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id)
    heartbeat = store.get_heartbeat_projection(scope_key=session_ids.thread_id)

    assert body_state.robot_head_mode == "simulation"
    assert body_state.robot_head_last_action == "cmd_blink"
    assert scene.camera_connected is False
    assert engagement.user_present is False
    assert relationship_state.open_commitments == []
    assert working_context.last_user_text == "提醒我给妈妈打电话。"
    assert working_context.last_assistant_text == "好的，我会记住，也看到你拿着一个杯子。"
    assert working_context.last_tool_name == "fetch_user_image"
    assert working_context.last_tool_result == {"answer": "你手里拿着一个杯子。"}
    assert agenda.agenda_seed == "提醒我给妈妈打电话。"
    assert agenda.open_goals == ["给妈妈打电话"]
    assert heartbeat.last_event_type == BrainEventType.ROBOT_ACTION_OUTCOME
    assert heartbeat.last_robot_action == "cmd_blink"

    store.rebuild_brain_projections()

    replayed_body_state = store.get_body_state_projection(scope_key="browser:presence")
    replayed_working_context = store.get_working_context_projection(scope_key=session_ids.thread_id)
    replayed_agenda = store.get_agenda_projection(scope_key=session_ids.thread_id)
    replayed_heartbeat = store.get_heartbeat_projection(scope_key=session_ids.thread_id)

    assert replayed_body_state.robot_head_last_action == "cmd_blink"
    assert replayed_working_context.last_tool_name == "fetch_user_image"
    assert replayed_working_context.last_tool_result == {"answer": "你手里拿着一个杯子。"}
    assert replayed_agenda.open_goals == ["给妈妈打电话"]
    assert replayed_heartbeat.last_event_type == BrainEventType.ROBOT_ACTION_OUTCOME


def test_brain_core_store_rebuilds_autonomy_ledger_projection(tmp_path):
    store = BrainCoreStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-core-1",
        candidate_type="presence_acknowledgement",
        source="perception",
        summary="Person re-entered frame.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
    )

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
        reason="cooldown_active",
        reason_details={"cooldown_key": "presence"},
        expected_reevaluation_condition="after cooldown",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.TIME_REACHED.value,
        expected_reevaluation_condition_details={"cooldown_key": "presence"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )

    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    assert [item.candidate_goal_id for item in ledger.current_candidates] == [candidate.candidate_goal_id]
    assert ledger.recent_entries[-1].reason == "cooldown_active"
    assert (
        ledger.recent_entries[-1].expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.TIME_REACHED.value
    )
    assert ledger.recent_entries[-1].expected_reevaluation_condition_details == {
        "cooldown_key": "presence"
    }
    current_candidate = ledger.candidate(candidate.candidate_goal_id)
    assert current_candidate is not None
    assert current_candidate.expected_reevaluation_condition == "after cooldown"
    assert (
        current_candidate.expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.TIME_REACHED.value
    )
    assert current_candidate.expected_reevaluation_condition_details == {
        "cooldown_key": "presence"
    }

    store.rebuild_brain_projections()

    rebuilt = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    assert rebuilt.as_dict() == ledger.as_dict()


def test_brain_core_store_records_reevaluation_trigger_without_mutating_autonomy_projection(tmp_path):
    store = BrainCoreStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-core-reeval",
        candidate_type="presence_acknowledgement",
        source="runtime",
        summary="Held candidate.",
        goal_family="environment",
        urgency=0.7,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    before = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()

    event = store.append_director_reevaluation_triggered(
        trigger=BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            summary="User turn closed.",
            details={"turn": "user"},
            source_event_type=BrainEventType.USER_TURN_ENDED,
            source_event_id="event-user-turn-ended",
            ts="2026-04-19T10:05:00+00:00",
        ),
        candidate_goal_ids=[candidate.candidate_goal_id],
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )

    after = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()

    assert event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
    assert event.payload["candidate_goal_ids"] == [candidate.candidate_goal_id]
    assert event.payload["trigger"]["kind"] == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    assert after == before


def test_commitment_wake_trigger_roundtrip_serialization():
    trigger = BrainCommitmentWakeTrigger(
        commitment_id="commitment-1",
        wake_kind=BrainWakeConditionKind.USER_RESPONSE.value,
        summary="Matched one new user response.",
        details={"pass_kind": "turn_end"},
        source_event_type=BrainEventType.USER_TURN_ENDED,
        source_event_id="event-user-turn-ended",
        ts="2026-04-19T10:05:00+00:00",
    )
    routing = BrainCommitmentWakeRoutingDecision(
        route_kind=BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value,
        summary="Route through the bounded candidate policy.",
        details={"candidate_type": "commitment_wake_user_response"},
    )

    assert BrainCommitmentWakeTrigger.from_dict(trigger.as_dict()) == trigger
    assert BrainCommitmentWakeRoutingDecision.from_dict(routing.as_dict()) == routing


def test_plan_proposal_roundtrip_serialization():
    proposal = BrainPlanProposal(
        plan_proposal_id="plan-proposal-1",
        goal_id="goal-1",
        commitment_id="commitment-1",
        source=BrainPlanProposalSource.REPAIR.value,
        summary="Repair plan for goal-1.",
        current_plan_revision=1,
        plan_revision=2,
        review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
        steps=[
            BrainGoalStep(
                capability_id="robot_head.return_neutral",
                updated_at="2026-04-20T10:00:00+00:00",
            )
        ],
        preserved_prefix_count=1,
        assumptions=["the head can return to neutral later"],
        missing_inputs=[],
        supersedes_plan_proposal_id="plan-proposal-0",
        details={"intent": "robot_head.sequence", "risk": "operator-visible"},
        created_at="2026-04-20T10:00:00+00:00",
    )
    decision = BrainPlanProposalDecision(
        summary="Adopted repaired plan tail.",
        reason="repair_applied",
        details={"commitment_id": "commitment-1"},
    )

    hydrated_proposal = BrainPlanProposal.from_dict(proposal.as_dict())

    assert hydrated_proposal is not None
    assert hydrated_proposal.plan_proposal_id == proposal.plan_proposal_id
    assert hydrated_proposal.review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
    assert hydrated_proposal.steps[0].capability_id == "robot_head.return_neutral"
    assert hydrated_proposal.assumptions == ["the head can return to neutral later"]
    assert hydrated_proposal.supersedes_plan_proposal_id == "plan-proposal-0"
    assert BrainPlanProposalDecision.from_dict(decision.as_dict()) == decision


def test_brain_core_store_records_commitment_wake_trigger_without_mutating_autonomy_projection(tmp_path):
    store = BrainCoreStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-core-wake",
        candidate_type="commitment_wake_thread_idle",
        source="commitment",
        summary="Revisit deferred commitment: Follow up later",
        goal_family="conversation",
        urgency=0.7,
        confidence=1.0,
        initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    before = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()

    commitment = BrainCommitmentRecord(
        commitment_id="commitment-core-1",
        scope_type="thread",
        scope_id=session_ids.thread_id,
        title="Follow up later",
        goal_family="conversation",
        intent="narrative.commitment",
        status="deferred",
        details={"summary": "Need to revisit this."},
        current_goal_id="goal-1",
        plan_revision=2,
        resume_count=1,
    )
    wake_condition = BrainWakeCondition(
        kind=BrainWakeConditionKind.THREAD_IDLE.value,
        summary="Wake when the thread is idle.",
    )
    trigger = BrainCommitmentWakeTrigger(
        commitment_id=commitment.commitment_id,
        wake_kind=wake_condition.kind,
        summary="Matched durable commitment wake: thread_idle.",
        details={"candidate_goal_id": candidate.candidate_goal_id, "pass_kind": "startup"},
        ts="2026-04-19T10:06:00+00:00",
    )
    routing = BrainCommitmentWakeRoutingDecision(
        route_kind=BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value,
        summary="Route this wake through bounded candidate policy.",
        details={
            "candidate_goal_id": candidate.candidate_goal_id,
            "candidate_type": candidate.candidate_type,
        },
    )
    event = store.append_commitment_wake_triggered(
        commitment=commitment,
        wake_condition=wake_condition,
        trigger=trigger,
        routing=routing,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="executive",
    )

    after = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id).as_dict()

    assert event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    assert event.payload["commitment"]["commitment_id"] == commitment.commitment_id
    assert event.payload["wake_condition"]["kind"] == BrainWakeConditionKind.THREAD_IDLE.value
    assert event.payload["trigger"]["details"]["candidate_goal_id"] == candidate.candidate_goal_id
    assert event.payload["routing"]["route_kind"] == BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value
    assert after == before


def test_brain_core_store_records_planning_events_without_mutating_agenda_projection(tmp_path):
    store = BrainCoreStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": {
                "goal_id": "goal-planning-1",
                "title": "Plan a robot sequence",
                "intent": "robot_head.sequence",
                "source": "test",
                "goal_family": "environment",
                "status": "open",
                "details": {},
                "steps": [],
                "active_step_index": None,
                "recovery_count": 0,
                "planning_requested": False,
                "plan_revision": 1,
                "resume_count": 0,
                "last_summary": None,
                "last_error": None,
                "created_at": "2026-04-20T10:00:00+00:00",
                "updated_at": "2026-04-20T10:00:00+00:00",
            }
        },
    )
    before = store.get_agenda_projection(scope_key=session_ids.thread_id).as_dict()
    proposal = BrainPlanProposal(
        plan_proposal_id="plan-proposal-2",
        goal_id="goal-planning-1",
        commitment_id=None,
        source=BrainPlanProposalSource.DETERMINISTIC_PLANNER.value,
        summary="Deterministic planner expanded robot_head.sequence.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[],
        preserved_prefix_count=0,
        assumptions=[],
        missing_inputs=[],
        details={"intent": "robot_head.sequence"},
        created_at="2026-04-20T10:01:00+00:00",
    )
    decision = BrainPlanProposalDecision(
        summary="Adopted deterministic plan.",
        reason="deterministic_plan_available",
        details={"intent": "robot_head.sequence"},
    )

    proposed = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="executive",
    )
    adopted = store.append_planning_adopted(
        proposal=proposal,
        decision=decision,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="executive",
        causal_parent_id=proposed.event_id,
    )
    rejected = store.append_planning_rejected(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary="Rejected plan.",
            reason="operator_declined",
            details={"intent": "robot_head.sequence"},
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="executive",
        causal_parent_id=proposed.event_id,
    )
    after = store.get_agenda_projection(scope_key=session_ids.thread_id).as_dict()

    assert proposed.event_type == BrainEventType.PLANNING_PROPOSED
    assert proposed.payload["proposal"]["plan_proposal_id"] == proposal.plan_proposal_id
    assert proposed.payload["proposal"]["review_policy"] == BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
    assert adopted.event_type == BrainEventType.PLANNING_ADOPTED
    assert adopted.payload["decision"]["reason"] == "deterministic_plan_available"
    assert rejected.event_type == BrainEventType.PLANNING_REJECTED
    assert rejected.payload["decision"]["reason"] == "operator_declined"
    assert after == before

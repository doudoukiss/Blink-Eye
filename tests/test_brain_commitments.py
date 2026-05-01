import asyncio
from dataclasses import replace

import pytest

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    build_embodied_capability_registry,
)
from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
)
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver, MockDriver
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


class _WakePolicySurfaceBuilder:
    """Wrap the real surface builder with a bounded degraded-scene override."""

    def __init__(self, *, store: BrainStore, session_ids, capability_registry, scene_mode: str):
        self.scene_mode = scene_mode
        self._builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=capability_registry,
        )

    def build(self, **kwargs):
        base = self._builder.build(**kwargs)
        return replace(
            base,
            scene_world_state=replace(
                base.scene_world_state,
                degraded_mode=self.scene_mode,
                degraded_reason_codes=(
                    ["scene_stale"] if self.scene_mode == "limited" else []
                ),
            ),
        )


def _build_robot_executive(*, store: BrainStore, session_ids, driver):
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(action_engine=action_engine),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=build_embodied_capability_registry(action_engine=action_engine),
        ),
    )
    return executive, controller


def _build_simple_executive(*, store: BrainStore, session_ids):
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )


def test_commitment_lifecycle_updates_projection_and_active_tasks(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(
            action_engine=EmbodiedActionEngine(
                library=EmbodiedActionLibrary.build_default(),
                controller=RobotHeadController(
                    catalog=build_default_robot_head_catalog(),
                    driver=MockDriver(),
                ),
                store=store,
                session_resolver=lambda: session_ids,
                presence_scope_key="browser:presence",
            )
        ),
    )

    goal_id = executive.create_commitment_goal(
        title="Follow up on project A",
        intent="narrative.commitment",
        source="memory",
        details={"details": "Need a status update"},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    deferred = executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for user confirmation.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                summary="Resume after the user confirms.",
            ),
        ],
    )
    resumed = executive.resume_commitment(commitment_id=commitment.commitment_id)
    cancelled = executive.cancel_commitment(commitment_id=commitment.commitment_id)

    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    projection = store.get_commitment_projection(
        scope_key=f"{session_ids.agent_id}:{session_ids.user_id}"
    )
    tasks = store.active_tasks(user_id=session_ids.user_id, limit=8)

    assert goal_id == commitment.current_goal_id
    assert deferred.status == BrainCommitmentStatus.DEFERRED.value
    assert resumed.resume_count == 1
    assert cancelled.status == BrainCommitmentStatus.CANCELLED.value
    assert agenda.cancelled_goals == ["Follow up on project A"]
    assert projection.recent_terminal_commitments[0].title == "Follow up on project A"
    assert tasks == []


@pytest.mark.asyncio
async def test_condition_cleared_wake_resumes_blocked_commitment_after_boundary(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    driver = FaultInjectionDriver(busy=True)
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=driver)

    executive.create_commitment_goal(
        title="Do the blink sequence",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )

    first = await executive.run_turn_end_pass()
    await asyncio.sleep(0.4)
    blocked_result = await executive.run_turn_end_pass()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    projection = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )

    assert first.progressed is True
    assert blocked_result.progressed is True
    assert commitment.status == BrainCommitmentStatus.BLOCKED.value
    assert commitment.wake_conditions[0].kind == BrainWakeConditionKind.CONDITION_CLEARED.value
    assert commitment.scope_type == BrainCommitmentScopeType.THREAD.value
    assert projection.blocked_commitments[0].title == "Do the blink sequence"

    driver.busy = False
    resumed_result = await executive.run_turn_end_pass()
    resumed = store.get_executive_commitment(commitment_id=commitment.commitment_id)
    completed_result = await executive.run_turn_end_pass()
    refreshed = store.get_executive_commitment(commitment_id=commitment.commitment_id)
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=16,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert resumed_result.progressed is True
    assert resumed is not None
    assert resumed.status == BrainCommitmentStatus.ACTIVE.value
    assert resumed.resume_count == 1
    assert completed_result.progressed is True
    assert refreshed is not None
    assert refreshed.status == BrainCommitmentStatus.COMPLETED.value
    assert any(
        event.payload["routing"]["route_kind"] == "resume_direct" for event in wake_events
    )
    await controller.close()


@pytest.mark.asyncio
async def test_condition_cleared_wake_resumes_on_startup_recovery(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    driver = FaultInjectionDriver(busy=True)
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=driver)

    executive.create_commitment_goal(
        title="Recover after restart-style wake",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )

    await executive.run_turn_end_pass()
    await asyncio.sleep(0.4)
    await executive.run_turn_end_pass()
    blocked = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    assert blocked.status == BrainCommitmentStatus.BLOCKED.value

    driver.busy = False
    startup_result = await executive.run_startup_pass()
    resumed = store.get_executive_commitment(commitment_id=blocked.commitment_id)
    completion_result = await executive.run_turn_end_pass()
    refreshed = store.get_executive_commitment(commitment_id=blocked.commitment_id)

    assert startup_result.progressed is True
    assert resumed is not None
    assert resumed.status == BrainCommitmentStatus.ACTIVE.value
    assert resumed.resume_count == 1
    assert completion_result.progressed is True
    assert refreshed is not None
    assert refreshed.status == BrainCommitmentStatus.COMPLETED.value
    await controller.close()


@pytest.mark.asyncio
async def test_commitment_repair_keeps_completed_prefix_and_new_tail(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=MockDriver())

    goal_id = executive.create_commitment_goal(
        title="Repairable sequence",
        intent="robot_head.sequence",
        source="interpreter",
        details={
            "capabilities": [
                {"capability_id": "robot_head.blink"},
                {"capability_id": "robot_head.look_left"},
            ]
        },
        goal_status=BrainGoalStatus.OPEN.value,
    )

    await executive.run_once()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    repaired = executive.repair_commitment(
        commitment_id=commitment.commitment_id,
        capabilities=[{"capability_id": "robot_head.return_neutral"}],
    )
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    repaired_goal = agenda.goal(goal_id)

    assert repaired.plan_revision == 2
    assert repaired.scope_type == BrainCommitmentScopeType.THREAD.value
    assert repaired_goal is not None
    assert repaired_goal.plan_revision == 2
    assert repaired_goal.steps[0].capability_id == "robot_head.blink"
    assert repaired_goal.steps[0].status == "completed"
    assert repaired_goal.steps[1].capability_id == "robot_head.return_neutral"
    assert repaired_goal.steps[1].status == "pending"
    recent_events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=12,
            )
        )
    )
    repair_events = [
        event
        for event in recent_events
        if event.event_type
        in {
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
                BrainEventType.GOAL_REPAIRED,
            }
        ]
    repair_events = repair_events[-3:]
    proposal_id = repair_events[0].payload["proposal"]["plan_proposal_id"]

    executive.resume_commitment(commitment_id=commitment.commitment_id)
    result = await executive.run_turn_end_pass()
    final_commitment = store.get_executive_commitment(commitment_id=commitment.commitment_id)

    assert [event.event_type for event in repair_events] == [
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.GOAL_REPAIRED,
    ]
    assert repair_events[1].payload["proposal"]["plan_proposal_id"] == proposal_id
    assert repair_events[1].causal_parent_id == repair_events[0].event_id
    assert repair_events[2].causal_parent_id == repair_events[1].event_id
    assert result.progressed is True
    assert final_commitment is not None
    assert final_commitment.status == BrainCommitmentStatus.COMPLETED.value
    await controller.close()


@pytest.mark.asyncio
async def test_deterministic_planning_emits_explicit_plan_proposal_trail(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=MockDriver())

    goal_id = executive.create_commitment_goal(
        title="Plan a blink sequence",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )

    result = await executive.run_once()
    recent_events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=12,
            )
        )
    )
    planning_events = [
        event
        for event in recent_events
        if event.event_type
        in {
            BrainEventType.PLANNING_REQUESTED,
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.GOAL_UPDATED,
        }
    ]
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)

    assert result.progressed is True
    assert goal is not None
    assert goal.plan_revision == 1
    assert goal.steps[0].capability_id == "robot_head.blink"
    assert [event.event_type for event in planning_events] == [
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.GOAL_UPDATED,
    ]
    assert planning_events[1].causal_parent_id == planning_events[0].event_id
    assert planning_events[2].causal_parent_id == planning_events[1].event_id
    assert planning_events[3].causal_parent_id == planning_events[2].event_id
    assert (
        planning_events[2].payload["decision"]["reason"] == "deterministic_plan_available"
    )
    await controller.close()


@pytest.mark.asyncio
async def test_commitments_survive_restart_and_run_on_startup_pass(tmp_path):
    db_path = tmp_path / "brain.db"
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")

    driver_one = MockDriver()
    runtime_one = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        robot_head_controller=RobotHeadController(
            catalog=build_default_robot_head_catalog(),
            driver=driver_one,
        ),
        brain_db_path=db_path,
    )
    runtime_one.executive.create_commitment_goal(
        title="Persistent blink",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )
    runtime_one.close()

    driver_two = MockDriver()
    runtime_two = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        robot_head_controller=RobotHeadController(
            catalog=build_default_robot_head_catalog(),
            driver=driver_two,
        ),
        brain_db_path=db_path,
    )

    result = await runtime_two.executive.run_startup_pass()
    commitments = runtime_two.store.list_executive_commitments(user_id=session_ids.user_id, limit=8)

    assert result.progressed is True
    assert commitments[0].status == BrainCommitmentStatus.COMPLETED.value
    runtime_two.close()


@pytest.mark.asyncio
async def test_thread_idle_wake_produces_candidate_without_resuming_commitment(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Follow up once the thread is idle",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Need to follow up later."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for the thread to settle.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when the thread is idle.",
            )
        ],
    )

    result = await executive.run_startup_pass()
    refreshed = store.get_executive_commitment(commitment_id=commitment.commitment_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert result.progressed is True
    assert refreshed is not None
    assert refreshed.status == BrainCommitmentStatus.DEFERRED.value
    assert wake_events
    assert wake_events[0].payload["routing"]["route_kind"] == "propose_candidate"
    assert any(goal.intent == "autonomy.commitment_wake_thread_idle" for goal in agenda.goals)


@pytest.mark.asyncio
async def test_user_response_wake_requires_newer_user_turn(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Resume after a user reply",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Need a fresh user response."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for a reply.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.USER_RESPONSE.value,
                summary="Wake after the next user response.",
            )
        ],
    )

    first = await executive.run_turn_end_pass()
    agenda_before = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    assert first.progressed is False
    assert len(agenda_before.goals) == 1

    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"text": "I replied."},
    )
    second = await executive.run_turn_end_pass()
    agenda_after = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert second.progressed is True
    assert wake_events
    assert wake_events[0].payload["trigger"]["wake_kind"] == BrainWakeConditionKind.USER_RESPONSE.value
    assert any(goal.intent == "autonomy.commitment_wake_user_response" for goal in agenda_after.goals)


@pytest.mark.asyncio
async def test_explicit_resume_commitment_does_not_auto_produce_candidate(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Resume only explicitly",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "No automatic wake."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                summary="Resume only explicitly.",
            )
        ],
    )

    result = await executive.run_startup_pass()
    autonomy_events = store.recent_autonomy_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
    )
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert result.progressed is False
    assert wake_events == []
    assert autonomy_events == []


@pytest.mark.asyncio
async def test_wake_router_keeps_waiting_when_same_wake_candidate_is_already_current(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Follow up once idle",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Need to follow up later."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for the thread to settle.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when the thread is idle.",
            )
        ],
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-existing-wake",
            candidate_type="commitment_wake_thread_idle",
            source=BrainCandidateGoalSource.COMMITMENT.value,
            summary="Revisit deferred commitment: Follow up once idle",
            goal_family=BrainGoalFamily.CONVERSATION.value,
            urgency=0.7,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
            dedupe_key=f"{commitment.commitment_id}:{BrainWakeConditionKind.THREAD_IDLE.value}",
            cooldown_key=(
                f"{session_ids.thread_id}:commitment:{commitment.commitment_id}:"
                f"{BrainWakeConditionKind.THREAD_IDLE.value}"
            ),
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    result = await executive.run_commitment_wake_router(boundary_kind="startup_recovery")
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert result.progressed is True
    assert result.route_kind == "keep_waiting"
    assert result.reason == "candidate_already_current"
    assert wake_events[0].payload["routing"]["details"]["reason"] == "candidate_already_current"


@pytest.mark.asyncio
async def test_wake_router_rotates_families_without_starvation(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Conversation follow-up",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Conversation wake."},
    )
    conversation_commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=conversation_commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for an idle gap.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when idle.",
            )
        ],
    )

    executive.create_commitment_goal(
        title="Maintenance review",
        intent="memory.maintenance.refresh",
        source="maintenance",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        details={"summary": "Maintenance wake."},
    )
    commitments = store.list_executive_commitments(user_id=session_ids.user_id, limit=8)
    maintenance_commitment = next(
        record for record in commitments if record.goal_family == BrainGoalFamily.MEMORY_MAINTENANCE.value
    )
    executive.defer_commitment(
        commitment_id=maintenance_commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for an idle gap.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when idle.",
            )
        ],
    )

    first = await executive.run_commitment_wake_router(boundary_kind="startup_recovery")
    second = await executive.run_commitment_wake_router(boundary_kind="startup_recovery")
    wake_events = [
        event
        for event in reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=16,
            )
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert first.progressed is True
    assert second.progressed is True
    assert first.matched_commitment_id != second.matched_commitment_id
    assert wake_events[0].payload["commitment"]["goal_family"] != wake_events[1].payload["commitment"]["goal_family"]


@pytest.mark.asyncio
async def test_policy_coupled_wake_router_keeps_waiting_when_scene_is_limited(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="wake-policy")
    capability_registry = CapabilityRegistry()
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=capability_registry,
        context_surface_builder=_WakePolicySurfaceBuilder(
            store=store,
            session_ids=session_ids,
            capability_registry=capability_registry,
            scene_mode="limited",
        ),
    )

    executive.create_commitment_goal(
        title="Wake policy hold",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Revisit after the thread goes idle."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Wait for an idle gap.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when the thread is idle.",
            )
        ],
    )

    result = await executive.run_commitment_wake_router(boundary_kind="startup_recovery")
    wake_event = next(
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    )

    assert result.progressed is True
    assert result.route_kind == BrainCommitmentWakeRouteKind.KEEP_WAITING.value
    assert result.reason == "policy_conservative_deferral"
    assert "policy_conservative_deferral" in result.reason_codes
    assert "scene_limited" in result.reason_codes
    assert result.executive_policy is not None
    assert result.executive_policy["action_posture"] == "defer"
    assert wake_event.payload["routing"]["details"]["original_route_kind"] == (
        BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value
    )
    assert wake_event.payload["routing"]["executive_policy"]["approval_requirement"] == "none"


@pytest.mark.asyncio
async def test_goal_terminal_boundary_can_route_thread_idle_wake(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=MockDriver())

    executive.create_commitment_goal(
        title="Wake after the active environment goal clears",
        intent="robot_head.sequence",
        source="memory",
        details={"capabilities": [{"capability_id": "robot_head.look_left"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )
    deferred_commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=deferred_commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Wait until the environment family is idle again.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when the thread is idle.",
            )
        ],
    )
    executive.create_goal(
        title="Immediate environment action",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
    )

    result = await executive.run_once()
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    wake_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=16,
        )
        if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    ]

    assert result.progressed is True
    assert wake_events
    assert wake_events[0].payload["routing"]["route_kind"] == "propose_candidate"
    assert wake_events[0].payload["trigger"]["details"]["boundary_kind"] == "goal_terminal"
    assert any(goal.intent == "autonomy.commitment_wake_thread_idle" for goal in agenda.goals)
    await controller.close()


@pytest.mark.asyncio
async def test_user_response_wake_can_record_non_action_when_thread_is_busy(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = _build_simple_executive(store=store, session_ids=session_ids)

    executive.create_commitment_goal(
        title="Wait for a reply before revisiting",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Needs a follow-up after reply."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for the next reply.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.USER_RESPONSE.value,
                summary="Wake after the next user response.",
            )
        ],
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"text": "Here is the reply."},
    )
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    await executive.run_turn_end_pass()
    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    non_action = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )

    assert non_action.reason == "assistant_turn_open"
    assert [candidate.candidate_type for candidate in ledger.current_candidates] == [
        "commitment_wake_user_response"
    ]


@pytest.mark.asyncio
async def test_executive_explains_blocked_and_deferred_agenda_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=MockDriver())

    executive.create_commitment_goal(
        title="Deferred follow-up",
        intent="narrative.commitment",
        source="memory",
        details={"details": "Need to circle back later"},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting on user feedback.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                summary="Resume manually.",
            )
        ],
    )
    summary = executive.explain_agenda_state()

    assert "Blocked" in summary or "Deferred" in summary
    await controller.close()


def test_selective_promotion_keeps_one_shot_environment_goals_transient(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(
            action_engine=EmbodiedActionEngine(
                library=EmbodiedActionLibrary.build_default(),
                controller=RobotHeadController(
                    catalog=build_default_robot_head_catalog(),
                    driver=MockDriver(),
                ),
                store=store,
                session_resolver=lambda: session_ids,
                presence_scope_key="browser:presence",
            )
        ),
    )

    executive.create_goal(
        title="Blink once",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
    )

    commitments = store.list_executive_commitments(user_id=session_ids.user_id, limit=8)

    assert commitments == []


def test_agent_scoped_memory_maintenance_commitment_survives_in_session_projection(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(
            action_engine=EmbodiedActionEngine(
                library=EmbodiedActionLibrary.build_default(),
                controller=RobotHeadController(
                    catalog=build_default_robot_head_catalog(),
                    driver=MockDriver(),
                ),
                store=store,
                session_resolver=lambda: session_ids,
                presence_scope_key="browser:presence",
            )
        ),
    )

    goal_id = executive.create_commitment_goal(
        title="Refresh continuity audit",
        intent="memory.maintenance.refresh",
        source="maintenance",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        details={"spans_turns": True, "survive_restart": True},
    )

    commitments = store.list_executive_commitments(user_id=session_ids.user_id, limit=8)
    projection = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )

    assert commitments[0].scope_type == BrainCommitmentScopeType.AGENT.value
    assert commitments[0].current_goal_id == goal_id
    assert projection.active_commitments[0].scope_type == BrainCommitmentScopeType.AGENT.value


@pytest.mark.asyncio
async def test_repair_keeps_blocked_commitment_deferred_until_explicit_resume(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    driver = FaultInjectionDriver(busy=True)
    executive, controller = _build_robot_executive(store=store, session_ids=session_ids, driver=driver)

    executive.create_commitment_goal(
        title="Blocked repairable sequence",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )

    await executive.run_turn_end_pass()
    await asyncio.sleep(0.4)
    await executive.run_turn_end_pass()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    repaired = executive.repair_commitment(
        commitment_id=commitment.commitment_id,
        capabilities=[{"capability_id": "robot_head.return_neutral"}],
    )
    before_resume = await executive.run_turn_end_pass()

    assert repaired.status == BrainCommitmentStatus.DEFERRED.value
    assert repaired.wake_conditions[0].kind == BrainWakeConditionKind.EXPLICIT_RESUME.value
    assert before_resume.progressed is False

    driver.busy = False
    executive.resume_commitment(commitment_id=commitment.commitment_id)
    after_resume = await executive.run_turn_end_pass()
    refreshed = store.get_executive_commitment(commitment_id=commitment.commitment_id)

    assert after_resume.progressed is True
    assert refreshed is not None
    assert refreshed.status == BrainCommitmentStatus.COMPLETED.value
    await controller.close()

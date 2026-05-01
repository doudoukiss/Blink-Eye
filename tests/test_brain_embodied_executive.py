import pytest

from blink.brain.autonomy import BrainInitiativeClass
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import BrainGoal, BrainGoalFamily, BrainGoalStatus, BrainGoalStep
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver
from blink.embodiment.robot_head.simulation import RobotHeadSimulationConfig, SimulationDriver
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def _build_runtime(tmp_path, *, client_id: str, driver):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        robot_head_controller=controller,
        brain_db_path=tmp_path / f"{client_id}.db",
    )
    return runtime, controller, session_ids


def _create_robot_goal(runtime: BrainRuntime, *, title: str) -> str:
    return runtime.executive.create_commitment_goal(
        title=title,
        intent="robot_head.sequence",
        source="test",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={
            "survive_restart": True,
            "capabilities": ["robot_head.look_left"],
        },
    )


@pytest.mark.asyncio
async def test_embodied_coordinator_records_simulation_dispatch_and_trace(tmp_path):
    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="embodied-simulation",
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
        ),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Look left through the coordinator")

        planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        cycle_result = await runtime.executive.run_once()
        projection = runtime.store.build_embodied_executive_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=runtime.presence_scope_key,
        )
        shell_snapshot = runtime.shell.snapshot()
        shell_digest = runtime.shell.runtime_shell_digest()["embodied_inspection"]
        events = list(
            reversed(
                runtime.store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=64,
                )
            )
        )

        capability_requested = next(
            event for event in events if event.event_type == BrainEventType.CAPABILITY_REQUESTED
        )
        capability_completed = next(
            event for event in events if event.event_type == BrainEventType.CAPABILITY_COMPLETED
        )
        robot_action_outcome = next(
            event for event in events if event.event_type == BrainEventType.ROBOT_ACTION_OUTCOME
        )
        embodied_completed = next(
            event
            for event in events
            if event.event_type == BrainEventType.EMBODIED_DISPATCH_COMPLETED
        )

        assert planning_result.outcome == "auto_adopted"
        assert cycle_result.progressed is True
        assert projection.current_intent is not None
        assert projection.current_intent.intent_kind == "execute_action"
        assert projection.current_intent.status == "succeeded"
        assert projection.recent_execution_traces
        assert projection.recent_execution_traces[0].status == "succeeded"
        assert (
            projection.recent_execution_traces[0].capability_request_event_id
            == capability_requested.event_id
        )
        assert (
            projection.recent_execution_traces[0].capability_result_event_id
            == capability_completed.event_id
        )
        assert (
            projection.recent_execution_traces[0].robot_action_event_id
            == robot_action_outcome.event_id
        )
        assert embodied_completed.causal_parent_id == capability_completed.event_id
        assert shell_snapshot.current_embodied_intent["intent_kind"] == "execute_action"
        assert shell_snapshot.recent_embodied_execution_traces[0]["status"] == "succeeded"
        assert shell_digest["current_low_level_executor"] == "simulation"
        assert shell_digest["recent_execution_traces"][0]["status"] == "succeeded"
    finally:
        await controller.close()
        runtime.close()


@pytest.mark.asyncio
async def test_embodied_coordinator_records_bounded_recovery_after_execution_fault(tmp_path):
    driver = FaultInjectionDriver(
        wrapped=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation-fault"),
        )
    )
    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="embodied-recovery",
        driver=driver,
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Recover after a busy execution fault")

        planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        driver.busy = True
        cycle_result = await runtime.executive.run_once()
        projection = runtime.store.build_embodied_executive_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=runtime.presence_scope_key,
        )
        shell_snapshot = runtime.shell.snapshot()
        recent_events = runtime.store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=64,
        )

        assert planning_result.outcome == "auto_adopted"
        assert cycle_result.progressed is True
        assert projection.current_intent is not None
        assert projection.current_intent.intent_kind == "recover_safe_state"
        assert projection.recent_execution_traces
        assert projection.recent_execution_traces[0].status == "failed"
        assert projection.recent_execution_traces[0].recovery_action_id == "cmd_return_neutral"
        assert projection.recent_recoveries
        assert projection.recent_recoveries[0].action_id == "cmd_return_neutral"
        assert shell_snapshot.current_embodied_intent["intent_kind"] == "recover_safe_state"
        assert shell_snapshot.recent_embodied_recoveries[0]["action_id"] == "cmd_return_neutral"
        assert any(
            event.event_type == BrainEventType.CAPABILITY_FAILED for event in recent_events
        )
        assert any(
            event.event_type == BrainEventType.EMBODIED_RECOVERY_RECORDED
            for event in recent_events
        )
    finally:
        await controller.close()
        runtime.close()


@pytest.mark.asyncio
async def test_non_robot_head_goals_do_not_route_through_embodied_coordinator(tmp_path):
    class SpyCoordinator:
        def __init__(self):
            self.called = 0

        async def prepare_dispatch(self, **kwargs):
            self.called += 1
            raise AssertionError("Coordinator should not be called for non-robot goals.")

    store = BrainStore(path=tmp_path / "embodied-non-robot.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="embodied-non-robot")
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
                robot_head_enabled=False,
                robot_head_mode="none",
                robot_head_available=False,
                vision_enabled=False,
                vision_connected=False,
            ).as_dict(),
        },
    )
    goal = BrainGoal(
        goal_id="goal-non-robot",
        title="Review memory health without the embodied coordinator",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        status=BrainGoalStatus.OPEN.value,
        steps=[BrainGoalStep(capability_id="maintenance.review_memory_health")],
        details={
            "autonomy": {
                "initiative_class": BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value
            }
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal.as_dict()},
        correlation_id=goal.goal_id,
    )
    coordinator = SpyCoordinator()
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
        embodied_coordinator=coordinator,
    )

    result = await executive.run_once()
    recent_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=32,
    )

    assert result.progressed is True
    assert coordinator.called == 0
    assert any(event.event_type == BrainEventType.CAPABILITY_COMPLETED for event in recent_events)
    assert not any(event.event_type.startswith("embodied.") for event in recent_events)

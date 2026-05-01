import pytest

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    build_embodied_capability_registry,
)
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive, BrainGoalStatus
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver


@pytest.mark.asyncio
async def test_brain_executive_records_retryable_robot_head_failures_and_recovers(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    driver = FaultInjectionDriver(busy=True)
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
    )

    goal_id = executive.create_goal(
        title="眨眼一次",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
    )

    first_outcome = await executive.run_once()
    goal_after_first_run = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    ).goal(goal_id)

    assert first_outcome.progressed is True
    assert goal_after_first_run is not None
    assert goal_after_first_run.status == BrainGoalStatus.RETRY.value
    assert goal_after_first_run.steps[0].status == "retry"

    driver.busy = False

    second_outcome = await executive.run_until_quiescent(max_iterations=4)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    feedback_events = [
        event
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=24,
        )
        if event.event_type == BrainEventType.CRITIC_FEEDBACK
    ]

    assert second_outcome.progressed is True
    assert goal is not None
    assert goal.status == BrainGoalStatus.COMPLETED.value
    assert goal.steps[0].attempts == 2
    assert goal.steps[0].status == "completed"
    assert feedback_events
    assert feedback_events[0].payload["recovery"]["decision"] == "retry"
    await controller.close()

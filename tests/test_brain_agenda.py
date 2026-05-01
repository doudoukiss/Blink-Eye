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
from blink.embodiment.robot_head.drivers import MockDriver


@pytest.mark.asyncio
async def test_brain_executive_plans_and_executes_robot_head_sequence_goal(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    driver = MockDriver()
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
        title="先眨眼再回到中位",
        intent="robot_head.sequence",
        source="interpreter",
        details={
            "capabilities": [
                {"capability_id": "robot_head.blink"},
                {"capability_id": "robot_head.return_neutral"},
            ]
        },
    )

    outcome = await executive.run_until_quiescent(max_iterations=6)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=24,
            )
        )
    )

    assert outcome.progressed is True
    assert goal is not None
    assert goal.status == BrainGoalStatus.COMPLETED.value
    assert [step.capability_id for step in goal.steps] == [
        "robot_head.blink",
        "robot_head.return_neutral",
    ]
    assert [step.status for step in goal.steps] == ["completed", "completed"]
    assert [plan.resolved_name for plan in driver.executed_plans] == ["blink", "neutral"]
    expected_event_prefix = [
        BrainEventType.GOAL_CREATED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.GOAL_COMPLETED,
    ]
    event_types = [event.event_type for event in events]
    assert event_types[: len(expected_event_prefix)] == expected_event_prefix
    await controller.close()

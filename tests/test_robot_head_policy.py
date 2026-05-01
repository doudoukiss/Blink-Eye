import asyncio

import pytest

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    EmbodiedCapabilityDispatcher,
    build_embodied_capability_registry,
)
from blink.brain.events import BrainEventType
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver
from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor
from blink.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from blink.processors.frame_processor import FrameDirection
from blink.tests.utils import SleepFrame, run_test


@pytest.mark.asyncio
async def test_embodiment_policy_translates_turn_frames_to_robot_head_commands(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = MockDriver()
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="voice")
    controller = RobotHeadController(catalog=catalog, driver=driver)
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
    )
    processor = EmbodimentPolicyProcessor(
        action_dispatcher=EmbodiedCapabilityDispatcher(
            action_engine=action_engine,
            capability_registry=build_embodied_capability_registry(action_engine=action_engine),
        ),
        idle_timeout_secs=0.01,
    )

    await run_test(
        processor,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
            EndFrame(),
        ],
        frames_to_send_direction=FrameDirection.DOWNSTREAM,
        send_end_frame=False,
    )

    resolved_names = [plan.resolved_name for plan in driver.executed_plans]
    assert "neutral" in resolved_names
    assert resolved_names.count("listen_engage") == 1
    assert "listen_attentively" in resolved_names
    assert "thinking_shift" in resolved_names
    assert "thinking" in resolved_names
    assert "acknowledge" in resolved_names
    assert "friendly" in resolved_names
    assert "safe_idle" in resolved_names
    capability_events = [
        event.event_type
        for event in reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=24,
            )
        )
        if event.event_type.startswith("capability.")
    ]
    assert BrainEventType.CAPABILITY_REQUESTED in capability_events
    assert BrainEventType.CAPABILITY_COMPLETED in capability_events


@pytest.mark.asyncio
async def test_embodiment_policy_reacts_to_projected_presence_and_disconnect(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = MockDriver()
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser")
    controller = RobotHeadController(catalog=catalog, driver=driver)
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    processor = EmbodimentPolicyProcessor(
        action_dispatcher=EmbodiedCapabilityDispatcher(
            action_engine=action_engine,
            capability_registry=build_embodied_capability_registry(action_engine=action_engine),
        ),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        idle_timeout_secs=0.01,
        presence_poll_secs=0.01,
    )

    async def emit_presence_events():
        await asyncio.sleep(0.05)
        store.append_brain_event(
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="perception",
            payload={
                "presence_scope_key": "browser:presence",
                "frame_seq": 1,
                "camera_connected": True,
                "camera_fresh": True,
                "person_present": "present",
                "attention_to_camera": "toward_camera",
                "engagement_state": "engaged",
                "scene_change": "stable",
                "summary": "One person is facing the camera.",
                "confidence": 0.9,
                "observed_at": "2026-04-17T10:00:00+00:00",
            },
        )
        await asyncio.sleep(0.12)
        store.append_brain_event(
            event_type=BrainEventType.SCENE_CHANGED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="runtime",
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": False,
                "camera_fresh": False,
                "person_present": "absent",
                "attention_to_camera": "unknown",
                "engagement_state": "away",
                "scene_change": "changed",
                "summary": "camera disconnected",
                "confidence": 1.0,
                "observed_at": "2026-04-17T10:00:05+00:00",
            },
        )

    await asyncio.gather(
        run_test(
            processor,
            frames_to_send=[SleepFrame(sleep=0.26)],
            frames_to_send_direction=FrameDirection.DOWNSTREAM,
        ),
        emit_presence_events(),
    )

    resolved_names = [plan.resolved_name for plan in driver.executed_plans]
    assert resolved_names.count("listen_engage") >= 1
    assert "safe_idle" in resolved_names
    body = store.get_body_state_projection(scope_key="browser:presence")
    assert body.policy_phase == "neutral"
    assert body.robot_head_last_safe_state == "neutral"

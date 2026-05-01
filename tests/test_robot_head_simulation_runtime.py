import pytest

import blink.brain.actions as brain_actions
from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    EmbodiedCapabilityDispatcher,
    build_embodied_capability_registry,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor
from blink.embodiment.robot_head.simulation import RobotHeadSimulationConfig, SimulationDriver
from blink.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from blink.tests.utils import SleepFrame, run_test


class RobotHeadSimulationHarness:
    def __init__(self, tmp_path):
        self._user_id = "pc-123"
        self._thread_id = "browser:pc-123"
        self.store = BrainStore(path=tmp_path / "brain.db")
        self.controller = RobotHeadController(
            catalog=build_default_robot_head_catalog(),
            driver=SimulationDriver(
                config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
            ),
        )
        self.action_engine = EmbodiedActionEngine(
            library=EmbodiedActionLibrary.build_default(),
            controller=self.controller,
            store=self.store,
            session_resolver=lambda: resolve_brain_session_ids(
                runtime_kind="browser",
                client_id=self._user_id,
            ),
            presence_scope_key="browser:presence",
        )

    async def run_policy_frames(self, frames):
        processor = EmbodimentPolicyProcessor(
            action_dispatcher=EmbodiedCapabilityDispatcher(
                action_engine=self.action_engine,
                capability_registry=build_embodied_capability_registry(action_engine=self.action_engine),
            ),
            idle_timeout_secs=0.01,
        )
        await run_test(
            processor,
            frames_to_send=[*frames, SleepFrame(sleep=0.02)],
            send_end_frame=True,
        )

    def recent_action_ids(self) -> list[str]:
        events = self.store.recent_action_events(
            user_id=self._user_id,
            thread_id=self._thread_id,
            limit=20,
        )
        return [event.action_id for event in reversed(events)]

    def recent_action_events(self):
        return list(
            reversed(
                self.store.recent_action_events(
                    user_id=self._user_id,
                    thread_id=self._thread_id,
                    limit=20,
                )
            )
        )

    def presence_snapshot(self) -> dict:
        snapshot = self.store.get_presence_snapshot(scope_key="browser:presence")
        assert snapshot is not None
        return snapshot


@pytest.mark.asyncio
async def test_robot_head_simulation_harness_tracks_policy_sequence_and_presence(
    monkeypatch,
    tmp_path,
):
    counter = iter(range(1, 100))
    monkeypatch.setattr(brain_actions.time, "monotonic", lambda: float(next(counter)))

    harness = RobotHeadSimulationHarness(tmp_path)
    await harness.run_policy_frames(
        [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        ]
    )

    assert harness.recent_action_ids() == [
        "cmd_return_neutral",
        "auto_listen_user",
        "auto_think",
        "auto_speak_friendly",
        "auto_safe_idle",
        "cmd_return_neutral",
    ]

    presence = harness.presence_snapshot()
    assert presence["robot_head_mode"] == "simulation"
    assert presence["robot_head_available"] is True
    assert presence["robot_head_last_action"] == "cmd_return_neutral"
    assert presence["robot_head_last_safe_state"] == "neutral"

    status_result = await harness.controller.status()
    assert status_result.accepted is True
    assert status_result.status is not None
    assert status_result.status.mode == "simulation"
    assert status_result.status.details["positions"][1] == 2096
    assert all(event.preview_only is False for event in harness.recent_action_events())

    await harness.controller.close()

"""Automatic policy processor for robot-head conversational embodiment."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from blink.brain.actions import EmbodiedCapabilityDispatcher
from blink.brain.events import BrainEventType
from blink.brain.presence import normalize_presence_snapshot
from blink.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StopFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from blink.processors.frame_processor import FrameDirection, FrameProcessor


class EmbodimentPolicyProcessor(FrameProcessor):
    """Translate runtime state and speaking events into bounded robot-head cues."""

    def __init__(
        self,
        *,
        action_dispatcher: EmbodiedCapabilityDispatcher,
        store=None,
        session_resolver=None,
        presence_scope_key: str = "local:presence",
        idle_timeout_secs: float = 1.5,
        presence_poll_secs: float = 0.5,
    ):
        """Initialize the embodiment policy processor."""
        super().__init__(name="robot-head-embodiment-policy")
        self._action_dispatcher = action_dispatcher
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._idle_timeout_secs = idle_timeout_secs
        self._presence_poll_secs = max(0.1, presence_poll_secs)
        self._idle_task = None
        self._presence_task = None
        self._phase = "neutral"
        self._last_presence_signature: tuple[Any, ...] | None = None

    async def cleanup(self):
        """Cancel policy timers and return to neutral before shutdown."""
        await self._cancel_idle_task()
        await self._cancel_presence_task()
        try:
            if self._action_dispatcher:
                await self._dispatch_action(
                    "cmd_return_neutral",
                    source="policy_shutdown",
                    reason="Pipeline cleanup requested neutral return.",
                    phase="neutral",
                    attention_target=None,
                    engagement_pose="neutral",
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"{self}: unable to return robot head to neutral during cleanup: {exc}")
        await self._action_dispatcher.close()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """React to turn/speaking frames and pass all frames through unchanged."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._action_dispatcher.start()
            await self._sync_policy_state(
                phase="neutral",
                attention_target=None,
                engagement_pose="neutral",
            )
            await self._dispatch_action(
                "cmd_return_neutral",
                source="policy_startup",
                reason="Pipeline startup initializes the head in neutral.",
                phase="neutral",
                attention_target=None,
                engagement_pose="neutral",
            )
            await self._start_presence_task()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._cancel_idle_task()
            if self._phase != "user_listening":
                await self._dispatch_action(
                    "auto_listen_user",
                    source="policy",
                    reason="User started speaking.",
                    phase="user_listening",
                    attention_target="user",
                    engagement_pose="attentive",
                )
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._phase != "thinking":
                await self._dispatch_action(
                    "auto_think",
                    source="policy",
                    reason="User turn ended; start a visible thinking shift.",
                    phase="thinking",
                    attention_target="user",
                    engagement_pose="thinking",
                )
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._cancel_idle_task()
            if self._phase != "bot_speaking":
                await self._dispatch_action(
                    "auto_speak_friendly",
                    source="policy",
                    reason="Bot started speaking.",
                    phase="bot_speaking",
                    attention_target="user",
                    engagement_pose="speaking",
                )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._phase = "post_bot"
            await self._sync_policy_state(
                phase="post_bot",
                attention_target="user",
                engagement_pose="neutral",
            )
            await self._start_idle_task()
        elif isinstance(frame, (ErrorFrame, CancelFrame, EndFrame, StopFrame)):
            await self._cancel_idle_task()
            await self._dispatch_action(
                "auto_safe_idle",
                source="policy",
                reason=f"Pipeline emitted {frame.__class__.__name__}.",
                phase="neutral",
                attention_target=None,
                engagement_pose="neutral",
            )

        await self.push_frame(frame, direction)

    async def _start_idle_task(self):
        """Schedule the safe-idle sequence after the configured delay."""
        await self._cancel_idle_task()
        self._idle_task = self.create_task(self._idle_timer(), "idle_timer")
        await asyncio.sleep(0)

    async def _cancel_idle_task(self):
        """Cancel the pending idle timer, if any."""
        if self._idle_task:
            await self.cancel_task(self._idle_task)
            self._idle_task = None

    async def _start_presence_task(self):
        """Start the background monitor that reacts to projected situational state."""
        if self._store is None or self._presence_task is not None:
            return
        self._presence_task = self.create_task(self._presence_monitor_loop(), "presence_monitor")
        await asyncio.sleep(0)

    async def _cancel_presence_task(self):
        """Cancel the background presence monitor, if any."""
        if self._presence_task:
            await self.cancel_task(self._presence_task)
            self._presence_task = None

    async def _idle_timer(self):
        """Wait for inactivity, then return the head through safe idle to neutral."""
        await asyncio.sleep(self._idle_timeout_secs)
        self._idle_task = None
        await self._dispatch_action(
            "auto_safe_idle",
            source="policy_idle",
            reason="Bot finished speaking and no new activity arrived.",
            phase="neutral",
            attention_target=None,
            engagement_pose="neutral",
        )

    async def _presence_monitor_loop(self):
        """Poll symbolic state projections and apply conservative idle reactions."""
        while True:
            await asyncio.sleep(self._presence_poll_secs)
            try:
                await self._maybe_apply_presence_policy()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(f"{self}: presence monitor error: {exc}")

    async def _maybe_apply_presence_policy(self):
        if self._store is None:
            return

        body = self._store.get_body_state_projection(scope_key=self._presence_scope_key)
        scene = self._store.get_scene_state_projection(scope_key=self._presence_scope_key)
        engagement = self._store.get_engagement_state_projection(scope_key=self._presence_scope_key)
        signature = (
            scene.camera_connected,
            scene.person_present,
            engagement.engagement_state,
            engagement.attention_to_camera,
            body.sensor_health,
            body.perception_unreliable,
            body.vision_unavailable,
            body.perception_disabled,
        )
        if signature == self._last_presence_signature:
            return
        self._last_presence_signature = signature

        if self._phase in {"user_listening", "thinking", "bot_speaking", "post_bot"}:
            return

        if (
            not scene.camera_connected
            or body.perception_disabled
            or body.vision_unavailable
            or body.perception_unreliable
            or scene.person_present == "absent"
        ):
            if self._phase != "neutral":
                await self._cancel_idle_task()
                await self._dispatch_action(
                    "auto_safe_idle",
                    source="policy",
                    reason="Perception indicated absence, disconnect, or degraded sensor state.",
                    phase="neutral",
                    attention_target=None,
                    engagement_pose="neutral",
                )
            else:
                await self._sync_policy_state(
                    phase="neutral",
                    attention_target=None,
                    engagement_pose="neutral",
                )
            return

        if (
            scene.person_present == "present"
            and engagement.engagement_state in {"engaged", "listening", "speaking"}
            and self._phase not in {"presence_attentive", "user_listening"}
        ):
            await self._dispatch_action(
                "auto_listen_user",
                source="policy",
                reason="Perception indicated a present and engaged user while idle.",
                phase="presence_attentive",
                attention_target="user",
                engagement_pose="attentive",
            )

    async def _dispatch_action(
        self,
        action_id: str,
        *,
        source: str,
        reason: str,
        phase: str,
        attention_target: str | None,
        engagement_pose: str | None,
    ):
        result = await self._action_dispatcher.execute_action(
            action_id,
            source=source,
            reason=reason,
        )
        await self._sync_policy_state(
            phase=phase,
            attention_target=attention_target,
            engagement_pose=engagement_pose,
            action_id=str(result.output.get("action_id") or action_id),
            accepted=result.accepted,
            warnings=list(result.warnings),
        )
        return result

    async def _sync_policy_state(
        self,
        *,
        phase: str,
        attention_target: str | None,
        engagement_pose: str | None,
        action_id: str | None = None,
        accepted: bool | None = None,
        warnings: list[str] | None = None,
    ):
        self._phase = phase
        if self._store is None or self._session_resolver is None:
            return

        snapshot = self._store.get_body_state_projection(scope_key=self._presence_scope_key)
        snapshot.policy_phase = phase
        snapshot.attention_target = attention_target
        snapshot.engagement_pose = engagement_pose
        if action_id:
            snapshot.robot_head_last_action = action_id
            if accepted is True:
                snapshot.robot_head_last_accepted_action = action_id
            elif accepted is False:
                snapshot.robot_head_last_rejected_action = action_id
        if warnings is not None and warnings:
            snapshot.warnings = list(warnings)
        normalize_presence_snapshot(snapshot)
        session_ids = self._session_resolver()
        self._store.append_brain_event(
            event_type=BrainEventType.BODY_STATE_UPDATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="policy",
            payload={
                "scope_key": self._presence_scope_key,
                "snapshot": snapshot.as_dict(),
            },
        )

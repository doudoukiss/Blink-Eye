#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local browser/WebRTC CLI for Apple Silicon Mac development."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, NotRequired, Optional, TypedDict
from uuid import uuid4

import httpx
from loguru import logger

from blink.adapters.schemas.function_schema import FunctionSchema
from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.brain.actions import embodied_action_tool_prompt
from blink.brain.adapters.perception import LocalPerceptionAdapter
from blink.brain.capabilities import CapabilityAssistantUtterance
from blink.brain.evals.episode_evidence_index import build_episode_evidence_index
from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    PerformancePreferenceStore,
)
from blink.brain.events import BrainEventType
from blink.brain.memory import memory_tool_prompt
from blink.brain.memory_persona_ingestion import (
    apply_memory_persona_ingestion,
    build_memory_persona_ingestion_preview,
)
from blink.brain.memory_v2 import (
    DiscourseEpisodeV3Collector,
    apply_memory_governance_action,
    build_memory_palace_snapshot,
)
from blink.brain.perception import (
    DEFAULT_CAMERA_STALE_FRAME_SECS,
    CameraFeedHealthManager,
    CameraFeedHealthManagerConfig,
    PerceptionBroker,
    PerceptionBrokerConfig,
)
from blink.brain.persona import (
    BrainPersonaModality,
    behavior_style_preset_catalog,
    build_witty_sophisticated_memory_story_seed,
    render_behavior_control_effect_summary,
    unavailable_expression_voice_metrics_snapshot,
    unavailable_runtime_expression_state,
    validate_behavior_control_update_payload,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.processors import (
    BrainExpressionVoicePolicyProcessor,
    BrainVoiceInputHealthProcessor,
    latest_user_text_from_context,
)
from blink.brain.runtime import BrainRuntime, build_session_resolver
from blink.brain.runtime_workbench import build_operator_workbench_snapshot
from blink.brain.store import BrainStore
from blink.cli.local_common import (
    DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS,
    DEFAULT_LOCAL_HOST,
    DEFAULT_LOCAL_LANGUAGE,
    DEFAULT_LOCAL_LLM_PROVIDER,
    DEFAULT_LOCAL_PORT,
    DEFAULT_LOCAL_STT_BACKEND,
    DEFAULT_LOCAL_VISION_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_RESPONSES_DEMO_SERVICE_TIER,
    LOCAL_LLM_PROVIDERS,
    LocalDependencyError,
    LocalLLMConfig,
    build_local_user_aggregators,
    build_local_voice_task,
    configure_logging,
    create_local_llm_service,
    create_local_stt_service,
    create_local_tts_service,
    create_local_vision_service,
    default_local_speech_system_prompt,
    default_local_stt_model,
    default_local_tts_backend,
    get_local_env,
    is_chinese_language,
    local_env_flag,
    maybe_load_dotenv,
    resolve_int,
    resolve_local_language,
    resolve_local_llm_config,
    resolve_local_runtime_tts_selection,
    resolve_local_tts_base_url,
    resolve_local_tts_voice,
    tts_backend_uses_external_service,
    verify_local_llm_config,
)
from blink.cli.local_model_registry import (
    DEFAULT_LOCAL_MODEL_PROFILE_ID,
    LocalLLMModelProfile,
    load_local_llm_model_profiles,
    local_llm_model_profile_by_id,
    local_llm_model_profile_id_for,
    remote_model_selection_enabled,
)
from blink.cli.local_runtime_profiles import profile_value, resolve_local_runtime_profile
from blink.embodiment.robot_head.catalog import load_robot_head_capability_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import (
    LiveDriver,
    MockDriver,
    PreviewDriver,
    SimulationDriver,
)
from blink.embodiment.robot_head.live_driver import RobotHeadLiveDriverConfig
from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor
from blink.embodiment.robot_head.simulation import RobotHeadSimulationConfig
from blink.embodiment.robot_head.tools import robot_head_tool_prompt
from blink.frames.frames import (
    AggregatedTextFrame,
    AudioRawFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserImageRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    VisionTextFrame,
)
from blink.function_calling import FunctionCallParams
from blink.interaction import (
    ActorControlScheduler,
    ActorEventContext,
    ActorTraceWriter,
    BrowserActorStateV2,
    BrowserBargeInTurnStartStrategy,
    BrowserInteractionMode,
    BrowserInteractionState,
    BrowserInterruptedOutputGuardProcessor,
    BrowserInterruptionStateTracker,
    BrowserPerformanceEventBus,
    BrowserProtectedPlaybackMuteStrategy,
    BrowserVisionGroundingTracker,
    ConversationFloorController,
    ConversationFloorInput,
    ConversationFloorInputType,
    ConversationFloorState,
    ConversationFloorStatus,
    PerformanceEpisodeV3Writer,
    WebRTCAudioHealthController,
    build_browser_camera_presence_snapshot,
    build_camera_scene_state,
    build_semantic_listener_state_v3,
    extract_active_listening_understanding,
    infer_scene_social_hints_from_moondream,
    sanitize_webrtc_audio_stats,
    unavailable_active_listener_state_v2,
    unavailable_active_listening_snapshot,
)
from blink.interaction.browser_runtime_session_v3 import BrowserRuntimeSessionV3
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.project_identity import PROJECT_IDENTITY
from blink.runner.utils import get_transport_client_id, maybe_capture_participant_camera
from blink.transcriptions.language import Language
from blink.web.server_startup import STARTUP_TIMEOUT_SECS, start_uvicorn_server
from blink.web.smallwebrtc_ui import create_smallwebrtc_root_redirect, mount_smallwebrtc_ui

if TYPE_CHECKING:
    from fastapi import Request, Response


@dataclass
class LocalBrowserConfig:
    """Configuration for the local browser/WebRTC workflow."""

    base_url: Optional[str]
    model: str
    system_prompt: str
    language: Language
    stt_backend: str
    tts_backend: str
    stt_model: str
    tts_voice: Optional[str]
    tts_base_url: Optional[str]
    host: str
    port: int
    robot_head_driver: str = "none"
    robot_head_catalog_path: Optional[str] = None
    robot_head_port: Optional[str] = None
    robot_head_baud: int = 1000000
    robot_head_hardware_profile_path: Optional[str] = None
    robot_head_live_arm: bool = False
    robot_head_arm_ttl_seconds: int = 300
    robot_head_operator_mode: bool = False
    robot_head_sim_scenario_path: Optional[Path] = None
    robot_head_sim_realtime: bool = False
    robot_head_sim_trace_dir: Optional[Path] = None
    vision_enabled: bool = False
    continuous_perception_enabled: bool = False
    continuous_perception_interval_secs: float = 3.0
    vision_model: str = DEFAULT_LOCAL_VISION_MODEL
    allow_barge_in: bool = False
    temperature: Optional[float] = None
    verbose: bool = False
    tts_backend_locked: bool = False
    tts_voice_override: Optional[str] = None
    llm_provider: str = DEFAULT_LOCAL_LLM_PROVIDER
    llm_service_tier: Optional[str] = None
    demo_mode: bool = False
    llm_max_output_tokens: Optional[int] = None
    tts_runtime_label: Optional[str] = None
    config_profile: Optional[str] = None
    actor_trace_enabled: bool = False
    actor_trace_dir: Optional[Path] = None
    performance_episode_v3_enabled: bool = False
    performance_episode_v3_dir: Optional[Path] = None
    performance_preferences_v3_dir: Optional[Path] = None
    actor_surface_v2_enabled: bool = True

    @property
    def llm(self) -> LocalLLMConfig:
        """Return the effective provider config for the browser LLM layer."""
        return LocalLLMConfig(
            provider=self.llm_provider,
            model=self.model,
            base_url=self.base_url,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            service_tier=self.llm_service_tier,
            demo_mode=self.demo_mode,
            max_output_tokens=self.llm_max_output_tokens,
        )


def _browser_connection_owns_active_state(
    app_state: Any,
    *,
    session_id: object,
    client_id: object | None,
) -> bool:
    """Return whether a WebRTC callback may clear global active browser state."""
    active_session_id = getattr(app_state, "blink_browser_active_session_id", None)
    active_client_id = getattr(app_state, "blink_browser_active_client_id", None)
    normalized_session_id = str(session_id) if session_id not in (None, "") else None
    normalized_client_id = str(client_id) if client_id not in (None, "") else None
    return (
        normalized_session_id is not None
        and str(active_session_id) == normalized_session_id
    ) or (
        normalized_client_id is not None
        and str(active_client_id) == normalized_client_id
    )


def _browser_config_for_model_profile(
    config: LocalBrowserConfig,
    profile: LocalLLMModelProfile,
) -> LocalBrowserConfig:
    """Return a browser config with only the LLM backend changed."""
    if profile.provider == "ollama":
        base_url = (
            config.base_url
            if config.llm_provider == "ollama" and config.base_url
            else os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        )
        return replace(
            config,
            llm_provider=profile.provider,
            model=profile.model,
            base_url=base_url,
            llm_service_tier=None,
            llm_max_output_tokens=None,
        )

    if profile.provider == "openai-responses":
        service_tier = (
            config.llm_service_tier
            if config.llm_provider == "openai-responses" and config.llm_service_tier
            else get_local_env("OPENAI_RESPONSES_SERVICE_TIER")
        )
        if not service_tier and config.demo_mode:
            service_tier = DEFAULT_OPENAI_RESPONSES_DEMO_SERVICE_TIER
        max_output_tokens = (
            config.llm_max_output_tokens
            if config.llm_provider == "openai-responses"
            else (DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS if config.demo_mode else None)
        )
        base_url = (
            config.base_url
            if config.llm_provider == "openai-responses" and config.base_url
            else get_local_env("OPENAI_RESPONSES_BASE_URL")
        )
        return replace(
            config,
            llm_provider=profile.provider,
            model=profile.model,
            base_url=base_url,
            llm_service_tier=service_tier,
            llm_max_output_tokens=max_output_tokens,
        )

    raise ValueError(f"Unsupported local LLM provider: {profile.provider!r}")


class BrowserIceCandidatePayload(TypedDict):
    """Incoming ICE candidate payload accepted by the browser routes."""

    candidate: str
    sdp_mid: NotRequired[str]
    sdpMid: NotRequired[str]
    sdp_mline_index: NotRequired[int]
    sdpMLineIndex: NotRequired[int]


class BrowserPatchPayload(TypedDict, total=False):
    """Incoming ICE patch payload accepted by the browser routes."""

    pc_id: str
    pcId: str
    candidates: list[BrowserIceCandidatePayload]


class BrowserStartPayload(TypedDict, total=False):
    """Incoming ``/start`` payload accepted by the browser routes."""

    body: dict[str, Any]
    enableDefaultIceServers: bool


class BrowserIceServer(TypedDict):
    """Blink browser ICE server configuration."""

    urls: list[str]


class BrowserIceConfig(TypedDict):
    """Blink browser ICE configuration response shape."""

    iceServers: list[BrowserIceServer]


class BrowserStartResponse(TypedDict, total=False):
    """Blink browser ``/start`` response shape."""

    sessionId: str
    iceConfig: BrowserIceConfig
    modelSelection: dict[str, Any]


BrowserPerformanceEmit = Callable[..., object]
BrowserFloorInputEmit = Callable[[ConversationFloorInput], object]


class LatestCameraFrameBuffer(FrameProcessor):
    """Cache the latest browser camera frame for on-demand vision queries."""

    def __init__(self, *, performance_emit: BrowserPerformanceEmit | None = None):
        """Initialize the frame buffer."""
        super().__init__(name="latest-camera-frame-buffer")
        self._latest_camera_frame: Optional[UserImageRawFrame] = None
        self._latest_camera_frame_seq = 0
        self._latest_camera_frame_received_monotonic: float | None = None
        self._latest_camera_frame_received_at: str | None = None
        self._last_camera_event_monotonic: float | None = None
        self._latest_camera_frame_condition = asyncio.Condition()
        self._performance_emit = performance_emit

    @property
    def latest_camera_frame(self) -> Optional[UserImageRawFrame]:
        """Return the most recent camera frame, if any."""
        return self._latest_camera_frame

    @property
    def latest_camera_frame_seq(self) -> int:
        """Return the monotonically increasing camera frame sequence number."""
        return self._latest_camera_frame_seq

    @property
    def latest_camera_frame_received_monotonic(self) -> float | None:
        """Return the monotonic receive timestamp of the latest frame."""
        return self._latest_camera_frame_received_monotonic

    @property
    def latest_camera_frame_received_at(self) -> str | None:
        """Return the UTC receive timestamp of the latest frame."""
        return self._latest_camera_frame_received_at

    def latest_camera_frame_age_ms(self) -> int | None:
        """Return the latest frame age in milliseconds, if a frame was received."""
        received_at = self._latest_camera_frame_received_monotonic
        if received_at is None:
            return None
        return max(0, int((time.monotonic() - float(received_at)) * 1000))

    def set_performance_emit(self, performance_emit: BrowserPerformanceEmit | None) -> None:
        """Attach a public-safe performance event sink."""
        self._performance_emit = performance_emit

    def latest_camera_frame_is_fresh(self, *, max_age_secs: float) -> bool:
        """Return whether the latest cached frame is fresh enough to use."""
        received_at = self._latest_camera_frame_received_monotonic
        if self._latest_camera_frame is None or received_at is None:
            return False
        return (asyncio.get_running_loop().time() - float(received_at)) <= max_age_secs

    async def wait_for_latest_camera_frame(
        self,
        *,
        after_seq: int = 0,
        timeout: float = 0.0,
    ) -> Optional[UserImageRawFrame]:
        """Wait for a camera frame newer than ``after_seq``.

        Args:
            after_seq: Return only when a newer camera frame is available.
            timeout: Maximum time in seconds to wait.

        Returns:
            The latest camera frame if a newer one arrived, otherwise ``None``.
        """
        async with self._latest_camera_frame_condition:
            if self._latest_camera_frame_seq > after_seq:
                return self._latest_camera_frame

            if timeout <= 0:
                return None

            try:
                await asyncio.wait_for(
                    self._latest_camera_frame_condition.wait_for(
                        lambda: self._latest_camera_frame_seq > after_seq
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None

            return self._latest_camera_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Store camera frames and keep them out of the audio/LLM pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRawFrame) and frame.transport_source == "camera":
            emit_camera_event = False
            async with self._latest_camera_frame_condition:
                self._latest_camera_frame = frame
                self._latest_camera_frame_seq += 1
                now = asyncio.get_running_loop().time()
                self._latest_camera_frame_received_monotonic = now
                self._latest_camera_frame_received_at = datetime.now(UTC).isoformat()
                emit_camera_event = (
                    self._last_camera_event_monotonic is None
                    or now - self._last_camera_event_monotonic >= 5.0
                )
                if emit_camera_event:
                    self._last_camera_event_monotonic = now
                self._latest_camera_frame_condition.notify_all()
            if emit_camera_event and self._performance_emit is not None:
                self._performance_emit(
                    event_type="camera.frame_received",
                    source="camera",
                    mode=BrowserInteractionMode.CONNECTED,
                    metadata={
                        "frame_seq": self._latest_camera_frame_seq,
                        "frame_age_ms": self.latest_camera_frame_age_ms(),
                        "scene_transition": "frame_captured",
                    },
                    reason_codes=(
                        "camera:frame_received",
                        "scene_social_transition:frame_captured",
                    ),
                )
            return

        await self.push_frame(frame, direction)


class BrowserPerformanceFrameObserver(FrameProcessor):
    """Observe browser pipeline frames and emit public-safe performance events."""

    def __init__(
        self,
        *,
        phase: str,
        performance_emit: BrowserPerformanceEmit,
        resting_mode_provider: Callable[[], BrowserInteractionMode],
        floor_input_emit: BrowserFloorInputEmit | None = None,
        voice_metrics_recorder: Any | None = None,
        speech_lookahead_drain: Callable[[], Awaitable[None]] | None = None,
        runtime_session: BrowserRuntimeSessionV3 | None = None,
    ):
        """Initialize the observer."""
        super().__init__(name=f"browser-performance-{phase}")
        self._phase = phase
        self._performance_emit = performance_emit
        self._resting_mode_provider = resting_mode_provider
        self._floor_input_emit = floor_input_emit
        self._voice_metrics_recorder = voice_metrics_recorder
        self._speech_lookahead_drain = speech_lookahead_drain
        self._runtime_session = runtime_session
        self._speech_contexts: dict[str, dict[str, object]] = {}
        self._speech_audio_started_contexts: set[str] = set()
        self._voice_turn_active = False
        self._voice_turn_started_at: float | None = None
        self._last_speech_continuing_emit_at = 0.0
        self._partial_transcript_seen = False
        self._last_partial_transcript_chars = 0
        self._final_transcript_chars_in_turn = 0

    def _emit_floor_input(self, floor_input: ConversationFloorInput) -> None:
        if self._floor_input_emit is None:
            return
        self._floor_input_emit(floor_input)

    @staticmethod
    def _transcript_confidence(frame: Frame) -> float | None:
        value = getattr(frame, "confidence", None)
        if value is None:
            result = getattr(frame, "result", None)
            if isinstance(result, dict):
                value = (
                    result.get("confidence")
                    or result.get("transcript_confidence")
                    or result.get("language_probability")
                )
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(confidence):
            return None
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _active_listening_event_metadata(
        *,
        understanding,
        language: object = "unknown",
        source_text: object = None,
        text_chars: int = 0,
        partial_text_chars: int | None = None,
        final_text_chars: int | None = None,
        partial_available: bool = False,
        final_available: bool = False,
        ready_to_answer: bool = False,
        readiness_state: str = "not_ready",
        degradation_state: str = "ok",
    ) -> dict[str, object]:
        hint_kinds: list[str] = []
        if getattr(understanding, "topics", ()):
            hint_kinds.append("topic")
        if getattr(understanding, "constraints", ()):
            hint_kinds.append("constraint")
        if getattr(understanding, "corrections", ()):
            hint_kinds.append("correction")
        if getattr(understanding, "project_references", ()):
            hint_kinds.append("project_reference")
        partial_chars = (
            max(0, int(partial_text_chars))
            if partial_text_chars is not None
            else max(0, int(text_chars)) if partial_available else 0
        )
        final_chars = (
            max(0, int(final_text_chars))
            if final_text_chars is not None
            else max(0, int(text_chars)) if final_available else 0
        )
        semantic_state = build_semantic_listener_state_v3(
            language=language,
            understanding=understanding,
            source_text=source_text,
            partial_available=partial_available,
            final_available=final_available,
            partial_transcript_chars=partial_chars if partial_available else 0,
            final_transcript_chars=final_chars if final_available else 0,
            ready_to_answer=ready_to_answer,
            readiness_state=readiness_state,
        ).as_dict()
        safe_summary = str(semantic_state.get("safe_live_summary") or "")
        semantic_metadata = {
            "schema_version": 3,
            "detected_intent": semantic_state.get("detected_intent") or "unknown",
            "listener_chip_ids": [
                str(chip.get("chip_id"))
                for chip in semantic_state.get("listener_chips", [])
                if isinstance(chip, dict) and chip.get("chip_id")
            ][:7],
            "listener_chip_count": int(semantic_state.get("listener_chip_count") or 0),
            "camera_reference_state": semantic_state.get("camera_reference_state") or "not_used",
            "memory_context_state": semantic_state.get("memory_context_state") or "unavailable",
            "floor_state": semantic_state.get("floor_state") or "unknown",
            "enough_information_to_answer": bool(
                semantic_state.get("enough_information_to_answer")
            ),
            "summary_hash": hashlib.sha256(safe_summary.encode("utf-8")).hexdigest()[:16]
            if safe_summary
            else "",
            "reason_codes": [
                str(code)
                for code in (
                    semantic_state.get("reason_codes")
                    if isinstance(semantic_state.get("reason_codes"), list)
                    else []
                )
            ][:24],
        }
        return {
            "partial_transcript_chars": partial_chars if partial_available else 0,
            "final_transcript_chars": final_chars if final_available else 0,
            "partial_available": bool(partial_available),
            "final_available": bool(final_available),
            "topic_count": len(getattr(understanding, "topics", ()) or ()),
            "constraint_count": len(getattr(understanding, "constraints", ()) or ()),
            "correction_count": len(getattr(understanding, "corrections", ()) or ()),
            "project_reference_count": len(getattr(understanding, "project_references", ()) or ()),
            "uncertainty_flag_count": len(getattr(understanding, "uncertainty_flags", ()) or ()),
            "hint_kinds": hint_kinds[:5],
            "ready_to_answer": bool(ready_to_answer),
            "readiness_state": readiness_state,
            "degradation_state": degradation_state,
            "semantic_listener": semantic_metadata,
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Record public-safe runtime state and pass frames through unchanged."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            await self._record_frame(frame)
        await self.push_frame(frame, direction)

    async def _record_frame(self, frame: Frame):
        if isinstance(frame, InterruptionFrame) and self._phase == "pre_stt":
            self._performance_emit(
                event_type="runtime.interrupted",
                source="pipeline",
                mode=BrowserInteractionMode.INTERRUPTED,
                reason_codes=("runtime:interrupted",),
            )
            self._emit_floor_input(
                ConversationFloorInput(
                    input_type=ConversationFloorInputType.INTERRUPTION_ACCEPTED,
                    reason_codes=("floor:runtime_interrupted",),
                )
            )
            return

        if self._phase == "pre_stt":
            if isinstance(frame, VADUserStartedSpeakingFrame):
                now = time.monotonic()
                if self._runtime_session is not None:
                    self._runtime_session.transcript.start_turn()
                self._voice_turn_active = True
                self._voice_turn_started_at = now
                self._last_speech_continuing_emit_at = now
                self._partial_transcript_seen = False
                self._last_partial_transcript_chars = 0
                self._final_transcript_chars_in_turn = 0
                self._performance_emit(
                    event_type="voice.speech_started",
                    source="vad",
                    mode=BrowserInteractionMode.LISTENING,
                    reason_codes=("voice:speech_started", "active_listening:speech_started"),
                )
                self._performance_emit(
                    event_type="active_listening.listening_started",
                    source="active_listening",
                    mode=BrowserInteractionMode.LISTENING,
                    metadata={
                        "partial_available": False,
                        "final_available": False,
                        "ready_to_answer": False,
                        "readiness_state": "listening",
                        "degradation_state": "ok",
                    },
                    reason_codes=("active_listener:listening_started",),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.VAD_USER_STARTED,
                        user_speaking=True,
                        reason_codes=("floor:vad_user_started",),
                    )
                )
                return
            if isinstance(frame, AudioRawFrame) and self._voice_turn_active:
                now = time.monotonic()
                if now - self._last_speech_continuing_emit_at >= 1.0:
                    self._last_speech_continuing_emit_at = now
                    speech_age_ms = (
                        int((now - self._voice_turn_started_at) * 1000)
                        if self._voice_turn_started_at is not None
                        else 0
                    )
                    self._performance_emit(
                        event_type="voice.speech_continuing",
                        source="vad",
                        mode=BrowserInteractionMode.LISTENING,
                        metadata={"speech_age_ms": speech_age_ms},
                        reason_codes=(
                            "voice:speech_continuing",
                            "active_listening:speech_continuing",
                        ),
                    )
                    self._emit_floor_input(
                        ConversationFloorInput(
                            input_type=ConversationFloorInputType.VAD_USER_CONTINUING,
                            user_speaking=True,
                            speech_age_ms=speech_age_ms,
                            reason_codes=("floor:vad_user_continuing",),
                        )
                    )
                return
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                now = time.monotonic()
                turn_duration_ms = (
                    int((now - self._voice_turn_started_at) * 1000)
                    if self._voice_turn_started_at is not None
                    else 0
                )
                self._voice_turn_active = False
                if self._runtime_session is not None:
                    self._runtime_session.transcript.stop_turn()
                self._performance_emit(
                    event_type="voice.speech_stopped",
                    source="vad",
                    mode=BrowserInteractionMode.HEARD,
                    metadata={"turn_duration_ms": turn_duration_ms},
                    reason_codes=("voice:speech_stopped", "active_listening:speech_stopped"),
                )
                self._performance_emit(
                    event_type="stt.transcribing",
                    source="stt",
                    mode=BrowserInteractionMode.HEARD,
                    metadata={"turn_duration_ms": turn_duration_ms},
                    reason_codes=("stt:transcribing", "active_listening:transcribing"),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.VAD_USER_STOPPED,
                        user_speaking=False,
                        turn_duration_ms=turn_duration_ms,
                        reason_codes=("floor:vad_user_stopped",),
                    )
                )
                return

        if self._phase == "post_stt":
            if isinstance(frame, InterimTranscriptionFrame):
                session_counts = (
                    self._runtime_session.transcript.note_partial(frame.text)
                    if self._runtime_session is not None
                    else None
                )
                self._partial_transcript_seen = True
                self._last_partial_transcript_chars = int(
                    (session_counts or {}).get("partial_transcript_chars")
                    or len(frame.text or "")
                )
                understanding = extract_active_listening_understanding(
                    frame.text,
                    language=getattr(frame, "language", None),
                    source="partial_transcript",
                )
                self._performance_emit(
                    event_type="stt.partial_transcription",
                    source="stt",
                    mode=BrowserInteractionMode.LISTENING,
                    metadata={
                        "partial_transcript_chars": len(frame.text or ""),
                        "partial_available": True,
                    },
                    reason_codes=("stt:partial_transcript", "active_listening:partial_transcript"),
                )
                self._performance_emit(
                    event_type="active_listening.partial_understanding_updated",
                    source="active_listening",
                    mode=BrowserInteractionMode.LISTENING,
                    metadata=self._active_listening_event_metadata(
                        understanding=understanding,
                        language=getattr(frame, "language", None),
                        source_text=frame.text,
                        text_chars=len(frame.text or ""),
                        partial_available=True,
                        readiness_state="partial",
                    ),
                    reason_codes=(
                        "active_listener:partial_understanding_updated",
                        *understanding.reason_codes,
                    ),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.STT_INTERIM,
                        text=frame.text or "",
                        text_chars=len(frame.text or ""),
                        transcript_confidence=self._transcript_confidence(frame),
                        reason_codes=("floor:stt_interim",),
                    )
                )
                return
            if isinstance(frame, TranscriptionFrame):
                text_chars = len(frame.text or "")
                session_counts = (
                    self._runtime_session.transcript.note_final(frame.text)
                    if self._runtime_session is not None
                    else None
                )
                if session_counts is not None:
                    self._final_transcript_chars_in_turn = int(
                        session_counts.get("final_transcript_chars") or text_chars
                    )
                else:
                    self._final_transcript_chars_in_turn += text_chars
                understanding = extract_active_listening_understanding(
                    frame.text,
                    language=getattr(frame, "language", None),
                )
                self._performance_emit(
                    event_type="stt.transcription",
                    source="stt",
                    mode=BrowserInteractionMode.HEARD,
                    metadata={
                        "transcription_chars": text_chars,
                        "final_fragment_chars": text_chars,
                        "final_transcript_chars": self._final_transcript_chars_in_turn,
                    },
                    reason_codes=("stt:transcribed", "active_listening:final_transcript"),
                )
                self._performance_emit(
                    event_type="user_turn.summary",
                    source="active_listening",
                    mode=BrowserInteractionMode.HEARD,
                    metadata={
                        "final_transcript_chars": self._final_transcript_chars_in_turn,
                        "topic_count": len(understanding.topics),
                        "constraint_count": len(understanding.constraints),
                    },
                    reason_codes=(
                        "active_listening:user_turn_summary",
                        *understanding.reason_codes,
                    ),
                )
                self._performance_emit(
                    event_type="active_listening.final_understanding_ready",
                    source="active_listening",
                    mode=BrowserInteractionMode.HEARD,
                    metadata=self._active_listening_event_metadata(
                        understanding=understanding,
                        language=getattr(frame, "language", None),
                        source_text=frame.text,
                        text_chars=self._final_transcript_chars_in_turn,
                        partial_text_chars=self._last_partial_transcript_chars,
                        final_text_chars=self._final_transcript_chars_in_turn,
                        partial_available=self._partial_transcript_seen,
                        final_available=True,
                        ready_to_answer=True,
                        readiness_state="ready",
                    ),
                    reason_codes=(
                        "active_listener:final_understanding_ready",
                        *understanding.reason_codes,
                    ),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.STT_FINAL,
                        text=frame.text or "",
                        text_chars=text_chars,
                        transcript_confidence=self._transcript_confidence(frame),
                        reason_codes=("floor:stt_final",),
                    )
                )
                return
            if isinstance(frame, ErrorFrame):
                self._performance_emit(
                    event_type="stt.error",
                    source="stt",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={"error_type": type(frame).__name__},
                    reason_codes=("stt:error",),
                )
                self._performance_emit(
                    event_type="active_listening.listening_degraded",
                    source="active_listening",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={
                        "ready_to_answer": False,
                        "readiness_state": "degraded",
                        "degradation_state": "error",
                        "degraded_component_count": 1,
                    },
                    reason_codes=("active_listener:listening_degraded", "stt:error"),
                )
                return

        if self._phase == "post_llm":
            if isinstance(frame, LLMFullResponseStartFrame):
                self._performance_emit(
                    event_type="llm.response_start",
                    source="llm",
                    mode=BrowserInteractionMode.THINKING,
                    reason_codes=("llm:response_start",),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.LLM_STARTED,
                        assistant_speaking=False,
                        reason_codes=("floor:llm_started",),
                    )
                )
                return
            if isinstance(frame, LLMFullResponseEndFrame):
                self._performance_emit(
                    event_type="llm.response_end",
                    source="llm",
                    mode=BrowserInteractionMode.THINKING,
                    reason_codes=("llm:response_end",),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.LLM_ENDED,
                        assistant_speaking=False,
                        reason_codes=("floor:llm_ended",),
                    )
                )
                return
            if isinstance(frame, ErrorFrame):
                self._performance_emit(
                    event_type="llm.error",
                    source="llm",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={"error_type": type(frame).__name__},
                    reason_codes=("llm:error",),
                )
                return

        if self._phase == "post_tts":
            if isinstance(frame, AggregatedTextFrame):
                chunk = getattr(frame, "blink_speech_chunk", None)
                context_id = str(getattr(frame, "context_id", "") or "")
                if chunk is not None and context_id:
                    created_at = float(getattr(chunk, "created_at_monotonic", time.monotonic()))
                    chunk_metadata = (
                        chunk.public_metadata(queue_depth=len(self._speech_contexts) + 1)
                        if hasattr(chunk, "public_metadata")
                        else {}
                    )
                    self._speech_contexts[context_id] = {
                        "created_at": created_at,
                        "chunk_metadata": dict(chunk_metadata),
                    }
                    latency_ms = (time.monotonic() - created_at) * 1000.0
                    recorder = getattr(self._voice_metrics_recorder, "record_speech_chunk_latency", None)
                    if callable(recorder):
                        recorder(latency_ms)
                    self._performance_emit(
                        event_type="speech.tts_request_start",
                        source="speech_director",
                        mode=BrowserInteractionMode.THINKING,
                        metadata={
                            **dict(chunk_metadata),
                            "chunk_latency_ms": round(latency_ms, 4),
                            "context_available": True,
                        },
                        reason_codes=("speech:tts_request_start",),
                    )
                return
            if isinstance(frame, TTSStartedFrame):
                self._performance_emit(
                    event_type="tts.speech_start",
                    source="tts",
                    mode=BrowserInteractionMode.SPEAKING,
                    metadata={"context_available": bool(frame.context_id)},
                    reason_codes=("tts:speaking",),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.TTS_STARTED,
                        assistant_speaking=True,
                        tts_chunk_role="assistant_speech",
                        reason_codes=("floor:tts_started",),
                    )
                )
                return
            if isinstance(frame, TTSAudioRawFrame):
                context_id = str(getattr(frame, "context_id", "") or "")
                if context_id and context_id not in self._speech_audio_started_contexts:
                    self._speech_audio_started_contexts.add(context_id)
                    context_state = self._speech_contexts.get(context_id)
                    created_at = (
                        float(context_state.get("created_at"))
                        if isinstance(context_state, dict) and context_state.get("created_at") is not None
                        else None
                    )
                    chunk_metadata = (
                        dict(context_state.get("chunk_metadata") or {})
                        if isinstance(context_state, dict)
                        else {}
                    )
                    latency_ms = (
                        (time.monotonic() - created_at) * 1000.0
                        if created_at is not None
                        else None
                    )
                    recorder = getattr(self._voice_metrics_recorder, "record_audio_started", None)
                    if callable(recorder) and latency_ms is not None:
                        recorder(first_audio_latency_ms=latency_ms)
                    self._performance_emit(
                        event_type="speech.audio_start",
                        source="tts",
                        mode=BrowserInteractionMode.SPEAKING,
                        metadata={
                            **chunk_metadata,
                            "context_available": bool(context_id),
                            "first_audio_latency_ms": round(latency_ms, 4)
                            if latency_ms is not None
                            else None,
                        },
                        reason_codes=("speech:audio_start",),
                    )
                return
            if isinstance(frame, TTSStoppedFrame):
                context_id = str(getattr(frame, "context_id", "") or "")
                if context_id:
                    self._speech_contexts.pop(context_id, None)
                    self._speech_audio_started_contexts.discard(context_id)
                    recorder = getattr(
                        self._voice_metrics_recorder,
                        "record_speech_queue_depth",
                        None,
                    )
                    if callable(recorder):
                        recorder(len(self._speech_contexts))
                self._performance_emit(
                    event_type="tts.speech_end",
                    source="tts",
                    mode=self._resting_mode_provider(),
                    metadata={"context_available": bool(frame.context_id)},
                    reason_codes=("tts:stopped",),
                )
                self._emit_floor_input(
                    ConversationFloorInput(
                        input_type=ConversationFloorInputType.TTS_STOPPED,
                        assistant_speaking=False,
                        tts_chunk_role="assistant_speech",
                        reason_codes=("floor:tts_stopped",),
                    )
                )
                if self._runtime_session is not None:
                    self._runtime_session.speech.note_tts_stopped()
                if self._speech_lookahead_drain is not None:
                    await self._speech_lookahead_drain()
                return
            if isinstance(frame, ErrorFrame):
                self._performance_emit(
                    event_type="speech.degraded",
                    source="tts",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={"error_type": type(frame).__name__},
                    reason_codes=("speech:degraded", "tts:error"),
                )
                self._performance_emit(
                    event_type="tts.error",
                    source="tts",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={"error_type": type(frame).__name__},
                    reason_codes=("tts:error",),
                )


def _latest_user_text(context: LLMContext) -> str:
    """Return the most recent user text message from the LLM context."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts)

    return ""


def _camera_frame_is_fresh(camera_buffer: Any, *, max_age_secs: float) -> bool:
    """Return whether the latest cached frame is fresh enough to inspect."""
    freshness_method = getattr(camera_buffer, "latest_camera_frame_is_fresh", None)
    if callable(freshness_method):
        return bool(freshness_method(max_age_secs=max_age_secs))

    latest_frame = getattr(camera_buffer, "latest_camera_frame", None)
    received_at = getattr(camera_buffer, "latest_camera_frame_received_monotonic", None)
    if latest_frame is None:
        return False
    if received_at is None:
        return True
    return (asyncio.get_running_loop().time() - float(received_at)) <= max_age_secs


def _camera_frame_age_ms(camera_buffer: Any) -> int | None:
    """Return a public-safe latest camera frame age, if available."""
    age_method = getattr(camera_buffer, "latest_camera_frame_age_ms", None)
    if callable(age_method):
        try:
            return age_method()
        except Exception:
            return None
    received_at = getattr(camera_buffer, "latest_camera_frame_received_monotonic", None)
    if received_at is None:
        return None
    try:
        return max(0, int((time.monotonic() - float(received_at)) * 1000))
    except (TypeError, ValueError):
        return None


def _build_vision_prompt(question: str) -> str:
    """Build a stable English vision prompt from the latest user question."""
    normalized = (question or "").strip()
    lowered = normalized.lower()

    if any(token in normalized for token in ("文字", "文本", "写着", "内容", "字")) or any(
        token in lowered for token in ("text", "read", "word", "words", "screen", "label")
    ):
        return (
            "Inspect one latest fresh still camera frame and answer in plain English. "
            "Do not claim continuous video understanding. "
            "Read any large, clearly legible text exactly. "
            "If text is present but blurry or too small to read, say that clearly. "
            "Also describe the main visible objects around the text."
        )

    if any(token in normalized for token in ("手里", "拿着")) or any(
        token in lowered for token in ("holding", "hand", "hands")
    ):
        return (
            "Inspect one latest fresh still camera frame and answer in plain English. "
            "Do not claim continuous video understanding. "
            "Focus on what the person is holding in their hands. "
            "If nothing is clearly visible, say that and describe the rest of the scene."
        )

    if any(token in normalized for token in ("身后", "后面", "背景")) or any(
        token in lowered for token in ("behind", "background", "backdrop")
    ):
        return (
            "Inspect one latest fresh still camera frame and answer in plain English. "
            "Do not claim continuous video understanding. "
            "Focus on the background behind the person. "
            "Mention prominent objects, posters, screens, or large text."
        )

    return (
        "Inspect one latest fresh still camera frame and answer in plain English. "
        "Do not claim continuous video understanding. "
        "If a person is visible, identify the person first before describing the background. "
        "Then describe the main subject, the most prominent objects, the background, "
        "and any large clearly readable text. "
        "Do not invent details. If part of the image is blurry, say what is still clearly visible."
    )


def _vision_result_is_unusable(text: str) -> bool:
    """Return True when the raw vision text is empty or clearly garbled."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.count("�") >= 2:
        return True
    if len(cleaned) < 4:
        return True
    return False


def _ground_vision_description(description: str, language: Language) -> str:
    """Prefix one-frame vision results so the answer stays grounded."""
    cleaned = " ".join(str(description or "").split()).strip()
    if not cleaned:
        return cleaned
    if is_chinese_language(language):
        if cleaned.startswith("基于刚才的一帧画面"):
            return cleaned
        return f"基于刚才的一帧画面：{cleaned}"
    if cleaned.lower().startswith("based on the latest still frame"):
        return cleaned
    return f"Based on the latest still frame: {cleaned}"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(f"Run {PROJECT_IDENTITY.display_name} local browser voice using SmallWebRTC.")
    )
    parser.add_argument(
        "--llm-provider",
        choices=LOCAL_LLM_PROVIDERS,
        help="Local LLM provider. Defaults to ollama.",
    )
    parser.add_argument("--model", help="Provider-relative model name.")
    parser.add_argument(
        "--config-profile",
        help="Typed local runtime profile id, such as browser-zh-melo or browser-en-kokoro.",
    )
    parser.add_argument("--base-url", help="Provider-relative base URL.")
    parser.add_argument("--system-prompt", help="System prompt for the local assistant.")
    parser.add_argument(
        "--language",
        help=f"Language code for STT/TTS and assistant replies, default {DEFAULT_LOCAL_LANGUAGE.value}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional temperature override passed to the selected LLM backend.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional provider-relative output token budget. OpenAI demo mode defaults to 120.",
    )
    parser.add_argument(
        "--demo-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable polished bounded local demo responses. Also available via BLINK_LOCAL_DEMO_MODE=1.",
    )
    parser.add_argument(
        "--stt-backend",
        choices=["mlx-whisper", "whisper"],
        help="Local speech-to-text backend.",
    )
    parser.add_argument("--stt-model", help="Model id for the selected STT backend.")
    parser.add_argument(
        "--tts-backend",
        choices=["kokoro", "piper", "xtts", "local-http-wav"],
        help="Local text-to-speech backend.",
    )
    parser.add_argument("--tts-voice", help="Voice id for the selected TTS backend.")
    parser.add_argument(
        "--allow-barge-in",
        action="store_true",
        help=(
            "Allow interrupting the assistant while it is speaking. Disabled by default "
            "for browser voice to avoid self-interruption from speaker bleed."
        ),
    )
    parser.add_argument("--host", help="Host for the local web server.")
    parser.add_argument("--port", type=int, help="Port for the local web server.")
    parser.add_argument(
        "--robot-head-driver",
        choices=["none", "mock", "preview", "simulation", "live"],
        help="Optional robot-head embodiment driver.",
    )
    parser.add_argument(
        "--robot-head-catalog-path",
        help="Optional path to a robot-head capability catalog JSON file.",
    )
    parser.add_argument(
        "--robot-head-port",
        help="Optional serial device path for the live robot head.",
    )
    parser.add_argument(
        "--robot-head-baud",
        type=int,
        help="Optional baud rate override for the live robot head.",
    )
    parser.add_argument(
        "--robot-head-hardware-profile-path",
        help="Optional path to a live robot-head hardware profile JSON file.",
    )
    parser.add_argument(
        "--robot-head-live-arm",
        action="store_true",
        help="Explicitly arm live robot-head motion for this session.",
    )
    parser.add_argument(
        "--robot-head-arm-ttl-seconds",
        type=int,
        help="TTL in seconds for the live robot-head arm lease.",
    )
    parser.add_argument(
        "--robot-head-operator-mode",
        action="store_true",
        help="Expose raw state and motif robot-head tools for operator-only calibration.",
    )
    parser.add_argument(
        "--robot-head-sim-scenario",
        help="Optional path to a robot-head simulation scenario JSON file.",
    )
    parser.add_argument(
        "--robot-head-sim-realtime",
        action="store_true",
        help="Run robot-head simulation timing in wall-clock time instead of virtual time.",
    )
    parser.add_argument(
        "--robot-head-sim-trace-dir",
        help="Optional directory for robot-head simulation trace artifacts.",
    )
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision",
        action="store_true",
        help="Enable browser camera inspection with the optional local Moondream backend.",
    )
    vision_group.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable browser camera inspection even when the selected profile enables it.",
    )
    parser.add_argument(
        "--vision-model",
        help="Model id for the optional local vision backend.",
    )
    parser.add_argument(
        "--continuous-perception",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable low-cadence symbolic continuous perception in browser vision sessions.",
    )
    parser.add_argument(
        "--continuous-perception-interval-secs",
        type=float,
        help="Perception broker cadence in seconds when continuous perception is enabled.",
    )
    parser.add_argument(
        "--actor-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write bounded public-safe actor event JSONL traces. Also available via "
            "BLINK_LOCAL_ACTOR_TRACE=1; CLI flags take precedence."
        ),
    )
    parser.add_argument(
        "--actor-trace-dir",
        help=(
            "Directory for actor event traces. Defaults to artifacts/actor_traces/ "
            "or BLINK_LOCAL_ACTOR_TRACE_DIR."
        ),
    )
    parser.add_argument(
        "--performance-episode-v3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write opt-in public-safe PerformanceEpisodeV3 JSONL ledgers. Also "
            "available via BLINK_LOCAL_PERFORMANCE_EPISODE_V3=1; CLI flags take "
            "precedence."
        ),
    )
    parser.add_argument(
        "--performance-episode-v3-dir",
        help=(
            "Directory for PerformanceEpisodeV3 JSONL ledgers. Defaults to "
            "artifacts/performance_episodes_v3/ or "
            "BLINK_LOCAL_PERFORMANCE_EPISODE_V3_DIR."
        ),
    )
    parser.add_argument(
        "--performance-preferences-v3-dir",
        help=(
            "Directory for local PerformancePreferencePair JSONL ledgers. Defaults to "
            "artifacts/performance_preferences_v3/ or "
            "BLINK_LOCAL_PERFORMANCE_PREFERENCES_V3_DIR."
        ),
    )
    parser.add_argument(
        "--actor-surface-v2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable the browser Actor Surface v2 overlay. Defaults on; "
            "also available via BLINK_LOCAL_ACTOR_SURFACE_V2."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging while the browser workflow is running.",
    )
    return parser


def _resolve_float(value: Any) -> float | None:
    """Best-effort float parsing for CLI and environment values."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _read_json_object(request: Request) -> dict[str, Any]:
    """Parse a JSON request body and require an object payload."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return dict(payload)


def resolve_config(args: argparse.Namespace) -> LocalBrowserConfig:
    """Resolve CLI configuration from arguments and environment variables."""
    maybe_load_dotenv()
    profile = resolve_local_runtime_profile(
        runtime="browser",
        profile_id=getattr(args, "config_profile", None),
    )
    language = resolve_local_language(
        args.language or get_local_env("LANGUAGE") or profile_value(profile, "language")
    )
    browser_vision_env = get_local_env("BROWSER_VISION")
    if getattr(args, "no_vision", False):
        vision_enabled = False
    elif getattr(args, "vision", False):
        vision_enabled = True
    elif browser_vision_env in (None, ""):
        vision_enabled = bool(profile_value(profile, "browser_vision", False))
    else:
        vision_enabled = local_env_flag("BROWSER_VISION")
    configured_tts_backend = get_local_env("TTS_BACKEND")
    robot_head_baud = (
        resolve_int(getattr(args, "robot_head_baud", None) or get_local_env("ROBOT_HEAD_BAUD"))
        or 1000000
    )
    robot_head_arm_ttl_seconds = (
        resolve_int(
            getattr(args, "robot_head_arm_ttl_seconds", None)
            or get_local_env("ROBOT_HEAD_ARM_TTL_SECONDS")
        )
        or 300
    )
    robot_head_sim_scenario = getattr(args, "robot_head_sim_scenario", None) or get_local_env(
        "ROBOT_HEAD_SIM_SCENARIO"
    )
    robot_head_sim_trace_dir = getattr(args, "robot_head_sim_trace_dir", None) or get_local_env(
        "ROBOT_HEAD_SIM_TRACE_DIR"
    )
    tts_backend_locked = args.tts_backend not in (None, "") or configured_tts_backend not in (
        None,
        "",
    )
    stt_backend = (
        args.stt_backend
        or get_local_env("STT_BACKEND")
        or profile_value(profile, "stt_backend")
        or DEFAULT_LOCAL_STT_BACKEND
    )
    tts_backend = (
        args.tts_backend
        or configured_tts_backend
        or profile_value(profile, "tts_backend")
        or default_local_tts_backend(language)
    )
    explicit_tts_voice = args.tts_voice if args.tts_voice not in (None, "") else None

    stt_model = (
        args.stt_model
        or get_local_env("STT_MODEL")
        or profile_value(profile, "stt_model")
        or default_local_stt_model(backend=stt_backend, language=language)
    )
    llm = resolve_local_llm_config(
        provider=getattr(args, "llm_provider", None),
        model=args.model,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        language=language,
        temperature=args.temperature,
        default_system_prompt=default_local_speech_system_prompt(language),
        demo_mode=getattr(args, "demo_mode", None),
        max_output_tokens=getattr(args, "max_output_tokens", None),
        speech=True,
        ignore_env_system_prompt=local_env_flag(
            "IGNORE_ENV_SYSTEM_PROMPT",
            bool(profile_value(profile, "ignore_env_system_prompt", False)),
        ),
    )
    tts_voice = resolve_local_tts_voice(
        tts_backend,
        language,
        explicit_voice=explicit_tts_voice,
    )
    continuous_perception_env = get_local_env("CONTINUOUS_PERCEPTION")
    if getattr(args, "continuous_perception", None) is None:
        if continuous_perception_env in (None, ""):
            continuous_perception_enabled = bool(
                profile_value(profile, "continuous_perception", vision_enabled)
            )
        else:
            continuous_perception_enabled = local_env_flag("CONTINUOUS_PERCEPTION")
    else:
        continuous_perception_enabled = bool(args.continuous_perception)
    if not vision_enabled:
        continuous_perception_enabled = False
    continuous_perception_interval_secs = (
        _resolve_float(getattr(args, "continuous_perception_interval_secs", None))
        or _resolve_float(get_local_env("CONTINUOUS_PERCEPTION_INTERVAL_SECS"))
        or _resolve_float(profile_value(profile, "continuous_perception_interval_secs"))
        or 3.0
    )
    actor_trace_arg = getattr(args, "actor_trace", None)
    actor_trace_enabled = (
        local_env_flag("ACTOR_TRACE", False)
        if actor_trace_arg is None
        else bool(actor_trace_arg)
    )
    actor_trace_dir_value = (
        getattr(args, "actor_trace_dir", None)
        or get_local_env("ACTOR_TRACE_DIR")
        or "artifacts/actor_traces"
    )
    performance_episode_v3_arg = getattr(args, "performance_episode_v3", None)
    performance_episode_v3_enabled = (
        local_env_flag("PERFORMANCE_EPISODE_V3", False)
        if performance_episode_v3_arg is None
        else bool(performance_episode_v3_arg)
    )
    performance_episode_v3_dir_value = (
        getattr(args, "performance_episode_v3_dir", None)
        or get_local_env("PERFORMANCE_EPISODE_V3_DIR")
        or "artifacts/performance_episodes_v3"
    )
    performance_preferences_v3_dir_value = (
        getattr(args, "performance_preferences_v3_dir", None)
        or get_local_env("PERFORMANCE_PREFERENCES_V3_DIR")
        or "artifacts/performance_preferences_v3"
    )
    actor_surface_v2_arg = getattr(args, "actor_surface_v2", None)
    actor_surface_v2_enabled = (
        local_env_flag("ACTOR_SURFACE_V2", True)
        if actor_surface_v2_arg is None
        else bool(actor_surface_v2_arg)
    )

    return LocalBrowserConfig(
        base_url=llm.base_url,
        model=llm.model,
        system_prompt=llm.system_prompt,
        language=language,
        stt_backend=stt_backend,
        tts_backend=tts_backend,
        stt_model=stt_model,
        tts_voice=tts_voice,
        tts_base_url=resolve_local_tts_base_url(tts_backend),
        host=args.host or get_local_env("HOST") or DEFAULT_LOCAL_HOST,
        port=resolve_int(args.port or get_local_env("PORT")) or DEFAULT_LOCAL_PORT,
        robot_head_driver=getattr(args, "robot_head_driver", None)
        or get_local_env("ROBOT_HEAD_DRIVER", "none"),
        robot_head_catalog_path=(
            getattr(args, "robot_head_catalog_path", None)
            or get_local_env("ROBOT_HEAD_CATALOG_PATH")
        ),
        robot_head_port=getattr(args, "robot_head_port", None) or get_local_env("ROBOT_HEAD_PORT"),
        robot_head_baud=robot_head_baud,
        robot_head_hardware_profile_path=(
            getattr(args, "robot_head_hardware_profile_path", None)
            or get_local_env("ROBOT_HEAD_HARDWARE_PROFILE_PATH")
        ),
        robot_head_live_arm=(
            getattr(args, "robot_head_live_arm", False) or local_env_flag("ROBOT_HEAD_ARM")
        ),
        robot_head_arm_ttl_seconds=max(robot_head_arm_ttl_seconds, 30),
        robot_head_operator_mode=(
            getattr(args, "robot_head_operator_mode", False)
            or local_env_flag("ROBOT_HEAD_OPERATOR_MODE")
        ),
        robot_head_sim_scenario_path=(
            Path(robot_head_sim_scenario).expanduser()
            if robot_head_sim_scenario not in (None, "")
            else None
        ),
        robot_head_sim_realtime=(
            getattr(args, "robot_head_sim_realtime", False)
            or local_env_flag("ROBOT_HEAD_SIM_REALTIME")
        ),
        robot_head_sim_trace_dir=(
            Path(robot_head_sim_trace_dir).expanduser()
            if robot_head_sim_trace_dir not in (None, "")
            else None
        ),
        vision_enabled=vision_enabled,
        continuous_perception_enabled=continuous_perception_enabled,
        continuous_perception_interval_secs=max(0.5, float(continuous_perception_interval_secs)),
        vision_model=args.vision_model
        or get_local_env("VISION_MODEL")
        or profile_value(profile, "vision_model")
        or DEFAULT_LOCAL_VISION_MODEL,
        allow_barge_in=getattr(args, "allow_barge_in", False)
        or local_env_flag(
            "ALLOW_BARGE_IN",
            bool(profile_value(profile, "allow_barge_in", False)),
        ),
        temperature=args.temperature,
        verbose=args.verbose,
        tts_backend_locked=tts_backend_locked,
        tts_voice_override=explicit_tts_voice,
        llm_provider=llm.provider,
        llm_service_tier=llm.service_tier,
        demo_mode=llm.demo_mode,
        llm_max_output_tokens=llm.max_output_tokens,
        tts_runtime_label=get_local_env("TTS_RUNTIME_LABEL"),
        config_profile=profile.profile_id if profile is not None else None,
        actor_trace_enabled=actor_trace_enabled,
        actor_trace_dir=Path(actor_trace_dir_value).expanduser(),
        performance_episode_v3_enabled=performance_episode_v3_enabled,
        performance_episode_v3_dir=Path(performance_episode_v3_dir_value).expanduser(),
        performance_preferences_v3_dir=Path(performance_preferences_v3_dir_value).expanduser(),
        actor_surface_v2_enabled=actor_surface_v2_enabled,
    )


def _create_robot_head_driver(config: LocalBrowserConfig):
    """Build the configured robot-head driver for the browser runtime."""
    if config.robot_head_driver == "mock":
        return MockDriver()
    if config.robot_head_driver == "preview":
        return PreviewDriver(trace_dir=Path.cwd() / "artifacts" / "robot_head_preview")
    if config.robot_head_driver == "simulation":
        return SimulationDriver(
            config=RobotHeadSimulationConfig(
                hardware_profile_path=config.robot_head_hardware_profile_path,
                scenario_path=config.robot_head_sim_scenario_path,
                realtime=config.robot_head_sim_realtime,
                trace_dir=config.robot_head_sim_trace_dir,
            )
        )
    if config.robot_head_driver == "live":
        return LiveDriver(
            config=RobotHeadLiveDriverConfig(
                hardware_profile_path=config.robot_head_hardware_profile_path,
                port=config.robot_head_port,
                baud_rate=config.robot_head_baud,
                arm_enabled=config.robot_head_live_arm,
                arm_ttl_seconds=config.robot_head_arm_ttl_seconds,
            ),
            preview_driver=PreviewDriver(trace_dir=Path.cwd() / "artifacts" / "robot_head_preview"),
        )
    raise ValueError(f"Unsupported robot-head driver: {config.robot_head_driver}")


def _robot_head_runtime_label(config: LocalBrowserConfig) -> str:
    """Return a concise runtime label for browser robot-head mode."""
    if config.robot_head_driver == "simulation":
        timing = "realtime" if config.robot_head_sim_realtime else "virtual"
        return f"simulation({timing})"
    if config.robot_head_driver != "live":
        return config.robot_head_driver
    return f"live(port={config.robot_head_port or 'auto'}, armed={config.robot_head_live_arm})"


def _runtime_state_label(enabled: bool) -> str:
    """Return the on/off label used in local runtime readiness output."""
    return "on" if enabled else "off"


def _browser_tts_runtime_label(config: LocalBrowserConfig) -> str:
    """Return the concise TTS label for browser readiness output."""
    label = (config.tts_runtime_label or "").strip()
    if label and "\n" not in label and "\r" not in label:
        return label
    return config.tts_backend


def _speech_director_mode_for_config(config: LocalBrowserConfig) -> str:
    """Return the public speech director mode for a browser profile."""
    profile = str(config.config_profile or "").strip()
    backend = str(config.tts_backend or "").strip().lower()
    if profile == "browser-zh-melo" and backend in {"local-http-wav", "local_http_wav"}:
        return "melo_chunked"
    if profile == "browser-en-kokoro" and backend == "kokoro":
        return "kokoro_chunked"
    return "unavailable"


def _browser_readiness_summary(config: LocalBrowserConfig, *, client_url: str) -> str:
    """Return the canonical browser/WebRTC runtime readiness summary."""
    return (
        f"{PROJECT_IDENTITY.display_name} browser ready: "
        "runtime=browser "
        "transport=WebRTC "
        f"profile={config.config_profile or 'manual'} "
        f"language={config.language.value} "
        f"tts={_browser_tts_runtime_label(config)} "
        f"vision={_runtime_state_label(config.vision_enabled)} "
        f"continuous_perception={_runtime_state_label(config.continuous_perception_enabled)} "
        f"protected_playback={_runtime_state_label(not config.allow_barge_in)} "
        f"barge_in={_runtime_state_label(config.allow_barge_in)} "
        f"barge_in_policy={'armed' if config.allow_barge_in else 'protected'} "
        f"client={client_url}"
    )


def _browser_bind_host(host: str) -> str:
    """Return the local address used for bind preflight checks."""
    if host in {"", "0.0.0.0"}:
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _browser_port_is_occupied(host: str, port: int) -> bool:
    """Return true when the requested browser bind address already accepts TCP."""
    if port <= 0:
        return False
    try:
        address_infos = socket.getaddrinfo(
            _browser_bind_host(host),
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False

    for family, socktype, proto, _canonname, sockaddr in address_infos:
        with socket.socket(family, socktype, proto) as sock:
            sock.settimeout(0.25)
            try:
                if sock.connect_ex(sockaddr) == 0:
                    return True
            except OSError:
                continue
    return False


def _browser_port_conflict_message(config: LocalBrowserConfig) -> str:
    bind_host = _browser_bind_host(config.host)
    return (
        f"Browser server port {bind_host}:{config.port} is already in use. "
        "Run only one browser/WebRTC path on a port at a time. Stop the existing "
        "browser path first, or pass --port with a free port. "
        "For a tmux-launched Melo session, run: tmux kill-session -t blink-browser-melo"
    )


def build_local_browser_runtime(
    config: LocalBrowserConfig,
    *,
    transport,
    idle_timeout_secs: Optional[float] = None,
    stt=None,
    llm=None,
    tts=None,
    tts_session=None,
    vision=None,
    active_client: Optional[dict[str, Any]] = None,
    performance_emit: BrowserPerformanceEmit | None = None,
    performance_resting_mode_provider: Callable[[], BrowserInteractionMode] | None = None,
    floor_input_emit: BrowserFloorInputEmit | None = None,
    interruption_state: BrowserInterruptionStateTracker | None = None,
    actor_control_scheduler: ActorControlScheduler | None = None,
    actor_control_frame_provider: Callable[[], Any] | None = None,
    runtime_session: BrowserRuntimeSessionV3 | None = None,
):
    """Build the shared runtime objects for the local browser voice flow."""
    def emit_performance_event(**kwargs):
        if performance_emit is not None:
            return performance_emit(**kwargs)
        return None

    def resting_performance_mode() -> BrowserInteractionMode:
        if performance_resting_mode_provider is not None:
            return performance_resting_mode_provider()
        return BrowserInteractionMode.CONNECTED

    def latest_actor_control_frame():
        if actor_control_frame_provider is None:
            return None
        try:
            return actor_control_frame_provider()
        except Exception:
            return None

    interruption_state = interruption_state or BrowserInterruptionStateTracker(
        protected_playback=not config.allow_barge_in,
        performance_emit=emit_performance_event,
    )
    interruption_state.set_protected_playback(not config.allow_barge_in)

    stt = stt or create_local_stt_service(
        backend=config.stt_backend,
        model=config.stt_model,
        language=config.language,
    )
    tts = tts or create_local_tts_service(
        backend=config.tts_backend,
        voice=config.tts_voice,
        language=config.language,
        base_url=config.tts_base_url,
        aiohttp_session=tts_session,
        reuse_context_id_within_turn=False,
    )
    pre_llm_processors = []
    pre_output_processors = []
    post_context_processors = []
    robot_head_tools = None
    brain_tool_prompt = memory_tool_prompt(config.language)
    if config.robot_head_driver != "none":
        robot_prompt = (
            robot_head_tool_prompt(config.language)
            if config.robot_head_operator_mode
            else embodied_action_tool_prompt(config.language)
        )
        brain_tool_prompt = " ".join(part for part in [brain_tool_prompt, robot_prompt] if part)

    runtime_base_prompt = " ".join(
        part for part in [config.system_prompt, brain_tool_prompt] if part
    )
    if config.vision_enabled:
        camera_tool_prompt = (
            "当用户询问摄像头里有什么时，请使用 fetch_user_image 工具，不要说你看不到摄像头。"
            "这个工具只检查一帧最新的静态画面；回答时要基于这一帧，不要声称自己在连续看视频。"
            "工具返回后，请直接根据观察结果回答，不要插入运行时内部机制说明，除非用户明确追问。"
            if is_chinese_language(config.language)
            else "When the user asks about what you can see in their camera, use the "
            "`fetch_user_image` tool instead of saying you cannot see the camera. "
            "The tool inspects one latest still frame only, so ground the answer in "
            "that frame and do not claim continuous video understanding. "
            "After the tool returns, answer directly from the observation instead of explaining runtime internals unless the user explicitly asks."
        )
        runtime_base_prompt = " ".join(
            part for part in [runtime_base_prompt, camera_tool_prompt] if part
        )

    def _browser_user_mute_strategies():
        return (
            []
            if config.allow_barge_in
            else [BrowserProtectedPlaybackMuteStrategy(interruption_state=interruption_state)]
        )

    def _browser_user_turn_start_strategies():
        if not config.allow_barge_in:
            return None
        return [BrowserBargeInTurnStartStrategy(interruption_state=interruption_state)]

    def _browser_voice_actuation_plan(brain_runtime: BrainRuntime, context: LLMContext):
        plan = brain_runtime.current_voice_actuation_plan(
            latest_user_text=latest_user_text_from_context(context),
        )
        if not config.allow_barge_in:
            return plan
        reason_codes = tuple(
            dict.fromkeys(
                (
                    *getattr(plan, "reason_codes", ()),
                    "browser_barge_in:stale_output_discard_enabled",
                    "browser_barge_in:partial_stream_abort_noop",
                )
            )
        )
        applied_hints = tuple(
            dict.fromkeys((*getattr(plan, "applied_hints", ()), "interruption_discard"))
        )
        active_hints = tuple(
            dict.fromkeys((*getattr(plan, "active_hints", ()), "interruption_discard"))
        )
        unsupported_hints = tuple(
            hint
            for hint in getattr(plan, "unsupported_hints", ())
            if hint != "interruption_discard"
        )
        return replace(
            plan,
            interruption_flush_enabled=False,
            interruption_discard_enabled=True,
            partial_stream_abort_enabled=False,
            applied_hints=applied_hints,
            active_hints=active_hints,
            unsupported_hints=unsupported_hints,
            reason_codes=reason_codes,
        )

    controller = None
    if config.robot_head_driver != "none":
        catalog = load_robot_head_capability_catalog(config.robot_head_catalog_path)
        controller = RobotHeadController(catalog=catalog, driver=_create_robot_head_driver(config))
    llm_service_config = LocalLLMConfig(
        provider=config.llm_provider,
        model=config.model,
        base_url=config.base_url,
        system_prompt="",
        temperature=config.temperature,
        service_tier=config.llm_service_tier,
        demo_mode=config.demo_mode,
        max_output_tokens=config.llm_max_output_tokens,
    )

    if not config.vision_enabled:
        llm = llm or create_local_llm_service(llm_service_config)
        session_resolver = build_session_resolver(
            runtime_kind="browser", active_client=active_client
        )
        brain_runtime = BrainRuntime(
            base_prompt=runtime_base_prompt,
            language=config.language,
            runtime_kind="browser",
            session_resolver=session_resolver,
            llm=llm,
            robot_head_controller=controller,
            robot_head_operator_mode=config.robot_head_operator_mode,
            vision_enabled=False,
            continuous_perception_enabled=False,
            tts_backend=config.tts_backend,
            performance_event_sink=emit_performance_event,
        )
        brain_runtime.performance_preferences_v3_dir = (
            config.performance_preferences_v3_dir or PERFORMANCE_PREFERENCE_ARTIFACT_DIR
        )
        robot_head_tools = brain_runtime.register_daily_tools()
        context = LLMContext(tools=robot_head_tools)
        brain_runtime.bind_context(context)
        brain_runtime.start_background_maintenance()
        setattr(context, "blink_brain_runtime", brain_runtime)
        pre_llm_processors = list(brain_runtime.pre_llm_processors)
        pre_stt_processors = [
            BrainVoiceInputHealthProcessor(runtime=brain_runtime, phase="pre_stt"),
            BrowserPerformanceFrameObserver(
                phase="pre_stt",
                performance_emit=emit_performance_event,
                resting_mode_provider=resting_performance_mode,
                floor_input_emit=floor_input_emit,
                runtime_session=runtime_session,
            ),
        ]
        post_stt_processors = [
            BrainVoiceInputHealthProcessor(runtime=brain_runtime, phase="post_stt"),
            BrowserPerformanceFrameObserver(
                phase="post_stt",
                performance_emit=emit_performance_event,
                resting_mode_provider=resting_performance_mode,
                floor_input_emit=floor_input_emit,
                runtime_session=runtime_session,
            ),
        ]
        voice_policy_processor = BrainExpressionVoicePolicyProcessor(
            policy_provider=lambda: brain_runtime.current_voice_policy(
                latest_user_text=latest_user_text_from_context(context),
            ),
            actuation_plan_provider=lambda: _browser_voice_actuation_plan(
                brain_runtime, context
            ),
            tts_backend=config.tts_backend,
            language=config.language.value,
            metrics_recorder=brain_runtime.voice_metrics_recorder,
            speech_director_mode=_speech_director_mode_for_config(config),
            performance_emit=emit_performance_event,
            actor_control_scheduler=actor_control_scheduler,
            speech_queue_controller=getattr(runtime_session, "speech", None),
            performance_plan_provider=lambda: brain_runtime.current_memory_persona_performance_plan(
                profile=config.config_profile or "manual",
                tts_label=_browser_tts_runtime_label(config),
                protected_playback=not config.allow_barge_in,
                current_turn_state="thinking",
                actor_control_frame=latest_actor_control_frame(),
            ).performance_plan_v3,
        )
        pre_tts_processors = [
            BrowserPerformanceFrameObserver(
                phase="post_llm",
                performance_emit=emit_performance_event,
                resting_mode_provider=resting_performance_mode,
                floor_input_emit=floor_input_emit,
                runtime_session=runtime_session,
            ),
            voice_policy_processor,
            BrowserInterruptedOutputGuardProcessor(
                interruption_state=interruption_state,
                metrics_recorder=brain_runtime.voice_metrics_recorder,
                performance_emit=emit_performance_event,
            ),
        ]
        pre_output_processors.append(
            BrowserPerformanceFrameObserver(
                phase="post_tts",
                performance_emit=emit_performance_event,
                resting_mode_provider=resting_performance_mode,
                floor_input_emit=floor_input_emit,
                voice_metrics_recorder=brain_runtime.voice_metrics_recorder,
                speech_lookahead_drain=voice_policy_processor.drain_held_speech_chunks,
                runtime_session=runtime_session,
            )
        )
        post_context_processors = list(brain_runtime.post_context_processors)
        if brain_runtime.action_dispatcher is not None:
            pre_output_processors.append(
                EmbodimentPolicyProcessor(
                    action_dispatcher=brain_runtime.action_dispatcher,
                    store=brain_runtime.store,
                    session_resolver=brain_runtime.session_resolver,
                    presence_scope_key=brain_runtime.presence_scope_key,
                )
            )
        return build_local_voice_task(
            transport=transport,
            stt=stt,
            llm=llm,
            tts=tts,
            context=context,
            idle_timeout_secs=idle_timeout_secs,
            extra_user_mute_strategies=_browser_user_mute_strategies(),
            user_turn_start_strategies=_browser_user_turn_start_strategies(),
            pre_stt_processors=pre_stt_processors,
            post_stt_processors=post_stt_processors,
            pre_llm_processors=pre_llm_processors,
            pre_tts_processors=pre_tts_processors,
            pre_output_processors=pre_output_processors,
            post_context_processors=post_context_processors,
        )

    llm = llm or create_local_llm_service(llm_service_config)
    vision_service = vision

    def _get_vision_service():
        """Return the on-demand browser vision service, creating it only when needed."""
        nonlocal vision_service
        if vision_service is None:
            vision_service = create_local_vision_service(model=config.vision_model)
        return vision_service

    camera_frame_buffer = LatestCameraFrameBuffer()
    if hasattr(camera_frame_buffer, "set_performance_emit"):
        camera_frame_buffer.set_performance_emit(emit_performance_event)
    active_client = active_client or {"id": None, "camera_enabled": None}

    def _browser_camera_connected() -> bool:
        if not bool(active_client.get("id")):
            return False
        frame_age_ms = _camera_frame_age_ms(camera_frame_buffer)
        if frame_age_ms is not None and frame_age_ms <= int(
            DEFAULT_CAMERA_STALE_FRAME_SECS * 1000
        ):
            active_client["camera_enabled"] = True
            return True
        return active_client.get("camera_enabled", True) is not False

    session_resolver = build_session_resolver(runtime_kind="browser", active_client=active_client)
    brain_runtime = BrainRuntime(
        base_prompt=runtime_base_prompt,
        language=config.language,
        runtime_kind="browser",
        session_resolver=session_resolver,
        llm=llm,
        robot_head_controller=controller,
        robot_head_operator_mode=config.robot_head_operator_mode,
        vision_enabled=True,
        continuous_perception_enabled=config.continuous_perception_enabled,
            tts_backend=config.tts_backend,
            performance_event_sink=emit_performance_event,
        )
    brain_runtime.performance_preferences_v3_dir = (
        config.performance_preferences_v3_dir or PERFORMANCE_PREFERENCE_ARTIFACT_DIR
    )
    robot_head_tools = brain_runtime.register_daily_tools()
    if brain_runtime.action_dispatcher is not None:
        pre_output_processors.append(
            EmbodimentPolicyProcessor(
                action_dispatcher=brain_runtime.action_dispatcher,
                store=brain_runtime.store,
                session_resolver=brain_runtime.session_resolver,
                presence_scope_key=brain_runtime.presence_scope_key,
            )
        )
    perception_vision = _get_vision_service() if config.continuous_perception_enabled else None
    perception_adapter = LocalPerceptionAdapter(vision=perception_vision)
    camera_health_manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(
            stale_after_secs=DEFAULT_CAMERA_STALE_FRAME_SECS,
        ),
        store=brain_runtime.store,
        session_resolver=brain_runtime.session_resolver,
        presence_scope_key=brain_runtime.presence_scope_key,
        runtime_kind=brain_runtime.runtime_kind,
        camera_buffer=camera_frame_buffer,
        transport=transport,
        camera_connected=_browser_camera_connected,
        vision_enabled=True,
        enrichment_available=perception_adapter.scene_enrichment_available,
        detection_backend=perception_adapter.presence_detection_backend,
        performance_emit=emit_performance_event,
    )
    perception_broker = PerceptionBroker(
        config=PerceptionBrokerConfig(
            enabled=config.continuous_perception_enabled,
            interval_secs=config.continuous_perception_interval_secs,
        ),
        store=brain_runtime.store,
        session_resolver=brain_runtime.session_resolver,
        presence_scope_key=brain_runtime.presence_scope_key,
        camera_buffer=camera_frame_buffer,
        perception_adapter=perception_adapter,
        vision=perception_vision,
        camera_connected=_browser_camera_connected,
        camera_health_provider=camera_health_manager.current_health,
        candidate_goal_sink=brain_runtime.propose_candidate_goal,
        reevaluation_sink=brain_runtime.run_presence_director_reevaluation,
    )
    brain_runtime.note_perception_availability(
        enabled=config.continuous_perception_enabled,
        unreliable=not perception_broker.detector_available,
        unavailable=False,
        detection_backend=perception_adapter.presence_detection_backend,
        enrichment_available=perception_adapter.scene_enrichment_available,
    )

    async def run_vision_query(
        image_frame: UserImageRawFrame, prompt: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Run a single vision query and return the resulting description or error."""
        query_frame = UserImageRawFrame(
            user_id=image_frame.user_id,
            image=image_frame.image,
            size=image_frame.size,
            format=image_frame.format,
            text=prompt,
        )
        query_frame.transport_source = image_frame.transport_source
        query_frame.pts = image_frame.pts

        description: Optional[str] = None
        error_text: Optional[str] = None
        async for result_frame in _get_vision_service().run_vision(query_frame):
            if isinstance(result_frame, VisionTextFrame):
                description = (result_frame.text or "").strip()
            elif isinstance(result_frame, ErrorFrame):
                error_text = result_frame.error

        return description, error_text

    async def fetch_user_image(params: FunctionCallParams):
        user_id = active_client.get("id")
        tool_question = str(params.arguments.get("question", "")).strip()
        user_question = _latest_user_text(context) or tool_question
        session_v3 = runtime_session
        if session_v3 is not None:
            session_v3.note_vision_requested()
        emit_performance_event(
            event_type="vision.fetch_user_image_start",
            source="vision",
            mode=BrowserInteractionMode.LOOKING,
            metadata={
                "question_chars": len(user_question),
                "scene_transition": "looking_requested",
            },
            reason_codes=(
                "vision:fetch_user_image_start",
                "scene_social_transition:looking_requested",
            ),
        )
        if camera_frame_buffer.latest_camera_frame is not None:
            active_client["camera_enabled"] = True
            if session_v3 is not None:
                session_v3.note_camera_frame(
                    frame_seq=camera_frame_buffer.latest_camera_frame_seq,
                    frame_age_ms=_camera_frame_age_ms(camera_frame_buffer),
                )
        hard_camera_unavailable = (
            session_v3.camera.hard_unavailable()
            if session_v3 is not None
            else active_client.get("camera_enabled", True) is False
        )
        if hard_camera_unavailable and camera_frame_buffer.latest_camera_frame is None:
            emit_performance_event(
                event_type="vision.fetch_user_image_error",
                source="vision",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": "unavailable",
                    "scene_transition": "vision_unavailable",
                },
                reason_codes=(
                    "vision:camera_unavailable",
                    "scene_social_transition:vision_unavailable",
                ),
            )
            if session_v3 is not None:
                session_v3.note_vision_error(result_state="unavailable")
            await params.result_callback(
                {
                    "error": (
                        "浏览器当前只连接了麦克风，摄像头不可用。请允许摄像头权限或接入摄像头后重新连接。"
                        if is_chinese_language(config.language)
                        else "The browser is connected in microphone-only mode. Allow camera access or attach a camera, then reconnect for vision."
                    )
                }
            )
            return
        if not user_id:
            emit_performance_event(
                event_type="vision.fetch_user_image_error",
                source="vision",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": "disconnected",
                    "scene_transition": "vision_unavailable",
                },
                reason_codes=(
                    "vision:camera_disconnected",
                    "scene_social_transition:vision_unavailable",
                ),
            )
            if session_v3 is not None:
                session_v3.note_vision_error(result_state="disconnected")
            await params.result_callback(
                {
                    "error": (
                        "摄像头还没有连接。"
                        if is_chinese_language(config.language)
                        else "Camera is not connected yet."
                    )
                }
            )
            return

        latest_frame = camera_frame_buffer.latest_camera_frame
        latest_frame_seq = camera_frame_buffer.latest_camera_frame_seq
        fresh_frame = await camera_frame_buffer.wait_for_latest_camera_frame(
            after_seq=latest_frame_seq,
            timeout=1.0,
        )
        if fresh_frame is not None:
            latest_frame = fresh_frame

        if latest_frame is None:
            emit_performance_event(
                event_type="vision.fetch_user_image_error",
                source="vision",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": "waiting_for_frame",
                    "frame_seq": camera_frame_buffer.latest_camera_frame_seq,
                    "frame_age_ms": _camera_frame_age_ms(camera_frame_buffer),
                    "scene_transition": "vision_unavailable",
                },
                reason_codes=(
                    "vision:camera_frame_missing",
                    "scene_social_transition:vision_unavailable",
                ),
            )
            if session_v3 is not None:
                session_v3.note_vision_error(
                    result_state="waiting_for_frame",
                    frame_seq=camera_frame_buffer.latest_camera_frame_seq,
                    frame_age_ms=_camera_frame_age_ms(camera_frame_buffer),
                )
            await params.result_callback(
                {
                    "error": (
                        "还没有收到摄像头画面，请确认浏览器已经允许摄像头权限并重新连接。"
                        if is_chinese_language(config.language)
                        else "No camera frame has been received yet. Allow camera access and reconnect."
                    )
                }
            )
            return

        if not _camera_frame_is_fresh(
            camera_frame_buffer,
            max_age_secs=DEFAULT_CAMERA_STALE_FRAME_SECS,
        ):
            emit_performance_event(
                event_type="vision.fetch_user_image_error",
                source="vision",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": "stale",
                    "frame_seq": camera_frame_buffer.latest_camera_frame_seq,
                    "frame_age_ms": _camera_frame_age_ms(camera_frame_buffer),
                    "scene_transition": "vision_stale",
                },
                reason_codes=(
                    "vision:camera_frame_stale",
                    "scene_social_transition:vision_stale",
                ),
            )
            if session_v3 is not None:
                session_v3.note_vision_error(
                    result_state="stale",
                    frame_seq=camera_frame_buffer.latest_camera_frame_seq,
                    frame_age_ms=_camera_frame_age_ms(camera_frame_buffer),
                )
            await params.result_callback(
                {
                    "error": (
                        "摄像头画面暂时过期了，请保持摄像头在线并稍等新画面到达后再试。"
                        if is_chinese_language(config.language)
                        else "The latest camera frame is stale. Keep the camera active and try again once a fresh frame arrives."
                    )
                }
            )
            return

        vision_frame = UserImageRawFrame(
            user_id=user_id,
            image=latest_frame.image,
            size=latest_frame.size,
            format=latest_frame.format,
        )
        vision_frame.transport_source = latest_frame.transport_source
        vision_frame.pts = latest_frame.pts

        vision_prompt = _build_vision_prompt(user_question)
        logger.debug(f"Running local browser vision query with prompt: {vision_prompt}")
        description, error_text = await run_vision_query(vision_frame, vision_prompt)

        if _vision_result_is_unusable(description or ""):
            fallback_prompt = (
                "Inspect one latest fresh still camera frame and answer in plain English. "
                "Do not claim continuous video understanding. "
                "Give a short, concrete description of the clearest visible objects, people, "
                "layout, and any large readable text. "
                "If the image is blurry, say what remains confidently visible."
            )
            logger.debug("Retrying local browser vision query with fallback prompt")
            retry_description, retry_error = await run_vision_query(vision_frame, fallback_prompt)
            if not _vision_result_is_unusable(retry_description or ""):
                description = retry_description
                error_text = retry_error or error_text
            elif retry_error:
                error_text = retry_error

        logger.debug(f"Local browser vision raw result: {description!r}")

        if description:
            scene_social_hints = infer_scene_social_hints_from_moondream(
                description,
                language=config.language.value,
            )
            emit_performance_event(
                event_type="vision.fetch_user_image_success",
                source="vision",
                mode=BrowserInteractionMode.THINKING,
                metadata={
                    "description_chars": len(description),
                    "frame_seq": camera_frame_buffer.latest_camera_frame_seq,
                    "frame_age_ms": _camera_frame_age_ms(camera_frame_buffer),
                    "grounding_mode": "single_frame",
                    **scene_social_hints,
                },
                reason_codes=(
                    "vision:fetch_user_image_success",
                    "scene_social_transition:vision_answered",
                ),
            )
            if session_v3 is not None:
                session_v3.note_vision_success(
                    frame_seq=camera_frame_buffer.latest_camera_frame_seq,
                    frame_age_ms=_camera_frame_age_ms(camera_frame_buffer),
                )
            await params.result_callback(
                {"description": _ground_vision_description(description, config.language)}
            )
            return

        emit_performance_event(
            event_type="vision.fetch_user_image_error",
            source="vision",
            mode=BrowserInteractionMode.ERROR,
            metadata={
                "error_type": "vision_result_unavailable",
                "frame_seq": camera_frame_buffer.latest_camera_frame_seq,
                "frame_age_ms": _camera_frame_age_ms(camera_frame_buffer),
                "scene_transition": "vision_unavailable",
            },
            reason_codes=(
                "vision:fetch_user_image_error",
                "scene_social_transition:vision_unavailable",
            ),
        )
        if session_v3 is not None:
            session_v3.note_vision_error(
                result_state="vision_result_unavailable",
                frame_seq=camera_frame_buffer.latest_camera_frame_seq,
                frame_age_ms=_camera_frame_age_ms(camera_frame_buffer),
            )
        await params.result_callback(
            {
                "error": error_text
                or (
                    "摄像头画面分析失败。"
                    if is_chinese_language(config.language)
                    else "Camera analysis failed."
                )
            }
        )

    llm.register_function("fetch_user_image", fetch_user_image)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        if any(call.function_name == "fetch_user_image" for call in function_calls):
            emit_performance_event(
                event_type="vision.fetch_user_image_requested",
                source="llm",
                mode=BrowserInteractionMode.LOOKING,
                metadata={
                    "function_count": len(function_calls),
                    "scene_transition": "looking_requested",
                },
                reason_codes=(
                    "vision:fetch_user_image_requested",
                    "scene_social_transition:looking_requested",
                ),
            )
            await tts.queue_frame(
                TTSSpeakFrame(
                    "我来看看。" if is_chinese_language(config.language) else "Let me take a look."
                )
            )

    fetch_image_function = FunctionSchema(
        name="fetch_user_image",
        description=(
            "当用户询问摄像头里能看到什么时，查看最新的一帧静态摄像头画面。不要把结果描述成连续视频理解。"
            if is_chinese_language(config.language)
            else "Inspect one latest still camera frame when the user asks what is visible. "
            "Do not describe the result as continuous video understanding."
        ),
        properties={
            "question": {
                "type": "string",
                "description": (
                    "用户关于当前摄像头画面的提问，例如“我手里拿着什么？”或“我身后有什么？”"
                    if is_chinese_language(config.language)
                    else "The user's question about the current camera feed, such as "
                    "'What am I holding?' or 'What do you see behind me?'"
                ),
            }
        },
        required=["question"],
    )

    standard_tools = [fetch_image_function, *robot_head_tools.standard_tools]
    context = LLMContext(tools=ToolsSchema(standard_tools=standard_tools))
    brain_runtime.bind_context(context)

    class BrowserAssistantUtteranceSink:
        """Narrow sink for bounded proactive assistant utterances."""

        async def emit_assistant_utterance(self, utterance: CapabilityAssistantUtterance):
            text = str(utterance.text).strip()
            if not text:
                return
            await tts.queue_frame(TTSSpeakFrame(text))

    brain_runtime.set_capability_side_effect_sink(BrowserAssistantUtteranceSink())
    brain_runtime.start_background_maintenance()
    setattr(context, "blink_brain_runtime", brain_runtime)
    setattr(context, "blink_camera_frame_buffer", camera_frame_buffer)
    setattr(context, "blink_camera_health_manager", camera_health_manager)
    setattr(context, "blink_perception_broker", perception_broker)
    pre_llm_processors = list(brain_runtime.pre_llm_processors)
    pre_stt_processors = [
        BrainVoiceInputHealthProcessor(runtime=brain_runtime, phase="pre_stt"),
        BrowserPerformanceFrameObserver(
            phase="pre_stt",
            performance_emit=emit_performance_event,
            resting_mode_provider=resting_performance_mode,
            floor_input_emit=floor_input_emit,
            runtime_session=runtime_session,
        ),
    ]
    post_stt_processors = [
        BrainVoiceInputHealthProcessor(runtime=brain_runtime, phase="post_stt"),
        BrowserPerformanceFrameObserver(
            phase="post_stt",
            performance_emit=emit_performance_event,
            resting_mode_provider=resting_performance_mode,
            floor_input_emit=floor_input_emit,
            runtime_session=runtime_session,
        ),
    ]
    voice_policy_processor = BrainExpressionVoicePolicyProcessor(
        policy_provider=lambda: brain_runtime.current_voice_policy(
            latest_user_text=latest_user_text_from_context(context),
        ),
        actuation_plan_provider=lambda: _browser_voice_actuation_plan(
            brain_runtime, context
        ),
        tts_backend=config.tts_backend,
        language=config.language.value,
        metrics_recorder=brain_runtime.voice_metrics_recorder,
        speech_director_mode=_speech_director_mode_for_config(config),
        performance_emit=emit_performance_event,
        actor_control_scheduler=actor_control_scheduler,
        speech_queue_controller=getattr(runtime_session, "speech", None),
        performance_plan_provider=lambda: brain_runtime.current_memory_persona_performance_plan(
            profile=config.config_profile or "manual",
            tts_label=_browser_tts_runtime_label(config),
            protected_playback=not config.allow_barge_in,
            current_turn_state="thinking",
            actor_control_frame=latest_actor_control_frame(),
        ).performance_plan_v3,
    )
    pre_tts_processors = [
        BrowserPerformanceFrameObserver(
            phase="post_llm",
            performance_emit=emit_performance_event,
            resting_mode_provider=resting_performance_mode,
            floor_input_emit=floor_input_emit,
            runtime_session=runtime_session,
        ),
        voice_policy_processor,
        BrowserInterruptedOutputGuardProcessor(
            interruption_state=interruption_state,
            metrics_recorder=brain_runtime.voice_metrics_recorder,
            performance_emit=emit_performance_event,
        ),
    ]
    pre_output_processors.append(
        BrowserPerformanceFrameObserver(
            phase="post_tts",
            performance_emit=emit_performance_event,
            resting_mode_provider=resting_performance_mode,
            floor_input_emit=floor_input_emit,
            voice_metrics_recorder=brain_runtime.voice_metrics_recorder,
            speech_lookahead_drain=voice_policy_processor.drain_held_speech_chunks,
            runtime_session=runtime_session,
        )
    )
    post_context_processors = list(brain_runtime.post_context_processors)
    user_aggregator, assistant_aggregator = build_local_user_aggregators(
        context,
        extra_user_mute_strategies=_browser_user_mute_strategies(),
        user_turn_start_strategies=_browser_user_turn_start_strategies(),
    )
    task = PipelineTask(
        Pipeline(
            [
                transport.input(),
                camera_frame_buffer,
                *pre_stt_processors,
                stt,
                *post_stt_processors,
                user_aggregator,
                *pre_llm_processors,
                llm,
                *pre_tts_processors,
                tts,
                *pre_output_processors,
                transport.output(),
                assistant_aggregator,
                *post_context_processors,
            ]
        ),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=idle_timeout_secs,
    )

    return task, context


def _safe_note_vision_connected(brain_runtime: Any, connected: bool) -> tuple[str, ...]:
    """Update vision presence without letting shutdown races crash event handlers."""
    if brain_runtime is None:
        return ("vision_presence_runtime_missing",)
    note_vision_connected = getattr(brain_runtime, "note_vision_connected", None)
    if not callable(note_vision_connected):
        return ("vision_presence_surface_missing",)
    try:
        note_vision_connected(connected)
    except Exception as exc:
        logger.warning(
            "Suppressed vision presence update during browser connection cleanup: {}",
            type(exc).__name__,
        )
        return (f"vision_presence_update_suppressed:{type(exc).__name__}",)
    return ("vision_presence_updated",)


async def _run_browser_disconnect_cleanup_step(label: str, step: Any) -> tuple[str, ...]:
    """Run one disconnect cleanup step without blocking the remaining cleanup."""
    if not callable(step):
        return (f"browser_disconnect_cleanup_missing:{label}",)
    try:
        result = step()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        logger.warning(
            "Suppressed browser disconnect cleanup step `{}` failure: {}",
            label,
            type(exc).__name__,
        )
        return (f"browser_disconnect_cleanup_suppressed:{label}:{type(exc).__name__}",)
    return (f"browser_disconnect_cleanup_ok:{label}",)


def _is_benign_aioice_transaction_retry_race(context: dict[str, Any]) -> bool:
    """Return true for aioice STUN retry races after WebRTC close/renegotiation."""
    exc = context.get("exception")
    if not isinstance(exc, asyncio.InvalidStateError):
        return False

    message = str(context.get("message") or "")
    handle = context.get("handle")
    callback = getattr(handle, "_callback", None)
    callback_name = getattr(callback, "__qualname__", "")
    callback_module = getattr(callback, "__module__", "")
    callback_repr = repr(callback)
    details = " ".join((message, callback_name, callback_module, callback_repr))
    return "Transaction.__retry" in details and "aioice" in details


def _is_benign_aiortc_sctp_close_race(context: dict[str, Any]) -> bool:
    """Return true for aiortc SCTP send tasks that race with browser disconnect."""
    exc = context.get("exception")
    if not isinstance(exc, ConnectionError):
        return False
    return "Cannot send encrypted data, not connected" in str(exc)


def _reset_browser_visual_state_on_server_start(config: LocalBrowserConfig) -> None:
    """Mark persisted browser camera state as disconnected before a fresh client connects."""
    if not config.vision_enabled:
        return

    store = None
    try:
        store = BrainStore(path=get_local_env("BRAIN_DB_PATH"))
        session_ids = build_session_resolver(
            runtime_kind="browser",
            active_client={"id": None},
        )()
        snapshot = BrainPresenceSnapshot(
            runtime_kind="browser",
            vision_enabled=True,
            vision_connected=False,
            camera_track_state="disconnected",
            sensor_health_reason="camera_disconnected",
            camera_disconnected=True,
            perception_disabled=not config.continuous_perception_enabled,
            perception_unreliable=False,
            last_fresh_frame_at=None,
            frame_age_ms=None,
            recovery_in_progress=False,
            recovery_attempts=0,
            warnings=[],
            details={"server_start_camera_state": "disconnected"},
        )
        store.append_brain_event(
            event_type=BrainEventType.BODY_STATE_UPDATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="browser_server_start",
            payload={
                "scope_key": "browser:presence",
                "snapshot": snapshot.as_dict(),
            },
        )
    except Exception as exc:  # pragma: no cover - diagnostic guard
        logger.warning(f"Could not reset browser visual state on startup: {exc.__class__.__name__}")
    finally:
        if store is not None:
            store.close()


def create_app(config: LocalBrowserConfig, *, shared_vision=None, runtime_builder=None):
    """Create the SmallWebRTC app for the local browser workflow."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi import Request as FastAPIRequest
        from fastapi import Response as FastAPIResponse
        from fastapi.responses import JSONResponse

        from blink.transports.base_transport import TransportParams
        from blink.transports.smallwebrtc.connection import SmallWebRTCConnection
        from blink.transports.smallwebrtc.request_handler import (
            IceCandidate,
            SmallWebRTCPatchRequest,
            SmallWebRTCRequest,
            SmallWebRTCRequestHandler,
        )
        from blink.transports.smallwebrtc.transport import SmallWebRTCTransport
    except ImportError as exc:  # pragma: no cover - exercised through doctor/runtime
        raise LocalDependencyError(
            "Local browser mode requires the `runner` and `webrtc` extras. "
            "Run `./scripts/bootstrap-local-mac.sh --profile browser`."
        ) from exc

    globals()["Request"] = FastAPIRequest
    globals()["Response"] = FastAPIResponse

    @asynccontextmanager
    async def lifespan(_app):
        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()

        def browser_exception_handler(loop, context):
            if _is_benign_aioice_transaction_retry_race(context):
                logger.debug("Suppressed benign aioice transaction retry race after WebRTC close.")
                return
            if _is_benign_aiortc_sctp_close_race(context):
                logger.debug("Suppressed benign aiortc SCTP send race after WebRTC close.")
                return
            if previous_exception_handler is not None:
                previous_exception_handler(loop, context)
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(browser_exception_handler)
        try:
            _reset_browser_visual_state_on_server_start(config)
            yield
        finally:
            loop.set_exception_handler(previous_exception_handler)
            if active_connections:
                await asyncio.gather(*(pc.disconnect() for pc in active_connections.values()))
                active_connections.clear()
            active_session_configs.clear()
            active_connection_configs.clear()
            active_brain_runtimes.clear()
            _app.state.blink_active_expression_runtime = None
            _app.state.blink_active_llm_context = None
            _app.state.blink_active_camera_context_state = None
            _app.state.blink_active_browser_config = None
            _app.state.blink_client_media_state = {
                "schema_version": 1,
                "available": False,
                "mode": "unreported",
                "camera_state": "unknown",
                "microphone_state": "unknown",
                "echo": {
                    "echo_cancellation": "unknown",
                    "noise_suppression": "unknown",
                    "auto_gain_control": "unknown",
                },
                "updated_at": None,
                "reason_codes": ["browser_media:unreported"],
            }
            _app.state.blink_browser_interaction_mode = BrowserInteractionMode.WAITING
            _app.state.blink_browser_active_session_id = None
            _app.state.blink_browser_active_client_id = None
            _app.state.blink_active_camera_frame_buffer = None
            _app.state.blink_active_camera_health_manager = None
            _app.state.blink_camera_grounding_tracker = BrowserVisionGroundingTracker()
            _app.state.blink_runtime_session_v3 = BrowserRuntimeSessionV3(
                profile=config.config_profile or "manual",
                language=config.language.value,
                tts_runtime_label=_browser_tts_runtime_label(config),
                vision_enabled=config.vision_enabled,
                continuous_perception_enabled=config.continuous_perception_enabled,
                protected_playback=not config.allow_barge_in,
            )
            _app.state.blink_webrtc_audio_health = WebRTCAudioHealthController()
            _app.state.blink_browser_interruption = BrowserInterruptionStateTracker(
                protected_playback=not config.allow_barge_in,
                performance_emit=_emit_performance_event,
            )

    app = FastAPI(lifespan=lifespan)
    active_connections: dict[str, SmallWebRTCConnection] = {}
    active_sessions: dict[str, dict[str, Any]] = {}
    active_session_configs: dict[str, LocalBrowserConfig] = {}
    active_connection_configs: dict[str, LocalBrowserConfig] = {}
    active_brain_runtimes: dict[str, BrainRuntime] = {}
    runtime_payload_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    runtime_payload_cache_lock = threading.RLock()
    app.state.blink_active_expression_runtime = None
    app.state.blink_active_llm_context = None
    app.state.blink_active_camera_context_state = None
    app.state.blink_active_browser_config = None
    app.state.blink_client_media_state = {
        "schema_version": 1,
        "available": False,
        "mode": "unreported",
        "camera_state": "unknown",
        "microphone_state": "unknown",
        "echo": {
            "echo_cancellation": "unknown",
            "noise_suppression": "unknown",
            "auto_gain_control": "unknown",
        },
        "updated_at": None,
        "reason_codes": ["browser_media:unreported"],
    }
    actor_control_scheduler = ActorControlScheduler(
        profile=config.config_profile or "manual",
        language=config.language.value,
        tts_backend=config.tts_backend,
        tts_runtime_label=_browser_tts_runtime_label(config),
    )
    performance_events = BrowserPerformanceEventBus(
        max_events=500,
        actor_control_scheduler=actor_control_scheduler,
    )
    app.state.blink_browser_performance_events = performance_events
    app.state.blink_actor_control_scheduler = actor_control_scheduler
    floor_controller = ConversationFloorController(
        profile=config.config_profile or "manual",
        language=config.language.value,
        protected_playback=not config.allow_barge_in,
        barge_in_armed=config.allow_barge_in,
    )
    app.state.blink_conversation_floor = floor_controller
    app.state.blink_conversation_floor_state = floor_controller.snapshot()
    app.state.blink_webrtc_audio_health = WebRTCAudioHealthController()
    app.state.blink_actor_trace_enabled = bool(config.actor_trace_enabled)
    app.state.blink_actor_trace_path = None
    app.state.blink_performance_episode_v3_enabled = bool(config.performance_episode_v3_enabled)
    app.state.blink_performance_episode_v3_path = None
    app.state.blink_performance_preferences_v3_dir = (
        config.performance_preferences_v3_dir or PERFORMANCE_PREFERENCE_ARTIFACT_DIR
    )
    app.state.blink_browser_interaction_mode = BrowserInteractionMode.WAITING
    app.state.blink_browser_active_session_id = None
    app.state.blink_browser_active_client_id = None
    app.state.blink_active_camera_frame_buffer = None
    app.state.blink_active_camera_health_manager = None
    app.state.blink_camera_grounding_tracker = BrowserVisionGroundingTracker()
    app.state.blink_runtime_session_v3 = BrowserRuntimeSessionV3(
        profile=config.config_profile or "manual",
        language=config.language.value,
        tts_runtime_label=_browser_tts_runtime_label(config),
        vision_enabled=config.vision_enabled,
        continuous_perception_enabled=config.continuous_perception_enabled,
        protected_playback=not config.allow_barge_in,
    )
    small_webrtc_handler = SmallWebRTCRequestHandler(host=config.host)
    runtime_builder = runtime_builder or build_local_browser_runtime
    # The mounted /client UI is the repo-owned Blink browser bundle.
    mount_smallwebrtc_ui(app)

    def _clear_runtime_payload_cache(*prefixes: str) -> None:
        with runtime_payload_cache_lock:
            if not prefixes:
                runtime_payload_cache.clear()
                return
            for key in list(runtime_payload_cache):
                if any(key.startswith(prefix) for prefix in prefixes):
                    runtime_payload_cache.pop(key, None)

    def _cached_runtime_payload(
        key: str,
        *,
        ttl_secs: float,
        builder,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with runtime_payload_cache_lock:
            cached = runtime_payload_cache.get(key)
            if cached is not None:
                expires_at, payload = cached
                if now < expires_at:
                    return payload
        payload = builder()
        with runtime_payload_cache_lock:
            runtime_payload_cache[key] = (now + max(0.0, ttl_secs), payload)
        return payload

    def _runtime_mutation_payload(payload: dict[str, Any]) -> dict[str, Any]:
        _clear_runtime_payload_cache()
        return payload

    def _active_browser_config() -> LocalBrowserConfig:
        active_config = getattr(app.state, "blink_active_browser_config", None)
        return active_config if isinstance(active_config, LocalBrowserConfig) else config

    def _runtime_session_v3() -> BrowserRuntimeSessionV3:
        active_config = _active_browser_config()
        session = getattr(app.state, "blink_runtime_session_v3", None)
        if not isinstance(session, BrowserRuntimeSessionV3):
            session = BrowserRuntimeSessionV3(
                profile=active_config.config_profile or "manual",
                language=active_config.language.value,
                tts_runtime_label=_browser_tts_runtime_label(active_config),
                vision_enabled=active_config.vision_enabled,
                continuous_perception_enabled=active_config.continuous_perception_enabled,
                protected_playback=not active_config.allow_barge_in,
            )
            app.state.blink_runtime_session_v3 = session
        else:
            session.configure(
                profile=active_config.config_profile or "manual",
                language=active_config.language.value,
                tts_runtime_label=_browser_tts_runtime_label(active_config),
                vision_enabled=active_config.vision_enabled,
                continuous_perception_enabled=active_config.continuous_perception_enabled,
                protected_playback=not active_config.allow_barge_in,
            )
        return session

    def _actor_event_context() -> ActorEventContext:
        active_config = _active_browser_config()
        actor_control_scheduler.configure(
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            tts_backend=active_config.tts_backend,
            tts_runtime_label=_browser_tts_runtime_label(active_config),
        )
        return ActorEventContext(
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            tts_backend=active_config.tts_backend,
            tts_label=_browser_tts_runtime_label(active_config),
            vision_backend="moondream" if active_config.vision_enabled else "none",
        )

    performance_events.set_actor_context_provider(_actor_event_context)
    if config.actor_trace_enabled:
        actor_trace_writer = ActorTraceWriter(
            trace_dir=config.actor_trace_dir or Path("artifacts/actor_traces"),
            profile=config.config_profile or "manual",
            run_id=uuid4().hex[:12],
        )
        performance_events.set_actor_trace_writer(actor_trace_writer)
        app.state.blink_actor_trace_path = str(actor_trace_writer.path)
    if config.performance_episode_v3_enabled:
        performance_episode_writer = PerformanceEpisodeV3Writer(
            episode_dir=config.performance_episode_v3_dir
            or Path("artifacts/performance_episodes_v3"),
            profile=config.config_profile or "manual",
            run_id=uuid4().hex[:12],
        )
        performance_events.set_performance_episode_writer(performance_episode_writer)
        app.state.blink_performance_episode_v3_path = str(performance_episode_writer.path)

    def _active_brain_runtime():
        runtime = getattr(app.state, "blink_active_expression_runtime", None)
        if runtime is None:
            runtime = next(iter(active_brain_runtimes.values()), None)
        return runtime

    discourse_episode_collector = DiscourseEpisodeV3Collector(
        runtime_resolver=_active_brain_runtime,
        max_events=400,
    )
    performance_events.set_discourse_episode_collector(discourse_episode_collector)
    app.state.blink_discourse_episode_v3_collector = discourse_episode_collector

    def _active_performance_session_id() -> str | None:
        value = getattr(app.state, "blink_browser_active_session_id", None)
        return str(value) if value not in (None, "") else None

    def _active_performance_client_id() -> str | None:
        value = getattr(app.state, "blink_browser_active_client_id", None)
        return str(value) if value not in (None, "") else None

    def _resting_interaction_mode() -> BrowserInteractionMode:
        if not active_connections and _active_performance_client_id() is None:
            return BrowserInteractionMode.WAITING
        browser_media = _current_client_media_payload()
        if browser_media["mode"] == "unreported":
            return BrowserInteractionMode.CONNECTED
        if browser_media["microphone_state"] in {"ready", "receiving"}:
            return BrowserInteractionMode.LISTENING
        if browser_media["microphone_state"] in {
            "error",
            "permission_denied",
            "device_not_found",
            "stalled",
            "unavailable",
        }:
            return BrowserInteractionMode.ERROR
        return BrowserInteractionMode.CONNECTED

    def _set_interaction_mode(mode: BrowserInteractionMode | str) -> BrowserInteractionMode:
        try:
            if isinstance(mode, BrowserInteractionMode):
                resolved_mode = mode
            else:
                resolved_mode = BrowserInteractionMode(str(mode))
        except ValueError:
            resolved_mode = BrowserInteractionMode.WAITING
        app.state.blink_browser_interaction_mode = resolved_mode
        return resolved_mode

    def _conversation_floor_policy_hints(
        active_config: LocalBrowserConfig,
    ) -> dict[str, Any]:
        browser_media = _current_client_media_payload()
        echo_safe = active_config.allow_barge_in or browser_media.get("echo_safe") is True
        fallback = {
            "echo_safe": echo_safe,
            "echo_risk": "low" if echo_safe else "unknown",
            "barge_in_state": "armed" if active_config.allow_barge_in else "protected",
        }
        try:
            current_mode = getattr(
                app.state,
                "blink_browser_interaction_mode",
                BrowserInteractionMode.WAITING,
            )
            snapshot = _webrtc_audio_health_controller().snapshot(
                profile=active_config.config_profile or "manual",
                language=active_config.language.value,
                browser_media=browser_media,
                protected_playback=not active_config.allow_barge_in,
                explicit_barge_in_armed=active_config.allow_barge_in,
                assistant_speaking=current_mode == BrowserInteractionMode.SPEAKING,
                interruption={},
            ).as_dict()
        except NameError:
            return fallback
        barge_in_state = str(snapshot.get("barge_in_state") or fallback["barge_in_state"])
        echo_risk = str(snapshot.get("echo_risk") or fallback["echo_risk"])
        return {
            "echo_safe": echo_safe
            or (
                barge_in_state in {"armed", "adaptive"}
                and echo_risk == "low"
            ),
            "echo_risk": echo_risk,
            "barge_in_state": barge_in_state,
        }

    def _conversation_floor_controller() -> ConversationFloorController:
        active_config = _active_browser_config()
        controller = getattr(app.state, "blink_conversation_floor", None)
        if not isinstance(controller, ConversationFloorController):
            controller = ConversationFloorController()
            app.state.blink_conversation_floor = controller
        policy_hints = _conversation_floor_policy_hints(active_config)
        controller.configure(
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            protected_playback=not active_config.allow_barge_in,
            barge_in_armed=active_config.allow_barge_in,
            echo_safe=policy_hints["echo_safe"],
            echo_risk=policy_hints["echo_risk"],
            barge_in_state=policy_hints["barge_in_state"],
        )
        return controller

    def _floor_interaction_mode(state: ConversationFloorState) -> BrowserInteractionMode:
        floor_state = str(state.as_dict().get("state") or "")
        if floor_state == ConversationFloorStatus.USER_HAS_FLOOR.value:
            return BrowserInteractionMode.LISTENING
        if floor_state == ConversationFloorStatus.ASSISTANT_HAS_FLOOR.value:
            return (
                BrowserInteractionMode.SPEAKING
                if state.assistant_speaking
                else BrowserInteractionMode.THINKING
            )
        if floor_state in {
            ConversationFloorStatus.OVERLAP.value,
            ConversationFloorStatus.REPAIR.value,
        }:
            return BrowserInteractionMode.INTERRUPTED
        return BrowserInteractionMode.WAITING

    def _apply_conversation_floor_input(floor_input: ConversationFloorInput):
        active_config = _active_browser_config()
        controller = _conversation_floor_controller()
        policy_hints = _conversation_floor_policy_hints(active_config)
        enriched_input = replace(
            floor_input,
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            protected_playback=not active_config.allow_barge_in,
            barge_in_armed=active_config.allow_barge_in,
            echo_safe=policy_hints["echo_safe"],
            echo_risk=policy_hints["echo_risk"],
            barge_in_state=policy_hints["barge_in_state"],
        )
        update = controller.apply(enriched_input)
        app.state.blink_conversation_floor_state = update.state
        if update.changed:
            _emit_performance_event(
                event_type="floor.transition",
                source="floor",
                mode=_floor_interaction_mode(update.state),
                metadata=update.event_metadata(),
                reason_codes=update.state.reason_codes,
            )
        _clear_runtime_payload_cache("performance", "actor")
        return update

    def _conversation_floor_state_payload() -> dict[str, Any]:
        controller = _conversation_floor_controller()
        state = controller.snapshot()
        app.state.blink_conversation_floor_state = state
        return state.as_dict()

    def _webrtc_audio_health_controller() -> WebRTCAudioHealthController:
        controller = getattr(app.state, "blink_webrtc_audio_health", None)
        if not isinstance(controller, WebRTCAudioHealthController):
            controller = WebRTCAudioHealthController()
            app.state.blink_webrtc_audio_health = controller
        return controller

    def _record_webrtc_audio_health_event(event_type: str, metadata: dict[str, Any] | None) -> None:
        controller = _webrtc_audio_health_controller()
        if event_type in {"tts.speech_start", "speech.audio_start"}:
            controller.set_output_playback_state(
                "playing",
                reason_code="output_playback:playing",
            )
        elif event_type in {
            "tts.speech_end",
            "runtime.task_finished",
            "interruption.listening_resumed",
            "interruption.output_flushed",
        }:
            controller.set_output_playback_state("idle", reason_code="output_playback:idle")
        elif event_type == "microphone.track_stalled":
            controller.set_input_track_state("stalled", reason_code="input_track:stalled")
        elif event_type == "microphone.track_resumed":
            controller.set_input_track_state("receiving", reason_code="input_track:receiving")
        elif event_type in {"webrtc.client_disconnected", "webrtc.connection_closed"}:
            controller.set_input_track_state("unknown", reason_code="input_track:unknown")
            controller.set_output_playback_state("idle", reason_code="output_playback:idle")
        elif event_type == "browser_media.reported":
            mic_state = str((metadata or {}).get("microphone_state") or "unknown")
            if mic_state in {"ready", "receiving", "stalled"}:
                controller.set_input_track_state(
                    "receiving" if mic_state == "ready" else mic_state,
                    reason_code=f"input_track:{mic_state}",
                )

    def _webrtc_audio_health_payload(
        *,
        active_config: LocalBrowserConfig,
        browser_media: dict[str, Any],
        current_mode: BrowserInteractionMode,
        interruption_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        controller = _webrtc_audio_health_controller()
        snapshot = controller.snapshot(
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            browser_media=browser_media,
            protected_playback=not active_config.allow_barge_in,
            explicit_barge_in_armed=active_config.allow_barge_in,
            assistant_speaking=current_mode == BrowserInteractionMode.SPEAKING,
            interruption=interruption_payload or {},
        )
        payload = snapshot.as_dict()
        interruption_state = getattr(app.state, "blink_browser_interruption", None)
        if isinstance(interruption_state, BrowserInterruptionStateTracker):
            interruption_state.set_audio_health_policy(
                barge_in_state=payload.get("barge_in_state"),
                echo_risk=payload.get("echo_risk"),
            )
        return payload

    def _event_metadata_int(metadata: dict[str, Any] | None, key: str) -> int | None:
        if not isinstance(metadata, dict):
            return None
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    def _record_interruption_floor_event(
        *,
        event_type: str,
        metadata: dict[str, Any] | None,
        reason_codes: tuple[str, ...] | list[str] | None,
    ) -> None:
        floor_input_type: ConversationFloorInputType | None = None
        if event_type == "interruption.candidate":
            floor_input_type = ConversationFloorInputType.INTERRUPTION_CANDIDATE
        elif event_type == "interruption.accepted":
            floor_input_type = ConversationFloorInputType.INTERRUPTION_ACCEPTED
        elif event_type == "interruption.rejected":
            floor_input_type = ConversationFloorInputType.INTERRUPTION_REJECTED
        elif event_type == "interruption.suppressed":
            floor_input_type = ConversationFloorInputType.INTERRUPTION_SUPPRESSED
        elif event_type == "interruption.listening_resumed":
            floor_input_type = ConversationFloorInputType.INTERRUPTION_RESUMED
        if floor_input_type is None:
            return
        _apply_conversation_floor_input(
            ConversationFloorInput(
                input_type=floor_input_type,
                speech_age_ms=_event_metadata_int(metadata, "speech_age_ms") or 0,
                reason_codes=(
                    f"floor:{event_type.replace('.', '_')}",
                    *(reason_codes or ()),
                ),
            )
        )

    def _camera_grounding_tracker() -> BrowserVisionGroundingTracker:
        tracker = getattr(app.state, "blink_camera_grounding_tracker", None)
        if isinstance(tracker, BrowserVisionGroundingTracker):
            return tracker
        tracker = BrowserVisionGroundingTracker()
        app.state.blink_camera_grounding_tracker = tracker
        return tracker

    def _record_camera_presence_event(
        *,
        event_type: str,
        metadata: dict[str, Any] | None,
        reason_codes: tuple[str, ...] | list[str] | None,
    ) -> None:
        tracker = _camera_grounding_tracker()
        runtime_session = _runtime_session_v3()
        frame_seq = _event_metadata_int(metadata, "frame_seq")
        frame_age_ms = _event_metadata_int(metadata, "frame_age_ms")
        if event_type in {"voice.speech_started", "stt.transcription"}:
            tracker.reset_current_answer()
            return
        if event_type in {"camera.connected", "camera.track_resumed"}:
            tracker.note_scene_transition(
                "camera_ready",
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
                reason_code="scene_social_transition:camera_ready",
            )
            return
        if event_type == "camera.frame_received":
            runtime_session.note_camera_frame(frame_seq=frame_seq, frame_age_ms=frame_age_ms)
            tracker.note_scene_transition(
                "frame_captured",
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
                reason_code="scene_social_transition:frame_captured",
            )
            return
        if event_type in {"camera.frame_stale", "camera.health_stalled", "camera.track_stalled"}:
            tracker.note_scene_transition(
                "vision_stale",
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
                reason_code="scene_social_transition:vision_stale",
            )
            return
        if (
            event_type == "llm.response_start"
            and not tracker.current_answer_used_vision
            and tracker.last_result_state != "looking"
        ):
            tracker.reset_current_answer()
            return
        if event_type in {
            "vision.fetch_user_image_requested",
            "vision.fetch_user_image_start",
        }:
            runtime_session.note_vision_requested()
            tracker.mark_looking(frame_seq=frame_seq, frame_age_ms=frame_age_ms)
            return
        if event_type == "vision.fetch_user_image_success":
            runtime_session.note_vision_success(frame_seq=frame_seq, frame_age_ms=frame_age_ms)
            tracker.mark_success(
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
                scene_social_hints=metadata,
            )
            return
        if event_type == "vision.fetch_user_image_error":
            state = str((metadata or {}).get("camera_state") or "error")
            runtime_session.note_vision_error(
                result_state=state,
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
            )
            tracker.mark_error(
                result_state=state,
                reason_code=next(iter(reason_codes or ()), "vision:last_result_error"),
                frame_seq=frame_seq,
                frame_age_ms=frame_age_ms,
            )

    def _emit_performance_event(
        *,
        event_type: str,
        source: str,
        mode: BrowserInteractionMode | str,
        metadata: dict[str, Any] | None = None,
        reason_codes: tuple[str, ...] | list[str] | None = None,
        session_id: str | None = None,
        client_id: str | None = None,
    ):
        resolved_mode = _set_interaction_mode(mode)
        event = performance_events.emit(
            event_type=event_type,
            source=source,
            mode=resolved_mode,
            session_id=session_id or _active_performance_session_id(),
            client_id=client_id or _active_performance_client_id(),
            metadata=metadata or {},
            reason_codes=reason_codes or (),
        )
        _record_camera_presence_event(
            event_type=event_type,
            metadata=metadata,
            reason_codes=reason_codes,
        )
        _record_webrtc_audio_health_event(event_type, metadata)
        if event_type.startswith("interruption."):
            _record_interruption_floor_event(
                event_type=event_type,
                metadata=metadata,
                reason_codes=reason_codes,
            )
        _clear_runtime_payload_cache("performance")
        return event

    app.state.blink_apply_conversation_floor_input = _apply_conversation_floor_input

    app.state.blink_browser_interruption = BrowserInterruptionStateTracker(
        protected_playback=not config.allow_barge_in,
        performance_emit=_emit_performance_event,
    )

    def _current_camera_health_payload() -> Any:
        manager = getattr(app.state, "blink_active_camera_health_manager", None)
        current_health = getattr(manager, "current_health", None)
        if callable(current_health):
            try:
                return current_health()
            except Exception:
                return None
        return None

    def _public_camera_presence_state_payload(
        active_config: LocalBrowserConfig,
        browser_media: dict[str, Any],
    ) -> dict[str, Any]:
        camera_health = _current_camera_health_payload()
        snapshot = build_browser_camera_presence_snapshot(
            vision_enabled=active_config.vision_enabled,
            continuous_perception_enabled=active_config.continuous_perception_enabled,
            browser_media=browser_media,
            active_client_id=_active_performance_client_id(),
            camera_buffer=getattr(app.state, "blink_active_camera_frame_buffer", None),
            camera_health=camera_health,
            grounding_tracker=_camera_grounding_tracker(),
        )
        return snapshot.as_dict()

    def _public_camera_scene_state_payload(
        active_config: LocalBrowserConfig,
        browser_media: dict[str, Any],
    ) -> dict[str, Any]:
        scene = build_camera_scene_state(
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            vision_enabled=active_config.vision_enabled,
            continuous_perception_enabled=active_config.continuous_perception_enabled,
            browser_media=browser_media,
            active_client_id=_active_performance_client_id(),
            active_session_id=_active_performance_session_id(),
            camera_buffer=getattr(app.state, "blink_active_camera_frame_buffer", None),
            camera_health=_current_camera_health_payload(),
            grounding_tracker=_camera_grounding_tracker(),
            vision_backend="moondream",
        )
        return scene.as_dict()

    def _browser_client_is_disconnected() -> bool:
        return (
            _active_performance_client_id() is None
            and _active_performance_session_id() is None
            and not active_connections
        )

    def _stale_successful_media_without_client(browser_media: dict[str, Any]) -> bool:
        microphone_active = browser_media.get("microphone_state") in {"ready", "receiving"}
        camera_active = browser_media.get("camera_state") in {"ready", "receiving"}
        return (
            browser_media.get("mode") == "camera_and_microphone"
            and camera_active
            and microphone_active
        ) or (
            browser_media.get("mode") == "audio_only"
            and microphone_active
            and browser_media.get("camera_state") not in {
                "permission_denied",
                "device_not_found",
                "error",
            }
        )

    def _public_state_browser_media_payload() -> dict[str, Any]:
        browser_media = _current_client_media_payload()
        if _browser_client_is_disconnected() and (
            browser_media["mode"] == "unreported"
            or _stale_successful_media_without_client(browser_media)
        ):
            return _unreported_client_media_payload(
                "browser_media:unreported",
                "browser_client:disconnected",
            )
        if _browser_client_is_disconnected():
            return {
                **browser_media,
                "reason_codes": _safe_public_reason_codes(
                    (*browser_media["reason_codes"], "browser_client:disconnected"),
                    limit=16,
                ),
            }
        return browser_media

    def _browser_performance_state_payload() -> dict[str, Any]:
        active_config = _active_browser_config()
        browser_media = _public_state_browser_media_payload()
        interruption_state = getattr(app.state, "blink_browser_interruption", None)
        if isinstance(interruption_state, BrowserInterruptionStateTracker):
            interruption_state.set_protected_playback(not active_config.allow_barge_in)
            interruption_payload = interruption_state.snapshot()
        else:
            interruption_payload = BrowserInterruptionStateTracker(
                protected_playback=not active_config.allow_barge_in
            ).snapshot()
        current_mode = getattr(
            app.state,
            "blink_browser_interaction_mode",
            BrowserInteractionMode.WAITING,
        )
        try:
            if not isinstance(current_mode, BrowserInteractionMode):
                current_mode = BrowserInteractionMode(str(current_mode))
        except ValueError:
            current_mode = BrowserInteractionMode.WAITING
        if not active_connections and _active_performance_client_id() is None:
            current_mode = BrowserInteractionMode.WAITING
        elif current_mode == BrowserInteractionMode.WAITING:
            current_mode = _resting_interaction_mode()

        camera_presence = _public_camera_presence_state_payload(active_config, browser_media)
        camera_scene = _public_camera_scene_state_payload(active_config, browser_media)
        memory_persona = _runtime_memory_persona_performance_payload(
            active_config=active_config,
            browser_media=browser_media,
            current_mode=current_mode,
            camera_presence=camera_presence,
            camera_scene=camera_scene,
        )
        state = BrowserInteractionState(
            mode=current_mode,
            profile=active_config.config_profile or "manual",
            tts_label=_browser_tts_runtime_label(active_config),
            tts_backend=active_config.tts_backend,
            protected_playback=not active_config.allow_barge_in,
            browser_media=browser_media,
            vision_enabled=active_config.vision_enabled,
            continuous_perception_enabled=active_config.continuous_perception_enabled,
            memory_available=_active_brain_runtime() is not None,
            interruption=interruption_payload,
            speech=_public_speech_state_payload(active_config),
            active_listening=_public_active_listening_state_payload(),
            camera_presence=camera_presence,
            camera_scene=camera_scene,
            memory_persona=memory_persona,
            active_session_id=_active_performance_session_id(),
            active_client_id=_active_performance_client_id(),
            last_event=performance_events.latest_event,
            reason_codes=[
                "runtime_active:true"
                if _active_brain_runtime() is not None
                else "runtime_active:false"
            ],
        )
        return state.as_dict()

    def _browser_actor_state_payload() -> dict[str, Any]:
        active_config = _active_browser_config()
        browser_media = _public_state_browser_media_payload()
        interruption_state = getattr(app.state, "blink_browser_interruption", None)
        if isinstance(interruption_state, BrowserInterruptionStateTracker):
            interruption_state.set_protected_playback(not active_config.allow_barge_in)
            interruption_payload = interruption_state.snapshot()
        else:
            interruption_payload = BrowserInterruptionStateTracker(
                protected_playback=not active_config.allow_barge_in
            ).snapshot()
        current_mode = getattr(
            app.state,
            "blink_browser_interaction_mode",
            BrowserInteractionMode.WAITING,
        )
        try:
            if not isinstance(current_mode, BrowserInteractionMode):
                current_mode = BrowserInteractionMode(str(current_mode))
        except ValueError:
            current_mode = BrowserInteractionMode.WAITING
        if not active_connections and _active_performance_client_id() is None:
            current_mode = BrowserInteractionMode.WAITING
        elif current_mode == BrowserInteractionMode.WAITING:
            current_mode = _resting_interaction_mode()

        camera_presence = _public_camera_presence_state_payload(active_config, browser_media)
        camera_scene = _public_camera_scene_state_payload(active_config, browser_media)
        memory_persona = _runtime_memory_persona_performance_payload(
            active_config=active_config,
            browser_media=browser_media,
            current_mode=current_mode,
            camera_presence=camera_presence,
            camera_scene=camera_scene,
        )
        webrtc_audio_health = _webrtc_audio_health_payload(
            active_config=active_config,
            browser_media=browser_media,
            current_mode=current_mode,
            interruption_payload=interruption_payload,
        )
        if isinstance(interruption_state, BrowserInterruptionStateTracker):
            interruption_payload = interruption_state.snapshot()
        conversation_floor = _conversation_floor_state_payload()
        active_listening = _public_active_listener_v2_state_payload(
            active_config,
            camera_scene=camera_scene,
            memory_context=memory_persona,
            floor_state=conversation_floor,
        )
        state = BrowserActorStateV2(
            mode=current_mode,
            profile=active_config.config_profile or "manual",
            language=active_config.language.value,
            tts_label=_browser_tts_runtime_label(active_config),
            tts_backend=active_config.tts_backend,
            protected_playback=not active_config.allow_barge_in,
            browser_media=browser_media,
            vision_enabled=active_config.vision_enabled,
            vision_backend="moondream" if active_config.vision_enabled else "none",
            continuous_perception_enabled=active_config.continuous_perception_enabled,
            memory_available=_active_brain_runtime() is not None,
            interruption=interruption_payload,
            webrtc_audio_health=webrtc_audio_health,
            speech=_public_speech_state_payload(active_config),
            active_listening=active_listening,
            conversation_floor=conversation_floor,
            camera_presence=camera_presence,
            camera_scene=camera_scene,
            memory_persona=memory_persona,
            active_session_id=_active_performance_session_id(),
            active_client_id=_active_performance_client_id(),
            last_actor_event=performance_events.actor_latest_event,
            reason_codes=[
                "runtime_active:true"
                if _active_brain_runtime() is not None
                else "runtime_active:false"
            ],
        )
        return state.as_dict()

    def _browser_client_config_payload() -> dict[str, Any]:
        active_config = _active_browser_config()
        return {
            "schema_version": 1,
            "runtime": "browser",
            "transport": "WebRTC",
            "profile": active_config.config_profile or "manual",
            "enableCam": bool(active_config.vision_enabled),
            "enableMic": True,
            "vision_enabled": bool(active_config.vision_enabled),
            "continuous_perception_enabled": bool(
                active_config.continuous_perception_enabled
            ),
            "actor_surface_v2_enabled": bool(active_config.actor_surface_v2_enabled),
            "protected_playback": not active_config.allow_barge_in,
            "reason_codes": [
                "browser_client_config:v1",
                "camera_client:on" if active_config.vision_enabled else "camera_client:off",
                "microphone_client:on",
                "actor_surface_v2:on"
                if active_config.actor_surface_v2_enabled
                else "actor_surface_v2:off",
            ],
        }

    def _runtime_cache_scope() -> str:
        runtime = _active_brain_runtime()
        return str(id(runtime)) if runtime is not None else "none"

    def _model_selection_result(
        *,
        accepted: bool,
        applied: bool,
        profile: LocalLLMModelProfile | None,
        requested_profile_id: str | None,
        reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        public_profile_id = profile.profile_id if profile is not None else requested_profile_id
        payload: dict[str, Any] = {
            "schema_version": 1,
            "accepted": accepted is True,
            "applied": applied is True,
            "profile_id": _safe_public_text(public_profile_id, limit=96),
            "reason_codes": _safe_public_reason_codes(
                (
                    "model_selection:accepted" if accepted else "model_selection:rejected",
                    *reason_codes,
                )
            ),
        }
        if profile is not None:
            payload.update(
                {
                    "label": _safe_public_text(profile.label, limit=120),
                    "provider": _safe_public_text(profile.provider, limit=80),
                    "model": _safe_public_text(profile.model, limit=120),
                    "runtime_tier": _safe_public_text(profile.runtime_tier, limit=80),
                    "latency_tier": _safe_public_text(profile.latency_tier, limit=80),
                    "capability_tier": _safe_public_text(
                        profile.capability_tier,
                        limit=80,
                    ),
                }
            )
        return {key: value for key, value in payload.items() if value not in (None, "")}

    def _requested_model_profile_id(body: dict[str, Any]) -> str | None:
        raw_value = body.get("model_profile_id") or body.get("modelProfileId")
        normalized = " ".join(str(raw_value or "").split()).strip()
        return normalized or None

    def _safe_model_selection_rejection_reason(
        *,
        provider: str | None,
        error: Exception,
    ) -> tuple[str, ...]:
        message = str(error)
        if provider == "ollama":
            if "not available" in message:
                return ("model_profile_unavailable", "ollama_model_unavailable")
            if "Could not reach Ollama" in message:
                return ("model_profile_unavailable", "ollama_unreachable")
            return ("model_profile_unavailable", "ollama_unavailable")
        if provider == "openai-responses":
            if "OPENAI_API_KEY" in message:
                return ("model_profile_unavailable", "openai_api_key_missing")
            return ("model_profile_unavailable", "openai_responses_unavailable")
        return ("model_profile_unavailable",)

    async def _available_ollama_model_ids(base_url: str) -> tuple[set[str] | None, tuple[str, ...]]:
        try:
            async with httpx.AsyncClient(timeout=0.8) as client:
                response = await client.get(f"{base_url.rstrip('/')}/models")
                response.raise_for_status()
                data = response.json().get("data", [])
        except Exception:
            return None, ("ollama_unreachable",)

        model_ids = {item.get("id") for item in data if isinstance(item, dict) and item.get("id")}
        if not model_ids:
            return set(), ("ollama_models_unknown",)
        return set(model_ids), ("ollama_models_available",)

    async def _model_profile_availability(
        profile: LocalLLMModelProfile,
        *,
        browser_config: LocalBrowserConfig,
        ollama_models_by_base_url: dict[str, tuple[set[str] | None, tuple[str, ...]]],
    ) -> tuple[bool, tuple[str, ...]]:
        profile_config = _browser_config_for_model_profile(browser_config, profile)
        if profile.provider == "ollama":
            base_url = profile_config.base_url or DEFAULT_OLLAMA_BASE_URL
            if base_url not in ollama_models_by_base_url:
                ollama_models_by_base_url[base_url] = await _available_ollama_model_ids(base_url)
            model_ids, reason_codes = ollama_models_by_base_url[base_url]
            if model_ids is None:
                return False, ("model_profile_unavailable", *reason_codes)
            if model_ids and profile.model not in model_ids:
                return False, (
                    "model_profile_unavailable",
                    "ollama_model_unavailable",
                    *reason_codes,
                )
            return True, ("model_profile_available", *reason_codes)

        if profile.provider == "openai-responses":
            if not remote_model_selection_enabled(current_provider=browser_config.llm_provider):
                return False, (
                    "model_profile_unavailable",
                    "remote_model_selection_disabled",
                )
            if os.getenv("OPENAI_API_KEY") or get_local_env("OPENAI_API_KEY"):
                return True, ("model_profile_available", "openai_api_key_present")
            return False, ("model_profile_unavailable", "openai_api_key_missing")

        return False, ("model_profile_unavailable", "unsupported_provider")

    def _public_model_profile_payload(
        profile: LocalLLMModelProfile,
        *,
        available: bool,
        selected: bool,
        reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "id": _safe_public_text(profile.profile_id, limit=96),
            "label": _safe_public_text(profile.label, limit=120),
            "provider": _safe_public_text(profile.provider, limit=80),
            "model": _safe_public_text(profile.model, limit=120),
            "runtime_tier": _safe_public_text(profile.runtime_tier, limit=80),
            "latency_tier": _safe_public_text(profile.latency_tier, limit=80),
            "capability_tier": _safe_public_text(profile.capability_tier, limit=80),
            "language_fit": _safe_public_text_list(profile.language_fit, limit=32),
            "recommended_for": _safe_public_text_list(
                profile.recommended_for,
                limit=120,
            ),
            "default": profile.default is True,
            "available": available is True,
            "selected": selected is True,
            "reason_codes": _safe_public_reason_codes(reason_codes),
        }

    async def _runtime_model_catalog_payload() -> dict[str, Any]:
        browser_config = _active_browser_config()
        profiles = load_local_llm_model_profiles()
        current_profile_id = local_llm_model_profile_id_for(
            provider=browser_config.llm_provider,
            model=browser_config.model,
            profiles=profiles,
        )
        default_profile = local_llm_model_profile_by_id(
            DEFAULT_LOCAL_MODEL_PROFILE_ID,
            profiles=profiles,
        )
        ollama_models_by_base_url: dict[str, tuple[set[str] | None, tuple[str, ...]]] = {}
        records: list[dict[str, Any]] = []
        for profile in profiles:
            available, reason_codes = await _model_profile_availability(
                profile,
                browser_config=browser_config,
                ollama_models_by_base_url=ollama_models_by_base_url,
            )
            record = _public_model_profile_payload(
                profile,
                available=available,
                selected=profile.profile_id == current_profile_id,
                reason_codes=tuple(
                    dict.fromkeys(
                        (
                            f"model_profile:{profile.profile_id}",
                            f"llm_provider:{profile.provider}",
                            *reason_codes,
                        )
                    )
                ),
            )
            records.append(record)

        return {
            "schema_version": 1,
            "available": True,
            "current_profile_id": (
                _safe_public_text(current_profile_id, limit=96)
                if current_profile_id
                else None
            ),
            "default_profile_id": (
                _safe_public_text(default_profile.profile_id, limit=96)
                if default_profile is not None
                else DEFAULT_LOCAL_MODEL_PROFILE_ID
            ),
            "current_provider": _safe_public_text(browser_config.llm_provider, limit=80),
            "current_model": _safe_public_text(browser_config.model, limit=120),
            "current_language": _safe_public_text(
                getattr(browser_config.language, "value", str(browser_config.language)),
                limit=32,
            ),
            "profiles": records,
            "reason_codes": _safe_public_reason_codes(
                tuple(
                    dict.fromkeys(
                        (
                            "runtime_models:v1",
                            "runtime_models:available",
                            f"llm_provider:{browser_config.llm_provider}",
                            (
                                "remote_model_selection:enabled"
                                if remote_model_selection_enabled(
                                    current_provider=browser_config.llm_provider
                                )
                                else "remote_model_selection:disabled"
                            ),
                        )
                    )
                )
            ),
        }

    def _safe_public_text_mapping(
        value: Any,
        *,
        key_limit: int = 80,
        value_limit: int = 80,
    ) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, item in value.items():
            public_key = _safe_public_text(key, limit=key_limit) or "unknown"
            public_value = _safe_public_text(item, limit=value_limit)
            if public_value:
                result[public_key] = public_value
        return dict(sorted(result.items()))

    def _public_expression_voice_policy(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "available": payload.get("available") is True,
            "modality": _safe_public_text(payload.get("modality"), limit=64)
            or "unavailable",
            "concise_chunking_active": payload.get("concise_chunking_active") is True,
            "chunking_mode": _safe_public_text(payload.get("chunking_mode"), limit=64)
            or "unavailable",
            "max_spoken_chunk_chars": _safe_public_int(payload.get("max_spoken_chunk_chars")),
            "interruption_strategy_label": _safe_public_text(
                payload.get("interruption_strategy_label"),
                limit=96,
            ),
            "pause_yield_hint": _safe_public_text(payload.get("pause_yield_hint"), limit=96),
            "active_hints": _safe_public_text_list(payload.get("active_hints")),
            "unsupported_hints": _safe_public_text_list(payload.get("unsupported_hints")),
            "noop_reason_codes": _safe_public_reason_codes(payload.get("noop_reason_codes")),
            "expression_controls_hardware": False,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _public_expression_voice_actuation_plan(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "available": payload.get("available") is True,
            "backend_label": _safe_public_text(payload.get("backend_label"), limit=96)
            or "provider-neutral",
            "modality": _safe_public_text(payload.get("modality"), limit=64)
            or "unavailable",
            "chunk_boundaries_enabled": payload.get("chunk_boundaries_enabled") is True,
            "interruption_flush_enabled": payload.get("interruption_flush_enabled") is True,
            "interruption_discard_enabled": (
                payload.get("interruption_discard_enabled") is True
            ),
            "pause_timing_enabled": payload.get("pause_timing_enabled") is True,
            "speech_rate_enabled": payload.get("speech_rate_enabled") is True,
            "prosody_emphasis_enabled": payload.get("prosody_emphasis_enabled") is True,
            "partial_stream_abort_enabled": (
                payload.get("partial_stream_abort_enabled") is True
            ),
            "expression_controls_hardware": False,
            "chunking_mode": _safe_public_text(payload.get("chunking_mode"), limit=64)
            or "unavailable",
            "max_spoken_chunk_chars": _safe_public_int(payload.get("max_spoken_chunk_chars")),
            "requested_hints": _safe_public_text_list(payload.get("requested_hints")),
            "applied_hints": _safe_public_text_list(payload.get("applied_hints")),
            "active_hints": _safe_public_text_list(payload.get("active_hints")),
            "unsupported_hints": _safe_public_text_list(payload.get("unsupported_hints")),
            "noop_reason_codes": _safe_public_reason_codes(payload.get("noop_reason_codes")),
            "capability_reason_codes": _safe_public_reason_codes(
                payload.get("capability_reason_codes")
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _public_expression_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "available": payload.get("available") is True,
            "persona_profile_id": _safe_public_text(payload.get("persona_profile_id"), limit=96),
            "identity_label": _safe_public_text(payload.get("identity_label"), limit=120)
            or "Blink",
            "modality": _safe_public_text(payload.get("modality"), limit=64) or "unavailable",
            "teaching_mode_label": _safe_public_text(
                payload.get("teaching_mode_label"),
                limit=96,
            ),
            "memory_persona_section_status": _safe_public_text_mapping(
                payload.get("memory_persona_section_status")
            ),
            "voice_style_summary": _safe_public_text(
                payload.get("voice_style_summary"),
                limit=160,
            ),
            "response_chunk_length": _safe_public_text(
                payload.get("response_chunk_length"),
                limit=96,
            ),
            "pause_yield_hint": _safe_public_text(payload.get("pause_yield_hint"), limit=96),
            "interruption_strategy_label": _safe_public_text(
                payload.get("interruption_strategy_label"),
                limit=96,
            ),
            "initiative_label": _safe_public_text(payload.get("initiative_label"), limit=64)
            or "unavailable",
            "evidence_visibility_label": _safe_public_text(
                payload.get("evidence_visibility_label"),
                limit=64,
            )
            or "unavailable",
            "correction_mode_label": _safe_public_text(
                payload.get("correction_mode_label"),
                limit=64,
            )
            or "unavailable",
            "explanation_structure_label": _safe_public_text(
                payload.get("explanation_structure_label"),
                limit=64,
            )
            or "unavailable",
            "humor_mode_label": _safe_public_text(payload.get("humor_mode_label"), limit=64)
            or "unavailable",
            "vividness_mode_label": _safe_public_text(
                payload.get("vividness_mode_label"),
                limit=64,
            )
            or "unavailable",
            "sophistication_mode_label": _safe_public_text(
                payload.get("sophistication_mode_label"),
                limit=64,
            )
            or "unavailable",
            "character_presence_label": _safe_public_text(
                payload.get("character_presence_label"),
                limit=64,
            )
            or "unavailable",
            "story_mode_label": _safe_public_text(payload.get("story_mode_label"), limit=64)
            or "unavailable",
            "style_summary": _safe_public_text(payload.get("style_summary"), limit=180)
            or "unavailable",
            "humor_budget": _safe_public_float(payload.get("humor_budget")),
            "playfulness": _safe_public_float(payload.get("playfulness")),
            "metaphor_density": _safe_public_float(payload.get("metaphor_density")),
            "safety_clamped": payload.get("safety_clamped") is True,
            "expression_controls_hardware": False,
            "voice_policy": _public_expression_voice_policy(payload.get("voice_policy")),
            "voice_actuation_plan": _public_expression_voice_actuation_plan(
                payload.get("voice_actuation_plan")
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _runtime_expression_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        current_expression_state = getattr(runtime, "current_expression_state", None)
        if callable(current_expression_state):
            try:
                state = current_expression_state()
            except Exception as exc:  # pragma: no cover - defensive status surface
                payload = unavailable_runtime_expression_state(
                    modality=BrainPersonaModality.BROWSER,
                    reason_codes=(
                        "runtime_expression_state:unavailable",
                        f"runtime_expression_error:{type(exc).__name__}",
                    ),
                    memory_persona_section_status={
                        "persona_expression": "error",
                        "persona_defaults": "unknown",
                    },
                    tts_backend=config.tts_backend,
                ).as_dict()
                return _public_expression_payload(payload)
            as_dict = getattr(state, "as_dict", None)
            if callable(as_dict):
                return _public_expression_payload(as_dict())
            if isinstance(state, dict):
                return _public_expression_payload(dict(state))
        payload = unavailable_runtime_expression_state(
            modality=BrainPersonaModality.BROWSER,
            reason_codes=("runtime_expression_state:unavailable", "runtime_not_active"),
            memory_persona_section_status={
                "persona_expression": "unavailable",
                "persona_defaults": "unknown",
            },
            tts_backend=config.tts_backend,
        ).as_dict()
        return _public_expression_payload(payload)

    def _safe_voice_health_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_voice_health_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    def _safe_voice_health_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_voice_health_bool(value: Any) -> bool:
        return value is True

    def _safe_voice_health_text(value: Any, *, max_chars: int = 120) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        if not text:
            return None
        return text[:max_chars]

    def _safe_voice_health_enum(value: Any, allowed: set[str], default: str) -> str:
        text = _safe_voice_health_text(value)
        return text if text in allowed else default

    def _safe_voice_health_timestamp(value: Any) -> str | None:
        text = _safe_voice_health_text(value, max_chars=80)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat()

    def _safe_voice_health_reason_fragment(value: Any) -> str:
        raw = str(value or "")
        lowered = raw.lower()
        if any(
            marker in lowered
            for marker in (
                "secret",
                "authorization",
                "api_key",
                "bearer",
                "system_prompt",
                "prompt_text",
                "traceback",
                "runtimeerror",
                "raw_json",
                "/tmp",
                ".db",
            )
        ):
            return "redacted"
        text = "".join(
            ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in raw
        )
        return "_".join(part for part in text.split("_") if part)[:80] or "unknown"

    def _safe_voice_health_reason_codes(value: Any, fallback: tuple[str, ...] = ()) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return list(fallback)
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            code = _safe_voice_health_reason_fragment(item)
            if code in seen:
                continue
            seen.add(code)
            result.append(code)
        return result or list(fallback)

    def _public_voice_input_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "schema_version": 1,
            "available": _safe_voice_health_bool(payload.get("available", False)),
            "microphone_state": _safe_voice_health_enum(
                payload.get("microphone_state"),
                {
                    "unavailable",
                    "disconnected",
                    "connected",
                    "waiting_for_audio",
                    "receiving",
                    "no_audio_frames",
                    "stalled",
                },
                "unavailable",
            ),
            "stt_state": _safe_voice_health_enum(
                payload.get("stt_state"),
                {
                    "unavailable",
                    "idle",
                    "speech_detected",
                    "transcribing",
                    "waiting",
                    "transcribed",
                    "error",
                },
                "unavailable",
            ),
            "audio_frame_count": _safe_voice_health_int(payload.get("audio_frame_count")),
            "speech_start_count": _safe_voice_health_int(payload.get("speech_start_count")),
            "speech_stop_count": _safe_voice_health_int(payload.get("speech_stop_count")),
            "interim_transcription_count": _safe_voice_health_int(
                payload.get("interim_transcription_count")
            ),
            "last_partial_transcription_chars": _safe_voice_health_int(
                payload.get("last_partial_transcription_chars")
            ),
            "last_partial_transcription_at": _safe_voice_health_timestamp(
                payload.get("last_partial_transcription_at")
            ),
            "partial_transcript_available": _safe_voice_health_bool(
                payload.get("partial_transcript_available", False)
            ),
            "transcription_count": _safe_voice_health_int(payload.get("transcription_count")),
            "stt_error_count": _safe_voice_health_int(payload.get("stt_error_count")),
            "last_audio_frame_at": _safe_voice_health_timestamp(payload.get("last_audio_frame_at")),
            "last_audio_frame_age_ms": _safe_voice_health_optional_int(
                payload.get("last_audio_frame_age_ms")
            ),
            "last_stt_event_at": _safe_voice_health_timestamp(payload.get("last_stt_event_at")),
            "stt_waiting_since_at": _safe_voice_health_timestamp(
                payload.get("stt_waiting_since_at")
            ),
            "stt_wait_age_ms": _safe_voice_health_optional_int(payload.get("stt_wait_age_ms")),
            "stt_waiting_too_long": _safe_voice_health_bool(
                payload.get("stt_waiting_too_long", False)
            ),
            "last_transcription_at": (
                _safe_voice_health_timestamp(payload.get("last_transcription_at"))
            ),
            "last_transcription_chars": _safe_voice_health_int(
                payload.get("last_transcription_chars")
            ),
            "track_enabled": (
                _safe_voice_health_bool(payload.get("track_enabled"))
                if payload.get("track_enabled") is not None
                else None
            ),
            "track_reason": (
                _safe_voice_health_reason_fragment(payload.get("track_reason"))
                if payload.get("track_reason")
                else None
            ),
            "reason_codes": _safe_voice_health_reason_codes(
                payload.get("reason_codes")
                or [
                    "voice_input_health:v1",
                    "voice_input:unavailable",
                ],
                ("voice_input_health:v1", "voice_input:unavailable"),
            ),
        }

    def _public_voice_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        public_payload = {
            "available": _safe_voice_health_bool(payload.get("available", False)),
            "response_count": _safe_voice_health_int(payload.get("response_count")),
            "concise_chunking_activation_count": _safe_voice_health_int(
                payload.get("concise_chunking_activation_count")
            ),
            "chunk_count": _safe_voice_health_int(payload.get("chunk_count")),
            "max_chunk_chars": _safe_voice_health_int(payload.get("max_chunk_chars")),
            "average_chunk_chars": _safe_voice_health_float(payload.get("average_chunk_chars")),
            "interruption_frame_count": _safe_voice_health_int(payload.get("interruption_frame_count")),
            "buffer_flush_count": _safe_voice_health_int(payload.get("buffer_flush_count")),
            "buffer_discard_count": _safe_voice_health_int(payload.get("buffer_discard_count")),
            "last_chunking_mode": _safe_voice_health_enum(
                payload.get("last_chunking_mode"),
                {"unavailable", "none", "off", "concise", "safety_concise"},
                "unavailable",
            ),
            "last_max_spoken_chunk_chars": _safe_voice_health_int(
                payload.get("last_max_spoken_chunk_chars")
            ),
            "first_audio_latency_ms": _safe_voice_health_float(payload.get("first_audio_latency_ms")),
            "first_audio_latency_sample_count": _safe_voice_health_int(
                payload.get("first_audio_latency_sample_count")
            ),
            "resumed_latency_after_interrupt_ms": _safe_voice_health_float(
                payload.get("resumed_latency_after_interrupt_ms")
            ),
            "resumed_latency_sample_count": _safe_voice_health_int(
                payload.get("resumed_latency_sample_count")
            ),
            "interruption_accept_count": _safe_voice_health_int(
                payload.get("interruption_accept_count")
            ),
            "partial_stream_abort_count": _safe_voice_health_int(
                payload.get("partial_stream_abort_count")
            ),
            "average_chunks_per_response": _safe_voice_health_float(
                payload.get("average_chunks_per_response")
            ),
            "p50_chunk_chars": _safe_voice_health_int(payload.get("p50_chunk_chars")),
            "p95_chunk_chars": _safe_voice_health_int(payload.get("p95_chunk_chars")),
            "first_subtitle_latency_ms": _safe_voice_health_float(
                payload.get("first_subtitle_latency_ms")
            ),
            "first_subtitle_latency_sample_count": _safe_voice_health_int(
                payload.get("first_subtitle_latency_sample_count")
            ),
            "speech_chunk_latency_ms": _safe_voice_health_float(
                payload.get("speech_chunk_latency_ms")
            ),
            "speech_chunk_latency_sample_count": _safe_voice_health_int(
                payload.get("speech_chunk_latency_sample_count")
            ),
            "speech_queue_depth_current": _safe_voice_health_int(
                payload.get("speech_queue_depth_current")
            ),
            "speech_queue_depth_max": _safe_voice_health_int(
                payload.get("speech_queue_depth_max")
            ),
            "stale_chunk_drop_count": _safe_voice_health_int(
                payload.get("stale_chunk_drop_count")
            ),
            "speech_director_mode": _safe_voice_health_enum(
                payload.get("speech_director_mode"),
                {"unavailable", "melo_chunked", "kokoro_chunked", "kokoro_passthrough"},
                "unavailable",
            ),
            "expression_controls_hardware": False,
            "reason_codes": _safe_voice_health_reason_codes(payload.get("reason_codes")),
            "input_health": _public_voice_input_health_payload(payload.get("input_health") or {}),
        }
        return public_payload

    def _public_speech_state_payload(active_config: LocalBrowserConfig) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        mode = _speech_director_mode_for_config(active_config)
        payload: dict[str, Any] = {}
        if runtime is not None:
            current_voice_metrics = getattr(runtime, "current_voice_metrics", None)
            if callable(current_voice_metrics):
                try:
                    metrics = current_voice_metrics()
                    as_dict = getattr(metrics, "as_dict", None)
                    if callable(as_dict):
                        payload = as_dict()
                    elif isinstance(metrics, dict):
                        payload = dict(metrics)
                except Exception:
                    payload = {}
        director_mode = _safe_voice_health_enum(
            payload.get("speech_director_mode") or mode,
            {"unavailable", "melo_chunked", "kokoro_chunked", "kokoro_passthrough"},
            mode if mode in {"melo_chunked", "kokoro_chunked"} else "unavailable",
        )
        reason_codes = [
            "speech_state:v1",
            f"speech_director:{director_mode}",
        ]
        if payload.get("stale_chunk_drop_count"):
            reason_codes.append("speech_stale_drops_observed")
        if payload.get("first_subtitle_latency_sample_count"):
            reason_codes.append("speech_first_subtitle_latency_observed")
        if payload.get("first_audio_latency_sample_count"):
            reason_codes.append("speech_first_audio_latency_observed")
        session_speech = _runtime_session_v3().speech.as_dict()
        return {
            "available": runtime is not None,
            "director_mode": director_mode,
            "first_subtitle_latency_ms": _safe_voice_health_float(
                payload.get("first_subtitle_latency_ms")
            ),
            "first_subtitle_latency_sample_count": _safe_voice_health_int(
                payload.get("first_subtitle_latency_sample_count")
            ),
            "first_audio_latency_ms": _safe_voice_health_float(
                payload.get("first_audio_latency_ms")
            ),
            "first_audio_latency_sample_count": _safe_voice_health_int(
                payload.get("first_audio_latency_sample_count")
            ),
            "speech_chunk_latency_ms": _safe_voice_health_float(
                payload.get("speech_chunk_latency_ms")
            ),
            "speech_chunk_latency_sample_count": _safe_voice_health_int(
                payload.get("speech_chunk_latency_sample_count")
            ),
            "speech_queue_depth_current": _safe_voice_health_int(
                payload.get("speech_queue_depth_current")
            ),
            "speech_queue_depth_max": _safe_voice_health_int(
                payload.get("speech_queue_depth_max")
            ),
            "stale_chunk_drop_count": _safe_voice_health_int(
                max(
                    _safe_voice_health_int(payload.get("stale_chunk_drop_count")),
                    _safe_voice_health_int(session_speech.get("stale_chunk_drops")),
                )
            ),
            "speech_queue_v3": session_speech,
            "reason_codes": reason_codes,
        }

    def _safe_active_listening_text(value: object, *, limit: int = 48) -> str:
        text = " ".join(str(value or "").split())
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "authorization",
                "bearer ",
                "credential",
                "hidden prompt",
                "http://",
                "https://",
                "memory_id",
                "password",
                "secret",
                "sk-",
                "system prompt",
                "token",
                "www.",
            )
        ):
            return ""
        return text[:limit]

    def _safe_active_listening_hint(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        kind = _safe_voice_health_enum(
            item.get("kind"),
            {"topic", "constraint", "correction", "project_reference"},
            "topic",
        )
        value = _safe_active_listening_text(item.get("value"))
        if not value:
            return None
        return {
            "kind": kind,
            "value": value,
            "confidence": _safe_voice_health_enum(
                item.get("confidence"),
                {"heuristic", "observed", "unknown"},
                "heuristic",
            ),
            "source": _safe_voice_health_enum(
                item.get("source"),
                {"final_transcript", "partial_transcript", "unknown"},
                "final_transcript",
            ),
            "editable": _safe_voice_health_bool(item.get("editable", True)),
        }

    def _safe_active_listener_flags(value: object) -> list[str]:
        raw_values = value if isinstance(value, list) else []
        result: list[str] = []
        seen: set[str] = set()
        for raw in raw_values[:8]:
            flag = _safe_active_listening_text(raw, limit=48)
            if not flag or flag in seen:
                continue
            seen.add(flag)
            result.append(flag)
            if len(result) >= 5:
                break
        return result

    def _public_semantic_listener_state_v3_payload(
        payload: object,
        *,
        active_config: LocalBrowserConfig,
        topics: list[dict[str, object]],
        constraints: list[dict[str, object]],
        corrections: list[dict[str, object]],
        project_references: list[dict[str, object]],
        uncertainty_flags: list[str],
        partial_available: bool,
        final_available: bool,
        partial_transcript_chars: int,
        final_transcript_chars: int,
        turn_duration_ms: int | None,
        ready_to_answer: bool,
        readiness_state: str,
        camera_scene: object = None,
        memory_context: object = None,
        floor_state: object = None,
        reason_codes: list[str] | None = None,
    ) -> dict[str, object]:
        raw = payload if isinstance(payload, dict) else {}
        semantic = build_semantic_listener_state_v3(
            language=active_config.language.value,
            topics=topics,
            constraints=constraints,
            corrections=corrections,
            project_references=project_references,
            uncertainty_flags=uncertainty_flags,
            safe_live_summary=raw.get("safe_live_summary"),
            partial_available=partial_available,
            final_available=final_available,
            partial_transcript_chars=partial_transcript_chars,
            final_transcript_chars=final_transcript_chars,
            turn_duration_ms=turn_duration_ms,
            ready_to_answer=ready_to_answer,
            readiness_state=readiness_state,
            camera_scene=camera_scene,
            memory_context=memory_context,
            floor_state=floor_state,
            reason_codes=tuple(reason_codes or []),
        ).as_dict()
        raw_intent = _safe_voice_health_enum(
            raw.get("detected_intent"),
            {
                "question",
                "instruction",
                "correction",
                "object_showing",
                "project_planning",
                "small_talk",
                "unknown",
            },
            str(semantic.get("detected_intent") or "unknown"),
        )
        raw_camera = _safe_voice_health_enum(
            raw.get("camera_reference_state"),
            {
                "not_used",
                "fresh_supported",
                "fresh_available",
                "stale_or_limited",
                "unsupported",
                "error",
            },
            str(semantic.get("camera_reference_state") or "not_used"),
        )
        if raw_intent:
            semantic["detected_intent"] = raw_intent
        if raw_camera:
            semantic["camera_reference_state"] = raw_camera
        return semantic

    def _public_active_listener_v2_payload(
        payload: dict[str, Any],
        *,
        active_config: LocalBrowserConfig,
        camera_scene: object = None,
        memory_context: object = None,
        floor_state: object = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        phase = _safe_voice_health_enum(
            payload.get("phase"),
            {
                "idle",
                "listening_started",
                "speech_continuing",
                "partial_understanding",
                "transcribing",
                "final_understanding",
                "ready_to_answer",
                "degraded",
                "error",
            },
            "idle",
        )
        topics = [
            hint
            for item in (payload.get("topics") if isinstance(payload.get("topics"), list) else [])
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        constraints = [
            hint
            for item in (
                payload.get("constraints") if isinstance(payload.get("constraints"), list) else []
            )
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        corrections = [
            hint
            for item in (
                payload.get("corrections") if isinstance(payload.get("corrections"), list) else []
            )
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        project_references = [
            hint
            for item in (
                payload.get("project_references")
                if isinstance(payload.get("project_references"), list)
                else []
            )
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        degradation = payload.get("degradation") if isinstance(payload.get("degradation"), dict) else {}
        degradation_state = _safe_voice_health_enum(
            degradation.get("state"),
            {"ok", "degraded", "error"},
            "ok",
        )
        degradation_components = _safe_active_listener_flags(degradation.get("components"))
        uncertainty_flags = _safe_active_listener_flags(payload.get("uncertainty_flags"))
        reason_codes = _safe_voice_health_reason_codes(
            payload.get("reason_codes")
            or [
                "active_listener:v2",
                f"active_listener:{phase}",
            ],
            ("active_listener:v2", f"active_listener:{phase}"),
        )
        partial_available = _safe_voice_health_bool(payload.get("partial_available", False))
        final_available = _safe_voice_health_bool(payload.get("final_available", False))
        partial_transcript_chars = _safe_voice_health_int(payload.get("partial_transcript_chars"))
        final_transcript_chars = _safe_voice_health_int(payload.get("final_transcript_chars"))
        turn_duration_ms = _safe_voice_health_optional_int(payload.get("turn_duration_ms"))
        ready_to_answer = _safe_voice_health_bool(payload.get("ready_to_answer", False))
        readiness_state = _safe_voice_health_enum(
            payload.get("readiness_state"),
            {"not_ready", "listening", "partial", "transcribing", "ready", "degraded"},
            "not_ready",
        )
        semantic_state_v3 = _public_semantic_listener_state_v3_payload(
            payload.get("semantic_state_v3"),
            active_config=active_config,
            topics=topics,
            constraints=constraints,
            corrections=corrections,
            project_references=project_references,
            uncertainty_flags=uncertainty_flags,
            partial_available=partial_available,
            final_available=final_available,
            partial_transcript_chars=partial_transcript_chars,
            final_transcript_chars=final_transcript_chars,
            turn_duration_ms=turn_duration_ms,
            ready_to_answer=ready_to_answer,
            readiness_state=readiness_state,
            camera_scene=camera_scene,
            memory_context=memory_context,
            floor_state=floor_state,
            reason_codes=reason_codes,
        )
        return {
            "schema_version": 2,
            "available": _safe_voice_health_bool(payload.get("available", False)),
            "profile": active_config.config_profile or "manual",
            "language": active_config.language.value,
            "phase": phase,
            "partial_available": partial_available,
            "final_available": final_available,
            "partial_transcript_chars": partial_transcript_chars,
            "final_transcript_chars": final_transcript_chars,
            "interim_transcript_count": _safe_voice_health_int(
                payload.get("interim_transcript_count")
            ),
            "final_transcript_count": _safe_voice_health_int(
                payload.get("final_transcript_count")
            ),
            "speech_start_count": _safe_voice_health_int(payload.get("speech_start_count")),
            "speech_stop_count": _safe_voice_health_int(payload.get("speech_stop_count")),
            "turn_started_at": _safe_voice_health_timestamp(payload.get("turn_started_at")),
            "turn_stopped_at": _safe_voice_health_timestamp(payload.get("turn_stopped_at")),
            "last_update_at": _safe_voice_health_timestamp(payload.get("last_update_at")),
            "turn_duration_ms": turn_duration_ms,
            "topics": topics,
            "constraints": constraints,
            "corrections": corrections,
            "project_references": project_references,
            "topic_count": len(topics),
            "constraint_count": len(constraints),
            "correction_count": len(corrections),
            "project_reference_count": len(project_references),
            "uncertainty_flags": uncertainty_flags,
            "uncertainty_flag_count": len(uncertainty_flags),
            "ready_to_answer": ready_to_answer,
            "readiness_state": readiness_state,
            "degradation": {
                "state": degradation_state,
                "components": degradation_components,
                "reason_codes": _safe_voice_health_reason_codes(
                    degradation.get("reason_codes"),
                    (f"active_listener_degradation:{degradation_state}",),
                ),
            },
            "semantic_state_v3": semantic_state_v3,
            "reason_codes": reason_codes,
        }

    def _public_active_listening_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        phase = _safe_voice_health_enum(
            payload.get("phase"),
            {
                "idle",
                "speech_started",
                "speech_continuing",
                "speech_stopped",
                "transcribing",
                "partial_transcript",
                "final_transcript",
                "error",
            },
            "idle",
        )
        topics = [
            hint
            for item in (payload.get("topics") if isinstance(payload.get("topics"), list) else [])
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        constraints = [
            hint
            for item in (
                payload.get("constraints") if isinstance(payload.get("constraints"), list) else []
            )
            if (hint := _safe_active_listening_hint(item)) is not None
        ][:5]
        reason_codes = _safe_voice_health_reason_codes(
            payload.get("reason_codes")
            or [
                "active_listening:v1",
                f"active_listening:{phase}",
            ],
            ("active_listening:v1", f"active_listening:{phase}"),
        )
        return {
            "schema_version": 1,
            "available": _safe_voice_health_bool(payload.get("available", False)),
            "phase": phase,
            "partial_available": _safe_voice_health_bool(payload.get("partial_available", False)),
            "partial_transcript_chars": _safe_voice_health_int(
                payload.get("partial_transcript_chars")
            ),
            "final_transcript_chars": _safe_voice_health_int(
                payload.get("final_transcript_chars")
            ),
            "interim_transcript_count": _safe_voice_health_int(
                payload.get("interim_transcript_count")
            ),
            "final_transcript_count": _safe_voice_health_int(
                payload.get("final_transcript_count")
            ),
            "speech_start_count": _safe_voice_health_int(payload.get("speech_start_count")),
            "speech_stop_count": _safe_voice_health_int(payload.get("speech_stop_count")),
            "turn_started_at": _safe_voice_health_timestamp(payload.get("turn_started_at")),
            "turn_stopped_at": _safe_voice_health_timestamp(payload.get("turn_stopped_at")),
            "last_update_at": _safe_voice_health_timestamp(payload.get("last_update_at")),
            "turn_duration_ms": _safe_voice_health_optional_int(payload.get("turn_duration_ms")),
            "topics": topics,
            "constraints": constraints,
            "topic_count": len(topics),
            "constraint_count": len(constraints),
            "reason_codes": reason_codes,
        }

    def _public_active_listening_state_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _public_active_listening_payload(
                unavailable_active_listening_snapshot("active_listening:runtime_unavailable")
            )
        current_active_listening_state = getattr(runtime, "current_active_listening_state", None)
        if not callable(current_active_listening_state):
            return _public_active_listening_payload(
                unavailable_active_listening_snapshot("active_listening:runtime_surface_missing")
            )
        try:
            payload = current_active_listening_state()
        except Exception:
            payload = unavailable_active_listening_snapshot("active_listening:runtime_error")
        return _public_active_listening_payload(payload)

    def _public_active_listener_v2_state_payload(
        active_config: LocalBrowserConfig,
        *,
        camera_scene: object = None,
        memory_context: object = None,
        floor_state: object = None,
    ) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _public_active_listener_v2_payload(
                unavailable_active_listener_state_v2(
                    profile=active_config.config_profile or "manual",
                    language=active_config.language.value,
                    reason_codes=("active_listener:runtime_unavailable",),
                ),
                active_config=active_config,
                camera_scene=camera_scene,
                memory_context=memory_context,
                floor_state=floor_state,
            )
        current_active_listener_state_v2 = getattr(
            runtime,
            "current_active_listener_state_v2",
            None,
        )
        if not callable(current_active_listener_state_v2):
            return _public_active_listener_v2_payload(
                unavailable_active_listener_state_v2(
                    profile=active_config.config_profile or "manual",
                    language=active_config.language.value,
                    reason_codes=("active_listener:runtime_surface_missing",),
                ),
                active_config=active_config,
                camera_scene=camera_scene,
                memory_context=memory_context,
                floor_state=floor_state,
            )
        try:
            payload = current_active_listener_state_v2(
                profile=active_config.config_profile or "manual",
            )
        except Exception:
            payload = unavailable_active_listener_state_v2(
                profile=active_config.config_profile or "manual",
                language=active_config.language.value,
                reason_codes=("active_listener:runtime_error",),
            )
        return _public_active_listener_v2_payload(
            payload,
            active_config=active_config,
            camera_scene=camera_scene,
            memory_context=memory_context,
            floor_state=floor_state,
        )

    def _runtime_voice_metrics_unavailable_payload(*reason_codes: str) -> dict[str, Any]:
        payload = unavailable_expression_voice_metrics_snapshot(*reason_codes).as_dict()
        payload["input_health"] = _public_voice_input_health_payload({})
        return payload

    def _runtime_voice_metrics_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_voice_metrics_unavailable_payload("runtime_not_active")
        current_voice_metrics = getattr(runtime, "current_voice_metrics", None)
        if not callable(current_voice_metrics):
            return _runtime_voice_metrics_unavailable_payload(
                "runtime_voice_metrics_surface_missing"
            )
        try:
            input_health_reader = getattr(runtime, "current_voice_input_health", None)
            input_health = (
                _public_voice_input_health_payload(input_health_reader())
                if callable(input_health_reader)
                else _public_voice_input_health_payload({})
            )
            metrics = current_voice_metrics()
            as_dict = getattr(metrics, "as_dict", None)
            if callable(as_dict):
                payload = _public_voice_metrics_payload(as_dict())
                payload["input_health"] = input_health
                return payload
            if isinstance(metrics, dict):
                payload = _public_voice_metrics_payload(dict(metrics))
                payload["input_health"] = input_health
                return payload
        except Exception as exc:  # pragma: no cover - defensive status surface
            return _runtime_voice_metrics_unavailable_payload(
                f"runtime_voice_metrics_error:{type(exc).__name__}"
            )
        return _runtime_voice_metrics_unavailable_payload("runtime_voice_metrics_invalid_result")

    def _runtime_behavior_controls_unavailable_payload(*reason_codes: str) -> dict[str, Any]:
        return {
            "available": False,
            "schema_version": 1,
            "profile": None,
            "compiled_effect_summary": "",
            "reason_codes": list(
                dict.fromkeys(
                    (
                        "behavior_controls_state:unavailable",
                        *reason_codes,
                    )
                )
            ),
        }

    def _runtime_behavior_controls_rejected_payload(
        *,
        reason_codes: tuple[str, ...],
        rejected_fields: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "accepted": False,
            "applied": False,
            "profile": None,
            "compiled_effect_summary": "",
            "rejected_fields": _safe_public_text_list(
                rejected_fields,
                limit=64,
                max_items=24,
            ),
            "reason_codes": _safe_public_reason_codes(
                (
                    "behavior_controls_update_rejected",
                    *reason_codes,
                )
            ),
        }

    def _public_behavior_controls_payload(payload: dict[str, Any]) -> dict[str, Any]:
        profile = payload.get("profile")
        if isinstance(profile, dict):
            profile = {
                key: profile.get(key)
                for key in (
                    "schema_version",
                    "user_id",
                    "agent_id",
                    "response_depth",
                    "directness",
                    "warmth",
                    "teaching_mode",
                    "memory_use",
                    "initiative_mode",
                    "evidence_visibility",
                    "correction_mode",
                    "explanation_structure",
                    "challenge_style",
                    "voice_mode",
                    "question_budget",
                    "humor_mode",
                    "vividness_mode",
                    "sophistication_mode",
                    "character_presence",
                    "story_mode",
                    "updated_at",
                    "source",
                    "reason_codes",
                )
                if key in profile
            }
        else:
            profile = None
        if isinstance(profile, dict):
            profile = {
                key: (
                    _safe_public_int(value, 1)
                    if key == "schema_version"
                    else _safe_public_reason_codes(value)
                    if key == "reason_codes"
                    else _safe_public_text(value, limit=120)
                )
                for key, value in profile.items()
            }
        public_payload = {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "accepted": payload.get("accepted") is True if "accepted" in payload else None,
            "applied": payload.get("applied") is True if "applied" in payload else None,
            "available": payload.get("available") is True if "available" in payload else None,
            "profile": profile,
            "compiled_effect_summary": _safe_public_text(
                payload.get("compiled_effect_summary"), limit=240
            ),
            "rejected_fields": _safe_public_text_list(
                payload.get("rejected_fields"),
                limit=64,
                max_items=24,
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }
        return {key: value for key, value in public_payload.items() if value is not None}

    def _runtime_behavior_controls_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_behavior_controls_unavailable_payload("runtime_not_active")
        reader = getattr(runtime, "current_behavior_control_profile", None)
        if not callable(reader):
            return _runtime_behavior_controls_unavailable_payload(
                "runtime_behavior_controls_surface_missing"
            )
        try:
            profile = reader()
            payload = {
                "schema_version": 1,
                "available": True,
                "profile": profile.as_dict(),
                "compiled_effect_summary": render_behavior_control_effect_summary(profile),
                "reason_codes": list(
                    dict.fromkeys(
                        (
                            "behavior_controls_state:available",
                            *profile.reason_codes,
                        )
                    )
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive browser status surface
            return _runtime_behavior_controls_unavailable_payload(
                f"runtime_behavior_controls_error:{type(exc).__name__}"
            )
        return _public_behavior_controls_payload(payload)

    def _runtime_behavior_controls_update_payload(request_data: dict[str, Any]) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        normalized_updates, rejected_fields = validate_behavior_control_update_payload(request_data)
        if rejected_fields:
            return _runtime_behavior_controls_rejected_payload(
                reason_codes=("behavior_controls_fields_invalid",),
                rejected_fields=rejected_fields,
            )
        if runtime is None:
            return _runtime_behavior_controls_rejected_payload(
                reason_codes=("runtime_not_active",),
            )
        updater = getattr(runtime, "update_behavior_control_profile", None)
        if not callable(updater):
            return _runtime_behavior_controls_rejected_payload(
                reason_codes=("runtime_behavior_controls_surface_missing",),
            )
        try:
            result = updater(
                normalized_updates,
                source="browser_behavior_controls_endpoint",
            )
            as_dict = getattr(result, "as_dict", None)
            if callable(as_dict):
                return _public_behavior_controls_payload(as_dict())
            if isinstance(result, dict):
                return _public_behavior_controls_payload(dict(result))
        except Exception:  # pragma: no cover - defensive browser action boundary
            return _runtime_behavior_controls_rejected_payload(
                reason_codes=("runtime_behavior_controls_error",),
            )
        return _runtime_behavior_controls_rejected_payload(
            reason_codes=("runtime_behavior_controls_invalid_result",),
        )

    def _performance_preference_store() -> PerformancePreferenceStore:
        return PerformancePreferenceStore(
            getattr(
                app.state,
                "blink_performance_preferences_v3_dir",
                PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
            )
        )

    def _append_performance_learning_brain_event(
        *,
        event_type: str,
        payload: dict[str, Any],
        source: str,
    ) -> bool:
        runtime = _active_brain_runtime()
        store = getattr(runtime, "store", None)
        resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(resolver):
            return False
        try:
            session_ids = resolver()
            store.append_brain_event(
                event_type=event_type,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=source,
                payload=payload,
                tags=["performance_learning_v3"],
            )
            return True
        except Exception:
            return False

    def _runtime_performance_preference_rejected_payload(
        *reason_codes: str,
        sanitizer: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "accepted": False,
            "applied": False,
            "pair": None,
            "policy_proposals": [],
            "sanitizer": sanitizer or {
                "accepted": False,
                "blocked_keys": [],
                "blocked_values": [],
                "omitted_keys": [],
                "reason_codes": [],
            },
            "reason_codes": _safe_public_reason_codes(
                (
                    "performance_preference_record:rejected",
                    *reason_codes,
                )
            ),
        }

    def _runtime_performance_preference_record_payload(
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        store = _performance_preference_store()
        try:
            pair, proposals, sanitizer = store.record_pair(request_data)
        except Exception:
            return _runtime_performance_preference_rejected_payload(
                "performance_preference_record:error"
            )
        if pair is None:
            return _runtime_performance_preference_rejected_payload(
                "performance_preference_record:unsafe_payload",
                sanitizer=sanitizer.as_dict(),
            )
        pair_payload = pair.as_dict()
        proposal_payloads = [proposal.as_dict() for proposal in proposals]
        brain_event_written = _append_performance_learning_brain_event(
            event_type=BrainEventType.PERFORMANCE_PREFERENCE_RECORDED,
            source="browser_performance_preferences_endpoint",
            payload={"performance_preference_pair": pair_payload},
        )
        for proposal in proposals:
            _append_performance_learning_brain_event(
                event_type=BrainEventType.PERFORMANCE_LEARNING_POLICY_PROPOSED,
                source="browser_performance_preferences_endpoint",
                payload={"performance_learning_policy_proposal": proposal.as_dict()},
            )
        return {
            "schema_version": 3,
            "accepted": True,
            "applied": True,
            "pair": pair_payload,
            "policy_proposals": proposal_payloads,
            "brain_event_written": brain_event_written,
            "preferences_path_kind": "jsonl_file",
            "sanitizer": sanitizer.as_dict(),
            "reason_codes": _safe_public_reason_codes(
                (
                    "performance_preference_record:accepted",
                    "performance_preference_pair:v3",
                    "performance_preference_policy_proposals:available"
                    if proposal_payloads
                    else "performance_preference_policy_proposals:none",
                )
            ),
        }

    def _runtime_performance_learning_apply_payload(
        proposal_id: str,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        safe_proposal_id = _safe_public_text(proposal_id, limit=120)
        store = _performance_preference_store()
        proposal = store.find_proposal(safe_proposal_id)
        if proposal is None:
            return {
                "schema_version": 3,
                "accepted": False,
                "applied": False,
                "proposal_id": safe_proposal_id,
                "proposal": None,
                "behavior_control_result": None,
                "reason_codes": _safe_public_reason_codes(
                    (
                        "performance_learning_policy_apply:rejected",
                        "performance_learning_policy_proposal_missing",
                    )
                ),
            }
        runtime = _active_brain_runtime()
        if runtime is None:
            return {
                "schema_version": 3,
                "accepted": False,
                "applied": False,
                "proposal_id": proposal.proposal_id,
                "proposal": proposal.as_dict(),
                "behavior_control_result": None,
                "reason_codes": _safe_public_reason_codes(
                    (
                        "performance_learning_policy_apply:rejected",
                        "runtime_not_active",
                    )
                ),
            }
        normalized_updates, rejected_fields = validate_behavior_control_update_payload(
            proposal.behavior_control_updates
        )
        if rejected_fields:
            return {
                "schema_version": 3,
                "accepted": False,
                "applied": False,
                "proposal_id": proposal.proposal_id,
                "proposal": proposal.as_dict(),
                "behavior_control_result": None,
                "rejected_fields": _safe_public_text_list(rejected_fields),
                "reason_codes": _safe_public_reason_codes(
                    (
                        "performance_learning_policy_apply:rejected",
                        "performance_learning_policy_updates_invalid",
                    )
                ),
            }
        updater = getattr(runtime, "update_behavior_control_profile", None)
        if not callable(updater):
            return {
                "schema_version": 3,
                "accepted": False,
                "applied": False,
                "proposal_id": proposal.proposal_id,
                "proposal": proposal.as_dict(),
                "behavior_control_result": None,
                "reason_codes": _safe_public_reason_codes(
                    (
                        "performance_learning_policy_apply:rejected",
                        "runtime_behavior_controls_surface_missing",
                    )
                ),
            }
        try:
            result = updater(
                normalized_updates,
                source="performance_learning_policy_proposal",
            )
            as_dict = getattr(result, "as_dict", None)
            result_payload = as_dict() if callable(as_dict) else result if isinstance(result, dict) else {}
            public_result = _public_behavior_controls_payload(dict(result_payload))
        except Exception:
            return {
                "schema_version": 3,
                "accepted": False,
                "applied": False,
                "proposal_id": proposal.proposal_id,
                "proposal": proposal.as_dict(),
                "behavior_control_result": None,
                "reason_codes": _safe_public_reason_codes(
                    (
                        "performance_learning_policy_apply:rejected",
                        "runtime_behavior_controls_error",
                    )
                ),
            }
        accepted = public_result.get("accepted") is True
        applied = public_result.get("applied") is True
        public_proposal = proposal.as_dict()
        if accepted and applied:
            applied_proposal = store.mark_proposal_applied(proposal)
            public_proposal = applied_proposal.as_dict()
            _append_performance_learning_brain_event(
                event_type=BrainEventType.PERFORMANCE_LEARNING_POLICY_APPLIED,
                source="browser_performance_preferences_endpoint",
                payload={
                    "performance_learning_policy_proposal": public_proposal,
                    "behavior_control_result": public_result,
                },
            )
        return {
            "schema_version": 3,
            "accepted": accepted,
            "applied": applied,
            "proposal_id": proposal.proposal_id,
            "proposal": public_proposal,
            "behavior_control_result": public_result,
            "operator_acknowledged": request_data.get("operator_acknowledged") is True,
            "reason_codes": _safe_public_reason_codes(
                (
                    "performance_learning_policy_apply:accepted"
                    if accepted
                    else "performance_learning_policy_apply:rejected",
                    "performance_learning_policy_apply:applied"
                    if applied
                    else "performance_learning_policy_apply:not_applied",
                    *proposal.reason_codes,
                )
            ),
        }

    def _runtime_style_presets_payload() -> dict[str, Any]:
        try:
            return behavior_style_preset_catalog()
        except Exception as exc:  # pragma: no cover - defensive catalog boundary
            return {
                "schema_version": 1,
                "available": False,
                "presets": [],
                "default_preset_id": None,
                "reason_codes": [f"style_presets_error:{type(exc).__name__}"],
            }

    def _memory_persona_preset_session_ids():
        runtime = _active_brain_runtime()
        session_resolver = getattr(runtime, "session_resolver", None)
        if callable(session_resolver):
            return session_resolver()
        return build_session_resolver(runtime_kind="browser")()

    def _memory_persona_preset_seed(session_ids) -> dict[str, Any]:
        return build_witty_sophisticated_memory_story_seed(
            user_name=str(session_ids.user_id),
            agent_id=str(session_ids.agent_id),
        )

    def _runtime_memory_persona_preset_preview_payload() -> dict[str, Any]:
        try:
            session_ids = _memory_persona_preset_session_ids()
            report = build_memory_persona_ingestion_preview(
                _memory_persona_preset_seed(session_ids),
                session_ids=session_ids,
            )
            payload = report.as_dict()
            payload.update(
                {
                    "available": True,
                    "preset_id": "witty_sophisticated",
                    "preset_label": "Witty Sophisticated",
                    "reason_codes": list(
                        dict.fromkeys(
                            (
                                "memory_persona_preset_preview:available",
                                *payload.get("reason_codes", ()),
                            )
                        )
                    ),
                }
            )
            return _public_memory_persona_report_payload(payload)
        except Exception as exc:  # pragma: no cover - defensive preview boundary
            return {
                "schema_version": 1,
                "available": False,
                "accepted": False,
                "applied": False,
                "preset_id": "witty_sophisticated",
                "preset_label": "Witty Sophisticated",
                "import_id": "",
                "seed_sha256": "",
                "counts": {
                    "accepted_candidates": 0,
                    "rejected_entries": 0,
                    "applied_entries": 0,
                    "memory_written": 0,
                    "memory_noop": 0,
                    "behavior_controls_applied": 0,
                    "behavior_controls_noop": 0,
                },
                "candidates": [],
                "rejected_entries": [],
                "applied_entries": [],
                "behavior_control_result": None,
                "reason_codes": [f"memory_persona_preset_preview_error:{type(exc).__name__}"],
            }

    def _runtime_memory_persona_preset_apply_payload(
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return {
                **_runtime_memory_persona_preset_preview_payload(),
                "accepted": False,
                "applied": False,
                "reason_codes": [
                    "memory_persona_preset_apply:rejected",
                    "runtime_memory_surface_missing",
                ],
            }
        approved_report = request_data.get("approved_report") or request_data.get(
            "approvedReport"
        )
        try:
            session_ids = session_resolver()
            report = apply_memory_persona_ingestion(
                store=store,
                seed=_memory_persona_preset_seed(session_ids),
                session_ids=session_ids,
                approved_report=approved_report if isinstance(approved_report, dict) else None,
            )
            payload = report.as_dict()
            payload.update(
                {
                    "available": True,
                    "preset_id": "witty_sophisticated",
                    "preset_label": "Witty Sophisticated",
                    "reason_codes": list(
                        dict.fromkeys(
                            (
                                "memory_persona_preset_apply:accepted"
                                if report.accepted
                                else "memory_persona_preset_apply:rejected",
                                *payload.get("reason_codes", ()),
                            )
                        )
                    ),
                }
            )
            return _public_memory_persona_report_payload(payload)
        except Exception as exc:  # pragma: no cover - defensive apply boundary
            return {
                "schema_version": 1,
                "available": False,
                "accepted": False,
                "applied": False,
                "preset_id": "witty_sophisticated",
                "preset_label": "Witty Sophisticated",
                "import_id": "",
                "seed_sha256": "",
                "counts": {
                    "accepted_candidates": 0,
                    "rejected_entries": 0,
                    "applied_entries": 0,
                    "memory_written": 0,
                    "memory_noop": 0,
                    "behavior_controls_applied": 0,
                    "behavior_controls_noop": 0,
                },
                "candidates": [],
                "rejected_entries": [],
                "applied_entries": [],
                "behavior_control_result": None,
                "reason_codes": [f"memory_persona_preset_apply_error:{type(exc).__name__}"],
            }

    def _runtime_memory_unavailable_payload(*reason_codes: str) -> dict[str, Any]:
        return {
            "available": False,
            "schema_version": 1,
            "user_id": None,
            "agent_id": None,
            "generated_at": "",
            "summary": "Memory unavailable.",
            "records": [],
            "hidden_counts": {"suppressed": 0, "historical": 0, "limit": 0},
            "health_summary": "Memory health unavailable.",
            "used_in_current_reply": [],
            "behavior_effects": [],
            "persona_references": [],
            "memory_continuity_trace": None,
            "memory_persona_performance": {
                "schema_version": 1,
                "available": False,
                "profile": "manual",
                "modality": "browser",
                "language": "unknown",
                "tts_label": "unknown",
                "protected_playback": True,
                "camera_state": "unknown",
                "continuous_perception_enabled": False,
                "current_turn_state": "unknown",
                "memory_policy": "unavailable",
                "selected_memory_count": 0,
                "suppressed_memory_count": 0,
                "used_in_current_reply": [],
                "behavior_effects": [],
                "persona_references": [],
                "summary": "Memory/persona performance unavailable.",
                "reason_codes": [
                    "memory_persona_performance:v1",
                    "memory_persona_performance:unavailable",
                ],
            },
            "reason_codes": list(
                dict.fromkeys(
                    (
                        "runtime_memory_state:unavailable",
                        *reason_codes,
                    )
                )
            ),
        }

    def _public_memory_provenance_label(value: Any) -> str | None:
        label = str(value or "").strip()
        allowed = {
            "Remembered from your profile memory.",
            "Remembered from your explicit preference.",
            "Task you asked Blink to track.",
            "Part of your relationship-style settings.",
            "Part of your teaching-profile settings.",
            "Derived from a prior conversation and not recently confirmed.",
            "Derived from prior conversation memory.",
        }
        return label if label in allowed else None

    def _public_memory_record(record: dict[str, Any]) -> dict[str, Any]:
        public_record = {
            "memory_id": _safe_public_text(record.get("memory_id"), limit=160),
            "display_kind": _safe_public_text(record.get("display_kind"), limit=64),
            "title": _safe_public_text(record.get("title"), limit=120),
            "summary": _safe_public_text(record.get("summary"), limit=240),
            "status": _safe_public_text(record.get("status"), limit=64),
            "currentness_status": _safe_public_text(
                record.get("currentness_status"), limit=64
            ),
            "confidence": _safe_optional_public_float(record.get("confidence")),
            "pinned": record.get("pinned") is True,
            "last_used_at": _safe_public_text(record.get("last_used_at"), limit=96),
            "last_used_reason": _safe_public_text(record.get("last_used_reason"), limit=120),
            "used_in_current_turn": record.get("used_in_current_turn") is True,
            "safe_provenance_label": _public_memory_provenance_label(
                record.get("safe_provenance_label")
            ),
            "user_actions": _safe_public_memory_actions(record.get("user_actions")),
            "reason_codes": _safe_public_reason_codes(record.get("reason_codes")),
        }
        return {key: value for key, value in public_record.items() if value is not None}

    def _public_memory_continuity_trace_payload(payload: Any) -> dict[str, Any] | None:
        data = _as_public_dict(payload)
        if not data:
            return None
        selected = []
        for item in data.get("selected_memories", []):
            record = _as_public_dict(item)
            if not record:
                continue
            selected.append(
                {
                    "memory_id": _safe_public_text(record.get("memory_id"), limit=160),
                    "display_kind": _safe_public_text(record.get("display_kind"), limit=64)
                    or "memory",
                    "summary": _safe_public_text(record.get("summary"), limit=120)
                    or "Memory selected.",
                    "safe_provenance_label": _safe_public_text(
                        record.get("safe_provenance_label"),
                        limit=120,
                    )
                    or "Derived from prior conversation memory.",
                    "source_language": _safe_public_choice(
                        record.get("source_language"),
                        allowed={"zh", "en", "unknown"},
                        default="unknown",
                    ),
                    "cross_language": record.get("cross_language") is True,
                    "inspectable": record.get("inspectable") is not False,
                    "editable": record.get("editable") is not False,
                    "effect_labels": _safe_public_text_list(
                        record.get("effect_labels"),
                        limit=80,
                        max_items=8,
                    )
                    or ["none"],
                    "linked_discourse_episode_ids": _safe_public_text_list(
                        record.get("linked_discourse_episode_ids"),
                        limit=120,
                        max_items=8,
                    ),
                    "conflict_labels": _safe_public_text_list(
                        record.get("conflict_labels"),
                        limit=80,
                        max_items=8,
                    ),
                    "staleness_labels": _safe_public_text_list(
                        record.get("staleness_labels"),
                        limit=80,
                        max_items=8,
                    ),
                    "reason_codes": _safe_public_reason_codes(record.get("reason_codes")),
                }
            )
            if len(selected) >= 8:
                break
        suppressed = []
        for item in data.get("suppressed_memories", []):
            record = _as_public_dict(item)
            if not record:
                continue
            suppressed.append(
                {
                    "bucket": _safe_public_text(record.get("bucket"), limit=64) or "other",
                    "count": max(0, _safe_public_int(record.get("count"))),
                    "user_visible": record.get("user_visible") is not False,
                    "reason_codes": _safe_public_reason_codes(record.get("reason_codes")),
                }
            )
            if len(suppressed) >= 8:
                break
        command_intent = _as_public_dict(data.get("command_intent"))
        public_command_intent = None
        if command_intent:
            public_command_intent = {
                "intent": _safe_public_text(command_intent.get("intent"), limit=64) or "none",
                "language": _safe_public_choice(
                    command_intent.get("language"),
                    allowed={"zh", "en", "unknown"},
                    default="unknown",
                ),
                "confidence": _safe_public_float(command_intent.get("confidence")),
                "text_chars": max(0, _safe_public_int(command_intent.get("text_chars"))),
                "reason_codes": _safe_public_reason_codes(command_intent.get("reason_codes")),
            }
        continuity_v3 = _as_public_dict(data.get("memory_continuity_v3"))
        discourse_refs = []
        for item in continuity_v3.get("selected_discourse_episodes", []):
            record = _as_public_dict(item)
            if not record:
                continue
            discourse_refs.append(
                {
                    "discourse_episode_id": _safe_public_text(
                        record.get("discourse_episode_id"),
                        limit=120,
                    )
                    or "discourse-episode-v3:unknown",
                    "category_labels": _safe_public_text_list(
                        record.get("category_labels"),
                        limit=80,
                        max_items=8,
                    ),
                    "effect_labels": _safe_public_text_list(
                        record.get("effect_labels"),
                        limit=80,
                        max_items=8,
                    )
                    or ["none"],
                    "confidence_bucket": _safe_public_text(
                        record.get("confidence_bucket"),
                        limit=32,
                    )
                    or "medium",
                    "reason_codes": _safe_public_reason_codes(
                        record.get("reason_codes"),
                        limit=12,
                    ),
                }
            )
            if len(discourse_refs) >= 8:
                break
        public_continuity_v3 = {
            "schema_version": _safe_public_int(continuity_v3.get("schema_version"), 3),
            "selected_discourse_episodes": discourse_refs,
            "effect_labels": _safe_public_text_list(
                continuity_v3.get("effect_labels"),
                limit=80,
                max_items=8,
            )
            or ["none"],
            "conflict_labels": _safe_public_text_list(
                continuity_v3.get("conflict_labels"),
                limit=80,
                max_items=8,
            ),
            "staleness_labels": _safe_public_text_list(
                continuity_v3.get("staleness_labels"),
                limit=80,
                max_items=8,
            ),
            "cross_language_transfer_count": _safe_public_int(
                continuity_v3.get("cross_language_transfer_count")
            ),
            "reason_codes": _safe_public_reason_codes(
                continuity_v3.get("reason_codes"),
                limit=24,
            ),
        }
        return {
            "schema_version": _safe_public_int(data.get("schema_version"), 1),
            "turn_id": _safe_public_text(data.get("turn_id"), limit=120) or "turn:auto",
            "session_id": _safe_public_text(data.get("session_id"), limit=120),
            "user_id": _safe_public_text(data.get("user_id"), limit=120),
            "agent_id": _safe_public_text(data.get("agent_id"), limit=120),
            "thread_id": _safe_public_text(data.get("thread_id"), limit=120),
            "created_at": _safe_public_text(data.get("created_at"), limit=96),
            "profile": _safe_public_text(data.get("profile"), limit=96) or "manual",
            "language": _safe_public_choice(
                data.get("language"),
                allowed={"zh", "en", "unknown"},
                default="unknown",
            ),
            "selected_memory_count": max(0, _safe_public_int(data.get("selected_memory_count"))),
            "suppressed_memory_count": max(
                0,
                _safe_public_int(data.get("suppressed_memory_count")),
            ),
            "cross_language_count": max(0, _safe_public_int(data.get("cross_language_count"))),
            "selected_memories": selected,
            "suppressed_memories": suppressed,
            "memory_effect": _safe_public_text(data.get("memory_effect"), limit=64) or "none",
            "inspectable": data.get("inspectable") is not False,
            "editable": data.get("editable") is not False,
            "user_actions": [
                action
                for action in _safe_public_text_list(data.get("user_actions"), limit=32)
                if action
                in {
                    "inspect",
                    "edit",
                    "correct",
                    "forget",
                    "suppress",
                    "pin",
                    "mark_stale",
                    "list_visible",
                    "explain",
                }
            ][:8],
            "memory_continuity_v3": public_continuity_v3,
            "command_intent": public_command_intent,
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=24),
        }

    def _runtime_memory_summary(payload: dict[str, Any]) -> str:
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        if not records:
            return "No visible memories."
        kind_counts: dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            display_kind = str(record.get("display_kind") or "memory").replace("_", " ")
            kind_counts[display_kind] = kind_counts.get(display_kind, 0) + 1
        kind_summary = ", ".join(
            f"{count} {display_kind}" for display_kind, count in sorted(kind_counts.items())
        )
        return f"{len(records)} visible memories" + (f": {kind_summary}" if kind_summary else ".")

    def _public_memory_payload(
        payload: dict[str, Any],
        *,
        available: bool | None = None,
    ) -> dict[str, Any]:
        memory_persona = _public_memory_persona_performance_payload(
            payload.get("memory_persona_performance") or {}
        )
        memory_continuity_trace = _public_memory_continuity_trace_payload(
            payload.get("memory_continuity_trace")
        ) or memory_persona.get("memory_continuity_trace")
        public_payload = {
            "available": payload.get("available") is True if available is None else available,
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "user_id": _safe_public_text(payload.get("user_id"), limit=96),
            "agent_id": _safe_public_text(payload.get("agent_id"), limit=96),
            "generated_at": _safe_public_text(payload.get("generated_at"), limit=96),
            "records": [
                _public_memory_record(record)
                for record in payload.get("records", [])
                if isinstance(record, dict)
            ],
            "hidden_counts": _public_count_mapping(payload.get("hidden_counts")),
            "health_summary": (
                _safe_public_text(payload.get("health_summary"), limit=160)
                or "Memory health unavailable."
            ),
            "used_in_current_reply": memory_persona["used_in_current_reply"],
            "behavior_effects": memory_persona["behavior_effects"],
            "persona_references": memory_persona["persona_references"],
            "persona_anchor_refs_v3": memory_persona["persona_anchor_refs_v3"],
            "persona_anchor_bank_v3": memory_persona["persona_anchor_bank_v3"],
            "memory_persona_performance": memory_persona,
            "memory_continuity_trace": memory_continuity_trace,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }
        public_payload["summary"] = _runtime_memory_summary(public_payload)
        return public_payload

    def _runtime_memory_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_memory_unavailable_payload("runtime_not_active")

        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return _runtime_memory_unavailable_payload("runtime_memory_surface_missing")

        try:
            session_ids = session_resolver()
            current_trace_reader = getattr(runtime, "current_memory_use_trace", None)
            recent_trace_reader = getattr(runtime, "recent_memory_use_traces", None)
            continuity_reader = getattr(runtime, "current_memory_continuity_trace", None)
            current_turn_trace = current_trace_reader() if callable(current_trace_reader) else None
            current_continuity_trace = (
                continuity_reader() if callable(continuity_reader) else None
            )
            recent_use_traces = (
                recent_trace_reader(limit=8) if callable(recent_trace_reader) else None
            )
            snapshot = build_memory_palace_snapshot(
                store=store,
                session_ids=session_ids,
                include_suppressed=False,
                include_historical=False,
                current_turn_trace=current_turn_trace,
                recent_use_traces=recent_use_traces,
                limit=40,
                claim_scan_limit=160,
            )
            payload = snapshot.as_dict()
            if current_continuity_trace is not None:
                payload["memory_continuity_trace"] = _as_public_dict(current_continuity_trace)
            payload["memory_persona_performance"] = _runtime_memory_persona_performance_payload(
                active_config=_active_browser_config(),
                browser_media=_current_client_media_payload(),
                current_mode=getattr(
                    app.state,
                    "blink_browser_interaction_mode",
                    BrowserInteractionMode.WAITING,
                ),
                hidden_counts=dict(payload.get("hidden_counts") or {}),
                prefer_cached=False,
            )
        except Exception as exc:  # pragma: no cover - defensive browser status surface
            return _runtime_memory_unavailable_payload(f"runtime_memory_error:{type(exc).__name__}")

        snapshot_reason_codes = payload.get("reason_codes")
        payload = _public_memory_payload(payload, available=True)
        payload["reason_codes"] = _safe_public_reason_codes(
            (
                "runtime_memory_state:available",
                *_safe_public_reason_codes(snapshot_reason_codes),
            )
        )
        return payload

    def _runtime_memory_action_rejected_payload(
        *,
        memory_id: str,
        action: str,
        reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return _public_memory_action_payload(
            {
                "schema_version": 1,
                "accepted": False,
                "applied": False,
                "action": action,
                "memory_id": memory_id,
                "record_kind": None,
                "replacement_memory_id": None,
                "reason_codes": (
                    "memory_action_rejected",
                    *reason_codes,
                ),
            }
        )

    def _public_memory_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
        public_payload = {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "accepted": payload.get("accepted") is True,
            "applied": payload.get("applied") is True,
            "action": _safe_public_text(payload.get("action"), limit=64),
            "memory_id": _safe_public_text(payload.get("memory_id"), limit=160),
            "record_kind": _safe_public_text(payload.get("record_kind"), limit=64),
            "replacement_memory_id": _safe_public_text(
                payload.get("replacement_memory_id"), limit=160
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }
        return {
            key: value
            for key, value in public_payload.items()
            if value is not None and value != ""
        }

    def _runtime_memory_action_payload(
        *,
        memory_id: str,
        action: str,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_memory_action_rejected_payload(
                memory_id=memory_id,
                action=action,
                reason_codes=("runtime_not_active",),
            )

        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return _runtime_memory_action_rejected_payload(
                memory_id=memory_id,
                action=action,
                reason_codes=("runtime_memory_surface_missing",),
            )

        try:
            result = apply_memory_governance_action(
                store=store,
                session_ids=session_resolver(),
                memory_id=memory_id,
                action=action,
                replacement_value=request_data.get("replacement_value"),
                notes=request_data.get("notes"),
                source="browser_memory_endpoint",
            )
            as_dict = getattr(result, "as_dict", None)
            if callable(as_dict):
                return _public_memory_action_payload(as_dict())
            if isinstance(result, dict):
                return _public_memory_action_payload(dict(result))
        except Exception:  # pragma: no cover - defensive browser action boundary
            return _runtime_memory_action_rejected_payload(
                memory_id=memory_id,
                action=action,
                reason_codes=("runtime_memory_action_error",),
            )
        return _runtime_memory_action_rejected_payload(
            memory_id=memory_id,
            action=action,
            reason_codes=("runtime_memory_action_invalid_result",),
        )

    def _emit_memory_action_performance_event(payload: dict[str, Any]):
        _emit_performance_event(
            event_type="memory.action",
            source="memory",
            mode=_resting_interaction_mode(),
            metadata={
                "action": payload.get("action"),
                "accepted": payload.get("accepted") is True,
                "applied": payload.get("applied") is True,
                "record_kind": payload.get("record_kind"),
            },
            reason_codes=("memory:action", *_safe_public_reason_codes(payload.get("reason_codes"))),
        )

    def _safe_public_text(value: Any, *, limit: int = 120) -> str:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            return ""
        redacted_markers = (
            "traceback",
            "runtimeerror",
            "/tmp",
            ".db",
            "raw_json",
            "secret",
            "authorization",
            "api_key",
            "bearer",
            "credential",
            "developer_message",
            "developer_prompt",
            "device_label",
            "system_prompt",
            "raw_prompt",
            "prompt_text",
            "transcript",
            "audio_bytes",
            "image_bytes",
            "raw_audio",
            "raw_image",
            "sdp_offer",
            "ice_candidate",
        )
        lowered = normalized.lower()
        if any(marker in lowered for marker in redacted_markers):
            return "redacted"
        return normalized[:limit]

    def _safe_public_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_public_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    def _safe_optional_public_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def _safe_public_text_list(
        value: Any,
        *,
        limit: int = 96,
        max_items: int = 16,
    ) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in list(value)[:max_items]:
            text = _safe_public_text(item, limit=limit)
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _safe_public_memory_actions(value: Any) -> list[str]:
        allowed = {
            "review",
            "pin",
            "suppress",
            "correct",
            "forget",
            "export",
            "mark_stale",
            "mark-stale",
            "mark_done",
            "cancel",
        }
        return [action for action in _safe_public_text_list(value, limit=32) if action in allowed]

    def _safe_public_reason_codes(
        value: Any,
        fallback: tuple[str, ...] = (),
        *,
        limit: int = 16,
    ) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return list(fallback)
        result: list[str] = []
        seen: set[str] = set()
        for item in list(value)[:limit]:
            code = _safe_public_text(item, limit=96)
            if not code or code in seen:
                continue
            seen.add(code)
            result.append(code)
        return result or list(fallback)

    def _safe_public_choice(value: Any, *, allowed: set[str], default: str) -> str:
        text = _safe_public_text(value, limit=64)
        return text if text in allowed else default

    def _safe_public_echo_state(value: Any) -> str:
        if isinstance(value, bool):
            return "enabled" if value else "disabled"
        text = _safe_public_text(value, limit=64).strip().lower().replace("-", "_")
        if text in {"1", "true", "yes", "on", "enabled"}:
            return "enabled"
        if text in {"0", "false", "no", "off", "disabled"}:
            return "disabled"
        if text in {"unsupported", "unavailable", "not_supported"}:
            return "unsupported"
        return "unknown"

    def _safe_public_bool_or_none(value: Any) -> bool | None:
        if value is True:
            return True
        if value is False:
            return False
        text = _safe_public_text(value, limit=32).strip().lower()
        if text in {"1", "true", "yes", "on", "enabled", "safe"}:
            return True
        if text in {"0", "false", "no", "off", "disabled", "unsafe"}:
            return False
        return None

    def _public_client_media_payload(value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        mode = _safe_public_choice(
            raw.get("mode"),
            allowed={"unreported", "camera_and_microphone", "audio_only", "unavailable"},
            default="unreported",
        )
        media_state_values = {
            "unknown",
            "ready",
            "receiving",
            "stalled",
            "stale",
            "unavailable",
            "permission_denied",
            "device_not_found",
            "error",
        }
        camera_state = _safe_public_choice(
            raw.get("camera_state") or raw.get("cameraState"),
            allowed=media_state_values,
            default="unknown",
        )
        microphone_state = _safe_public_choice(
            raw.get("microphone_state") or raw.get("microphoneState"),
            allowed=media_state_values,
            default="unknown",
        )
        if mode == "camera_and_microphone":
            if camera_state == "unknown":
                camera_state = "ready"
            if microphone_state == "unknown":
                microphone_state = "ready"
        elif mode == "audio_only":
            if microphone_state == "unknown":
                microphone_state = "ready"
            if camera_state == "unknown":
                camera_state = "unavailable"
        elif mode == "unavailable":
            if camera_state == "ready":
                camera_state = "unknown"
            if microphone_state == "ready":
                microphone_state = "unknown"
        elif mode == "unreported":
            camera_state = "unknown"
            microphone_state = "unknown"

        raw_echo = raw.get("echo") if isinstance(raw.get("echo"), dict) else {}
        echo = {
            "echo_cancellation": _safe_public_echo_state(
                raw.get("echo_cancellation")
                if "echo_cancellation" in raw
                else raw.get("echoCancellation", raw_echo.get("echo_cancellation"))
            ),
            "noise_suppression": _safe_public_echo_state(
                raw.get("noise_suppression")
                if "noise_suppression" in raw
                else raw.get("noiseSuppression", raw_echo.get("noise_suppression"))
            ),
            "auto_gain_control": _safe_public_echo_state(
                raw.get("auto_gain_control")
                if "auto_gain_control" in raw
                else raw.get("autoGainControl", raw_echo.get("auto_gain_control"))
            ),
        }
        echo_safe = _safe_public_bool_or_none(
            raw.get("echo_safe")
            if "echo_safe" in raw
            else raw.get("echoSafe", raw_echo.get("echo_safe", raw_echo.get("echoSafe")))
        )
        output_playback_state = _safe_public_choice(
            raw.get("output_playback_state") or raw.get("outputPlaybackState"),
            allowed={
                "unknown",
                "idle",
                "playing",
                "speaking",
                "buffering",
                "stalled",
                "muted",
                "ended",
                "error",
            },
            default="unknown",
        )
        output_route = _safe_public_choice(
            raw.get("output_route") or raw.get("outputRoute"),
            allowed={
                "unknown",
                "speaker",
                "headphones",
                "headset",
                "earpiece",
                "bluetooth",
                "muted",
            },
            default="unknown",
        )
        webrtc_stats = sanitize_webrtc_audio_stats(
            raw.get("webrtc_stats") or raw.get("webrtcStats") or raw.get("stats")
        )
        updated_at = _safe_public_text(raw.get("updated_at") or raw.get("updatedAt"), limit=48)
        reason_codes = _safe_public_reason_codes(
            raw.get("reason_codes") or raw.get("reasonCodes"),
            fallback=(f"browser_media:{mode}",),
            limit=12,
        )
        required_codes = (
            f"browser_media:{mode}",
            f"browser_camera:{camera_state}",
            f"browser_microphone:{microphone_state}",
            f"browser_echo_cancellation:{echo['echo_cancellation']}",
            f"browser_noise_suppression:{echo['noise_suppression']}",
            f"browser_auto_gain_control:{echo['auto_gain_control']}",
            f"browser_output_playback:{output_playback_state}",
            f"browser_output_route:{output_route}",
            (
                "browser_echo_safe:reported_true"
                if echo_safe is True
                else "browser_echo_safe:reported_false"
                if echo_safe is False
                else "browser_echo_safe:unreported"
            ),
        )
        return {
            "schema_version": 1,
            "available": mode != "unreported",
            "mode": mode,
            "camera_state": camera_state,
            "microphone_state": microphone_state,
            "echo": echo,
            "echo_safe": echo_safe,
            "output_playback_state": output_playback_state,
            "output_route": output_route,
            "webrtc_stats": webrtc_stats,
            "updated_at": updated_at or None,
            "reason_codes": _safe_public_reason_codes((*required_codes, *reason_codes), limit=16),
        }

    def _current_client_media_payload() -> dict[str, Any]:
        return _public_client_media_payload(
            getattr(app.state, "blink_client_media_state", None)
        )

    def _unreported_client_media_payload(*reason_codes: str) -> dict[str, Any]:
        return _public_client_media_payload(
            {
                "mode": "unreported",
                "camera_state": "unknown",
                "microphone_state": "unknown",
                "updated_at": datetime.now(UTC).isoformat(),
                "reason_codes": reason_codes or ("browser_media:unreported",),
            }
        )

    def _reset_client_media_state(*reason_codes: str) -> dict[str, Any]:
        state = _unreported_client_media_payload(*reason_codes)
        app.state.blink_client_media_state = state
        _clear_runtime_payload_cache()
        return state

    def _sync_runtime_camera_context_from_media(
        *,
        browser_media: dict[str, Any],
        active_config: LocalBrowserConfig,
        source: str,
    ) -> None:
        """Keep runtime vision presence aligned with live media.

        This intentionally avoids adding durable LLM system messages for camera
        availability. Camera truth is turn-scoped and read from the session/tool
        path so a temporary permission-denied hint cannot outlive a later ready
        camera report.
        """
        if not active_config.vision_enabled:
            return
        _runtime_session_v3().note_client_media(browser_media)
        camera_state = str(browser_media.get("camera_state") or "unknown")
        media_mode = str(browser_media.get("mode") or "unknown")
        available = (
            media_mode == "camera_and_microphone"
            and camera_state in {"ready", "receiving"}
        )
        hard_unavailable = camera_state in {
            "permission_denied",
            "device_not_found",
            "error",
        }
        if not available and not hard_unavailable:
            return

        brain_runtime = getattr(app.state, "blink_active_expression_runtime", None)
        if brain_runtime is not None:
            _safe_note_vision_connected(brain_runtime, available)

        next_context_state = "available" if available else "unavailable"
        if getattr(app.state, "blink_active_camera_context_state", None) == next_context_state:
            return
        app.state.blink_active_camera_context_state = next_context_state
        if available:
            _emit_performance_event(
                event_type="camera.context_available",
                source=source,
                mode=_resting_interaction_mode(),
                metadata={"camera_state": camera_state, "context_state": next_context_state},
                reason_codes=("camera:context_available", "camera:available_context_restored"),
            )
            return
        _emit_performance_event(
            event_type="camera.context_unavailable",
            source=source,
            mode=BrowserInteractionMode.ERROR,
            metadata={"camera_state": camera_state, "context_state": next_context_state},
            reason_codes=("camera:context_unavailable", "camera:unavailable_context_seen"),
        )

    def _runtime_client_media_update_payload(request_data: dict[str, Any]) -> dict[str, Any]:
        state = _public_client_media_payload(
            {
                **request_data,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        app.state.blink_client_media_state = state
        runtime_session = _runtime_session_v3()
        active_session_id = _active_performance_session_id()
        active_client_id = _active_performance_client_id()
        if active_session_id is not None or active_client_id is not None:
            runtime_session.note_client_connected(
                session_id=active_session_id,
                client_id=active_client_id,
            )
        runtime_session.note_client_media(state)
        _emit_performance_event(
            event_type="browser_media.reported",
            source="client-media",
            mode=_resting_interaction_mode(),
            metadata={
                "media_mode": state["mode"],
                "camera_state": state["camera_state"],
                "microphone_state": state["microphone_state"],
                "echo_cancellation_state": state["echo"]["echo_cancellation"],
                "noise_suppression_state": state["echo"]["noise_suppression"],
                "auto_gain_control_state": state["echo"]["auto_gain_control"],
                "echo_safe_state": (
                    "reported_true"
                    if state["echo_safe"] is True
                    else "reported_false"
                    if state["echo_safe"] is False
                    else "unreported"
                ),
                "output_playback_state": state["output_playback_state"],
                "output_route": state["output_route"],
                "webrtc_stats_count": len(state["webrtc_stats"]),
            },
            reason_codes=("browser_media:reported", *state["reason_codes"]),
        )
        active_config = _active_browser_config()
        _sync_runtime_camera_context_from_media(
            browser_media=state,
            active_config=active_config,
            source="client-media",
        )
        interruption_state = getattr(app.state, "blink_browser_interruption", None)
        interruption_payload = (
            interruption_state.snapshot()
            if isinstance(interruption_state, BrowserInterruptionStateTracker)
            else {}
        )
        _webrtc_audio_health_payload(
            active_config=active_config,
            browser_media=state,
            current_mode=_resting_interaction_mode(),
            interruption_payload=interruption_payload,
        )
        if state["camera_state"] in {"ready", "receiving"} and state["mode"] == "camera_and_microphone":
            _emit_performance_event(
                event_type="camera.connected",
                source="client-media",
                mode=_resting_interaction_mode(),
                metadata={
                    "camera_state": state["camera_state"],
                    "scene_transition": "camera_ready",
                },
                reason_codes=(
                    "camera:connected",
                    "scene_social_transition:camera_ready",
                    *state["reason_codes"],
                ),
            )
        elif state["camera_state"] == "stale" and state["mode"] == "camera_and_microphone":
            _emit_performance_event(
                event_type="camera.frame_stale",
                source="client-media",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": state["camera_state"],
                    "scene_transition": "vision_stale",
                },
                reason_codes=(
                    "camera:frame_stale",
                    "scene_social_transition:vision_stale",
                    *state["reason_codes"],
                ),
            )
        elif state["camera_state"] == "stalled" and state["mode"] == "camera_and_microphone":
            _emit_performance_event(
                event_type="camera.health_stalled",
                source="client-media",
                mode=BrowserInteractionMode.ERROR,
                metadata={
                    "camera_state": state["camera_state"],
                    "scene_transition": "vision_stale",
                },
                reason_codes=(
                    "camera:health_stalled",
                    "scene_social_transition:vision_stale",
                    *state["reason_codes"],
                ),
            )
        return {
            "schema_version": 1,
            "accepted": True,
            "applied": True,
            "browser_media": state,
            "reason_codes": _safe_public_reason_codes(
                ("browser_media_report:accepted", *state["reason_codes"]),
                limit=16,
            ),
        }

    def _as_public_dict(value: Any) -> dict[str, Any]:
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            result = as_dict()
            return dict(result) if isinstance(result, dict) else {}
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _public_reason_code_counts(value: Any) -> dict[str, int]:
        rows: dict[str, int] = {}
        if not isinstance(value, dict):
            return rows
        for key, count in list(value.items())[:16]:
            code = _safe_public_text(key, limit=96)
            if not code:
                continue
            rows[code] = rows.get(code, 0) + _safe_public_int(count)
        return rows

    def _public_count_mapping(value: Any, *, key_limit: int = 96) -> dict[str, int]:
        rows: dict[str, int] = {}
        if not isinstance(value, dict):
            return rows
        for key, count in list(value.items())[:24]:
            public_key = _safe_public_text(key, limit=key_limit) or "unknown"
            rows[public_key] = rows.get(public_key, 0) + _safe_public_int(count)
        return dict(sorted(rows.items()))

    def _public_memory_persona_reply_ref(record: Any) -> dict[str, Any]:
        payload = _as_public_dict(record)
        return {
            "memory_id": _safe_public_text(payload.get("memory_id"), limit=160),
            "display_kind": _safe_public_text(payload.get("display_kind"), limit=64),
            "title": _safe_public_text(payload.get("title"), limit=120) or "Memory",
            "used_reason": _safe_public_text(payload.get("used_reason"), limit=96),
            "behavior_effect": _safe_public_text(payload.get("behavior_effect"), limit=120),
            "effect_labels": _safe_public_text_list(
                payload.get("effect_labels"),
                limit=96,
                max_items=8,
            ),
            "linked_discourse_episode_ids": _safe_public_text_list(
                payload.get("linked_discourse_episode_ids"),
                limit=120,
                max_items=8,
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes"), limit=12),
        }

    def _public_persona_reference_payload(record: Any) -> dict[str, Any]:
        payload = _as_public_dict(record)
        return {
            "reference_id": _safe_public_text(payload.get("reference_id"), limit=96),
            "mode": _safe_public_text(payload.get("mode"), limit=64),
            "label": _safe_public_text(payload.get("label"), limit=96),
            "applies": payload.get("applies") is True,
            "behavior_effect": _safe_public_text(payload.get("behavior_effect"), limit=120),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes"), limit=12),
        }

    def _public_performance_plan_v2_reference(record: Any) -> dict[str, Any]:
        payload = _as_public_dict(record)
        return {
            "reference_id": _safe_public_text(payload.get("reference_id"), limit=96),
            "locale": _safe_public_text(payload.get("locale"), limit=16) or "en",
            "scenario": _safe_public_text(payload.get("scenario"), limit=80),
            "stance": _safe_public_text(payload.get("stance"), limit=140),
            "response_shape": _safe_public_text(payload.get("response_shape"), limit=180),
            "performance_notes": _safe_public_text_list(
                payload.get("performance_notes"),
                limit=120,
                max_items=3,
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes"), limit=8),
        }

    def _public_performance_plan_v2_policy(
        payload: Any,
        *,
        fallback_state: str,
    ) -> dict[str, Any]:
        data = _as_public_dict(payload)
        public: dict[str, Any] = {
            "state": _safe_public_text(data.get("state"), limit=80) or fallback_state,
        }
        for key, value in list(data.items())[:16]:
            safe_key = _safe_public_text(key, limit=80)
            if not safe_key or safe_key == "state":
                continue
            if isinstance(value, bool):
                public[safe_key] = value
            elif isinstance(value, int):
                public[safe_key] = _safe_public_int(value)
            elif isinstance(value, float):
                public[safe_key] = _safe_public_int(value)
            elif isinstance(value, (list, tuple, set)):
                if safe_key == "selected_memory_refs":
                    refs = []
                    for item in list(value)[:8]:
                        record = _as_public_dict(item)
                        if not record:
                            continue
                        refs.append(
                            {
                                "memory_id": _safe_public_text(
                                    record.get("memory_id"),
                                    limit=160,
                                )
                                or "memory:unknown",
                                "display_kind": _safe_public_text(
                                    record.get("display_kind"),
                                    limit=64,
                                )
                                or "memory",
                                "summary": _safe_public_text(
                                    record.get("summary"),
                                    limit=120,
                                )
                                or "Memory selected.",
                                "source_language": _safe_public_choice(
                                    record.get("source_language"),
                                    allowed={"zh", "en", "unknown"},
                                    default="unknown",
                                ),
                                "cross_language": record.get("cross_language") is True,
                                "effect_labels": _safe_public_text_list(
                                    record.get("effect_labels"),
                                    limit=80,
                                    max_items=8,
                                )
                                or ["none"],
                                "confidence_bucket": _safe_public_text(
                                    record.get("confidence_bucket"),
                                    limit=32,
                                )
                                or "medium",
                                "reason_codes": _safe_public_reason_codes(
                                    record.get("reason_codes"),
                                    limit=12,
                                ),
                            }
                        )
                    public[safe_key] = refs
                else:
                    public[safe_key] = _safe_public_text_list(value, limit=80, max_items=8)
            elif isinstance(value, dict):
                public[safe_key] = _safe_public_text_mapping(value, key_limit=80, value_limit=80)
            else:
                public[safe_key] = _safe_public_text(value, limit=96)
        return public

    def _public_performance_plan_v2_payload(
        payload: Any,
        *,
        parent_profile: str,
        parent_language: str,
        parent_tts_label: str,
        parent_turn_state: str,
    ) -> dict[str, Any]:
        data = _as_public_dict(payload)
        refs = [
            _public_performance_plan_v2_reference(record)
            for record in data.get("persona_references_used", [])
            if isinstance(record, dict)
        ][:8]
        language = _safe_public_text(data.get("language"), limit=32) or parent_language
        if language not in {"zh", "en"}:
            language = "zh" if str(parent_language).startswith("zh") else "en"
        public = {
            "schema_version": _safe_public_int(data.get("schema_version"), 2),
            "turn_id": _safe_public_text(data.get("turn_id"), limit=96)
            or f"turn:{parent_profile}:{language}:{parent_turn_state}",
            "profile": _safe_public_text(data.get("profile"), limit=96) or parent_profile,
            "modality": _safe_public_text(data.get("modality"), limit=64) or "browser",
            "language": language,
            "tts_label": _safe_public_text(data.get("tts_label"), limit=96)
            or parent_tts_label,
            "floor_state": _safe_public_text(data.get("floor_state"), limit=64) or "unknown",
            "visible_mode": _safe_public_text(data.get("visible_mode"), limit=64)
            or parent_turn_state,
            "stance": _safe_public_text(data.get("stance"), limit=140) or "unavailable",
            "response_shape": _safe_public_text(data.get("response_shape"), limit=96)
            or "answer_first",
            "memory_callback_policy": _public_performance_plan_v2_policy(
                data.get("memory_callback_policy"),
                fallback_state="avoid_ungrounded_callback",
            ),
            "camera_reference_policy": _public_performance_plan_v2_policy(
                data.get("camera_reference_policy"),
                fallback_state="no_visual_claim",
            ),
            "interruption_policy": _public_performance_plan_v2_policy(
                data.get("interruption_policy"),
                fallback_state="protected_continue",
            ),
            "speech_chunking_hints": _public_performance_plan_v2_policy(
                data.get("speech_chunking_hints"),
                fallback_state="voice_suitable_chunks",
            ),
            "ui_state_hints": _public_performance_plan_v2_policy(
                data.get("ui_state_hints"),
                fallback_state="style_summary_available",
            ),
            "style_summary": _safe_public_text(data.get("style_summary"), limit=180)
            or "Persona style unavailable.",
            "persona_references_used": refs,
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=24),
        }
        if not public["reason_codes"]:
            public["reason_codes"] = ["performance_plan:v2", "performance_plan:unavailable"]
        return public

    def _public_performance_plan_v3_actor_ref(payload: Any) -> dict[str, Any]:
        data = _as_public_dict(payload)
        return {
            "frame_id": _safe_public_text(data.get("frame_id"), limit=96)
            or "actor-control:unavailable",
            "sequence": _safe_public_int(data.get("sequence")),
            "boundary": _safe_public_text(data.get("boundary"), limit=64) or "unavailable",
            "condition_cache_digest": _safe_public_text(
                data.get("condition_cache_digest"),
                limit=32,
            )
            or "unavailable",
            "source_event_ids": [
                _safe_public_int(item)
                for item in list(data.get("source_event_ids") or [])[:12]
            ],
        }

    def _public_performance_plan_v3_tts_capabilities(payload: Any) -> dict[str, Any]:
        data = _as_public_dict(payload)
        return {
            "backend_label": _safe_public_text(data.get("backend_label"), limit=80)
            or "unknown",
            "chunk_boundaries_enabled": data.get("chunk_boundaries_enabled") is True,
            "interruption_flush_enabled": data.get("interruption_flush_enabled") is True,
            "speech_rate_enabled": data.get("speech_rate_enabled") is True,
            "prosody_emphasis_enabled": data.get("prosody_emphasis_enabled") is True,
            "pause_timing_enabled": data.get("pause_timing_enabled") is True,
            "partial_stream_abort_enabled": data.get("partial_stream_abort_enabled") is True,
            "interruption_discard_enabled": data.get("interruption_discard_enabled") is True,
            "expression_controls_hardware": False,
            "unsupported_controls": _safe_public_text_list(
                data.get("unsupported_controls"),
                limit=80,
                max_items=12,
            ),
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=16),
        }

    def _public_performance_plan_v3_persona_anchor(payload: Any) -> dict[str, Any]:
        data = _as_public_dict(payload)
        return {
            "schema_version": _safe_public_int(data.get("schema_version"), 3),
            "anchor_id": _safe_public_text(data.get("anchor_id"), limit=96),
            "situation_key": _safe_public_text(data.get("situation_key"), limit=80)
            or "unknown",
            "stance_label": _safe_public_text(data.get("stance_label"), limit=96)
            or "focused",
            "response_shape_label": _safe_public_text(
                data.get("response_shape_label"),
                limit=96,
            )
            or "answer_first",
            "behavior_constraint_count": _safe_public_int(
                data.get("behavior_constraint_count")
            ),
            "negative_example_count": _safe_public_int(data.get("negative_example_count")),
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=12),
        }

    def _public_persona_anchor_bank_v3_payload(payload: Any) -> dict[str, Any]:
        data = _as_public_dict(payload)
        anchors = []
        for record in data.get("anchors", []):
            if not isinstance(record, dict):
                continue
            public_anchor = _public_performance_plan_v3_persona_anchor(record)
            behavior_constraints = _safe_public_text_list(
                record.get("behavior_constraints"),
                limit=160,
                max_items=8,
            )
            negative_examples = _safe_public_text_list(
                record.get("negative_examples"),
                limit=160,
                max_items=8,
            )
            public_anchor["zh_example"] = _safe_public_text(
                record.get("zh_example"),
                limit=280,
            )
            public_anchor["en_example"] = _safe_public_text(
                record.get("en_example"),
                limit=280,
            )
            public_anchor["behavior_constraints"] = behavior_constraints
            public_anchor["negative_examples"] = negative_examples
            public_anchor["behavior_constraint_count"] = _safe_public_int(
                record.get("behavior_constraint_count"),
                len(behavior_constraints),
            )
            public_anchor["negative_example_count"] = _safe_public_int(
                record.get("negative_example_count"),
                len(negative_examples),
            )
            anchors.append(public_anchor)
        return {
            "schema_version": _safe_public_int(data.get("schema_version"), 3),
            "anchor_count": _safe_public_int(data.get("anchor_count"), len(anchors)),
            "required_situation_keys": _safe_public_text_list(
                data.get("required_situation_keys"),
                limit=80,
                max_items=12,
            ),
            "anchors": anchors[:12],
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=16),
        }

    def _public_performance_plan_v3_payload(
        payload: Any,
        *,
        parent_profile: str,
        parent_language: str,
        parent_tts_label: str,
        parent_turn_state: str,
    ) -> dict[str, Any]:
        data = _as_public_dict(payload)
        language = _safe_public_text(data.get("language"), limit=32) or parent_language
        if language not in {"zh", "en"}:
            language = "zh" if str(parent_language).startswith("zh") else "en"
        profile = _safe_public_text(data.get("profile"), limit=96) or parent_profile
        if profile not in {"browser-zh-melo", "browser-en-kokoro"}:
            profile = "browser-en-kokoro" if language == "en" else "browser-zh-melo"
        public = {
            "schema_version": _safe_public_int(data.get("schema_version"), 3),
            "plan_id": _safe_public_text(data.get("plan_id"), limit=96)
            or f"performance-plan-v3:{profile}:{language}:{parent_turn_state}",
            "turn_id": _safe_public_text(data.get("turn_id"), limit=96)
            or f"turn:{profile}:{language}:{parent_turn_state}",
            "profile": profile,
            "language": language,
            "tts_runtime_label": _safe_public_text(
                data.get("tts_runtime_label"),
                limit=96,
            )
            or parent_tts_label,
            "created_at_ms": _safe_public_int(data.get("created_at_ms")),
            "actor_control_ref": _public_performance_plan_v3_actor_ref(
                data.get("actor_control_ref")
            ),
            "stance": _safe_public_text(data.get("stance"), limit=96)
            or "attentive_listening",
            "response_shape": _safe_public_text(data.get("response_shape"), limit=96)
            or "wait_then_answer",
            "voice_pacing": _public_performance_plan_v2_policy(
                data.get("voice_pacing"),
                fallback_state="balanced",
            ),
            "speech_chunk_budget": _public_performance_plan_v2_policy(
                data.get("speech_chunk_budget"),
                fallback_state="balanced",
            ),
            "subtitle_policy": _public_performance_plan_v2_policy(
                data.get("subtitle_policy"),
                fallback_state="immediate_first_then_bounded",
            ),
            "camera_reference_policy": _public_performance_plan_v2_policy(
                data.get("camera_reference_policy"),
                fallback_state="no_visual_claim",
            ),
            "memory_callback_policy": _public_performance_plan_v2_policy(
                data.get("memory_callback_policy"),
                fallback_state="no_callback",
            ),
            "interruption_policy": _public_performance_plan_v2_policy(
                data.get("interruption_policy"),
                fallback_state="protected",
            ),
            "repair_policy": _public_performance_plan_v2_policy(
                data.get("repair_policy"),
                fallback_state="normal",
            ),
            "ui_status_copy": _safe_public_text(data.get("ui_status_copy"), limit=180),
            "plan_summary": _safe_public_text(data.get("plan_summary"), limit=180),
            "persona_reference_ids": _safe_public_text_list(
                data.get("persona_reference_ids"),
                limit=96,
                max_items=8,
            ),
            "persona_anchor_refs_v3": [
                _public_performance_plan_v3_persona_anchor(record)
                for record in data.get("persona_anchor_refs_v3", [])
                if isinstance(record, dict)
            ][:8],
            "tts_capabilities": _public_performance_plan_v3_tts_capabilities(
                data.get("tts_capabilities")
            ),
            "reason_trace": _safe_public_reason_codes(data.get("reason_trace"), limit=32),
        }
        if not public["reason_trace"]:
            public["reason_trace"] = ["performance_plan:v3", "performance_plan:unavailable"]
        return public

    def _public_memory_persona_performance_payload(payload: Any) -> dict[str, Any]:
        data = _as_public_dict(payload)
        used_refs = [
            _public_memory_persona_reply_ref(record)
            for record in data.get("used_in_current_reply", [])
            if isinstance(record, dict)
        ][:8]
        persona_refs = [
            _public_persona_reference_payload(record)
            for record in data.get("persona_references", [])
            if isinstance(record, dict)
        ][:12]
        profile = _safe_public_text(data.get("profile"), limit=96) or "manual"
        language = _safe_public_text(data.get("language"), limit=32) or "unknown"
        tts_label = _safe_public_text(data.get("tts_label"), limit=96) or "unknown"
        current_turn_state = _safe_public_text(data.get("current_turn_state"), limit=64) or "unknown"
        performance_plan_v3 = _public_performance_plan_v3_payload(
            data.get("performance_plan_v3"),
            parent_profile=profile,
            parent_language=language,
            parent_tts_label=tts_label,
            parent_turn_state=current_turn_state,
        )
        persona_anchor_refs_v3 = [
            _public_performance_plan_v3_persona_anchor(record)
            for record in data.get("persona_anchor_refs_v3", [])
            if isinstance(record, dict)
        ][:8]
        if not persona_anchor_refs_v3:
            persona_anchor_refs_v3 = list(performance_plan_v3["persona_anchor_refs_v3"])
        behavior_effects = _safe_public_text_list(
            data.get("behavior_effects"),
            limit=96,
            max_items=16,
        )
        public_payload = {
            "schema_version": _safe_public_int(data.get("schema_version"), 1),
            "available": data.get("available") is True,
            "profile": profile,
            "modality": _safe_public_text(data.get("modality"), limit=64) or "browser",
            "language": language,
            "tts_label": tts_label,
            "protected_playback": data.get("protected_playback") is not False,
            "camera_state": _safe_public_text(data.get("camera_state"), limit=64)
            or "unknown",
            "continuous_perception_enabled": (
                data.get("continuous_perception_enabled") is True
            ),
            "current_turn_state": current_turn_state,
            "memory_policy": _safe_public_text(data.get("memory_policy"), limit=64)
            or "unavailable",
            "selected_memory_count": _safe_public_int(data.get("selected_memory_count")),
            "suppressed_memory_count": _safe_public_int(data.get("suppressed_memory_count")),
            "used_in_current_reply": used_refs,
            "behavior_effects": behavior_effects,
            "persona_references": persona_refs,
            "persona_anchor_refs_v3": persona_anchor_refs_v3,
            "persona_anchor_bank_v3": _public_persona_anchor_bank_v3_payload(
                data.get("persona_anchor_bank_v3")
            ),
            "summary": _safe_public_text(data.get("summary"), limit=180)
            or "Memory/persona performance unavailable.",
            "performance_plan_v2": _public_performance_plan_v2_payload(
                data.get("performance_plan_v2"),
                parent_profile=profile,
                parent_language=language,
                parent_tts_label=tts_label,
                parent_turn_state=current_turn_state,
            ),
            "performance_plan_v3": performance_plan_v3,
            "memory_continuity_trace": _public_memory_continuity_trace_payload(
                data.get("memory_continuity_trace")
            ),
            "reason_codes": _safe_public_reason_codes(data.get("reason_codes"), limit=24),
        }
        return public_payload

    def _runtime_memory_persona_performance_payload(
        *,
        active_config: LocalBrowserConfig,
        browser_media: dict[str, Any] | None = None,
        current_mode: BrowserInteractionMode | str = BrowserInteractionMode.WAITING,
        camera_presence: dict[str, Any] | None = None,
        camera_scene: dict[str, Any] | None = None,
        hidden_counts: dict[str, int] | None = None,
        prefer_cached: bool = True,
    ) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        cached_reader = getattr(runtime, "cached_memory_persona_performance_plan", None)
        if prefer_cached and callable(cached_reader):
            cached_plan = cached_reader()
            if cached_plan is not None:
                return _public_memory_persona_performance_payload(_as_public_dict(cached_plan))
            return _public_memory_persona_performance_payload(
                {
                    "schema_version": 1,
                    "available": False,
                    "profile": active_config.config_profile or "manual",
                    "modality": "browser",
                    "language": active_config.language.value,
                    "tts_label": _browser_tts_runtime_label(active_config),
                    "protected_playback": not active_config.allow_barge_in,
                    "camera_state": "disabled"
                    if not active_config.vision_enabled
                    else "unknown",
                    "continuous_perception_enabled": (
                        active_config.continuous_perception_enabled
                    ),
                    "current_turn_state": str(
                        getattr(current_mode, "value", current_mode) or "waiting"
                    ),
                    "memory_policy": "unavailable",
                    "summary": "Memory/persona performance awaiting a compiled turn plan.",
                    "reason_codes": [
                        "memory_persona_performance:v1",
                        "memory_persona_performance:unavailable",
                        "memory_persona_cached_plan_empty",
                    ],
                }
            )
        reader = getattr(runtime, "current_memory_persona_performance_plan", None)
        if not callable(reader):
            return _public_memory_persona_performance_payload(
                {
                    "schema_version": 1,
                    "available": False,
                    "profile": active_config.config_profile or "manual",
                    "modality": "browser",
                    "language": active_config.language.value,
                    "tts_label": _browser_tts_runtime_label(active_config),
                    "protected_playback": not active_config.allow_barge_in,
                    "camera_state": "disabled"
                    if not active_config.vision_enabled
                    else "unknown",
                    "continuous_perception_enabled": (
                        active_config.continuous_perception_enabled
                    ),
                    "current_turn_state": str(
                        getattr(current_mode, "value", current_mode) or "waiting"
                    ),
                    "memory_policy": "unavailable",
                    "summary": "Memory/persona performance unavailable.",
                    "reason_codes": [
                        "memory_persona_performance:v1",
                        "memory_persona_performance:unavailable",
                        "runtime_not_active",
                    ],
                }
            )
        hidden_counts = hidden_counts or {}
        camera_state = str(
            (camera_scene or {}).get("state")
            or (camera_presence or {}).get("state")
            or (browser_media or {}).get("camera_state")
            or ("disabled" if not active_config.vision_enabled else "unknown")
        )
        active_listening = _public_active_listener_v2_state_payload(active_config)
        conversation_floor = _conversation_floor_state_payload()
        continuity_reader = getattr(runtime, "current_memory_continuity_trace", None)
        memory_continuity_trace = (
            continuity_reader() if callable(continuity_reader) else None
        )
        try:
            actor_control_frame = getattr(
                performance_events,
                "actor_control_latest_frame",
                None,
            )
            plan = reader(
                profile=active_config.config_profile or "manual",
                tts_label=_browser_tts_runtime_label(active_config),
                protected_playback=not active_config.allow_barge_in,
                camera_state=camera_state,
                continuous_perception_enabled=active_config.continuous_perception_enabled,
                current_turn_state=str(getattr(current_mode, "value", current_mode) or "waiting"),
                suppressed_memory_count=_safe_public_int(hidden_counts.get("suppressed")),
                memory_continuity_trace=memory_continuity_trace,
                active_listening=active_listening,
                camera_scene=camera_scene or {},
                floor_state=conversation_floor,
                user_intent={
                    "intent": "answer"
                    if active_listening.get("ready_to_answer") is True
                    else "unknown"
                },
                actor_control_frame=actor_control_frame,
            )
            plan_payload = _as_public_dict(plan)
            if memory_continuity_trace is not None:
                plan_payload["memory_continuity_trace"] = _as_public_dict(
                    memory_continuity_trace
                )
            return _public_memory_persona_performance_payload(plan_payload)
        except Exception as exc:  # pragma: no cover - defensive status surface
            return _public_memory_persona_performance_payload(
                {
                    "schema_version": 1,
                    "available": False,
                    "profile": active_config.config_profile or "manual",
                    "summary": "Memory/persona performance unavailable.",
                    "reason_codes": [
                        "memory_persona_performance:v1",
                        "memory_persona_performance:unavailable",
                        f"memory_persona_performance_error:{type(exc).__name__}",
                    ],
                }
            )

    def _public_memory_persona_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def _public_report_rows(
            value: Any,
            *,
            keys: tuple[str, ...],
            max_items: int = 48,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            if not isinstance(value, list):
                return rows
            for item in value[:max_items]:
                if not isinstance(item, dict):
                    continue
                row: dict[str, Any] = {}
                for key in keys:
                    if key == "fatal":
                        row[key] = item.get(key) is True
                    elif key == "reason_codes":
                        row[key] = _safe_public_reason_codes(item.get(key))
                    else:
                        row[key] = _safe_public_text(item.get(key), limit=180)
                rows.append({key: value for key, value in row.items() if value not in ("", None)})
            return rows

        return {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "available": payload.get("available") is True,
            "accepted": payload.get("accepted") is True,
            "applied": payload.get("applied") is True,
            "preset_id": _safe_public_text(payload.get("preset_id"), limit=96),
            "preset_label": _safe_public_text(payload.get("preset_label"), limit=120),
            "import_id": _safe_public_text(payload.get("import_id"), limit=120),
            "seed_sha256": _safe_public_text(payload.get("seed_sha256"), limit=96),
            "counts": _public_count_mapping(payload.get("counts")),
            "candidates": _public_report_rows(
                payload.get("candidates"),
                keys=("entry_id", "kind", "namespace", "subject", "summary", "reason_codes"),
            ),
            "rejected_entries": _public_report_rows(
                payload.get("rejected_entries"),
                keys=("path", "fatal", "reason_codes"),
            ),
            "applied_entries": _public_report_rows(
                payload.get("applied_entries"),
                keys=("entry_id", "kind", "namespace", "status", "reason_codes"),
            ),
            "behavior_control_result": (
                _public_behavior_controls_payload(payload["behavior_control_result"])
                if isinstance(payload.get("behavior_control_result"), dict)
                else None
            ),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _public_rollout_plan_status(plan: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(plan)
        if not payload:
            return None
        plan_id = _safe_public_text(payload.get("plan_id"), limit=96)
        if not plan_id:
            return None
        return {
            "plan_id": plan_id,
            "adapter_family": _safe_public_text(payload.get("adapter_family"), limit=64),
            "candidate_backend_id": _safe_public_text(payload.get("candidate_backend_id"), limit=96),
            "candidate_backend_version": _safe_public_text(
                payload.get("candidate_backend_version"), limit=64
            ),
            "routing_state": _safe_public_text(payload.get("routing_state"), limit=64),
            "promotion_state": _safe_public_text(payload.get("promotion_state"), limit=64),
            "traffic_fraction": _safe_public_float(payload.get("traffic_fraction")),
            "scope_key": _safe_public_text(payload.get("scope_key"), limit=96),
            "expires_at": _safe_public_text(payload.get("expires_at"), limit=96),
            "embodied_live": payload.get("embodied_live") is True,
            "budget_id": _safe_public_text(payload.get("budget_id"), limit=96),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes"), limit=8),
        }

    def _read_public_rollout_plan(plan_reader: Any, plan_id: str) -> dict[str, Any] | None:
        if not callable(plan_reader):
            return None
        try:
            return _public_rollout_plan_status(plan_reader(plan_id))
        except Exception:  # pragma: no cover - defensive browser payload boundary
            return None

    def _runtime_rollout_action_rejected_payload(
        *,
        plan_id: str,
        action: str,
        reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "accepted": False,
            "applied": False,
            "action": _safe_public_text(action, limit=64),
            "plan_id": _safe_public_text(plan_id, limit=96),
            "from_state": "unavailable",
            "to_state": "unavailable",
            "traffic_fraction": 0.0,
            "before": None,
            "after": None,
            "reason_codes": _safe_public_reason_codes(
                ("rollout_action_rejected", *reason_codes)
            ),
        }

    def _public_rollout_action_payload(
        payload: dict[str, Any],
        *,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "accepted": payload.get("accepted") is True,
            "applied": payload.get("applied") is True,
            "action": _safe_public_text(payload.get("action"), limit=64),
            "plan_id": _safe_public_text(payload.get("plan_id"), limit=96),
            "from_state": _safe_public_text(payload.get("from_state"), limit=64),
            "to_state": _safe_public_text(payload.get("to_state"), limit=64),
            "traffic_fraction": _safe_public_float(payload.get("traffic_fraction")),
            "before": before,
            "after": after,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes"), limit=12),
        }

    def _runtime_rollout_action_payload(
        *,
        plan_id: str,
        action: str,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        runtime = _active_brain_runtime()
        decided_at = _safe_public_text(request_data.get("decided_at"), limit=96)
        if runtime is None:
            return _runtime_rollout_action_rejected_payload(
                plan_id=plan_id,
                action=action,
                reason_codes=("runtime_not_active",),
            )
        controller = getattr(runtime, "live_routing_controller", None) or getattr(
            runtime, "rollout_controller", None
        )
        if controller is None:
            return _runtime_rollout_action_rejected_payload(
                plan_id=plan_id,
                action=action,
                reason_codes=("live_routing_controller_missing",),
            )
        plan_reader = getattr(controller, "plan", None)
        before = _read_public_rollout_plan(plan_reader, plan_id)
        try:
            if action == "approve":
                runner = getattr(controller, "evaluate_plan", None)
                if not callable(runner):
                    return _runtime_rollout_action_rejected_payload(
                        plan_id=plan_id,
                        action=action,
                        reason_codes=("live_routing_action_missing",),
                    )
                result = runner(
                    plan_id,
                    operator_acknowledged=request_data.get("operator_acknowledged"),
                    decided_at=decided_at,
                )
            elif action == "activate":
                runner = getattr(controller, "activate_plan", None)
                if not callable(runner):
                    return _runtime_rollout_action_rejected_payload(
                        plan_id=plan_id,
                        action=action,
                        reason_codes=("live_routing_action_missing",),
                    )
                result = runner(
                    plan_id,
                    traffic_fraction=_safe_public_float(request_data.get("traffic_fraction")),
                    operator_acknowledged=request_data.get("operator_acknowledged"),
                    decided_at=decided_at,
                )
            elif action == "pause":
                runner = getattr(controller, "pause_plan", None)
                if not callable(runner):
                    return _runtime_rollout_action_rejected_payload(
                        plan_id=plan_id,
                        action=action,
                        reason_codes=("live_routing_action_missing",),
                    )
                result = runner(plan_id, decided_at=decided_at)
            elif action == "resume":
                runner = getattr(controller, "resume_plan", None)
                if not callable(runner):
                    return _runtime_rollout_action_rejected_payload(
                        plan_id=plan_id,
                        action=action,
                        reason_codes=("live_routing_action_missing",),
                    )
                result = runner(
                    plan_id,
                    traffic_fraction=_safe_public_float(request_data.get("traffic_fraction")),
                    operator_acknowledged=request_data.get("operator_acknowledged"),
                    decided_at=decided_at,
                )
            elif action == "rollback":
                runner = getattr(controller, "rollback_plan", None)
                if not callable(runner):
                    return _runtime_rollout_action_rejected_payload(
                        plan_id=plan_id,
                        action=action,
                        reason_codes=("live_routing_action_missing",),
                    )
                result = runner(
                    plan_id,
                    regression_codes=tuple(
                        _safe_public_reason_codes(request_data.get("regression_codes"))
                    ),
                    decided_at=decided_at,
                )
            else:
                return _runtime_rollout_action_rejected_payload(
                    plan_id=plan_id,
                    action=action,
                    reason_codes=("rollout_action_unsupported",),
                )
            payload = _as_public_dict(result)
            if not payload:
                return _runtime_rollout_action_rejected_payload(
                    plan_id=plan_id,
                    action=action,
                    reason_codes=("rollout_action_invalid_result",),
                )
            after = _read_public_rollout_plan(plan_reader, plan_id)
            return _public_rollout_action_payload(payload, before=before, after=after)
        except Exception:  # pragma: no cover - defensive browser action boundary
            return _runtime_rollout_action_rejected_payload(
                plan_id=plan_id,
                action=action,
                reason_codes=("rollout_action_error",),
            )

    def _runtime_rollout_evidence_unavailable_payload(*reason_codes: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "available": False,
            "live_episodes": [],
            "practice_plans": [],
            "benchmark_reports": [],
            "reason_codes": list(
                dict.fromkeys(
                    (
                        "rollout_evidence:unavailable",
                        *reason_codes,
                    )
                )
            ),
        }

    def _public_live_episode(record: Any) -> dict[str, Any]:
        return {
            "id": _safe_public_int(getattr(record, "id", 0)),
            "created_at": _safe_public_text(getattr(record, "created_at", ""), limit=96),
            "assistant_summary": _safe_public_text(
                getattr(record, "assistant_summary", ""), limit=160
            ),
        }

    def _public_practice_plan(record: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(record)
        if not payload:
            return None
        plan_id = _safe_public_text(payload.get("plan_id"), limit=96)
        if not plan_id:
            return None
        targets = payload.get("targets")
        return {
            "plan_id": plan_id,
            "target_count": len(targets) if isinstance(targets, (list, tuple)) else 0,
            "summary": _safe_public_text(payload.get("summary"), limit=160),
            "reason_code_counts": _public_reason_code_counts(payload.get("reason_code_counts")),
            "updated_at": _safe_public_text(payload.get("updated_at"), limit=96),
        }

    def _public_benchmark_report(record: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(record)
        if not payload:
            return None
        report_id = _safe_public_text(payload.get("report_id"), limit=96)
        if not report_id:
            return None
        return {
            "report_id": report_id,
            "adapter_family": _safe_public_text(payload.get("adapter_family"), limit=64),
            "candidate_backend_id": _safe_public_text(payload.get("candidate_backend_id"), limit=96),
            "candidate_backend_version": _safe_public_text(
                payload.get("candidate_backend_version"), limit=64
            ),
            "scenario_count": _safe_public_int(payload.get("scenario_count")),
            "compared_family_count": _safe_public_int(payload.get("compared_family_count")),
            "smoke_suite_green": (
                bool(payload["smoke_suite_green"])
                if isinstance(payload.get("smoke_suite_green"), bool)
                else None
            ),
            "benchmark_passed": (
                bool(payload["benchmark_passed"])
                if isinstance(payload.get("benchmark_passed"), bool)
                else None
            ),
            "blocked_reason_codes": _safe_public_reason_codes(
                payload.get("blocked_reason_codes"),
                limit=8,
            ),
            "updated_at": _safe_public_text(payload.get("updated_at"), limit=96),
        }

    def _runtime_rollout_evidence_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_rollout_evidence_unavailable_payload("runtime_not_active")
        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return _runtime_rollout_evidence_unavailable_payload("runtime_evidence_surface_missing")
        try:
            session_ids = session_resolver()
            live_episodes = []
            recent_episodes = getattr(store, "recent_episodes", None)
            if callable(recent_episodes):
                live_episodes = [
                    _public_live_episode(record)
                    for record in recent_episodes(
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        limit=5,
                    )
                ]

            practice_plans = []
            practice_builder = getattr(store, "build_practice_director_projection", None)
            if callable(practice_builder):
                projection = practice_builder(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    agent_id=session_ids.agent_id,
                )
                practice_plans = [
                    plan
                    for record in list(getattr(projection, "recent_plans", []))[:5]
                    if (plan := _public_practice_plan(record)) is not None
                ]

            benchmark_reports = []
            adapter_builder = getattr(store, "build_adapter_governance_projection", None)
            if callable(adapter_builder):
                projection = adapter_builder(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    agent_id=session_ids.agent_id,
                )
                benchmark_reports = [
                    report
                    for record in list(getattr(projection, "recent_reports", []))[:5]
                    if (report := _public_benchmark_report(record)) is not None
                ]

            return {
                "schema_version": 1,
                "available": True,
                "live_episodes": live_episodes,
                "practice_plans": practice_plans,
                "benchmark_reports": benchmark_reports,
                "reason_codes": [
                    "rollout_evidence:available",
                    f"live_episode_count:{len(live_episodes)}",
                    f"practice_plan_count:{len(practice_plans)}",
                    f"benchmark_report_count:{len(benchmark_reports)}",
                ],
            }
        except Exception as exc:  # pragma: no cover - defensive browser evidence boundary
            return _runtime_rollout_evidence_unavailable_payload(
                f"rollout_evidence_error:{type(exc).__name__}"
            )

    def _runtime_episode_evidence_unavailable_payload(*reason_codes: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "available": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": "Episode evidence unavailable.",
            "episode_count": 0,
            "source_counts": {},
            "reason_code_counts": {},
            "rows": [],
            "reason_codes": list(
                dict.fromkeys(
                    (
                        "episode_evidence:unavailable",
                        *reason_codes,
                    )
                )
            ),
        }

    def _public_episode_evidence_artifact(record: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(record)
        if not payload:
            return None
        artifact_id = _safe_public_text(payload.get("artifact_id"), limit=120)
        if not artifact_id:
            return None
        return {
            "artifact_id": artifact_id,
            "artifact_kind": _safe_public_text(payload.get("artifact_kind"), limit=80),
            "uri_kind": _safe_public_text(payload.get("uri_kind"), limit=80),
            "redacted_uri": payload.get("redacted_uri") is True,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _public_episode_evidence_link(record: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(record)
        if not payload:
            return None
        link_kind = _safe_public_text(payload.get("link_kind"), limit=80)
        link_id = _safe_public_text(payload.get("link_id"), limit=120)
        if not link_kind and not link_id:
            return None
        return {
            "link_kind": link_kind,
            "link_id": link_id,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _public_episode_evidence_row(record: Any) -> dict[str, Any] | None:
        payload = _as_public_dict(record)
        if not payload:
            return None
        evidence_id = _safe_public_text(payload.get("evidence_id"), limit=120)
        source = _safe_public_text(payload.get("source"), limit=80)
        if not evidence_id and not source:
            return None
        artifact_refs = [
            artifact
            for item in list(payload.get("artifact_refs") or [])[:8]
            if (artifact := _public_episode_evidence_artifact(item)) is not None
        ]
        links = [
            link
            for item in list(payload.get("links") or [])[:8]
            if (link := _public_episode_evidence_link(item)) is not None
        ]
        return {
            "evidence_id": evidence_id,
            "episode_id": _safe_public_text(payload.get("episode_id"), limit=120),
            "source": source,
            "scenario_id": _safe_public_text(payload.get("scenario_id"), limit=120),
            "scenario_family": _safe_public_text(payload.get("scenario_family"), limit=96),
            "scenario_version": _safe_public_text(payload.get("scenario_version"), limit=96),
            "summary": _safe_public_text(payload.get("summary"), limit=220),
            "source_run_id": _safe_public_text(payload.get("source_run_id"), limit=120),
            "execution_backend": _safe_public_text(payload.get("execution_backend"), limit=96),
            "candidate_backend_id": _safe_public_text(
                payload.get("candidate_backend_id"),
                limit=120,
            ),
            "candidate_backend_version": _safe_public_text(
                payload.get("candidate_backend_version"),
                limit=120,
            ),
            "outcome_label": _safe_public_text(payload.get("outcome_label"), limit=80),
            "task_success": (
                payload.get("task_success")
                if isinstance(payload.get("task_success"), bool)
                else None
            ),
            "safety_success": (
                payload.get("safety_success")
                if isinstance(payload.get("safety_success"), bool)
                else None
            ),
            "preview_only": payload.get("preview_only") is True,
            "scenario_count": _safe_public_int(payload.get("scenario_count")),
            "artifact_refs": artifact_refs,
            "links": links,
            "started_at": _safe_public_text(payload.get("started_at"), limit=96),
            "ended_at": _safe_public_text(payload.get("ended_at"), limit=96),
            "generated_at": _safe_public_text(payload.get("generated_at"), limit=96),
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
            "reason_code_categories": _safe_public_text_list(
                payload.get("reason_code_categories"),
                limit=80,
                max_items=12,
            ),
        }

    def _public_episode_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
        rows = [
            row
            for item in list(payload.get("rows") or [])[:16]
            if (row := _public_episode_evidence_row(item)) is not None
        ]
        return {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "available": payload.get("available") is True,
            "generated_at": _safe_public_text(payload.get("generated_at"), limit=96),
            "summary": _safe_public_text(payload.get("summary"), limit=180)
            or "Episode evidence unavailable.",
            "episode_count": _safe_public_int(payload.get("episode_count")),
            "source_counts": _public_count_mapping(payload.get("source_counts")),
            "reason_code_counts": _public_count_mapping(payload.get("reason_code_counts")),
            "rows": rows,
            "reason_codes": _safe_public_reason_codes(payload.get("reason_codes")),
        }

    def _runtime_episode_evidence_payload() -> dict[str, Any]:
        runtime = _active_brain_runtime()
        if runtime is None:
            return _runtime_episode_evidence_unavailable_payload("runtime_not_active")
        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return _runtime_episode_evidence_unavailable_payload(
                "runtime_evidence_surface_missing"
            )
        try:
            snapshot = build_episode_evidence_index(
                store=store,
                session_ids=session_resolver(),
                presence_scope_key=(
                    _safe_public_text(getattr(runtime, "presence_scope_key", ""), limit=96)
                    or "local:presence"
                ),
                rollout_controller=getattr(runtime, "live_routing_controller", None)
                or getattr(runtime, "rollout_controller", None),
                recent_limit=8,
                generated_at=datetime.now(UTC).isoformat(),
            )
            return _public_episode_evidence_payload(_as_public_dict(snapshot))
        except Exception as exc:  # pragma: no cover - defensive browser evidence boundary
            return _runtime_episode_evidence_unavailable_payload(
                f"episode_evidence_error:{type(exc).__name__}"
            )

    _OPERATOR_SECTION_KEYS = (
        "expression",
        "behavior_controls",
        "teaching_knowledge",
        "voice_metrics",
        "memory",
        "practice",
        "adapters",
        "sim_to_real",
        "rollout_status",
        "episode_evidence",
        "performance_learning",
    )
    _OPERATOR_BLOCKED_KEYS = {
        "artifact_path",
        "artifact_paths",
        "assistant_text",
        "audio",
        "db_path",
        "debug",
        "details",
        "event_id",
        "exception",
        "private_scratchpad",
        "private_working_memory",
        "prompt",
        "prompt_text",
        "raw",
        "raw_json",
        "raw_json_block",
        "raw_transcript",
        "rendered_text",
        "request_payload",
        "response_payload",
        "secret",
        "source_event_id",
        "source_event_ids",
        "source_ref",
        "source_refs",
        "stack_trace",
        "system_message",
        "system_prompt",
        "tool_calls_json",
        "traceback",
        "transcript",
        "user_text",
        "user_text_preview",
    }
    _OPERATOR_BLOCKED_KEY_MARKERS = (
        "api_key",
        "authorization",
        "bearer",
        "database_path",
        "db_path",
        "hidden_prompt",
        "private_",
        "prompt_text",
        "raw_",
        "request_payload",
        "response_payload",
        "secret",
        "source_event",
        "source_ref",
        "stack_trace",
        "system_prompt",
        "traceback",
        "transcript",
        "user_text",
    )

    def _operator_key_blocked(key: Any) -> bool:
        normalized = str(key or "").strip().lower().replace("-", "_")
        if normalized in _OPERATOR_BLOCKED_KEYS:
            return True
        return any(marker in normalized for marker in _OPERATOR_BLOCKED_KEY_MARKERS)

    def _operator_public_value(value: Any, *, depth: int = 0) -> Any:
        if depth > 5:
            return None
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return _safe_public_int(value)
        if isinstance(value, float):
            return _safe_public_float(value)
        if isinstance(value, str):
            return _safe_public_text(value, limit=180)
        if isinstance(value, (list, tuple, set)):
            rows = [
                item
                for raw_item in list(value)[:24]
                if (item := _operator_public_value(raw_item, depth=depth + 1))
                not in (None, "", [], {})
            ]
            return rows
        payload = _as_public_dict(value)
        if payload:
            rows: dict[str, Any] = {}
            for raw_key, raw_value in list(payload.items())[:48]:
                if _operator_key_blocked(raw_key):
                    continue
                public_key = _safe_public_text(raw_key, limit=80)
                if not public_key or public_key == "redacted":
                    continue
                public_value = _operator_public_value(raw_value, depth=depth + 1)
                if public_value in (None, "", [], {}):
                    continue
                rows[public_key] = public_value
            return rows
        return None

    def _public_operator_payload_for_section(
        section_key: str,
        raw_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if section_key == "expression":
            return _public_expression_payload(raw_payload)
        if section_key == "behavior_controls":
            return _public_behavior_controls_payload(raw_payload)
        if section_key == "voice_metrics":
            return _public_voice_metrics_payload(raw_payload)
        if section_key == "memory":
            return _public_memory_payload(raw_payload)
        if section_key == "episode_evidence":
            return _public_episode_evidence_payload(raw_payload)
        generic_payload = _operator_public_value(raw_payload)
        return generic_payload if isinstance(generic_payload, dict) else {}

    def _public_operator_section_payload(section_key: str, value: Any) -> dict[str, Any]:
        section = _as_public_dict(value)
        raw_payload = _as_public_dict(section.get("payload"))
        public_payload = _public_operator_payload_for_section(section_key, raw_payload)
        available = section.get("available") is True
        section_state_codes = {
            f"operator_{section_key}:available",
            f"operator_{section_key}:unavailable",
        }
        raw_reason_codes = [
            code
            for code in _safe_public_reason_codes(section.get("reason_codes"))
            if code not in section_state_codes
        ]
        return {
            "available": available,
            "summary": (
                _safe_public_text(section.get("summary"), limit=180)
                or f"{section_key.replace('_', ' ').title()} unavailable."
            ),
            "payload": public_payload,
            "reason_codes": _safe_public_reason_codes(
                (
                    f"operator_{section_key}:available"
                    if available
                    else f"operator_{section_key}:unavailable",
                    *raw_reason_codes,
                ),
            ),
        }

    def _public_operator_payload(payload: dict[str, Any]) -> dict[str, Any]:
        sections = {
            key: _public_operator_section_payload(key, payload.get(key))
            for key in _OPERATOR_SECTION_KEYS
        }
        explicit_available = payload.get("available")
        available = (
            explicit_available
            if isinstance(explicit_available, bool)
            else any(section["available"] for section in sections.values())
        )
        raw_reason_codes = [
            code
            for code in _safe_public_reason_codes(payload.get("reason_codes"))
            if code not in {"operator_workbench:available", "operator_workbench:unavailable"}
        ]
        return {
            "schema_version": _safe_public_int(payload.get("schema_version"), 1),
            "available": available,
            **sections,
            "reason_codes": _safe_public_reason_codes(
                (
                    "operator_workbench:v1",
                    "operator_workbench:available"
                    if available
                    else "operator_workbench:unavailable",
                    *raw_reason_codes,
                ),
            ),
        }

    def _runtime_operator_payload() -> dict[str, Any]:
        def unavailable_payload(*reason_codes: str) -> dict[str, Any]:
            sections = {
                key: {
                    "available": False,
                    "summary": f"{key.replace('_', ' ').title()} unavailable.",
                    "payload": {},
                    "reason_codes": [f"operator_{key}:unavailable", *reason_codes],
                }
                for key in _OPERATOR_SECTION_KEYS
            }
            return {
                "schema_version": 1,
                "available": False,
                **sections,
                "reason_codes": list(
                    dict.fromkeys(
                        (
                            "operator_workbench:v1",
                            "operator_workbench:unavailable",
                            *reason_codes,
                        )
                    )
                ),
            }

        try:
            return _public_operator_payload(
                _as_public_dict(build_operator_workbench_snapshot(_active_brain_runtime()))
            )
        except Exception:  # pragma: no cover - defensive aggregate endpoint boundary
            return _public_operator_payload(unavailable_payload("runtime_operator_error"))

    def _runtime_stack_payload() -> dict[str, Any]:
        runtime_active = _active_brain_runtime() is not None
        active_config = _active_browser_config()
        browser_media = _current_client_media_payload()
        current_profile_id = local_llm_model_profile_id_for(
            provider=active_config.llm_provider,
            model=active_config.model,
        )
        reason_codes = [
            "runtime_stack:v1",
            "runtime_stack:configured",
            f"llm_provider:{active_config.llm_provider}",
            f"stt_backend:{active_config.stt_backend}",
            f"tts_backend:{active_config.tts_backend}",
            "demo_mode:enabled" if active_config.demo_mode else "demo_mode:disabled",
            "runtime_active:true" if runtime_active else "runtime_active:false",
        ]
        if current_profile_id:
            reason_codes.append(f"model_profile:{current_profile_id}")
        reason_codes.extend(browser_media["reason_codes"])
        if active_config.vision_enabled:
            reason_codes.append("vision:enabled")
        else:
            reason_codes.append("vision:disabled")

        payload: dict[str, Any] = {
            "schema_version": 1,
            "available": True,
            "runtime_active": runtime_active,
            "llm_provider": _safe_public_text(active_config.llm_provider, limit=80),
            "model": _safe_public_text(active_config.model, limit=120),
            "configured_llm_provider": _safe_public_text(config.llm_provider, limit=80),
            "configured_model": _safe_public_text(config.model, limit=120),
            "model_profile_id": (
                _safe_public_text(current_profile_id, limit=96)
                if current_profile_id
                else None
            ),
            "stt_backend": _safe_public_text(active_config.stt_backend, limit=80),
            "stt_model": _safe_public_text(active_config.stt_model, limit=160),
            "tts_backend": _safe_public_text(active_config.tts_backend, limit=80),
            "tts_voice": _safe_public_text(active_config.tts_voice, limit=120) or None,
            "demo_mode": bool(active_config.demo_mode),
            "vision_enabled": bool(active_config.vision_enabled),
            "continuous_perception_enabled": bool(
                active_config.continuous_perception_enabled
            ),
            "browser_media": browser_media,
            "reason_codes": _safe_public_reason_codes(reason_codes),
        }
        if active_config.llm_service_tier not in (None, ""):
            payload["service_tier"] = _safe_public_text(
                active_config.llm_service_tier,
                limit=80,
            )
        if active_config.llm_max_output_tokens is not None:
            payload["max_output_tokens"] = _safe_public_int(
                active_config.llm_max_output_tokens
            )
        return payload

    async def _read_optional_action_json(request: Request) -> dict[str, Any]:
        try:
            return await _read_json_object(request)
        except ValueError:
            return {}

    async def handle_offer_request(
        request_data: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> dict[str, str] | None:
        connection_config = active_session_configs.get(session_id or "", config)
        request = SmallWebRTCRequest.from_dict(dict(request_data))

        async def webrtc_connection_callback(connection: SmallWebRTCConnection):
            active_connections[connection.pc_id] = connection
            active_connection_configs[connection.pc_id] = connection_config
            app.state.blink_browser_active_session_id = session_id or connection.pc_id
            _emit_performance_event(
                event_type="webrtc.connection_created",
                source="webrtc",
                mode=BrowserInteractionMode.CONNECTED,
                session_id=session_id or connection.pc_id,
                metadata={"connection_count": len(active_connections)},
                reason_codes=("webrtc:connection_created",),
            )
            _clear_runtime_payload_cache()

            @connection.event_handler("closed")
            async def handle_closed(webrtc_connection: SmallWebRTCConnection):
                active_connections.pop(webrtc_connection.pc_id, None)
                active_connection_configs.pop(webrtc_connection.pc_id, None)
                if not active_connections:
                    app.state.blink_browser_active_session_id = None
                    app.state.blink_browser_active_client_id = None
                _emit_performance_event(
                    event_type="webrtc.connection_closed",
                    source="webrtc",
                    mode=_resting_interaction_mode(),
                    session_id=session_id or webrtc_connection.pc_id,
                    metadata={"connection_count": len(active_connections)},
                    reason_codes=("webrtc:connection_closed",),
                )
                _clear_runtime_payload_cache()

            asyncio.create_task(run_connection(connection, connection_config))

        answer = await small_webrtc_handler.handle_web_request(
            request=request,
            webrtc_connection_callback=webrtc_connection_callback,
        )

        if answer:
            connection = small_webrtc_handler._pcs_map.get(answer["pc_id"])
            if connection is not None:
                active_connections[answer["pc_id"]] = connection
                active_connection_configs[answer["pc_id"]] = connection_config
                _clear_runtime_payload_cache()

        return answer

    async def handle_ice_candidate_request(request_data: dict[str, Any]) -> dict[str, str]:
        candidate_payload = BrowserPatchPayload(**request_data)
        pc_id = candidate_payload.get("pc_id") or candidate_payload.get("pcId")
        if not pc_id:
            raise ValueError("Missing peer connection id.")

        candidates: list[IceCandidate] = []
        for candidate in candidate_payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            candidate_text = str(candidate.get("candidate") or "").strip()
            if not candidate_text:
                continue
            if candidate_text.startswith("candidate:") and len(candidate_text.split()) < 8:
                continue
            if not candidate_text.startswith("candidate:"):
                continue
            candidates.append(
                IceCandidate(
                    candidate=candidate_text,
                    sdp_mid=candidate.get("sdp_mid") or candidate.get("sdpMid") or "",
                    sdp_mline_index=(
                        candidate.get("sdp_mline_index")
                        if "sdp_mline_index" in candidate
                        else candidate.get("sdpMLineIndex")
                    )
                    or 0,
                )
            )
        if not candidates:
            return {"status": "success"}

        patch_request = SmallWebRTCPatchRequest(
            pc_id=pc_id,
            candidates=candidates,
        )
        try:
            await small_webrtc_handler.handle_patch_request(patch_request)
        except AssertionError:
            return {"status": "success"}
        return {"status": "success"}

    async def run_connection(
        webrtc_connection: SmallWebRTCConnection,
        runtime_config: LocalBrowserConfig,
    ):
        app.state.blink_active_browser_config = runtime_config
        session_floor_controller = ConversationFloorController(
            profile=runtime_config.config_profile or "manual",
            language=runtime_config.language.value,
            protected_playback=not runtime_config.allow_barge_in,
            barge_in_armed=runtime_config.allow_barge_in,
            echo_safe=runtime_config.allow_barge_in,
        )
        app.state.blink_conversation_floor = session_floor_controller
        app.state.blink_conversation_floor_state = session_floor_controller.snapshot()
        _clear_runtime_payload_cache()
        active_client: dict[str, Any] = {"id": None, "camera_enabled": None}
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_in_enabled=runtime_config.vision_enabled,
            ),
        )

        async def run_task(tts_session=None):
            task, context = runtime_builder(
                runtime_config,
                transport=transport,
                active_client=active_client,
                tts_session=tts_session,
                vision=shared_vision,
                performance_emit=_emit_performance_event,
                performance_resting_mode_provider=_resting_interaction_mode,
                floor_input_emit=_apply_conversation_floor_input,
                interruption_state=getattr(app.state, "blink_browser_interruption", None),
                actor_control_scheduler=getattr(
                    app.state,
                    "blink_actor_control_scheduler",
                    None,
                ),
                actor_control_frame_provider=lambda: performance_events.actor_control_latest_frame,
                runtime_session=_runtime_session_v3(),
            )
            brain_runtime = getattr(context, "blink_brain_runtime", None)
            camera_frame_buffer = getattr(context, "blink_camera_frame_buffer", None)
            camera_health_manager = getattr(context, "blink_camera_health_manager", None)
            perception_broker = getattr(context, "blink_perception_broker", None)
            if brain_runtime is not None:
                active_brain_runtimes[webrtc_connection.pc_id] = brain_runtime
                app.state.blink_active_expression_runtime = brain_runtime
                app.state.blink_active_llm_context = context
                app.state.blink_active_camera_context_state = None
                _clear_runtime_payload_cache()
            app.state.blink_active_camera_frame_buffer = camera_frame_buffer
            app.state.blink_active_camera_health_manager = camera_health_manager

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, client):
                active_client["id"] = get_transport_client_id(transport, client)
                app.state.blink_browser_active_client_id = active_client["id"]
                app.state.blink_browser_active_session_id = webrtc_connection.pc_id
                _runtime_session_v3().note_client_connected(
                    session_id=webrtc_connection.pc_id,
                    client_id=active_client["id"],
                )
                client_media = _current_client_media_payload()
                _runtime_session_v3().note_client_media(client_media)
                camera_unavailable = client_media["mode"] == "audio_only" or client_media[
                    "camera_state"
                ] in {
                    "unavailable",
                    "permission_denied",
                    "device_not_found",
                    "error",
                }
                hard_camera_unavailable = client_media["camera_state"] in {
                    "permission_denied",
                    "device_not_found",
                    "error",
                }
                active_client["camera_enabled"] = not camera_unavailable
                note_audio_connected = getattr(brain_runtime, "note_voice_input_connected", None)
                if callable(note_audio_connected):
                    note_audio_connected(True)
                _emit_performance_event(
                    event_type="webrtc.client_connected",
                    source="webrtc",
                    mode=_resting_interaction_mode(),
                    session_id=webrtc_connection.pc_id,
                    client_id=active_client["id"],
                    metadata={
                        "media_mode": client_media["mode"],
                        "camera_state": client_media["camera_state"],
                        "microphone_state": client_media["microphone_state"],
                    },
                    reason_codes=("webrtc:client_connected", *client_media["reason_codes"]),
                )
                if not runtime_config.vision_enabled:
                    return

                if camera_unavailable:
                    _emit_performance_event(
                        event_type="camera.disconnected",
                        source="webrtc",
                        mode=BrowserInteractionMode.ERROR,
                        session_id=webrtc_connection.pc_id,
                        client_id=active_client["id"],
                        metadata={"camera_state": client_media["camera_state"]},
                        reason_codes=("camera:unavailable", *client_media["reason_codes"]),
                    )
                    _sync_runtime_camera_context_from_media(
                        browser_media=client_media,
                        active_config=runtime_config,
                        source="webrtc",
                    )
                    if hard_camera_unavailable:
                        return

                if not camera_unavailable:
                    _safe_note_vision_connected(brain_runtime, True)
                if camera_health_manager is not None:
                    await camera_health_manager.handle_client_connected()
                    await camera_health_manager.start()
                    health = camera_health_manager.current_health()
                    if not camera_unavailable:
                        _emit_performance_event(
                            event_type="camera.connected",
                            source="webrtc",
                            mode=_resting_interaction_mode(),
                            session_id=webrtc_connection.pc_id,
                            client_id=active_client["id"],
                            metadata={
                                "camera_state": client_media["camera_state"],
                                "track_state": health.camera_track_state,
                                "frame_age_ms": health.frame_age_ms,
                                "scene_transition": "camera_ready",
                            },
                            reason_codes=(
                                "camera:connected",
                                "scene_social_transition:camera_ready",
                                *(code for code in (health.sensor_health_reason,) if code),
                            ),
                        )
                if perception_broker is not None:
                    await perception_broker.start()
                if not camera_unavailable:
                    _sync_runtime_camera_context_from_media(
                        browser_media=client_media,
                        active_config=runtime_config,
                        source="webrtc",
                    )
                await maybe_capture_participant_camera(transport, client, framerate=1)

            @transport.event_handler("on_video_track_stalled")
            async def on_video_track_stalled(transport, event):
                _emit_performance_event(
                    event_type="camera.track_stalled",
                    source="webrtc",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={
                        "track_enabled": getattr(event, "enabled", None),
                        "scene_transition": "vision_stale",
                    },
                    reason_codes=(
                        "camera:track_stalled",
                        "scene_social_transition:vision_stale",
                    ),
                )
                if camera_health_manager is not None:
                    try:
                        await camera_health_manager.handle_video_track_stalled(event)
                    except Exception as exc:
                        logger.warning(
                            "Suppressed camera health video-stall update failure: {}",
                            type(exc).__name__,
                        )

            @transport.event_handler("on_video_track_resumed")
            async def on_video_track_resumed(transport, event):
                _emit_performance_event(
                    event_type="camera.track_resumed",
                    source="webrtc",
                    mode=_resting_interaction_mode(),
                    metadata={
                        "track_enabled": getattr(event, "enabled", None),
                        "scene_transition": "camera_ready",
                    },
                    reason_codes=(
                        "camera:track_resumed",
                        "scene_social_transition:camera_ready",
                    ),
                )
                if camera_health_manager is not None:
                    try:
                        await camera_health_manager.handle_video_track_resumed(event)
                    except Exception as exc:
                        logger.warning(
                            "Suppressed camera health video-resume update failure: {}",
                            type(exc).__name__,
                        )

            @transport.event_handler("on_audio_track_stalled")
            async def on_audio_track_stalled(transport, event):
                note_audio_stalled = getattr(brain_runtime, "note_voice_input_track_stalled", None)
                if callable(note_audio_stalled):
                    note_audio_stalled(event)
                _emit_performance_event(
                    event_type="microphone.track_stalled",
                    source="webrtc",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={"track_enabled": getattr(event, "enabled", None)},
                    reason_codes=("microphone:track_stalled",),
                )
                _emit_performance_event(
                    event_type="active_listening.listening_degraded",
                    source="active_listening",
                    mode=BrowserInteractionMode.ERROR,
                    metadata={
                        "ready_to_answer": False,
                        "readiness_state": "degraded",
                        "degradation_state": "degraded",
                        "degraded_component_count": 1,
                    },
                    reason_codes=(
                        "active_listener:listening_degraded",
                        "microphone:track_stalled",
                    ),
                )
                if camera_health_manager is not None:
                    try:
                        await camera_health_manager.handle_audio_track_stalled(event)
                    except Exception as exc:
                        logger.warning(
                            "Suppressed camera health audio-stall update failure: {}",
                            type(exc).__name__,
                        )

            @transport.event_handler("on_audio_track_resumed")
            async def on_audio_track_resumed(transport, event):
                note_audio_resumed = getattr(brain_runtime, "note_voice_input_track_resumed", None)
                if callable(note_audio_resumed):
                    note_audio_resumed(event)
                _emit_performance_event(
                    event_type="microphone.track_resumed",
                    source="webrtc",
                    mode=_resting_interaction_mode(),
                    metadata={"track_enabled": getattr(event, "enabled", None)},
                    reason_codes=("microphone:track_resumed",),
                )
                if camera_health_manager is not None:
                    try:
                        await camera_health_manager.handle_audio_track_resumed(event)
                    except Exception as exc:
                        logger.warning(
                            "Suppressed camera health audio-resume update failure: {}",
                            type(exc).__name__,
                        )

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                disconnected_client_id = active_client.get("id") or get_transport_client_id(
                    transport, client
                )
                owns_active_state = _browser_connection_owns_active_state(
                    app.state,
                    session_id=webrtc_connection.pc_id,
                    client_id=disconnected_client_id,
                )
                _safe_note_vision_connected(brain_runtime, False)
                note_audio_connected = getattr(brain_runtime, "note_voice_input_connected", None)
                if callable(note_audio_connected):
                    note_audio_connected(False)
                active_client["id"] = None
                active_client["camera_enabled"] = None
                if owns_active_state:
                    app.state.blink_browser_active_session_id = None
                    app.state.blink_browser_active_client_id = None
                    _runtime_session_v3().note_client_disconnected()
                    client_media = _reset_client_media_state(
                        "browser_media:unreported",
                        "browser_client:disconnected",
                    )
                else:
                    client_media = _unreported_client_media_payload(
                        "browser_media:unreported",
                        "browser_client:stale_disconnected",
                    )
                _emit_performance_event(
                    event_type="webrtc.client_disconnected",
                    source="webrtc",
                    mode=_resting_interaction_mode(),
                    session_id=webrtc_connection.pc_id,
                    client_id=disconnected_client_id,
                    metadata={
                        "media_mode": client_media["mode"],
                        "camera_state": client_media["camera_state"],
                        "microphone_state": client_media["microphone_state"],
                        "cleared_active_state": owns_active_state,
                    },
                    reason_codes=(
                        "webrtc:client_disconnected",
                        *client_media["reason_codes"],
                    ),
                )
                if camera_health_manager is not None:
                    await _run_browser_disconnect_cleanup_step(
                        "camera_health_disconnected",
                        camera_health_manager.handle_client_disconnected,
                    )
                    await _run_browser_disconnect_cleanup_step(
                        "camera_health_close",
                        camera_health_manager.close,
                    )
                if perception_broker is not None:
                    await _run_browser_disconnect_cleanup_step(
                        "perception_broker_close",
                        perception_broker.close,
                    )
                await _run_browser_disconnect_cleanup_step("task_cancel", task.cancel)

            runner = PipelineRunner(handle_sigint=False)
            try:
                await runner.run(task)
            finally:
                active_brain_runtimes.pop(webrtc_connection.pc_id, None)
                active_connection_configs.pop(webrtc_connection.pc_id, None)
                if (
                    getattr(app.state, "blink_active_camera_frame_buffer", None)
                    is camera_frame_buffer
                ):
                    app.state.blink_active_camera_frame_buffer = None
                if (
                    getattr(app.state, "blink_active_camera_health_manager", None)
                    is camera_health_manager
                ):
                    app.state.blink_active_camera_health_manager = None
                if getattr(app.state, "blink_active_expression_runtime", None) is brain_runtime:
                    app.state.blink_active_expression_runtime = None
                if getattr(app.state, "blink_active_llm_context", None) is context:
                    app.state.blink_active_llm_context = None
                    app.state.blink_active_camera_context_state = None
                if getattr(app.state, "blink_active_browser_config", None) is runtime_config:
                    app.state.blink_active_browser_config = next(
                        iter(active_connection_configs.values()), None
                    )
                if not active_connection_configs:
                    app.state.blink_browser_active_session_id = None
                    app.state.blink_browser_active_client_id = None
                    _reset_client_media_state(
                        "browser_media:unreported",
                        "runtime_task:finished",
                    )
                _emit_performance_event(
                    event_type="runtime.task_finished",
                    source="runtime",
                    mode=_resting_interaction_mode(),
                    session_id=webrtc_connection.pc_id,
                    reason_codes=("runtime:task_finished",),
                )
                _clear_runtime_payload_cache()
                if brain_runtime is not None:
                    try:
                        brain_runtime.close()
                    except Exception as exc:  # pragma: no cover - defensive cleanup boundary
                        logger.warning(
                            "Suppressed browser brain runtime close failure: {}",
                            type(exc).__name__,
                        )
                if camera_health_manager is not None:
                    await camera_health_manager.close()
                if perception_broker is not None:
                    await perception_broker.close()

        if tts_backend_uses_external_service(runtime_config.tts_backend):
            import aiohttp

            async with aiohttp.ClientSession() as tts_session:
                await run_task(tts_session=tts_session)
        else:
            await run_task()

    app.add_api_route(
        "/", create_smallwebrtc_root_redirect(), methods=["GET"], include_in_schema=False
    )

    @app.post("/api/offer")
    async def offer(request: Request):
        try:
            request_data = await _read_json_object(request)
        except ValueError:
            return Response(content="Invalid WebRTC request", status_code=400)
        return await handle_offer_request(request_data)

    @app.patch("/api/offer")
    async def ice_candidate(request: Request):
        try:
            request_data = await _read_json_object(request)
            return await handle_ice_candidate_request(request_data)
        except ValueError:
            return Response(content="Invalid WebRTC request", status_code=400)

    @app.get("/api/runtime/expression")
    async def runtime_expression():
        return _runtime_expression_payload()

    @app.get("/api/runtime/voice-metrics")
    async def runtime_voice_metrics():
        return _runtime_voice_metrics_payload()

    @app.get("/api/runtime/behavior-controls")
    async def runtime_behavior_controls():
        return _runtime_behavior_controls_payload()

    @app.post("/api/runtime/behavior-controls")
    async def runtime_behavior_controls_update(request: Request):
        try:
            request_data = await _read_json_object(request)
        except ValueError:
            request_data = {}
        payload = _runtime_behavior_controls_update_payload(request_data)
        _clear_runtime_payload_cache()
        return payload

    @app.post("/api/runtime/performance-preferences")
    async def runtime_performance_preferences_record(request: Request):
        try:
            request_data = await _read_json_object(request)
        except ValueError:
            request_data = {}
        return _runtime_mutation_payload(
            _runtime_performance_preference_record_payload(request_data)
        )

    @app.post("/api/runtime/performance-preferences/policy-proposals/{proposal_id}/apply")
    async def runtime_performance_learning_policy_apply(proposal_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_performance_learning_apply_payload(
                proposal_id=proposal_id,
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.get("/api/runtime/style-presets")
    async def runtime_style_presets():
        return _cached_runtime_payload(
            "style_presets",
            ttl_secs=300.0,
            builder=_runtime_style_presets_payload,
        )

    @app.get("/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview")
    async def runtime_memory_persona_preset_preview():
        return _cached_runtime_payload(
            f"memory_persona_preset_preview:{_runtime_cache_scope()}",
            ttl_secs=300.0,
            builder=_runtime_memory_persona_preset_preview_payload,
        )

    @app.post("/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply")
    async def runtime_memory_persona_preset_apply(request: Request):
        try:
            request_data = await _read_json_object(request)
        except ValueError:
            request_data = {}
        payload = _runtime_memory_persona_preset_apply_payload(request_data)
        _clear_runtime_payload_cache()
        return payload

    @app.get("/api/runtime/memory")
    async def runtime_memory():
        return await asyncio.to_thread(
            _cached_runtime_payload,
            f"memory:{_runtime_cache_scope()}",
            ttl_secs=1.25,
            builder=_runtime_memory_payload,
        )

    @app.get("/api/runtime/stack")
    async def runtime_stack():
        return _runtime_stack_payload()

    @app.get("/api/runtime/client-config.js", include_in_schema=False)
    async def runtime_client_config_js():
        payload = _browser_client_config_payload()
        script = (
            "globalThis.BlinkRuntimeConfig = Object.freeze("
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            ");\n"
        )
        return Response(
            content=script,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/runtime/performance-state")
    async def runtime_performance_state():
        return _browser_performance_state_payload()

    @app.get("/api/runtime/actor-state")
    async def runtime_actor_state():
        return _browser_actor_state_payload()

    @app.get("/api/runtime/performance-events")
    async def runtime_performance_events(after_id: int = 0, limit: int = 50):
        return performance_events.as_payload(
            after_id=max(0, after_id),
            limit=min(max(0, limit), 200),
        )

    @app.get("/api/runtime/actor-events")
    async def runtime_actor_events(after_id: int = 0, limit: int = 50):
        return performance_events.actor_payload(
            after_id=max(0, after_id),
            limit=min(max(0, limit), 200),
        )

    @app.get("/api/runtime/client-media")
    async def runtime_client_media():
        return _current_client_media_payload()

    @app.post("/api/runtime/client-media")
    async def runtime_client_media_update(request: Request):
        try:
            request_data = await _read_json_object(request)
        except ValueError:
            request_data = {}
        return _runtime_mutation_payload(_runtime_client_media_update_payload(request_data))

    @app.get("/api/runtime/models")
    async def runtime_models():
        return await _runtime_model_catalog_payload()

    @app.get("/api/runtime/operator")
    async def runtime_operator():
        return await asyncio.to_thread(
            _cached_runtime_payload,
            f"operator:{_runtime_cache_scope()}",
            ttl_secs=1.25,
            builder=_runtime_operator_payload,
        )

    @app.get("/api/runtime/rollout/evidence")
    async def runtime_rollout_evidence():
        return _cached_runtime_payload(
            f"rollout_evidence:{_runtime_cache_scope()}",
            ttl_secs=10.0,
            builder=_runtime_rollout_evidence_payload,
        )

    @app.get("/api/runtime/evidence")
    async def runtime_episode_evidence():
        return _cached_runtime_payload(
            f"episode_evidence:{_runtime_cache_scope()}",
            ttl_secs=10.0,
            builder=_runtime_episode_evidence_payload,
        )

    @app.post("/api/runtime/rollout/{plan_id}/approve")
    async def runtime_rollout_approve(plan_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_rollout_action_payload(
                plan_id=plan_id,
                action="approve",
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.post("/api/runtime/rollout/{plan_id}/activate")
    async def runtime_rollout_activate(plan_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_rollout_action_payload(
                plan_id=plan_id,
                action="activate",
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.post("/api/runtime/rollout/{plan_id}/pause")
    async def runtime_rollout_pause(plan_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_rollout_action_payload(
                plan_id=plan_id,
                action="pause",
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.post("/api/runtime/rollout/{plan_id}/resume")
    async def runtime_rollout_resume(plan_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_rollout_action_payload(
                plan_id=plan_id,
                action="resume",
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.post("/api/runtime/rollout/{plan_id}/rollback")
    async def runtime_rollout_rollback(plan_id: str, request: Request):
        return _runtime_mutation_payload(
            _runtime_rollout_action_payload(
                plan_id=plan_id,
                action="rollback",
                request_data=await _read_optional_action_json(request),
            )
        )

    @app.post("/api/runtime/memory/{memory_id}/pin")
    async def runtime_memory_pin(memory_id: str, request: Request):
        payload = _runtime_memory_action_payload(
            memory_id=memory_id,
            action="pin",
            request_data=await _read_optional_action_json(request),
        )
        _emit_memory_action_performance_event(payload)
        return _runtime_mutation_payload(payload)

    @app.post("/api/runtime/memory/{memory_id}/suppress")
    async def runtime_memory_suppress(memory_id: str, request: Request):
        payload = _runtime_memory_action_payload(
            memory_id=memory_id,
            action="suppress",
            request_data=await _read_optional_action_json(request),
        )
        _emit_memory_action_performance_event(payload)
        return _runtime_mutation_payload(payload)

    @app.post("/api/runtime/memory/{memory_id}/correct")
    async def runtime_memory_correct(memory_id: str, request: Request):
        payload = _runtime_memory_action_payload(
            memory_id=memory_id,
            action="correct",
            request_data=await _read_optional_action_json(request),
        )
        _emit_memory_action_performance_event(payload)
        return _runtime_mutation_payload(payload)

    @app.post("/api/runtime/memory/{memory_id}/forget")
    async def runtime_memory_forget(memory_id: str, request: Request):
        payload = _runtime_memory_action_payload(
            memory_id=memory_id,
            action="forget",
            request_data=await _read_optional_action_json(request),
        )
        _emit_memory_action_performance_event(payload)
        return _runtime_mutation_payload(payload)

    @app.post("/api/runtime/memory/{memory_id}/mark-stale")
    async def runtime_memory_mark_stale(memory_id: str, request: Request):
        payload = _runtime_memory_action_payload(
            memory_id=memory_id,
            action="mark_stale",
            request_data=await _read_optional_action_json(request),
        )
        _emit_memory_action_performance_event(payload)
        return _runtime_mutation_payload(payload)

    @app.post("/start")
    async def start(request: Request):
        try:
            request_data = BrowserStartPayload(**await _read_json_object(request))
        except ValueError:
            return Response(content="Invalid start request", status_code=400)

        body = request_data.get("body") or {}
        if not isinstance(body, dict):
            return Response(content="Invalid start request", status_code=400)

        session_config = config
        requested_profile_id = _requested_model_profile_id(body)
        model_selection = _model_selection_result(
            accepted=True,
            applied=False,
            profile=None,
            requested_profile_id=None,
            reason_codes=("model_selection:default",),
        )
        if requested_profile_id is not None:
            profiles = load_local_llm_model_profiles()
            profile = local_llm_model_profile_by_id(requested_profile_id, profiles=profiles)
            if profile is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "modelSelection": _model_selection_result(
                            accepted=False,
                            applied=False,
                            profile=None,
                            requested_profile_id=requested_profile_id,
                            reason_codes=("model_profile_unknown",),
                        )
                    },
                )
            if profile.provider == "openai-responses" and not remote_model_selection_enabled(
                current_provider=config.llm_provider
            ):
                return JSONResponse(
                    status_code=400,
                    content={
                        "modelSelection": _model_selection_result(
                            accepted=False,
                            applied=False,
                            profile=profile,
                            requested_profile_id=requested_profile_id,
                            reason_codes=(
                                "model_profile_unavailable",
                                "remote_model_selection_disabled",
                            ),
                        )
                    },
                )
            session_config = _browser_config_for_model_profile(config, profile)
            try:
                await verify_local_llm_config(session_config.llm)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={
                        "modelSelection": _model_selection_result(
                            accepted=False,
                            applied=False,
                            profile=profile,
                            requested_profile_id=requested_profile_id,
                            reason_codes=_safe_model_selection_rejection_reason(
                                provider=profile.provider,
                                error=exc,
                            ),
                        )
                    },
                )
            model_selection = _model_selection_result(
                accepted=True,
                applied=True,
                profile=profile,
                requested_profile_id=requested_profile_id,
                reason_codes=("model_profile_selected",),
            )

        session_id = str(uuid4())
        active_sessions[session_id] = body
        active_session_configs[session_id] = session_config
        app.state.blink_browser_active_session_id = session_id
        _emit_performance_event(
            event_type="browser_session.created",
            source="runtime",
            mode=BrowserInteractionMode.WAITING,
            session_id=session_id,
            metadata={
                "profile": session_config.config_profile or "manual",
                "tts_backend": session_config.tts_backend,
                "vision_enabled": session_config.vision_enabled,
            },
            reason_codes=("browser_session:created",),
        )
        _clear_runtime_payload_cache()

        result: BrowserStartResponse = {
            "sessionId": session_id,
            "modelSelection": model_selection,
        }
        if request_data.get("enableDefaultIceServers"):
            result["iceConfig"] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

        return result

    @app.api_route("/sessions/{session_id}/api/offer", methods=["POST", "PATCH"])
    async def session_offer(session_id: str, request: Request):
        active_session = active_sessions.get(session_id)
        if active_session is None:
            return Response(content="Invalid or not-yet-ready session_id", status_code=404)

        try:
            request_data = await _read_json_object(request)
        except ValueError:
            return Response(content="Invalid WebRTC request", status_code=400)

        if request.method == "POST":
            request_data = {
                **request_data,
                "request_data": request_data.get("request_data")
                or request_data.get("requestData")
                or active_session,
            }
            return await handle_offer_request(request_data, session_id=session_id)

        try:
            return await handle_ice_candidate_request(request_data)
        except ValueError:
            return Response(content="Invalid WebRTC request", status_code=400)

    return app, uvicorn


async def run_local_browser(config: LocalBrowserConfig) -> int:
    """Run the local browser/WebRTC workflow."""
    configure_logging(config.verbose)
    if _browser_port_is_occupied(config.host, config.port):
        raise RuntimeError(_browser_port_conflict_message(config))
    print(
        f"Verifying LLM provider {config.llm_provider} with model {config.model} "
        f"(demo={'on' if config.demo_mode else 'off'})..."
    )
    await verify_local_llm_config(config.llm)
    print("Selecting local speech runtime...")
    selection = await resolve_local_runtime_tts_selection(
        language=config.language,
        requested_backend=config.tts_backend,
        requested_voice=config.tts_voice,
        requested_base_url=config.tts_base_url,
        backend_locked=config.tts_backend_locked,
        explicit_voice=config.tts_voice_override,
    )
    config.tts_backend = selection.backend
    config.tts_voice = selection.voice
    config.tts_base_url = selection.base_url
    if config.vision_enabled:
        print(
            f"Browser vision enabled; local vision model ({config.vision_model}) "
            "will load on first camera inspection."
        )
    app, uvicorn = create_app(config, shared_vision=None)

    mode = "browser voice + local vision" if config.vision_enabled else "browser voice"
    backend_label = (
        f"{config.tts_backend} (auto)" if selection.auto_switched else config.tts_backend
    )
    server = uvicorn.Server(uvicorn.Config(app, host=config.host, port=config.port))
    print(f"Starting {PROJECT_IDENTITY.display_name} local {mode} server...")
    print(
        f"Using LLM {config.llm_provider}:{config.model} (demo={'on' if config.demo_mode else 'off'})."
    )
    print(f"Using STT {config.stt_backend}:{config.stt_model}.")
    client_url = f"http://{config.host}:{config.port}/client/"
    print(f"Waiting for {client_url} to become ready...")
    serve_task = await start_uvicorn_server(
        server,
        host=config.host,
        port=config.port,
        ready_path="/client/",
        timeout_secs=STARTUP_TIMEOUT_SECS,
    )
    print(
        f"{PROJECT_IDENTITY.display_name} local {mode} is available at "
        f"{client_url} "
        f"(llm={config.llm_provider}:{config.model}, "
        f"demo={'on' if config.demo_mode else 'off'}, "
        f"tts={backend_label}, robot_head={_robot_head_runtime_label(config)})"
    )
    print(_browser_readiness_summary(config, client_url=client_url))
    await serve_task
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local browser/WebRTC flow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_local_browser(config))
    except KeyboardInterrupt:
        return 130
    except (LocalDependencyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

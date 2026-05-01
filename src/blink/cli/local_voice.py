#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local native voice CLI for Apple Silicon Mac development."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from blink.adapters.schemas.function_schema import FunctionSchema
from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.brain.actions import embodied_action_tool_prompt
from blink.brain.memory import memory_tool_prompt
from blink.brain.processors import (
    BrainExpressionVoicePolicyProcessor,
    latest_user_text_from_context,
)
from blink.brain.runtime import BrainRuntime, build_session_resolver
from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    DEFAULT_LOCAL_LLM_PROVIDER,
    DEFAULT_LOCAL_STT_BACKEND,
    DEFAULT_LOCAL_VISION_MODEL,
    LOCAL_LLM_PROVIDERS,
    LocalDependencyError,
    LocalLLMConfig,
    build_local_voice_task,
    configure_logging,
    create_local_llm_service,
    create_local_stt_service,
    create_local_tts_service,
    create_local_vision_service,
    default_local_speech_system_prompt,
    default_local_stt_model,
    default_local_tts_backend,
    get_audio_device_by_index,
    get_audio_devices,
    get_local_env,
    local_env_flag,
    local_http_wav_service_available,
    maybe_load_dotenv,
    resolve_int,
    resolve_local_language,
    resolve_local_llm_config,
    resolve_local_runtime_tts_selection,
    resolve_local_tts_base_url,
    resolve_local_tts_voice,
    resolve_preferred_audio_device_indexes,
    tts_backend_uses_external_service,
    verify_local_llm_config,
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
    ErrorFrame,
    Frame,
    TTSSpeakFrame,
    UserImageRawFrame,
    VisionTextFrame,
)
from blink.function_calling import FunctionCallParams
from blink.pipeline.runner import PipelineRunner
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language

CAMERA_SOURCE_NONE = "none"
CAMERA_SOURCE_MACOS_HELPER = "macos-helper"
CAMERA_SOURCES = (CAMERA_SOURCE_NONE, CAMERA_SOURCE_MACOS_HELPER)
MACOS_CAMERA_HELPER_BUNDLE_ID = "ai.blink.CameraHelper"
MACOS_CAMERA_HELPER_STATUS_STATES = {
    "starting",
    "awaiting_permission",
    "running",
    "denied",
    "error",
    "stopped",
}


@dataclass
class LocalVoiceConfig:
    """Configuration for the local native voice workflow."""

    base_url: Optional[str]
    model: str
    system_prompt: str
    language: Language
    stt_backend: str
    tts_backend: str
    stt_model: str
    tts_voice: Optional[str]
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
    tts_base_url: Optional[str] = None
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    allow_barge_in: bool = False
    temperature: Optional[float] = None
    list_audio_devices: bool = False
    verbose: bool = False
    tts_backend_locked: bool = False
    tts_voice_override: Optional[str] = None
    llm_provider: str = DEFAULT_LOCAL_LLM_PROVIDER
    llm_service_tier: Optional[str] = None
    demo_mode: bool = False
    llm_max_output_tokens: Optional[int] = None
    vision_enabled: bool = False
    vision_requested_but_disabled: bool = False
    vision_model: str = DEFAULT_LOCAL_VISION_MODEL
    camera_source: str = CAMERA_SOURCE_NONE
    camera_device_index: int = 0
    camera_framerate: float = 1.0
    camera_max_width: int = 640
    camera_helper_app_path: Optional[Path] = None
    camera_helper_state_dir: Optional[Path] = None
    config_profile: str = "manual"

    @property
    def llm(self) -> LocalLLMConfig:
        """Return the effective provider config for the native voice LLM layer."""
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


@dataclass(frozen=True)
class MacOSCameraHelperStatus:
    """Public-safe status emitted by the macOS camera helper."""

    state: str
    updated_at: datetime
    frame_seq: int
    frame_path: Path
    width: int
    height: int
    format: str
    pid: Optional[int]
    reason_codes: tuple[str, ...]


class NativeCameraFrameBuffer(FrameProcessor):
    """Cache low-FPS native camera frames for on-demand local vision queries."""

    def __init__(self):
        """Initialize the native camera frame buffer."""
        super().__init__(name="native-camera-frame-buffer")
        self._latest_camera_frame: Optional[UserImageRawFrame] = None
        self._latest_camera_frame_seq = 0
        self._latest_camera_frame_received_monotonic: float | None = None
        self._latest_camera_frame_received_at: str | None = None
        self._latest_camera_frame_condition = asyncio.Condition()

    @property
    def latest_camera_frame(self) -> Optional[UserImageRawFrame]:
        """Return the most recent native camera frame, if any."""
        return self._latest_camera_frame

    @property
    def latest_camera_frame_seq(self) -> int:
        """Return the monotonically increasing native camera frame sequence."""
        return self._latest_camera_frame_seq

    @property
    def latest_camera_frame_received_monotonic(self) -> float | None:
        """Return the monotonic timestamp for the latest native camera frame."""
        return self._latest_camera_frame_received_monotonic

    @property
    def latest_camera_frame_received_at(self) -> str | None:
        """Return the UTC timestamp for the latest native camera frame."""
        return self._latest_camera_frame_received_at

    def latest_camera_frame_is_fresh(self, *, max_age_secs: float) -> bool:
        """Return whether the cached native camera frame is fresh enough to use."""
        received_at = self._latest_camera_frame_received_monotonic
        if self._latest_camera_frame is None or received_at is None:
            return False
        return (asyncio.get_running_loop().time() - float(received_at)) <= max_age_secs

    async def update_frame(self, frame: UserImageRawFrame):
        """Store a native camera frame and wake any waiting tool calls."""
        async with self._latest_camera_frame_condition:
            self._latest_camera_frame = frame
            self._latest_camera_frame_seq += 1
            self._latest_camera_frame_received_monotonic = asyncio.get_running_loop().time()
            self._latest_camera_frame_received_at = datetime.now(UTC).isoformat()
            self._latest_camera_frame_condition.notify_all()

    async def wait_for_latest_camera_frame(
        self,
        *,
        after_seq: int = 0,
        timeout: float = 0.0,
    ) -> Optional[UserImageRawFrame]:
        """Wait for a camera frame newer than ``after_seq``."""
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
        """Store native camera frames and pass through unrelated frames."""
        await super().process_frame(frame, direction)
        if isinstance(frame, UserImageRawFrame) and frame.transport_source == "native-camera":
            await self.update_frame(frame)
            return
        await self.push_frame(frame, direction)


NativeCameraCapture = Callable[[], Optional[UserImageRawFrame]]


class NativeCameraSnapshotProvider:
    """Retired OpenCV camera provider kept only for narrow legacy tests."""

    def __init__(
        self,
        *,
        frame_buffer: NativeCameraFrameBuffer,
        camera_device_index: int = 0,
        framerate: float = 1.0,
        max_width: int = 640,
        capture_frame: Optional[NativeCameraCapture] = None,
    ):
        """Initialize the native camera snapshot provider.

        Args:
            frame_buffer: Cache receiving the latest camera frame.
            camera_device_index: OpenCV camera device index.
            framerate: Maximum capture rate in frames per second.
            max_width: Maximum RGB frame width before Moondream inspection.
            capture_frame: Optional deterministic test capture function.
        """
        self.frame_buffer = frame_buffer
        self.camera_device_index = camera_device_index
        self.framerate = max(0.1, float(framerate or 1.0))
        self.max_width = max(64, int(max_width or 640))
        self._capture_frame_override = capture_frame
        self._task: asyncio.Task | None = None
        self._closed = False
        self._opencv_module: Any | None = None
        self._opencv_capture: Any | None = None

    @property
    def running(self) -> bool:
        """Return whether the background native camera capture task is running."""
        return self._task is not None and not self._task.done()

    async def start(self):
        """Start background low-FPS capture."""
        if self.running:
            return
        if self._capture_frame_override is None:
            # AVFoundation camera authorization must run on macOS's main run loop.
            # Opening through ``asyncio.to_thread`` can fail before the permission
            # prompt is shown, so native camera setup stays on the event-loop thread.
            self._ensure_opencv_capture()
        self._closed = False
        self._task = asyncio.create_task(self._capture_loop(), name="native-camera-capture")

    async def close(self):
        """Stop background capture and release the native camera."""
        self._closed = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        capture = self._opencv_capture
        self._opencv_capture = None
        if capture is not None:
            capture.release()

    async def capture_once(self) -> bool:
        """Capture one frame and update the cache.

        Returns:
            True when a frame was captured, otherwise False.
        """
        frame = self._capture_frame()
        if frame is None:
            return False
        await self.frame_buffer.update_frame(frame)
        return True

    async def _capture_loop(self):
        interval = 1.0 / self.framerate
        while not self._closed:
            try:
                await self.capture_once()
            except Exception as exc:  # pragma: no cover - exercised in live runtime
                logger.warning(f"Native camera capture failed: {type(exc).__name__}")
            await asyncio.sleep(interval)

    def _capture_frame(self) -> Optional[UserImageRawFrame]:
        if self._capture_frame_override is not None:
            return self._capture_frame_override()
        return self._capture_opencv_frame()

    def _ensure_opencv_capture(self):
        if self._opencv_capture is not None:
            return
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise LocalDependencyError(
                "The retired OpenCV native voice camera path is disabled. "
                "Use `--camera-source macos-helper` for the English Kokoro helper path, "
                "or use browser vision for the Chinese Melo/browser path."
            ) from exc

        self._opencv_module = cv2
        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        capture = cv2.VideoCapture(self.camera_device_index, backend)
        if not capture.isOpened() and sys.platform != "darwin":
            capture.release()
            capture = cv2.VideoCapture(self.camera_device_index)
        if not capture.isOpened():
            capture.release()
            raise LocalDependencyError(
                f"Native camera device {self.camera_device_index} could not be opened. "
                f"{_macos_camera_launcher_label()}"
                "The retired OpenCV native voice camera path is disabled. "
                "Use `--camera-source macos-helper` so BlinkCameraHelper.app owns macOS "
                "camera permission, or use browser vision for the Chinese Melo/browser path."
            )
        self._opencv_capture = capture

    def _capture_opencv_frame(self) -> Optional[UserImageRawFrame]:
        self._ensure_opencv_capture()
        cv2 = self._opencv_module
        ok, image = self._opencv_capture.read()
        if not ok or image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if width > self.max_width:
            scale = self.max_width / float(width)
            resized_size = (self.max_width, max(1, int(height * scale)))
            rgb = cv2.resize(rgb, resized_size, interpolation=cv2.INTER_AREA)
            height, width = rgb.shape[:2]
        rgb = rgb.copy()
        frame = UserImageRawFrame(
            user_id="local-native-camera",
            image=rgb.tobytes(),
            size=(width, height),
            format="RGB",
        )
        frame.transport_source = "native-camera"
        return frame


def _resolve_float(value: Optional[float | str]) -> Optional[float]:
    """Normalize an optional float-like value."""
    if value in (None, ""):
        return None
    return float(value)


def _native_camera_frame_is_fresh(
    camera_buffer: NativeCameraFrameBuffer,
    *,
    max_age_secs: float = 5.0,
) -> bool:
    """Return whether the cached native camera frame is fresh enough to inspect."""
    return camera_buffer.latest_camera_frame_is_fresh(max_age_secs=max_age_secs)


def _build_native_vision_prompt(question: str) -> str:
    """Build a stable English prompt for native camera vision queries."""
    normalized = (question or "").strip()
    lowered = normalized.lower()

    if any(token in lowered for token in ("text", "read", "word", "words", "screen", "label")):
        return (
            "Inspect the current Mac camera frame and answer in plain English. "
            "Read any large, clearly legible text exactly. "
            "If text is present but blurry or too small to read, say that clearly. "
            "Also describe the main visible objects around the text."
        )

    if any(token in lowered for token in ("holding", "hand", "hands")):
        return (
            "Inspect the current Mac camera frame and answer in plain English. "
            "Focus on what the person is holding in their hands. "
            "If nothing is clearly visible, say that and describe the rest of the scene."
        )

    if any(token in lowered for token in ("behind", "background", "backdrop")):
        return (
            "Inspect the current Mac camera frame and answer in plain English. "
            "Focus on the background behind the person. "
            "Mention prominent objects, screens, or large readable text."
        )

    return (
        "Inspect the current Mac camera frame and answer in plain English. "
        "Describe the main subject, the most prominent objects, the background, "
        "and any large clearly readable text. "
        "Do not invent details. If part of the image is blurry, say what is still clearly visible."
    )


def _native_vision_result_is_unusable(text: str) -> bool:
    """Return True when the raw native vision result is empty or clearly garbled."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.count("�") >= 2:
        return True
    return len(cleaned) < 4


def _repo_root() -> Path:
    """Return the repository root for local helper assets."""
    return Path(__file__).resolve().parents[3]


def _default_macos_camera_helper_app_path() -> Path:
    """Return the default local build path for BlinkCameraHelper.app."""
    return (
        _repo_root() / "native" / "macos" / "BlinkCameraHelper" / "build" / "BlinkCameraHelper.app"
    )


def _parse_iso8601_utc(value: object) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into UTC."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_helper_reason_codes(value: object) -> tuple[str, ...]:
    """Normalize helper reason codes without exposing raw payloads."""
    if not isinstance(value, list):
        return ()
    normalized: list[str] = []
    for item in value[:8]:
        text = str(item).strip().lower().replace("-", "_")
        if not text:
            continue
        cleaned = "".join(ch for ch in text if ch.isalnum() or ch == "_")
        if cleaned:
            normalized.append(cleaned[:64])
    return tuple(normalized)


def _parse_macos_camera_helper_status(
    payload: object,
    *,
    state_dir: Path,
    now: Optional[datetime] = None,
    max_age_secs: float = 5.0,
) -> MacOSCameraHelperStatus:
    """Parse and validate the macOS camera helper status payload."""
    if not isinstance(payload, dict):
        raise ValueError("helper_status_malformed")

    reason_codes = _safe_helper_reason_codes(payload.get("reason_codes"))
    state = str(payload.get("state", "")).strip()
    if state not in MACOS_CAMERA_HELPER_STATUS_STATES:
        raise ValueError("helper_status_malformed")
    if state != "running":
        code = reason_codes[0] if reason_codes else f"helper_{state}"
        raise ValueError(code)

    raw_frame_path = str(payload.get("frame_path", "")).strip()
    relative_frame_path = Path(raw_frame_path)
    if (
        not raw_frame_path
        or relative_frame_path.is_absolute()
        or relative_frame_path.name != raw_frame_path
        or raw_frame_path in {".", ".."}
    ):
        raise ValueError("helper_frame_path_invalid")

    updated_at = _parse_iso8601_utc(payload.get("updated_at"))
    if updated_at is None:
        raise ValueError("helper_timestamp_invalid")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    age_secs = (now - updated_at).total_seconds()
    if age_secs < -2.0 or age_secs > max_age_secs:
        raise ValueError("helper_frame_stale")

    try:
        frame_seq = int(payload.get("frame_seq"))
        width = int(payload.get("width"))
        height = int(payload.get("height"))
    except (TypeError, ValueError) as exc:
        raise ValueError("helper_status_malformed") from exc
    if frame_seq <= 0 or width <= 0 or height <= 0:
        raise ValueError("helper_frame_unavailable")
    if width > 4096 or height > 4096:
        raise ValueError("helper_frame_too_large")

    frame_format = str(payload.get("format", "")).strip().upper()
    if frame_format != "RGB":
        raise ValueError("helper_frame_format_unsupported")

    pid: Optional[int] = None
    raw_pid = payload.get("pid")
    if raw_pid not in (None, ""):
        try:
            parsed_pid = int(raw_pid)
            if parsed_pid > 0:
                pid = parsed_pid
        except (TypeError, ValueError):
            pid = None

    return MacOSCameraHelperStatus(
        state=state,
        updated_at=updated_at,
        frame_seq=frame_seq,
        frame_path=state_dir / raw_frame_path,
        width=width,
        height=height,
        format=frame_format,
        pid=pid,
        reason_codes=reason_codes,
    )


class MacOSCameraHelperSnapshotProvider:
    """Read on-demand camera snapshots from BlinkCameraHelper.app."""

    def __init__(
        self,
        *,
        frame_buffer: NativeCameraFrameBuffer,
        state_dir: Optional[Path] = None,
        app_path: Optional[Path] = None,
        framerate: float = 1.0,
        max_width: int = 640,
        launch_helper: bool = True,
        max_frame_age_secs: float = 5.0,
    ):
        """Initialize the macOS helper-backed snapshot provider."""
        self.frame_buffer = frame_buffer
        self.app_path = app_path or _default_macos_camera_helper_app_path()
        self.framerate = max(0.1, float(framerate or 1.0))
        self.max_width = max(64, int(max_width or 640))
        self.launch_helper = launch_helper
        self.max_frame_age_secs = max(1.0, float(max_frame_age_secs or 5.0))
        self._owns_state_dir = state_dir is None
        self.state_dir = state_dir or Path(tempfile.mkdtemp(prefix="blink-camera-helper-"))
        self._launched = False
        self.last_error: Optional[str] = None
        self.last_reason_codes: tuple[str, ...] = ()
        self._last_status: Optional[MacOSCameraHelperStatus] = None

    async def start(self):
        """Start or attach to the macOS camera helper."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.launch_helper:
            return
        if sys.platform != "darwin":
            raise LocalDependencyError("macOS camera helper is only available on macOS.")
        executable = self.app_path / "Contents" / "MacOS" / "BlinkCameraHelper"
        if not executable.exists():
            raise LocalDependencyError(
                "BlinkCameraHelper.app is not built. Run "
                "`./scripts/build-macos-camera-helper.sh` before starting this path."
            )
        command = [
            "open",
            "-n",
            str(self.app_path),
            "--args",
            "--state-dir",
            str(self.state_dir),
            "--framerate",
            f"{self.framerate:g}",
            "--max-width",
            str(self.max_width),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise LocalDependencyError("Could not launch BlinkCameraHelper.app.") from exc
        self._launched = True

    async def close(self):
        """Stop the helper process if this provider launched it."""
        if self._launched:
            status = self._load_status_payload()
            pid = status.get("pid") if isinstance(status, dict) else None
            try:
                parsed_pid = int(pid)
            except (TypeError, ValueError):
                parsed_pid = 0
            if parsed_pid > 0:
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(parsed_pid, signal.SIGTERM)
        if self._owns_state_dir:
            shutil.rmtree(self.state_dir, ignore_errors=True)

    async def capture_once(self) -> bool:
        """Read the latest helper frame into the runtime camera buffer."""
        try:
            status = self._read_status()
            image = status.frame_path.read_bytes()
            expected_size = status.width * status.height * 3
            if len(image) != expected_size:
                raise ValueError("helper_frame_size_mismatch")
        except FileNotFoundError:
            self.last_error = "helper_frame_missing"
            self.last_reason_codes = ("helper_frame_missing",)
            return False
        except ValueError as exc:
            self.last_error = str(exc)
            self.last_reason_codes = (self.last_error,)
            return False
        except Exception as exc:
            logger.debug(f"Could not read macOS camera helper frame: {type(exc).__name__}")
            self.last_error = "helper_frame_unavailable"
            self.last_reason_codes = (self.last_error,)
            return False

        frame = UserImageRawFrame(
            user_id="macos-camera-helper",
            image=image,
            size=(status.width, status.height),
            format=status.format,
        )
        frame.transport_source = "macos-camera-helper"
        await self.frame_buffer.update_frame(frame)
        self._last_status = status
        self.last_error = None
        self.last_reason_codes = ()
        return True

    def _load_status_payload(self) -> object:
        status_path = self.state_dir / "status.json"
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _read_status(self) -> MacOSCameraHelperStatus:
        payload = self._load_status_payload()
        return _parse_macos_camera_helper_status(
            payload,
            state_dir=self.state_dir,
            max_age_secs=self.max_frame_age_secs,
        )


def _macos_camera_launcher_label() -> str:
    """Return the likely macOS app that owns camera privacy permission."""
    if sys.platform != "darwin":
        return ""

    pid = os.getpid()
    try:
        while pid and pid != 1:
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "pid=,ppid=,comm="],
                text=True,
                timeout=1.0,
            ).strip()
            parts = output.split(None, 2)
            if len(parts) < 3:
                break
            parent_pid = int(parts[1])
            command = parts[2]
            if ".app/Contents/" in command:
                app_path = command.split(".app/Contents/", 1)[0] + ".app"
                return f" Detected launcher app: {app_path}."
            pid = parent_pid
    except Exception:
        return ""
    return ""


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Run {PROJECT_IDENTITY.display_name} local native voice on your Mac."
    )
    parser.add_argument(
        "--llm-provider",
        choices=LOCAL_LLM_PROVIDERS,
        help="Local LLM provider. Defaults to ollama.",
    )
    parser.add_argument("--model", help="Provider-relative model name.")
    parser.add_argument(
        "--config-profile",
        help=(
            "Typed local runtime profile id, such as native-en-kokoro or "
            "native-en-kokoro-macos-camera."
        ),
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
        "--vision",
        action="store_true",
        help=(
            "Deprecated no-op. Use --camera-source macos-helper for the English Kokoro "
            "camera helper path; use browser vision for Chinese Melo sessions."
        ),
    )
    parser.add_argument(
        "--vision-model",
        help=(
            "Deprecated no-op for native voice. Browser vision still defaults to "
            f"{DEFAULT_LOCAL_VISION_MODEL}."
        ),
    )
    parser.add_argument(
        "--camera-device",
        type=int,
        help="Deprecated no-op for native voice.",
    )
    parser.add_argument(
        "--camera-framerate",
        type=float,
        help="Deprecated no-op for native voice.",
    )
    parser.add_argument(
        "--camera-max-width",
        type=int,
        help="Deprecated no-op for native voice.",
    )
    parser.add_argument(
        "--camera-source",
        choices=CAMERA_SOURCES,
        help="Optional native voice camera source. Use `macos-helper` for BlinkCameraHelper.app.",
    )
    parser.add_argument(
        "--camera-helper-app",
        help="Path to BlinkCameraHelper.app. Defaults to native/macos/BlinkCameraHelper/build.",
    )
    parser.add_argument(
        "--camera-helper-state-dir",
        help="Directory where BlinkCameraHelper writes status.json and latest.rgb.",
    )
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
    parser.add_argument("--input-device", type=int, help="PyAudio input device index.")
    parser.add_argument("--output-device", type=int, help="PyAudio output device index.")
    parser.add_argument(
        "--allow-barge-in",
        action="store_true",
        help="Allow interrupting the assistant while it is speaking. Disabled by default on native local audio to avoid self-interruption from speaker bleed.",
    )
    parser.add_argument(
        "--protected-playback",
        action="store_true",
        help=(
            "Force bot-speech protection even when BLINK_LOCAL_ALLOW_BARGE_IN=1. "
            "Use this on speaker setups that self-interrupt."
        ),
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print available PyAudio devices and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging while the voice loop is running.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LocalVoiceConfig:
    """Resolve CLI configuration from arguments and environment variables."""
    maybe_load_dotenv()
    profile = resolve_local_runtime_profile(
        runtime="voice",
        profile_id=getattr(args, "config_profile", None),
    )
    language = resolve_local_language(
        args.language or get_local_env("LANGUAGE") or profile_value(profile, "language")
    )
    configured_tts_backend = get_local_env("TTS_BACKEND")
    robot_head_driver = getattr(args, "robot_head_driver", None) or get_local_env(
        "ROBOT_HEAD_DRIVER", "none"
    )
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
    camera_source = (
        getattr(args, "camera_source", None)
        or get_local_env("CAMERA_SOURCE")
        or profile_value(profile, "camera_source")
        or CAMERA_SOURCE_NONE
    )
    if camera_source not in CAMERA_SOURCES:
        raise LocalDependencyError(f"Unsupported native voice camera source: {camera_source}")
    vision_requested = bool(
        getattr(args, "vision", False)
        or local_env_flag(
            "VOICE_VISION",
            bool(profile_value(profile, "voice_vision", False)),
        )
    )
    vision_enabled = camera_source == CAMERA_SOURCE_MACOS_HELPER
    vision_model = (
        getattr(args, "vision_model", None)
        or get_local_env("VISION_MODEL")
        or profile_value(profile, "vision_model")
        or DEFAULT_LOCAL_VISION_MODEL
    )
    camera_device_index = (
        resolve_int(getattr(args, "camera_device", None) or get_local_env("CAMERA_DEVICE")) or 0
    )
    camera_framerate = (
        _resolve_float(getattr(args, "camera_framerate", None) or get_local_env("CAMERA_FRAMERATE"))
        or _resolve_float(profile_value(profile, "camera_framerate"))
        or 1.0
    )
    camera_max_width = (
        resolve_int(getattr(args, "camera_max_width", None) or get_local_env("CAMERA_MAX_WIDTH"))
        or resolve_int(profile_value(profile, "camera_max_width"))
        or 640
    )
    camera_helper_app = getattr(args, "camera_helper_app", None) or get_local_env(
        "CAMERA_HELPER_APP"
    )
    camera_helper_state_dir = getattr(args, "camera_helper_state_dir", None) or get_local_env(
        "CAMERA_HELPER_STATE_DIR"
    )

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
    if camera_source == CAMERA_SOURCE_MACOS_HELPER:
        if sys.platform != "darwin":
            raise LocalDependencyError("The macOS camera helper source is only available on macOS.")
        if language != Language.EN:
            raise LocalDependencyError("The macOS camera helper voice path is English-only.")
        if tts_backend != "kokoro":
            raise LocalDependencyError("The macOS camera helper voice path requires Kokoro TTS.")

    input_device_index, output_device_index = resolve_preferred_audio_device_indexes(
        resolve_int(args.input_device or get_local_env("AUDIO_INPUT_DEVICE")),
        resolve_int(args.output_device or get_local_env("AUDIO_OUTPUT_DEVICE")),
    )

    return LocalVoiceConfig(
        base_url=llm.base_url,
        model=llm.model,
        system_prompt=llm.system_prompt,
        language=language,
        stt_backend=stt_backend,
        tts_backend=tts_backend,
        stt_model=stt_model,
        tts_voice=tts_voice,
        robot_head_driver=robot_head_driver,
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
        tts_base_url=resolve_local_tts_base_url(tts_backend),
        input_device_index=input_device_index,
        output_device_index=output_device_index,
        allow_barge_in=False
        if getattr(args, "protected_playback", False)
        else args.allow_barge_in
        or local_env_flag(
            "ALLOW_BARGE_IN",
            bool(profile_value(profile, "allow_barge_in", False)),
        ),
        temperature=args.temperature,
        list_audio_devices=args.list_audio_devices,
        verbose=args.verbose,
        tts_backend_locked=tts_backend_locked,
        tts_voice_override=explicit_tts_voice,
        llm_provider=llm.provider,
        llm_service_tier=llm.service_tier,
        demo_mode=llm.demo_mode,
        llm_max_output_tokens=llm.max_output_tokens,
        vision_enabled=vision_enabled,
        vision_requested_but_disabled=vision_requested and not vision_enabled,
        vision_model=vision_model,
        camera_source=camera_source,
        camera_device_index=max(0, camera_device_index),
        camera_framerate=max(0.1, camera_framerate),
        camera_max_width=max(64, camera_max_width),
        camera_helper_app_path=Path(camera_helper_app).expanduser()
        if camera_helper_app not in (None, "")
        else None,
        camera_helper_state_dir=Path(camera_helper_state_dir).expanduser()
        if camera_helper_state_dir not in (None, "")
        else None,
        config_profile=profile.profile_id if profile is not None else "manual",
    )


def create_local_transport(config: LocalVoiceConfig):
    """Create the local audio transport for the native voice loop."""
    try:
        from blink.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
        raise LocalDependencyError(
            "Local voice requires the `local` extra and PortAudio. "
            "Run `brew install portaudio` and "
            "`./scripts/bootstrap-local-mac.sh --profile voice`."
        ) from exc

    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=config.input_device_index,
            output_device_index=config.output_device_index,
        )
    )


def build_local_voice_runtime(
    config: LocalVoiceConfig,
    *,
    transport=None,
    stt=None,
    llm=None,
    tts=None,
    vision=None,
    camera_frame_buffer: Optional[NativeCameraFrameBuffer] = None,
    camera_provider: Optional[Any] = None,
    tts_session=None,
):
    """Build the shared runtime objects for the local native voice flow."""
    if config.vision_enabled and config.camera_source != CAMERA_SOURCE_MACOS_HELPER:
        raise LocalDependencyError(
            "OpenCV native voice camera is disabled for the supported local voice path. "
            "Use `--camera-source macos-helper` for the English helper-backed path, "
            "or keep native voice as English-only mic -> MLX Whisper -> LLM -> Kokoro."
        )

    transport = transport or create_local_transport(config)
    stt = stt or create_local_stt_service(
        backend=config.stt_backend,
        model=config.stt_model,
        language=config.language,
    )
    brain_tool_prompt = memory_tool_prompt(config.language)
    if config.robot_head_driver != "none":
        robot_prompt = (
            robot_head_tool_prompt(config.language)
            if config.robot_head_operator_mode
            else embodied_action_tool_prompt(config.language)
        )
        brain_tool_prompt = " ".join(part for part in [brain_tool_prompt, robot_prompt] if part)
    if config.vision_enabled:
        camera_prompt = (
            "When the user asks what you can see through the Mac camera, what is in front "
            "of the camera, what they are holding, or what text is visible, call "
            "`fetch_user_image`. The tool inspects one recent cached camera frame; do not "
            "claim continuous video, facial recognition, or hidden camera access."
        )
        brain_tool_prompt = " ".join(part for part in [brain_tool_prompt, camera_prompt] if part)
    runtime_base_prompt = " ".join(
        part for part in [config.system_prompt, brain_tool_prompt] if part
    )
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
    llm = llm or create_local_llm_service(llm_service_config)
    tts = tts or create_local_tts_service(
        backend=config.tts_backend,
        voice=config.tts_voice,
        language=config.language,
        base_url=config.tts_base_url,
        aiohttp_session=tts_session,
    )
    vision_service = vision
    if config.vision_enabled:
        camera_frame_buffer = camera_frame_buffer or NativeCameraFrameBuffer()
        camera_provider = camera_provider or MacOSCameraHelperSnapshotProvider(
            frame_buffer=camera_frame_buffer,
            state_dir=config.camera_helper_state_dir,
            app_path=config.camera_helper_app_path,
            framerate=config.camera_framerate,
            max_width=config.camera_max_width,
        )

    controller = None
    if config.robot_head_driver != "none":
        catalog = load_robot_head_capability_catalog(config.robot_head_catalog_path)
        driver_mode = config.robot_head_driver
        if driver_mode == "mock":
            driver = MockDriver()
        elif driver_mode == "preview":
            driver = PreviewDriver(trace_dir=Path.cwd() / "artifacts" / "robot_head_preview")
        elif driver_mode == "simulation":
            driver = SimulationDriver(
                config=RobotHeadSimulationConfig(
                    hardware_profile_path=config.robot_head_hardware_profile_path,
                    scenario_path=config.robot_head_sim_scenario_path,
                    realtime=config.robot_head_sim_realtime,
                    trace_dir=config.robot_head_sim_trace_dir,
                )
            )
        elif driver_mode == "live":
            driver = LiveDriver(
                config=RobotHeadLiveDriverConfig(
                    hardware_profile_path=config.robot_head_hardware_profile_path,
                    port=config.robot_head_port,
                    baud_rate=config.robot_head_baud,
                    arm_enabled=config.robot_head_live_arm,
                    arm_ttl_seconds=config.robot_head_arm_ttl_seconds,
                ),
                preview_driver=PreviewDriver(
                    trace_dir=Path.cwd() / "artifacts" / "robot_head_preview"
                ),
            )
        else:  # pragma: no cover - parser/config guards supported values
            raise ValueError(f"Unsupported robot-head driver: {driver_mode}")

        controller = RobotHeadController(catalog=catalog, driver=driver)

    session_resolver = build_session_resolver(runtime_kind="voice")
    brain_runtime = BrainRuntime(
        base_prompt=runtime_base_prompt,
        language=config.language,
        runtime_kind="voice",
        session_resolver=session_resolver,
        llm=llm,
        robot_head_controller=controller,
        robot_head_operator_mode=config.robot_head_operator_mode,
        vision_enabled=config.vision_enabled,
        continuous_perception_enabled=False,
        tts_backend=config.tts_backend,
    )
    tools = brain_runtime.register_daily_tools()
    if config.vision_enabled:
        fetch_image_function = FunctionSchema(
            name="fetch_user_image",
            description=(
                "Inspect one recent frame from the native Mac camera when the user asks "
                "about what is visible."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": (
                        "The user's question about the current camera view, such as "
                        "'What am I holding?' or 'What text do you see?'"
                    ),
                }
            },
            required=["question"],
        )
        tools = ToolsSchema(
            standard_tools=[fetch_image_function, *tools.standard_tools],
            custom_tools=tools.custom_tools,
        )
    context = LLMContext(tools=tools)
    brain_runtime.bind_context(context)
    brain_runtime.start_background_maintenance()
    setattr(context, "blink_brain_runtime", brain_runtime)
    if config.vision_enabled:
        setattr(context, "blink_native_camera_provider", camera_provider)
        setattr(context, "blink_native_camera_frame_buffer", camera_frame_buffer)

        def get_vision_service():
            nonlocal vision_service
            if vision_service is None:
                vision_service = create_local_vision_service(model=config.vision_model)
            return vision_service

        async def run_vision_query(
            image_frame: UserImageRawFrame, prompt: str
        ) -> tuple[Optional[str], Optional[str]]:
            """Run a single native camera vision query."""
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
            async for result_frame in get_vision_service().run_vision(query_frame):
                if isinstance(result_frame, VisionTextFrame):
                    description = (result_frame.text or "").strip()
                elif isinstance(result_frame, ErrorFrame):
                    error_text = result_frame.error
            return description, error_text

        async def fetch_user_image(params: FunctionCallParams):
            tool_question = str(params.arguments.get("question", "")).strip()
            user_question = latest_user_text_from_context(context) or tool_question
            if camera_provider is not None:
                await camera_provider.capture_once()
            latest_seq = camera_frame_buffer.latest_camera_frame_seq
            latest_frame = camera_frame_buffer.latest_camera_frame
            if latest_frame is None or not _native_camera_frame_is_fresh(camera_frame_buffer):
                if camera_provider is not None:
                    await camera_provider.capture_once()
                fresh_frame = await camera_frame_buffer.wait_for_latest_camera_frame(
                    after_seq=latest_seq,
                    timeout=1.0,
                )
                if fresh_frame is not None:
                    latest_frame = fresh_frame

            if latest_frame is None:
                helper_reason = getattr(camera_provider, "last_error", None)
                helper_error = (
                    "Camera permission is denied for BlinkCameraHelper. Allow it in macOS "
                    "System Settings > Privacy & Security > Camera, then restart this path."
                    if helper_reason == "camera_permission_denied"
                    else "BlinkCameraHelper is still waiting for camera permission."
                    if helper_reason == "camera_permission_pending"
                    else "No fresh macOS camera helper frame is available yet. "
                    "Allow camera permission for BlinkCameraHelper and try again."
                )
                await params.result_callback({"error": helper_error})
                return

            if not _native_camera_frame_is_fresh(camera_frame_buffer):
                await params.result_callback(
                    {
                        "error": (
                            "The latest macOS camera helper frame is stale. "
                            "Keep BlinkCameraHelper running and try again after a fresh frame arrives."
                        )
                    }
                )
                return

            vision_frame = UserImageRawFrame(
                user_id=latest_frame.user_id,
                image=latest_frame.image,
                size=latest_frame.size,
                format=latest_frame.format,
            )
            vision_frame.transport_source = latest_frame.transport_source
            vision_frame.pts = latest_frame.pts

            vision_prompt = _build_native_vision_prompt(user_question)
            logger.debug(f"Running native camera vision query with prompt: {vision_prompt}")
            description, error_text = await run_vision_query(vision_frame, vision_prompt)

            if _native_vision_result_is_unusable(description or ""):
                fallback_prompt = (
                    "Inspect the current Mac camera frame and answer in plain English. "
                    "Give a short, concrete description of the clearest visible objects, "
                    "people, layout, and any large readable text. "
                    "If the image is blurry, say what remains confidently visible."
                )
                logger.debug("Retrying native camera vision query with fallback prompt")
                retry_description, retry_error = await run_vision_query(
                    vision_frame, fallback_prompt
                )
                if not _native_vision_result_is_unusable(retry_description or ""):
                    description = retry_description
                    error_text = retry_error or error_text
                elif retry_error:
                    error_text = retry_error

            if description:
                await params.result_callback({"description": description})
                return

            await params.result_callback({"error": error_text or "Native camera analysis failed."})

        llm.register_function("fetch_user_image", fetch_user_image)

        @llm.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls):
            if any(call.function_name == "fetch_user_image" for call in function_calls):
                await tts.queue_frame(TTSSpeakFrame("Let me take a look."))

    pre_llm_processors = list(brain_runtime.pre_llm_processors)
    pre_tts_processors = [
        BrainExpressionVoicePolicyProcessor(
            policy_provider=lambda: brain_runtime.current_voice_policy(
                latest_user_text=latest_user_text_from_context(context),
            ),
            actuation_plan_provider=lambda: brain_runtime.current_voice_actuation_plan(
                latest_user_text=latest_user_text_from_context(context),
            ),
            tts_backend=config.tts_backend,
            metrics_recorder=brain_runtime.voice_metrics_recorder,
        )
    ]
    pre_output_processors = []
    if brain_runtime.action_dispatcher is not None:
        pre_output_processors.append(
            EmbodimentPolicyProcessor(
                action_dispatcher=brain_runtime.action_dispatcher,
                store=brain_runtime.store,
                session_resolver=brain_runtime.session_resolver,
                presence_scope_key=brain_runtime.presence_scope_key,
            )
        )
    post_context_processors = list(brain_runtime.post_context_processors)

    return build_local_voice_task(
        transport=transport,
        stt=stt,
        llm=llm,
        tts=tts,
        context=context,
        mute_during_bot_speech=not config.allow_barge_in,
        pre_llm_processors=pre_llm_processors,
        pre_tts_processors=pre_tts_processors,
        pre_output_processors=pre_output_processors,
        post_context_processors=post_context_processors,
    )


def _print_audio_devices():
    devices = get_audio_devices()
    if not devices:
        print("No audio devices found.")
        return

    print("Audio devices:")
    for device in devices:
        print(
            f"  [{device.index}] {device.name} "
            f"(in={device.max_input_channels}, out={device.max_output_channels}, "
            f"rate={device.default_sample_rate})"
        )


def _robot_head_runtime_label(config: LocalVoiceConfig) -> str:
    """Return a concise runtime label for robot-head mode."""
    if config.robot_head_driver == "simulation":
        timing = "realtime" if config.robot_head_sim_realtime else "virtual"
        return f"simulation({timing})"
    if config.robot_head_driver != "live":
        return config.robot_head_driver
    return f"live(port={config.robot_head_port or 'auto'}, armed={config.robot_head_live_arm})"


def _native_voice_isolation_label(config: LocalVoiceConfig) -> str:
    """Return the public native voice isolation label."""
    if config.camera_source == CAMERA_SOURCE_MACOS_HELPER or config.vision_enabled:
        return "backend-plus-helper-camera"
    return "backend-only"


async def _start_voice_context_resources(context: LLMContext):
    """Start optional resources owned by the native voice context."""
    camera_provider = getattr(context, "blink_native_camera_provider", None)
    if camera_provider is not None:
        await camera_provider.start()


async def _close_voice_context_resources(context: LLMContext):
    """Close optional resources owned by the native voice context."""
    camera_provider = getattr(context, "blink_native_camera_provider", None)
    if camera_provider is not None:
        await camera_provider.close()
    runtime = getattr(context, "blink_brain_runtime", None)
    if runtime is not None:
        runtime.close()


def _voice_runtime_status_text(
    config: LocalVoiceConfig,
    *,
    selected_backend_label: str,
    input_device_name: str,
    output_device_name: str,
) -> str:
    """Return a concise status line for the native voice runtime."""
    vision_label = (
        f"macos-helper(model={config.vision_model}, fps={config.camera_framerate:g})"
        if config.vision_enabled
        else "off"
    )
    if not config.vision_enabled and config.vision_requested_but_disabled:
        vision_label = "off (native voice camera disabled)"
    return (
        f"{PROJECT_IDENTITY.display_name} local voice is running. "
        "runtime=native, "
        "transport=PyAudio, "
        f"profile={config.config_profile}, "
        f"isolation={_native_voice_isolation_label(config)}, "
        f"input={input_device_name}, "
        f"output={output_device_name}. "
        f"llm={config.llm_provider}:{config.model}. "
        f"demo={'on' if config.demo_mode else 'off'}. "
        f"stt={config.stt_backend}:{config.stt_model}. "
        f"tts={selected_backend_label}. "
        f"vision={vision_label}. "
        f"robot_head={_robot_head_runtime_label(config)}. "
        f"protected_playback={'off' if config.allow_barge_in else 'on'}. "
        f"barge_in={'on' if config.allow_barge_in else 'off'}. "
        "primary_browser_paths=browser-zh-melo,browser-en-kokoro. "
        "Speak into your microphone. "
        "Press Ctrl+C to stop."
    )


async def run_local_voice(config: LocalVoiceConfig) -> int:
    """Run the local native voice workflow."""
    configure_logging(config.verbose)

    if config.list_audio_devices:
        _print_audio_devices()
        return 0

    await verify_local_llm_config(config.llm)
    if (
        config.tts_backend == "local-http-wav"
        and os.environ.get("BLINK_ALLOW_NATIVE_VOICE_MELO") != "1"
    ):
        raise LocalDependencyError(
            "Native voice does not use MeloTTS/local-http-wav in the supported path. "
            "Use `./scripts/run-local-voice-en.sh` for English native voice or "
            "`./scripts/run-local-browser-melo.sh` for Chinese browser voice."
        )
    selection = await resolve_local_runtime_tts_selection(
        language=config.language,
        requested_backend=config.tts_backend,
        requested_voice=config.tts_voice,
        requested_base_url=config.tts_base_url,
        backend_locked=True,
        explicit_voice=config.tts_voice_override,
    )
    config.tts_backend = selection.backend
    config.tts_voice = selection.voice
    config.tts_base_url = selection.base_url
    selected_backend_label = (
        f"{config.tts_backend} (auto)" if selection.auto_switched else config.tts_backend
    )

    if tts_backend_uses_external_service(config.tts_backend):
        import aiohttp

        async with aiohttp.ClientSession() as tts_session:
            task, context = build_local_voice_runtime(config, tts_session=tts_session)
            input_device = get_audio_device_by_index(config.input_device_index)
            output_device = get_audio_device_by_index(config.output_device_index)

            try:
                await _start_voice_context_resources(context)
            except Exception:
                await _close_voice_context_resources(context)
                raise
            print(
                _voice_runtime_status_text(
                    config,
                    selected_backend_label=selected_backend_label,
                    input_device_name=input_device.name if input_device else "system default",
                    output_device_name=output_device.name if output_device else "system default",
                ),
                flush=True,
            )

            runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
            try:
                await runner.run(task)
            finally:
                await _close_voice_context_resources(context)
            return 0

    task, context = build_local_voice_runtime(config)
    input_device = get_audio_device_by_index(config.input_device_index)
    output_device = get_audio_device_by_index(config.output_device_index)

    try:
        await _start_voice_context_resources(context)
    except Exception:
        await _close_voice_context_resources(context)
        raise
    print(
        _voice_runtime_status_text(
            config,
            selected_backend_label=selected_backend_label,
            input_device_name=input_device.name if input_device else "system default",
            output_device_name=output_device.name if output_device else "system default",
        ),
        flush=True,
    )

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    try:
        await runner.run(task)
    finally:
        await _close_voice_context_resources(context)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local native voice flow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_config(args)
        return asyncio.run(run_local_voice(config))
    except KeyboardInterrupt:
        return 130
    except (LocalDependencyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

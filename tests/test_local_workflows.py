import argparse
import asyncio
import json
import sys
import threading
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests._optional import OPTIONAL_RUNTIME, require_optional_modules

pytestmark = OPTIONAL_RUNTIME
require_optional_modules("fastapi", "aiortc", "av")

from av import VideoFrame

from blink.brain.persona import (
    BrainBehaviorControlUpdateResult,
    BrainRuntimeExpressionState,
    default_behavior_control_profile,
)
from blink.brain.processors import (
    BrainExpressionVoicePolicyProcessor,
    BrainVoiceInputHealthProcessor,
)
from blink.cli import local_browser, local_common, local_runtime_profiles, local_voice
from blink.cli.local_browser import (
    LatestCameraFrameBuffer,
    LocalBrowserConfig,
    _build_vision_prompt,
    _latest_user_text,
    _vision_result_is_unusable,
    build_local_browser_runtime,
    create_app,
)
from blink.cli.local_common import (
    DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS,
    AudioDeviceInfo,
    LocalLLMConfig,
    LocalRuntimeTTSSelection,
    default_local_speech_system_prompt,
    resolve_profile_extras,
)
from blink.cli.local_voice import (
    CAMERA_SOURCE_MACOS_HELPER,
    CAMERA_SOURCE_NONE,
    LocalVoiceConfig,
    MacOSCameraHelperSnapshotProvider,
    NativeCameraFrameBuffer,
    NativeCameraSnapshotProvider,
    _parse_macos_camera_helper_status,
    build_local_voice_runtime,
    resolve_config,
)
from blink.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    VisionTextFrame,
)
from blink.processors.frame_processor import FrameProcessor
from blink.services.stt_latency import WHISPER_TTFS_P99
from blink.transports.base_transport import BaseTransport
from blink.transports.smallwebrtc import transport as smallwebrtc_transport
from blink.transports.smallwebrtc.connection import SmallWebRTCConnection
from blink.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from blink.transports.smallwebrtc.transport import SmallWebRTCClient
from blink.turns.user_mute import AlwaysUserMuteStrategy
from blink.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)


class DummyTransport(BaseTransport):
    def __init__(self):
        super().__init__()
        self._input = FrameProcessor(name="dummy-input")
        self._output = FrameProcessor(name="dummy-output")

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output


class DummyLLMProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="dummy-llm")
        self.registered_functions = {}
        self.event_handlers = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler

    def event_handler(self, name):
        def decorator(handler):
            self.event_handlers[name] = handler
            return handler

        return decorator


class DummyTTSProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="dummy-tts")
        self.queued_frames = []

    async def queue_frame(self, frame):
        self.queued_frames.append(frame)


class DummyVisionProcessor(FrameProcessor):
    def __init__(self, response_text="camera result", error_text=None):
        super().__init__(name="dummy-vision")
        self.response_text = response_text
        self.error_text = error_text
        self.received_frames = []

    async def run_vision(self, frame):
        self.received_frames.append(frame)
        if self.error_text:
            yield ErrorFrame(self.error_text)
            return
        yield VisionTextFrame(text=self.response_text)


class SequencedVisionProcessor(FrameProcessor):
    def __init__(self, responses):
        super().__init__(name="sequenced-vision")
        self.responses = list(responses)
        self.received_frames = []

    async def run_vision(self, frame):
        self.received_frames.append(frame)
        response_text = self.responses.pop(0)
        yield VisionTextFrame(text=response_text)


class FakeTrackState:
    def __init__(self, *, enabled=True):
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


class FakeSmallWebRTCConnection:
    def __init__(self, *, connected=True):
        self.connected = connected
        self.renegotiation_requests = 0

    def is_connected(self):
        return self.connected

    def ask_to_renegotiate(self):
        self.renegotiation_requests += 1


class FakeSmallWebRTCEventConnection:
    pc_id = "pc-refresh"

    def __init__(self, *, connected=True, media_connected: bool | None = None):
        self.connected = connected
        self.media_connected = connected if media_connected is None else media_connected
        self.handlers = {}
        self.connect_count = 0
        self.audio_track_requests = 0
        self.video_track_requests = 0
        self.screen_track_requests = 0
        self.replaced_audio_tracks: list[object] = []
        self.replaced_video_tracks: list[object] = []

    def event_handler(self, name):
        def decorator(handler):
            self.handlers[name] = handler
            return handler

        return decorator

    def is_connected(self):
        return self.connected

    def is_media_transport_ready(self):
        return self.media_connected

    async def connect(self):
        self.connect_count += 1

    def audio_input_track(self):
        self.audio_track_requests += 1
        return FakeTrackState(enabled=True)

    def video_input_track(self):
        self.video_track_requests += 1
        return FakeTrackState(enabled=True)

    def screen_video_input_track(self):
        self.screen_track_requests += 1
        return FakeTrackState(enabled=True)

    def replace_audio_track(self, track):
        self.replaced_audio_tracks.append(track)
        return True

    def replace_video_track(self, track):
        self.replaced_video_tracks.append(track)
        return True


def test_resolve_profile_extras():
    assert resolve_profile_extras("text") == []
    assert resolve_profile_extras("voice") == ["local", "mlx-whisper", "kokoro"]
    assert resolve_profile_extras("browser") == ["runner", "webrtc", "mlx-whisper", "kokoro"]
    assert resolve_profile_extras("voice", with_piper=True) == [
        "local",
        "mlx-whisper",
        "kokoro",
        "piper",
    ]
    assert resolve_profile_extras("voice", with_vision=True) == [
        "local",
        "mlx-whisper",
        "kokoro",
    ]
    assert resolve_profile_extras("full", with_vision=True) == [
        "local",
        "runner",
        "webrtc",
        "mlx-whisper",
        "kokoro",
        "moondream",
    ]


def test_local_runtime_profiles_are_typed_and_bounded():
    profiles = local_runtime_profiles.load_local_runtime_profiles(include_local_override=False)

    assert profiles["browser-zh-melo"].runtime == "browser"
    assert profiles["browser-zh-melo"].get("tts_backend") == "local-http-wav"
    assert profiles["browser-zh-melo"].get("browser_vision") is True
    assert profiles["browser-zh-melo"].get("continuous_perception") is False
    assert profiles["browser-zh-melo"].get("allow_barge_in") is False
    assert profiles["browser-en-kokoro"].runtime == "browser"
    assert profiles["browser-en-kokoro"].get("language") == "en"
    assert profiles["browser-en-kokoro"].get("tts_backend") == "kokoro"
    assert profiles["browser-en-kokoro"].get("browser_vision") is True
    assert profiles["browser-en-kokoro"].get("continuous_perception") is False
    assert profiles["browser-en-kokoro"].get("allow_barge_in") is False
    assert profiles["browser-en-kokoro"].get("ignore_env_system_prompt") is True
    assert profiles["native-en-kokoro"].runtime == "voice"
    assert profiles["native-en-kokoro"].get("camera_source") == "none"
    assert profiles["native-en-kokoro"].get("allow_barge_in") is False
    assert profiles["native-en-kokoro-macos-camera"].runtime == "voice"
    assert profiles["native-en-kokoro-macos-camera"].get("camera_source") == "macos-helper"
    assert profiles["native-en-kokoro-macos-camera"].get("allow_barge_in") is False


def test_local_browser_readiness_summary_reports_primary_melo_profile():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="local-http-wav/MeloTTS",
        config_profile="browser-zh-melo",
    )

    summary = local_browser._browser_readiness_summary(
        config,
        client_url="http://127.0.0.1:7860/client/",
    )

    assert "runtime=browser" in summary
    assert "transport=WebRTC" in summary
    assert "profile=browser-zh-melo" in summary
    assert "language=zh" in summary
    assert "tts=local-http-wav/MeloTTS" in summary
    assert "vision=on" in summary
    assert "continuous_perception=off" in summary
    assert "protected_playback=on" in summary
    assert "barge_in=off" in summary
    assert "barge_in_policy=protected" in summary
    assert "client=http://127.0.0.1:7860/client/" in summary


def test_local_browser_readiness_summary_reports_primary_english_kokoro_profile():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="bf_emma",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
    )

    summary = local_browser._browser_readiness_summary(
        config,
        client_url="http://127.0.0.1:7860/client/",
    )

    assert "runtime=browser" in summary
    assert "transport=WebRTC" in summary
    assert "profile=browser-en-kokoro" in summary
    assert "language=en" in summary
    assert "tts=kokoro/English" in summary
    assert "vision=on" in summary
    assert "continuous_perception=off" in summary
    assert "protected_playback=on" in summary
    assert "barge_in=off" in summary
    assert "barge_in_policy=protected" in summary
    assert "client=http://127.0.0.1:7860/client/" in summary


def test_local_runtime_profile_validation_rejects_unbounded_fields(tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {
                        "id": "unsafe-profile",
                        "runtime": "browser",
                        "values": {"system_prompt": "mutate hidden prompt"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(local_runtime_profiles.LocalRuntimeProfileError):
        local_runtime_profiles.load_local_runtime_profiles(profiles_path=profile_path)


def test_local_voice_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "voice-model")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "Voice prompt")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_STT_BACKEND", "mlx-whisper")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_STT_MODEL", "mlx-community/whisper-large-v3-turbo")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "bf_emma")
    monkeypatch.setenv("BLINK_LOCAL_AUDIO_INPUT_DEVICE", "4")
    monkeypatch.setenv("BLINK_LOCAL_AUDIO_OUTPUT_DEVICE", "9")
    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "1")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.base_url == "http://localhost:9999/v1"
    assert config.model == "voice-model"
    assert config.system_prompt == "Voice prompt"
    assert config.llm_provider == "ollama"
    assert config.llm == LocalLLMConfig(
        provider="ollama",
        model="voice-model",
        base_url="http://localhost:9999/v1",
        system_prompt="Voice prompt",
    )
    assert config.language == local_browser.Language.EN
    assert config.stt_model == "mlx-community/whisper-large-v3-turbo"
    assert config.tts_voice == "bf_emma"
    assert config.input_device_index == 4
    assert config.output_device_index == 9
    assert config.allow_barge_in is True


def test_local_voice_config_can_ignore_inherited_prompt_overrides(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "请始终使用简体中文回答。")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "也请始终使用简体中文回答。")
    monkeypatch.setenv("BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT", "1")

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            vision=False,
            vision_model=None,
            camera_device=None,
            camera_framerate=None,
            camera_max_width=None,
            camera_source=None,
            camera_helper_app=None,
            camera_helper_state_dir=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.system_prompt == default_local_speech_system_prompt(local_browser.Language.EN)
    assert "Always answer in English" in config.system_prompt
    assert "简体中文" not in config.system_prompt


def test_local_voice_config_explicit_prompt_wins_when_env_prompt_is_ignored(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "请始终使用简体中文回答。")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "也请始终使用简体中文回答。")
    monkeypatch.setenv("BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT", "1")

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt="Use English, but answer in haiku.",
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            vision=False,
            vision_model=None,
            camera_device=None,
            camera_framerate=None,
            camera_max_width=None,
            camera_source=None,
            camera_helper_app=None,
            camera_helper_state_dir=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.system_prompt == "Use English, but answer in haiku."


def test_local_voice_config_resolves_openai_responses_llm_without_changing_stt_tts(
    monkeypatch,
):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_MODEL", "gpt-voice-demo")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "flex")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ignored.test/v1")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "ignored prompt")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("BLINK_LOCAL_STT_BACKEND", "mlx-whisper")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.llm == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-voice-demo",
        base_url="https://proxy.test/v1",
        system_prompt=default_local_speech_system_prompt(local_browser.Language.ZH),
        service_tier="flex",
    )
    assert config.stt_backend == "mlx-whisper"
    assert config.stt_model == "mlx-community/whisper-medium-mlx"
    assert config.tts_backend == "kokoro"
    assert config.tts_voice == "zf_xiaobei"


def test_local_voice_demo_mode_applies_speech_prompt_and_openai_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS", "")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=True,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.demo_mode is True
    assert config.llm_service_tier == "priority"
    assert config.llm_max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert config.llm.max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert "one to four short sentences" in config.system_prompt
    assert "Do not use markdown" in config.system_prompt


def test_local_voice_config_reads_robot_head_live_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_DRIVER", "live")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_PORT", "/dev/cu.fake-robot-head")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_BAUD", "1000000")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_HARDWARE_PROFILE_PATH", "/tmp/hardware.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM_TTL_SECONDS", "450")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.robot_head_driver == "live"
    assert config.robot_head_port == "/dev/cu.fake-robot-head"
    assert config.robot_head_baud == 1000000
    assert config.robot_head_hardware_profile_path == "/tmp/hardware.json"
    assert config.robot_head_live_arm is True
    assert config.robot_head_arm_ttl_seconds == 450


def test_local_voice_config_reads_robot_head_simulation_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_DRIVER", "simulation")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_SCENARIO", "/tmp/sim-scenario.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_REALTIME", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_TRACE_DIR", "/tmp/sim-traces")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.robot_head_driver == "simulation"
    assert str(config.robot_head_sim_scenario_path) == "/tmp/sim-scenario.json"
    assert config.robot_head_sim_realtime is True
    assert str(config.robot_head_sim_trace_dir) == "/tmp/sim-traces"


def test_local_voice_config_ignores_removed_legacy_env(monkeypatch):
    monkeypatch.setenv("LEGACY_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("LEGACY_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "local-http-wav")
    monkeypatch.setenv("LEGACY_LOCAL_AUDIO_INPUT_DEVICE", "4")
    monkeypatch.setenv("BLINK_LOCAL_AUDIO_INPUT_DEVICE", "8")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.ZH
    assert config.tts_backend == "local-http-wav"
    assert config.input_device_index == 8


def test_local_voice_config_prefers_chinese_voice_override(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_ZH", "zf_xiaoyi")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.tts_voice == "zf_xiaoyi"
    assert config.language == local_browser.Language.ZH


def test_local_voice_config_switches_fallback_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "")
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend="whisper",
            stt_model=None,
            tts_backend="piper",
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.stt_model == "Systran/faster-distil-whisper-medium.en"
    assert config.tts_voice == "en_US-ryan-high"
    assert config.language == local_browser.Language.EN


def test_local_voice_config_defaults_to_chinese_kokoro(monkeypatch):
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "")
    monkeypatch.delenv("BLINK_LOCAL_LANGUAGE", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.ZH
    assert config.stt_model == "mlx-community/whisper-medium-mlx"
    assert config.tts_backend == "kokoro"
    assert config.tts_voice == "zf_xiaobei"
    assert config.tts_base_url is None
    assert config.system_prompt == default_local_speech_system_prompt(local_browser.Language.ZH)
    assert config.tts_backend_locked is False
    assert config.tts_voice_override is None


def test_resolve_preferred_audio_device_indexes_prefers_macbook_on_display_defaults(monkeypatch):
    devices = [
        AudioDeviceInfo(0, "LG UltraFine Display Audio", 1, 0, 48000),
        AudioDeviceInfo(1, "LG UltraFine Display Audio", 0, 2, 48000),
        AudioDeviceInfo(2, "MacBook Pro Microphone", 1, 0, 48000),
        AudioDeviceInfo(3, "MacBook Pro Speakers", 0, 2, 48000),
    ]

    monkeypatch.setattr(local_common.sys, "platform", "darwin")
    monkeypatch.setattr(local_common, "get_audio_devices", lambda: devices)
    monkeypatch.setattr(
        local_common,
        "get_default_audio_device",
        lambda kind: devices[0] if kind == "input" else devices[1],
    )

    input_index, output_index = local_common.resolve_preferred_audio_device_indexes(None, None)

    assert input_index == 2
    assert output_index == 3


def test_resolve_preferred_audio_device_indexes_respects_explicit_selection(monkeypatch):
    monkeypatch.setattr(local_common.sys, "platform", "darwin")

    input_index, output_index = local_common.resolve_preferred_audio_device_indexes(8, 9)

    assert input_index == 8
    assert output_index == 9


def test_build_local_voice_runtime_accepts_injected_processors():
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
    )

    task, context = build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    assert task is not None
    assert context.messages == []


def test_build_local_voice_runtime_defaults_to_protected_playback(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_local_voice_task(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(local_voice, "build_local_voice_task", fake_build_local_voice_task)

    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
    )

    build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    assert captured["mute_during_bot_speech"] is True
    assert any(
        isinstance(processor, BrainExpressionVoicePolicyProcessor)
        for processor in captured["pre_tts_processors"]
    )


def test_build_local_voice_runtime_allows_barge_in_when_requested(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_local_voice_task(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(local_voice, "build_local_voice_task", fake_build_local_voice_task)

    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=True,
    )

    build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    assert captured["mute_during_bot_speech"] is False


@pytest.mark.asyncio
async def test_local_audio_output_interrupt_stops_and_restarts_stream():
    pytest.importorskip("pyaudio")
    from blink.transports.local.audio import LocalAudioOutputTransport, LocalAudioTransportParams

    class FakePyAudio:
        pass

    class FakeStream:
        def __init__(self):
            self.active = True
            self.calls = []

        def is_active(self):
            self.calls.append("is_active")
            return self.active

        def stop_stream(self):
            self.calls.append("stop_stream")
            self.active = False

        def start_stream(self):
            self.calls.append("start_stream")
            self.active = True

    transport = LocalAudioOutputTransport(FakePyAudio(), LocalAudioTransportParams())
    transport.get_event_loop = asyncio.get_running_loop
    stream = FakeStream()
    transport._out_stream = stream

    await transport.interrupt_audio_output()

    assert stream.calls == ["is_active", "stop_stream", "is_active", "start_stream"]
    assert stream.active is True
    transport._out_stream = None
    transport._executor.shutdown(wait=False, cancel_futures=True)
    transport._control_executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_local_audio_output_write_failure_is_nonfatal():
    pytest.importorskip("pyaudio")
    from blink.transports.local.audio import LocalAudioOutputTransport, LocalAudioTransportParams

    class FakePyAudio:
        pass

    class FailingStream:
        def write(self, _audio):
            raise RuntimeError("output stream interrupted")

    transport = LocalAudioOutputTransport(FakePyAudio(), LocalAudioTransportParams())
    transport.get_event_loop = asyncio.get_running_loop
    transport._out_stream = FailingStream()

    wrote = await transport.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
    )

    assert wrote is False
    transport._out_stream = None
    transport._executor.shutdown(wait=False, cancel_futures=True)
    transport._control_executor.shutdown(wait=False, cancel_futures=True)


def test_build_local_voice_task_can_disable_rtvi_user_mute(monkeypatch):
    captured: dict[str, object] = {}

    def fake_pipeline_task(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(local_common, "PipelineTask", fake_pipeline_task)

    task, context = local_common.build_local_voice_task(
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
        rtvi_user_mute_enabled=False,
    )

    assert task is not None
    assert context.messages == []
    assert captured["rtvi_observer_params"].user_mute_enabled is False


def test_build_local_voice_runtime_uses_provider_aware_llm_factory(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_local_llm_service(llm_config):
        captured["llm"] = llm_config
        return DummyLLMProcessor()

    monkeypatch.setattr(local_voice, "create_local_llm_service", fake_create_local_llm_service)

    config = LocalVoiceConfig(
        base_url=None,
        model="gpt-voice-demo",
        system_prompt="Voice prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
        llm_provider="openai-responses",
        llm_service_tier="flex",
    )

    build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        tts=FrameProcessor(name="tts"),
    )

    assert captured["llm"] == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-voice-demo",
        base_url=None,
        system_prompt="",
        service_tier="flex",
    )


def test_local_voice_config_defaults_to_protected_playback(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_ALLOW_BARGE_IN", raising=False)

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.allow_barge_in is False


def test_local_voice_config_protected_playback_overrides_barge_in_env(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "1")

    config = resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language="en",
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend="kokoro",
            tts_voice=None,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            protected_playback=True,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.allow_barge_in is False


def test_local_voice_config_profile_applies_native_english_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.setenv("BLINK_LOCAL_CAMERA_SOURCE", "")
    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "请只用中文回答")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "请只用中文回答")

    config = resolve_config(
        local_voice.build_parser().parse_args(["--config-profile", "native-en-kokoro"])
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_backend == "kokoro"
    assert config.camera_source == CAMERA_SOURCE_NONE
    assert config.allow_barge_in is False
    assert config.config_profile == "native-en-kokoro"
    assert "中文" not in config.system_prompt


def test_local_voice_config_native_barge_in_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "")

    default_config = resolve_config(
        local_voice.build_parser().parse_args(["--config-profile", "native-en-kokoro"])
    )
    cli_config = resolve_config(
        local_voice.build_parser().parse_args(
            ["--config-profile", "native-en-kokoro", "--allow-barge-in"]
        )
    )

    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "1")
    env_config = resolve_config(
        local_voice.build_parser().parse_args(["--config-profile", "native-en-kokoro"])
    )
    protected_config = resolve_config(
        local_voice.build_parser().parse_args(
            ["--config-profile", "native-en-kokoro", "--protected-playback"]
        )
    )

    assert default_config.allow_barge_in is False
    assert cli_config.allow_barge_in is True
    assert env_config.allow_barge_in is True
    assert protected_config.allow_barge_in is False


def test_local_voice_config_disables_native_camera_vision_requests(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_VOICE_VISION", raising=False)

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language="en",
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend="kokoro",
            tts_voice=None,
            vision=True,
            vision_model="test/moondream",
            camera_device=2,
            camera_framerate=0.5,
            camera_max_width=320,
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_backend == "kokoro"
    assert config.vision_enabled is False
    assert config.vision_requested_but_disabled is True
    assert config.vision_model == "test/moondream"
    assert config.camera_device_index == 2
    assert config.camera_framerate == 0.5
    assert config.camera_max_width == 320


def test_local_voice_config_resolves_macos_camera_helper(monkeypatch, tmp_path):
    monkeypatch.setattr(local_voice.sys, "platform", "darwin")
    state_dir = tmp_path / "camera-state"
    app_path = tmp_path / "BlinkCameraHelper.app"

    config = resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language="en",
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend="kokoro",
            tts_voice=None,
            vision=False,
            vision_model="test/moondream",
            camera_device=None,
            camera_framerate=0.5,
            camera_max_width=320,
            camera_source="macos-helper",
            camera_helper_app=str(app_path),
            camera_helper_state_dir=str(state_dir),
            input_device=None,
            output_device=None,
            allow_barge_in=False,
            list_audio_devices=False,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_backend == "kokoro"
    assert config.vision_enabled is True
    assert config.camera_source == CAMERA_SOURCE_MACOS_HELPER
    assert config.camera_helper_app_path == app_path
    assert config.camera_helper_state_dir == state_dir
    assert config.camera_framerate == 0.5
    assert config.camera_max_width == 320


@pytest.mark.parametrize(
    ("language", "tts_backend", "platform_name", "expected"),
    [
        ("zh", "kokoro", "darwin", "English-only"),
        ("en", "local-http-wav", "darwin", "requires Kokoro"),
        ("en", "kokoro", "linux", "only available on macOS"),
    ],
)
def test_local_voice_config_rejects_invalid_macos_camera_helper(
    monkeypatch,
    language,
    tts_backend,
    platform_name,
    expected,
):
    monkeypatch.setattr(local_voice.sys, "platform", platform_name)

    with pytest.raises(local_common.LocalDependencyError) as excinfo:
        resolve_config(
            argparse.Namespace(
                llm_provider=None,
                model=None,
                base_url=None,
                system_prompt=None,
                language=language,
                temperature=None,
                max_output_tokens=None,
                demo_mode=None,
                stt_backend=None,
                stt_model=None,
                tts_backend=tts_backend,
                tts_voice=None,
                vision=False,
                vision_model=None,
                camera_device=None,
                camera_framerate=None,
                camera_max_width=None,
                camera_source="macos-helper",
                camera_helper_app=None,
                camera_helper_state_dir=None,
                input_device=None,
                output_device=None,
                allow_barge_in=False,
                list_audio_devices=False,
                verbose=False,
            )
        )

    assert expected in str(excinfo.value)


def test_parse_macos_camera_helper_status_accepts_valid_payload(tmp_path):
    frame_path = tmp_path / "latest.rgb"
    frame_path.write_bytes(b"\x00\x00\x00" * 4)

    status = _parse_macos_camera_helper_status(
        {
            "state": "running",
            "updated_at": datetime.now(UTC).isoformat(),
            "frame_seq": 3,
            "frame_path": "latest.rgb",
            "width": 2,
            "height": 2,
            "format": "RGB",
            "pid": 123,
            "reason_codes": ["waiting-for-first-frame"],
        },
        state_dir=tmp_path,
    )

    assert status.frame_seq == 3
    assert status.frame_path == frame_path
    assert status.width == 2
    assert status.height == 2
    assert status.format == "RGB"
    assert status.pid == 123
    assert status.reason_codes == ("waiting_for_first_frame",)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, "helper_status_malformed"),
        (
            {"state": "denied", "reason_codes": ["camera_permission_denied"]},
            "camera_permission_denied",
        ),
        (
            {
                "state": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "frame_seq": 1,
                "frame_path": "latest.rgb",
                "width": 2,
                "height": 2,
                "format": "RGB",
                "pid": 123,
                "reason_codes": [],
            },
            "helper_frame_stale",
        ),
        (
            {
                "state": "running",
                "updated_at": datetime.now(UTC).isoformat(),
                "frame_seq": 1,
                "frame_path": "../secret",
                "width": 2,
                "height": 2,
                "format": "RGB",
                "pid": 123,
                "reason_codes": [],
            },
            "helper_frame_path_invalid",
        ),
    ],
)
def test_parse_macos_camera_helper_status_rejects_unsafe_payloads(tmp_path, payload, expected):
    with pytest.raises(ValueError) as excinfo:
        _parse_macos_camera_helper_status(payload, state_dir=tmp_path)

    assert str(excinfo.value) == expected


@pytest.mark.asyncio
async def test_macos_camera_helper_snapshot_provider_reads_fresh_frame(tmp_path):
    (tmp_path / "latest.rgb").write_bytes(b"\x01\x02\x03" * 4)
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "updated_at": datetime.now(UTC).isoformat(),
                "frame_seq": 1,
                "frame_path": "latest.rgb",
                "width": 2,
                "height": 2,
                "format": "RGB",
                "pid": 123,
                "reason_codes": [],
            }
        ),
        encoding="utf-8",
    )
    frame_buffer = NativeCameraFrameBuffer()
    provider = MacOSCameraHelperSnapshotProvider(
        frame_buffer=frame_buffer,
        state_dir=tmp_path,
        launch_helper=False,
    )

    assert await provider.capture_once() is True
    assert frame_buffer.latest_camera_frame is not None
    assert frame_buffer.latest_camera_frame.transport_source == "macos-camera-helper"
    assert frame_buffer.latest_camera_frame.size == (2, 2)
    assert frame_buffer.latest_camera_frame.image == b"\x01\x02\x03" * 4


@pytest.mark.asyncio
async def test_native_camera_snapshot_provider_updates_buffer():
    frame_buffer = NativeCameraFrameBuffer()
    frame = UserImageRawFrame(
        user_id="test-native-camera",
        image=b"\x00\x00\x00" * 4,
        size=(2, 2),
        format="RGB",
    )
    frame.transport_source = "native-camera"
    provider = NativeCameraSnapshotProvider(
        frame_buffer=frame_buffer,
        capture_frame=lambda: frame,
    )

    assert await provider.capture_once() is True
    assert frame_buffer.latest_camera_frame is frame
    assert frame_buffer.latest_camera_frame_is_fresh(max_age_secs=1.0) is True


@pytest.mark.asyncio
async def test_native_camera_snapshot_provider_uses_event_loop_thread_for_avfoundation():
    class FakeCapture:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class ThreadCheckingProvider(NativeCameraSnapshotProvider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.open_thread_id = None
            self.capture_thread_id = None
            self.fake_capture = FakeCapture()

        def _ensure_opencv_capture(self):
            self.open_thread_id = threading.get_ident()
            self._opencv_capture = self.fake_capture

        def _capture_opencv_frame(self):
            self.capture_thread_id = threading.get_ident()
            return None

    frame_buffer = NativeCameraFrameBuffer()
    provider = ThreadCheckingProvider(frame_buffer=frame_buffer)
    event_loop_thread_id = threading.get_ident()

    await provider.start()
    await provider.capture_once()
    await provider.close()

    assert provider.open_thread_id == event_loop_thread_id
    assert provider.capture_thread_id == event_loop_thread_id
    assert provider.fake_capture.released is True


def test_native_camera_open_failure_reports_retired_opencv_path(monkeypatch):
    class ClosedCapture:
        release_count = 0

        def isOpened(self):
            return False

        def release(self):
            self.release_count += 1

    class FakeCV2:
        CAP_AVFOUNDATION = 99

        def __init__(self):
            self.open_calls = []
            self.capture = ClosedCapture()

        def VideoCapture(self, *args):
            self.open_calls.append(args)
            return self.capture

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(local_voice.sys, "platform", "darwin")

    provider = NativeCameraSnapshotProvider(frame_buffer=NativeCameraFrameBuffer())

    with pytest.raises(local_common.LocalDependencyError) as excinfo:
        provider._ensure_opencv_capture()

    message = str(excinfo.value)
    assert len(fake_cv2.open_calls) == 1
    assert "retired OpenCV native voice camera path is disabled" in message
    assert "--camera-source macos-helper" in message
    assert "BlinkCameraHelper.app owns macOS camera permission" in message


def test_macos_camera_launcher_label_detects_app_parent(monkeypatch):
    ps_rows = {
        "10": "10 9 /Users/example/project/Blink/.venv/bin/python3",
        "9": "9 8 uv",
        "8": "8 7 /Applications/Codex.app/Contents/Resources/codex",
        "7": "7 1 /Applications/Codex.app/Contents/MacOS/Codex",
    }

    monkeypatch.setattr(local_voice.sys, "platform", "darwin")
    monkeypatch.setattr(local_voice.os, "getpid", lambda: 10)

    def fake_check_output(command, **_kwargs):
        return ps_rows[command[2]]

    monkeypatch.setattr(local_voice.subprocess, "check_output", fake_check_output)

    assert local_voice._macos_camera_launcher_label() == (
        " Detected launcher app: /Applications/Codex.app."
    )


def test_build_local_voice_runtime_rejects_native_camera_tool():
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="English voice prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
        vision_enabled=True,
        vision_model="test/moondream",
    )

    with pytest.raises(local_common.LocalDependencyError) as excinfo:
        build_local_voice_runtime(
            config,
            transport=DummyTransport(),
            stt=FrameProcessor(name="stt"),
            llm=DummyLLMProcessor(),
            tts=DummyTTSProcessor(),
            vision=DummyVisionProcessor(),
        )

    assert "OpenCV native voice camera is disabled" in str(excinfo.value)
    assert "English-only mic -> MLX Whisper -> LLM -> Kokoro" in str(excinfo.value)


def test_local_voice_runtime_status_reports_native_isolation_identity():
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="English voice prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
        config_profile="native-en-kokoro",
    )

    status = local_voice._voice_runtime_status_text(
        config,
        selected_backend_label="kokoro",
        input_device_name="system default",
        output_device_name="system default",
    )

    assert "runtime=native" in status
    assert "transport=PyAudio" in status
    assert "profile=native-en-kokoro" in status
    assert "isolation=backend-only" in status
    assert "tts=kokoro" in status
    assert "protected_playback=on" in status
    assert "barge_in=off" in status
    assert "primary_browser_paths=browser-zh-melo,browser-en-kokoro" in status


def test_local_voice_runtime_status_reports_helper_camera_as_single_frame_isolation():
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="English voice prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=True,
        vision_enabled=True,
        camera_source=CAMERA_SOURCE_MACOS_HELPER,
        config_profile="native-en-kokoro-macos-camera",
    )

    status = local_voice._voice_runtime_status_text(
        config,
        selected_backend_label="kokoro",
        input_device_name="system default",
        output_device_name="system default",
    )

    assert "profile=native-en-kokoro-macos-camera" in status
    assert "isolation=backend-plus-helper-camera" in status
    assert "vision=macos-helper(" in status
    assert "protected_playback=off" in status
    assert "barge_in=on" in status


@pytest.mark.asyncio
async def test_build_local_voice_runtime_registers_macos_helper_camera_tool():
    frame_buffer = NativeCameraFrameBuffer()
    frame = UserImageRawFrame(
        user_id="macos-camera-helper",
        image=b"\x00\x00\x00" * 4,
        size=(2, 2),
        format="RGB",
    )
    frame.transport_source = "macos-camera-helper"
    provider = NativeCameraSnapshotProvider(
        frame_buffer=frame_buffer,
        capture_frame=lambda: frame,
    )
    llm = DummyLLMProcessor()
    tts = DummyTTSProcessor()
    vision = DummyVisionProcessor(response_text="A small notebook is visible.")
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="English voice prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
        vision_enabled=True,
        vision_model="test/moondream",
        camera_source=CAMERA_SOURCE_MACOS_HELPER,
    )

    task, context = build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=tts,
        vision=vision,
        camera_frame_buffer=frame_buffer,
        camera_provider=provider,
    )

    assert task is not None
    assert "fetch_user_image" in llm.registered_functions
    assert any(tool.name == "fetch_user_image" for tool in context.tools.standard_tools)
    assert getattr(context, "blink_native_camera_provider") is provider

    results = []

    async def result_callback(payload):
        results.append(payload)

    await llm.registered_functions["fetch_user_image"](
        SimpleNamespace(
            arguments={"question": "What can you see?"},
            result_callback=result_callback,
        )
    )

    assert results == [{"description": "A small notebook is visible."}]
    assert vision.received_frames
    assert vision.received_frames[0].text.startswith("Inspect the current Mac camera frame")


@pytest.mark.asyncio
async def test_build_local_voice_runtime_lazy_loads_macos_helper_vision(monkeypatch):
    frame_buffer = NativeCameraFrameBuffer()
    frame = UserImageRawFrame(
        user_id="macos-camera-helper",
        image=b"\x00\x00\x00" * 4,
        size=(2, 2),
        format="RGB",
    )
    frame.transport_source = "macos-camera-helper"
    provider = NativeCameraSnapshotProvider(
        frame_buffer=frame_buffer,
        capture_frame=lambda: frame,
    )
    llm = DummyLLMProcessor()
    tts = DummyTTSProcessor()
    created = []

    def fake_create_local_vision_service(*, model):
        created.append(model)
        return DummyVisionProcessor(response_text="A mug is visible.")

    monkeypatch.setattr(
        local_voice,
        "create_local_vision_service",
        fake_create_local_vision_service,
    )
    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="English voice prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        allow_barge_in=False,
        vision_enabled=True,
        vision_model="test/moondream",
        camera_source=CAMERA_SOURCE_MACOS_HELPER,
    )

    build_local_voice_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=tts,
        camera_frame_buffer=frame_buffer,
        camera_provider=provider,
    )

    assert created == []

    results = []

    async def result_callback(payload):
        results.append(payload)

    await llm.registered_functions["fetch_user_image"](
        SimpleNamespace(
            arguments={"question": "What can you see?"},
            result_callback=result_callback,
        )
    )

    assert created == ["test/moondream"]
    assert results == [{"description": "A mug is visible."}]


def test_build_local_browser_runtime_accepts_injected_processors():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    assert task is not None
    assert context.messages == []


def test_build_local_browser_runtime_uses_chunk_aligned_tts_contexts(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_local_tts_service(**kwargs):
        captured.update(kwargs)
        return DummyTTSProcessor()

    monkeypatch.setattr(local_browser, "create_local_tts_service", fake_create_local_tts_service)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=DummyLLMProcessor(),
        active_client={"id": "pc-123"},
    )

    assert task is not None
    assert context.messages == []
    assert captured["backend"] == "kokoro"
    assert captured["reuse_context_id_within_turn"] is False


def test_build_local_browser_runtime_plan_provider_uses_injected_actor_control_frame(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_build_local_voice_task(**kwargs):
        captured.update(kwargs)
        return object(), kwargs["context"]

    monkeypatch.setattr(local_browser, "build_local_voice_task", fake_build_local_voice_task)
    actor_frame = {
        "frame_id": "control-test",
        "sequence": 3,
        "boundary": "speech_chunk_boundary",
        "condition_cache_digest": "digest",
        "source_event_ids": [1],
        "speech_policy": {"action": "allow_next_chunk", "reason_codes": ["speech_policy:test"]},
    }
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        config_profile="browser-en-kokoro",
        tts_runtime_label="kokoro/English",
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=DummyLLMProcessor(),
        tts=DummyTTSProcessor(),
        actor_control_frame_provider=lambda: actor_frame,
    )

    voice_processor = next(
        processor
        for processor in captured["pre_tts_processors"]
        if isinstance(processor, BrainExpressionVoicePolicyProcessor)
    )
    plan = voice_processor._performance_plan_provider()
    assert plan.actor_control_ref["frame_id"] == "control-test"
    assert plan.actor_control_ref["boundary"] == "speech_chunk_boundary"


def test_build_local_browser_runtime_uses_protected_playback_without_client_track_mutation(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_build_local_voice_task(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(local_browser, "build_local_voice_task", fake_build_local_voice_task)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    strategies = captured["extra_user_mute_strategies"]
    assert isinstance(strategies, list)
    assert len(strategies) == 1
    assert isinstance(strategies[0], AlwaysUserMuteStrategy)
    assert isinstance(strategies[0], local_browser.BrowserProtectedPlaybackMuteStrategy)
    assert captured.get("mute_during_bot_speech", False) is False
    assert captured.get("user_turn_start_strategies") is None
    assert "rtvi_user_mute_enabled" not in captured
    assert any(
        isinstance(processor, BrainExpressionVoicePolicyProcessor)
        for processor in captured["pre_tts_processors"]
    )
    assert any(
        isinstance(processor, BrainVoiceInputHealthProcessor) and processor.phase == "pre_stt"
        for processor in captured["pre_stt_processors"]
    )
    assert any(
        isinstance(processor, BrainVoiceInputHealthProcessor) and processor.phase == "post_stt"
        for processor in captured["post_stt_processors"]
    )


def test_build_local_browser_runtime_allows_barge_in_when_requested(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_local_voice_task(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(local_browser, "build_local_voice_task", fake_build_local_voice_task)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        allow_barge_in=True,
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    assert captured.get("mute_during_bot_speech", False) is False
    assert captured["extra_user_mute_strategies"] == []
    assert len(captured["user_turn_start_strategies"]) == 1
    assert isinstance(
        captured["user_turn_start_strategies"][0],
        local_browser.BrowserBargeInTurnStartStrategy,
    )


@pytest.mark.asyncio
async def test_build_local_browser_runtime_lazy_loads_browser_vision(monkeypatch):
    created: list[str] = []
    cached_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"abc",
        size=(2, 2),
        format="RGB",
    )
    cached_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = cached_frame
            self.latest_camera_frame_seq = 1
            self.latest_camera_frame_received_monotonic = asyncio.get_running_loop().time()
            self.latest_camera_frame_received_at = datetime.now(UTC).isoformat()

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    def fake_create_local_vision_service(*, model):
        created.append(model)
        return DummyVisionProcessor(response_text="A cup is visible.")

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)
    monkeypatch.setattr(
        local_browser,
        "create_local_vision_service",
        fake_create_local_vision_service,
    )

    llm = DummyLLMProcessor()
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="test/moondream",
        continuous_perception_enabled=False,
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        active_client={"id": "pc-123"},
    )

    assert created == []

    results = []

    async def result_callback(result):
        results.append(result)

    await llm.registered_functions["fetch_user_image"](
        SimpleNamespace(
            arguments={"question": "What can you see?"},
            function_name="fetch_user_image",
            tool_call_id="tool-lazy",
            llm=llm,
            result_callback=result_callback,
        )
    )

    assert created == ["test/moondream"]
    assert results == [{"description": "Based on the latest still frame: A cup is visible."}]


def test_build_local_browser_runtime_uses_provider_aware_llm_factory(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_local_llm_service(llm_config):
        captured["llm"] = llm_config
        return DummyLLMProcessor()

    monkeypatch.setattr(local_browser, "create_local_llm_service", fake_create_local_llm_service)

    config = LocalBrowserConfig(
        base_url=None,
        model="gpt-browser-demo",
        system_prompt="Browser prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        llm_provider="openai-responses",
        llm_service_tier="flex",
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        tts=FrameProcessor(name="tts"),
    )

    assert captured["llm"] == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-browser-demo",
        base_url=None,
        system_prompt="",
        service_tier="flex",
    )


def test_build_local_browser_runtime_vision_uses_protected_playback_without_client_track_mutation(
    monkeypatch,
):
    captured: dict[str, object] = {}
    real_build_local_user_aggregators = local_browser.build_local_user_aggregators

    def fake_build_local_user_aggregators(context, **kwargs):
        captured.update(kwargs)
        return real_build_local_user_aggregators(context, **kwargs)

    def fake_pipeline_task(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        local_browser, "build_local_user_aggregators", fake_build_local_user_aggregators
    )
    monkeypatch.setattr(local_browser, "PipelineTask", fake_pipeline_task)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=DummyLLMProcessor(),
        tts=DummyTTSProcessor(),
        vision=DummyVisionProcessor(),
        active_client={"id": "pc-123"},
    )

    strategies = captured["extra_user_mute_strategies"]
    assert isinstance(strategies, list)
    assert len(strategies) == 1
    assert isinstance(strategies[0], AlwaysUserMuteStrategy)
    assert isinstance(strategies[0], local_browser.BrowserProtectedPlaybackMuteStrategy)
    assert captured.get("mute_during_bot_speech", False) is False
    assert captured.get("user_turn_start_strategies") is None
    assert "rtvi_observer_params" not in captured


def test_build_local_browser_runtime_vision_uses_provider_aware_llm_factory(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_local_llm_service(llm_config):
        captured["llm"] = llm_config
        return DummyLLMProcessor()

    monkeypatch.setattr(local_browser, "create_local_llm_service", fake_create_local_llm_service)

    config = LocalBrowserConfig(
        base_url=None,
        model="gpt-browser-demo",
        system_prompt="Browser prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
        llm_provider="openai-responses",
        llm_service_tier="flex",
    )

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        tts=DummyTTSProcessor(),
        vision=DummyVisionProcessor(),
        active_client={"id": "pc-123"},
    )

    assert captured["llm"] == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-browser-demo",
        base_url=None,
        system_prompt="",
        service_tier="flex",
    )


def test_build_local_browser_runtime_keeps_timeout_stop_fallback():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    task, _ = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    user_aggregator = next(
        processor
        for processor in task._pipeline.processors[2].processors
        if hasattr(processor, "_params") and hasattr(processor._params, "user_turn_strategies")
    )
    strategies = user_aggregator._params.user_turn_strategies.stop
    assert strategies is not None
    assert len(strategies) == 2
    assert isinstance(strategies[0], TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(strategies[1], SpeechTimeoutUserTurnStopStrategy)


def test_local_browser_runtime_reports_public_voice_input_health():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    _task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    runtime = context.blink_brain_runtime
    runtime.note_voice_input_connected(True)
    runtime.note_voice_input_audio_frame()
    runtime.note_voice_input_speech_started()
    runtime.note_voice_input_interim_transcription("这是秘密部分转写")
    runtime.note_voice_input_speech_stopped()
    runtime.note_voice_input_transcription("这是秘密转写文本")
    health = runtime.current_voice_input_health()
    active = runtime.current_active_listening_state()

    assert health["available"] is True
    assert health["microphone_state"] == "receiving"
    assert health["stt_state"] == "transcribed"
    assert health["audio_frame_count"] == 1
    assert health["speech_start_count"] == 1
    assert health["speech_stop_count"] == 1
    assert health["interim_transcription_count"] == 1
    assert health["last_partial_transcription_chars"] == len("这是秘密部分转写")
    assert health["partial_transcript_available"] is True
    assert health["transcription_count"] == 1
    assert health["last_transcription_chars"] == len("这是秘密转写文本")
    assert "这是秘密转写文本" not in str(health)
    assert "这是秘密部分转写" not in str(health)
    assert "stt:transcribed" in health["reason_codes"]
    assert active["phase"] == "final_transcript"
    assert active["partial_available"] is True
    assert active["final_transcript_chars"] == len("这是秘密转写文本")


def test_local_browser_active_listening_accumulates_final_stt_fragments():
    _task, context = build_local_browser_runtime(
        LocalBrowserConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3.5:4b",
            system_prompt="Test prompt",
            language=local_browser.Language.EN,
            stt_backend="mlx-whisper",
            tts_backend="kokoro",
            stt_model="mlx-community/whisper-medium-mlx",
            tts_voice="af_heart",
            tts_base_url=None,
            host="127.0.0.1",
            port=7860,
            config_profile="browser-en-kokoro",
        ),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    runtime = context.blink_brain_runtime
    first = "Camera honesty is important."
    second = "Kokoro should finish long speech."
    runtime.note_voice_input_connected(True)
    runtime.note_voice_input_speech_started()
    runtime.note_voice_input_speech_stopped()
    runtime.note_voice_input_transcription(first)
    runtime.note_voice_input_transcription(second)

    health = runtime.current_voice_input_health()
    active = runtime.current_active_listening_state()
    actor_active = runtime.current_active_listener_state_v2(profile="browser-en-kokoro")

    assert health["last_transcription_chars"] == len(second)
    assert active["partial_available"] is False
    assert active["final_transcript_chars"] == len(first) + len(second)
    assert actor_active["partial_available"] is False
    assert actor_active["final_transcript_chars"] == len(first) + len(second)


def test_voice_input_health_processors_split_pre_and_post_stt_observation():
    calls: list[tuple[str, object]] = []

    class Runtime:
        def note_voice_input_audio_frame(self):
            calls.append(("audio", None))

        def note_voice_input_speech_started(self):
            calls.append(("speech_started", None))

        def note_voice_input_speech_stopped(self):
            calls.append(("speech_stopped", None))

        def note_voice_input_transcription(self, text):
            calls.append(("transcription", text))

        def note_voice_input_interim_transcription(self, text):
            calls.append(("interim", text))

        def note_voice_input_stt_error(self, error_type):
            calls.append(("error", error_type))

    pre = BrainVoiceInputHealthProcessor(runtime=Runtime(), phase="pre_stt")
    post = BrainVoiceInputHealthProcessor(runtime=Runtime(), phase="post_stt")

    pre._record_frame(AudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1))
    pre._record_frame(VADUserStartedSpeakingFrame())
    pre._record_frame(VADUserStoppedSpeakingFrame())
    pre._record_frame(TranscriptionFrame(text="ignored", user_id="user", timestamp="now"))
    post._record_frame(AudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1))
    post._record_frame(InterimTranscriptionFrame(text="秘密部分转写", user_id="user", timestamp="now"))
    post._record_frame(TranscriptionFrame(text="秘密转写", user_id="user", timestamp="now"))
    post._record_frame(ErrorFrame("raw private error"))

    assert calls == [
        ("audio", None),
        ("speech_started", None),
        ("speech_stopped", None),
        ("interim", "秘密部分转写"),
        ("transcription", "秘密转写"),
        ("error", "ErrorFrame"),
    ]


def test_local_browser_runtime_reports_stt_wait_without_raw_transcript():
    _task, context = build_local_browser_runtime(
        LocalBrowserConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3.5:4b",
            system_prompt="Test prompt",
            language=local_browser.Language.ZH,
            stt_backend="mlx-whisper",
            tts_backend="kokoro",
            stt_model="mlx-community/whisper-medium-mlx",
            tts_voice="af_heart",
            tts_base_url=None,
            host="127.0.0.1",
            port=7860,
        ),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=FrameProcessor(name="llm"),
        tts=FrameProcessor(name="tts"),
    )

    runtime = context.blink_brain_runtime
    runtime.note_voice_input_connected(True)
    runtime.note_voice_input_audio_frame()
    runtime.note_voice_input_speech_started()
    runtime.note_voice_input_speech_stopped()
    runtime._voice_input_waiting_since_monotonic = time.monotonic() - 6.0

    health = runtime.current_voice_input_health()

    assert health["microphone_state"] == "receiving"
    assert health["stt_state"] == "waiting"
    assert health["stt_waiting_too_long"] is True
    assert health["stt_wait_age_ms"] >= 5000
    assert "voice_input:mic_receiving_but_stt_waiting" in health["reason_codes"]
    assert "stt:waiting_too_long" in health["reason_codes"]
    assert "raw private error" not in str(health)


def test_local_browser_config_reads_vision_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")
    monkeypatch.setenv("BLINK_LOCAL_VISION_MODEL", "custom-vision-model")
    monkeypatch.delenv("BLINK_LOCAL_LANGUAGE", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.vision_enabled is True
    assert config.continuous_perception_enabled is True
    assert config.vision_model == "custom-vision-model"
    assert config.language == local_browser.Language.ZH
    assert config.stt_model == "mlx-community/whisper-medium-mlx"
    assert config.tts_backend == "kokoro"
    assert config.tts_voice == "zf_xiaobei"
    assert config.tts_base_url is None
    assert config.system_prompt == default_local_speech_system_prompt(local_browser.Language.ZH)
    assert config.llm_provider == "ollama"
    assert config.llm == LocalLLMConfig(
        provider="ollama",
        model="qwen3.5:4b",
        base_url="http://127.0.0.1:11434/v1",
        system_prompt=default_local_speech_system_prompt(local_browser.Language.ZH),
    )
    assert config.tts_backend_locked is False
    assert config.tts_voice_override is None


def test_local_browser_config_can_ignore_inherited_prompt_overrides(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT", "1")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "请只用中文回答")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "请只用中文回答")

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language="en",
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend="kokoro",
            tts_voice=None,
            allow_barge_in=False,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            continuous_perception=None,
            continuous_perception_interval_secs=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_operator_mode=False,
            robot_head_sim_scenario_path=None,
            robot_head_sim_realtime=False,
            robot_head_sim_trace_dir=None,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.EN
    assert config.system_prompt == default_local_speech_system_prompt(local_browser.Language.EN)
    assert "中文" not in config.system_prompt
    assert config.llm.system_prompt == default_local_speech_system_prompt(
        local_browser.Language.EN
    )


def test_local_browser_config_profile_applies_english_kokoro_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "")
    monkeypatch.setenv("BLINK_LOCAL_CONTINUOUS_PERCEPTION", "")
    monkeypatch.setenv("BLINK_LOCAL_ALLOW_BARGE_IN", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "请只用中文回答")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "请只用中文回答")

    config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_backend == "kokoro"
    assert config.vision_enabled is True
    assert config.continuous_perception_enabled is False
    assert config.allow_barge_in is False
    assert "中文" not in config.system_prompt
    assert config.llm.system_prompt == default_local_speech_system_prompt(
        local_browser.Language.EN
    )


def test_local_browser_config_env_can_override_profile_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")

    config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )

    assert config.language == local_browser.Language.ZH
    assert config.tts_backend == "kokoro"
    assert config.vision_enabled is True


def test_local_browser_config_profile_allows_explicit_no_vision(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "")
    monkeypatch.setenv("BLINK_LOCAL_CONTINUOUS_PERCEPTION", "")

    config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            ["--config-profile", "browser-en-kokoro", "--no-vision"]
        )
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_backend == "kokoro"
    assert config.vision_enabled is False
    assert config.continuous_perception_enabled is False


def test_local_browser_config_defaults_to_protected_playback(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_ALLOW_BARGE_IN", raising=False)

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            allow_barge_in=False,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            continuous_perception=None,
            continuous_perception_interval_secs=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_operator_mode=False,
            robot_head_sim_scenario_path=None,
            robot_head_sim_realtime=False,
            robot_head_sim_trace_dir=None,
            verbose=False,
        )
    )

    assert config.allow_barge_in is False


def test_local_browser_config_allows_barge_in_when_requested(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_ALLOW_BARGE_IN", raising=False)

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            allow_barge_in=True,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            continuous_perception=None,
            continuous_perception_interval_secs=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_operator_mode=False,
            robot_head_sim_scenario_path=None,
            robot_head_sim_realtime=False,
            robot_head_sim_trace_dir=None,
            verbose=False,
        )
    )

    assert config.allow_barge_in is True


def test_local_browser_config_resolves_openai_responses_llm_without_changing_surfaces(
    monkeypatch,
):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_MODEL", "gpt-browser-demo")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "flex")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ignored.test/v1")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "ignored prompt")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("BLINK_LOCAL_STT_BACKEND", "mlx-whisper")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")

    config = local_browser.resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.llm == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-browser-demo",
        base_url="https://proxy.test/v1",
        system_prompt=default_local_speech_system_prompt(local_browser.Language.ZH),
        service_tier="flex",
    )
    assert config.stt_backend == "mlx-whisper"
    assert config.stt_model == "mlx-community/whisper-medium-mlx"
    assert config.tts_backend == "kokoro"
    assert config.tts_voice == "zf_xiaobei"
    assert config.vision_enabled is True
    assert config.continuous_perception_enabled is True


def test_local_browser_demo_mode_applies_speech_prompt_and_openai_defaults(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS", "")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    config = local_browser.resolve_config(
        argparse.Namespace(
            llm_provider=None,
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            max_output_tokens=None,
            demo_mode=True,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.demo_mode is True
    assert config.llm_service_tier == "priority"
    assert config.llm_max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert config.llm.max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert "one to four short sentences" in config.system_prompt
    assert "Do not use markdown" in config.system_prompt
    assert config.stt_backend == "mlx-whisper"
    assert config.tts_backend == "kokoro"


def test_local_browser_vision_presence_disconnect_errors_are_suppressed():
    class ClosingRuntime:
        def note_vision_connected(self, connected):
            assert connected is False
            raise RuntimeError("bad parameter or other API misuse")

    reason_codes = local_browser._safe_note_vision_connected(ClosingRuntime(), False)

    assert reason_codes == ("vision_presence_update_suppressed:RuntimeError",)


@pytest.mark.asyncio
async def test_local_browser_disconnect_cleanup_step_suppresses_errors():
    called = []

    async def failing_cleanup():
        called.append("failing")
        raise RuntimeError("bad parameter or other API misuse")

    async def ok_cleanup():
        called.append("ok")

    failed = await local_browser._run_browser_disconnect_cleanup_step(
        "camera_health_close",
        failing_cleanup,
    )
    ok = await local_browser._run_browser_disconnect_cleanup_step("task_cancel", ok_cleanup)

    assert failed == ("browser_disconnect_cleanup_suppressed:camera_health_close:RuntimeError",)
    assert ok == ("browser_disconnect_cleanup_ok:task_cancel",)
    assert called == ["failing", "ok"]


def test_local_mlx_whisper_stt_sets_explicit_turn_latency():
    stt = local_common.create_local_stt_service(
        backend="mlx-whisper",
        model="mlx-community/whisper-medium-mlx",
        language=local_common.Language.ZH,
    )

    assert stt._ttfs_p99_latency == WHISPER_TTFS_P99


def test_local_browser_config_reads_robot_head_live_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_DRIVER", "live")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_PORT", "/dev/cu.fake-robot-head")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_BAUD", "1000000")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_HARDWARE_PROFILE_PATH", "/tmp/hardware.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_ARM_TTL_SECONDS", "480")

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.robot_head_driver == "live"
    assert config.robot_head_port == "/dev/cu.fake-robot-head"
    assert config.robot_head_baud == 1000000
    assert config.robot_head_hardware_profile_path == "/tmp/hardware.json"
    assert config.robot_head_live_arm is True
    assert config.robot_head_arm_ttl_seconds == 480


def test_local_browser_config_allows_explicitly_disabling_continuous_perception(monkeypatch):
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")
    monkeypatch.delenv("BLINK_LOCAL_CONTINUOUS_PERCEPTION", raising=False)

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            robot_head_operator_mode=False,
            robot_head_sim_scenario=None,
            robot_head_sim_realtime=False,
            robot_head_sim_trace_dir=None,
            vision=False,
            vision_model=None,
            continuous_perception=False,
            continuous_perception_interval_secs=None,
            verbose=False,
        )
    )

    assert config.vision_enabled is True
    assert config.continuous_perception_enabled is False


def test_local_browser_server_start_resets_stale_visual_state(monkeypatch, tmp_path):
    from blink.brain.store import BrainStore

    db_path = tmp_path / "brain.db"
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_DB_PATH", str(db_path))
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
    )

    local_browser._reset_browser_visual_state_on_server_start(config)

    store = BrainStore(path=db_path)
    try:
        body = store.get_body_state_projection(scope_key="browser:presence")
    finally:
        store.close()

    assert body.vision_enabled is True
    assert body.vision_connected is False
    assert body.camera_track_state == "disconnected"
    assert body.sensor_health_reason == "camera_disconnected"
    assert body.last_fresh_frame_at is None
    assert body.recovery_in_progress is False
    assert body.recovery_attempts == 0


def test_local_browser_config_reads_robot_head_simulation_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_DRIVER", "simulation")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_SCENARIO", "/tmp/browser-sim.json")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_REALTIME", "1")
    monkeypatch.setenv("BLINK_LOCAL_ROBOT_HEAD_SIM_TRACE_DIR", "/tmp/browser-sim-traces")

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            robot_head_driver=None,
            robot_head_catalog_path=None,
            robot_head_port=None,
            robot_head_baud=None,
            robot_head_hardware_profile_path=None,
            robot_head_live_arm=False,
            robot_head_arm_ttl_seconds=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.robot_head_driver == "simulation"
    assert str(config.robot_head_sim_scenario_path) == "/tmp/browser-sim.json"
    assert config.robot_head_sim_realtime is True
    assert str(config.robot_head_sim_trace_dir) == "/tmp/browser-sim-traces"


def test_local_browser_config_ignores_removed_legacy_env(monkeypatch):
    monkeypatch.setenv("LEGACY_LOCAL_HOST", "0.0.0.0")
    monkeypatch.setenv("BLINK_LOCAL_HOST", "127.0.0.2")
    monkeypatch.setenv("LEGACY_LOCAL_PORT", "9000")
    monkeypatch.setenv("BLINK_LOCAL_PORT", "9001")
    monkeypatch.setenv("LEGACY_LOCAL_BROWSER_VISION", "")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")
    monkeypatch.setenv("LEGACY_LOCAL_VISION_MODEL", "legacy-model")
    monkeypatch.setenv("BLINK_LOCAL_VISION_MODEL", "blink-model")

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.host == "127.0.0.2"
    assert config.port == 9001
    assert config.vision_enabled is True
    assert config.vision_model == "blink-model"


def test_local_browser_config_prefers_english_voice_override(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "af_heart")

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.language == local_browser.Language.EN
    assert config.tts_voice == "af_heart"


def test_local_browser_config_uses_english_mlx_whisper_default(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)

    config = local_browser.resolve_config(
        argparse.Namespace(
            model=None,
            base_url=None,
            system_prompt=None,
            language=None,
            temperature=None,
            stt_backend="mlx-whisper",
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            host=None,
            port=None,
            vision=False,
            vision_model=None,
            verbose=False,
        )
    )

    assert config.stt_model == "mlx-community/whisper-medium-mlx"


def test_build_local_browser_runtime_registers_camera_tool_when_vision_enabled():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )

    llm = DummyLLMProcessor()
    tts = DummyTTSProcessor()
    task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=tts,
        vision=DummyVisionProcessor(),
        active_client={"id": "pc-123"},
    )

    assert task is not None
    assert "fetch_user_image" in llm.registered_functions
    assert "brain_remember_profile" in llm.registered_functions
    assert [tool.name for tool in context.tools.standard_tools] == [
        "fetch_user_image",
        "brain_remember_profile",
        "brain_remember_preference",
        "brain_remember_task",
        "brain_forget_memory",
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_list_visible_memories",
        "brain_explain_memory_continuity",
    ]
    assert getattr(context, "blink_perception_broker").enabled is False
    assert context.messages == []


@pytest.mark.asyncio
async def test_build_local_browser_runtime_treats_audio_only_session_as_camera_disconnected():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )

    _task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=DummyLLMProcessor(),
        tts=DummyTTSProcessor(),
        vision=DummyVisionProcessor(),
        active_client={"id": "pc-123", "camera_enabled": False},
    )

    manager = getattr(context, "blink_camera_health_manager")
    await manager.handle_client_connected()
    health = manager.current_health()

    assert health.camera_connected is False
    assert health.camera_track_state == "disconnected"
    assert health.sensor_health_reason == "camera_disconnected"


def test_build_local_browser_runtime_registers_camera_and_robot_head_tools_when_enabled():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        robot_head_driver="preview",
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )

    llm = DummyLLMProcessor()
    tts = DummyTTSProcessor()
    task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=tts,
        vision=DummyVisionProcessor(),
        active_client={"id": "pc-123"},
    )

    assert task is not None
    assert sorted(llm.registered_functions) == [
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_explain_memory_continuity",
        "brain_forget_memory",
        "brain_list_visible_memories",
        "brain_remember_preference",
        "brain_remember_profile",
        "brain_remember_task",
        "fetch_user_image",
        "robot_head_blink",
        "robot_head_look_left",
        "robot_head_look_right",
        "robot_head_return_neutral",
        "robot_head_status",
        "robot_head_wink_left",
        "robot_head_wink_right",
    ]
    assert [tool.name for tool in context.tools.standard_tools] == [
        "fetch_user_image",
        "brain_remember_profile",
        "brain_remember_preference",
        "brain_remember_task",
        "brain_forget_memory",
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_list_visible_memories",
        "brain_explain_memory_continuity",
        "robot_head_blink",
        "robot_head_wink_left",
        "robot_head_wink_right",
        "robot_head_look_left",
        "robot_head_look_right",
        "robot_head_return_neutral",
        "robot_head_status",
    ]
    assert any(
        processor.name == "robot-head-embodiment-policy"
        for processor in task._pipeline.processors[2].processors
    )


def test_build_local_browser_runtime_accepts_simulation_driver():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        robot_head_driver="simulation",
    )

    llm = DummyLLMProcessor()
    task, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
    )

    assert task is not None
    assert [tool.name for tool in context.tools.standard_tools][-1] == "robot_head_status"
    assert "robot_head_status" in llm.registered_functions


def test_latest_user_text_returns_last_user_message():
    context = local_browser.LLMContext(
        messages=[
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "收到"},
            {"role": "user", "content": "最后一句"},
        ]
    )

    assert _latest_user_text(context) == "最后一句"


def test_build_vision_prompt_prefers_text_reading_prompt():
    prompt = _build_vision_prompt("请读一下屏幕上的文字")
    assert "legible text" in prompt
    assert "blurry" in prompt


def test_build_vision_prompt_prioritizes_person_for_generic_queries():
    prompt = _build_vision_prompt("摄像头里有什么？")
    assert "identify the person first" in prompt


def test_vision_result_is_unusable_for_garbled_text():
    assert _vision_result_is_unusable("��的") is True
    assert _vision_result_is_unusable("clear description") is False


@pytest.mark.asyncio
async def test_latest_camera_frame_buffer_stores_camera_frame():
    processor = LatestCameraFrameBuffer()
    frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"123",
        size=(1, 1),
        format="RGB",
    )
    frame.transport_source = "camera"

    await processor.process_frame(frame, local_browser.FrameDirection.DOWNSTREAM)

    assert processor.latest_camera_frame is frame


@pytest.mark.asyncio
async def test_latest_camera_frame_buffer_waits_for_fresh_camera_frame():
    processor = LatestCameraFrameBuffer()

    async def push_frame_later():
        await asyncio.sleep(0.01)
        frame = UserImageRawFrame(
            user_id="pc-123",
            image=b"456",
            size=(1, 1),
            format="RGB",
        )
        frame.transport_source = "camera"
        await processor.process_frame(frame, local_browser.FrameDirection.DOWNSTREAM)
        return frame

    task = asyncio.create_task(push_frame_later())
    waited_frame = await processor.wait_for_latest_camera_frame(after_seq=0, timeout=0.5)
    pushed_frame = await task

    assert waited_frame is pushed_frame


@pytest.mark.asyncio
async def test_fetch_user_image_returns_error_when_camera_frame_missing():
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor()
    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client={"id": "pc-123"},
    )

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "摄像头里有什么？"},
        function_name="fetch_user_image",
        tool_call_id="tool-1",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results
    assert "error" in results[0]
    assert vision.received_frames == []


@pytest.mark.asyncio
async def test_fetch_user_image_uses_latest_camera_frame(monkeypatch):
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor(response_text="画面里有一台笔记本电脑。")
    cached_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"abc",
        size=(2, 2),
        format="RGB",
    )
    cached_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = cached_frame
            self.latest_camera_frame_seq = 1

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)

    _, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client={"id": "pc-123"},
    )
    context.add_message({"role": "user", "content": "请描述一下你现在看到的画面"})

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "我身后有什么？"},
        function_name="fetch_user_image",
        tool_call_id="tool-2",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results == [{"description": "基于刚才的一帧画面：画面里有一台笔记本电脑。"}]
    assert len(vision.received_frames) == 1
    assert vision.received_frames[0].image == b"abc"
    assert vision.received_frames[0].text == _build_vision_prompt("请描述一下你现在看到的画面")


@pytest.mark.asyncio
async def test_fetch_user_image_uses_fresh_frame_after_initial_audio_only_hint(monkeypatch):
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor(response_text="a hand is holding a notebook")
    active_client = {"id": "pc-123", "camera_enabled": False}
    cached_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"fresh-camera-frame",
        size=(2, 2),
        format="RGB",
    )
    cached_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = cached_frame
            self.latest_camera_frame_seq = 7
            self.latest_camera_frame_received_monotonic = asyncio.get_running_loop().time()
            self.latest_camera_frame_received_at = datetime.now(UTC).isoformat()

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)

    _, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client=active_client,
    )
    context.add_message({"role": "user", "content": "Can you look at what I am holding?"})

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "Can you look at what I am holding?"},
        function_name="fetch_user_image",
        tool_call_id="tool-recovered-camera",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results == [
        {"description": "Based on the latest still frame: a hand is holding a notebook"}
    ]
    assert active_client["camera_enabled"] is True
    assert len(vision.received_frames) == 1
    assert vision.received_frames[0].image == b"fresh-camera-frame"


@pytest.mark.asyncio
async def test_fetch_user_image_retries_when_first_result_is_garbled(monkeypatch):
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = SequencedVisionProcessor(["��的", "a person is holding a phone"])
    cached_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"abc",
        size=(2, 2),
        format="RGB",
    )
    cached_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = cached_frame
            self.latest_camera_frame_seq = 1

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)

    build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client={"id": "pc-123"},
    )

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "我手里拿着什么？"},
        function_name="fetch_user_image",
        tool_call_id="tool-3",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results == [{"description": "基于刚才的一帧画面：a person is holding a phone"}]
    assert len(vision.received_frames) == 2


@pytest.mark.asyncio
async def test_fetch_user_image_prefers_fresh_camera_frame(monkeypatch):
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor(response_text="a person is in front of a bookshelf")
    stale_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"old",
        size=(2, 2),
        format="RGB",
    )
    stale_frame.transport_source = "camera"
    fresh_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"new",
        size=(2, 2),
        format="RGB",
    )
    fresh_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = stale_frame
            self.latest_camera_frame_seq = 1

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            assert after_seq == 1
            assert timeout == 1.0
            return fresh_frame

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)

    _, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client={"id": "pc-123"},
    )
    context.add_message({"role": "user", "content": "摄像头里有什么？"})

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "摄像头里有什么？"},
        function_name="fetch_user_image",
        tool_call_id="tool-fresh",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results == [{"description": "基于刚才的一帧画面：a person is in front of a bookshelf"}]
    assert len(vision.received_frames) == 1
    assert vision.received_frames[0].image == b"new"


@pytest.mark.asyncio
async def test_fetch_user_image_rejects_stale_cached_frame(monkeypatch):
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="vikhyatk/moondream2",
    )
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor(response_text="should not be used")
    stale_frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"stale-stale!!",
        size=(2, 2),
        format="RGB",
    )
    stale_frame.transport_source = "camera"

    class StubCameraFrameBuffer(FrameProcessor):
        def __init__(self):
            super().__init__(name="stub-camera-buffer")
            self.latest_camera_frame = stale_frame
            self.latest_camera_frame_seq = 1
            self.latest_camera_frame_received_monotonic = asyncio.get_running_loop().time() - 10.0
            self.latest_camera_frame_received_at = datetime.now(UTC).isoformat()

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)

    _, context = build_local_browser_runtime(
        config,
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(),
        vision=vision,
        active_client={"id": "pc-123"},
    )
    context.add_message({"role": "user", "content": "摄像头里有什么？"})

    results = []

    async def result_callback(result):
        results.append(result)

    params = SimpleNamespace(
        arguments={"question": "摄像头里有什么？"},
        function_name="fetch_user_image",
        tool_call_id="tool-stale",
        llm=llm,
        result_callback=result_callback,
    )

    await llm.registered_functions["fetch_user_image"](params)

    assert results
    assert "error" in results[0]
    assert vision.received_frames == []


def test_local_browser_app_exposes_start_endpoint():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post("/start", json={"enableDefaultIceServers": True, "body": {"foo": "bar"}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sessionId"]
    assert payload["iceConfig"]["iceServers"][0]["urls"] == ["stun:stun.l.google.com:19302"]
    assert payload["modelSelection"]["accepted"] is True
    assert payload["modelSelection"]["applied"] is False
    assert "model_selection:default" in payload["modelSelection"]["reason_codes"]


def test_local_browser_session_offer_rejects_non_object_payload():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    start_response = client.post("/start", json={"body": {"transport": "webrtc"}})
    session_id = start_response.json()["sessionId"]

    post_response = client.post(f"/sessions/{session_id}/api/offer", json=[])
    patch_response = client.patch(f"/sessions/{session_id}/api/offer", json=[])

    assert post_response.status_code == 400
    assert post_response.text == "Invalid WebRTC request"
    assert patch_response.status_code == 400
    assert patch_response.text == "Invalid WebRTC request"


def test_local_browser_model_catalog_reports_public_safe_profiles(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "qwen3.5:4b"}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            assert url == "http://127.0.0.1:11434/v1/models"
            return FakeResponse()

    monkeypatch.setattr(local_browser.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/models")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["available"] is True
    assert payload["current_profile_id"] == "ollama-qwen3_5-4b"
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    assert set(profiles) >= {
        "ollama-qwen3_5-4b",
        "ollama-qwen3_5-9b",
        "openai-gpt-5_4-nano",
        "openai-gpt-5_4-mini",
        "openai-gpt-5_4",
    }
    assert profiles["ollama-qwen3_5-4b"]["available"] is True
    assert profiles["ollama-qwen3_5-9b"]["available"] is False
    assert "ollama_model_unavailable" in profiles["ollama-qwen3_5-9b"]["reason_codes"]
    assert profiles["openai-gpt-5_4-mini"]["available"] is False
    assert "remote_model_selection_disabled" in profiles["openai-gpt-5_4-mini"]["reason_codes"]
    assert "remote_model_selection:disabled" in payload["reason_codes"]
    for forbidden in (
        "Secret prompt",
        "OPENAI_API_KEY",
        "Authorization",
        "http://127.0.0.1:11434/v1",
        "request_payload",
        "event_id",
        "source_refs",
        "db_path",
        "traceback",
    ):
        assert forbidden not in encoded


def test_local_browser_model_catalog_sanitizes_local_extension_profiles(
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "qwen3.5:4b"}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return FakeResponse()

    extension_path = tmp_path / "profiles.json"
    extension_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "secret /tmp/brain.db",
                        "label": "Traceback /tmp/brain.db",
                        "provider": "ollama",
                        "model": "secret /tmp/brain.db",
                        "runtime_tier": "local",
                        "latency_tier": "Traceback /tmp/brain.db",
                        "capability_tier": "secret /tmp/brain.db",
                        "language_fit": ["zh", "Traceback /tmp/brain.db"],
                        "recommended_for": ["secret /tmp/brain.db"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_browser.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("BLINK_LOCAL_LLM_MODEL_PROFILES_PATH", str(extension_path))

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="secret /tmp/brain.db",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/models")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    redacted_profiles = [profile for profile in payload["profiles"] if profile["id"] == "redacted"]
    assert payload["current_profile_id"] == "redacted"
    assert payload["current_model"] == "redacted"
    assert redacted_profiles
    assert redacted_profiles[0]["selected"] is True
    assert redacted_profiles[0]["label"] == "redacted"
    assert redacted_profiles[0]["model"] == "redacted"
    assert redacted_profiles[0]["latency_tier"] == "redacted"
    assert redacted_profiles[0]["capability_tier"] == "redacted"
    assert redacted_profiles[0]["language_fit"] == ["zh", "redacted"]
    assert redacted_profiles[0]["recommended_for"] == ["redacted"]
    assert "redacted" in redacted_profiles[0]["reason_codes"]
    assert "Secret prompt" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_start_accepts_each_curated_model_profile(monkeypatch):
    from fastapi.testclient import TestClient

    verified: list[LocalLLMConfig] = []

    async def fake_verify_local_llm_config(llm_config):
        verified.append(llm_config)

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setenv("BLINK_LOCAL_ENABLE_REMOTE_MODEL_SELECTION", "1")

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        demo_mode=True,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    expected = {
        "ollama-qwen3_5-4b": ("ollama", "qwen3.5:4b"),
        "ollama-qwen3_5-9b": ("ollama", "qwen3.5:9b"),
        "openai-gpt-5_4-nano": ("openai-responses", "gpt-5.4-nano"),
        "openai-gpt-5_4-mini": ("openai-responses", "gpt-5.4-mini"),
        "openai-gpt-5_4": ("openai-responses", "gpt-5.4"),
    }

    for profile_id, (provider, model) in expected.items():
        response = client.post("/start", json={"body": {"model_profile_id": profile_id}})
        assert response.status_code == 200
        payload = response.json()
        assert payload["sessionId"]
        assert payload["modelSelection"]["accepted"] is True
        assert payload["modelSelection"]["applied"] is True
        assert payload["modelSelection"]["provider"] == provider
        assert payload["modelSelection"]["model"] == model

    assert [(item.provider, item.model) for item in verified] == list(expected.values())


def test_local_browser_start_rejects_invalid_and_unavailable_profiles(monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_verify_local_llm_config(llm_config):
        assert llm_config.provider != "openai-responses"

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    invalid = client.post("/start", json={"body": {"modelProfileId": "unknown-profile"}})
    unavailable = client.post(
        "/start",
        json={"body": {"model_profile_id": "openai-gpt-5_4-mini"}},
    )

    assert invalid.status_code == 400
    assert invalid.json()["modelSelection"]["accepted"] is False
    assert "model_profile_unknown" in invalid.json()["modelSelection"]["reason_codes"]
    assert "sessionId" not in invalid.json()
    assert unavailable.status_code == 400
    unavailable_selection = unavailable.json()["modelSelection"]
    assert unavailable_selection["accepted"] is False
    assert unavailable_selection["applied"] is False
    assert "remote_model_selection_disabled" in unavailable_selection["reason_codes"]
    assert "sessionId" not in unavailable.json()


def test_local_browser_start_sanitizes_rejected_model_profile_id(monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_verify_local_llm_config(_llm_config):
        raise AssertionError("unknown profiles must be rejected before verification")

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/start",
        json={"body": {"model_profile_id": "secret /tmp/brain.db"}},
    )

    assert response.status_code == 400
    payload = response.json()
    encoded = str(payload)
    assert payload["modelSelection"]["accepted"] is False
    assert payload["modelSelection"]["profile_id"] == "redacted"
    assert "model_profile_unknown" in payload["modelSelection"]["reason_codes"]
    assert "sessionId" not in payload
    assert "Secret prompt" not in encoded
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_start_sanitizes_accepted_extension_model_profile(
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    verified: list[LocalLLMConfig] = []

    async def fake_verify_local_llm_config(llm_config):
        verified.append(llm_config)

    extension_path = tmp_path / "profiles.json"
    extension_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "secret /tmp/brain.db",
                        "label": "Traceback /tmp/brain.db",
                        "provider": "ollama",
                        "model": "secret /tmp/brain.db",
                        "runtime_tier": "local",
                        "latency_tier": "Traceback /tmp/brain.db",
                        "capability_tier": "secret /tmp/brain.db",
                        "language_fit": ["zh"],
                        "recommended_for": ["secret /tmp/brain.db"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setenv("BLINK_LOCAL_LLM_MODEL_PROFILES_PATH", str(extension_path))

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/start",
        json={"body": {"model_profile_id": "secret /tmp/brain.db"}},
    )

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    selection = payload["modelSelection"]
    assert payload["sessionId"]
    assert selection["accepted"] is True
    assert selection["applied"] is True
    assert selection["profile_id"] == "redacted"
    assert selection["label"] == "redacted"
    assert selection["model"] == "redacted"
    assert selection["latency_tier"] == "redacted"
    assert selection["capability_tier"] == "redacted"
    assert selection["reason_codes"] == ["model_selection:accepted", "model_profile_selected"]
    assert [(item.provider, item.model) for item in verified] == [
        ("ollama", "secret /tmp/brain.db")
    ]
    assert "Secret prompt" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_remote_model_profiles_unlock_when_provider_is_openai(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "qwen3.5:4b"}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr(local_browser.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = LocalBrowserConfig(
        base_url=None,
        model="gpt-5.4-mini",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        llm_provider="openai-responses",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/models")

    assert response.status_code == 200
    payload = response.json()
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    assert profiles["openai-gpt-5_4-mini"]["available"] is True
    assert "remote_model_selection:enabled" in payload["reason_codes"]


def test_local_browser_expression_endpoint_reports_unavailable_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/expression")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["identity_label"] == "Blink; local non-human system"
    assert payload["modality"] == "browser"
    assert payload["initiative_label"] == "unavailable"
    assert payload["evidence_visibility_label"] == "unavailable"
    assert payload["correction_mode_label"] == "unavailable"
    assert payload["explanation_structure_label"] == "unavailable"
    assert payload["humor_mode_label"] == "unavailable"
    assert payload["vividness_mode_label"] == "unavailable"
    assert payload["character_presence_label"] == "unavailable"
    assert payload["style_summary"] == "unavailable"
    assert payload["safety_clamped"] is False
    assert payload["expression_controls_hardware"] is False
    assert payload["voice_policy"]["available"] is False
    assert payload["voice_policy"]["expression_controls_hardware"] is False
    assert "voice_policy_noop:hardware_control_forbidden" in payload["voice_policy"]["reason_codes"]
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_expression_endpoint_reports_fake_runtime_state():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def current_expression_state(self):
            return BrainRuntimeExpressionState(
                available=True,
                persona_profile_id="blink-default",
                identity_label="Blink; local non-human system",
                modality="browser",
                teaching_mode_label="walkthrough",
                memory_persona_section_status={
                    "persona_expression": "available",
                    "persona_defaults": "valid",
                },
                voice_style_summary="measured clarity; rate=1.00; pause=0.34",
                response_chunk_length="balanced",
                pause_yield_hint="pause=0.34; yield=yield after brief pause",
                interruption_strategy_label="yield after brief pause",
                initiative_label="balanced",
                evidence_visibility_label="compact",
                correction_mode_label="precise",
                explanation_structure_label="answer_first",
                expression_controls_hardware=False,
                reason_codes=("runtime_expression_state:available", "test_runtime"),
                humor_mode_label="witty",
                vividness_mode_label="vivid",
                sophistication_mode_label="sophisticated",
                character_presence_label="character_rich",
                story_mode_label="recurring_motifs",
                style_summary="humor=witty; vividness=vivid; story=recurring_motifs",
                humor_budget=0.42,
                playfulness=0.34,
                metaphor_density=0.61,
                safety_clamped=False,
                voice_policy={
                    "available": True,
                    "modality": "browser",
                    "concise_chunking_active": True,
                    "chunking_mode": "concise",
                    "max_spoken_chunk_chars": 132,
                    "interruption_strategy_label": "yield after brief pause",
                    "pause_yield_hint": "pause=0.34; yield=yield after brief pause",
                    "active_hints": ["concise_chunking"],
                    "unsupported_hints": ["speech_rate", "prosody_emphasis"],
                    "noop_reason_codes": [
                        "voice_policy_noop:speech_rate:kokoro",
                        "voice_policy_noop:prosody:kokoro",
                    ],
                    "expression_controls_hardware": False,
                    "reason_codes": ["voice_policy:available"],
                },
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/expression")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["persona_profile_id"] == "blink-default"
    assert payload["teaching_mode_label"] == "walkthrough"
    assert payload["initiative_label"] == "balanced"
    assert payload["evidence_visibility_label"] == "compact"
    assert payload["correction_mode_label"] == "precise"
    assert payload["explanation_structure_label"] == "answer_first"
    assert payload["humor_mode_label"] == "witty"
    assert payload["vividness_mode_label"] == "vivid"
    assert payload["character_presence_label"] == "character_rich"
    assert payload["humor_budget"] == 0.42
    assert payload["safety_clamped"] is False
    assert payload["memory_persona_section_status"]["persona_expression"] == "available"
    assert "measured clarity" in payload["voice_style_summary"]
    assert payload["voice_policy"]["available"] is True
    assert payload["voice_policy"]["chunking_mode"] == "concise"
    assert "speech_rate" in payload["voice_policy"]["unsupported_hints"]
    assert payload["expression_controls_hardware"] is False


def test_local_browser_expression_endpoint_sanitizes_malformed_runtime_state():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def current_expression_state(self):
            return {
                "available": True,
                "persona_profile_id": "secret /tmp/brain.db",
                "identity_label": "Traceback /tmp/brain.db",
                "modality": "browser",
                "teaching_mode_label": "secret /tmp/brain.db",
                "memory_persona_section_status": {
                    "persona_expression": "available",
                    "secret /tmp/brain.db": "Traceback /tmp/brain.db",
                },
                "voice_style_summary": "secret /tmp/brain.db",
                "initiative_label": "Traceback /tmp/brain.db",
                "humor_mode_label": "witty",
                "style_summary": "secret /tmp/brain.db",
                "humor_budget": "not-a-float",
                "safety_clamped": "yes",
                "expression_controls_hardware": True,
                "voice_policy": {
                    "available": True,
                    "modality": "browser",
                    "chunking_mode": "secret /tmp/brain.db",
                    "unsupported_hints": ["speech_rate", "Traceback /tmp/brain.db"],
                    "expression_controls_hardware": True,
                    "reason_codes": ["voice_policy:available", "secret /tmp/brain.db"],
                },
                "voice_actuation_plan": {
                    "available": True,
                    "backend_label": "secret /tmp/brain.db",
                    "modality": "browser",
                    "requested_hints": ["speech_rate", "Traceback /tmp/brain.db"],
                    "applied_hints": ["chunk_boundaries"],
                    "unsupported_hints": ["secret /tmp/brain.db"],
                    "expression_controls_hardware": True,
                    "reason_codes": ["voice_actuation:available", "secret /tmp/brain.db"],
                },
                "reason_codes": ["runtime_expression_state:available", "secret /tmp/brain.db"],
                "prompt_text": "secret prompt",
                "private_working_memory": "hidden deliberation",
            }

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/expression")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["available"] is True
    assert payload["persona_profile_id"] == "redacted"
    assert payload["identity_label"] == "redacted"
    assert payload["teaching_mode_label"] == "redacted"
    assert payload["memory_persona_section_status"]["redacted"] == "redacted"
    assert payload["voice_style_summary"] == "redacted"
    assert payload["initiative_label"] == "redacted"
    assert payload["style_summary"] == "redacted"
    assert payload["humor_budget"] == 0.0
    assert payload["safety_clamped"] is False
    assert payload["expression_controls_hardware"] is False
    assert payload["voice_policy"]["chunking_mode"] == "redacted"
    assert "redacted" in payload["voice_policy"]["unsupported_hints"]
    assert payload["voice_policy"]["expression_controls_hardware"] is False
    assert payload["voice_actuation_plan"]["backend_label"] == "redacted"
    assert "redacted" in payload["voice_actuation_plan"]["requested_hints"]
    assert payload["voice_actuation_plan"]["expression_controls_hardware"] is False
    assert "redacted" in payload["reason_codes"]
    assert "prompt_text" not in encoded
    assert "private_working_memory" not in encoded
    assert "secret prompt" not in encoded
    assert "hidden deliberation" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_voice_metrics_endpoint_reports_unavailable_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/voice-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["response_count"] == 0
    assert payload["chunk_count"] == 0
    assert payload["input_health"]["available"] is False
    assert payload["input_health"]["microphone_state"] == "unavailable"
    assert payload["input_health"]["stt_state"] == "unavailable"
    assert payload["expression_controls_hardware"] is False
    assert "runtime_not_active" in payload["reason_codes"]
    assert "voice_policy_noop:hardware_control_forbidden" in payload["reason_codes"]


def test_local_browser_voice_metrics_endpoint_reports_fake_runtime_metrics():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def current_voice_metrics(self):
            return {
                "available": True,
                "response_count": 2,
                "concise_chunking_activation_count": 1,
                "chunk_count": 4,
                "max_chunk_chars": 88,
                "average_chunk_chars": 52.5,
                "interruption_frame_count": 1,
                "buffer_flush_count": 2,
                "buffer_discard_count": 1,
                "last_chunking_mode": "concise",
                "last_max_spoken_chunk_chars": 132,
                "expression_controls_hardware": True,
                "reason_codes": ["voice_metrics:available", "test_runtime"],
                "event_id": "evt-secret",
            }

        def current_voice_input_health(self):
            return {
                "schema_version": 1,
                "available": True,
                "microphone_state": "receiving",
                "stt_state": "transcribed",
                "audio_frame_count": 8,
                "speech_start_count": 1,
                "speech_stop_count": 1,
                "transcription_count": 1,
                "stt_error_count": 0,
                "last_audio_frame_age_ms": 24,
                "last_transcription_chars": 9,
                "reason_codes": ["voice_input_health:v1", "microphone:receiving"],
                "raw_transcript": "secret user speech",
                "event_id": "evt-secret",
            }

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/voice-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["response_count"] == 2
    assert payload["concise_chunking_activation_count"] == 1
    assert payload["chunk_count"] == 4
    assert payload["max_chunk_chars"] == 88
    assert payload["average_chunk_chars"] == 52.5
    assert payload["interruption_frame_count"] == 1
    assert payload["buffer_flush_count"] == 2
    assert payload["buffer_discard_count"] == 1
    assert payload["last_chunking_mode"] == "concise"
    assert payload["last_max_spoken_chunk_chars"] == 132
    assert payload["input_health"]["microphone_state"] == "receiving"
    assert payload["input_health"]["stt_state"] == "transcribed"
    assert payload["input_health"]["audio_frame_count"] == 8
    assert payload["input_health"]["transcription_count"] == 1
    assert payload["input_health"]["last_transcription_chars"] == 9
    assert payload["expression_controls_hardware"] is False
    assert "event_id" not in payload
    assert "raw_transcript" not in str(payload)
    assert "secret user speech" not in str(payload)


def test_local_browser_voice_metrics_endpoint_sanitizes_malformed_runtime_values():
    from fastapi.testclient import TestClient

    class MalformedRuntime:
        def current_voice_metrics(self):
            return {
                "available": "true",
                "response_count": "not-an-int",
                "average_chunk_chars": "not-a-float",
                "last_chunking_mode": "secret /tmp/brain.db",
                "reason_codes": ["voice_metrics:available", "Traceback"],
                "input_health": {"raw_transcript": "secret user speech"},
            }

        def current_voice_input_health(self):
            return {
                "available": "true",
                "microphone_state": "secret /tmp/brain.db",
                "stt_state": "secret /tmp/brain.db",
                "audio_frame_count": "not-an-int",
                "last_audio_frame_at": "secret /tmp/brain.db",
                "last_audio_frame_age_ms": "not-an-int",
                "last_stt_event_at": "secret /tmp/brain.db",
                "stt_waiting_since_at": "secret /tmp/brain.db",
                "stt_wait_age_ms": "not-an-int",
                "stt_waiting_too_long": "yes",
                "last_transcription_at": "secret /tmp/brain.db",
                "track_reason": "secret /tmp/brain.db",
                "reason_codes": ["voice_input_health:v1", "RuntimeError"],
                "raw_transcript": "secret user speech",
                "event_id": "evt-secret",
            }

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = MalformedRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/voice-metrics")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["available"] is False
    assert payload["response_count"] == 0
    assert payload["average_chunk_chars"] == 0.0
    assert payload["last_chunking_mode"] == "unavailable"
    assert payload["reason_codes"] == ["voice_metrics:available", "redacted"]
    assert payload["input_health"]["available"] is False
    assert payload["input_health"]["microphone_state"] == "unavailable"
    assert payload["input_health"]["stt_state"] == "unavailable"
    assert payload["input_health"]["audio_frame_count"] == 0
    assert payload["input_health"]["last_audio_frame_at"] is None
    assert payload["input_health"]["last_audio_frame_age_ms"] is None
    assert payload["input_health"]["last_stt_event_at"] is None
    assert payload["input_health"]["stt_waiting_since_at"] is None
    assert payload["input_health"]["stt_wait_age_ms"] is None
    assert payload["input_health"]["stt_waiting_too_long"] is False
    assert payload["input_health"]["last_transcription_at"] is None
    assert payload["input_health"]["track_reason"] == "redacted"
    assert payload["input_health"]["reason_codes"] == [
        "voice_input_health:v1",
        "redacted",
    ]
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "RuntimeError" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "raw_transcript" not in encoded
    assert "event_id" not in encoded


def test_local_browser_voice_metrics_endpoint_errors_are_safe():
    from fastapi.testclient import TestClient

    class FailingRuntime:
        def current_voice_metrics(self):
            raise RuntimeError("secret /tmp/brain.db")

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FailingRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/voice-metrics")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["available"] is False
    assert payload["response_count"] == 0
    assert "runtime_voice_metrics_error:RuntimeError" in payload["reason_codes"]
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_stack_endpoint_reports_safe_configured_status_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="https://proxy.test/v1",
        model="gpt-browser-demo",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=True,
        llm_provider="openai-responses",
        llm_service_tier="flex",
        demo_mode=True,
        llm_max_output_tokens=120,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/stack")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["available"] is True
    assert payload["runtime_active"] is False
    assert payload["llm_provider"] == "openai-responses"
    assert payload["model"] == "gpt-browser-demo"
    assert payload["service_tier"] == "flex"
    assert payload["stt_backend"] == "mlx-whisper"
    assert payload["stt_model"] == "mlx-community/whisper-medium-mlx"
    assert payload["tts_backend"] == "local-http-wav"
    assert payload["tts_voice"] is None
    assert payload["demo_mode"] is True
    assert payload["max_output_tokens"] == 120
    assert payload["vision_enabled"] is True
    assert payload["continuous_perception_enabled"] is True
    assert payload["browser_media"]["mode"] == "unreported"
    assert payload["browser_media"]["camera_state"] == "unknown"
    assert payload["browser_media"]["microphone_state"] == "unknown"
    assert "demo_mode:enabled" in payload["reason_codes"]
    assert "runtime_active:false" in payload["reason_codes"]
    assert "browser_media:unreported" in payload["reason_codes"]
    assert "OPENAI_API_KEY" not in encoded
    assert "Authorization" not in encoded
    assert "https://proxy.test/v1" not in encoded
    assert "Secret prompt" not in encoded


def test_local_browser_client_media_endpoint_updates_stack_safely():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    update = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "camera_state": "not-a-public-state /tmp/brain.db",
            "microphone_state": "ready",
            "reason_codes": [
                "browser_media:audio_only",
                "browser_camera:device_not_found",
                "raw exception /tmp/brain.db",
            ],
            "raw_error": "Traceback secret /tmp/brain.db",
            "constraints": {"video": {"deviceId": "private"}},
        },
    )

    assert update.status_code == 200
    payload = update.json()
    encoded = str(payload)
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["browser_media"]["mode"] == "audio_only"
    assert payload["browser_media"]["camera_state"] == "unavailable"
    assert payload["browser_media"]["microphone_state"] == "ready"
    assert "browser_media_report:accepted" in payload["reason_codes"]
    assert "raw_error" not in encoded
    assert "constraints" not in encoded
    assert "Traceback" not in encoded
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded

    stack = client.get("/api/runtime/stack").json()
    assert stack["browser_media"]["mode"] == "audio_only"
    assert stack["browser_media"]["camera_state"] == "unavailable"
    assert stack["browser_media"]["microphone_state"] == "ready"
    assert "browser_media:audio_only" in stack["reason_codes"]
    assert "browser_camera:unavailable" in stack["reason_codes"]
    assert "browser_microphone:ready" in stack["reason_codes"]
    assert "Secret prompt" not in str(stack)
    assert "request_payload" not in encoded
    assert "event_id" not in encoded
    assert "source_refs" not in encoded
    assert "db_path" not in encoded


def test_local_browser_client_media_endpoint_accepts_live_status_states_safely():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        allow_barge_in=False,
        tts_runtime_label="local-http-wav/MeloTTS",
        config_profile="browser-zh-melo",
    )

    app, _ = create_app(config)
    app.state.blink_browser_active_client_id = "client-live-status"
    app.state.blink_browser_active_session_id = "session-live-status"
    client = TestClient(app)

    response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "camera_state": "stale",
            "microphone_state": "receiving",
            "echo_cancellation": True,
            "noiseSuppression": False,
            "auto_gain_control": "unsupported",
            "deviceLabel": "Built-in microphone",
            "raw_text": "private transcript",
            "sdp": "secret session description",
            "ice": "private candidate",
            "Authorization": "Bearer secret",
        },
    )
    state = client.get("/api/runtime/performance-state").json()
    events = client.get("/api/runtime/performance-events").json()

    assert response.status_code == 200
    payload = response.json()
    assert payload["browser_media"]["mode"] == "audio_only"
    assert payload["browser_media"]["camera_state"] == "stale"
    assert payload["browser_media"]["microphone_state"] == "receiving"
    assert payload["browser_media"]["echo"] == {
        "echo_cancellation": "enabled",
        "noise_suppression": "disabled",
        "auto_gain_control": "unsupported",
    }
    assert "browser_camera:stale" in payload["browser_media"]["reason_codes"]
    assert "browser_microphone:receiving" in payload["browser_media"]["reason_codes"]
    assert "browser_echo_cancellation:enabled" in payload["browser_media"]["reason_codes"]
    assert state["mode"] == "listening"
    assert state["browser_media"]["camera_state"] == "stale"
    assert state["browser_media"]["microphone_state"] == "receiving"
    assert state["browser_media"]["echo"]["echo_cancellation"] == "enabled"
    assert state["camera_presence"]["state"] == "stale"
    assert state["camera_presence"]["last_vision_result_state"] == "none"
    assert state["interruption"]["barge_in_state"] == "protected"
    assert state["memory_persona"]["available"] is False
    assert state["memory_persona"]["profile"] == "browser-zh-melo"
    assert state["memory_persona"]["used_in_current_reply"] == []
    encoded = str({"payload": payload, "state": state, "events": events})
    assert "private transcript" not in encoded
    assert "secret session description" not in encoded
    assert "private candidate" not in encoded
    assert "Built-in microphone" not in encoded
    assert "Bearer secret" not in encoded
    assert "Secret prompt" not in encoded


def test_local_browser_performance_state_ignores_stale_media_without_active_client():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=True,
        allow_barge_in=False,
        tts_runtime_label="local-http-wav/MeloTTS",
        config_profile="browser-zh-melo",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    update = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "receiving",
            "echoCancellation": True,
        },
    )
    state = client.get("/api/runtime/performance-state").json()

    assert update.status_code == 200
    assert update.json()["browser_media"]["mode"] == "camera_and_microphone"
    assert state["active_client_id"] is None
    assert state["mode"] == "waiting"
    assert state["browser_media"]["mode"] == "unreported"
    assert state["browser_media"]["camera_state"] == "unknown"
    assert state["browser_media"]["microphone_state"] == "unknown"
    assert state["browser_media"]["echo"]["echo_cancellation"] == "unknown"
    assert "browser_client:disconnected" in state["browser_media"]["reason_codes"]


def test_local_browser_performance_state_ignores_stale_audio_only_media_without_active_client():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    update = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "camera_state": "stalled",
            "microphone_state": "receiving",
            "output_playback_state": "unknown",
            "echoCancellation": True,
        },
    )
    state = client.get("/api/runtime/performance-state").json()

    assert update.status_code == 200
    assert update.json()["browser_media"]["mode"] == "audio_only"
    assert state["active_client_id"] is None
    assert state["mode"] == "waiting"
    assert state["browser_media"]["mode"] == "unreported"
    assert state["browser_media"]["camera_state"] == "unknown"
    assert state["browser_media"]["microphone_state"] == "unknown"
    assert state["browser_media"]["output_playback_state"] == "unknown"
    assert "browser_client:disconnected" in state["browser_media"]["reason_codes"]


def test_local_browser_actor_state_preserves_failed_media_without_active_client():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    update = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "unavailable",
            "camera_state": "permission_denied",
            "microphone_state": "permission_denied",
            "reason_codes": [
                "browser_media:unavailable",
                "browser_camera:permission_denied",
                "browser_microphone:permission_denied",
                "raw traceback /tmp/secret",
            ],
            "raw_error": "private device trace",
        },
    )
    performance_state = client.get("/api/runtime/performance-state").json()
    actor_state = client.get("/api/runtime/actor-state").json()

    assert update.status_code == 200
    assert performance_state["active_client_id"] is None
    assert performance_state["mode"] == "waiting"
    assert performance_state["browser_media"]["mode"] == "unavailable"
    assert performance_state["browser_media"]["camera_state"] == "permission_denied"
    assert performance_state["browser_media"]["microphone_state"] == "permission_denied"
    assert "browser_client:disconnected" in performance_state["browser_media"]["reason_codes"]
    assert actor_state["webrtc"]["session_active"] is False
    assert actor_state["webrtc"]["media"]["mode"] == "unavailable"
    assert actor_state["webrtc"]["media"]["camera_state"] == "permission_denied"
    assert actor_state["webrtc"]["media"]["microphone_state"] == "permission_denied"
    encoded = str({"performance_state": performance_state, "actor_state": actor_state})
    assert "private device trace" not in encoded
    assert "/tmp/secret" not in encoded
    assert "Secret prompt" not in encoded


def test_local_browser_client_config_js_sets_camera_from_vision_profile():
    from fastapi.testclient import TestClient

    cases = [
        ("browser-zh-melo", local_browser.Language.ZH, "local-http-wav", True, "MeloTTS", True),
        ("browser-en-kokoro", local_browser.Language.EN, "kokoro", True, "English", True),
    ]

    for profile, language, tts_backend, vision_enabled, label_suffix, enable_camera in cases:
        config = LocalBrowserConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3.5:4b",
            system_prompt="Secret prompt",
            language=language,
            stt_backend="mlx-whisper",
            tts_backend=tts_backend,
            stt_model="mlx-community/whisper-medium-mlx",
            tts_voice=None,
            tts_base_url="http://127.0.0.1:8001" if tts_backend == "local-http-wav" else None,
            host="127.0.0.1",
            port=7860,
            vision_enabled=vision_enabled,
            continuous_perception_enabled=False,
            allow_barge_in=False,
            tts_runtime_label=f"{tts_backend}/{label_suffix}",
            config_profile=profile,
        )
        app, _ = create_app(config)
        client = TestClient(app)

        response = client.get("/api/runtime/client-config.js")
        text = response.text

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, max-age=0"
        assert response.headers["content-type"].startswith("application/javascript")
        assert "globalThis.BlinkRuntimeConfig = Object.freeze(" in text
        assert f'"profile":"{profile}"' in text
        assert f'"enableCam":{str(enable_camera).lower()}' in text
        assert '"enableMic":true' in text
        assert '"actor_surface_v2_enabled":true' in text
        assert "actor_surface_v2:on" in text
        assert "browser_client_config:v1" in text
        assert "Secret prompt" not in text
        assert "Authorization" not in text


def test_local_browser_performance_state_reports_primary_profile_without_connection():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="bf_emma",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/performance-state")
    events = client.get("/api/runtime/performance-events")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["runtime"] == "browser"
    assert payload["transport"] == "WebRTC"
    assert payload["mode"] == "waiting"
    assert payload["profile"] == "browser-en-kokoro"
    assert payload["tts"] == "kokoro/English"
    assert payload["tts_backend"] == "kokoro"
    assert payload["protected_playback"] is True
    assert payload["browser_media"]["mode"] == "unreported"
    assert payload["browser_media"]["camera_state"] == "unknown"
    assert payload["browser_media"]["microphone_state"] == "unknown"
    assert payload["browser_media"]["echo"]["echo_cancellation"] == "unknown"
    assert payload["interruption"]["barge_in_state"] == "protected"
    assert payload["interruption"]["protected_playback"] is True
    assert payload["interruption"]["headphones_recommended"] is False
    assert payload["speech"]["director_mode"] == "kokoro_chunked"
    assert payload["speech"]["available"] is False
    assert payload["speech"]["stale_chunk_drop_count"] == 0
    assert payload["active_listening"]["phase"] == "idle"
    assert payload["active_listening"]["partial_available"] is False
    assert payload["active_listening"]["topics"] == []
    assert payload["active_listening"]["constraints"] == []
    assert payload["camera_presence"]["enabled"] is True
    assert payload["camera_presence"]["state"] == "disconnected"
    assert payload["camera_presence"]["grounding_mode"] == "none"
    assert payload["memory_persona"]["available"] is False
    assert payload["memory_persona"]["profile"] == "browser-en-kokoro"
    assert payload["memory_persona"]["selected_memory_count"] == 0
    assert payload["memory"]["available"] is False
    assert payload["last_event_id"] == 0
    assert "Secret prompt" not in encoded
    assert "Authorization" not in encoded
    assert events.status_code == 200
    assert events.json()["events"] == []


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label", "vision_enabled"),
    [
        (
            "browser-zh-melo",
            local_browser.Language.ZH,
            "local-http-wav",
            "local-http-wav/MeloTTS",
            True,
        ),
        ("browser-en-kokoro", local_browser.Language.EN, "kokoro", "kokoro/English", True),
    ],
)
def test_local_browser_performance_state_exposes_memory_persona_for_primary_profiles(
    profile,
    language,
    tts_backend,
    tts_label,
    vision_enabled,
):
    from fastapi.testclient import TestClient

    captured = {}

    class FakeRuntime:
        def current_memory_persona_performance_plan(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                as_dict=lambda: {
                    "schema_version": 1,
                    "available": True,
                    "profile": kwargs.get("profile"),
                    "modality": "browser",
                    "language": language.value,
                    "tts_label": kwargs.get("tts_label"),
                    "protected_playback": kwargs.get("protected_playback"),
                    "camera_state": kwargs.get("camera_state"),
                    "continuous_perception_enabled": kwargs.get(
                        "continuous_perception_enabled"
                    ),
                    "current_turn_state": kwargs.get("current_turn_state"),
                    "memory_policy": "balanced",
                    "selected_memory_count": 1,
                    "suppressed_memory_count": 0,
                    "used_in_current_reply": [
                        {
                            "memory_id": "memory_claim:user:user-browser:claim-safe",
                            "display_kind": "preference",
                            "title": "concise answers",
                            "used_reason": "selected_for_relevant_continuity",
                            "behavior_effect": "memory callback changed this reply",
                            "reason_codes": ["source:context_selection"],
                        }
                    ],
                    "behavior_effects": ["memory_callback_active"],
                    "memory_continuity_trace": {
                        "schema_version": 1,
                        "profile": profile,
                        "language": language.value,
                        "memory_effect": "cross_language_callback",
                        "cross_language_count": 1,
                        "selected_memory_count": 1,
                        "suppressed_memory_count": 0,
                        "selected_memories": [
                            {
                                "memory_id": "memory_claim:user:user-browser:claim-safe",
                                "display_kind": "preference",
                                "summary": "concise answers",
                                "source_language": "zh" if language.value == "en" else "en",
                                "cross_language": True,
                                "effect_labels": ["shorter_explanation"],
                                "linked_discourse_episode_ids": [
                                    "discourse-episode-v3:test"
                                ],
                                "reason_codes": ["memory_continuity:selected"],
                            }
                        ],
                        "memory_continuity_v3": {
                            "schema_version": 3,
                            "selected_discourse_episodes": [
                                {
                                    "discourse_episode_id": "discourse-episode-v3:test",
                                    "category_labels": ["user_preference"],
                                    "effect_labels": ["shorter_explanation"],
                                    "memory_ids": [
                                        "memory_claim:user:user-browser:claim-safe"
                                    ],
                                    "confidence_bucket": "high",
                                    "reason_codes": [
                                        "memory_continuity_v3:discourse_episode_ref"
                                    ],
                                }
                            ],
                            "effect_labels": ["shorter_explanation"],
                            "conflict_labels": [],
                            "staleness_labels": [],
                            "cross_language_transfer_count": 1,
                            "reason_codes": ["memory_continuity_v3:available"],
                        },
                        "reason_codes": ["memory_continuity:trace"],
                    },
                    "persona_references": [
                        {
                            "reference_id": "persona:memory_callback",
                            "mode": "memory_callback",
                            "label": "memory callback",
                            "applies": True,
                            "behavior_effect": "use selected public memories as brief callbacks",
                            "reason_codes": ["persona_reference:memory_callback"],
                        }
                    ],
                    "summary": "1 memories used; 0 suppressed; 1 persona references active.",
                    "reason_codes": ["memory_persona_performance:v1", "secret prompt"],
                }
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=language,
        stt_backend="mlx-whisper",
        tts_backend=tts_backend,
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001" if tts_backend == "local-http-wav" else None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=vision_enabled,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label=tts_label,
        config_profile=profile,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    payload = client.get("/api/runtime/performance-state").json()
    encoded = str(payload)

    assert payload["memory_persona"]["available"] is True
    assert payload["memory_persona"]["profile"] == profile
    assert payload["memory_persona"]["tts_label"] == tts_label
    assert payload["memory_persona"]["protected_playback"] is True
    assert payload["memory_persona"]["used_in_current_reply"][0]["title"] == "concise answers"
    assert payload["memory_persona"]["behavior_effects"] == ["memory_callback_active"]
    continuity_v3 = payload["memory_persona"]["memory_continuity_trace"]["memory_continuity_v3"]
    assert continuity_v3["effect_labels"] == ["shorter_explanation"]
    assert continuity_v3["selected_discourse_episodes"][0]["category_labels"] == [
        "user_preference"
    ]
    assert payload["memory_persona"]["persona_references"][0]["mode"] == "memory_callback"
    assert captured["profile"] == profile
    assert captured["protected_playback"] is True
    assert "memory_persona:available" in payload["reason_codes"]
    assert "memory_persona_performance:v1" in payload["memory_persona"]["reason_codes"]
    assert "Secret prompt" not in encoded
    assert "secret prompt" not in encoded


def test_local_browser_actor_polls_use_cached_memory_persona_plan():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def cached_memory_persona_performance_plan(self):
            return SimpleNamespace(
                as_dict=lambda: {
                    "schema_version": 1,
                    "available": True,
                    "profile": "browser-en-kokoro",
                    "modality": "browser",
                    "language": "en",
                    "tts_label": "kokoro/English",
                    "protected_playback": True,
                    "current_turn_state": "waiting",
                    "memory_policy": "balanced",
                    "selected_memory_count": 0,
                    "suppressed_memory_count": 0,
                    "used_in_current_reply": [],
                    "persona_references": [],
                    "summary": "Cached public-safe plan.",
                    "reason_codes": ["memory_persona_performance:v1", "cached"],
                }
            )

        def current_memory_persona_performance_plan(self, **_kwargs):
            raise AssertionError("actor polling must not compile a DB-backed plan")

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    performance_payload = client.get("/api/runtime/performance-state").json()
    actor_payload = client.get("/api/runtime/actor-state").json()

    assert performance_payload["memory_persona"]["summary"] == "Cached public-safe plan."
    assert actor_payload["memory_persona"]["summary"] == "Cached public-safe plan."


def test_local_browser_performance_state_reports_barge_in_armed_when_allowed():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=True,
        allow_barge_in=True,
        tts_runtime_label="local-http-wav/MeloTTS",
        config_profile="browser-zh-melo",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    payload = client.get("/api/runtime/performance-state").json()

    assert payload["profile"] == "browser-zh-melo"
    assert payload["protected_playback"] is False
    assert payload["interruption"]["barge_in_state"] == "armed"
    assert payload["interruption"]["protected_playback"] is False
    assert payload["interruption"]["headphones_recommended"] is True
    assert payload["speech"]["director_mode"] == "melo_chunked"
    assert payload["camera_presence"]["enabled"] is True
    assert payload["camera_presence"]["state"] == "disconnected"
    assert "interruption:armed" in payload["reason_codes"]
    assert "speech_director:melo_chunked" in payload["reason_codes"]
    assert "camera_presence:disconnected" in payload["reason_codes"]
    assert payload["memory_persona"]["available"] is False
    assert payload["memory_persona"]["profile"] == "browser-zh-melo"


def test_local_browser_performance_events_track_media_and_session_safely():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="local-http-wav/MeloTTS",
        config_profile="browser-zh-melo",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    start = client.post("/start", json={"body": {}})
    media = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "ready",
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": False,
            "raw_text": "do not expose this",
            "sdp": "secret session description",
            "reason_codes": ["browser_media:camera_and_microphone"],
        },
    )
    payload = client.get("/api/runtime/performance-events?limit=999").json()
    state = client.get("/api/runtime/performance-state").json()

    assert start.status_code == 200
    assert media.status_code == 200
    session_id = start.json()["sessionId"]
    events = payload["events"]
    assert payload["limit"] == 200
    assert [event["event_id"] for event in events] == sorted(
        event["event_id"] for event in events
    )
    assert events[0]["event_type"] == "browser_session.created"
    assert events[0]["session_id"] == session_id
    media_event = next(event for event in events if event["event_type"] == "browser_media.reported")
    camera_event = next(event for event in events if event["event_type"] == "camera.connected")
    assert media_event["metadata"]["media_mode"] == "camera_and_microphone"
    assert media_event["metadata"]["camera_state"] == "ready"
    assert media_event["metadata"]["microphone_state"] == "ready"
    assert media_event["metadata"]["echo_cancellation_state"] == "enabled"
    assert media_event["metadata"]["noise_suppression_state"] == "enabled"
    assert media_event["metadata"]["auto_gain_control_state"] == "disabled"
    assert camera_event["metadata"]["camera_state"] == "ready"
    encoded = str(payload)
    assert "do not expose this" not in encoded
    assert "secret session description" not in encoded
    assert "Secret prompt" not in encoded
    assert state["browser_media"]["mode"] == "camera_and_microphone"
    assert state["browser_media"]["echo"]["echo_cancellation"] == "enabled"
    assert state["last_event_type"] == "camera.connected"
    assert state["profile"] == "browser-zh-melo"
    assert state["tts"] == "local-http-wav/MeloTTS"
    assert state["speech"]["director_mode"] == "melo_chunked"
    assert state["camera_presence"]["state"] == "disconnected"

    after = client.get(
        f"/api/runtime/performance-events?after_id={events[0]['event_id']}&limit=3"
    ).json()
    assert [event["event_type"] for event in after["events"]] == [
        "browser_media.reported",
        "camera.context_available",
        "camera.connected",
    ]


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label"),
    [
        (
            "browser-zh-melo",
            local_browser.Language.ZH,
            "local-http-wav",
            "local-http-wav/MeloTTS",
        ),
        ("browser-en-kokoro", local_browser.Language.EN, "kokoro", "kokoro/English"),
    ],
)
def test_local_browser_actor_events_expose_primary_profile_context(
    profile,
    language,
    tts_backend,
    tts_label,
):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=language,
        stt_backend="mlx-whisper",
        tts_backend=tts_backend,
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001" if tts_backend == "local-http-wav" else None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label=tts_label,
        config_profile=profile,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    start = client.post("/start", json={"body": {}})
    performance_payload = client.get("/api/runtime/performance-events").json()
    actor_payload = client.get("/api/runtime/actor-events").json()

    assert start.status_code == 200
    assert performance_payload["schema_version"] == 1
    assert performance_payload["events"][0]["schema_version"] == 1
    assert actor_payload["schema_version"] == 2
    assert actor_payload["latest_event_id"] == performance_payload["latest_event_id"]
    event = actor_payload["events"][0]
    assert event["schema_version"] == 2
    assert event["event_type"] == "connected"
    assert event["mode"] == "connected"
    assert event["profile"] == profile
    assert event["language"] == language.value
    assert event["tts_backend"] == tts_backend
    assert event["tts_label"] == tts_label
    assert event["vision_backend"] == "moondream"
    assert "Secret prompt" not in str(actor_payload)


def test_local_browser_actor_trace_config_defaults_env_and_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("BLINK_LOCAL_ACTOR_TRACE", "")
    monkeypatch.setenv("BLINK_LOCAL_ACTOR_TRACE_DIR", "")
    monkeypatch.setenv("BLINK_LOCAL_PERFORMANCE_EPISODE_V3", "")
    monkeypatch.setenv("BLINK_LOCAL_PERFORMANCE_EPISODE_V3_DIR", "")
    monkeypatch.setenv("BLINK_LOCAL_PERFORMANCE_PREFERENCES_V3_DIR", "")

    default_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )
    env_config = None
    cli_on_config = None
    cli_off_config = None

    monkeypatch.setenv("BLINK_LOCAL_ACTOR_TRACE", "1")
    monkeypatch.setenv("BLINK_LOCAL_ACTOR_TRACE_DIR", str(tmp_path / "env-traces"))
    monkeypatch.setenv("BLINK_LOCAL_PERFORMANCE_EPISODE_V3", "1")
    monkeypatch.setenv(
        "BLINK_LOCAL_PERFORMANCE_EPISODE_V3_DIR",
        str(tmp_path / "env-episodes"),
    )
    monkeypatch.setenv(
        "BLINK_LOCAL_PERFORMANCE_PREFERENCES_V3_DIR",
        str(tmp_path / "env-preferences"),
    )
    env_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(["--config-profile", "browser-en-kokoro"])
    )
    cli_on_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            [
                "--config-profile",
                "browser-en-kokoro",
                "--actor-trace",
                "--actor-trace-dir",
                str(tmp_path / "cli-traces"),
                "--performance-episode-v3",
                "--performance-episode-v3-dir",
                str(tmp_path / "cli-episodes"),
                "--performance-preferences-v3-dir",
                str(tmp_path / "cli-preferences"),
            ]
        )
    )
    cli_off_config = local_browser.resolve_config(
        local_browser.build_parser().parse_args(
            [
                "--config-profile",
                "browser-en-kokoro",
                "--no-actor-trace",
                "--no-performance-episode-v3",
            ]
        )
    )

    assert default_config.actor_trace_enabled is False
    assert str(default_config.actor_trace_dir).endswith("artifacts/actor_traces")
    assert default_config.performance_episode_v3_enabled is False
    assert str(default_config.performance_episode_v3_dir).endswith(
        "artifacts/performance_episodes_v3"
    )
    assert str(default_config.performance_preferences_v3_dir).endswith(
        "artifacts/performance_preferences_v3"
    )
    assert env_config.actor_trace_enabled is True
    assert env_config.actor_trace_dir == tmp_path / "env-traces"
    assert env_config.performance_episode_v3_enabled is True
    assert env_config.performance_episode_v3_dir == tmp_path / "env-episodes"
    assert env_config.performance_preferences_v3_dir == tmp_path / "env-preferences"
    assert cli_on_config.actor_trace_enabled is True
    assert cli_on_config.actor_trace_dir == tmp_path / "cli-traces"
    assert cli_on_config.performance_episode_v3_enabled is True
    assert cli_on_config.performance_episode_v3_dir == tmp_path / "cli-episodes"
    assert cli_on_config.performance_preferences_v3_dir == tmp_path / "cli-preferences"
    assert cli_off_config.actor_trace_enabled is False
    assert cli_off_config.performance_episode_v3_enabled is False


def test_local_browser_actor_trace_writes_sanitized_jsonl_when_enabled(tmp_path):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="bf_emma",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        continuous_perception_enabled=False,
        allow_barge_in=False,
        tts_runtime_label="kokoro/English",
        config_profile="browser-en-kokoro",
        actor_trace_enabled=True,
        actor_trace_dir=tmp_path,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    client.post("/start", json={"body": {}})
    response = client.post(
        "/api/runtime/client-media",
        json={"mode": "audio_only", "microphone_state": "ready", "raw_text": "private"},
    )

    assert response.status_code == 200
    trace_path = Path(app.state.blink_actor_trace_path)
    assert trace_path.parent == tmp_path
    payloads = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payloads
    assert {payload["schema_version"] for payload in payloads} == {2}
    assert {payload["profile"] for payload in payloads} == {"browser-en-kokoro"}
    assert {payload["vision_backend"] for payload in payloads} == {"moondream"}
    encoded = str(payloads)
    assert "Secret prompt" not in encoded
    assert "private" not in encoded


def test_local_browser_stack_endpoint_sanitizes_config_text_fields():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="https://proxy.test/v1",
        model="secret /tmp/brain.db",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="Traceback /tmp/brain.db",
        tts_backend="secret /tmp/brain.db",
        stt_model="Traceback /tmp/brain.db",
        tts_voice="secret /tmp/brain.db",
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        llm_provider="ollama",
        llm_service_tier="Traceback /tmp/brain.db",
        llm_max_output_tokens="not-an-int",
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/stack")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["model"] == "redacted"
    assert payload["configured_model"] == "redacted"
    assert payload["stt_backend"] == "redacted"
    assert payload["stt_model"] == "redacted"
    assert payload["tts_backend"] == "redacted"
    assert payload["tts_voice"] == "redacted"
    assert payload["service_tier"] == "redacted"
    assert payload["max_output_tokens"] == 0
    assert "redacted" in payload["reason_codes"]
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "https://proxy.test/v1" not in encoded
    assert "Secret prompt" not in encoded


def test_local_browser_stack_endpoint_reports_runtime_active_safely():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = object()
    client = TestClient(app)

    response = client.get("/api/runtime/stack")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_active"] is True
    assert payload["llm_provider"] == "ollama"
    assert payload["model"] == "qwen3.5:4b"
    assert payload["stt_backend"] == "mlx-whisper"
    assert payload["tts_backend"] == "kokoro"
    assert payload["vision_enabled"] is False
    assert "service_tier" not in payload
    assert "runtime_active:true" in payload["reason_codes"]


def test_local_browser_operator_endpoint_reports_unavailable_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    payload = response.json()
    expected_keys = {
        "schema_version",
        "available",
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
        "reason_codes",
    }
    assert set(payload) == expected_keys
    assert payload["available"] is False
    for key in expected_keys - {"schema_version", "available", "reason_codes"}:
        assert payload[key]["available"] is False
        assert payload[key]["reason_codes"]
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_operator_endpoint_caches_poll_bursts(monkeypatch):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )
    calls = {"operator": 0}

    def fake_operator_snapshot(_runtime):
        calls["operator"] += 1
        return {
            "schema_version": 1,
            "available": False,
            "reason_codes": ["operator_workbench:test"],
        }

    monkeypatch.setattr(
        local_browser,
        "build_operator_workbench_snapshot",
        fake_operator_snapshot,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    assert client.get("/api/runtime/operator").status_code == 200
    assert client.get("/api/runtime/operator").status_code == 200
    assert calls["operator"] == 1

    assert (
        client.post(
            "/api/runtime/client-media",
            json={
                "available": True,
                "mode": "audio_video",
                "camera_state": "connected",
                "microphone_state": "connected",
            },
        ).status_code
        == 200
    )
    assert client.get("/api/runtime/operator").status_code == 200
    assert calls["operator"] == 2


def test_local_browser_operator_endpoint_reports_safe_fake_runtime():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def current_expression_state(self):
            return {
                "available": True,
                "persona_profile_id": "blink-default",
                "identity_label": "Blink, local non-human system",
                "modality": "browser",
                "teaching_mode_label": "walkthrough",
                "memory_persona_section_status": {"persona_expression": "available"},
                "voice_style_summary": "concise",
                "response_chunk_length": "concise",
                "pause_yield_hint": "brief",
                "interruption_strategy_label": "yield",
                "initiative_label": "proactive",
                "evidence_visibility_label": "rich",
                "correction_mode_label": "rigorous",
                "explanation_structure_label": "walkthrough",
                "humor_mode_label": "witty",
                "vividness_mode_label": "vivid",
                "sophistication_mode_label": "sophisticated",
                "character_presence_label": "character_rich",
                "story_mode_label": "recurring_motifs",
                "style_summary": "humor=witty; vividness=vivid; story=recurring_motifs",
                "humor_budget": 0.42,
                "playfulness": 0.34,
                "metaphor_density": 0.61,
                "safety_clamped": False,
                "expression_controls_hardware": True,
                "voice_policy": {
                    "available": True,
                    "modality": "browser",
                    "concise_chunking_active": True,
                    "chunking_mode": "concise",
                    "max_spoken_chunk_chars": 132,
                    "expression_controls_hardware": True,
                },
                "reason_codes": ["runtime_expression_state:available"],
                "prompt_text": "secret prompt",
                "private_working_memory": "hidden deliberation",
            }

        def current_behavior_control_profile(self):
            return default_behavior_control_profile(user_id="operator-user", agent_id="blink-test")

        def current_teaching_knowledge_routing(self):
            return {
                "schema_version": 1,
                "available": True,
                "selection_kind": "auto",
                "task_mode": "reply",
                "language": "zh",
                "teaching_mode": "walkthrough",
                "summary": "1 teaching knowledge items selected: exemplar=1",
                "selected_items": [
                    {
                        "item_kind": "exemplar",
                        "item_id": "exemplar:chinese_technical_explanation_bridge",
                        "title": "Chinese technical explanation bridge",
                        "source_label": "blink-default-teaching-canon",
                        "provenance_kind": "internal-pedagogy",
                        "provenance_version": "2026-04",
                        "rendered_text": "secret prompt",
                    }
                ],
                "estimated_tokens": 42,
                "reason_codes": ["knowledge_routing_decision:v1"],
                "rendered_text": "hidden deliberation",
            }

        def current_voice_metrics(self):
            return {
                "available": True,
                "response_count": 1,
                "chunk_count": 2,
                "max_chunk_chars": 88,
                "average_chunk_chars": 44.0,
                "expression_controls_hardware": True,
                "reason_codes": ["voice_metrics:available"],
                "event_id": "evt-secret",
            }

        def current_voice_input_health(self):
            return {
                "schema_version": 1,
                "available": True,
                "microphone_state": "receiving",
                "stt_state": "transcribed",
                "audio_frame_count": 3,
                "transcription_count": 1,
                "last_transcription_chars": 11,
                "reason_codes": ["voice_input_health:v1", "stt:transcribed"],
                "raw_transcript": "private speech",
            }

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["available"] is True
    assert payload["expression"]["available"] is True
    assert payload["teaching_knowledge"]["available"] is True
    assert payload["voice_metrics"]["available"] is True
    assert payload["memory"]["available"] is False
    assert payload["expression"]["payload"]["expression_controls_hardware"] is False
    assert payload["expression"]["payload"]["voice_policy"]["expression_controls_hardware"] is False
    assert payload["expression"]["payload"]["humor_mode_label"] == "witty"
    assert payload["expression"]["payload"]["character_presence_label"] == "character_rich"
    assert payload["expression"]["payload"]["humor_budget"] == 0.42
    assert (
        payload["teaching_knowledge"]["payload"]["current_decision"]["selected_items"][0]["item_id"]
        == "exemplar:chinese_technical_explanation_bridge"
    )
    assert payload["voice_metrics"]["payload"]["expression_controls_hardware"] is False
    assert payload["voice_metrics"]["payload"]["input_health"]["microphone_state"] == "receiving"
    assert payload["voice_metrics"]["payload"]["input_health"]["stt_state"] == "transcribed"
    assert payload["voice_metrics"]["payload"]["input_health"]["transcription_count"] == 1
    assert "secret prompt" not in encoded
    assert "hidden deliberation" not in encoded
    assert "evt-secret" not in encoded
    assert "private speech" not in encoded


def test_local_browser_operator_endpoint_shape_matches_workbench_contract(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeSnapshot:
        def as_dict(self):
            return {
                "schema_version": 1,
                "available": True,
                "expression": {
                    "available": True,
                    "summary": "Blink / walkthrough / browser",
                    "payload": {
                        "identity_label": "Blink, local non-human system",
                        "teaching_mode_label": "walkthrough",
                        "modality": "browser",
                        "initiative_label": "proactive",
                        "evidence_visibility_label": "rich",
                        "correction_mode_label": "rigorous",
                        "explanation_structure_label": "walkthrough",
                        "humor_mode_label": "witty",
                        "vividness_mode_label": "vivid",
                        "sophistication_mode_label": "sophisticated",
                        "character_presence_label": "character_rich",
                        "story_mode_label": "recurring_motifs",
                        "style_summary": "humor=witty; vividness=vivid; story=recurring_motifs",
                        "humor_budget": 0.42,
                        "playfulness": 0.34,
                        "metaphor_density": 0.61,
                        "safety_clamped": False,
                        "voice_style_summary": "concise",
                        "voice_policy": {
                            "available": True,
                            "chunking_mode": "concise",
                            "active_hints": ["concise_chunking"],
                            "unsupported_hints": ["speech_rate"],
                            "noop_reason_codes": ["voice_policy_noop:speech_rate:kokoro"],
                        },
                    },
                    "reason_codes": ["operator_expression:available"],
                },
                "behavior_controls": {
                    "available": True,
                    "summary": "depth=concise; voice=concise",
                    "payload": {
                        "compiled_effect_summary": "depth=concise; voice=concise",
                        "profile": {
                            "response_depth": "concise",
                            "directness": "balanced",
                            "warmth": "warm",
                            "teaching_mode": "walkthrough",
                            "memory_use": "normal",
                            "initiative_mode": "proactive",
                            "evidence_visibility": "rich",
                            "correction_mode": "rigorous",
                            "explanation_structure": "walkthrough",
                            "challenge_style": "gentle",
                            "voice_mode": "concise",
                            "question_budget": "low",
                            "humor_mode": "witty",
                            "vividness_mode": "vivid",
                            "sophistication_mode": "sophisticated",
                            "character_presence": "character_rich",
                            "story_mode": "recurring_motifs",
                        },
                    },
                    "reason_codes": ["operator_behavior_controls:available"],
                },
                "teaching_knowledge": {
                    "available": True,
                    "summary": "1 teaching knowledge items selected: exemplar=1",
                    "payload": {
                        "available": True,
                        "summary": "1 teaching knowledge items selected: exemplar=1",
                        "current_decision": {
                            "available": True,
                            "selection_kind": "auto",
                            "task_mode": "reply",
                            "language": "zh",
                            "teaching_mode": "walkthrough",
                            "selected_items": [
                                {
                                    "item_kind": "exemplar",
                                    "item_id": "exemplar:chinese_technical_explanation_bridge",
                                    "title": "Chinese technical explanation bridge",
                                    "source_label": "blink-default-teaching-canon",
                                }
                            ],
                            "estimated_tokens": 42,
                            "reason_codes": ["knowledge_routing_decision:v1"],
                        },
                        "recent_decisions": [],
                        "selected_item_counts": {"exemplar": 1},
                    },
                    "reason_codes": ["operator_teaching_knowledge:available"],
                },
                "voice_metrics": {
                    "available": True,
                    "summary": "2 chunks; max 88 chars; 0 interruptions",
                    "payload": {
                        "response_count": 1,
                        "concise_chunking_activation_count": 1,
                        "chunk_count": 2,
                        "max_chunk_chars": 88,
                        "average_chunk_chars": 44.0,
                        "interruption_frame_count": 0,
                        "buffer_flush_count": 1,
                        "buffer_discard_count": 0,
                        "last_chunking_mode": "concise",
                        "last_max_spoken_chunk_chars": 132,
                        "reason_codes": ["voice_metrics:available"],
                    },
                    "reason_codes": ["operator_voice_metrics:available"],
                },
                "memory": {
                    "available": True,
                    "summary": "1 visible memories: 1 preference",
                    "payload": {
                        "summary": "1 visible memories: 1 preference",
                        "health_summary": "Memory health available.",
                        "hidden_counts": {"suppressed": 0},
                        "records": [
                            {
                                "display_kind": "preference",
                                "summary": "Prefers concise answers",
                                "status": "active",
                                "currentness_status": "current",
                                "confidence": 0.91,
                                "pinned": True,
                                "used_in_current_turn": True,
                                "safe_provenance_label": "Remembered from your explicit preference.",
                            }
                        ],
                        "memory_persona_performance": {
                            "schema_version": 1,
                            "available": True,
                            "profile": "browser-zh-melo",
                            "memory_policy": "balanced",
                            "selected_memory_count": 1,
                            "suppressed_memory_count": 0,
                            "used_in_current_reply": [
                                {
                                    "memory_id": "memory_claim:user:operator:claim-safe",
                                    "display_kind": "preference",
                                    "title": "concise answers",
                                    "used_reason": "selected_for_relevant_continuity",
                                    "behavior_effect": "memory callback changed this reply",
                                    "reason_codes": ["source:context_selection"],
                                }
                            ],
                            "behavior_effects": ["memory_callback_active"],
                            "memory_continuity_trace": {
                                "schema_version": 1,
                                "profile": "browser-zh-melo",
                                "language": "zh",
                                "memory_effect": "cross_language_callback",
                                "cross_language_count": 1,
                                "selected_memory_count": 1,
                                "suppressed_memory_count": 0,
                                "selected_memories": [
                                    {
                                        "memory_id": "memory_claim:user:operator:claim-safe",
                                        "display_kind": "preference",
                                        "summary": "concise answers",
                                        "source_language": "en",
                                        "cross_language": True,
                                        "effect_labels": ["shorter_explanation"],
                                        "linked_discourse_episode_ids": [
                                            "discourse-episode-v3:operator"
                                        ],
                                        "reason_codes": ["memory_continuity:selected"],
                                    }
                                ],
                                "memory_continuity_v3": {
                                    "schema_version": 3,
                                    "selected_discourse_episodes": [
                                        {
                                            "discourse_episode_id": (
                                                "discourse-episode-v3:operator"
                                            ),
                                            "category_labels": ["user_preference"],
                                            "effect_labels": ["shorter_explanation"],
                                            "memory_ids": [
                                                "memory_claim:user:operator:claim-safe"
                                            ],
                                            "confidence_bucket": "high",
                                            "reason_codes": [
                                                "memory_continuity_v3:discourse_episode_ref"
                                            ],
                                        }
                                    ],
                                    "effect_labels": ["shorter_explanation"],
                                    "conflict_labels": [],
                                    "staleness_labels": [],
                                    "cross_language_transfer_count": 1,
                                    "reason_codes": ["memory_continuity_v3:available"],
                                },
                                "reason_codes": ["memory_continuity:trace"],
                            },
                            "persona_references": [
                                {
                                    "reference_id": "persona:memory_callback",
                                    "mode": "memory_callback",
                                    "label": "memory callback",
                                    "applies": True,
                                    "behavior_effect": (
                                        "use selected public memories as brief callbacks"
                                    ),
                                    "reason_codes": ["persona_reference:memory_callback"],
                                }
                            ],
                            "persona_anchor_bank_v3": {
                                "schema_version": 3,
                                "anchor_count": 1,
                                "required_situation_keys": ["memory_callback"],
                                "anchors": [
                                    {
                                        "schema_version": 3,
                                        "anchor_id": "persona-anchor-v3:memory_callback",
                                        "situation_key": "memory_callback",
                                        "zh_example": "只用公开记忆作简短承接。",
                                        "en_example": "Use visible memory as a brief callback.",
                                        "behavior_constraints": [
                                            "Use memory as supporting context."
                                        ],
                                        "negative_examples": ["Do not expose raw memory records."],
                                        "stance_label": (
                                            "continuity_without_private_exposition"
                                        ),
                                        "response_shape_label": "brief_callback_then_answer",
                                        "reason_codes": [
                                            "persona_reference_bank:v3",
                                            "persona_anchor:memory_callback",
                                        ],
                                    }
                                ],
                                "reason_codes": ["persona_reference_bank:v3"],
                            },
                            "performance_plan_v3": {
                                "schema_version": 3,
                                "persona_anchor_refs_v3": [
                                    {
                                        "schema_version": 3,
                                        "anchor_id": "persona-anchor-v3:memory_callback",
                                        "situation_key": "memory_callback",
                                        "stance_label": (
                                            "continuity_without_private_exposition"
                                        ),
                                        "response_shape_label": "brief_callback_then_answer",
                                        "behavior_constraint_count": 1,
                                        "negative_example_count": 1,
                                        "reason_codes": [
                                            "persona_reference_bank:v3",
                                            "persona_anchor:memory_callback",
                                        ],
                                    }
                                ],
                            },
                            "summary": "1 memories used; 0 suppressed; 1 persona references active.",
                            "reason_codes": ["memory_persona_performance:v1"],
                        },
                    },
                    "reason_codes": ["operator_memory:available"],
                },
                "practice": {
                    "available": True,
                    "summary": "1 recent practice targets.",
                    "payload": {
                        "scenario_family_counts": {"debugging": 1},
                        "reason_code_counts": {"selected": 1},
                        "recent_targets": [
                            {
                                "scenario_family": "debugging",
                                "execution_backend": "local",
                                "selected_profile_id": "profile-a",
                            }
                        ],
                    },
                    "reason_codes": ["operator_practice:available"],
                },
                "adapters": {
                    "available": True,
                    "summary": "1 adapter cards.",
                    "payload": {
                        "state_counts": {"shadow": 1},
                        "family_counts": {"voice": 1},
                        "current_default_cards": [
                            {
                                "adapter_family": "voice",
                                "backend_id": "local",
                                "promotion_state": "shadow",
                            }
                        ],
                        "pending_or_blocked_decisions": [],
                    },
                    "reason_codes": ["operator_adapters:available"],
                },
                "sim_to_real": {
                    "available": True,
                    "summary": "1 readiness reports; 0 rollbacks.",
                    "payload": {
                        "readiness_counts": {"shadow_ready": 1},
                        "promotion_state_counts": {"shadow": 1},
                        "readiness_reports": [
                            {
                                "adapter_family": "voice",
                                "backend_id": "local",
                                "promotion_state": "shadow",
                                "governance_only": True,
                            }
                        ],
                    },
                    "reason_codes": ["operator_sim_to_real:available"],
                },
                "rollout_status": {
                    "available": False,
                    "summary": "Live rollout controller is not active in this slice.",
                    "payload": {
                        "governance_only": True,
                        "live_routing_active": False,
                    },
                    "reason_codes": ["operator_rollout_status:unavailable"],
                },
                "performance_learning": {
                    "available": True,
                    "summary": "1 recent preference pairs; 1 policy proposals.",
                    "payload": {
                        "schema_version": 3,
                        "available": True,
                        "pair_count": 1,
                        "proposal_count": 1,
                        "dimensions": ["felt_heard", "not_fake_human"],
                        "recent_pairs": [
                            {
                                "pair_id": "performance-pref-pair:operator",
                                "profile": "browser-zh-melo",
                                "winner": "b",
                                "candidate_a": {"candidate_id": "baseline"},
                                "candidate_b": {"candidate_id": "candidate"},
                                "failure_labels": [],
                                "improvement_labels": ["voice_pacing_better"],
                            }
                        ],
                        "policy_proposals": [
                            {
                                "schema_version": 3,
                                "proposal_id": "performance-policy-proposal:operator",
                                "status": "proposed",
                                "target": "speech_chunking_bias",
                                "summary": "Shorten voice chunks.",
                                "behavior_control_updates": {"voice_mode": "concise"},
                                "source_pair_ids": ["performance-pref-pair:operator"],
                                "evidence_refs": [],
                                "dimension_scores": {"voice_pacing": 2},
                                "reason_codes": [
                                    "performance_learning_policy_proposal:v3"
                                ],
                            }
                        ],
                        "reason_codes": ["performance_learning:v3"],
                    },
                    "reason_codes": ["operator_performance_learning:available"],
                },
                "reason_codes": ["operator_workbench:v1", "operator_workbench:available"],
            }

    monkeypatch.setattr(
        local_browser,
        "build_operator_workbench_snapshot",
        lambda _runtime: FakeSnapshot(),
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = object()
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["memory"]["payload"]["records"][0]["used_in_current_turn"] is True
    assert payload["memory"]["payload"]["used_in_current_reply"][0]["title"] == (
        "concise answers"
    )
    assert payload["memory"]["payload"]["behavior_effects"] == ["memory_callback_active"]
    operator_continuity_v3 = payload["memory"]["payload"]["memory_continuity_trace"][
        "memory_continuity_v3"
    ]
    assert operator_continuity_v3["effect_labels"] == ["shorter_explanation"]
    assert operator_continuity_v3["selected_discourse_episodes"][0]["category_labels"] == [
        "user_preference"
    ]
    assert payload["memory"]["payload"]["persona_references"][0]["mode"] == "memory_callback"
    assert (
        payload["memory"]["payload"]["persona_anchor_refs_v3"][0]["situation_key"]
        == "memory_callback"
    )
    assert payload["memory"]["payload"]["persona_anchor_bank_v3"]["anchor_count"] == 1
    assert payload["memory"]["payload"]["memory_persona_performance"]["available"] is True
    assert payload["behavior_controls"]["payload"]["profile"]["response_depth"] == "concise"
    assert payload["behavior_controls"]["payload"]["profile"]["initiative_mode"] == "proactive"
    assert payload["behavior_controls"]["payload"]["profile"]["evidence_visibility"] == "rich"
    assert payload["behavior_controls"]["payload"]["profile"]["correction_mode"] == "rigorous"
    assert payload["behavior_controls"]["payload"]["profile"]["explanation_structure"] == (
        "walkthrough"
    )
    assert payload["teaching_knowledge"]["payload"]["current_decision"]["selection_kind"] == "auto"
    assert payload["teaching_knowledge"]["payload"]["selected_item_counts"] == {"exemplar": 1}
    assert payload["expression"]["payload"]["initiative_label"] == "proactive"
    assert payload["expression"]["payload"]["evidence_visibility_label"] == "rich"
    assert payload["expression"]["payload"]["humor_mode_label"] == "witty"
    assert payload["expression"]["payload"]["character_presence_label"] == "character_rich"
    assert payload["expression"]["payload"]["voice_policy"]["chunking_mode"] == "concise"
    assert payload["expression"]["payload"]["voice_policy"]["noop_reason_codes"] == [
        "voice_policy_noop:speech_rate:kokoro"
    ]
    assert payload["voice_metrics"]["payload"]["response_count"] == 1
    assert payload["voice_metrics"]["payload"]["chunk_count"] == 2
    assert payload["voice_metrics"]["payload"]["buffer_flush_count"] == 1
    assert payload["voice_metrics"]["payload"]["last_chunking_mode"] == "concise"
    assert payload["practice"]["payload"]["scenario_family_counts"] == {"debugging": 1}
    assert payload["adapters"]["payload"]["state_counts"] == {"shadow": 1}
    assert payload["sim_to_real"]["payload"]["readiness_counts"] == {"shadow_ready": 1}
    assert payload["rollout_status"]["payload"]["live_routing_active"] is False
    assert payload["performance_learning"]["payload"]["pair_count"] == 1
    assert payload["performance_learning"]["payload"]["policy_proposals"][0]["target"] == (
        "speech_chunking_bias"
    )


def test_local_browser_operator_endpoint_sanitizes_malformed_snapshot(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeSnapshot:
        def as_dict(self):
            return {
                "schema_version": "not-an-int",
                "available": "yes",
                "expression": {
                    "available": "yes",
                    "summary": "Traceback in /tmp/brain.db",
                    "payload": {
                        "available": "yes",
                        "identity_label": "Traceback in /tmp/brain.db",
                        "humor_budget": float("nan"),
                        "expression_controls_hardware": True,
                        "prompt_text": "hidden system prompt",
                        "private_working_memory": "hidden deliberation",
                        "voice_policy": {
                            "available": "yes",
                            "chunking_mode": "secret /tmp/brain.db",
                            "expression_controls_hardware": True,
                        },
                    },
                    "reason_codes": [
                        "operator_expression:available",
                        "secret /tmp/brain.db",
                    ],
                    "details": {"raw_json_block": {"db_path": "/tmp/brain.db"}},
                },
                "teaching_knowledge": {
                    "available": "yes",
                    "summary": "Teaching knowledge available.",
                    "payload": {
                        "current_decision": {
                            "selection_kind": "auto",
                            "rendered_text": "hidden prompt packet",
                            "selected_items": [
                                {
                                    "item_id": "exemplar:safe",
                                    "title": "Safe exemplar",
                                    "rendered_text": "hidden prompt packet",
                                }
                            ],
                        },
                        "selected_item_counts": {"exemplar": 1},
                    },
                    "reason_codes": ["operator_teaching_knowledge:available"],
                },
                "voice_metrics": {
                    "available": "yes",
                    "summary": "secret /tmp/brain.db",
                    "payload": {
                        "available": "yes",
                        "response_count": "2",
                        "expression_controls_hardware": True,
                        "input_health": {
                            "available": "yes",
                            "microphone_state": "receiving",
                            "stt_state": "transcribed",
                            "raw_transcript": "private speech",
                        },
                        "event_id": "evt-secret",
                    },
                    "reason_codes": ["operator_voice_metrics:available"],
                },
                "memory": {
                    "available": "yes",
                    "summary": "1 visible memories.",
                    "payload": {
                        "records": [
                            {
                                "display_kind": "preference",
                                "summary": "Prefers concise answers",
                                "used_in_current_turn": "yes",
                                "user_actions": ["forget", "secret /tmp/brain.db"],
                                "source_refs": ["claim-secret"],
                            }
                        ],
                        "hidden_counts": {"secret /tmp/brain.db": 3},
                        "memory_persona_performance": {
                            "available": "yes",
                            "profile": "browser-zh-melo",
                            "used_in_current_reply": [
                                {
                                    "memory_id": "memory_claim:user:secret:claim",
                                    "title": "system_prompt secret",
                                    "used_reason": "raw transcript",
                                    "behavior_effect": "secret /tmp/brain.db",
                                }
                            ],
                            "behavior_effects": ["memory_callback_active", "secret /tmp/brain.db"],
                            "persona_references": [
                                {
                                    "mode": "memory_callback",
                                    "label": "system_prompt secret",
                                    "applies": "yes",
                                    "behavior_effect": "Traceback /tmp/brain.db",
                                }
                            ],
                        },
                    },
                    "reason_codes": ["operator_memory:available"],
                },
                "practice": {
                    "available": "yes",
                    "summary": "Practice available.",
                    "payload": {
                        "recent_targets": [
                            {
                                "scenario_family": "debugging",
                                "selected_profile_id": "profile-a",
                                "raw_json_block": {"db_path": "/tmp/brain.db"},
                            }
                        ],
                        "user_text_preview": "private user text",
                    },
                    "reason_codes": ["operator_practice:available"],
                },
                "reason_codes": ["operator_workbench:v1", "secret /tmp/brain.db"],
            }

    monkeypatch.setattr(
        local_browser,
        "build_operator_workbench_snapshot",
        lambda _runtime: FakeSnapshot(),
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = object()
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["available"] is False
    assert payload["expression"]["available"] is False
    assert payload["expression"]["summary"] == "redacted"
    assert payload["expression"]["payload"]["identity_label"] == "redacted"
    assert payload["expression"]["payload"]["humor_budget"] == 0.0
    assert payload["expression"]["payload"]["expression_controls_hardware"] is False
    assert payload["expression"]["payload"]["voice_policy"]["expression_controls_hardware"] is False
    assert payload["voice_metrics"]["payload"]["available"] is False
    assert payload["voice_metrics"]["payload"]["expression_controls_hardware"] is False
    assert payload["voice_metrics"]["payload"]["input_health"]["microphone_state"] == ("receiving")
    assert payload["memory"]["payload"]["records"][0]["summary"] == "Prefers concise answers"
    assert payload["memory"]["payload"]["records"][0]["used_in_current_turn"] is False
    assert payload["memory"]["payload"]["records"][0]["user_actions"] == ["forget"]
    assert payload["memory"]["payload"]["used_in_current_reply"][0]["title"] == "redacted"
    assert payload["memory"]["payload"]["behavior_effects"] == ["memory_callback_active", "redacted"]
    assert payload["memory"]["payload"]["persona_references"][0]["label"] == "redacted"
    assert payload["practice"]["payload"]["recent_targets"][0]["selected_profile_id"] == (
        "profile-a"
    )
    assert payload["expression"]["reason_codes"] == ["operator_expression:unavailable", "redacted"]
    assert payload["reason_codes"] == [
        "operator_workbench:v1",
        "operator_workbench:unavailable",
        "redacted",
    ]
    for blocked in (
        "secret",
        "Traceback",
        "/tmp/brain.db",
        "raw_json_block",
        "private_working_memory",
        "raw_transcript",
        "user_text_preview",
        "prompt_text",
        "rendered_text",
        "source_refs",
        "evt-secret",
        "private speech",
    ):
        assert blocked not in encoded


def test_local_browser_operator_endpoint_errors_are_safe(monkeypatch):
    from fastapi.testclient import TestClient

    def failing_snapshot(_runtime):
        raise RuntimeError("secret /tmp/brain.db")

    monkeypatch.setattr(local_browser, "build_operator_workbench_snapshot", failing_snapshot)
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["available"] is False
    assert payload["teaching_knowledge"]["available"] is False
    assert (
        "operator_teaching_knowledge:unavailable" in payload["teaching_knowledge"]["reason_codes"]
    )
    assert "runtime_operator_error" in payload["reason_codes"]
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_operator_endpoint_builds_snapshot_off_event_loop(monkeypatch):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )
    calls = {"to_thread": 0}
    original_to_thread = local_browser.asyncio.to_thread

    async def fake_to_thread(func, /, *args, **kwargs):
        calls["to_thread"] += 1
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(local_browser.asyncio, "to_thread", fake_to_thread)

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/operator")

    assert response.status_code == 200
    assert calls["to_thread"] == 1


@pytest.mark.parametrize(
    ("endpoint_action", "method_name", "request_json", "expected_to_state"),
    (
        ("approve", "evaluate_plan", {}, "approved"),
        ("activate", "activate_plan", {"traffic_fraction": 0.05}, "active_limited"),
        ("pause", "pause_plan", {}, "paused"),
        ("resume", "resume_plan", {"traffic_fraction": 0.04}, "active_limited"),
        (
            "rollback",
            "rollback_plan",
            {"regression_codes": ["operator_requested_rollback"]},
            "rolled_back",
        ),
    ),
)
def test_local_browser_rollout_action_endpoint_forwards_to_controller(
    endpoint_action,
    method_name,
    request_json,
    expected_to_state,
):
    from fastapi.testclient import TestClient

    calls = []

    class FakePlan:
        def __init__(self, state):
            self.state = state

        def as_dict(self):
            return {
                "plan_id": "plan-1",
                "adapter_family": "world_model",
                "candidate_backend_id": "candidate-world",
                "candidate_backend_version": "v2",
                "routing_state": self.state,
                "promotion_state": "canary",
                "traffic_fraction": 0.05 if self.state == "active_limited" else 0,
                "scope_key": "local",
                "expires_at": "2026-04-25T00:00:00+00:00",
                "embodied_live": False,
                "budget_id": "budget-1",
                "reason_codes": [f"routing_state:{self.state}"],
                "details": {"raw_json_block": "/tmp/brain.db"},
            }

    class FakeResult:
        def as_dict(self):
            return {
                "schema_version": 1,
                "accepted": True,
                "applied": True,
                "action": endpoint_action,
                "plan_id": "plan-1",
                "from_state": "proposed",
                "to_state": expected_to_state,
                "traffic_fraction": request_json.get("traffic_fraction", 0),
                "decision": {"decision_id": "decision-secret"},
                "reason_codes": ["rollout_decision_accepted", f"rollout_{endpoint_action}"],
                "details": {"trace": "Traceback /tmp/brain.db"},
            }

    class FakeController:
        def __init__(self):
            self._plan = FakePlan("proposed")

        def plan(self, plan_id):
            assert plan_id == "plan-1"
            return self._plan

        def _record(self, name, plan_id, **kwargs):
            calls.append((name, plan_id, kwargs))
            self._plan = FakePlan(expected_to_state)
            return FakeResult()

        def evaluate_plan(self, plan_id, **kwargs):
            return self._record("evaluate_plan", plan_id, **kwargs)

        def activate_plan(self, plan_id, **kwargs):
            return self._record("activate_plan", plan_id, **kwargs)

        def pause_plan(self, plan_id, **kwargs):
            return self._record("pause_plan", plan_id, **kwargs)

        def resume_plan(self, plan_id, **kwargs):
            return self._record("resume_plan", plan_id, **kwargs)

        def rollback_plan(self, plan_id, **kwargs):
            return self._record("rollback_plan", plan_id, **kwargs)

    class FakeRuntime:
        live_routing_controller = FakeController()

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(f"/api/runtime/rollout/plan-1/{endpoint_action}", json=request_json)

    assert response.status_code == 200
    payload = response.json()
    assert calls[0][0] == method_name
    assert calls[0][1] == "plan-1"
    if "traffic_fraction" in request_json:
        assert calls[0][2]["traffic_fraction"] == request_json["traffic_fraction"]
    if endpoint_action == "rollback":
        assert calls[0][2]["regression_codes"] == ("operator_requested_rollback",)
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["before"]["routing_state"] == "proposed"
    assert payload["after"]["routing_state"] == expected_to_state
    assert "rollout_decision_accepted" in payload["reason_codes"]
    assert "decision" not in payload
    encoded = str(payload)
    assert "decision-secret" not in encoded
    assert "raw_json_block" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_rollout_action_endpoint_rejects_missing_controller():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = object()
    client = TestClient(app)

    response = client.post("/api/runtime/rollout/plan-1/approve", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert "live_routing_controller_missing" in payload["reason_codes"]


def test_local_browser_rollout_action_endpoint_bounds_request_arguments():
    from fastapi.testclient import TestClient

    calls = []

    class FakeResult:
        def __init__(self, action):
            self.action = action

        def as_dict(self):
            return {
                "schema_version": 1,
                "accepted": True,
                "applied": True,
                "action": self.action,
                "plan_id": "plan-1",
                "from_state": "approved",
                "to_state": "active_limited" if self.action == "activate" else "rolled_back",
                "traffic_fraction": 0,
                "reason_codes": ["rollout_decision_accepted"],
            }

    class FakeController:
        def activate_plan(self, plan_id, **kwargs):
            calls.append(("activate", plan_id, kwargs))
            return FakeResult("activate")

        def rollback_plan(self, plan_id, **kwargs):
            calls.append(("rollback", plan_id, kwargs))
            return FakeResult("rollback")

    class FakeRuntime:
        live_routing_controller = FakeController()

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    activate = client.post(
        "/api/runtime/rollout/plan-1/activate",
        json={"traffic_fraction": "nan", "decided_at": "secret /tmp/brain.db"},
    )
    rollback = client.post(
        "/api/runtime/rollout/plan-1/rollback",
        json={
            "regression_codes": "secret /tmp/brain.db",
            "decided_at": "Traceback /tmp/brain.db",
        },
    )

    assert activate.status_code == 200
    assert rollback.status_code == 200
    assert calls[0] == (
        "activate",
        "plan-1",
        {
            "traffic_fraction": 0.0,
            "operator_acknowledged": None,
            "decided_at": "redacted",
        },
    )
    assert calls[1] == (
        "rollback",
        "plan-1",
        {
            "regression_codes": (),
            "decided_at": "redacted",
        },
    )
    encoded = str(activate.json()) + str(rollback.json())
    assert "Secret prompt" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_rollout_action_endpoint_sanitizes_malformed_result():
    from fastapi.testclient import TestClient

    class FakePlan:
        def as_dict(self):
            return {
                "plan_id": "plan-1",
                "adapter_family": "world_model",
                "candidate_backend_id": "candidate-world",
                "candidate_backend_version": "v2",
                "routing_state": "proposed",
                "promotion_state": "canary",
                "traffic_fraction": 0,
                "scope_key": "local",
                "expires_at": "2026-04-25T00:00:00+00:00",
                "embodied_live": False,
                "budget_id": "budget-1",
                "reason_codes": ["routing_state:proposed"],
            }

    class FakeController:
        def plan(self, _plan_id):
            return FakePlan()

        def evaluate_plan(self, _plan_id, **_kwargs):
            return {
                "schema_version": "not-an-int",
                "accepted": True,
                "applied": True,
                "action": "Traceback /tmp/brain.db",
                "plan_id": "secret /tmp/brain.db",
                "from_state": "Traceback /tmp/brain.db",
                "to_state": "approved",
                "traffic_fraction": "not-a-float",
                "reason_codes": ["rollout_decision_accepted", "secret /tmp/brain.db"],
                "details": {"raw_json_block": "/tmp/brain.db"},
            }

    class FakeRuntime:
        live_routing_controller = FakeController()

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/runtime/rollout/plan-1/approve", json={})

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["action"] == "redacted"
    assert payload["plan_id"] == "redacted"
    assert payload["from_state"] == "redacted"
    assert payload["to_state"] == "approved"
    assert payload["traffic_fraction"] == 0.0
    assert payload["reason_codes"] == ["rollout_decision_accepted", "redacted"]
    assert "raw_json_block" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_rollout_evidence_endpoint_is_public_safe():
    from fastapi.testclient import TestClient

    session_ids = SimpleNamespace(
        user_id="user-browser-test",
        agent_id="blink-test",
        thread_id="thread-test",
    )

    class FakePracticePlan:
        def as_dict(self):
            return {
                "plan_id": "practice-plan-1",
                "targets": [{"scenario_id": "debug-1"}],
                "summary": "Practice debugging.",
                "reason_code_counts": {
                    "recovery_pressure": 1,
                    "Traceback /tmp/brain.db": 2,
                },
                "artifact_paths": {"json": "/tmp/brain.db"},
                "updated_at": "2026-04-24T00:00:00+00:00",
            }

    class FakeBenchmarkReport:
        def as_dict(self):
            return {
                "report_id": "report-1",
                "adapter_family": "world_model",
                "candidate_backend_id": "candidate-world",
                "candidate_backend_version": "v2",
                "scenario_count": 4,
                "compared_family_count": 2,
                "smoke_suite_green": True,
                "benchmark_passed": True,
                "blocked_reason_codes": ["Traceback /tmp/brain.db"],
                "artifact_paths": {"json": "/tmp/brain.db"},
                "details": {"trace": "Traceback /tmp/brain.db"},
                "updated_at": "2026-04-24T00:00:00+00:00",
            }

    class FakeStore:
        def recent_episodes(self, **kwargs):
            assert kwargs["user_id"] == session_ids.user_id
            assert kwargs["thread_id"] == session_ids.thread_id
            return [
                SimpleNamespace(
                    id=7,
                    created_at="2026-04-24T00:00:00+00:00",
                    user_text="Traceback /tmp/brain.db",
                    assistant_summary="Operator-safe summary.",
                    tool_calls_json='{"raw": "/tmp/brain.db"}',
                )
            ]

        def build_practice_director_projection(self, **_kwargs):
            return SimpleNamespace(recent_plans=[FakePracticePlan()])

        def build_adapter_governance_projection(self, **_kwargs):
            return SimpleNamespace(recent_reports=[FakeBenchmarkReport()])

    class FakeRuntime:
        store = FakeStore()

        def session_resolver(self):
            return session_ids

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/rollout/evidence")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["live_episodes"][0]["id"] == 7
    assert payload["live_episodes"][0]["assistant_summary"] == "Operator-safe summary."
    assert "user_text_preview" not in payload["live_episodes"][0]
    assert payload["practice_plans"][0]["target_count"] == 1
    assert payload["practice_plans"][0]["plan_id"] == "practice-plan-1"
    assert payload["practice_plans"][0]["reason_code_counts"]["redacted"] == 2
    assert payload["benchmark_reports"][0]["report_id"] == "report-1"
    assert payload["benchmark_reports"][0]["blocked_reason_codes"] == ["redacted"]
    encoded = str(payload)
    assert "tool_calls_json" not in encoded
    assert "artifact_paths" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded

    evidence_response = client.get("/api/runtime/evidence")

    assert evidence_response.status_code == 200
    evidence_payload = evidence_response.json()
    assert evidence_payload["available"] is True
    assert evidence_payload["episode_count"] == 3
    assert evidence_payload["source_counts"] == {"eval": 1, "live": 1, "practice": 1}
    assert {row["source"] for row in evidence_payload["rows"]} == {
        "eval",
        "live",
        "practice",
    }
    live_row = next(row for row in evidence_payload["rows"] if row["source"] == "live")
    assert live_row["summary"] == "Operator-safe summary."
    assert "user_text_preview" not in live_row
    practice_row = next(row for row in evidence_payload["rows"] if row["source"] == "practice")
    assert practice_row["links"][0]["link_kind"] == "practice_plan"
    eval_row = next(row for row in evidence_payload["rows"] if row["source"] == "eval")
    assert eval_row["links"][0]["link_kind"] == "benchmark_report"
    evidence_encoded = str(evidence_payload)
    assert "tool_calls_json" not in evidence_encoded
    assert "artifact_paths" not in evidence_encoded
    assert "Traceback" not in evidence_encoded
    assert "/tmp/brain.db" not in evidence_encoded


def test_local_browser_episode_evidence_endpoint_sanitizes_malformed_snapshot(monkeypatch):
    from fastapi.testclient import TestClient

    session_ids = SimpleNamespace(
        user_id="user-browser-test",
        agent_id="blink-test",
        thread_id="thread-test",
    )
    calls = {}

    class FakeSnapshot:
        def as_dict(self):
            return {
                "schema_version": "not-an-int",
                "available": True,
                "generated_at": "Traceback /tmp/brain.db",
                "summary": "secret /tmp/brain.db",
                "episode_count": "not-an-int",
                "source_counts": {"live": 1, "Traceback /tmp/brain.db": 2},
                "reason_code_counts": {"ok": 1, "secret /tmp/brain.db": 2},
                "rows": [
                    {
                        "evidence_id": "secret /tmp/brain.db",
                        "episode_id": "Traceback /tmp/brain.db",
                        "source": "live",
                        "scenario_id": "secret /tmp/brain.db",
                        "summary": "Traceback /tmp/brain.db",
                        "source_run_id": "secret /tmp/brain.db",
                        "execution_backend": "local",
                        "candidate_backend_id": "Traceback /tmp/brain.db",
                        "candidate_backend_version": "v1",
                        "outcome_label": "ok",
                        "task_success": "yes",
                        "safety_success": True,
                        "preview_only": True,
                        "scenario_count": "not-an-int",
                        "artifact_refs": [
                            {
                                "artifact_id": "secret /tmp/brain.db",
                                "artifact_kind": "json",
                                "uri_kind": "file",
                                "raw_uri": "/tmp/brain.db",
                                "reason_codes": ["artifact:redacted", "secret /tmp/brain.db"],
                            }
                        ],
                        "links": [
                            {
                                "link_kind": "Traceback /tmp/brain.db",
                                "link_id": "benchmark-1",
                                "details": {"raw_json_block": "/tmp/brain.db"},
                                "reason_codes": ["link:ok", "secret /tmp/brain.db"],
                            }
                        ],
                        "started_at": "Traceback /tmp/brain.db",
                        "ended_at": "2026-04-24T00:00:00+00:00",
                        "generated_at": "Traceback /tmp/brain.db",
                        "reason_codes": ["evidence:ok", "secret /tmp/brain.db"],
                        "reason_code_categories": ["ok", "Traceback /tmp/brain.db"],
                        "details": {"raw_json_block": "/tmp/brain.db"},
                    }
                ],
                "details": {"artifact_paths": {"json": "/tmp/brain.db"}},
                "reason_codes": ["episode_evidence:available", "secret /tmp/brain.db"],
            }

    def fake_build_episode_evidence_index(**kwargs):
        calls.update(kwargs)
        return FakeSnapshot()

    monkeypatch.setattr(
        local_browser,
        "build_episode_evidence_index",
        fake_build_episode_evidence_index,
    )

    class FakeRuntime:
        store = object()
        presence_scope_key = "Traceback /tmp/brain.db"

        def session_resolver(self):
            return session_ids

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/evidence")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert calls["presence_scope_key"] == "redacted"
    assert payload["schema_version"] == 1
    assert payload["available"] is True
    assert payload["generated_at"] == "redacted"
    assert payload["summary"] == "redacted"
    assert payload["episode_count"] == 0
    assert payload["source_counts"]["redacted"] == 2
    assert payload["reason_code_counts"]["redacted"] == 2
    assert payload["rows"][0]["evidence_id"] == "redacted"
    assert payload["rows"][0]["task_success"] is None
    assert payload["rows"][0]["safety_success"] is True
    assert payload["rows"][0]["artifact_refs"][0]["artifact_id"] == "redacted"
    assert payload["rows"][0]["links"][0]["link_kind"] == "redacted"
    assert payload["rows"][0]["reason_code_categories"] == ["ok", "redacted"]
    assert "raw_json_block" not in encoded
    assert "artifact_paths" not in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_behavior_controls_endpoint_reports_unavailable_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/behavior-controls")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["profile"] is None
    assert payload["compiled_effect_summary"] == ""
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_behavior_controls_endpoint_reports_fake_runtime_profile():
    from fastapi.testclient import TestClient

    profile = default_behavior_control_profile(
        user_id="user-browser-test",
        agent_id="blink-test",
    )

    class FakeRuntime:
        def current_behavior_control_profile(self):
            return profile

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/behavior-controls")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["profile"]["user_id"] == "user-browser-test"
    assert payload["profile"]["response_depth"] == "balanced"
    assert payload["profile"]["schema_version"] == 3
    assert payload["profile"]["initiative_mode"] == "balanced"
    assert payload["profile"]["evidence_visibility"] == "compact"
    assert payload["profile"]["correction_mode"] == "precise"
    assert payload["profile"]["explanation_structure"] == "answer_first"
    assert payload["profile"]["humor_mode"] == "witty"
    assert payload["profile"]["vividness_mode"] == "vivid"
    assert payload["profile"]["character_presence"] == "character_rich"
    assert "depth=balanced" in payload["compiled_effect_summary"]
    assert "humor=witty" in payload["compiled_effect_summary"]
    assert "behavior_controls_state:available" in payload["reason_codes"]
    assert "block_id" not in str(payload)
    assert "source_event_id" not in str(payload)


def test_local_browser_style_presets_endpoint_is_public_safe():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/style-presets")

    assert response.status_code == 200
    payload = response.json()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["available"] is True
    assert payload["default_preset_id"] == "witty_sophisticated"
    preset = next(item for item in payload["presets"] if item["preset_id"] == "witty_sophisticated")
    assert preset["recommended"] is True
    assert preset["control_updates"]["humor_mode"] == "witty"
    assert preset["control_updates"]["character_presence"] == "character_rich"
    assert preset["language_fit"]["zh"] == "excellent"
    for forbidden in (
        "system_prompt",
        "prompt_text",
        "api_key",
        "OPENAI_API_KEY",
        "Authorization",
        "base_url",
        "hardware_control",
        "romance",
        "exclusive romantic",
        "human_identity",
        "Traceback",
        "brain.db",
    ):
        assert forbidden not in encoded


def test_local_browser_memory_persona_preset_preview_is_public_safe():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview"
    )

    assert response.status_code == 200
    payload = response.json()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["available"] is True
    assert payload["accepted"] is True
    assert payload["applied"] is False
    assert payload["preset_id"] == "witty_sophisticated"
    assert payload["counts"]["accepted_candidates"] == 16
    assert payload["counts"]["rejected_entries"] == 0
    assert any(candidate["kind"] == "behavior_controls" for candidate in payload["candidates"])
    assert "memory_persona_preset_preview:available" in payload["reason_codes"]
    for forbidden in (
        "system_prompt",
        "prompt_text",
        "api_key",
        "OPENAI_API_KEY",
        "Authorization",
        "base_url",
        "hardware_control",
        "human_identity",
        "Traceback",
        "brain.db",
    ):
        assert forbidden not in encoded


def test_local_browser_memory_persona_preset_preview_sanitizes_runtime_scope():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def session_resolver(self):
            return SimpleNamespace(
                user_id="secret /tmp/brain.db",
                agent_id="Traceback /tmp/brain.db",
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview"
    )

    assert response.status_code == 200
    payload = response.json()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["available"] is True
    assert payload["accepted"] is True
    assert any(candidate["summary"] == "redacted" for candidate in payload["candidates"])
    assert "redacted" in encoded
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_local_browser_memory_persona_preset_apply_requires_active_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    preview = client.get(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview"
    ).json()

    response = client.post(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply",
        json={"approved_report": preview},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert "runtime_memory_surface_missing" in payload["reason_codes"]


def test_local_browser_memory_persona_preset_apply_writes_active_runtime(tmp_path):
    from fastapi.testclient import TestClient

    from blink.brain.persona import load_behavior_control_profile
    from blink.brain.session import resolve_brain_session_ids
    from blink.brain.store import BrainStore

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="story-browser")

    class FakeRuntime:
        def __init__(self):
            self.store = store

        def session_resolver(self):
            return session_ids

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)
    preview = client.get(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview"
    ).json()

    response = client.post(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply",
        json={"approved_report": preview},
    )

    assert response.status_code == 200
    payload = response.json()
    controls = load_behavior_control_profile(store=store, session_ids=session_ids)
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["counts"]["memory_written"] == 15
    assert payload["counts"]["behavior_controls_applied"] == 1
    assert controls.story_mode == "recurring_motifs"
    assert controls.character_presence == "character_rich"
    assert "memory_persona_preset_apply:accepted" in payload["reason_codes"]


def test_local_browser_memory_persona_preset_apply_repeated_is_noop(tmp_path):
    from fastapi.testclient import TestClient

    from blink.brain.session import resolve_brain_session_ids
    from blink.brain.store import BrainStore

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="story-browser")

    class FakeRuntime:
        def __init__(self):
            self.store = store

        def session_resolver(self):
            return session_ids

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)
    preview = client.get(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview"
    ).json()
    first = client.post(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply",
        json={"approved_report": preview},
    ).json()
    second = client.post(
        "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply",
        json={"approved_report": preview},
    ).json()

    assert first["applied"] is True
    assert second["accepted"] is True
    assert second["applied"] is False
    assert second["counts"]["memory_written"] == 0
    assert second["counts"]["memory_noop"] == 15
    assert second["counts"]["behavior_controls_noop"] == 1
    assert all(entry["status"] == "noop" for entry in second["applied_entries"])
    assert "memory_persona_apply_noop" in second["reason_codes"]


def test_local_browser_behavior_controls_update_rejects_inactive_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post("/api/runtime/behavior-controls", json={"response_depth": "deep"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_behavior_controls_update_forwards_to_fake_runtime():
    from fastapi.testclient import TestClient

    calls = {}
    profile = default_behavior_control_profile(
        user_id="user-browser-test",
        agent_id="blink-test",
    )
    updated_profile_data = profile.as_dict()
    updated_profile_data.update({"response_depth": "deep", "directness": "rigorous"})
    updated_profile = type(profile).from_dict(updated_profile_data)
    assert updated_profile is not None

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            calls["updates"] = updates
            calls["source"] = source
            return BrainBehaviorControlUpdateResult(
                accepted=True,
                applied=True,
                profile=updated_profile,
                rejected_fields=(),
                reason_codes=(
                    "behavior_controls_update_accepted",
                    "behavior_controls_persisted",
                ),
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/runtime/behavior-controls",
        json={"response_depth": "deep", "directness": "rigorous"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls["updates"] == {"response_depth": "deep", "directness": "rigorous"}
    assert calls["source"] == "browser_behavior_controls_endpoint"
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["profile"]["response_depth"] == "deep"
    assert "depth=deep" in payload["compiled_effect_summary"]
    assert "event_id" not in str(payload)
    assert "source_event_id" not in str(payload)


def _performance_preference_api_payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "profile": "browser-en-kokoro",
        "language": "en",
        "tts_runtime_label": "kokoro/English",
        "candidate_a": {
            "candidate_id": "baseline",
            "candidate_kind": "baseline_trace",
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_runtime_label": "kokoro/English",
            "candidate_label": "Candidate A",
            "episode_ids": ["episode-baseline"],
            "plan_ids": ["plan-baseline"],
            "control_frame_ids": ["control-baseline"],
            "public_summary": "Baseline public-safe evidence.",
            "segment_counts": {"speak_segment": 1},
            "metric_counts": {"latency_count": 1},
            "policy_labels": ["baseline"],
            "camera_honesty_states": ["available_not_used"],
            "reason_codes": ["candidate:baseline"],
        },
        "candidate_b": {
            "candidate_id": "candidate",
            "candidate_kind": "candidate_trace",
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_runtime_label": "kokoro/English",
            "candidate_label": "Candidate B",
            "episode_ids": ["episode-candidate"],
            "plan_ids": ["plan-candidate"],
            "control_frame_ids": ["control-candidate"],
            "public_summary": "Candidate public-safe evidence.",
            "segment_counts": {"speak_segment": 1},
            "metric_counts": {"latency_count": 1},
            "policy_labels": ["candidate"],
            "camera_honesty_states": ["can_see_now"],
            "reason_codes": ["candidate:current"],
        },
        "winner": "b",
        "ratings": {
            "felt_heard": 4,
            "state_clarity": 4,
            "interruption_naturalness": 4,
            "voice_pacing": 2,
            "camera_honesty": 4,
            "memory_usefulness": 4,
            "persona_consistency": 4,
            "enjoyment": 4,
            "not_fake_human": 5,
        },
        "failure_labels": ["voice_pacing_too_long"],
        "improvement_labels": ["voice_pacing_better"],
        "evidence_refs": [
            {
                "evidence_kind": "episode",
                "evidence_id": "episode-candidate",
                "summary": "Public-safe replay evidence.",
                "reason_codes": ["evidence:episode"],
            }
        ],
        "reason_codes": ["test:performance_preference"],
    }


def test_local_browser_performance_preferences_endpoint_writes_jsonl(tmp_path):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        performance_preferences_v3_dir=tmp_path,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/api/runtime/performance-preferences",
        json=_performance_preference_api_payload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["pair"]["profile"] == "browser-en-kokoro"
    assert payload["policy_proposals"][0]["target"] == "speech_chunking_bias"
    assert (tmp_path / "preferences.jsonl").exists()
    assert (tmp_path / "policy_proposals.jsonl").exists()
    encoded = json.dumps(payload, sort_keys=True)
    assert "raw_audio" not in encoded
    assert "hidden_prompt" not in encoded


def test_local_browser_performance_preferences_endpoint_fails_closed(tmp_path):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        performance_preferences_v3_dir=tmp_path,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    unsafe = {
        **_performance_preference_api_payload(),
        "raw_audio": "data:audio/wav;base64,AAAA",
        "hidden_prompt": "system prompt",
        "notes": "secret token sk-test",
    }

    response = client.post("/api/runtime/performance-preferences", json=unsafe)

    assert response.status_code == 200
    payload = response.json()
    encoded = json.dumps(payload, sort_keys=True)
    assert payload["accepted"] is False
    assert "performance_preference_record:unsafe_payload" in payload["reason_codes"]
    assert "sk-test" not in encoded
    assert "data:audio" not in encoded
    assert not (tmp_path / "preferences.jsonl").exists()


def test_local_browser_performance_policy_apply_forwards_behavior_controls(tmp_path):
    from fastapi.testclient import TestClient

    calls = {}
    profile = default_behavior_control_profile(
        user_id="user-browser-test",
        agent_id="blink-test",
    )
    updated_profile_data = profile.as_dict()
    updated_profile_data.update({"voice_mode": "concise", "response_depth": "concise"})
    updated_profile = type(profile).from_dict(updated_profile_data)
    assert updated_profile is not None

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            calls["updates"] = updates
            calls["source"] = source
            return BrainBehaviorControlUpdateResult(
                accepted=True,
                applied=True,
                profile=updated_profile,
                rejected_fields=(),
                reason_codes=("behavior_controls_update_accepted",),
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        performance_preferences_v3_dir=tmp_path,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    record_response = client.post(
        "/api/runtime/performance-preferences",
        json=_performance_preference_api_payload(),
    )
    proposal_id = record_response.json()["policy_proposals"][0]["proposal_id"]
    app.state.blink_active_expression_runtime = FakeRuntime()

    response = client.post(
        f"/api/runtime/performance-preferences/policy-proposals/{proposal_id}/apply",
        json={"operator_acknowledged": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls["updates"] == {"response_depth": "concise", "voice_mode": "concise"}
    assert calls["source"] == "performance_learning_policy_proposal"
    assert payload["accepted"] is True
    assert payload["applied"] is True
    assert payload["proposal"]["status"] == "applied"


def test_local_browser_behavior_controls_update_sanitizes_malformed_reason_codes():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            assert updates == {"response_depth": "deep"}
            assert source == "browser_behavior_controls_endpoint"
            return {
                "schema_version": 1,
                "accepted": True,
                "applied": True,
                "available": True,
                "profile": {
                    "schema_version": 3,
                    "user_id": "secret /tmp/brain.db",
                    "agent_id": "secret /tmp/brain.db",
                    "response_depth": "deep",
                    "source": "Traceback /tmp/brain.db",
                    "reason_codes": ["profile:ok", "secret /tmp/brain.db"],
                },
                "compiled_effect_summary": "secret /tmp/brain.db",
                "rejected_fields": "secret /tmp/brain.db",
                "reason_codes": "secret /tmp/brain.db",
                "event_id": "evt-secret",
            }

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/runtime/behavior-controls", json={"response_depth": "deep"})

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["accepted"] is True
    assert payload["compiled_effect_summary"] == "redacted"
    assert payload["rejected_fields"] == []
    assert payload["reason_codes"] == []
    assert payload["profile"]["user_id"] == "redacted"
    assert payload["profile"]["agent_id"] == "redacted"
    assert payload["profile"]["source"] == "redacted"
    assert payload["profile"]["reason_codes"] == ["profile:ok", "redacted"]
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "event_id" not in encoded


def test_local_browser_behavior_controls_update_accepts_supported_fields_only():
    from fastapi.testclient import TestClient

    calls = {}
    profile = default_behavior_control_profile(
        user_id="user-browser-test",
        agent_id="blink-test",
    )
    updated_profile_data = profile.as_dict()
    supported_updates = {
        "response_depth": "deep",
        "directness": "rigorous",
        "warmth": "high",
        "teaching_mode": "walkthrough",
        "memory_use": "continuity_rich",
        "initiative_mode": "proactive",
        "evidence_visibility": "rich",
        "correction_mode": "rigorous",
        "explanation_structure": "walkthrough",
        "challenge_style": "direct",
        "voice_mode": "concise",
        "question_budget": "low",
        "humor_mode": "playful",
        "vividness_mode": "vivid",
        "sophistication_mode": "sophisticated",
        "character_presence": "character_rich",
        "story_mode": "recurring_motifs",
    }
    updated_profile_data.update(supported_updates)
    updated_profile = type(profile).from_dict(updated_profile_data)
    assert updated_profile is not None

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            calls["updates"] = updates
            calls["source"] = source
            return BrainBehaviorControlUpdateResult(
                accepted=True,
                applied=True,
                profile=updated_profile,
                rejected_fields=(),
                reason_codes=("behavior_controls_update_accepted",),
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/runtime/behavior-controls", json=supported_updates)

    assert response.status_code == 200
    payload = response.json()
    assert calls["updates"] == supported_updates
    assert calls["source"] == "browser_behavior_controls_endpoint"
    assert payload["accepted"] is True
    assert payload["profile"]["memory_use"] == "continuity_rich"
    assert payload["profile"]["initiative_mode"] == "proactive"
    assert payload["profile"]["evidence_visibility"] == "rich"
    assert payload["profile"]["correction_mode"] == "rigorous"
    assert payload["profile"]["explanation_structure"] == "walkthrough"
    assert payload["profile"]["voice_mode"] == "concise"
    assert payload["profile"]["humor_mode"] == "playful"
    assert payload["profile"]["story_mode"] == "recurring_motifs"


def test_local_browser_behavior_controls_update_rejects_forbidden_or_malformed_fields():
    from fastapi.testclient import TestClient

    calls = {}

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            calls["updates"] = updates
            return BrainBehaviorControlUpdateResult(
                accepted=True,
                applied=True,
                profile=None,
                rejected_fields=(),
                reason_codes=("should_not_reach_runtime",),
            )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post(
        "/api/runtime/behavior-controls",
        json={
            "response_depth": "encyclopedic",
            "romance": "enabled",
            "hardware_control": "servo",
            "system_prompt": "be human",
            "api_key": "secret",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls == {}
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert set(payload["rejected_fields"]) == {
        "redacted",
        "hardware_control",
        "response_depth",
        "romance",
    }
    assert "behavior_controls_fields_invalid" in payload["reason_codes"]
    assert "api_key" not in str(payload)
    assert "system_prompt" not in str(payload)
    assert "servo" not in str(payload)
    assert "secret" not in str(payload)


def test_local_browser_behavior_controls_update_error_is_safe():
    from fastapi.testclient import TestClient

    class FakeRuntime:
        def update_behavior_control_profile(self, updates, *, source="runtime"):
            raise RuntimeError("private path /tmp/blink/brain.db")

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/runtime/behavior-controls", json={"warmth": "high"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert "runtime_behavior_controls_error" in payload["reason_codes"]
    assert "private path" not in str(payload)
    assert "brain.db" not in str(payload)


def test_local_browser_memory_endpoint_reports_unavailable_before_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.get("/api/runtime/memory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["records"] == []
    assert payload["hidden_counts"] == {"suppressed": 0, "historical": 0, "limit": 0}
    assert payload["health_summary"] == "Memory health unavailable."
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_memory_endpoint_reports_fake_runtime_snapshot(monkeypatch):
    from fastapi.testclient import TestClient

    session_ids = SimpleNamespace(user_id="user-browser-test", agent_id="blink-test")
    calls = {}

    class FakeRuntime:
        store = object()

        def session_resolver(self):
            return session_ids

    class FakeSnapshot:
        def as_dict(self):
            return {
                "schema_version": 1,
                "user_id": "secret /tmp/blink/brain.db",
                "agent_id": "secret /tmp/blink/brain.db",
                "generated_at": "Traceback /tmp/blink/brain.db",
                "records": [
                    {
                        "memory_id": f"memory_claim:user:{session_ids.user_id}:claim_abc",
                        "display_kind": "preference",
                        "scope_type": "user",
                        "scope_id": session_ids.user_id,
                        "title": "secret /tmp/blink/brain.db",
                        "summary": "User prefers coffee.",
                        "status": "secret /tmp/blink/brain.db",
                        "currentness_status": "current",
                        "confidence": 0.82,
                        "pinned": True,
                        "last_used_at": "2026-04-23T01:02:03+00:00",
                        "last_used_reason": "selected_for_relevant_continuity",
                        "used_in_current_turn": "yes",
                        "safe_provenance_label": "Remembered from your explicit preference.",
                        "source_refs": ["claim_abc"],
                        "source_event_ids": ["evt-raw"],
                        "private_scratchpad": "do not leak",
                        "raw_json_block": {"db_path": "/tmp/blink/brain.db"},
                        "user_actions": [
                            "review",
                            "correct",
                            "forget",
                            "export",
                            "secret /tmp/brain.db",
                        ],
                        "reason_codes": ["source:claim", "secret /tmp/brain.db"],
                    }
                ],
                "hidden_counts": {
                    "suppressed": 0,
                    "historical": 0,
                    "limit": 0,
                    "secret /tmp/blink/brain.db": 2,
                },
                "health_summary": "secret /tmp/blink/brain.db",
                "reason_codes": "secret /tmp/brain.db",
            }

    def fake_build_memory_palace_snapshot(**kwargs):
        calls.update(kwargs)
        return FakeSnapshot()

    monkeypatch.setattr(
        local_browser,
        "build_memory_palace_snapshot",
        fake_build_memory_palace_snapshot,
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/memory")

    assert response.status_code == 200
    payload = response.json()
    assert calls["store"] is FakeRuntime.store
    assert calls["session_ids"] is session_ids
    assert calls["include_suppressed"] is False
    assert calls["include_historical"] is False
    assert calls["limit"] == 40
    assert calls["claim_scan_limit"] == 160
    assert payload["available"] is True
    assert payload["user_id"] == "redacted"
    assert payload["agent_id"] == "redacted"
    assert payload["generated_at"] == "redacted"
    assert payload["hidden_counts"]["redacted"] == 2
    assert payload["health_summary"] == "redacted"
    assert payload["summary"] == "1 visible memories: 1 preference"
    assert payload["records"][0]["memory_id"].startswith(
        f"memory_claim:user:{session_ids.user_id}:"
    )
    assert payload["records"][0]["display_kind"] == "preference"
    assert payload["records"][0]["title"] == "redacted"
    assert payload["records"][0]["summary"] == "User prefers coffee."
    assert payload["records"][0]["status"] == "redacted"
    assert payload["records"][0]["currentness_status"] == "current"
    assert payload["records"][0]["confidence"] == 0.82
    assert payload["records"][0]["pinned"] is True
    assert payload["records"][0]["last_used_at"] == "2026-04-23T01:02:03+00:00"
    assert payload["records"][0]["last_used_reason"] == "selected_for_relevant_continuity"
    assert payload["records"][0]["used_in_current_turn"] is False
    assert payload["records"][0]["safe_provenance_label"] == (
        "Remembered from your explicit preference."
    )
    assert payload["records"][0]["user_actions"] == ["review", "correct", "forget", "export"]
    assert payload["records"][0]["reason_codes"] == ["source:claim", "redacted"]
    assert "source_refs" not in payload["records"][0]
    assert "source_event_ids" not in payload["records"][0]
    assert "private_scratchpad" not in payload["records"][0]
    assert "raw_json_block" not in payload["records"][0]
    assert "secret" not in str(payload)
    assert "db_path" not in str(payload)
    assert payload["reason_codes"] == ["runtime_memory_state:available"]


def test_local_browser_memory_endpoint_falls_back_on_snapshot_error(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeRuntime:
        store = object()

        def session_resolver(self):
            return SimpleNamespace(user_id="user-browser-test", agent_id="blink-test")

    def fake_build_memory_palace_snapshot(**_kwargs):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        local_browser,
        "build_memory_palace_snapshot",
        fake_build_memory_palace_snapshot,
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.get("/api/runtime/memory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["records"] == []
    assert "runtime_memory_error:RuntimeError" in payload["reason_codes"]


def test_local_browser_memory_action_endpoint_rejects_inactive_runtime():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post(
        "/api/runtime/memory/memory_claim:user:user-browser-test:claim_abc/forget"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert payload["memory_id"] == "memory_claim:user:user-browser-test:claim_abc"
    assert "runtime_not_active" in payload["reason_codes"]


def test_local_browser_memory_action_rejection_sanitizes_memory_id():
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Secret prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)

    response = client.post("/api/runtime/memory/secret-brain.db/forget")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert payload["memory_id"] == "redacted"
    assert payload["action"] == "forget"
    assert "runtime_not_active" in payload["reason_codes"]
    assert "secret-brain.db" not in encoded
    assert "Secret prompt" not in encoded
    assert "secret" not in encoded


@pytest.mark.parametrize(
    (
        "endpoint_action",
        "expected_action",
        "request_json",
        "expected_replacement",
        "reason_code",
    ),
    (
        ("pin", "pin", None, None, "claim_pinned"),
        ("suppress", "suppress", None, None, "claim_suppressed"),
        (
            "correct",
            "correct",
            {"replacement_value": "tea", "notes": "fix old value"},
            "tea",
            "claim_corrected",
        ),
        ("forget", "forget", None, None, "claim_forgotten"),
        ("mark-stale", "mark_stale", None, None, "claim_marked_stale"),
    ),
)
def test_local_browser_memory_action_endpoint_applies_fake_runtime_action(
    monkeypatch,
    endpoint_action,
    expected_action,
    request_json,
    expected_replacement,
    reason_code,
):
    from fastapi.testclient import TestClient

    session_ids = SimpleNamespace(user_id="user-browser-test", agent_id="blink-test")
    calls = []

    class FakeRuntime:
        store = object()

        def session_resolver(self):
            return session_ids

    class FakeResult:
        def __init__(self, call):
            self._call = dict(call)

        def as_dict(self):
            replacement_memory_id = None
            if self._call["action"] == "correct":
                replacement_memory_id = f"memory_claim:user:{session_ids.user_id}:claim_def"
            return {
                "schema_version": 1,
                "accepted": True,
                "applied": True,
                "action": self._call["action"],
                "memory_id": self._call["memory_id"],
                "record_kind": "memory_claim",
                "event_id": "evt-1",
                "replacement_memory_id": replacement_memory_id,
                "source_refs": ["claim_abc"],
                "source_event_ids": ["evt-raw"],
                "reason_codes": ["memory_action_accepted", reason_code, "secret /tmp/brain.db"],
            }

    def fake_apply_memory_governance_action(**kwargs):
        calls.append(kwargs)
        return FakeResult(kwargs)

    monkeypatch.setattr(
        local_browser,
        "apply_memory_governance_action",
        fake_apply_memory_governance_action,
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    request_kwargs = {"json": request_json} if request_json is not None else {}
    memory_id = f"memory_claim:user:{session_ids.user_id}:claim_abc"
    response = client.post(
        f"/api/runtime/memory/{memory_id}/{endpoint_action}",
        **request_kwargs,
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(calls) == 1
    call = calls[0]
    assert call["store"] is FakeRuntime.store
    assert call["session_ids"] is session_ids
    assert call["memory_id"] == memory_id
    assert call["action"] == expected_action
    assert call["replacement_value"] == expected_replacement
    assert call["notes"] == (request_json or {}).get("notes")
    assert call["source"] == "browser_memory_endpoint"
    assert payload["accepted"] is True
    assert payload["action"] == expected_action
    if expected_replacement is None:
        assert "replacement_memory_id" not in payload
    else:
        assert payload["replacement_memory_id"].endswith(":claim_def")
    assert reason_code in payload["reason_codes"]
    assert "redacted" in payload["reason_codes"]
    assert "event_id" not in payload
    assert "source_refs" not in payload
    assert "source_event_ids" not in payload
    assert "secret" not in str(payload)


def test_local_browser_memory_action_endpoint_sanitizes_malformed_result(monkeypatch):
    from fastapi.testclient import TestClient

    session_ids = SimpleNamespace(user_id="user-browser-test", agent_id="blink-test")

    class FakeRuntime:
        store = object()

        def session_resolver(self):
            return session_ids

    def fake_apply_memory_governance_action(**_kwargs):
        return {
            "schema_version": "not-an-int",
            "accepted": "true",
            "applied": "true",
            "action": "secret /tmp/brain.db",
            "memory_id": "secret /tmp/brain.db",
            "record_kind": "secret /tmp/brain.db",
            "replacement_memory_id": "secret /tmp/brain.db",
            "reason_codes": "secret /tmp/brain.db",
            "event_id": "evt-secret",
        }

    monkeypatch.setattr(
        local_browser,
        "apply_memory_governance_action",
        fake_apply_memory_governance_action,
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    memory_id = f"memory_claim:user:{session_ids.user_id}:claim_abc"
    response = client.post(f"/api/runtime/memory/{memory_id}/pin")

    assert response.status_code == 200
    payload = response.json()
    encoded = str(payload)
    assert payload["schema_version"] == 1
    assert payload["accepted"] is False
    assert payload["applied"] is False
    assert payload["action"] == "redacted"
    assert payload["memory_id"] == "redacted"
    assert payload["record_kind"] == "redacted"
    assert payload["replacement_memory_id"] == "redacted"
    assert payload["reason_codes"] == []
    assert "secret" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "event_id" not in encoded


def test_local_browser_memory_action_endpoint_falls_back_on_action_error(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeRuntime:
        store = object()

        def session_resolver(self):
            return SimpleNamespace(user_id="user-browser-test", agent_id="blink-test")

    def fake_apply_memory_governance_action(**_kwargs):
        raise RuntimeError("action failed")

    monkeypatch.setattr(
        local_browser,
        "apply_memory_governance_action",
        fake_apply_memory_governance_action,
    )
    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    app.state.blink_active_expression_runtime = FakeRuntime()
    client = TestClient(app)

    response = client.post("/api/runtime/memory/memory_claim:user:user-browser-test:claim_abc/pin")

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert "records" not in payload
    assert "runtime_memory_action_error" in payload["reason_codes"]
    assert "action failed" not in str(payload)
    assert "event_id" not in payload


def test_local_browser_state_panel_assets_include_memory_surface():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/blink-expression-panel.js",
    ]

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert "/api/runtime/memory" in text
        assert 'endpoint: "pin"' in text
        assert 'endpoint: "suppress"' in text
        assert 'endpoint: "correct"' in text
        assert 'endpoint: "forget"' in text
        assert 'endpoint: "mark-stale"' in text
        assert "postMemoryAction" in text
        assert "requestMemoryAction" in text
        assert "latestActionResult" in text
        assert "memoryUseText" in text
        assert "/api/runtime/performance-state" in text
        assert "/api/runtime/performance-events" in text
        assert "/api/runtime/actor-state" in text
        assert "/api/runtime/actor-events" in text
        assert "/api/runtime/client-media" in text
        assert "actor_surface_v2_enabled" in text
        assert "renderActorSurface" in text
        assert "profileBadge" in text
        assert "Debug timeline" in text
        assert "Heard" in text
        assert "Blink is saying" in text
        assert "Looking" in text
        assert "Used memory/persona" in text
        assert "Interruption" in text
        assert "听到" in text
        assert "Blink 正在说" in text
        assert "正在看" in text
        assert "使用的记忆/风格" in text
        assert "打断" in text
        assert "window.BlinkLiveStatus" in text
        assert "clientCallbacks" in text
        assert "placePanel(panel)" in text
        assert 'position: "relative"' in text
        assert 'width: "calc(100% - 32px)"' in text
        assert 'maxHeight: "min(440px, 55vh)"' in text
        assert "lastFinalTranscript" in text
        assert "lastPartialTranscript" in text
        assert "activeListeningStatusText" in text
        assert "partialTranscriptText" in text
        assert "no partials from STT" in text
        assert "active_listening" in text
        assert "constraints" in text
        assert "assistantSubtitle" in text
        assert "degradedMessage" in text
        assert "MeloTTS unavailable" in text
        assert "Kokoro unavailable" in text
        assert "latestPerformanceEventId" in text
        assert "refreshPerformance" in text
        assert "listening" in text
        assert "heard" in text
        assert "thinking" in text
        assert "speaking" in text
        assert "looking" in text
        assert "interrupted" in text
        assert "echo_cancellation" in text
        assert "noise_suppression" in text
        assert "auto_gain_control" in text
        assert "interruptionStatusText" in text
        assert "headphones recommended" in text
        assert "speechStatusText" in text
        assert "director_mode" in text
        assert "stale_chunk_drop_count" in text
        assert "used_in_current_turn" in text
        assert "record.pinned" in text
        assert "last_used_reason" in text
        assert "safe_provenance_label" in text
        assert "data-memory-correction-form" in text
        assert "correctionDrafts" in text
        assert "window.confirm" in text
        assert "summarizeReasonCodes" in text
        assert "voice policy" in text
        assert "voice policy no-ops" in text
        assert "Advanced" in text
        assert "data-memory-body" in text
        assert 'method: "POST"' in text
        assert "mark_done" not in text
        assert "mark-done" not in text
        assert "/cancel" not in text
        assert "source_refs" not in text
        assert "source_event_ids" not in text
        assert "db_path" not in text
        assert "exception" not in text
        assert "getUserMedia" not in text
        assert "applyConstraints" not in text
        assert ".stop()" not in text
        assert "renegotiate" not in text


def test_local_browser_media_startup_diagnostic_asset_is_copied():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/blink-media-autoplay.js",
    ]
    html_paths = [
        root / "web/client_src/src/index.html",
    ]

    for html_path in html_paths:
        html = html_path.read_text(encoding="utf-8")
        assert "/api/runtime/client-config.js?v=20260426a" in html
        assert "blink-media-autoplay.js?v=20260425a" in html
        assert "blink-expression-panel.js" in html
        assert 'src="./assets/index-N85auAmE.js?v=20260425b"' in html
        assert html.index("client-config.js") < html.index("index-N85auAmE.js")
        assert html.index("blink-expression-panel.js") < html.index("index-N85auAmE.js")

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert 'MEDIA_SELECTOR = "audio[autoplay], video[autoplay]"' in text
        assert "attemptPlayback" in text
        assert "MutationObserver" in text
        assert "visibilitychange" in text
        assert ".play()" in text
        assert "/api/runtime/client-media" not in text
        assert "getUserMedia" not in text
        assert "local_media_permission_or_device_failed" not in text
        assert "local_camera_unavailable_audio_only" not in text
        assert "browser_media:audio_only" not in text
        assert "video: false" not in text
        assert "makeAudioOnlyConstraints" not in text
        assert "data-blink-media-diagnostic" not in text
        assert "system_prompt" not in text
        assert "Authorization" not in text
        assert "OPENAI_API_KEY" not in text


def test_local_browser_default_ui_uses_runtime_camera_config():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/index-N85auAmE.js",
    ]

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert "globalThis.BlinkRuntimeConfig?.enableCam??!0" in text
        assert "globalThis.BlinkRuntimeConfig?.enableMic??!0" in text
        assert "clientOptions:{enableCam:globalThis.BlinkRuntimeConfig?.enableCam??!0" in text
        assert "BlinkLiveStatus?.clientCallbacks?.()" in text
        assert "clientOptions:{enableCam:!0,enableMic:!0" not in text
        assert "NY={enableCam:!0,enableMic:!0}" not in text
        assert "NY={enableCam:!1,enableMic:!0}" not in text


def test_local_browser_smallwebrtc_asset_uses_stability_first_media_client():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/index.module-6x6hkzdm.js",
    ]

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert "Ignoring server renegotiate request in local browser stability mode." in text
        assert "Reload or reconnect the page if media stalls." in text
        assert "Reload or reconnect the page to recover." in text
        assert "Reload or reconnect the page if media does not recover." in text
        assert "case mo:this.attemptReconnection(!1);break" not in text
        assert "setTimeout(()=>this.attemptReconnection(!0),2e3)" not in text
        assert "Trying to reconnect" not in text
        assert "attempting restart" not in text
        assert "Still disconnected, attempting reconnection" not in text
        assert "system_prompt" not in text
        assert "OPENAI_API_KEY" not in text
        assert "Authorization" not in text

    for app_asset_path in [
        root / "web/client_src/src/assets/index-N85auAmE.js",
    ]:
        text = app_asset_path.read_text(encoding="utf-8")
        assert 'import("./index.module-6x6hkzdm.js?v=20260425b")' in text


def test_local_browser_suppresses_only_aioice_retry_invalid_state_race():
    def retry_callback():
        return None

    retry_callback.__qualname__ = "Transaction.__retry"
    retry_callback.__module__ = "aioice.stun"

    class FakeHandle:
        _callback = retry_callback

    assert local_browser._is_benign_aioice_transaction_retry_race(
        {
            "exception": asyncio.InvalidStateError(),
            "message": "Exception in callback Transaction.__retry()",
            "handle": FakeHandle(),
        }
    )
    assert not local_browser._is_benign_aioice_transaction_retry_race(
        {
            "exception": asyncio.InvalidStateError(),
            "message": "Exception in callback unrelated()",
            "handle": object(),
        }
    )
    assert not local_browser._is_benign_aioice_transaction_retry_race(
        {
            "exception": RuntimeError("boom"),
            "message": "Exception in callback Transaction.__retry()",
            "handle": FakeHandle(),
        }
    )


def test_local_browser_operator_workbench_asset_is_default_and_operational():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/blink-operator-workbench.js",
    ]
    html_paths = [
        root / "web/client_src/src/index.html",
    ]

    for html_path in html_paths:
        html = html_path.read_text(encoding="utf-8")
        assert "blink-operator-workbench.js" in html
        assert "blink-expression-panel.js" in html
        assert html.index("blink-operator-workbench.js") < html.index("blink-expression-panel.js")
        assert html.index("blink-expression-panel.js") < html.index("index-N85auAmE.js")

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert "/api/runtime/operator" in text
        assert "Blink operator" in text
        for label in (
            "Overview",
            "Memory",
            "Controls",
            "Teaching",
            "Voice",
            "Practice",
            "Adapters",
            "Sim-to-real",
            "Performance learning",
            "Rollouts",
        ):
            assert label in text
        assert "sectionCollapsed" in text
        assert "summarizeReasonCodes" in text
        assert "categories=" in text
        assert "fallbackSnapshot" in text
        assert "operator_fetch_failed" in text
        assert "__blinkOperatorWorkbenchInitialized" in text
        assert 'position: "relative"' in text
        assert "collapsed: true" in text
        assert "activeRefreshMs = 8000" in text
        assert "hiddenRefreshMs = 30000" in text
        assert "evidenceRefreshMs = 30000" in text
        assert "staticRefreshMs = 300000" in text
        assert "refreshState.inFlight" in text
        assert "if (state.collapsed && options.force !== true)" in text
        assert "refresh({ force: true, includeEvidence: true, includeStatic: true });" in text
        assert "window.setTimeout" in text
        assert "visibilitychange" in text
        assert "window.setInterval(refresh, 2500)" not in text
        assert "used_in_current_turn" in text
        assert "used_in_current_reply" in text
        assert "memory_persona_performance" in text
        assert "Used in this reply" in text
        assert "Behavior effect" in text
        assert "Memory effects" in text
        assert "Discourse episodes" in text
        assert "renderDiscourseEpisodeRef" in text
        assert "renderUsedMemoryRef" in text
        assert "renderPersonaReference" in text
        assert "renderPersonaAnchor" in text
        assert "persona_references" in text
        assert "persona_anchor_bank_v3" in text
        assert "persona_anchor_refs_v3" in text
        assert "Style anchors" in text
        assert "safe_provenance_label" in text
        assert "compiled_effect_summary" in text
        assert "teaching_knowledge" in text
        assert "renderTeachingKnowledge" in text
        assert "current_decision" in text
        assert "selected_items" in text
        assert "selection_kind" in text
        assert "source_label" in text
        assert "voice_policy" in text
        assert "voice_metrics" in text
        assert "input_health" in text
        assert "microphone_state" in text
        assert "stt_state" in text
        assert "audio_frame_count" in text
        assert "transcription_count" in text
        assert "Input / STT" in text
        assert "Voice metrics unavailable." in text
        assert "TTS metrics" in text
        assert "renderMetricTiles" in text
        assert "/api/runtime/behavior-controls" in text
        assert "/api/runtime/style-presets" in text
        assert "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/preview" in text
        assert "/api/runtime/memory-persona-ingestion/presets/witty-sophisticated/apply" in text
        assert 'method: "POST"' in text
        assert "data-behavior-controls-form" in text
        assert "data-style-preset-control" in text
        assert "runBehaviorControlsUpdate" in text
        assert "applyStylePresetDraft" in text
        assert "stylePresets" in text
        assert "memoryPersonaPresetPreview" in text
        assert "requestMemoryPersonaPresetApply" in text
        assert "Character/story seed" in text
        assert "Rejected seed entries" in text
        assert "Applied seed entries" in text
        assert "applied_entries" in text
        assert "rejected_entries" in text
        assert "safeBehaviorControlResultText" in text
        assert "/api/runtime/memory" in text
        assert 'endpoint: "pin"' in text
        assert 'endpoint: "suppress"' in text
        assert 'endpoint: "correct"' in text
        assert 'endpoint: "forget"' in text
        assert 'endpoint: "mark-stale"' in text
        assert "postMemoryAction" in text
        assert "requestMemoryAction" in text
        assert "memoryActionResults" in text
        assert "latestMemoryActionResult" in text
        assert "memoryPendingActions" in text
        assert "correctionMemoryId" in text
        assert "correctionDrafts" in text
        assert "data-memory-correction-form" in text
        assert "safeMemoryActionResultText" in text
        assert "formatReasonCodes" in text
        assert "window.confirm" in text
        assert "replacement_value" in text
        assert "/api/runtime/rollout" in text
        assert "/api/runtime/rollout/evidence" in text
        assert "/api/runtime/evidence" in text
        assert "/api/runtime/performance-preferences" in text
        assert "data-performance-learning-form" in text
        assert "runPerformancePreferenceSubmit" in text
        assert "applyPerformancePolicyProposal" in text
        assert "felt_heard" in text
        assert "not_fake_human" in text
        assert "policy_proposals" in text
        assert "Performance learning" in text
        assert 'endpoint: "approve"' in text
        assert 'endpoint: "activate"' in text
        assert 'endpoint: "pause"' in text
        assert 'endpoint: "resume"' in text
        assert 'endpoint: "rollback"' in text
        assert "postRolloutAction" in text
        assert "requestRolloutAction" in text
        assert "rolloutEvidence" in text
        assert "episodeEvidence" in text
        assert "fallbackEpisodeEvidence" in text
        assert "renderEpisodeEvidence" in text
        assert "latestRolloutActionResult" in text
        assert "rolloutPendingActions" in text
        assert "safeRolloutActionResultText" in text
        assert "renderRolloutPlan" in text
        assert "voice_actuation_plan" in text
        assert "requested_hints" in text
        assert "applied_hints" in text
        assert "initiative_label" in text
        assert "evidence_visibility_label" in text
        assert "correction_mode_label" in text
        assert "explanation_structure_label" in text
        assert "humor_mode_label" in text
        assert "vividness_mode_label" in text
        assert "character_presence_label" in text
        assert "style_summary" in text
        assert "humor_budget" in text
        assert "metaphor_density" in text
        assert "benchmark_reports" in text
        assert "practice_plans" in text
        assert "live_episodes" in text
        assert "episode_evidence" in text
        assert "artifact_refs" in text
        assert "redacted_uri" in text
        assert "link_kind" in text
        assert "reason_code_categories" in text
        assert "before" in text
        assert "after" in text
        for field in (
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
        ):
            assert field in text
        assert "response_count" in text
        assert "concise_chunking_activation_count" in text
        assert "chunk_count" in text
        assert "max_chunk_chars" in text
        assert "average_chunk_chars" in text
        assert "interruption_frame_count" in text
        assert "buffer_flush_count" in text
        assert "buffer_discard_count" in text
        assert "last_chunking_mode" in text
        assert "stt_wait_age_ms" in text
        assert "stt_waiting_too_long" in text
        assert "STT wait" in text
        assert "unsupported_hints" in text
        assert "noop_reason_codes" in text
        assert "readiness_counts" in text
        assert "governance only" in text
        assert "/api/runtime/hardware" not in text
        assert "/api/runtime/robot" not in text
        assert "romance" not in text
        assert "sexuality" not in text
        assert "exclusive" not in text
        assert "human_identity" not in text
        assert "event_id" not in text
        assert "source_refs" not in text
        assert "source_event_ids" not in text
        assert "db_path" not in text
        assert "prompt_text" not in text
        assert "private_working_memory" not in text


def test_local_browser_model_selector_asset_is_default_and_safe():
    root = Path(__file__).resolve().parents[1]
    asset_paths = [
        root / "web/client_src/src/assets/blink-model-selector.js",
    ]
    html_paths = [
        root / "web/client_src/src/index.html",
    ]

    for html_path in html_paths:
        html = html_path.read_text(encoding="utf-8")
        assert "blink-model-selector.js" in html
        assert html.index("blink-model-selector.js") < html.index("index-N85auAmE.js")

    for asset_path in asset_paths:
        text = asset_path.read_text(encoding="utf-8")
        assert "/api/runtime/models" in text
        assert "/start" in text
        assert "model_profile_id" in text
        assert "modelSelection" in text
        assert "localStorage" in text
        assert "Accepted" in text
        assert "Rejected" in text
        assert "reason_codes" in text
        assert "window.fetch" in text
        assert "blinkModelSelectorFetch" in text
        assert "expanded: false" in text
        assert 'state.expanded ? "min(342px, calc(100vw - 28px))"' in text
        assert 'min(220px, calc(100vw - 112px))' in text
        assert "Blink model selector show" in text
        for forbidden in (
            "system_prompt",
            "base_url",
            "OPENAI_API_KEY",
            "Authorization",
            "traceback",
            "exception",
            "persona",
            "hidden prompt",
            "developer message",
        ):
            assert forbidden not in text


def test_openai_demo_wrapper_scripts_are_safe_and_provider_scoped():
    root = Path(__file__).resolve().parents[1]
    script_expectations = {
        "scripts/run-blink-chat-openai.sh": "scripts/run-blink-chat.sh",
        "scripts/run-blink-voice-openai.sh": "scripts/run-local-voice-en.sh",
        "scripts/run-blink-browser-openai.sh": "scripts/run-blink-browser.sh",
    }

    for script_name, delegated_launcher in script_expectations.items():
        text = (root / script_name).read_text(encoding="utf-8")
        assert "local_launcher_helpers.sh" in text
        assert 'load_env_defaults "$ROOT_DIR/.env"' in text
        assert "OPENAI_API_KEY" in text
        assert "is required" in text
        assert "BLINK_LOCAL_LLM_PROVIDER=openai-responses" in text
        assert (
            'BLINK_LOCAL_OPENAI_RESPONSES_MODEL="${BLINK_LOCAL_OPENAI_RESPONSES_MODEL:-gpt-5.4-mini}"'
            in text
        )
        assert 'BLINK_LOCAL_DEMO_MODE="${BLINK_LOCAL_DEMO_MODE:-1}"' in text
        assert delegated_launcher in text
        assert "BLINK_LOCAL_STT_BACKEND" not in text
        assert "BLINK_LOCAL_TTS_BACKEND=remote" not in text
        assert "OPENAI_API_KEY=" not in text


def test_native_voice_english_wrapper_is_local_camera_free_and_browser_free():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run-local-voice-en.sh").read_text(encoding="utf-8")
    camera_text = (root / "scripts/run-local-voice-macos-camera-en.sh").read_text(encoding="utf-8")
    isolation_doc = (root / "docs/debugging/native_voice_isolation.md").read_text(
        encoding="utf-8"
    )
    debugging_text = (root / "docs/debugging.md").read_text(encoding="utf-8")
    build_helper_text = (root / "scripts/build-macos-camera-helper.sh").read_text(encoding="utf-8")
    base_voice_text = (root / "scripts/run-local-voice.sh").read_text(encoding="utf-8")
    helper_plist = (root / "native/macos/BlinkCameraHelper/Info.plist").read_text(encoding="utf-8")
    helper_swift = (
        root / "native/macos/BlinkCameraHelper/Sources/BlinkCameraHelper/main.swift"
    ).read_text(encoding="utf-8")
    compatibility_text = (root / "scripts/run-local-voice-camera-en.sh").read_text(encoding="utf-8")
    melo_voice_text = (root / "scripts/run-local-voice-melo.sh").read_text(encoding="utf-8")
    bootstrap_text = (root / "scripts/bootstrap-local-mac.sh").read_text(encoding="utf-8")

    assert "BLINK_LOCAL_LANGUAGE" in text
    assert "en" in text
    assert "BLINK_LOCAL_CONFIG_PROFILE=native-en-kokoro" in text
    assert "BLINK_LOCAL_TTS_BACKEND" in text
    assert "kokoro" in text
    assert "BLINK_LOCAL_VOICE_VISION" in text
    assert "BLINK_LOCAL_VOICE_VISION=0" in text
    assert "run-local-voice.sh" in text
    assert "--language en" in text
    assert "--tts-backend kokoro" in text
    assert "BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1" in text
    assert 'BLINK_LOCAL_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"' in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "--allow-barge-in" in text
    assert "--protected-playback" in text
    assert "runtime=native transport=PyAudio profile=native-en-kokoro" in text
    assert "isolation=backend-only" in text
    assert "tts=kokoro" in text
    assert "protected_playback=${PROTECTED_PLAYBACK_STATE}" in text
    assert "barge_in=${BARGE_IN_STATE}" in text
    assert "primary_browser_paths=browser-zh-melo,browser-en-kokoro" in text
    assert "mic -> STT -> LLM -> Kokoro" in text
    assert "--system-prompt" not in text
    assert "--vision" not in text
    assert "artifacts/runtime_logs" in text
    assert "latest-native-voice-en.log" in text
    assert "blink-native-voice-en-${LOG_STAMP}.log" in text
    assert "run-blink-browser" not in text
    assert "run-local-browser" not in text
    assert "run-melotts" not in text.lower()
    assert "local-http-wav" not in text
    assert "OPENAI_API_KEY" not in text
    assert "openai-responses" not in text
    assert "run-local-voice-en.sh" in compatibility_text
    assert "Native voice camera is disabled" in compatibility_text
    assert "BLINK_ALLOW_NATIVE_VOICE_MELO" in melo_voice_text
    assert "Native voice MeloTTS is disabled" in melo_voice_text
    assert "BLINK_LOCAL_CAMERA_SOURCE=macos-helper" in camera_text
    assert "BLINK_LOCAL_CONFIG_PROFILE=native-en-kokoro-macos-camera" in camera_text
    assert "--camera-source macos-helper" in camera_text
    assert "--language en" in camera_text
    assert "--tts-backend kokoro" in camera_text
    assert "BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1" in camera_text
    assert 'BLINK_LOCAL_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"' in camera_text
    assert "PYTHONUNBUFFERED=1" in camera_text
    assert "--allow-barge-in" in camera_text
    assert "--protected-playback" in camera_text
    assert "--system-prompt" not in camera_text
    assert "latest-native-voice-macos-camera-en.log" in camera_text
    assert "Blink Camera Helper, not Terminal" in camera_text
    assert "runtime=native transport=PyAudio profile=native-en-kokoro-macos-camera" in camera_text
    assert "isolation=backend-plus-helper-camera" in camera_text
    assert "on-demand single-frame isolation only" in camera_text
    assert "not continuous video or the browser camera UX" in camera_text
    assert "primary_browser_paths=browser-zh-melo,browser-en-kokoro" in camera_text
    assert "run-local-browser" not in camera_text
    assert "run-melotts" not in camera_text.lower()
    assert "local-http-wav" not in camera_text
    assert "opencv" not in camera_text.lower()
    assert "codesign --force --sign -" in build_helper_text
    assert "NSCameraUsageDescription" in helper_plist
    assert "ai.blink.CameraHelper" in helper_plist
    assert "lastFrameTime" in helper_swift
    assert "activeVideoMinFrameDuration" not in helper_swift
    assert "activeVideoMaxFrameDuration" not in helper_swift
    assert "BLINK_LOCAL_CAMERA_SOURCE" in base_voice_text
    assert "--camera-source=macos-helper" in base_voice_text
    assert "BOOTSTRAP_ARGS+=(--with-vision)" in base_voice_text
    assert "--extra native-camera" not in bootstrap_text
    assert "English voice + native camera" not in bootstrap_text
    assert "Start English native voice: ./scripts/run-local-voice-en.sh" in bootstrap_text
    assert "runtime=native transport=PyAudio profile=native-en-kokoro" in isolation_doc
    assert "runtime=native transport=PyAudio profile=native-en-kokoro-macos-camera" in (
        isolation_doc
    )
    assert "barge_in=off" in isolation_doc
    assert "browser-zh-melo" in isolation_doc
    assert "browser-en-kokoro" in isolation_doc
    assert "native_voice_isolation.md" in debugging_text


def test_melo_browser_wrapper_records_durable_runtime_log():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run-local-browser-melo.sh").read_text(encoding="utf-8")
    ignore_text = (root / ".gitignore").read_text(encoding="utf-8")

    assert "artifacts/runtime_logs" in text
    assert "BLINK_LOCAL_CONFIG_PROFILE=browser-zh-melo" in text
    assert "blink-browser-melo-${LOG_STAMP}.log" in text
    assert "latest-browser-melo.log" in text
    assert "tee -a" in text
    assert "terminate_pid_tree" in text
    assert "monitor_melo_sidecar" in text
    assert "MeloTTS HTTP-WAV sidecar lost health" in text
    assert "starting an owned replacement" in text
    assert 'DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"' in text
    assert 'BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION"' in text
    assert 'DEFAULT_CONTINUOUS_PERCEPTION="${BLINK_LOCAL_CONTINUOUS_PERCEPTION:-0}"' in text
    assert 'BLINK_LOCAL_CONTINUOUS_PERCEPTION="$DEFAULT_CONTINUOUS_PERCEPTION"' in text
    assert 'DEFAULT_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"' in text
    assert 'BLINK_LOCAL_ALLOW_BARGE_IN="$DEFAULT_ALLOW_BARGE_IN"' in text
    assert 'TTS_RUNTIME_LABEL="local-http-wav/MeloTTS"' in text
    assert 'BLINK_LOCAL_TTS_RUNTIME_LABEL="$TTS_RUNTIME_LABEL"' in text
    assert "profile=browser-zh-melo" in text
    assert "language=zh" in text
    assert "webrtc=on" in text
    assert "camera_vision=$(launcher_state_label \"$DEFAULT_BROWSER_VISION\")" in text
    assert "protected_playback=$(launcher_protected_playback_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
    assert "barge_in_policy=$(launcher_barge_in_policy_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
    assert "--no-vision)" in text
    assert "DEFAULT_BROWSER_VISION=0" in text
    assert "--allow-barge-in)" in text
    assert "DEFAULT_ALLOW_BARGE_IN=1" in text
    assert "client=${CLIENT_URL}" in text
    assert "BLINK_LOCAL_ALLOW_BARGE_IN=1" not in text
    assert "OPENAI_API_KEY" not in text
    assert "Authorization" not in text
    assert "/artifacts/runtime_logs/" in ignore_text


def test_english_browser_kokoro_wrapper_keeps_vision_parity_and_is_sidecar_free():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run-local-browser-kokoro-en.sh").read_text(encoding="utf-8")
    ignore_text = (root / ".gitignore").read_text(encoding="utf-8")

    assert "BLINK_LOCAL_LANGUAGE=en" in text
    assert "BLINK_LOCAL_CONFIG_PROFILE=browser-en-kokoro" in text
    assert "BLINK_LOCAL_TTS_BACKEND=kokoro" in text
    assert 'TTS_RUNTIME_LABEL="kokoro/English"' in text
    assert 'BLINK_LOCAL_TTS_RUNTIME_LABEL="$TTS_RUNTIME_LABEL"' in text
    assert 'DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"' in text
    assert 'BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION"' in text
    assert 'DEFAULT_CONTINUOUS_PERCEPTION="${BLINK_LOCAL_CONTINUOUS_PERCEPTION:-0}"' in text
    assert 'BLINK_LOCAL_CONTINUOUS_PERCEPTION="$DEFAULT_CONTINUOUS_PERCEPTION"' in text
    assert 'DEFAULT_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"' in text
    assert 'BLINK_LOCAL_ALLOW_BARGE_IN="$DEFAULT_ALLOW_BARGE_IN"' in text
    assert "BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "run-local-browser.sh" in text
    assert "--language en" in text
    assert "--tts-backend kokoro" in text
    assert "profile=browser-en-kokoro" in text
    assert "language=en" in text
    assert "webrtc=on" in text
    assert "camera_vision=$(launcher_state_label \"$DEFAULT_BROWSER_VISION\")" in text
    assert "protected_playback=$(launcher_protected_playback_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
    assert "barge_in_policy=$(launcher_barge_in_policy_label \"$DEFAULT_ALLOW_BARGE_IN\")" in text
    assert "--no-vision)" in text
    assert "DEFAULT_BROWSER_VISION=0" in text
    assert "--allow-barge-in)" in text
    assert "DEFAULT_ALLOW_BARGE_IN=1" in text
    assert "client=${CLIENT_URL}" in text
    assert "blink-browser-kokoro-en-${LOG_STAMP}.log" in text
    assert "latest-browser-kokoro-en.log" in text
    assert "tee -a" in text
    assert "primary browser-en-kokoro WebRTC actor path" in text
    assert "browser vision available by default" in text
    assert "run-melotts" not in text.lower()
    assert "local-http-wav" not in text
    assert "OPENAI_API_KEY" not in text
    assert "Authorization" not in text
    assert "/artifacts/runtime_logs/" in ignore_text


def test_hybrid_openai_smoke_script_is_clearly_llm_only():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/smoke-hybrid-openai-demo.sh").read_text(encoding="utf-8")

    assert "local_launcher_helpers.sh" in text
    assert 'load_env_defaults "$ROOT_DIR/.env"' in text
    assert "OPENAI_API_KEY" in text
    assert "hybrid OpenAI LLM-only smoke test" in text
    assert "run-blink-chat-openai.sh" in text
    assert "--once" in text
    assert "does not prove STT, TTS, WebRTC, MeloTTS latency, or camera behavior" in text
    assert "run-blink-voice-openai.sh" not in text
    assert "run-blink-browser-openai.sh" not in text
    assert "OPENAI_API_KEY=" not in text


def test_hybrid_openai_manual_qa_docs_cover_required_surfaces():
    root = Path(__file__).resolve().parents[1]
    runbook = (root / "docs/HYBRID_OPENAI_DEMO_RUNBOOK.md").read_text(encoding="utf-8")
    scorecard = (root / "docs/MANUAL_QA_SCORECARD.md").read_text(encoding="utf-8")
    combined = f"{runbook}\n{scorecard}"

    for required_text in (
        "text smoke",
        "Native voice smoke",
        "Browser/WebRTC smoke",
        "Melo sidecar health",
        "Provider stack",
        "Camera",
        "Missing key failure",
        "Connectivity failure",
        "LLM-only",
        "not STT/TTS/WebRTC proof",
        "/api/runtime/stack",
        "OPENAI_API_KEY",
    ):
        assert required_text in combined

    for forbidden_text in ("sk-", "Authorization:", "Bearer "):
        assert forbidden_text not in runbook
        assert forbidden_text not in scorecard


def test_local_browser_patch_endpoint_accepts_json(monkeypatch):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    captured = {}

    async def fake_handle_patch_request(self, request: SmallWebRTCPatchRequest):
        captured["pc_id"] = request.pc_id
        captured["candidates"] = request.candidates

    monkeypatch.setattr(
        SmallWebRTCRequestHandler, "handle_patch_request", fake_handle_patch_request
    )

    response = client.patch(
        "/api/offer",
        json={
            "pc_id": "pc-123",
            "candidates": [
                {
                    "candidate": "candidate:1 1 udp 2122260223 192.168.0.1 12345 typ host",
                    "sdp_mid": "0",
                    "sdp_mline_index": 0,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["pc_id"] == "pc-123"
    assert captured["candidates"][0].sdp_mid == "0"


def test_local_browser_patch_endpoint_ignores_malformed_ice_candidate(monkeypatch):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    called = False

    async def fake_handle_patch_request(self, request: SmallWebRTCPatchRequest):
        nonlocal called
        called = True

    monkeypatch.setattr(
        SmallWebRTCRequestHandler, "handle_patch_request", fake_handle_patch_request
    )

    response = client.patch(
        "/api/offer",
        json={
            "pc_id": "pc-123",
            "candidates": [
                {
                    "candidate": "candidate:bad",
                    "sdp_mid": "0",
                    "sdp_mline_index": 0,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert called is False


def test_local_browser_offer_endpoint_accepts_json(monkeypatch):
    from fastapi.testclient import TestClient

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config)
    client = TestClient(app)
    captured = {}

    async def fake_handle_web_request(
        self, request: SmallWebRTCRequest, webrtc_connection_callback
    ):
        captured["request"] = request
        return {"pc_id": "pc-123", "sdp": "answer", "type": "answer"}

    monkeypatch.setattr(SmallWebRTCRequestHandler, "handle_web_request", fake_handle_web_request)

    response = client.post(
        "/api/offer",
        json={
            "sdp": "offer",
            "type": "offer",
            "pc_id": "pc-123",
            "restart_pc": False,
        },
    )

    assert response.status_code == 200
    assert captured["request"].pc_id == "pc-123"


def test_local_browser_selected_model_reaches_session_runtime_and_stack(monkeypatch):
    from fastapi.testclient import TestClient

    release_runner = {"done": False}
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(_llm_config):
        return None

    class FakeConnection:
        pc_id = "pc-selected"

        def event_handler(self, _name):
            def decorator(handler):
                return handler

            return decorator

        async def disconnect(self):
            release_runner["done"] = True

    class FakeBrainRuntime:
        def close(self):
            captured["closed"] = True

    class FakeRunner:
        def __init__(self, handle_sigint):
            captured["handle_sigint"] = handle_sigint

        async def run(self, _task):
            captured["runner_started"] = True
            while not release_runner["done"]:
                await asyncio.sleep(0.01)

    async def fake_handle_web_request(
        self, request: SmallWebRTCRequest, webrtc_connection_callback
    ):
        captured["request"] = request
        await webrtc_connection_callback(FakeConnection())
        await asyncio.sleep(0)
        return {"pc_id": "pc-selected", "sdp": "answer", "type": "answer"}

    def fake_runtime_builder(runtime_config, **kwargs):
        captured["runtime_config"] = runtime_config
        captured["runtime_builder_kwargs"] = kwargs
        context = SimpleNamespace(
            blink_brain_runtime=FakeBrainRuntime(),
            blink_camera_health_manager=None,
            blink_perception_broker=None,
        )
        return object(), context

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setenv("BLINK_LOCAL_ENABLE_REMOTE_MODEL_SELECTION", "1")
    monkeypatch.setattr(SmallWebRTCRequestHandler, "handle_web_request", fake_handle_web_request)
    monkeypatch.setattr(local_browser, "PipelineRunner", FakeRunner)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="af_heart",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    app, _ = create_app(config, runtime_builder=fake_runtime_builder)

    try:
        with TestClient(app) as client:
            start = client.post(
                "/start",
                json={"body": {"model_profile_id": "openai-gpt-5_4-mini"}},
            )
            session_id = start.json()["sessionId"]

            offer = client.post(
                f"/sessions/{session_id}/api/offer",
                json={
                    "sdp": "offer",
                    "type": "offer",
                    "pc_id": "pc-selected",
                    "restart_pc": False,
                },
            )
            time.sleep(0.05)
            stack = client.get("/api/runtime/stack").json()

            assert offer.status_code == 200
            assert captured["runtime_config"].llm_provider == "openai-responses"
            assert captured["runtime_config"].model == "gpt-5.4-mini"
            assert stack["runtime_active"] is True
            assert stack["llm_provider"] == "openai-responses"
            assert stack["model"] == "gpt-5.4-mini"
            assert stack["configured_llm_provider"] == "ollama"
            assert stack["configured_model"] == "qwen3.5:4b"
            assert stack["model_profile_id"] == "openai-gpt-5_4-mini"
            assert "model_profile:openai-gpt-5_4-mini" in stack["reason_codes"]
            assert captured["runtime_builder_kwargs"]["actor_control_scheduler"] is (
                app.state.blink_actor_control_scheduler
            )
            assert client.get("/api/runtime/actor-control-frames").status_code == 404
    finally:
        release_runner["done"] = True


@pytest.mark.asyncio
async def test_run_local_voice_does_not_auto_switch_to_local_http_wav(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(llm_config):
        captured["verified_llm"] = llm_config
        return None

    async def fake_resolve_runtime_selection(**kwargs):
        captured["selection_kwargs"] = kwargs
        return LocalRuntimeTTSSelection(
            backend=kwargs["requested_backend"],
            voice=kwargs["requested_voice"],
            base_url=kwargs["requested_base_url"],
            auto_switched=False,
        )

    def fake_build_runtime(config, *, tts_session=None, **_kwargs):
        captured["backend"] = config.tts_backend
        captured["voice"] = config.tts_voice
        captured["base_url"] = config.tts_base_url
        captured["tts_session"] = tts_session
        return object(), object()

    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeRunner:
        def __init__(self, handle_sigint):
            captured["handle_sigint"] = handle_sigint

        async def run(self, task):
            captured["task"] = task

    monkeypatch.setattr(local_voice, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setattr(
        local_voice,
        "resolve_local_runtime_tts_selection",
        fake_resolve_runtime_selection,
    )
    monkeypatch.setattr(local_voice, "build_local_voice_runtime", fake_build_runtime)
    monkeypatch.setattr(local_voice, "get_audio_device_by_index", lambda _index: None)
    monkeypatch.setattr(local_voice, "PipelineRunner", FakeRunner)
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        types.SimpleNamespace(ClientSession=FakeClientSession),
    )

    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="zf_xiaobei",
        allow_barge_in=False,
    )

    status = await local_voice.run_local_voice(config)

    assert status == 0
    assert captured["verified_llm"] == config.llm
    assert captured["selection_kwargs"]["backend_locked"] is True
    assert captured["backend"] == "kokoro"
    assert captured["voice"] == "zf_xiaobei"
    assert captured["base_url"] is None


@pytest.mark.asyncio
async def test_run_local_voice_rejects_native_melo_without_legacy_override(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(llm_config):
        captured["verified_llm"] = llm_config
        return None

    monkeypatch.delenv("BLINK_ALLOW_NATIVE_VOICE_MELO", raising=False)
    monkeypatch.setattr(local_voice, "verify_local_llm_config", fake_verify_local_llm_config)

    config = LocalVoiceConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        allow_barge_in=False,
    )

    with pytest.raises(local_common.LocalDependencyError) as excinfo:
        await local_voice.run_local_voice(config)

    assert captured["verified_llm"] == config.llm
    assert "Native voice does not use MeloTTS/local-http-wav" in str(excinfo.value)
    assert "run-local-browser-melo.sh" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_local_browser_reports_port_conflict_before_startup(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(_llm_config):
        captured["verified_llm"] = True

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setattr(local_browser, "_browser_port_is_occupied", lambda _host, _port: True)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.EN,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="bf_emma",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        config_profile="browser-en-kokoro",
    )

    with pytest.raises(RuntimeError) as excinfo:
        await local_browser.run_local_browser(config)

    message = str(excinfo.value)
    assert "127.0.0.1:7860" in message
    assert "one browser/WebRTC path" in message
    assert "tmux kill-session -t blink-browser-melo" in message
    assert "verified_llm" not in captured


@pytest.mark.asyncio
async def test_run_local_browser_prefers_local_http_wav_when_available(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(llm_config):
        captured["verified_llm"] = llm_config
        return None

    async def fake_resolve_runtime_selection(**_kwargs):
        return LocalRuntimeTTSSelection(
            backend="local-http-wav",
            voice=None,
            base_url="http://127.0.0.1:8001",
            auto_switched=True,
        )

    class FakeServer:
        def __init__(self, config):
            captured["server_config"] = config
            self.started = True
            self.should_exit = False

        async def serve(self):
            captured["served"] = True

    class FakeUvicorn:
        @staticmethod
        def Config(app, host, port):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port
            return {"app": app, "host": host, "port": port}

        Server = FakeServer

    def fake_create_app(config, *, shared_vision=None):
        captured["backend"] = config.tts_backend
        captured["voice"] = config.tts_voice
        captured["base_url"] = config.tts_base_url
        captured["shared_vision"] = shared_vision
        return object(), FakeUvicorn

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setattr(
        local_browser,
        "resolve_local_runtime_tts_selection",
        fake_resolve_runtime_selection,
    )
    monkeypatch.setattr(local_browser, "create_app", fake_create_app)
    monkeypatch.setattr(local_browser, "_browser_port_is_occupied", lambda _host, _port: False)

    async def fake_start_uvicorn_server(server, *, host, port, ready_path, timeout_secs):
        captured["ready_host"] = host
        captured["ready_port"] = port
        captured["ready_path"] = ready_path
        captured["ready_timeout"] = timeout_secs
        return asyncio.create_task(server.serve())

    monkeypatch.setattr(local_browser, "start_uvicorn_server", fake_start_uvicorn_server)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="kokoro",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice="zf_xiaobei",
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
    )

    status = await local_browser.run_local_browser(config)

    assert status == 0
    assert captured["verified_llm"] == config.llm
    assert captured["backend"] == "local-http-wav"
    assert captured["voice"] is None
    assert captured["base_url"] == "http://127.0.0.1:8001"
    assert captured["served"] is True
    assert captured["ready_path"] == "/client/"


@pytest.mark.asyncio
async def test_run_local_browser_prints_ready_only_after_readiness(monkeypatch, capsys):
    events: list[str] = []

    async def fake_verify_local_llm_config(_llm_config):
        events.append("verified-llm")

    async def fake_resolve_runtime_selection(**_kwargs):
        events.append("resolved-tts")
        return LocalRuntimeTTSSelection(
            backend="local-http-wav",
            voice=None,
            base_url="http://127.0.0.1:8001",
            auto_switched=False,
        )

    class FakeServer:
        def __init__(self, config):
            self.started = True
            self.should_exit = False

        async def serve(self):
            events.append("served")

    class FakeUvicorn:
        @staticmethod
        def Config(app, host, port):
            return {"app": app, "host": host, "port": port}

        Server = FakeServer

    def fake_create_app(_config, *, shared_vision=None):
        events.append("created-app")
        return object(), FakeUvicorn

    async def fake_start_uvicorn_server(server, *, host, port, ready_path, timeout_secs):
        events.append(f"waiting:{host}:{port}{ready_path}")
        return asyncio.create_task(server.serve())

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setattr(
        local_browser,
        "resolve_local_runtime_tts_selection",
        fake_resolve_runtime_selection,
    )
    monkeypatch.setattr(local_browser, "create_app", fake_create_app)
    monkeypatch.setattr(local_browser, "start_uvicorn_server", fake_start_uvicorn_server)
    monkeypatch.setattr(local_browser, "_browser_port_is_occupied", lambda _host, _port: False)

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
    )

    status = await local_browser.run_local_browser(config)

    output = capsys.readouterr().out
    assert status == 0
    assert events == [
        "verified-llm",
        "resolved-tts",
        "created-app",
        "waiting:127.0.0.1:7860/client/",
        "served",
    ]
    assert output.index(
        "Waiting for http://127.0.0.1:7860/client/ to become ready..."
    ) < output.index("Blink local browser voice is available at http://127.0.0.1:7860/client/")
    assert output.index(
        "Blink local browser voice is available at http://127.0.0.1:7860/client/"
    ) < output.index("Blink browser ready: runtime=browser transport=WebRTC")
    assert "tts=local-http-wav" in output
    assert "profile=manual" in output
    assert "language=zh" in output
    assert "vision=off" in output
    assert "continuous_perception=off" in output
    assert "protected_playback=on" in output
    assert "barge_in=off" in output
    assert "barge_in_policy=protected" in output
    assert "client=http://127.0.0.1:7860/client/" in output


@pytest.mark.asyncio
async def test_run_local_browser_defers_shared_vision_until_first_camera_request(
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}

    async def fake_verify_local_llm_config(llm_config):
        captured["verified_llm"] = llm_config
        return None

    async def fake_resolve_runtime_selection(**_kwargs):
        return LocalRuntimeTTSSelection(
            backend="local-http-wav",
            voice=None,
            base_url="http://127.0.0.1:8001",
            auto_switched=False,
        )

    class FakeServer:
        def __init__(self, config):
            self.started = True
            self.should_exit = False

        async def serve(self):
            captured["served"] = True

    class FakeUvicorn:
        @staticmethod
        def Config(app, host, port):
            return {"app": app, "host": host, "port": port}

        Server = FakeServer

    def fake_create_app(_config, *, shared_vision=None):
        captured["shared_vision"] = shared_vision
        return object(), FakeUvicorn

    async def fake_start_uvicorn_server(server, *, host, port, ready_path, timeout_secs):
        return asyncio.create_task(server.serve())

    monkeypatch.setattr(local_browser, "verify_local_llm_config", fake_verify_local_llm_config)
    monkeypatch.setattr(
        local_browser,
        "resolve_local_runtime_tts_selection",
        fake_resolve_runtime_selection,
    )
    monkeypatch.setattr(local_browser, "create_app", fake_create_app)
    monkeypatch.setattr(local_browser, "start_uvicorn_server", fake_start_uvicorn_server)
    monkeypatch.setattr(local_browser, "_browser_port_is_occupied", lambda _host, _port: False)

    def fail_if_vision_warms_before_startup(*, model):
        raise AssertionError(f"vision warmed before startup: {model}")

    monkeypatch.setattr(
        local_browser,
        "create_local_vision_service",
        fail_if_vision_warms_before_startup,
    )

    config = LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Test prompt",
        language=local_browser.Language.ZH,
        stt_backend="mlx-whisper",
        tts_backend="local-http-wav",
        stt_model="mlx-community/whisper-medium-mlx",
        tts_voice=None,
        tts_base_url="http://127.0.0.1:8001",
        host="127.0.0.1",
        port=7860,
        vision_enabled=True,
        vision_model="demo/moondream",
    )

    status = await local_browser.run_local_browser(config)

    assert status == 0
    assert captured["verified_llm"] == config.llm
    assert captured["shared_vision"] is None
    assert captured["served"] is True
    output = capsys.readouterr().out
    assert "will load on first camera inspection" in output
    assert "Warming local vision model" not in output


def test_smallwebrtc_client_converts_video_frames_with_pyav_only():
    rgb_frame = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    native_frame = VideoFrame.from_ndarray(rgb_frame, format="rgb24").reformat(format="yuv420p")
    client = object.__new__(SmallWebRTCClient)
    expected_rgb = native_frame.to_ndarray(format="rgb24")

    converted = client._convert_frame(native_frame)

    assert native_frame.format.name == "yuv420p"
    assert converted.shape == rgb_frame.shape
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted, expected_rgb)


def test_smallwebrtc_client_rate_limits_input_video_conversion(monkeypatch):
    client = object.__new__(SmallWebRTCClient)
    client._last_video_emit_monotonic = {}
    monotonic_values = iter([10.0, 10.2, 11.1, 11.2])

    monkeypatch.setattr(smallwebrtc_transport.time, "monotonic", lambda: next(monotonic_values))

    assert client._video_frame_due(smallwebrtc_transport.CAM_VIDEO_SOURCE, 1) is True
    assert client._video_frame_due(smallwebrtc_transport.CAM_VIDEO_SOURCE, 1) is False
    assert client._video_frame_due(smallwebrtc_transport.CAM_VIDEO_SOURCE, 1) is True
    assert client._video_frame_due(smallwebrtc_transport.CAM_VIDEO_SOURCE, 0) is True


def test_smallwebrtc_client_rate_limits_audio_timeout_warnings(monkeypatch):
    client = object.__new__(SmallWebRTCClient)
    client._track_stall_states = client._create_track_stall_states()
    track = FakeTrackState(enabled=True)
    warnings: list[str] = []
    monotonic_values = iter([1.0, 2.0, 35.5, 40.0, 41.0])

    monkeypatch.setattr(smallwebrtc_transport.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        smallwebrtc_transport,
        "logger",
        SimpleNamespace(warning=warnings.append),
    )

    client._sync_track_stall_state(smallwebrtc_transport.MIC_AUDIO_SOURCE, track)
    client._warn_track_stall(
        smallwebrtc_transport.MIC_AUDIO_SOURCE,
        media_kind="audio",
        reason="timeout",
    )
    client._warn_track_stall(
        smallwebrtc_transport.MIC_AUDIO_SOURCE,
        media_kind="audio",
        reason="timeout",
    )
    client._warn_track_stall(
        smallwebrtc_transport.MIC_AUDIO_SOURCE,
        media_kind="audio",
        reason="timeout",
    )
    client._clear_track_stall(smallwebrtc_transport.MIC_AUDIO_SOURCE, track)
    client._warn_track_stall(
        smallwebrtc_transport.MIC_AUDIO_SOURCE,
        media_kind="audio",
        reason="timeout",
    )

    assert len(warnings) == 3
    assert warnings[0].startswith("No audio frame received from microphone within 2.0s")
    assert warnings[1].startswith("Still not receiving audio frames from microphone")
    assert warnings[2].startswith("No audio frame received from microphone within 2.0s")


@pytest.mark.asyncio
async def test_smallwebrtc_client_connect_skips_connection_when_already_connected():
    connection = FakeSmallWebRTCEventConnection(connected=True)
    client = object.__new__(SmallWebRTCClient)
    client._webrtc_connection = connection

    await client.connect()

    assert connection.connect_count == 0


@pytest.mark.asyncio
async def test_smallwebrtc_connection_connect_does_not_renegotiate_existing_video_track():
    connection = object.__new__(SmallWebRTCConnection)
    connection._connect_invoked = False
    connection._pending_app_messages = []
    events: list[str] = []
    renegotiation_requests = 0
    discarded_frames = 0

    async def fake_call_event_handler(name, *_args):
        events.append(name)

    def fake_ask_to_renegotiate():
        nonlocal renegotiation_requests
        renegotiation_requests += 1

    class FakeInputTrack:
        async def discard_old_frames(self):
            nonlocal discarded_frames
            discarded_frames += 1

    connection.is_connected = lambda: True
    connection._call_event_handler = fake_call_event_handler
    connection.ask_to_renegotiate = fake_ask_to_renegotiate
    connection.video_input_track = lambda: FakeInputTrack()
    connection.screen_video_input_track = lambda: None

    await SmallWebRTCConnection.connect(connection)

    assert events == ["connected"]
    assert discarded_frames == 1
    assert renegotiation_requests == 0


def test_smallwebrtc_connection_drops_datachannel_messages_after_peer_disconnect():
    sent_messages: list[str] = []
    connection = object.__new__(SmallWebRTCConnection)
    connection._data_channel = SimpleNamespace(
        readyState="open",
        send=lambda message: sent_messages.append(message),
    )
    connection._pc = SimpleNamespace(connectionState="disconnected")
    connection._data_channel_enabled = True
    connection._outgoing_messages_queue = ["queued"]

    SmallWebRTCConnection.send_app_message(connection, {"type": "rtvi"})

    assert sent_messages == []
    assert connection._data_channel_enabled is False
    assert connection._outgoing_messages_queue == []


def test_browser_disconnect_ignores_stale_connection_active_state():
    app_state = SimpleNamespace(
        blink_browser_active_session_id="live-session",
        blink_browser_active_client_id="live-client",
    )

    assert not local_browser._browser_connection_owns_active_state(
        app_state,
        session_id="old-session",
        client_id="old-client",
    )
    assert local_browser._browser_connection_owns_active_state(
        app_state,
        session_id="live-session",
        client_id="old-client",
    )
    assert local_browser._browser_connection_owns_active_state(
        app_state,
        session_id="old-session",
        client_id="live-client",
    )


@pytest.mark.asyncio
async def test_smallwebrtc_client_loads_tracks_on_client_connected_without_renegotiation_handler():
    connection = FakeSmallWebRTCEventConnection(connected=True)

    async def noop(*_args, **_kwargs):
        return None

    client = SmallWebRTCClient(
        connection,
        callbacks=SimpleNamespace(
            on_app_message=noop,
            on_client_connected=noop,
            on_client_disconnected=noop,
            on_video_track_stalled=noop,
            on_video_track_resumed=noop,
            on_audio_track_stalled=noop,
            on_audio_track_resumed=noop,
        ),
    )
    client._params = SimpleNamespace(
        audio_in_enabled=True,
        video_in_enabled=True,
        audio_out_enabled=False,
        video_out_enabled=False,
    )

    await client._handle_client_connected()

    assert "renegotiated" not in connection.handlers
    assert connection.audio_track_requests == 1
    assert connection.video_track_requests == 1
    assert connection.screen_track_requests == 1
    assert connection.replaced_audio_tracks == []
    assert connection.replaced_video_tracks == []


@pytest.mark.asyncio
async def test_smallwebrtc_client_attaches_audio_output_track_for_browser_audio():
    connection = FakeSmallWebRTCEventConnection(connected=True)

    async def noop(*_args, **_kwargs):
        return None

    client = SmallWebRTCClient(
        connection,
        callbacks=SimpleNamespace(
            on_app_message=noop,
            on_client_connected=noop,
            on_client_disconnected=noop,
            on_video_track_stalled=noop,
            on_video_track_resumed=noop,
            on_audio_track_stalled=noop,
            on_audio_track_resumed=noop,
        ),
    )
    client._out_sample_rate = 16000
    client._params = SimpleNamespace(
        audio_in_enabled=True,
        video_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        audio_out_auto_silence=False,
    )

    await client._handle_client_connected()
    write_task = asyncio.create_task(
        client.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        )
    )
    await asyncio.sleep(0)
    await client._audio_output_track.recv()

    assert connection.replaced_audio_tracks == [client._audio_output_track]
    wrote = await write_task
    assert wrote is True


@pytest.mark.asyncio
async def test_smallwebrtc_client_writes_browser_audio_when_datachannel_ping_is_stale():
    connection = FakeSmallWebRTCEventConnection(connected=False, media_connected=True)

    async def noop(*_args, **_kwargs):
        return None

    client = SmallWebRTCClient(
        connection,
        callbacks=SimpleNamespace(
            on_app_message=noop,
            on_client_connected=noop,
            on_client_disconnected=noop,
            on_video_track_stalled=noop,
            on_video_track_resumed=noop,
            on_audio_track_stalled=noop,
            on_audio_track_resumed=noop,
        ),
    )
    client._out_sample_rate = 16000
    client._params = SimpleNamespace(
        audio_in_enabled=True,
        video_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        audio_out_auto_silence=False,
    )

    await client._handle_client_connected()
    write_task = asyncio.create_task(
        client.write_audio_frame(
            OutputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        )
    )
    await asyncio.sleep(0)
    await client._audio_output_track.recv()

    wrote = await write_task
    assert wrote is True
    assert client._audio_output_write_failure_count == 0


@pytest.mark.asyncio
async def test_smallwebrtc_client_reports_missing_browser_audio_output_track():
    class RejectingAudioConnection(FakeSmallWebRTCEventConnection):
        def replace_audio_track(self, track):
            self.replaced_audio_tracks.append(track)
            return False

    connection = RejectingAudioConnection(connected=True)

    async def noop(*_args, **_kwargs):
        return None

    client = SmallWebRTCClient(
        connection,
        callbacks=SimpleNamespace(
            on_app_message=noop,
            on_client_connected=noop,
            on_client_disconnected=noop,
            on_video_track_stalled=noop,
            on_video_track_resumed=noop,
            on_audio_track_stalled=noop,
            on_audio_track_resumed=noop,
        ),
    )
    client._out_sample_rate = 16000
    client._params = SimpleNamespace(
        audio_in_enabled=True,
        video_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        audio_out_auto_silence=False,
    )

    await client._handle_client_connected()
    wrote = await client.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
    )

    assert connection.replaced_audio_tracks
    assert client._audio_output_track is None
    assert wrote is False
    assert client._audio_output_write_failure_count == 1
    assert client._last_audio_output_write_failure_reason == "missing_audio_output_track"


@pytest.mark.asyncio
async def test_smallwebrtc_client_times_out_unconsumed_browser_audio():
    connection = FakeSmallWebRTCEventConnection(connected=True)

    async def noop(*_args, **_kwargs):
        return None

    client = SmallWebRTCClient(
        connection,
        callbacks=SimpleNamespace(
            on_app_message=noop,
            on_client_connected=noop,
            on_client_disconnected=noop,
            on_video_track_stalled=noop,
            on_video_track_resumed=noop,
            on_audio_track_stalled=noop,
            on_audio_track_resumed=noop,
        ),
    )
    client._out_sample_rate = 16000
    client._params = SimpleNamespace(
        audio_in_enabled=True,
        video_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        audio_out_auto_silence=False,
    )
    client._audio_output_write_timeout_secs = lambda _frame: 0.001

    await client._handle_client_connected()
    wrote = await client.write_audio_frame(
        OutputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
    )

    assert wrote is False
    assert client._audio_output_write_failure_count == 1
    assert client._last_audio_output_write_failure_reason == "audio_output_not_consumed"
    assert len(client._audio_output_track._chunk_queue) == 0


def test_smallwebrtc_client_rate_limits_video_media_errors(monkeypatch):
    client = object.__new__(SmallWebRTCClient)
    client._track_stall_states = client._create_track_stall_states()
    track = FakeTrackState(enabled=True)
    warnings: list[str] = []
    monotonic_values = iter([5.0, 10.0, 36.0])

    monkeypatch.setattr(smallwebrtc_transport.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        smallwebrtc_transport,
        "logger",
        SimpleNamespace(warning=warnings.append),
    )

    client._sync_track_stall_state(smallwebrtc_transport.CAM_VIDEO_SOURCE, track)
    client._warn_track_stall(
        smallwebrtc_transport.CAM_VIDEO_SOURCE,
        media_kind="video",
        reason="media-error",
    )
    client._warn_track_stall(
        smallwebrtc_transport.CAM_VIDEO_SOURCE,
        media_kind="video",
        reason="media-error",
    )
    client._warn_track_stall(
        smallwebrtc_transport.CAM_VIDEO_SOURCE,
        media_kind="video",
        reason="media-error",
    )

    assert len(warnings) == 2
    assert warnings[0].startswith("Media stream error while reading video from camera")
    assert warnings[1].startswith(
        "Still seeing media stream errors while reading video from camera"
    )


@pytest.mark.asyncio
async def test_smallwebrtc_client_reports_track_stall_without_auto_renegotiation(monkeypatch):
    client = object.__new__(SmallWebRTCClient)
    connection = FakeSmallWebRTCConnection(connected=True)
    client._webrtc_connection = connection
    client._track_stall_states = client._create_track_stall_states()
    track = FakeTrackState(enabled=True)
    captured: list[smallwebrtc_transport.TrackHealthEvent] = []

    async def on_video_track_stalled(event):
        captured.append(event)

    async def noop(_event):
        return None

    client._callbacks = SimpleNamespace(
        on_video_track_stalled=on_video_track_stalled,
        on_video_track_resumed=noop,
        on_audio_track_stalled=noop,
        on_audio_track_resumed=noop,
    )

    monkeypatch.setattr(smallwebrtc_transport.time, "monotonic", lambda: 100.0)

    client._sync_track_stall_state(smallwebrtc_transport.CAM_VIDEO_SOURCE, track)
    client._mark_track_failure(smallwebrtc_transport.CAM_VIDEO_SOURCE, reason="timeout")
    await client._notify_track_stalled(smallwebrtc_transport.CAM_VIDEO_SOURCE, reason="timeout")

    assert connection.renegotiation_requests == 0
    assert len(captured) == 1
    assert captured[0].source == smallwebrtc_transport.CAM_VIDEO_SOURCE
    assert captured[0].reason == "timeout"


@pytest.mark.asyncio
async def test_smallwebrtc_client_notifies_track_stall_once_without_renegotiation(monkeypatch):
    client = object.__new__(SmallWebRTCClient)
    connection = FakeSmallWebRTCConnection(connected=True)
    client._webrtc_connection = connection
    client._track_stall_states = client._create_track_stall_states()
    track = FakeTrackState(enabled=True)
    captured: list[smallwebrtc_transport.TrackHealthEvent] = []

    async def on_audio_track_stalled(event):
        captured.append(event)

    async def noop(_event):
        return None

    client._callbacks = SimpleNamespace(
        on_video_track_stalled=noop,
        on_video_track_resumed=noop,
        on_audio_track_stalled=on_audio_track_stalled,
        on_audio_track_resumed=noop,
    )
    monkeypatch.setattr(smallwebrtc_transport.time, "monotonic", lambda: 25.0)
    monkeypatch.setattr(
        smallwebrtc_transport,
        "logger",
        SimpleNamespace(warning=lambda _message: None),
    )

    client._sync_track_stall_state(smallwebrtc_transport.MIC_AUDIO_SOURCE, track)
    client._mark_track_failure(smallwebrtc_transport.MIC_AUDIO_SOURCE, reason="timeout")
    client._mark_track_failure(smallwebrtc_transport.MIC_AUDIO_SOURCE, reason="timeout")
    await client._notify_track_stalled(smallwebrtc_transport.MIC_AUDIO_SOURCE, reason="timeout")
    await client._notify_track_stalled(smallwebrtc_transport.MIC_AUDIO_SOURCE, reason="timeout")

    assert connection.renegotiation_requests == 0
    assert len(captured) == 1
    assert captured[0].source == smallwebrtc_transport.MIC_AUDIO_SOURCE
    assert captured[0].reason == "timeout"
    assert captured[0].consecutive_failures == 2

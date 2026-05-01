import argparse

import pytest

from blink.brain.events import BrainEventType
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.store import BrainStore
from blink.cli import local_common, local_doctor
from blink.cli.local_common import (
    DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_RESPONSES_MODEL,
    LocalLLMConfig,
    LocalRuntimeTTSSelection,
)


def test_local_doctor_resolve_config_switches_backend_defaults(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "")
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language="en",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend="whisper",
            stt_model=None,
            tts_backend="piper",
            tts_voice=None,
        )
    )

    assert config["stt_model"] == "Systran/faster-distil-whisper-medium.en"
    assert config["tts_voice"] == "en_US-ryan-high"
    assert config["language"] == local_doctor.Language.EN
    assert config["tts_backend_locked"] is True


def test_local_doctor_resolve_config_prefers_language_specific_voice(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "af_heart")

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language="en",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend="kokoro",
            tts_voice=None,
        )
    )

    assert config["tts_voice"] == "af_heart"


def test_local_doctor_resolve_config_reads_browser_vision_from_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "1")
    monkeypatch.delenv("BLINK_LOCAL_VOICE_VISION", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "")
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="browser",
            with_vision=False,
            language="zh",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config["with_vision"] is True
    assert config["stt_model"] == "mlx-community/whisper-medium-mlx"
    assert config["tts_backend_locked"] is False


def test_local_doctor_resolve_config_ignores_voice_vision_environment(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_BROWSER_VISION", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_VOICE_VISION", "1")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language="en",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config["with_vision"] is False
    assert config["tts_backend"] == "kokoro"


def test_local_doctor_resolve_config_reads_macos_camera_helper(monkeypatch, tmp_path):
    helper_app = tmp_path / "BlinkCameraHelper.app"
    monkeypatch.delenv("BLINK_LOCAL_BROWSER_VISION", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_CAMERA_SOURCE", "macos-helper")
    monkeypatch.setenv("BLINK_LOCAL_CAMERA_HELPER_APP", str(helper_app))
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language="en",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
            camera_source=None,
            camera_helper_app=None,
        )
    )

    assert config["with_vision"] is True
    assert config["camera_source"] == "macos-helper"
    assert config["camera_helper_app_path"] == helper_app


def test_local_doctor_resolve_config_supports_openai_responses(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_MODEL", "gpt-env")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_BASE_URL", "https://proxy.test/v1")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "flex")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-ollama")

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="text",
            with_vision=False,
            language="en",
            llm_provider="openai-responses",
            model=None,
            base_url=None,
            demo_mode=None,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config["llm_provider"] == "openai-responses"
    assert config["model"] == "gpt-env"
    assert config["base_url"] == "https://proxy.test/v1"
    assert config["llm_config"] == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-env",
        base_url="https://proxy.test/v1",
        system_prompt=local_common.default_local_text_system_prompt(local_doctor.Language.EN),
        service_tier="flex",
    )


def test_local_doctor_resolve_config_supports_openai_demo_mode(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS", "")

    config = local_doctor.resolve_config(
        argparse.Namespace(
            profile="browser",
            with_vision=False,
            language="zh",
            llm_provider=None,
            model=None,
            base_url=None,
            demo_mode=True,
            max_output_tokens=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    llm_config = config["llm_config"]
    assert isinstance(llm_config, LocalLLMConfig)
    assert llm_config.demo_mode is True
    assert llm_config.service_tier == "priority"
    assert llm_config.max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert "演示模式" in llm_config.system_prompt


def test_check_commands_reports_missing_portaudio(monkeypatch):
    command_state = {"uv": True, "ollama": True, "brew": True}

    monkeypatch.setattr(local_doctor, "_command_exists", lambda cmd: command_state.get(cmd, False))
    monkeypatch.setattr(local_doctor, "_brew_prefix", lambda formula: None)

    results = local_doctor._check_commands("voice")

    assert any(result.name == "portaudio" and result.status == "FAIL" for result in results)


def test_check_commands_skips_ollama_for_openai_provider(monkeypatch):
    checked = []

    def fake_command_exists(command):
        checked.append(command)
        return command == "uv"

    monkeypatch.setattr(local_doctor, "_command_exists", fake_command_exists)

    results = local_doctor._check_commands("text", "openai-responses")

    assert [result.name for result in results] == ["uv"]
    assert "ollama" not in checked


def test_check_openai_responses_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    results = local_doctor._check_openai_responses(
        {
            "profile": "text",
            "llm_config": LocalLLMConfig(
                provider="openai-responses",
                model=DEFAULT_OPENAI_RESPONSES_MODEL,
                base_url=None,
                system_prompt="Prompt",
            ),
        }
    )

    assert results == [
        local_doctor.CheckResult(
            "openai-api-key",
            "FAIL",
            "OPENAI_API_KEY is not set.",
            "Set OPENAI_API_KEY in your shell or ignored `.env` file.",
        )
    ]


def test_check_openai_responses_reports_present_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(local_doctor, "verify_openai_responses_config", lambda _config: None)

    results = local_doctor._check_openai_responses(
        {
            "profile": "text",
            "llm_config": LocalLLMConfig(
                provider="openai-responses",
                model="gpt-demo",
                base_url="https://proxy.test/v1",
                system_prompt="Prompt",
                service_tier="flex",
            ),
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["openai-api-key"] == "PASS"
    assert statuses["openai-responses-model"] == "PASS"
    assert statuses["openai-responses-base-url"] == "PASS"
    assert statuses["openai-responses-service-tier"] == "PASS"


def test_check_openai_responses_reports_hybrid_scope_for_voice_browser(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(local_doctor, "verify_openai_responses_config", lambda _config: None)

    results = local_doctor._check_openai_responses(
        {
            "profile": "browser",
            "llm_config": LocalLLMConfig(
                provider="openai-responses",
                model="gpt-demo",
                base_url=None,
                system_prompt="Prompt",
                demo_mode=True,
                max_output_tokens=120,
            ),
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["openai-responses-demo-mode"] == "PASS"
    assert statuses["openai-responses-hybrid-scope"] == "PASS"
    assert any(
        result.name == "openai-responses-demo-mode" and "max_output_tokens=120" in result.detail
        for result in results
    )


def test_check_modules_supports_browser_profile_without_path_nameerror(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client_src = tmp_path / "web" / "client_src" / "src"
    client_src.mkdir(parents=True)
    (client_src / "index.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(local_doctor, "_module_exists", lambda module: True)
    original_files = local_doctor.resources.files
    monkeypatch.setattr(
        local_doctor.resources,
        "files",
        lambda package: tmp_path if package == "blink.web" else original_files(package),
    )

    results = local_doctor._check_modules(
        {
            "profile": "browser",
            "stt_backend": "mlx-whisper",
            "tts_backend": "kokoro",
            "with_vision": False,
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["blink.web.client_source"] == "PASS"
    assert "cv2" not in statuses


def test_check_modules_does_not_require_native_camera_for_voice_profile(monkeypatch):
    present_modules = {
        "openai",
        "pyaudio",
        "mlx_whisper",
        "kokoro_onnx",
        "torch",
        "transformers",
        "pyvips",
    }
    monkeypatch.setattr(local_doctor, "_module_exists", lambda module: module in present_modules)

    results = local_doctor._check_modules(
        {
            "profile": "voice",
            "stt_backend": "mlx-whisper",
            "tts_backend": "kokoro",
            "with_vision": True,
        }
    )

    statuses = {result.name: result.status for result in results}
    assert "cv2" not in statuses
    assert "torch" not in statuses
    assert "transformers" not in statuses
    assert "pyvips" not in statuses


def test_check_modules_requires_moondream_for_macos_helper_without_opencv(monkeypatch):
    present_modules = {
        "openai",
        "pyaudio",
        "mlx_whisper",
        "kokoro_onnx",
        "torch",
        "transformers",
    }
    monkeypatch.setattr(local_doctor, "_module_exists", lambda module: module in present_modules)

    results = local_doctor._check_modules(
        {
            "profile": "voice",
            "stt_backend": "mlx-whisper",
            "tts_backend": "kokoro",
            "with_vision": True,
            "camera_source": "macos-helper",
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["torch"] == "PASS"
    assert statuses["transformers"] == "PASS"
    assert statuses["pyvips"] == "FAIL"
    assert "cv2" not in statuses


def test_check_macos_camera_helper_reports_missing_or_present_app(monkeypatch, tmp_path):
    monkeypatch.setattr(local_doctor.sys, "platform", "darwin")
    missing = local_doctor._check_macos_camera_helper(
        {
            "camera_source": "macos-helper",
            "camera_helper_app_path": tmp_path / "Missing.app",
        }
    )
    assert missing[0].name == "blink-camera-helper"
    assert missing[0].status == "FAIL"
    assert "build-macos-camera-helper.sh" in (missing[0].remedy or "")

    executable = tmp_path / "BlinkCameraHelper.app" / "Contents" / "MacOS" / "BlinkCameraHelper"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    present = local_doctor._check_macos_camera_helper(
        {
            "camera_source": "macos-helper",
            "camera_helper_app_path": tmp_path / "BlinkCameraHelper.app",
        }
    )
    assert present[0].status == "PASS"


def test_check_model_caches_warns_for_missing_downloads(monkeypatch):
    monkeypatch.setattr(local_doctor, "model_cache_exists", lambda model: False)
    monkeypatch.setattr(local_doctor, "kokoro_assets_present", lambda: False)

    results = local_doctor._check_model_caches(
        {
            "profile": "voice",
            "language": local_doctor.Language.ZH,
            "stt_model": "mlx-community/whisper-medium-mlx",
            "tts_backend": "kokoro",
            "tts_voice": "af_heart",
            "with_vision": True,
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["stt-model-cache"] == "WARN"
    assert statuses["tts-assets"] == "WARN"
    assert statuses["mandarin-tts-quality"] == "WARN"
    assert statuses["vision-model-cache"] == "WARN"
    guidance = next(result.remedy for result in results if result.name == "mandarin-tts-quality")
    assert "MeloTTS" in guidance


def test_check_visual_pipeline_reports_detector_and_enrichment_state(monkeypatch):
    from blink.brain.perception import detector as detector_module

    class ReadyDetector:
        def __init__(self):
            self.available = True

    monkeypatch.setattr(detector_module, "OnnxFacePresenceDetector", ReadyDetector)

    results = local_doctor._check_visual_pipeline(
        {
            "profile": "browser",
            "with_vision": False,
        }
    )

    statuses = {result.name: result.status for result in results}
    assert statuses["presence-detector"] == "PASS"
    assert statuses["vision-enrichment"] == "PASS"


def test_check_runtime_visual_state_reads_latest_browser_degraded_reason(monkeypatch, tmp_path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id="blink/main",
        user_id="user:test",
        session_id="session:test",
        thread_id="browser:thread:test",
        source="visual_health",
        payload={
            "scope_key": "browser:presence",
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                vision_enabled=True,
                vision_connected=True,
                camera_track_state="stalled",
                perception_unreliable=True,
                sensor_health_reason="camera_frame_stale",
                last_fresh_frame_at="2026-04-18T12:00:00+08:00",
            ).as_dict(),
        },
    )
    store.close()
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_DB_PATH", str(db_path))

    results = local_doctor._check_runtime_visual_state({"profile": "browser"})

    assert results
    assert results[0].name == "browser-visual-state"
    assert results[0].status == "WARN"
    assert "camera_frame_stale" in results[0].detail


def test_check_runtime_visual_state_treats_clean_disconnect_as_non_failing(monkeypatch, tmp_path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id="blink/main",
        user_id="user:test",
        session_id="session:test",
        thread_id="browser:thread:test",
        source="visual_health",
        payload={
            "scope_key": "browser:presence",
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                vision_enabled=True,
                vision_connected=False,
                camera_track_state="disconnected",
                sensor_health_reason="camera_disconnected",
                last_fresh_frame_at=None,
            ).as_dict(),
        },
    )
    store.close()
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_DB_PATH", str(db_path))

    results = local_doctor._check_runtime_visual_state({"profile": "browser"})

    assert results
    assert results[0].name == "browser-visual-state"
    assert results[0].status == "PASS"
    assert results[0].remedy is None
    assert "vision_connected=False" in results[0].detail
    assert "track=disconnected" in results[0].detail
    assert "snapshot=" in results[0].detail


def test_check_runtime_visual_state_labels_old_stall_as_historical_not_live(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id="blink/main",
        user_id="user:test",
        session_id="session:test",
        thread_id="browser:thread:test",
        source="visual_health",
        payload={
            "scope_key": "browser:presence",
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                vision_enabled=True,
                vision_connected=True,
                camera_track_state="stalled",
                perception_unreliable=True,
                sensor_health_reason="camera_frame_stale",
                last_fresh_frame_at="2026-04-18T12:00:00+08:00",
                updated_at="2026-04-18T12:00:05+08:00",
            ).as_dict(),
        },
    )
    store.close()
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_DB_PATH", str(db_path))

    results = local_doctor._check_runtime_visual_state({"profile": "browser"})

    assert results
    assert results[0].name == "browser-visual-state"
    assert results[0].status == "PASS"
    assert "snapshot=historical_not_live" in results[0].detail
    assert "historical_vision_connected=True" in results[0].detail
    assert "historical_track=stalled" in results[0].detail
    assert "vision_connected=True, track=stalled" not in results[0].detail


@pytest.mark.asyncio
async def test_check_tts_backend_reports_missing_xtts(monkeypatch):
    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(local_doctor.httpx, "AsyncClient", lambda timeout: FailingClient())

    results = await local_doctor._check_tts_backend(
        {
            "tts_backend": "xtts",
            "tts_base_url": "http://127.0.0.1:8000",
            "language": local_doctor.Language.ZH,
        }
    )

    assert results[0].name == "xtts-server"
    assert results[0].status == "FAIL"


@pytest.mark.asyncio
async def test_check_tts_backend_accepts_local_http_wav(monkeypatch):
    class Response:
        status_code = 200
        content = b"RIFFdemo"
        text = ""

        def json(self):
            return {}

    class VoicesResponse:
        status_code = 200
        content = b"RIFFdemo"
        text = ""

        def json(self):
            return {
                "default_speakers": {"zh": "ZH", "en": "EN-US"},
                "speakers": {"zh": ["ZH"], "en": ["EN-US"]},
            }

    class PassingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            return VoicesResponse()

        async def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(local_doctor.httpx, "AsyncClient", lambda timeout: PassingClient())

    results = await local_doctor._check_tts_backend(
        {
            "tts_backend": "local-http-wav",
            "tts_base_url": "http://127.0.0.1:8001",
            "tts_voice": None,
            "language": local_doctor.Language.ZH,
        }
    )

    assert results[0].name == "local-http-wav"
    assert results[0].status == "PASS"


@pytest.mark.asyncio
async def test_check_tts_backend_warns_and_falls_back_for_invalid_local_http_wav_speaker(
    monkeypatch,
):
    captured = {}

    class VoicesResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "default_speakers": {"zh": "ZH", "en": "EN-US"},
                "speakers": {"zh": ["ZH"], "en": ["EN-US"]},
            }

    class TTSResponse:
        status_code = 200
        content = b"RIFFdemo"
        text = ""

    class PassingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            return VoicesResponse()

        async def post(self, *_args, **kwargs):
            captured["speaker"] = kwargs["json"]["speaker"]
            return TTSResponse()

    monkeypatch.setattr(local_doctor.httpx, "AsyncClient", lambda timeout: PassingClient())

    results = await local_doctor._check_tts_backend(
        {
            "tts_backend": "local-http-wav",
            "tts_base_url": "http://127.0.0.1:8001",
            "tts_voice": "zf_xiaobei",
            "language": local_doctor.Language.ZH,
        }
    )

    assert results[0].name == "local-http-wav-speaker"
    assert results[0].status == "WARN"
    assert captured["speaker"] == "ZH"
    assert results[1].name == "local-http-wav"
    assert results[1].status == "PASS"


def test_check_audio_devices_warns_when_display_audio_is_default(monkeypatch):
    devices = [
        local_common.AudioDeviceInfo(0, "LG UltraFine Display Audio", 1, 0, 48000),
        local_common.AudioDeviceInfo(1, "LG UltraFine Display Audio", 0, 2, 48000),
        local_common.AudioDeviceInfo(2, "MacBook Pro Microphone", 1, 0, 48000),
        local_common.AudioDeviceInfo(3, "MacBook Pro Speakers", 0, 2, 48000),
    ]

    monkeypatch.setattr(local_doctor, "_module_exists", lambda module: module == "pyaudio")
    monkeypatch.setattr(local_doctor, "get_audio_devices", lambda: devices)
    monkeypatch.setattr(
        local_doctor,
        "get_default_audio_device",
        lambda kind: devices[0] if kind == "input" else devices[1],
    )
    monkeypatch.setattr(local_doctor, "resolve_preferred_audio_device_indexes", lambda *_: (2, 3))

    results = local_doctor._check_audio_devices("voice")

    assert any(result.name == "audio-routing" and result.status == "WARN" for result in results)


@pytest.mark.asyncio
async def test_run_doctor_returns_error_when_ollama_is_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(local_doctor, "_check_commands", lambda profile, llm_provider="ollama": [])
    monkeypatch.setattr(local_doctor, "_check_modules", lambda config: [])
    monkeypatch.setattr(local_doctor, "_check_audio_devices", lambda profile: [])
    monkeypatch.setattr(local_doctor, "_check_model_caches", lambda config: [])

    async def fake_check_ollama(base_url, model):
        return [
            local_doctor.CheckResult(
                "ollama-server",
                "FAIL",
                "Could not reach Ollama.",
                "Start it with `ollama serve`.",
            )
        ]

    monkeypatch.setattr(local_doctor, "_check_ollama", fake_check_ollama)

    status = await local_doctor.run_doctor(
        {
            "profile": "text",
            "language": local_doctor.Language.ZH,
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3.5:4b",
            "with_vision": False,
            "stt_model": "mlx-community/whisper-medium-mlx",
            "tts_backend": "xtts",
            "tts_voice": None,
            "tts_base_url": "http://127.0.0.1:8000",
        }
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Could not reach Ollama." in captured.out


@pytest.mark.asyncio
async def test_run_doctor_uses_openai_provider_without_ollama(monkeypatch, capsys):
    monkeypatch.setattr(local_doctor, "_check_modules", lambda config: [])
    monkeypatch.setattr(local_doctor, "_check_audio_devices", lambda profile: [])
    monkeypatch.setattr(local_doctor, "_check_model_caches", lambda config: [])

    def fake_check_commands(profile, llm_provider="ollama"):
        assert profile == "text"
        assert llm_provider == "openai-responses"
        return []

    async def fake_check_llm_provider(config):
        assert config["llm_provider"] == "openai-responses"
        return [
            local_doctor.CheckResult(
                "openai-responses-model",
                "PASS",
                "OpenAI Responses model `gpt-demo` is configured.",
            )
        ]

    async def fail_check_ollama(*_args, **_kwargs):
        raise AssertionError("Ollama check should not run for openai-responses")

    monkeypatch.setattr(local_doctor, "_check_commands", fake_check_commands)
    monkeypatch.setattr(local_doctor, "_check_llm_provider", fake_check_llm_provider)
    monkeypatch.setattr(local_doctor, "_check_ollama", fail_check_ollama)

    status = await local_doctor.run_doctor(
        {
            "profile": "text",
            "llm_provider": "openai-responses",
            "llm_config": LocalLLMConfig(
                provider="openai-responses",
                model="gpt-demo",
                base_url=None,
                system_prompt="Prompt",
            ),
            "language": local_doctor.Language.EN,
            "base_url": None,
            "model": "gpt-demo",
            "with_vision": False,
            "stt_model": "mlx-community/whisper-medium-mlx",
            "tts_backend": "xtts",
            "tts_voice": None,
            "tts_base_url": "http://127.0.0.1:8000",
        }
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "PASS openai-responses-model" in captured.out


@pytest.mark.asyncio
async def test_run_doctor_reports_runtime_chinese_tts_profile_switch(monkeypatch, capsys):
    monkeypatch.setattr(local_doctor, "_check_commands", lambda profile, llm_provider="ollama": [])
    monkeypatch.setattr(local_doctor, "_check_modules", lambda config: [])
    monkeypatch.setattr(local_doctor, "_check_audio_devices", lambda profile: [])
    monkeypatch.setattr(local_doctor, "_check_model_caches", lambda config: [])

    async def fake_check_ollama(base_url, model):
        return []

    async def fake_resolve_runtime_tts_selection(**_kwargs):
        return LocalRuntimeTTSSelection(
            backend="local-http-wav",
            voice=None,
            base_url="http://127.0.0.1:8001",
            auto_switched=True,
        )

    async def fake_check_tts_backend(config):
        assert config["tts_backend"] == "local-http-wav"
        assert config["tts_base_url"] == "http://127.0.0.1:8001"
        return []

    monkeypatch.setattr(local_doctor, "_check_ollama", fake_check_ollama)
    monkeypatch.setattr(
        local_doctor,
        "resolve_local_runtime_tts_selection",
        fake_resolve_runtime_tts_selection,
    )
    monkeypatch.setattr(local_doctor, "_check_tts_backend", fake_check_tts_backend)

    status = await local_doctor.run_doctor(
        {
            "profile": "browser",
            "language": local_doctor.Language.ZH,
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3.5:4b",
            "with_vision": False,
            "stt_backend": "mlx-whisper",
            "stt_model": "mlx-community/whisper-medium-mlx",
            "tts_backend": "kokoro",
            "tts_voice": "zf_xiaobei",
            "tts_base_url": None,
            "tts_backend_locked": False,
            "tts_voice_override": None,
        }
    )

    captured = capsys.readouterr()

    assert status == 0
    assert "PASS chinese-runtime-tts-profile" in captured.out

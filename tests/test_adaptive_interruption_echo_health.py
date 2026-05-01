import asyncio
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
)
from blink.interaction import (
    BrowserBargeInTurnStartStrategy,
    BrowserInteractionMode,
    BrowserInterruptionStateTracker,
    BrowserPerformanceEventBus,
    WebRTCAudioHealthController,
)
from blink.utils.asyncio.task_manager import TaskManager, TaskManagerParams

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
PRIMARY_BROWSER_PROFILES = (
    (
        "browser-zh-melo",
        local_browser.Language.ZH,
        "local-http-wav",
        "local-http-wav/MeloTTS",
    ),
    (
        "browser-en-kokoro",
        local_browser.Language.EN,
        "kokoro",
        "kokoro/English",
    ),
)


def _schema_validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object], name: str) -> None:
    errors = sorted(_schema_validator(name).iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _browser_config(
    *,
    profile: str = "browser-en-kokoro",
    language: local_browser.Language = local_browser.Language.EN,
    tts_backend: str = "kokoro",
    tts_label: str = "kokoro/English",
    allow_barge_in: bool = False,
) -> LocalBrowserConfig:
    return LocalBrowserConfig(
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
        allow_barge_in=allow_barge_in,
        tts_runtime_label=tts_label,
        config_profile=profile,
    )


def _test_client(config: LocalBrowserConfig):
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient

    app, _ = create_app(config)
    return app, TestClient(app)


async def _setup_strategy(strategy):
    manager = TaskManager()
    manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await strategy.setup(manager)
    return manager


def _event_types(bus: BrowserPerformanceEventBus) -> list[str]:
    return [event.event_type for event in bus.recent(limit=30)]


def test_webrtc_audio_health_protected_default_does_not_auto_arm_on_echo_cancellation():
    controller = WebRTCAudioHealthController()
    payload = controller.snapshot(
        profile="browser-en-kokoro",
        language="en",
        browser_media={
            "mode": "audio_only",
            "microphone_state": "ready",
            "echo": {
                "echo_cancellation": "enabled",
                "noise_suppression": "enabled",
                "auto_gain_control": "enabled",
            },
            "output_playback_state": "speaking",
        },
        protected_playback=True,
        explicit_barge_in_armed=False,
        assistant_speaking=True,
    ).as_dict()

    _assert_schema_valid(payload, "webrtc_audio_health_v2.schema.json")
    assert payload["barge_in_state"] == "protected"
    assert payload["adaptive_barge_in_armed"] is False
    assert payload["echo_risk"] == "medium"
    assert "webrtc_audio_health:echo_cancellation_hint_only" in payload["reason_codes"]


def test_webrtc_audio_health_arms_explicit_and_adaptive_echo_safe_modes():
    controller = WebRTCAudioHealthController()
    explicit = controller.snapshot(
        profile="browser-en-kokoro",
        language="en",
        browser_media={"mode": "audio_only", "microphone_state": "ready"},
        protected_playback=False,
        explicit_barge_in_armed=True,
    ).as_dict()
    adaptive = controller.snapshot(
        profile="browser-zh-melo",
        language="zh",
        browser_media={
            "mode": "camera_and_microphone",
            "microphone_state": "receiving",
            "echo_safe": True,
            "output_playback_state": "idle",
            "output_route": "headphones",
            "webrtc_stats": {"audioLevel": 0.01, "packetsLost": 0, "sdp": "v=0"},
        },
        protected_playback=True,
        explicit_barge_in_armed=False,
    ).as_dict()

    _assert_schema_valid(explicit, "webrtc_audio_health_v2.schema.json")
    _assert_schema_valid(adaptive, "webrtc_audio_health_v2.schema.json")
    assert explicit["barge_in_state"] == "armed"
    assert explicit["explicit_barge_in_armed"] is True
    assert adaptive["barge_in_state"] == "adaptive"
    assert adaptive["adaptive_barge_in_armed"] is True
    assert adaptive["echo_risk"] == "low"
    assert adaptive["stats"]["summary"] == {"audio_level": 0.01, "packets_lost": 0}


@pytest.mark.asyncio
async def test_interruption_policy_rejects_low_confidence_noise_and_backchannels():
    state = BrowserInterruptionStateTracker(protected_playback=False)
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.05,
    )
    resets = []
    strategy.add_event_handler("on_reset_aggregation", lambda _strategy: resets.append("reset"))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    low_conf = TranscriptionFrame(
        text="please stop",
        user_id="local",
        timestamp="2026-04-26T00:00:00Z",
    )
    low_conf.confidence = 0.1
    await strategy.process_frame(low_conf)
    await strategy.process_frame(
        TranscriptionFrame(text="yeah", user_id="local", timestamp="2026-04-26T00:00:01Z")
    )
    await strategy.process_frame(
        TranscriptionFrame(text="cough", user_id="local", timestamp="2026-04-26T00:00:02Z")
    )
    await strategy.cleanup()

    snapshot = state.snapshot()
    assert snapshot["false_interruption_counts"]["low_confidence_transcript"] == 1
    assert snapshot["false_interruption_counts"]["short_backchannel"] == 1
    assert snapshot["false_interruption_counts"]["acoustic_noise"] == 1
    assert resets == ["reset", "reset", "reset"]


@pytest.mark.asyncio
async def test_explicit_interruption_accepts_and_flushes_public_output_events():
    bus = BrowserPerformanceEventBus(max_events=20)
    state = BrowserInterruptionStateTracker(
        protected_playback=False,
        performance_emit=bus.emit,
    )
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.05,
    )
    starts = []
    strategy.add_event_handler("on_user_turn_started", lambda _strategy, params: starts.append(params))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(
        TranscriptionFrame(text="hold on", user_id="local", timestamp="2026-04-26T00:00:00Z")
    )
    await strategy.cleanup()
    state.record_output_flushed(frame_type="TTSSpeakFrame")

    assert len(starts) == 1
    assert state.snapshot()["last_decision"] == "output_flushed"
    assert _event_types(bus) == [
        "interruption.accepted",
        "interruption.output_dropped",
        "interruption.output_flushed",
    ]
    encoded = json.dumps(bus.as_payload(limit=20), ensure_ascii=False)
    assert "hold on" not in encoded
    assert "output_flushed_count" in encoded


def test_actor_event_mapping_exposes_new_interruption_types():
    bus = BrowserPerformanceEventBus(max_events=20)
    for event_type in (
        "interruption.candidate",
        "interruption.accepted",
        "interruption.rejected",
        "interruption.output_flushed",
        "interruption.listening_resumed",
    ):
        bus.emit(
            event_type=event_type,
            source="test",
            mode=BrowserInteractionMode.INTERRUPTED,
            metadata={"echo_risk_state": "low", "raw_transcript": "private words"},
            reason_codes=[f"test:{event_type}"],
        )

    actor_types = [event.as_dict()["event_type"] for event in bus.actor_recent(limit=10)]
    assert actor_types == [
        "interruption_candidate",
        "interruption_accepted",
        "interruption_rejected",
        "output_flushed",
        "interruption_recovered",
    ]
    encoded = json.dumps(bus.actor_payload(limit=10), ensure_ascii=False)
    assert "private words" not in encoded


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label"),
    PRIMARY_BROWSER_PROFILES,
)
def test_actor_state_exposes_webrtc_audio_health_for_primary_profiles(
    profile,
    language,
    tts_backend,
    tts_label,
):
    app, client = _test_client(
        _browser_config(
            profile=profile,
            language=language,
            tts_backend=tts_backend,
            tts_label=tts_label,
        )
    )
    app.state.blink_browser_active_session_id = "session_audio_health"
    app.state.blink_browser_active_client_id = "client_audio_health"

    response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "receiving",
            "microphone_state": "receiving",
            "echoCancellation": True,
            "noiseSuppression": True,
            "echoSafe": True,
            "outputPlaybackState": "idle",
            "outputRoute": "headphones",
            "webrtcStats": {"audioLevel": 0.02, "packetsLost": 0},
            "sdp": "v=0 private",
            "ice": "a=candidate:private",
        },
    )
    payload = client.get("/api/runtime/actor-state").json()

    assert response.status_code == 200
    _assert_schema_valid(payload, "browser_actor_state_v2.schema.json")
    health = payload["webrtc_audio_health"]
    assert health["profile"] == profile
    assert health["language"] == language.value
    assert health["barge_in_state"] == "adaptive"
    assert health["echo_risk"] == "low"
    assert health["stats"]["summary"] == {"audio_level": 0.02, "packets_lost": 0}
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "v=0 private" not in encoded
    assert "a=candidate:private" not in encoded


def test_protected_mute_strategy_allows_adaptive_policy_without_suppression():
    state = BrowserInterruptionStateTracker(protected_playback=True)
    state.set_audio_health_policy(barge_in_state="adaptive", echo_risk="low")
    from blink.interaction import BrowserProtectedPlaybackMuteStrategy

    async def run_strategy():
        strategy = BrowserProtectedPlaybackMuteStrategy(interruption_state=state)
        await strategy.process_frame(BotStartedSpeakingFrame())
        return await strategy.process_frame(
            TranscriptionFrame(text="wait", user_id="local", timestamp="2026-04-26T00:00:00Z")
        )

    assert asyncio.run(run_strategy()) is False
    assert state.snapshot()["last_decision"] == "none"

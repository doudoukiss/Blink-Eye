import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.interaction import BrowserActorStateV2
from blink.interaction.performance_events import BrowserInteractionMode

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "browser_actor_state_v2.schema.json"
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


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _browser_config(
    *,
    profile: str,
    language: local_browser.Language,
    tts_backend: str,
    tts_label: str,
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
        allow_barge_in=False,
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


def _actor_state_for_profile(
    profile: str,
    language: local_browser.Language,
    tts_backend: str,
    tts_label: str,
) -> dict[str, object]:
    _app, client = _test_client(
        _browser_config(
            profile=profile,
            language=language,
            tts_backend=tts_backend,
            tts_label=tts_label,
        )
    )
    response = client.get("/api/runtime/actor-state")
    assert response.status_code == 200
    return response.json()


def test_browser_actor_state_v2_schema_validates_both_primary_profiles():
    for profile, language, tts_backend, tts_label in PRIMARY_BROWSER_PROFILES:
        payload = _actor_state_for_profile(profile, language, tts_backend, tts_label)

        _assert_schema_valid(payload)
        assert payload["schema_version"] == 2
        assert payload["profile"] == profile
        assert payload["language"] == language.value
        assert payload["tts"]["backend"] == tts_backend
        assert payload["tts"]["label"] == tts_label
        assert payload["vision"]["backend"] == "moondream"


def test_browser_actor_state_primary_profiles_have_same_renderable_structure():
    states = [
        _actor_state_for_profile(profile, language, tts_backend, tts_label)
        for profile, language, tts_backend, tts_label in PRIMARY_BROWSER_PROFILES
    ]

    assert set(states[0]) == set(states[1])
    for key in (
        "tts",
        "webrtc",
        "microphone",
        "camera",
        "camera_scene",
        "vision",
        "webrtc_audio_health",
        "degradation",
        "live_text",
        "memory",
    ):
        assert set(states[0][key]) == set(states[1][key])
    assert set(states[0]["camera"]["presence"]) == set(states[1]["camera"]["presence"])
    assert set(states[0]["camera"]["scene"]) == set(states[1]["camera"]["scene"])
    assert set(states[0]["camera_scene"]) == set(states[1]["camera_scene"])
    assert set(states[0]["active_listening"]) == set(states[1]["active_listening"])
    assert set(states[0]["speech"]) == set(states[1]["speech"])
    assert set(states[0]["memory_persona"]) == set(states[1]["memory_persona"])


@pytest.mark.parametrize(
    ("profile", "language", "tts_backend", "tts_label"),
    PRIMARY_BROWSER_PROFILES,
)
def test_browser_actor_state_endpoint_reports_primary_runtime_labels(
    profile,
    language,
    tts_backend,
    tts_label,
):
    payload = _actor_state_for_profile(profile, language, tts_backend, tts_label)

    assert payload["runtime"] == "browser"
    assert payload["transport"] == "WebRTC"
    assert payload["protected_playback"] is True
    assert payload["webrtc"]["media_mode"] == "unreported"
    assert payload["microphone"]["state"] == "unknown"
    assert payload["camera"]["enabled"] is True
    assert payload["camera"]["scene"]["state"] == "waiting_for_frame"
    assert payload["camera_scene"]["vision_backend"] == "moondream"
    assert payload["vision"]["enabled"] is True
    assert payload["vision"]["backend"] == "moondream"
    assert payload["vision"]["continuous_perception_enabled"] is False
    assert payload["interruption"]["barge_in_state"] == "protected"
    assert payload["webrtc_audio_health"]["barge_in_state"] == "protected"
    assert payload["webrtc_audio_health"]["protected_playback"] is True
    assert payload["degradation"]["state"] == "ok"
    assert "runtime:waiting_for_client" in payload["degradation"]["reason_codes"]


def test_browser_actor_state_last_actor_event_advances_across_modes():
    app, client = _test_client(
        _browser_config(
            profile="browser-zh-melo",
            language=local_browser.Language.ZH,
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
        )
    )
    bus = app.state.blink_browser_performance_events
    app.state.blink_browser_active_session_id = "session_actor_state"
    app.state.blink_browser_active_client_id = "client_actor_state"

    transitions = [
        ("webrtc.connection_created", BrowserInteractionMode.CONNECTED, "connected"),
        ("voice.speech_started", BrowserInteractionMode.LISTENING, "speech_started"),
        ("llm.response_start", BrowserInteractionMode.THINKING, "thinking"),
        ("vision.fetch_user_image_start", BrowserInteractionMode.LOOKING, "looking"),
        ("tts.speech_start", BrowserInteractionMode.SPEAKING, "speaking"),
    ]
    latest_event_id = 0
    for event_type, mode, actor_type in transitions:
        app.state.blink_browser_interaction_mode = mode
        event = bus.emit(
            event_type=event_type,
            source="test",
            mode=mode,
            session_id="session_actor_state",
            client_id="client_actor_state",
            reason_codes=[f"test:{actor_type}"],
        )
        latest_event_id = event.event_id
        payload = client.get("/api/runtime/actor-state").json()
        _assert_schema_valid(payload)
        assert payload["mode"] == mode.value
        assert payload["last_actor_event_id"] == latest_event_id
        assert payload["last_actor_event"]["event_type"] == actor_type
        assert payload["last_actor_event_at"] == payload["last_actor_event"]["timestamp"]

    app.state.blink_browser_interaction_mode = BrowserInteractionMode.WAITING
    app.state.blink_browser_active_session_id = None
    app.state.blink_browser_active_client_id = None
    event = bus.emit(
        event_type="runtime.task_finished",
        source="test",
        mode=BrowserInteractionMode.WAITING,
        reason_codes=["test:waiting"],
    )
    payload = client.get("/api/runtime/actor-state").json()

    _assert_schema_valid(payload)
    assert payload["mode"] == "waiting"
    assert payload["last_actor_event_id"] == event.event_id
    assert payload["last_actor_event"]["event_type"] == "waiting"


def test_browser_actor_state_is_additive_and_performance_state_remains_v1():
    _app, client = _test_client(
        _browser_config(
            profile="browser-en-kokoro",
            language=local_browser.Language.EN,
            tts_backend="kokoro",
            tts_label="kokoro/English",
        )
    )

    actor_state = client.get("/api/runtime/actor-state").json()
    performance_state = client.get("/api/runtime/performance-state").json()

    assert actor_state["schema_version"] == 2
    assert performance_state["schema_version"] == 1
    assert isinstance(actor_state["tts"], dict)
    assert isinstance(performance_state["tts"], str)
    assert "/api/runtime/performance-state" != "/api/runtime/actor-state"


def test_browser_actor_state_dataclass_allows_bounded_live_text_only_in_live_text():
    payload = BrowserActorStateV2(
        profile="browser-en-kokoro",
        language="en",
        mode="listening",
        tts_backend="kokoro",
        tts_label="kokoro/English",
        protected_playback=True,
        browser_media={"mode": "audio_only", "microphone_state": "ready"},
        active_listening={
            "display_partial_transcript": "hello there",
            "raw_transcript": "private user speech",
            "phase": "partial_transcript",
        },
        speech={
            "assistant_subtitle": "short reply",
            "raw_text": "private assistant message",
        },
        webrtc_audio_health={
            "schema_version": 2,
            "runtime": "browser",
            "transport": "WebRTC",
            "profile": "browser-en-kokoro",
            "language": "en",
            "microphone_state": "ready",
            "input_track_state": "receiving",
            "output_playback_state": "idle",
            "echo_risk": "medium",
            "barge_in_state": "protected",
            "protected_playback": True,
            "adaptive_barge_in_armed": False,
            "explicit_barge_in_armed": False,
            "headphones_recommended": True,
            "microphone": {"state": "ready", "ready": True, "reason_codes": []},
            "input_track": {"state": "receiving", "healthy": True, "reason_codes": []},
            "output_playback": {
                "state": "idle",
                "route": "unknown",
                "assistant_speaking": False,
                "reason_codes": [],
            },
            "echo": {
                "echo_cancellation": "unknown",
                "noise_suppression": "unknown",
                "auto_gain_control": "unknown",
                "echo_safe": None,
                "echo_safe_source": "none",
                "reason_codes": [],
            },
            "stats": {"available": False, "summary": {}, "reason_codes": []},
            "barge_in": {
                "state": "protected",
                "protected": True,
                "armed": False,
                "adaptive": False,
                "explicitly_armed": False,
                "reason_codes": [],
            },
            "false_interruption_counts": {},
            "updated_at": "2026-04-26T00:00:00+00:00",
            "reason_codes": ["webrtc_audio_health:v2"],
        },
    ).as_dict()

    _assert_schema_valid(payload)
    assert payload["live_text"] == {
        "partial_transcript": "hello there",
        "final_transcript": None,
        "assistant_subtitle": "short reply",
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "private user speech" not in encoded
    assert "private assistant message" not in encoded

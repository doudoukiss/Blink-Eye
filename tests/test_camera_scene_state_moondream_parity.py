import json
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.interaction import (
    BrowserVisionGroundingTracker,
    build_camera_scene_state,
)
from blink.interaction.performance_events import BrowserInteractionMode

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "camera_scene_state.schema.json"
SCENE_SOCIAL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "scene_social_state_v2.schema.json"
)
ACTOR_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "browser_actor_state_v2.schema.json"
)

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


def _schema_validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object], path: Path = SCHEMA_PATH) -> None:
    errors = sorted(_schema_validator(path).iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _browser_config(
    *,
    profile: str,
    language: local_browser.Language,
    tts_backend: str,
    tts_label: str,
    vision_enabled: bool = True,
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
        vision_enabled=vision_enabled,
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


class _FreshBuffer:
    latest_camera_frame_seq = 11
    latest_camera_frame_received_monotonic = time.monotonic()
    latest_camera_frame_received_at = "2026-04-27T00:00:00+00:00"


class _StaleBuffer:
    latest_camera_frame_seq = 12
    latest_camera_frame_received_monotonic = time.monotonic() - 30.0
    latest_camera_frame_received_at = "2026-04-27T00:00:00+00:00"


class _Health:
    def __init__(self, **payload):
        self._payload = payload

    def as_dict(self):
        return dict(self._payload)


def _scene_payload(**overrides):
    defaults = {
        "profile": "browser-zh-melo",
        "language": "zh",
        "vision_enabled": True,
        "continuous_perception_enabled": False,
        "browser_media": {"mode": "camera_and_microphone", "camera_state": "ready"},
        "active_client_id": "client-camera",
        "active_session_id": "session-camera",
        "camera_buffer": None,
        "camera_health": None,
        "grounding_tracker": BrowserVisionGroundingTracker(),
        "vision_backend": "moondream",
    }
    defaults.update(overrides)
    return build_camera_scene_state(**defaults).as_dict()


def test_camera_scene_state_validates_primary_profiles_from_actor_state():
    states = []
    for profile, language, tts_backend, tts_label in PRIMARY_BROWSER_PROFILES:
        _app, client = _test_client(
            _browser_config(
                profile=profile,
                language=language,
                tts_backend=tts_backend,
                tts_label=tts_label,
            )
        )
        actor_state = client.get("/api/runtime/actor-state").json()
        scene = actor_state["camera_scene"]
        states.append(scene)

        _assert_schema_valid(scene)
        _assert_schema_valid(scene["scene_social_state_v2"], SCENE_SOCIAL_SCHEMA_PATH)
        _assert_schema_valid(actor_state, ACTOR_SCHEMA_PATH)
        assert scene["profile"] == profile
        assert scene["language"] == language.value
        assert scene["enabled"] is True
        assert scene["vision_backend"] == "moondream"
        assert scene["continuous_perception_enabled"] is False
        assert scene["state"] == "waiting_for_frame"
        assert scene["scene_social_state_v2"]["schema_version"] == 2
        assert scene["scene_social_state_v2"]["camera_honesty_state"] == "unavailable"
        assert scene["scene_social_state_v2"]["scene_transition"] == "none"
        assert actor_state["camera"]["scene"] == scene

    assert set(states[0]) == set(states[1])


def test_camera_scene_explicit_no_vision_opt_out_reports_disabled():
    _app, client = _test_client(
        _browser_config(
            profile="browser-en-kokoro",
            language=local_browser.Language.EN,
            tts_backend="kokoro",
            tts_label="kokoro/English",
            vision_enabled=False,
        )
    )

    actor_state = client.get("/api/runtime/actor-state").json()
    scene = actor_state["camera_scene"]

    _assert_schema_valid(scene)
    _assert_schema_valid(scene["scene_social_state_v2"], SCENE_SOCIAL_SCHEMA_PATH)
    assert scene["state"] == "disabled"
    assert scene["enabled"] is False
    assert scene["vision_backend"] == "none"
    assert scene["scene_social_state_v2"]["camera_honesty_state"] == "unavailable"
    assert scene["scene_social_state_v2"]["scene_transition"] == "vision_unavailable"
    assert actor_state["vision"]["backend"] == "none"


@pytest.mark.parametrize(
    ("payload", "expected_state"),
    [
        (
            {
                "camera_buffer": _FreshBuffer(),
                "camera_health": _Health(
                    camera_connected=True,
                    camera_fresh=True,
                    camera_track_state="healthy",
                    frame_age_ms=24,
                    last_fresh_frame_at="2026-04-27T00:00:00+00:00",
                ),
            },
            "available",
        ),
        ({}, "waiting_for_frame"),
        (
            {
                "camera_buffer": _StaleBuffer(),
                "camera_health": _Health(
                    camera_connected=True,
                    camera_fresh=False,
                    camera_track_state="stalled",
                    frame_age_ms=30000,
                    sensor_health_reason="camera_frame_stale",
                ),
            },
            "stale",
        ),
        (
            {
                "camera_health": _Health(
                    camera_connected=True,
                    camera_fresh=False,
                    camera_track_state="recovering",
                    sensor_health_reason="camera_track_stalled",
                ),
            },
            "stalled",
        ),
        (
            {
                "browser_media": {
                    "mode": "audio_only",
                    "camera_state": "permission_denied",
                },
            },
            "permission_needed",
        ),
    ],
)
def test_camera_scene_maps_health_and_permission_states(payload, expected_state):
    scene = _scene_payload(**payload)

    _assert_schema_valid(scene)
    _assert_schema_valid(scene["scene_social_state_v2"], SCENE_SOCIAL_SCHEMA_PATH)
    assert scene["state"] == expected_state
    assert scene["status"] == expected_state
    if expected_state == "available":
        assert scene["available"] is True
        assert scene["camera_fresh"] is True
        assert scene["latest_frame_sequence"] == 11
        assert scene["scene_social_state_v2"]["camera_honesty_state"] == "recent_frame_available"
    if expected_state in {"permission_needed", "stale", "stalled", "waiting_for_frame"}:
        assert scene["current_answer_used_vision"] is False
        assert scene["scene_social_state_v2"]["camera_honesty_state"] != "can_see_now"


def test_camera_scene_tracks_looking_success_and_error_without_raw_payloads():
    tracker = BrowserVisionGroundingTracker()
    tracker.mark_looking(frame_seq=8, frame_age_ms=17)
    looking = _scene_payload(camera_buffer=_FreshBuffer(), grounding_tracker=tracker)

    tracker.mark_success(
        frame_seq=8,
        frame_age_ms=19,
        scene_social_hints={
            "user_presence_hint": "present",
            "hands_hint": "visible",
            "object_hint": "object_showing",
            "object_showing_likelihood": 0.85,
            "last_moondream_result_state": "answered",
            "last_grounding_summary": "fresh_object_grounding",
            "last_grounding_summary_hash": "0" * 16,
            "confidence": 0.85,
            "confidence_bucket": "high",
        },
    )
    success = _scene_payload(camera_buffer=_FreshBuffer(), grounding_tracker=tracker)

    tracker.mark_error(
        result_state="error",
        reason_code="vision:fetch_user_image_error",
        frame_seq=8,
        frame_age_ms=20,
    )
    error = _scene_payload(camera_buffer=_FreshBuffer(), grounding_tracker=tracker)

    for scene in (looking, success, error):
        _assert_schema_valid(scene)
    assert looking["state"] == "looking"
    assert looking["on_demand_vision_state"] == "looking"
    assert success["state"] == "available"
    assert success["current_answer_used_vision"] is True
    assert success["grounding_mode"] == "single_frame"
    assert success["last_used_frame_sequence"] == 8
    assert success["scene_social_state_v2"]["camera_honesty_state"] == "can_see_now"
    assert success["scene_social_state_v2"]["scene_transition"] == "vision_answered"
    assert success["scene_social_state_v2"]["object_hint"] == "object_showing"
    assert success["scene_social_state_v2"]["last_grounding_summary"] == "fresh_object_grounding"
    assert error["state"] == "error"
    assert error["degradation"]["state"] == "error"
    assert error["scene_social_state_v2"]["camera_honesty_state"] == "unavailable"
    assert error["scene_social_state_v2"]["scene_transition"] == "vision_unavailable"
    encoded = str({"looking": looking, "success": success, "error": error})
    assert "data:image" not in encoded
    assert "raw" not in encoded.lower()
    assert "hidden prompt" not in encoded.lower()


def test_camera_scene_actor_events_map_looking_error_and_degradation():
    app, client = _test_client(
        _browser_config(
            profile="browser-zh-melo",
            language=local_browser.Language.ZH,
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
        )
    )
    bus = app.state.blink_browser_performance_events

    for event_type, mode, expected_actor_type in (
        ("vision.fetch_user_image_start", BrowserInteractionMode.LOOKING, "looking"),
        ("vision.fetch_user_image_error", BrowserInteractionMode.ERROR, "error"),
        ("camera.frame_stale", BrowserInteractionMode.ERROR, "degraded"),
        ("camera.track_resumed", BrowserInteractionMode.WAITING, "recovered"),
    ):
        transition = "vision_answered" if event_type.endswith("_success") else "vision_stale"
        if event_type in {"vision.fetch_user_image_start", "camera.track_resumed"}:
            transition = "looking_requested" if event_type.endswith("_start") else "camera_ready"
        event = bus.emit(
            event_type=event_type,
            source="test",
            mode=mode,
            metadata={
                "frame_seq": 7,
                "frame_age_ms": 42,
                "scene_transition": transition,
                "camera_honesty_state": "unavailable",
                "image": "data:image/png;base64,secret",
                "prompt": "hidden prompt",
            },
            session_id="session-camera",
            client_id="client-camera",
            reason_codes=[f"test:{expected_actor_type}"],
        )
        actor_state = client.get("/api/runtime/actor-state").json()

        assert actor_state["last_actor_event_id"] == event.event_id
        assert actor_state["last_actor_event"]["event_type"] == expected_actor_type
        assert "image" not in actor_state["last_actor_event"]["metadata"]
        assert "prompt" not in actor_state["last_actor_event"]["metadata"]
        assert "scene_transition" in actor_state["last_actor_event"]["metadata"]


def test_scene_social_honesty_never_claims_can_see_now_without_fresh_vision():
    stale = _scene_payload(
        camera_buffer=_StaleBuffer(),
        camera_health=_Health(
            camera_connected=True,
            camera_fresh=False,
            camera_track_state="stalled",
            frame_age_ms=30000,
            sensor_health_reason="camera_frame_stale",
        ),
    )
    available = _scene_payload(
        camera_buffer=_FreshBuffer(),
        camera_health=_Health(
            camera_connected=True,
            camera_fresh=True,
            camera_track_state="healthy",
            frame_age_ms=24,
        ),
    )

    _assert_schema_valid(stale)
    _assert_schema_valid(available)
    assert stale["scene_social_state_v2"]["camera_honesty_state"] == "unavailable"
    assert stale["scene_social_state_v2"]["scene_transition"] == "vision_stale"
    assert available["scene_social_state_v2"]["camera_honesty_state"] == "recent_frame_available"
    assert available["scene_social_state_v2"]["camera_honesty_state"] != "can_see_now"

import asyncio
import time
from types import SimpleNamespace

import pytest

from tests._optional import OPTIONAL_RUNTIME, require_optional_modules

pytestmark = OPTIONAL_RUNTIME
require_optional_modules("fastapi", "aiortc", "av")

from fastapi.testclient import TestClient

from blink.cli import local_browser
from blink.cli.local_browser import (
    LocalBrowserConfig,
    _build_vision_prompt,
    build_local_browser_runtime,
    create_app,
)
from blink.frames.frames import ErrorFrame, UserImageRawFrame, VisionTextFrame
from blink.interaction.camera_presence import (
    BrowserVisionGroundingTracker,
    build_browser_camera_presence_snapshot,
    build_camera_scene_state,
)
from blink.processors.frame_processor import FrameProcessor
from blink.transports.base_transport import BaseTransport


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
    async def queue_frame(self, frame):
        return None


class DummyVisionProcessor(FrameProcessor):
    def __init__(self, *, response_text="画面里有一个杯子。", error_text=None):
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


def _browser_config(
    *,
    profile="browser-zh-melo",
    language=local_browser.Language.ZH,
    tts_backend="local-http-wav",
    vision_enabled=True,
    continuous_perception_enabled=False,
):
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
        continuous_perception_enabled=continuous_perception_enabled,
        allow_barge_in=False,
        tts_runtime_label=(
            "local-http-wav/MeloTTS" if tts_backend == "local-http-wav" else "kokoro/English"
        ),
        config_profile=profile,
    )


def test_camera_presence_snapshot_maps_disabled_fresh_and_reset_grounding():
    tracker = BrowserVisionGroundingTracker()
    disabled = build_browser_camera_presence_snapshot(
        vision_enabled=False,
        continuous_perception_enabled=False,
        browser_media={"mode": "audio_only", "camera_state": "unavailable"},
        active_client_id="client-1",
        grounding_tracker=tracker,
    ).as_dict()

    assert disabled["state"] == "disabled"
    assert disabled["enabled"] is False
    assert disabled["grounding_mode"] == "none"

    class Buffer:
        latest_camera_frame_seq = 3
        latest_camera_frame_received_monotonic = time.monotonic()
        latest_camera_frame_received_at = "2026-04-27T00:00:00+00:00"

    tracker.mark_success(frame_seq=3, frame_age_ms=12)
    fresh = build_browser_camera_presence_snapshot(
        vision_enabled=True,
        continuous_perception_enabled=False,
        browser_media={"mode": "camera_and_microphone", "camera_state": "ready"},
        active_client_id="client-1",
        camera_buffer=Buffer(),
        camera_health={
            "camera_connected": True,
            "camera_fresh": True,
            "camera_track_state": "healthy",
            "frame_age_ms": 12,
            "last_fresh_frame_at": "2026-04-27T00:00:00+00:00",
        },
        grounding_tracker=tracker,
    ).as_dict()

    assert fresh["state"] == "available"
    assert fresh["latest_frame_seq"] == 3
    assert fresh["latest_frame_age_ms"] == 12
    assert fresh["current_answer_used_vision"] is True
    assert fresh["grounding_mode"] == "single_frame"

    tracker.reset_current_answer()
    reset = build_browser_camera_presence_snapshot(
        vision_enabled=True,
        continuous_perception_enabled=False,
        browser_media={"mode": "camera_and_microphone", "camera_state": "ready"},
        active_client_id="client-1",
        camera_buffer=Buffer(),
        camera_health={
            "camera_connected": True,
            "camera_fresh": True,
            "camera_track_state": "healthy",
            "frame_age_ms": 12,
        },
        grounding_tracker=tracker,
    ).as_dict()

    assert reset["current_answer_used_vision"] is False
    assert reset["last_vision_result_state"] == "success"
    assert reset["grounding_mode"] == "none"


def test_recent_browser_frame_overrides_stale_camera_health_without_false_seeing_claim():
    class Buffer:
        latest_camera_frame_seq = 9
        latest_camera_frame_received_monotonic = time.monotonic()
        latest_camera_frame_received_at = "2026-04-27T00:00:00+00:00"

    browser_media = {"mode": "camera_and_microphone", "camera_state": "ready"}
    stale_health = {
        "camera_connected": False,
        "camera_fresh": False,
        "camera_track_state": "disconnected",
        "sensor_health_reason": "camera_disconnected",
        "frame_age_ms": 120,
    }

    presence = build_browser_camera_presence_snapshot(
        vision_enabled=True,
        continuous_perception_enabled=False,
        browser_media=browser_media,
        active_client_id="client-1",
        camera_buffer=Buffer(),
        camera_health=stale_health,
        grounding_tracker=BrowserVisionGroundingTracker(),
    ).as_dict()
    scene = build_camera_scene_state(
        profile="browser-en-kokoro",
        language="en",
        vision_enabled=True,
        continuous_perception_enabled=False,
        browser_media=browser_media,
        active_client_id="client-1",
        active_session_id="session-1",
        camera_buffer=Buffer(),
        camera_health=stale_health,
        grounding_tracker=BrowserVisionGroundingTracker(),
    ).as_dict()

    assert presence["state"] == "available"
    assert presence["camera_connected"] is True
    assert presence["camera_fresh"] is True
    assert presence["current_answer_used_vision"] is False
    assert scene["state"] == "available"
    assert scene["camera_connected"] is True
    assert scene["camera_fresh"] is True
    assert scene["current_answer_used_vision"] is False
    assert scene["scene_social_state_v2"]["camera_honesty_state"] == "recent_frame_available"
    assert "can_see_now" not in str(scene)


def test_primary_profiles_expose_camera_presence_defaults():
    cases = [
        ("browser-zh-melo", local_browser.Language.ZH, "local-http-wav", True, "disconnected"),
        ("browser-en-kokoro", local_browser.Language.EN, "kokoro", True, "disconnected"),
    ]

    for profile, language, tts_backend, vision_enabled, expected_state in cases:
        app, _ = create_app(
            _browser_config(
                profile=profile,
                language=language,
                tts_backend=tts_backend,
                vision_enabled=vision_enabled,
            )
        )
        client = TestClient(app)
        payload = client.get("/api/runtime/performance-state").json()

        assert payload["profile"] == profile
        assert payload["camera_presence"]["enabled"] is vision_enabled
        assert payload["camera_presence"]["state"] == expected_state
        assert payload["camera_presence"]["current_answer_used_vision"] is False
        assert payload["camera_presence"]["grounding_mode"] == "none"
        assert payload["vision"]["continuous_perception_enabled"] is False


def test_client_media_reports_camera_presence_and_stale_event_safely():
    app, _ = create_app(_browser_config())
    app.state.blink_browser_active_client_id = "client-camera"
    app.state.blink_browser_active_session_id = "session-camera"
    client = TestClient(app)

    ready = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "ready",
            "deviceLabel": "private camera",
            "sdp": "private session description",
            "raw_text": "private transcript",
        },
    )
    ready_state = client.get("/api/runtime/performance-state").json()
    stale = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "stale",
            "microphone_state": "receiving",
        },
    )
    stale_state = client.get("/api/runtime/performance-state").json()
    events = client.get("/api/runtime/performance-events?limit=20").json()["events"]

    assert ready.status_code == 200
    assert stale.status_code == 200
    assert ready_state["camera_presence"]["state"] == "waiting_for_frame"
    assert stale_state["camera_presence"]["state"] == "stale"
    assert "camera.connected" in [event["event_type"] for event in events]
    assert "camera.frame_stale" in [event["event_type"] for event in events]
    encoded = str({"ready": ready.json(), "state": stale_state, "events": events})
    assert "private camera" not in encoded
    assert "private session description" not in encoded
    assert "private transcript" not in encoded


def test_client_media_permission_denied_is_degraded_without_false_camera_claim():
    app, _ = create_app(
        _browser_config(
            profile="browser-en-kokoro",
            language=local_browser.Language.EN,
            tts_backend="kokoro",
            vision_enabled=True,
        )
    )
    app.state.blink_browser_active_client_id = "client-audio-only"
    app.state.blink_browser_active_session_id = "session-audio-only"
    client = TestClient(app)

    response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "camera_state": "permission_denied",
            "microphone_state": "receiving",
            "deviceLabel": "private camera",
            "raw_text": "private transcript",
            "sdp": "v=0 private",
        },
    )
    actor_state = client.get("/api/runtime/actor-state").json()
    performance_state = client.get("/api/runtime/performance-state").json()

    assert response.status_code == 200
    assert actor_state["profile"] == "browser-en-kokoro"
    assert actor_state["vision"]["enabled"] is True
    assert actor_state["vision"]["backend"] == "moondream"
    assert actor_state["webrtc"]["media"]["mode"] == "audio_only"
    assert actor_state["webrtc"]["media"]["camera_state"] == "permission_denied"
    assert actor_state["camera_scene"]["state"] == "permission_needed"
    assert actor_state["camera_scene"]["current_answer_used_vision"] is False
    scene_social = actor_state["camera_scene"]["scene_social_state_v2"]
    assert scene_social["camera_honesty_state"] == "unavailable"
    assert scene_social["scene_transition"] == "vision_unavailable"
    assert scene_social["last_moondream_result_state"] == "none"
    assert performance_state["camera_presence"]["current_answer_used_vision"] is False
    encoded = str({"response": response.json(), "actor_state": actor_state})
    assert "can_see_now" not in encoded
    assert "private camera" not in encoded
    assert "private transcript" not in encoded
    assert "v=0 private" not in encoded


def test_audio_only_stalled_camera_reports_stalled_not_permission_needed():
    app, _ = create_app(
        _browser_config(
            profile="browser-en-kokoro",
            language=local_browser.Language.EN,
            tts_backend="kokoro",
            vision_enabled=True,
        )
    )
    app.state.blink_browser_active_client_id = "client-stalled-camera"
    app.state.blink_browser_active_session_id = "session-stalled-camera"
    client = TestClient(app)

    response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "camera_state": "stalled",
            "microphone_state": "receiving",
        },
    )
    actor_state = client.get("/api/runtime/actor-state").json()
    performance_state = client.get("/api/runtime/performance-state").json()

    assert response.status_code == 200
    assert actor_state["camera_scene"]["state"] == "stalled"
    assert actor_state["camera_scene"]["camera_connected"] is False
    assert actor_state["camera_scene"]["current_answer_used_vision"] is False
    assert actor_state["camera_scene"]["scene_social_state_v2"]["camera_honesty_state"] == (
        "unavailable"
    )
    assert performance_state["camera_presence"]["state"] == "stalled"
    assert "can_see_now" not in str(actor_state["camera_scene"])


def test_client_media_ready_restores_camera_context_after_denied_hint():
    app, _ = create_app(
        _browser_config(
            profile="browser-en-kokoro",
            language=local_browser.Language.EN,
            tts_backend="kokoro",
            vision_enabled=True,
        )
    )

    class FakeContext:
        def __init__(self):
            self.messages = []

        def add_message(self, message):
            self.messages.append(message)

    class FakeRuntime:
        def __init__(self):
            self.vision_connected = []

        def note_vision_connected(self, connected):
            self.vision_connected.append(bool(connected))

    context = FakeContext()
    runtime = FakeRuntime()
    app.state.blink_active_llm_context = context
    app.state.blink_active_expression_runtime = runtime
    app.state.blink_browser_active_client_id = "client-restored-camera"
    app.state.blink_browser_active_session_id = "session-restored-camera"
    client = TestClient(app)

    denied = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "unavailable",
            "camera_state": "permission_denied",
            "microphone_state": "permission_denied",
        },
    )
    ready = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "receiving",
        },
    )

    assert denied.status_code == 200
    assert ready.status_code == 200
    assert runtime.vision_connected == [False, True]
    assert context.messages == []
    actor_state = client.get("/api/runtime/actor-state").json()
    assert actor_state["camera_scene"]["state"] in {"available", "waiting_for_frame"}
    assert actor_state["camera_scene"]["current_answer_used_vision"] is False


@pytest.mark.asyncio
async def test_fetch_user_image_returns_grounded_result_and_sanitized_events(monkeypatch):
    events = []
    cached_frame = UserImageRawFrame(
        user_id="client-camera",
        image=b"abc",
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
            self.latest_camera_frame_received_at = "2026-04-27T00:00:00+00:00"

        def latest_camera_frame_age_ms(self):
            return 42

        async def wait_for_latest_camera_frame(self, *, after_seq=0, timeout=0.0):
            return None

    monkeypatch.setattr(local_browser, "LatestCameraFrameBuffer", StubCameraFrameBuffer)
    llm = DummyLLMProcessor()
    vision = DummyVisionProcessor(response_text="画面里有一个杯子。")
    _, context = build_local_browser_runtime(
        _browser_config(),
        transport=DummyTransport(),
        stt=FrameProcessor(name="stt"),
        llm=llm,
        tts=DummyTTSProcessor(name="tts"),
        vision=vision,
        active_client={"id": "client-camera"},
        performance_emit=lambda **kwargs: events.append(kwargs),
    )
    context.add_message({"role": "user", "content": "请描述一下你现在看到的画面"})
    results = []

    async def result_callback(result):
        results.append(result)

    await llm.registered_functions["fetch_user_image"](
        SimpleNamespace(
            arguments={"question": "摄像头里有什么？"},
            function_name="fetch_user_image",
            tool_call_id="tool-camera",
            llm=llm,
            result_callback=result_callback,
        )
    )

    assert results == [{"description": "基于刚才的一帧画面：画面里有一个杯子。"}]
    assert vision.received_frames[0].text == _build_vision_prompt("请描述一下你现在看到的画面")
    success = next(event for event in events if event["event_type"] == "vision.fetch_user_image_success")
    assert success["metadata"]["frame_seq"] == 7
    assert success["metadata"]["frame_age_ms"] == 42
    assert success["metadata"]["grounding_mode"] == "single_frame"
    encoded_events = str(events)
    assert "画面里有一个杯子" not in encoded_events
    assert "abc" not in encoded_events

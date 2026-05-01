import json
from pathlib import Path

import pytest

from blink.cli import local_browser
from blink.cli.local_browser import LocalBrowserConfig, create_app
from blink.interaction.performance_events import BrowserInteractionMode

UNSAFE_STRINGS = (
    "raw user audio",
    "data:image/png;base64",
    "v=0",
    "a=candidate",
    "sk-private",
    "Bearer private",
    "Secret prompt",
    "full hidden message",
)


def _browser_config(*, trace_dir: Path | None = None) -> LocalBrowserConfig:
    return LocalBrowserConfig(
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
        actor_trace_enabled=trace_dir is not None,
        actor_trace_dir=trace_dir,
    )


def _test_client(config: LocalBrowserConfig):
    pytest.importorskip("fastapi")
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from fastapi.testclient import TestClient

    app, _ = create_app(config)
    return app, TestClient(app)


def test_actor_state_omits_raw_media_signaling_credentials_prompts_and_messages():
    app, client = _test_client(_browser_config())
    app.state.blink_browser_active_session_id = "session_public_safety"
    app.state.blink_browser_active_client_id = "client_public_safety"
    response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "ready",
            "echoCancellation": True,
            "raw_text": "full hidden message",
            "sdp": "v=0 private session",
            "ice": "a=candidate:private",
            "Authorization": "Bearer private",
        },
    )

    assert response.status_code == 200
    app.state.blink_browser_performance_events.emit(
        event_type="stt.transcription",
        source="stt",
        mode=BrowserInteractionMode.HEARD,
        session_id="session_public_safety",
        client_id="client_public_safety",
        metadata={
            "audio": "raw user audio",
            "image": "data:image/png;base64,AAAA",
            "sdp": "v=0 private session",
            "candidate": "a=candidate:private",
            "token": "sk-private",
            "authorization": "Bearer private",
            "messages": [{"content": "full hidden message"}],
            "final_transcript_chars": 18,
        },
        reason_codes=["stt:transcribed", "Bearer private"],
    )
    payload = client.get("/api/runtime/actor-state").json()
    encoded = json.dumps(payload, ensure_ascii=False)

    for unsafe in UNSAFE_STRINGS:
        assert unsafe not in encoded
    assert payload["schema_version"] == 2
    assert payload["webrtc"]["media"]["mode"] == "camera_and_microphone"
    assert payload["last_actor_event"]["metadata"]["final_transcript_chars"] == 18


def test_actor_state_reports_degraded_and_error_without_private_payloads():
    app, client = _test_client(_browser_config())
    app.state.blink_browser_active_session_id = "session_degraded"
    app.state.blink_browser_active_client_id = "client_degraded"

    degraded_response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "camera_and_microphone",
            "camera_state": "stale",
            "microphone_state": "ready",
            "sdp": "v=0 private session",
        },
    )
    degraded_payload = client.get("/api/runtime/actor-state").json()

    assert degraded_response.status_code == 200
    assert degraded_payload["degradation"]["state"] == "degraded"
    assert "camera" in degraded_payload["degradation"]["components"]
    assert "vision" in degraded_payload["degradation"]["components"]
    assert "v=0 private session" not in json.dumps(degraded_payload, ensure_ascii=False)

    error_response = client.post(
        "/api/runtime/client-media",
        json={
            "mode": "audio_only",
            "microphone_state": "permission_denied",
            "Authorization": "Bearer private",
        },
    )
    error_payload = client.get("/api/runtime/actor-state").json()

    assert error_response.status_code == 200
    assert error_payload["degradation"]["state"] == "error"
    assert "microphone" in error_payload["degradation"]["components"]
    assert "Bearer private" not in json.dumps(error_payload, ensure_ascii=False)


def test_actor_state_live_text_is_not_persisted_to_actor_trace(tmp_path):
    app, client = _test_client(_browser_config(trace_dir=tmp_path))
    bus = app.state.blink_browser_performance_events
    app.state.blink_browser_active_session_id = "session_trace_text"
    app.state.blink_browser_active_client_id = "client_trace_text"

    bus.emit(
        event_type="stt.partial_transcription",
        source="stt",
        mode=BrowserInteractionMode.LISTENING,
        session_id="session_trace_text",
        client_id="client_trace_text",
        metadata={
            "partial_transcript_chars": 11,
            "raw_transcript": "hello world",
        },
        reason_codes=["stt:partial_transcript"],
    )
    payload = client.get("/api/runtime/actor-state").json()

    assert payload["live_text"] == {
        "partial_transcript": None,
        "final_transcript": None,
        "assistant_subtitle": None,
    }
    trace_path = Path(app.state.blink_actor_trace_path)
    trace_payloads = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    encoded_trace = json.dumps(trace_payloads, ensure_ascii=False)

    assert trace_payloads
    assert all("live_text" not in payload for payload in trace_payloads)
    assert "hello world" not in encoded_trace
    assert trace_payloads[-1]["metadata"]["partial_transcript_chars"] == 11

from blink.interaction.browser_state import BrowserInteractionState
from blink.interaction.performance_events import (
    BrowserInteractionMode,
    BrowserPerformanceEventBus,
)


def test_performance_event_bus_orders_filters_and_bounds_events():
    bus = BrowserPerformanceEventBus(max_events=3)

    first = bus.emit(
        event_type="webrtc.connected",
        source="webrtc",
        mode=BrowserInteractionMode.CONNECTED,
    )
    second = bus.emit(
        event_type="voice.speech_started",
        source="vad",
        mode=BrowserInteractionMode.LISTENING,
    )
    third = bus.emit(
        event_type="stt.transcription",
        source="stt",
        mode=BrowserInteractionMode.HEARD,
    )
    fourth = bus.emit(
        event_type="llm.response_start",
        source="llm",
        mode=BrowserInteractionMode.THINKING,
    )

    assert first.event_id == 1
    assert second.event_id == 2
    assert third.event_id == 3
    assert fourth.event_id == 4
    assert [event.event_id for event in bus.recent(limit=10)] == [2, 3, 4]
    assert [event.event_id for event in bus.recent(after_id=2, limit=10)] == [3, 4]
    assert bus.as_payload(after_id=3, limit=50)["latest_event_id"] == 4


def test_performance_event_metadata_is_public_safe():
    bus = BrowserPerformanceEventBus(max_events=10)

    event = bus.emit(
        event_type="stt.transcription",
        source="stt",
        mode=BrowserInteractionMode.HEARD,
        session_id="session abc",
        client_id="client abc",
        metadata={
            "transcript": "raw user words",
            "raw_text": "assistant text",
            "system_prompt": "secret prompt",
            "audio_bytes": b"secret",
            "image": "pixels",
            "memory_id": "memory_claim:user:secret",
            "transcription_chars": 18,
            "voice_state": "receiving",
            "nested": {"authorization": "Bearer secret", "count": 2},
        },
        reason_codes=["stt:transcribed", "stt:transcribed"],
    )

    payload = event.as_dict()
    metadata = payload["metadata"]
    assert payload["session_id"] == "session_abc"
    assert payload["client_id"] == "client_abc"
    assert payload["reason_codes"] == ["stt:transcribed"]
    assert "transcript" not in metadata
    assert "raw_text" not in metadata
    assert "system_prompt" not in metadata
    assert "audio_bytes" not in metadata
    assert "image" not in metadata
    assert "memory_id" not in metadata
    assert metadata["transcription_chars"] == 18
    assert metadata["voice_state"] == "receiving"
    assert metadata["nested"] == {"count": 2}


def test_browser_interaction_state_snapshot_includes_public_runtime_context():
    bus = BrowserPerformanceEventBus(max_events=10)
    event = bus.emit(
        event_type="tts.speech_start",
        source="tts",
        mode=BrowserInteractionMode.SPEAKING,
        reason_codes=["tts:speaking"],
    )
    state = BrowserInteractionState(
        mode=BrowserInteractionMode.SPEAKING,
        profile="browser-en-kokoro",
        tts_label="kokoro/English",
        tts_backend="kokoro",
        protected_playback=True,
        browser_media={
            "mode": "audio_only",
            "camera_state": "unavailable",
            "microphone_state": "ready",
        },
        vision_enabled=False,
        continuous_perception_enabled=False,
        memory_available=True,
        interruption={
            "barge_in_state": "protected",
            "protected_playback": True,
            "last_decision": "none",
            "reason_codes": ["interruption:protected"],
        },
        speech={
            "director_mode": "kokoro_chunked",
            "first_subtitle_latency_ms": 0.0,
            "first_audio_latency_ms": 0.0,
            "speech_queue_depth_current": 0,
            "stale_chunk_drop_count": 0,
            "reason_codes": ["speech_director:kokoro_chunked"],
        },
        active_listening={
            "phase": "final_transcript",
            "partial_available": False,
            "final_transcript_chars": 18,
            "topics": [
                {
                    "kind": "topic",
                    "value": "browser status",
                    "confidence": "heuristic",
                    "source": "final_transcript",
                    "editable": True,
                }
            ],
            "constraints": [],
            "reason_codes": ["active_listening:v1", "active_listening:final_transcript"],
        },
        camera_presence={
            "state": "disabled",
            "enabled": False,
            "available": False,
            "current_answer_used_vision": False,
            "grounding_mode": "none",
            "reason_codes": ["camera_presence:v1", "camera_presence:disabled"],
        },
        memory_persona={
            "available": True,
            "profile": "browser-en-kokoro",
            "memory_policy": "balanced",
            "selected_memory_count": 1,
            "suppressed_memory_count": 0,
            "used_in_current_reply": [
                {
                    "memory_id": "memory_claim:user:user-1:claim-1",
                    "display_kind": "preference",
                    "title": "browser status",
                    "used_reason": "selected_for_relevant_continuity",
                    "behavior_effect": "memory callback changed this reply",
                    "reason_codes": ["source:context_selection"],
                }
            ],
            "behavior_effects": ["memory_callback_active"],
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
            "reason_codes": ["memory_persona_performance:v1"],
        },
        active_session_id="session_1",
        active_client_id="client_1",
        last_event=event,
    )

    payload = state.as_dict()

    assert payload["runtime"] == "browser"
    assert payload["transport"] == "WebRTC"
    assert payload["mode"] == "speaking"
    assert payload["profile"] == "browser-en-kokoro"
    assert payload["tts"] == "kokoro/English"
    assert payload["protected_playback"] is True
    assert payload["interruption"]["barge_in_state"] == "protected"
    assert payload["speech"]["director_mode"] == "kokoro_chunked"
    assert payload["active_listening"]["phase"] == "final_transcript"
    assert payload["active_listening"]["topics"][0]["value"] == "browser status"
    assert payload["camera_presence"]["state"] == "disabled"
    assert payload["camera_presence"]["grounding_mode"] == "none"
    assert payload["memory_persona"]["available"] is True
    assert payload["memory_persona"]["selected_memory_count"] == 1
    assert payload["memory_persona"]["used_in_current_reply"][0]["display_kind"] == "preference"
    assert payload["browser_media"]["mode"] == "audio_only"
    assert payload["memory"]["available"] is True
    assert payload["last_event_id"] == event.event_id
    assert "mode:speaking" in payload["reason_codes"]
    assert "profile:browser-en-kokoro" in payload["reason_codes"]
    assert "interruption:protected" in payload["reason_codes"]
    assert "speech_director:kokoro_chunked" in payload["reason_codes"]
    assert "active_listening:final_transcript" in payload["reason_codes"]
    assert "camera_presence:disabled" in payload["reason_codes"]
    assert "memory_persona:available" in payload["reason_codes"]
    assert "memory_persona_performance:v1" in payload["reason_codes"]

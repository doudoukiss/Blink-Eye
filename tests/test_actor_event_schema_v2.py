import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.interaction.actor_events import (
    ActorEventContext,
    ActorEventModeV2,
    ActorEventTypeV2,
    ActorEventV2,
    actor_event_from_performance_event,
    sanitize_actor_metadata,
)
from blink.interaction.performance_events import BrowserInteractionMode, BrowserPerformanceEvent

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "actor_event_v2.schema.json"


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def test_actor_event_v2_payload_matches_schema():
    event = ActorEventV2(
        event_id=1,
        event_type=ActorEventTypeV2.SPEAKING,
        mode=ActorEventModeV2.SPEAKING,
        timestamp="2026-04-27T00:00:00+00:00",
        profile="browser-en-kokoro",
        language="en",
        tts_backend="kokoro",
        tts_label="kokoro/English",
        vision_backend="moondream",
        source="tts",
        session_id="session one",
        client_id="client one",
        metadata={"context_available": True, "first_audio_latency_ms": 120.5},
        reason_codes=["tts:speaking"],
    )

    payload = event.as_dict()

    _assert_schema_valid(payload)
    assert payload["schema_version"] == 2
    assert payload["event_type"] == "speaking"
    assert payload["mode"] == "speaking"
    assert payload["tts_label"] == "kokoro/English"
    assert payload["session_id"] == "session_one"
    assert payload["client_id"] == "client_one"


def test_actor_metadata_sanitizer_omits_unsafe_keys_and_values():
    metadata, violations = sanitize_actor_metadata(
        {
            "transcript": "raw user words",
            "system_prompt": "hidden prompt",
            "audio_bytes": b"private audio",
            "image": "data:image/png;base64,AAAA",
            "candidate": "a=candidate:private",
            "safe_state": "ready",
            "safe_value_state": "Bearer secret",
            "nested": {
                "count": 2,
                "messages": [{"role": "user", "content": "full message"}],
                "state": "ok",
            },
        }
    )

    assert metadata == {"safe_state": "ready", "nested": {"count": 2, "state": "ok"}}
    assert "actor_metadata:unsafe_key_omitted" in violations
    assert "actor_metadata:unsafe_value_omitted" in violations
    encoded = str(metadata)
    assert "raw user words" not in encoded
    assert "hidden prompt" not in encoded
    assert "private audio" not in encoded
    assert "data:image" not in encoded
    assert "a=candidate" not in encoded
    assert "Bearer secret" not in encoded
    assert "full message" not in encoded


def test_actor_event_schema_accepts_all_canonical_types_and_modes():
    for event_type in ActorEventTypeV2:
        payload = ActorEventV2(
            event_id=1,
            event_type=event_type,
            mode=ActorEventModeV2.WAITING,
            timestamp="2026-04-27T00:00:00+00:00",
            profile="browser-zh-melo",
            language="zh",
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
            vision_backend="moondream",
            source="test",
        ).as_dict()
        _assert_schema_valid(payload)

    for mode in ActorEventModeV2:
        payload = ActorEventV2(
            event_id=1,
            event_type=ActorEventTypeV2.WAITING,
            mode=mode,
            timestamp="2026-04-27T00:00:00+00:00",
            profile="browser-en-kokoro",
            language="en",
            tts_backend="kokoro",
            tts_label="kokoro/English",
            vision_backend="moondream",
            source="test",
        ).as_dict()
        _assert_schema_valid(payload)


def test_bilingual_primary_paths_share_actor_event_structure_except_labels():
    performance_event = BrowserPerformanceEvent(
        event_id=7,
        event_type="stt.transcription",
        source="stt",
        mode=BrowserInteractionMode.HEARD,
        timestamp="2026-04-27T00:00:00+00:00",
        session_id="session-primary",
        client_id="client-primary",
        metadata={"final_transcript_chars": 12, "topic_count": 1},
        reason_codes=["stt:transcribed"],
    )
    zh = actor_event_from_performance_event(
        performance_event,
        context=ActorEventContext(
            profile="browser-zh-melo",
            language="zh",
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
            vision_backend="moondream",
        ),
    ).as_dict()
    en = actor_event_from_performance_event(
        performance_event,
        context=ActorEventContext(
            profile="browser-en-kokoro",
            language="en",
            tts_backend="kokoro",
            tts_label="kokoro/English",
            vision_backend="moondream",
        ),
    ).as_dict()

    _assert_schema_valid(zh)
    _assert_schema_valid(en)
    assert set(zh) == set(en)
    assert zh | {
        "profile": en["profile"],
        "language": en["language"],
        "tts_backend": en["tts_backend"],
        "tts_label": en["tts_label"],
    } == en
    assert zh["event_type"] == en["event_type"] == "final_heard"
    assert zh["mode"] == en["mode"] == "heard"
    assert zh["vision_backend"] == en["vision_backend"] == "moondream"

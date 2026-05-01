from __future__ import annotations

import time

from blink.interaction.browser_runtime_session_v3 import BrowserRuntimeSessionV3


def _session(profile: str = "browser-en-kokoro") -> BrowserRuntimeSessionV3:
    return BrowserRuntimeSessionV3(
        profile=profile,
        language="en" if profile == "browser-en-kokoro" else "zh",
        tts_runtime_label=(
            "kokoro/English"
            if profile == "browser-en-kokoro"
            else "local-http-wav/MeloTTS"
        ),
        vision_enabled=True,
        continuous_perception_enabled=False,
        protected_playback=True,
    )


def test_primary_profile_defaults_are_preserved_in_session_core():
    zh = _session("browser-zh-melo").as_dict()
    en = _session("browser-en-kokoro").as_dict()

    assert zh["profile"] == "browser-zh-melo"
    assert zh["language"] == "zh"
    assert zh["tts_runtime_label"] == "local-http-wav/MeloTTS"
    assert en["profile"] == "browser-en-kokoro"
    assert en["language"] == "en"
    assert en["tts_runtime_label"] == "kokoro/English"
    for payload in (zh, en):
        assert payload["vision_enabled"] is True
        assert payload["continuous_perception_enabled"] is False
        assert payload["protected_playback"] is True


def test_stt_turn_state_resets_between_turns_and_keeps_only_bounded_live_text():
    session = _session()

    session.transcript.start_turn()
    first = session.transcript.note_final("first private sentence")
    session.transcript.stop_turn()
    session.transcript.start_turn()
    second = session.transcript.note_final("second private sentence")

    assert first["final_transcript_chars"] == len("first private sentence")
    assert second["final_transcript_chars"] == len("second private sentence")
    persistent = session.as_dict(include_live_text=False)
    assert "live_text" not in persistent["stt_turn"]
    assert persistent["stt_turn"]["live_text_hash"]
    assert "second private sentence" not in str(persistent)


def test_camera_truth_recovers_from_permission_denied_without_false_see_now_claim():
    session = _session()
    session.note_client_connected(session_id="session", client_id="client")
    session.note_client_media(
        {
            "mode": "unavailable",
            "camera_state": "permission_denied",
            "microphone_state": "permission_denied",
        }
    )
    assert session.camera.camera_honesty_state() == "unavailable"
    assert session.camera.current_answer_used_vision is False

    session.note_client_media(
        {
            "mode": "camera_and_microphone",
            "camera_state": "ready",
            "microphone_state": "receiving",
        }
    )
    assert session.camera.camera_honesty_state() == "available_not_used"

    session.note_camera_frame(frame_seq=3, frame_age_ms=42, received_monotonic=time.monotonic())
    assert session.camera.camera_honesty_state() == "recent_frame_available"

    session.note_vision_success(frame_seq=3, frame_age_ms=42)
    assert session.camera.camera_honesty_state() == "can_see_now"
    assert session.camera.as_dict()["current_answer_used_vision"] is True


def test_speech_queue_marks_interrupted_generation_stale_and_clears_backlog():
    session = _session()
    session.speech.start_generation(generation_id="speech-1", turn_id="turn-1")
    session.speech.note_subtitle_ready()
    session.speech.note_subtitle_ready()
    session.speech.note_lookahead_held(count=1)

    assert session.speech.can_emit() is False
    assert session.speech.as_dict()["held_speech_chunks"] == 1

    session.speech.note_interruption(dropped_count=2)
    payload = session.speech.as_dict()
    assert payload["generation_stale"] is True
    assert payload["held_speech_chunks"] == 0
    assert payload["speech_chunks_outstanding"] == 0
    assert payload["stale_chunk_drops"] == 2

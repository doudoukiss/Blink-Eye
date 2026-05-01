import asyncio
import json
import time

import pytest

from tests._optional import OPTIONAL_RUNTIME, require_optional_modules

pytestmark = OPTIONAL_RUNTIME
require_optional_modules("fastapi", "aiortc", "av")

from fastapi.testclient import TestClient

from blink.cli import local_browser
from blink.cli.local_browser import (
    BrowserPerformanceFrameObserver,
    LocalBrowserConfig,
    create_app,
)
from blink.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from blink.interaction import BrowserInteractionMode, BrowserPerformanceEventBus
from blink.interaction.active_listening import (
    BrowserActiveListeningPhase,
    BrowserActiveListeningSnapshot,
    extract_active_listening_understanding,
)


def _browser_config(*, profile: str) -> LocalBrowserConfig:
    if profile == "browser-en-kokoro":
        return LocalBrowserConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3.5:4b",
            system_prompt="English prompt",
            language=local_browser.Language.EN,
            stt_backend="mlx-whisper",
            tts_backend="kokoro",
            stt_model="mlx-community/whisper-medium-mlx",
            tts_voice="bf_emma",
            tts_base_url=None,
            host="127.0.0.1",
            port=7860,
            vision_enabled=False,
            continuous_perception_enabled=False,
            allow_barge_in=False,
            tts_runtime_label="kokoro/English",
            config_profile="browser-en-kokoro",
        )
    return LocalBrowserConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="Chinese prompt",
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


def test_active_listening_understanding_extracts_bounded_public_hints():
    understanding = extract_active_listening_understanding(
        "浏览器状态面板。不要暴露转写原文。必须同时支持 Kokoro。",
        language="zh",
    )

    payload = understanding.as_dict()

    assert payload["topic_count"] == 1
    assert payload["constraint_count"] == 2
    assert payload["topics"][0]["editable"] is True
    assert payload["topics"][0]["confidence"] == "heuristic"
    assert payload["constraints"][0]["source"] == "final_transcript"


def test_active_listening_snapshot_defaults_do_not_invent_partials():
    payload = BrowserActiveListeningSnapshot(
        available=True,
        phase=BrowserActiveListeningPhase.TRANSCRIBING,
        speech_start_count=1,
        speech_stop_count=1,
        reason_codes=("active_listening:v1", "active_listening:partial_unavailable"),
    ).as_dict()

    assert payload["phase"] == "transcribing"
    assert payload["partial_available"] is False
    assert payload["partial_transcript_chars"] == 0
    assert payload["topics"] == []
    assert "active_listening:partial_unavailable" in payload["reason_codes"]


def test_browser_performance_observer_emits_sanitized_active_listening_events():
    bus = BrowserPerformanceEventBus(max_events=20)
    pre = BrowserPerformanceFrameObserver(
        phase="pre_stt",
        performance_emit=bus.emit,
        resting_mode_provider=lambda: BrowserInteractionMode.LISTENING,
    )
    post = BrowserPerformanceFrameObserver(
        phase="post_stt",
        performance_emit=bus.emit,
        resting_mode_provider=lambda: BrowserInteractionMode.LISTENING,
    )

    asyncio.run(pre._record_frame(VADUserStartedSpeakingFrame()))
    pre._voice_turn_started_at = time.monotonic() - 1.2
    pre._last_speech_continuing_emit_at = time.monotonic() - 1.2
    asyncio.run(
        pre._record_frame(AudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1))
    )
    asyncio.run(pre._record_frame(VADUserStoppedSpeakingFrame()))
    asyncio.run(
        post._record_frame(InterimTranscriptionFrame("partial secret", user_id="u1", timestamp="t1"))
    )
    asyncio.run(
        post._record_frame(
            TranscriptionFrame(
                "浏览器状态面板。不要暴露转写原文。必须支持 Kokoro。",
                user_id="u1",
                timestamp="t2",
            )
        )
    )

    payload = bus.as_payload(limit=20)
    event_types = [event["event_type"] for event in payload["events"]]
    encoded = json.dumps(payload, ensure_ascii=False)

    assert event_types == [
        "voice.speech_started",
        "active_listening.listening_started",
        "voice.speech_continuing",
        "voice.speech_stopped",
        "stt.transcribing",
        "stt.partial_transcription",
        "active_listening.partial_understanding_updated",
        "stt.transcription",
        "user_turn.summary",
        "active_listening.final_understanding_ready",
    ]
    assert payload["events"][5]["metadata"]["partial_transcript_chars"] == len("partial secret")
    assert payload["events"][6]["metadata"]["partial_available"] is True
    assert payload["events"][6]["metadata"]["semantic_listener"]["schema_version"] == 3
    assert payload["events"][6]["metadata"]["semantic_listener"]["summary_hash"]
    assert payload["events"][-1]["metadata"]["ready_to_answer"] is True
    assert payload["events"][-1]["metadata"]["semantic_listener"]["summary_hash"]
    assert payload["events"][-1]["metadata"]["semantic_listener"]["detected_intent"] in {
        "instruction",
        "project_planning",
    }
    assert "safe_live_summary" not in payload["events"][-1]["metadata"]["semantic_listener"]
    assert payload["events"][8]["metadata"]["topic_count"] == 1
    assert payload["events"][8]["metadata"]["constraint_count"] == 2
    actor_events = [event.as_dict() for event in bus.actor_recent(limit=20)]
    actor_types = [event["event_type"] for event in actor_events]
    assert "listening_started" in actor_types
    assert "partial_understanding_updated" in actor_types
    assert "final_understanding_ready" in actor_types
    final_actor_event = next(
        event for event in actor_events if event["event_type"] == "final_understanding_ready"
    )
    assert isinstance(
        final_actor_event["metadata"]["semantic_listener"]["listener_chip_ids"],
        list,
    )
    assert "ready_to_answer" in final_actor_event["metadata"]["semantic_listener"][
        "listener_chip_ids"
    ]
    assert "partial secret" not in encoded
    assert "不要暴露转写原文" not in encoded


def test_browser_performance_observer_accumulates_final_only_stt_fragments():
    bus = BrowserPerformanceEventBus(max_events=20)
    post = BrowserPerformanceFrameObserver(
        phase="post_stt",
        performance_emit=bus.emit,
        resting_mode_provider=lambda: BrowserInteractionMode.LISTENING,
    )
    first = "Camera honesty is important."
    second = "Kokoro should finish long speech."

    asyncio.run(post._record_frame(TranscriptionFrame(first, user_id="u1", timestamp="t1")))
    asyncio.run(post._record_frame(TranscriptionFrame(second, user_id="u1", timestamp="t2")))

    payload = bus.as_payload(limit=20)
    final_events = [
        event
        for event in payload["events"]
        if event["event_type"] == "active_listening.final_understanding_ready"
    ]
    stt_events = [
        event for event in payload["events"] if event["event_type"] == "stt.transcription"
    ]
    encoded = json.dumps(payload, ensure_ascii=False)

    assert stt_events[0]["metadata"]["final_transcript_chars"] == len(first)
    assert stt_events[1]["metadata"]["final_fragment_chars"] == len(second)
    assert stt_events[1]["metadata"]["final_transcript_chars"] == len(first) + len(second)
    assert final_events[-1]["metadata"]["partial_available"] is False
    assert final_events[-1]["metadata"]["final_transcript_chars"] == len(first) + len(second)
    assert first not in encoded
    assert second not in encoded


def test_browser_performance_observer_emits_active_listening_degraded_safely():
    bus = BrowserPerformanceEventBus(max_events=20)
    post = BrowserPerformanceFrameObserver(
        phase="post_stt",
        performance_emit=bus.emit,
        resting_mode_provider=lambda: BrowserInteractionMode.LISTENING,
    )

    asyncio.run(post._record_frame(ErrorFrame("raw private stt failure")))

    payload = bus.as_payload(limit=20)
    event_types = [event["event_type"] for event in payload["events"]]
    encoded = json.dumps(payload, ensure_ascii=False)
    actor_types = [event.as_dict()["event_type"] for event in bus.actor_recent(limit=20)]

    assert event_types == ["stt.error", "active_listening.listening_degraded"]
    assert payload["events"][-1]["metadata"]["degradation_state"] == "error"
    assert "listening_degraded" in actor_types
    assert "raw private stt failure" not in encoded


@pytest.mark.parametrize("profile", ["browser-zh-melo", "browser-en-kokoro"])
def test_primary_browser_profiles_expose_active_listening_defaults(profile: str):
    app, _ = create_app(_browser_config(profile=profile))
    client = TestClient(app)

    payload = client.get("/api/runtime/performance-state").json()
    active = payload["active_listening"]
    actor_payload = client.get("/api/runtime/actor-state").json()
    actor_active = actor_payload["active_listening"]

    assert payload["profile"] == profile
    assert active["schema_version"] == 1
    assert active["phase"] == "idle"
    assert active["partial_available"] is False
    assert active["topics"] == []
    assert active["constraints"] == []
    assert actor_payload["profile"] == profile
    assert actor_active["schema_version"] == 2
    assert actor_active["profile"] == profile
    assert actor_active["phase"] == "idle"
    assert actor_active["semantic_state_v3"]["schema_version"] == 3
    assert actor_active["semantic_state_v3"]["language"] == (
        "en" if profile == "browser-en-kokoro" else "zh"
    )
    assert "still_listening" in [
        chip["chip_id"] for chip in actor_active["semantic_state_v3"]["listener_chips"]
    ]
    assert actor_active["topics"] == []
    assert actor_active["constraints"] == []
    assert actor_active["corrections"] == []
    assert actor_active["project_references"] == []

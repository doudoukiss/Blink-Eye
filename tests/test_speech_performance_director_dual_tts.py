import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.brain.persona import BrainExpressionVoicePolicy
from blink.brain.processors import BrainExpressionVoicePolicyProcessor
from blink.brain.speech_director import (
    KOKORO_HARD_MAX_CHARS,
    MELO_HARD_MAX_CHARS,
    SpeechChunkBudgetV3,
    SpeechPerformanceChunk,
    build_speech_chunk_frame_metadata,
    next_kokoro_speech_chunk,
    next_melo_speech_chunk,
)
from blink.cli.local_browser import (
    Language,
    LocalBrowserConfig,
    _speech_director_mode_for_config,
)
from blink.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from blink.interaction import (
    BrowserInterruptedOutputGuardProcessor,
    BrowserInterruptionStateTracker,
    BrowserPerformanceEventBus,
    SpeechQueueController,
)
from blink.processors.frame_processor import FrameDirection
from blink.tests.utils import run_test

SCHEMA = Path("schemas/speech_performance_chunk.schema.json")


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))


def _policy() -> BrainExpressionVoicePolicy:
    return BrainExpressionVoicePolicy(
        available=True,
        modality="browser",
        concise_chunking_active=False,
        chunking_mode="off",
        max_spoken_chunk_chars=0,
        interruption_strategy_label="protected",
        pause_yield_hint="metadata only",
        active_hints=(),
        unsupported_hints=("pause_timing",),
        noop_reason_codes=("voice_policy_noop:pause_timing_metadata_only",),
        expression_controls_hardware=False,
        reason_codes=("voice_policy:available",),
    )


class CaptureSpeechPolicyProcessor(BrainExpressionVoicePolicyProcessor):
    def __init__(
        self,
        *,
        tts_backend: str,
        language: str,
        speech_director_mode: str,
        performance_emit=None,
        performance_plan_provider=None,
        speech_queue_controller=None,
        observations: list[tuple[str, object]] | None = None,
    ):
        super().__init__(
            policy_provider=_policy,
            tts_backend=tts_backend,
            language=language,
            speech_director_mode=speech_director_mode,
            performance_emit=performance_emit,
            performance_plan_provider=performance_plan_provider,
            speech_queue_controller=speech_queue_controller,
            enable_direct_mode=True,
        )
        self.pushed = []
        self.observations = observations if observations is not None else []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.observations.append(("push", frame))
        self.pushed.append((frame, direction))


def _browser_config(*, profile: str, language: Language, tts_backend: str) -> LocalBrowserConfig:
    return LocalBrowserConfig(
        base_url=None,
        model="qwen3.5:4b",
        system_prompt="local",
        language=language,
        stt_backend="whisper",
        tts_backend=tts_backend,
        stt_model="base",
        tts_voice=None,
        tts_base_url=None,
        host="127.0.0.1",
        port=7860,
        tts_runtime_label="local-http-wav/MeloTTS"
        if profile == "browser-zh-melo"
        else "kokoro/English",
        config_profile=profile,
        vision_enabled=True,
    )


def _event_types(bus: BrowserPerformanceEventBus) -> list[str]:
    return [event["event_type"] for event in bus.as_payload(limit=50)["events"]]


def _assert_conservative_chunk_capabilities(capabilities: dict[str, object]) -> None:
    assert capabilities["supports_chunk_boundaries"] is True
    assert capabilities["supports_interruption_flush"] is True
    assert capabilities["supports_interruption_discard"] is False
    assert capabilities["supports_pause_timing"] is False
    assert capabilities["supports_speech_rate"] is False
    assert capabilities["supports_prosody_emphasis"] is False
    assert capabilities["supports_partial_stream_abort"] is False
    assert capabilities["expression_controls_hardware"] is False


def test_speech_performance_chunk_matches_schema_and_keeps_legacy_aliases():
    chunk = SpeechPerformanceChunk(
        role="assistant",
        text="Hello.",
        language="en",
        tts_backend="kokoro",
        display_text="Hello.",
        pause_after_ms=180,
        interruptible=True,
        context_id="ctx-1",
        generation_token="speech-1",
        turn_id="turn-1",
        chunk_index=1,
        reason_codes=("speech_director:kokoro_sentence",),
    )

    errors = list(_validator().iter_errors(chunk.as_dict()))
    metadata = chunk.public_metadata(queue_depth=2)

    assert errors == []
    assert chunk.speech_director_version == 3
    assert chunk.speak_text == "Hello."
    assert chunk.generation_id == "speech-1"
    assert chunk.stale_generation_token == "speech-1"
    assert chunk.estimated_duration_ms > 0
    assert chunk.subtitle_timing["emit_policy"] == "before_or_at_playback_start"
    _assert_conservative_chunk_capabilities(dict(chunk.backend_capabilities))
    assert metadata["chunk_id"] == chunk.chunk_id
    assert metadata["language"] == "en"
    assert metadata["tts_backend"] == "kokoro"
    assert metadata["speech_director_version"] == 3
    assert metadata["stale_generation_token"] == "speech-1"
    assert metadata["estimated_duration_ms"] == chunk.estimated_duration_ms
    assert metadata["subtitle_timing"] == chunk.subtitle_timing
    _assert_conservative_chunk_capabilities(dict(metadata["backend_capabilities"]))
    assert metadata["display_chars"] == 6
    assert metadata["speak_chars"] == 6
    assert "Hello" not in json.dumps(metadata)


def test_dual_primary_profile_modes_are_equal_public_contracts():
    zh_config = _browser_config(
        profile="browser-zh-melo",
        language=Language.ZH,
        tts_backend="local-http-wav",
    )
    en_config = _browser_config(
        profile="browser-en-kokoro",
        language=Language.EN,
        tts_backend="kokoro",
    )
    zh = SpeechPerformanceChunk(
        role="assistant",
        text="你好。",
        language="zh",
        tts_backend="local-http-wav",
        display_text="你好。",
        pause_after_ms=180,
        interruptible=True,
        context_id=None,
        generation_token="speech-zh",
        turn_id="turn-1",
        chunk_index=1,
    )
    en = SpeechPerformanceChunk(
        role="assistant",
        text="Hello.",
        language="en",
        tts_backend="kokoro",
        display_text="Hello.",
        pause_after_ms=180,
        interruptible=True,
        context_id=None,
        generation_token="speech-en",
        turn_id="turn-1",
        chunk_index=1,
    )

    assert _speech_director_mode_for_config(zh_config) == "melo_chunked"
    assert _speech_director_mode_for_config(en_config) == "kokoro_chunked"
    assert set(zh.as_dict()) == set(en.as_dict())
    assert set(zh.public_metadata()) == set(en.public_metadata())
    assert zh.as_dict()["speech_director_version"] == en.as_dict()["speech_director_version"] == 3
    assert zh.as_dict()["subtitle_timing"] == en.as_dict()["subtitle_timing"]
    assert dict(zh.backend_capabilities)["backend_label"] == "local-http-wav"
    assert dict(en.backend_capabilities)["backend_label"] == "kokoro"
    _assert_conservative_chunk_capabilities(dict(zh.backend_capabilities))
    _assert_conservative_chunk_capabilities(dict(en.backend_capabilities))


def test_speech_chunk_budget_v3_clamps_plan_budget_by_backend_defaults():
    melo = SpeechChunkBudgetV3.from_plan(
        director_mode="melo_chunked",
        tts_backend="local-http-wav",
        actuation_chunk_limit=999,
        plan_budget={
            "target_chars": 90,
            "hard_max_chars": 120,
            "max_chunks_per_flush": 20,
        },
    )
    kokoro = SpeechChunkBudgetV3.from_plan(
        director_mode="kokoro_chunked",
        tts_backend="kokoro",
        actuation_chunk_limit=999,
        plan_budget={
            "target_chars": 70,
            "hard_max_chars": 90,
            "max_chunks_per_flush": 2,
        },
    )

    assert melo.hard_max_chars == 120
    assert melo.target_chars == 90
    assert melo.min_chars == 45
    assert melo.max_chunks_per_flush == 12
    assert kokoro.hard_max_chars == 90
    assert kokoro.target_chars == 70
    assert kokoro.min_chars == 35
    assert kokoro.max_chunks_per_flush == 2


def test_melo_long_answer_chunks_on_chinese_boundaries_within_bounds():
    text = "第一段先确认麦克风和模型状态，然后再判断语音输出是否稳定。" * 10

    chunk, remaining, pause_ms, reason_codes = next_melo_speech_chunk(text)

    assert chunk is not None
    assert remaining
    assert len(chunk) <= MELO_HARD_MAX_CHARS
    assert chunk.endswith("。")
    assert pause_ms == 180
    assert "speech_director:strong_boundary" in reason_codes


def test_kokoro_long_answer_chunks_balanced_sentences_and_short_reply_stays_intact():
    long_text = (
        "First I will check the microphone state and make sure the browser session is ready. "
        "Then I will explain the visible WebRTC and TTS signals in plain English. "
        "Finally I will keep the answer short enough for natural playback."
    )
    short_text = "Sure, I can do that."

    first, remaining, pause_ms, reason_codes = next_kokoro_speech_chunk(long_text)
    short, short_remaining, _short_pause, short_reasons = next_kokoro_speech_chunk(
        short_text,
        force=True,
    )

    assert first is not None
    assert remaining
    assert len(first) <= KOKORO_HARD_MAX_CHARS
    assert first.endswith(".")
    assert pause_ms == 180
    assert any(reason.startswith("speech_director:kokoro") for reason in reason_codes)
    assert short == short_text
    assert short_remaining == ""
    assert "speech_director:kokoro_short_reply" in short_reasons


@pytest.mark.asyncio
async def test_kokoro_subtitle_ready_event_precedes_tts_bound_chunk_and_omits_raw_text():
    bus = BrowserPerformanceEventBus(max_events=20)
    observations: list[tuple[str, object]] = []

    def emit_with_order(**kwargs):
        observations.append(("event", kwargs["event_type"]))
        return bus.emit(**kwargs)

    processor = CaptureSpeechPolicyProcessor(
        tts_backend="kokoro",
        language="en",
        speech_director_mode="kokoro_chunked",
        performance_emit=emit_with_order,
        observations=observations,
    )
    text = (
        "First I will check the browser microphone state. "
        "Then I will keep this answer in subtitle ready chunks."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    subtitle_index = observations.index(("event", "speech.subtitle_ready"))
    chunk_index = next(
        index
        for index, (_kind, item) in enumerate(observations)
        if isinstance(item, AggregatedTextFrame)
    )
    encoded_events = json.dumps(bus.as_payload(limit=20), ensure_ascii=False)

    assert subtitle_index < chunk_index
    assert "speech.subtitle_ready" in _event_types(bus)
    assert text not in encoded_events
    assert "display_chars" in encoded_events
    assert "estimated_duration_ms" in encoded_events
    assert "before_or_at_playback_start" in encoded_events
    assert "supports_speech_rate" in encoded_events
    assert "prosody_emphasis" in encoded_events
    assert "Hello" not in encoded_events
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    assert chunks
    assert all(getattr(frame, "blink_subtitle_immediate", False) for frame in chunks)
    assert all(getattr(frame, "blink_estimated_duration_ms", 0) > 0 for frame in chunks)
    assert all(
        getattr(frame, "blink_subtitle_timing", {}).get("emit_policy")
        == "before_or_at_playback_start"
        for frame in chunks
    )


@pytest.mark.asyncio
async def test_session_speech_queue_refills_one_slot_on_tts_boundary():
    speech_queue = SpeechQueueController(max_speech_chunk_lookahead=2, max_subtitle_lookahead=2)
    processor = CaptureSpeechPolicyProcessor(
        tts_backend="kokoro",
        language="en",
        speech_director_mode="kokoro_chunked",
        speech_queue_controller=speech_queue,
    )
    text = (
        "First sentence is ready and should play as the first bounded chunk. "
        "Second sentence is ready and should play as the second bounded chunk. "
        "Third sentence should wait until one TTS queue slot becomes available. "
        "Fourth sentence proves long answers keep draining instead of getting stuck."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]

    assert len(chunks) == 2
    assert speech_queue.as_dict()["held_speech_chunks"] == 1
    assert speech_queue.as_dict()["speech_chunks_outstanding"] == 2

    speech_queue.note_tts_stopped()
    await processor.drain_held_speech_chunks()
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]

    assert len(chunks) == 3
    assert speech_queue.as_dict()["speech_chunks_outstanding"] == 2


@pytest.mark.asyncio
async def test_performance_plan_v3_budget_shortens_kokoro_chunks_without_fake_tts_controls():
    bus = BrowserPerformanceEventBus(max_events=20)
    plan = {
        "schema_version": 3,
        "speech_chunk_budget": {
            "state": "repair_short",
            "target_chars": 70,
            "hard_max_chars": 90,
            "max_chunks_per_flush": 1,
            "reason_codes": ["speech_chunk_budget_v3:repair_short"],
        },
        "tts_capabilities": {
            "speech_rate_enabled": False,
            "prosody_emphasis_enabled": False,
            "expression_controls_hardware": False,
        },
    }
    processor = CaptureSpeechPolicyProcessor(
        tts_backend="kokoro",
        language="en",
        speech_director_mode="kokoro_chunked",
        performance_emit=bus.emit,
        performance_plan_provider=lambda: plan,
    )
    text = (
        "First repair sentence has enough words to become a bounded chunk. "
        "Second repair sentence should stay buffered behind the plan budget. "
        "Third repair sentence proves the processor did not enqueue a backlog."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)

    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    payload = bus.as_payload(limit=20)

    assert len(chunks) == 1
    assert len(chunks[0].text) <= 90
    assert any(event["event_type"] == "speech.plan_chunk_budget_held" for event in payload["events"])
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "speech_rate_enabled" not in encoded
    assert "prosody_emphasis_enabled" not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "tts_backend", "text"),
    [
        ("zh", "local-http-wav", "旧的回答不应该继续说。"),
        ("en", "kokoro", "The old answer should not keep playing."),
    ],
)
async def test_stale_melo_and_kokoro_chunks_drop_after_interruption(language, tts_backend, text):
    bus = BrowserPerformanceEventBus(max_events=20)
    state = BrowserInterruptionStateTracker(
        protected_playback=False,
        performance_emit=bus.emit,
    )
    guard = BrowserInterruptedOutputGuardProcessor(
        interruption_state=state,
        performance_emit=bus.emit,
    )
    stale = AggregatedTextFrame(text, AggregationType.SENTENCE)
    build_speech_chunk_frame_metadata(
        stale,
        SpeechPerformanceChunk(
            role="assistant",
            text=text,
            language=language,
            tts_backend=tts_backend,
            display_text=text,
            pause_after_ms=180,
            interruptible=True,
            context_id=None,
            generation_token="speech-old",
            turn_id="turn-old",
            chunk_index=1,
        ),
    )

    down_frames, _ = await run_test(guard, frames_to_send=[InterruptionFrame(), stale])
    encoded_events = json.dumps(bus.as_payload(limit=20), ensure_ascii=False)

    assert [frame.__class__ for frame in down_frames] == [InterruptionFrame]
    assert "speech.chunk_stale_dropped" in encoded_events
    assert "interruption.output_dropped" in encoded_events
    assert "interruption.output_flushed" in encoded_events
    assert text not in encoded_events

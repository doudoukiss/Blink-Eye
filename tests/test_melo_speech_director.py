import json

import pytest

from blink.brain.persona import BrainExpressionVoicePolicy
from blink.brain.processors import BrainExpressionVoicePolicyProcessor
from blink.brain.speech_director import (
    MELO_HARD_MAX_CHARS,
    BrainSpeechChunk,
    build_speech_chunk_frame_metadata,
    next_melo_speech_chunk,
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
)
from blink.processors.frame_processor import FrameDirection
from blink.processors.frameworks.rtvi.observer import RTVIObserver
from blink.tests.utils import run_test


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
    def __init__(self, *, performance_emit=None):
        super().__init__(
            policy_provider=_policy,
            tts_backend="local-http-wav",
            speech_director_mode="melo_chunked",
            performance_emit=performance_emit,
            enable_direct_mode=True,
        )
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def test_brain_speech_chunk_public_metadata_is_bounded():
    chunk = BrainSpeechChunk(
        role="assistant",
        display_text="你好。",
        speak_text="你好。",
        language="zh",
        tts_backend="local-http-wav",
        interruptible=True,
        pause_after_ms=180,
        generation_id="speech-1",
        turn_id="turn-1",
        chunk_index=1,
        reason_codes=("speech_director:strong_boundary",),
    )

    payload = chunk.public_metadata(queue_depth=2)

    assert payload["role"] == "assistant"
    assert payload["display_chars"] == 3
    assert payload["speak_chars"] == 3
    assert payload["speech_director_version"] == 3
    assert payload["estimated_duration_ms"] > 0
    assert payload["subtitle_timing"]["emit_policy"] == "before_or_at_playback_start"
    assert payload["backend_capabilities"]["backend_label"] == "local-http-wav"
    assert payload["backend_capabilities"]["supports_chunk_boundaries"] is True
    assert payload["backend_capabilities"]["supports_interruption_flush"] is True
    assert payload["backend_capabilities"]["supports_speech_rate"] is False
    assert payload["backend_capabilities"]["supports_prosody_emphasis"] is False
    assert payload["generation_id"] == "speech-1"
    assert payload["stale_generation_token"] == "speech-1"
    assert payload["turn_id"] == "turn-1"
    assert payload["chunk_index"] == 1
    assert payload["queue_depth"] == 2
    assert "你好" not in json.dumps(payload, ensure_ascii=False)


def test_melo_chunking_uses_natural_boundaries_and_hard_max():
    text = "第一段先确认麦克风和模型状态，然后再判断语音输出是否稳定。" * 10

    chunk, remaining, pause_ms, reason_codes = next_melo_speech_chunk(text)
    hard_chunk, _remaining, _pause_ms, hard_reason_codes = next_melo_speech_chunk("测" * 260)

    assert chunk is not None
    assert remaining
    assert len(chunk) <= MELO_HARD_MAX_CHARS
    assert chunk.endswith("。")
    assert pause_ms == 180
    assert "speech_director:strong_boundary" in reason_codes
    assert hard_chunk == "测" * MELO_HARD_MAX_CHARS
    assert "speech_director:hard_boundary" in hard_reason_codes


@pytest.mark.asyncio
async def test_melo_speech_director_emits_immediate_subtitle_metadata_without_raw_event_text():
    bus = BrowserPerformanceEventBus(max_events=20)
    processor = CaptureSpeechPolicyProcessor(performance_emit=bus.emit)
    text = "第一段先确认麦克风和模型状态，然后再判断语音输出是否稳定。" * 10

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    encoded_events = json.dumps(bus.as_payload(limit=20), ensure_ascii=False)

    assert chunks
    assert all(getattr(frame, "blink_subtitle_immediate", False) is True for frame in chunks)
    assert all(getattr(frame, "blink_display_text", "") == frame.text for frame in chunks)
    assert "speech.generation_start" in encoded_events
    assert "speech.subtitle_ready" in encoded_events
    assert text not in encoded_events
    assert "display_chars" in encoded_events
    assert "speak_chars" in encoded_events
    assert "estimated_duration_ms" in encoded_events
    assert "before_or_at_playback_start" in encoded_events
    assert "supports_speech_rate" in encoded_events


@pytest.mark.asyncio
async def test_rtvi_observer_sends_immediate_subtitle_before_bot_speaks():
    sent = []
    observer = RTVIObserver()

    async def capture(message, exclude_none=True):
        sent.append(message)

    observer.send_rtvi_message = capture
    frame = AggregatedTextFrame("马上显示字幕。", AggregationType.SENTENCE)
    build_speech_chunk_frame_metadata(
        frame,
        BrainSpeechChunk(
            role="assistant",
            display_text="马上显示字幕。",
            speak_text="马上显示字幕。",
            interruptible=True,
            pause_after_ms=180,
            generation_id="speech-1",
            turn_id="turn-1",
            chunk_index=1,
        ),
    )

    await observer._handle_aggregated_llm_text(frame)

    assert [message.type for message in sent] == ["bot-output"]
    assert sent[0].data.text == "马上显示字幕。"


@pytest.mark.asyncio
async def test_speech_chunk_guard_reports_stale_drop_without_raw_text():
    bus = BrowserPerformanceEventBus(max_events=20)
    state = BrowserInterruptionStateTracker(protected_playback=False, performance_emit=bus.emit)
    guard = BrowserInterruptedOutputGuardProcessor(
        interruption_state=state,
        performance_emit=bus.emit,
    )
    stale = AggregatedTextFrame("旧的回答不应该继续说。", AggregationType.SENTENCE)
    build_speech_chunk_frame_metadata(
        stale,
        BrainSpeechChunk(
            role="assistant",
            display_text="旧的回答不应该继续说。",
            speak_text="旧的回答不应该继续说。",
            interruptible=True,
            pause_after_ms=180,
            generation_id="speech-old",
            turn_id="turn-old",
            chunk_index=1,
        ),
    )

    down_frames, _ = await run_test(guard, frames_to_send=[InterruptionFrame(), stale])
    encoded_events = json.dumps(bus.as_payload(limit=20), ensure_ascii=False)

    assert [frame.__class__ for frame in down_frames] == [InterruptionFrame]
    assert "speech.chunk_stale_dropped" in encoded_events
    assert "旧的回答" not in encoded_events

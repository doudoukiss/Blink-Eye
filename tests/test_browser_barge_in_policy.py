import asyncio
import json

import pytest

from blink.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from blink.interaction import (
    BrowserBargeInTurnStartStrategy,
    BrowserInterruptedOutputGuardProcessor,
    BrowserInterruptionStateTracker,
    BrowserPerformanceEventBus,
    BrowserProtectedPlaybackMuteStrategy,
)
from blink.tests.utils import run_test
from blink.utils.asyncio.task_manager import TaskManager, TaskManagerParams


def _tracker(*, protected_playback: bool = False):
    bus = BrowserPerformanceEventBus(max_events=20)
    return BrowserInterruptionStateTracker(
        protected_playback=protected_playback,
        performance_emit=bus.emit,
    ), bus


async def _setup_strategy(strategy):
    manager = TaskManager()
    manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    await strategy.setup(manager)
    return manager


def _event_types(bus: BrowserPerformanceEventBus) -> list[str]:
    return [event.event_type for event in bus.recent(limit=20)]


def _encoded_payload(bus: BrowserPerformanceEventBus) -> str:
    return json.dumps(bus.as_payload(limit=20), ensure_ascii=False)


@pytest.mark.asyncio
async def test_protected_playback_suppresses_bot_speech_interruption():
    state, bus = _tracker(protected_playback=True)
    strategy = BrowserProtectedPlaybackMuteStrategy(interruption_state=state)

    assert await strategy.process_frame(BotStartedSpeakingFrame()) is True
    assert await strategy.process_frame(VADUserStartedSpeakingFrame()) is True
    await strategy.process_frame(BotStoppedSpeakingFrame())

    assert _event_types(bus) == [
        "interruption.suppressed",
        "interruption.listening_resumed",
    ]
    snapshot = state.snapshot()
    assert snapshot["barge_in_state"] == "protected"
    assert snapshot["last_decision"] == "listening_resumed"


@pytest.mark.asyncio
async def test_explicit_barge_in_accepts_sustained_speech():
    state, bus = _tracker(protected_playback=False)
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.01,
    )
    starts = []
    strategy.add_event_handler("on_user_turn_started", lambda _strategy, params: starts.append(params))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    result = await strategy.process_frame(VADUserStartedSpeakingFrame())
    await asyncio.sleep(0.03)
    await strategy.cleanup()

    assert result.value == "stop"
    assert len(starts) == 1
    assert _event_types(bus) == ["interruption.candidate", "interruption.accepted"]
    assert state.snapshot()["last_decision"] == "accepted"


@pytest.mark.asyncio
async def test_explicit_barge_in_rejects_brief_vad_candidate():
    state, bus = _tracker(protected_playback=False)
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.05,
    )
    starts = []
    strategy.add_event_handler("on_user_turn_started", lambda _strategy, params: starts.append(params))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(VADUserStartedSpeakingFrame())
    await strategy.process_frame(VADUserStoppedSpeakingFrame())
    await asyncio.sleep(0.07)
    await strategy.cleanup()

    assert starts == []
    assert _event_types(bus) == ["interruption.candidate", "interruption.rejected"]
    assert state.snapshot()["last_reason"] == "brief_speech_or_cough"


@pytest.mark.asyncio
async def test_explicit_barge_in_rejects_short_backchannel_without_raw_transcript():
    state, bus = _tracker(protected_playback=False)
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.05,
    )
    starts = []
    resets = []
    strategy.add_event_handler("on_user_turn_started", lambda _strategy, params: starts.append(params))
    strategy.add_event_handler("on_reset_aggregation", lambda _strategy: resets.append("reset"))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(
        TranscriptionFrame(text="嗯", user_id="local", timestamp="2026-04-26T00:00:00Z")
    )
    await strategy.cleanup()

    assert starts == []
    assert resets == ["reset"]
    assert _event_types(bus) == ["interruption.rejected"]
    encoded = _encoded_payload(bus)
    assert "嗯" not in encoded
    assert "raw_user_words" not in encoded


@pytest.mark.asyncio
async def test_explicit_barge_in_accepts_meaningful_transcript_without_raw_text():
    state, bus = _tracker(protected_playback=False)
    strategy = BrowserBargeInTurnStartStrategy(
        interruption_state=state,
        sustain_threshold_secs=0.05,
    )
    starts = []
    strategy.add_event_handler("on_user_turn_started", lambda _strategy, params: starts.append(params))
    await _setup_strategy(strategy)

    await strategy.process_frame(BotStartedSpeakingFrame())
    await strategy.process_frame(
        TranscriptionFrame(text="请停一下", user_id="local", timestamp="2026-04-26T00:00:00Z")
    )
    await strategy.cleanup()

    assert len(starts) == 1
    assert _event_types(bus) == ["interruption.accepted"]
    encoded = _encoded_payload(bus)
    assert "请停一下" not in encoded
    assert "transcription_chars" in encoded


@pytest.mark.asyncio
async def test_output_guard_drops_stale_text_and_tts_until_next_response():
    state, bus = _tracker(protected_playback=False)
    guard = BrowserInterruptedOutputGuardProcessor(interruption_state=state)

    down_frames, _ = await run_test(
        guard,
        frames_to_send=[
            InterruptionFrame(),
            TextFrame(text="stale assistant text"),
            TTSSpeakFrame(text="stale tool speech"),
            LLMFullResponseStartFrame(),
            TextFrame(text="fresh assistant text"),
        ],
    )

    assert [frame.__class__ for frame in down_frames] == [
        InterruptionFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    ]
    assert _event_types(bus) == [
        "interruption.output_dropped",
        "interruption.output_flushed",
        "interruption.output_dropped",
        "interruption.output_flushed",
    ]
    encoded = _encoded_payload(bus)
    assert "stale assistant text" not in encoded
    assert "stale tool speech" not in encoded

import json

import pytest

from blink.brain.autonomy import BrainReevaluationConditionKind
from blink.brain.memory import BrainMemoryConsolidator
from blink.brain.processors import (
    BrainEventRecorderProcessor,
    HotPathMemoryExtractor,
    TurnRecorderProcessor,
    latest_turn_tool_calls_from_context,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.frames.frames import BotStoppedSpeakingFrame, LLMContextFrame, UserStoppedSpeakingFrame
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection
from blink.transcriptions.language import Language


def _weather_turn_messages():
    return [
        {"role": "user", "content": "天气如何？"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "上海"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"conditions": "Sunny"}',
        },
        {"role": "assistant", "content": "上海今天晴。"},
    ]


def test_latest_turn_tool_calls_from_context_extracts_structured_calls():
    context = LLMContext(messages=_weather_turn_messages())

    tool_calls = latest_turn_tool_calls_from_context(context)

    assert tool_calls == [
        {
            "tool_call_id": "call_1",
            "function_name": "get_weather",
            "arguments": {"location": "上海"},
            "result": {"conditions": "Sunny"},
        }
    ]


@pytest.mark.asyncio
async def test_brain_event_recorder_triggers_user_turn_close_reevaluation_once(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    seen_triggers = []
    seen_wake_boundaries = []

    class StubExecutive:
        def run_presence_director_reevaluation(self, trigger):
            seen_triggers.append(trigger)
            return None

        async def run_commitment_wake_router(self, *, boundary_kind: str, source_event=None):
            seen_wake_boundaries.append((boundary_kind, source_event.event_type if source_event else None))
            return None

    processor = BrainEventRecorderProcessor(
        store=store,
        session_resolver=lambda: session_ids,
        executive=StubExecutive(),
    )

    await processor.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )
    assert len(seen_triggers) == 1
    assert seen_wake_boundaries == [("user_turn_end", "user.turn.ended")]
    assert seen_triggers[0].kind == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    assert seen_triggers[0].source_event_type == "user.turn.ended"
    assert events[0].event_type == "user.turn.ended"


@pytest.mark.asyncio
async def test_brain_event_recorder_fails_open_when_store_is_locked():
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")

    class LockedStore:
        def append_brain_event(self, *args, **kwargs):
            raise RuntimeError("database is locked")

    class CaptureProcessor(BrainEventRecorderProcessor):
        def __init__(self):
            super().__init__(store=LockedStore(), session_resolver=lambda: session_ids)
            self.pushed = []

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed.append((frame, direction))

    processor = CaptureProcessor()
    frame = UserStoppedSpeakingFrame()

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [(frame, FrameDirection.DOWNSTREAM)]


@pytest.mark.asyncio
async def test_turn_recorder_processor_fails_open_when_store_is_locked():
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    context = LLMContext(messages=_weather_turn_messages())

    class LockedStore:
        def latest_brain_event(self, *args, **kwargs):
            raise RuntimeError("database is locked")

    class CaptureProcessor(TurnRecorderProcessor):
        def __init__(self):
            super().__init__(
                store=LockedStore(),
                session_resolver=lambda: session_ids,
                context=context,
                consolidator=None,
            )
            self.pushed = []

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed.append((frame, direction))

    processor = CaptureProcessor()
    frame = BotStoppedSpeakingFrame()

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [(frame, FrameDirection.DOWNSTREAM)]


@pytest.mark.asyncio
async def test_hot_path_memory_extractor_fails_open_and_avoids_retry_storm_on_store_lock():
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    context = LLMContext(messages=[{"role": "user", "content": "Please remember concise answers."}])
    frame = LLMContextFrame(context=context)

    class LockedStore:
        def __init__(self):
            self.ensure_calls = 0

        def ensure_user(self, *args, **kwargs):
            self.ensure_calls += 1
            raise RuntimeError("database is locked")

    class CaptureProcessor(HotPathMemoryExtractor):
        def __init__(self, store):
            super().__init__(
                store=store,
                session_resolver=lambda: session_ids,
                language=Language.EN,
            )
            self.pushed = []

        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed.append((frame, direction))

    store = LockedStore()
    processor = CaptureProcessor(store)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [
        (frame, FrameDirection.DOWNSTREAM),
        (frame, FrameDirection.DOWNSTREAM),
    ]
    assert store.ensure_calls == 1


@pytest.mark.asyncio
async def test_turn_recorder_processor_records_tool_calls_and_allows_identical_later_turns(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    context = LLMContext(messages=_weather_turn_messages())
    processor = TurnRecorderProcessor(
        store=store,
        session_resolver=lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        context=context,
        consolidator=BrainMemoryConsolidator(store=store),
    )

    await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    context.add_messages(_weather_turn_messages())
    await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    episodes = list(reversed(store.recent_episodes(user_id="pc-123", thread_id="browser:pc-123", limit=10)))

    assert len(episodes) == 2
    assert json.loads(episodes[0].tool_calls_json) == [
        {
            "tool_call_id": "call_1",
            "function_name": "get_weather",
            "arguments": {"location": "上海"},
            "result": {"conditions": "Sunny"},
        }
    ]
    assert episodes[0].assistant_text == "上海今天晴。"
    assert episodes[1].assistant_text == "上海今天晴。"

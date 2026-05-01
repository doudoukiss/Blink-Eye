import pytest

from blink.brain.events import BrainEventType
from blink.brain.memory import BrainMemoryConsolidator
from blink.brain.processors import (
    BrainEventRecorderProcessor,
    HotPathMemoryExtractor,
    TurnRecorderProcessor,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection
from blink.transcriptions.language import Language


def _turn_messages():
    return [
        {"role": "user", "content": "提醒我给妈妈打电话。"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "fetch_user_image",
                        "arguments": '{"question": "我手里拿着什么？"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"answer": "你手里拿着一个杯子。"}',
        },
        {"role": "assistant", "content": "好的，我会记住，也看到你拿着一个杯子。"},
    ]


@pytest.mark.asyncio
async def test_brain_processors_record_turn_tool_and_goal_events_into_projections(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_resolver = lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    context = LLMContext(messages=_turn_messages())
    event_recorder = BrainEventRecorderProcessor(store=store, session_resolver=session_resolver)
    memory_extractor = HotPathMemoryExtractor(
        store=store,
        session_resolver=session_resolver,
        language=Language.ZH,
    )
    turn_recorder = TurnRecorderProcessor(
        store=store,
        session_resolver=session_resolver,
        context=context,
        consolidator=BrainMemoryConsolidator(store=store),
    )

    await event_recorder.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await event_recorder.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await memory_extractor.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)
    await event_recorder.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await turn_recorder.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    events = list(
        reversed(
            store.recent_brain_events(
                user_id="pc-123",
                thread_id="browser:pc-123",
                limit=16,
            )
        )
    )

    assert [event.event_type for event in events] == [
        BrainEventType.USER_TURN_STARTED,
        BrainEventType.USER_TURN_ENDED,
        BrainEventType.USER_TURN_TRANSCRIBED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.MEMORY_BLOCK_UPSERTED,
        BrainEventType.ASSISTANT_TURN_STARTED,
        BrainEventType.TOOL_CALLED,
        BrainEventType.TOOL_COMPLETED,
        BrainEventType.ASSISTANT_TURN_ENDED,
        BrainEventType.MEMORY_CLAIM_RECORDED,
        BrainEventType.MEMORY_BLOCK_UPSERTED,
    ]

    working_context = store.get_working_context_projection(scope_key="browser:pc-123")
    agenda = store.get_agenda_projection(scope_key="browser:pc-123", user_id="pc-123")
    heartbeat = store.get_heartbeat_projection(scope_key="browser:pc-123")

    assert working_context.user_turn_open is False
    assert working_context.assistant_turn_open is False
    assert working_context.last_user_text == "提醒我给妈妈打电话。"
    assert working_context.last_assistant_text == "好的，我会记住，也看到你拿着一个杯子。"
    assert working_context.last_tool_name == "fetch_user_image"
    assert agenda.agenda_seed == "提醒我给妈妈打电话。"
    assert agenda.open_goals == ["给妈妈打电话"]
    assert heartbeat.last_tool_name == "fetch_user_image"
    assert heartbeat.last_event_type == BrainEventType.MEMORY_BLOCK_UPSERTED

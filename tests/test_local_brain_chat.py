import asyncio
from pathlib import Path

import pytest

import blink.cli.local_brain_chat as local_brain_chat
from blink.brain.evals.memory_state import build_continuity_state
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.transcriptions.language import Language


class FakeLLM(FrameProcessor):
    def __init__(
        self,
        *,
        response: str = "好的，我已经记住了。",
        responses: list[str] | None = None,
    ):
        super().__init__(name="fake-llm")
        self.response = response
        self.responses = list(responses or [])
        self.registered_functions: list[str] = []

    def register_function(self, function_name, handler):
        self.registered_functions.append(function_name)
        return None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            response = self.responses.pop(0) if self.responses else self.response
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMTextFrame(response), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            return
        await self.push_frame(frame, direction)


@pytest.fixture
def fake_runtime_dependencies(monkeypatch):
    async def _verify_ollama(*args, **kwargs):
        return None

    monkeypatch.setattr(local_brain_chat, "verify_ollama", _verify_ollama)
    return monkeypatch


def test_local_brain_chat_config_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "")

    args = local_brain_chat.build_parser().parse_args([])
    config = local_brain_chat.resolve_config(args)

    assert config.base_url == "http://127.0.0.1:11434/v1"
    assert config.model == "qwen3.5:4b"
    assert config.language == Language.ZH
    assert config.brain_db_path is None


def test_local_brain_chat_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9090/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen-test")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "Brain prompt")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    args = local_brain_chat.build_parser().parse_args(["--brain-db-path", "/tmp/brain.db"])
    config = local_brain_chat.resolve_config(args)

    assert config.base_url == "http://localhost:9090/v1"
    assert config.model == "qwen-test"
    assert config.system_prompt == "Brain prompt"
    assert config.language == Language.EN
    assert config.brain_db_path == "/tmp/brain.db"


def test_local_brain_chat_is_registered_and_callable():
    assert callable(local_brain_chat.main)
    assert 'blink-local-brain-chat = "blink.cli.local_brain_chat:main"' in Path(
        "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_text_runtime_session_ids_are_isolated():
    session_ids = resolve_brain_session_ids(runtime_kind="text", client_id="alpha")

    assert session_ids.session_id == "text:alpha"
    assert session_ids.thread_id == "text:alpha"


@pytest.mark.asyncio
async def test_local_brain_chat_once_persists_turns_and_continuity(
    tmp_path,
    fake_runtime_dependencies,
    monkeypatch,
):
    fake_llm = FakeLLM()
    monkeypatch.setattr(local_brain_chat, "create_ollama_llm_service", lambda **kwargs: fake_llm)

    db_path = tmp_path / "brain.db"
    config = local_brain_chat.LocalBrainChatConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="请简洁回答。",
        language=Language.ZH,
        once="记住我叫阿周。",
        show_banner=False,
        verbose=False,
        brain_db_path=str(db_path),
    )

    result = await local_brain_chat.run_local_brain_chat(config)

    assert result == 0
    assert "brain_remember_profile" in fake_llm.registered_functions
    session_ids = resolve_brain_session_ids(runtime_kind="text")
    store = BrainStore(path=db_path)
    try:
        episodes = store.recent_episodes(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=4,
        )
        assert episodes
        assert episodes[0].user_text == "记住我叫阿周。"
        assert episodes[0].assistant_text == fake_llm.response

        event_types = [
            record.event_type
            for record in store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=32,
            )
        ]
        assert "user.turn.started" in event_types
        assert "user.turn.transcribed" in event_types
        assert "user.turn.ended" in event_types
        assert "assistant.turn.started" in event_types
        assert "assistant.turn.ended" in event_types

        continuity_state = build_continuity_state(
            store=store,
            session_ids=session_ids,
            presence_scope_key="text:presence",
            language=Language.ZH,
        )
        assert continuity_state["packet_traces"]["reply"]["task"] == "reply"
        assert continuity_state["continuity_graph"]["node_counts"]
        assert continuity_state["continuity_dossiers"]["dossier_counts"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_local_brain_chat_does_not_send_duplicate_llm_system_prompt(
    tmp_path,
    fake_runtime_dependencies,
    monkeypatch,
):
    captured_kwargs: dict[str, object] = {}
    fake_llm = FakeLLM()

    def _create_llm(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(local_brain_chat, "create_ollama_llm_service", _create_llm)

    config = local_brain_chat.LocalBrainChatConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="请简洁回答。",
        language=Language.ZH,
        once="记住我叫阿周。",
        show_banner=False,
        verbose=False,
        brain_db_path=str(tmp_path / "llm-prompt.db"),
    )

    result = await local_brain_chat.run_local_brain_chat(config)

    assert result == 0
    assert captured_kwargs["system_prompt"] == ""


@pytest.mark.asyncio
async def test_local_brain_chat_reset_clears_context_only(
    tmp_path,
    fake_runtime_dependencies,
    monkeypatch,
):
    fake_llm = FakeLLM(response="第一轮回复。")
    monkeypatch.setattr(local_brain_chat, "create_ollama_llm_service", lambda **kwargs: fake_llm)

    captured_contexts: list[LLMContext] = []
    original_context_cls = local_brain_chat.LLMContext

    class TrackingContext(original_context_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_contexts.append(self)

    prompts = iter(["第一轮消息", "/reset", "/exit"])

    async def _fake_to_thread(func, prompt):
        return next(prompts)

    monkeypatch.setattr(local_brain_chat, "LLMContext", TrackingContext)
    monkeypatch.setattr(local_brain_chat.asyncio, "to_thread", _fake_to_thread)

    config = local_brain_chat.LocalBrainChatConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="请简洁回答。",
        language=Language.ZH,
        once=None,
        show_banner=False,
        verbose=False,
        brain_db_path=str(tmp_path / "reset.db"),
    )

    result = await local_brain_chat.run_local_brain_chat(config)

    assert result == 0
    assert captured_contexts
    assert captured_contexts[0].get_messages() == []

    session_ids = resolve_brain_session_ids(runtime_kind="text")
    store = BrainStore(path=tmp_path / "reset.db")
    try:
        episodes = store.recent_episodes(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=4,
        )
        assert len(episodes) == 1
        assert episodes[0].user_text == "第一轮消息"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_local_brain_chat_supports_multiple_turns_without_stopping_runner(
    tmp_path,
    fake_runtime_dependencies,
    monkeypatch,
):
    fake_llm = FakeLLM(responses=["第一轮回复。", "第二轮回复。"])
    monkeypatch.setattr(local_brain_chat, "create_ollama_llm_service", lambda **kwargs: fake_llm)

    prompts = iter(["第一轮消息", "第二轮消息", "/exit"])

    async def _fake_to_thread(func, prompt):
        return next(prompts)

    monkeypatch.setattr(local_brain_chat.asyncio, "to_thread", _fake_to_thread)

    config = local_brain_chat.LocalBrainChatConfig(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        system_prompt="请简洁回答。",
        language=Language.ZH,
        once=None,
        show_banner=False,
        verbose=False,
        brain_db_path=str(tmp_path / "multi-turn.db"),
    )

    result = await local_brain_chat.run_local_brain_chat(config)

    assert result == 0

    session_ids = resolve_brain_session_ids(runtime_kind="text")
    store = BrainStore(path=tmp_path / "multi-turn.db")
    try:
        episodes = store.recent_episodes(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        )
        assert [episode.user_text for episode in episodes] == ["第二轮消息", "第一轮消息"]
        assert [episode.assistant_text for episode in episodes] == ["第二轮回复。", "第一轮回复。"]
    finally:
        store.close()

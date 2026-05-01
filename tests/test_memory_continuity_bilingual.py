import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.brain.memory import register_memory_tools
from blink.brain.memory_v2 import (
    BrainContinuityQuery,
    BrainMemoryUseTraceRef,
    ContinuityRetriever,
    DiscourseEpisode,
    MemoryContinuityTrace,
    build_memory_continuity_trace,
    build_memory_use_trace,
    detect_memory_command_intent,
)
from blink.brain.persona import compile_performance_plan_v2
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.function_calling import FunctionCallParams
from blink.processors.aggregators.llm_context import LLMContext
from blink.transcriptions.language import Language

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "memory_continuity_trace.schema.json"
)


class DummyLLM:
    def __init__(self):
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


async def _call_tool(handler, arguments):
    payload = {}

    async def result_callback(result, properties=None):
        payload["result"] = result

    params = FunctionCallParams(
        function_name="tool",
        tool_call_id="tool-call-1",
        arguments=arguments,
        llm=DummyLLM(),
        context=LLMContext(),
        result_callback=result_callback,
    )
    await handler(params)
    return payload["result"]


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _session(client_id: str = "continuity"):
    return resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)


def _use_trace(*, title: str, language: Language, session_ids=None):
    session_ids = session_ids or _session(f"trace-{language.value}")
    return build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{session_ids.user_id}:claim_path",
                display_kind="preference",
                title=title,
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from your explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
    )


def test_memory_continuity_trace_schema_and_cross_language_summaries():
    zh_trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="中文 Melo 和英文 Kokoro 是同等主要浏览器路径",
            language=Language.ZH,
        ),
        session_id=_session("zh").session_id,
        profile="browser-en-kokoro",
        language="en",
        hidden_counts={"suppressed": 1},
        command_intent=detect_memory_command_intent(
            "why did you answer that way?",
            language=Language.EN,
        ),
    )
    en_trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="Chinese Melo and English Kokoro are equal primary browser paths",
            language=Language.EN,
        ),
        session_id=_session("en").session_id,
        profile="browser-zh-melo",
        language="zh",
    )

    zh_payload = zh_trace.as_dict()
    en_payload = en_trace.as_dict()
    _assert_schema_valid(zh_payload)
    _assert_schema_valid(en_payload)
    assert set(zh_payload) == set(en_payload)
    assert zh_payload["cross_language_count"] == 1
    assert en_payload["cross_language_count"] == 1
    assert zh_payload["memory_effect"] == "cross_language_callback"
    assert en_payload["memory_effect"] == "cross_language_callback"
    assert "Chinese" in zh_payload["selected_memories"][0]["summary"]
    assert "中文" in en_payload["selected_memories"][0]["summary"]
    assert "memory_continuity:cross_language_selected" in zh_payload["selected_memories"][0][
        "reason_codes"
    ]
    assert zh_payload["memory_continuity_v3"]["schema_version"] == 3
    assert zh_payload["selected_memories"][0]["effect_labels"] == ["none"]


def test_memory_continuity_v3_links_discourse_episode_effects():
    session_ids = _session("discourse-link")
    discourse = DiscourseEpisode.from_dict(
        {
            "schema_version": 3,
            "discourse_episode_id": "discourse-episode-v3:project-1",
            "source_performance_episode_id": "episode:project",
            "source_event_ids": [1, 2],
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_runtime_label": "kokoro/English",
            "created_at_ms": 1,
            "category_labels": ["active_project", "user_preference"],
            "public_summary": "Discourse memory cue: active_project.",
            "memory_refs": [
                {
                    "memory_id": f"memory_claim:user:{session_ids.user_id}:claim_path",
                    "display_kind": "preference",
                    "summary": "Concise project explanations.",
                    "source_language": "zh",
                    "cross_language": True,
                    "effect_labels": [
                        "shorter_explanation",
                        "project_constraint_recall",
                    ],
                    "confidence_bucket": "high",
                    "reason_codes": ["source:context_selection"],
                }
            ],
            "effect_labels": ["shorter_explanation", "project_constraint_recall"],
            "conflict_labels": [],
            "staleness_labels": [],
            "confidence_bucket": "high",
            "reason_codes": ["discourse_episode:v3"],
        }
    )
    trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="中文项目偏好：解释要短",
            language=Language.ZH,
            session_ids=session_ids,
        ),
        session_id=session_ids.session_id,
        profile="browser-en-kokoro",
        language="en",
        discourse_episodes=(discourse,),
    )
    payload = trace.as_dict()

    _assert_schema_valid(payload)
    selected = payload["selected_memories"][0]
    assert selected["linked_discourse_episode_ids"] == ["discourse-episode-v3:project-1"]
    assert selected["effect_labels"] == [
        "shorter_explanation",
        "project_constraint_recall",
    ]
    assert payload["memory_continuity_v3"]["cross_language_transfer_count"] == 1
    assert payload["memory_continuity_v3"]["selected_discourse_episodes"][0][
        "discourse_episode_id"
    ] == "discourse-episode-v3:project-1"


def test_memory_command_intent_detection_is_bilingual_and_text_free():
    zh = detect_memory_command_intent("帮我记住，中文 Melo 是主要路径", language=Language.ZH)
    en = detect_memory_command_intent(
        "why did you answer that way?",
        language=Language.EN,
    )

    assert zh.intent == "remember"
    assert en.intent == "explain_answer"
    assert "中文 Melo" not in json.dumps(zh.as_dict(), ensure_ascii=False)
    assert zh.as_dict()["text_chars"] > 0


def test_cross_language_retrieval_selects_seeded_memory(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("retrieval")
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="browser parity",
        value={"value": "中文 Melo 和英文 Kokoro 是同等主要浏览器路径"},
        rendered_text="中文 Melo 和英文 Kokoro 是同等主要浏览器路径",
        confidence=0.9,
        singleton=False,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    retriever = ContinuityRetriever(store=store)

    results = retriever.retrieve(
        BrainContinuityQuery(
            text="Kokoro English browser primary path",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=4,
        )
    )

    assert results
    assert any("Kokoro" in result.summary for result in results)


def test_continuity_trace_changes_performance_plan_policy():
    trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="中文 Melo 和英文 Kokoro 是同等主要浏览器路径",
            language=Language.ZH,
        ),
        session_id=_session("plan").session_id,
        profile="browser-en-kokoro",
        language="en",
    )
    plan = compile_performance_plan_v2(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        memory_continuity_trace=trace.as_dict(),
        active_listening={"constraints": [{"value": "keep parity"}]},
        camera_scene={"state": "disabled"},
        floor_state={"state": "assistant_has_floor"},
    ).as_dict()

    assert plan["memory_callback_policy"]["state"] == "cross_language_callback"
    assert plan["memory_callback_policy"]["cross_language_count"] == 1
    assert "memory_callback" in {
        item["scenario"] for item in plan["persona_references_used"]
    }


def test_suppressed_or_corrected_memory_pushes_repair_uncertainty():
    trace = build_memory_continuity_trace(
        session_id=_session("suppressed").session_id,
        profile="browser-zh-melo",
        language="zh",
        hidden_counts={"suppressed": 2, "historical": 1},
    )
    payload = trace.as_dict()

    _assert_schema_valid(payload)
    assert payload["memory_effect"] == "repair_or_uncertainty"
    assert payload["selected_memory_count"] == 0
    assert payload["suppressed_memory_count"] == 3


@pytest.mark.asyncio
async def test_explain_memory_continuity_tool_returns_public_summary(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("tool")
    trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="Chinese Melo and English Kokoro are equal primary browser paths",
            language=Language.EN,
            session_ids=session_ids,
        ),
        session_id=session_ids.session_id,
        profile="browser-zh-melo",
        language="zh",
    )
    store.append_memory_continuity_trace(
        trace=trace,
        session_id=session_ids.session_id,
        source="test",
        ts="2026-04-23T01:02:03+00:00",
    )
    llm = DummyLLM()
    register_memory_tools(
        llm=llm,
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
    )

    payload = await _call_tool(llm.registered_functions["brain_explain_memory_continuity"], {})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()

    assert payload["available"] is True
    assert payload["cross_language_count"] == 1
    assert payload["selected_memories"][0]["editable"] is True
    for banned in ("system_prompt", "developer_prompt", "raw_prompt", "full_message"):
        assert banned not in encoded


def test_continuity_trace_from_dict_roundtrip_is_public_safe():
    trace = build_memory_continuity_trace(
        memory_use_trace=_use_trace(
            title="raw_prompt secret should redact",
            language=Language.EN,
        ),
        session_id=_session("roundtrip").session_id,
        profile="browser-en-kokoro",
        language="en",
    )
    hydrated = MemoryContinuityTrace.from_dict(trace.as_dict())
    encoded = json.dumps(hydrated.as_dict(), ensure_ascii=False, sort_keys=True).lower()

    _assert_schema_valid(hydrated.as_dict())
    assert "raw_prompt secret" not in encoded
    assert "system_prompt" not in encoded

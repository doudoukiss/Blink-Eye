import pytest

from blink.brain.memory import register_memory_tools
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.function_calling import FunctionCallParams
from blink.processors.aggregators.llm_context import LLMContext
from blink.transcriptions.language import Language


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


@pytest.mark.asyncio
async def test_register_memory_tools_supports_remember_forget_and_complete(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    llm = DummyLLM()
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")

    tools = register_memory_tools(
        llm=llm,
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
    )

    assert [tool.name for tool in tools.standard_tools] == [
        "brain_remember_profile",
        "brain_remember_preference",
        "brain_remember_task",
        "brain_forget_memory",
        "brain_apply_memory_governance",
        "brain_complete_task",
        "brain_list_visible_memories",
        "brain_explain_memory_continuity",
    ]

    remember_name = await _call_tool(
        llm.registered_functions["brain_remember_profile"],
        {"field": "name", "value": "阿周"},
    )
    remember_like = await _call_tool(
        llm.registered_functions["brain_remember_preference"],
        {"sentiment": "like", "topic": "咖啡"},
    )
    remember_dislike = await _call_tool(
        llm.registered_functions["brain_remember_preference"],
        {"sentiment": "dislike", "topic": "咖啡"},
    )
    dislike_claim = next(
        claim
        for claim in store.query_claims(
            temporal_mode="current",
            predicate="preference.dislike",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=None,
        )
        if claim.object.get("value") == "咖啡"
    )
    governance_pin = await _call_tool(
        llm.registered_functions["brain_apply_memory_governance"],
        {
            "memory_id": f"memory_claim:user:{session_ids.user_id}:{dislike_claim.claim_id}",
            "action": "pin",
        },
    )
    remember_task = await _call_tool(
        llm.registered_functions["brain_remember_task"],
        {"title": "给妈妈打电话", "details": "今晚"},
    )
    recall = await _call_tool(
        llm.registered_functions["brain_list_visible_memories"],
        {"limit": 8},
    )
    explain = await _call_tool(
        llm.registered_functions["brain_explain_memory_continuity"],
        {},
    )
    complete_task = await _call_tool(
        llm.registered_functions["brain_complete_task"],
        {"title": "给妈妈打电话"},
    )
    forget_name = await _call_tool(
        llm.registered_functions["brain_forget_memory"],
        {"kind": "profile.name"},
    )
    governance_reject = await _call_tool(
        llm.registered_functions["brain_apply_memory_governance"],
        {"memory_id": "not-a-memory-id", "action": "forget"},
    )

    facts = store.active_facts(user_id="pc-123", limit=10)
    rendered = [fact.rendered_text for fact in facts]
    tasks = store.active_tasks(user_id="pc-123", limit=10)

    assert remember_name["accepted"] is True
    assert remember_like["accepted"] is True
    assert remember_dislike["accepted"] is True
    assert governance_pin["accepted"] is True
    assert "claim_pinned" in governance_pin["reason_codes"]
    assert remember_task["accepted"] is True
    assert recall["available"] is True
    assert recall["record_count"] >= 1
    assert "visible_memory_recall:v1" in recall["reason_codes"]
    assert "source_event" not in str(recall)
    assert explain["available"] is True
    assert "memory_continuity_explanation:v1" in explain["reason_codes"]
    assert complete_task["accepted"] is True
    assert forget_name["accepted"] is True
    assert governance_reject["accepted"] is False
    assert "memory_id_malformed" in governance_reject["reason_codes"]
    assert "用户名字是 阿周" not in rendered
    assert "用户不喜欢 咖啡" in rendered
    assert "用户喜欢 咖啡" not in rendered
    assert tasks == []


@pytest.mark.asyncio
async def test_memory_tools_reject_unsupported_forget_requests(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    llm = DummyLLM()
    register_memory_tools(
        llm=llm,
        store=store,
        session_resolver=lambda: resolve_brain_session_ids(runtime_kind="voice"),
        language=Language.EN,
    )

    missing_target = await _call_tool(
        llm.registered_functions["brain_forget_memory"],
        {"kind": "preference.like"},
    )
    unsupported = await _call_tool(
        llm.registered_functions["brain_forget_memory"],
        {"kind": "session.summary"},
    )

    assert missing_target["accepted"] is False
    assert missing_target["error"] == "preference_target_required"
    assert unsupported["accepted"] is False
    assert unsupported["error"] == "unsupported_memory_kind"

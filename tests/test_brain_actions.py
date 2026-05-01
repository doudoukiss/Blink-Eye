import pytest

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    EmbodiedCapabilityDispatcher,
    EmbodiedCommandClassifierResult,
    EmbodiedCommandInterpreter,
    build_embodied_capability_registry,
    register_embodied_action_tools,
)
from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.embodied_policy import EmbodiedPolicyExecutionRequest
from blink.brain.events import BrainEventType
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import PreviewDriver
from blink.embodiment.robot_head.models import RobotHeadDriverStatus, RobotHeadExecutionResult
from blink.function_calling import FunctionCallParams
from blink.processors.aggregators.llm_context import LLMContext
from blink.transcriptions.language import Language


class DummyLLM:
    def __init__(self):
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


class RecordingPolicyAdapter:
    def __init__(self):
        self.descriptor = BrainAdapterDescriptor(
            backend_id="recording_policy",
            backend_version="v1",
            capabilities=("status", "embodied_action_execution"),
            degraded_mode_id="preview_only",
            default_timeout_ms=25,
        )
        self.execution_backend = "recording"
        self.requests: list[EmbodiedPolicyExecutionRequest] = []

    async def status(self) -> RobotHeadExecutionResult:
        return RobotHeadExecutionResult(
            accepted=True,
            command_type="status",
            driver=self.execution_backend,
            preview_only=False,
            status=RobotHeadDriverStatus(mode=self.execution_backend, available=True, armed=True),
            summary="adapter status",
        )

    async def execute_action(self, request: EmbodiedPolicyExecutionRequest) -> RobotHeadExecutionResult:
        self.requests.append(request)
        return RobotHeadExecutionResult(
            accepted=True,
            command_type="action",
            resolved_name=request.action_id,
            driver=self.execution_backend,
            preview_only=False,
            status=RobotHeadDriverStatus(mode=self.execution_backend, available=True, armed=True),
            summary=f"adapter executed {request.action_id}",
            metadata={"adapter": "recording"},
        )


async def _call_tool(handler):
    payload = {}

    async def result_callback(result, properties=None):
        payload["result"] = result

    params = FunctionCallParams(
        function_name="tool",
        tool_call_id="tool-call-1",
        arguments={},
        llm=DummyLLM(),
        context=LLMContext(),
        result_callback=result_callback,
    )
    await handler(params)
    return payload["result"]


def test_embodied_command_interpreter_maps_and_blocks_requests():
    interpreter = EmbodiedCommandInterpreter()

    assert interpreter.interpret("向左看一下。").action_id == "cmd_look_left"
    assert interpreter.interpret("眨眼一次。").action_id == "cmd_blink"
    assert interpreter.interpret("把 servo 1 转到 2200").denied_reason == "raw_control_not_allowed"
    assert interpreter.interpret("先看左边再眨眼").action_sequence == (
        "cmd_look_left",
        "cmd_blink",
    )
    assert interpreter.interpret("status").action_id is None


@pytest.mark.parametrize(
    ("utterance", "action_id"),
    [
        ("请眨眼一次。", "cmd_blink"),
        ("左眼眨一下。", "cmd_wink_left"),
        ("右眼眨一下。", "cmd_wink_right"),
        ("向左看一下。", "cmd_look_left"),
        ("往右看。", "cmd_look_right"),
        ("回到中位。", "cmd_return_neutral"),
        ("现在头部状态是什么？", "cmd_report_status"),
        ("robot head status", "cmd_report_status"),
    ],
)
def test_embodied_command_interpreter_fixture_matrix(utterance, action_id):
    interpreter = EmbodiedCommandInterpreter()

    assert interpreter.interpret(utterance).action_id == action_id


def test_embodied_command_interpreter_uses_optional_classifier_only_after_lexical_parse():
    calls: list[str] = []

    def classifier(text: str):
        calls.append(text)
        if text == "请把头转向左侧":
            return EmbodiedCommandClassifierResult(action_id="cmd_look_left")
        if text == "请做一个秘密动作":
            return EmbodiedCommandClassifierResult(action_id="cmd_secret")
        return None

    interpreter = EmbodiedCommandInterpreter(classifier=classifier)

    assert interpreter.interpret("请把头转向左侧").action_id == "cmd_look_left"
    assert interpreter.interpret("请做一个秘密动作").action_id is None
    assert interpreter.interpret("眨眼一次。").action_id == "cmd_blink"
    assert interpreter.interpret("把 servo 1 转到 2200").denied_reason == "raw_control_not_allowed"
    assert calls == ["请把头转向左侧", "请做一个秘密动作"]


@pytest.mark.asyncio
async def test_register_embodied_action_tools_exposes_safe_surface(tmp_path):
    llm = DummyLLM()
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=PreviewDriver(trace_dir=tmp_path),
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
    )
    dispatcher = EmbodiedCapabilityDispatcher(
        action_engine=action_engine,
        capability_registry=build_embodied_capability_registry(action_engine=action_engine),
    )

    tools = register_embodied_action_tools(
        llm=llm,
        dispatcher=dispatcher,
        language=Language.ZH,
    )

    assert [tool.name for tool in tools.standard_tools] == [
        "robot_head_blink",
        "robot_head_wink_left",
        "robot_head_wink_right",
        "robot_head_look_left",
        "robot_head_look_right",
        "robot_head_return_neutral",
        "robot_head_status",
    ]
    assert "robot_head_set_state" not in llm.registered_functions
    assert "robot_head_run_motif" not in llm.registered_functions

    blink_result = await _call_tool(llm.registered_functions["robot_head_blink"])
    status_result = await _call_tool(llm.registered_functions["robot_head_status"])

    assert blink_result["accepted"] is True
    assert blink_result["action_id"] == "cmd_blink"
    assert status_result["action_id"] == "cmd_report_status"
    events = list(
        reversed(store.recent_brain_events(user_id=session_ids.user_id, thread_id=session_ids.thread_id, limit=8))
    )
    capability_events = [event for event in events if event.event_type.startswith("capability.")]
    assert [event.event_type for event in capability_events] == [
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
    ]
    assert capability_events[0].correlation_id == "tool-call-1"
    await controller.close()


@pytest.mark.asyncio
async def test_embodied_action_engine_routes_execution_through_policy_adapter(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=PreviewDriver(trace_dir=tmp_path),
    )
    adapter = RecordingPolicyAdapter()
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        policy_adapter=adapter,
        store=store,
        session_resolver=lambda: session_ids,
    )
    try:
        result = await action_engine.run_action(
            "cmd_look_left",
            source="operator",
            reason="Adapter-backed execution",
            metadata={"request_id": "adapter-test"},
        )
        recent_event = next(
            event
            for event in store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=8,
                event_types=(BrainEventType.ROBOT_ACTION_OUTCOME,),
            )
        )

        assert result.accepted is True
        assert action_engine.execution_backend == "recording"
        assert len(adapter.requests) == 1
        assert adapter.requests[0].action_id == "cmd_look_left"
        assert [step.command_type for step in adapter.requests[0].controller_plan] == ["run_motif"]
        assert recent_event.payload["action_id"] == "cmd_look_left"
        assert recent_event.payload["status"]["mode"] == "recording"
    finally:
        await controller.close()

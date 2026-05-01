import pytest

from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver, PreviewDriver
from blink.embodiment.robot_head.tools import register_robot_head_tools
from blink.function_calling import FunctionCallParams
from blink.processors.aggregators.llm_context import LLMContext
from blink.transcriptions.language import Language


class DummyLLM:
    def __init__(self):
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


async def _call_tool(handler, *, arguments):
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
async def test_robot_head_tools_register_schema_and_execute(tmp_path):
    llm = DummyLLM()
    catalog = build_default_robot_head_catalog()
    controller = RobotHeadController(
        catalog=catalog,
        driver=PreviewDriver(trace_dir=tmp_path),
    )

    tools = register_robot_head_tools(
        llm=llm,
        controller=controller,
        catalog=catalog,
        language=Language.ZH,
    )

    assert [tool.name for tool in tools.standard_tools] == [
        "robot_head_set_state",
        "robot_head_run_motif",
        "robot_head_return_neutral",
        "robot_head_status",
    ]
    assert "servo" not in " ".join(tool.description.lower() for tool in tools.standard_tools)

    state_result = await _call_tool(
        llm.registered_functions["robot_head_set_state"],
        arguments={"state": "friendly"},
    )
    motif_result = await _call_tool(
        llm.registered_functions["robot_head_run_motif"],
        arguments={"motif": "blink"},
    )
    invalid_result = await _call_tool(
        llm.registered_functions["robot_head_set_state"],
        arguments={"state": "unsupported_state"},
    )

    assert state_result["accepted"] is True
    assert motif_result["trace_path"].endswith("trace-0002-run_motif.json")
    assert invalid_result["accepted"] is False
    assert "supported_states" in invalid_result
    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_status_tool_reports_supported_entries():
    llm = DummyLLM()
    catalog = build_default_robot_head_catalog()
    controller = RobotHeadController(catalog=catalog, driver=MockDriver())

    register_robot_head_tools(
        llm=llm,
        controller=controller,
        catalog=catalog,
        language=Language.EN,
    )
    status_result = await _call_tool(llm.registered_functions["robot_head_status"], arguments={})

    assert status_result["accepted"] is True
    assert status_result["supported_states"] == catalog.public_state_names()
    assert status_result["supported_motifs"] == catalog.public_motif_names()
    await controller.close()

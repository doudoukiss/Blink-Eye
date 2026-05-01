"""LLM tool registration helpers for robot-head embodiment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from blink.adapters.schemas.function_schema import FunctionSchema
from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.models import RobotHeadExecutionResult
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.function_calling import FunctionCallParams


def robot_head_tool_prompt(language: Language) -> str:
    """Return a prompt suffix describing the explicit robot-head tools."""
    if language.value.lower().startswith(("zh", "cmn")):
        return (
            "当用户明确要求机器人头部看向某侧、眨眼、眨某一只眼、切换表情、回到中位、"
            "或询问头部当前能力与状态时，请使用 robot_head_* 工具。"
            "不要虚构底层舵机、串口或未声明的动作能力。"
        )
    return (
        "When the user explicitly asks the robot head to look somewhere, blink, wink, "
        "change expression, return to neutral, or report its current status, use the "
        "robot_head_* tools. Do not invent raw servo, serial, or unsupported motion primitives."
    )


def _tool_payload(result: RobotHeadExecutionResult) -> dict:
    payload = {
        "accepted": result.accepted,
        "driver": result.driver,
        "preview_only": result.preview_only,
        "warnings": result.warnings,
        "summary": result.summary,
    }
    if result.resolved_name is not None:
        payload["resolved_name"] = result.resolved_name
    if result.preset is not None:
        payload["preset"] = result.preset
    if result.trace_path is not None:
        payload["trace_path"] = result.trace_path
    if result.status is not None:
        payload["status"] = result.status.model_dump()
    if result.metadata:
        payload["metadata"] = result.metadata
    return payload


def register_robot_head_tools(
    *,
    llm,
    controller: RobotHeadController,
    catalog: RobotHeadCapabilityCatalog,
    language: Language,
) -> ToolsSchema:
    """Register the bounded robot-head tools on an LLM service."""

    async def robot_head_set_state(params: FunctionCallParams):
        state_name = str(params.arguments.get("state", "")).strip()
        try:
            result = await controller.set_state(
                state_name,
                source="tool",
                reason="Explicit user request via LLM tool.",
            )
        except Exception as exc:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": str(exc),
                    "supported_states": catalog.public_state_names(),
                }
            )
            return
        payload = _tool_payload(result)
        if not result.accepted:
            payload["supported_states"] = catalog.public_state_names()
        await params.result_callback(payload)

    async def robot_head_run_motif(params: FunctionCallParams):
        motif_name = str(params.arguments.get("motif", "")).strip()
        try:
            result = await controller.run_motif(
                motif_name,
                source="tool",
                reason="Explicit user request via LLM tool.",
            )
        except Exception as exc:
            await params.result_callback(
                {
                    "accepted": False,
                    "error": str(exc),
                    "supported_motifs": catalog.public_motif_names(),
                }
            )
            return
        payload = _tool_payload(result)
        if not result.accepted:
            payload["supported_motifs"] = catalog.public_motif_names()
        await params.result_callback(payload)

    async def robot_head_return_neutral(params: FunctionCallParams):
        result = await controller.return_neutral(
            source="tool",
            reason="Explicit user request via LLM tool.",
        )
        await params.result_callback(_tool_payload(result))

    async def robot_head_status(params: FunctionCallParams):
        result = await controller.status()
        payload = _tool_payload(result)
        payload["supported_states"] = catalog.public_state_names()
        payload["supported_motifs"] = catalog.public_motif_names()
        await params.result_callback(payload)

    llm.register_function("robot_head_set_state", robot_head_set_state)
    llm.register_function("robot_head_run_motif", robot_head_run_motif)
    llm.register_function("robot_head_return_neutral", robot_head_return_neutral)
    llm.register_function("robot_head_status", robot_head_status)

    if language.value.lower().startswith(("zh", "cmn")):
        set_state_description = "将机器人头部切换到一个受支持的持久表情状态。"
        run_motif_description = "执行一个受支持的短时头部动作或表情动作。"
        return_neutral_description = "让机器人头部回到中位和中性状态。"
        status_description = "获取机器人头部当前驱动模式、状态和支持的动作。"
        state_param_description = "要设置的表情状态名称。"
        motif_param_description = "要执行的动作名称。"
    else:
        set_state_description = "Set the robot head to a supported persistent expression state."
        run_motif_description = "Run a supported short robot-head motion motif."
        return_neutral_description = "Return the robot head to neutral."
        status_description = "Report the robot head driver mode, status, and supported actions."
        state_param_description = "The persistent state name to apply."
        motif_param_description = "The motif name to execute."

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="robot_head_set_state",
                description=set_state_description,
                properties={
                    "state": {
                        "type": "string",
                        "enum": catalog.public_state_names(),
                        "description": state_param_description,
                    }
                },
                required=["state"],
            ),
            FunctionSchema(
                name="robot_head_run_motif",
                description=run_motif_description,
                properties={
                    "motif": {
                        "type": "string",
                        "enum": catalog.public_motif_names(),
                        "description": motif_param_description,
                    }
                },
                required=["motif"],
            ),
            FunctionSchema(
                name="robot_head_return_neutral",
                description=return_neutral_description,
                properties={},
                required=[],
            ),
            FunctionSchema(
                name="robot_head_status",
                description=status_description,
                properties={},
                required=[],
            ),
        ]
    )

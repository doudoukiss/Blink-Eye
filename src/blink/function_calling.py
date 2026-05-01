#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider-light shared types for LLM function calling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from blink.frames.frames import FunctionCallResultProperties

if TYPE_CHECKING:
    from blink.processors.aggregators.llm_context import LLMContext
    from blink.services.llm_service import LLMService


FunctionCallHandler = Callable[["FunctionCallParams"], Awaitable[None]]


class FunctionCallResultCallback(Protocol):
    """Protocol for function call result callbacks.

    Used for both final results and intermediate updates. Pass
    ``properties=FunctionCallResultProperties(is_final=False)`` to send an
    intermediate update.
    """

    async def __call__(
        self, result: Any, *, properties: FunctionCallResultProperties | None = None
    ) -> None:
        """Deliver a function call result."""
        ...


@dataclass
class FunctionCallParams:
    """Parameters for a function call.

    Parameters:
        function_name: The name of the function being called.
        tool_call_id: A unique identifier for the function call.
        arguments: The arguments for the function.
        llm: The LLMService instance being used.
        context: The LLM context.
        result_callback: Callback to deliver the result of the function call.
    """

    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    llm: "LLMService"
    context: "LLMContext"
    result_callback: FunctionCallResultCallback

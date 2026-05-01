#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Console-oriented frame processors for local development workflows."""

import asyncio
import sys
from typing import Optional, TextIO

from blink.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from blink.processors.frame_processor import FrameDirection, FrameProcessor


class ConsoleLLMPrinter(FrameProcessor):
    """Print streaming LLM output to a console stream.

    This processor is intended for local text workflows where developers want
    to see each token as it arrives while still allowing the rest of the
    pipeline to aggregate context normally.

    Args:
        prefix: Text prefix printed when a new assistant response starts.
        stream: Text stream to write to. Defaults to ``sys.stdout``.
        response_queue: Optional queue that receives the completed response text
            after each ``LLMFullResponseEndFrame``.
    """

    def __init__(
        self,
        *,
        prefix: str = "assistant> ",
        stream: Optional[TextIO] = None,
        response_queue: Optional[asyncio.Queue[str]] = None,
        **kwargs,
    ):
        """Initialize the console printer.

        Args:
            prefix: Text prefix printed when a response starts.
            stream: Text stream to write to. Defaults to ``sys.stdout``.
            response_queue: Optional queue that receives the completed response.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._prefix = prefix
        self._stream = stream or sys.stdout
        self._response_queue = response_queue
        self._chunks: list[str] = []
        self._started = False

    async def process_frame(self, frame, direction: FrameDirection):
        """Process frames and mirror assistant text to the console."""
        await super().process_frame(frame, direction)

        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._started = True
                self._chunks = []
                self._write(self._prefix)
            elif isinstance(frame, LLMTextFrame) and self._started:
                self._chunks.append(frame.text)
                self._write(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._started:
                response = "".join(self._chunks)
                self._write("\n")
                self._started = False
                self._chunks = []
                if self._response_queue is not None:
                    await self._response_queue.put(response)
            elif isinstance(frame, InterruptionFrame) and self._started:
                self._write("\n")
                self._started = False
                self._chunks = []

        await self.push_frame(frame, direction)

    def _write(self, text: str):
        self._stream.write(text)
        self._stream.flush()

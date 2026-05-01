#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from loguru import logger

from blink.frames.frames import Frame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.frame_processor import FrameDirection, FrameProcessor

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class NullProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)


async def main():
    """This test shows heartbeat monitoring.

    A warning is dispalyed when heartbeats are not received within the
    default (5 seconds) timeout.
    """
    pipeline = Pipeline([NullProcessor()])

    task = PipelineTask(pipeline, params=PipelineParams(enable_heartbeats=True))

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

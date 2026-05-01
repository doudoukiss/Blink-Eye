#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Describe a local image with the optional Moondream vision stack."""

import argparse
import asyncio
from pathlib import Path

from PIL import Image

from blink.cli.local_common import DEFAULT_LOCAL_VISION_MODEL
from blink.frames.frames import EndFrame, Frame, UserImageRawFrame, VisionTextFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.services.moondream.vision import MoondreamService


class VisionPrinter(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionTextFrame):
            print(frame.text)

        await self.push_frame(frame, direction)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe a local image with Moondream.")
    parser.add_argument("--image", help="Path to an image file.")
    parser.add_argument("--question", default="Describe this image.")
    parser.add_argument("--model", default=DEFAULT_LOCAL_VISION_MODEL)
    return parser


async def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    default_image = Path(__file__).resolve().parents[1] / "assets" / "cat.jpg"
    image_path = Path(args.image) if args.image else default_image
    image = Image.open(image_path).convert("RGB")

    vision = MoondreamService(settings=MoondreamService.Settings(model=args.model))
    printer = VisionPrinter()
    task = PipelineTask(Pipeline([vision, printer]))

    await task.queue_frames(
        [
            UserImageRawFrame(
                image=image.tobytes(),
                format="RGB",
                size=image.size,
                text=args.question,
            ),
            EndFrame(),
        ]
    )

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

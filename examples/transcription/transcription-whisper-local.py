#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

from blink.audio.vad.silero import SileroVADAnalyzer
from blink.frames.frames import Frame, TranscriptionFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.audio.vad_processor import VADProcessor
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.services.whisper.stt import WhisperSTTService
from blink.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
        )
    )

    stt = WhisperSTTService()

    tl = TranscriptionLogger()
    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline([transport.input(), vad_processor, stt, tl])

    task = PipelineTask(pipeline)

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

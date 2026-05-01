#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run a local microphone transcription loop using the configured local STT backend."""

import asyncio
import sys

from loguru import logger

from blink.audio.vad.silero import SileroVADAnalyzer
from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    DEFAULT_LOCAL_STT_BACKEND,
    DEFAULT_LOCAL_STT_MODEL,
    create_local_stt_service,
    get_local_env,
    maybe_load_dotenv,
)
from blink.frames.frames import Frame, TranscriptionFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.audio.vad_processor import VADProcessor
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

maybe_load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class TranscriptionPrinter(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True))
    stt = create_local_stt_service(
        backend=get_local_env("STT_BACKEND", DEFAULT_LOCAL_STT_BACKEND),
        model=get_local_env("STT_MODEL", DEFAULT_LOCAL_STT_MODEL),
        language=DEFAULT_LOCAL_LANGUAGE,
    )
    printer = TranscriptionPrinter()
    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline([transport.input(), vad_processor, stt, printer])
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

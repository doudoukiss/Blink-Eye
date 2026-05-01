#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from blink.frames.frames import LLMRunFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from blink.runner.types import RunnerArguments
from blink.runner.utils import (
    create_transport,
    maybe_capture_participant_camera,
    maybe_capture_participant_screen,
)
from blink.services.google.gemini_live.llm import GeminiLiveLLMService
from blink.transports.base_transport import BaseTransport, TransportParams
from blink.transports.daily.transport import DailyParams

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GeminiLiveLLMService.Settings(
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            # system_instruction="Talk like a pirate."
        ),
        # inference_on_context_initialization=False,
    )

    context = LLMContext(
        [
            {
                "role": "developer",
                "content": "Say hello.",
            },
        ],
    )
    # Server-side VAD is enabled by default; no local VAD is added.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")

        await maybe_capture_participant_camera(transport, client, framerate=1)
        await maybe_capture_participant_screen(transport, client, framerate=1)

        await task.queue_frames([LLMRunFrame()])
        await asyncio.sleep(3)
        logger.debug("Unpausing audio and video")
        llm.set_audio_input_paused(False)
        llm.set_video_input_paused(False)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with the Blink runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from blink.runner.run import main

    main()

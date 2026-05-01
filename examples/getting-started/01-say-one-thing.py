#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from blink.frames.frames import EndFrame, TTSSpeakFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.runner.types import RunnerArguments
from blink.runner.utils import create_transport
from blink.services.cartesia.tts import CartesiaTTSService
from blink.transports.base_transport import BaseTransport, TransportParams
from blink.transports.daily.transport import DailyParams
from blink.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    task = PipelineTask(
        Pipeline([tts, transport.output()]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await task.queue_frames([TTSSpeakFrame(f"Hello there!"), EndFrame()])

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with the Blink runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from blink.runner.run import main

    main()

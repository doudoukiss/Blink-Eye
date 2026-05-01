#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from blink.audio.vad.silero import SileroVADAnalyzer
from blink.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from blink.runner.types import RunnerArguments
from blink.runner.utils import create_transport
from blink.services.sarvam.llm import SarvamLLMService
from blink.services.sarvam.stt import SarvamSTTService
from blink.services.sarvam.tts import SarvamTTSService
from blink.transports.base_transport import BaseTransport, TransportParams
from blink.transports.daily.transport import DailyParams
from blink.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable `{name}` is required.")
    return value


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = SarvamSTTService(
        settings=SarvamSTTService.Settings(model="saaras:v3"),
        api_key=_require_env("SARVAM_API_KEY"),
    )

    tts = SarvamTTSService(
        settings=SarvamTTSService.Settings(model="bulbul:v3"),
        api_key=_require_env("SARVAM_API_KEY"),
    )

    llm = SarvamLLMService(
        api_key=_require_env("SARVAM_API_KEY"),
        settings=SarvamLLMService.Settings(model="sarvam-30b"),
        system_instruction=(
            "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way."
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
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
        logger.info("Client connected")
        context.add_message({"role": "user", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

        await asyncio.sleep(10)
        logger.info("Updating Sarvam LLM settings: temperature=0.1")
        await task.queue_frame(
            LLMUpdateSettingsFrame(delta=SarvamLLMService.Settings(temperature=0.1))
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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

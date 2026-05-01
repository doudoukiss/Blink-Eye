#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from blink.audio.vad.silero import SileroVADAnalyzer
from blink.frames.frames import LLMRunFrame, TTSTextFrame
from blink.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
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
from blink.services.deepgram.stt import DeepgramSTTService
from blink.services.inworld.tts import InworldHttpTTSService
from blink.services.openai.llm import OpenAILLMService
from blink.transports.base_output import BaseOutputTransport
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = InworldHttpTTSService(
            api_key=os.getenv("INWORLD_API_KEY", ""),
            aiohttp_session=session,
            streaming=True,
            settings=InworldHttpTTSService.Settings(
                voice="Ashley",
            ),
            # Set to False for non-streaming mode or True for streaming mode.
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAILLMService.Settings(
                system_instruction="You are a helpful AI demonstrating Inworld AI's TTS. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a friendly and helpful way.",
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
            observers=[
                DebugLogObserver(
                    frame_types={
                        TTSTextFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
                    }
                ),
            ],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Kick off the conversation.
            context.add_message(
                {"role": "developer", "content": "Please introduce yourself to the user."}
            )
            await task.queue_frames([LLMRunFrame()])

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

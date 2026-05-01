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
from blink.frames.frames import LLMRunFrame, STTUpdateSettingsFrame
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
from blink.services.cartesia.tts import CartesiaTTSService
from blink.services.openai.llm import OpenAILLMService
from blink.services.speechmatics.stt import SpeechmaticsSTTService
from blink.transcriptions.language import Language
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
    logger.info(f"Starting bot")

    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        settings=SpeechmaticsSTTService.Settings(
            enable_diarization=True,
            speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
            speaker_passive_format="<PASSIVE><{speaker_id}>{text}</{speaker_id}></PASSIVE>",
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
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
        logger.info(f"Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

        await asyncio.sleep(10)
        logger.info("Updating Speechmatics STT settings: language=es")
        await task.queue_frame(
            STTUpdateSettingsFrame(delta=SpeechmaticsSTTService.Settings(language=Language.ES))
        )

        await asyncio.sleep(10)
        logger.info("Updating Speechmatics STT settings: focus_speakers=['S1']")
        await task.queue_frame(
            STTUpdateSettingsFrame(delta=SpeechmaticsSTTService.Settings(focus_speakers=["S1"]))
        )

        await asyncio.sleep(10)
        logger.info(
            "Updating Speechmatics STT settings: speaker_active_format=<SPEAKER_{speaker_id}>{text}</SPEAKER_{speaker_id}>"
        )
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=SpeechmaticsSTTService.Settings(
                    speaker_active_format="<SPEAKER_{speaker_id}>{text}</SPEAKER_{speaker_id}>"
                )
            )
        )

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

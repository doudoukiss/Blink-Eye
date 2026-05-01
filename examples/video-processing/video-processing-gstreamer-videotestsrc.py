#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from dotenv import load_dotenv
from loguru import logger

from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.gstreamer.pipeline_source import GStreamerPipelineSource
from blink.runner.types import RunnerArguments
from blink.runner.utils import create_transport
from blink.transports.base_transport import BaseTransport, TransportParams
from blink.transports.daily.transport import DailyParams

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
    "webrtc": lambda: TransportParams(
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with video test source")

    gst = GStreamerPipelineSource(
        pipeline='videotestsrc ! capsfilter caps="video/x-raw,width=1280,height=720,framerate=30/1"',
        out_params=GStreamerPipelineSource.OutputParams(
            video_width=1280, video_height=720, clock_sync=False
        ),
    )

    pipeline = Pipeline(
        [
            gst,  # GStreamer test source
            transport.output(),  # Transport bot output
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with the Blink runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from blink.runner.run import main

    main()

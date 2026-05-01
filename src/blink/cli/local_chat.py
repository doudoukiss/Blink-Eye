#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local terminal chat for Blink development."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Optional

from blink.cli.local_common import (
    LOCAL_LLM_PROVIDERS,
    LocalLLMConfig,
    configure_logging,
    create_local_llm_service,
    get_local_env,
    maybe_load_dotenv,
    resolve_local_language,
    resolve_local_llm_config,
    verify_local_llm_config,
)
from blink.frames.frames import LLMContextFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.console import ConsoleLLMPrinter
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language


@dataclass
class LocalChatConfig:
    """Configuration for the local terminal chat command.

    Args:
        llm: Effective local LLM provider configuration.
        language: Default reply language.
        temperature: Optional sampling temperature override.
        once: Optional single prompt to send before exiting.
        show_banner: Whether to print the startup banner.
        demo_mode: Whether local demo-mode response shaping is active.
        verbose: Whether to emit debug logging.
    """

    llm: LocalLLMConfig
    language: Language
    temperature: Optional[float] = None
    once: Optional[str] = None
    show_banner: bool = True
    demo_mode: bool = False
    verbose: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(f"Run {PROJECT_IDENTITY.display_name} local text chat in the terminal.")
    )
    parser.add_argument(
        "--llm-provider",
        choices=LOCAL_LLM_PROVIDERS,
        help="Local LLM provider. Defaults to ollama.",
    )
    parser.add_argument("--model", help="Provider-relative model name.")
    parser.add_argument("--base-url", help="Provider-relative base URL.")
    parser.add_argument("--system-prompt", help="System prompt for the local assistant.")
    parser.add_argument(
        "--language",
        help="Default reply language for the local assistant, e.g. zh or en.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Optional temperature override passed to the LLM service.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional provider-relative output token budget. OpenAI demo mode defaults to 120.",
    )
    parser.add_argument(
        "--demo-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable polished bounded local demo responses. Also available via BLINK_LOCAL_DEMO_MODE=1.",
    )
    parser.add_argument("--once", help="Send one prompt, print the response, and exit.")
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the interactive startup banner.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging while the chat is running.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LocalChatConfig:
    """Resolve CLI configuration from arguments and environment variables."""
    maybe_load_dotenv()
    language = resolve_local_language(args.language or get_local_env("LANGUAGE"))
    llm = resolve_local_llm_config(
        provider=args.llm_provider,
        model=args.model,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        language=language,
        temperature=args.temperature,
        demo_mode=args.demo_mode,
        max_output_tokens=args.max_output_tokens,
    )

    return LocalChatConfig(
        llm=llm,
        language=language,
        temperature=args.temperature,
        once=args.once,
        show_banner=not args.no_banner,
        demo_mode=llm.demo_mode,
        verbose=args.verbose,
    )


async def _await_response(
    response_queue: asyncio.Queue[str], runner_task: asyncio.Task[None]
) -> str:
    queue_task = asyncio.create_task(response_queue.get())
    done, _ = await asyncio.wait({queue_task, runner_task}, return_when=asyncio.FIRST_COMPLETED)

    if queue_task in done:
        return queue_task.result()

    if runner_task in done:
        queue_task.cancel()
        await asyncio.gather(queue_task, return_exceptions=True)
        try:
            await runner_task
        except asyncio.CancelledError as exc:
            raise RuntimeError(
                "The conversation pipeline stopped before producing a response."
            ) from exc
        raise RuntimeError("The conversation pipeline exited before producing a response.")

    raise RuntimeError("The conversation pipeline did not produce a response.")


async def run_local_chat(config: LocalChatConfig) -> int:
    """Run the local terminal chat session."""
    configure_logging(config.verbose)
    await verify_local_llm_config(config.llm)

    response_queue: asyncio.Queue[str] = asyncio.Queue()
    context = LLMContext()

    llm = create_local_llm_service(config.llm)
    printer = ConsoleLLMPrinter(response_queue=response_queue)

    task = PipelineTask(
        Pipeline(
            [
                llm,
                printer,
            ]
        )
    )

    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    if config.show_banner and not config.once:
        print(
            f"{PROJECT_IDENTITY.display_name} local chat using "
            f"{config.llm.provider} model {config.llm.model!r} "
            f"(demo={'on' if config.demo_mode else 'off'})"
        )
        print("Commands: /reset to clear context, /exit to quit")

    try:
        if config.once:
            context.add_message({"role": "user", "content": config.once})
            await task.queue_frames([LLMContextFrame(context)])
            response = await _await_response(response_queue, runner_task)
            context.add_message({"role": "assistant", "content": response})
            return 0

        while True:
            try:
                user_input = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                break
            if user_input == "/reset":
                context.set_messages([])
                print("context cleared")
                continue

            context.add_message({"role": "user", "content": user_input})
            await task.queue_frames([LLMContextFrame(context)])
            response = await _await_response(response_queue, runner_task)
            context.add_message({"role": "assistant", "content": response})

        return 0
    finally:
        await task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local terminal chat."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_local_chat(config))
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

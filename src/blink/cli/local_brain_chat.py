"""Local terminal chat for Blink development on the full brain runtime."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from blink.brain.memory import memory_tool_prompt
from blink.brain.processors import latest_assistant_text_from_context
from blink.brain.runtime import BrainRuntime, build_session_resolver
from blink.cli.local_chat import LocalChatConfig, _await_response
from blink.cli.local_chat import resolve_config as resolve_local_chat_config
from blink.cli.local_common import (
    DEFAULT_LOCAL_LLM_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LocalLLMConfig,
    configure_logging,
    create_ollama_llm_service,
    verify_ollama,
)
from blink.frames.frames import LLMContextAssistantTimestampFrame, LLMContextFrame
from blink.pipeline.pipeline import Pipeline
from blink.pipeline.runner import PipelineRunner
from blink.pipeline.task import PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from blink.processors.console import ConsoleLLMPrinter
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.project_identity import PROJECT_IDENTITY


@dataclass(init=False)
class LocalBrainChatConfig(LocalChatConfig):
    """Configuration for the brain-backed local terminal chat command."""

    brain_db_path: str | None = None

    def __init__(
        self,
        *,
        llm: LocalLLMConfig | None = None,
        base_url: str | None = None,
        model: str | None = None,
        system_prompt: str = "",
        language,
        temperature: float | None = None,
        once: str | None = None,
        show_banner: bool = True,
        demo_mode: bool | None = None,
        verbose: bool = False,
        brain_db_path: str | None = None,
    ) -> None:
        """Initialize from the current local LLM config or legacy flat fields."""
        resolved_llm = llm or LocalLLMConfig(
            provider=DEFAULT_LOCAL_LLM_PROVIDER,
            model=model or DEFAULT_OLLAMA_MODEL,
            base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
            system_prompt=system_prompt,
            temperature=temperature,
            demo_mode=bool(demo_mode),
        )
        super().__init__(
            llm=resolved_llm,
            language=language,
            temperature=temperature if temperature is not None else resolved_llm.temperature,
            once=once,
            show_banner=show_banner,
            demo_mode=resolved_llm.demo_mode if demo_mode is None else bool(demo_mode),
            verbose=verbose,
        )
        self.brain_db_path = brain_db_path

    @property
    def base_url(self) -> str:
        """Return the provider base URL for legacy callers."""
        return self.llm.base_url or DEFAULT_OLLAMA_BASE_URL

    @property
    def model(self) -> str:
        """Return the provider model for legacy callers."""
        return self.llm.model

    @property
    def system_prompt(self) -> str:
        """Return the configured local system prompt for legacy callers."""
        return self.llm.system_prompt


class _BrainTextResponseCollector(FrameProcessor):
    """Queue one completed assistant response after context aggregation finishes."""

    def __init__(self, *, context: LLMContext, response_queue: asyncio.Queue[str]):
        super().__init__(name="brain-text-response-collector")
        self._context = context
        self._response_queue = response_queue

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextAssistantTimestampFrame):
            await self._response_queue.put(latest_assistant_text_from_context(self._context))
        await self.push_frame(frame, direction)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run {PROJECT_IDENTITY.display_name} local text chat on the brain-backed "
            "runtime against Ollama."
        )
    )
    parser.add_argument("--model", help="Ollama model name.")
    parser.add_argument("--base-url", help="OpenAI-compatible Ollama base URL.")
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
    parser.add_argument(
        "--brain-db-path",
        help="Optional SQLite store path override for the brain-backed runtime.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LocalBrainChatConfig:
    """Resolve CLI configuration from arguments and environment variables."""
    for name in ("llm_provider", "demo_mode", "max_output_tokens"):
        if not hasattr(args, name):
            setattr(args, name, None)
    base_config = resolve_local_chat_config(args)
    return LocalBrainChatConfig(
        llm=base_config.llm,
        language=base_config.language,
        temperature=base_config.temperature,
        once=base_config.once,
        show_banner=base_config.show_banner,
        demo_mode=base_config.demo_mode,
        verbose=base_config.verbose,
        brain_db_path=args.brain_db_path,
    )


async def run_local_brain_chat(config: LocalBrainChatConfig) -> int:
    """Run the local terminal chat session on the brain-backed runtime."""
    configure_logging(config.verbose)
    await verify_ollama(config.base_url, config.model)

    runtime_base_prompt = " ".join(
        part for part in [config.system_prompt, memory_tool_prompt(config.language)] if part
    )
    llm = create_ollama_llm_service(
        base_url=config.base_url,
        model=config.model,
        system_prompt="",
        temperature=config.temperature,
    )
    session_resolver = build_session_resolver(runtime_kind="text")
    brain_runtime = BrainRuntime(
        base_prompt=runtime_base_prompt,
        language=config.language,
        runtime_kind="text",
        session_resolver=session_resolver,
        llm=llm,
        brain_db_path=Path(config.brain_db_path) if config.brain_db_path else None,
    )
    tools = brain_runtime.register_daily_tools()
    context = LLMContext(tools=tools)
    brain_runtime.bind_context(context)
    brain_runtime.start_background_maintenance()
    setattr(context, "blink_brain_runtime", brain_runtime)

    response_queue: asyncio.Queue[str] = asyncio.Queue()
    printer = ConsoleLLMPrinter()
    assistant_aggregator = LLMContextAggregatorPair(context).assistant()
    response_collector = _BrainTextResponseCollector(
        context=context,
        response_queue=response_queue,
    )
    task = PipelineTask(
        Pipeline(
            [
                *brain_runtime.pre_llm_processors,
                llm,
                printer,
                *brain_runtime.post_context_processors,
                assistant_aggregator,
                *brain_runtime.post_aggregation_processors,
                response_collector,
            ]
        )
    )
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    if config.show_banner and not config.once:
        print(f"{PROJECT_IDENTITY.display_name} local brain chat using Ollama model {config.model!r}")
        print("Commands: /reset to clear context, /exit to quit")

    try:
        if config.once:
            context.add_message({"role": "user", "content": config.once})
            await task.queue_frames([LLMContextFrame(context)])
            await _await_response(response_queue, runner_task)
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
            await _await_response(response_queue, runner_task)

        return 0
    finally:
        await task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)
        brain_runtime.close()


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the brain-backed local terminal chat."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_local_brain_chat(config))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

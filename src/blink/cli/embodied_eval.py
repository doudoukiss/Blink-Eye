"""Run bounded embodied eval suites over Blink's simulation-backed runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from blink.brain.evals.embodied_arena import EmbodiedEvalArena
from blink.brain.evals.embodied_scenarios import load_builtin_embodied_eval_suite
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Run bounded embodied eval suites for {PROJECT_IDENTITY.display_name}."
    )
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=("smoke", "benchmark"),
        help="Built-in embodied eval suite to run.",
    )
    parser.add_argument(
        "--scenario",
        help="Optional scenario id filter inside the selected suite.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override. Defaults to artifacts/brain_evals/<suite>.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language passed to the runtime and audit surfaces.",
    )
    parser.add_argument(
        "--runtime-kind",
        default="browser",
        help="Runtime kind used for session id resolution.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    suite = load_builtin_embodied_eval_suite(str(args.suite))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("artifacts") / "brain_evals" / str(args.suite)
    )
    arena = EmbodiedEvalArena(
        language=Language(str(args.language).lower()),
        runtime_kind=str(args.runtime_kind),
    )
    result = await arena.run_suite(
        suite=suite,
        output_dir=output_dir,
        scenario_id=args.scenario,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    any_failures = any(
        not run.expectation_passed
        for report in result.reports
        for run in report.runs
    )
    return 1 if any_failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for bounded embodied eval suites."""
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

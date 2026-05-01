#!/usr/bin/env python3
"""Replay a saved Blink actor-event JSONL trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.interaction.actor_events import replay_actor_trace  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a public-safe Blink actor-event JSONL trace."
    )
    parser.add_argument("trace", type=Path, help="Path to actor-trace-*.jsonl.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Replay the trace and print a compact JSON summary."""
    args = _build_parser().parse_args(argv)
    summary = replay_actor_trace(args.trace)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

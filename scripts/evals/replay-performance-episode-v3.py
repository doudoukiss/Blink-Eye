#!/usr/bin/env python3
"""Replay or build Blink PerformanceEpisodeV3 JSONL artifacts offline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.interaction.performance_episode_v3 import (  # noqa: E402
    PERFORMANCE_EPISODE_V3_SCHEMA_VERSION,
    compile_performance_episode_v3_from_actor_trace,
    render_performance_episode_v3_jsonl,
    replay_performance_episode_v3_jsonl,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay PerformanceEpisodeV3 JSONL or convert actor traces offline."
    )
    parser.add_argument("input", type=Path, help="Episode JSONL or actor trace JSONL.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "episode-jsonl", "actor-trace"),
        default="auto",
        help="Input format. Auto sniffs the first JSON object.",
    )
    parser.add_argument(
        "--output-episodes",
        type=Path,
        help="Optional path to write converted PerformanceEpisodeV3 JSONL.",
    )
    return parser


def _sniff_input_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                return "episode-jsonl"
            if (
                isinstance(payload, dict)
                and payload.get("schema_version") == PERFORMANCE_EPISODE_V3_SCHEMA_VERSION
                and "segments" in payload
            ):
                return "episode-jsonl"
            if isinstance(payload, dict) and payload.get("schema_version") == 2:
                return "actor-trace"
            return "episode-jsonl"
    return "episode-jsonl"


def main(argv: list[str] | None = None) -> int:
    """Replay or convert without browser, model, TTS, camera, or audio services."""
    args = _build_parser().parse_args(argv)
    input_format = (
        _sniff_input_format(args.input) if args.input_format == "auto" else args.input_format
    )

    if input_format == "actor-trace":
        episode = compile_performance_episode_v3_from_actor_trace(args.input)
        if args.output_episodes is not None and episode.sanitizer.passed:
            args.output_episodes.parent.mkdir(parents=True, exist_ok=True)
            args.output_episodes.write_text(
                render_performance_episode_v3_jsonl([episode]),
                encoding="utf-8",
            )
        if args.output_episodes is not None:
            summary_input = args.output_episodes
            if not episode.sanitizer.passed:
                summary = {
                    "schema_version": PERFORMANCE_EPISODE_V3_SCHEMA_VERSION,
                    "episode_count": 0,
                    "sanitizer": episode.sanitizer.as_dict(),
                    "failure_labels": list(episode.failure_labels),
                }
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
        else:
            temp_summary = {
                "schema_version": PERFORMANCE_EPISODE_V3_SCHEMA_VERSION,
                "episode_count": 1,
                "profiles": [episode.profile],
                "languages": [episode.language],
                "tts_labels": [episode.tts_runtime_label],
                "segment_counts": episode.metrics.get("segment_count", 0),
                "failure_labels": list(episode.failure_labels),
                "sanitizer": episode.sanitizer.as_dict(),
            }
            print(json.dumps(temp_summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if episode.sanitizer.passed else 1
    else:
        summary_input = args.input

    summary = replay_performance_episode_v3_jsonl(summary_input)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.sanitizer.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

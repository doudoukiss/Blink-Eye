#!/usr/bin/env python3
"""Replay or build Blink ActorControlFrameV3 JSONL artifacts offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blink.interaction.actor_control_frame_v3 import (
    compile_actor_control_frames_v3,
    load_actor_control_frames_v3_jsonl,
    load_actor_events_for_control_v3,
    render_actor_control_frames_v3_jsonl,
    summarize_actor_control_frames_v3,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay ActorControlFrameV3 JSONL or convert actor traces offline."
    )
    parser.add_argument("input", type=Path, help="Input actor trace or control-frame JSONL.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "actor-trace", "control-jsonl"),
        default="auto",
        help="Input format. Auto detects schema_version=3 control JSONL.",
    )
    parser.add_argument(
        "--output-control-frames",
        type=Path,
        help="Optional path to write converted ActorControlFrameV3 JSONL.",
    )
    return parser


def _detect_input_format(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                return "actor-trace"
            if isinstance(payload, dict) and payload.get("schema_version") == 3:
                return "control-jsonl"
            return "actor-trace"
    return "control-jsonl"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_format = (
        _detect_input_format(args.input) if args.input_format == "auto" else args.input_format
    )

    if input_format == "actor-trace":
        events, violations = load_actor_events_for_control_v3(args.input)
        if violations:
            summary = summarize_actor_control_frames_v3((), safety_violations=violations)
            print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
            return 2
        frames = compile_actor_control_frames_v3(events)
        if args.output_control_frames is not None:
            args.output_control_frames.parent.mkdir(parents=True, exist_ok=True)
            args.output_control_frames.write_text(
                render_actor_control_frames_v3_jsonl(frames),
                encoding="utf-8",
            )
        summary = summarize_actor_control_frames_v3(frame.as_dict() for frame in frames)
        print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    frames, violations = load_actor_control_frames_v3_jsonl(args.input)
    summary = summarize_actor_control_frames_v3(frames, safety_violations=violations)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 2 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

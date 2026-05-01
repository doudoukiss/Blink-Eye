"""Export canonical brain episodes from bounded source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from blink.brain.evals.episode_dataset import export_episode_dataset
from blink.project_identity import PROJECT_IDENTITY


def build_parser() -> argparse.ArgumentParser:
    """Build the episode-export CLI parser."""
    parser = argparse.ArgumentParser(
        description=f"Export canonical brain episodes for {PROJECT_IDENTITY.display_name}."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("embodied-eval", "replay", "live", "practice"),
        help="Artifact family to export from.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one run/scenario/suite eval artifact or one replay artifact JSON.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for episode JSON and dataset manifest artifacts.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for Phase 22 episode export."""
    args = build_parser().parse_args(argv)
    result = export_episode_dataset(
        source=str(args.source),
        input_path=Path(args.input),
        output_dir=Path(args.out),
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

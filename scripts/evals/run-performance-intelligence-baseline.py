#!/usr/bin/env python3
"""Regenerate or check the deterministic Performance Intelligence V3 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals.performance_intelligence_baseline import (  # noqa: E402
    PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH,
    build_performance_intelligence_baseline,
    render_performance_intelligence_baseline_json,
    write_performance_intelligence_baseline,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate Blink's deterministic PerformanceIntelligenceBaseline fixture."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH,
        help="Path to the baseline_profiles.json fixture.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the fixture does not match the deterministic baseline.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Regenerate or validate the checked-in baseline without live services."""
    args = _build_parser().parse_args(argv)
    baseline = build_performance_intelligence_baseline()
    rendered = render_performance_intelligence_baseline_json(baseline)

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Could not read baseline fixture: {args.output}: {exc}", file=sys.stderr)
            return 1
        if existing != rendered:
            print(f"Baseline fixture is out of date: {args.output}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "baseline_id": baseline.baseline_id,
                    "checked": str(args.output),
                    "matched": True,
                    "profiles": [profile.profile for profile in baseline.profiles],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    write_performance_intelligence_baseline(args.output, baseline=baseline)
    print(
        json.dumps(
            {
                "baseline_id": baseline.baseline_id,
                "profiles": [profile.profile for profile in baseline.profiles],
                "written": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

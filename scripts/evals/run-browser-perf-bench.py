#!/usr/bin/env python3
"""Run Blink's deterministic browser/WebRTC performance bench."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals.browser_perf_bench import (  # noqa: E402
    BROWSER_PERF_BENCH_ARTIFACT_DIR,
    BROWSER_PERF_BENCH_PROFILES,
    evaluate_browser_perf_bench_suite,
    render_browser_perf_bench_metrics_rows,
    write_browser_perf_bench_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Blink's offline browser/WebRTC performance bench."
    )
    parser.add_argument(
        "--profile",
        choices=("all", *BROWSER_PERF_BENCH_PROFILES),
        default="all",
        help="Profile fixture subset to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BROWSER_PERF_BENCH_ARTIFACT_DIR,
        help="Directory for latest.json, latest.jsonl, Markdown summary, and rating forms.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the provider-free suite, write artifacts, and print compact metrics."""
    args = _build_parser().parse_args(argv)
    report = evaluate_browser_perf_bench_suite(profile=args.profile)
    artifact_links = write_browser_perf_bench_artifacts(report, output_dir=args.output_dir)
    payload = {
        "suite_id": report.suite_id,
        "profile_filter": report.profile_filter,
        "passed": report.passed,
        "failed_gates": list(report.failed_gates()),
        "aggregate_metrics": report.aggregate_metrics(),
        "artifact_links": artifact_links,
        "metrics_rows": render_browser_perf_bench_metrics_rows(report),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

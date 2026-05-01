#!/usr/bin/env python3
"""Run Blink's deterministic bilingual actor bench and release gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals.bilingual_actor_bench import (  # noqa: E402
    BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR,
    BILINGUAL_ACTOR_RELEASE_THRESHOLD,
    BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD,
    evaluate_bilingual_actor_bench_suite,
    evaluate_bilingual_performance_bench_v3,
    render_bilingual_actor_bench_metrics_rows,
    write_bilingual_actor_bench_artifacts,
    write_bilingual_performance_bench_v3_artifacts,
)
from blink.brain.evals.performance_preferences import (  # noqa: E402
    PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Blink's deterministic bilingual actor bench."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=BILINGUAL_ACTOR_RELEASE_THRESHOLD,
        help="Minimum required v1 compatibility score per quality dimension and profile.",
    )
    parser.add_argument(
        "--v3-threshold",
        type=float,
        default=BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD,
        help="Minimum required V3 performance score per quality dimension and profile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR,
        help="Directory for JSON, JSONL, Markdown, and rating-form artifacts.",
    )
    parser.add_argument(
        "--preferences-dir",
        type=Path,
        default=PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
        help="Optional Phase 11 preference JSONL directory used to enrich V3 evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the provider-free suite, write artifacts, and print compact JSON."""
    args = _build_parser().parse_args(argv)
    report = evaluate_bilingual_actor_bench_suite(threshold=args.threshold)
    artifact_links = write_bilingual_actor_bench_artifacts(report, output_dir=args.output_dir)
    v3_report = evaluate_bilingual_performance_bench_v3(
        threshold=args.v3_threshold,
        preferences_dir=args.preferences_dir,
    )
    v3_artifact_links = write_bilingual_performance_bench_v3_artifacts(
        v3_report,
        output_dir=args.output_dir,
    )
    payload = {
        "suite_id": v3_report.suite_id,
        "run_id": v3_report.run_id,
        "passed": v3_report.passed,
        "threshold": v3_report.release_gate.threshold,
        "hard_blockers": list(v3_report.release_gate.hard_blockers),
        "profile_failures": v3_report.release_gate.profile_failures,
        "aggregate_metrics": v3_report.aggregate_metrics,
        "paired_comparison": v3_report.paired_comparison,
        "artifact_links": v3_artifact_links,
        "compatibility_v1": {
            "suite_id": report.suite_id,
            "run_id": report.run_id,
            "passed": report.passed,
            "threshold": report.release_gate.threshold,
            "hard_blockers": list(report.release_gate.hard_blockers),
            "profile_failures": report.release_gate.profile_failures,
            "artifact_links": artifact_links,
            "metrics_rows": render_bilingual_actor_bench_metrics_rows(report),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if v3_report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

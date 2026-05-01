#!/usr/bin/env python3
"""Run Blink's deterministic autonomy benchmark program."""

from __future__ import annotations

import json
import os

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals import (  # noqa: E402
    evaluate_autonomy_benchmark_program,
    render_autonomy_benchmark_metrics_rows,
    write_autonomy_benchmark_artifacts,
)


def main() -> int:
    """Run the benchmark program, write artifacts, and print compact metrics."""
    report = evaluate_autonomy_benchmark_program()
    paths = write_autonomy_benchmark_artifacts(report)
    payload = {
        "suite_id": report.suite_id,
        "passed": report.passed,
        "aggregate_score": report.aggregate_score,
        "gating_failures": list(report.gating_failures),
        "artifact_links": paths,
        "metrics_rows": render_autonomy_benchmark_metrics_rows(report),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

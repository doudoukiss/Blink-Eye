#!/usr/bin/env python3
"""Run Blink's deterministic frontier behavior workbench eval report."""

from __future__ import annotations

import json
import os

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals import (
    evaluate_frontier_behavior_workbench_suite,
    render_frontier_behavior_workbench_metrics_rows,
    write_frontier_behavior_workbench_artifacts,
)


def main() -> int:
    """Run the provider-free suite, write artifacts, and print compact metrics."""
    report = evaluate_frontier_behavior_workbench_suite()
    write_frontier_behavior_workbench_artifacts(report)
    payload = {
        "suite_id": report.suite_id,
        "passed": report.passed,
        "aggregate_metrics": report.aggregate_metrics(),
        "metrics_rows": render_frontier_behavior_workbench_metrics_rows(report),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

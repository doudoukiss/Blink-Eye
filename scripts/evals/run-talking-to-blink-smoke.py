#!/usr/bin/env python3
"""Run the deterministic Talking To Blink manual smoke suite."""

from __future__ import annotations

import json
import os

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals import (  # noqa: E402
    evaluate_talking_to_blink_manual_suite,
    render_talking_to_blink_manual_metrics_rows,
)


def main() -> int:
    """Run the suite and print compact JSON metrics."""
    report = evaluate_talking_to_blink_manual_suite()
    payload = {
        "suite_id": report.suite_id,
        "passed": report.passed,
        "aggregate_metrics": report.aggregate_metrics(),
        "manual_followups": list(report.manual_followups),
        "metrics_rows": render_talking_to_blink_manual_metrics_rows(report),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

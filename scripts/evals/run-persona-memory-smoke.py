#!/usr/bin/env python3
"""Run Blink's deterministic persona-memory eval smoke suite."""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.brain.evals.persona_memory import (
    evaluate_persona_memory_eval_suite,
    render_persona_memory_metrics_rows,
)


def main() -> int:
    """Run the local provider-free smoke suite and print compact JSON rows."""
    report = evaluate_persona_memory_eval_suite()
    payload = {
        "suite_id": report.suite_id,
        "passed": report.passed,
        "aggregate_metrics": report.aggregate_metrics(),
        "metrics_rows": render_persona_memory_metrics_rows(report),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

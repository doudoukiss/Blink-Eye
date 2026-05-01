"""CLI entrypoint for Blink release-gate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blink.brain.evals.autonomy_benchmark_program import evaluate_autonomy_benchmark_program
from blink.brain.evals.frontier_behavior_workbench import evaluate_frontier_behavior_workbench_suite
from blink.brain.evals.release_gate import (
    RELEASE_GATE_ARTIFACT_DIR,
    build_release_gate_report,
    write_release_gate_artifacts,
)


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else None


def build_parser() -> argparse.ArgumentParser:
    """Build the release-gate CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build Blink's deterministic release-gate report.",
    )
    parser.add_argument("--frontier-report-json", help="Existing frontier behavior report JSON.")
    parser.add_argument("--autonomy-report-json", help="Existing autonomy benchmark report JSON.")
    parser.add_argument("--sim-to-real-json", help="Existing sim-to-real digest JSON.")
    parser.add_argument("--adapter-governance-json", help="Adapter governance projection JSON.")
    parser.add_argument("--rollout-plan-json", help="Adapter routing plan JSON.")
    parser.add_argument("--rollout-budget-json", help="Rollout budget JSON.")
    parser.add_argument("--voice-backend", default="provider-neutral", help="TTS backend label.")
    parser.add_argument("--voice-capabilities-json", help="Voice capabilities override JSON.")
    parser.add_argument(
        "--generated-at",
        default="",
        help="Optional ISO timestamp to include in the report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RELEASE_GATE_ARTIFACT_DIR),
        help="Directory for latest.json and latest.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the report, write artifacts, and print a compact payload."""
    args = build_parser().parse_args(argv)
    frontier_report = _load_json(args.frontier_report_json)
    if frontier_report is None:
        frontier_report = evaluate_frontier_behavior_workbench_suite().as_dict()
    autonomy_report = _load_json(args.autonomy_report_json)
    if autonomy_report is None:
        autonomy_report = evaluate_autonomy_benchmark_program().as_dict()
    output_dir = Path(args.output_dir)
    artifact_links = {
        "json": str(output_dir / "latest.json"),
        "markdown": str(output_dir / "latest.md"),
    }
    report = build_release_gate_report(
        frontier_behavior_report=frontier_report,
        autonomy_benchmark_report=autonomy_report,
        sim_to_real_digest=_load_json(args.sim_to_real_json),
        adapter_governance=_load_json(args.adapter_governance_json),
        rollout_plan=_load_json(args.rollout_plan_json),
        rollout_budget=_load_json(args.rollout_budget_json),
        tts_backend=args.voice_backend,
        voice_capabilities=_load_json(args.voice_capabilities_json),
        generated_at=args.generated_at,
        artifact_links=artifact_links,
    )
    paths = write_release_gate_artifacts(report, output_dir=output_dir)
    payload = {
        "suite_id": report.suite_id,
        "report_id": report.report_id,
        "outcome": report.outcome,
        "passed": report.passed,
        "rollout_reference": report.rollout_reference(),
        "blocking_reason_codes": list(report.blocking_reason_codes),
        "artifact_links": paths,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

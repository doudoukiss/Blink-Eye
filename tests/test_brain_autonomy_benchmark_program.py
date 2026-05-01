from __future__ import annotations

import json

from blink.brain.evals import (
    AUTONOMY_BENCHMARK_FAMILIES,
    AUTONOMY_BENCHMARK_PROGRAM_SUITE_ID,
    BrainAutonomyBenchmarkFamilyScore,
    build_autonomy_benchmark_report,
    evaluate_autonomy_benchmark_program,
    render_autonomy_benchmark_markdown,
    render_autonomy_benchmark_metrics_rows,
    write_autonomy_benchmark_artifacts,
)


def test_autonomy_benchmark_report_structure_and_family_order():
    report = evaluate_autonomy_benchmark_program()
    payload = report.as_dict()

    assert report.suite_id == AUTONOMY_BENCHMARK_PROGRAM_SUITE_ID
    assert report.passed is True
    assert payload["aggregate_score"] == 1.0
    assert [record["family_id"] for record in payload["family_scores"]] == list(
        AUTONOMY_BENCHMARK_FAMILIES
    )
    assert set(payload["aggregate_metrics"]) == {"aggregate_score", *AUTONOMY_BENCHMARK_FAMILIES}
    assert payload["gating_failures"] == []
    assert payload["artifact_links"] == {
        "json": "artifacts/brain_evals/autonomy_benchmark_program/latest.json",
        "markdown": "artifacts/brain_evals/autonomy_benchmark_program/latest.md",
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)) == payload


def test_autonomy_benchmark_family_aggregation_correctness():
    report = build_autonomy_benchmark_report(
        (
            BrainAutonomyBenchmarkFamilyScore(
                family_id="memory_continuity",
                display_name="Memory",
                score=1.0,
                regression_delta=0.0,
                passed=True,
                metrics={"a": 1.0},
            ),
            BrainAutonomyBenchmarkFamilyScore(
                family_id="teaching_quality",
                display_name="Teaching",
                score=0.5,
                regression_delta=-0.01,
                passed=True,
                metrics={"b": 0.5},
            ),
        ),
        artifact_links={"json": "artifact/latest.json"},
    )

    assert report.aggregate_score == 0.75
    assert report.aggregate_metrics() == {
        "aggregate_score": 0.75,
        "memory_continuity": 1.0,
        "teaching_quality": 0.5,
    }
    assert report.regression_deltas() == {
        "memory_continuity": 0.0,
        "teaching_quality": -0.01,
    }
    assert report.passed is True


def test_autonomy_benchmark_gating_logic_does_not_bury_failures():
    report = build_autonomy_benchmark_report(
        (
            BrainAutonomyBenchmarkFamilyScore(
                family_id="rollout_safety",
                display_name="Rollout",
                score=0.9,
                regression_delta=-0.2,
                passed=False,
                gating_failures=("rollback_missing",),
                metrics={"rollout": 0.9},
            ),
        )
    )
    markdown = render_autonomy_benchmark_markdown(report)

    assert report.passed is False
    assert report.gating_failures == (
        "family_failed:rollout_safety",
        "rollout_safety:rollback_missing",
        "rollout_safety:regression_delta_below_floor",
    )
    assert markdown.index("## Gating Failures") < markdown.index("## Family Scores")
    assert "`rollout_safety:rollback_missing`" in markdown


def test_autonomy_benchmark_regression_baselines_gate_release():
    report = evaluate_autonomy_benchmark_program(
        baseline_scores={"voice_interruption_behavior": 1.1}
    )

    assert report.passed is False
    assert "voice_interruption_behavior:regression_delta_below_floor" in report.gating_failures
    voice = next(
        family
        for family in render_autonomy_benchmark_metrics_rows(report)
        if family["family_id"] == "voice_interruption_behavior"
    )
    assert voice["regression_delta"] == -0.1
    assert "regression_delta_below_floor" in voice["gating_failures"]


def test_autonomy_benchmark_artifact_writer_is_stable(tmp_path):
    report = evaluate_autonomy_benchmark_program()
    paths = write_autonomy_benchmark_artifacts(report, output_dir=tmp_path)
    markdown = render_autonomy_benchmark_markdown(report)

    assert paths == {
        "json": str(tmp_path / "latest.json"),
        "markdown": str(tmp_path / "latest.md"),
    }
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == report.as_dict()
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == f"{markdown}\n"
    assert write_autonomy_benchmark_artifacts(report, output_dir=tmp_path) == paths
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == f"{markdown}\n"


def test_autonomy_benchmark_payload_excludes_raw_internals_and_prompts():
    report = evaluate_autonomy_benchmark_program()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True)

    banned = (
        "source_event_ids",
        "source_refs",
        "source_event_id",
        "event_id",
        "raw_json",
        "memory_claim:",
        "brain.db",
        ".db",
        "/tmp",
        "```json",
        "[BLINK_BRAIN_CONTEXT]",
        "private_scratchpad",
        "Traceback",
        "RuntimeError",
        "You are Blink. Always follow",
    )
    assert all(token not in encoded for token in banned)
    assert len(render_autonomy_benchmark_metrics_rows(report)) == len(AUTONOMY_BENCHMARK_FAMILIES)

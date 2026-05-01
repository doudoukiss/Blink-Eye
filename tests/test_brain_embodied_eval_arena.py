from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blink.brain.evals.embodied_arena import EmbodiedEvalArena
from blink.brain.evals.embodied_scenarios import (
    build_benchmark_embodied_eval_suite,
    build_smoke_embodied_eval_suite,
)
from blink.transcriptions.language import Language


def _latest_reflection_draft_path(run) -> str | None:
    audit_json = run.artifact_paths["audit_json"]
    assert audit_json is not None
    payload = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    return payload["continuity_state"]["latest_reflection_draft_path"]


@pytest.mark.asyncio
async def test_embodied_eval_arena_runs_shared_compare_scenario_and_writes_artifacts(tmp_path):
    suite = build_smoke_embodied_eval_suite()
    scenario = suite.scenario("robot_head_look_left_compare")
    assert scenario is not None

    report = await EmbodiedEvalArena(language=Language.EN).run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "compare",
    )

    assert len(report.runs) == 2
    assert all(run.expectation_passed for run in report.runs)
    by_profile = {run.profile_id: run for run in report.runs}
    assert by_profile["incumbent_simulation"].metrics.task_success is True
    assert by_profile["incumbent_simulation"].metrics.preview_only is False
    assert by_profile["candidate_preview"].metrics.task_success is True
    assert by_profile["candidate_preview"].metrics.preview_only is True
    assert report.benchmark_report.comparisons
    comparison = report.benchmark_report.comparisons[0]
    assert comparison.incumbent_profile_id == "incumbent_simulation"
    assert comparison.candidate_profile_id == "candidate_preview"
    assert report.report_json_path is not None and report.report_json_path.exists()
    assert report.report_markdown_path is not None and report.report_markdown_path.exists()

    payload = json.loads(report.report_json_path.read_text(encoding="utf-8"))
    assert payload["scenario"]["scenario_id"] == "robot_head_look_left_compare"
    assert len(payload["runs"]) == 2
    assert all("episode" not in key for key in payload["runs"][0]["artifact_paths"])


@pytest.mark.asyncio
async def test_embodied_eval_arena_busy_fault_run_records_bounded_review_floor(tmp_path):
    suite = build_smoke_embodied_eval_suite()
    scenario = suite.scenario("robot_head_busy_fault")
    assert scenario is not None

    report = await EmbodiedEvalArena(language=Language.EN).run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "busy",
    )

    assert len(report.runs) == 1
    run = report.runs[0]
    assert run.expectation_passed is True
    assert run.metrics.task_success is False
    assert run.metrics.safety_success is True
    assert run.metrics.recovery_count == 0
    assert run.metrics.review_floor_count == 1
    assert run.metrics.trace_status is None
    assert run.artifact_paths["run_json"] is not None
    assert run.artifact_paths["audit_json"] is not None


@pytest.mark.asyncio
async def test_embodied_eval_arena_runs_benchmark_suite_skeleton_with_scenario_filter(tmp_path):
    suite = build_benchmark_embodied_eval_suite()

    result = await EmbodiedEvalArena(language=Language.EN).run_suite(
        suite=suite,
        output_dir=tmp_path / "benchmark",
        scenario_id="robot_head_degraded_comparison",
    )

    assert len(result.reports) == 1
    report = result.reports[0]
    assert report.scenario.scenario_id == "robot_head_degraded_comparison"
    assert len(report.runs) == 2
    assert result.report_json_path is not None and result.report_json_path.exists()
    assert result.report_markdown_path is not None and result.report_markdown_path.exists()


@pytest.mark.asyncio
async def test_embodied_eval_arena_reuses_output_dir_without_stale_runtime_state(tmp_path):
    suite = build_smoke_embodied_eval_suite()
    scenario = suite.scenario("robot_head_look_left_compare")
    assert scenario is not None

    arena = EmbodiedEvalArena(language=Language.EN)
    first = await arena.run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "repeatable",
    )
    first_reflection_paths = [_latest_reflection_draft_path(first_run) for first_run in first.runs]
    second = await arena.run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "repeatable",
    )
    second_reflection_paths = [
        _latest_reflection_draft_path(second_run) for second_run in second.runs
    ]

    assert all(run.expectation_passed for run in first.runs)
    assert all(run.expectation_passed for run in second.runs)
    assert [run.metrics.task_success for run in second.runs] == [True, True]
    assert [run.metrics.review_floor_count for run in second.runs] == [0, 0]
    assert first_reflection_paths == second_reflection_paths


def test_runtime_module_imports_cleanly_after_eval_reexports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from blink.brain.runtime import BrainRuntime; print(BrainRuntime.__name__)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "BrainRuntime" in result.stdout

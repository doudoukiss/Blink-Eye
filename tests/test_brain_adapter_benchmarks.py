from __future__ import annotations

from blink.brain.evals.adapter_benchmarks import build_adapter_benchmark_report
from blink.brain.evals.embodied_metrics import BrainEmbodiedEvalMetricRow


def _row(
    *,
    run_id: str,
    scenario_id: str,
    family: str,
    profile_id: str,
    matrix_index: int,
    backend: str,
    task_success: bool,
    safety_success: bool = True,
    recovery_count: int = 0,
    review_floor_count: int = 0,
    preview_only: bool = False,
) -> BrainEmbodiedEvalMetricRow:
    return BrainEmbodiedEvalMetricRow(
        run_id=run_id,
        scenario_id=scenario_id,
        scenario_family=family,
        scenario_version="v1",
        profile_id=profile_id,
        matrix_index=matrix_index,
        execution_backend=backend,
        perception_backend_id=None,
        world_model_backend_id="local_world_model",
        embodied_policy_backend_id="local_robot_head_policy",
        task_success=task_success,
        safety_success=safety_success,
        preview_only=preview_only,
        operator_intervention_count=0,
        recovery_count=recovery_count,
        step_count=1,
        review_floor_count=review_floor_count,
        skill_reuse_detected=False,
    )


def test_adapter_benchmark_report_builds_deterministic_incumbent_candidate_rows():
    report = build_adapter_benchmark_report(
        (
            _row(
                run_id="candidate",
                scenario_id="robot_head_look_left_compare",
                family="robot_head_single_step",
                profile_id="candidate_preview",
                matrix_index=1,
                backend="preview",
                task_success=True,
                preview_only=True,
            ),
            _row(
                run_id="incumbent",
                scenario_id="robot_head_look_left_compare",
                family="robot_head_single_step",
                profile_id="incumbent_simulation",
                matrix_index=0,
                backend="simulation",
                task_success=True,
            ),
            _row(
                run_id="single",
                scenario_id="robot_head_busy_fault",
                family="robot_head_busy_fault",
                profile_id="fault_busy",
                matrix_index=0,
                backend="fault",
                task_success=False,
                recovery_count=1,
            ),
        )
    )

    assert report.scenario_count == 2
    assert report.compared_family_count == 1
    assert len(report.comparisons) == 1
    row = report.comparisons[0]
    assert row.scenario_id == "robot_head_look_left_compare"
    assert row.incumbent_profile_id == "incumbent_simulation"
    assert row.candidate_profile_id == "candidate_preview"
    assert row.preview_only_delta == 1
    assert row.task_success_delta == 0

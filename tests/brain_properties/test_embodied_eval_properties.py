from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from blink.brain.evals.adapter_benchmarks import build_adapter_benchmark_report
from blink.brain.evals.embodied_metrics import BrainEmbodiedEvalMetricRow

pytestmark = pytest.mark.brain_property


def _rows() -> tuple[BrainEmbodiedEvalMetricRow, ...]:
    return (
        BrainEmbodiedEvalMetricRow(
            run_id="incumbent",
            scenario_id="robot_head_look_left_compare",
            scenario_family="robot_head_single_step",
            scenario_version="v1",
            profile_id="incumbent_simulation",
            matrix_index=0,
            execution_backend="simulation",
            perception_backend_id=None,
            world_model_backend_id="local_world_model",
            embodied_policy_backend_id="local_robot_head_policy",
            task_success=True,
            safety_success=True,
            preview_only=False,
            operator_intervention_count=0,
            recovery_count=0,
            step_count=1,
            review_floor_count=0,
            skill_reuse_detected=False,
        ),
        BrainEmbodiedEvalMetricRow(
            run_id="candidate",
            scenario_id="robot_head_look_left_compare",
            scenario_family="robot_head_single_step",
            scenario_version="v1",
            profile_id="candidate_preview",
            matrix_index=1,
            execution_backend="preview",
            perception_backend_id=None,
            world_model_backend_id="local_world_model",
            embodied_policy_backend_id="local_robot_head_policy",
            task_success=True,
            safety_success=True,
            preview_only=True,
            operator_intervention_count=0,
            recovery_count=0,
            step_count=1,
            review_floor_count=0,
            skill_reuse_detected=False,
        ),
        BrainEmbodiedEvalMetricRow(
            run_id="single",
            scenario_id="robot_head_busy_fault",
            scenario_family="robot_head_busy_fault",
            scenario_version="v1",
            profile_id="fault_busy",
            matrix_index=0,
            execution_backend="fault",
            perception_backend_id=None,
            world_model_backend_id="local_world_model",
            embodied_policy_backend_id="local_robot_head_policy",
            task_success=False,
            safety_success=True,
            preview_only=False,
            operator_intervention_count=0,
            recovery_count=1,
            step_count=1,
            review_floor_count=0,
            skill_reuse_detected=False,
        ),
    )


@given(order=st.permutations((0, 1, 2)))
def test_adapter_benchmark_report_is_invariant_to_input_order(order):
    rows = _rows()
    baseline = build_adapter_benchmark_report(rows).as_dict()
    shuffled = build_adapter_benchmark_report([rows[index] for index in order]).as_dict()

    assert shuffled == baseline

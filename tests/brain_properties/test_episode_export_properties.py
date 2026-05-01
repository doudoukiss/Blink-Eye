from __future__ import annotations

import copy

import pytest
from hypothesis import given
from hypothesis import strategies as st

from blink.brain.evals.dataset_manifest import build_episode_dataset_manifest
from blink.brain.evals.episode_export import (
    BrainEpisodeActionSummary,
    BrainEpisodeOutcomeSummary,
    BrainEpisodePredictionSummary,
    BrainEpisodeRecord,
    BrainEpisodeRehearsalSummary,
    BrainEpisodeSafetySummary,
    build_episode_from_embodied_eval_run_payload,
    build_episode_from_replay_artifact_payload,
)
from blink.brain.evals.failure_clusters import build_failure_clusters

pytestmark = pytest.mark.brain_property


def _episode(
    *,
    episode_id: str,
    scenario_family: str,
    task_success: bool | None,
    safety_success: bool | None,
    review_floor_count: int = 0,
    recovery_count: int = 0,
) -> BrainEpisodeRecord:
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version="brain_episode/v1",
        origin="simulation",
        scenario_id=f"{scenario_family}-scenario",
        scenario_family=scenario_family,
        scenario_version="v1",
        execution_backend="simulation",
        embodied_policy_backend_id="local_robot_head_policy",
        prediction_summary=BrainEpisodePredictionSummary(),
        rehearsal_summary=BrainEpisodeRehearsalSummary(
            calibration_bucket_counts={"aligned": 1} if task_success else {"overconfident": 1}
        ),
        action_summary=BrainEpisodeActionSummary(
            trace_ids=(f"trace-{episode_id}",),
            trace_status="succeeded" if task_success else "failed",
            recovery_count=recovery_count,
        ),
        outcome_summary=BrainEpisodeOutcomeSummary(
            task_success=task_success,
            trace_status="succeeded" if task_success else "failed",
            operator_review_floored=review_floor_count > 0,
            calibration_bucket_counts={"aligned": 1} if task_success else {"overconfident": 1},
        ),
        safety_summary=BrainEpisodeSafetySummary(
            safety_success=safety_success,
            review_floor_count=review_floor_count,
            recovery_count=recovery_count,
        ),
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _eval_run_payload() -> dict:
    return {
        "run_id": "scenario:v1:0:incumbent_simulation",
        "scenario_id": "robot_head_look_left_compare",
        "scenario_family": "robot_head_single_step",
        "profile_id": "incumbent_simulation",
        "matrix_index": 0,
        "goal_status": "completed",
        "planning_outcome": "accepted",
        "metrics": {
            "scenario_version": "v1",
            "execution_backend": "simulation",
            "perception_backend_id": None,
            "world_model_backend_id": "local_world_model",
            "embodied_policy_backend_id": "local_robot_head_policy",
            "task_success": True,
            "safety_success": True,
            "preview_only": False,
            "operator_intervention_count": 0,
            "recovery_count": 0,
            "review_floor_count": 0,
            "trace_status": "succeeded",
            "mismatch_codes": [],
            "repair_codes": [],
        },
        "event_slice": [
            {
                "event_id": "evt-1",
                "event_type": "body.state.updated",
                "ts": "2026-01-01T00:00:00+00:00",
                "source": "runtime",
                "payload": {"presence_scope_key": "browser:presence", "snapshot": {"sensor_health": "healthy"}},
            },
            {
                "event_id": "evt-2",
                "event_type": "scene.changed",
                "ts": "2026-01-01T00:00:01+00:00",
                "source": "perception",
                "payload": {"summary": "Ada is visible.", "presence_scope_key": "browser:presence"},
            },
            {
                "event_id": "evt-3",
                "event_type": "goal.created",
                "ts": "2026-01-01T00:00:02+00:00",
                "source": "executive",
                "payload": {"goal": {"goal_id": "goal-1", "plan_proposal_id": "plan-1"}},
            },
        ],
        "shell_snapshot": {
            "recent_embodied_execution_traces": [{"trace_id": "trace-1", "status": "succeeded", "goal_id": "goal-1"}],
            "recent_embodied_recoveries": [],
        },
        "shell_digest": {
            "predictive_inspection": {
                "active_kind_counts": {"action_outcome": 1},
                "active_confidence_band_counts": {"high": 1},
                "resolution_kind_counts": {},
                "highest_risk_prediction_ids": ["prediction-1"],
            },
            "rehearsal_inspection": {
                "recent_rehearsals": [
                    {
                        "rehearsal_id": "rehearsal-1",
                        "plan_proposal_id": "plan-1",
                        "simulated_backend": "robot_head_simulation",
                        "decision_recommendation": "proceed",
                    }
                ],
                "recent_comparisons": [],
                "recommendation_counts": {"proceed": 1},
                "calibration_bucket_counts": {},
                "observed_outcome_counts": {},
                "risk_code_counts": {},
                "recurrent_mismatch_patterns": {},
            },
            "embodied_inspection": {
                "current_intent": {"goal_id": "goal-1", "plan_proposal_id": "plan-1"},
                "recent_execution_traces": [{"trace_id": "trace-1", "status": "succeeded", "goal_id": "goal-1"}],
                "recent_recoveries": [],
                "recent_low_level_embodied_actions": [{"action_id": "cmd_look_left", "source": "capability"}],
                "last_action_envelope": {"goal_id": "goal-1", "plan_proposal_id": "plan-1"},
            },
        },
        "artifact_paths": {"run_json": "/tmp/run.json", "brain_db": "/tmp/brain.db"},
    }


def _replay_payload() -> dict:
    return {
        "scenario": {"name": "phase1_turn_tool_robot_action", "description": "fixture replay"},
        "events": [
            {
                "event_id": "replay-1",
                "event_type": "body.state.updated",
                "ts": "2026-01-01T00:00:00+00:00",
                "source": "runtime",
                "payload": {"presence_scope_key": "browser:presence", "snapshot": {"sensor_health": "healthy"}},
            },
            {
                "event_id": "replay-2",
                "event_type": "goal.completed",
                "ts": "2026-01-01T00:00:01+00:00",
                "source": "executive",
                "payload": {"goal": {"goal_id": "goal-r"}},
            },
        ],
        "continuity_state": {
            "predictive_world_model": {
                "active_predictions": [],
                "recent_resolutions": [{"prediction_id": "prediction-r"}],
            },
            "counterfactual_rehearsal": {
                "recent_rehearsals": [{"rehearsal_id": "rehearsal-r", "decision_recommendation": "proceed"}],
                "recent_comparisons": [{"comparison_id": "comparison-r", "calibration_bucket": "aligned"}],
            },
            "embodied_executive": {
                "recent_execution_traces": [{"trace_id": "trace-r", "status": "succeeded", "goal_id": "goal-r"}],
            },
            "predictive_digest": {
                "active_kind_counts": {},
                "active_confidence_band_counts": {},
                "resolution_kind_counts": {"confirmed": 1},
                "highest_risk_prediction_ids": [],
            },
            "rehearsal_digest": {
                "recent_rehearsals": [{"rehearsal_id": "rehearsal-r", "decision_recommendation": "proceed"}],
                "recent_comparisons": [{"comparison_id": "comparison-r", "calibration_bucket": "aligned"}],
                "recommendation_counts": {"proceed": 1},
                "calibration_bucket_counts": {"aligned": 1},
                "observed_outcome_counts": {"success": 1},
                "risk_code_counts": {},
                "recurrent_mismatch_patterns": {},
            },
            "embodied_digest": {
                "current_intent": {"goal_id": "goal-r"},
                "current_low_level_executor": "simulation",
                "recent_execution_traces": [{"trace_id": "trace-r", "status": "succeeded", "goal_id": "goal-r"}],
                "recent_recoveries": [],
                "recent_low_level_embodied_actions": [{"action_id": "cmd_blink", "source": "capability"}],
            },
            "planning_digest": {"outcome": "accepted"},
        },
    }


@given(order=st.permutations((0, 1, 2)))
def test_episode_dataset_manifest_and_failure_clusters_are_order_invariant(order):
    episodes = (
        _episode(episode_id="episode-1", scenario_family="family-a", task_success=True, safety_success=True),
        _episode(
            episode_id="episode-2",
            scenario_family="family-b",
            task_success=False,
            safety_success=False,
            review_floor_count=1,
        ),
        _episode(
            episode_id="episode-3",
            scenario_family="family-b",
            task_success=False,
            safety_success=False,
            recovery_count=1,
        ),
    )
    baseline_manifest = build_episode_dataset_manifest(episodes).as_dict()
    baseline_clusters = [row.as_dict() for row in build_failure_clusters(episodes)]
    shuffled = [episodes[index] for index in order]

    assert build_episode_dataset_manifest(shuffled).as_dict() == baseline_manifest
    assert [row.as_dict() for row in build_failure_clusters(shuffled)] == baseline_clusters


def test_equivalent_source_payloads_produce_identical_episode_ids():
    eval_first = build_episode_from_embodied_eval_run_payload(copy.deepcopy(_eval_run_payload()))
    eval_second = build_episode_from_embodied_eval_run_payload(copy.deepcopy(_eval_run_payload()))
    replay_first = build_episode_from_replay_artifact_payload(copy.deepcopy(_replay_payload()))
    replay_second = build_episode_from_replay_artifact_payload(copy.deepcopy(_replay_payload()))

    assert eval_first.episode_id == eval_second.episode_id
    assert replay_first.episode_id == replay_second.episode_id

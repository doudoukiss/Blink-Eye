from __future__ import annotations

from blink.brain.evals.episode_export import (
    BrainEpisodeActionSummary,
    BrainEpisodeOutcomeSummary,
    BrainEpisodePredictionSummary,
    BrainEpisodeRecord,
    BrainEpisodeRehearsalSummary,
    BrainEpisodeSafetySummary,
)
from blink.brain.evals.failure_clusters import build_failure_clusters


def _episode(
    *,
    episode_id: str,
    mismatch_codes: tuple[str, ...] = (),
    repair_codes: tuple[str, ...] = (),
    risk_codes: tuple[str, ...] = (),
    calibration_bucket_counts: dict[str, int] | None = None,
    execution_backend: str = "simulation",
    embodied_policy_backend_id: str = "local_robot_head_policy",
    task_success: bool | None = False,
    safety_success: bool | None = False,
    review_floor_count: int = 0,
    recovery_count: int = 0,
) -> BrainEpisodeRecord:
    prediction_summary = BrainEpisodePredictionSummary()
    rehearsal_summary = BrainEpisodeRehearsalSummary(
        calibration_bucket_counts=dict(calibration_bucket_counts or {})
    )
    action_summary = BrainEpisodeActionSummary(
        trace_ids=(f"trace-{episode_id}",),
        trace_status="failed" if task_success is False else "succeeded",
        mismatch_codes=mismatch_codes,
        repair_codes=repair_codes,
        execution_backend=execution_backend,
        recovery_count=recovery_count,
    )
    outcome_summary = BrainEpisodeOutcomeSummary(
        task_success=task_success,
        trace_status=action_summary.trace_status,
        operator_review_floored=review_floor_count > 0,
        calibration_bucket_counts=dict(calibration_bucket_counts or {}),
    )
    safety_summary = BrainEpisodeSafetySummary(
        safety_success=safety_success,
        review_floor_count=review_floor_count,
        recovery_count=recovery_count,
        risk_codes=risk_codes,
        mismatch_codes=mismatch_codes,
        repair_codes=repair_codes,
    )
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version="brain_episode/v1",
        origin="simulation",
        scenario_id="robot_head_busy_fault",
        scenario_family="robot_head_fault_family",
        scenario_version="v1",
        execution_backend=execution_backend,
        embodied_policy_backend_id=embodied_policy_backend_id,
        prediction_summary=prediction_summary,
        rehearsal_summary=rehearsal_summary,
        action_summary=action_summary,
        outcome_summary=outcome_summary,
        safety_summary=safety_summary,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_failure_clusters_collapse_same_signature_rows():
    first = _episode(
        episode_id="episode-a",
        mismatch_codes=("robot_head_busy",),
        risk_codes=("safety_review",),
        calibration_bucket_counts={"overconfident": 1},
        review_floor_count=1,
    )
    second = _episode(
        episode_id="episode-b",
        mismatch_codes=("robot_head_busy",),
        risk_codes=("safety_review",),
        calibration_bucket_counts={"overconfident": 1},
        review_floor_count=1,
    )

    rows = build_failure_clusters([first, second])

    assert len(rows) == 1
    row = rows[0]
    assert row.episode_count == 2
    assert row.episode_ids == ("episode-a", "episode-b")
    assert row.mismatch_codes == ("robot_head_busy",)
    assert row.risk_codes == ("safety_review",)
    assert row.calibration_bucket_counts == {"overconfident": 2}


def test_failure_clusters_split_different_signatures():
    busy = _episode(
        episode_id="episode-busy",
        mismatch_codes=("robot_head_busy",),
        calibration_bucket_counts={"overconfident": 1},
        review_floor_count=1,
    )
    unsafe = _episode(
        episode_id="episode-unsafe",
        mismatch_codes=("unsafe",),
        calibration_bucket_counts={"underconfident": 1},
        execution_backend="preview",
    )

    rows = build_failure_clusters([busy, unsafe])

    assert len(rows) == 2
    assert {row.episode_ids for row in rows} == {("episode-busy",), ("episode-unsafe",)}
    assert {row.execution_backend for row in rows} == {"simulation", "preview"}


def test_failure_clusters_sort_and_ids_are_stable():
    first = _episode(
        episode_id="episode-2",
        mismatch_codes=("robot_head_busy",),
        calibration_bucket_counts={"overconfident": 1},
        review_floor_count=1,
    )
    second = _episode(
        episode_id="episode-1",
        mismatch_codes=("robot_head_busy",),
        calibration_bucket_counts={"overconfident": 1},
        review_floor_count=1,
    )
    third = _episode(
        episode_id="episode-3",
        mismatch_codes=("unsafe",),
        repair_codes=("return_neutral",),
        recovery_count=1,
    )

    baseline = [row.as_dict() for row in build_failure_clusters([first, second, third])]
    reversed_rows = [row.as_dict() for row in build_failure_clusters([third, second, first])]

    assert reversed_rows == baseline

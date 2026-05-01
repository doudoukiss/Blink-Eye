from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.counterfactual_rehearsal_digest import build_counterfactual_rehearsal_digest
from blink.brain.counterfactuals import (
    append_outcome_comparison,
    append_rehearsal_request,
    append_rehearsal_result,
)
from blink.brain.projections import (
    BrainActionOutcomeComparisonRecord,
    BrainActionRehearsalRequest,
    BrainActionRehearsalResult,
    BrainCalibrationBucket,
    BrainCounterfactualRehearsalKind,
    BrainCounterfactualRehearsalProjection,
    BrainObservedActionOutcomeKind,
    BrainPredictionConfidenceBand,
    BrainRehearsalDecisionRecommendation,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _request(rehearsal_id: str, *, index: int) -> BrainActionRehearsalRequest:
    return BrainActionRehearsalRequest(
        rehearsal_id=rehearsal_id,
        goal_id=f"goal-{index}",
        commitment_id=f"commitment-{index}",
        plan_proposal_id=f"proposal-{index}",
        step_index=index,
        candidate_action_id=f"cmd-{index}",
        fallback_action_ids=["cmd_return_neutral"],
        rehearsal_kind=BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
        simulated_backend="robot_head_simulation",
        requested_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _result(
    rehearsal_id: str,
    *,
    index: int,
    recommendation: str,
) -> BrainActionRehearsalResult:
    return BrainActionRehearsalResult(
        rehearsal_id=rehearsal_id,
        goal_id=f"goal-{index}",
        commitment_id=f"commitment-{index}",
        plan_proposal_id=f"proposal-{index}",
        step_index=index,
        candidate_action_id=f"cmd-{index}",
        fallback_action_ids=["cmd_return_neutral"],
        rehearsal_kind=BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
        simulated_backend="robot_head_simulation",
        predicted_success_probability=0.65,
        confidence_band=BrainPredictionConfidenceBand.MEDIUM.value,
        decision_recommendation=recommendation,
        summary=f"Rehearsal {index}",
        completed_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


def _comparison(
    rehearsal_id: str,
    *,
    index: int,
    observed_outcome_kind: str,
    calibration_bucket: str,
) -> BrainActionOutcomeComparisonRecord:
    return BrainActionOutcomeComparisonRecord(
        comparison_id=f"comparison-{index}-{rehearsal_id}",
        rehearsal_id=rehearsal_id,
        goal_id=f"goal-{index}",
        commitment_id=f"commitment-{index}",
        plan_proposal_id=f"proposal-{index}",
        step_index=index,
        candidate_action_id=f"cmd-{index}",
        observed_outcome_kind=observed_outcome_kind,
        predicted_success_probability=0.65,
        confidence_band=BrainPredictionConfidenceBand.MEDIUM.value,
        decision_recommendation=BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
        calibration_bucket=calibration_bucket,
        comparison_summary=f"Comparison {index}",
        observed_event_id=f"evt-{index}",
        compared_at=f"2026-01-01T00:00:{index:02d}+00:00",
        updated_at=f"2026-01-01T00:00:{index:02d}+00:00",
    )


@given(
    rehearsal_ids=st.lists(
        st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=8),
        min_size=1,
        max_size=8,
    )
)
@_SETTINGS
def test_rehearsal_projection_keeps_one_open_request_per_rehearsal_id(rehearsal_ids):
    projection = BrainCounterfactualRehearsalProjection(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    for index, rehearsal_id in enumerate(rehearsal_ids):
        append_rehearsal_request(projection, _request(rehearsal_id, index=index))

    assert len(projection.open_rehearsal_ids) == len(set(rehearsal_ids))
    assert len(set(projection.open_rehearsal_ids)) == len(projection.open_rehearsal_ids)


@given(
    recommendations=st.lists(
        st.sampled_from(
            [
                BrainRehearsalDecisionRecommendation.PROCEED.value,
                BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
                BrainRehearsalDecisionRecommendation.WAIT.value,
                BrainRehearsalDecisionRecommendation.REPAIR.value,
            ]
        ),
        min_size=1,
        max_size=6,
    ),
    calibration_buckets=st.lists(
        st.sampled_from(
            [
                BrainCalibrationBucket.ALIGNED.value,
                BrainCalibrationBucket.OVERCONFIDENT.value,
                BrainCalibrationBucket.UNDERCONFIDENT.value,
                BrainCalibrationBucket.NOT_CALIBRATED.value,
            ]
        ),
        min_size=1,
        max_size=6,
    ),
)
@_SETTINGS
def test_rehearsal_digest_counts_match_projection_rows(recommendations, calibration_buckets):
    projection = BrainCounterfactualRehearsalProjection(
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    rehearsal_ids: list[str] = []
    for index, recommendation in enumerate(recommendations):
        rehearsal_id = f"rehearsal-{index}"
        rehearsal_ids.append(rehearsal_id)
        append_rehearsal_result(
            projection,
            _result(rehearsal_id, index=index, recommendation=recommendation),
        )
    for index, calibration_bucket in enumerate(calibration_buckets):
        append_outcome_comparison(
            projection,
            _comparison(
                rehearsal_ids[index % len(rehearsal_ids)],
                index=index,
                observed_outcome_kind=(
                    BrainObservedActionOutcomeKind.SUCCESS.value
                    if calibration_bucket != BrainCalibrationBucket.NOT_CALIBRATED.value
                    else BrainObservedActionOutcomeKind.PREVIEW_ONLY.value
                ),
                calibration_bucket=calibration_bucket,
            ),
        )

    digest = build_counterfactual_rehearsal_digest(
        counterfactual_rehearsal=projection.as_dict(),
    )

    assert len(set(projection.recent_rehearsal_ids)) == len(projection.recent_rehearsal_ids)
    assert len(set(projection.recent_comparison_ids)) == len(projection.recent_comparison_ids)
    assert sum(digest["recommendation_counts"].values()) == len(projection.recent_rehearsals)
    assert sum(digest["calibration_bucket_counts"].values()) == len(projection.recent_comparisons)

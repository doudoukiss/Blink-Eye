from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

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

pytestmark = pytest.mark.brain_stateful


class CounterfactualRehearsalStateMachine(RuleBasedStateMachine):
    """Exercise bounded rehearsal request/result/comparison lifecycles."""

    requests = Bundle("requests")
    completed = Bundle("completed")

    def __init__(self):
        super().__init__()
        self.projection = BrainCounterfactualRehearsalProjection(
            scope_key="thread-1",
            presence_scope_key="browser:presence",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        self._counter = 0

    @rule(target=requests)
    def request_rehearsal(self):
        rehearsal_id = f"rehearsal-{self._counter}"
        self._counter += 1
        append_rehearsal_request(
            self.projection,
            BrainActionRehearsalRequest(
                rehearsal_id=rehearsal_id,
                goal_id=f"goal-{self._counter}",
                commitment_id=f"commitment-{self._counter}",
                plan_proposal_id=f"proposal-{self._counter}",
                step_index=self._counter,
                candidate_action_id="cmd_look_left",
                fallback_action_ids=["cmd_return_neutral"],
                rehearsal_kind=BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
                simulated_backend="robot_head_simulation",
                requested_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
                updated_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
            ),
        )
        return rehearsal_id

    @rule(
        target=completed,
        rehearsal_id=requests,
        recommendation=st.sampled_from(
            [
                BrainRehearsalDecisionRecommendation.PROCEED.value,
                BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
                BrainRehearsalDecisionRecommendation.WAIT.value,
                BrainRehearsalDecisionRecommendation.REPAIR.value,
                BrainRehearsalDecisionRecommendation.ABORT.value,
            ]
        ),
    )
    def complete_rehearsal(self, rehearsal_id: str, recommendation: str):
        append_rehearsal_result(
            self.projection,
            BrainActionRehearsalResult(
                rehearsal_id=rehearsal_id,
                goal_id=f"goal-{rehearsal_id}",
                commitment_id=f"commitment-{rehearsal_id}",
                plan_proposal_id=f"proposal-{rehearsal_id}",
                step_index=self._counter,
                candidate_action_id="cmd_look_left",
                fallback_action_ids=["cmd_return_neutral"],
                rehearsal_kind=BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
                simulated_backend="robot_head_simulation",
                predicted_success_probability=0.65,
                confidence_band=BrainPredictionConfidenceBand.MEDIUM.value,
                decision_recommendation=recommendation,
                summary=f"Completed {rehearsal_id}",
                completed_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
                updated_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
            ),
        )
        return rehearsal_id

    @rule(
        rehearsal_id=completed,
        observed_outcome_kind=st.sampled_from(
            [
                BrainObservedActionOutcomeKind.SUCCESS.value,
                BrainObservedActionOutcomeKind.FAILURE.value,
                BrainObservedActionOutcomeKind.PREVIEW_ONLY.value,
            ]
        ),
        calibration_bucket=st.sampled_from(
            [
                BrainCalibrationBucket.ALIGNED.value,
                BrainCalibrationBucket.OVERCONFIDENT.value,
                BrainCalibrationBucket.UNDERCONFIDENT.value,
                BrainCalibrationBucket.NOT_CALIBRATED.value,
            ]
        ),
    )
    def compare_outcome(
        self,
        rehearsal_id: str,
        observed_outcome_kind: str,
        calibration_bucket: str,
    ):
        append_outcome_comparison(
            self.projection,
            BrainActionOutcomeComparisonRecord(
                comparison_id=f"comparison-{rehearsal_id}-{observed_outcome_kind}",
                rehearsal_id=rehearsal_id,
                goal_id=f"goal-{rehearsal_id}",
                commitment_id=f"commitment-{rehearsal_id}",
                plan_proposal_id=f"proposal-{rehearsal_id}",
                step_index=self._counter,
                candidate_action_id="cmd_look_left",
                observed_outcome_kind=observed_outcome_kind,
                predicted_success_probability=0.65,
                confidence_band=BrainPredictionConfidenceBand.MEDIUM.value,
                decision_recommendation=BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
                calibration_bucket=calibration_bucket,
                comparison_summary=f"Compared {rehearsal_id}",
                observed_event_id=f"evt-{rehearsal_id}-{observed_outcome_kind}",
                compared_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
                updated_at=f"2026-01-01T00:00:{self._counter:02d}+00:00",
            ),
        )

    @invariant()
    def open_requests_stay_unique(self):
        assert len(set(self.projection.open_rehearsal_ids)) == len(self.projection.open_rehearsal_ids)

    @invariant()
    def terminal_rows_stay_unique(self):
        assert len(set(self.projection.recent_rehearsal_ids)) == len(
            self.projection.recent_rehearsal_ids
        )
        assert len(set(self.projection.recent_comparison_ids)) == len(
            self.projection.recent_comparison_ids
        )


TestCounterfactualRehearsalStateMachine = CounterfactualRehearsalStateMachine.TestCase
TestCounterfactualRehearsalStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)

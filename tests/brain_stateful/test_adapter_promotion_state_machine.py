from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain.adapters.cards import BrainAdapterPromotionState
from blink.brain.evals.adapter_promotion import (
    BrainAdapterGovernanceProjection,
    append_adapter_benchmark_report,
    append_adapter_card,
    append_adapter_promotion_decision,
    apply_promotion_decision_to_card,
    build_adapter_promotion_decision,
    build_embodied_policy_benchmark_comparison_report,
    with_card_benchmark_summary,
)
from tests.phase24_fixtures import build_candidate_card, make_metric_row

pytestmark = pytest.mark.brain_stateful


def _state_rank(state: str) -> int:
    return {
        BrainAdapterPromotionState.EXPERIMENTAL.value: 0,
        BrainAdapterPromotionState.SHADOW.value: 1,
        BrainAdapterPromotionState.CANARY.value: 2,
        BrainAdapterPromotionState.DEFAULT.value: 3,
        BrainAdapterPromotionState.ROLLED_BACK.value: 4,
    }[state]


class AdapterPromotionStateMachine(RuleBasedStateMachine):
    """Exercise monotonic adapter promotion and rollback governance."""

    def __init__(self):
        super().__init__()
        self._counter = 2
        self.projection = BrainAdapterGovernanceProjection(scope_key="thread-phase24")
        self.card = build_candidate_card(updated_at="2026-04-22T00:00:01+00:00")
        append_adapter_card(self.projection, self.card)
        self._state_history = [self.card.promotion_state]
        self.last_report = None
        self.last_decision = None

    def _ts(self) -> str:
        self._counter += 1
        return f"2026-04-22T00:00:{self._counter:02d}+00:00"

    def _report(self, *, passing: bool, safety_regression: bool):
        if safety_regression:
            rows = [
                make_metric_row(
                    run_id="incumbent-safety",
                    scenario_id="robot_head_safety_compare",
                    scenario_family="robot_head_safety_critical",
                    profile_id="incumbent",
                    matrix_index=0,
                    embodied_policy_backend_id="local_robot_head_policy",
                    task_success=True,
                    safety_success=True,
                ),
                make_metric_row(
                    run_id="candidate-safety",
                    scenario_id="robot_head_safety_compare",
                    scenario_family="robot_head_safety_critical",
                    profile_id="candidate",
                    matrix_index=1,
                    embodied_policy_backend_id="candidate_robot_head_policy",
                    task_success=False,
                    safety_success=False,
                    mismatch_codes=("unsafe",),
                ),
            ]
            target_families = ("robot_head_safety_critical",)
        elif passing:
            rows = [
                make_metric_row(
                    run_id="incumbent",
                    scenario_id="robot_head_look_left_compare",
                    scenario_family="robot_head_single_step",
                    profile_id="incumbent",
                    matrix_index=0,
                    embodied_policy_backend_id="local_robot_head_policy",
                    task_success=False,
                    review_floor_count=1,
                ),
                make_metric_row(
                    run_id="candidate",
                    scenario_id="robot_head_look_left_compare",
                    scenario_family="robot_head_single_step",
                    profile_id="candidate",
                    matrix_index=1,
                    embodied_policy_backend_id="candidate_robot_head_policy",
                    task_success=True,
                ),
            ]
            target_families = ("robot_head_single_step",)
        else:
            rows = []
            target_families = ("robot_head_single_step",)
        return build_embodied_policy_benchmark_comparison_report(
            rows,
            incumbent_backend_id="local_robot_head_policy",
            candidate_backend_id="candidate_robot_head_policy",
            target_families=target_families,
            smoke_suite_green=passing and not safety_regression,
            updated_at=self._ts(),
        )

    @rule(passing=st.booleans(), safety_regression=st.booleans())
    def publish_report(self, passing: bool, safety_regression: bool):
        report = self._report(passing=passing, safety_regression=safety_regression)
        append_adapter_benchmark_report(self.projection, report)
        self.card = with_card_benchmark_summary(self.card, report)
        append_adapter_card(self.projection, self.card)
        self.last_report = report

    @rule()
    def promote(self):
        if self.card.promotion_state == BrainAdapterPromotionState.ROLLED_BACK.value:
            return
        decision = build_adapter_promotion_decision(
            card=self.card,
            outcome="promote",
            report=self.last_report,
            updated_at=self._ts(),
        )
        append_adapter_promotion_decision(self.projection, decision)
        self.card = apply_promotion_decision_to_card(self.card, decision)
        append_adapter_card(self.projection, self.card)
        self.last_decision = decision
        self._state_history.append(self.card.promotion_state)

    @rule()
    def hold(self):
        decision = build_adapter_promotion_decision(
            card=self.card,
            outcome="hold",
            report=self.last_report,
            blocked_reason_codes=("missing_shared_family_evidence",),
            updated_at=self._ts(),
        )
        append_adapter_promotion_decision(self.projection, decision)
        self.card = apply_promotion_decision_to_card(self.card, decision)
        append_adapter_card(self.projection, self.card)
        self.last_decision = decision
        self._state_history.append(self.card.promotion_state)

    @rule()
    def rollback(self):
        decision = build_adapter_promotion_decision(
            card=self.card,
            outcome="rollback",
            report=self.last_report,
            blocked_reason_codes=("safety_critical_regression",),
            updated_at=self._ts(),
        )
        append_adapter_promotion_decision(self.projection, decision)
        self.card = apply_promotion_decision_to_card(self.card, decision)
        append_adapter_card(self.projection, self.card)
        self.last_decision = decision
        self._state_history.append(self.card.promotion_state)

    @rule()
    def duplicate_last_decision(self):
        if self.last_decision is None:
            return
        append_adapter_promotion_decision(self.projection, self.last_decision)

    @invariant()
    def states_remain_unique_and_monotonic(self):
        assert len(self.projection.adapter_cards) == 1
        decision_ids = [record.decision_id for record in self.projection.recent_decisions]
        assert len(set(decision_ids)) == len(decision_ids)
        ranks = [_state_rank(state) for state in self._state_history]
        for previous, current in zip(ranks, ranks[1:]):
            assert current >= previous
            if current != _state_rank(BrainAdapterPromotionState.ROLLED_BACK.value):
                assert current - previous <= 1
        if BrainAdapterPromotionState.ROLLED_BACK.value in self._state_history:
            assert self._state_history[-1] == BrainAdapterPromotionState.ROLLED_BACK.value


TestAdapterPromotionStateMachine = AdapterPromotionStateMachine.TestCase
TestAdapterPromotionStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)

from __future__ import annotations

from blink.brain.adapters.cards import BrainAdapterPromotionState, build_default_adapter_cards
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
from blink.brain.evals.sim_to_real_report import (
    build_sim_to_real_digest,
    build_sim_to_real_readiness_reports,
)
from tests.phase24_fixtures import build_candidate_card, make_metric_row


def test_sim_to_real_report_stays_governance_only_for_shadow_and_canary():
    projection = BrainAdapterGovernanceProjection(scope_key="thread-phase24")
    for card in build_default_adapter_cards(updated_at="2026-04-22T00:00:01+00:00"):
        append_adapter_card(projection, card)

    candidate = build_candidate_card(updated_at="2026-04-22T00:00:02+00:00")
    report = build_embodied_policy_benchmark_comparison_report(
        [
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
        ],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_single_step",),
        smoke_suite_green=True,
        updated_at="2026-04-22T00:00:03+00:00",
    )
    shadow = apply_promotion_decision_to_card(
        with_card_benchmark_summary(candidate, report),
        build_adapter_promotion_decision(
            card=with_card_benchmark_summary(candidate, report),
            outcome="promote",
            report=report,
            updated_at="2026-04-22T00:00:04+00:00",
        ),
    )
    append_adapter_card(projection, shadow)
    append_adapter_benchmark_report(projection, report)

    reports = build_sim_to_real_readiness_reports(adapter_governance=projection)
    shadow_report = next(
        report for report in reports if report.backend_id == "candidate_robot_head_policy"
    )
    assert shadow_report.governance_only is True
    assert shadow_report.shadow_ready is True
    assert shadow_report.canary_ready is False
    assert shadow_report.default_ready is False
    assert shadow_report.promotion_state == BrainAdapterPromotionState.SHADOW.value


def test_sim_to_real_digest_surfaces_weak_families_and_blocked_reasons():
    projection = BrainAdapterGovernanceProjection(scope_key="thread-phase24")
    candidate = build_candidate_card(updated_at="2026-04-22T00:00:02+00:00")
    report = build_embodied_policy_benchmark_comparison_report(
        [
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
        ],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_safety_critical",),
        smoke_suite_green=False,
        updated_at="2026-04-22T00:00:03+00:00",
    )
    candidate = with_card_benchmark_summary(candidate, report)
    rollback = build_adapter_promotion_decision(
        card=candidate,
        outcome="rollback",
        report=report,
        updated_at="2026-04-22T00:00:04+00:00",
    )
    candidate = apply_promotion_decision_to_card(candidate, rollback)
    append_adapter_card(projection, candidate)
    append_adapter_benchmark_report(projection, report)
    append_adapter_promotion_decision(projection, rollback)

    digest = build_sim_to_real_digest(adapter_governance=projection)
    row = digest["readiness_reports"][0]

    assert digest["blocked_reason_counts"]["safety_critical_regression"] >= 1
    assert row["rollback_required"] is True
    assert "robot_head_safety_critical" in row["weak_families"]
    assert row["governance_only"] is True

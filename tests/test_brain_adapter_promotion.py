from __future__ import annotations

from blink.brain.adapters.cards import BrainAdapterPromotionState
from blink.brain.evals.adapter_promotion import (
    BrainAdapterGovernanceProjection,
    append_adapter_card,
    append_adapter_promotion_decision,
    apply_promotion_decision_to_card,
    build_adapter_promotion_decision,
    build_embodied_policy_benchmark_comparison_report,
    with_card_benchmark_summary,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from tests.phase24_fixtures import build_candidate_card, make_metric_row


def test_adapter_cards_seed_deterministic_local_baselines(tmp_path):
    store = BrainStore(path=tmp_path / "brain_phase24_seed.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase24-seed")

    store.ensure_default_adapter_cards(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        updated_at="2026-04-22T00:00:01+00:00",
    )
    event_count_before = int(
        store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    )
    projection = store.get_adapter_governance_projection(scope_key=session_ids.thread_id)

    assert [
        (card.adapter_family, card.backend_id, card.promotion_state)
        for card in projection.adapter_cards
    ] == [
        ("embodied_policy", "local_robot_head_policy", "default"),
        ("perception", "local_perception", "default"),
        ("world_model", "local_world_model", "default"),
    ]
    assert projection.state_counts == {"default": 3}

    store.ensure_default_adapter_cards(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        updated_at="2026-04-22T00:00:01+00:00",
    )
    event_count_after = int(store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0])
    assert event_count_after == event_count_before


def test_embodied_policy_report_requires_explicit_decisions_to_reach_default():
    candidate_card = build_candidate_card(updated_at="2026-04-22T00:00:02+00:00")
    report = build_embodied_policy_benchmark_comparison_report(
        [
            make_metric_row(
                run_id="incumbent-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=False,
                review_floor_count=1,
            ),
            make_metric_row(
                run_id="candidate-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="candidate",
                matrix_index=1,
                embodied_policy_backend_id="candidate_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="incumbent-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="candidate-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
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

    assert report.benchmark_passed is True
    assert not report.blocked_reason_codes
    primary_family = next(
        row for row in report.family_rows if row.scenario_family == "robot_head_single_step"
    )
    assert primary_family.task_success_delta_points >= 100.0

    candidate_with_report = with_card_benchmark_summary(candidate_card, report)
    assert candidate_with_report.promotion_state == BrainAdapterPromotionState.EXPERIMENTAL.value

    first = build_adapter_promotion_decision(
        card=candidate_with_report,
        outcome="promote",
        report=report,
        updated_at="2026-04-22T00:00:04+00:00",
    )
    after_first = apply_promotion_decision_to_card(candidate_with_report, first)
    assert first.to_state == BrainAdapterPromotionState.SHADOW.value
    assert after_first.promotion_state == BrainAdapterPromotionState.SHADOW.value

    second = build_adapter_promotion_decision(
        card=after_first,
        outcome="promote",
        report=report,
        updated_at="2026-04-22T00:00:05+00:00",
    )
    after_second = apply_promotion_decision_to_card(after_first, second)
    assert second.to_state == BrainAdapterPromotionState.CANARY.value
    assert after_second.promotion_state == BrainAdapterPromotionState.CANARY.value

    third = build_adapter_promotion_decision(
        card=after_second,
        outcome="promote",
        report=report,
        updated_at="2026-04-22T00:00:06+00:00",
    )
    after_third = apply_promotion_decision_to_card(after_second, third)
    assert third.to_state == BrainAdapterPromotionState.DEFAULT.value
    assert after_third.promotion_state == BrainAdapterPromotionState.DEFAULT.value


def test_adapter_promotion_blocks_on_safety_regression_and_missing_evidence():
    candidate_card = build_candidate_card(updated_at="2026-04-22T00:00:02+00:00")
    safety_regression = build_embodied_policy_benchmark_comparison_report(
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
                trace_status="failed",
                mismatch_codes=("unsafe",),
            ),
        ],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_safety_critical",),
        smoke_suite_green=True,
        updated_at="2026-04-22T00:00:03+00:00",
    )
    assert "safety_critical_regression" in safety_regression.blocked_reason_codes
    assert "new_critical_failure_signature" in safety_regression.blocked_reason_codes
    assert safety_regression.benchmark_passed is False

    held = build_adapter_promotion_decision(
        card=candidate_card,
        outcome="hold",
        report=safety_regression,
        updated_at="2026-04-22T00:00:04+00:00",
    )
    blocked_card = apply_promotion_decision_to_card(candidate_card, held)
    assert blocked_card.promotion_state == BrainAdapterPromotionState.EXPERIMENTAL.value

    missing_evidence = build_embodied_policy_benchmark_comparison_report(
        [],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_single_step",),
        smoke_suite_green=True,
        updated_at="2026-04-22T00:00:05+00:00",
    )
    assert "missing_shared_family_evidence" in missing_evidence.blocked_reason_codes
    assert missing_evidence.benchmark_passed is False

    rollback = build_adapter_promotion_decision(
        card=blocked_card,
        outcome="rollback",
        report=safety_regression,
        updated_at="2026-04-22T00:00:06+00:00",
    )
    rolled_back = apply_promotion_decision_to_card(blocked_card, rollback)
    assert rollback.to_state == BrainAdapterPromotionState.ROLLED_BACK.value
    assert rolled_back.promotion_state == BrainAdapterPromotionState.ROLLED_BACK.value


def test_adapter_governance_projection_dedupes_duplicate_decisions():
    candidate_card = build_candidate_card(updated_at="2026-04-22T00:00:02+00:00")
    projection = BrainAdapterGovernanceProjection(scope_key="thread-phase24")
    append_adapter_card(projection, candidate_card)

    hold = build_adapter_promotion_decision(
        card=candidate_card,
        outcome="hold",
        blocked_reason_codes=("missing_shared_family_evidence",),
        updated_at="2026-04-22T00:00:03+00:00",
    )
    append_adapter_promotion_decision(projection, hold)
    append_adapter_promotion_decision(projection, hold)

    assert len(projection.recent_decisions) == 1
    assert projection.recent_decisions[0].decision_outcome == "hold"

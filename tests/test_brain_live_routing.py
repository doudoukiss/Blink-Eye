from dataclasses import replace

import pytest

from blink.brain.adapters import LOCAL_WORLD_MODEL_DESCRIPTOR
from blink.brain.adapters.cards import (
    BrainAdapterBenchmarkSummary,
    BrainAdapterFamily,
    BrainAdapterPromotionState,
    build_adapter_card,
)
from blink.brain.adapters.live_controller import LiveRoutingController
from blink.brain.adapters.live_routing import (
    AdapterRoutingPlan,
    AdapterRoutingState,
    RolloutDecisionRecord,
    active_routing_plan_for_family,
    apply_rollout_decision,
    build_adapter_routing_plan,
    build_rollout_decision,
)
from blink.brain.adapters.rollout_budget import RolloutBudget, build_rollout_budget
from blink.brain.evals import build_live_routing_report, render_live_routing_report_markdown


def _candidate_card(
    *,
    adapter_family: str = BrainAdapterFamily.WORLD_MODEL.value,
    backend_id: str = "candidate_world_model",
    promotion_state: str = BrainAdapterPromotionState.CANARY.value,
    benchmark_passed: bool | None = True,
    smoke_suite_green: bool | None = True,
    scenario_count: int = 4,
    compared_family_count: int = 2,
):
    descriptor = replace(
        LOCAL_WORLD_MODEL_DESCRIPTOR,
        backend_id=backend_id,
        backend_version="candidate-v1",
    )
    return build_adapter_card(
        adapter_family=adapter_family,
        descriptor=descriptor,
        promotion_state=promotion_state,
        latest_benchmark_summary=BrainAdapterBenchmarkSummary(
            report_id=f"report-{adapter_family}-{backend_id}",
            adapter_family=adapter_family,
            scenario_count=scenario_count,
            compared_family_count=compared_family_count,
            benchmark_passed=benchmark_passed,
            smoke_suite_green=smoke_suite_green,
            target_families=("baseline",),
            updated_at="2026-04-23T00:00:00+00:00",
        ),
        updated_at="2026-04-23T00:00:00+00:00",
    )


def _budget(
    *,
    adapter_family: str = BrainAdapterFamily.WORLD_MODEL.value,
    max_traffic_fraction: float = 0.1,
    allow_embodied_live: bool = False,
) -> RolloutBudget:
    return build_rollout_budget(
        adapter_family=adapter_family,
        max_traffic_fraction=max_traffic_fraction,
        eligible_scopes=("local", "lab"),
        allow_embodied_live=allow_embodied_live,
        minimum_scenario_count=2,
        minimum_compared_family_count=1,
    )


def _plan(
    *,
    adapter_family: str = BrainAdapterFamily.WORLD_MODEL.value,
    budget: RolloutBudget | None = None,
    embodied_live: bool = False,
    recovery_floor_passed: bool = True,
) -> AdapterRoutingPlan:
    rollout_budget = budget or _budget(adapter_family=adapter_family)
    return build_adapter_routing_plan(
        card=_candidate_card(adapter_family=adapter_family),
        budget=rollout_budget,
        incumbent_backend_id="local_world_model",
        scope_key="local",
        created_at="2026-04-23T00:00:00+00:00",
        expires_at="2026-04-24T00:00:00+00:00",
        embodied_live=embodied_live,
        operator_acknowledged=True,
        recovery_floor_passed=recovery_floor_passed,
    )


def test_live_routing_state_transitions_require_explicit_decisions():
    budget = _budget(max_traffic_fraction=1.0)
    plan = _plan(budget=budget)

    direct_default = build_rollout_decision(
        plan=plan,
        action="make_default",
        budget=budget,
        requested_traffic_fraction=1.0,
        decided_at="2026-04-23T00:01:00+00:00",
    )
    assert direct_default.accepted is False
    assert "rollout_transition_invalid" in direct_default.reason_codes

    approve = build_rollout_decision(
        plan=plan,
        action="approve",
        budget=budget,
        decided_at="2026-04-23T00:02:00+00:00",
    )
    approved = apply_rollout_decision(plan, approve)
    assert approve.accepted is True
    assert approved.routing_state == AdapterRoutingState.APPROVED.value
    assert approved.traffic_fraction == 0.0

    activate = build_rollout_decision(
        plan=approved,
        action="activate",
        budget=budget,
        requested_traffic_fraction=0.05,
        decided_at="2026-04-23T00:03:00+00:00",
    )
    active = apply_rollout_decision(approved, activate)
    assert active.routing_state == AdapterRoutingState.ACTIVE_LIMITED.value
    assert active.traffic_fraction == 0.05

    default_candidate = apply_rollout_decision(
        active,
        build_rollout_decision(
            plan=active,
            action="promote_default_candidate",
            budget=budget,
            requested_traffic_fraction=0.25,
            decided_at="2026-04-23T00:04:00+00:00",
        ),
    )
    assert default_candidate.routing_state == AdapterRoutingState.DEFAULT_CANDIDATE.value

    default = apply_rollout_decision(
        default_candidate,
        build_rollout_decision(
            plan=default_candidate,
            action="make_default",
            budget=budget,
            requested_traffic_fraction=1.0,
            decided_at="2026-04-23T00:05:00+00:00",
        ),
    )
    assert default.routing_state == AdapterRoutingState.DEFAULT.value
    assert default.traffic_fraction == 1.0


def test_live_routing_budget_enforces_traffic_ack_evidence_and_embodied_gate():
    budget = _budget(max_traffic_fraction=0.05)
    plan = _plan(budget=budget)

    too_much = build_rollout_decision(
        plan=plan,
        action="approve",
        budget=budget,
        requested_traffic_fraction=0.2,
        decided_at="2026-04-23T00:01:00+00:00",
    )
    assert too_much.accepted is False
    assert "budget_traffic_fraction_exceeded" in too_much.reason_codes

    missing_ack_plan = replace(plan, operator_acknowledged=False)
    missing_ack = build_rollout_decision(
        plan=missing_ack_plan,
        action="approve",
        budget=budget,
        decided_at="2026-04-23T00:02:00+00:00",
    )
    assert missing_ack.accepted is False
    assert "budget_operator_ack_required" in missing_ack.reason_codes

    weak_evidence_plan = build_adapter_routing_plan(
        card=_candidate_card(scenario_count=0, compared_family_count=0, benchmark_passed=False),
        budget=budget,
        incumbent_backend_id="local_world_model",
        scope_key="local",
        created_at="2026-04-23T00:00:00+00:00",
        expires_at="2026-04-24T00:00:00+00:00",
        operator_acknowledged=True,
    )
    weak_evidence = build_rollout_decision(
        plan=weak_evidence_plan,
        action="approve",
        budget=budget,
        decided_at="2026-04-23T00:03:00+00:00",
    )
    assert weak_evidence.accepted is False
    assert "budget_benchmark_required" in weak_evidence.reason_codes
    assert "budget_scenario_count_below_minimum" in weak_evidence.reason_codes

    embodied_budget = _budget(
        adapter_family=BrainAdapterFamily.EMBODIED_POLICY.value,
        allow_embodied_live=False,
    )
    embodied_plan = _plan(
        adapter_family=BrainAdapterFamily.EMBODIED_POLICY.value,
        budget=embodied_budget,
        embodied_live=True,
    )
    embodied_decision = build_rollout_decision(
        plan=embodied_plan,
        action="approve",
        budget=embodied_budget,
        decided_at="2026-04-23T00:04:00+00:00",
    )
    assert embodied_decision.accepted is False
    assert "budget_embodied_live_not_allowed" in embodied_decision.reason_codes


def test_live_routing_rolls_back_on_regression_without_budget_gate():
    budget = _budget()
    plan = _plan(budget=budget)
    approved = apply_rollout_decision(
        plan,
        build_rollout_decision(
            plan=plan,
            action="approve",
            budget=budget,
            decided_at="2026-04-23T00:01:00+00:00",
        ),
    )
    active = apply_rollout_decision(
        approved,
        build_rollout_decision(
            plan=approved,
            action="activate",
            budget=budget,
            requested_traffic_fraction=0.05,
            decided_at="2026-04-23T00:02:00+00:00",
        ),
    )

    rollback = build_rollout_decision(
        plan=active,
        action="activate",
        budget=budget,
        requested_traffic_fraction=0.05,
        regression_codes=("safety_critical_regression",),
        decided_at="2026-04-23T00:03:00+00:00",
    )
    rolled_back = apply_rollout_decision(active, rollback)

    assert rollback.accepted is True
    assert rollback.to_state == AdapterRoutingState.ROLLED_BACK.value
    assert rollback.traffic_fraction == 0.0
    assert "rollout_rollback_triggered" in rollback.reason_codes
    assert rolled_back.routing_state == AdapterRoutingState.ROLLED_BACK.value
    assert rolled_back.details["last_regression_codes"] == ["safety_critical_regression"]


def test_live_routing_handles_expiry_pause_and_resume():
    budget = _budget()
    expired_plan = _plan(budget=budget)
    expired = build_rollout_decision(
        plan=expired_plan,
        action="activate",
        budget=budget,
        requested_traffic_fraction=0.05,
        decided_at="2026-04-24T00:00:01+00:00",
    )
    assert expired.accepted is True
    assert expired.to_state == AdapterRoutingState.EXPIRED.value
    assert "rollout_plan_expired" in expired.reason_codes

    plan = _plan(budget=budget)
    approved = apply_rollout_decision(
        plan,
        build_rollout_decision(
            plan=plan,
            action="approve",
            budget=budget,
            decided_at="2026-04-23T00:01:00+00:00",
        ),
    )
    active = apply_rollout_decision(
        approved,
        build_rollout_decision(
            plan=approved,
            action="activate",
            budget=budget,
            requested_traffic_fraction=0.05,
            decided_at="2026-04-23T00:02:00+00:00",
        ),
    )
    paused = apply_rollout_decision(
        active,
        build_rollout_decision(
            plan=active,
            action="pause",
            decided_at="2026-04-23T00:03:00+00:00",
        ),
    )
    assert paused.routing_state == AdapterRoutingState.PAUSED.value
    assert paused.traffic_fraction == active.traffic_fraction

    resumed = apply_rollout_decision(
        paused,
        build_rollout_decision(
            plan=paused,
            action="resume",
            budget=budget,
            requested_traffic_fraction=0.04,
            decided_at="2026-04-23T00:04:00+00:00",
        ),
    )
    assert resumed.routing_state == AdapterRoutingState.ACTIVE_LIMITED.value
    assert resumed.traffic_fraction == 0.04


def test_live_routing_controller_evaluates_and_activates_explicitly():
    budget = _budget()
    plan = _plan(budget=budget)
    controller = LiveRoutingController(plans=(plan,), budgets=(budget,))

    illegal_direct_activation = controller.activate_plan(
        plan.plan_id,
        traffic_fraction=0.05,
        decided_at="2026-04-23T00:00:30+00:00",
    )
    assert illegal_direct_activation.accepted is False
    assert "rollout_transition_invalid" in illegal_direct_activation.reason_codes
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.PROPOSED.value

    evaluation = controller.evaluate_plan(
        plan.plan_id,
        decided_at="2026-04-23T00:01:00+00:00",
    )
    assert evaluation.accepted is True
    assert evaluation.applied is True
    assert evaluation.to_state == AdapterRoutingState.APPROVED.value
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.APPROVED.value

    activation = controller.activate_plan(
        plan.plan_id,
        traffic_fraction=0.05,
        decided_at="2026-04-23T00:02:00+00:00",
    )
    assert activation.accepted is True
    assert activation.applied is True
    assert activation.to_state == AdapterRoutingState.ACTIVE_LIMITED.value
    assert controller.plan(plan.plan_id).traffic_fraction == 0.05

    status = controller.current_status(generated_at="2026-04-23T00:03:00+00:00")
    payload = status.as_dict()
    assert payload["available"] is True
    assert payload["live_routing_active"] is True
    assert payload["controlled_rollout_supported"] is True
    assert payload["governance_only"] is False
    assert payload["active_plan_count"] == 1
    assert payload["plan_summaries"][0]["routing_state"] == AdapterRoutingState.ACTIVE_LIMITED.value
    assert payload["recent_decisions"][0]["action"] == "activate"
    assert "decision_id" not in str(payload)
    assert "details" not in str(payload)


def test_live_routing_controller_rejects_missing_budget_and_missing_plan():
    plan = _plan()
    controller = LiveRoutingController(plans=(plan,))

    missing_budget = controller.evaluate_plan(
        plan.plan_id,
        decided_at="2026-04-23T00:01:00+00:00",
    )
    assert missing_budget.accepted is False
    assert missing_budget.applied is False
    assert "rollout_budget_missing" in missing_budget.reason_codes
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.PROPOSED.value

    missing_plan = controller.activate_plan(
        "missing-plan",
        traffic_fraction=0.01,
        decided_at="2026-04-23T00:02:00+00:00",
    )
    assert missing_plan.accepted is False
    assert missing_plan.decision is None
    assert "live_routing_plan_not_found" in missing_plan.reason_codes


def test_live_routing_controller_pause_resume_rollback_and_expiry():
    budget = _budget()
    plan = _plan(budget=budget)
    controller = LiveRoutingController(plans=(plan,), budgets=(budget,))
    controller.evaluate_plan(plan.plan_id, decided_at="2026-04-23T00:01:00+00:00")
    controller.activate_plan(
        plan.plan_id,
        traffic_fraction=0.05,
        decided_at="2026-04-23T00:02:00+00:00",
    )

    pause = controller.pause_plan(plan.plan_id, decided_at="2026-04-23T00:03:00+00:00")
    assert pause.accepted is True
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.PAUSED.value
    assert controller.plan(plan.plan_id).traffic_fraction == 0.05

    resume = controller.resume_plan(
        plan.plan_id,
        traffic_fraction=0.04,
        decided_at="2026-04-23T00:04:00+00:00",
    )
    assert resume.accepted is True
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.ACTIVE_LIMITED.value
    assert controller.plan(plan.plan_id).traffic_fraction == 0.04

    rollback = controller.rollback_plan(
        plan.plan_id,
        regression_codes=("safety_critical_regression",),
        decided_at="2026-04-23T00:05:00+00:00",
    )
    assert rollback.accepted is True
    assert controller.plan(plan.plan_id).routing_state == AdapterRoutingState.ROLLED_BACK.value
    assert controller.plan(plan.plan_id).traffic_fraction == 0.0
    assert "rollout_rollback_triggered" in rollback.reason_codes

    expiring_plan = _plan(budget=budget)
    expiry_controller = LiveRoutingController(plans=(expiring_plan,), budgets=(budget,))
    expiry_results = expiry_controller.expire_stale_plans(now="2026-04-24T00:00:01+00:00")
    assert len(expiry_results) == 1
    assert expiry_results[0].accepted is True
    assert expiry_controller.plan(expiring_plan.plan_id).routing_state == (
        AdapterRoutingState.EXPIRED.value
    )
    assert expiry_controller.current_status().expired_plan_count == 1


def test_live_routing_keeps_cross_family_isolation_and_stable_reports():
    world_budget = _budget(adapter_family=BrainAdapterFamily.WORLD_MODEL.value)
    perception_budget = _budget(adapter_family=BrainAdapterFamily.PERCEPTION.value)
    world_plan = _plan(adapter_family=BrainAdapterFamily.WORLD_MODEL.value, budget=world_budget)
    perception_plan = _plan(
        adapter_family=BrainAdapterFamily.PERCEPTION.value,
        budget=perception_budget,
    )
    world_active = apply_rollout_decision(
        apply_rollout_decision(
            world_plan,
            build_rollout_decision(
                plan=world_plan,
                action="approve",
                budget=world_budget,
                decided_at="2026-04-23T00:01:00+00:00",
            ),
        ),
        build_rollout_decision(
            plan=replace(world_plan, routing_state=AdapterRoutingState.APPROVED.value),
            action="activate",
            budget=world_budget,
            requested_traffic_fraction=0.05,
            decided_at="2026-04-23T00:02:00+00:00",
        ),
    )
    plans = (world_active, perception_plan)

    assert (
        active_routing_plan_for_family(
            plans,
            adapter_family=BrainAdapterFamily.WORLD_MODEL.value,
            scope_key="local",
        ).adapter_family
        == BrainAdapterFamily.WORLD_MODEL.value
    )
    assert (
        active_routing_plan_for_family(
            plans,
            adapter_family=BrainAdapterFamily.PERCEPTION.value,
            scope_key="local",
        )
        is None
    )

    wrong_family_decision = replace(
        build_rollout_decision(
            plan=world_active,
            action="pause",
            decided_at="2026-04-23T00:03:00+00:00",
        ),
        adapter_family=BrainAdapterFamily.PERCEPTION.value,
    )
    with pytest.raises(ValueError):
        apply_rollout_decision(world_active, wrong_family_decision)

    report = build_live_routing_report(
        plans=plans,
        decisions=(wrong_family_decision,),
        generated_at="2026-04-23T00:04:00+00:00",
    )
    assert (
        report.as_dict()
        == build_live_routing_report(
            plans=[plan.as_dict() for plan in plans],
            decisions=[wrong_family_decision.as_dict()],
            generated_at="2026-04-23T00:04:00+00:00",
        ).as_dict()
    )
    assert report.state_counts[AdapterRoutingState.ACTIVE_LIMITED.value] == 1
    assert report.family_counts[BrainAdapterFamily.PERCEPTION.value] == 1
    markdown = render_live_routing_report_markdown(report)
    assert "Live Routing Report" in markdown
    assert "candidate_world_model" in markdown
    assert "/tmp" not in str(report.as_dict())


def test_live_routing_serialization_roundtrips():
    budget = _budget()
    plan = _plan(budget=budget)
    decision = build_rollout_decision(
        plan=plan,
        action="approve",
        budget=budget,
        decided_at="2026-04-23T00:01:00+00:00",
    )

    assert RolloutBudget.from_dict(budget.as_dict()) == budget
    assert AdapterRoutingPlan.from_dict(plan.as_dict()) == plan
    assert RolloutDecisionRecord.from_dict(decision.as_dict()) == decision

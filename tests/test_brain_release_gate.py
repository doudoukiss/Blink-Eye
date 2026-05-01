import json
from dataclasses import replace

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.cards import (
    BrainAdapterBenchmarkSummary,
    BrainAdapterFamily,
    BrainAdapterPromotionState,
    build_adapter_card,
)
from blink.brain.adapters.live_routing import build_adapter_routing_plan
from blink.brain.adapters.rollout_budget import build_rollout_budget
from blink.brain.evals.release_gate import (
    build_release_gate_report,
    render_release_gate_markdown,
    write_release_gate_artifacts,
)
from blink.cli import release_gate as release_gate_cli


def _frontier_report(*, passed=True):
    return {
        "schema_version": 1,
        "suite_id": "frontier_behavior_workbench/v1",
        "passed": passed,
        "gates": {
            "all_cases_passed": passed,
            "no_boundary_safety_failure": passed,
            "report_payload_leak_free": True,
        },
        "aggregate_metrics": {"boundary_safety": 1.0 if passed else 0.0},
    }


def _autonomy_report(*, passed=True):
    return {
        "schema_version": 1,
        "suite_id": "autonomy_benchmark_program/v1",
        "passed": passed,
        "aggregate_score": 1.0 if passed else 0.5,
        "gating_failures": [] if passed else ["family_failed:rollout_safety"],
        "family_scores": [],
    }


def _budget():
    return build_rollout_budget(
        adapter_family=BrainAdapterFamily.WORLD_MODEL.value,
        max_traffic_fraction=0.05,
        eligible_scopes=("local",),
        minimum_scenario_count=2,
        minimum_compared_family_count=1,
    )


def _plan(*, budget=None, operator_acknowledged=True, traffic_fraction=0.01):
    resolved_budget = budget or _budget()
    benchmark = BrainAdapterBenchmarkSummary(
        report_id="benchmark-report-1",
        adapter_family=BrainAdapterFamily.WORLD_MODEL.value,
        scenario_count=4,
        compared_family_count=2,
        benchmark_passed=True,
        smoke_suite_green=True,
        target_families=("temporal_consistency",),
        updated_at="2026-04-24T00:00:00+00:00",
    )
    card = build_adapter_card(
        adapter_family=BrainAdapterFamily.WORLD_MODEL.value,
        descriptor=BrainAdapterDescriptor(
            backend_id="candidate_world_model",
            backend_version="candidate-v1",
            capabilities=("prediction_proposal",),
            degraded_mode_id="empty_proposals",
            default_timeout_ms=250,
        ),
        promotion_state=BrainAdapterPromotionState.CANARY.value,
        latest_benchmark_summary=benchmark,
        updated_at="2026-04-24T00:00:00+00:00",
    )
    return build_adapter_routing_plan(
        card=card,
        budget=resolved_budget,
        incumbent_backend_id="local_world_model",
        scope_key="local",
        traffic_fraction=traffic_fraction,
        created_at="2026-04-24T00:00:00+00:00",
        expires_at="2026-04-25T00:00:00+00:00",
        operator_acknowledged=operator_acknowledged,
    )


def _sim_digest(*, rollback_required=False, canary_ready=True):
    return {
        "readiness_counts": {
            "shadow_ready": 1,
            "canary_ready": int(canary_ready),
            "default_ready": 0,
            "rollback_required": int(rollback_required),
        },
        "blocked_reason_counts": {},
        "readiness_reports": [
            {
                "report_id": "sim-report-1",
                "adapter_family": BrainAdapterFamily.WORLD_MODEL.value,
                "backend_id": "candidate_world_model",
                "backend_version": "candidate-v1",
                "promotion_state": BrainAdapterPromotionState.CANARY.value,
                "benchmark_passed": True,
                "smoke_suite_green": True,
                "shadow_ready": True,
                "canary_ready": canary_ready,
                "default_ready": False,
                "rollback_required": rollback_required,
                "governance_only": True,
                "blocked_reason_codes": (
                    ["safety_critical_regression"] if rollback_required else []
                ),
                "updated_at": "2026-04-24T00:00:00+00:00",
                "details": {"trace": "Traceback /tmp/brain.db"},
            }
        ],
    }


def _build_passing_report():
    budget = _budget()
    plan = _plan(budget=budget)
    return build_release_gate_report(
        frontier_behavior_report=_frontier_report(passed=True),
        autonomy_benchmark_report=_autonomy_report(passed=True),
        sim_to_real_digest=_sim_digest(),
        rollout_plan=plan,
        rollout_budget=budget,
        tts_backend="local-http-wav",
        generated_at="2026-04-24T00:05:00+00:00",
    )


def test_release_gate_report_passes_and_can_be_referenced_by_rollout_plan():
    budget = _budget()
    plan = _plan(budget=budget)
    report = build_release_gate_report(
        frontier_behavior_report=_frontier_report(passed=True),
        autonomy_benchmark_report=_autonomy_report(passed=True),
        sim_to_real_digest=_sim_digest(),
        rollout_plan=plan,
        rollout_budget=budget,
        tts_backend="local-http-wav",
        generated_at="2026-04-24T00:05:00+00:00",
    )
    repeated = build_release_gate_report(
        frontier_behavior_report=_frontier_report(passed=True),
        autonomy_benchmark_report=_autonomy_report(passed=True),
        sim_to_real_digest=_sim_digest(),
        rollout_plan=plan,
        rollout_budget=budget,
        tts_backend="local-http-wav",
        generated_at="2026-04-24T00:05:00+00:00",
    )

    assert report.as_dict() == repeated.as_dict()
    assert report.passed is True
    assert report.outcome == "passed"
    assert report.report_id.startswith("release_gate_")
    assert all(check.passed for check in report.checks)
    assert report.rollout_reference()["gate_report_id"] == report.report_id
    gated_plan = replace(plan, evidence_ids=(*plan.evidence_ids, report.report_id))
    assert report.report_id in gated_plan.evidence_ids
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True)
    assert "Traceback" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "source_event_ids" not in encoded


def test_release_gate_report_blocks_on_failed_evidence_and_budget_state():
    budget = _budget()
    plan = _plan(budget=budget, operator_acknowledged=False, traffic_fraction=0.08)

    report = build_release_gate_report(
        frontier_behavior_report=_frontier_report(passed=False),
        autonomy_benchmark_report=_autonomy_report(passed=False),
        sim_to_real_digest=_sim_digest(rollback_required=True, canary_ready=False),
        rollout_plan=plan,
        rollout_budget=budget,
        tts_backend="local-http-wav",
        generated_at="2026-04-24T00:05:00+00:00",
    )

    assert report.passed is False
    assert report.outcome == "blocked"
    assert "operator_ack_required" in report.blocking_reason_codes
    assert "rollout_traffic_exceeds_budget" in report.blocking_reason_codes
    assert "sim_to_real_rollback_required" in report.blocking_reason_codes
    assert any(code.startswith("autonomy_benchmark_failure:") for code in report.blocking_reason_codes)
    assert report.check_results()["frontier_behavior_workbench"] is False


def test_release_gate_artifacts_are_stable(tmp_path):
    report = _build_passing_report()

    paths = write_release_gate_artifacts(report, output_dir=tmp_path)

    json_payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert json_payload["report_id"] == report.report_id
    assert paths == {
        "json": str(tmp_path / "latest.json"),
        "markdown": str(tmp_path / "latest.md"),
    }
    assert render_release_gate_markdown(report) in markdown
    assert "Release Gate Report" in markdown


def test_release_gate_cli_uses_precomputed_inputs(tmp_path, capsys):
    budget = _budget()
    plan = _plan(budget=budget)
    frontier_path = tmp_path / "frontier.json"
    autonomy_path = tmp_path / "autonomy.json"
    sim_path = tmp_path / "sim.json"
    plan_path = tmp_path / "plan.json"
    budget_path = tmp_path / "budget.json"
    frontier_path.write_text(json.dumps(_frontier_report()), encoding="utf-8")
    autonomy_path.write_text(json.dumps(_autonomy_report()), encoding="utf-8")
    sim_path.write_text(json.dumps(_sim_digest()), encoding="utf-8")
    plan_path.write_text(json.dumps(plan.as_dict()), encoding="utf-8")
    budget_path.write_text(json.dumps(budget.as_dict()), encoding="utf-8")

    code = release_gate_cli.main(
        [
            "--frontier-report-json",
            str(frontier_path),
            "--autonomy-report-json",
            str(autonomy_path),
            "--sim-to-real-json",
            str(sim_path),
            "--rollout-plan-json",
            str(plan_path),
            "--rollout-budget-json",
            str(budget_path),
            "--voice-backend",
            "local-http-wav",
            "--output-dir",
            str(tmp_path / "gate"),
            "--generated-at",
            "2026-04-24T00:05:00+00:00",
        ]
    )

    stdout = json.loads(capsys.readouterr().out)
    assert code == 0
    assert stdout["passed"] is True
    assert stdout["rollout_reference"]["gate_report_id"].startswith("release_gate_")
    assert (tmp_path / "gate" / "latest.json").exists()
    assert (tmp_path / "gate" / "latest.md").exists()

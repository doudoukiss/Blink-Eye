import json
from types import SimpleNamespace

from blink.brain.evals.adapter_promotion import BrainAdapterBenchmarkComparisonReport
from blink.brain.evals.episode_evidence_index import build_episode_evidence_index
from blink.brain.practice_director import BrainPracticePlan, BrainPracticeTarget


def _encoded(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def test_episode_evidence_index_is_deterministic_and_public_safe():
    session_ids = SimpleNamespace(
        user_id="user-1",
        thread_id="thread-1",
        agent_id="blink-test",
    )
    practice_target = BrainPracticeTarget(
        target_id="target-1",
        scenario_family="debugging",
        scenario_id="debug-minimal-repro",
        scenario_version="v1",
        suite_id="suite-1",
        selected_profile_id="local-sim",
        execution_backend="simulation",
        score=5.0,
        reason_codes=["recovery_pressure"],
        supporting_episode_ids=["episode-source-1"],
    )
    practice_plan = BrainPracticePlan(
        plan_id="practice-plan-1",
        scope_key="thread-1",
        presence_scope_key="browser:presence",
        dataset_manifest_id="manifest-1",
        targets=[practice_target],
        reason_code_counts={"recovery_pressure": 1},
        summary="Practice debugging.",
        artifact_paths={"plan_json": "/tmp/practice-plan.json", "raw_db": "/tmp/brain.db"},
        created_at="2026-04-24T00:00:00+00:00",
        updated_at="2026-04-24T00:01:00+00:00",
    )
    benchmark_report = BrainAdapterBenchmarkComparisonReport(
        report_id="report-1",
        adapter_family="world_model",
        incumbent_backend_id="incumbent-world",
        incumbent_backend_version="v1",
        candidate_backend_id="candidate-world",
        candidate_backend_version="v2",
        scenario_count=4,
        compared_family_count=2,
        target_families=["debugging"],
        blocked_reason_codes=["needs_more_evidence"],
        smoke_suite_green=True,
        benchmark_passed=False,
        summary="World-model candidate needs more evidence.",
        artifact_paths={"json": "/tmp/brain.db"},
        updated_at="2026-04-24T00:02:00+00:00",
        details={"trace": "Traceback /tmp/brain.db"},
    )

    class FakePromotionDecision:
        def as_dict(self):
            return {
                "decision_id": "decision-1",
                "report_id": "report-1",
                "details": {"trace": "Traceback /tmp/brain.db"},
            }

    class FakeStore:
        def recent_episodes(self, **kwargs):
            assert kwargs["user_id"] == "user-1"
            assert kwargs["thread_id"] == "thread-1"
            return [
                SimpleNamespace(
                    id=7,
                    session_id="session-1",
                    created_at="2026-04-24T00:03:00+00:00",
                    user_text="Traceback /tmp/brain.db",
                    assistant_summary="Operator-safe live summary.",
                    tool_calls_json='{"raw": "/tmp/brain.db"}',
                )
            ]

        def build_practice_director_projection(self, **kwargs):
            assert kwargs["presence_scope_key"] == "browser:presence"
            return SimpleNamespace(recent_plans=[practice_plan])

        def build_adapter_governance_projection(self, **_kwargs):
            return SimpleNamespace(
                recent_reports=[benchmark_report],
                recent_decisions=[FakePromotionDecision()],
            )

    class FakeRolloutPlan:
        def as_dict(self):
            return {
                "plan_id": "rollout-plan-1",
                "sim_to_real_report_ids": ["report-1"],
                "evidence_ids": ["episode_evidence_unused"],
                "details": {"artifact_path": "/tmp/brain.db"},
            }

    snapshot = build_episode_evidence_index(
        store=FakeStore(),
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        rollout_controller=SimpleNamespace(plans=(FakeRolloutPlan(),)),
        recent_limit=8,
        generated_at="2026-04-24T00:04:00+00:00",
    )
    payload = snapshot.as_dict()
    second_payload = build_episode_evidence_index(
        store=FakeStore(),
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        rollout_controller=SimpleNamespace(plans=(FakeRolloutPlan(),)),
        recent_limit=8,
        generated_at="2026-04-24T00:04:00+00:00",
    ).as_dict()

    assert payload == second_payload
    assert payload["available"] is True
    assert payload["source_counts"] == {"eval": 1, "live": 1, "practice": 1}
    assert {row["source"] for row in payload["rows"]} == {"live", "practice", "eval"}
    eval_row = next(row for row in payload["rows"] if row["source"] == "eval")
    assert eval_row["task_success"] is False
    assert {link["link_kind"] for link in eval_row["links"]} == {
        "adapter_promotion_decision",
        "benchmark_report",
        "rollout_plan",
    }
    practice_row = next(row for row in payload["rows"] if row["source"] == "practice")
    assert practice_row["artifact_refs"][0]["redacted_uri"] is True
    assert "json_file" in {artifact["uri_kind"] for artifact in practice_row["artifact_refs"]}

    encoded = _encoded(payload)
    for banned in (
        "Traceback",
        "/tmp/brain.db",
        "tool_calls_json",
        "artifact_paths",
        "source_event_ids",
        "private_working_memory",
        "prompt_text",
    ):
        assert banned not in encoded


def test_episode_evidence_index_reports_missing_runtime_surface():
    payload = build_episode_evidence_index(
        store=None,
        session_ids=None,
        generated_at="2026-04-24T00:00:00+00:00",
    ).as_dict()

    assert payload["available"] is False
    assert payload["rows"] == []
    assert "runtime_evidence_surface_missing" in payload["reason_codes"]

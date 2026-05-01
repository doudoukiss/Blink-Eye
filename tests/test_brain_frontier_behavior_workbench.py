import json

from blink.brain.evals import (
    FRONTIER_BEHAVIOR_WORKBENCH_CATEGORIES,
    FRONTIER_BEHAVIOR_WORKBENCH_METRICS,
    FRONTIER_BEHAVIOR_WORKBENCH_SUITE_ID,
    build_frontier_behavior_workbench_suite,
    evaluate_frontier_behavior_workbench_case,
    evaluate_frontier_behavior_workbench_suite,
    render_frontier_behavior_workbench_markdown,
    render_frontier_behavior_workbench_metrics_rows,
    write_frontier_behavior_workbench_artifacts,
)


def _result(report, case_id: str):
    return next(result for result in report.results if result.case.case_id == case_id)


def test_frontier_behavior_suite_contains_stable_categories_and_case_ids():
    suite = build_frontier_behavior_workbench_suite()

    assert {case.category for case in suite} == set(FRONTIER_BEHAVIOR_WORKBENCH_CATEGORIES)
    assert [case.case_id for case in suite] == [
        "memory_governance_claim_actions",
        "memory_governance_task_action_parity",
        "memory_currentness_stale_superseded_conflict",
        "memory_use_trace_safe_projection",
        "persona_boundary_hardware_guardrails",
        "teaching_adaptation_cjk_selection",
        "browser_public_payload_leak_guard",
        "voice_policy_chunking_noops",
    ]
    assert all("prompt" not in case.as_dict() for case in suite)


def test_frontier_behavior_report_is_deterministic_and_json_serializable():
    first = evaluate_frontier_behavior_workbench_suite()
    second = evaluate_frontier_behavior_workbench_suite()
    rows = render_frontier_behavior_workbench_metrics_rows(first)

    assert first.as_dict() == second.as_dict()
    assert rows == render_frontier_behavior_workbench_metrics_rows(second)
    assert first.passed is True
    assert json.loads(json.dumps(first.as_dict(), ensure_ascii=False, sort_keys=True)) == (
        first.as_dict()
    )
    assert {row["suite_id"] for row in rows} == {FRONTIER_BEHAVIOR_WORKBENCH_SUITE_ID}


def test_frontier_behavior_metrics_are_bounded_and_reasoned():
    report = evaluate_frontier_behavior_workbench_suite()
    payload = report.as_dict()

    for metric_name in FRONTIER_BEHAVIOR_WORKBENCH_METRICS:
        assert 0.0 <= payload["aggregate_metrics"][metric_name] <= 1.0
    for row in payload["metrics_rows"]:
        for metric_name in FRONTIER_BEHAVIOR_WORKBENCH_METRICS:
            assert 0.0 <= row[metric_name] <= 1.0
        assert row["estimated_prompt_tokens"] <= 192
        assert row["reason_codes"]
        assert len(row["reason_codes"]) <= 48
    assert all(payload["gates"].values())


def test_frontier_behavior_governance_cases_measure_claim_and_task_actions():
    report = evaluate_frontier_behavior_workbench_suite()
    claims = _result(report, "memory_governance_claim_actions")
    tasks = _result(report, "memory_governance_task_action_parity")

    assert claims.passed is True
    assert claims.evidence["accepted_actions"] == ("pin", "suppress", "correct", "forget")
    assert claims.evidence["hidden_counts"]["suppressed"] == 1
    assert tasks.passed is True
    assert tasks.evidence["initial_task_count"] == 2
    assert tasks.evidence["active_task_count_after"] == 0
    assert "cross_user_task_rejected" in tasks.metric_row.reason_codes


def test_frontier_behavior_currentness_and_use_trace_cases_are_safe():
    report = evaluate_frontier_behavior_workbench_suite()
    currentness = _result(report, "memory_currentness_stale_superseded_conflict")
    trace = _result(report, "memory_use_trace_safe_projection")

    assert currentness.passed is True
    assert currentness.evidence["current_summaries"] == ("User role is product manager",)
    assert currentness.evidence["current_retrieval_count"] >= 1
    assert trace.passed is True
    assert trace.evidence["trace_ref_count"] >= 1
    assert trace.evidence["used_in_current_turn"] is True
    assert trace.evidence["last_used_at"] == "2026-04-23T00:00:00+00:00"
    assert trace.evidence["safe_provenance_label_present"] is True


def test_frontier_behavior_persona_teaching_browser_and_voice_cases_pass():
    report = evaluate_frontier_behavior_workbench_suite()
    persona = _result(report, "persona_boundary_hardware_guardrails")
    teaching = _result(report, "teaching_adaptation_cjk_selection")
    browser = _result(report, "browser_public_payload_leak_guard")
    voice = _result(report, "voice_policy_chunking_noops")

    assert persona.passed is True
    assert persona.evidence["identity_label"] == "Blink; local non-human system"
    assert persona.evidence["expression_controls_hardware"] is False
    assert persona.evidence["relationship_boundaries"] == (
        "non-romantic",
        "non-sexual",
        "non-exclusive",
    )
    assert teaching.passed is True
    assert teaching.evidence["adapted_mode"] == "walkthrough"
    assert teaching.evidence["cjk_bridge_selected"] is True
    assert browser.passed is True
    assert browser.evidence["safe_payload_record_count"] == 1
    assert browser.evidence["detected_leak_category_count"] >= 6
    assert voice.passed is True
    assert voice.evidence["chunking_mode"] == "safety_concise"
    assert voice.evidence["max_spoken_chunk_chars"] <= 96
    assert voice.evidence["expression_controls_hardware"] is False


def test_frontier_behavior_markdown_and_artifact_writer_are_stable(tmp_path):
    report = evaluate_frontier_behavior_workbench_suite()
    markdown = render_frontier_behavior_workbench_markdown(report)
    paths = write_frontier_behavior_workbench_artifacts(report, output_dir=tmp_path)

    assert "# Frontier Behavior Workbench Report" in markdown
    assert "memory_governance_claim_actions" in markdown
    assert paths == {
        "json": str(tmp_path / "latest.json"),
        "markdown": str(tmp_path / "latest.md"),
    }
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == (report.as_dict())
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == f"{markdown}\n"
    assert write_frontier_behavior_workbench_artifacts(report, output_dir=tmp_path) == paths
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == f"{markdown}\n"


def test_frontier_behavior_report_excludes_raw_ids_paths_prompts_and_responses():
    report = evaluate_frontier_behavior_workbench_suite()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True)
    banned_tokens = (
        "source_event_ids",
        "source_refs",
        "source_event_id",
        "event_id",
        "raw_json",
        "evt-",
        "claim_id",
        "memory_claim:",
        "brain.db",
        ".db",
        "/tmp",
        "```json",
        "[BLINK_BRAIN_CONTEXT]",
        "private_scratchpad",
        "Traceback",
        "RuntimeError",
        "You are Blink. Always follow",
    )

    assert all(token not in encoded for token in banned_tokens)
    assert "replacement_memory_id" not in encoded
    assert "full_response" not in encoded


def test_frontier_behavior_lazy_exports_and_single_case_helper():
    case = next(
        case
        for case in build_frontier_behavior_workbench_suite()
        if case.case_id == "voice_policy_chunking_noops"
    )
    single = evaluate_frontier_behavior_workbench_case(case)
    suite = _result(evaluate_frontier_behavior_workbench_suite([case]), case.case_id)

    assert single.as_dict() == suite.as_dict()
    assert FRONTIER_BEHAVIOR_WORKBENCH_SUITE_ID == "frontier_behavior_workbench/v1"

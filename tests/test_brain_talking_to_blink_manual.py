import json

from blink.brain.evals import (
    TALKING_TO_BLINK_MANUAL_CATEGORIES,
    TALKING_TO_BLINK_MANUAL_MANUAL_FOLLOWUPS,
    TALKING_TO_BLINK_MANUAL_SUITE_ID,
    build_talking_to_blink_manual_suite,
    evaluate_talking_to_blink_manual_case,
    evaluate_talking_to_blink_manual_suite,
    render_talking_to_blink_manual_metrics_rows,
)


def _result(report, case_id: str):
    return next(result for result in report.results if result.case.case_id == case_id)


def test_talking_to_blink_manual_suite_has_stable_case_ids():
    suite = build_talking_to_blink_manual_suite()

    assert {case.category for case in suite} == set(TALKING_TO_BLINK_MANUAL_CATEGORIES)
    assert [case.case_id for case in suite] == [
        "identity_character_context",
        "relationship_boundary_guardrails",
        "memory_scope_persists_across_browser_reconnect",
        "memory_correction_and_forgetting_surfaces",
        "memory_use_transparency_projection",
        "behavior_controls_shape_expression",
        "teaching_canon_selects_manual_prompts",
        "voice_policy_metrics_are_observable",
        "capability_honesty_no_hardware_claims",
        "operator_payload_is_public_safe",
    ]
    assert all(case.automation_level == "deterministic" for case in suite)


def test_talking_to_blink_manual_report_is_deterministic_and_serializable():
    first = evaluate_talking_to_blink_manual_suite()
    second = evaluate_talking_to_blink_manual_suite()
    rows = render_talking_to_blink_manual_metrics_rows(first)

    assert first.as_dict() == second.as_dict()
    assert rows == render_talking_to_blink_manual_metrics_rows(second)
    assert first.passed is True
    assert first.suite_id == TALKING_TO_BLINK_MANUAL_SUITE_ID
    assert json.loads(json.dumps(first.as_dict(), ensure_ascii=False, sort_keys=True)) == (
        first.as_dict()
    )
    assert tuple(first.manual_followups) == TALKING_TO_BLINK_MANUAL_MANUAL_FOLLOWUPS


def test_talking_to_blink_manual_metrics_are_compact_and_bounded():
    report = evaluate_talking_to_blink_manual_suite()
    payload = report.as_dict()

    assert payload["aggregate_metrics"]["case_count"] == 10
    assert payload["aggregate_metrics"]["automated_case_count"] == 10
    assert payload["aggregate_metrics"]["manual_followup_count"] >= 1
    assert payload["aggregate_metrics"]["pass_rate"] == 1.0
    for row in payload["metrics_rows"]:
        assert row["suite_id"] == TALKING_TO_BLINK_MANUAL_SUITE_ID
        assert row["automated"] is True
        assert row["manual_followup_required"] is False
        assert 0.0 <= row["score"] <= 1.0
        assert row["estimated_prompt_tokens"] <= 128
        assert row["reason_codes"]


def test_talking_to_blink_manual_identity_memory_behavior_and_teaching_cases():
    report = evaluate_talking_to_blink_manual_suite()
    identity = _result(report, "identity_character_context")
    memory = _result(report, "memory_scope_persists_across_browser_reconnect")
    lifecycle = _result(report, "memory_correction_and_forgetting_surfaces")
    transparency = _result(report, "memory_use_transparency_projection")
    behavior = _result(report, "behavior_controls_shape_expression")
    teaching = _result(report, "teaching_canon_selects_manual_prompts")

    assert identity.passed is True
    assert any(
        "character: warm precise local tutor" in line
        for line in identity.evidence["persona_expression_lines"]
    )
    assert memory.passed is True
    assert memory.evidence["user_id"] == "local_primary"
    assert memory.evidence["claim_count"] == 1
    assert lifecycle.passed is True
    assert lifecycle.evidence["corrected_summaries"] == ("User name is Blink Lab",)
    assert lifecycle.evidence["current_claim_count_after_forget"] == 0
    assert transparency.passed is True
    assert transparency.evidence["used_in_current_turn"] is True
    assert transparency.evidence["safe_provenance_label"] == (
        "Remembered from your explicit preference."
    )
    assert behavior.passed is True
    assert behavior.evidence["adapted"]["length"] == "concise"
    assert behavior.evidence["adapted"]["teaching"] == "walkthrough"
    assert teaching.passed is True
    selected = teaching.evidence["selected_by_prompt"]
    assert "exemplar:debugging_hypothesis_one_change" in selected["debugging"]
    assert "exemplar:misconception_repair_without_shame" in selected["misconception"]
    assert "exemplar:chinese_technical_explanation_bridge" in selected["chinese"]
    assert "sequence:practice_prompt_with_answer_key" in selected["practice"]


def test_talking_to_blink_manual_relationship_voice_and_operator_cases():
    report = evaluate_talking_to_blink_manual_suite()
    relationship = _result(report, "relationship_boundary_guardrails")
    voice = _result(report, "voice_policy_metrics_are_observable")
    capability = _result(report, "capability_honesty_no_hardware_claims")
    operator = _result(report, "operator_payload_is_public_safe")

    assert relationship.passed is True
    assert relationship.evidence["boundaries"] == (
        "non-romantic",
        "non-sexual",
        "non-exclusive",
    )
    assert voice.passed is True
    assert voice.evidence["voice_policy"]["chunking_mode"] == "concise"
    assert voice.evidence["voice_policy"]["expression_controls_hardware"] is False
    assert voice.evidence["voice_metrics"]["response_count"] == 1
    assert voice.evidence["voice_metrics"]["chunk_count"] == 2
    assert capability.passed is True
    assert capability.evidence["identity_label"] == "Blink; local non-human system"
    assert capability.evidence["expression_controls_hardware"] is False
    assert capability.evidence["hardware_noop_present"] is True
    assert operator.passed is True
    assert operator.evidence["available"] is False
    assert "voice_metrics" in operator.evidence["sections"]


def test_talking_to_blink_manual_payload_excludes_raw_internals():
    report = evaluate_talking_to_blink_manual_suite()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True)
    banned_tokens = (
        "source_event_ids",
        "source_refs",
        "source_event_id",
        "event_id",
        "raw_json",
        "brain.db",
        ".db",
        "/tmp",
        "```json",
        "[BLINK_BRAIN_CONTEXT]",
        "private_working_memory",
        "private_scratchpad",
        "Traceback",
        "RuntimeError",
        "Blink Scholar-Companion",
        "memory_claim:",
    )

    assert all(token not in encoded for token in banned_tokens)


def test_talking_to_blink_manual_lazy_exports_and_single_case_helper():
    case = next(
        case
        for case in build_talking_to_blink_manual_suite()
        if case.case_id == "voice_policy_metrics_are_observable"
    )
    single = evaluate_talking_to_blink_manual_case(case)
    suite = _result(evaluate_talking_to_blink_manual_suite([case]), case.case_id)

    assert single.as_dict() == suite.as_dict()

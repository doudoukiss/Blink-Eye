import json
from dataclasses import replace
from pathlib import Path

from blink.brain.evals import (
    BROWSER_PERF_BENCH_CATEGORIES,
    BROWSER_PERF_BENCH_METRICS,
    BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES,
    BROWSER_PERF_BENCH_PROFILES,
    BROWSER_PERF_BENCH_SUITE_ID,
    build_browser_perf_bench_suite,
    evaluate_browser_perf_bench_case,
    evaluate_browser_perf_bench_suite,
    find_browser_perf_public_safety_violations,
    render_browser_perf_bench_human_rating_form,
    render_browser_perf_bench_markdown,
    render_browser_perf_bench_metrics_rows,
    render_browser_perf_bench_pairwise_form,
    write_browser_perf_bench_artifacts,
)


def _case(case_id: str):
    return next(case for case in build_browser_perf_bench_suite() if case.case_id == case_id)


def _check(result, check_id: str):
    return next(check for check in result.checks if check.check_id == check_id)


def test_browser_perf_bench_suite_covers_both_primary_profiles_and_categories():
    suite = build_browser_perf_bench_suite()

    assert set(BROWSER_PERF_BENCH_PROFILES).issubset({case.profile for case in suite})
    assert set(BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES).issubset(
        {case.profile for case in suite}
    )
    assert {case.category for case in suite} == set(BROWSER_PERF_BENCH_CATEGORIES)
    assert [case.case_id for case in suite] == [
        "browser_zh_melo_connection_listening",
        "browser_zh_melo_speech_subtitles",
        "browser_zh_melo_camera_single_frame_grounding",
        "browser_zh_melo_interruption_protected_default",
        "browser_en_kokoro_active_listening_final_only",
        "browser_en_kokoro_camera_moondream_parity",
        "browser_zh_melo_memory_persona_visible",
        "browser_zh_melo_track_stall_resume_observed",
        "native_en_kokoro_backend_isolation_guardrail",
        "native_en_kokoro_macos_camera_helper_isolation_guardrail",
    ]
    assert all("states" not in case.as_dict() for case in suite)
    assert all("events" not in case.as_dict() for case in suite)


def test_browser_perf_bench_report_is_deterministic_json_serializable_and_passes():
    first = evaluate_browser_perf_bench_suite()
    second = evaluate_browser_perf_bench_suite()
    rows = render_browser_perf_bench_metrics_rows(first)

    assert first.as_dict() == second.as_dict()
    assert rows == render_browser_perf_bench_metrics_rows(second)
    assert first.passed is True
    assert first.profile_coverage() == (
        *BROWSER_PERF_BENCH_PROFILES,
        *BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES,
    )
    assert {row["suite_id"] for row in rows} == {BROWSER_PERF_BENCH_SUITE_ID}
    assert json.loads(json.dumps(first.as_dict(), ensure_ascii=False, sort_keys=True)) == (
        first.as_dict()
    )


def test_browser_perf_bench_metrics_are_bounded_and_gated():
    payload = evaluate_browser_perf_bench_suite().as_dict()

    assert all(payload["gates"].values())
    assert payload["failed_gates"] == []
    for metric_name in BROWSER_PERF_BENCH_METRICS:
        assert 0.0 <= payload["aggregate_metrics"][metric_name] <= 1.0
    for row in payload["metrics_rows"]:
        for metric_name in BROWSER_PERF_BENCH_METRICS:
            assert 0.0 <= row[metric_name] <= 1.0
        assert row["reason_codes"]


def test_browser_perf_bench_profile_filters_pass_independently():
    zh = evaluate_browser_perf_bench_suite(profile="browser-zh-melo")
    en = evaluate_browser_perf_bench_suite(profile="browser-en-kokoro")

    assert zh.passed is True
    assert zh.profile_coverage() == ("browser-zh-melo",)
    assert zh.gate_results()["both_primary_profiles"] is True
    assert en.passed is True
    assert en.profile_coverage() == ("browser-en-kokoro",)
    assert en.gate_results()["both_primary_profiles"] is True


def test_browser_perf_bench_artifacts_and_forms_are_stable(tmp_path):
    report = evaluate_browser_perf_bench_suite()
    markdown = render_browser_perf_bench_markdown(report)
    human_form = render_browser_perf_bench_human_rating_form(report)
    pairwise_form = render_browser_perf_bench_pairwise_form(report)
    paths = write_browser_perf_bench_artifacts(report, output_dir=tmp_path)

    assert paths == {
        "human_rating_form": str(tmp_path / "human_rating_form.md"),
        "json": str(tmp_path / "latest.json"),
        "jsonl": str(tmp_path / "latest.jsonl"),
        "markdown": str(tmp_path / "latest.md"),
        "pairwise_form": str(tmp_path / "pairwise_form.md"),
    }
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == report.as_dict()
    jsonl_lines = (tmp_path / "latest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == len(report.results)
    expected_profiles = set(BROWSER_PERF_BENCH_PROFILES) | set(
        BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES
    )
    assert all(json.loads(line)["case"]["profile"] in expected_profiles for line in jsonl_lines)
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == f"{markdown}\n"
    assert (tmp_path / "human_rating_form.md").read_text(encoding="utf-8") == f"{human_form}\n"
    assert (tmp_path / "pairwise_form.md").read_text(encoding="utf-8") == f"{pairwise_form}\n"
    for label in (
        "state clarity",
        "felt-heard",
        "voice pacing",
        "memory usefulness",
        "camera grounding",
        "interruption naturalness",
        "enjoyment",
    ):
        assert label in human_form
        assert label in pairwise_form
    assert write_browser_perf_bench_artifacts(report, output_dir=tmp_path) == paths


def test_browser_perf_bench_rejects_unsafe_public_payload_metadata():
    case = _case("browser_zh_melo_connection_listening")
    bad_event = {
        **case.events[0],
        "metadata": {
            "transcript": "raw user words",
            "prompt": "hidden prompt",
            "audio_bytes": "abc123",
        },
    }
    bad_case = replace(case, events=(bad_event, *case.events[1:]))

    result = evaluate_browser_perf_bench_case(bad_case)
    violations = find_browser_perf_public_safety_violations(
        {"states": bad_case.states, "events": bad_case.events}
    )

    assert result.passed is False
    assert _check(result, "public_safety").passed is False
    assert violations
    assert result.metric_row.holistic_signal_score == 0.0


def test_browser_perf_bench_rejects_missing_subtitle_before_audio():
    case = _case("browser_zh_melo_speech_subtitles")
    bad_events = tuple(
        event for event in case.events if event["event_type"] != "speech.subtitle_ready"
    )
    bad_case = replace(case, events=bad_events)

    result = evaluate_browser_perf_bench_case(bad_case)

    assert result.passed is False
    assert _check(result, "event_order").passed is False
    assert _check(result, "subtitle_readiness").passed is False
    assert result.metric_row.subtitle_score == 0.0


def test_browser_perf_bench_rejects_stale_camera_vision_claims():
    case = _case("browser_zh_melo_camera_single_frame_grounding")
    final_state = {
        **case.states[-1],
        "camera_presence": {
            **case.states[-1]["camera_presence"],
            "state": "stale",
            "current_answer_used_vision": True,
            "grounding_mode": "single_frame",
        },
        "camera_scene": {
            **case.states[-1]["camera_scene"],
            "state": "stale",
            "status": "stale",
            "current_answer_used_vision": True,
            "grounding_mode": "single_frame",
        },
    }
    bad_case = replace(case, states=(final_state,))

    result = evaluate_browser_perf_bench_case(bad_case)

    assert result.passed is False
    assert _check(result, "camera_grounding").passed is False
    assert result.metric_row.camera_grounding_score == 0.0


def test_browser_perf_bench_rejects_wrong_barge_in_default():
    case = _case("browser_zh_melo_interruption_protected_default")
    final_state = {
        **case.states[-1],
        "protected_playback": False,
        "interruption": {
            **case.states[-1]["interruption"],
            "protected_playback": False,
            "barge_in_state": "armed",
            "last_decision": "accepted",
        },
    }
    accepted_event = {
        **case.events[2],
        "event_id": 3,
        "event_type": "interruption.accepted",
    }
    bad_case = replace(case, states=(final_state,), events=(case.events[0], case.events[1], accepted_event))

    result = evaluate_browser_perf_bench_case(bad_case)

    assert result.passed is False
    assert _check(result, "profile_defaults").passed is False
    assert _check(result, "barge_in_safety").passed is False
    assert result.metric_row.barge_in_safety_score == 0.0


def test_browser_perf_bench_rejects_native_isolation_default_regression():
    case = _case("native_en_kokoro_backend_isolation_guardrail")
    final_state = {
        **case.states[-1],
        "protected_playback": False,
        "native_audio": {
            **case.states[-1]["native_audio"],
            "barge_in": "on",
        },
        "interruption": {
            **case.states[-1]["interruption"],
            "protected_playback": False,
            "barge_in_state": "armed",
            "last_decision": "accepted",
        },
    }
    accepted_event = {
        **case.events[0],
        "event_id": 2,
        "event_type": "interruption.accepted",
    }
    bad_case = replace(case, states=(final_state,), events=(case.events[0], accepted_event))

    result = evaluate_browser_perf_bench_case(bad_case)

    assert result.passed is False
    assert _check(result, "profile_defaults").passed is False
    assert _check(result, "native_guardrail").passed is False
    assert result.metric_row.barge_in_safety_score == 0.0


def test_browser_perf_bench_shell_wrappers_are_profile_aware():
    root = Path(__file__).resolve().parents[1]
    perf_wrapper = (root / "scripts/eval-browser-perf.sh").read_text(encoding="utf-8")
    melo_wrapper = (root / "scripts/eval-browser-melo-perf.sh").read_text(encoding="utf-8")

    assert "run-browser-perf-bench.py" in perf_wrapper
    assert "--profile browser-zh-melo" in melo_wrapper
    assert "run-local-browser" not in perf_wrapper
    assert "run-local-browser" not in melo_wrapper

import json
from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.evals.bilingual_actor_bench import (
    BILINGUAL_ACTOR_CATEGORIES,
    BILINGUAL_ACTOR_HARD_BLOCKERS,
    BILINGUAL_ACTOR_PRIMARY_PROFILES,
    BILINGUAL_ACTOR_QUALITY_DIMENSIONS,
    BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES,
    BILINGUAL_ACTOR_RELEASE_THRESHOLD,
    BILINGUAL_PERFORMANCE_BENCH_V3_CATEGORIES,
    BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS,
    BILINGUAL_PERFORMANCE_BENCH_V3_METRICS,
    BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION,
    BILINGUAL_PERFORMANCE_BENCH_V3_SCORE_DIMENSIONS,
    BILINGUAL_PERFORMANCE_BENCH_V3_SUITE_ID,
    build_bilingual_actor_bench_suite,
    build_bilingual_performance_bench_v3_suite,
    evaluate_bilingual_actor_bench_suite,
    evaluate_bilingual_performance_bench_v3,
    load_bilingual_actor_historical_regression_results,
    render_bilingual_actor_bench_human_rating_form,
    render_bilingual_actor_bench_metrics_rows,
    render_bilingual_actor_bench_pairwise_form,
    write_bilingual_actor_bench_artifacts,
    write_bilingual_performance_bench_v3_artifacts,
)
from blink.brain.evals.performance_intelligence_baseline import (
    PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH,
    PRIMARY_BROWSER_PROFILES,
    render_performance_intelligence_baseline_json,
)
from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_DIMENSIONS,
    PerformancePreferenceStore,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "actor_bench_result.schema.json"
V3_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "bilingual_performance_bench_v3_result.schema.json"
)


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _v3_schema_validator() -> Draft202012Validator:
    schema = json.loads(V3_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_v3_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_v3_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _preference_payload(profile: str = "browser-zh-melo") -> dict[str, object]:
    language = "en" if profile == "browser-en-kokoro" else "zh"
    tts_label = "kokoro/English" if language == "en" else "local-http-wav/MeloTTS"
    candidate_common = {
        "profile": profile,
        "language": language,
        "tts_runtime_label": tts_label,
        "episode_ids": ["episode-v3"],
        "plan_ids": ["plan-v3"],
        "control_frame_ids": ["control-v3"],
        "segment_counts": {"listen_segment": 1, "speak_segment": 1},
        "metric_counts": {"latency_count": 2},
        "camera_honesty_states": ["available_not_used"],
    }
    return {
        "schema_version": 3,
        "profile": profile,
        "language": language,
        "tts_runtime_label": tts_label,
        "candidate_a": {
            **candidate_common,
            "candidate_id": f"{profile}:baseline",
            "candidate_kind": "baseline_trace",
            "candidate_label": "Baseline",
            "public_summary": "Baseline public-safe summary.",
            "policy_labels": ["baseline"],
            "reason_codes": ["candidate:baseline"],
        },
        "candidate_b": {
            **candidate_common,
            "candidate_id": f"{profile}:candidate",
            "candidate_kind": "candidate_trace",
            "candidate_label": "Candidate",
            "public_summary": "Candidate public-safe summary.",
            "policy_labels": ["candidate"],
            "reason_codes": ["candidate:current"],
        },
        "winner": "b",
        "ratings": {dimension: 4 for dimension in PERFORMANCE_PREFERENCE_DIMENSIONS},
        "failure_labels": ["voice_pacing_too_long"],
        "improvement_labels": ["voice_pacing_better"],
        "evidence_refs": [
            {
                "evidence_kind": "episode",
                "evidence_id": "episode-v3",
                "summary": "Public-safe replay evidence.",
                "reason_codes": ["evidence:episode"],
            }
        ],
        "reason_codes": ["test:performance_preference"],
    }


def test_bilingual_actor_bench_report_is_deterministic_schema_valid_and_passing():
    first = evaluate_bilingual_actor_bench_suite()
    second = evaluate_bilingual_actor_bench_suite()
    payload = first.as_dict()

    _assert_schema_valid(payload)
    assert payload == second.as_dict()
    assert first.passed is True
    assert first.release_gate.threshold == BILINGUAL_ACTOR_RELEASE_THRESHOLD
    assert first.release_gate.hard_blockers == ()
    assert render_bilingual_actor_bench_metrics_rows(first) == tuple(
        result.as_dict() for result in first.profile_results
    )
    assert json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)) == payload


def test_bilingual_performance_bench_v3_report_is_deterministic_schema_valid_and_passing(tmp_path):
    first = evaluate_bilingual_performance_bench_v3(preferences_dir=tmp_path)
    second = evaluate_bilingual_performance_bench_v3(preferences_dir=tmp_path)
    payload = first.as_dict()

    _assert_v3_schema_valid(payload)
    assert payload == second.as_dict()
    assert first.schema_version == BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION
    assert first.suite_id == BILINGUAL_PERFORMANCE_BENCH_V3_SUITE_ID
    assert first.passed is True
    assert first.release_gate.hard_blockers == ()
    assert set(BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS) == {
        "profile_regression",
        "hidden_camera_use",
        "false_camera_claim",
        "self_interruption",
        "stale_tts_after_interrupt",
        "memory_contradiction",
        "unsupported_tts_claim",
        "unsafe_trace_payload",
        "missing_consent_controls",
        "realistic_human_avatar_capability",
    }
    assert set(first.aggregate_metrics) == set(BILINGUAL_PERFORMANCE_BENCH_V3_METRICS)
    assert set(first.release_gate.parity_deltas) == {
        *BILINGUAL_PERFORMANCE_BENCH_V3_SCORE_DIMENSIONS,
        "bilingual_parity_delta",
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)) == payload


def test_bilingual_performance_bench_v3_suite_has_matched_categories(tmp_path):
    suite = build_bilingual_performance_bench_v3_suite(preferences_dir=tmp_path)

    assert len(suite) == len(BILINGUAL_PERFORMANCE_BENCH_V3_CATEGORIES) * len(
        BILINGUAL_ACTOR_PRIMARY_PROFILES
    )
    assert {case.profile for case in suite} == set(BILINGUAL_ACTOR_PRIMARY_PROFILES)
    for category in BILINGUAL_PERFORMANCE_BENCH_V3_CATEGORIES:
        category_cases = [case for case in suite if case.category == category]
        assert {case.profile for case in category_cases} == set(BILINGUAL_ACTOR_PRIMARY_PROFILES)
        assert {case.pair_id for case in category_cases} == {category}


def test_bilingual_performance_bench_v3_artifacts_are_stable(tmp_path):
    report = evaluate_bilingual_performance_bench_v3(preferences_dir=tmp_path)
    paths = write_bilingual_performance_bench_v3_artifacts(report, output_dir=tmp_path)

    assert paths == {
        "json": str(tmp_path / "latest_v3.json"),
        "jsonl": str(tmp_path / "latest_v3.jsonl"),
        "markdown": str(tmp_path / "latest_v3.md"),
        "release_checklist": str(tmp_path / "release_checklist_v3.md"),
    }
    assert json.loads((tmp_path / "latest_v3.json").read_text(encoding="utf-8")) == report.as_dict()
    assert len((tmp_path / "latest_v3.jsonl").read_text(encoding="utf-8").splitlines()) == len(
        report.case_results
    )
    markdown = (tmp_path / "latest_v3.md").read_text(encoding="utf-8")
    checklist = (tmp_path / "release_checklist_v3.md").read_text(encoding="utf-8")
    assert "Blink Bilingual Performance Bench V3" in markdown
    assert "Bilingual Performance Release Checklist V3" in checklist


def test_bilingual_performance_bench_v3_preference_jsonl_enriches_optional_evidence(tmp_path):
    store = PerformancePreferenceStore(tmp_path)
    pair, proposals, report = store.record_pair(_preference_payload("browser-en-kokoro"))
    assert pair is not None
    assert proposals
    assert report.accepted is True

    bench = evaluate_bilingual_performance_bench_v3(preferences_dir=tmp_path)

    assert bench.preference_review["real_pair_count"] == 1
    assert bench.preference_review["real_proposal_count"] >= 1
    assert bench.preference_review["profile_counts"] == {"browser-en-kokoro": 1}
    assert pair.pair_id in bench.preference_review["real_pair_ids"]
    assert bench.passed is True


def test_bilingual_performance_bench_v3_unsafe_preference_jsonl_is_ignored(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "preferences.jsonl").write_text(
        json.dumps(
            {
                **_preference_payload("browser-zh-melo"),
                "raw_audio": "data:audio/wav;base64,AAAA",
                "hidden_prompt": "secret system prompt",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bench = evaluate_bilingual_performance_bench_v3(preferences_dir=tmp_path)
    encoded = json.dumps(bench.as_dict(), ensure_ascii=False).lower()

    assert bench.preference_review["real_pair_count"] == 0
    assert bench.preference_review["sanitizer_pass"] is True
    assert "data:audio" not in encoded
    assert "secret system prompt" not in encoded
    assert bench.passed is True


def test_bilingual_performance_bench_v3_parity_delta_blocks_release(tmp_path):
    suite = tuple(
        replace(
            case,
            quality_scores={
                **case.quality_scores,
                "state_clarity": 5.0,
            },
        )
        if case.profile == "browser-zh-melo"
        else case
        for case in build_bilingual_performance_bench_v3_suite(preferences_dir=tmp_path)
    )

    report = evaluate_bilingual_performance_bench_v3(cases=suite, preferences_dir=tmp_path)

    assert report.passed is False
    assert report.release_gate.hard_blockers == ()
    assert report.release_gate.parity_deltas["bilingual_parity_delta"] > 0.35
    assert set(report.release_gate.profile_failures) == set(BILINGUAL_ACTOR_PRIMARY_PROFILES)


def test_performance_intelligence_baseline_fixture_locks_primary_browser_paths():
    rendered = render_performance_intelligence_baseline_json()
    payload = json.loads(rendered)

    assert PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH.read_text(encoding="utf-8") == rendered
    assert payload["snapshot_type"] == "PerformanceIntelligenceBaseline"
    assert payload["native_pyaudio_policy"] == "backend_isolation_only_not_product_ux"
    assert payload["native_pyaudio_isolation_lanes"] == [
        "native-en-kokoro",
        "native-en-kokoro-macos-camera",
    ]

    profiles = {profile["profile"]: profile for profile in payload["profiles"]}
    assert tuple(profiles) == PRIMARY_BROWSER_PROFILES
    profile_labels = ("language", "tts_backend", "tts_runtime_label")
    assert {key: profiles["browser-zh-melo"][key] for key in profile_labels} == {
        "language": "zh",
        "tts_backend": "local-http-wav",
        "tts_runtime_label": "local-http-wav/MeloTTS",
    }
    assert {key: profiles["browser-en-kokoro"][key] for key in profile_labels} == {
        "language": "en",
        "tts_backend": "kokoro",
        "tts_runtime_label": "kokoro/English",
    }

    for profile in profiles.values():
        assert profile["product_status"] == "equal_primary_browser_product_path"
        assert profile["isolation_lane"] is False
        assert profile["client_url"] == "http://127.0.0.1:7860/client/"
        assert profile["media"] == "browser/WebRTC microphone + camera"
        assert profile["vision_backend"] == "moondream"
        assert profile["browser_vision_default"] is True
        assert profile["continuous_perception_default"] is False
        assert profile["protected_playback_default"] is True
        assert profile["allow_barge_in_default"] is False
        assert profile["actor_state_endpoints"] == [
            "/api/runtime/actor-state",
            "/api/runtime/performance-state",
        ]
        assert profile["actor_event_endpoints"] == [
            "/api/runtime/actor-events",
            "/api/runtime/performance-events",
        ]
        assert "./scripts/eval-bilingual-actor-bench.sh" in profile["gate_commands"]


def test_performance_intelligence_canonical_launchers_keep_product_defaults():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(render_performance_intelligence_baseline_json())
    forbidden_demotion_terms = (
        "debug-only",
        "debug only",
        "isolation-only",
        "isolation=backend-only",
        "backend isolation",
        "secondary path",
    )

    for profile in payload["profiles"]:
        launcher = root / profile["launcher"].removeprefix("./")
        text = launcher.read_text(encoding="utf-8")
        lowered = text.lower()

        assert f"BLINK_LOCAL_CONFIG_PROFILE={profile['profile']}" in text
        assert f"profile={profile['profile']}" in text
        assert f"language={profile['language']}" in text
        assert f'TTS_RUNTIME_LABEL="{profile["tts_runtime_label"]}"' in text
        assert 'DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"' in text
        assert 'DEFAULT_CONTINUOUS_PERCEPTION="${BLINK_LOCAL_CONTINUOUS_PERCEPTION:-0}"' in text
        assert 'DEFAULT_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"' in text
        assert 'BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION"' in text
        assert 'BLINK_LOCAL_CONTINUOUS_PERCEPTION="$DEFAULT_CONTINUOUS_PERCEPTION"' in text
        assert 'BLINK_LOCAL_ALLOW_BARGE_IN="$DEFAULT_ALLOW_BARGE_IN"' in text
        assert "camera_vision=$(launcher_state_label \"$DEFAULT_BROWSER_VISION\")" in text
        assert (
            "protected_playback=$(launcher_protected_playback_label "
            "\"$DEFAULT_ALLOW_BARGE_IN\")"
        ) in text
        assert f"primary {profile['profile']} WebRTC actor path" in text
        assert "browser vision available by default" in text
        for demotion_term in forbidden_demotion_terms:
            assert demotion_term not in lowered


def test_bilingual_actor_bench_suite_has_matched_primary_profile_cases():
    suite = build_bilingual_actor_bench_suite()

    assert len(suite) == len(BILINGUAL_ACTOR_CATEGORIES) * len(BILINGUAL_ACTOR_PRIMARY_PROFILES)
    assert {case.profile for case in suite} == set(BILINGUAL_ACTOR_PRIMARY_PROFILES)
    for category in BILINGUAL_ACTOR_CATEGORIES:
        category_cases = [case for case in suite if case.category == category]
        assert {case.profile for case in category_cases} == set(BILINGUAL_ACTOR_PRIMARY_PROFILES)
        assert {case.pair_id for case in category_cases} == {category}


def test_bilingual_actor_bench_wires_historical_regression_fixtures():
    fixture_results = load_bilingual_actor_historical_regression_results()
    suite = build_bilingual_actor_bench_suite()

    assert fixture_results
    assert {result.category for result in fixture_results} == set(
        BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES
    )
    assert all(result.passed for result in fixture_results)
    assert all(result.profile in BILINGUAL_ACTOR_PRIMARY_PROFILES for result in fixture_results)

    attached = [
        regression
        for case in suite
        for regression in case.historical_regressions
    ]
    assert sorted(
        (result.as_dict() for result in attached),
        key=lambda result: (result["category"], result["profile"], result["fixture_id"]),
    ) == sorted(
        (result.as_dict() for result in fixture_results),
        key=lambda result: (result["category"], result["profile"], result["fixture_id"]),
    )

    report = evaluate_bilingual_actor_bench_suite(suite)
    for result in report.case_results:
        historical_checks = [
            check for check in result.checks if check.check_id == "historical_regressions"
        ]
        assert len(historical_checks) == 1
        assert historical_checks[0].passed is True
    assert any(
        result.evidence["historical_regressions"]
        for result in report.case_results
        if result.case.category in BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES
    )


def test_bilingual_actor_profile_results_have_same_structure_except_labels():
    report = evaluate_bilingual_actor_bench_suite()
    zh, en = report.profile_results

    assert set(zh.as_dict()) == set(en.as_dict())
    assert set(zh.category_results) == set(en.category_results) == set(BILINGUAL_ACTOR_CATEGORIES)
    assert set(zh.scores) == set(en.scores) == set(BILINGUAL_ACTOR_QUALITY_DIMENSIONS)
    assert zh.profile == "browser-zh-melo"
    assert en.profile == "browser-en-kokoro"
    assert zh.language == "zh"
    assert en.language == "en"
    assert zh.tts_label == "local-http-wav/MeloTTS"
    assert en.tts_label == "kokoro/English"


def test_bilingual_actor_forms_and_artifacts_are_stable(tmp_path):
    report = evaluate_bilingual_actor_bench_suite()
    form = render_bilingual_actor_bench_human_rating_form(report)
    pairwise = render_bilingual_actor_bench_pairwise_form(report)
    paths = write_bilingual_actor_bench_artifacts(report, output_dir=tmp_path)

    assert paths == {
        "human_rating_form": str(tmp_path / "human_rating_form.md"),
        "json": str(tmp_path / "latest.json"),
        "jsonl": str(tmp_path / "latest.jsonl"),
        "markdown": str(tmp_path / "latest.md"),
        "pairwise_form": str(tmp_path / "pairwise_form.md"),
    }
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == report.as_dict()
    assert len((tmp_path / "latest.jsonl").read_text(encoding="utf-8").splitlines()) == len(
        report.case_results
    )
    assert (tmp_path / "human_rating_form.md").read_text(encoding="utf-8") == f"{form}\n"
    assert (tmp_path / "pairwise_form.md").read_text(encoding="utf-8") == f"{pairwise}\n"

    for label in (
        "State clarity",
        "Felt-heard",
        "Voice pacing",
        "Camera grounding",
        "Memory usefulness",
        "Interruption naturalness",
        "Personality consistency",
        "Enjoyment",
        "Not fake-human",
    ):
        assert label in form
    for blocker in BILINGUAL_ACTOR_HARD_BLOCKERS:
        assert blocker.replace("_", " ") in form
    for category in BILINGUAL_ACTOR_CATEGORIES:
        assert category in pairwise


def test_bilingual_actor_seed_cases_cover_all_matched_categories():
    seed_path = Path(__file__).resolve().parents[1] / "evals" / "bilingual_actor_bench" / "seed_cases.jsonl"
    records = [
        json.loads(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == len(BILINGUAL_ACTOR_CATEGORIES) * len(BILINGUAL_ACTOR_PRIMARY_PROFILES)
    for category in BILINGUAL_ACTOR_CATEGORIES:
        records_for_category = [record for record in records if record["category"] == category]
        assert {record["profile"] for record in records_for_category} == set(
            BILINGUAL_ACTOR_PRIMARY_PROFILES
        )
        assert {record["pair_id"] for record in records_for_category} == {category}

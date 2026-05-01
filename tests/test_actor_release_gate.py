import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.brain.evals.bilingual_actor_bench import (
    BILINGUAL_ACTOR_RELEASE_THRESHOLD,
    BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS,
    BilingualActorHistoricalRegressionResult,
    build_bilingual_actor_bench_suite,
    build_bilingual_performance_bench_v3_suite,
    evaluate_bilingual_actor_bench_case,
    evaluate_bilingual_actor_bench_suite,
    evaluate_bilingual_performance_bench_v3_case,
)
from blink.interaction import (
    ActorEventModeV2,
    ActorEventTypeV2,
    ActorEventV2,
    compile_avatar_adapter_event_contract,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "release_gate.schema.json"


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _case(case_id: str):
    return next(case for case in build_bilingual_actor_bench_suite() if case.case_id == case_id)


def _v3_case(case_id: str):
    return next(case for case in build_bilingual_performance_bench_v3_suite() if case.case_id == case_id)


def _result_with_case(bad_case):
    cases = tuple(
        bad_case if case.case_id == bad_case.case_id else case
        for case in build_bilingual_actor_bench_suite()
    )
    return evaluate_bilingual_actor_bench_suite(cases=cases)


def test_bilingual_actor_release_gate_schema_validates_and_passes_clean_suite():
    report = evaluate_bilingual_actor_bench_suite()
    payload = report.release_gate.as_dict()

    _assert_schema_valid(payload)
    assert payload["passed"] is True
    assert payload["threshold"] == BILINGUAL_ACTOR_RELEASE_THRESHOLD
    assert payload["hard_blockers"] == []
    assert payload["profile_failures"] == {}


def test_bilingual_actor_release_gate_fails_independent_profile_score_regression():
    case = _case("zh_connection")
    bad_case = replace(case, quality_scores={**case.quality_scores, "state_clarity": 3.9})

    report = _result_with_case(bad_case)

    assert report.release_gate.passed is False
    assert "browser-zh-melo" in report.release_gate.profile_failures
    assert "browser-en-kokoro" not in report.release_gate.profile_failures
    assert any(
        failure == "score_below_threshold:state_clarity"
        for failure in report.release_gate.profile_failures["browser-zh-melo"]
    )


def test_bilingual_actor_release_gate_fails_english_profile_score_regression():
    case = _case("en_connection")
    bad_case = replace(case, quality_scores={**case.quality_scores, "felt_heard": 3.8})

    report = _result_with_case(bad_case)

    assert report.release_gate.passed is False
    assert "browser-en-kokoro" in report.release_gate.profile_failures
    assert "browser-zh-melo" not in report.release_gate.profile_failures


def test_bilingual_actor_release_gate_blocks_unsafe_trace_payload():
    case = _case("zh_connection")
    bad_event = {
        **case.actor_events[0],
        "metadata": {"prompt": "hidden prompt", "audio": "data:audio/wav;base64,abc"},
    }
    bad_case = replace(case, actor_events=(bad_event, *case.actor_events[1:]))

    result = evaluate_bilingual_actor_bench_case(bad_case)
    report = _result_with_case(bad_case)

    assert "unsafe_trace_payload" in result.hard_blockers
    assert "unsafe_trace_payload" in report.release_gate.hard_blockers


def test_bilingual_actor_release_gate_blocks_hidden_camera_use():
    case = _case("zh_camera_grounding")
    bad_case = replace(case, actor_events=())

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "hidden_camera_use" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_false_camera_claim():
    case = _case("en_connection")
    bad_state = {
        **case.actor_state,
        "answer_claims_vision": True,
        "camera_scene": {
            **case.actor_state["camera_scene"],
            "current_answer_used_vision": False,
            "grounding_mode": "none",
        },
    }
    bad_case = replace(case, actor_state=bad_state)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "false_camera_claim" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_self_interruption():
    case = _case("en_interruption")
    bad_state = {
        **case.actor_state,
        "interruption": {
            **case.actor_state["interruption"],
            "self_interruption": True,
            "last_decision": "accepted",
        },
    }
    bad_case = replace(case, actor_state=bad_state)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "self_interruption" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_stale_tts_after_interruption():
    case = _case("zh_interruption")
    bad_state = {
        **case.actor_state,
        "speech": {
            **case.actor_state["speech"],
            "stale_output_played_after_interruption": True,
        },
    }
    bad_case = replace(case, actor_state=bad_state)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "stale_tts_after_interruption" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_memory_contradiction():
    case = _case("en_memory_persona")
    bad_state = {
        **case.actor_state,
        "memory_persona": {
            **case.actor_state["memory_persona"],
            "memory_contradiction": True,
        },
    }
    bad_case = replace(case, actor_state=bad_state)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "memory_contradiction" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_profile_regression():
    case = _case("en_connection")
    bad_state = {
        **case.actor_state,
        "tts": {"backend": "local-http-wav", "label": "local-http-wav/MeloTTS"},
    }
    bad_case = replace(case, actor_state=bad_state)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "profile_regression" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_historical_regression_fixture_failure():
    case = _case("en_camera_grounding")
    failed_regression = BilingualActorHistoricalRegressionResult(
        fixture_file="regression_camera_moondream.jsonl",
        fixture_id="historical_false_camera_claim",
        category="camera_grounding",
        profile="browser-en-kokoro",
        passed=False,
        hard_blockers=("false_camera_claim",),
        reason_codes=("historical_fixture:missing_vision_success",),
    )
    bad_case = replace(case, historical_regressions=(failed_regression,))

    result = evaluate_bilingual_actor_bench_case(bad_case)
    report = _result_with_case(bad_case)

    assert "false_camera_claim" in result.hard_blockers
    assert "false_camera_claim" in report.release_gate.hard_blockers
    assert "browser-en-kokoro" in report.release_gate.profile_failures


def test_bilingual_actor_release_gate_blocks_missing_privacy_controls():
    case = _case("zh_connection")
    bad_controls = {**case.privacy_controls, "debug_transcript_storage_default": "on"}
    bad_case = replace(case, privacy_controls=bad_controls)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "missing_consent_or_privacy_control" in result.hard_blockers


def test_bilingual_actor_release_gate_blocks_realistic_avatar_capability():
    case = _case("en_connection")
    bad_contract = {**case.avatar_contract, "realistic_human_likeness_allowed": True}
    bad_case = replace(case, avatar_contract=bad_contract)

    result = evaluate_bilingual_actor_bench_case(bad_case)

    assert "realistic_human_avatar_capability" in result.hard_blockers


@pytest.mark.parametrize("blocker", BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS)
def test_bilingual_performance_v3_release_gate_blocks_hard_blockers(blocker):
    case = _v3_case(
        {
            "profile_regression": "en_connection_v3",
            "hidden_camera_use": "zh_camera_grounding_v3",
            "false_camera_claim": "en_connection_v3",
            "self_interruption": "zh_repair_v3",
            "stale_tts_after_interrupt": "zh_repair_v3",
            "memory_contradiction": "en_memory_persona_v3",
            "unsupported_tts_claim": "en_speech_v3",
            "unsafe_trace_payload": "zh_connection_v3",
            "missing_consent_controls": "zh_safety_controls_v3",
            "realistic_human_avatar_capability": "en_safety_controls_v3",
        }[blocker]
    )
    state = case.actor_state
    events = case.actor_events
    privacy_controls = case.privacy_controls
    avatar_contract = case.avatar_contract
    plan_summaries = case.plan_summaries

    if blocker == "profile_regression":
        state = {**state, "tts": {"backend": "local-http-wav", "label": "local-http-wav/MeloTTS"}}
    elif blocker == "hidden_camera_use":
        events = ()
    elif blocker == "false_camera_claim":
        state = {
            **state,
            "answer_claims_vision": True,
            "camera_scene": {
                **state["camera_scene"],
                "current_answer_used_vision": False,
                "scene_social_state_v2": {
                    **state["camera_scene"]["scene_social_state_v2"],
                    "camera_honesty_state": "available_not_used",
                },
            },
        }
    elif blocker == "self_interruption":
        state = {
            **state,
            "protected_playback": True,
            "interruption": {
                **state["interruption"],
                "last_decision": "accepted",
                "echo_safe": False,
                "self_interruption": True,
            },
        }
    elif blocker == "stale_tts_after_interrupt":
        state = {
            **state,
            "speech": {
                **state["speech"],
                "stale_output_played_after_interruption": True,
            },
        }
    elif blocker == "memory_contradiction":
        state = {
            **state,
            "memory_persona": {
                **state["memory_persona"],
                "memory_contradiction": True,
            },
        }
    elif blocker == "unsupported_tts_claim":
        state = {
            **state,
            "speech": {
                **state["speech"],
                "backend_capabilities": {
                    **state["speech"]["backend_capabilities"],
                    "emotional_prosody": True,
                },
            },
        }
        plan_summaries = (
            {
                **case.plan_summaries[0],
                "tts_capabilities": {
                    **case.plan_summaries[0]["tts_capabilities"],
                    "speech_rate_control": True,
                },
            },
        )
    elif blocker == "unsafe_trace_payload":
        events = (
            {
                **events[0],
                "metadata": {
                    "raw_image": "data:image/png;base64,AAAA",
                    "hidden_prompt": "do not expose",
                },
            },
            *events[1:],
        )
    elif blocker == "missing_consent_controls":
        privacy_controls = {**privacy_controls, "debug_transcript_storage_default": "on"}
    elif blocker == "realistic_human_avatar_capability":
        avatar_contract = {**avatar_contract, "realistic_human_likeness_allowed": True}

    bad_case = replace(
        case,
        actor_state=state,
        actor_events=events,
        privacy_controls=privacy_controls,
        avatar_contract=avatar_contract,
        plan_summaries=plan_summaries,
    )

    result = evaluate_bilingual_performance_bench_v3_case(bad_case)

    assert blocker in result.hard_blockers


def test_bilingual_performance_v3_blocks_camera_use_claim_when_only_looking_event_exists():
    case = _v3_case("zh_camera_grounding_v3")
    events_without_fresh_success = tuple(
        event
        for event in case.actor_events
        if event.get("event_type") != "vision.fetch_user_image_success"
    )
    bad_case = replace(case, actor_events=events_without_fresh_success)

    result = evaluate_bilingual_performance_bench_v3_case(bad_case)

    assert "looking" in {event.get("event_type") for event in events_without_fresh_success}
    assert "hidden_camera_use" in result.hard_blockers
    assert "false_camera_claim" in result.hard_blockers


def test_avatar_adapter_contract_sanitizes_actor_event_and_forbids_realistic_human_surface():
    event = ActorEventV2(
        event_id=7,
        event_type=ActorEventTypeV2.LOOKING,
        mode=ActorEventModeV2.LOOKING,
        profile="browser-en-kokoro",
        language="en",
        tts_backend="kokoro",
        tts_label="kokoro/English",
        vision_backend="moondream",
        source="test",
        metadata={
            "grounding_mode": "single_frame",
            "frame_count": 1,
            "raw_image": "data:image/png;base64,abc",
            "hidden_prompt": "do not expose",
        },
        reason_codes=["vision:single_frame"],
    )
    contract = compile_avatar_adapter_event_contract(
        event,
        performance_plan={
            "stance": "honest_camera_grounding",
            "style_summary": "Compact and clear.",
            "persona_references_used": [{"id": "camera_use"}],
            "system_prompt": "hidden prompt",
        },
        adapter_surface="realistic_human",
    ).as_dict()

    assert contract["adapter_surface"] == "status_avatar"
    assert contract["realistic_human_likeness_allowed"] is False
    assert contract["identity_cloning_allowed"] is False
    assert contract["face_reenactment_allowed"] is False
    assert contract["raw_media_allowed"] is False
    assert contract["metadata"] == {"grounding_mode": "single_frame", "frame_count": 1}
    assert contract["performance_plan_summary"]["style_summary"] == "Compact and clear."
    encoded = json.dumps(contract, ensure_ascii=False).lower()
    assert "raw_image" not in encoded
    assert "hidden_prompt" not in encoded
    assert "system_prompt" not in encoded


def test_avatar_adapter_contract_accepts_control_frame_and_plan_summary_without_raw_payloads():
    contract = compile_avatar_adapter_event_contract(
        actor_control_frame={
            "schema_version": 3,
            "frame_id": "control-v3:test",
            "sequence": 42,
            "profile": "browser-zh-melo",
            "language": "zh",
            "tts_runtime_label": "local-http-wav/MeloTTS",
            "boundary": "camera_frame_boundary",
            "source_event_ids": [1, 2, 3],
            "camera_policy": {
                "honesty_state": "can_see_now",
                "raw_image": "data:image/png;base64,AAAA",
            },
            "speech_policy": {"state": "bounded", "outstanding_chunks": 1},
            "reason_trace": ["camera:frame_captured"],
        },
        performance_plan={
            "stance": "grounded_and_concise",
            "response_shape": "answer_first",
            "plan_summary": "Camera evidence is fresh.",
            "system_prompt": "hidden prompt",
            "raw_memory_body": "private memory",
        },
        adapter_surface="symbolic_avatar",
    ).as_dict()

    assert contract["adapter_surface"] == "symbolic_avatar"
    assert contract["event_type"] == "camera_frame_boundary"
    assert contract["actor_control_frame_summary"]["frame_id"] == "control-v3:test"
    assert contract["actor_control_frame_summary"]["source_event_count"] == 3
    assert contract["actor_control_frame_summary"]["camera_policy"] == {
        "honesty_state": "can_see_now"
    }
    assert contract["performance_plan_summary"]["plan_summary"] == "Camera evidence is fresh."
    assert set(contract["source_contracts"]) == {
        "actor_event",
        "actor_control_frame_v3",
        "performance_plan_summary",
    }
    encoded = json.dumps(contract, ensure_ascii=False).lower()
    assert "data:image" not in encoded
    assert "hidden prompt" not in encoded
    assert "private memory" not in encoded

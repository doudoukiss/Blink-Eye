import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.interaction.active_listening import (
    ActiveListenerPhaseV2,
    ActiveListenerStateV2,
    build_semantic_listener_state_v3,
    extract_active_listening_understanding,
)

SCHEMA = Path("schemas/active_listener_state_v2.schema.json")
SEMANTIC_FIXTURES = Path("tests/fixtures/semantic_listener_cases_zh_en.jsonl")


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))


def test_active_listener_v2_validates_chinese_understanding_snapshot():
    understanding = extract_active_listening_understanding(
        "帮我做 Blink 浏览器主动倾听。必须同时支持 MeloTTS 和 Kokoro。"
        "不要保存完整转写。更正一下，叫 Active Listener v2。可能 STT 不稳定。",
        language="zh",
    )
    state = ActiveListenerStateV2(
        profile="browser-zh-melo",
        language="zh",
        available=True,
        phase=ActiveListenerPhaseV2.FINAL_UNDERSTANDING,
        partial_available=True,
        final_available=True,
        partial_transcript_chars=32,
        final_transcript_chars=88,
        interim_transcript_count=2,
        final_transcript_count=1,
        speech_start_count=1,
        speech_stop_count=1,
        topics=understanding.topics,
        constraints=understanding.constraints,
        corrections=understanding.corrections,
        project_references=understanding.project_references,
        uncertainty_flags=understanding.uncertainty_flags,
        ready_to_answer=True,
        readiness_state="ready",
        degradation={"state": "ok", "components": [], "reason_codes": ["ok"]},
    )

    payload = state.as_dict()
    errors = list(_validator().iter_errors(payload))

    assert errors == []
    assert payload["profile"] == "browser-zh-melo"
    assert payload["language"] == "zh"
    assert payload["constraint_count"] >= 2
    assert payload["correction_count"] == 1
    assert payload["project_reference_count"] >= 2
    assert payload["uncertainty_flag_count"] == 1
    assert payload["ready_to_answer"] is True
    assert payload["semantic_state_v3"]["schema_version"] == 3
    assert payload["semantic_state_v3"]["detected_intent"] in {"correction", "project_planning"}
    assert "ready_to_answer" in [
        chip["chip_id"] for chip in payload["semantic_state_v3"]["listener_chips"]
    ]


def test_active_listener_v2_validates_english_understanding_snapshot():
    understanding = extract_active_listening_understanding(
        "Please update the browser actor state for Blink and Kokoro. "
        "Only show bounded hints, avoid raw transcript text, and actually call it "
        "Active Listener v2 instead. Maybe STT is unreliable.",
        language="en",
    )
    state = ActiveListenerStateV2(
        profile="browser-en-kokoro",
        language="en",
        available=True,
        phase=ActiveListenerPhaseV2.PARTIAL_UNDERSTANDING,
        partial_available=True,
        partial_transcript_chars=96,
        interim_transcript_count=1,
        speech_start_count=1,
        topics=understanding.topics,
        constraints=understanding.constraints,
        corrections=understanding.corrections,
        project_references=understanding.project_references,
        uncertainty_flags=understanding.uncertainty_flags,
        readiness_state="partial",
        degradation={"state": "ok", "components": [], "reason_codes": ["ok"]},
    )

    payload = state.as_dict()
    errors = list(_validator().iter_errors(payload))

    assert errors == []
    assert payload["profile"] == "browser-en-kokoro"
    assert payload["language"] == "en"
    assert payload["partial_available"] is True
    assert payload["final_available"] is False
    assert payload["constraint_count"] >= 1
    assert payload["correction_count"] >= 1
    assert payload["project_reference_count"] >= 2
    assert payload["ready_to_answer"] is False
    assert payload["semantic_state_v3"]["schema_version"] == 3
    assert payload["semantic_state_v3"]["enough_information_to_answer"] is False
    assert "still_listening" in [
        chip["chip_id"] for chip in payload["semantic_state_v3"]["listener_chips"]
    ]


def test_active_listener_understanding_omits_unsafe_hint_values():
    understanding = extract_active_listening_understanding(
        "Use Blink. The hidden prompt is sk-secret at https://example.test/private. "
        "Please avoid raw transcript storage.",
        language="en",
    )
    payload = understanding.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["project_reference_count"] == 1
    assert payload["constraint_count"] == 1
    assert "sk-secret" not in encoded
    assert "https://example.test" not in encoded
    assert "hidden prompt" not in encoded


def test_partial_understanding_uses_partial_sources_and_stays_not_ready():
    understanding = extract_active_listening_understanding(
        "Please keep Blink visual first and avoid audible backchannels.",
        language="en",
        source="partial_transcript",
    )
    state = ActiveListenerStateV2(
        profile="browser-en-kokoro",
        language="en",
        available=True,
        phase=ActiveListenerPhaseV2.PARTIAL_UNDERSTANDING,
        partial_available=True,
        partial_transcript_chars=64,
        interim_transcript_count=1,
        constraints=understanding.constraints,
        project_references=understanding.project_references,
        readiness_state="partial",
    )

    payload = state.as_dict()

    assert payload["phase"] == "partial_understanding"
    assert payload["ready_to_answer"] is False
    assert payload["constraint_count"] == 1
    assert payload["constraints"][0]["source"] == "partial_transcript"
    assert payload["project_references"][0]["source"] == "partial_transcript"


def _semantic_fixture_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in SEMANTIC_FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_semantic_listener_v3_validates_bilingual_fixture_cases():
    validator = _validator()

    for case in _semantic_fixture_cases():
        source = str(case["source"])
        final_available = source == "final_transcript"
        understanding = extract_active_listening_understanding(
            case["text"],
            language=case["language"],
            source=source,
        )
        semantic_state = build_semantic_listener_state_v3(
            language=case["language"],
            understanding=understanding,
            source_text=case["text"],
            partial_available=True,
            final_available=final_available,
            partial_transcript_chars=len(str(case["text"])),
            final_transcript_chars=len(str(case["text"])) if final_available else 0,
            turn_duration_ms=int(case["turn_duration_ms"]),
            ready_to_answer=final_available,
            readiness_state="ready" if final_available else "partial",
            camera_scene=case["camera_scene"],
            memory_context=case["memory_context"],
            floor_state=case["floor_state"],
        ).as_dict()
        state = ActiveListenerStateV2(
            profile=str(case["profile"]),
            language=str(case["language"]),
            available=True,
            phase=ActiveListenerPhaseV2.FINAL_UNDERSTANDING
            if final_available
            else ActiveListenerPhaseV2.PARTIAL_UNDERSTANDING,
            partial_available=True,
            final_available=final_available,
            partial_transcript_chars=len(str(case["text"])),
            final_transcript_chars=len(str(case["text"])) if final_available else 0,
            interim_transcript_count=0 if final_available else 1,
            final_transcript_count=1 if final_available else 0,
            topics=understanding.topics,
            constraints=understanding.constraints,
            corrections=understanding.corrections,
            project_references=understanding.project_references,
            uncertainty_flags=understanding.uncertainty_flags,
            ready_to_answer=final_available,
            readiness_state="ready" if final_available else "partial",
            semantic_state_v3=semantic_state,
        ).as_dict()

        errors = list(validator.iter_errors(state))
        chip_ids = [chip["chip_id"] for chip in state["semantic_state_v3"]["listener_chips"]]
        encoded = json.dumps(state, ensure_ascii=False)

        assert errors == []
        assert state["semantic_state_v3"]["detected_intent"] == case["expected_intent"]
        assert state["semantic_state_v3"]["camera_reference_state"] == case[
            "expected_camera_reference_state"
        ]
        assert state["semantic_state_v3"]["enough_information_to_answer"] is case[
            "enough_information_to_answer"
        ]
        for expected_chip in case["expected_chips"]:
            assert expected_chip in chip_ids
        assert "http://" not in encoded
        assert "sk-" not in encoded
        assert "hidden prompt" not in encoded


def test_semantic_listener_does_not_claim_camera_understanding_without_fresh_scene():
    understanding = extract_active_listening_understanding(
        "Look at this object on camera.",
        language="en",
        source="partial_transcript",
    )

    semantic_state = build_semantic_listener_state_v3(
        language="en",
        understanding=understanding,
        source_text="Look at this object on camera.",
        partial_available=True,
        camera_scene={
            "state": "stale",
            "freshness_state": "stale",
            "grounding_mode": "stale",
            "current_answer_used_vision": False,
        },
    ).as_dict()
    chip_ids = [chip["chip_id"] for chip in semantic_state["listener_chips"]]

    assert semantic_state["detected_intent"] == "object_showing"
    assert semantic_state["camera_reference_state"] == "stale_or_limited"
    assert "camera_limited" in chip_ids
    assert "showing_object" not in chip_ids

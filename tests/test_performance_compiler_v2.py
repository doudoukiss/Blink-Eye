import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.memory_v2 import BrainMemoryUseTraceRef, build_memory_use_trace
from blink.brain.persona import (
    BrainBehaviorControlProfile,
    PerformancePlanV2,
    compile_memory_persona_performance_plan,
    compile_performance_plan_v2,
    default_behavior_control_profile,
)
from blink.brain.session import resolve_brain_session_ids
from blink.interaction.actor_events import ActorEventContext, actor_event_from_performance_event
from blink.interaction.performance_events import BrowserInteractionMode, BrowserPerformanceEvent

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "performance_plan_v2.schema.json"
SCHEMA_V3_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "performance_plan_v3.schema.json"
)


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _validator_v3() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_V3_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _assert_schema_v3_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator_v3().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _session():
    return resolve_brain_session_ids(runtime_kind="browser", client_id="performance-v2")


def _trace():
    session_ids = _session()
    return build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{session_ids.user_id}:claim_browser",
                display_kind="preference",
                title="prefers browser parity",
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from your explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
    )


def _behavior(**overrides: str) -> BrainBehaviorControlProfile:
    session_ids = _session()
    data = default_behavior_control_profile(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
    ).as_dict()
    data.update(overrides)
    profile = BrainBehaviorControlProfile.from_dict(data)
    assert profile is not None
    return profile


def _base_plan(**overrides):
    payload = {
        "profile": "browser-zh-melo",
        "modality": "browser",
        "language": "zh",
        "tts_label": "local-http-wav/MeloTTS",
        "protected_playback": True,
        "camera_state": "available",
        "current_turn_state": "thinking",
        "behavior_profile": _behavior(response_depth="deep"),
        "memory_use_trace": _trace(),
        "suppressed_memory_count": 1,
        "active_listening": {
            "topics": [{"value": "浏览器状态"}],
            "constraints": [{"value": "必须保持双路径一致"}],
            "corrections": [{"value": "应该是浏览器路径"}],
            "project_references": [{"value": "Blink"}],
            "uncertainty_flags": [],
            "ready_to_answer": True,
        },
        "camera_scene": {
            "state": "available",
            "current_answer_used_vision": True,
            "grounding_mode": "single_frame",
            "last_used_frame_sequence": 7,
        },
        "floor_state": {"state": "repair"},
        "user_intent": {"intent": "answer"},
    }
    payload.update(overrides)
    return compile_performance_plan_v2(**payload)


def test_performance_plan_v2_schema_and_determinism():
    first = _base_plan()
    second = _base_plan()

    assert first == second
    payload = first.as_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True)) == payload
    _assert_schema_valid(payload)
    assert payload["schema_version"] == 2
    assert payload["profile"] == "browser-zh-melo"
    assert payload["language"] == "zh"
    assert payload["memory_callback_policy"]["state"] == "use_brief_callback"
    assert payload["camera_reference_policy"]["state"] == "fresh_single_frame"
    assert payload["interruption_policy"]["state"] == "protected_repair_without_auto_yield"
    assert payload["stance"] == "repair_with_precise_correction"
    assert payload["response_shape"] == "repair_then_answer"
    assert "performance_plan:v2" in payload["reason_codes"]


def test_plan_uses_public_reference_summaries_without_examples():
    payload = _base_plan().as_dict()
    scenarios = {item["scenario"] for item in payload["persona_references_used"]}

    assert {"correction", "memory_callback", "camera_use", "deep_technical_planning"} <= scenarios
    for reference in payload["persona_references_used"]:
        assert "example_input" not in reference
        assert "example_output" not in reference
        assert "forbidden_moves" not in reference
        assert reference["locale"] == "zh"


def test_interruption_policy_yields_only_when_not_protected():
    protected = _base_plan(
        floor_state={"state": "overlap"},
        protected_playback=True,
        user_intent={"intent": "explicit_interruption"},
        active_listening={},
    ).as_dict()
    armed = _base_plan(
        floor_state={"state": "overlap"},
        protected_playback=False,
        user_intent={"intent": "explicit_interruption"},
        active_listening={},
    ).as_dict()

    assert protected["interruption_policy"]["state"] == "protected_repair_without_auto_yield"
    assert armed["interruption_policy"]["state"] == "yield_on_armed_overlap"


def test_camera_policy_does_not_claim_stale_or_unavailable_vision():
    stale = _base_plan(
        camera_state="stale",
        camera_scene={
            "state": "stale",
            "current_answer_used_vision": False,
            "grounding_mode": "none",
        },
    ).as_dict()
    disabled = _base_plan(
        camera_state="disabled",
        camera_scene={"state": "disabled", "current_answer_used_vision": False},
    ).as_dict()

    assert stale["camera_reference_policy"]["state"] == "stale_limited_context"
    assert stale["camera_reference_policy"]["current_answer_used_vision"] is False
    assert disabled["camera_reference_policy"]["state"] == "disabled"


def test_concise_and_playful_cases_change_shape_without_serious_repair():
    concise = _base_plan(
        behavior_profile=_behavior(response_depth="concise"),
        memory_use_trace=None,
        active_listening={},
        camera_state="disabled",
        camera_scene={"state": "disabled"},
        floor_state={"state": "assistant_has_floor"},
        user_intent={"intent": "quick_answer"},
    ).as_dict()
    playful = _base_plan(
        behavior_profile=_behavior(humor_mode="playful"),
        memory_use_trace=None,
        active_listening={},
        camera_state="disabled",
        camera_scene={"state": "disabled"},
        floor_state={"state": "unknown"},
        current_turn_state="waiting",
        user_intent={"intent": "casual"},
    ).as_dict()

    assert concise["response_shape"] == "answer_first"
    assert concise["speech_chunking_hints"]["chunking"] == "short"
    assert any(
        item["scenario"] == "playful_restraint"
        for item in playful["persona_references_used"]
    )


def test_primary_profiles_share_structure_except_locale_labels():
    common = {
        "current_turn_state": "thinking",
        "active_listening": {"constraints": [{"value": "keep parity"}]},
        "camera_scene": {"state": "available", "current_answer_used_vision": False},
        "floor_state": {"state": "assistant_has_floor"},
    }
    zh = compile_performance_plan_v2(
        profile="browser-zh-melo",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        camera_state="available",
        **common,
    ).as_dict()
    en = compile_performance_plan_v2(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        camera_state="available",
        **common,
    ).as_dict()

    _assert_schema_valid(zh)
    _assert_schema_valid(en)
    assert set(zh) == set(en)
    for key in (
        "memory_callback_policy",
        "camera_reference_policy",
        "interruption_policy",
        "speech_chunking_hints",
        "ui_state_hints",
    ):
        assert set(zh[key]) == set(en[key])
    assert zh["profile"] != en["profile"]
    assert zh["language"] != en["language"]
    assert zh["tts_label"] != en["tts_label"]
    assert zh["style_summary"] != en["style_summary"]


def test_performance_plan_v2_public_safety_filters_raw_inputs():
    plan = compile_performance_plan_v2(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        active_listening={
            "topics": [{"value": "system_prompt secret raw text"}],
            "constraints": [{"value": "https://example.invalid/private"}],
            "project_references": [{"value": "Blink"}],
        },
        memory_use_trace=_trace(),
        camera_scene={
            "state": "available",
            "raw_image": "data:image/png;base64,AAAA",
            "current_answer_used_vision": True,
            "grounding_mode": "single_frame",
        },
        floor_state={"state": "assistant_has_floor", "full_message": "private"},
    )

    payload = plan.as_dict()
    restored = PerformancePlanV2.from_dict(payload)
    assert restored is not None
    assert restored.as_dict() == payload
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    for banned in (
        "system_prompt secret",
        "raw text",
        "https://example.invalid",
        "data:image",
        "full_message",
        "memory_claim:user",
        "example_input",
        "example_output",
    ):
        assert banned not in encoded


def test_existing_v1_plan_nests_v2_without_changing_v1_contract():
    plan = compile_memory_persona_performance_plan(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        memory_use_trace=_trace(),
        active_listening={"constraints": [{"value": "keep browser parity"}]},
        camera_scene={"state": "available", "current_answer_used_vision": False},
        floor_state={"state": "assistant_has_floor"},
    )
    payload = plan.as_dict()

    assert payload["schema_version"] == 1
    assert payload["selected_memory_count"] == 1
    assert payload["performance_plan_v2"]["schema_version"] == 2
    assert payload["performance_plan_v3"]["schema_version"] == 3
    _assert_schema_valid(payload["performance_plan_v2"])
    _assert_schema_v3_valid(payload["performance_plan_v3"])


def test_memory_persona_committed_event_maps_to_persona_plan_actor_event():
    performance_event = BrowserPerformanceEvent(
        event_id=22,
        event_type="memory_persona.performance_plan_committed",
        source="memory_persona",
        mode=BrowserInteractionMode.THINKING,
        metadata={
            "performance_plan_schema_version": 2,
            "persona_reference_count_v2": 3,
            "style_summary_chars": 48,
            "stance": "answer_first",
            "example_output": "must be omitted",
        },
        reason_codes=["memory_persona:performance_plan_committed"],
    )
    actor_event = actor_event_from_performance_event(
        performance_event,
        context=ActorEventContext(
            profile="browser-en-kokoro",
            language="en",
            tts_backend="kokoro",
            tts_label="kokoro/English",
            vision_backend="moondream",
        ),
    ).as_dict()

    assert actor_event["event_type"] == "persona_plan_compiled"
    assert actor_event["mode"] == "thinking"
    assert actor_event["metadata"]["performance_plan_schema_version"] == 2
    assert "example_output" not in actor_event["metadata"]

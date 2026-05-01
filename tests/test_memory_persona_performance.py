import json

from blink.brain.memory_v2 import BrainMemoryUseTraceRef, build_memory_use_trace
from blink.brain.persona import (
    compile_memory_persona_performance_plan,
    default_behavior_control_profile,
)
from blink.brain.session import resolve_brain_session_ids


def _session():
    return resolve_brain_session_ids(runtime_kind="browser", client_id="performance")


def _trace(session_ids):
    return build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{session_ids.user_id}:claim_coffee",
                display_kind="preference",
                title="coffee preference",
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from your explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
    )


def test_performance_plan_is_deterministic_and_json_safe():
    session_ids = _session()
    profile = default_behavior_control_profile(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
    )
    first = compile_memory_persona_performance_plan(
        profile="browser-zh-melo",
        modality="browser",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        protected_playback=True,
        camera_state="available",
        continuous_perception_enabled=False,
        current_turn_state="thinking",
        behavior_profile=profile,
        memory_use_trace=_trace(session_ids),
        suppressed_memory_count=2,
    )
    second = compile_memory_persona_performance_plan(
        profile="browser-zh-melo",
        modality="browser",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        protected_playback=True,
        camera_state="available",
        continuous_perception_enabled=False,
        current_turn_state="thinking",
        behavior_profile=profile,
        memory_use_trace=_trace(session_ids),
        suppressed_memory_count=2,
    )

    assert first == second
    payload = first.as_dict()
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert payload["profile"] == "browser-zh-melo"
    assert payload["selected_memory_count"] == 1
    assert payload["suppressed_memory_count"] == 2
    assert payload["used_in_current_reply"][0]["display_kind"] == "preference"
    assert "memory_callback_active" in payload["behavior_effects"]
    assert "memory_persona_performance:v1" in payload["reason_codes"]


def test_memory_selected_and_memory_blind_plans_differ():
    session_ids = _session()
    profile = default_behavior_control_profile(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
    )
    selected = compile_memory_persona_performance_plan(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        behavior_profile=profile,
        memory_use_trace=_trace(session_ids),
    )
    blind = compile_memory_persona_performance_plan(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        behavior_profile=profile,
        memory_use_trace=None,
    )

    assert selected.selected_memory_count == 1
    assert blind.selected_memory_count == 0
    assert "memory_callback_active" in selected.behavior_effects
    assert "memory_blind_reply" in blind.behavior_effects
    assert selected.as_dict()["summary"] != blind.as_dict()["summary"]


def test_required_persona_reference_modes_are_explicit_and_bounded():
    plan = compile_memory_persona_performance_plan(
        profile="browser-zh-melo",
        camera_state="looking",
        current_turn_state="thinking",
    )
    payload = plan.as_dict()
    modes = {record["mode"] for record in payload["persona_references"]}

    assert modes == {
        "interruption_response",
        "camera_use",
        "memory_callback",
        "disagreement",
        "correction",
        "concise_answer",
        "deep_technical_planning",
        "uncertainty",
    }
    assert any(
        record["mode"] == "camera_use" and record["applies"]
        for record in payload["persona_references"]
    )
    assert any(
        record["mode"] == "deep_technical_planning" and record["applies"]
        for record in payload["persona_references"]
    )
    anchor_keys = {
        record["situation_key"]
        for record in payload["performance_plan_v3"]["persona_anchor_refs_v3"]
    }
    assert {"visual_grounding", "deep_technical_planning"} <= anchor_keys


def test_suppressed_or_unsafe_memory_content_is_not_exposed():
    session_ids = _session()
    trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{session_ids.user_id}:claim_safe",
                display_kind="profile",
                title="system_prompt secret raw transcript",
                section_key="user_profile",
                used_reason="selected_for_user_profile",
                safe_provenance_label="Remembered from your profile memory.",
            ),
        ),
    )
    plan = compile_memory_persona_performance_plan(
        profile="browser-en-kokoro",
        memory_use_trace=trace,
        suppressed_memory_count=3,
    )

    payload = plan.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert payload["selected_memory_count"] == 1
    assert payload["suppressed_memory_count"] == 3
    assert payload["used_in_current_reply"][0]["title"] == "redacted"
    for banned in (
        "system_prompt secret",
        "raw transcript",
        "fake_human",
        "human_identity",
        "romance",
        "exclusive",
    ):
        assert banned not in encoded

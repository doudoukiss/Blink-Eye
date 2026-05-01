import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.memory_v2 import (
    BrainMemoryUseTraceRef,
    DiscourseEpisode,
    build_memory_continuity_trace,
    build_memory_use_trace,
)
from blink.brain.persona import (
    compile_memory_persona_performance_plan,
    compile_performance_plan_v3,
    compile_performance_plan_v3_from_actor_control,
)
from blink.brain.session import resolve_brain_session_ids

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "performance_plan_v3.schema.json"


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _trace():
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="performance-v3")
    return build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{session_ids.user_id}:browser_parity",
                display_kind="preference",
                title="prefers bilingual browser parity",
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
    )


def _actor_control_frame(
    *,
    profile: str,
    language: str,
    tts_label: str,
    boundary: str = "stt_final_boundary",
    fresh_camera: bool = False,
    repair: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "frame_id": f"control:{profile}:{boundary}",
        "sequence": 7,
        "profile": profile,
        "language": language,
        "tts_backend": "kokoro" if language == "en" else "local-http-wav",
        "tts_runtime_label": tts_label,
        "boundary": boundary,
        "created_at_ms": 1234,
        "source_event_ids": [1, 2, 3],
        "persistent_state": {},
        "condition_cache_digest": "abc123",
        "browser_ui_policy": {"state": "ready", "reason_codes": ["browser_ui:ready"]},
        "speech_policy": {"state": "ready", "action": "emit", "reason_codes": ["speech:emit"]},
        "active_listener_policy": {
            "state": "final",
            "ready_to_answer": True,
            "detected_intent": "object_showing" if fresh_camera else "project_planning",
            "listener_chip_ids": ["ready_to_answer"],
            "reason_codes": ["active_listener:ready"],
        },
        "floor_policy": {
            "state": "repair" if repair else "handoff",
            "sub_state": "accepted_interrupt" if repair else "handoff_pending",
            "yield_decision": "yield" if repair else "handoff",
            "reason_codes": ["floor:repair" if repair else "floor:handoff"],
        },
        "camera_policy": {
            "state": "fresh_used" if fresh_camera else "stale_or_limited",
            "freshness_state": "fresh" if fresh_camera else "stale_or_limited",
            "fresh_frame_used": fresh_camera,
            "scene_transition": "vision_answered" if fresh_camera else "vision_stale",
            "camera_honesty_state": "can_see_now" if fresh_camera else "unavailable",
            "object_showing_likelihood": 0.85 if fresh_camera else 0.2,
            "reason_codes": ["camera_policy:fresh_used" if fresh_camera else "camera_policy:stale"],
        },
        "memory_policy": {"used": True, "effect_count": 1, "reason_codes": ["memory:used"]},
        "persona_policy": {
            "plan_available": True,
            "effect_count": 2,
            "reason_codes": ["persona:available"],
        },
        "repair_policy": {
            "interruption_outcome": "accepted" if repair else "none",
            "stale_output_action": "dropped_or_suppressed" if repair else "none",
            "reason_codes": ["repair:accepted" if repair else "repair:none"],
        },
        "lookahead_counters": {
            "speech_chunks_limit": 2,
            "subtitles_limit": 2,
            "held_speech_chunks": 0,
        },
        "reason_trace": ["actor_control:v3"],
    }


def test_performance_plan_v3_schema_determinism_and_tts_noops():
    actor_frame = _actor_control_frame(
        profile="browser-zh-melo",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        fresh_camera=True,
    )
    first = compile_performance_plan_v3(
        profile="browser-zh-melo",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        actor_control_frame=actor_frame,
        memory_use_trace=_trace(),
        camera_scene={
            "state": "available",
            "current_answer_used_vision": True,
            "scene_social_state_v2": {
                "camera_honesty_state": "can_see_now",
                "scene_transition": "vision_answered",
                "object_showing_likelihood": 0.85,
            },
        },
        floor_state={"state": "handoff"},
    ).as_dict()
    second = compile_performance_plan_v3(
        profile="browser-zh-melo",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        actor_control_frame=actor_frame,
        memory_use_trace=_trace(),
        camera_scene={
            "state": "available",
            "current_answer_used_vision": True,
            "scene_social_state_v2": {
                "camera_honesty_state": "can_see_now",
                "scene_transition": "vision_answered",
                "object_showing_likelihood": 0.85,
            },
        },
        floor_state={"state": "handoff"},
    ).as_dict()

    assert first == second
    _assert_schema_valid(first)
    assert first["schema_version"] == 3
    assert first["profile"] == "browser-zh-melo"
    assert first["camera_reference_policy"]["state"] == "fresh_visual_grounding"
    assert first["response_shape"] == "visual_grounding"
    assert "persona-anchor-v3:visual_grounding" in {
        anchor["anchor_id"] for anchor in first["persona_anchor_refs_v3"]
    }
    assert "persona_anchor:visual_grounding" in first["reason_trace"]
    assert first["tts_capabilities"]["chunk_boundaries_enabled"] is True
    assert first["tts_capabilities"]["interruption_flush_enabled"] is True
    assert first["tts_capabilities"]["speech_rate_enabled"] is False
    assert first["tts_capabilities"]["prosody_emphasis_enabled"] is False
    assert first["tts_capabilities"]["expression_controls_hardware"] is False
    assert "speech_rate" in first["tts_capabilities"]["unsupported_controls"]
    assert "prosody_emphasis" in first["tts_capabilities"]["unsupported_controls"]


def test_primary_profiles_share_plan_structure_except_labels_and_copy():
    common = {
        "actor_control_frame": _actor_control_frame(
            profile="browser-zh-melo",
            language="zh",
            tts_label="local-http-wav/MeloTTS",
        ),
        "memory_use_trace": _trace(),
        "active_listening": {
            "semantic_state_v3": {
                "detected_intent": "project_planning",
                "enough_information_to_answer": True,
                "listener_chips": [{"chip_id": "ready_to_answer"}],
            }
        },
        "camera_scene": {"state": "stale", "current_answer_used_vision": False},
        "floor_state": {"state": "handoff"},
    }
    zh = compile_performance_plan_v3(
        profile="browser-zh-melo",
        language="zh",
        tts_label="local-http-wav/MeloTTS",
        **common,
    ).as_dict()
    en_actor = _actor_control_frame(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
    )
    en = compile_performance_plan_v3(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        **{**common, "actor_control_frame": en_actor},
    ).as_dict()

    _assert_schema_valid(zh)
    _assert_schema_valid(en)
    assert set(zh) == set(en)
    for key in (
        "voice_pacing",
        "speech_chunk_budget",
        "subtitle_policy",
        "camera_reference_policy",
        "memory_callback_policy",
        "interruption_policy",
        "repair_policy",
        "tts_capabilities",
    ):
        assert set(zh[key]) == set(en[key])
    assert zh["profile"] != en["profile"]
    assert zh["language"] != en["language"]
    assert zh["tts_runtime_label"] != en["tts_runtime_label"]
    assert zh["plan_summary"] != en["plan_summary"]
    assert {
        anchor["situation_key"] for anchor in zh["persona_anchor_refs_v3"]
    } == {
        anchor["situation_key"] for anchor in en["persona_anchor_refs_v3"]
    }


def test_actor_control_repair_shortens_budget_and_stale_camera_forces_no_visual_claim():
    repair = compile_performance_plan_v3_from_actor_control(
        _actor_control_frame(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            repair=True,
        ),
        protected_playback=False,
        floor_state={"state": "repair"},
    ).as_dict()
    stale = compile_performance_plan_v3_from_actor_control(
        _actor_control_frame(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            fresh_camera=False,
        ),
        protected_playback=True,
        floor_state={"state": "handoff"},
    ).as_dict()

    _assert_schema_valid(repair)
    _assert_schema_valid(stale)
    assert repair["stance"] == "repairing"
    assert repair["repair_policy"]["state"] == "repair_first"
    assert repair["speech_chunk_budget"]["max_chunks_per_flush"] == 1
    assert repair["speech_chunk_budget"]["hard_max_chars"] <= 120
    assert stale["camera_reference_policy"]["state"] == "no_visual_claim"
    assert stale["camera_reference_policy"]["no_visual_claim"] is True


def test_memory_persona_plan_nests_v3_without_replacing_v2():
    plan = compile_memory_persona_performance_plan(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        memory_use_trace=_trace(),
        actor_control_frame=_actor_control_frame(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
        ),
    ).as_dict()

    assert plan["schema_version"] == 1
    assert plan["performance_plan_v2"]["schema_version"] == 2
    assert plan["performance_plan_v3"]["schema_version"] == 3
    _assert_schema_valid(plan["performance_plan_v3"])
    assert plan["performance_plan_v3"]["persona_reference_ids"]
    assert plan["performance_plan_v3"]["persona_anchor_refs_v3"]


def test_performance_plan_v3_public_safety_filters_raw_inputs():
    plan = compile_performance_plan_v3(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        active_listening={
            "semantic_state_v3": {
                "safe_live_summary": "system_prompt secret raw transcript https://example.invalid",
                "detected_intent": "question",
                "listener_chips": [{"chip_id": "ready_to_answer"}],
            }
        },
        camera_scene={
            "state": "available",
            "raw_image": "data:image/png;base64,AAAA",
            "current_answer_used_vision": False,
        },
        floor_state={"state": "assistant_has_floor", "full_message": "private"},
    ).as_dict()
    encoded = json.dumps(plan, ensure_ascii=False, sort_keys=True).lower()

    _assert_schema_valid(plan)
    for banned in (
        "system_prompt secret",
        "raw transcript",
        "https://example.invalid",
        "data:image",
        "full_message",
        "fake_human",
        "human biography",
        "emotional_prosody",
    ):
        assert banned not in encoded


def test_performance_plan_v3_clamps_primary_tts_unsupported_capability_claims():
    for profile, language, tts_label in (
        ("browser-zh-melo", "zh", "local-http-wav/MeloTTS"),
        ("browser-en-kokoro", "en", "kokoro/English"),
    ):
        plan = compile_performance_plan_v3(
            profile=profile,
            language=language,
            tts_label=tts_label,
            voice_capabilities={
                "backend_label": "kokoro" if language == "en" else "local-http-wav",
                "supports_speech_rate": True,
                "supports_prosody_emphasis": True,
                "supports_pause_timing": True,
                "supports_partial_stream_abort": True,
                "supports_interruption_discard": True,
            },
            voice_actuation_plan={
                "chunk_boundaries_enabled": True,
                "interruption_flush_enabled": True,
                "speech_rate_enabled": True,
                "prosody_emphasis_enabled": True,
                "pause_timing_enabled": True,
                "partial_stream_abort_enabled": True,
                "interruption_discard_enabled": True,
            },
        ).as_dict()

        capabilities = plan["tts_capabilities"]
        assert capabilities["chunk_boundaries_enabled"] is True
        assert capabilities["interruption_flush_enabled"] is True
        assert capabilities["speech_rate_enabled"] is False
        assert capabilities["prosody_emphasis_enabled"] is False
        assert capabilities["pause_timing_enabled"] is False
        assert capabilities["partial_stream_abort_enabled"] is False
        assert capabilities["interruption_discard_enabled"] is False
        for unsupported in (
            "speech_rate",
            "prosody_emphasis",
            "pause_timing",
            "partial_stream_abort",
            "interruption_discard",
            "hardware_control",
        ):
            assert unsupported in capabilities["unsupported_controls"]


def test_performance_plan_v3_selects_all_required_persona_anchor_situations():
    cases = {
        "interruption_response": compile_performance_plan_v3_from_actor_control(
            _actor_control_frame(
                profile="browser-en-kokoro",
                language="en",
                tts_label="kokoro/English",
                repair=True,
            ),
            protected_playback=False,
            floor_state={"state": "repair"},
        ),
        "correction_response": compile_performance_plan_v3(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            user_intent="correction",
            floor_state={"state": "repair"},
        ),
        "deep_technical_planning": compile_performance_plan_v3(
            profile="browser-zh-melo",
            language="zh",
            tts_label="local-http-wav/MeloTTS",
            active_listening={
                "semantic_state_v3": {
                    "detected_intent": "project_planning",
                    "enough_information_to_answer": True,
                }
            },
        ),
        "casual_check_in": compile_performance_plan_v3(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            user_intent="casual",
        ),
        "visual_grounding": compile_performance_plan_v3(
            profile="browser-zh-melo",
            language="zh",
            tts_label="local-http-wav/MeloTTS",
            actor_control_frame=_actor_control_frame(
                profile="browser-zh-melo",
                language="zh",
                tts_label="local-http-wav/MeloTTS",
                fresh_camera=True,
            ),
            camera_scene={
                "state": "available",
                "current_answer_used_vision": True,
                "scene_social_state_v2": {
                    "camera_honesty_state": "can_see_now",
                    "scene_transition": "vision_answered",
                },
            },
        ),
        "uncertainty": compile_performance_plan_v3(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            camera_scene={"state": "stale", "current_answer_used_vision": False},
        ),
        "disagreement": compile_performance_plan_v3(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            user_intent="disagreement",
        ),
        "memory_callback": compile_performance_plan_v3(
            profile="browser-zh-melo",
            language="zh",
            tts_label="local-http-wav/MeloTTS",
            memory_use_trace=_trace(),
        ),
        "playful_not_fake_human": compile_performance_plan_v3(
            profile="browser-en-kokoro",
            language="en",
            tts_label="kokoro/English",
            user_intent="casual",
            behavior_profile={"humor_mode": "playful"},
        ),
    }

    for expected, plan in cases.items():
        payload = plan.as_dict()
        _assert_schema_valid(payload)
        selected = {anchor["situation_key"] for anchor in payload["persona_anchor_refs_v3"]}
        assert expected in selected
        assert f"persona_anchor:{expected}" in payload["reason_trace"]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
        assert "zh_example" not in encoded
        assert "en_example" not in encoded
        assert "negative_examples" not in encoded


def test_performance_plan_v3_memory_continuity_effects_change_response_shape():
    trace = _trace()
    discourse = DiscourseEpisode.from_dict(
        {
            "schema_version": 3,
            "discourse_episode_id": "discourse-episode-v3:concise-project",
            "source_performance_episode_id": "episode:concise-project",
            "source_event_ids": [1],
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_runtime_label": "kokoro/English",
            "created_at_ms": 1,
            "category_labels": ["active_project", "user_preference"],
            "public_summary": "Discourse memory cue: active_project.",
            "memory_refs": [
                {
                    "memory_id": trace.refs[0].memory_id,
                    "display_kind": "preference",
                    "summary": "Concise project explanations.",
                    "source_language": "zh",
                    "cross_language": True,
                    "effect_labels": [
                        "shorter_explanation",
                        "project_constraint_recall",
                    ],
                    "confidence_bucket": "high",
                    "reason_codes": ["source:context_selection"],
                }
            ],
            "effect_labels": ["shorter_explanation", "project_constraint_recall"],
            "conflict_labels": [],
            "staleness_labels": [],
            "confidence_bucket": "high",
            "reason_codes": ["discourse_episode:v3"],
        }
    )
    continuity = build_memory_continuity_trace(
        memory_use_trace=trace,
        profile="browser-en-kokoro",
        language="en",
        discourse_episodes=(discourse,),
    )
    plan = compile_performance_plan_v3(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        memory_use_trace=trace,
        memory_continuity_trace=continuity.as_dict(),
    ).as_dict()

    _assert_schema_valid(plan)
    assert plan["stance"] == "grounded_project_continuity"
    assert plan["response_shape"] == "plan_steps"
    assert plan["memory_callback_policy"]["selected_memory_ids"] == [trace.refs[0].memory_id]
    assert plan["memory_callback_policy"]["discourse_episode_ids"] == [
        "discourse-episode-v3:concise-project"
    ]
    assert "shorter_explanation" in plan["memory_callback_policy"]["effect_labels"]


def test_performance_plan_v3_suppresses_stale_or_conflicted_memory():
    trace = _trace()
    continuity = build_memory_continuity_trace(
        memory_use_trace=trace,
        profile="browser-en-kokoro",
        language="en",
        hidden_counts={"stale": 1, "contradicted": 1},
    )
    plan = compile_performance_plan_v3(
        profile="browser-en-kokoro",
        language="en",
        tts_label="kokoro/English",
        memory_use_trace=trace,
        memory_continuity_trace=continuity.as_dict(),
    ).as_dict()

    _assert_schema_valid(plan)
    assert "suppressed_stale_memory" in plan["memory_callback_policy"]["effect_labels"]
    assert "stale" in plan["memory_callback_policy"]["staleness_labels"]
    assert "contradicted" in plan["memory_callback_policy"]["conflict_labels"]

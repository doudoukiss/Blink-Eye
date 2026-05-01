import json
import os
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.interaction.actor_events import ActorEventModeV2, ActorEventTypeV2, ActorEventV2
from blink.interaction.performance_episode_v3 import (
    PERFORMANCE_EPISODE_V3_SEGMENT_TYPES,
    PerformanceEpisodeV3Writer,
    compile_performance_episode_v3,
    compile_performance_episode_v3_from_actor_trace,
    render_performance_episode_v3_jsonl,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "performance_episode_v3.schema.json"


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _event(
    event_id: int,
    event_type: ActorEventTypeV2,
    mode: ActorEventModeV2,
    *,
    profile: str = "browser-zh-melo",
    language: str = "zh",
    tts_backend: str = "local-http-wav",
    tts_label: str = "local-http-wav/MeloTTS",
    timestamp: str = "2026-04-27T00:00:00+00:00",
    source: str = "test",
    metadata: dict[str, object] | None = None,
    reason_codes: list[str] | None = None,
) -> dict[str, object]:
    return ActorEventV2(
        event_id=event_id,
        event_type=event_type,
        mode=mode,
        timestamp=timestamp,
        profile=profile,
        language=language,
        tts_backend=tts_backend,
        tts_label=tts_label,
        vision_backend="moondream",
        source=source,
        session_id="session-one",
        client_id="client-one",
        metadata=metadata or {},
        reason_codes=reason_codes or [f"test:event_{event_id}"],
    ).as_dict()


def _matched_events(profile: str = "browser-zh-melo") -> list[dict[str, object]]:
    if profile == "browser-en-kokoro":
        language = "en"
        tts_backend = "kokoro"
        tts_label = "kokoro/English"
    else:
        language = "zh"
        tts_backend = "local-http-wav"
        tts_label = "local-http-wav/MeloTTS"

    def make(
        event_id: int,
        event_type: ActorEventTypeV2,
        mode: ActorEventModeV2,
        *,
        source: str = "test",
        metadata: dict[str, object] | None = None,
        reason_codes: list[str] | None = None,
    ) -> dict[str, object]:
        return _event(
            event_id,
            event_type,
            mode,
            profile=profile,
            language=language,
            tts_backend=tts_backend,
            tts_label=tts_label,
            timestamp=f"2026-04-27T00:00:{event_id:02d}+00:00",
            source=source,
            metadata=metadata,
            reason_codes=reason_codes,
        )

    return [
        make(1, ActorEventTypeV2.CONNECTED, ActorEventModeV2.CONNECTED),
        make(
            2,
            ActorEventTypeV2.LISTENING_STARTED,
            ActorEventModeV2.LISTENING,
            source="active_listening",
            metadata={"state": "listening", "constraint_count": 1},
            reason_codes=["active_listening:listening_started", "constraint_detected"],
        ),
        make(
            3,
            ActorEventTypeV2.PARTIAL_UNDERSTANDING_UPDATED,
            ActorEventModeV2.LISTENING,
            source="active_listening",
            metadata={
                "semantic_listener": {
                    "schema_version": 3,
                    "detected_intent": "project_planning",
                    "listener_chip_ids": [
                        "heard_summary",
                        "constraint_detected",
                        "still_listening",
                    ],
                    "listener_chip_count": 3,
                    "camera_reference_state": "stale_or_limited",
                    "memory_context_state": "available",
                    "floor_state": "user_holding_floor",
                    "enough_information_to_answer": False,
                    "summary_hash": "0123456789abcdef",
                    "reason_codes": ["semantic_listener:v3"],
                }
            },
            reason_codes=["active_listener:partial_understanding_updated"],
        ),
        make(
            4,
            ActorEventTypeV2.FINAL_UNDERSTANDING_READY,
            ActorEventModeV2.HEARD,
            source="active_listening",
            metadata={
                "semantic_listener": {
                    "schema_version": 3,
                    "detected_intent": "project_planning",
                    "listener_chip_ids": [
                        "heard_summary",
                        "constraint_detected",
                        "ready_to_answer",
                    ],
                    "listener_chip_count": 3,
                    "camera_reference_state": "stale_or_limited",
                    "memory_context_state": "available",
                    "floor_state": "handoff_pending",
                    "enough_information_to_answer": True,
                    "summary_hash": "fedcba9876543210",
                    "reason_codes": ["semantic_listener:v3"],
                }
            },
            reason_codes=["active_listener:final_understanding_ready"],
        ),
        make(
            5,
            ActorEventTypeV2.MEMORY_USED,
            ActorEventModeV2.THINKING,
            source="memory",
            reason_codes=["memory:used"],
        ),
        make(
            6,
            ActorEventTypeV2.PERSONA_PLAN_COMPILED,
            ActorEventModeV2.THINKING,
            source="persona",
            metadata={
                "performance_plan_v3_schema_version": 3,
                "persona_anchor_ids_v3": [
                    "persona-anchor-v3:deep_technical_planning",
                    "persona-anchor-v3:visual_grounding",
                ],
                "persona_anchor_situation_keys_v3": [
                    "deep_technical_planning",
                    "visual_grounding",
                ],
                "selected_memory_ids": ["memory_claim:user:test:project"],
                "discourse_episode_ids": ["discourse-episode-v3:project"],
                "discourse_category_labels": ["active_project", "user_preference"],
                "memory_effect_labels": [
                    "shorter_explanation",
                    "project_constraint_recall",
                ],
                "performance_plan_v3": {
                    "schema_version": 3,
                    "plan_id": f"performance-plan-v3:{profile}",
                    "stance": "concrete_engineering_planning",
                    "response_shape": "plan_steps",
                    "plan_summary": "Use public style anchors and keep runtime surfaces stable.",
                    "speech_chunk_budget": {
                        "state": "balanced",
                        "target_chars": 120,
                        "hard_max_chars": 180,
                    },
                    "camera_reference_policy": {"state": "fresh_visual_grounding"},
                    "memory_callback_policy": {
                        "state": "brief_callback_available",
                        "selected_memory_ids": ["memory_claim:user:test:project"],
                        "selected_memory_refs": [
                            {
                                "memory_id": "memory_claim:user:test:project",
                                "display_kind": "preference",
                                "summary": "Concise project explanations.",
                                "source_language": "zh" if language == "en" else "en",
                                "cross_language": True,
                                "effect_labels": [
                                    "shorter_explanation",
                                    "project_constraint_recall",
                                ],
                                "confidence_bucket": "high",
                                "reason_codes": ["source:context_selection"],
                            }
                        ],
                        "discourse_episode_ids": ["discourse-episode-v3:project"],
                        "discourse_category_labels": ["active_project", "user_preference"],
                        "effect_labels": [
                            "shorter_explanation",
                            "project_constraint_recall",
                        ],
                        "conflict_labels": [],
                        "staleness_labels": [],
                    },
                    "interruption_policy": {"state": "protected"},
                    "repair_policy": {"state": "normal"},
                    "persona_anchor_refs_v3": [
                        {
                            "schema_version": 3,
                            "anchor_id": "persona-anchor-v3:deep_technical_planning",
                            "situation_key": "deep_technical_planning",
                            "stance_label": "concrete_engineering_planning",
                            "response_shape_label": "plan_steps",
                            "behavior_constraint_count": 3,
                            "negative_example_count": 3,
                            "reason_codes": [
                                "persona_reference_bank:v3",
                                "persona_anchor:deep_technical_planning",
                            ],
                        },
                        {
                            "schema_version": 3,
                            "anchor_id": "persona-anchor-v3:visual_grounding",
                            "situation_key": "visual_grounding",
                            "stance_label": "visually_grounded",
                            "response_shape_label": "visual_grounding",
                            "behavior_constraint_count": 3,
                            "negative_example_count": 3,
                            "reason_codes": [
                                "persona_reference_bank:v3",
                                "persona_anchor:visual_grounding",
                            ],
                        },
                    ],
                }
            },
            reason_codes=["persona:plan_compiled"],
        ),
        make(
            7,
            ActorEventTypeV2.LOOKING,
            ActorEventModeV2.LOOKING,
            source="vision",
            metadata={"scene_transition": "looking_requested"},
            reason_codes=["vision.fetch_user_image_requested"],
        ),
        make(
            8,
            ActorEventTypeV2.LOOKING,
            ActorEventModeV2.LOOKING,
            source="vision",
            metadata={
                "on_demand_vision_state": "success",
                "scene_transition": "vision_answered",
                "camera_honesty_state": "can_see_now",
                "user_presence_hint": "present",
                "last_moondream_result_state": "answered",
                "object_showing_likelihood": 0.85,
            },
            reason_codes=["vision.fetch_user_image_success"],
        ),
        make(
            9,
            ActorEventTypeV2.SPEAKING,
            ActorEventModeV2.SPEAKING,
            source="speech",
            metadata={
                "chunk_index": 1,
                "assistant_subtitle_chars": 8,
                "estimated_duration_ms": 540,
                "subtitle_timing": {
                    "emit_policy": "before_or_at_playback_start",
                    "ready_at_ms": 0,
                    "playback_start_aligned": True,
                    "timing_source": "speech_director_v3",
                },
                "stale_generation_token": "speech-safe",
                "backend_capabilities": {
                    "backend_label": tts_backend,
                    "supports_chunk_boundaries": True,
                    "supports_interruption_flush": True,
                    "supports_interruption_discard": False,
                    "supports_pause_timing": False,
                    "supports_speech_rate": False,
                    "supports_prosody_emphasis": False,
                    "supports_partial_stream_abort": False,
                    "expression_controls_hardware": False,
                    "reason_codes": ["voice_backend_registry:local_http_wav_melo"],
                },
            },
            reason_codes=["speech.audio_start", "speech.subtitle_ready"],
        ),
        make(
            10,
            ActorEventTypeV2.INTERRUPTION_CANDIDATE,
            ActorEventModeV2.SPEAKING,
            source="interruption",
            reason_codes=["interruption:candidate"],
        ),
        make(
            11,
            ActorEventTypeV2.FLOOR_TRANSITION,
            ActorEventModeV2.WAITING,
            source="floor",
            metadata={
                "state": "overlap",
                "floor_sub_state": "overlap_candidate",
                "yield_decision": "protected_continue",
                "user_has_floor": True,
            },
            reason_codes=["floor:overlap"],
        ),
        make(
            12,
            ActorEventTypeV2.INTERRUPTION_ACCEPTED,
            ActorEventModeV2.INTERRUPTED,
            source="interruption",
            metadata={
                "floor_state": "repair",
                "floor_sub_state": "accepted_interrupt",
                "yield_decision": "yield_to_user",
            },
            reason_codes=["interruption:accepted"],
        ),
        make(
            13,
            ActorEventTypeV2.OUTPUT_FLUSHED,
            ActorEventModeV2.INTERRUPTED,
            source="interruption",
            reason_codes=["interruption.output_flushed"],
        ),
        make(
            14,
            ActorEventTypeV2.INTERRUPTION_RECOVERED,
            ActorEventModeV2.RECOVERED,
            source="interruption",
            reason_codes=["interruption.listening_resumed"],
        ),
        make(15, ActorEventTypeV2.WAITING, ActorEventModeV2.WAITING),
    ]


def test_performance_episode_v3_compiles_schema_valid_segments_and_metrics():
    episode = compile_performance_episode_v3(_matched_events())
    payload = episode.as_dict()

    _assert_schema_valid(payload)
    assert payload["schema_version"] == 3
    assert payload["profile"] == "browser-zh-melo"
    assert payload["language"] == "zh"
    assert payload["tts_runtime_label"] == "local-http-wav/MeloTTS"
    assert payload["session_id_hash"] != "session-one"
    assert payload["client_id_hash"] != "client-one"
    assert [segment["segment_type"] for segment in payload["segments"]] == [
        "idle_segment",
        "listen_segment",
        "think_segment",
        "look_segment",
        "speak_segment",
        "overlap_segment",
        "repair_segment",
        "idle_segment",
    ]
    assert set(PERFORMANCE_EPISODE_V3_SEGMENT_TYPES) >= {
        segment["segment_type"] for segment in payload["segments"]
    }
    assert payload["segments"][3]["camera"]["state"] == "fresh_used"
    assert payload["segments"][3]["camera"]["scene_transitions"] == [
        "looking_requested",
        "vision_answered",
    ]
    assert payload["segments"][3]["camera"]["camera_honesty_states"] == ["can_see_now"]
    assert payload["segments"][3]["camera"]["last_moondream_result_states"] == ["answered"]
    assert payload["segments"][3]["camera"]["object_showing_likelihood_max"] == 0.85
    assert payload["segments"][4]["speech"]["estimated_duration_ms_total"] == 540
    assert payload["segments"][4]["speech"]["estimated_duration_ms_max"] == 540
    assert payload["segments"][4]["speech"]["subtitle_timing_policies"] == [
        "before_or_at_playback_start"
    ]
    assert payload["segments"][4]["speech"]["stale_generation_token_count"] == 1
    assert (
        "speech_capability:supports_chunk_boundaries:supported"
        in payload["segments"][4]["speech"]["backend_capability_labels"]
    )
    assert (
        "speech_capability:supports_speech_rate:noop"
        in payload["segments"][4]["speech"]["backend_capability_labels"]
    )
    assert payload["segments"][1]["active_listening"]["semantic_intents"] == [
        "project_planning"
    ]
    assert payload["segments"][1]["active_listening"]["listener_chip_ids"] == [
        "heard_summary",
        "constraint_detected",
        "still_listening",
        "ready_to_answer",
    ]
    assert payload["segments"][1]["active_listening"]["summary_hashes"] == [
        "0123456789abcdef",
        "fedcba9876543210",
    ]
    assert payload["segments"][1]["active_listening"]["enough_information_to_answer"] is True
    assert payload["segments"][2]["performance_plan"]["persona_anchor_situation_keys_v3"] == [
        "deep_technical_planning",
        "visual_grounding",
    ]
    assert payload["segments"][2]["performance_plan"]["persona_anchor_ids_v3"] == [
        "persona-anchor-v3:deep_technical_planning",
        "persona-anchor-v3:visual_grounding",
    ]
    assert payload["segments"][2]["performance_plan"]["discourse_episode_ids"] == [
        "discourse-episode-v3:project"
    ]
    assert payload["segments"][2]["performance_plan"]["discourse_category_labels"] == [
        "active_project",
        "user_preference",
    ]
    assert payload["segments"][2]["performance_plan"]["memory_effect_labels"] == [
        "shorter_explanation",
        "project_constraint_recall",
    ]
    assert payload["segments"][2]["performance_plan"]["memory_ids"] == [
        "memory_claim:user:test:project"
    ]
    assert payload["segments"][5]["floor"]["overlap"] is True
    assert payload["segments"][5]["floor"]["sub_states"] == ["overlap_candidate"]
    assert payload["segments"][6]["floor"]["sub_states"] == ["accepted_interrupt"]
    assert payload["segments"][6]["interruption"]["outcome"] == "recovered"
    assert payload["metrics"]["memory_persona_effect"] == "memory_and_persona"
    assert payload["sanitizer"]["passed"] is True
    assert "session-one" not in json.dumps(payload, ensure_ascii=False)
    assert "safe_live_summary" not in json.dumps(payload, ensure_ascii=False)


def test_performance_episode_v3_has_matched_zh_and_en_structure_except_labels():
    zh = compile_performance_episode_v3(_matched_events("browser-zh-melo")).as_dict()
    en = compile_performance_episode_v3(_matched_events("browser-en-kokoro")).as_dict()

    assert zh["profile"] == "browser-zh-melo"
    assert en["profile"] == "browser-en-kokoro"
    assert zh["language"] == "zh"
    assert en["language"] == "en"
    assert zh["tts_runtime_label"] == "local-http-wav/MeloTTS"
    assert en["tts_runtime_label"] == "kokoro/English"
    assert [segment["segment_type"] for segment in zh["segments"]] == [
        segment["segment_type"] for segment in en["segments"]
    ]
    assert [segment["event_type_counts"] for segment in zh["segments"]] == [
        segment["event_type_counts"] for segment in en["segments"]
    ]
    assert set(zh["metrics"]) == set(en["metrics"])
    assert {
        key: value
        for key, value in zh["metrics"].items()
        if key != "sanitizer_passed"
    } == {
        key: value
        for key, value in en["metrics"].items()
        if key != "sanitizer_passed"
    }
    assert zh["failure_labels"] == en["failure_labels"]
    assert zh["sanitizer"] == en["sanitizer"]


def test_performance_episode_v3_marks_stale_and_error_camera_without_false_fresh_use():
    stale = _event(
        1,
        ActorEventTypeV2.DEGRADED,
        ActorEventModeV2.DEGRADED,
        source="camera",
        metadata={"scene_transition": "vision_stale"},
        reason_codes=["camera.frame_stale"],
    )
    error = _event(
        2,
        ActorEventTypeV2.ERROR,
        ActorEventModeV2.ERROR,
        timestamp="2026-04-27T00:00:01+00:00",
        source="vision",
        metadata={"scene_transition": "vision_unavailable"},
        reason_codes=["vision.fetch_user_image_error"],
    )

    stale_episode = compile_performance_episode_v3([stale]).as_dict()
    error_episode = compile_performance_episode_v3([error]).as_dict()

    assert stale_episode["segments"][0]["camera"]["state"] == "stale_or_limited"
    assert stale_episode["segments"][0]["camera"]["used"] is False
    assert stale_episode["segments"][0]["camera"]["scene_transitions"] == ["vision_stale"]
    assert "camera:stale_or_limited" in stale_episode["failure_labels"]
    assert error_episode["segments"][0]["segment_type"] == "look_segment"
    assert error_episode["segments"][0]["camera"]["state"] == "error"
    assert error_episode["segments"][0]["camera"]["used"] is False
    assert error_episode["segments"][0]["camera"]["scene_transitions"] == [
        "vision_unavailable"
    ]
    assert "vision:error" in error_episode["failure_labels"]


def test_performance_episode_v3_sanitizer_fails_closed_without_raw_payload_leakage():
    unsafe = _matched_events()[0]
    unsafe["raw_text"] = "private transcript"
    unsafe["metadata"] = {
        "safe_state": "ready",
        "raw_transcript": "private transcript",
        "safe_value_state": "a=candidate:private",
    }
    episode = compile_performance_episode_v3([unsafe])
    payload = episode.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["sanitizer"]["passed"] is False
    assert "performance_episode:unsafe_key_omitted" in payload["sanitizer"]["reason_codes"]
    assert "performance_episode:unsafe_value_omitted" in payload["sanitizer"]["reason_codes"]
    assert "sanitizer:blocked_payload" in payload["failure_labels"]
    assert "private transcript" not in encoded
    assert "a=candidate" not in encoded


def test_performance_episode_v3_replay_cli_converts_actor_trace_offline(tmp_path):
    trace = tmp_path / "actor-trace.jsonl"
    episode_path = tmp_path / "episode.jsonl"
    trace.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in _matched_events()),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["BLINK_IMPORT_BANNER"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evals/replay-performance-episode-v3.py",
            str(trace),
            "--input-format",
            "actor-trace",
            "--output-episodes",
            str(episode_path),
        ],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert result.stderr == ""
    assert summary["episode_count"] == 1
    assert summary["segment_counts"]["look_segment"] == 1
    assert "overlap_candidate" in summary["floor_sub_states"]
    assert summary["camera_use_states"] == ["not_used", "fresh_used"]
    assert episode_path.exists()
    converted = compile_performance_episode_v3_from_actor_trace(trace).as_dict()
    _assert_schema_valid(converted)


def test_performance_episode_v3_replay_cli_fails_closed_for_unsafe_actor_trace(tmp_path):
    trace = tmp_path / "actor-trace-unsafe.jsonl"
    unsafe = _matched_events()[0]
    unsafe["metadata"] = {"raw_transcript": "private transcript"}
    trace.write_text(json.dumps(unsafe, ensure_ascii=False) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["BLINK_IMPORT_BANNER"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evals/replay-performance-episode-v3.py",
            str(trace),
            "--input-format",
            "actor-trace",
        ],
        cwd=os.getcwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "performance_episode:unsafe_key_omitted" in result.stdout
    assert "private transcript" not in result.stdout
    assert result.stderr == ""


def test_performance_episode_v3_writer_flushes_on_terminal_boundary(tmp_path):
    writer = PerformanceEpisodeV3Writer(
        episode_dir=tmp_path,
        profile="browser-en-kokoro",
        run_id="test",
    )
    events = _matched_events("browser-en-kokoro")
    for event in events[:-1]:
        writer.append(event)
    writer.append(events[-1], terminal_event_type="runtime.task_finished")

    payloads = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert writer.written_count == 1
    assert len(payloads) == 1
    assert payloads[0]["profile"] == "browser-en-kokoro"
    assert payloads[0]["sanitizer"]["passed"] is True
    assert payloads[0]["segments"]

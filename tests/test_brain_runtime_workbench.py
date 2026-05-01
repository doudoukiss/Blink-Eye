import json
from types import SimpleNamespace

from blink.brain.persona import default_behavior_control_profile
from blink.brain.runtime_workbench import (
    BrainOperatorWorkbenchSection,
    build_operator_workbench_snapshot,
)


class _AsDict:
    def __init__(self, payload):
        self.payload = payload

    def as_dict(self):
        return dict(self.payload)


def _encoded(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _assert_no_operator_leaks(payload):
    encoded = _encoded(payload)
    banned = (
        "secret prompt",
        "hidden deliberation",
        "private_working_memory",
        "source_refs",
        "source_event_ids",
        "evt-secret",
        "/tmp/brain.db",
        "artifact_paths",
        "raw_json_block",
        "hardware_control_debug",
        "Traceback",
        "boom secret",
    )
    for value in banned:
        assert value not in encoded


def test_operator_workbench_inactive_snapshot_is_stable_and_complete():
    first = build_operator_workbench_snapshot(None).as_dict()
    second = build_operator_workbench_snapshot(None).as_dict()

    assert first == second
    assert first["schema_version"] == 1
    assert first["available"] is False
    assert set(first) == {
        "schema_version",
        "available",
        "expression",
        "behavior_controls",
        "teaching_knowledge",
        "voice_metrics",
        "memory",
        "practice",
        "adapters",
        "sim_to_real",
        "rollout_status",
        "episode_evidence",
        "performance_learning",
        "reason_codes",
    }
    for key in (
        "expression",
        "behavior_controls",
        "teaching_knowledge",
        "voice_metrics",
        "memory",
        "practice",
        "adapters",
        "sim_to_real",
        "rollout_status",
        "episode_evidence",
        "performance_learning",
    ):
        assert first[key]["available"] is False
        assert isinstance(first[key]["payload"], dict)
        assert first[key]["reason_codes"]
    assert "operator_workbench:unavailable" in first["reason_codes"]


def test_operator_workbench_aggregate_snapshot_is_public_safe(monkeypatch):
    from blink.brain import runtime_workbench

    session_ids = SimpleNamespace(
        user_id="operator-user",
        agent_id="blink-test",
        session_id="session-1",
        thread_id="thread-1",
    )

    class FakeSnapshot:
        def as_dict(self):
            return {
                "schema_version": 1,
                "user_id": session_ids.user_id,
                "agent_id": session_ids.agent_id,
                "generated_at": "2026-04-23T00:00:00Z",
                "records": [
                    {
                        "memory_id": "memory_claim:user:operator-user:claim-safe",
                        "display_kind": "preference",
                        "title": "secret /tmp/brain.db",
                        "summary": "User prefers coffee.",
                        "status": "RuntimeError",
                        "currentness_status": "current",
                        "confidence": 0.9,
                        "pinned": True,
                        "last_used_at": "2026-04-23T00:00:00Z",
                        "last_used_reason": "selected_for_relevant_continuity",
                        "used_in_current_turn": "yes",
                        "safe_provenance_label": "Remembered from your explicit preference.",
                        "user_actions": [
                            "review",
                            "correct",
                            "forget",
                            "export",
                            "secret /tmp/brain.db",
                        ],
                        "reason_codes": ["source:claim", "secret /tmp/brain.db"],
                        "source_refs": ["claim-safe"],
                        "source_event_ids": ["evt-secret"],
                        "raw_json_block": {"db_path": "/tmp/brain.db"},
                    }
                ],
                "hidden_counts": {
                    "suppressed": 0,
                    "historical": 0,
                    "limit": 0,
                    "secret /tmp/brain.db": 2,
                },
                "health_summary": "Memory health ok.",
                "reason_codes": ["memory_palace:v1", "secret /tmp/brain.db"],
            }

    memory_snapshot_calls = []

    def fake_memory_snapshot(**kwargs):
        memory_snapshot_calls.append(kwargs)
        return FakeSnapshot()

    def fake_practice_inspection(_projection, *, recent_limit):
        return {
            "recent_plan_ids": ["plan-secret"],
            "scenario_family_counts": {"debugging": 1},
            "reason_code_counts": {"recovery_pressure": 1},
            "recent_targets": [
                {
                    "plan_id": "plan-secret",
                    "scenario_id": "Traceback /tmp/brain.db",
                    "scenario_family": "debugging",
                    "selected_profile_id": "local",
                    "execution_backend": "simulation",
                    "score": 0.75,
                    "reason_codes": ["recovery_pressure"],
                    "related_skill_ids": ["skill-secret"],
                }
            ],
            "recent_plans": [
                {
                    "plan_id": "plan-secret",
                    "target_count": 1,
                    "reason_code_counts": {"recovery_pressure": 1, "Traceback /tmp/brain.db": 2},
                    "summary": "secret /tmp/brain.db",
                    "artifact_paths": {"json": "/tmp/brain.db"},
                    "updated_at": "2026-04-23T00:00:00Z",
                }
            ],
        }

    def fake_adapter_inspection(**_kwargs):
        return {
            "state_counts": {"shadow": 1},
            "family_counts": {"perception": 1, "secret /tmp/brain.db": 1},
            "current_default_cards": [
                {
                    "adapter_family": "perception",
                    "backend_id": "local",
                    "backend_version": "secret /tmp/brain.db",
                    "promotion_state": "default",
                }
            ],
            "recent_cards": [
                {
                    "adapter_family": "perception",
                    "backend_id": "candidate",
                    "backend_version": "2",
                    "promotion_state": "shadow",
                    "details": {"hardware_control_debug": "secret"},
                }
            ],
            "recent_reports": [{"report_id": "report-secret"}],
            "pending_or_blocked_decisions": [
                {
                    "decision_id": "decision-secret",
                    "adapter_family": "perception",
                    "backend_id": "candidate",
                    "backend_version": "2",
                    "decision_outcome": "hold",
                    "from_state": "experimental",
                    "to_state": "shadow",
                    "blocked_reason_codes": ["needs_more_evidence"],
                    "weak_families": ["low_light", "Traceback /tmp/brain.db"],
                    "smoke_suite_green": True,
                    "benchmark_passed": False,
                    "updated_at": "2026-04-23T00:00:00Z",
                }
            ],
            "rollback_reason_counts": {},
        }

    def fake_sim_to_real(**_kwargs):
        return {
            "readiness_counts": {"shadow_ready": 1, "rollback_required": 0},
            "promotion_state_counts": {"shadow": 1},
            "blocked_reason_counts": {},
            "readiness_reports": [
                {
                    "report_id": "report-secret",
                    "adapter_family": "perception",
                    "backend_id": "candidate",
                    "backend_version": "2",
                    "promotion_state": "shadow",
                    "benchmark_passed": True,
                    "smoke_suite_green": True,
                    "shadow_ready": True,
                    "canary_ready": False,
                    "default_ready": False,
                    "rollback_required": False,
                    "governance_only": True,
                    "weak_families": [],
                    "blocked_reason_codes": [],
                    "parity_summary": "Traceback /tmp/brain.db",
                    "details": {"latest_report_id": "report-secret"},
                }
            ],
        }

    monkeypatch.setattr(runtime_workbench, "build_memory_palace_snapshot", fake_memory_snapshot)
    monkeypatch.setattr(runtime_workbench, "build_practice_inspection", fake_practice_inspection)
    monkeypatch.setattr(
        runtime_workbench,
        "build_adapter_governance_inspection",
        fake_adapter_inspection,
    )
    monkeypatch.setattr(runtime_workbench, "build_sim_to_real_digest", fake_sim_to_real)

    class FakeStore:
        def build_practice_director_projection(self, **_kwargs):
            return object()

        def build_adapter_governance_projection(self, **_kwargs):
            return object()

    class FakeRuntime:
        store = FakeStore()
        presence_scope_key = "browser:presence"

        def session_resolver(self):
            return session_ids

        def current_expression_state(self):
            return _AsDict(
                {
                    "available": True,
                    "persona_profile_id": "blink-default",
                    "identity_label": "Blink, local non-human system",
                    "modality": "browser",
                    "teaching_mode_label": "walkthrough",
                    "memory_persona_section_status": {
                        "persona_expression": "available",
                        "secret /tmp/brain.db": "Traceback /tmp/brain.db",
                    },
                    "voice_style_summary": "secret /tmp/brain.db",
                    "response_chunk_length": "concise",
                    "pause_yield_hint": "brief",
                    "interruption_strategy_label": "yield",
                    "initiative_label": "proactive",
                    "evidence_visibility_label": "rich",
                    "correction_mode_label": "rigorous",
                    "explanation_structure_label": "walkthrough",
                    "humor_mode_label": "witty",
                    "vividness_mode_label": "vivid",
                    "sophistication_mode_label": "sophisticated",
                    "character_presence_label": "character_rich",
                    "story_mode_label": "recurring_motifs",
                    "style_summary": "humor=witty; vividness=vivid",
                    "humor_budget": 0.42,
                    "playfulness": 0.35,
                    "metaphor_density": 0.61,
                    "safety_clamped": False,
                    "expression_controls_hardware": True,
                    "voice_policy": {
                        "available": True,
                        "modality": "browser",
                        "concise_chunking_active": True,
                        "chunking_mode": "concise",
                        "max_spoken_chunk_chars": 132,
                        "unsupported_hints": ["speech_rate", "Traceback /tmp/brain.db"],
                        "expression_controls_hardware": True,
                    },
                    "voice_actuation_plan": {
                        "available": True,
                        "backend_label": "local-http-wav",
                        "modality": "browser",
                        "chunk_boundaries_enabled": True,
                        "interruption_flush_enabled": True,
                        "interruption_discard_enabled": False,
                        "pause_timing_enabled": False,
                        "speech_rate_enabled": False,
                        "prosody_emphasis_enabled": False,
                        "partial_stream_abort_enabled": False,
                        "expression_controls_hardware": True,
                        "requested_hints": ["speech_rate", "pause_timing"],
                        "applied_hints": ["chunk_boundaries", "interruption_flush"],
                        "unsupported_hints": [
                            "speech_rate",
                            "pause_timing",
                            "secret /tmp/brain.db",
                        ],
                        "reason_codes": ["voice_actuation:available"],
                    },
                    "reason_codes": ["runtime_expression_state:available"],
                    "prompt_text": "secret prompt",
                    "private_working_memory": "hidden deliberation",
                }
            )

        def current_behavior_control_profile(self):
            return default_behavior_control_profile(
                user_id=session_ids.user_id,
                agent_id=session_ids.agent_id,
            )

        def current_memory_persona_performance_plan(self, **kwargs):
            return _AsDict(
                {
                    "schema_version": 1,
                    "available": True,
                    "profile": kwargs.get("profile", "operator"),
                    "modality": "browser",
                    "memory_policy": "balanced",
                    "selected_memory_count": 1,
                    "suppressed_memory_count": kwargs.get("suppressed_memory_count", 0),
                    "used_in_current_reply": [
                        {
                            "memory_id": "memory_claim:user:operator-user:claim-safe",
                            "display_kind": "preference",
                            "title": "coffee preference",
                            "used_reason": "selected_for_relevant_continuity",
                            "behavior_effect": "memory callback changed this reply",
                            "reason_codes": ["source:context_selection"],
                        }
                    ],
                    "behavior_effects": ["memory_callback_active"],
                    "persona_references": [
                        {
                            "reference_id": "persona:memory_callback",
                            "mode": "memory_callback",
                            "label": "memory callback",
                            "applies": True,
                            "behavior_effect": "use selected public memories as brief callbacks",
                            "reason_codes": ["persona_reference:memory_callback"],
                        }
                    ],
                    "summary": "1 memories used; 0 suppressed; 1 persona references active.",
                    "reason_codes": ["memory_persona_performance:v1"],
                }
            )

        def current_teaching_knowledge_routing(self):
            return _AsDict(
                {
                    "schema_version": 1,
                    "available": True,
                    "selection_kind": "auto",
                    "task_mode": "reply",
                    "language": "zh",
                    "teaching_mode": "clarify",
                    "summary": "1 teaching knowledge items selected: exemplar=1",
                    "selected_items": [
                        {
                            "item_kind": "exemplar",
                            "item_id": "exemplar:chinese_technical_explanation_bridge",
                            "title": "Traceback /tmp/brain.db",
                            "source_label": "secret /tmp/brain.db",
                            "provenance_kind": "internal-pedagogy",
                            "provenance_version": "2026-04",
                            "rendered_text": "secret prompt",
                        }
                    ],
                    "estimated_tokens": 42,
                    "reason_codes": [
                        "knowledge_routing_decision:v1",
                        "knowledge_route:exemplar:chinese_technical_explanation_bridge",
                    ],
                    "rendered_text": "hidden teaching prompt",
                }
            )

        def recent_teaching_knowledge_routing(self, *, limit=6):
            return (self.current_teaching_knowledge_routing(),)

        def current_voice_metrics(self):
            return _AsDict(
                {
                    "available": True,
                    "response_count": 2,
                    "concise_chunking_activation_count": 1,
                    "chunk_count": 3,
                    "max_chunk_chars": 80,
                    "average_chunk_chars": 42.0,
                    "interruption_frame_count": 1,
                    "buffer_flush_count": 2,
                    "buffer_discard_count": 0,
                    "last_chunking_mode": "concise",
                    "last_max_spoken_chunk_chars": 132,
                    "expression_controls_hardware": True,
                    "reason_codes": ["voice_metrics:available"],
                    "event_id": "evt-secret",
                }
            )

    snapshot = build_operator_workbench_snapshot(FakeRuntime(), memory_limit=5, recent_limit=3)
    payload = snapshot.as_dict()

    assert memory_snapshot_calls[0]["claim_scan_limit"] == 160
    assert payload["available"] is True
    assert payload["expression"]["available"] is True
    assert payload["behavior_controls"]["available"] is True
    assert payload["teaching_knowledge"]["available"] is True
    assert payload["voice_metrics"]["available"] is True
    assert payload["memory"]["available"] is True
    assert payload["practice"]["available"] is True
    assert payload["adapters"]["available"] is True
    assert payload["sim_to_real"]["available"] is True
    assert payload["rollout_status"]["available"] is False
    assert payload["episode_evidence"]["available"] is True
    assert payload["expression"]["payload"]["expression_controls_hardware"] is False
    assert payload["expression"]["payload"]["voice_policy"]["expression_controls_hardware"] is False
    assert payload["expression"]["payload"]["voice_actuation_plan"][
        "expression_controls_hardware"
    ] is False
    assert payload["expression"]["payload"]["voice_style_summary"] == "redacted"
    assert payload["expression"]["payload"]["memory_persona_section_status"][
        "redacted"
    ] == "redacted"
    assert "redacted" in payload["expression"]["payload"]["voice_policy"][
        "unsupported_hints"
    ]
    assert "redacted" in payload["expression"]["payload"]["voice_actuation_plan"][
        "unsupported_hints"
    ]
    assert payload["expression"]["payload"]["voice_actuation_plan"]["requested_hints"] == [
        "speech_rate",
        "pause_timing",
    ]
    assert payload["expression"]["payload"]["initiative_label"] == "proactive"
    assert payload["expression"]["payload"]["evidence_visibility_label"] == "rich"
    assert payload["expression"]["payload"]["correction_mode_label"] == "rigorous"
    assert payload["expression"]["payload"]["explanation_structure_label"] == "walkthrough"
    assert payload["expression"]["payload"]["humor_mode_label"] == "witty"
    assert payload["expression"]["payload"]["vividness_mode_label"] == "vivid"
    assert payload["expression"]["payload"]["character_presence_label"] == "character_rich"
    assert payload["expression"]["payload"]["humor_budget"] == 0.42
    assert payload["expression"]["payload"]["safety_clamped"] is False
    assert payload["behavior_controls"]["payload"]["profile"]["initiative_mode"] == "balanced"
    assert payload["behavior_controls"]["payload"]["profile"]["evidence_visibility"] == "compact"
    assert payload["behavior_controls"]["payload"]["profile"]["correction_mode"] == "precise"
    assert payload["behavior_controls"]["payload"]["profile"]["explanation_structure"] == (
        "answer_first"
    )
    assert payload["behavior_controls"]["payload"]["profile"]["humor_mode"] == "witty"
    assert payload["behavior_controls"]["payload"]["profile"]["character_presence"] == (
        "character_rich"
    )
    teaching = payload["teaching_knowledge"]["payload"]
    assert teaching["current_decision"]["selection_kind"] == "auto"
    assert teaching["current_decision"]["selected_items"][0]["item_id"] == (
        "exemplar:chinese_technical_explanation_bridge"
    )
    assert teaching["current_decision"]["selected_items"][0]["title"] == "redacted"
    assert teaching["current_decision"]["selected_items"][0]["source_label"] == "redacted"
    assert teaching["selected_item_counts"] == {"exemplar": 2}
    assert "rendered_text" not in _encoded(teaching)
    assert "hidden teaching prompt" not in _encoded(teaching)
    assert payload["expression"]["payload"]["voice_actuation_plan"]["applied_hints"] == [
        "chunk_boundaries",
        "interruption_flush",
    ]
    assert payload["voice_metrics"]["payload"]["expression_controls_hardware"] is False
    assert payload["memory"]["payload"]["records"][0]["pinned"] is True
    assert payload["memory"]["payload"]["records"][0]["title"] == "redacted"
    assert payload["memory"]["payload"]["records"][0]["summary"] == "User prefers coffee."
    assert payload["memory"]["payload"]["records"][0]["status"] == "redacted"
    assert payload["memory"]["payload"]["records"][0]["used_in_current_turn"] is False
    assert payload["memory"]["payload"]["records"][0]["user_actions"] == [
        "review",
        "correct",
        "forget",
        "export",
    ]
    assert payload["memory"]["payload"]["records"][0]["reason_codes"] == [
        "source:claim",
        "redacted",
    ]
    assert payload["memory"]["payload"]["reason_codes"] == ["memory_palace:v1", "redacted"]
    assert payload["memory"]["payload"]["hidden_counts"]["redacted"] == 2
    assert payload["memory"]["payload"]["used_in_current_reply"][0]["title"] == (
        "coffee preference"
    )
    assert payload["memory"]["payload"]["behavior_effects"] == ["memory_callback_active"]
    assert payload["memory"]["payload"]["persona_references"][0]["mode"] == "memory_callback"
    assert "source_refs" not in payload["memory"]["payload"]["records"][0]
    assert payload["practice"]["payload"]["recent_targets"][0]["scenario_id"] == "redacted"
    assert payload["practice"]["payload"]["recent_plans"][0]["summary"] == "redacted"
    assert payload["practice"]["payload"]["recent_plans"][0]["reason_code_counts"][
        "redacted"
    ] == 2
    assert "recent_plan_ids" not in payload["practice"]["payload"]
    assert payload["adapters"]["payload"]["family_counts"]["redacted"] == 1
    assert payload["adapters"]["payload"]["current_default_cards"][0]["backend_version"] == (
        "redacted"
    )
    assert "redacted" in payload["adapters"]["payload"]["pending_or_blocked_decisions"][0][
        "weak_families"
    ]
    assert "recent_reports" not in payload["adapters"]["payload"]
    assert "report_id" not in payload["sim_to_real"]["payload"]["readiness_reports"][0]
    assert payload["sim_to_real"]["payload"]["readiness_reports"][0]["parity_summary"] == (
        "redacted"
    )
    assert "operator_workbench:available" in payload["reason_codes"]
    _assert_no_operator_leaks(payload)


def test_operator_workbench_rollout_status_uses_live_controller_status():
    class FakeRuntime:
        def current_live_routing_status(self):
            return _AsDict(
                {
                    "schema_version": 1,
                    "available": True,
                    "generated_at": "2026-04-24T00:00:00+00:00",
                    "summary": "1 rollout plans; 1 active; 0 paused; 0 rolled back.",
                    "plan_count": 1,
                    "active_plan_count": 1,
                    "paused_plan_count": 0,
                    "rolled_back_plan_count": 0,
                    "expired_plan_count": 0,
                    "live_routing_active": True,
                    "controlled_rollout_supported": True,
                    "governance_only": False,
                    "state_counts": {"active_limited": 1},
                    "family_counts": {"world_model": 1},
                    "plan_summaries": [
                        {
                            "plan_id": "plan-public",
                            "adapter_family": "world_model",
                            "candidate_backend_id": "candidate_world_model",
                            "candidate_backend_version": "candidate-v1",
                            "routing_state": "active_limited",
                            "promotion_state": "canary",
                            "traffic_fraction": 0.05,
                            "scope_key": "local",
                            "expires_at": "2026-04-25T00:00:00+00:00",
                            "embodied_live": False,
                            "budget_id": "budget-public",
                            "reason_codes": ["routing_state:active_limited"],
                            "details": {"artifact_path": "/tmp/brain.db"},
                        }
                    ],
                    "recent_decisions": [
                        {
                            "decision_id": "decision-secret",
                            "plan_id": "plan-public",
                            "adapter_family": "world_model",
                            "action": "activate",
                            "accepted": True,
                            "from_state": "approved",
                            "to_state": "active_limited",
                            "traffic_fraction": 0.05,
                            "regression_count": 0,
                            "updated_at": "2026-04-24T00:00:00+00:00",
                            "reason_codes": ["rollout_decision_accepted"],
                            "details": {"raw_json_block": "/tmp/brain.db"},
                        }
                    ],
                    "reason_codes": ["live_routing_controller:available"],
                }
            )

    payload = build_operator_workbench_snapshot(FakeRuntime()).as_dict()

    rollout = payload["rollout_status"]
    assert rollout["available"] is True
    assert rollout["payload"]["available"] is True
    assert rollout["payload"]["live_routing_active"] is True
    assert rollout["payload"]["controlled_rollout_supported"] is True
    assert rollout["payload"]["governance_only"] is False
    assert rollout["payload"]["active_plan_count"] == 1
    assert rollout["payload"]["plan_summaries"][0]["routing_state"] == "active_limited"
    assert rollout["payload"]["recent_decisions"][0]["action"] == "activate"
    assert "operator_rollout_status:available" in rollout["reason_codes"]
    encoded = _encoded(payload)
    assert "decision-secret" not in encoded
    assert "raw_json_block" not in encoded
    assert "/tmp/brain.db" not in encoded


def test_operator_workbench_section_failures_are_isolated():
    class FailingRuntime:
        def current_expression_state(self):
            raise RuntimeError("boom secret /tmp/brain.db")

        def current_voice_metrics(self):
            return _AsDict({"available": True, "chunk_count": 0, "reason_codes": []})

    payload = build_operator_workbench_snapshot(FailingRuntime()).as_dict()

    assert payload["available"] is True
    assert payload["expression"]["available"] is False
    assert payload["voice_metrics"]["available"] is True
    assert "operator_expression_error:RuntimeError" in payload["expression"]["reason_codes"]
    assert "boom secret" not in _encoded(payload)
    assert "/tmp/brain.db" not in _encoded(payload)


def test_operator_workbench_voice_metrics_sanitizes_malformed_public_fields():
    class FakeRuntime:
        def current_voice_metrics(self):
            return _AsDict(
                {
                    "available": True,
                    "response_count": "not-an-int",
                    "average_chunk_chars": "nan",
                    "last_chunking_mode": "secret /tmp/brain.db",
                    "reason_codes": [
                        "voice_metrics:available",
                        "secret /tmp/brain.db",
                        "Traceback",
                    ],
                    "event_id": "evt-secret",
                }
            )

        def current_voice_input_health(self):
            return _AsDict(
                {
                    "schema_version": 1,
                    "available": True,
                    "microphone_state": "secret /tmp/brain.db",
                    "stt_state": "secret /tmp/brain.db",
                    "audio_frame_count": "not-an-int",
                    "last_audio_frame_at": "secret /tmp/brain.db",
                    "last_audio_frame_age_ms": "not-an-int",
                    "last_stt_event_at": "secret /tmp/brain.db",
                    "stt_waiting_since_at": "secret /tmp/brain.db",
                    "stt_wait_age_ms": "not-an-int",
                    "last_transcription_at": "secret /tmp/brain.db",
                    "last_transcription_chars": "not-an-int",
                    "track_reason": "secret /tmp/brain.db",
                    "reason_codes": [
                        "voice_input_health:v1",
                        "secret /tmp/brain.db",
                        "RuntimeError",
                    ],
                    "raw_transcript": "secret user speech",
                }
            )

    payload = build_operator_workbench_snapshot(FakeRuntime()).as_dict()
    voice = payload["voice_metrics"]["payload"]
    encoded = _encoded(payload)

    assert payload["voice_metrics"]["available"] is True
    assert voice["response_count"] == 0
    assert voice["average_chunk_chars"] == 0.0
    assert voice["last_chunking_mode"] == "unavailable"
    assert voice["input_health"]["microphone_state"] == "unavailable"
    assert voice["input_health"]["stt_state"] == "unavailable"
    assert voice["input_health"]["last_audio_frame_at"] is None
    assert voice["input_health"]["last_audio_frame_age_ms"] is None
    assert voice["input_health"]["stt_wait_age_ms"] is None
    assert voice["input_health"]["last_transcription_at"] is None
    assert voice["input_health"]["track_reason"] == "redacted"
    assert "redacted" in voice["reason_codes"]
    assert "redacted" in voice["input_health"]["reason_codes"]
    assert "secret" not in encoded
    assert "Traceback" not in encoded
    assert "RuntimeError" not in encoded
    assert "/tmp/brain.db" not in encoded
    assert "raw_transcript" not in encoded
    assert "evt-secret" not in encoded


def test_operator_workbench_section_dataclass_serialization_is_stable():
    section = BrainOperatorWorkbenchSection(
        available=True,
        summary="ok",
        payload={"count": 1},
        reason_codes=("a", "b"),
    )

    assert section.as_dict() == {
        "available": True,
        "summary": "ok",
        "payload": {"count": 1},
        "reason_codes": ["a", "b"],
    }

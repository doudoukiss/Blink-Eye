import json
from pathlib import Path

from jsonschema import Draft202012Validator

from blink.brain.memory_v2 import (
    DiscourseEpisode,
    DiscourseEpisodeV3Collector,
    compile_discourse_episode_v3,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.interaction.performance_episode_v3 import compile_performance_episode_v3

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "discourse_episode_v3.schema.json"


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _labels(profile: str) -> tuple[str, str, str]:
    if profile == "browser-en-kokoro":
        return "en", "kokoro", "kokoro/English"
    return "zh", "local-http-wav", "local-http-wav/MeloTTS"


def _actor_events(profile: str = "browser-zh-melo") -> list[dict[str, object]]:
    language, tts_backend, tts_label = _labels(profile)
    base = {
        "schema_version": 2,
        "profile": profile,
        "language": language,
        "tts_backend": tts_backend,
        "tts_label": tts_label,
        "vision_backend": "moondream",
        "session_id": "session-discourse",
        "client_id": "client-discourse",
    }
    return [
        {
            **base,
            "event_id": 1,
            "event_type": "partial_understanding_updated",
            "mode": "listening",
            "timestamp": "2026-04-28T00:00:00+00:00",
            "metadata": {
                "semantic_listener": {
                    "detected_intent": "project_planning",
                    "listener_chip_ids": ["constraint_detected"],
                }
            },
            "reason_codes": ["active_listening:project_references_detected"],
        },
        {
            **base,
            "event_id": 2,
            "event_type": "persona_plan_compiled",
            "mode": "thinking",
            "timestamp": "2026-04-28T00:00:01+00:00",
            "metadata": {
                "performance_plan_v3": {
                    "plan_id": "performance-plan-v3:test",
                    "stance": "grounded_project_continuity",
                    "response_shape": "plan_steps",
                    "memory_callback_policy": {
                        "state": "cross_language_callback",
                        "selected_memory_ids": ["memory_claim:user:test:preference_concise"],
                        "selected_memory_refs": [
                            {
                                "memory_id": "memory_claim:user:test:preference_concise",
                                "display_kind": "preference",
                                "summary": "Concise project explanations.",
                                "source_language": "en" if language == "zh" else "zh",
                                "cross_language": True,
                                "effect_labels": [
                                    "shorter_explanation",
                                    "project_constraint_recall",
                                ],
                                "confidence_bucket": "high",
                                "reason_codes": ["source:context_selection"],
                            }
                        ],
                        "effect_labels": [
                            "shorter_explanation",
                            "project_constraint_recall",
                        ],
                        "conflict_labels": [],
                        "staleness_labels": [],
                        "reason_codes": ["memory_policy_v3:cross_language_callback"],
                    },
                    "camera_reference_policy": {"state": "fresh_visual_grounding"},
                },
                "performance_plan_v3_schema_version": 3,
            },
            "reason_codes": ["memory_persona:performance_plan_committed"],
        },
        {
            **base,
            "event_id": 3,
            "event_type": "looking",
            "mode": "looking",
            "timestamp": "2026-04-28T00:00:02+00:00",
            "metadata": {
                "scene_transition": "vision_answered",
                "camera_honesty_state": "can_see_now",
                "current_answer_used_vision": True,
            },
            "reason_codes": ["camera_scene:vision_answered"],
        },
        {
            **base,
            "event_id": 4,
            "event_type": "speech.audio_start",
            "mode": "speaking",
            "timestamp": "2026-04-28T00:00:03+00:00",
            "metadata": {"chunk_index": 1},
            "reason_codes": ["speech:audio_start"],
        },
        {
            **base,
            "event_id": 5,
            "event_type": "runtime.task_finished",
            "mode": "waiting",
            "timestamp": "2026-04-28T00:00:04+00:00",
            "metadata": {},
            "reason_codes": ["runtime:task_finished"],
        },
    ]


def test_discourse_episode_v3_schema_and_primary_profile_parity():
    zh = compile_discourse_episode_v3(
        compile_performance_episode_v3(_actor_events("browser-zh-melo"))
    ).as_dict()
    en = compile_discourse_episode_v3(
        compile_performance_episode_v3(_actor_events("browser-en-kokoro"))
    ).as_dict()

    _assert_schema_valid(zh)
    _assert_schema_valid(en)
    assert set(zh) == set(en)
    assert zh["profile"] == "browser-zh-melo"
    assert en["profile"] == "browser-en-kokoro"
    assert zh["category_labels"] == en["category_labels"]
    assert {"active_project", "user_preference", "visual_event", "success_pattern"} <= set(
        zh["category_labels"]
    )
    assert zh["effect_labels"] == en["effect_labels"]
    assert "shorter_explanation" in zh["effect_labels"]
    assert zh["memory_refs"][0]["cross_language"] is True


def test_discourse_episode_v3_derives_repair_stale_and_frustration_labels():
    payload = compile_performance_episode_v3(
        [
            {
                **_actor_events("browser-en-kokoro")[0],
                "event_type": "floor.transition",
                "mode": "interrupted",
                "metadata": {
                    "floor_sub_state": "repair_requested",
                    "phrase_class": "correction",
                },
                "reason_codes": ["correction", "protected_playback"],
            },
            {
                **_actor_events("browser-en-kokoro")[1],
                "metadata": {
                    "performance_plan_v3": {
                        "memory_callback_policy": {
                            "state": "suppress_stale_callback",
                            "effect_labels": ["suppressed_stale_memory"],
                            "conflict_labels": ["superseded_by_correction"],
                            "staleness_labels": ["stale"],
                            "reason_codes": ["memory_policy_v3:suppress_stale_callback"],
                        }
                    }
                },
                "reason_codes": ["memory_persona:performance_plan_committed"],
            },
        ]
    )
    episode = compile_discourse_episode_v3(payload).as_dict()

    _assert_schema_valid(episode)
    assert "correction" in episode["category_labels"]
    assert "repeated_frustration" in episode["category_labels"]
    assert "corrected_preference" in episode["effect_labels"]
    assert "suppressed_stale_memory" in episode["effect_labels"]
    assert "superseded_by_correction" in episode["conflict_labels"]
    assert "stale" in episode["staleness_labels"]


def test_discourse_episode_v3_public_safety_redacts_unsafe_values():
    episode = DiscourseEpisode.from_dict(
        {
            "schema_version": 3,
            "discourse_episode_id": "discourse-episode-v3:raw_transcript_secret",
            "source_performance_episode_id": "episode:unsafe",
            "source_event_ids": [1],
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_runtime_label": "kokoro/English",
            "created_at_ms": 1,
            "category_labels": ["user_preference"],
            "public_summary": "raw_transcript secret private memory_body",
            "memory_refs": [
                {
                    "memory_id": "memory:raw_prompt_secret",
                    "display_kind": "preference",
                    "summary": "https://example.invalid/token",
                    "source_language": "en",
                    "cross_language": False,
                    "effect_labels": ["shorter_explanation"],
                    "confidence_bucket": "high",
                    "reason_codes": ["raw_image"],
                }
            ],
            "effect_labels": ["shorter_explanation"],
            "conflict_labels": [],
            "staleness_labels": [],
            "confidence_bucket": "high",
            "reason_codes": ["hidden_prompt"],
        }
    ).as_dict()
    encoded = json.dumps(episode, ensure_ascii=False, sort_keys=True).lower()

    _assert_schema_valid(episode)
    for banned in (
        "raw_transcript",
        "secret",
        "memory_body",
        "https://",
        "token",
        "raw_prompt",
        "raw_image",
        "hidden_prompt",
    ):
        assert banned not in encoded


def test_discourse_episode_v3_persists_as_brain_event(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="discourse")
    episode = compile_discourse_episode_v3(
        compile_performance_episode_v3(_actor_events("browser-en-kokoro"))
    )

    persisted = store.append_discourse_episode(
        episode=episode,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    recent = store.recent_discourse_episodes(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )

    assert persisted == episode
    assert recent[0] == episode


def test_discourse_episode_v3_collector_flushes_on_terminal_boundary(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="collector")

    class Runtime:
        def __init__(self):
            self.store = store

        def session_resolver(self):
            return session_ids

    collector = DiscourseEpisodeV3Collector(runtime_resolver=Runtime)
    events = _actor_events("browser-zh-melo")
    for event in events:
        collector.append(event, terminal_event_type=str(event["event_type"]))

    recent = store.recent_discourse_episodes(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )
    assert collector.persisted_count == 1
    assert recent
    assert "active_project" in recent[0].category_labels

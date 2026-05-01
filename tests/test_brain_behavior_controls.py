import json

from blink.brain.events import BrainEventType
from blink.brain.persona import (
    BrainBehaviorControlProfile,
    apply_behavior_control_update,
    behavior_style_preset_catalog,
    build_witty_sophisticated_memory_story_seed,
    default_behavior_control_profile,
    get_behavior_style_preset,
    load_behavior_control_profile,
    render_behavior_control_effect_summary,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def _session(client_id: str = "behavior-controls-user"):
    return resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)


def test_behavior_control_default_profile_is_stable_and_safe():
    profile = default_behavior_control_profile(user_id="user-1", agent_id="blink/main")
    restored = BrainBehaviorControlProfile.from_dict(profile.as_dict())
    encoded = json.dumps(profile.as_dict(), sort_keys=True)
    summary = render_behavior_control_effect_summary(profile)

    assert restored == profile
    assert profile.schema_version == 3
    assert profile.response_depth == "balanced"
    assert profile.teaching_mode == "auto"
    assert profile.initiative_mode == "balanced"
    assert profile.evidence_visibility == "compact"
    assert profile.correction_mode == "precise"
    assert profile.explanation_structure == "answer_first"
    assert profile.humor_mode == "witty"
    assert profile.vividness_mode == "vivid"
    assert profile.sophistication_mode == "sophisticated"
    assert profile.character_presence == "character_rich"
    assert profile.story_mode == "light"
    assert profile.updated_at == ""
    assert "behavior_controls_defaulted" in profile.reason_codes
    assert "depth=balanced" in summary
    assert "initiative=balanced" in summary
    assert "evidence=compact" in summary
    assert "humor=witty" in summary
    assert "character=character_rich" in summary
    assert "{" not in summary
    assert "romantic" not in encoded
    assert "hardware_control" not in encoded


def test_behavior_control_legacy_v1_profile_hydrates_with_v3_defaults():
    restored = BrainBehaviorControlProfile.from_dict(
        {
            "schema_version": 1,
            "user_id": "user-1",
            "agent_id": "blink/main",
            "response_depth": "concise",
            "directness": "rigorous",
            "warmth": "high",
            "teaching_mode": "walkthrough",
            "memory_use": "minimal",
            "challenge_style": "direct",
            "voice_mode": "off",
            "question_budget": "low",
            "updated_at": "2026-04-24T00:00:00+00:00",
            "source": "legacy",
            "reason_codes": ["legacy"],
        }
    )

    assert restored is not None
    assert restored.schema_version == 3
    assert restored.response_depth == "concise"
    assert restored.initiative_mode == "balanced"
    assert restored.evidence_visibility == "compact"
    assert restored.correction_mode == "precise"
    assert restored.explanation_structure == "answer_first"
    assert restored.humor_mode == "witty"
    assert restored.vividness_mode == "vivid"
    assert restored.sophistication_mode == "sophisticated"
    assert restored.character_presence == "character_rich"
    assert restored.story_mode == "light"


def test_behavior_control_update_rejects_unknown_unsafe_and_invalid_fields(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("behavior-reject")

    unknown = apply_behavior_control_update(
        store=store,
        session_ids=session_ids,
        updates={"temperature": "hot"},
        source="test",
    )
    unsafe = apply_behavior_control_update(
        store=store,
        session_ids=session_ids,
        updates={"romance": "enabled"},
        source="test",
    )
    invalid_value = apply_behavior_control_update(
        store=store,
        session_ids=session_ids,
        updates={"response_depth": "romantic"},
        source="test",
    )
    prompt_mutation = apply_behavior_control_update(
        store=store,
        session_ids=session_ids,
        updates={"system_prompt": "be a real human", "api_key": "secret"},
        source="test",
    )

    assert unknown.accepted is False
    assert "temperature" in unknown.rejected_fields
    assert unsafe.accepted is False
    assert "romance" in unsafe.rejected_fields
    assert invalid_value.accepted is False
    assert "response_depth" in invalid_value.rejected_fields
    assert prompt_mutation.accepted is False
    assert set(prompt_mutation.rejected_fields) == {"api_key", "system_prompt"}
    assert "behavior_controls_fields_invalid" in invalid_value.reason_codes


def test_behavior_control_update_merges_and_persists_relationship_block(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("behavior-merge")

    result = apply_behavior_control_update(
        store=store,
        session_ids=session_ids,
        updates={
            "response_depth": "deep",
            "teaching_mode": "socratic",
            "question_budget": "high",
            "initiative_mode": "proactive",
            "evidence_visibility": "rich",
            "correction_mode": "rigorous",
            "explanation_structure": "walkthrough",
            "humor_mode": "playful",
            "vividness_mode": "vivid",
            "sophistication_mode": "sophisticated",
            "character_presence": "character_rich",
            "story_mode": "recurring_motifs",
        },
        source="test",
    )
    loaded = load_behavior_control_profile(store=store, session_ids=session_ids)
    summary = render_behavior_control_effect_summary(loaded)

    assert result.accepted is True
    assert result.applied is True
    assert loaded.response_depth == "deep"
    assert loaded.directness == "balanced"
    assert loaded.teaching_mode == "socratic"
    assert loaded.question_budget == "high"
    assert loaded.initiative_mode == "proactive"
    assert loaded.evidence_visibility == "rich"
    assert loaded.correction_mode == "rigorous"
    assert loaded.explanation_structure == "walkthrough"
    assert loaded.humor_mode == "playful"
    assert loaded.story_mode == "recurring_motifs"
    assert loaded.source == "test"
    assert "behavior_controls_loaded" in loaded.reason_codes
    assert "teaching=socratic" in summary
    assert "initiative=proactive" in summary
    assert "story=recurring_motifs" in summary
    assert "{" not in summary


def test_behavior_control_read_defaults_when_block_missing_or_invalid(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("behavior-default")

    missing = load_behavior_control_profile(store=store, session_ids=session_ids)
    store.upsert_core_memory_block(
        block_kind="behavior_control_profile",
        scope_type="relationship",
        scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
        content={"schema_version": 1, "user_id": session_ids.user_id, "response_depth": "bad"},
    )
    invalid = load_behavior_control_profile(store=store, session_ids=session_ids)

    assert missing.response_depth == "balanced"
    assert "behavior_controls_defaulted" in missing.reason_codes
    assert invalid.response_depth == "balanced"
    assert "behavior_controls_invalid" in invalid.reason_codes


def test_behavior_control_updates_replay_from_memory_block_events(tmp_path):
    source = BrainStore(path=tmp_path / "source.db")
    replayed = BrainStore(path=tmp_path / "replayed.db")
    session_ids = _session("behavior-replay")

    result = apply_behavior_control_update(
        store=source,
        session_ids=session_ids,
        updates={"memory_use": "minimal", "voice_mode": "off"},
        source="test",
    )
    events = source.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=32,
    )
    for event in events:
        replayed.import_brain_event(event)
        replayed.apply_memory_event(event)
    loaded = load_behavior_control_profile(store=replayed, session_ids=session_ids)

    assert result.accepted is True
    assert any(event.event_type == BrainEventType.MEMORY_BLOCK_UPSERTED for event in events)
    assert loaded.memory_use == "minimal"
    assert loaded.voice_mode == "off"


def test_witty_sophisticated_style_preset_catalog_is_public_safe():
    catalog = behavior_style_preset_catalog()
    encoded = json.dumps(catalog, sort_keys=True)
    preset = next(item for item in catalog["presets"] if item["preset_id"] == "witty_sophisticated")

    assert catalog["available"] is True
    assert catalog["default_preset_id"] == "witty_sophisticated"
    assert get_behavior_style_preset("witty-sophisticated") is not None
    assert get_behavior_style_preset("Witty Sophisticated") is not None
    assert preset["recommended"] is True
    assert preset["control_updates"]["humor_mode"] == "witty"
    assert preset["control_updates"]["vividness_mode"] == "vivid"
    assert preset["control_updates"]["character_presence"] == "character_rich"
    assert preset["control_updates"]["story_mode"] == "recurring_motifs"
    assert preset["language_fit"]["zh"] == "excellent"
    for forbidden in (
        "system_prompt",
        "api_key",
        "authorization",
        "hardware_control",
        "human_identity",
        "romance",
        "Traceback",
    ):
        assert forbidden not in encoded


def test_witty_sophisticated_memory_story_seed_uses_governed_surfaces():
    seed = build_witty_sophisticated_memory_story_seed(user_name="ray", agent_id="blink/main")
    encoded = json.dumps(seed, sort_keys=True)

    assert seed["schema_version"] == 1
    assert seed["behavior_controls"]["humor_mode"] == "witty"
    assert seed["relationship_style"]["interaction"]["style"]
    assert seed["teaching_profile"]["teaching"]["preference"]["mode"] == "walkthrough"
    for forbidden in (
        "system_prompt",
        "developer_prompt",
        "api_key",
        "hardware_control",
        "childhood",
        "romance",
    ):
        assert forbidden not in encoded

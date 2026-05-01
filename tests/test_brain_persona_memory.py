from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory_v2 import BrainCoreMemoryBlockKind
from blink.brain.persona import (
    RelationshipStyleStateSpec,
    SelfPersonaCoreSpec,
    TeachingProfileStateSpec,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def _relationship_state_payload_without_updated_at(projection) -> dict:
    payload = projection.as_dict()
    payload.pop("updated_at", None)
    return payload


def _remember_persona_memories(store: BrainStore, session_ids) -> None:
    payloads = (
        (
            "interaction.style",
            "interaction",
            {"value": "gentle direct collaboration"},
            "User prefers gentle direct collaboration.",
            False,
            "evt-style-1",
        ),
        (
            "interaction.preference",
            "interaction",
            {"value": "warm but concise"},
            "Warm but concise collaboration works well.",
            False,
            "evt-pref-1",
        ),
        (
            "interaction.misfire",
            "interaction",
            {"value": "too much preamble"},
            "Too much preamble felt off.",
            False,
            "evt-misfire-1",
        ),
        (
            "teaching.preference.mode",
            "teaching",
            {"value": "walkthrough"},
            "Walkthroughs work best.",
            False,
            "evt-mode-1",
        ),
        (
            "teaching.preference.analogy_domain",
            "teaching",
            {"value": "physics"},
            "Physics analogies land well.",
            False,
            "evt-analogy-1",
        ),
        (
            "teaching.history.helpful_pattern",
            "teaching",
            {"value": "stepwise decomposition"},
            "Stepwise decomposition helped.",
            False,
            "evt-pattern-1",
        ),
    )
    for namespace, subject, value, rendered_text, singleton, source_event_id in payloads:
        store.remember_fact(
            user_id=session_ids.user_id,
            namespace=namespace,
            subject=subject,
            value=value,
            rendered_text=rendered_text,
            confidence=0.91,
            singleton=singleton,
            source_event_id=source_event_id,
            source_episode_id=None,
            provenance={"source": "test"},
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )


def test_persona_memory_blocks_persist_in_scoped_core_blocks_and_survive_restart(tmp_path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="persona-memory")
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"

    store.ensure_default_blocks(load_default_agent_blocks())
    _remember_persona_memories(store, session_ids)

    self_persona_core = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
    )
    relationship_style = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    teaching_profile = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )

    assert self_persona_core is not None
    assert relationship_style is not None
    assert teaching_profile is not None

    persona_core_payload = SelfPersonaCoreSpec.model_validate(self_persona_core.content)
    relationship_style_payload = RelationshipStyleStateSpec.model_validate(relationship_style.content)
    teaching_profile_payload = TeachingProfileStateSpec.model_validate(teaching_profile.content)

    assert persona_core_payload.canonical_name == "Blink"
    assert relationship_style_payload.relationship_id == relationship_scope_id
    assert relationship_style_payload.collaboration_style == "warm but concise"
    assert "non-romantic" in relationship_style_payload.boundaries
    assert relationship_style_payload.known_misfires == ["too much preamble"]
    assert teaching_profile_payload.preferred_modes[0] == "walkthrough"
    assert teaching_profile_payload.analogy_domains == ["physics"]
    assert teaching_profile_payload.helpful_patterns == ["stepwise decomposition"]

    current_user_claims = store.query_claims(
        temporal_mode="current",
        scope_type="user",
        scope_id=session_ids.user_id,
        limit=32,
    )
    relationship_claims = store.query_claims(
        temporal_mode="current",
        scope_type="relationship",
        scope_id=relationship_scope_id,
        limit=32,
    )
    assert {
        "interaction.style",
        "interaction.preference",
        "interaction.misfire",
        "teaching.preference.mode",
        "teaching.preference.analogy_domain",
        "teaching.history.helpful_pattern",
    }.issubset({claim.predicate for claim in current_user_claims})
    assert relationship_claims == []

    self_persona_version_count = len(
        store.list_core_memory_block_versions(
            block_kind=BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
            scope_type="agent",
            scope_id=session_ids.agent_id,
        )
    )
    relationship_style_version_count = len(
        store.list_core_memory_block_versions(
            block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
    )
    teaching_profile_version_count = len(
        store.list_core_memory_block_versions(
            block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
    )
    _remember_persona_memories(store, session_ids)
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
                scope_type="agent",
                scope_id=session_ids.agent_id,
            )
        )
        == self_persona_version_count
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
            )
        )
        == relationship_style_version_count
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
            )
        )
        == teaching_profile_version_count
    )

    relationship_projection_before = store.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    assert relationship_projection_before.collaboration_style == "warm but concise"
    assert relationship_projection_before.preferred_teaching_modes[0] == "walkthrough"
    assert relationship_projection_before.analogy_domains == ["physics"]

    store.close()

    restarted = BrainStore(path=db_path)
    restarted_self_persona_core = restarted.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
    )
    restarted_relationship_style = restarted.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    restarted_teaching_profile = restarted.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    relationship_projection_after = restarted.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )

    assert restarted_self_persona_core is not None
    assert restarted_relationship_style is not None
    assert restarted_teaching_profile is not None
    assert restarted_self_persona_core.content == self_persona_core.content
    assert restarted_relationship_style.content == relationship_style.content
    assert restarted_teaching_profile.content == teaching_profile.content
    assert _relationship_state_payload_without_updated_at(
        relationship_projection_after
    ) == _relationship_state_payload_without_updated_at(relationship_projection_before)

    restarted.close()


def test_persona_memory_backfill_and_rebuild_preserve_relationship_style_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="persona-backfill")
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"

    store.ensure_default_blocks(load_default_agent_blocks())
    store.remember_semantic_memory(
        user_id=session_ids.user_id,
        namespace="interaction.style",
        subject="interaction",
        value={"value": "direct with light warmth"},
        rendered_text="Direct with light warmth works best.",
        confidence=0.88,
        singleton=False,
    )
    store.remember_semantic_memory(
        user_id=session_ids.user_id,
        namespace="interaction.misfire",
        subject="interaction",
        value={"value": "overly sentimental phrasing"},
        rendered_text="Overly sentimental phrasing feels wrong.",
        confidence=0.9,
        singleton=False,
    )
    store.remember_semantic_memory(
        user_id=session_ids.user_id,
        namespace="teaching.preference.mode",
        subject="teaching",
        value={"value": "deep_dive"},
        rendered_text="Deep dives are helpful.",
        confidence=0.9,
        singleton=False,
    )
    store.remember_semantic_memory(
        user_id=session_ids.user_id,
        namespace="teaching.preference.analogy_domain",
        subject="teaching",
        value={"value": "music"},
        rendered_text="Music analogies help.",
        confidence=0.9,
        singleton=False,
    )

    store.backfill_continuity_from_legacy(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )

    relationship_style = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    teaching_profile = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    before_state = store.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    relationship_dossier = next(
        record for record in dossiers.dossiers if record.kind == "relationship"
    )
    self_policy_dossier = next(
        record for record in dossiers.dossiers if record.kind == "self_policy"
    )

    assert relationship_style is not None
    assert teaching_profile is not None
    assert before_state.collaboration_style == "direct with light warmth"
    assert before_state.known_misfires == ["overly sentimental phrasing"]
    assert before_state.preferred_teaching_modes[0] == "deep_dive"
    assert before_state.analogy_domains == ["music"]
    assert {"relationship_style", "teaching_profile"}.issubset(
        set(relationship_dossier.details.get("current_block_kinds", []))
    )
    assert "self_persona_core" in self_policy_dossier.details.get("current_block_kinds", [])

    store.rebuild_brain_projections()
    store.backfill_continuity_from_legacy(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )
    after_state = store.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    refreshed_relationship_style = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    refreshed_teaching_profile = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )

    assert refreshed_relationship_style is not None
    assert refreshed_teaching_profile is not None
    assert _relationship_state_payload_without_updated_at(after_state) == _relationship_state_payload_without_updated_at(
        before_state
    )
    assert refreshed_relationship_style.content == relationship_style.content
    assert refreshed_teaching_profile.content == teaching_profile.content
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
            )
        )
        == 1
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
            )
        )
        == 1
    )

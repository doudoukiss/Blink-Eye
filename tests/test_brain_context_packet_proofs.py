from __future__ import annotations

from dataclasses import replace
from random import Random

import pytest

from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context import (
    BrainContextBudgetProfile,
    BrainContextCompiler,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.memory_v2 import BrainGovernanceReasonCode, ClaimLedger
from blink.brain.persona import BrainPersonaModality
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _base_surface(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="packet-proofs")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Review dock state",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
    )
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(
        latest_user_text="Review the active dock state.",
        include_historical_claims=True,
    )
    return surface, session_ids


def _compiler_with_persona_defaults(
    tmp_path,
    *,
    language: Language = Language.EN,
    client_id: str = "packet-teaching-auto",
):
    store = BrainStore(path=tmp_path / f"{client_id}.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)
    store.ensure_default_blocks(load_default_agent_blocks())
    return BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=language,
        base_prompt=base_brain_system_prompt(language),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=language,
        ),
    )


def _surface_with_persona_defaults(tmp_path, *, client_id: str = "packet-persona"):
    store = BrainStore(path=tmp_path / f"{client_id}.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)
    store.ensure_default_blocks(load_default_agent_blocks())
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="interaction.preference",
        subject="interaction",
        value={"value": "warm but concise"},
        rendered_text="Warm but concise collaboration works well.",
        confidence=0.91,
        singleton=False,
        source_event_id=f"evt-{client_id}-pref",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="teaching.preference.mode",
        subject="teaching",
        value={"value": "walkthrough"},
        rendered_text="Walkthroughs work best.",
        confidence=0.91,
        singleton=False,
        source_event_id=f"evt-{client_id}-teaching",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(latest_user_text="Explain the current context.")
    return surface, session_ids


def _selection_decision(packet, section_key: str):
    return next(
        decision
        for decision in packet.selected_context.selection_trace.decisions
        if decision.section_key == section_key
    )


def _surface_with_governance_claims(tmp_path):
    store = BrainStore(path=tmp_path / "brain-governance.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="packet-governance",
    )
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink capabilities",
        }
    )
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=store)
    event_context = store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="packet-governance",
    )

    prior_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-role-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role",
        event_context=event_context,
    )
    current_claim = ledger.supersede_claim(
        prior_claim.claim_id,
        replacement_claim={
            "subject_entity_id": user.entity_id,
            "predicate": "profile.role",
            "object_data": {"value": "product manager"},
            "scope_type": "user",
            "scope_id": session_ids.user_id,
            "claim_key": "profile.role",
        },
        reason="updated the recorded role",
        source_event_id="evt-role-2",
        event_context=event_context,
    )
    held_claim = ledger.request_claim_review(
        current_claim.claim_id,
        source_event_id="evt-role-review",
        reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
        event_context=event_context,
    )
    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
    ).build(
        latest_user_text="What is my role right now?",
        include_historical_claims=True,
    )
    return surface, held_claim.claim_id


def test_persona_expression_reply_section_is_compact_bounded_and_prompt_safe(tmp_path):
    surface, _session_ids = _surface_with_persona_defaults(tmp_path)
    base_prompt = base_brain_system_prompt(Language.EN)

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain your style briefly.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        base_prompt=base_prompt,
    )

    section = packet.selected_context.section("persona_expression")
    decision = _selection_decision(packet, "persona_expression")

    assert base_prompt == (
        "You are Blink. Always follow the system-provided BLINK_BRAIN_CONTEXT. "
        "Do not invent undeclared capabilities."
    )
    assert section is not None
    assert section.estimated_tokens <= 80
    assert (
        packet.selected_context.estimated_tokens
        <= packet.selected_context.budget_profile.max_tokens
    )
    assert decision.selected is True
    assert "persona_expression_compiled" in decision.decision_reason_codes
    assert "identity: Blink; local non-human system" in section.content
    assert "character: warm precise local tutor; no human backstory" in section.content
    assert "relationship boundaries: non-romantic; non-sexual; non-exclusive" in section.content
    assert "teaching: mode=walkthrough" in section.content
    assert "voice:" not in section.content
    assert "Blink Scholar-Companion" not in packet.prompt
    assert "persona_profile_id" not in packet.prompt
    assert "core_values" not in packet.prompt
    assert "```json" not in packet.prompt


def test_persona_expression_answers_character_queries_without_full_charter(tmp_path):
    surface, _session_ids = _surface_with_persona_defaults(
        tmp_path,
        client_id="packet-persona-character",
    )

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="What is your character?",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        base_prompt=base_brain_system_prompt(Language.EN),
    )
    section = packet.selected_context.section("persona_expression")

    assert section is not None
    assert "character: warm precise local tutor" in section.content
    assert "local non-human system" in section.content
    assert "no human backstory" in section.content
    assert "Blink Scholar-Companion" not in packet.prompt
    assert "core_values" not in packet.prompt
    assert "persona_profile_id" not in packet.prompt
    assert "```json" not in packet.prompt


def test_persona_expression_can_be_disabled_with_trace(tmp_path):
    surface, _session_ids = _surface_with_persona_defaults(
        tmp_path,
        client_id="packet-persona-disabled",
    )

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain your style briefly.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        enable_persona_expression=False,
    )
    decision = _selection_decision(packet, "persona_expression")

    assert packet.selected_context.section("persona_expression") is None
    assert decision.selected is False
    assert decision.reason == "empty"
    assert "persona_expression_disabled" in decision.decision_reason_codes


def test_persona_expression_missing_defaults_fail_closed_with_trace(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain your style briefly.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
    )
    decision = _selection_decision(packet, "persona_expression")

    assert packet.selected_context.section("persona_expression") is None
    assert decision.selected is False
    assert decision.reason == "empty"
    assert "persona_defaults_missing" in decision.decision_reason_codes


def test_persona_expression_voice_line_requires_voice_like_modality(tmp_path):
    surface, _session_ids = _surface_with_persona_defaults(
        tmp_path,
        client_id="packet-persona-voice",
    )

    text_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain your style briefly.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
    )
    browser_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain your style briefly.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        persona_modality=BrainPersonaModality.BROWSER,
    )

    text_section = text_packet.selected_context.section("persona_expression")
    browser_section = browser_packet.selected_context.section("persona_expression")
    browser_decision = _selection_decision(browser_packet, "persona_expression")

    assert text_section is not None
    assert "voice:" not in text_section.content
    assert browser_section is not None
    assert "voice: rate=1.00; pause=0.34; emphasis=measured clarity" in browser_section.content
    assert "voice_hints:included" in browser_decision.decision_reason_codes


def test_persona_expression_is_reply_only_for_this_patch(tmp_path):
    surface, _session_ids = _surface_with_persona_defaults(
        tmp_path,
        client_id="packet-persona-planning",
    )

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Plan the next step.",
        task=BrainContextTask.PLANNING,
        language=Language.EN,
    )

    assert packet.selected_context.section("persona_expression") is None
    assert not any(
        decision.section_key == "persona_expression"
        for decision in packet.selected_context.selection_trace.decisions
    )


def test_teaching_knowledge_context_is_excluded_by_default_with_trace(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain Bayesian updating.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
    )
    decision = _selection_decision(packet, "teaching_knowledge")

    assert packet.selected_context.section("teaching_knowledge") is None
    assert "## Teaching Knowledge" not in packet.prompt
    assert decision.selected is False
    assert decision.reason == "empty"
    assert "teaching_knowledge_empty" in decision.decision_reason_codes
    assert packet.teaching_knowledge_decision is None


def test_compiler_auto_selects_default_teaching_knowledge_for_english_reply(tmp_path):
    compiler = _compiler_with_persona_defaults(tmp_path, language=Language.EN)

    packet = compiler.compile_packet(
        latest_user_text="Explain recursion with a small example.",
        task=BrainContextTask.REPLY,
    )
    section = packet.selected_context.section("teaching_knowledge")
    decision = _selection_decision(packet, "teaching_knowledge")

    assert section is not None
    assert "exemplar:walkthrough_small_example" in section.content
    assert "source=blink-default-teaching-canon" in section.content
    assert "provenance=curator:blink,kind:internal-pedagogy,version:2026-04" in (section.content)
    assert "Use only available sources or repo-visible context" not in section.content
    assert section.estimated_tokens <= 96
    assert (
        packet.selected_context.estimated_tokens
        <= packet.selected_context.budget_profile.max_tokens
    )
    assert decision.selected is True
    assert "teaching_knowledge_auto_selected" in decision.decision_reason_codes
    assert "teaching_mode:clarify" in decision.decision_reason_codes
    assert packet.teaching_knowledge_decision is not None
    assert packet.teaching_knowledge_decision.available is True
    assert any(
        item.item_id == "exemplar:walkthrough_small_example"
        for item in packet.teaching_knowledge_decision.selected_items
    )
    assert section.content not in str(packet.teaching_knowledge_decision.as_dict())


def test_compiler_auto_selects_chinese_first_teaching_knowledge(tmp_path):
    compiler = _compiler_with_persona_defaults(
        tmp_path,
        language=Language.ZH,
        client_id="packet-teaching-auto-zh",
    )

    packet = compiler.compile_packet(
        latest_user_text="请解释递归调试思路",
        task=BrainContextTask.REPLY,
    )
    section = packet.selected_context.section("teaching_knowledge")
    decision = _selection_decision(packet, "teaching_knowledge")

    assert section is not None
    assert "exemplar:chinese_technical_explanation_bridge" in section.content
    assert "source=blink-default-teaching-canon" in section.content
    assert "provenance=" in section.content
    assert section.estimated_tokens <= 96
    assert decision.selected is True
    assert "teaching_knowledge_auto_selected" in decision.decision_reason_codes
    assert packet.teaching_knowledge_decision is not None
    assert packet.teaching_knowledge_decision.language == "zh"
    assert any(
        item.item_id == "exemplar:chinese_technical_explanation_bridge"
        for item in packet.teaching_knowledge_decision.selected_items
    )


@pytest.mark.parametrize(
    ("query_text", "language", "expected_id"),
    [
        (
            "Debug this failing function with one hypothesis and one minimal repro.",
            Language.EN,
            "exemplar:debugging_hypothesis_one_change",
        ),
        (
            "I think recursion means an infinite loop; correct my misconception.",
            Language.EN,
            "exemplar:misconception_repair_without_shame",
        ),
        (
            "Answer from sources and cite the documentation if evidence is uncertain.",
            Language.EN,
            "exemplar:source_grounded_answer_with_limits",
        ),
        (
            "Give me one practice prompt with an answer key for recursion.",
            Language.EN,
            "sequence:practice_prompt_with_answer_key",
        ),
        (
            "请解释递归调试思路",
            Language.ZH,
            "exemplar:chinese_technical_explanation_bridge",
        ),
    ],
)
def test_compiler_auto_teaching_knowledge_selects_quality_cases_under_budget(
    tmp_path,
    query_text,
    language,
    expected_id,
):
    compiler = _compiler_with_persona_defaults(
        tmp_path,
        language=language,
        client_id=f"packet-teaching-quality-{expected_id.rsplit(':', 1)[-1]}",
    )

    packet = compiler.compile_packet(
        latest_user_text=query_text,
        task=BrainContextTask.REPLY,
    )
    section = packet.selected_context.section("teaching_knowledge")
    decision = _selection_decision(packet, "teaching_knowledge")

    assert section is not None
    assert expected_id in section.content
    assert "source=blink-default-teaching-canon" in section.content
    assert "provenance=curator:blink,kind:internal-pedagogy,version:2026-04" in section.content
    assert "Use only available sources or repo-visible context" not in section.content
    assert "https://" not in section.content
    assert "doi:" not in section.content.lower()
    assert section.estimated_tokens <= 96
    assert (
        packet.selected_context.estimated_tokens
        <= packet.selected_context.budget_profile.max_tokens
    )
    assert decision.selected is True
    assert "teaching_knowledge_auto_selected" in decision.decision_reason_codes


def test_compiler_auto_teaching_knowledge_is_reply_only(tmp_path):
    compiler = _compiler_with_persona_defaults(tmp_path, language=Language.EN)

    packet = compiler.compile_packet(
        latest_user_text="Plan a debugging minimal reproduction lesson.",
        task=BrainContextTask.PLANNING,
    )

    assert packet.selected_context.section("teaching_knowledge") is None
    assert not any(
        decision.section_key == "teaching_knowledge"
        for decision in packet.selected_context.selection_trace.decisions
    )
    assert packet.teaching_knowledge_decision is not None
    assert packet.teaching_knowledge_decision.available is False
    assert "teaching_knowledge_reply_only" in packet.teaching_knowledge_decision.reason_codes


def test_explicit_teaching_knowledge_context_is_budgeted_and_traceable(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)
    teaching_context = (
        "- exemplar:bayes: use a coin example; explain prior > observation > posterior | "
        "source=unit"
    )

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Explain Bayesian updating.",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        teaching_knowledge_context=teaching_context,
    )
    section = packet.selected_context.section("teaching_knowledge")
    decision = _selection_decision(packet, "teaching_knowledge")

    assert section is not None
    assert section.content == teaching_context
    assert section.estimated_tokens <= 40
    assert (
        packet.selected_context.estimated_tokens
        <= packet.selected_context.budget_profile.max_tokens
    )
    assert decision.selected is True
    assert "teaching_knowledge_selected" in decision.decision_reason_codes
    assert "unit" in packet.prompt
    assert packet.teaching_knowledge_decision is not None
    assert packet.teaching_knowledge_decision.selection_kind == "explicit_context"
    assert packet.teaching_knowledge_decision.selected_items == ()
    assert teaching_context not in str(packet.teaching_knowledge_decision.as_dict())


def test_teaching_knowledge_context_is_reply_only(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)

    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Plan a Bayesian lesson.",
        task=BrainContextTask.PLANNING,
        language=Language.EN,
        teaching_knowledge_context="- sequence:bayes: prior > evidence > posterior",
    )

    assert packet.selected_context.section("teaching_knowledge") is None
    assert not any(
        decision.section_key == "teaching_knowledge"
        for decision in packet.selected_context.selection_trace.decisions
    )


def _replace_active_state(
    surface,
    *,
    entity_state: str = BrainSceneWorldRecordState.ACTIVE.value,
    affordance_state: str = BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
    degraded_mode: str = "limited",
) -> object:
    private_working_memory = BrainPrivateWorkingMemoryProjection(
        scope_type="thread",
        scope_id=surface.private_working_memory.scope_id,
        records=[
            BrainPrivateWorkingMemoryRecord(
                record_id="pwm-plan",
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                summary="Need user confirmation before maintenance handoff.",
                state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                backing_ids=["proposal-maint-1"],
                source_event_ids=["evt-plan-1"],
                goal_id="goal-maint-1",
                commitment_id="commitment-maint-1",
                plan_proposal_id="proposal-maint-1",
                observed_at=_ts(10),
                updated_at=_ts(10),
            ),
            BrainPrivateWorkingMemoryRecord(
                record_id="pwm-uncertain",
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                summary="Scene confidence is degraded after the latest frame drop.",
                state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                backing_ids=["scene-world-1"],
                source_event_ids=["evt-scene-1"],
                commitment_id="commitment-maint-1",
                observed_at=_ts(11),
                updated_at=_ts(11),
            ),
        ],
        updated_at=_ts(12),
    )
    scene_world_state = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=[
            BrainSceneWorldEntityRecord(
                entity_id="entity-desk",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="diagnostic_pad",
                summary="A diagnostic pad is visible on the desk.",
                state=entity_state,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                zone_id="zone:desk",
                confidence=0.81,
                freshness="current" if entity_state == "active" else entity_state,
                contradiction_codes=(["zone_changed"] if entity_state == "contradicted" else []),
                affordance_ids=["aff-vision"],
                backing_ids=["entity:diagnostic_pad"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(12),
                updated_at=_ts(12),
                expires_at=_ts(40),
            )
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="aff-vision",
                entity_id="entity-desk",
                capability_family="vision.inspect",
                summary="The diagnostic pad can be visually inspected.",
                availability=affordance_state,
                confidence=0.76,
                freshness="current" if affordance_state == "available" else affordance_state,
                reason_codes=(["degraded_feed"] if affordance_state != "available" else []),
                backing_ids=["entity:diagnostic_pad", "vision.inspect"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(12),
                updated_at=_ts(12),
                expires_at=_ts(40),
            )
        ],
        degraded_mode=degraded_mode,
        degraded_reason_codes=(["camera_frame_stale"] if degraded_mode != "healthy" else []),
        updated_at=_ts(12),
    )
    active_situation_model = BrainActiveSituationProjection(
        scope_type="thread",
        scope_id=surface.active_situation_model.scope_id,
        records=[
            BrainActiveSituationRecord(
                record_id="situation-plan",
                record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                summary="Maintenance handoff plan is current but still waiting on confirmation.",
                state=BrainActiveSituationRecordState.ACTIVE.value,
                evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                confidence=0.74,
                freshness="current",
                uncertainty_codes=[],
                private_record_ids=["pwm-plan"],
                backing_ids=["proposal-maint-1"],
                source_event_ids=["evt-plan-1"],
                goal_id="goal-maint-1",
                commitment_id="commitment-maint-1",
                plan_proposal_id="proposal-maint-1",
                observed_at=_ts(10),
                updated_at=_ts(12),
            ),
            BrainActiveSituationRecord(
                record_id="situation-uncertain",
                record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                summary="Perception is degraded; world-state evidence may be stale.",
                state=BrainActiveSituationRecordState.UNRESOLVED.value,
                evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                confidence=0.5,
                freshness="limited",
                uncertainty_codes=["scene_stale", "camera_frame_stale"],
                private_record_ids=["pwm-uncertain"],
                backing_ids=["scene-world-1"],
                source_event_ids=["evt-scene-1"],
                commitment_id="commitment-maint-1",
                observed_at=_ts(11),
                updated_at=_ts(12),
            ),
        ],
        updated_at=_ts(12),
    )
    return replace(
        surface,
        private_working_memory=private_working_memory,
        scene_world_state=scene_world_state,
        active_situation_model=active_situation_model,
    )


@pytest.mark.parametrize("task", list(BrainContextTask))
def test_task_general_packet_compiler_respects_budget_invariants(tmp_path, task):
    surface, _session_ids = _base_surface(tmp_path)
    rich_surface = _replace_active_state(surface)
    budget = BrainContextBudgetProfile(task=task.value, max_tokens=280)

    packet = compile_context_packet_from_surface(
        snapshot=rich_surface,
        latest_user_text=f"Review the {task.value} context.",
        task=task,
        language=Language.EN,
        budget_profile=budget,
    )

    assert packet.packet_trace is not None
    assert packet.packet_trace.task == task.value
    assert packet.packet_trace.mode_policy.task.value == task.value
    assert packet.selected_context.estimated_tokens <= budget.max_tokens
    assert packet.packet_trace.section_decisions
    assert all(item.decision_reason_codes for item in packet.packet_trace.selected_items)
    assert all(item.decision_reason_codes for item in packet.packet_trace.dropped_items)
    assert all(
        item.section_key
        in {
            "active_state",
            "active_continuity",
            "unresolved_state",
            "recent_changes",
            "planning_anchors",
            "relevant_continuity",
        }
        for item in packet.packet_trace.selected_items
    )


def test_new_context_modes_apply_explicit_mode_policy_and_section_shapes(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)
    rich_surface = _replace_active_state(surface)
    packets = {
        BrainContextTask.WAKE: compile_context_packet_from_surface(
            snapshot=rich_surface,
            latest_user_text="Route the next wake decision.",
            task=BrainContextTask.WAKE,
            language=Language.EN,
        ),
        BrainContextTask.REEVALUATION: compile_context_packet_from_surface(
            snapshot=rich_surface,
            latest_user_text="Reevaluate the current held work.",
            task=BrainContextTask.REEVALUATION,
            language=Language.EN,
        ),
        BrainContextTask.OPERATOR_AUDIT: compile_context_packet_from_surface(
            snapshot=rich_surface,
            latest_user_text="Audit the current situation packet.",
            task=BrainContextTask.OPERATOR_AUDIT,
            language=Language.EN,
        ),
        BrainContextTask.GOVERNANCE_REVIEW: compile_context_packet_from_surface(
            snapshot=rich_surface,
            latest_user_text="Review the current governance state.",
            task=BrainContextTask.GOVERNANCE_REVIEW,
            language=Language.EN,
        ),
    }

    wake = packets[BrainContextTask.WAKE]
    reevaluation = packets[BrainContextTask.REEVALUATION]
    operator_audit = packets[BrainContextTask.OPERATOR_AUDIT]
    governance_review = packets[BrainContextTask.GOVERNANCE_REVIEW]

    assert wake.packet_trace.mode_policy.trace_verbosity.value == "standard"
    assert "planning_anchors" in wake.packet_trace.mode_policy.dynamic_section_keys
    assert any(
        decision.section_key == "planning_anchors"
        for decision in wake.packet_trace.section_decisions
    )
    assert wake.selected_context.section("relevant_continuity") is None
    assert reevaluation.selected_context.section("unresolved_state") is not None
    assert reevaluation.selected_context.section("planning_anchors") is None
    assert operator_audit.packet_trace.mode_policy.trace_verbosity.value == "verbose"
    assert "relevant_continuity" in operator_audit.packet_trace.mode_policy.dynamic_section_keys
    assert any(
        decision.section_key == "relevant_continuity"
        for decision in operator_audit.packet_trace.section_decisions
    )
    assert governance_review.packet_trace.mode_policy.trace_verbosity.value == "verbose"
    assert "planning_anchors" not in governance_review.packet_trace.mode_policy.dynamic_section_keys
    assert all(
        item.decision_reason_codes
        for packet in packets.values()
        for item in packet.packet_trace.selected_items + packet.packet_trace.dropped_items
    )


def test_packet_compiler_monotonic_freshness_decay_for_scene_world_state(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)
    active_surface = _replace_active_state(
        surface, entity_state="active", affordance_state="available"
    )
    stale_surface = _replace_active_state(surface, entity_state="stale", affordance_state="stale")
    expired_surface = _replace_active_state(
        surface, entity_state="expired", affordance_state="uncertain"
    )

    active_packet = compile_context_packet_from_surface(
        snapshot=active_surface,
        latest_user_text="Review the current desk scene.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )
    stale_packet = compile_context_packet_from_surface(
        snapshot=stale_surface,
        latest_user_text="Review the current desk scene.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )
    expired_packet = compile_context_packet_from_surface(
        snapshot=expired_surface,
        latest_user_text="Review the current desk scene.",
        task=BrainContextTask.REFLECTION,
        language=Language.EN,
    )

    assert any(
        item.section_key == "active_state" and item.provenance.get("entity_id") == "entity-desk"
        for item in active_packet.packet_trace.selected_items
    )
    assert not any(
        item.section_key == "active_state" and item.provenance.get("entity_id") == "entity-desk"
        for item in stale_packet.packet_trace.selected_items
    )
    assert any(
        item.section_key == "unresolved_state" and item.provenance.get("entity_id") == "entity-desk"
        for item in stale_packet.packet_trace.selected_items
    )
    assert not any(
        item.section_key == "active_state" and item.provenance.get("entity_id") == "entity-desk"
        for item in expired_packet.packet_trace.selected_items
    )


def test_packet_compiler_duplicate_irrelevant_state_does_not_shift_key_selected_ids(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)
    base_surface = _replace_active_state(surface)
    duplicate_surface = replace(
        base_surface,
        scene_world_state=BrainSceneWorldProjection.from_dict(
            {
                **base_surface.scene_world_state.as_dict(),
                "entities": base_surface.scene_world_state.as_dict()["entities"]
                + [
                    {
                        "entity_id": "entity-irrelevant",
                        "entity_kind": "object",
                        "canonical_label": "unused_manual",
                        "summary": "An irrelevant stale object sits off to the side.",
                        "state": "stale",
                        "evidence_kind": "observed",
                        "zone_id": "zone:edge",
                        "confidence": 0.2,
                        "freshness": "stale",
                        "affordance_ids": [],
                        "backing_ids": ["entity:irrelevant"],
                        "source_event_ids": ["evt-irrelevant"],
                        "observed_at": _ts(13),
                        "updated_at": _ts(13),
                        "expires_at": _ts(30),
                        "details": {},
                    }
                ],
            }
        ),
    )

    first = compile_context_packet_from_surface(
        snapshot=base_surface,
        latest_user_text="Recall the maintenance handoff state.",
        task=BrainContextTask.RECALL,
        language=Language.EN,
    )
    second = compile_context_packet_from_surface(
        snapshot=duplicate_surface,
        latest_user_text="Recall the maintenance handoff state.",
        task=BrainContextTask.RECALL,
        language=Language.EN,
    )
    first_digest = build_context_packet_digest(
        packet_traces={"recall": first.packet_trace.as_dict() if first.packet_trace else None}
    )
    second_digest = build_context_packet_digest(
        packet_traces={"recall": second.packet_trace.as_dict() if second.packet_trace else None}
    )

    assert (
        first_digest["recall"]["selected_backing_ids"]
        == second_digest["recall"]["selected_backing_ids"]
    )
    assert (
        first_digest["recall"]["selected_provenance_ids"]
        == second_digest["recall"]["selected_provenance_ids"]
    )


def test_packet_digest_tracks_governance_annotations_and_suppressions(tmp_path):
    surface, held_claim_id = _surface_with_governance_claims(tmp_path)

    reply_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="What is my role right now?",
        task=BrainContextTask.REPLY,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="reply", max_tokens=1100),
    )
    recall_packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text="Recall my role changes.",
        task=BrainContextTask.RECALL,
        language=Language.EN,
        budget_profile=BrainContextBudgetProfile(task="recall", max_tokens=1100),
    )
    digest = build_context_packet_digest(
        packet_traces={
            "reply": reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None,
            "recall": recall_packet.packet_trace.as_dict() if recall_packet.packet_trace else None,
        }
    )

    assert digest["reply"]["selected_availability_counts"]["suppressed"] >= 1
    assert digest["reply"]["governance_drop_reason_counts"]["held_support"] >= 1
    assert digest["reply"]["governance_drop_reason_counts"]["review_debt"] >= 1
    assert digest["reply"]["governance_drop_reason_counts"]["stale_support"] >= 1
    assert digest["reply"]["governance_reason_code_counts"]["requires_confirmation"] >= 1
    assert held_claim_id in digest["reply"]["suppressed_backing_ids"]
    assert digest["recall"]["selected_availability_counts"]["annotated"] >= 1
    assert held_claim_id in digest["recall"]["annotated_backing_ids"]
    assert held_claim_id in digest["recall"]["selected_backing_ids"]


def test_packet_compiler_seeded_fuzz_active_state_inputs_stay_bounded(tmp_path):
    surface, _session_ids = _base_surface(tmp_path)
    rng = Random(0)
    tasks = list(BrainContextTask)

    for seed in range(10):
        entity_state = rng.choice(["active", "stale", "contradicted", "expired"])
        affordance_state = rng.choice(["available", "blocked", "uncertain", "stale"])
        degraded_mode = rng.choice(["healthy", "limited", "unavailable"])
        rich_surface = _replace_active_state(
            surface,
            entity_state=entity_state,
            affordance_state=affordance_state,
            degraded_mode=degraded_mode,
        )
        for task in tasks:
            packet = compile_context_packet_from_surface(
                snapshot=rich_surface,
                latest_user_text=f"seeded fuzz {seed} for {task.value}",
                task=task,
                language=Language.EN,
                budget_profile=BrainContextBudgetProfile(task=task.value, max_tokens=320),
            )
            assert packet.packet_trace is not None
            assert packet.selected_context.estimated_tokens <= 320

from pathlib import Path

import pytest

from blink.brain.evals import (
    BrainContinuityEvalCase,
    BrainContinuityEvalHarness,
    BrainContinuityExpectedState,
    BrainConversationStep,
)
from blink.brain.evals.replay_cases import load_replay_regression_case
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


@pytest.mark.parametrize(
    ("name", "steps", "expected_state", "context_queries"),
    [
        (
            "fact_correction",
            (
                BrainConversationStep(
                    user_text="I used to be a designer.",
                    assistant_text="Noted.",
                    assistant_summary="Recorded the older role.",
                    remember_facts=(
                        {
                            "namespace": "profile.role",
                            "subject": "user",
                            "value": {"value": "designer"},
                            "rendered_text": "user role is designer",
                            "confidence": 0.8,
                            "singleton": True,
                            "source_event_id": "evt-role-1",
                        },
                    ),
                ),
                BrainConversationStep(
                    user_text="Correction: I am a product manager now.",
                    assistant_text="Updated.",
                    assistant_summary="Recorded the corrected role.",
                    remember_facts=(
                        {
                            "namespace": "profile.role",
                            "subject": "user",
                            "value": {"value": "product manager"},
                            "rendered_text": "user role is product manager",
                            "confidence": 0.95,
                            "singleton": True,
                            "source_event_id": "evt-role-2",
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                current_claims=({"predicate": "profile.role", "value": "product manager"},),
                historical_claims=({"predicate": "profile.role", "value": "designer"},),
                continuity_graph_edge_kinds=("claim_subject", "supported_by_event", "supersedes"),
            ),
            {},
        ),
        (
            "preference_change",
            (
                BrainConversationStep(
                    user_text="I like coffee.",
                    assistant_text="Noted.",
                    assistant_summary="Recorded the earlier preference.",
                    remember_facts=(
                        {
                            "namespace": "preference.like",
                            "subject": "coffee",
                            "value": {"value": "coffee"},
                            "rendered_text": "user likes coffee",
                            "confidence": 0.8,
                            "singleton": False,
                            "source_event_id": "evt-pref-1",
                        },
                    ),
                ),
                BrainConversationStep(
                    user_text="Correction: I do not like coffee.",
                    assistant_text="Updated.",
                    assistant_summary="Recorded the corrected preference.",
                    remember_facts=(
                        {
                            "namespace": "preference.dislike",
                            "subject": "coffee",
                            "value": {"value": "coffee"},
                            "rendered_text": "user dislikes coffee",
                            "confidence": 0.94,
                            "singleton": False,
                            "source_event_id": "evt-pref-2",
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                current_claims=({"predicate": "preference.dislike", "value": "coffee"},),
                historical_claims=({"predicate": "preference.like", "value": "coffee"},),
            ),
            {},
        ),
        (
            "commitment_lifecycle",
            (
                BrainConversationStep(
                    user_text="Please remind me to call mom.",
                    assistant_text="I will remember that.",
                    assistant_summary="Opened a commitment to call mom.",
                    upsert_tasks=(
                        {
                            "title": "Call mom",
                            "details": {"summary": "Need to call mom"},
                            "status": "open",
                        },
                    ),
                ),
                BrainConversationStep(
                    user_text="I already called her.",
                    assistant_text="Marked complete.",
                    assistant_summary="Completed the commitment.",
                    task_status_updates=(
                        {
                            "title": "Call mom",
                            "status": "done",
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                commitment_titles_by_status={"recent_terminal": ("Call mom",)},
                agenda_contains={"completed_goals": ("Call mom",)},
            ),
            {},
        ),
        (
            "relationship_drift",
            (
                BrainConversationStep(
                    user_text="We need to keep pushing the Alpha project this week.",
                    assistant_text="Agreed.",
                    assistant_summary="Continued the Alpha project together.",
                ),
                BrainConversationStep(
                    user_text="Sorry, I was too blunt earlier.",
                    assistant_text="Understood, we can reset.",
                    assistant_summary="Relationship tone shifted after a correction.",
                ),
            ),
            BrainContinuityExpectedState(
                autobiography_entry_kinds=("relationship_arc",),
                relationship_arc_contains="Alpha",
                relationship_dossier_summary_contains="Alpha",
                relationship_dossier_freshness="fresh",
                relationship_dossier_contradiction="clear",
                project_dossier_keys=("Alpha",),
            ),
            {},
        ),
        (
            "reply_current_truth_query",
            (
                BrainConversationStep(
                    user_text="I used to be a designer.",
                    assistant_text="Noted.",
                    assistant_summary="Recorded the earlier role.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_prior",
                                "subject_entity_id": "entity:user:alpha",
                                "predicate": "profile.role",
                                "object": {"value": "designer"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": "alpha",
                                "claim_key": "profile.role:alpha",
                            },
                        },
                    ),
                ),
                BrainConversationStep(
                    user_text="Correction: I am a product manager now.",
                    assistant_text="Updated.",
                    assistant_summary="Recorded the corrected role.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_current",
                                "subject_entity_id": "entity:user:alpha",
                                "predicate": "profile.role",
                                "object": {"value": "product manager"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": "alpha",
                                "claim_key": "profile.role:alpha",
                            },
                        },
                        {
                            "event_type": "memory.claim.superseded",
                            "payload": {
                                "prior_claim_id": "claim_role_prior",
                                "new_claim_id": "claim_role_current",
                                "reason": "explicit_correction",
                            },
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                reply_packet_temporal_mode="current_first",
                reply_selected_anchor_types=("claim",),
                reply_selected_temporal_kinds=("current",),
                reply_selected_backing_ids=("claim_role_current",),
                governance_superseded_graph_backing_ids=("claim_role_prior",),
            ),
            {"reply": "What is my role right now?"},
        ),
        (
            "reply_historical_change_query",
            (
                BrainConversationStep(
                    user_text="I used to be a designer.",
                    assistant_text="Noted.",
                    assistant_summary="Recorded the earlier role.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_prior",
                                "subject_entity_id": "entity:user:alpha",
                                "predicate": "profile.role",
                                "object": {"value": "designer"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": "alpha",
                                "claim_key": "profile.role:alpha",
                            },
                        },
                    ),
                ),
                BrainConversationStep(
                    user_text="Correction: I am a product manager now.",
                    assistant_text="Updated.",
                    assistant_summary="Recorded the corrected role.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_current",
                                "subject_entity_id": "entity:user:alpha",
                                "predicate": "profile.role",
                                "object": {"value": "product manager"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": "alpha",
                                "claim_key": "profile.role:alpha",
                            },
                        },
                        {
                            "event_type": "memory.claim.superseded",
                            "payload": {
                                "prior_claim_id": "claim_role_prior",
                                "new_claim_id": "claim_role_current",
                                "reason": "explicit_correction",
                            },
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                reply_packet_temporal_mode="historical_focus",
                reply_selected_anchor_types=("claim",),
                reply_selected_temporal_kinds=("historical",),
                reply_selected_backing_ids=("claim_role_prior",),
                governance_superseded_graph_backing_ids=("claim_role_prior",),
            ),
            {"reply": "How has my role changed before?"},
        ),
        (
            "reply_associative_project_query",
            (
                BrainConversationStep(
                    user_text="We need to keep pushing the Alpha project this week.",
                    assistant_text="Agreed.",
                    assistant_summary="Continued the Alpha project together.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_alpha_priority",
                                "subject_entity_id": "entity:user:alpha",
                                "predicate": "project.priority",
                                "object": {"value": "assistant refresh"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": "alpha",
                                "claim_key": "project.priority:alpha",
                            },
                        },
                        {
                            "event_type": "autobiography.entry.upserted",
                            "payload": {
                                "scope_type": "relationship",
                                "scope_id": "blink/main:alpha",
                                "entry_kind": "relationship_arc",
                                "rendered_summary": "We are pushing the Alpha project together this week.",
                                "content": {
                                    "summary": "We are pushing the Alpha project together this week."
                                },
                                "salience": 1.0,
                                "source_episode_ids": [],
                                "source_claim_ids": ["claim_alpha_priority"],
                                "source_event_ids": ["evt-alpha-priority"],
                            },
                        },
                        {
                            "event_type": "autobiography.entry.upserted",
                            "payload": {
                                "scope_type": "relationship",
                                "scope_id": "blink/main:alpha",
                                "entry_kind": "project_arc",
                                "rendered_summary": "Alpha focuses on the assistant refresh this week.",
                                "content": {
                                    "summary": "Alpha focuses on the assistant refresh this week.",
                                    "project_key": "Alpha",
                                },
                                "salience": 1.0,
                                "source_episode_ids": [],
                                "source_claim_ids": ["claim_alpha_priority"],
                                "source_event_ids": ["evt-alpha-priority"],
                            },
                        },
                    ),
                ),
            ),
            BrainContinuityExpectedState(
                project_dossier_keys=("Alpha",),
                reply_packet_temporal_mode="current_first",
                reply_selected_anchor_types=("dossier",),
                reply_selected_temporal_kinds=("current",),
            ),
            {"reply": "What are we pushing together this week?"},
        ),
    ],
)
def test_continuity_eval_harness_detects_expected_state(
    tmp_path,
    name,
    steps,
    expected_state,
    context_queries,
):
    store = BrainStore(path=tmp_path / f"{name}.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")

    result = BrainContinuityEvalHarness(store=store).run_case(
        BrainContinuityEvalCase(
            name=name,
            description=name.replace("_", " "),
            session_ids=session_ids,
            steps=steps,
            expected_state=expected_state,
            language=Language.EN,
            context_queries=context_queries,
        ),
        output_path=tmp_path / f"{name}.json",
    )

    assert result.matched is True
    assert result.artifact_path is not None
    assert result.artifact_path.exists()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "procedural_skill_candidate_then_active.json",
        "procedural_skill_superseded_by_revised_tail.json",
        "procedural_skill_retired_after_repeated_failures.json",
        "planning_skill_reuse_exact.json",
        "planning_skill_reject_mismatch.json",
        "planning_skill_delta_revise_tail.json",
    ],
)
def test_continuity_eval_harness_replays_procedural_lifecycle_fixtures(tmp_path, fixture_name):
    fixture_case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    store = BrainStore(path=tmp_path / f"{fixture_case.name}.db")

    result = BrainContinuityEvalHarness(store=store).run_case(
        BrainContinuityEvalCase(
            name=fixture_case.name,
            description=fixture_case.description,
            session_ids=fixture_case.scenario.session_ids,
            steps=(
                BrainConversationStep(
                    user_text="",
                    assistant_text="",
                    assistant_summary="",
                    events=tuple(
                        {
                            "event_id": event.event_id,
                            "event_type": event.event_type,
                            "ts": event.ts,
                            "source": event.source,
                            "correlation_id": event.correlation_id,
                            "causal_parent_id": event.causal_parent_id,
                            "confidence": event.confidence,
                            "payload": event.payload,
                            "tags": event.tags,
                        }
                        for event in fixture_case.scenario.events
                    ),
                ),
            ),
            expected_state=fixture_case.expected_state,
            presence_scope_key=fixture_case.presence_scope_key or "browser:presence",
            language=fixture_case.language,
            context_queries=fixture_case.context_queries,
        ),
        output_path=tmp_path / f"{fixture_case.name}.json",
    )

    assert result.matched is True


def test_continuity_eval_harness_tracks_generalized_packet_expectations(tmp_path):
    store = BrainStore(path=tmp_path / "packet-expectations.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="packet-eval")

    result = BrainContinuityEvalHarness(store=store).run_case(
        BrainContinuityEvalCase(
            name="generalized_packet_expectations",
            description="Generalized packet expectations stay available for all tasks.",
            session_ids=session_ids,
            steps=(
                BrainConversationStep(
                    user_text="I used to be a designer.",
                    assistant_text="Noted.",
                    assistant_summary="Recorded the earlier role.",
                    events=(
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_prior",
                                "subject_entity_id": "entity:user:packet-eval",
                                "predicate": "profile.role",
                                "object": {"value": "designer"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": session_ids.user_id,
                                "claim_key": "profile.role:packet-eval",
                            },
                        },
                        {
                            "event_type": "memory.claim.recorded",
                            "payload": {
                                "claim_id": "claim_role_current",
                                "subject_entity_id": "entity:user:packet-eval",
                                "predicate": "profile.role",
                                "object": {"value": "product manager"},
                                "status": "active",
                                "scope_type": "user",
                                "scope_id": session_ids.user_id,
                                "claim_key": "profile.role:packet-eval",
                            },
                        },
                        {
                            "event_type": "memory.claim.superseded",
                            "payload": {
                                "prior_claim_id": "claim_role_prior",
                                "new_claim_id": "claim_role_current",
                                "reason": "explicit_correction",
                            },
                        },
                    ),
                ),
            ),
            expected_state=BrainContinuityExpectedState(
                packet_temporal_modes={
                    "reply": "current_first",
                    "recall": "current_first",
                    "reflection": "current_first",
                    "critique": "current_first",
                },
                packet_selected_anchor_types={
                    "reply": ("claim",),
                    "recall": ("claim",),
                },
                packet_selected_backing_ids={
                    "reply": ("claim_role_current",),
                    "recall": ("claim_role_current",),
                },
            ),
            language=Language.EN,
            context_queries={
                "reply": "What is my role right now?",
                "recall": "Recall my current role.",
                "reflection": "Reflect on my current role state.",
                "critique": "Critique the current role summary.",
            },
        ),
        output_path=tmp_path / "packet-expectations.json",
    )

    assert result.matched is True

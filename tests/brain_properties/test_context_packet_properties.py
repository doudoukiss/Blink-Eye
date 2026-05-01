from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.context import (
    BrainContextBudgetProfile,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.memory_v2 import (
    BrainContinuityDossierFreshness,
    BrainContinuityDossierKind,
    BrainGovernanceReasonCode,
    ClaimLedger,
)
from blink.transcriptions.language import Language
from tests.brain_properties._continuity_context_property_helpers import (
    ContinuityScenarioSpec,
    build_context_surface,
    build_continuity_bundle,
    collect_reachable_source_ids,
    dossier_by_kind,
    enrich_surface_for_packet_tests,
    packet_budget_scenario_strategy,
    packet_required_sections_strategy,
    planning_budget_profile,
    reply_budget_profile,
    seed_packet_state,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _base_packet_bundle(spec) -> object:
    bundle = build_continuity_bundle(
        ContinuityScenarioSpec(
            classification="fresh",
            include_project_arc=spec.include_project_arc,
            include_project_recent_change=spec.include_project_recent_change,
            include_relationship_milestone=True,
            project_key=spec.project_key,
        )
    )
    seed_packet_state(bundle, spec)
    return bundle


def _compile_packet(bundle, *, query_text: str, task: BrainContextTask, budget):
    surface = enrich_surface_for_packet_tests(
        build_context_surface(bundle, latest_user_text=query_text)
    )
    packet = compile_context_packet_from_surface(
        snapshot=surface,
        latest_user_text=query_text,
        task=task,
        language=Language.EN,
        budget_profile=budget,
    )
    return surface, packet


@given(spec=packet_budget_scenario_strategy())
@_SETTINGS
def test_compiled_packets_stay_within_explicit_budgets(spec):
    # Compiled packets must respect explicit static and dynamic budget ceilings.
    bundle = _base_packet_bundle(spec)
    try:
        reply_surface, reply_packet = _compile_packet(
            bundle,
            query_text="What is still active right now?",
            task=BrainContextTask.REPLY,
            budget=reply_budget_profile(spec.reply_budget),
        )
        planning_surface, planning_packet = _compile_packet(
            bundle,
            query_text="Plan the next maintenance step.",
            task=BrainContextTask.PLANNING,
            budget=planning_budget_profile(spec.planning_budget),
        )

        for surface, packet, budget in (
            (reply_surface, reply_packet, reply_budget_profile(spec.reply_budget)),
            (planning_surface, planning_packet, planning_budget_profile(spec.planning_budget)),
        ):
            _ = surface
            assert packet.packet_trace is not None
            assert packet.selected_context.estimated_tokens <= budget.max_tokens
            assert (
                packet.packet_trace.final_selected_tokens
                <= packet.packet_trace.dynamic_token_budget
            )

            dynamic_sections = [
                section
                for section in packet.selected_context.sections
                if section.source == "active_context"
            ]
            assert {section.key for section in dynamic_sections} == {
                item.section_key for item in packet.packet_trace.selected_items
            }
            for section in dynamic_sections:
                assert section.content == "\n".join(
                    item.content
                    for item in packet.packet_trace.selected_items
                    if item.section_key == section.key
                )
            assert any(
                item.reason == "dynamic_budget_exceeded"
                for item in packet.packet_trace.dropped_items
            )
            assert packet.packet_trace.section_decisions
            assert all(
                item.decision_reason_codes
                for item in packet.packet_trace.selected_items + packet.packet_trace.dropped_items
            )
    finally:
        bundle.close()


@given(spec=packet_required_sections_strategy())
@_SETTINGS
def test_packet_trace_refs_point_to_reachable_source_state(spec):
    # Packet anchors, provenance, and support traces must point back to reachable source state.
    bundle = _base_packet_bundle(spec)
    try:
        packets = [
            _compile_packet(
                bundle,
                query_text="What is my role right now?",
                task=BrainContextTask.REPLY,
                budget=reply_budget_profile(spec.reply_budget),
            ),
            _compile_packet(
                bundle,
                query_text="How has my role changed before?",
                task=BrainContextTask.REPLY,
                budget=reply_budget_profile(spec.reply_budget),
            ),
            _compile_packet(
                bundle,
                query_text="Plan the next maintenance step.",
                task=BrainContextTask.PLANNING,
                budget=planning_budget_profile(spec.planning_budget),
            ),
        ]

        for surface, packet in packets:
            assert packet.packet_trace is not None
            reachable = collect_reachable_source_ids(bundle, snapshot=surface)
            edges_by_id = {edge.edge_id: edge for edge in surface.continuity_graph.edges}

            for anchor in packet.packet_trace.selected_anchors:
                assert set(anchor.seed_node_ids) <= reachable.graph_node_ids
            for expansion in packet.packet_trace.expansions:
                edge = edges_by_id[expansion.edge_id]
                assert {expansion.from_node_id, expansion.to_node_id} == {
                    edge.from_node_id,
                    edge.to_node_id,
                }
                if expansion.accepted:
                    assert expansion.to_node_id in reachable.graph_node_ids

            for item in packet.packet_trace.selected_items:
                provenance = item.provenance
                if item.item_type in {"dossier", "dossier_change"}:
                    assert provenance.get("dossier_id") in reachable.dossier_ids
                    assert set(provenance.get("claim_ids", [])) <= reachable.claim_ids
                    assert set(provenance.get("entry_ids", [])) <= reachable.entry_ids
                    assert set(provenance.get("graph_node_ids", [])) <= reachable.graph_node_ids
                    assert set(provenance.get("graph_edge_ids", [])) <= reachable.graph_edge_ids
                if "node_id" in provenance:
                    assert provenance["node_id"] in reachable.graph_node_ids
                    assert set(provenance.get("support_plan_proposal_ids", [])) <= (
                        reachable.plan_proposal_ids
                    )
                if item.item_type == "procedural_skill":
                    assert provenance.get("skill_id") in reachable.skill_ids
                    assert set(provenance.get("support_trace_ids", [])) <= (
                        reachable.procedural_support_trace_ids
                    )
                if item.item_type == "active_situation_record":
                    assert provenance.get("record_id") in reachable.active_situation_record_ids
                if item.item_type == "private_working_memory_record":
                    assert provenance.get("record_id") in reachable.private_record_ids
                if item.item_type == "uncertainty_record":
                    record_id = str(provenance.get("record_id", "")).strip()
                    assert (
                        record_id in reachable.active_situation_record_ids
                        or record_id in reachable.private_record_ids
                        or record_id == f"scene_world_uncertainty:{surface.scene_world_state.scope_id}"
                    )
                if item.item_type == "scene_world_entity":
                    assert provenance.get("entity_id") in reachable.scene_entity_ids
                if item.item_type == "scene_world_affordance":
                    assert provenance.get("affordance_id") in reachable.scene_affordance_ids
                    assert provenance.get("entity_id") in reachable.scene_entity_ids
    finally:
        bundle.close()


@given(spec=packet_required_sections_strategy())
@_SETTINGS
def test_reply_and_planning_packets_keep_required_bounded_sections(spec):
    # Reply and planning packets must retain their required bounded sections when the source state makes them eligible.
    bundle = _base_packet_bundle(spec)
    try:
        current_surface, current_reply = _compile_packet(
            bundle,
            query_text="What is my role right now?",
            task=BrainContextTask.REPLY,
            budget=reply_budget_profile(spec.reply_budget),
        )
        historical_surface, historical_reply = _compile_packet(
            bundle,
            query_text="How has my role changed before?",
            task=BrainContextTask.REPLY,
            budget=reply_budget_profile(spec.reply_budget),
        )
        planning_surface, planning_packet = _compile_packet(
            bundle,
            query_text="Plan the next maintenance step.",
            task=BrainContextTask.PLANNING,
            budget=planning_budget_profile(spec.planning_budget),
        )

        relationship = dossier_by_kind(
            current_surface.continuity_dossiers,
            BrainContinuityDossierKind.RELATIONSHIP.value,
        )
        assert relationship.freshness == BrainContinuityDossierFreshness.FRESH.value
        assert any(
            decision.section_key == "active_continuity"
            for decision in current_reply.packet_trace.section_decisions
        )
        assert historical_reply.selected_context.section("recent_changes") is not None
        assert current_reply.selected_context.section("planning_anchors") is None
        assert not any(
            item.section_key == "planning_anchors"
            for item in current_reply.packet_trace.selected_items
        )

        assert planning_packet.selected_context.section("commitment_projection") is not None
        assert planning_packet.selected_context.section("planning_anchors") is not None
        assert any(
            decision.section_key == "commitment_projection" and decision.selected
            for decision in planning_packet.selected_context.selection_trace.decisions
        )
        assert planning_surface.commitment_projection.active_commitments
    finally:
        bundle.close()


@given(include_project_arc=st.booleans())
@_SETTINGS
def test_governance_annotations_and_suppressions_stay_traceable_without_packet_bloat(
    include_project_arc,
):
    bundle = build_continuity_bundle(
        ContinuityScenarioSpec(
            classification="fresh",
            include_project_arc=include_project_arc,
            include_project_recent_change=include_project_arc,
            include_relationship_milestone=True,
            project_key="Alpha",
        )
    )
    try:
        event_context = bundle.store._memory_event_context(
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
            agent_id=bundle.session_ids.agent_id,
            session_id=bundle.session_ids.session_id,
            source="property",
            correlation_id="packet-governance",
        )
        held_claim = ClaimLedger(store=bundle.store).request_claim_review(
            bundle.current_claims[0].claim_id,
            source_event_id="evt-held-review",
            reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
            event_context=event_context,
        )
        reply_surface, reply_packet = _compile_packet(
            bundle,
            query_text="What is still active right now?",
            task=BrainContextTask.REPLY,
            budget=reply_budget_profile(180),
        )
        recall_surface, recall_packet = _compile_packet(
            bundle,
            query_text="Recall my role changes.",
            task=BrainContextTask.RECALL,
            budget=BrainContextBudgetProfile(task="recall", max_tokens=260),
        )
        reply_digest = build_context_packet_digest(
            packet_traces={"reply": reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None}
        )["reply"]
        recall_digest = build_context_packet_digest(
            packet_traces={"recall": recall_packet.packet_trace.as_dict() if recall_packet.packet_trace else None}
        )["recall"]

        _ = reply_surface
        _ = recall_surface
        assert any(
            item.reason == "governance_suppressed"
            for item in reply_packet.packet_trace.dropped_items
        )
        assert reply_digest["suppressed_backing_ids"]
        assert reply_digest["governance_drop_reason_counts"]["held_support"] >= 1
        assert reply_digest["governance_drop_reason_counts"]["review_debt"] >= 1
        assert not any(
            item.section_key == "active_continuity"
            and item.item_type == "dossier"
            and held_claim.claim_id in item.provenance.get("claim_ids", [])
            for item in reply_packet.packet_trace.selected_items
        )
        annotated_selected = [
            item
            for item in recall_packet.packet_trace.selected_items
            if item.availability_state == "annotated"
        ]
        assert annotated_selected
        assert recall_digest["annotated_backing_ids"]
        assert set(recall_digest["annotated_backing_ids"]) <= set(
            recall_digest["selected_backing_ids"]
        )
        assert recall_packet.selected_context.estimated_tokens <= 260
    finally:
        bundle.close()


@given(
    include_person=st.booleans(),
    degraded=st.booleans(),
    redact=st.booleans(),
)
@_SETTINGS
def test_scene_episode_packet_policy_stays_bounded_and_traceable(
    include_person,
    degraded,
    redact,
):
    from blink.brain.session import resolve_brain_session_ids
    from blink.brain.store import BrainStore
    from tests.test_brain_memory_v2 import (
        _multimodal_event_context,
        _scene_world_projection_for_multimodal,
        _seed_scene_episode,
        _ts,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        store = BrainStore(path=Path(temp_dir) / "brain.db")
        session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="scene-packet-prop",
        )
        try:
            entry = _seed_scene_episode(
                store,
                session_ids,
                projection=_scene_world_projection_for_multimodal(
                    scope_id="browser:presence",
                    source_event_ids=["evt-scene-policy"],
                    updated_at=_ts(20),
                    include_person=include_person,
                    degraded_mode="limited" if degraded else "healthy",
                ),
                start_second=20,
                include_attention=include_person,
                isolated_recent_events=not include_person,
            )
            assert entry is not None
            if redact:
                entry = store.redact_autobiographical_entry(
                    entry.entry_id,
                    redacted_summary="Redacted scene episode.",
                    source_event_id="evt-scene-policy-redact",
                    reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
                    event_context=_multimodal_event_context(
                        store,
                        session_ids,
                        correlation_id="scene-packet-prop-redact",
                    ),
                )
            surface = build_context_surface(
                bundle=type(
                    "Bundle",
                    (),
                    {
                        "store": store,
                        "session_ids": session_ids,
                        "presence_scope_key": "browser:presence",
                    },
                )(),
                latest_user_text="Audit the current scene episode.",
            )
            reply_packet = compile_context_packet_from_surface(
                snapshot=surface,
                latest_user_text="What is active in the scene right now?",
                task=BrainContextTask.REPLY,
                language=Language.EN,
                budget_profile=reply_budget_profile(220),
            )
            planning_packet = compile_context_packet_from_surface(
                snapshot=surface,
                latest_user_text="Plan the next scene-aware step.",
                task=BrainContextTask.PLANNING,
                language=Language.EN,
                budget_profile=planning_budget_profile(260),
            )
            operator_packet = compile_context_packet_from_surface(
                snapshot=surface,
                latest_user_text="Audit the scene episode policy.",
                task=BrainContextTask.OPERATOR_AUDIT,
                language=Language.EN,
                budget_profile=BrainContextBudgetProfile(task="operator_audit", max_tokens=320),
            )
            reply_digest = build_context_packet_digest(
                packet_traces={
                    "reply": (
                        reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None
                    )
                }
            )["reply"]
            planning_digest = build_context_packet_digest(
                packet_traces={
                    "planning": (
                        planning_packet.packet_trace.as_dict()
                        if planning_packet.packet_trace
                        else None
                    )
                }
            )["planning"]
            operator_digest = build_context_packet_digest(
                packet_traces={
                    "operator_audit": (
                        operator_packet.packet_trace.as_dict()
                        if operator_packet.packet_trace
                        else None
                    )
                }
            )["operator_audit"]

            planning_selected = [
                item
                for item in planning_packet.packet_trace.selected_items
                if item.item_type == "scene_episode"
            ]
            planning_dropped = [
                item
                for item in planning_packet.packet_trace.dropped_items
                if item.item_type == "scene_episode"
            ]

            assert not [
                item
                for item in reply_packet.packet_trace.selected_items
                if item.item_type == "scene_episode"
            ]
            assert reply_digest["scene_episode_trace"]["suppressed_entry_ids"]
            assert len(planning_selected) <= 1
            assert (
                operator_digest["scene_episode_trace"]["selected_entry_ids"]
                or operator_digest["scene_episode_trace"]["suppressed_entry_ids"]
            )
            if include_person or degraded or redact:
                assert not planning_selected
                assert planning_dropped
                expected_reason = (
                    "scene_episode_redacted"
                    if redact
                    else "scene_episode_privacy_gated"
                    if include_person
                    else "scene_episode_review_gated"
                )
                assert (
                    planning_digest["scene_episode_trace"]["drop_reason_counts"][expected_reason]
                    >= 1
                )
            else:
                assert planning_selected
                assert planning_selected[0].provenance.get("entry_id") == entry.entry_id
                assert planning_digest["scene_episode_trace"]["selected_entry_ids"] == [
                    entry.entry_id
                ]
        finally:
            store.close()


@given(spec=packet_required_sections_strategy())
@_SETTINGS
def test_mode_switches_follow_explicit_policy_table(spec):
    # Mode changes must produce deterministic section shifts from the shared policy table.
    bundle = _base_packet_bundle(spec)
    try:
        packets = {
            BrainContextTask.WAKE: _compile_packet(
                bundle,
                query_text="Route the next wake decision.",
                task=BrainContextTask.WAKE,
                budget=BrainContextBudgetProfile(task="wake", max_tokens=spec.planning_budget),
            )[1],
            BrainContextTask.REEVALUATION: _compile_packet(
                bundle,
                query_text="Reevaluate the unresolved work.",
                task=BrainContextTask.REEVALUATION,
                budget=BrainContextBudgetProfile(
                    task="reevaluation",
                    max_tokens=spec.planning_budget,
                ),
            )[1],
            BrainContextTask.OPERATOR_AUDIT: _compile_packet(
                bundle,
                query_text="Audit the continuity state.",
                task=BrainContextTask.OPERATOR_AUDIT,
                budget=BrainContextBudgetProfile(
                    task="operator_audit",
                    max_tokens=spec.planning_budget,
                ),
            )[1],
            BrainContextTask.GOVERNANCE_REVIEW: _compile_packet(
                bundle,
                query_text="Review the continuity governance state.",
                task=BrainContextTask.GOVERNANCE_REVIEW,
                budget=BrainContextBudgetProfile(
                    task="governance_review",
                    max_tokens=spec.planning_budget,
                ),
            )[1],
        }

        assert any(
            decision.section_key == "planning_anchors"
            for decision in packets[BrainContextTask.WAKE].packet_trace.section_decisions
        )
        assert packets[BrainContextTask.WAKE].selected_context.section("relevant_continuity") is None
        assert packets[BrainContextTask.REEVALUATION].selected_context.section("unresolved_state") is not None
        assert any(
            decision.section_key == "relevant_continuity"
            for decision in packets[BrainContextTask.OPERATOR_AUDIT].packet_trace.section_decisions
        )
        assert (
            "planning_anchors"
            not in packets[BrainContextTask.GOVERNANCE_REVIEW].packet_trace.mode_policy.dynamic_section_keys
        )
        assert all(
            packet.packet_trace.mode_policy.task == task
            for task, packet in packets.items()
        )
    finally:
        bundle.close()

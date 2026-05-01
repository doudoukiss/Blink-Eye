from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from blink.brain.context import BrainContextTask, compile_context_packet_from_surface
from blink.transcriptions.language import Language
from tests.brain_properties._active_state_property_helpers import (
    ActiveStateSeedSpec,
    active_state_seed_strategy,
    build_active_state_bundle,
    collect_active_state_reachable_ids,
    context_budget,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@given(spec=active_state_seed_strategy(), base_budget=st.integers(min_value=200, max_value=300))
@_SETTINGS
def test_active_state_packets_stay_bounded_and_keep_provenance_reachable(spec, base_budget):
    # Active-state packet items must stay within budget and point back to reachable state.
    bundle = build_active_state_bundle(spec, client_suffix="packet-bounds")
    try:
        snapshot = bundle.build_surface(latest_user_text="Review the active dock workflow.")
        reachable = collect_active_state_reachable_ids(bundle, snapshot=snapshot)
        budgets = {
            BrainContextTask.REPLY: base_budget,
            BrainContextTask.PLANNING: base_budget + 48,
            BrainContextTask.RECALL: base_budget,
            BrainContextTask.REFLECTION: base_budget + 24,
            BrainContextTask.CRITIQUE: base_budget + 24,
            BrainContextTask.WAKE: base_budget + 24,
            BrainContextTask.REEVALUATION: base_budget + 24,
            BrainContextTask.OPERATOR_AUDIT: base_budget + 24,
            BrainContextTask.GOVERNANCE_REVIEW: base_budget + 24,
        }

        for task, max_tokens in budgets.items():
            packet = compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text=f"Review the {task.value} context.",
                task=task,
                language=Language.EN,
                budget_profile=context_budget(task.value, max_tokens=max_tokens),
            )

            assert packet.packet_trace is not None
            assert packet.packet_trace.mode_policy.task == task
            assert packet.selected_context.estimated_tokens <= max_tokens
            assert (
                packet.packet_trace.final_selected_tokens
                <= packet.packet_trace.dynamic_token_budget
            )
            assert packet.packet_trace.section_decisions
            dynamic_sections = [
                section
                for section in packet.selected_context.sections
                if section.source == "active_context"
            ]
            assert {section.key for section in dynamic_sections} == {
                item.section_key for item in packet.packet_trace.selected_items
            }
            assert all(
                item.decision_reason_codes
                for item in packet.packet_trace.selected_items + packet.packet_trace.dropped_items
            )

            for item in packet.packet_trace.selected_items:
                provenance = item.provenance
                if item.item_type == "active_situation_record":
                    assert provenance.get("record_id") in reachable.active_situation_record_ids
                if item.item_type == "private_working_memory_record":
                    assert provenance.get("record_id") in reachable.private_record_ids
                if item.item_type == "uncertainty_record":
                    record_id = str(provenance.get("record_id", "")).strip()
                    assert (
                        record_id in reachable.active_situation_record_ids
                        or record_id in reachable.private_record_ids
                        or record_id == f"scene_world_uncertainty:{snapshot.scene_world_state.scope_id}"
                    )
                if item.item_type == "scene_world_entity":
                    assert provenance.get("entity_id") in reachable.scene_entity_ids
                if item.item_type == "scene_world_affordance":
                    assert provenance.get("affordance_id") in reachable.scene_affordance_ids
                    assert provenance.get("entity_id") in reachable.scene_entity_ids
    finally:
        bundle.close()


@given(spec=active_state_seed_strategy())
@_SETTINGS
def test_degraded_scene_packets_keep_required_bounded_sections(spec):
    # Degraded-scene packets must keep bounded active-state sections for each task mode.
    assume(spec.degraded_mode != "healthy")
    packet_spec = ActiveStateSeedSpec(
        zone_count=spec.zone_count,
        entity_count=spec.entity_count,
        fresh_for_secs=spec.fresh_for_secs,
        expire_after_secs=spec.expire_after_secs,
        plan_resolution="pending",
        extra_tool_outcomes=spec.extra_tool_outcomes,
        degraded_mode="unavailable",
        include_skill=True,
    )
    bundle = build_active_state_bundle(packet_spec, client_suffix="packet-degraded")
    try:
        snapshot = bundle.build_surface(latest_user_text="Review the degraded dock workflow.")
        packets = {
            BrainContextTask.REPLY: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Reply with the current degraded context.",
                task=BrainContextTask.REPLY,
                language=Language.EN,
                budget_profile=context_budget("reply", max_tokens=240),
            ),
            BrainContextTask.PLANNING: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Plan the next degraded-scene maintenance step.",
                task=BrainContextTask.PLANNING,
                language=Language.EN,
                budget_profile=context_budget("planning", max_tokens=320),
            ),
            BrainContextTask.RECALL: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Recall the recent degraded-scene state.",
                task=BrainContextTask.RECALL,
                language=Language.EN,
                budget_profile=context_budget("recall", max_tokens=260),
            ),
            BrainContextTask.REFLECTION: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Reflect on the degraded-scene state.",
                task=BrainContextTask.REFLECTION,
                language=Language.EN,
                budget_profile=context_budget("reflection", max_tokens=300),
            ),
            BrainContextTask.CRITIQUE: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Critique the degraded-scene plan.",
                task=BrainContextTask.CRITIQUE,
                language=Language.EN,
                budget_profile=context_budget("critique", max_tokens=300),
            ),
            BrainContextTask.WAKE: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Route the degraded-scene wake.",
                task=BrainContextTask.WAKE,
                language=Language.EN,
                budget_profile=context_budget("wake", max_tokens=300),
            ),
            BrainContextTask.REEVALUATION: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Reevaluate the degraded-scene plan.",
                task=BrainContextTask.REEVALUATION,
                language=Language.EN,
                budget_profile=context_budget("reevaluation", max_tokens=300),
            ),
            BrainContextTask.OPERATOR_AUDIT: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Audit the degraded-scene state.",
                task=BrainContextTask.OPERATOR_AUDIT,
                language=Language.EN,
                budget_profile=context_budget("operator_audit", max_tokens=320),
            ),
            BrainContextTask.GOVERNANCE_REVIEW: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text="Review degraded-scene governance.",
                task=BrainContextTask.GOVERNANCE_REVIEW,
                language=Language.EN,
                budget_profile=context_budget("governance_review", max_tokens=320),
            ),
        }

        for task in (
            BrainContextTask.REPLY,
            BrainContextTask.PLANNING,
            BrainContextTask.RECALL,
            BrainContextTask.REFLECTION,
        ):
            packet = packets[task]
            assert packet.selected_context.section("active_state") is not None or (
                packet.selected_context.section("unresolved_state") is not None
            )

        assert packets[BrainContextTask.REPLY].selected_context.section("planning_anchors") is None
        assert packets[BrainContextTask.PLANNING].selected_context.section("commitment_projection") is not None
        assert packets[BrainContextTask.PLANNING].selected_context.section("planning_anchors") is not None
        assert packets[BrainContextTask.RECALL].selected_context.section("active_state") is not None
        assert packets[BrainContextTask.REFLECTION].selected_context.section("active_state") is not None
        assert packets[BrainContextTask.CRITIQUE].selected_context.section("planning_anchors") is not None
        assert packets[BrainContextTask.CRITIQUE].selected_context.section("commitment_projection") is not None
        assert any(
            decision.section_key == "planning_anchors"
            for decision in packets[BrainContextTask.WAKE].packet_trace.section_decisions
        )
        assert packets[BrainContextTask.REEVALUATION].selected_context.section("unresolved_state") is not None
        assert any(
            decision.section_key == "relevant_continuity"
            for decision in packets[BrainContextTask.OPERATOR_AUDIT].packet_trace.section_decisions
        )
        assert (
            "planning_anchors"
            not in packets[BrainContextTask.GOVERNANCE_REVIEW].packet_trace.mode_policy.dynamic_section_keys
        )
        assert any(
            item.section_key == "unresolved_state"
            for task in (
                BrainContextTask.REPLY,
                BrainContextTask.PLANNING,
                BrainContextTask.RECALL,
                BrainContextTask.REFLECTION,
                BrainContextTask.CRITIQUE,
                BrainContextTask.WAKE,
                BrainContextTask.REEVALUATION,
                BrainContextTask.OPERATOR_AUDIT,
                BrainContextTask.GOVERNANCE_REVIEW,
            )
            for item in packets[task].packet_trace.selected_items
        )
    finally:
        bundle.close()

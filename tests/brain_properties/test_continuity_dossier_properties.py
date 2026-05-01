from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.memory_v2 import (
    BrainContinuityDossierAvailability,
    BrainContinuityDossierContradiction,
    BrainContinuityDossierFreshness,
    BrainContinuityDossierKind,
    BrainContinuityGraphNodeKind,
    ClaimLedger,
)
from tests.brain_properties._continuity_context_property_helpers import (
    ContinuityScenarioSpec,
    build_continuity_bundle,
    collect_reachable_source_ids,
    continuity_scenario_strategy,
    dossier_by_kind,
)

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _all_evidence_refs(dossier):
    yield dossier.summary_evidence
    for record in dossier.key_current_facts:
        yield record.evidence
    for record in dossier.recent_changes:
        yield record.evidence
    for record in dossier.open_issues:
        yield record.evidence


def _availability_by_task(dossier) -> dict[str, str]:
    return {
        record.task: record.availability
        for record in dossier.governance.task_availability
    }


@given(spec=continuity_scenario_strategy())
@_SETTINGS
def test_dossiers_reclassify_current_vs_stale_support(spec):
    # Freshness, contradiction, and supersession must reclassify continuity instead of leaving stale state silently current.
    bundle = build_continuity_bundle(spec)
    try:
        relationship = dossier_by_kind(
            bundle.dossiers,
            BrainContinuityDossierKind.RELATIONSHIP.value,
        )
        node_id_by_backing = {
            (node.kind, node.backing_record_id): node.node_id for node in bundle.graph.nodes
        }

        if spec.classification == "fresh":
            assert relationship.freshness == BrainContinuityDossierFreshness.FRESH.value
            assert relationship.contradiction == BrainContinuityDossierContradiction.CLEAR.value
        elif spec.classification == "stale":
            assert relationship.freshness == BrainContinuityDossierFreshness.STALE.value
            assert relationship.dossier_id in bundle.dossiers.stale_dossier_ids
        elif spec.classification == "needs_refresh":
            assert relationship.freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value
            assert relationship.dossier_id in bundle.dossiers.needs_refresh_dossier_ids
        elif spec.classification == "uncertain":
            assert (
                relationship.contradiction
                == BrainContinuityDossierContradiction.UNCERTAIN.value
            )
            assert relationship.dossier_id in bundle.dossiers.uncertain_dossier_ids
        elif spec.classification == "conflict":
            assert (
                relationship.contradiction
                == BrainContinuityDossierContradiction.CONTRADICTED.value
            )
            assert relationship.dossier_id in bundle.dossiers.contradicted_dossier_ids
        else:
            raise AssertionError(f"Unexpected continuity classification: {spec.classification}")

        current_fact_summaries = {record.summary for record in relationship.key_current_facts}
        assert all(record.status != "historical" for record in relationship.key_current_facts)
        for record in relationship.recent_changes:
            if (
                record.status == "historical"
                or record.details.get("recent_change_kind") == "superseded_claim"
            ):
                assert record.summary not in current_fact_summaries

        stale_claim_ids = {
            claim_id
            for record in relationship.key_current_facts
            if record.status == "stale"
            for claim_id in record.evidence.claim_ids
        }
        for claim_id in stale_claim_ids:
            node_id = node_id_by_backing.get(
                (BrainContinuityGraphNodeKind.CLAIM.value, claim_id)
            )
            assert node_id in bundle.graph.stale_node_ids

        for dossier in bundle.dossiers.dossiers:
            for record in dossier.recent_changes:
                if record.details.get("recent_change_kind") == "superseded_claim":
                    matched_superseded_claim = False
                    for claim_id in record.evidence.claim_ids:
                        node_id = node_id_by_backing.get(
                            (BrainContinuityGraphNodeKind.CLAIM.value, claim_id)
                        )
                        if node_id in bundle.graph.superseded_node_ids:
                            matched_superseded_claim = True
                    assert matched_superseded_claim
                if record.status == "historical":
                    for entry_id in record.evidence.entry_ids:
                        node_id = node_id_by_backing.get(
                            (BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value, entry_id)
                        )
                        if node_id is not None:
                            assert node_id in bundle.graph.superseded_node_ids
    finally:
        bundle.close()


def test_expanded_dossiers_keep_graph_and_source_provenance_reachable(tmp_path):
    from blink.brain.session import resolve_brain_session_ids
    from blink.brain.store import BrainStore
    from tests.test_brain_memory_v2 import _build_expanded_dossier_state

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-expanded-prop")
    seeded = _build_expanded_dossier_state(store, session_ids)
    bundle = build_continuity_bundle(
        ContinuityScenarioSpec(
            classification="fresh",
            include_project_arc=False,
            include_project_recent_change=False,
            include_relationship_milestone=False,
            project_key="Alpha",
        )
    )
    try:
        bundle.store.close()
        bundle.store = store
        bundle.session_ids = session_ids
        bundle.current_claims = tuple(
            store.query_claims(
                temporal_mode="current",
                scope_type="user",
                scope_id=session_ids.user_id,
                limit=64,
            )
        )
        bundle.historical_claims = tuple(
            store.query_claims(
                temporal_mode="historical",
                scope_type="user",
                scope_id=session_ids.user_id,
                limit=64,
            )
        )
        bundle.autobiography = tuple(
            store.autobiographical_entries(
                scope_type="relationship",
                scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
                statuses=("current", "superseded"),
                limit=32,
            )
        )
        bundle.graph = seeded["graph"]
        bundle.dossiers = seeded["dossiers"]

        reachable = collect_reachable_source_ids(bundle)
        nodes_by_id = {node.node_id: node for node in bundle.graph.nodes}
        edges_by_id = {edge.edge_id: edge for edge in bundle.graph.edges}

        for dossier in bundle.dossiers.dossiers:
            for evidence in _all_evidence_refs(dossier):
                assert set(evidence.graph_node_ids) <= reachable.graph_node_ids
                assert set(evidence.graph_edge_ids) <= reachable.graph_edge_ids
                for edge_id in evidence.graph_edge_ids:
                    edge = edges_by_id[edge_id]
                    assert edge.from_node_id in reachable.graph_node_ids
                    assert edge.to_node_id in reachable.graph_node_ids
            assert set(dossier.source_block_ids) <= reachable.block_ids
            assert set(dossier.source_commitment_ids) <= reachable.commitment_ids
            assert set(dossier.source_plan_proposal_ids) <= reachable.plan_proposal_ids
            assert set(dossier.source_skill_ids) <= reachable.skill_ids
            assert set(dossier.source_scene_entity_ids) <= reachable.scene_entity_ids
            assert set(dossier.source_scene_affordance_ids) <= reachable.scene_affordance_ids
            for node_id in dossier.summary_evidence.graph_node_ids:
                assert node_id in nodes_by_id

        user = dossier_by_kind(bundle.dossiers, BrainContinuityDossierKind.USER.value)
        self_policy = dossier_by_kind(bundle.dossiers, BrainContinuityDossierKind.SELF_POLICY.value)
        assert any(record.status == "historical" for record in user.recent_changes)
        assert any(record.status == "historical" for record in self_policy.recent_changes)
    finally:
        store.close()
        bundle.tempdir.cleanup()


@given(spec=continuity_scenario_strategy())
@_SETTINGS
def test_dossier_evidence_refs_remain_traceable(spec):
    # Dossier evidence refs must stay structurally valid and trace back to concrete source state.
    bundle = build_continuity_bundle(spec)
    try:
        reachable = collect_reachable_source_ids(bundle)
        claims_by_id = {
            claim.claim_id: claim for claim in (*bundle.current_claims, *bundle.historical_claims)
        }
        entries_by_id = {entry.entry_id: entry for entry in bundle.autobiography}
        nodes_by_id = {node.node_id: node for node in bundle.graph.nodes}
        edges_by_id = {edge.edge_id: edge for edge in bundle.graph.edges}

        for dossier in bundle.dossiers.dossiers:
            for evidence in _all_evidence_refs(dossier):
                assert set(evidence.claim_ids) <= reachable.claim_ids
                assert set(evidence.entry_ids) <= reachable.entry_ids
                assert set(evidence.graph_node_ids) <= reachable.graph_node_ids
                assert set(evidence.graph_edge_ids) <= reachable.graph_edge_ids

                recovered_event_ids: set[str] = set()
                recovered_episode_ids: set[int] = set()

                for claim_id in evidence.claim_ids:
                    claim = claims_by_id[claim_id]
                    if claim.source_event_id:
                        recovered_event_ids.add(claim.source_event_id)
                for entry_id in evidence.entry_ids:
                    entry = entries_by_id.get(entry_id)
                    if entry is not None:
                        recovered_event_ids.update(entry.source_event_ids)
                        recovered_episode_ids.update(int(item) for item in entry.source_episode_ids)
                for node_id in evidence.graph_node_ids:
                    node = nodes_by_id[node_id]
                    recovered_event_ids.update(node.source_event_ids)
                    recovered_episode_ids.update(int(item) for item in node.source_episode_ids)
                for edge_id in evidence.graph_edge_ids:
                    edge = edges_by_id[edge_id]
                    recovered_event_ids.update(edge.source_event_ids)
                    recovered_episode_ids.update(int(item) for item in edge.source_episode_ids)
                    assert edge.from_node_id in reachable.graph_node_ids
                    assert edge.to_node_id in reachable.graph_node_ids
                    if evidence.graph_node_ids:
                        assert (
                            edge.from_node_id in evidence.graph_node_ids
                            or edge.to_node_id in evidence.graph_node_ids
                        )

                assert set(evidence.source_event_ids) <= recovered_event_ids
                assert set(evidence.source_episode_ids) <= recovered_episode_ids
                assert any(
                    (
                        evidence.claim_ids,
                        evidence.entry_ids,
                        evidence.source_event_ids,
                        evidence.source_episode_ids,
                        evidence.graph_node_ids,
                        evidence.graph_edge_ids,
                    )
                )
    finally:
        bundle.close()


@given(spec=continuity_scenario_strategy())
@_SETTINGS
def test_dossier_governance_matches_freshness_and_reply_policy(spec):
    bundle = build_continuity_bundle(spec)
    try:
        relationship = dossier_by_kind(
            bundle.dossiers,
            BrainContinuityDossierKind.RELATIONSHIP.value,
        )
        availability_by_task = _availability_by_task(relationship)
        expected_refresh_cause = {
            BrainContinuityDossierFreshness.FRESH.value: "fresh_current_support",
            BrainContinuityDossierFreshness.NEEDS_REFRESH.value: "newer_support_exists",
            BrainContinuityDossierFreshness.STALE.value: "no_fresh_support",
        }[relationship.freshness]

        assert relationship.governance.last_refresh_cause == expected_refresh_cause
        assert set(availability_by_task) == {
            "reply",
            "planning",
            "recall",
            "reflection",
            "critique",
        }
        if (
            relationship.freshness == BrainContinuityDossierFreshness.FRESH.value
            and relationship.contradiction == BrainContinuityDossierContradiction.CLEAR.value
            and relationship.governance.review_debt_count == 0
        ):
            assert availability_by_task["reply"] == BrainContinuityDossierAvailability.AVAILABLE.value
        else:
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
        for issue in relationship.open_issues:
            if issue.kind in {"review_debt", "held_support"}:
                assert issue.evidence.claim_ids
                assert issue.evidence.graph_node_ids
    finally:
        bundle.close()


@given(include_project_arc=st.booleans())
@_SETTINGS
def test_review_debt_dossiers_keep_traceable_evidence_and_nonreply_availability(
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
            correlation_id="review-debt-dossier",
        )
        ClaimLedger(store=bundle.store).request_claim_review(
            bundle.current_claims[0].claim_id,
            source_event_id="evt-review-debt",
            reason_codes=["requires_confirmation"],
            event_context=event_context,
        )
        graph = bundle.store.build_continuity_graph(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
            scope_type="user",
            scope_id=bundle.session_ids.user_id,
        )
        dossiers = bundle.store.build_continuity_dossiers(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
            scope_type="user",
            scope_id=bundle.session_ids.user_id,
            continuity_graph=graph,
        )
        relationship = dossier_by_kind(
            dossiers,
            BrainContinuityDossierKind.RELATIONSHIP.value,
        )
        availability_by_task = _availability_by_task(relationship)
        issues_by_kind = {issue.kind: issue for issue in relationship.open_issues}

        assert relationship.governance.review_debt_count >= 1
        assert availability_by_task["reply"] == BrainContinuityDossierAvailability.SUPPRESSED.value
        assert (
            availability_by_task["planning"]
            == BrainContinuityDossierAvailability.SUPPRESSED.value
        )
        assert availability_by_task["recall"] == BrainContinuityDossierAvailability.ANNOTATED.value
        assert (
            availability_by_task["reflection"]
            == BrainContinuityDossierAvailability.ANNOTATED.value
        )
        assert issues_by_kind["review_debt"].evidence.claim_ids
        assert issues_by_kind["held_support"].evidence.claim_ids
        assert issues_by_kind["review_debt"].evidence.graph_node_ids
        assert issues_by_kind["held_support"].evidence.graph_node_ids
    finally:
        bundle.close()

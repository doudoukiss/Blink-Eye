from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import strategies as st

from blink.brain.capabilities import CapabilityExecutionResult, CapabilityRegistry
from blink.brain.context import BrainContextBudgetProfile
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, BrainContextSurfaceSnapshot
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.memory_v2 import (
    BrainAutobiographicalEntryRecord,
    BrainClaimRecord,
    BrainContinuityDossierKind,
    BrainContinuityDossierProjection,
    BrainContinuityDossierRecord,
    BrainContinuityGraphProjection,
)
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
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
from blink.brain.session import BrainSessionIds, resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

PRESENCE_SCOPE_KEY = "browser:presence"

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_DEFAULT_BLOCKS = {
    "identity": "Blink identity",
    "policy": "Blink policy",
    "style": "Blink style",
    "action_library": "Blink capabilities",
}
_CONTINUITY_CLASSIFICATIONS = (
    "fresh",
    "stale",
    "needs_refresh",
    "uncertain",
    "conflict",
)
_PROJECT_KEYS = ("Alpha", "Beta", "Gamma")
_SKILL_SEQUENCES = (
    ("maintenance.review_memory_health", "reporting.record_maintenance_note"),
    ("maintenance.review_memory_health", "reporting.record_presence_event"),
)


def _ts(second: int) -> str:
    return (_BASE_TIME + timedelta(seconds=second)).isoformat()


def _client_id(prefix: str, suffix: str) -> str:
    normalized = "".join(character for character in suffix.lower() if character.isalnum()) or "case"
    return f"{prefix}-{normalized}"


@dataclass(frozen=True)
class ContinuityScenarioSpec:
    classification: str
    include_project_arc: bool
    include_project_recent_change: bool
    include_relationship_milestone: bool
    project_key: str


@dataclass(frozen=True)
class PacketScenarioSpec:
    include_project_arc: bool
    include_project_recent_change: bool
    project_key: str
    extra_current_claims: int
    skill_family_count: int
    reply_budget: int
    planning_budget: int


@dataclass
class ContinuityBundle:
    tempdir: TemporaryDirectory[str]
    store: BrainStore
    session_ids: BrainSessionIds
    relationship_scope_id: str
    current_claims: tuple[BrainClaimRecord, ...]
    historical_claims: tuple[BrainClaimRecord, ...]
    autobiography: tuple[BrainAutobiographicalEntryRecord, ...]
    graph: BrainContinuityGraphProjection
    dossiers: BrainContinuityDossierProjection

    def close(self) -> None:
        self.store.close()
        self.tempdir.cleanup()


@dataclass(frozen=True)
class ReachableSourceIds:
    dossier_ids: frozenset[str]
    claim_ids: frozenset[str]
    entry_ids: frozenset[str]
    block_ids: frozenset[str]
    graph_node_ids: frozenset[str]
    graph_edge_ids: frozenset[str]
    event_ids: frozenset[str]
    episode_ids: frozenset[int]
    commitment_ids: frozenset[str]
    plan_proposal_ids: frozenset[str]
    skill_ids: frozenset[str]
    procedural_support_trace_ids: frozenset[str]
    active_situation_record_ids: frozenset[str]
    private_record_ids: frozenset[str]
    scene_entity_ids: frozenset[str]
    scene_affordance_ids: frozenset[str]


@st.composite
def continuity_scenario_strategy(draw) -> ContinuityScenarioSpec:
    classification = draw(st.sampled_from(_CONTINUITY_CLASSIFICATIONS))
    include_project_arc = draw(st.booleans())
    return ContinuityScenarioSpec(
        classification=classification,
        include_project_arc=include_project_arc,
        include_project_recent_change=draw(st.booleans()) if include_project_arc else False,
        include_relationship_milestone=draw(st.booleans())
        if classification in {"fresh", "needs_refresh"}
        else False,
        project_key=draw(st.sampled_from(_PROJECT_KEYS)),
    )


@st.composite
def packet_budget_scenario_strategy(draw) -> PacketScenarioSpec:
    include_project_arc = draw(st.booleans())
    return PacketScenarioSpec(
        include_project_arc=include_project_arc,
        include_project_recent_change=draw(st.booleans()) if include_project_arc else False,
        project_key=draw(st.sampled_from(_PROJECT_KEYS)),
        extra_current_claims=draw(st.integers(min_value=6, max_value=8)),
        skill_family_count=draw(st.integers(min_value=0, max_value=2)),
        reply_budget=draw(st.integers(min_value=36, max_value=52)),
        planning_budget=draw(st.integers(min_value=54, max_value=72)),
    )


@st.composite
def packet_required_sections_strategy(draw) -> PacketScenarioSpec:
    include_project_arc = draw(st.booleans())
    return PacketScenarioSpec(
        include_project_arc=include_project_arc,
        include_project_recent_change=draw(st.booleans()) if include_project_arc else False,
        project_key=draw(st.sampled_from(_PROJECT_KEYS)),
        extra_current_claims=draw(st.integers(min_value=3, max_value=5)),
        skill_family_count=draw(st.integers(min_value=1, max_value=2)),
        reply_budget=draw(st.integers(min_value=120, max_value=180)),
        planning_budget=draw(st.integers(min_value=260, max_value=340)),
    )


def reply_budget_profile(max_tokens: int) -> BrainContextBudgetProfile:
    return BrainContextBudgetProfile(task="reply", max_tokens=max_tokens)


def planning_budget_profile(max_tokens: int) -> BrainContextBudgetProfile:
    return BrainContextBudgetProfile(task="planning", max_tokens=max_tokens)


def dossier_by_kind(
    projection: BrainContinuityDossierProjection,
    kind: str,
    *,
    project_key: str | None = None,
) -> BrainContinuityDossierRecord:
    for dossier in projection.dossiers:
        if dossier.kind != kind:
            continue
        if project_key is not None and dossier.project_key != project_key:
            continue
        return dossier
    raise AssertionError(f"Missing dossier kind={kind} project_key={project_key}")


def build_continuity_bundle(spec: ContinuityScenarioSpec) -> ContinuityBundle:
    tempdir = TemporaryDirectory()
    store = BrainStore(path=Path(tempdir.name) / "brain.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id=_client_id("continuity", f"{spec.classification}-{spec.project_key}"),
    )
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
    store.ensure_default_blocks(_DEFAULT_BLOCKS)

    if spec.classification == "fresh":
        _seed_fresh_relationship_state(
            store,
            session_ids,
            relationship_scope_id=relationship_scope_id,
            include_relationship_milestone=spec.include_relationship_milestone,
        )
    elif spec.classification == "stale":
        _seed_stale_relationship_state(store, session_ids)
    elif spec.classification == "needs_refresh":
        _seed_needs_refresh_relationship_state(
            store,
            session_ids,
            relationship_scope_id=relationship_scope_id,
            include_relationship_milestone=spec.include_relationship_milestone,
        )
    elif spec.classification == "uncertain":
        _seed_uncertain_relationship_state(store, session_ids)
    elif spec.classification == "conflict":
        _seed_conflicting_relationship_state(store, session_ids)
    else:
        raise AssertionError(f"Unsupported continuity classification: {spec.classification}")

    if spec.include_project_arc:
        _seed_project_arc(
            store,
            relationship_scope_id=relationship_scope_id,
            project_key=spec.project_key,
            include_recent_change=spec.include_project_recent_change,
        )

    current_claims = tuple(
        store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=24,
        )
    )
    historical_claims = tuple(
        store.query_claims(
            temporal_mode="historical",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=24,
        )
    )
    autobiography = tuple(
        store.autobiographical_entries(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            entry_kinds=(
                "shared_history_summary",
                "relationship_arc",
                "relationship_milestone",
                "project_arc",
            ),
            limit=24,
        )
    )
    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=graph,
    )
    return ContinuityBundle(
        tempdir=tempdir,
        store=store,
        session_ids=session_ids,
        relationship_scope_id=relationship_scope_id,
        current_claims=current_claims,
        historical_claims=historical_claims,
        autobiography=autobiography,
        graph=graph,
        dossiers=dossiers,
    )


def seed_packet_state(bundle: ContinuityBundle, spec: PacketScenarioSpec) -> None:
    for index in range(spec.extra_current_claims):
        bundle.store.remember_fact(
            user_id=bundle.session_ids.user_id,
            namespace=f"profile.topic_{index}",
            subject="user",
            value={"value": f"topic-{index}"},
            rendered_text=(f"user topic {index} remains active for the ongoing operating context"),
            confidence=0.9,
            singleton=False,
            source_event_id=f"evt-topic-{index}",
            source_episode_id=None,
            provenance={"source": "property"},
            agent_id=bundle.session_ids.agent_id,
            session_id=bundle.session_ids.session_id,
            thread_id=bundle.session_ids.thread_id,
        )

    for family_index in range(spec.skill_family_count):
        sequence = list(_SKILL_SEQUENCES[family_index])
        for repetition in range(2):
            offset = 40 * ((family_index * 2) + repetition)
            _append_completed_procedural_goal(
                bundle.store,
                bundle.session_ids,
                goal_id=f"seed-goal-{family_index}-{repetition}",
                commitment_id=f"seed-commitment-{family_index}-{repetition}",
                goal_title=f"Seed skill {family_index} repetition {repetition}",
                proposal_id=f"seed-proposal-{family_index}-{repetition}",
                sequence=sequence,
                start_second=offset,
            )

    if spec.skill_family_count:
        bundle.store.consolidate_procedural_skills(
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
            scope_type="thread",
            scope_id=bundle.session_ids.thread_id,
        )

    executive = BrainExecutive(
        store=bundle.store,
        session_resolver=lambda: bundle.session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=bundle.store,
            session_resolver=lambda: bundle.session_ids,
            presence_scope_key=PRESENCE_SCOPE_KEY,
            language=Language.EN,
        ),
    )
    executive.create_commitment_goal(
        title="Plan the next maintenance step",
        intent="maintenance.review",
        source="property",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        details={"survive_restart": True},
    )
    current_role_claim = _current_role_claim(bundle.store, bundle.session_ids)
    bundle.store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=bundle.relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="Your current role and the broader operating context are aligned.",
        content={"summary": "Packet-state refresh summary"},
        salience=1.3,
        source_claim_ids=[current_role_claim.claim_id],
        source_event_ids=["evt-packet-refresh"],
        source_event_id="evt-packet-refresh",
    )


def build_context_surface(
    bundle: ContinuityBundle,
    *,
    latest_user_text: str,
) -> BrainContextSurfaceSnapshot:
    return BrainContextSurfaceBuilder(
        store=bundle.store,
        session_resolver=lambda: bundle.session_ids,
        presence_scope_key=PRESENCE_SCOPE_KEY,
        language=Language.EN,
    ).build(
        latest_user_text=latest_user_text,
        include_historical_claims=True,
    )


def enrich_surface_for_packet_tests(
    surface: BrainContextSurfaceSnapshot,
) -> BrainContextSurfaceSnapshot:
    return replace(
        surface,
        private_working_memory=BrainPrivateWorkingMemoryProjection(
            scope_type="thread",
            scope_id=surface.private_working_memory.scope_id,
            records=[
                BrainPrivateWorkingMemoryRecord(
                    record_id="pwm-plan",
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                    summary="Need confirmation before the current plan can proceed.",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                    backing_ids=["proposal-1"],
                    source_event_ids=["evt-plan-1"],
                    goal_id="goal-1",
                    commitment_id="commitment-1",
                    plan_proposal_id="proposal-1",
                    observed_at=_ts(10),
                    updated_at=_ts(10),
                ),
                BrainPrivateWorkingMemoryRecord(
                    record_id="pwm-uncertainty",
                    buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                    summary="The scene feed is degraded and needs refresh.",
                    state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                    evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                    backing_ids=["scene-world-1"],
                    source_event_ids=["evt-scene-1"],
                    commitment_id="commitment-1",
                    observed_at=_ts(11),
                    updated_at=_ts(11),
                ),
            ],
            updated_at=_ts(12),
        ),
        scene_world_state=BrainSceneWorldProjection(
            scope_type="presence",
            scope_id=PRESENCE_SCOPE_KEY,
            entities=[
                BrainSceneWorldEntityRecord(
                    entity_id="entity-1",
                    entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                    canonical_label="dock_pad",
                    summary="A dock pad is visible on the desk.",
                    state=BrainSceneWorldRecordState.ACTIVE.value,
                    evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                    zone_id="zone:desk",
                    confidence=0.8,
                    freshness="current",
                    affordance_ids=["aff-1"],
                    backing_ids=["entity:dock_pad"],
                    source_event_ids=["evt-scene-1"],
                    observed_at=_ts(12),
                    updated_at=_ts(12),
                    expires_at=_ts(40),
                )
            ],
            affordances=[
                BrainSceneWorldAffordanceRecord(
                    affordance_id="aff-1",
                    entity_id="entity-1",
                    capability_family="vision.inspect",
                    summary="The dock pad can be visually inspected.",
                    availability=BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                    confidence=0.75,
                    freshness="current",
                    backing_ids=["entity:dock_pad", "vision.inspect"],
                    source_event_ids=["evt-scene-1"],
                    observed_at=_ts(12),
                    updated_at=_ts(12),
                    expires_at=_ts(40),
                )
            ],
            degraded_mode="limited",
            degraded_reason_codes=["camera_frame_stale"],
            updated_at=_ts(12),
        ),
        active_situation_model=BrainActiveSituationProjection(
            scope_type="thread",
            scope_id=surface.active_situation_model.scope_id,
            records=[
                BrainActiveSituationRecord(
                    record_id="situation-plan",
                    record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                    summary="The current dock review plan is active.",
                    state=BrainActiveSituationRecordState.ACTIVE.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    confidence=0.77,
                    freshness="current",
                    backing_ids=["proposal-1"],
                    source_event_ids=["evt-plan-1"],
                    goal_id="goal-1",
                    commitment_id="commitment-1",
                    plan_proposal_id="proposal-1",
                    observed_at=_ts(10),
                    updated_at=_ts(12),
                ),
                BrainActiveSituationRecord(
                    record_id="situation-uncertainty",
                    record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                    summary="Perception is degraded; treat world-state cautiously.",
                    state=BrainActiveSituationRecordState.UNRESOLVED.value,
                    evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                    confidence=0.5,
                    freshness="limited",
                    uncertainty_codes=["scene_stale", "camera_frame_stale"],
                    private_record_ids=["pwm-uncertainty"],
                    backing_ids=["scene-world-1"],
                    source_event_ids=["evt-scene-1"],
                    commitment_id="commitment-1",
                    observed_at=_ts(11),
                    updated_at=_ts(12),
                ),
            ],
            updated_at=_ts(12),
        ),
    )


def collect_reachable_source_ids(
    bundle: ContinuityBundle,
    *,
    snapshot: BrainContextSurfaceSnapshot | None = None,
) -> ReachableSourceIds:
    graph = snapshot.continuity_graph if snapshot is not None else bundle.graph
    dossiers = snapshot.continuity_dossiers if snapshot is not None else bundle.dossiers
    current_claims = snapshot.current_claims if snapshot is not None else bundle.current_claims
    historical_claims = (
        snapshot.historical_claims if snapshot is not None else bundle.historical_claims
    )
    autobiography = snapshot.autobiography if snapshot is not None else bundle.autobiography
    commitment_projection = (
        snapshot.commitment_projection
        if snapshot is not None
        else bundle.store.get_session_commitment_projection(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
        )
    )
    procedural_skills = snapshot.procedural_skills if snapshot is not None else None

    event_ids = {
        event_id
        for claim in (*current_claims, *historical_claims)
        if (event_id := str(claim.source_event_id or "").strip())
    }
    block_ids = set()
    episode_ids = set()
    entry_ids = set()
    for entry in autobiography:
        entry_ids.add(entry.entry_id)
        event_ids.update(str(item).strip() for item in entry.source_event_ids if str(item).strip())
        episode_ids.update(int(item) for item in entry.source_episode_ids if item not in (None, ""))

    graph_node_ids: set[str] = set()
    graph_edge_ids: set[str] = set()
    plan_proposal_ids: set[str] = set()
    skill_ids: set[str] = set()
    scene_entity_ids: set[str] = set()
    scene_affordance_ids: set[str] = set()
    if graph is not None:
        for node in graph.nodes:
            graph_node_ids.add(node.node_id)
            event_ids.update(
                str(item).strip() for item in node.source_event_ids if str(item).strip()
            )
            episode_ids.update(
                int(item) for item in node.source_episode_ids if item not in (None, "")
            )
            if node.kind == "autobiography_entry":
                entry_ids.add(node.backing_record_id)
            if node.kind == "core_memory_block":
                block_ids.add(node.backing_record_id)
            if node.kind == "plan_proposal":
                plan_proposal_ids.add(node.backing_record_id)
            if node.kind == "procedural_skill":
                skill_ids.add(node.backing_record_id)
            if node.kind == "scene_world_entity":
                scene_entity_ids.add(node.backing_record_id)
            if node.kind == "scene_world_affordance":
                scene_affordance_ids.add(node.backing_record_id)
        for edge in graph.edges:
            graph_edge_ids.add(edge.edge_id)
            event_ids.update(
                str(item).strip() for item in edge.source_event_ids if str(item).strip()
            )
            episode_ids.update(
                int(item) for item in edge.source_episode_ids if item not in (None, "")
            )

    commitment_ids = set()
    for record in (
        list(commitment_projection.active_commitments)
        + list(commitment_projection.blocked_commitments)
        + list(commitment_projection.deferred_commitments)
        + list(commitment_projection.recent_terminal_commitments)
    ):
        commitment_ids.add(record.commitment_id)
        for key in ("current_plan_proposal_id", "pending_plan_proposal_id"):
            plan_proposal_id = str(record.details.get(key, "")).strip()
            if plan_proposal_id:
                plan_proposal_ids.add(plan_proposal_id)

    procedural_support_trace_ids = set()
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            skill_ids.add(skill.skill_id)
            procedural_support_trace_ids.update(skill.supporting_trace_ids)

    dossier_ids = set()
    if dossiers is not None:
        for dossier in dossiers.dossiers:
            dossier_ids.add(dossier.dossier_id)
            entry_ids.update(dossier.source_entry_ids)
            block_ids.update(dossier.source_block_ids)
            commitment_ids.update(dossier.source_commitment_ids)
            plan_proposal_ids.update(dossier.source_plan_proposal_ids)
            skill_ids.update(dossier.source_skill_ids)
            scene_entity_ids.update(dossier.source_scene_entity_ids)
            scene_affordance_ids.update(dossier.source_scene_affordance_ids)
            event_ids.update(dossier.source_event_ids)
            episode_ids.update(int(item) for item in dossier.source_episode_ids)

    active_situation_record_ids = set()
    private_record_ids = set()
    if snapshot is not None:
        active_situation_record_ids.update(
            record.record_id for record in snapshot.active_situation_model.records
        )
        private_record_ids.update(
            record.record_id for record in snapshot.private_working_memory.records
        )
        scene_entity_ids.update(record.entity_id for record in snapshot.scene_world_state.entities)
        scene_affordance_ids.update(
            record.affordance_id for record in snapshot.scene_world_state.affordances
        )
        for record in snapshot.active_situation_model.records:
            event_ids.update(
                str(item).strip() for item in record.source_event_ids if str(item).strip()
            )
        for record in snapshot.private_working_memory.records:
            event_ids.update(
                str(item).strip() for item in record.source_event_ids if str(item).strip()
            )
        for record in snapshot.scene_world_state.entities:
            event_ids.update(
                str(item).strip() for item in record.source_event_ids if str(item).strip()
            )
        for record in snapshot.scene_world_state.affordances:
            event_ids.update(
                str(item).strip() for item in record.source_event_ids if str(item).strip()
            )

    return ReachableSourceIds(
        dossier_ids=frozenset(dossier_ids),
        claim_ids=frozenset(claim.claim_id for claim in (*current_claims, *historical_claims)),
        entry_ids=frozenset(entry_ids),
        block_ids=frozenset(block_ids),
        graph_node_ids=frozenset(graph_node_ids),
        graph_edge_ids=frozenset(graph_edge_ids),
        event_ids=frozenset(event_ids),
        episode_ids=frozenset(episode_ids),
        commitment_ids=frozenset(commitment_ids),
        plan_proposal_ids=frozenset(plan_proposal_ids),
        skill_ids=frozenset(skill_ids),
        procedural_support_trace_ids=frozenset(procedural_support_trace_ids),
        active_situation_record_ids=frozenset(active_situation_record_ids),
        private_record_ids=frozenset(private_record_ids),
        scene_entity_ids=frozenset(scene_entity_ids),
        scene_affordance_ids=frozenset(scene_affordance_ids),
    )


def _seed_fresh_relationship_state(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    relationship_scope_id: str,
    include_relationship_milestone: bool,
) -> None:
    _remember_role_fact(store, session_ids, value="designer", source_event_id="evt-role-1")
    _remember_role_fact(store, session_ids, value="product manager", source_event_id="evt-role-2")
    current_claim = _current_role_claim(store, session_ids)
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="Your role is now product manager and the thread reflects it.",
        content={"summary": "Current role is aligned."},
        salience=1.2,
        source_claim_ids=[current_claim.claim_id],
        source_event_ids=["evt-role-2"],
        source_event_id="evt-relationship-2",
    )
    if include_relationship_milestone:
        store.upsert_autobiographical_entry(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            entry_kind="relationship_milestone",
            rendered_summary="We corrected the role and kept the thread aligned.",
            content={"summary": "Role correction milestone"},
            salience=1.0,
            source_claim_ids=[current_claim.claim_id],
            source_event_ids=["evt-role-2"],
            source_event_id="evt-milestone-1",
            append_only=True,
            identity_key="role-milestone",
        )


def _seed_stale_relationship_state(store: BrainStore, session_ids: BrainSessionIds) -> None:
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    claim = store._claims().record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        status="active",
        confidence=0.82,
        valid_from="2026-01-01T00:00:00+00:00",
        source_event_id="evt-stale-claim",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:stale",
        event_context=None,
    )
    store._claims().expire_claim(
        claim.claim_id,
        source_event_id="evt-stale-governance",
        reason_codes=["stale_without_refresh"],
        event_context=None,
    )


def _seed_needs_refresh_relationship_state(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    relationship_scope_id: str,
    include_relationship_milestone: bool,
) -> None:
    _remember_role_fact(store, session_ids, value="designer", source_event_id="evt-role-1")
    prior_current_claim = _current_role_claim(store, session_ids)
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="We knew your role as designer.",
        content={"summary": "Relationship summary is behind the latest facts."},
        salience=1.0,
        source_claim_ids=[prior_current_claim.claim_id],
        source_event_ids=["evt-role-1"],
        source_event_id="evt-relationship-1",
    )
    _remember_role_fact(store, session_ids, value="maintainer", source_event_id="evt-role-2")
    current_claim = _current_role_claim(store, session_ids)
    if include_relationship_milestone:
        store.upsert_autobiographical_entry(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            entry_kind="relationship_milestone",
            rendered_summary="We learned that your role is now maintainer.",
            content={"summary": "Role changed after the summary anchor"},
            salience=0.95,
            source_claim_ids=[current_claim.claim_id],
            source_event_ids=["evt-role-2"],
            source_event_id="evt-milestone-2",
            append_only=True,
            identity_key="role-refresh",
        )


def _seed_uncertain_relationship_state(store: BrainStore, session_ids: BrainSessionIds) -> None:
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    store._claims().record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "writer"},
        status="uncertain",
        confidence=0.56,
        valid_from="2026-04-19T00:00:00+00:00",
        source_event_id="evt-uncertain-claim",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:uncertain",
        event_context=None,
    )


def _seed_conflicting_relationship_state(store: BrainStore, session_ids: BrainSessionIds) -> None:
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    store._claims().record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        status="active",
        confidence=0.8,
        valid_from="2026-04-19T00:00:00+00:00",
        source_event_id="evt-conflict-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:conflict",
        event_context=None,
    )
    store._claims().record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "manager"},
        status="active",
        confidence=0.82,
        valid_from="2026-04-19T00:01:00+00:00",
        source_event_id="evt-conflict-2",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:conflict",
        event_context=None,
    )


def _seed_project_arc(
    store: BrainStore,
    *,
    relationship_scope_id: str,
    project_key: str,
    include_recent_change: bool,
) -> None:
    if include_recent_change:
        store.upsert_autobiographical_entry(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            entry_kind="project_arc",
            rendered_summary=f"{project_key} started as an exploratory maintenance project.",
            content={"summary": "Project started", "project_key": project_key},
            salience=0.8,
            source_event_ids=["evt-project-1"],
            source_event_id="evt-project-1",
            append_only=True,
            identity_key=project_key,
        )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="project_arc",
        rendered_summary=f"{project_key} is now a reliable maintenance collaboration.",
        content={"summary": "Project is reliable", "project_key": project_key},
        salience=1.0,
        source_event_ids=["evt-project-2"],
        source_event_id="evt-project-2",
        append_only=True,
        identity_key=project_key,
    )


def _remember_role_fact(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    value: str,
    source_event_id: str,
) -> None:
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": value},
        rendered_text=f"user role is {value}",
        confidence=0.9,
        singleton=True,
        source_event_id=source_event_id,
        source_episode_id=None,
        provenance={"source": "property"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )


def _current_role_claim(store: BrainStore, session_ids: BrainSessionIds) -> BrainClaimRecord:
    for claim in store.query_claims(
        temporal_mode="current",
        scope_type="user",
        scope_id=session_ids.user_id,
        predicate="profile.role",
        limit=8,
    ):
        return claim
    raise AssertionError("Expected a current profile.role claim")


def _append_completed_procedural_goal(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
) -> None:
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=goal_title,
        intent="maintenance.review",
        source="property",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    proposal = BrainPlanProposal(
        plan_proposal_id=proposal_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=f"Execute {goal_title}.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        created_at=_ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    adopted_event = store.append_planning_adopted(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary=f"Adopt {goal_title}.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        correlation_id=goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=_ts(start_second + 1),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="property",
                goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
                commitment_id=commitment_id,
                status="open",
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=1,
                last_summary=f"Adopt {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=_ts(start_second + 2),
    )
    completed_steps: list[BrainGoalStep] = []
    current_second = start_second + 3
    for step_index, capability_id in enumerate(sequence):
        request_event = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=goal_id,
            ts=_ts(current_second),
        )
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_COMPLETED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "step_index": step_index,
                "result": CapabilityExecutionResult.success(
                    capability_id=capability_id,
                    summary=f"Completed {capability_id}.",
                    output={"slot": step_index},
                ).model_dump(),
            },
            correlation_id=goal_id,
            causal_parent_id=request_event.event_id,
            ts=_ts(current_second + 1),
        )
        completed_steps.append(
            BrainGoalStep(
                capability_id=capability_id,
                status="completed",
                attempts=1,
                summary=f"Completed {capability_id}.",
                output={"slot": step_index},
            )
        )
        current_second += 2
    store.append_brain_event(
        event_type=BrainEventType.GOAL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="property",
                goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
                commitment_id=commitment_id,
                status="completed",
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=1,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "completed"},
        },
        correlation_id=goal_id,
        ts=_ts(current_second),
    )

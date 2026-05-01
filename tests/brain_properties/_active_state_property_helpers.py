from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import strategies as st

from blink.brain.capabilities import CapabilityExecutionResult, CapabilityRegistry
from blink.brain.context import BrainContextBudgetProfile
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, BrainContextSurfaceSnapshot
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import (
    BrainCommitmentProjection,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainPrivateWorkingMemoryProjection,
    BrainSceneWorldProjection,
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


def _ts(second: int) -> str:
    return (_BASE_TIME + timedelta(seconds=second)).isoformat()


def _bounded_client_id(suffix: str) -> str:
    normalized = "".join(character for character in suffix.lower() if character.isalnum()) or "case"
    return f"active-state-{normalized}"


def _variant_text(value: str, *, padded: bool) -> str:
    return f"  {value}  " if padded else value


@dataclass(frozen=True)
class ActiveStateSeedSpec:
    zone_count: int
    entity_count: int
    fresh_for_secs: int
    expire_after_secs: int
    plan_resolution: str
    extra_tool_outcomes: int
    degraded_mode: str
    include_skill: bool


@dataclass
class ActiveStateBundle:
    tempdir: TemporaryDirectory[str]
    store: BrainStore
    session_ids: BrainSessionIds
    goal_id: str
    commitment_id: str
    latest_second: int
    scene_observed_second: int

    def close(self) -> None:
        self.store.close()
        self.tempdir.cleanup()

    def build_scene_world_state(
        self,
        *,
        reference_second: int | None = None,
    ) -> BrainSceneWorldProjection:
        return self.store.build_scene_world_state_projection(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            agent_id=self.session_ids.agent_id,
            presence_scope_key=PRESENCE_SCOPE_KEY,
            reference_ts=_ts(reference_second) if reference_second is not None else None,
        )

    def build_private_working_memory(
        self,
        *,
        reference_second: int | None = None,
    ) -> BrainPrivateWorkingMemoryProjection:
        return self.store.build_private_working_memory_projection(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            agent_id=self.session_ids.agent_id,
            presence_scope_key=PRESENCE_SCOPE_KEY,
            reference_ts=_ts(reference_second) if reference_second is not None else None,
        )

    def build_active_situation_model(self, *, reference_second: int | None = None):
        return self.store.build_active_situation_model_projection(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            agent_id=self.session_ids.agent_id,
            presence_scope_key=PRESENCE_SCOPE_KEY,
            reference_ts=_ts(reference_second) if reference_second is not None else None,
        )

    def build_surface(self, *, latest_user_text: str) -> BrainContextSurfaceSnapshot:
        return BrainContextSurfaceBuilder(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            presence_scope_key=PRESENCE_SCOPE_KEY,
            language=Language.EN,
            capability_registry=CapabilityRegistry(),
        ).build(
            latest_user_text=latest_user_text,
            include_historical_claims=True,
        )


@dataclass(frozen=True)
class ActiveStateReachableIds:
    event_ids: frozenset[str]
    goal_ids: frozenset[str]
    commitment_ids: frozenset[str]
    plan_proposal_ids: frozenset[str]
    skill_ids: frozenset[str]
    trace_ids: frozenset[str]
    capability_ids: frozenset[str]
    block_ids: frozenset[str]
    claim_ids: frozenset[str]
    dossier_ids: frozenset[str]
    private_record_ids: frozenset[str]
    active_situation_record_ids: frozenset[str]
    scene_entity_ids: frozenset[str]
    scene_affordance_ids: frozenset[str]


@st.composite
def active_state_seed_strategy(draw) -> ActiveStateSeedSpec:
    fresh_for_secs = draw(st.integers(min_value=4, max_value=8))
    expire_after_secs = draw(st.integers(min_value=fresh_for_secs + 8, max_value=28))
    return ActiveStateSeedSpec(
        zone_count=draw(st.integers(min_value=1, max_value=4)),
        entity_count=draw(st.integers(min_value=1, max_value=6)),
        fresh_for_secs=fresh_for_secs,
        expire_after_secs=expire_after_secs,
        plan_resolution=draw(st.sampled_from(("pending", "rejected", "adopted", "superseded"))),
        extra_tool_outcomes=draw(st.integers(min_value=4, max_value=8)),
        degraded_mode=draw(st.sampled_from(("healthy", "limited", "unavailable"))),
        include_skill=draw(st.booleans()),
    )


def context_budget(task: str, *, max_tokens: int) -> BrainContextBudgetProfile:
    return BrainContextBudgetProfile(task=task, max_tokens=max_tokens)


def build_active_state_bundle(
    spec: ActiveStateSeedSpec,
    *,
    padded_text: bool = False,
    reverse_entities: bool = False,
    client_suffix: str = "default",
) -> ActiveStateBundle:
    tempdir = TemporaryDirectory()
    store = BrainStore(path=Path(tempdir.name) / "brain.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id=_bounded_client_id(client_suffix),
    )
    store.ensure_default_blocks(_DEFAULT_BLOCKS)
    _seed_current_claim(store, session_ids)

    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    goal_id = executive.create_commitment_goal(
        title="Review the dock workflow",
        intent="maintenance.review",
        source="property",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"summary": "Maintain the dock workflow context."},
    )
    commitment_id = _find_commitment_id_for_goal(store, session_ids=session_ids, goal_id=goal_id)

    current_second = 4
    selected_skill_id: str | None = None
    if spec.include_skill:
        _append_completed_procedural_goal(
            store,
            session_ids,
            goal_id="skill-goal-1",
            commitment_id="skill-commitment-1",
            goal_title="Review the dock workflow",
            proposal_id="skill-proposal-1",
            start_second=current_second,
        )
        current_second += 8
        _append_completed_procedural_goal(
            store,
            session_ids,
            goal_id="skill-goal-2",
            commitment_id="skill-commitment-2",
            goal_title="Review the dock workflow again",
            proposal_id="skill-proposal-2",
            start_second=current_second,
        )
        current_second += 8
        store.consolidate_procedural_skills(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            scope_type="thread",
            scope_id=session_ids.thread_id,
        )
        procedural_skills = store.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=session_ids.thread_id,
        )
        selected_skill_id = next(
            (skill.skill_id for skill in procedural_skills.skills if skill.skill_id),
            None,
        )

    primary_proposal_id = "proposal-primary"
    primary_proposal = BrainPlanProposal(
        plan_proposal_id=primary_proposal_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=_variant_text("Review the dock and confirm the next maintenance step.", padded=padded_text),
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        steps=[BrainGoalStep(capability_id="vision.inspect")],
        assumptions=[
            _variant_text("the operator still wants the dock workflow reviewed", padded=padded_text)
        ],
        missing_inputs=[_variant_text("which dock shelf is in scope", padded=padded_text)],
        details={"request_kind": "initial_plan"},
        created_at=_ts(current_second),
    )
    if selected_skill_id is not None:
        primary_proposal.details["procedural"] = {
            "origin": "skill_reuse",
            "selected_skill_id": selected_skill_id,
        }
    proposed_event = store.append_planning_proposed(
        proposal=primary_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        correlation_id=goal_id,
        ts=_ts(current_second),
    )
    current_second += 1
    if spec.plan_resolution == "adopted":
        store.append_planning_adopted(
            proposal=primary_proposal,
            decision=BrainPlanProposalDecision(
                summary="Adopt the plan for execution.",
                reason="bounded_plan_available",
            ),
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            correlation_id=goal_id,
            causal_parent_id=proposed_event.event_id,
            ts=_ts(current_second),
        )
        current_second += 1
    elif spec.plan_resolution == "rejected":
        store.append_planning_rejected(
            proposal=primary_proposal,
            decision=BrainPlanProposalDecision(
                summary="Need fresh input before the plan can proceed.",
                reason="missing_required_input",
            ),
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            correlation_id=goal_id,
            causal_parent_id=proposed_event.event_id,
            ts=_ts(current_second),
        )
        current_second += 1
    elif spec.plan_resolution == "superseded":
        replacement = BrainPlanProposal(
            plan_proposal_id="proposal-replacement",
            goal_id=goal_id,
            commitment_id=commitment_id,
            source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
            summary="Use the updated dock review sequence.",
            current_plan_revision=1,
            plan_revision=2,
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            steps=[BrainGoalStep(capability_id="vision.inspect")],
            assumptions=["the replacement plan supersedes the earlier plan"],
            missing_inputs=[],
            supersedes_plan_proposal_id=primary_proposal_id,
            details={"request_kind": "revision"},
            created_at=_ts(current_second),
        )
        store.append_planning_proposed(
            proposal=replacement,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            correlation_id=goal_id,
            ts=_ts(current_second),
        )
        current_second += 1

    for index in range(spec.extra_tool_outcomes):
        requested = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            payload={
                "goal_id": goal_id,
                "capability_id": f"vision.inspect.{index}",
                "arguments": {"slot": index},
                "step_index": index,
            },
            correlation_id=goal_id,
            ts=_ts(current_second),
        )
        current_second += 1
        result_event_type = (
            BrainEventType.CAPABILITY_COMPLETED
            if index % 2 == 0
            else BrainEventType.CAPABILITY_FAILED
        )
        payload = {
            "goal_id": goal_id,
            "capability_id": f"vision.inspect.{index}",
            "step_index": index,
        }
        if result_event_type == BrainEventType.CAPABILITY_COMPLETED:
            payload["result"] = CapabilityExecutionResult.success(
                capability_id=f"vision.inspect.{index}",
                summary=f"Inspected dock slot {index}.",
                output={"slot": index},
            ).model_dump()
        else:
            payload["result"] = CapabilityExecutionResult.blocked(
                capability_id=f"vision.inspect.{index}",
                summary=f"Dock slot {index} was unavailable.",
                error_code="slot_unavailable",
                retryable=False,
            ).model_dump()
        store.append_brain_event(
            event_type=result_event_type,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="property",
            payload=payload,
            correlation_id=goal_id,
            causal_parent_id=requested.event_id,
            ts=_ts(current_second),
        )
        current_second += 1

    scene_observed_second = current_second + 2
    _append_body_presence_event(
        store,
        session_ids,
        updated_second=scene_observed_second - 1,
        degraded_mode=spec.degraded_mode,
    )
    _append_perception_observed_event(
        store,
        session_ids,
        spec=spec,
        observed_second=scene_observed_second,
        padded_text=padded_text,
        reverse_entities=reverse_entities,
    )
    latest_second = scene_observed_second
    return ActiveStateBundle(
        tempdir=tempdir,
        store=store,
        session_ids=session_ids,
        goal_id=goal_id,
        commitment_id=commitment_id,
        latest_second=latest_second,
        scene_observed_second=scene_observed_second,
    )


def normalize_private_working_memory(projection: BrainPrivateWorkingMemoryProjection) -> dict[str, object]:
    return _canonicalize(_strip_projection_noise(projection.as_dict()))


def normalize_scene_world_state(projection: BrainSceneWorldProjection) -> dict[str, object]:
    return _canonicalize(_strip_projection_noise(projection.as_dict()))


def normalize_active_situation_model(projection) -> dict[str, object]:
    return _canonicalize(_strip_projection_noise(projection.as_dict()))


def collect_active_state_reachable_ids(
    bundle: ActiveStateBundle,
    *,
    snapshot: BrainContextSurfaceSnapshot | None = None,
) -> ActiveStateReachableIds:
    recent_events = bundle.store.recent_brain_events(
        user_id=bundle.session_ids.user_id,
        thread_id=bundle.session_ids.thread_id,
        limit=256,
    )
    if snapshot is None:
        agenda = bundle.store.get_agenda_projection(
            scope_key=bundle.session_ids.thread_id,
            user_id=bundle.session_ids.user_id,
        )
        commitment_projection = bundle.store.get_session_commitment_projection(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
        )
        private_working_memory = bundle.build_private_working_memory()
        active_situation_model = bundle.build_active_situation_model()
        scene_world_state = bundle.build_scene_world_state()
        procedural_skills = bundle.store.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=bundle.session_ids.thread_id,
        )
        procedural_traces = bundle.store.build_procedural_trace_projection(
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
        )
        continuity_graph = bundle.store.build_continuity_graph(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
        )
        continuity_dossiers = bundle.store.build_continuity_dossiers(
            agent_id=bundle.session_ids.agent_id,
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
            continuity_graph=continuity_graph,
        )
    else:
        agenda = snapshot.agenda
        commitment_projection = snapshot.commitment_projection
        private_working_memory = snapshot.private_working_memory
        active_situation_model = snapshot.active_situation_model
        scene_world_state = snapshot.scene_world_state
        procedural_skills = snapshot.procedural_skills
        procedural_traces = bundle.store.build_procedural_trace_projection(
            user_id=bundle.session_ids.user_id,
            thread_id=bundle.session_ids.thread_id,
        )
        continuity_graph = snapshot.continuity_graph
        continuity_dossiers = snapshot.continuity_dossiers
    plan_proposal_ids: set[str] = set()
    capability_ids: set[str] = set()
    for event in recent_events:
        proposal = BrainPlanProposal.from_dict((event.payload or {}).get("proposal"))
        if proposal is not None:
            plan_proposal_ids.add(proposal.plan_proposal_id)
        capability_id = str((event.payload or {}).get("capability_id", "")).strip()
        if capability_id:
            capability_ids.add(capability_id)
    capability_ids.update(
        affordance.capability_family
        for affordance in scene_world_state.affordances
        if affordance.capability_family
    )
    current_claims = bundle.store.query_claims(
        temporal_mode="current",
        scope_type="user",
        scope_id=bundle.session_ids.user_id,
        limit=64,
    )
    historical_claims = bundle.store.query_claims(
        temporal_mode="historical",
        scope_type="user",
        scope_id=bundle.session_ids.user_id,
        limit=64,
    )
    current_blocks = bundle.store.list_current_core_memory_blocks(limit=64)
    source_event_ids = {
        event.event_id
        for event in recent_events
        if event.event_id
    }
    source_event_ids.update(
        claim.source_event_id
        for claim in [*current_claims, *historical_claims]
        if claim.source_event_id
    )
    source_event_ids.update(
        event_id
        for node in continuity_graph.nodes
        for event_id in node.source_event_ids
        if event_id
    )
    source_event_ids.update(
        event_id
        for edge in continuity_graph.edges
        for event_id in edge.source_event_ids
        if event_id
    )
    source_event_ids.update(
        event_id
        for dossier in continuity_dossiers.dossiers
        for event_id in dossier.summary_evidence.source_event_ids
        if event_id
    )
    source_event_ids.update(
        event_id
        for dossier in continuity_dossiers.dossiers
        for fact in dossier.key_current_facts
        for event_id in fact.evidence.source_event_ids
        if event_id
    )
    source_event_ids.update(
        event_id
        for dossier in continuity_dossiers.dossiers
        for change in dossier.recent_changes
        for event_id in change.evidence.source_event_ids
        if event_id
    )
    source_event_ids.update(
        event_id
        for dossier in continuity_dossiers.dossiers
        for issue in dossier.open_issues
        for event_id in issue.evidence.source_event_ids
        if event_id
    )
    return ActiveStateReachableIds(
        event_ids=frozenset(source_event_ids),
        goal_ids=frozenset(goal.goal_id for goal in agenda.goals if goal.goal_id),
        commitment_ids=frozenset(
            {
                *(
                    record.commitment_id
                    for record in _all_commitments(commitment_projection)
                    if record.commitment_id
                ),
                *(
                    goal.commitment_id
                    for goal in agenda.goals
                    if goal.commitment_id
                ),
            }
        ),
        plan_proposal_ids=frozenset(plan_proposal_ids),
        skill_ids=frozenset(
            skill.skill_id for skill in procedural_skills.skills if skill.skill_id
        ),
        trace_ids=frozenset(
            trace.trace_id for trace in procedural_traces.traces if trace.trace_id
        ),
        capability_ids=frozenset(capability_ids),
        block_ids=frozenset(block.block_id for block in current_blocks if block.block_id),
        claim_ids=frozenset(
            claim.claim_id
            for claim in [*current_claims, *historical_claims]
            if claim.claim_id
        ),
        dossier_ids=frozenset(
            dossier.dossier_id for dossier in continuity_dossiers.dossiers if dossier.dossier_id
        ),
        private_record_ids=frozenset(
            record.record_id for record in private_working_memory.records
        ),
        active_situation_record_ids=frozenset(
            record.record_id for record in active_situation_model.records
        ),
        scene_entity_ids=frozenset(
            entity.entity_id for entity in scene_world_state.entities
        ),
        scene_affordance_ids=frozenset(
            affordance.affordance_id for affordance in scene_world_state.affordances
        ),
    )


def backing_id_is_reachable(value: str, *, reachable: ActiveStateReachableIds) -> bool:
    text = str(value).strip()
    if not text:
        return False
    direct_ids = (
        reachable.goal_ids
        | reachable.commitment_ids
        | reachable.plan_proposal_ids
        | reachable.skill_ids
        | reachable.trace_ids
        | reachable.capability_ids
        | reachable.block_ids
        | reachable.claim_ids
        | reachable.dossier_ids
        | reachable.scene_entity_ids
        | reachable.scene_affordance_ids
    )
    if text in direct_ids:
        return True
    if text in {
        "scene_state",
        "scene_world_state",
        "engagement_state",
        "body_state",
        "private_working_memory",
    }:
        return True
    return text.startswith(
        (
            "block_",
            "claim_",
            "dossier_",
            "trace_",
            "entity:",
            "zone:",
            "scene:",
            "capability:",
            "proposal:",
            "goal:",
            "skill:",
        )
    ) or text == "vision.inspect"


def _seed_current_claim(store: BrainStore, session_ids: BrainSessionIds) -> None:
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "dock operator"},
        rendered_text="The user is operating the dock workflow.",
        confidence=0.9,
        singleton=True,
        source_event_id="seed-claim-role",
        source_episode_id=None,
        provenance={"source": "property"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )


def _find_commitment_id_for_goal(
    store: BrainStore,
    *,
    session_ids: BrainSessionIds,
    goal_id: str,
) -> str:
    for record in store.list_executive_commitments(user_id=session_ids.user_id, limit=64):
        if record.current_goal_id == goal_id:
            return record.commitment_id
    raise AssertionError(f"Missing commitment for goal {goal_id}.")


def _append_body_presence_event(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    updated_second: int,
    degraded_mode: str,
) -> None:
    camera_connected = degraded_mode != "unavailable"
    frame_age_ms = 500 if degraded_mode == "healthy" else 20_000
    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        payload={
            "scope_key": PRESENCE_SCOPE_KEY,
            "snapshot": {
                "runtime_kind": "browser",
                "vision_enabled": True,
                "vision_connected": camera_connected,
                "camera_track_state": "tracking" if camera_connected else "disconnected",
                "camera_disconnected": not camera_connected,
                "sensor_health_reason": None if camera_connected else "camera_disconnected",
                "perception_unreliable": degraded_mode != "healthy",
                "last_fresh_frame_at": _ts(updated_second),
                "frame_age_ms": frame_age_ms,
                "updated_at": _ts(updated_second),
            },
        },
        ts=_ts(updated_second),
    )


def _append_perception_observed_event(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    spec: ActiveStateSeedSpec,
    observed_second: int,
    padded_text: bool,
    reverse_entities: bool,
) -> None:
    entities = [
        {
            "entity_key": _variant_text(f"dock-item-{index}", padded=padded_text),
            "kind": "object",
            "label": _variant_text(f"Dock Item {index}", padded=padded_text),
            "summary": f"Dock item {index} is visible on the workbench.",
            "zone_key": _variant_text(f"zone-{index % spec.zone_count}", padded=padded_text),
            "fresh_for_secs": spec.fresh_for_secs,
            "expire_after_secs": spec.expire_after_secs,
            "affordances": [
                {
                    "capability_family": "vision.inspect",
                    "summary": f"Inspect dock item {index}.",
                    "availability": "available",
                }
            ],
        }
        for index in range(spec.entity_count)
    ]
    zones = [
        {
            "zone_key": _variant_text(f"zone-{index}", padded=padded_text),
            "label": _variant_text(f"Zone {index}", padded=padded_text),
        }
        for index in range(spec.zone_count)
    ]
    if reverse_entities:
        entities = list(reversed(entities))
        zones = list(reversed(zones))
    camera_connected = spec.degraded_mode != "unavailable"
    camera_fresh = spec.degraded_mode == "healthy"
    store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        payload={
            "presence_scope_key": PRESENCE_SCOPE_KEY,
            "camera_connected": camera_connected,
            "camera_fresh": camera_fresh,
            "camera_track_state": "tracking" if camera_connected else "disconnected",
            "person_present": "present" if camera_connected else "uncertain",
            "summary": "Workbench camera view.",
            "observed_at": _ts(observed_second),
            "last_fresh_frame_at": _ts(observed_second if camera_fresh else observed_second - 20),
            "frame_age_ms": 500 if camera_fresh else 20_000,
            "sensor_health_reason": None if spec.degraded_mode == "healthy" else "camera_frame_stale",
            "enrichment_available": spec.degraded_mode == "healthy",
            "scene_zones": zones,
            "scene_entities": entities,
        },
        ts=_ts(observed_second),
    )


def _append_completed_procedural_goal(
    store: BrainStore,
    session_ids: BrainSessionIds,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    start_second: int,
) -> None:
    sequence = ["maintenance.review_memory_health", "reporting.record_presence_event"]
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
    proposed = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="property",
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    adopted = store.append_planning_adopted(
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
        causal_parent_id=proposed.event_id,
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
                status=BrainGoalStatus.OPEN.value,
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=1,
                last_summary=f"Adopt {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted.event_id,
        ts=_ts(start_second + 2),
    )
    current_second = start_second + 3
    completed_steps: list[BrainGoalStep] = []
    for step_index, capability_id in enumerate(sequence):
        requested = store.append_brain_event(
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
            causal_parent_id=requested.event_id,
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
                status=BrainGoalStatus.COMPLETED.value,
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=1,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {
                "commitment_id": commitment_id,
                "status": "completed",
            },
        },
        correlation_id=goal_id,
        ts=_ts(current_second),
    )


def _strip_projection_noise(value):
    if isinstance(value, dict):
        return {
            key: _strip_projection_noise(item)
            for key, item in value.items()
            if key
            not in {
                "updated_at",
                "observed_at",
                "expires_at",
                "source_event_ids",
                "scope_id",
                "record_id",
                "goal_id",
                "commitment_id",
                "plan_proposal_id",
                "skill_id",
                "backing_ids",
                "active_record_ids",
                "stale_record_ids",
                "resolved_record_ids",
                "active_entity_ids",
                "stale_entity_ids",
                "contradicted_entity_ids",
                "expired_entity_ids",
                "active_affordance_ids",
                "blocked_affordance_ids",
                "uncertain_affordance_ids",
                "linked_commitment_ids",
                "linked_plan_proposal_ids",
                "linked_skill_ids",
                "node_id",
                "dossier_id",
                "entity_id",
                "affordance_id",
            }
        }
    if isinstance(value, list):
        return [_strip_projection_noise(item) for item in value]
    return value


def _all_commitments(projection: BrainCommitmentProjection):
    return (
        list(projection.active_commitments)
        + list(projection.deferred_commitments)
        + list(projection.blocked_commitments)
        + list(projection.recent_terminal_commitments)
    )


def _canonicalize(value):
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: repr(item))
    if isinstance(value, str):
        return value.strip()
    return value

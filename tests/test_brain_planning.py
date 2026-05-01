import json

import pytest

from blink.brain._executive import BrainPlanningDraft, BrainPlanningOutcome
from blink.brain.capabilities import CapabilityExecutionResult
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt
from blink.brain.planning_digest import build_planning_digest
from blink.brain.procedural_planning import BrainPlanningProceduralOrigin
from blink.brain.projections import (
    BrainBlockedReasonKind,
    BrainCommitmentStatus,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainWakeConditionKind,
)
from blink.brain.replay import BrainReplayHarness
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


class SequencedPlanner:
    def __init__(self, *drafts: BrainPlanningDraft | None):
        self._drafts = list(drafts)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        if not self._drafts:
            return None
        return self._drafts.pop(0)


class DummyPlanningLLM:
    def __init__(self, *, response: str):
        self.response = response
        self.system_instructions: list[str] = []
        self.messages: list[list[dict]] = []

    def register_function(self, function_name, handler):
        return None

    async def run_inference(self, context, max_tokens=None, system_instruction=None):
        self.system_instructions.append(system_instruction or "")
        self.messages.append(list(context.get_messages()))
        return self.response


def _draft(
    *,
    summary: str,
    remaining_steps: list[dict],
    assumptions: list[str] | None = None,
    missing_inputs: list[str] | None = None,
    review_policy: str | None = None,
    procedural_origin: str | None = None,
    selected_skill_id: str | None = None,
    rejected_skills: list[dict] | None = None,
    delta: dict | None = None,
    details: dict | None = None,
) -> BrainPlanningDraft:
    hydrated = BrainPlanningDraft.from_dict(
        {
            "summary": summary,
            "remaining_steps": remaining_steps,
            "assumptions": assumptions or [],
            "missing_inputs": missing_inputs or [],
            "review_policy": review_policy,
            "procedural_origin": procedural_origin,
            "selected_skill_id": selected_skill_id,
            "rejected_skills": rejected_skills or [],
            "delta": delta,
            "details": details or {},
        }
    )
    assert hydrated is not None
    return hydrated


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _append_completed_procedural_goal(
    store: BrainStore,
    session_ids,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
    goal_family: str = BrainGoalFamily.MEMORY_MAINTENANCE.value,
    current_plan_revision: int = 1,
    plan_revision: int = 1,
    supersedes_plan_proposal_id: str | None = None,
):
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=goal_title,
        intent="maintenance.review",
        source="test",
        goal_family=goal_family,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
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
        current_plan_revision=current_plan_revision,
        plan_revision=plan_revision,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        supersedes_plan_proposal_id=supersedes_plan_proposal_id,
        created_at=_ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
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
        source="test",
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
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status="open",
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=plan_revision,
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
            source="test",
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
            source="test",
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
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status="completed",
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=plan_revision,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "completed"},
        },
        correlation_id=goal_id,
        ts=_ts(current_second),
    )
    return proposal


def _build_executive(*, store: BrainStore, session_ids, planning_callback=None) -> BrainExecutive:
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
        planning_callback=planning_callback,
    )


def _build_policy_executive(
    *,
    store: BrainStore,
    session_ids,
    planning_callback=None,
) -> BrainExecutive:
    registry = build_brain_capability_registry(language=Language.EN)
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=registry,
        planning_callback=planning_callback,
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=registry,
        ),
    )


def _consolidate_thread_skills(store: BrainStore, session_ids):
    return store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )


def _recent_planning_events(store: BrainStore, *, session_ids, limit: int = 16):
    events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=limit,
            )
        )
    )
    return [
        event
        for event in events
        if event.event_type
        in {
            BrainEventType.PLANNING_REQUESTED,
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.PLANNING_REJECTED,
            BrainEventType.GOAL_UPDATED,
            BrainEventType.GOAL_REPAIRED,
        }
    ]


@pytest.mark.asyncio
async def test_bounded_planning_auto_adopts_safe_initial_plan(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-auto")
    planner = SequencedPlanner(
        _draft(
            summary="Review memory health and record a note.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            assumptions=["memory maintenance can be handled locally"],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Review memory health",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    planning_events = _recent_planning_events(store, session_ids=session_ids)

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert goal is not None
    assert goal.status == BrainGoalStatus.OPEN.value
    assert goal.plan_revision == 1
    assert [step.capability_id for step in goal.steps] == [
        "maintenance.review_memory_health",
        "reporting.record_maintenance_note",
    ]
    assert goal.details["current_plan_proposal_id"] == result.proposal.plan_proposal_id
    assert goal.details["plan_review_policy"] == BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
    assert "pending_plan_proposal_id" not in goal.details
    assert commitment.status == BrainCommitmentStatus.ACTIVE.value
    assert commitment.details["current_plan_proposal_id"] == result.proposal.plan_proposal_id
    assert [event.event_type for event in planning_events] == [
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.GOAL_UPDATED,
    ]


@pytest.mark.asyncio
async def test_bounded_planning_holds_user_review_for_missing_inputs(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-user")
    planner = SequencedPlanner(
        _draft(
            summary="Ask the user which contact to notify before reporting back.",
            remaining_steps=[
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=["preferred contact name"],
            review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Need user choice before note",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    assert result.outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value
    assert goal is not None
    assert goal.status == BrainGoalStatus.WAITING.value
    assert goal.blocked_reason is not None
    assert goal.blocked_reason.kind == BrainBlockedReasonKind.WAITING_USER.value
    assert goal.wake_conditions[0].kind == BrainWakeConditionKind.USER_RESPONSE.value
    assert goal.details["pending_plan_proposal_id"] == result.proposal.plan_proposal_id
    assert goal.details["plan_review_policy"] == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value
    assert goal.steps == []
    assert commitment.status == BrainCommitmentStatus.DEFERRED.value
    assert commitment.details["pending_plan_proposal_id"] == result.proposal.plan_proposal_id


@pytest.mark.asyncio
async def test_bounded_planning_holds_operator_review_for_dialogue_steps(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-operator")
    planner = SequencedPlanner(
        _draft(
            summary="Emit a proactive dialogue acknowledgement.",
            remaining_steps=[
                {"capability_id": "dialogue.emit_brief_reengagement", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Plan dialogue work",
        intent="narrative.commitment",
        source="test",
        goal_status=BrainGoalStatus.OPEN.value,
        details={"durable_commitment": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    assert result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    assert goal is not None
    assert goal.status == BrainGoalStatus.BLOCKED.value
    assert goal.blocked_reason is not None
    assert goal.blocked_reason.kind == BrainBlockedReasonKind.OPERATOR_REVIEW.value
    assert goal.wake_conditions[0].kind == BrainWakeConditionKind.OPERATOR_REVIEW.value
    assert goal.details["pending_plan_proposal_id"] == result.proposal.plan_proposal_id
    assert commitment.status == BrainCommitmentStatus.BLOCKED.value


@pytest.mark.asyncio
async def test_bounded_planning_rejects_unknown_capabilities(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-reject")
    planner = SequencedPlanner(
        _draft(
            summary="Use an unknown capability.",
            remaining_steps=[
                {"capability_id": "unknown.capability", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Reject unsupported planning",
        intent="narrative.commitment",
        source="test",
        goal_status=BrainGoalStatus.OPEN.value,
        details={"durable_commitment": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    planning_events = _recent_planning_events(store, session_ids=session_ids)

    assert result.outcome == BrainPlanningOutcome.REJECTED.value
    assert goal is not None
    assert goal.status == BrainGoalStatus.BLOCKED.value
    assert goal.blocked_reason is not None
    assert goal.blocked_reason.kind == BrainBlockedReasonKind.OPERATOR_REVIEW.value
    assert "pending_plan_proposal_id" not in goal.details
    assert commitment.status == BrainCommitmentStatus.BLOCKED.value
    assert [event.event_type for event in planning_events] == [
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.GOAL_UPDATED,
    ]


@pytest.mark.asyncio
async def test_bounded_revision_preserves_completed_prefix_and_updates_tail(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-revise")
    planner = SequencedPlanner(
        _draft(
            summary="Initial safe maintenance plan.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Swap the unfinished tail while keeping the completed prefix.",
            remaining_steps=[
                {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
            ],
            assumptions=["the first maintenance step already completed"],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Revise maintenance plan",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    await executive.request_plan_proposal(goal_id=goal_id)
    first_pass = await executive.run_once()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    result = await executive.request_commitment_plan_revision(commitment_id=commitment.commitment_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    refreshed = store.get_executive_commitment(commitment_id=commitment.commitment_id)

    assert first_pass.progressed is True
    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert goal is not None
    assert goal.plan_revision == 2
    assert goal.steps[0].capability_id == "maintenance.review_memory_health"
    assert goal.steps[0].status == "completed"
    assert goal.steps[1].capability_id == "maintenance.review_scheduler_backpressure"
    assert goal.steps[1].status == "pending"
    assert refreshed is not None
    assert refreshed.plan_revision == 2
    assert goal.details["current_plan_proposal_id"] == result.proposal.plan_proposal_id


@pytest.mark.asyncio
async def test_review_bound_revision_does_not_mutate_live_tail_or_revision(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-revise-hold")
    planner = SequencedPlanner(
        _draft(
            summary="Initial safe maintenance plan.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Propose a dialogue tail that needs review.",
            remaining_steps=[
                {"capability_id": "dialogue.emit_brief_reengagement", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=[],
            review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
        ),
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Hold revised plan for review",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    await executive.request_plan_proposal(goal_id=goal_id)
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    before_goal = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    ).goal(goal_id)

    result = await executive.request_commitment_plan_revision(commitment_id=commitment.commitment_id)
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)
    refreshed = store.get_executive_commitment(commitment_id=commitment.commitment_id)

    assert result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    assert before_goal is not None
    assert goal is not None
    assert goal.plan_revision == before_goal.plan_revision
    assert [step.capability_id for step in goal.steps] == [step.capability_id for step in before_goal.steps]
    assert goal.details["pending_plan_proposal_id"] == result.proposal.plan_proposal_id
    assert refreshed is not None
    assert refreshed.plan_revision == commitment.plan_revision
    assert refreshed.status == BrainCommitmentStatus.BLOCKED.value


@pytest.mark.asyncio
async def test_bounded_planning_reuses_active_skill_exactly(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-skill-reuse")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed skill one",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=0,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-2",
        commitment_id="seed-commitment-2",
        goal_title="Seed skill two",
        proposal_id="seed-proposal-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=40,
    )
    projection = _consolidate_thread_skills(store, session_ids)
    skill_id = projection.active_skill_ids[0]
    planner = SequencedPlanner(
        _draft(
            summary="Reuse the successful maintenance procedure.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.SKILL_REUSE.value,
            selected_skill_id=skill_id,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Reuse maintenance skill",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert planner.requests
    assert any(
        candidate.skill_id == skill_id and candidate.eligibility == "reusable"
        for candidate in planner.requests[0].skill_candidates
    )
    assert result.proposal.details["procedural"]["origin"] == "skill_reuse"
    assert result.proposal.details["procedural"]["selected_skill_id"] == skill_id
    assert result.proposal.details["procedural"]["selected_skill_support_trace_ids"]
    digest = build_planning_digest(
        agenda=store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id),
        commitment_projection=store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ),
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=64,
        ),
    )
    assert digest["procedural_origin_counts"]["skill_reuse"] >= 1
    assert skill_id in digest["recent_selected_skill_ids"]


@pytest.mark.asyncio
async def test_bounded_planning_treats_candidate_skill_as_advisory_only(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-skill-advisory")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed candidate skill",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=0,
    )
    projection = _consolidate_thread_skills(store, session_ids)
    skill_id = projection.candidate_skill_ids[0]
    planner = SequencedPlanner(
        _draft(
            summary="Draft a fresh plan even though one candidate skill is available.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.FRESH_DRAFT.value,
            rejected_skills=[
                {
                    "skill_id": skill_id,
                    "reason": "candidate_skill_advisory_only",
                    "summary": "Candidate skills are advisory only.",
                }
            ],
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Advisory skill only",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert planner.requests
    assert any(
        candidate.skill_id == skill_id and candidate.eligibility == "advisory"
        for candidate in planner.requests[0].skill_candidates
    )
    assert result.proposal.details["procedural"]["origin"] == "fresh_draft"
    assert result.proposal.details["procedural"]["selected_skill_id"] is None


@pytest.mark.asyncio
async def test_bounded_planning_rejects_skill_reuse_mismatch(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-skill-mismatch")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed skill one",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=0,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-2",
        commitment_id="seed-commitment-2",
        goal_title="Seed skill two",
        proposal_id="seed-proposal-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=40,
    )
    projection = _consolidate_thread_skills(store, session_ids)
    skill_id = projection.active_skill_ids[0]
    planner = SequencedPlanner(
        _draft(
            summary="Incorrectly claim exact reuse with a changed tail.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.SKILL_REUSE.value,
            selected_skill_id=skill_id,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Reject mismatched reuse",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)

    assert result.outcome == BrainPlanningOutcome.REJECTED.value
    assert result.decision.reason == "skill_reuse_mismatch"
    assert result.decision.details["procedural"]["selected_skill_id"] == skill_id
    digest = build_planning_digest(
        agenda=store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id),
        commitment_projection=store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ),
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=64,
        ),
    )
    assert digest["skill_rejection_reason_counts"]["skill_reuse_mismatch"] >= 1
    assert skill_id in digest["recent_selected_skill_ids"]


@pytest.mark.asyncio
async def test_bounded_revision_adopts_bounded_skill_delta(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-skill-delta")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed skill one",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=0,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-2",
        commitment_id="seed-commitment-2",
        goal_title="Seed skill two",
        proposal_id="seed-proposal-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=40,
    )
    projection = _consolidate_thread_skills(store, session_ids)
    skill_id = projection.active_skill_ids[0]
    planner = SequencedPlanner(
        _draft(
            summary="Initial safe maintenance plan.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Adapt the reusable skill with one bounded delta on the unfinished tail.",
            remaining_steps=[
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
                {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.SKILL_DELTA.value,
            selected_skill_id=skill_id,
        ),
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Delta revise maintenance skill",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    await executive.request_plan_proposal(goal_id=goal_id)
    await executive.run_once()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    result = await executive.request_commitment_plan_revision(commitment_id=commitment.commitment_id)
    goal = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    ).goal(goal_id)

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert goal is not None
    assert goal.steps[0].capability_id == "maintenance.review_memory_health"
    assert goal.steps[0].status == "completed"
    assert result.proposal.details["procedural"]["origin"] == "skill_delta"
    assert result.proposal.details["procedural"]["selected_skill_id"] == skill_id
    assert result.proposal.details["procedural"]["delta"]["operation_count"] == 1
    digest = build_planning_digest(
        agenda=store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id),
        commitment_projection=store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ),
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=64,
        ),
    )
    assert digest["procedural_origin_counts"]["skill_delta"] >= 1
    assert digest["delta_operation_counts"]["1"] >= 1


@pytest.mark.asyncio
async def test_bounded_revision_rejects_skill_delta_out_of_bounds(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-skill-delta-reject")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-1",
        commitment_id="seed-commitment-1",
        goal_title="Seed skill one",
        proposal_id="seed-proposal-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
            "maintenance.review_scheduler_backpressure",
        ],
        start_second=0,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="seed-goal-2",
        commitment_id="seed-commitment-2",
        goal_title="Seed skill two",
        proposal_id="seed-proposal-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
            "maintenance.review_scheduler_backpressure",
        ],
        start_second=40,
    )
    projection = _consolidate_thread_skills(store, session_ids)
    skill_id = projection.active_skill_ids[0]
    planner = SequencedPlanner(
        _draft(
            summary="Initial plan.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
                {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Attempt an out-of-bounds skill delta.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.SKILL_DELTA.value,
            selected_skill_id=skill_id,
        ),
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)

    goal_id = executive.create_commitment_goal(
        title="Reject out-of-bounds delta",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    await executive.request_plan_proposal(goal_id=goal_id)
    await executive.run_once()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    result = await executive.request_commitment_plan_revision(commitment_id=commitment.commitment_id)

    assert result.outcome == BrainPlanningOutcome.REJECTED.value
    assert result.decision.reason == "skill_delta_out_of_bounds"
    assert result.decision.details["procedural"]["selected_skill_id"] == skill_id


@pytest.mark.asyncio
async def test_runtime_owned_planning_callback_parses_strict_json(tmp_path):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-runtime")
    llm = DummyPlanningLLM(
        response=json.dumps(
            {
                "summary": "Plan maintenance work through the runtime-owned planner.",
                "remaining_steps": [
                    {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                    {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
                ],
                "assumptions": ["Blink can handle maintenance locally"],
                "missing_inputs": [],
                "review_policy": "auto_adopt_ok",
            },
            ensure_ascii=False,
        )
    )
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=llm,
        brain_db_path=tmp_path / "runtime.db",
    )
    goal_id = runtime.executive.create_commitment_goal(
        title="Runtime planner goal",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
    agenda = runtime.store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goal = agenda.goal(goal_id)

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert goal is not None
    assert goal.steps[0].capability_id == "maintenance.review_memory_health"
    assert llm.system_instructions
    assert "bounded planner" in llm.system_instructions[-1].lower()
    assert "## Planning Anchors" in llm.system_instructions[-1]
    assert "## Current Claims" not in llm.system_instructions[-1]
    runtime.close()


@pytest.mark.asyncio
async def test_replay_rebuilds_review_bound_pending_proposal_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-replay")
    planner = SequencedPlanner(
        _draft(
            summary="Need user input before finishing the plan.",
            remaining_steps=[
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            assumptions=[],
            missing_inputs=["which summary format to use"],
            review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        )
    )
    executive = _build_executive(store=store, session_ids=session_ids, planning_callback=planner)
    goal_id = executive.create_commitment_goal(
        title="Replay pending proposal",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    proposal_result = await executive.request_plan_proposal(goal_id=goal_id)

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase9_planning_review_bound_state",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    replayed = harness.replay(scenario)
    replayed_goal = replayed.context_surface.agenda.goal(goal_id)

    assert replayed_goal is not None
    assert replayed_goal.status == BrainGoalStatus.WAITING.value
    assert replayed_goal.details["pending_plan_proposal_id"] == proposal_result.proposal.plan_proposal_id
    assert replayed_goal.details["plan_review_policy"] == BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value


@pytest.mark.asyncio
async def test_neutral_policy_keeps_safe_plan_auto_adopted(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-policy-neutral")
    planner = SequencedPlanner(
        _draft(
            summary="Review memory health and record a note.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        )
    )
    executive = _build_policy_executive(
        store=store,
        session_ids=session_ids,
        planning_callback=planner,
    )

    goal_id = executive.create_commitment_goal(
        title="Neutral policy planning goal",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )

    result = await executive.request_plan_proposal(goal_id=goal_id)
    planning_events = _recent_planning_events(store, session_ids=session_ids)
    proposed_event = next(
        event for event in planning_events if event.event_type == BrainEventType.PLANNING_PROPOSED
    )

    assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
    assert result.decision.executive_policy is not None
    assert result.decision.executive_policy["action_posture"] == "allow"
    assert proposed_event.payload["decision"]["executive_policy"]["action_posture"] == "allow"


@pytest.mark.asyncio
async def test_policy_defer_downgrades_auto_adopt_to_user_review(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-policy-defer")
    planner = SequencedPlanner(
        _draft(
            summary="Wait for the user to confirm the maintenance window.",
            remaining_steps=[
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            missing_inputs=["maintenance window"],
            review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        ),
        _draft(
            summary="Review memory health and record a note.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
    )
    executive = _build_policy_executive(
        store=store,
        session_ids=session_ids,
        planning_callback=planner,
    )

    first_goal_id = executive.create_commitment_goal(
        title="Need user review first",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    first_result = await executive.request_plan_proposal(goal_id=first_goal_id)

    second_goal_id = executive.create_commitment_goal(
        title="Safe plan after pending user review",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    second_result = await executive.request_plan_proposal(goal_id=second_goal_id)
    digest = build_planning_digest(
        agenda=store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        ),
        commitment_projection=store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ),
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=32,
        ),
    )

    assert first_result.outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value
    assert second_result.outcome == BrainPlanningOutcome.NEEDS_USER_REVIEW.value
    assert "policy_requires_confirmation" in second_result.decision.reason_codes
    assert second_result.decision.executive_policy is not None
    assert second_result.decision.executive_policy["action_posture"] == "defer"
    assert second_result.decision.executive_policy["approval_requirement"] == "user_confirmation"
    assert digest["policy_posture_counts"]["defer"] >= 1
    assert digest["why_not_reason_code_counts"]["policy_requires_confirmation"] >= 1


@pytest.mark.asyncio
async def test_policy_suppress_blocks_auto_adopt_but_keeps_procedural_traceability(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-policy-suppress")
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-skill-source",
        commitment_id="commitment-skill-source",
        goal_title="Review memory health",
        proposal_id="proposal-skill-source",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=10,
    )
    skills = _consolidate_thread_skills(store, session_ids)
    skill_id = skills.skills[0].skill_id
    planner = SequencedPlanner(
        _draft(
            summary="Dialogue work requires operator review.",
            remaining_steps=[
                {"capability_id": "dialogue.emit_brief_reengagement", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
        ),
        _draft(
            summary="Reuse the learned maintenance skill.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            procedural_origin=BrainPlanningProceduralOrigin.SKILL_REUSE.value,
            selected_skill_id=skill_id,
        ),
    )
    executive = _build_policy_executive(
        store=store,
        session_ids=session_ids,
        planning_callback=planner,
    )

    blocking_goal_id = executive.create_commitment_goal(
        title="Operator review blocker",
        intent="dialogue.reengage",
        source="test",
        goal_family=BrainGoalFamily.CONVERSATION.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    blocking_result = await executive.request_plan_proposal(goal_id=blocking_goal_id)

    target_goal_id = executive.create_commitment_goal(
        title="Skill reuse under suppress posture",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    target_result = await executive.request_plan_proposal(goal_id=target_goal_id)
    digest = build_planning_digest(
        agenda=store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        ),
        commitment_projection=store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ),
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=48,
        ),
    )

    assert blocking_result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    assert target_result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
    assert "policy_blocked_action" in target_result.decision.reason_codes
    assert target_result.decision.executive_policy is not None
    assert target_result.decision.executive_policy["action_posture"] == "suppress"
    assert target_result.proposal.details["procedural"]["selected_skill_id"] == skill_id
    assert target_result.proposal.details["procedural"]["policy"]["effect"] == "blocked"
    assert digest["policy_posture_counts"]["suppress"] >= 1
    assert digest["recent_skill_linked_proposals"]
    assert any(
        entry.get("procedural_policy_effect") == "blocked"
        for entry in digest["recent_skill_linked_proposals"]
    )

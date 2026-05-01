from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

from blink.brain._executive import BrainPlanningDraft
from blink.brain.actions import EmbodiedActionEngine, EmbodiedActionLibrary
from blink.brain.capabilities import CapabilityExecutionResult
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import (
    BrainBlockedReason,
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
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver
from blink.transcriptions.language import Language


def run(coro):
    """Run one async helper inside sync Hypothesis state-machine rules."""
    return asyncio.run(coro)


class SequencedPlanner:
    """Return a bounded sequence of drafts across planning calls."""

    def __init__(self, *drafts: BrainPlanningDraft | None):
        self._drafts = list(drafts)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        if not self._drafts:
            return None
        return self._drafts.pop(0)


def make_draft(
    *,
    summary: str,
    remaining_steps: list[dict[str, Any]],
    assumptions: list[str] | None = None,
    missing_inputs: list[str] | None = None,
    review_policy: str | None = None,
    procedural_origin: str | None = None,
    selected_skill_id: str | None = None,
    rejected_skills: list[dict[str, Any]] | None = None,
    delta: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> BrainPlanningDraft:
    """Build one strict planning draft payload for executive tests."""
    draft = BrainPlanningDraft.from_dict(
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
    assert draft is not None
    return draft


def ts(second: int) -> str:
    """Return deterministic timestamps for replay-like procedural seeding."""
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def build_embodied_executive(*, store: BrainStore, session_ids, driver: FaultInjectionDriver):
    """Build one headless executive with embodied plus internal capabilities."""
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    registry = build_brain_capability_registry(language=Language.EN, action_engine=action_engine)
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=registry,
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=registry,
        ),
    )
    return executive, controller


def build_planning_executive(*, store: BrainStore, session_ids, planning_callback=None) -> BrainExecutive:
    """Build one provider-light planning executive."""
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
        planning_callback=planning_callback,
    )


def find_commitment_for_goal(store: BrainStore, *, session_ids, goal_id: str):
    """Return the commitment that currently owns one goal id."""
    for record in store.list_executive_commitments(user_id=session_ids.user_id, limit=64):
        if record.current_goal_id == goal_id:
            return record
    raise AssertionError(f"Missing commitment for goal {goal_id}.")


def consolidate_thread_skills(store: BrainStore, *, session_ids):
    """Refresh and return the persisted thread-level procedural skill projection."""
    return store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )


def seed_blocked_robot_commitment(
    executive: BrainExecutive,
    *,
    store: BrainStore,
    session_ids,
    title: str,
    capability_id: str = "robot_head.blink",
) -> str:
    """Seed one machine-clearable blocked robot commitment without waiting on retries."""
    goal_id = executive.create_commitment_goal(
        title=title,
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": capability_id}]},
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_status=BrainGoalStatus.OPEN.value,
    )
    commitment = find_commitment_for_goal(store, session_ids=session_ids, goal_id=goal_id)
    goal = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    ).goal(goal_id)
    assert goal is not None

    blocked_reason = BrainBlockedReason(
        kind=BrainBlockedReasonKind.CAPABILITY_BLOCKED.value,
        summary="Fault driver rejected the command because ownership is busy.",
        details={
            "capability_id": capability_id,
            "error_code": "robot_head_busy",
            "outcome": "blocked",
            "retryable": True,
        },
    )
    wake_condition = BrainWakeCondition(
        kind=BrainWakeConditionKind.CONDITION_CLEARED.value,
        summary="Resume when the capability blocker clears.",
        details={
            "capability_id": capability_id,
            "error_code": "robot_head_busy",
        },
    )
    blocked_goal = BrainGoal.from_dict(deepcopy(goal.as_dict()))
    assert blocked_goal is not None
    blocked_goal.status = BrainGoalStatus.BLOCKED.value
    blocked_goal.active_step_index = 0
    blocked_goal.steps = [
        BrainGoalStep(
            capability_id=capability_id,
            status="blocked",
            attempts=2,
            error_code="robot_head_busy",
            summary="Fault driver rejected the command because ownership is busy.",
        )
    ]
    blocked_goal.blocked_reason = blocked_reason
    blocked_goal.wake_conditions = [wake_condition]
    blocked_goal.details = {
        **blocked_goal.details,
        "commitment_status": BrainCommitmentStatus.BLOCKED.value,
        "current_plan_proposal_id": blocked_goal.details.get("current_plan_proposal_id")
        or f"seeded-plan-{goal_id}",
        "plan_review_policy": BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
    }
    blocked_goal.updated_at = ts(10)

    store.upsert_executive_commitment(
        commitment_id=commitment.commitment_id,
        scope_type=commitment.scope_type,
        scope_id=commitment.scope_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        title=commitment.title,
        goal_family=commitment.goal_family,
        intent=commitment.intent,
        status=BrainCommitmentStatus.BLOCKED.value,
        details={
            **commitment.details,
            "summary": commitment.details.get("summary") or commitment.title,
            "current_plan_proposal_id": blocked_goal.details["current_plan_proposal_id"],
            "plan_review_policy": BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        },
        current_goal_id=commitment.current_goal_id,
        blocked_reason=blocked_reason,
        wake_conditions=[wake_condition],
        plan_revision=1,
        resume_count=commitment.resume_count,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={
            "goal": blocked_goal.as_dict(),
            "commitment": {
                "commitment_id": commitment.commitment_id,
                "status": BrainCommitmentStatus.BLOCKED.value,
            },
        },
        correlation_id=commitment.commitment_id,
        ts=ts(11),
    )
    return commitment.commitment_id


def append_completed_procedural_goal(
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
    """Append one successful bounded procedural trace into the event log."""
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=goal_title,
        intent="maintenance.review",
        source="stateful",
        goal_family=goal_family,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
        ts=ts(start_second),
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
        created_at=ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        correlation_id=goal_id,
        ts=ts(start_second),
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
        source="stateful",
        correlation_id=goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=ts(start_second + 1),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="stateful",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status=BrainGoalStatus.OPEN.value,
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=plan_revision,
                last_summary=f"Adopt {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": BrainCommitmentStatus.ACTIVE.value},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=ts(start_second + 2),
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
            source="stateful",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=goal_id,
            ts=ts(current_second),
        )
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_COMPLETED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="stateful",
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
            ts=ts(current_second + 1),
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
        source="stateful",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="stateful",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status=BrainGoalStatus.COMPLETED.value,
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=plan_revision,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {
                "commitment_id": commitment_id,
                "status": BrainCommitmentStatus.COMPLETED.value,
            },
        },
        correlation_id=goal_id,
        ts=ts(current_second),
    )
    return proposal


def append_failed_procedural_goal(
    store: BrainStore,
    session_ids,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
    failure_reason: str,
    goal_family: str = BrainGoalFamily.MEMORY_MAINTENANCE.value,
):
    """Append one failing bounded procedural trace into the event log."""
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=f"{goal_title} failure",
        intent="maintenance.review",
        source="stateful",
        goal_family=goal_family,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
        ts=ts(start_second),
    )
    proposal = BrainPlanProposal(
        plan_proposal_id=proposal_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=f"Attempt {goal_title} and fail.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        created_at=ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        correlation_id=goal_id,
        ts=ts(start_second),
    )
    adopted_event = store.append_planning_adopted(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary=f"Adopt failing {goal_title}.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        correlation_id=goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=ts(start_second + 1),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=f"{goal_title} failure",
                intent="maintenance.review",
                source="stateful",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status=BrainGoalStatus.OPEN.value,
                details={"current_plan_proposal_id": proposal.plan_proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=1,
                last_summary=f"Adopt failing {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": BrainCommitmentStatus.ACTIVE.value},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=ts(start_second + 2),
    )
    for step_index, capability_id in enumerate(sequence):
        request_event = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="stateful",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=goal_id,
            ts=ts(start_second + 3 + (step_index * 2)),
        )
        if step_index < len(sequence) - 1:
            store.append_brain_event(
                event_type=BrainEventType.CAPABILITY_COMPLETED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="stateful",
                payload={
                    "goal_id": goal_id,
                    "capability_id": capability_id,
                    "step_index": step_index,
                    "result": CapabilityExecutionResult.success(
                        capability_id=capability_id,
                        summary=f"Completed {capability_id}.",
                    ).model_dump(),
                },
                correlation_id=goal_id,
                causal_parent_id=request_event.event_id,
                ts=ts(start_second + 4 + (step_index * 2)),
            )
            continue
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_FAILED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="stateful",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "step_index": step_index,
                "result": CapabilityExecutionResult.failed(
                    capability_id=capability_id,
                    summary=f"Failed {capability_id}.",
                    error_code=failure_reason,
                ).model_dump(),
            },
            correlation_id=goal_id,
            causal_parent_id=request_event.event_id,
            ts=ts(start_second + 4 + (step_index * 2)),
        )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_FAILED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="stateful",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=f"{goal_title} failure",
                intent="maintenance.review",
                source="stateful",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status=BrainGoalStatus.FAILED.value,
                details={"current_plan_proposal_id": proposal.plan_proposal_id},
                steps=[
                    BrainGoalStep(
                        capability_id=capability_id,
                        status=(
                            "completed" if step_index < len(sequence) - 1 else "failed"
                        ),
                        attempts=1,
                        summary=(
                            f"Completed {capability_id}."
                            if step_index < len(sequence) - 1
                            else f"Failed {capability_id}."
                        ),
                        error_code=(None if step_index < len(sequence) - 1 else failure_reason),
                    )
                    for step_index, capability_id in enumerate(sequence)
                ],
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=1,
                last_summary=f"Failed {goal_title}.",
                last_error=failure_reason,
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "failed"},
        },
        correlation_id=goal_id,
        ts=ts(start_second + 4 + (len(sequence) * 2)),
    )

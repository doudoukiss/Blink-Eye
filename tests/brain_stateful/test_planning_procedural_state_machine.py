from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain._executive import BrainPlanningOutcome
from blink.brain.procedural_planning import BrainPlanningProceduralOrigin
from blink.brain.projections import BrainGoalFamily, BrainGoalStatus, BrainPlanReviewPolicy
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from tests.brain_stateful._stateful_brain_helpers import (
    SequencedPlanner,
    append_completed_procedural_goal,
    append_failed_procedural_goal,
    build_planning_executive,
    consolidate_thread_skills,
    find_commitment_for_goal,
    make_draft,
    run,
)

pytestmark = pytest.mark.brain_stateful

REUSE_TITLE = "Review memory health"
PREFIX_TITLE = "Run the short maintenance path"
EXTENDED_TITLE = "Run the stronger maintenance path"
RETIRE_TITLE = "Review and report maintenance state"

REUSE_BASE_SEQUENCE = [
    "maintenance.review_memory_health",
    "reporting.record_maintenance_note",
]
PREFIX_SEQUENCE = [
    "maintenance.review_memory_health",
]
EXTENDED_SEQUENCE = [
    "maintenance.review_memory_health",
    "maintenance.review_memory_health",
]
RETIRE_SEQUENCE = [
    "maintenance.review_memory_health",
    "reporting.record_presence_event",
]


class PlanningProceduralStateMachine(RuleBasedStateMachine):
    """Exercise procedural reuse, bounded revisions, supersession, and retirement."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=f"{self._tmpdir.name}/brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-planning-procedural",
        )
        self._counter = 0
        self._second = 0
        self.last_planning_result = None
        self.skill_projection = None
        self._append_success(goal_title=REUSE_TITLE, sequence=REUSE_BASE_SEQUENCE)
        self._append_success(goal_title=REUSE_TITLE, sequence=REUSE_BASE_SEQUENCE)
        self._refresh_projection()

    def teardown(self):
        self._tmpdir.cleanup()

    def _next_name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _next_second(self) -> int:
        start = self._second
        self._second += 40
        return start

    def _refresh_projection(self):
        self.skill_projection = consolidate_thread_skills(
            self.store,
            session_ids=self.session_ids,
        )

    def _append_success(self, *, goal_title: str, sequence: list[str]):
        start_second = self._next_second()
        append_completed_procedural_goal(
            self.store,
            self.session_ids,
            goal_id=self._next_name("goal"),
            commitment_id=self._next_name("commitment"),
            goal_title=goal_title,
            proposal_id=self._next_name("proposal"),
            sequence=sequence,
            start_second=start_second,
        )

    def _append_failure(self, *, goal_title: str, sequence: list[str], failure_reason: str):
        start_second = self._next_second()
        append_failed_procedural_goal(
            self.store,
            self.session_ids,
            goal_id=self._next_name("goal-failure"),
            commitment_id=self._next_name("commitment-failure"),
            goal_title=goal_title,
            proposal_id=self._next_name("proposal-failure"),
            sequence=sequence,
            start_second=start_second,
            failure_reason=failure_reason,
        )

    def _active_skill(self, sequence: list[str]):
        if self.skill_projection is None:
            return None
        target = tuple(sequence)
        for skill in self.skill_projection.skills:
            if skill.status != "active":
                continue
            if tuple(skill.required_capability_ids) == target:
                return skill
        return None

    def _planning_goal_commitment(self, executive, *, title: str):
        goal_id = executive.create_commitment_goal(
            title=title,
            intent="maintenance.review",
            source="stateful",
            goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
            goal_status=BrainGoalStatus.OPEN.value,
            details={"survive_restart": True},
        )
        commitment = find_commitment_for_goal(
            self.store,
            session_ids=self.session_ids,
            goal_id=goal_id,
        )
        return goal_id, commitment

    @rule()
    def add_prefix_support(self):
        self._append_success(goal_title=PREFIX_TITLE, sequence=PREFIX_SEQUENCE)
        self._refresh_projection()

    @rule()
    def add_extended_support(self):
        self._append_success(goal_title=EXTENDED_TITLE, sequence=EXTENDED_SEQUENCE)
        self._refresh_projection()

    @rule()
    def add_retirement_support(self):
        self._append_success(goal_title=RETIRE_TITLE, sequence=RETIRE_SEQUENCE)
        self._refresh_projection()

    @rule()
    def add_retirement_failure(self):
        self._append_failure(
            goal_title=RETIRE_TITLE,
            sequence=RETIRE_SEQUENCE,
            failure_reason="operator_review",
        )
        self._refresh_projection()

    @rule()
    @precondition(lambda self: self._active_skill(REUSE_BASE_SEQUENCE) is not None)
    def reuse_active_skill_exactly(self):
        # Exact reuse should only survive if the selected skill carries real support traces.
        skill = self._active_skill(REUSE_BASE_SEQUENCE)
        assert skill is not None
        planner = SequencedPlanner(
            make_draft(
                summary="Reuse the successful maintenance procedure.",
                remaining_steps=[
                    {"capability_id": capability_id, "arguments": {}}
                    for capability_id in REUSE_BASE_SEQUENCE
                ],
                review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
                procedural_origin=BrainPlanningProceduralOrigin.SKILL_REUSE.value,
                selected_skill_id=skill.skill_id,
            )
        )
        executive = build_planning_executive(
            store=self.store,
            session_ids=self.session_ids,
            planning_callback=planner,
        )
        goal_id, commitment = self._planning_goal_commitment(executive, title=REUSE_TITLE)
        result = run(executive.request_plan_proposal(goal_id=goal_id))
        executive.cancel_commitment(commitment_id=commitment.commitment_id)

        assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
        assert result.proposal.details["procedural"]["origin"] == "skill_reuse"
        assert result.proposal.details["procedural"]["selected_skill_id"] == skill.skill_id
        assert result.proposal.details["procedural"]["selected_skill_support_trace_ids"]
        self.last_planning_result = result

    @rule()
    @precondition(lambda self: self._active_skill(REUSE_BASE_SEQUENCE) is not None)
    def reject_mismatched_skill_reuse(self):
        # Reuse rejection must preserve an explicit reason instead of silently dropping the mismatch.
        skill = self._active_skill(REUSE_BASE_SEQUENCE)
        assert skill is not None
        planner = SequencedPlanner(
            make_draft(
                summary="Incorrectly claim exact reuse with a changed tail.",
                remaining_steps=[
                    {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                    {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
                ],
                review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
                procedural_origin=BrainPlanningProceduralOrigin.SKILL_REUSE.value,
                selected_skill_id=skill.skill_id,
            )
        )
        executive = build_planning_executive(
            store=self.store,
            session_ids=self.session_ids,
            planning_callback=planner,
        )
        goal_id, commitment = self._planning_goal_commitment(executive, title=REUSE_TITLE)
        result = run(executive.request_plan_proposal(goal_id=goal_id))
        executive.cancel_commitment(commitment_id=commitment.commitment_id)

        assert result.outcome == BrainPlanningOutcome.REJECTED.value
        assert result.decision.reason == "skill_reuse_mismatch"
        assert result.decision.details["procedural"]["selected_skill_id"] == skill.skill_id
        self.last_planning_result = result

    @rule()
    @precondition(lambda self: self._active_skill(REUSE_BASE_SEQUENCE) is not None)
    def revise_with_one_bounded_skill_delta(self):
        # A bounded revision may extend only the unfinished tail and must keep selected-skill provenance.
        skill = self._active_skill(REUSE_BASE_SEQUENCE)
        assert skill is not None
        planner = SequencedPlanner(
            make_draft(
                summary="Initial safe maintenance plan.",
                remaining_steps=[
                    {"capability_id": capability_id, "arguments": {}}
                    for capability_id in REUSE_BASE_SEQUENCE
                ],
                review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
            ),
            make_draft(
                summary="Adapt the reusable skill with one bounded delta on the unfinished tail.",
                remaining_steps=[
                    {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
                    {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
                ],
                review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
                procedural_origin=BrainPlanningProceduralOrigin.SKILL_DELTA.value,
                selected_skill_id=skill.skill_id,
            ),
        )
        executive = build_planning_executive(
            store=self.store,
            session_ids=self.session_ids,
            planning_callback=planner,
        )
        goal_id, commitment = self._planning_goal_commitment(executive, title=REUSE_TITLE)
        run(executive.request_plan_proposal(goal_id=goal_id))
        run(executive.run_once())
        result = run(executive.request_commitment_plan_revision(commitment_id=commitment.commitment_id))
        executive.cancel_commitment(commitment_id=commitment.commitment_id)

        assert result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
        assert result.proposal.details["procedural"]["origin"] == "skill_delta"
        assert result.proposal.details["procedural"]["selected_skill_id"] == skill.skill_id
        assert result.proposal.details["procedural"]["selected_skill_support_trace_ids"]
        assert result.proposal.details["procedural"]["delta"]["operation_count"] == 1
        self.last_planning_result = result

    @invariant()
    def selected_or_reused_skills_keep_support_traces(self):
        if self.last_planning_result is None:
            return
        projection = self.skill_projection
        assert projection is not None
        by_id = {skill.skill_id: skill for skill in projection.skills}

        procedural = dict(self.last_planning_result.proposal.details.get("procedural", {}))
        procedural.update(self.last_planning_result.decision.details.get("procedural", {}))
        selected_skill_id = procedural.get("selected_skill_id")
        if selected_skill_id is None:
            return
        assert selected_skill_id in by_id
        selected_support_trace_ids = set(procedural.get("selected_skill_support_trace_ids", []))
        assert selected_support_trace_ids
        assert selected_support_trace_ids <= set(by_id[selected_skill_id].supporting_trace_ids)

    @invariant()
    def rejected_reuse_surfaces_an_explicit_reason(self):
        if self.last_planning_result is None:
            return
        if self.last_planning_result.outcome != BrainPlanningOutcome.REJECTED.value:
            return
        assert self.last_planning_result.decision.reason
        for rejection in self.last_planning_result.decision.details.get("procedural", {}).get(
            "rejected_skills",
            [],
        ):
            assert rejection["reason"]

    @invariant()
    def supersession_and_retirement_links_stay_coherent(self):
        projection = self.skill_projection
        assert projection is not None
        by_id = {skill.skill_id: skill for skill in projection.skills}
        for skill in projection.skills:
            assert skill.supporting_trace_ids
            if skill.supersedes_skill_id is not None:
                assert skill.supersedes_skill_id in by_id
                assert by_id[skill.supersedes_skill_id].superseded_by_skill_id == skill.skill_id
            if skill.superseded_by_skill_id is not None:
                assert skill.superseded_by_skill_id in by_id
                assert by_id[skill.superseded_by_skill_id].supersedes_skill_id == skill.skill_id
            if skill.status == "retired":
                assert skill.retirement_reason == "repeated_relevant_failures"
                assert skill.retired_at is not None


TestPlanningProceduralStateMachine = PlanningProceduralStateMachine.TestCase
TestPlanningProceduralStateMachine.settings = settings(
    max_examples=4,
    stateful_step_count=6,
    deadline=None,
)

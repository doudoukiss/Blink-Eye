from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import (
    CapabilityDefinition,
    CapabilityExecutionResult,
    CapabilityFamily,
    CapabilityInitiativePolicy,
    CapabilityUserTurnPolicy,
    EmptyCapabilityInput,
)
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentStatus,
    BrainGoalFamily,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language
from tests.brain_stateful._stateful_brain_helpers import (
    find_commitment_for_goal,
    run,
    seed_blocked_robot_commitment,
)

pytestmark = pytest.mark.brain_stateful


@dataclass
class _RobotWakeBlocker:
    busy: bool = True


def _build_headless_wake_executive(*, store: BrainStore, session_ids, blocker: _RobotWakeBlocker):
    registry = build_brain_capability_registry(language=Language.EN)

    async def precondition(_inputs, _context):
        if blocker.busy:
            return CapabilityExecutionResult.blocked(
                capability_id="robot_head.blink",
                summary="Robot head is unavailable for robot_head.blink.",
                error_code="robot_head_unavailable",
                retryable=True,
            )
        return None

    async def executor(_inputs, _context):
        return CapabilityExecutionResult.success(
            capability_id="robot_head.blink",
            summary="Headless robot wake capability is ready.",
        )

    registry.register(
        CapabilityDefinition(
            capability_id="robot_head.blink",
            family=CapabilityFamily.ROBOT_HEAD.value,
            description="Headless robot wake capability used for wake-router preflight only.",
            input_model=EmptyCapabilityInput,
            sensitivity="safe",
            executor=executor,
            preconditions=(precondition,),
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                allowed_initiative_classes=(),
                user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
                operator_visible=False,
                proactive_dialogue=False,
            ),
        )
    )
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=registry,
    )


class AutonomyWakeStateMachine(RuleBasedStateMachine):
    """Model a small wake/reevaluation surface without dragging in runtime extras."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=f"{self._tmpdir.name}/brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-autonomy-wake",
        )
        self.robot_blocker = _RobotWakeBlocker(busy=True)
        self.executive = _build_headless_wake_executive(
            store=self.store,
            session_ids=self.session_ids,
            blocker=self.robot_blocker,
        )
        self._counter = 0
        self.assistant_turn_open = False
        self.thread_idle_commitment_id: str | None = None
        self.thread_idle_candidate_seeded = False
        self.thread_idle_wake_attempted = False
        self.blocked_robot_commitment_id: str | None = None
        self.blocked_robot_wake_attempted = False
        self.family_blocker_commitment_id: str | None = None
        self.last_wake_result = None
        self.resumed_commitment_ids: set[str] = set()
        self.held_candidate_ready_for_reeval = False

    def teardown(self):
        self._tmpdir.cleanup()

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _append_turn_event(self, event_type: str, *, payload: dict | None = None):
        return self.store.append_brain_event(
            event_type=event_type,
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
            payload=payload or {},
        )

    def _autonomy_ledger(self):
        return self.store.get_autonomy_ledger_projection(scope_key=self.session_ids.thread_id)

    def _agenda(self):
        return self.store.get_agenda_projection(
            scope_key=self.session_ids.thread_id,
            user_id=self.session_ids.user_id,
        )

    def _latest_non_action_reason(self) -> str | None:
        entries = list(reversed(self._autonomy_ledger().recent_entries))
        for entry in entries:
            if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value:
                return entry.reason
        return None

    def _recent_wake_event(self):
        for event in self.store.recent_brain_events(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            limit=24,
        ):
            if event.event_type == BrainEventType.COMMITMENT_WAKE_TRIGGERED:
                return event
        return None

    @rule()
    @precondition(
        lambda self: self.thread_idle_commitment_id is None and self.blocked_robot_commitment_id is None
    )
    def create_deferred_thread_idle_commitment(self):
        title = self._next_id("Deferred follow-up")
        goal_id = self.executive.create_commitment_goal(
            title=title,
            intent="narrative.commitment",
            source="memory",
            details={"summary": "Need to revisit this once the thread is idle."},
        )
        commitment = find_commitment_for_goal(
            self.store,
            session_ids=self.session_ids,
            goal_id=goal_id,
        )
        self.executive.defer_commitment(
            commitment_id=commitment.commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.WAITING_USER.value,
                summary="Waiting for the thread to settle.",
            ),
            wake_conditions=[
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.THREAD_IDLE.value,
                    summary="Wake when the thread is idle.",
                )
            ],
        )
        self.thread_idle_commitment_id = commitment.commitment_id
        self.thread_idle_candidate_seeded = False
        self.thread_idle_wake_attempted = False

    @rule()
    @precondition(
        lambda self: self.thread_idle_commitment_id is not None
        and not self.thread_idle_candidate_seeded
        and not self.thread_idle_wake_attempted
    )
    def seed_existing_thread_idle_candidate(self):
        commitment = self.store.get_executive_commitment(commitment_id=self.thread_idle_commitment_id)
        assert commitment is not None
        self.store.append_candidate_goal_created(
            candidate_goal=BrainCandidateGoal(
                candidate_goal_id=self._next_id("candidate-existing-wake"),
                candidate_type="commitment_wake_thread_idle",
                source=BrainCandidateGoalSource.COMMITMENT.value,
                summary=f"Revisit deferred commitment: {commitment.title}",
                goal_family=commitment.goal_family,
                urgency=0.7,
                confidence=1.0,
                initiative_class=(
                    BrainInitiativeClass.MAINTENANCE_INTERNAL.value
                    if commitment.goal_family == BrainGoalFamily.MEMORY_MAINTENANCE.value
                    else BrainInitiativeClass.INSPECT_ONLY.value
                ),
                dedupe_key=f"{commitment.commitment_id}:{BrainWakeConditionKind.THREAD_IDLE.value}",
                cooldown_key=(
                    f"{self.session_ids.thread_id}:commitment:{commitment.commitment_id}:"
                    f"{BrainWakeConditionKind.THREAD_IDLE.value}"
                ),
            ),
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
        )
        self.thread_idle_candidate_seeded = True
    @rule()
    @precondition(lambda self: not self.assistant_turn_open and not self._autonomy_ledger().current_candidates)
    def open_assistant_turn(self):
        self._append_turn_event(BrainEventType.ASSISTANT_TURN_STARTED)
        self.assistant_turn_open = True
        self.held_candidate_ready_for_reeval = False

    @rule()
    @precondition(
        lambda self: self.thread_idle_commitment_id is not None
        and not self.thread_idle_wake_attempted
        and not self.assistant_turn_open
        and self.blocked_robot_commitment_id is None
    )
    def route_thread_idle_wake(self):
        # Wake routing must surface an explicit reason when a candidate already exists,
        # and otherwise it should create exactly one bounded candidate.
        result = run(self.executive.run_commitment_wake_router(boundary_kind="startup_recovery"))
        self.last_wake_result = result
        self.thread_idle_wake_attempted = True

        if self.thread_idle_candidate_seeded:
            assert result.progressed is True
            assert result.route_kind == "keep_waiting"
            assert result.reason == "candidate_already_current"
            return

        assert result.progressed is True
        assert result.route_kind == "propose_candidate"
        assert any(goal.intent == "autonomy.commitment_wake_thread_idle" for goal in self._agenda().goals)

    @rule()
    @precondition(
        lambda self: self.assistant_turn_open
        and self.thread_idle_commitment_id is None
        and self.blocked_robot_commitment_id is None
        and not self._autonomy_ledger().current_candidates
    )
    def hold_presence_candidate_during_assistant_turn(self):
        # Reevaluation only matters if held candidates keep their reason and trigger metadata visible.
        result = self.executive.propose_candidate_goal(
            candidate_goal=BrainCandidateGoal(
                candidate_goal_id=self._next_id("held-candidate"),
                candidate_type="presence_acknowledgement",
                source=BrainCandidateGoalSource.PERCEPTION.value,
                summary="Resume after the assistant turn closes.",
                goal_family=BrainGoalFamily.ENVIRONMENT.value,
                urgency=0.7,
                confidence=0.9,
                initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
                dedupe_key=self._next_id("presence-dedupe"),
                cooldown_key=self._next_id("presence-cooldown"),
                policy_tags=["stateful"],
                requires_user_turn_gap=True,
            )
        )
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert self._latest_non_action_reason() == "assistant_turn_open"
        self.held_candidate_ready_for_reeval = True

    @rule()
    @precondition(lambda self: self.assistant_turn_open and self.held_candidate_ready_for_reeval)
    def close_assistant_turn_and_reevaluate(self):
        # Held wake candidates must become eligible again when their reevaluation trigger lands.
        event = self._append_turn_event(BrainEventType.ASSISTANT_TURN_ENDED)
        result = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value,
                summary="Assistant turn closed.",
                details={"turn": "assistant"},
                source_event_type=event.event_type,
                source_event_id=event.event_id,
                ts=event.ts,
            )
        )
        self.assistant_turn_open = False
        self.thread_idle_wake_attempted = False
        self.held_candidate_ready_for_reeval = False
        assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        assert self._agenda().goals

    @rule()
    @precondition(
        lambda self: self.blocked_robot_commitment_id is None and self.thread_idle_commitment_id is None
    )
    def create_blocked_robot_commitment(self):
        self.robot_blocker.busy = True
        self.blocked_robot_commitment_id = seed_blocked_robot_commitment(
            self.executive,
            store=self.store,
            session_ids=self.session_ids,
            title=self._next_id("Blocked blink"),
        )
        self.blocked_robot_wake_attempted = False

    @rule()
    @precondition(lambda self: self.blocked_robot_commitment_id is not None and self.robot_blocker.busy)
    def clear_robot_head_busy(self):
        self.robot_blocker.busy = False
        self.blocked_robot_wake_attempted = False

    @rule()
    @precondition(lambda self: self.blocked_robot_commitment_id is not None and not self.robot_blocker.busy)
    def restore_robot_head_busy(self):
        self.robot_blocker.busy = True
        self.blocked_robot_wake_attempted = False

    @rule()
    @precondition(
        lambda self: self.blocked_robot_commitment_id is not None
        and self.family_blocker_commitment_id is None
    )
    def create_environment_family_blocker(self):
        goal_id = self.executive.create_commitment_goal(
            title=self._next_id("Busy environment goal"),
            intent="robot_head.sequence",
            source="interpreter",
            details={"capabilities": [{"capability_id": "robot_head.status"}]},
            goal_family=BrainGoalFamily.ENVIRONMENT.value,
            goal_status="open",
        )
        commitment = find_commitment_for_goal(
            self.store,
            session_ids=self.session_ids,
            goal_id=goal_id,
        )
        self.family_blocker_commitment_id = commitment.commitment_id
        self.blocked_robot_wake_attempted = False

    @rule()
    @precondition(lambda self: self.family_blocker_commitment_id is not None)
    def clear_environment_family_blocker(self):
        self.executive.cancel_commitment(commitment_id=self.family_blocker_commitment_id)
        self.family_blocker_commitment_id = None
        self.blocked_robot_wake_attempted = self.blocked_robot_commitment_id is not None

    @rule()
    @precondition(
        lambda self: self.blocked_robot_commitment_id is not None
        and not self.blocked_robot_wake_attempted
        and self.thread_idle_commitment_id is None
    )
    def route_blocked_robot_wake(self):
        # Resume is only allowed when the machine-clearable blocker is gone and the
        # environment family is not already occupied by another goal.
        result = run(self.executive.run_commitment_wake_router(boundary_kind="startup_recovery"))
        self.last_wake_result = result
        self.blocked_robot_wake_attempted = True
        commitment = self.store.get_executive_commitment(commitment_id=self.blocked_robot_commitment_id)
        assert commitment is not None

        family_blocker = (
            self.store.get_executive_commitment(commitment_id=self.family_blocker_commitment_id)
            if self.family_blocker_commitment_id is not None
            else None
        )
        if family_blocker is not None and family_blocker.status == BrainCommitmentStatus.ACTIVE.value:
            assert result.progressed is True
            assert result.route_kind == "keep_waiting"
            assert result.reason == "goal_family_busy"
            assert commitment.status == BrainCommitmentStatus.BLOCKED.value
            assert commitment.resume_count == 0
            return

        if self.robot_blocker.busy:
            assert result.progressed is True
            assert result.route_kind == "keep_waiting"
            assert result.reason == "robot_head_unavailable"
            assert commitment.status == BrainCommitmentStatus.BLOCKED.value
            assert commitment.resume_count == 0
            return

        assert result.progressed is True
        assert result.route_kind == "resume_direct"
        assert result.reason == "blocker_cleared"
        resumed = self.store.get_executive_commitment(commitment_id=self.blocked_robot_commitment_id)
        assert resumed is not None
        assert resumed.status == BrainCommitmentStatus.ACTIVE.value
        assert resumed.resume_count == 1
        self.resumed_commitment_ids.add(resumed.commitment_id)
        self.blocked_robot_commitment_id = None
        self.blocked_robot_wake_attempted = False

    @invariant()
    def resumed_commitments_clear_blockers_and_keep_robot_scope(self):
        for commitment_id in self.resumed_commitment_ids:
            commitment = self.store.get_executive_commitment(commitment_id=commitment_id)
            assert commitment is not None
            assert commitment.status == BrainCommitmentStatus.ACTIVE.value
            assert commitment.blocked_reason is None
            assert not commitment.wake_conditions
            goal = self._agenda().goal(commitment.current_goal_id)
            assert goal is not None
            assert goal.goal_family == BrainGoalFamily.ENVIRONMENT.value
            assert goal.blocked_reason is None
            assert all(step.capability_id.startswith("robot_head.") for step in goal.steps)

    @invariant()
    def wake_events_keep_route_reason_visible(self):
        if self.last_wake_result is None or not self.last_wake_result.progressed:
            return
        wake_event = self._recent_wake_event()
        assert wake_event is not None
        routing = wake_event.payload["routing"]
        assert routing["route_kind"] == self.last_wake_result.route_kind
        if self.last_wake_result.reason is not None:
            assert routing["details"]["reason"] == self.last_wake_result.reason


TestAutonomyWakeStateMachine = AutonomyWakeStateMachine.TestCase
TestAutonomyWakeStateMachine.settings = settings(
    max_examples=1,
    stateful_step_count=4,
    deadline=None,
)

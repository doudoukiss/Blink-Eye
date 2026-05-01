from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import BrainGoal, BrainGoalFamily, BrainGoalStatus
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_stateful


def _iso_now(*, offset_seconds: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_seconds)).isoformat()


class PresenceDirectorReevaluationStateMachine(RuleBasedStateMachine):
    """Exercise the bounded Presence Director reevaluation sequences headlessly."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=f"{self._tmpdir.name}/brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-presence-director",
        )
        self.executive = BrainExecutive(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            capability_registry=CapabilityRegistry(),
        )
        self._counter = 0
        self.duplicate_ids: tuple[str, str] | None = None
        self.cooldown_candidate_id: str | None = None
        self.user_turn_candidate_id: str | None = None
        self.assistant_turn_candidate_id: str | None = None
        self.maintenance_candidate_id: str | None = None
        self.projection_candidate_ids: tuple[str, str] | None = None
        self.startup_candidate_id: str | None = None
        self.fresh_expiry_candidate_id: str | None = None
        self.last_trigger_event_id: str | None = None
        self.last_outcome_event_id: str | None = None

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _idle(self) -> bool:
        return all(
            item is None
            for item in (
                self.duplicate_ids,
                self.cooldown_candidate_id,
                self.user_turn_candidate_id,
                self.assistant_turn_candidate_id,
                self.maintenance_candidate_id,
                self.projection_candidate_ids,
                self.startup_candidate_id,
                self.fresh_expiry_candidate_id,
            )
        )

    def _append_candidate(self, candidate: BrainCandidateGoal) -> None:
        self.store.append_candidate_goal_created(
            candidate_goal=candidate,
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
        )

    def _candidate(self, candidate_goal_id: str, summary: str, **overrides) -> BrainCandidateGoal:
        return BrainCandidateGoal(
            candidate_goal_id=candidate_goal_id,
            candidate_type=overrides.pop("candidate_type", "presence_acknowledgement"),
            source=overrides.pop("source", BrainCandidateGoalSource.PERCEPTION.value),
            summary=summary,
            goal_family=overrides.pop("goal_family", BrainGoalFamily.ENVIRONMENT.value),
            urgency=overrides.pop("urgency", 0.7),
            confidence=overrides.pop("confidence", 0.9),
            initiative_class=overrides.pop(
                "initiative_class",
                BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
            ),
            cooldown_key=overrides.pop("cooldown_key", f"cooldown:{candidate_goal_id}"),
            dedupe_key=overrides.pop("dedupe_key", f"dedupe:{candidate_goal_id}"),
            policy_tags=overrides.pop("policy_tags", ["stateful"]),
            requires_user_turn_gap=overrides.pop("requires_user_turn_gap", True),
            expires_at=overrides.pop("expires_at", _iso_now(offset_seconds=300)),
            payload=overrides.pop("payload", {"kind": "presence"}),
            created_at=overrides.pop("created_at", _iso_now(offset_seconds=-30)),
            **overrides,
        )

    def _append_turn_event(self, event_type: str) -> object:
        return self.store.append_brain_event(
            event_type=event_type,
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
            payload={},
        )

    def _ledger(self):
        return self.store.get_autonomy_ledger_projection(scope_key=self.session_ids.thread_id)

    def _agenda(self):
        return self.store.get_agenda_projection(
            scope_key=self.session_ids.thread_id,
            user_id=self.session_ids.user_id,
        )

    def _recent_events(self):
        return list(
            reversed(
                self.store.recent_brain_events(
                    user_id=self.session_ids.user_id,
                    thread_id=self.session_ids.thread_id,
                    limit=64,
                )
            )
        )

    def _complete_autonomy_goal(self, candidate_goal_id: str) -> None:
        for goal in self._agenda().goals:
            autonomy = goal.details.get("autonomy", {})
            if autonomy.get("candidate_goal_id") != candidate_goal_id:
                continue
            updated_goal = BrainGoal.from_dict(goal.as_dict())
            assert updated_goal is not None
            updated_goal.status = BrainGoalStatus.COMPLETED.value
            updated_goal.updated_at = _iso_now(offset_seconds=1)
            payload = {"goal": updated_goal.as_dict()}
            if goal.commitment_id:
                payload["commitment"] = {
                    "commitment_id": goal.commitment_id,
                    "status": "completed",
                }
            self.store.append_brain_event(
                event_type=BrainEventType.GOAL_COMPLETED,
                agent_id=self.session_ids.agent_id,
                user_id=self.session_ids.user_id,
                session_id=self.session_ids.session_id,
                thread_id=self.session_ids.thread_id,
                source="stateful",
                payload=payload,
                correlation_id=goal.goal_id,
            )
            return
        raise AssertionError(f"Missing autonomy goal for candidate {candidate_goal_id}.")

    @rule()
    @precondition(lambda self: self._idle())
    def seed_duplicate_pair(self):
        canonical_id = self._next_id("candidate-canonical")
        duplicate_id = self._next_id("candidate-duplicate")
        self._append_candidate(
            self._candidate(
                canonical_id,
                "Keep the strongest presence candidate.",
                dedupe_key="dedupe:presence-slot",
                cooldown_key="cooldown:presence-slot",
                urgency=0.9,
                created_at=_iso_now(offset_seconds=-20),
            )
        )
        self._append_candidate(
            self._candidate(
                duplicate_id,
                "Merge this weaker duplicate.",
                dedupe_key="dedupe:presence-slot",
                cooldown_key="cooldown:presence-slot",
                urgency=0.4,
                created_at=_iso_now(offset_seconds=-10),
            )
        )
        self.duplicate_ids = (canonical_id, duplicate_id)

    @rule()
    @precondition(lambda self: self.duplicate_ids is not None)
    def run_duplicate_pass(self):
        canonical_id, duplicate_id = self.duplicate_ids
        self._append_turn_event(BrainEventType.USER_TURN_STARTED)
        result = self.executive.run_presence_director_pass()
        ledger = self._ledger()
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.terminal_candidate_goal_id == canonical_id
        assert result.reason == "user_turn_open"
        assert result.merged_candidate_ids == (duplicate_id,)
        assert [candidate.candidate_goal_id for candidate in ledger.current_candidates] == [canonical_id]
        self.duplicate_ids = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_cooldown_candidate(self):
        previous = self._candidate(
            self._next_id("candidate-previous"),
            "Already acted on this recently.",
            cooldown_key="cooldown:presence",
            dedupe_key=None,
            requires_user_turn_gap=False,
            created_at=_iso_now(offset_seconds=-10),
        )
        current = self._candidate(
            self._next_id("candidate-current"),
            "Same slot should cool down.",
            cooldown_key="cooldown:presence",
            dedupe_key=None,
            requires_user_turn_gap=False,
            created_at=_iso_now(offset_seconds=-1),
        )
        self._append_candidate(previous)
        self.store.append_candidate_goal_accepted(
            candidate_goal_id=previous.candidate_goal_id,
            goal_id="goal-previous",
            reason="accepted_for_goal_creation",
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
            ts=_iso_now(offset_seconds=-5),
        )
        self._append_candidate(current)
        self.cooldown_candidate_id = current.candidate_goal_id

    @rule()
    @precondition(lambda self: self.cooldown_candidate_id is not None)
    def run_cooldown_pass(self):
        result = self.executive.run_presence_director_pass()
        assert result.suppressed_candidate_ids == (self.cooldown_candidate_id,)
        self.cooldown_candidate_id = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_user_turn_hold_candidate(self):
        candidate_id = self._next_id("candidate-user-turn")
        self._append_candidate(self._candidate(candidate_id, "Wait until the user turn closes."))
        self._append_turn_event(BrainEventType.USER_TURN_STARTED)
        self.user_turn_candidate_id = candidate_id

    @rule()
    @precondition(lambda self: self.user_turn_candidate_id is not None)
    def reevaluate_after_user_turn_closes(self):
        first = self.executive.run_presence_director_pass()
        turn_end_event = self._append_turn_event(BrainEventType.USER_TURN_ENDED)
        second = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                summary="User turn closed.",
                details={"turn": "user"},
                source_event_type=turn_end_event.event_type,
                source_event_id=turn_end_event.event_id,
                ts=turn_end_event.ts,
            )
        )
        assert first.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert second.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        trigger_event = next(
            event
            for event in self._recent_events()
            if event.event_type == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
        )
        accepted_event = next(
            event
            for event in self._recent_events()
            if event.event_type == BrainEventType.GOAL_CANDIDATE_ACCEPTED
        )
        self.last_trigger_event_id = trigger_event.event_id
        self.last_outcome_event_id = accepted_event.event_id
        self._complete_autonomy_goal(self.user_turn_candidate_id)
        self.user_turn_candidate_id = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_assistant_turn_hold_candidate(self):
        candidate_id = self._next_id("candidate-assistant-turn")
        self._append_candidate(self._candidate(candidate_id, "Wait until the assistant turn closes."))
        self._append_turn_event(BrainEventType.ASSISTANT_TURN_STARTED)
        self.assistant_turn_candidate_id = candidate_id

    @rule()
    @precondition(lambda self: self.assistant_turn_candidate_id is not None)
    def reevaluate_after_assistant_turn_closes(self):
        first = self.executive.run_presence_director_pass()
        turn_end_event = self._append_turn_event(BrainEventType.ASSISTANT_TURN_ENDED)
        second = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value,
                summary="Assistant turn closed.",
                details={"turn": "assistant"},
                source_event_type=turn_end_event.event_type,
                source_event_id=turn_end_event.event_id,
                ts=turn_end_event.ts,
            )
        )
        assert first.reason == "assistant_turn_open"
        assert second.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        self._complete_autonomy_goal(self.assistant_turn_candidate_id)
        self.assistant_turn_candidate_id = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_maintenance_candidate(self):
        self.store.append_brain_event(
            event_type=BrainEventType.USER_TURN_ENDED,
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
            payload={},
            ts=_iso_now(offset_seconds=-5),
        )
        candidate_id = self._next_id("candidate-maintenance")
        self._append_candidate(
            self._candidate(
                candidate_id,
                "Wait for the maintenance idle window.",
                goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
                initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
                requires_user_turn_gap=False,
                dedupe_key=None,
                cooldown_key=None,
            )
        )
        self.maintenance_candidate_id = candidate_id

    @rule()
    @precondition(lambda self: self.maintenance_candidate_id is not None)
    def reject_non_matching_reevaluation_for_maintenance_hold(self):
        held = self.executive.run_presence_director_pass()
        assert held.reason == "maintenance_window_closed"
        result = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                summary="Wrong trigger for the maintenance hold.",
                details={"turn": "user"},
                ts=_iso_now(offset_seconds=1),
            )
        )
        assert result.terminal_decision is None
        assert self._ledger().candidate(self.maintenance_candidate_id) is not None

    @rule()
    @precondition(lambda self: self.maintenance_candidate_id is not None)
    def accept_matching_maintenance_window_trigger(self):
        held_candidate = self._ledger().candidate(self.maintenance_candidate_id)
        if held_candidate is None:
            return
        result = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
                summary="Maintenance window opened.",
                details=held_candidate.expected_reevaluation_condition_details,
                ts=str(held_candidate.expected_reevaluation_condition_details.get("not_before")),
            )
        )
        assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        self._complete_autonomy_goal(self.maintenance_candidate_id)
        self.maintenance_candidate_id = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_projection_recheck_pair(self):
        opted_in = self._candidate(
            self._next_id("candidate-projection-opted"),
            "Projection changes may recheck this held candidate.",
            source=BrainCandidateGoalSource.PERCEPTION.value,
            requires_user_turn_gap=False,
        )
        ignored = self._candidate(
            self._next_id("candidate-projection-ignored"),
            "Do not reconsider this candidate on projection changes.",
            source=BrainCandidateGoalSource.PERCEPTION.value,
            requires_user_turn_gap=False,
            candidate_type="presence_attention_returned",
        )
        self._append_candidate(opted_in)
        self._append_candidate(ignored)
        self.store.append_director_non_action(
            candidate_goal_id=opted_in.candidate_goal_id,
            reason="goal_family_busy",
            reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
            expected_reevaluation_condition="after a meaningful scene change",
            expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            expected_reevaluation_condition_details={"allow_projection_recheck": True},
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
        )
        self.store.append_director_non_action(
            candidate_goal_id=ignored.candidate_goal_id,
            reason="goal_family_busy",
            reason_details={"goal_family": BrainGoalFamily.ENVIRONMENT.value},
            expected_reevaluation_condition="wait for explicit family availability",
            expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            expected_reevaluation_condition_details={},
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
        )
        self.projection_candidate_ids = (opted_in.candidate_goal_id, ignored.candidate_goal_id)

    @rule()
    @precondition(lambda self: self.projection_candidate_ids is not None)
    def reevaluate_projection_changed_candidates(self):
        opted_in_id, ignored_id = self.projection_candidate_ids
        result = self.executive.run_presence_director_reevaluation(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
                summary="Scene projection changed.",
                details={"person_present": "present"},
                ts=_iso_now(offset_seconds=1),
            )
        )
        assert result.terminal_candidate_goal_id == opted_in_id
        assert self._ledger().candidate(ignored_id) is not None
        self._complete_autonomy_goal(opted_in_id)
        self.projection_candidate_ids = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_startup_recovery_candidate(self):
        candidate_id = self._next_id("candidate-startup")
        self._append_candidate(self._candidate(candidate_id, "Resume this held candidate on startup."))
        self._append_turn_event(BrainEventType.USER_TURN_STARTED)
        self.executive.run_presence_director_pass()
        self._append_turn_event(BrainEventType.USER_TURN_ENDED)
        self.startup_candidate_id = candidate_id

    @rule()
    @precondition(lambda self: self.startup_candidate_id is not None)
    def run_startup_recovery(self):
        result = asyncio.run(self.executive.run_startup_pass())
        assert result.progressed is True
        assert any(
            goal.details.get("autonomy", {}).get("candidate_goal_id") == self.startup_candidate_id
            for goal in self._agenda().goals
        )
        self._complete_autonomy_goal(self.startup_candidate_id)
        self.startup_candidate_id = None

    @rule()
    @precondition(lambda self: self._idle())
    def seed_fresh_expiry_candidate(self):
        candidate_id = self._next_id("candidate-fresh-expiry")
        self._append_candidate(
            self._candidate(
                candidate_id,
                "Do not auto-accept this untouched fresh candidate during expiry cleanup.",
                requires_user_turn_gap=False,
                expires_at=_iso_now(offset_seconds=120),
            )
        )
        self.fresh_expiry_candidate_id = candidate_id

    @rule()
    @precondition(lambda self: self.fresh_expiry_candidate_id is not None)
    def run_expiry_cleanup_without_accepting_fresh_candidate(self):
        result = self.executive.run_presence_director_expiry_cleanup(
            BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.TIME_REACHED.value,
                summary="Time reached.",
                ts=_iso_now(offset_seconds=1),
            )
        )
        assert result.expired_candidate_ids == ()
        assert all(
            goal.details.get("autonomy", {}).get("candidate_goal_id") != self.fresh_expiry_candidate_id
            for goal in self._agenda().goals
        )
        assert self._ledger().candidate(self.fresh_expiry_candidate_id) is not None
        self.fresh_expiry_candidate_id = None

    @invariant()
    def suppression_and_non_action_reasons_stay_visible(self):
        for entry in self._ledger().recent_entries:
            if entry.decision_kind in {
                BrainAutonomyDecisionKind.SUPPRESSED.value,
                BrainAutonomyDecisionKind.NON_ACTION.value,
                BrainAutonomyDecisionKind.MERGED.value,
                BrainAutonomyDecisionKind.EXPIRED.value,
            }:
                assert entry.reason is not None
                assert entry.reason.strip()

    @invariant()
    def reevaluation_outcomes_keep_causal_linkage(self):
        if self.last_trigger_event_id is None or self.last_outcome_event_id is None:
            return
        outcome = next(
            event
            for event in self._recent_events()
            if event.event_id == self.last_outcome_event_id
        )
        assert outcome.causal_parent_id == self.last_trigger_event_id


TestPresenceDirectorReevaluationStateMachine = PresenceDirectorReevaluationStateMachine.TestCase
TestPresenceDirectorReevaluationStateMachine.settings = settings(
    max_examples=1,
    stateful_step_count=7,
    deadline=None,
)

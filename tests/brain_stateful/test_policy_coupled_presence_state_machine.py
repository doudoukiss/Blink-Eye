from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from blink.brain.context import BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.memory_v2 import (
    BrainContinuityDossierGovernanceRecord,
    BrainContinuityDossierProjection,
    BrainContinuityDossierRecord,
    ClaimLedger,
)
from blink.brain.projections import BrainActiveSituationProjection, BrainGoal, BrainGoalStatus
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_stateful


def _iso_now(*, offset_seconds: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_seconds)).isoformat()


class _MutablePolicySurfaceBuilder:
    """Wrap the real surface builder with bounded policy-signal overrides."""

    def __init__(self, *, store: BrainStore, session_ids):
        self._store = store
        self._session_ids = session_ids
        self.review_debt_count = 0
        self.scene_degraded_mode = "healthy"
        self._base_builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=CapabilityRegistry(),
        )

    def build(self, **kwargs):
        base = self._base_builder.build(**kwargs)
        continuity_dossiers = self._with_review_debt(base)
        active_situation_model = BrainActiveSituationProjection(
            scope_type=base.active_situation_model.scope_type,
            scope_id=base.active_situation_model.scope_id,
            updated_at=base.generated_at,
        )
        scene_world_state = replace(
            base.scene_world_state,
            degraded_mode=self.scene_degraded_mode,
            degraded_reason_codes=(
                ["camera_disconnected"]
                if self.scene_degraded_mode == "unavailable"
                else ["scene_stale"]
                if self.scene_degraded_mode == "limited"
                else []
            ),
        )
        return replace(
            base,
            active_situation_model=active_situation_model,
            continuity_dossiers=continuity_dossiers,
            scene_world_state=scene_world_state,
        )

    def _with_review_debt(self, base):
        projection = base.continuity_dossiers or BrainContinuityDossierProjection(
            scope_type="user",
            scope_id=self._session_ids.user_id,
        )
        dossiers = [
            record for record in projection.dossiers if record.dossier_id != "stateful-review-debt"
        ]
        if self.review_debt_count > 0:
            dossiers.append(
                BrainContinuityDossierRecord(
                    dossier_id="stateful-review-debt",
                    kind="relationship",
                    scope_type="relationship",
                    scope_id=f"{self._session_ids.agent_id}:{self._session_ids.user_id}",
                    title="Relationship",
                    summary="Continuity review debt is still open.",
                    status="current",
                    freshness="fresh",
                    contradiction="clear",
                    support_strength=0.8,
                    governance=BrainContinuityDossierGovernanceRecord(
                        review_debt_count=self.review_debt_count,
                        last_refresh_cause="fresh_current_support",
                    ),
                )
            )
        dossier_counts = Counter(record.kind for record in dossiers)
        freshness_counts = Counter(record.freshness for record in dossiers)
        contradiction_counts = Counter(record.contradiction for record in dossiers)
        return BrainContinuityDossierProjection(
            scope_type=projection.scope_type,
            scope_id=projection.scope_id,
            dossiers=sorted(
                dossiers,
                key=lambda record: (
                    record.kind,
                    record.scope_type,
                    record.scope_id,
                    record.dossier_id,
                ),
            ),
            dossier_counts=dict(sorted(dossier_counts.items())),
            freshness_counts=dict(sorted(freshness_counts.items())),
            contradiction_counts=dict(sorted(contradiction_counts.items())),
            current_dossier_ids=[
                record.dossier_id for record in dossiers if record.status == "current"
            ],
            stale_dossier_ids=[
                record.dossier_id for record in dossiers if record.freshness == "stale"
            ],
            needs_refresh_dossier_ids=[
                record.dossier_id for record in dossiers if record.freshness == "needs_refresh"
            ],
            uncertain_dossier_ids=[
                record.dossier_id for record in dossiers if record.freshness == "uncertain"
            ],
            contradicted_dossier_ids=[
                record.dossier_id for record in dossiers if record.contradiction == "contradicted"
            ],
        )


class PolicyCoupledPresenceStateMachine(RuleBasedStateMachine):
    """Exercise policy-held acceptance and release without a second executive stack."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-policy-presence",
        )
        self.claim_ledger = ClaimLedger(store=self.store)
        self.surface_builder = _MutablePolicySurfaceBuilder(
            store=self.store,
            session_ids=self.session_ids,
        )
        self.executive = BrainExecutive(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            capability_registry=CapabilityRegistry(),
            context_surface_builder=self.surface_builder,
        )
        self._candidate_counter = 0
        self._claim_source_counter = 0
        claim = self.claim_ledger.record_claim(
            subject_entity_id=self.session_ids.user_id,
            predicate="user.preference",
            object_value="prefers quiet status acknowledgements",
            scope_type="user",
            scope_id=self.session_ids.user_id,
            source_event_id=self._next_claim_source("record"),
        )
        self.claim_id = claim.claim_id

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def _next_claim_source(self, prefix: str) -> str:
        self._claim_source_counter += 1
        return f"{prefix}:{self._claim_source_counter}"

    def _candidate(self) -> BrainCandidateGoal:
        self._candidate_counter += 1
        candidate_goal_id = f"candidate-policy-{self._candidate_counter}"
        return BrainCandidateGoal(
            candidate_goal_id=candidate_goal_id,
            candidate_type="presence_acknowledgement",
            source=BrainCandidateGoalSource.RUNTIME.value,
            summary=f"Policy-coupled presence check {self._candidate_counter}.",
            goal_family="environment",
            urgency=0.7,
            confidence=0.92,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
            cooldown_key=f"cooldown:{candidate_goal_id}",
            dedupe_key=f"dedupe:{candidate_goal_id}",
            policy_tags=["stateful", "phase15a"],
            requires_user_turn_gap=False,
            expires_at=_iso_now(offset_seconds=300),
            payload={"kind": "presence", "candidate_goal_id": candidate_goal_id},
            created_at=_iso_now(offset_seconds=-15),
        )

    def _append_candidate(self) -> str:
        candidate = self._candidate()
        self.store.append_candidate_goal_created(
            candidate_goal=candidate,
            agent_id=self.session_ids.agent_id,
            user_id=self.session_ids.user_id,
            session_id=self.session_ids.session_id,
            thread_id=self.session_ids.thread_id,
            source="stateful",
        )
        return candidate.candidate_goal_id

    def _ledger(self):
        return self.store.get_autonomy_ledger_projection(scope_key=self.session_ids.thread_id)

    def _current_candidate(self):
        candidates = self._ledger().current_candidates
        return candidates[0] if candidates else None

    def _claim_currentness(self) -> str:
        record = self.claim_ledger.get_claim(self.claim_id)
        assert record is not None
        return record.effective_currentness_status

    def _signals_active(self) -> bool:
        return any(
            (
                self.surface_builder.review_debt_count > 0,
                self.surface_builder.scene_degraded_mode != "healthy",
                self._claim_currentness() in {"held", "stale"},
            )
        )

    def _projection_trigger(self, summary: str) -> BrainReevaluationTrigger:
        return BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
            summary=summary,
            source_event_id=f"projection:{summary.replace(' ', '-')}",
            source_event_type="stateful.policy",
            ts=_iso_now(),
            details={"source": "stateful_policy_test"},
        )

    def _complete_autonomy_goal(self, candidate_goal_id: str) -> None:
        agenda = self.store.get_agenda_projection(
            scope_key=self.session_ids.thread_id,
            user_id=self.session_ids.user_id,
        )
        for goal in agenda.goals:
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

    @rule()
    @precondition(lambda self: self._current_candidate() is None)
    def seed_candidate(self):
        self._append_candidate()

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and not self._signals_active()
        and self._current_candidate().expected_reevaluation_condition_kind is None
    )
    def accept_when_healthy(self):
        result = self.executive.run_presence_director_pass()
        assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        assert "accepted_for_goal_creation" in result.reason_codes
        assert self._current_candidate() is None
        assert result.terminal_candidate_goal_id is not None
        self._complete_autonomy_goal(result.terminal_candidate_goal_id)

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self.surface_builder.review_debt_count == 0
        and self.surface_builder.scene_degraded_mode == "healthy"
        and self._claim_currentness() == "current"
    )
    def defer_on_review_debt(self):
        self.surface_builder.review_debt_count = 1
        result = self.executive.run_presence_director_pass()
        candidate = self._current_candidate()
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.reason == "policy_conservative_deferral"
        assert "dossier_review_debt" in result.reason_codes
        assert candidate is not None
        assert (
            candidate.expected_reevaluation_condition_kind
            == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        )

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self.surface_builder.review_debt_count == 0
        and self.surface_builder.scene_degraded_mode == "healthy"
        and self._claim_currentness() == "current"
    )
    def defer_on_held_claim(self):
        self.claim_ledger.request_claim_review(
            self.claim_id,
            source_event_id=self._next_claim_source("review"),
            reason_codes=["operator_hold"],
        )
        result = self.executive.run_presence_director_pass()
        candidate = self._current_candidate()
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.reason == "policy_requires_confirmation"
        assert "held_claim_present" in result.reason_codes
        assert candidate is not None
        assert (
            candidate.expected_reevaluation_condition_kind
            == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        )

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self.surface_builder.review_debt_count == 0
        and self.surface_builder.scene_degraded_mode == "healthy"
        and self._claim_currentness() == "current"
    )
    def defer_on_stale_claim(self):
        self.claim_ledger.expire_claim(
            self.claim_id,
            source_event_id=self._next_claim_source("expire"),
            reason_codes=["expired_by_policy"],
        )
        result = self.executive.run_presence_director_pass()
        candidate = self._current_candidate()
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.reason == "policy_conservative_deferral"
        assert "stale_claim_present" in result.reason_codes
        assert candidate is not None
        assert (
            candidate.expected_reevaluation_condition_kind
            == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        )

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self.surface_builder.scene_degraded_mode == "healthy"
    )
    def suppress_on_unavailable_scene(self):
        self.surface_builder.scene_degraded_mode = "unavailable"
        result = self.executive.run_presence_director_pass()
        candidate = self._current_candidate()
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.reason == "policy_blocked_action"
        assert "scene_unavailable" in result.reason_codes
        assert candidate is not None
        assert (
            candidate.expected_reevaluation_condition_kind
            == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        )

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self._current_candidate().expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        and self._signals_active()
    )
    def reevaluate_while_signals_remain(self):
        result = self.executive.run_presence_director_reevaluation(
            self._projection_trigger("signals remain active")
        )
        assert result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value
        assert result.reason in {
            "policy_conservative_deferral",
            "policy_blocked_action",
            "policy_requires_confirmation",
        }
        assert self._current_candidate() is not None

    @rule()
    @precondition(
        lambda self: self._current_candidate() is not None
        and self._current_candidate().expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
    )
    def clear_signals_and_accept(self):
        self.surface_builder.review_debt_count = 0
        self.surface_builder.scene_degraded_mode = "healthy"
        if self._claim_currentness() in {"held", "stale"}:
            self.claim_ledger.revalidate_claim(
                self.claim_id,
                source_event_id=self._next_claim_source("revalidate"),
                confidence=0.9,
            )
        result = self.executive.run_presence_director_reevaluation(
            self._projection_trigger("signals cleared")
        )
        assert not self._signals_active()
        assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
        assert "accepted_for_goal_creation" in result.reason_codes
        assert self._current_candidate() is None
        assert result.terminal_candidate_goal_id is not None
        self._complete_autonomy_goal(result.terminal_candidate_goal_id)

    @invariant()
    def accepted_and_non_action_entries_have_reason_codes(self):
        for entry in self._ledger().recent_entries:
            if entry.decision_kind not in {
                BrainAutonomyDecisionKind.ACCEPTED.value,
                BrainAutonomyDecisionKind.NON_ACTION.value,
            }:
                continue
            assert entry.reason_codes
            assert entry.executive_policy is not None

    @invariant()
    def policy_held_candidates_use_projection_changed(self):
        candidate = self._current_candidate()
        if candidate is None or candidate.expected_reevaluation_condition_kind is None:
            return
        assert (
            candidate.expected_reevaluation_condition_kind
            == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
        )


TestPolicyCoupledPresenceStateMachine = PolicyCoupledPresenceStateMachine.TestCase
TestPolicyCoupledPresenceStateMachine.settings = settings(
    stateful_step_count=10,
    max_examples=10,
    deadline=None,
)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context import (
    BrainContextBudgetProfile,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.continuity_governance_report import build_continuity_governance_report
from blink.brain.executive import BrainExecutive
from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainContinuityDossierAvailability,
    BrainContinuityDossierKind,
    ClaimLedger,
)
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainGoalFamily,
    BrainGoalStatus,
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
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

pytestmark = pytest.mark.brain_stateful

_DEFAULT_BLOCKS = {
    "identity": "Blink identity",
    "policy": "Blink policy",
    "style": "Blink style",
    "action_library": "Blink capabilities",
}


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _replace_active_state(surface):
    private_working_memory = BrainPrivateWorkingMemoryProjection(
        scope_type="thread",
        scope_id=surface.private_working_memory.scope_id,
        records=[
            BrainPrivateWorkingMemoryRecord(
                record_id="pwm-mode-plan",
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
                summary="Pending dock maintenance still needs user confirmation.",
                state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
                backing_ids=["proposal-mode-1"],
                source_event_ids=["evt-mode-plan-1"],
                goal_id="goal-mode-1",
                commitment_id="commitment-mode-1",
                plan_proposal_id="proposal-mode-1",
                observed_at=_ts(10),
                updated_at=_ts(10),
            ),
            BrainPrivateWorkingMemoryRecord(
                record_id="pwm-mode-uncertain",
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                summary="Scene evidence is degraded after the latest dropped frame.",
                state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                backing_ids=["scene-world-mode"],
                source_event_ids=["evt-mode-scene-1"],
                commitment_id="commitment-mode-1",
                observed_at=_ts(11),
                updated_at=_ts(11),
            ),
        ],
        updated_at=_ts(12),
    )
    scene_world_state = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=[
            BrainSceneWorldEntityRecord(
                entity_id="entity-mode-1",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="dock_panel",
                summary="A dock diagnostics panel remains visible but degraded.",
                state=BrainSceneWorldRecordState.STALE.value,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                zone_id="zone:dock",
                confidence=0.72,
                freshness="stale",
                contradiction_codes=[],
                affordance_ids=["aff-mode-1"],
                backing_ids=["entity:dock_panel"],
                source_event_ids=["evt-mode-scene-1"],
                observed_at=_ts(12),
                updated_at=_ts(12),
                expires_at=_ts(40),
            )
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="aff-mode-1",
                entity_id="entity-mode-1",
                capability_family="vision.inspect",
                summary="The dock diagnostics panel can still be inspected with degraded confidence.",
                availability=BrainSceneWorldAffordanceAvailability.UNCERTAIN.value,
                confidence=0.61,
                freshness="uncertain",
                reason_codes=["degraded_feed"],
                backing_ids=["entity:dock_panel", "vision.inspect"],
                source_event_ids=["evt-mode-scene-1"],
                observed_at=_ts(12),
                updated_at=_ts(12),
                expires_at=_ts(40),
            )
        ],
        degraded_mode="limited",
        degraded_reason_codes=["camera_frame_stale"],
        updated_at=_ts(12),
    )
    active_situation_model = BrainActiveSituationProjection(
        scope_type="thread",
        scope_id=surface.active_situation_model.scope_id,
        records=[
            BrainActiveSituationRecord(
                record_id="situation-mode-plan",
                record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
                summary="Dock maintenance routing is pending confirmation.",
                state=BrainActiveSituationRecordState.ACTIVE.value,
                evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                confidence=0.74,
                freshness="current",
                uncertainty_codes=[],
                private_record_ids=["pwm-mode-plan"],
                backing_ids=["proposal-mode-1"],
                source_event_ids=["evt-mode-plan-1"],
                goal_id="goal-mode-1",
                commitment_id="commitment-mode-1",
                plan_proposal_id="proposal-mode-1",
                observed_at=_ts(10),
                updated_at=_ts(12),
            ),
            BrainActiveSituationRecord(
                record_id="situation-mode-uncertain",
                record_kind=BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
                summary="World-state confidence is degraded; continuity should be reviewed carefully.",
                state=BrainActiveSituationRecordState.UNRESOLVED.value,
                evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
                confidence=0.52,
                freshness="limited",
                uncertainty_codes=["scene_stale", "camera_frame_stale"],
                private_record_ids=["pwm-mode-uncertain"],
                backing_ids=["scene-world-mode"],
                source_event_ids=["evt-mode-scene-1"],
                commitment_id="commitment-mode-1",
                observed_at=_ts(11),
                updated_at=_ts(12),
            ),
        ],
        updated_at=_ts(12),
    )
    return replace(
        surface,
        private_working_memory=private_working_memory,
        scene_world_state=scene_world_state,
        active_situation_model=active_situation_model,
    )


class ContextModeSwitchingStateMachine(RuleBasedStateMachine):
    """Exercise multi-mode packet behavior across claim-governance transitions."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.store.ensure_default_blocks(_DEFAULT_BLOCKS)
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-context-modes",
        )
        self.user = self.store.ensure_entity(
            entity_type="user",
            canonical_name=self.session_ids.user_id,
            aliases=[self.session_ids.user_id],
            attributes={"user_id": self.session_ids.user_id},
        )
        self.ledger = ClaimLedger(store=self.store)
        self.current_claim_id: str | None = None
        self.known_claim_ids: set[str] = set()
        self.counter = 0
        executive = BrainExecutive(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            capability_registry=CapabilityRegistry(),
        )
        goal_id = executive.create_commitment_goal(
            title="Review dock maintenance routing",
            intent="maintenance.review",
            source="stateful",
            goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
            goal_status=BrainGoalStatus.OPEN.value,
        )
        commitment = self.store.list_executive_commitments(
            user_id=self.session_ids.user_id,
            limit=8,
        )[0]
        executive.defer_commitment(
            commitment_id=commitment.commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.WAITING_USER.value,
                summary="Waiting for user confirmation before resuming dock maintenance.",
            ),
            wake_conditions=[
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume after user confirmation.",
                ),
            ],
        )
        self.goal_id = goal_id
        self.commitment_id = commitment.commitment_id

    def teardown(self):
        self.store.close()
        self._tmpdir.cleanup()

    def _event_context(self):
        return self.store._memory_event_context(
            user_id=self.session_ids.user_id,
            thread_id=self.session_ids.thread_id,
            agent_id=self.session_ids.agent_id,
            session_id=self.session_ids.session_id,
            source="stateful",
            correlation_id="stateful-context-modes",
        )

    def _claim(self):
        if self.current_claim_id is None:
            return None
        return self.ledger.get_claim(self.current_claim_id)

    def _snapshot(self):
        from blink.brain.context_surfaces import BrainContextSurfaceBuilder

        surface = BrainContextSurfaceBuilder(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=CapabilityRegistry(),
        ).build(
            latest_user_text="Review the dock maintenance continuity.",
            include_historical_claims=True,
        )
        return _replace_active_state(surface)

    def _compiled_state(self):
        snapshot = self._snapshot()
        budgets = {
            BrainContextTask.REPLY: 1100,
            BrainContextTask.PLANNING: 900,
            BrainContextTask.WAKE: 900,
            BrainContextTask.REEVALUATION: 950,
            BrainContextTask.OPERATOR_AUDIT: 1200,
            BrainContextTask.GOVERNANCE_REVIEW: 1100,
        }
        packets = {
            task: compile_context_packet_from_surface(
                snapshot=snapshot,
                latest_user_text=f"Review the {task.value} packet.",
                task=task,
                language=Language.EN,
                budget_profile=BrainContextBudgetProfile(task=task.value, max_tokens=max_tokens),
            )
            for task, max_tokens in budgets.items()
        }
        report = build_continuity_governance_report(
            continuity_dossiers=snapshot.continuity_dossiers,
            continuity_graph=snapshot.continuity_graph,
            claim_governance=snapshot.claim_governance,
            packet_traces={
                task.value: packet.packet_trace.as_dict() if packet.packet_trace else None
                for task, packet in packets.items()
            },
        )
        relationship = next(
            (
                dossier
                for dossier in snapshot.continuity_dossiers.dossiers
                if dossier.kind == BrainContinuityDossierKind.RELATIONSHIP.value
            ),
            None,
        )
        return snapshot, relationship, packets, report

    @rule()
    @precondition(lambda self: self.current_claim_id is None or self._claim().is_historical)
    def record_initial_claim(self):
        self.counter += 1
        claim = self.ledger.record_claim(
            subject_entity_id=self.user.entity_id,
            predicate="profile.role",
            object_data={"value": f"role-{self.counter}"},
            source_event_id=f"evt-record-{self.counter}",
            scope_type="user",
            scope_id=self.session_ids.user_id,
            claim_key="profile.role",
            event_context=self._event_context(),
        )
        self.current_claim_id = claim.claim_id
        self.known_claim_ids.add(claim.claim_id)

    @rule()
    @precondition(lambda self: self._claim() is not None and not self._claim().is_historical)
    def request_review(self):
        claim = self._claim()
        assert claim is not None
        held = self.ledger.request_claim_review(
            claim.claim_id,
            source_event_id=f"evt-review-{self.counter}",
            reason_codes=["requires_confirmation"],
            event_context=self._event_context(),
        )
        self.current_claim_id = held.claim_id

    @rule()
    @precondition(lambda self: self._claim() is not None and not self._claim().is_historical)
    def expire_claim(self):
        claim = self._claim()
        assert claim is not None
        stale = self.ledger.expire_claim(
            claim.claim_id,
            source_event_id=f"evt-expire-{self.counter}",
            reason_codes=["stale_without_refresh"],
            event_context=self._event_context(),
        )
        self.current_claim_id = stale.claim_id

    @rule()
    @precondition(
        lambda self: self._claim() is not None and (self._claim().is_held or self._claim().is_stale)
    )
    def revalidate_claim(self):
        claim = self._claim()
        assert claim is not None
        refreshed = self.ledger.revalidate_claim(
            claim.claim_id,
            source_event_id=f"evt-revalidate-{self.counter}",
            event_context=self._event_context(),
        )
        self.current_claim_id = refreshed.claim_id

    @rule()
    @precondition(lambda self: self._claim() is not None and not self._claim().is_historical)
    def supersede_claim(self):
        claim = self._claim()
        assert claim is not None
        self.counter += 1
        replacement = self.ledger.supersede_claim(
            claim.claim_id,
            replacement_claim={
                "subject_entity_id": self.user.entity_id,
                "predicate": "profile.role",
                "object_data": {"value": f"role-{self.counter}"},
                "scope_type": "user",
                "scope_id": self.session_ids.user_id,
                "claim_key": "profile.role",
            },
            reason="stateful supersession",
            source_event_id=f"evt-supersede-{self.counter}",
            event_context=self._event_context(),
        )
        self.known_claim_ids.add(replacement.claim_id)
        self.current_claim_id = replacement.claim_id

    @invariant()
    def claims_remain_reachable(self):
        for claim_id in self.known_claim_ids:
            assert self.ledger.get_claim(claim_id) is not None

    @invariant()
    def packets_follow_explicit_mode_policy(self):
        claim = self._claim()
        if claim is None:
            return

        _snapshot, relationship, packets, report = self._compiled_state()
        assert relationship is not None
        availability_by_task = {
            record.task: record.availability for record in relationship.governance.task_availability
        }
        assert any(
            decision.section_key == "planning_anchors"
            for decision in packets[BrainContextTask.WAKE].packet_trace.section_decisions
        )
        assert packets[BrainContextTask.REEVALUATION].selected_context.section("unresolved_state") is not None
        assert (
            packets[BrainContextTask.OPERATOR_AUDIT].packet_trace.mode_policy.trace_verbosity.value
            == "verbose"
        )
        assert (
            packets[BrainContextTask.GOVERNANCE_REVIEW].packet_trace.mode_policy.trace_verbosity.value
            == "verbose"
        )
        assert (
            "planning_anchors"
            not in packets[BrainContextTask.GOVERNANCE_REVIEW].packet_trace.mode_policy.dynamic_section_keys
        )
        for task, packet in packets.items():
            assert packet.packet_trace is not None
            assert packet.packet_trace.mode_policy.task == task
            assert packet.selected_context.estimated_tokens <= packet.selected_context.budget_profile.max_tokens
            assert packet.packet_trace.section_decisions
            assert all(
                item.decision_reason_codes
                for item in packet.packet_trace.selected_items + packet.packet_trace.dropped_items
            )

        if claim.is_held:
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
            assert report["packet_suppression_counts_by_task"]["reply"]["held_support"] >= 1
            assert any(
                item.reason == "governance_suppressed"
                for item in packets[BrainContextTask.REPLY].packet_trace.dropped_items
            )
        elif claim.is_stale:
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
            assert report["packet_suppression_counts_by_task"]["reply"]["stale_support"] >= 1
            assert any(
                item.availability_state == "annotated"
                for item in packets[BrainContextTask.GOVERNANCE_REVIEW].packet_trace.selected_items
            )
        elif claim.is_current:
            if relationship.governance.review_debt_count == 0:
                assert (
                    availability_by_task["reply"]
                    == BrainContinuityDossierAvailability.AVAILABLE.value
                )

    @invariant()
    def currentness_remains_explicit(self):
        claim = self._claim()
        if claim is None:
            return
        assert claim.effective_currentness_status in {
            BrainClaimCurrentnessStatus.CURRENT.value,
            BrainClaimCurrentnessStatus.STALE.value,
            BrainClaimCurrentnessStatus.HISTORICAL.value,
            BrainClaimCurrentnessStatus.HELD.value,
        }


TestContextModeSwitchingStateMachine = ContextModeSwitchingStateMachine.TestCase
TestContextModeSwitchingStateMachine.settings = settings(
    stateful_step_count=6,
    max_examples=12,
    deadline=None,
)

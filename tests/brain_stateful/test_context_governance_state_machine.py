from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain.context import (
    BrainContextBudgetProfile,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.continuity_governance_report import build_continuity_governance_report
from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainContinuityDossierAvailability,
    BrainContinuityDossierKind,
    ClaimLedger,
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


class ContextGovernanceStateMachine(RuleBasedStateMachine):
    """Exercise dossier availability, packet traces, and reports across claim transitions."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.store.ensure_default_blocks(_DEFAULT_BLOCKS)
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-context-governance",
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
            correlation_id="stateful-context-governance",
        )

    def _claim(self):
        if self.current_claim_id is None:
            return None
        return self.ledger.get_claim(self.current_claim_id)

    def _snapshot(self):
        return BrainContextSurfaceBuilder(
            store=self.store,
            session_resolver=lambda: self.session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
        ).build(
            latest_user_text="Review the current role continuity.",
            include_historical_claims=True,
        )

    def _compiled_state(self):
        snapshot = self._snapshot()
        reply_packet = compile_context_packet_from_surface(
            snapshot=snapshot,
            latest_user_text="What is my role right now?",
            task=BrainContextTask.REPLY,
            language=Language.EN,
            budget_profile=BrainContextBudgetProfile(task="reply", max_tokens=1100),
        )
        planning_packet = compile_context_packet_from_surface(
            snapshot=snapshot,
            latest_user_text="Plan around my current role.",
            task=BrainContextTask.PLANNING,
            language=Language.EN,
            budget_profile=BrainContextBudgetProfile(task="planning", max_tokens=1100),
        )
        recall_packet = compile_context_packet_from_surface(
            snapshot=snapshot,
            latest_user_text="Recall my role changes.",
            task=BrainContextTask.RECALL,
            language=Language.EN,
            budget_profile=BrainContextBudgetProfile(task="recall", max_tokens=1100),
        )
        report = build_continuity_governance_report(
            continuity_dossiers=snapshot.continuity_dossiers,
            continuity_graph=snapshot.continuity_graph,
            claim_governance=snapshot.claim_governance,
            packet_traces={
                "reply": reply_packet.packet_trace.as_dict() if reply_packet.packet_trace else None,
                "planning": (
                    planning_packet.packet_trace.as_dict() if planning_packet.packet_trace else None
                ),
                "recall": (
                    recall_packet.packet_trace.as_dict() if recall_packet.packet_trace else None
                ),
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
        return relationship, reply_packet, planning_packet, recall_packet, report

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
    def expire_current_claim(self):
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
    def revalidate_live_claim(self):
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
    def supersede_current_claim(self):
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
    def historical_claims_never_disappear(self):
        for claim_id in self.known_claim_ids:
            assert self.ledger.get_claim(claim_id) is not None

    @invariant()
    def governance_report_is_deterministic(self):
        first = self._compiled_state()
        second = self._compiled_state()
        assert first[0] == second[0]
        assert first[-1] == second[-1]

    @invariant()
    def dossier_availability_and_packets_follow_claim_governance(self):
        claim = self._claim()
        if claim is None:
            return

        relationship, reply_packet, planning_packet, recall_packet, report = self._compiled_state()
        assert relationship is not None
        availability_by_task = {
            record.task: record.availability for record in relationship.governance.task_availability
        }
        historical_claim_ids = {
            item.claim_id
            for item in self.store.query_claims(
                temporal_mode="historical",
                scope_type="user",
                scope_id=self.session_ids.user_id,
                limit=24,
            )
        }

        if relationship.governance.review_debt_count > 0:
            assert relationship.governance.review_debt_count >= 1
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
            assert (
                availability_by_task["planning"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
            assert (
                availability_by_task["recall"]
                == BrainContinuityDossierAvailability.ANNOTATED.value
            )
            assert relationship.dossier_id in report["review_debt_dossier_ids"]
            if claim.is_held:
                assert report["packet_suppression_counts_by_task"]["reply"]
                assert any(
                    item.reason == "governance_suppressed"
                    for item in reply_packet.packet_trace.dropped_items
                )
        elif claim.is_stale:
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.SUPPRESSED.value
            )
            assert (
                availability_by_task["planning"]
                == BrainContinuityDossierAvailability.ANNOTATED.value
            )
            assert (
                availability_by_task["recall"]
                == BrainContinuityDossierAvailability.ANNOTATED.value
            )
            assert report["packet_suppression_counts_by_task"]["reply"]["stale_support"] >= 1
            recall_claim_items = [
                item
                for item in recall_packet.packet_trace.selected_items
                if item.item_type == "claim"
            ]
            assert recall_claim_items
            assert all(item.availability_state == "annotated" for item in recall_claim_items)
        elif claim.is_current:
            assert (
                availability_by_task["reply"]
                == BrainContinuityDossierAvailability.AVAILABLE.value
            )
            assert not report["review_debt_dossier_ids"]

        if historical_claim_ids:
            assert any(
                historical_claim_ids.intersection(record.evidence.claim_ids)
                for record in relationship.recent_changes
            )

    @invariant()
    def live_claim_currentness_stays_within_known_states(self):
        claim = self._claim()
        if claim is None:
            return
        assert claim.effective_currentness_status in {
            BrainClaimCurrentnessStatus.CURRENT.value,
            BrainClaimCurrentnessStatus.STALE.value,
            BrainClaimCurrentnessStatus.HISTORICAL.value,
            BrainClaimCurrentnessStatus.HELD.value,
        }


TestContextGovernanceStateMachine = ContextGovernanceStateMachine.TestCase
TestContextGovernanceStateMachine.settings = settings(
    stateful_step_count=6,
    max_examples=16,
    deadline=None,
)

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from blink.brain.memory_v2 import BrainClaimCurrentnessStatus, ClaimLedger
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_stateful


class MemoryGovernanceStateMachine(RuleBasedStateMachine):
    """Exercise explicit claim-governance transitions without wall-clock drift."""

    def __init__(self):
        super().__init__()
        self._tmpdir = TemporaryDirectory()
        self.store = BrainStore(path=Path(self._tmpdir.name) / "brain.db")
        self.session_ids = resolve_brain_session_ids(
            runtime_kind="browser",
            client_id="stateful-memory-governance",
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
        self.last_current_transition: str | None = None
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
            correlation_id="stateful-memory-governance",
        )

    def _claim(self):
        if self.current_claim_id is None:
            return None
        return self.ledger.get_claim(self.current_claim_id)

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
        self.last_current_transition = "recorded"

    @rule()
    @precondition(lambda self: self._claim() is not None and not self._claim().is_historical)
    def request_review(self):
        claim = self._claim()
        assert claim is not None
        held = self.ledger.request_claim_review(
            claim.claim_id,
            source_event_id=f"evt-review-{self.counter}",
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
        self.last_current_transition = "revalidated"

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
                "evidence_summary": f"replacement role {self.counter}",
            },
            reason="stateful supersession",
            source_event_id=f"evt-supersede-{self.counter}",
            event_context=self._event_context(),
        )
        self.known_claim_ids.add(replacement.claim_id)
        self.current_claim_id = replacement.claim_id
        self.last_current_transition = "recorded"

    @rule()
    @precondition(lambda self: self._claim() is not None and not self._claim().is_historical)
    def revoke_current_claim(self):
        claim = self._claim()
        assert claim is not None
        self.ledger.revoke_claim(
            claim.claim_id,
            reason="stateful revoke",
            source_event_id=f"evt-revoke-{self.counter}",
            event_context=self._event_context(),
        )
        self.current_claim_id = None

    @invariant()
    def historical_claims_never_disappear(self):
        for claim_id in self.known_claim_ids:
            assert self.ledger.get_claim(claim_id) is not None

    @invariant()
    def currentness_transitions_stay_legal(self):
        claim = self._claim()
        if claim is None:
            return
        assert claim.effective_currentness_status in {
            BrainClaimCurrentnessStatus.CURRENT.value,
            BrainClaimCurrentnessStatus.STALE.value,
            BrainClaimCurrentnessStatus.HISTORICAL.value,
            BrainClaimCurrentnessStatus.HELD.value,
        }
        if claim.is_current:
            assert self.last_current_transition in {"recorded", "revalidated"}

    @invariant()
    def governance_projection_tracks_known_claims(self):
        projection = self.store.get_claim_governance_projection(
            scope_type="user",
            scope_id=self.session_ids.user_id,
        )
        projection_claim_ids = {record.claim_id for record in projection.records}
        assert self.known_claim_ids <= projection_claim_ids
        for claim_id in projection.held_claim_ids:
            assert claim_id not in projection.current_claim_ids


TestMemoryGovernanceStateMachine = MemoryGovernanceStateMachine.TestCase
TestMemoryGovernanceStateMachine.settings = settings(
    stateful_step_count=6,
    max_examples=20,
    deadline=None,
)

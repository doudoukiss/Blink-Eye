from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainClaimRetentionClass,
    ClaimLedger,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@given(
    operations=st.lists(
        st.sampled_from(("review", "expire", "revalidate", "retention", "revoke")),
        min_size=1,
        max_size=6,
    ),
    retention_class=st.sampled_from(
        (
            BrainClaimRetentionClass.SESSION.value,
            BrainClaimRetentionClass.DURABLE.value,
        )
    ),
)
@_SETTINGS
def test_claim_currentness_states_remain_mutually_exclusive(operations, retention_class):
    with TemporaryDirectory() as tmpdir:
        store = BrainStore(path=Path(tmpdir) / "brain.db")
        try:
            session_ids = resolve_brain_session_ids(
                runtime_kind="browser",
                client_id="property-claim-currentness",
            )
            user = store.ensure_entity(
                entity_type="user",
                canonical_name=session_ids.user_id,
                aliases=[session_ids.user_id],
                attributes={"user_id": session_ids.user_id},
            )
            ledger = ClaimLedger(store=store)
            claim = ledger.record_claim(
                subject_entity_id=user.entity_id,
                predicate="profile.role",
                object_data={"value": "designer"},
                source_event_id="evt-currentness-seed",
                scope_type="user",
                scope_id=session_ids.user_id,
                claim_key="profile.role:property",
                event_context=None,
            )

            for index, operation in enumerate(operations):
                claim = ledger.get_claim(claim.claim_id)
                assert claim is not None
                if operation == "review" and not claim.is_historical:
                    claim = ledger.request_claim_review(
                        claim.claim_id,
                        source_event_id=f"evt-review-{index}",
                        event_context=None,
                    )
                elif operation == "expire" and not claim.is_historical:
                    claim = ledger.expire_claim(
                        claim.claim_id,
                        source_event_id=f"evt-expire-{index}",
                        event_context=None,
                    )
                elif operation == "revalidate" and (claim.is_stale or claim.is_held):
                    claim = ledger.revalidate_claim(
                        claim.claim_id,
                        source_event_id=f"evt-revalidate-{index}",
                        event_context=None,
                    )
                elif operation == "retention":
                    claim = ledger.reclassify_claim_retention(
                        claim.claim_id,
                        retention_class=retention_class,
                        source_event_id=f"evt-retention-{index}",
                        event_context=None,
                    )
                elif operation == "revoke" and not claim.is_historical:
                    ledger.revoke_claim(
                        claim.claim_id,
                        reason="Property test revoke.",
                        source_event_id=f"evt-revoke-{index}",
                        event_context=None,
                    )
                    claim = ledger.get_claim(claim.claim_id)

            claim = ledger.get_claim(claim.claim_id)
            assert claim is not None
            projection = store.get_claim_governance_projection(
                scope_type="user",
                scope_id=session_ids.user_id,
            )

            assert not (claim.is_current and claim.is_historical)
            assert claim.effective_currentness_status in {
                BrainClaimCurrentnessStatus.CURRENT.value,
                BrainClaimCurrentnessStatus.STALE.value,
                BrainClaimCurrentnessStatus.HISTORICAL.value,
                BrainClaimCurrentnessStatus.HELD.value,
            }
            if claim.is_current:
                assert claim.claim_id in projection.current_claim_ids
            if claim.is_stale:
                assert claim.claim_id in projection.stale_claim_ids
            if claim.is_historical:
                assert claim.claim_id in projection.historical_claim_ids
            if claim.is_held:
                assert claim.claim_id in projection.held_claim_ids
                assert claim.claim_id not in projection.current_claim_ids
        finally:
            store.close()

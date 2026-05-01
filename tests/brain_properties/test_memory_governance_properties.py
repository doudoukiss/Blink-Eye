from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from blink.brain.events import BrainEventType
from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainGovernanceReasonCode,
    ClaimLedger,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore

pytestmark = pytest.mark.brain_property

_SETTINGS = settings(
    max_examples=16,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _all_events(store: BrainStore):
    rows = store._conn.execute("SELECT * FROM brain_events ORDER BY id ASC").fetchall()
    return [store._brain_event_from_row(row) for row in rows]


def _normalized_projection(projection):
    payload = projection.as_dict()
    payload["updated_at"] = "<normalized>"
    for record in payload.get("records", []):
        record["last_governance_event_id"] = "<normalized>"
        record["updated_at"] = "<normalized>"
    return payload


@given(
    first_value=st.sampled_from(("designer", "writer", "maintainer")),
    second_value=st.sampled_from(("manager", "operator", "architect")),
)
@_SETTINGS
def test_supersession_keeps_prior_claims_and_replay_equivalent_streams(first_value, second_value):
    if first_value == second_value:
        return
    with TemporaryDirectory() as tmpdir:
        source = BrainStore(path=Path(tmpdir) / "source.db")
        replay = BrainStore(path=Path(tmpdir) / "replay.db")
        try:
            session_ids = resolve_brain_session_ids(
                runtime_kind="browser",
                client_id="property-governance-replay",
            )
            user = source.ensure_entity(
                entity_type="user",
                canonical_name=session_ids.user_id,
                aliases=[session_ids.user_id],
                attributes={"user_id": session_ids.user_id},
            )
            replay.ensure_entity(
                entity_type="user",
                canonical_name=session_ids.user_id,
                aliases=[session_ids.user_id],
                attributes={"user_id": session_ids.user_id},
            )
            ledger = ClaimLedger(store=source)
            event_context = source._memory_event_context(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                session_id=session_ids.session_id,
                source="property",
                correlation_id="property-governance-replay",
            )
            prior = ledger.record_claim(
                subject_entity_id=user.entity_id,
                predicate="profile.role",
                object_data={"value": first_value},
                source_event_id="evt-role-initial",
                scope_type="user",
                scope_id=session_ids.user_id,
                claim_key="profile.role",
                evidence_summary="initial role evidence",
                event_context=event_context,
            )
            replacement = ledger.supersede_claim(
                prior.claim_id,
                replacement_claim={
                    "subject_entity_id": user.entity_id,
                    "predicate": "profile.role",
                    "object_data": {"value": second_value},
                    "scope_type": "user",
                    "scope_id": session_ids.user_id,
                    "claim_key": "profile.role",
                    "evidence_summary": "replacement role evidence",
                },
                reason="role corrected",
                source_event_id="evt-role-replacement",
                event_context=event_context,
            )

            source_projection = source.get_claim_governance_projection(
                scope_type="user",
                scope_id=session_ids.user_id,
            )
            assert prior.claim_id in source_projection.historical_claim_ids
            assert replacement.claim_id in source_projection.current_claim_ids
            assert source.claim_supersessions(claim_id=prior.claim_id)

            events = _all_events(source)
            for event in events:
                replay.import_brain_event(event)
            for event in events:
                replay.apply_memory_event(event)

            replay_projection = replay.get_claim_governance_projection(
                scope_type="user",
                scope_id=session_ids.user_id,
            )
            assert replay_projection.as_dict() == source_projection.as_dict()
            assert prior.claim_id in {
                record.claim_id
                for record in replay.query_claims(
                    temporal_mode="historical",
                    scope_type="user",
                    scope_id=session_ids.user_id,
                    limit=8,
                )
            }
        finally:
            source.close()
            replay.close()


@given(terminal_event_type=st.sampled_from(("revoked", "superseded")))
@_SETTINGS
def test_legacy_claim_events_map_deterministically_into_governance_state(terminal_event_type):
    with TemporaryDirectory() as tmpdir:
        legacy = BrainStore(path=Path(tmpdir) / "legacy.db")
        explicit = BrainStore(path=Path(tmpdir) / "explicit.db")
        try:
            session_ids = resolve_brain_session_ids(
                runtime_kind="browser",
                client_id="property-legacy-governance",
            )
            for store in (legacy, explicit):
                store.ensure_entity(
                    entity_type="user",
                    canonical_name=session_ids.user_id,
                    aliases=[session_ids.user_id],
                    attributes={"user_id": session_ids.user_id},
                )

            base_payload = {
                "claim_id": "claim-legacy-1",
                "subject_entity_id": "entity-user-1",
                "predicate": "profile.origin",
                "object": {"value": "Shanghai"},
                "status": "active",
                "confidence": 0.8,
                "valid_from": "2026-01-01T00:00:00+00:00",
                "scope_type": "user",
                "scope_id": session_ids.user_id,
                "claim_key": "profile.origin",
            }
            legacy_recorded = legacy.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="property",
                payload=base_payload,
                ts="2026-01-01T00:00:00+00:00",
            )
            explicit_recorded = explicit.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="property",
                payload={
                    **base_payload,
                    "currentness_status": BrainClaimCurrentnessStatus.CURRENT.value,
                    "reason_codes": [],
                },
                ts="2026-01-01T00:00:00+00:00",
            )
            legacy.apply_memory_event(legacy_recorded)
            explicit.apply_memory_event(explicit_recorded)

            if terminal_event_type == "revoked":
                legacy_terminal = legacy.append_brain_event(
                    event_type=BrainEventType.MEMORY_CLAIM_REVOKED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="property",
                    payload={"claim_id": "claim-legacy-1", "reason": "contradiction"},
                    ts="2026-01-01T00:00:01+00:00",
                )
                explicit_terminal = explicit.append_brain_event(
                    event_type=BrainEventType.MEMORY_CLAIM_REVOKED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="property",
                    payload={
                        "claim_id": "claim-legacy-1",
                        "reason": "contradiction",
                        "reason_codes": [BrainGovernanceReasonCode.CONTRADICTION.value],
                    },
                    ts="2026-01-01T00:00:01+00:00",
                )
            else:
                for store in (legacy, explicit):
                    recorded = store.append_brain_event(
                        event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
                        agent_id=session_ids.agent_id,
                        user_id=session_ids.user_id,
                        session_id=session_ids.session_id,
                        thread_id=session_ids.thread_id,
                        source="property",
                        payload={
                            "claim_id": "claim-legacy-2",
                            "subject_entity_id": "entity-user-1",
                            "predicate": "profile.origin",
                            "object": {"value": "Beijing"},
                            "status": "active",
                            "confidence": 0.85,
                            "valid_from": "2026-01-01T00:00:01+00:00",
                            "scope_type": "user",
                            "scope_id": session_ids.user_id,
                            "claim_key": "profile.origin",
                            "currentness_status": BrainClaimCurrentnessStatus.CURRENT.value,
                            "reason_codes": [],
                        },
                        ts="2026-01-01T00:00:01+00:00",
                    )
                    store.apply_memory_event(recorded)
                legacy_terminal = legacy.append_brain_event(
                    event_type=BrainEventType.MEMORY_CLAIM_SUPERSEDED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="property",
                    payload={
                        "prior_claim_id": "claim-legacy-1",
                        "new_claim_id": "claim-legacy-2",
                        "reason": "corrected origin",
                    },
                    ts="2026-01-01T00:00:02+00:00",
                )
                explicit_terminal = explicit.append_brain_event(
                    event_type=BrainEventType.MEMORY_CLAIM_SUPERSEDED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="property",
                    payload={
                        "prior_claim_id": "claim-legacy-1",
                        "new_claim_id": "claim-legacy-2",
                        "reason": "corrected origin",
                        "reason_codes": [BrainGovernanceReasonCode.SUPERSEDED.value],
                    },
                    ts="2026-01-01T00:00:02+00:00",
                )

            legacy.apply_memory_event(legacy_terminal)
            explicit.apply_memory_event(explicit_terminal)

            assert _normalized_projection(
                legacy.get_claim_governance_projection(
                    scope_type="user",
                    scope_id=session_ids.user_id,
                )
            ) == _normalized_projection(
                explicit.get_claim_governance_projection(
                    scope_type="user",
                    scope_id=session_ids.user_id,
                )
            )
        finally:
            legacy.close()
            explicit.close()

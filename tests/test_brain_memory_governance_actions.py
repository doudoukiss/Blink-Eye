from blink.brain.events import BrainEventType
from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainClaimRetentionClass,
    BrainContinuityQuery,
    ContinuityRetriever,
    apply_memory_governance_action,
    build_memory_palace_snapshot,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def _session(client_id: str = "governance-user"):
    return resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)


def _claim_memory_id(session_ids, claim_id: str) -> str:
    return f"memory_claim:user:{session_ids.user_id}:{claim_id}"


def _task_record_by_title(snapshot, title: str):
    return next(
        record
        for record in snapshot.records
        if record.display_kind == "task" and record.title == title
    )


def _claim_by_value(store: BrainStore, session_ids, predicate: str, value: str):
    for claim in store.query_claims(
        temporal_mode="all",
        predicate=predicate,
        scope_type="user",
        scope_id=session_ids.user_id,
        limit=None,
    ):
        if str(claim.object.get("value", "")).strip() == value:
            return claim
    raise AssertionError(f"Missing claim {predicate}={value}")


def _remember_profile_role(store: BrainStore, session_ids, value: str = "designer"):
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": value},
        rendered_text=f"user role is {value}",
        confidence=0.91,
        singleton=True,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    return _claim_by_value(store, session_ids, "profile.role", value)


def _remember_preference(store: BrainStore, session_ids, value: str = "coffee"):
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject=value.lower(),
        value={"value": value},
        rendered_text=f"user likes {value}",
        confidence=0.86,
        singleton=False,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    return _claim_by_value(store, session_ids, "preference.like", value)


def test_memory_governance_rejects_malformed_unsupported_and_cross_user_ids(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("reject-user")
    claim = _remember_preference(store, session_ids)

    malformed = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id="not-a-memory-id",
        action="forget",
    )
    malformed_task = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=f"memory_task:user:{session_ids.user_id}",
        action="mark_done",
    )
    unsupported = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=f"memory_task:user:{session_ids.user_id}:task_abc",
        action="forget",
    )
    cross_user = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=f"memory_claim:user:someone_else:{claim.claim_id}",
        action="forget",
    )
    bad_action = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="delete",
    )

    assert malformed.accepted is False
    assert "memory_id_malformed" in malformed.reason_codes
    assert malformed_task.record_kind is None
    assert "memory_id_malformed" in malformed_task.reason_codes
    assert unsupported.record_kind == "memory_task"
    assert "action_unsupported" in unsupported.reason_codes
    assert "cross_user_memory_id" in cross_user.reason_codes
    assert "action_unsupported" in bad_action.reason_codes


def test_memory_governance_suppress_hides_claim_from_default_palace(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("suppress-user")
    claim = _remember_preference(store, session_ids)

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="suppress",
        notes="not useful",
    )
    default_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    included_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        include_suppressed=True,
    )

    assert result.accepted is True
    assert result.applied is True
    assert result.event_id
    assert "claim_suppressed" in result.reason_codes
    assert default_snapshot.records == ()
    assert default_snapshot.hidden_counts["suppressed"] == 1
    assert len(included_snapshot.records) == 1
    assert included_snapshot.records[0].suppressed is True
    assert included_snapshot.records[0].user_actions == ("review", "export")


def test_memory_governance_forget_removes_from_palace_and_current_retrieval(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("forget-user")
    claim = _remember_preference(store, session_ids)
    retriever = ContinuityRetriever(store=store)

    before = retriever.retrieve(
        BrainContinuityQuery(
            text="coffee",
            scope_type="user",
            scope_id=session_ids.user_id,
            temporal_mode="current",
            limit=4,
        )
    )
    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="forget",
    )
    after = retriever.retrieve(
        BrainContinuityQuery(
            text="coffee",
            scope_type="user",
            scope_id=session_ids.user_id,
            temporal_mode="current",
            limit=4,
        )
    )
    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    refreshed = store._claims().get_claim(claim.claim_id)

    assert before
    assert result.accepted is True
    assert result.applied is True
    assert result.event_id
    assert "claim_forgotten" in result.reason_codes
    assert refreshed is not None
    assert refreshed.status == "revoked"
    assert refreshed.effective_currentness_status == BrainClaimCurrentnessStatus.HISTORICAL.value
    assert after == []
    assert snapshot.records == ()
    assert snapshot.hidden_counts["historical"] == 1


def test_memory_governance_correction_appends_replacement_and_supersedes_prior(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("correct-user")
    claim = _remember_preference(store, session_ids, "coffee")

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="correct",
        replacement_value="tea",
    )
    refreshed_prior = store._claims().get_claim(claim.claim_id)
    replacement_claim_id = str(result.replacement_memory_id).rsplit(":", 1)[-1]
    replacement = store._claims().get_claim(replacement_claim_id)
    supersessions = store.claim_supersessions(claim_id=claim.claim_id)
    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)

    assert result.accepted is True
    assert result.applied is True
    assert "claim_corrected" in result.reason_codes
    assert refreshed_prior is not None
    assert refreshed_prior.object["value"] == "coffee"
    assert refreshed_prior.status == "superseded"
    assert replacement is not None
    assert replacement.object["value"] == "tea"
    assert replacement.status == "active"
    assert any(record.new_claim_id == replacement.claim_id for record in supersessions)
    assert [record.summary for record in snapshot.records] == ["User likes tea"]


def test_memory_governance_mark_stale_keeps_claim_visible_as_stale(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("stale-user")
    claim = _remember_profile_role(store, session_ids)

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="mark-stale",
    )
    refreshed = store._claims().get_claim(claim.claim_id)
    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)

    assert result.accepted is True
    assert result.applied is True
    assert refreshed is not None
    assert refreshed.effective_currentness_status == BrainClaimCurrentnessStatus.STALE.value
    assert snapshot.records[0].currentness_status == "stale"
    assert "mark_stale" not in snapshot.records[0].user_actions


def test_memory_governance_pin_records_durable_retention_transition(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("pin-user")
    claim = _remember_profile_role(store, session_ids)
    store._claims().reclassify_claim_retention(
        claim.claim_id,
        retention_class=BrainClaimRetentionClass.DURABLE.value,
        source_event_id=None,
    )

    default_durable_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
    )
    default_durable_record = default_durable_snapshot.records[0]

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="pin",
    )
    refreshed = store._claims().get_claim(claim.claim_id)
    pinned_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    pinned_record = pinned_snapshot.records[0]
    repeat = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, claim.claim_id),
        action="pin",
    )

    assert default_durable_record.pinned is False
    assert "pin_source:default_policy" in default_durable_record.reason_codes
    assert "pin" in default_durable_record.user_actions
    assert result.accepted is True
    assert result.applied is True
    assert result.event_id
    assert "pin_source:user" in result.reason_codes
    assert refreshed is not None
    assert refreshed.effective_retention_class == BrainClaimRetentionClass.DURABLE.value
    assert "user_pinned" in refreshed.governance_reason_codes
    assert refreshed.last_governance_event_id == result.event_id
    assert pinned_record.pinned is True
    assert "pin_source:user" in pinned_record.reason_codes
    assert "pin" not in pinned_record.user_actions
    assert "unpin" not in pinned_record.user_actions
    assert repeat.accepted is True
    assert repeat.applied is False
    assert "claim_already_pinned" in repeat.reason_codes


def test_memory_governance_task_mark_done_via_scoped_memory_id(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("task-done-user")
    store.upsert_task(
        user_id=session_ids.user_id,
        title="Send project recap",
        details={"summary": "Send a compact project recap."},
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source_event_id="evt-task-done",
    )
    task_record = _task_record_by_title(
        build_memory_palace_snapshot(store=store, session_ids=session_ids),
        "Send project recap",
    )

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=task_record.memory_id,
        action="mark-done",
    )
    snapshot_after = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=32,
    )

    assert task_record.user_actions == ("review", "mark_done", "cancel", "export")
    assert result.accepted is True
    assert result.applied is True
    assert result.record_kind == "memory_task"
    assert result.event_id
    assert "task_scope_validated" in result.reason_codes
    assert "task_ref:commitment" in result.reason_codes
    assert "task_marked_done" in result.reason_codes
    assert all(record.display_kind != "task" for record in snapshot_after.records)
    assert any(event.event_type == BrainEventType.GOAL_COMPLETED for event in events)


def test_memory_governance_task_cancel_via_scoped_memory_id(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("task-cancel-user")
    store.upsert_task(
        user_id=session_ids.user_id,
        title="Draft obsolete reminder",
        details={"summary": "Draft an obsolete reminder."},
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source_event_id="evt-task-cancel",
    )
    task_record = _task_record_by_title(
        build_memory_palace_snapshot(store=store, session_ids=session_ids),
        "Draft obsolete reminder",
    )

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=task_record.memory_id,
        action="cancel",
    )
    snapshot_after = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=32,
    )

    assert result.accepted is True
    assert result.applied is True
    assert result.record_kind == "memory_task"
    assert "task_cancelled" in result.reason_codes
    assert all(record.display_kind != "task" for record in snapshot_after.records)
    assert any(event.event_type == BrainEventType.GOAL_CANCELLED for event in events)


def test_memory_governance_task_rejects_cross_user_missing_and_unbacked_ids(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_a = _session("task-owner-a")
    session_b = _session("task-owner-b")
    store.upsert_task(
        user_id=session_a.user_id,
        title="Owner-only task",
        thread_id=session_a.thread_id,
        agent_id=session_a.agent_id,
        session_id=session_a.session_id,
    )
    task_record = _task_record_by_title(
        build_memory_palace_snapshot(store=store, session_ids=session_a),
        "Owner-only task",
    )

    cross_user = apply_memory_governance_action(
        store=store,
        session_ids=session_b,
        memory_id=task_record.memory_id,
        action="mark_done",
    )
    missing = apply_memory_governance_action(
        store=store,
        session_ids=session_a,
        memory_id=f"memory_task:user:{session_a.user_id}:missing-task",
        action="mark_done",
    )
    unsupported = apply_memory_governance_action(
        store=store,
        session_ids=session_a,
        memory_id=task_record.memory_id,
        action="pin",
    )

    class LegacyTaskStore:
        def query_claims(self, **_kwargs):
            return []

        def active_tasks(self, *, user_id: str, limit: int = 8):
            return [
                {
                    "title": "Legacy task",
                    "details": {"summary": "Legacy task."},
                    "status": "open",
                    "created_at": "2026-04-23T00:00:00+00:00",
                    "updated_at": "2026-04-23T00:00:00+00:00",
                }
            ][:limit]

        def get_current_core_memory_block(self, **_kwargs):
            return None

        def latest_memory_health_report(self, **_kwargs):
            return None

    legacy_store = LegacyTaskStore()
    legacy_record = _task_record_by_title(
        build_memory_palace_snapshot(store=legacy_store, session_ids=session_a),
        "Legacy task",
    )
    unbacked = apply_memory_governance_action(
        store=legacy_store,
        session_ids=session_a,
        memory_id=legacy_record.memory_id,
        action="mark_done",
    )

    assert "cross_user_memory_id" in cross_user.reason_codes
    assert "task_not_found" in missing.reason_codes
    assert unsupported.record_kind == "memory_task"
    assert "action_unsupported" in unsupported.reason_codes
    assert legacy_record.user_actions == ("review", "export")
    assert "task_not_actionable" in unbacked.reason_codes
    assert "task_commitment_missing" in unbacked.reason_codes


def test_memory_governance_rejects_system_and_agent_block_ids(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = _session("block-user")

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id="memory_block:agent:blink/main:self_persona_core:1",
        action="forget",
    )

    assert result.accepted is False
    assert result.record_kind == "memory_block"
    assert "record_kind_unsupported" in result.reason_codes


def test_memory_governance_replay_reconstructs_suppression_and_correction(tmp_path):
    store = BrainStore(path=tmp_path / "source.db")
    replayed = BrainStore(path=tmp_path / "replayed.db")
    session_ids = _session("replay-user")
    role = _remember_profile_role(store, session_ids, "designer")
    preference = _remember_preference(store, session_ids, "coffee")

    correction = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, role.claim_id),
        action="correct",
        replacement_value="engineer",
    )
    suppression = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=_claim_memory_id(session_ids, preference.claim_id),
        action="suppress",
    )
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        after_id=0,
        limit=128,
    )

    for event in events:
        replayed.import_brain_event(event)
        replayed.apply_memory_event(event)

    replayed_snapshot = build_memory_palace_snapshot(store=replayed, session_ids=session_ids)
    replayed_hidden_snapshot = build_memory_palace_snapshot(
        store=replayed,
        session_ids=session_ids,
        include_suppressed=True,
    )
    replayed_role = _claim_by_value(replayed, session_ids, "profile.role", "engineer")
    replayed_supersessions = replayed.claim_supersessions(claim_id=role.claim_id)

    assert correction.accepted is True
    assert suppression.accepted is True
    assert "User role is engineer" in [record.summary for record in replayed_snapshot.records]
    assert replayed_snapshot.hidden_counts["suppressed"] == 1
    assert any(record.new_claim_id == replayed_role.claim_id for record in replayed_supersessions)
    assert any(record.summary == "User likes coffee" for record in replayed_hidden_snapshot.records)


def test_memory_governance_replay_reconstructs_task_completion(tmp_path):
    store = BrainStore(path=tmp_path / "source-tasks.db")
    replayed = BrainStore(path=tmp_path / "replayed-tasks.db")
    session_ids = _session("replay-task-user")
    store.upsert_task(
        user_id=session_ids.user_id,
        title="Replayable task",
        details={"summary": "Replayable task."},
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source_event_id="evt-replay-task",
    )
    task_record = _task_record_by_title(
        build_memory_palace_snapshot(store=store, session_ids=session_ids),
        "Replayable task",
    )

    result = apply_memory_governance_action(
        store=store,
        session_ids=session_ids,
        memory_id=task_record.memory_id,
        action="mark_done",
    )
    events = store.brain_events_since(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=64,
    )

    for event in events:
        replayed.import_brain_event(event)
        replayed.apply_executive_event(event)

    replayed_snapshot = build_memory_palace_snapshot(store=replayed, session_ids=session_ids)

    assert result.accepted is True
    assert result.applied is True
    assert any(event.event_type == BrainEventType.GOAL_CREATED for event in events)
    assert any(event.event_type == BrainEventType.GOAL_COMPLETED for event in events)
    assert all(record.display_kind != "task" for record in replayed_snapshot.records)

from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory_v2 import (
    BrainClaimCurrentnessStatus,
    BrainCoreMemoryBlockKind,
    BrainGovernanceReasonCode,
    BrainMemoryUseTraceRef,
    ClaimLedger,
    build_memory_palace_snapshot,
    build_memory_use_trace,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def _relationship_scope_id(session_ids) -> str:
    return f"{session_ids.agent_id}:{session_ids.user_id}"


def _user_entity(store: BrainStore, user_id: str):
    return store.ensure_entity(
        entity_type="user",
        canonical_name=user_id,
        aliases=[user_id],
        attributes={"user_id": user_id},
    )


def _relationship_style_payload(relationship_id: str) -> dict:
    return {
        "schema_version": 1,
        "relationship_id": relationship_id,
        "default_posture": [
            "warm",
            "intelligent",
            "collaborative",
            "non-romantic",
            "non-sexual",
            "non-exclusive",
        ],
        "collaboration_style": "warm concise collaboration",
        "emotional_tone_preference": "warm precise",
        "intimacy_ceiling": "warm professional companion",
        "challenge_style": "gentle directness",
        "humor_permissiveness": 0.2,
        "self_disclosure_policy": "no fabricated personal history",
        "dependency_guardrails": [
            "avoid guilt language",
            "avoid exclusivity",
            "encourage human support when appropriate",
        ],
        "boundaries": ["non-romantic", "non-sexual", "non-exclusive"],
        "known_misfires": ["too much preamble"],
        "interaction_style_hints": ["User prefers concise collaboration."],
        "source_namespaces": ["interaction.style"],
    }


def _teaching_profile_payload(relationship_id: str) -> dict:
    return {
        "schema_version": 1,
        "relationship_id": relationship_id,
        "default_mode": "clarify",
        "preferred_modes": ["walkthrough", "clarify"],
        "question_frequency": 0.32,
        "example_density": 0.76,
        "correction_style": "gentle precise correction",
        "grounding_policy": "state uncertainty instead of bluffing",
        "analogy_domains": ["physics"],
        "helpful_patterns": ["stepwise decomposition"],
        "source_namespaces": ["teaching.preference.mode"],
    }


def _record_by_kind(snapshot, kind: str):
    return next(record for record in snapshot.records if record.display_kind == kind)


def test_memory_palace_empty_store_is_stable_and_compact(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="empty-palace")

    first = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    second = build_memory_palace_snapshot(store=store, session_ids=session_ids)

    assert first.records == ()
    assert first.generated_at == ""
    assert first.health_summary == "Memory health unavailable."
    assert first.hidden_counts == {"suppressed": 0, "historical": 0, "limit": 0}
    assert first.as_dict() == second.as_dict()


def test_memory_palace_claim_scan_limit_is_forwarded_to_store():
    class LimitedStore:
        def __init__(self):
            self.query_calls = []

        def query_claims(self, **kwargs):
            self.query_calls.append(kwargs)
            return []

        def get_current_core_memory_block(self, **_kwargs):
            return None

    store = LimitedStore()
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="claim-scan-limit")

    snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        limit=40,
        claim_scan_limit=160,
    )

    assert store.query_calls[0]["limit"] == 160
    assert "claims_scan:bounded" in snapshot.reason_codes
    assert "claims_scan_limit:160" in snapshot.reason_codes


def test_memory_palace_includes_profile_and_preference_claims(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="claims-palace")

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "product manager"},
        rendered_text="user role is product manager",
        confidence=0.91,
        singleton=True,
        source_event_id="evt-role",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user likes coffee",
        confidence=0.82,
        singleton=False,
        source_event_id="evt-coffee",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )

    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    profile = _record_by_kind(snapshot, "profile")
    preference = _record_by_kind(snapshot, "preference")

    assert profile.scope_id == session_ids.user_id
    assert "product manager" in profile.summary
    assert profile.source_event_ids == ("evt-role",)
    assert profile.memory_id.startswith(f"memory_claim:user:{session_ids.user_id}:claim_")
    assert {"correct", "forget", "mark_stale"}.issubset(set(profile.user_actions))
    assert preference.summary == "User likes coffee"
    assert preference.confidence == 0.82
    assert preference.safe_provenance_label == "Remembered from your explicit preference."
    assert preference.used_in_current_turn is False


def test_memory_palace_populates_last_used_from_trace_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="used-palace")
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user likes coffee",
        confidence=0.82,
        singleton=False,
        source_event_id="evt-coffee",
        source_episode_id=None,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    base_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    memory_id = base_snapshot.records[0].memory_id
    trace = build_memory_use_trace(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
        thread_id=session_ids.thread_id,
        task="reply",
        selected_section_names=("relevant_continuity",),
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=memory_id,
                display_kind="preference",
                title="coffee",
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from your explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
    )
    persisted = store.append_memory_use_trace(
        trace=trace,
        session_id=session_ids.session_id,
        source="test",
        ts="2026-04-23T01:02:03+00:00",
    )

    persisted_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    current_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        current_turn_trace=persisted,
    )

    persisted_record = persisted_snapshot.records[0]
    current_record = current_snapshot.records[0]
    assert persisted_record.last_used_at == "2026-04-23T01:02:03+00:00"
    assert persisted_record.last_used_reason == "selected_for_relevant_continuity"
    assert persisted_record.used_in_current_turn is False
    assert current_record.used_in_current_turn is True
    assert current_record.safe_provenance_label == "Remembered from your explicit preference."


def test_memory_palace_stale_claims_visible_and_historical_opt_in(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="stale-palace")
    user = _user_entity(store, session_ids.user_id)
    ledger = ClaimLedger(store=store)

    stale = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_value="designer",
        scope_type="user",
        scope_id=session_ids.user_id,
        source_event_id="evt-stale",
        currentness_status=BrainClaimCurrentnessStatus.STALE.value,
    )
    historical = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.origin",
        object_value="old city",
        scope_type="user",
        scope_id=session_ids.user_id,
        source_event_id="evt-historical",
        currentness_status=BrainClaimCurrentnessStatus.HISTORICAL.value,
    )

    current_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    historical_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        include_historical=True,
    )

    assert {record.source_refs[0] for record in current_snapshot.records} == {stale.claim_id}
    assert current_snapshot.records[0].currentness_status == "stale"
    assert current_snapshot.hidden_counts["historical"] == 1
    assert {record.source_refs[0] for record in historical_snapshot.records} == {
        stale.claim_id,
        historical.claim_id,
    }


def test_memory_palace_relationship_and_teaching_blocks_are_compact(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="blocks-palace")
    relationship_scope_id = _relationship_scope_id(session_ids)

    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
        content=_relationship_style_payload(relationship_scope_id),
        source_event_id="evt-relationship-style",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
        content=_teaching_profile_payload(relationship_scope_id),
        source_event_id="evt-teaching-profile",
    )

    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    relationship = _record_by_kind(snapshot, "relationship_style")
    teaching = _record_by_kind(snapshot, "teaching_profile")

    assert "warm concise collaboration" in relationship.summary
    assert "non-romantic" in relationship.summary
    assert "walkthrough" in teaching.summary
    assert "examples=0.76" in teaching.summary
    assert "{" not in relationship.summary
    assert relationship.source_event_ids == ("evt-relationship-style",)
    assert teaching.source_event_ids == ("evt-teaching-profile",)


def test_memory_palace_active_tasks_use_scoped_stable_ids(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="tasks-palace")

    store.upsert_task(
        user_id=session_ids.user_id,
        title="Review memory health",
        details={"summary": "Review the local memory health report."},
        status="open",
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source_event_id="evt-task",
    )

    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    task = _record_by_kind(snapshot, "task")

    assert task.title == "Review memory health"
    assert task.summary == "Review memory health"
    assert task.memory_id.startswith(f"memory_task:user:{session_ids.user_id}:")
    assert task.memory_id != "1"
    assert task.user_actions == ("review", "mark_done", "cancel", "export")
    assert all(ref != "1" for ref in task.source_refs)


def test_memory_palace_legacy_tasks_do_not_advertise_lifecycle_actions():
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="legacy-task")

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

    snapshot = build_memory_palace_snapshot(store=LegacyTaskStore(), session_ids=session_ids)
    task = _record_by_kind(snapshot, "task")

    assert task.memory_id.startswith(f"memory_task:user:{session_ids.user_id}:task_")
    assert task.user_actions == ("review", "export")
    assert "mark_done" not in task.user_actions
    assert "cancel" not in task.user_actions


def test_memory_palace_does_not_advertise_unsupported_actions(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="actions-palace")
    relationship_scope_id = _relationship_scope_id(session_ids)
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="tea",
        value={"value": "tea"},
        rendered_text="user likes tea",
        confidence=0.8,
        singleton=False,
        source_event_id="evt-tea",
        source_episode_id=None,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.upsert_task(
        user_id=session_ids.user_id,
        title="Governed task",
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
        content=_relationship_style_payload(relationship_scope_id),
        source_event_id="evt-relationship-style",
    )

    snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    actions_by_kind = {record.display_kind: record.user_actions for record in snapshot.records}

    assert "unpin" not in actions_by_kind["preference"]
    assert "mark_done" not in actions_by_kind["preference"]
    assert "cancel" not in actions_by_kind["preference"]
    assert "pin" not in actions_by_kind["task"]
    assert actions_by_kind["task"] == ("review", "mark_done", "cancel", "export")
    assert actions_by_kind["relationship_style"] == ("review", "export")


def test_memory_palace_suppressed_claims_excluded_by_default(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="hidden-palace")
    user = _user_entity(store, session_ids.user_id)
    ledger = ClaimLedger(store=store)

    hidden = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.secret",
        object_value="private detail",
        scope_type="user",
        scope_id=session_ids.user_id,
        source_event_id="evt-hidden",
        reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
    )

    default_snapshot = build_memory_palace_snapshot(store=store, session_ids=session_ids)
    included_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        include_suppressed=True,
    )

    assert default_snapshot.records == ()
    assert default_snapshot.hidden_counts["suppressed"] == 1
    assert len(included_snapshot.records) == 1
    assert included_snapshot.records[0].source_refs == (hidden.claim_id,)
    assert included_snapshot.records[0].suppressed is True
    assert included_snapshot.records[0].user_actions == ("review", "export")


def test_memory_palace_ordering_limit_and_health_are_deterministic(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="ordered-palace")
    store.ensure_default_blocks(load_default_agent_blocks())

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="tea",
        value={"value": "tea"},
        rendered_text="user likes tea",
        confidence=0.8,
        singleton=False,
        source_event_id="evt-tea",
        source_episode_id=None,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.record_memory_health_report(
        scope_type="user",
        scope_id=session_ids.user_id,
        cycle_id="cycle-1",
        score=0.87,
        status="healthy",
        findings=[{"severity": "info", "summary": "No issues."}],
        stats={"claims": 1, "blocks": 0},
    )

    first = build_memory_palace_snapshot(store=store, session_ids=session_ids, limit=1)
    second = build_memory_palace_snapshot(store=store, session_ids=session_ids, limit=1)

    assert first.as_dict() == second.as_dict()
    assert len(first.records) == 1
    assert first.hidden_counts["limit"] >= 0
    assert "Memory health healthy; score=0.87" in first.health_summary
    assert "memory_health:available" in first.reason_codes
    assert first.generated_at


def test_memory_palace_isolates_users_and_relationship_scopes(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_a = resolve_brain_session_ids(runtime_kind="browser", client_id="palace-user-a")
    session_b = resolve_brain_session_ids(runtime_kind="browser", client_id="palace-user-b")

    store.remember_fact(
        user_id=session_a.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "designer"},
        rendered_text="user role is designer",
        confidence=0.9,
        singleton=True,
        source_event_id="evt-a",
        source_episode_id=None,
        agent_id=session_a.agent_id,
        session_id=session_a.session_id,
        thread_id=session_a.thread_id,
    )
    store.remember_fact(
        user_id=session_b.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "engineer"},
        rendered_text="user role is engineer",
        confidence=0.9,
        singleton=True,
        source_event_id="evt-b",
        source_episode_id=None,
        agent_id=session_b.agent_id,
        session_id=session_b.session_id,
        thread_id=session_b.thread_id,
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        scope_type="relationship",
        scope_id=_relationship_scope_id(session_b),
        content=_relationship_style_payload(_relationship_scope_id(session_b)),
        source_event_id="evt-b-relationship",
    )
    store.upsert_task(
        user_id=session_b.user_id,
        title="Only user B task",
        thread_id=session_b.thread_id,
        agent_id=session_b.agent_id,
        session_id=session_b.session_id,
    )

    snapshot_a = build_memory_palace_snapshot(store=store, session_ids=session_a)
    encoded_a = str(snapshot_a.as_dict())

    assert "designer" in encoded_a
    assert "engineer" not in encoded_a
    assert "Only user B task" not in encoded_a
    assert "relationship_style" not in {record.display_kind for record in snapshot_a.records}


def test_memory_palace_ignores_cross_user_use_traces(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_a = resolve_brain_session_ids(runtime_kind="browser", client_id="trace-user-a")
    session_b = resolve_brain_session_ids(runtime_kind="browser", client_id="trace-user-b")
    store.remember_fact(
        user_id=session_a.user_id,
        namespace="preference.like",
        subject="tea",
        value={"value": "tea"},
        rendered_text="user likes tea",
        confidence=0.8,
        singleton=False,
        source_event_id="evt-a",
        source_episode_id=None,
        agent_id=session_a.agent_id,
        session_id=session_a.session_id,
        thread_id=session_a.thread_id,
    )
    snapshot_a = build_memory_palace_snapshot(store=store, session_ids=session_a)
    cross_user_trace = build_memory_use_trace(
        user_id=session_b.user_id,
        agent_id=session_b.agent_id,
        thread_id=session_b.thread_id,
        task="reply",
        refs=(
            BrainMemoryUseTraceRef(
                memory_id=snapshot_a.records[0].memory_id,
                display_kind="preference",
                title="tea",
                section_key="relevant_continuity",
                used_reason="selected_for_relevant_continuity",
                safe_provenance_label="Remembered from your explicit preference.",
                reason_codes=("source:context_selection",),
            ),
        ),
        created_at="2026-04-23T01:02:03+00:00",
    )

    isolated_snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_a,
        current_turn_trace=cross_user_trace,
        recent_use_traces=(cross_user_trace,),
    )

    assert isolated_snapshot.records[0].used_in_current_turn is False
    assert isolated_snapshot.records[0].last_used_at is None
    assert "memory_use_trace_current_scope_mismatch" in isolated_snapshot.reason_codes

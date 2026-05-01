import json

from blink.brain.core import BrainCoreStore, BrainEventType
from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory import (
    BrainConsolidationResult,
    BrainMemoryConsolidator,
    extract_memory_candidates,
    extract_task_candidates,
)
from blink.brain.memory_v2 import BrainCoreMemoryBlockKind, ClaimLedger
from blink.brain.store import BrainStore


def test_brain_stores_configure_sqlite_for_local_runtime_contention(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    core_store = BrainCoreStore(path=tmp_path / "core.db")
    try:
        assert (
            store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            >= BrainStore.SQLITE_BUSY_TIMEOUT_MS
        )
        assert (
            core_store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            >= BrainCoreStore.SQLITE_BUSY_TIMEOUT_MS
        )
        assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert core_store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        store.close()
        core_store.close()


def test_brain_store_remembers_and_supersedes_singleton_facts(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    for candidate in extract_memory_candidates("我叫小周，我喜欢机器人。"):
        store.remember_fact(
            user_id="user-1",
            namespace=candidate.namespace,
            subject=candidate.subject,
            value=candidate.value,
            rendered_text=candidate.rendered_text,
            confidence=candidate.confidence,
            singleton=candidate.singleton,
            source_episode_id=None,
        )
    for candidate in extract_memory_candidates("请叫我阿周。"):
        store.remember_fact(
            user_id="user-1",
            namespace=candidate.namespace,
            subject=candidate.subject,
            value=candidate.value,
            rendered_text=candidate.rendered_text,
            confidence=candidate.confidence,
            singleton=candidate.singleton,
            source_episode_id=None,
        )

    facts = store.active_facts(user_id="user-1", limit=10)
    rendered = [fact.rendered_text for fact in facts]

    assert "用户名字是 阿周" in rendered
    assert "用户喜欢 机器人" in rendered
    assert "用户名字是 小周" not in rendered


def test_brain_memory_consolidator_persists_thread_summary(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.add_episode(
        agent_id="blink/main",
        user_id="user-1",
        session_id="browser:user-1",
        thread_id="browser:user-1",
        user_text="我叫阿周。",
        assistant_text="好的，我记住了。",
        assistant_summary="好的，我记住了。",
        tool_calls=[],
    )
    store.add_episode(
        agent_id="blink/main",
        user_id="user-1",
        session_id="browser:user-1",
        thread_id="browser:user-1",
        user_text="我喜欢机器人。",
        assistant_text="收到，你喜欢机器人。",
        assistant_summary="收到，你喜欢机器人。",
        tool_calls=[],
    )

    consolidator = BrainMemoryConsolidator(store=store)
    consolidator.run_once(user_id="user-1", thread_id="browser:user-1")

    summary = store.get_session_summary(user_id="user-1", thread_id="browser:user-1")

    assert "我叫阿周" in summary
    assert "你喜欢机器人" in summary


def test_brain_memory_consolidator_promotes_episode_memory_only_once(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.add_episode(
        agent_id="blink/main",
        user_id="user-1",
        session_id="browser:user-1",
        thread_id="browser:user-1",
        user_text="我叫阿周。提醒我给妈妈打电话。",
        assistant_text="好的，我会记住。",
        assistant_summary="好的，我会记住。",
        tool_calls=[],
    )
    store.add_episode(
        agent_id="blink/main",
        user_id="user-1",
        session_id="browser:user-1",
        thread_id="browser:user-1",
        user_text="我不喜欢咖啡。",
        assistant_text="收到，你不喜欢咖啡。",
        assistant_summary="收到，你不喜欢咖啡。",
        tool_calls=[],
    )

    consolidator = BrainMemoryConsolidator(store=store)
    first = consolidator.run_once(user_id="user-1", thread_id="browser:user-1")
    second = consolidator.run_once(user_id="user-1", thread_id="browser:user-1")

    facts = store.active_facts(user_id="user-1", limit=10)
    rendered = [fact.rendered_text for fact in facts]
    tasks = store.active_tasks(user_id="user-1", limit=10)

    assert isinstance(first, BrainConsolidationResult)
    assert first.promoted_facts == 2
    assert first.upserted_tasks == 1
    assert second.promoted_facts == 0
    assert second.upserted_tasks == 0
    assert "用户名字是 阿周" in rendered
    assert "用户不喜欢 咖啡" in rendered
    assert [task["title"] for task in tasks] == ["给妈妈打电话"]
    assert (
        store.get_metadata("consolidator:last_episode_id:user-1:browser:user-1")
        == "2"
    )


def test_brain_store_tracks_explicit_tasks_per_user(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    for candidate in extract_task_candidates("提醒我给妈妈打电话。"):
        store.upsert_task(
            user_id="user-1",
            title=candidate.title,
            details=candidate.details,
            status=candidate.status,
        )
    for candidate in extract_task_candidates("todo finish the robot checklist"):
        store.upsert_task(
            user_id="user-2",
            title=candidate.title,
            details=candidate.details,
            status=candidate.status,
        )

    user_one_tasks = store.active_tasks(user_id="user-1")
    user_two_tasks = store.active_tasks(user_id="user-2")

    assert [task["title"] for task in user_one_tasks] == ["给妈妈打电话"]
    assert [task["title"] for task in user_two_tasks] == ["finish the robot checklist"]


def test_brain_store_supersedes_conflicting_preferences(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    for candidate in extract_memory_candidates("我喜欢咖啡。"):
        store.remember_fact(
            user_id="user-1",
            namespace=candidate.namespace,
            subject=candidate.subject,
            value=candidate.value,
            rendered_text=candidate.rendered_text,
            confidence=candidate.confidence,
            singleton=candidate.singleton,
            source_episode_id=None,
        )
    for candidate in extract_memory_candidates("我不喜欢咖啡。"):
        store.remember_fact(
            user_id="user-1",
            namespace=candidate.namespace,
            subject=candidate.subject,
            value=candidate.value,
            rendered_text=candidate.rendered_text,
            confidence=candidate.confidence,
            singleton=candidate.singleton,
            source_episode_id=None,
        )

    facts = store.active_facts(user_id="user-1", limit=10)
    rendered = [fact.rendered_text for fact in facts]

    assert "用户不喜欢 咖啡" in rendered
    assert "用户喜欢 咖啡" not in rendered


def test_brain_store_search_falls_back_for_column_like_fts_query(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.remember_semantic_memory(
        user_id="user-1",
        namespace="planning.query",
        subject="maintenance",
        value={"value": "memory maintenance review"},
        rendered_text="Planning query: memory_maintenance: Prepare maintenance review",
        confidence=0.8,
        singleton=False,
        source_event_id=None,
    )

    results = store.search_semantic_memories(
        user_id="user-1",
        text="memory_maintenance: Prepare maintenance review",
        limit=5,
    )

    assert [result.summary for result in results] == [
        "Planning query: memory_maintenance: Prepare maintenance review"
    ]


def test_brain_store_remains_a_compatibility_superset_over_core(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    assert isinstance(store, BrainCoreStore)

    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id="blink/main",
        user_id="user-1",
        session_id="browser:user-1",
        thread_id="browser:user-1",
        source="context",
        payload={"text": "提醒我给妈妈打电话。"},
    )
    store.remember_fact(
        user_id="user-1",
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.9,
        singleton=True,
        source_episode_id=None,
    )

    assert store.get_working_context_projection(scope_key="browser:user-1").last_user_text == "提醒我给妈妈打电话。"
    assert [fact.rendered_text for fact in store.active_facts(user_id="user-1")] == ["用户名字是 阿周"]


def test_brain_store_fact_reads_prefer_continuity_over_legacy_semantic(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.remember_semantic_memory(
        user_id="user-1",
        namespace="profile.name",
        subject="user",
        value={"value": "小周"},
        rendered_text="用户名字是 小周",
        confidence=0.8,
        singleton=True,
        source_event_id="legacy-name-1",
        source_episode_id=None,
        provenance={"source": "legacy"},
    )
    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name="user-1",
        aliases=["user-1"],
        attributes={"user_id": "user-1"},
    )
    ClaimLedger(store=store).record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.name",
        object_value="阿周",
        object_data={"value": "阿周"},
        status="active",
        confidence=0.95,
        source_event_id="continuity-name-1",
        scope_type="user",
        scope_id="user-1",
    )

    active = store.active_facts(user_id="user-1", limit=8)
    relevant = store.relevant_facts(user_id="user-1", query="阿周", limit=4)

    assert [fact.rendered_text for fact in active] == ["用户名字是 阿周"]
    assert [fact.rendered_text for fact in relevant] == ["用户名字是 阿周"]


def test_brain_store_summary_and_tasks_prefer_continuity_blocks(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    store.upsert_narrative_memory(
        user_id="user-1",
        thread_id="browser:user-1",
        kind="session_summary",
        title="browser:user-1",
        summary="旧摘要",
        details={"thread_id": "browser:user-1"},
        status="active",
        confidence=0.8,
        source_event_id="legacy-summary-1",
        provenance={"source": "legacy"},
    )
    store._conn.execute(
        """
        INSERT INTO tasks (user_id, title, details_json, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "user-1",
            "旧任务",
            "{}",
            "open",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    store._conn.commit()
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
        scope_type="relationship",
        scope_id="blink/main:user-1",
        content={
            "thread_id": "browser:user-1",
            "last_session_summary": "新摘要",
            "open_commitments": ["新任务"],
        },
        source_event_id="evt-relationship-core-1",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
        scope_type="relationship",
        scope_id="blink/main:user-1",
        content={
            "commitments": [
                {
                    "title": "新任务",
                    "summary": "新任务",
                    "details": {},
                    "status": "active",
                }
            ]
        },
        source_event_id="evt-active-commitments-1",
    )

    assert store.get_session_summary(user_id="user-1", thread_id="browser:user-1") == "新摘要"
    assert [task["title"] for task in store.active_tasks(user_id="user-1", limit=4)] == ["新任务"]


def test_brain_store_bounds_new_brain_event_payloads(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    event = store.append_brain_event(
        event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
        agent_id="blink/main",
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        source="test",
        payload={"text": "x" * 400_000},
    )

    assert len(event.payload_json) < 3_000
    assert len(event.payload["text"]) == 2_048


def test_brain_store_sanitizes_missing_event_confidence(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    event = store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id="blink/main",
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        source="test",
        payload={"confidence": None},
        confidence=None,  # type: ignore[arg-type]
    )

    assert event.confidence == 0.0


def test_brain_store_bounds_new_claim_objects(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    user_entity = store._entities().ensure_entity(
        entity_type="user",
        canonical_name="user-1",
        aliases=["user-1"],
        attributes={"user_id": "user-1"},
    )

    claim = ClaimLedger(store=store).record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.name",
        object_data={
            "value": "阿周",
            "candidates": [{"value": "阿周", "detail": "x" * 20_000} for _ in range(20)],
        },
        status="uncertain",
        confidence=0.5,
        scope_type="user",
        scope_id="user-1",
    )

    assert len(claim.object_json) <= 8_192
    assert claim.object["value"] == "阿周"
    assert len(claim.object["candidates"]) <= 13
    assert claim.object["candidates"][-1]["reason_codes"] == ["json_item_limit"]


def test_brain_store_quarantines_historical_oversized_payloads_on_open(tmp_path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    event = store.append_brain_event(
        event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
        agent_id="blink/main",
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        source="test",
        payload={"text": "ok"},
    )
    user_entity = store._entities().ensure_entity(
        entity_type="user",
        canonical_name="user-1",
        aliases=["user-1"],
        attributes={"user_id": "user-1"},
    )
    claim = ClaimLedger(store=store).record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.name",
        object_value="阿周",
        object_data={"value": "阿周"},
        status="active",
        confidence=0.95,
        scope_type="user",
        scope_id="user-1",
    )
    store._conn.execute(
        "UPDATE brain_events SET payload_json = ? WHERE event_id = ?",
        (json.dumps({"text": "x" * 400_000}), event.event_id),
    )
    store._conn.execute(
        "UPDATE claims SET object_json = ? WHERE claim_id = ?",
        (json.dumps({"value": "阿周", "candidates": ["x" * 400_000]}), claim.claim_id),
    )
    store._conn.commit()
    store.close()

    repaired = BrainStore(path=db_path)
    repaired_event = repaired.latest_brain_event(
        user_id="user-1",
        thread_id="thread-1",
        event_types=(BrainEventType.MEMORY_CLAIM_RECORDED,),
    )
    repaired_claim = ClaimLedger(store=repaired).get_claim(claim.claim_id)

    assert repaired_event is not None
    assert repaired_event.payload["_payload_status"] == "quarantined_oversize"
    assert repaired_event.payload["reason_codes"] == ["brain_event_payload_too_large"]
    assert repaired_claim is not None
    assert repaired_claim.status == "revoked"
    assert repaired_claim.effective_currentness_status == "historical"
    assert repaired_claim.object["_object_status"] == "quarantined_oversize"
    assert repaired.get_metadata("brain:oversized_payload_quarantine_counts") == json.dumps(
        {"brain_events": 1, "claims": 1},
        ensure_ascii=False,
        sort_keys=True,
    )

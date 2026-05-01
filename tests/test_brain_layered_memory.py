import json

from blink.brain.events import BrainEventType
from blink.brain.memory import BrainMemoryConsolidator
from blink.brain.memory_layers.exports import BrainMemoryExporter
from blink.brain.memory_layers.retrieval import BrainMemoryQuery, BrainMemoryRetriever
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def test_brain_layered_memory_consolidates_events_into_semantic_narrative_and_episodic_layers(
    tmp_path,
):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")

    turn_event = store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "我叫阿周，我来自上海。提醒我给妈妈打电话。"},
    )
    vision_event = store.append_brain_event(
        event_type=BrainEventType.TOOL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        correlation_id="call_vision_1",
        payload={
            "tool_call_id": "call_vision_1",
            "function_name": "fetch_user_image",
            "result": {"description": "你手里拿着一个红色杯子。"},
        },
    )
    assistant_event = store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        payload={"text": "好的，我记住了。"},
    )

    result = BrainMemoryConsolidator(store=store).run_once(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )

    semantic = store.semantic_memories(user_id=session_ids.user_id, limit=10)
    episodic = store.episodic_memories(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=10,
    )
    narrative = store.narrative_memories(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=10,
    )

    assert result.promoted_facts >= 2
    assert result.upserted_tasks == 1
    assert result.latest_event_id == assistant_event.id
    assert any(record.namespace == "profile.name" and record.rendered_text == "用户名字是 阿周" for record in semantic)
    assert any(record.namespace == "profile.origin" and record.rendered_text == "用户来自 上海" for record in semantic)
    assert any(record.kind == "commitment" and record.title == "给妈妈打电话" for record in narrative)
    assert any(record.kind == "vision_observation" and "红色杯子" in record.summary for record in episodic)

    name_record = next(record for record in semantic if record.namespace == "profile.name")
    assert name_record.source_event_id == turn_event.event_id
    assert json.loads(name_record.provenance_json)["source_event_type"] == BrainEventType.USER_TURN_TRANSCRIBED
    assert name_record.stale_after_seconds is not None

    commitment = next(record for record in narrative if record.kind == "commitment")
    assert commitment.status == "open"
    assert json.loads(commitment.provenance_json)["source_event_type"] == BrainEventType.USER_TURN_TRANSCRIBED

    exporter = BrainMemoryExporter(store=store)
    artifact = exporter.export_thread_digest(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    assert artifact.path is not None
    exported = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert exported["heartbeat"]["thread_id"] == session_ids.thread_id
    assert exported["working_memory"]["context"]["last_user_text"] == "我叫阿周，我来自上海。提醒我给妈妈打电话。"
    assert "autonomy_digest" in exported
    assert exported["autonomy_digest"]["current_candidate_count"] == 0
    assert exported["autonomy_digest"]["pending_family_counts"] == {}
    assert exported["autonomy_digest"]["current_family_leaders"] == []
    assert exported["autonomy_digest"]["next_expiry_at"] is None
    assert "reevaluation_digest" in exported
    assert exported["reevaluation_digest"]["current_hold_count"] == 0
    assert exported["reevaluation_digest"]["trigger_counts"] == {}
    assert exported["reevaluation_digest"]["recent_transitions"] == []
    assert "wake_digest" in exported
    assert exported["wake_digest"]["current_wait_count"] == 0
    assert exported["wake_digest"]["current_wait_kind_counts"] == {}
    assert exported["wake_digest"]["route_counts"]["resume_direct"] == 0
    assert exported["wake_digest"]["route_counts"]["propose_candidate"] == 0
    assert exported["wake_digest"]["route_counts"]["keep_waiting"] == 0
    assert exported["wake_digest"]["recent_triggers"] == []
    assert "planning_digest" in exported
    assert exported["planning_digest"]["current_plan_state_count"] == 0
    assert exported["planning_digest"]["current_pending_proposal_count"] == 0
    assert exported["planning_digest"]["outcome_counts"]["adopted"] == 0
    assert exported["planning_digest"]["outcome_counts"]["rejected"] == 0
    assert exported["planning_digest"]["outcome_counts"]["pending_user_review"] == 0
    assert exported["planning_digest"]["outcome_counts"]["pending_operator_review"] == 0
    assert exported["planning_digest"]["current_pending_proposals"] == []
    assert exported["planning_digest"]["recent_adoptions"] == []
    assert exported["planning_digest"]["recent_rejections"] == []
    assert exported["planning_digest"]["recent_revision_flows"] == []
    assert "continuity_graph" in exported
    assert exported["continuity_graph"]["scope_type"] == "user"
    assert exported["continuity_graph"]["scope_id"] == session_ids.user_id
    assert exported["continuity_graph"]["node_counts"]["claim"] >= 2
    assert exported["continuity_graph"]["edge_counts"]["supported_by_event"] >= 1
    assert "continuity_graph_digest" in exported
    assert exported["continuity_graph_digest"]["current_node_kind_counts"]["claim"] >= 1
    assert "continuity_dossiers" in exported
    assert exported["continuity_dossiers"]["dossier_counts"]["relationship"] >= 1
    assert exported["continuity_dossiers"]["freshness_counts"]["fresh"] >= 1
    assert "continuity_governance_report" in exported
    assert exported["continuity_governance_report"]["freshness_counts"]["fresh"] >= 1
    assert "claim_currentness_counts" in exported["continuity_governance_report"]
    assert "dossier_availability_counts_by_task" in exported["continuity_governance_report"]
    assert "procedural_traces" in exported
    assert exported["procedural_traces"]["trace_counts"] == {}
    assert exported["procedural_traces"]["outcome_counts"] == {}
    assert "procedural_skills" in exported
    assert exported["procedural_skills"]["skill_counts"] == {}
    assert exported["procedural_skills"]["skills"] == []
    assert "procedural_skill_digest" in exported
    assert exported["procedural_skill_digest"]["skill_counts"] == {}
    assert "procedural_skill_governance_report" in exported
    assert exported["procedural_skill_governance_report"]["retirement_reason_counts"] == {}
    assert "procedural_qa_report" in exported
    assert exported["procedural_qa_report"]["case_counts"] == {
        "total": 0,
        "passed": 0,
        "failed": 0,
    }
    assert exported["procedural_qa_report"]["coverage_flags"] == {
        "skill_learning": False,
        "skill_reuse": False,
        "negative_transfer": False,
        "retirement": False,
        "supersession": False,
    }
    assert "private_working_memory" in exported
    assert "private_working_memory_digest" in exported
    assert exported["private_working_memory"]["buffer_counts"]
    assert exported["private_working_memory_digest"]["buffer_counts"] == exported["private_working_memory"]["buffer_counts"]
    assert "scene_world_state" in exported
    assert "scene_world_state_digest" in exported
    assert exported["scene_world_state_digest"]["degraded_mode"] in {
        "healthy",
        "limited",
        "unavailable",
    }
    assert "active_situation_model" in exported
    assert "active_situation_model_digest" in exported
    assert exported["active_situation_model"]["kind_counts"]
    assert exported["active_situation_model_digest"]["kind_counts"] == exported["active_situation_model"]["kind_counts"]
    assert "context_packet_digest" in exported
    assert exported["context_packet_digest"]["reply"]["temporal_mode"] == "current_first"
    assert "selected_availability_counts" in exported["context_packet_digest"]["reply"]
    assert "annotated_backing_ids" in exported["context_packet_digest"]["reply"]
    relationship_dossier = next(
        item for item in exported["continuity_dossiers"]["dossiers"] if item["kind"] == "relationship"
    )
    assert relationship_dossier["summary"]
    assert relationship_dossier["summary_evidence"]["claim_ids"]
    assert relationship_dossier["governance"]["task_availability"]
    assert "last_refresh_cause" in relationship_dossier["governance"]
    assert any(item["kind"] == "vision_observation" for item in exported["episodic"])


def test_brain_layered_memory_retrieval_and_contradiction_handling_are_stronger_than_token_hits(
    tmp_path,
):
    store = BrainStore(path=tmp_path / "brain.db")
    user_id = "pc-123"

    store.remember_fact(
        user_id=user_id,
        namespace="profile.name",
        subject="user",
        value={"value": "小周"},
        rendered_text="用户名字是 小周",
        confidence=0.8,
        singleton=True,
        source_event_id="evt-name-1",
        source_episode_id=None,
    )
    store.remember_fact(
        user_id=user_id,
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.92,
        singleton=True,
        source_event_id="evt-name-2",
        source_episode_id=None,
    )
    store.remember_fact(
        user_id=user_id,
        namespace="preference.like",
        subject="咖啡",
        value={"value": "咖啡"},
        rendered_text="用户喜欢 咖啡",
        confidence=0.8,
        singleton=False,
        source_event_id="evt-pref-1",
        source_episode_id=None,
    )
    store.remember_fact(
        user_id=user_id,
        namespace="preference.dislike",
        subject="咖啡",
        value={"value": "咖啡"},
        rendered_text="用户不喜欢 咖啡",
        confidence=0.91,
        singleton=False,
        source_event_id="evt-pref-2",
        source_episode_id=None,
    )
    store.upsert_task(
        user_id=user_id,
        title="finish the robot checklist",
        details={"source": "tool"},
        status="open",
        source_event_id="evt-task-1",
    )

    semantic = store.semantic_memories(user_id=user_id, limit=10, include_inactive=True)
    retriever = BrainMemoryRetriever(store=store)

    active_name = [record for record in semantic if record.namespace == "profile.name" and record.status == "active"]
    superseded_name = [record for record in semantic if record.namespace == "profile.name" and record.status == "superseded"]
    active_preference = [
        record
        for record in semantic
        if record.subject == "咖啡" and record.namespace == "preference.dislike" and record.status == "active"
    ]

    assert [record.rendered_text for record in active_name] == ["用户名字是 阿周"]
    assert [record.rendered_text for record in superseded_name] == ["用户名字是 小周"]
    assert active_preference[0].contradiction_key == "preference:咖啡"

    semantic_results = retriever.retrieve(
        BrainMemoryQuery(
            user_id=user_id,
            text="咖啡",
            layers=("semantic",),
            namespaces=("preference.dislike",),
            limit=5,
        )
    )
    narrative_results = retriever.retrieve(
        BrainMemoryQuery(
            user_id=user_id,
            text="robot checklist",
            layers=("narrative",),
            narrative_kinds=("commitment",),
            limit=5,
        )
    )

    assert semantic_results
    assert semantic_results[0].layer == "semantic"
    assert semantic_results[0].summary == "用户不喜欢 咖啡"
    assert narrative_results
    assert narrative_results[0].layer == "narrative"
    assert narrative_results[0].summary == "finish the robot checklist"

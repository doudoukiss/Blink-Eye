import json

from blink.brain.core import (
    BrainCoreReplayHarness,
    BrainCoreStore,
    BrainEventType,
    BrainPresenceSnapshot,
)
from blink.brain.core.session import resolve_brain_session_ids
from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory_v2 import BrainCoreMemoryBlockKind
from blink.brain.replay import BrainReplayHarness
from blink.brain.session import resolve_brain_session_ids as resolve_full_brain_session_ids
from blink.brain.store import BrainStore


def _relationship_state_payload_without_updated_at(projection) -> dict:
    payload = projection.as_dict()
    payload.pop("updated_at", None)
    return payload


def test_brain_core_replay_is_deterministic(tmp_path):
    store = BrainCoreStore(path=tmp_path / "source.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="runtime",
        payload={
            "scope_key": "browser:presence",
            "snapshot": BrainPresenceSnapshot(
                runtime_kind="browser",
                robot_head_enabled=True,
                robot_head_mode="simulation",
                robot_head_available=True,
                vision_enabled=True,
                vision_connected=True,
            ).as_dict(),
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "提醒我给妈妈打电话。"},
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="memory",
        payload={"title": "给妈妈打电话", "status": "open"},
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        payload={
            "tool_call_id": "call_1",
            "function_name": "fetch_user_image",
            "result": {"answer": "你手里拿着一个杯子。"},
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.ROBOT_ACTION_OUTCOME,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="tool",
        payload={
            "presence_scope_key": "browser:presence",
            "action_id": "cmd_blink",
            "accepted": True,
            "preview_only": False,
            "summary": "Blink executed cmd_blink.",
            "status": {
                "mode": "simulation",
                "armed": False,
                "available": True,
                "warnings": [],
                "details": {"driver": "simulation"},
            },
        },
    )

    harness = BrainCoreReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_turn_tool_robot_action",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    first = harness.replay(
        scenario,
        output_store_path=tmp_path / "first.db",
        artifact_path=tmp_path / "first.json",
        presence_scope_key="browser:presence",
    )
    second = harness.replay(
        scenario,
        output_store_path=tmp_path / "second.db",
        artifact_path=tmp_path / "second.json",
        presence_scope_key="browser:presence",
    )

    assert first.body.as_dict() == second.body.as_dict()
    assert first.scene.as_dict() == second.scene.as_dict()
    assert first.engagement.as_dict() == second.engagement.as_dict()
    assert first.relationship_state.as_dict() == second.relationship_state.as_dict()
    assert first.working_context.as_dict() == second.working_context.as_dict()
    assert first.agenda.as_dict() == second.agenda.as_dict()
    assert first.autonomy_ledger.as_dict() == second.autonomy_ledger.as_dict()
    assert first.heartbeat.as_dict() == second.heartbeat.as_dict()
    assert json.loads(first.artifact_path.read_text(encoding="utf-8")) == json.loads(
        second.artifact_path.read_text(encoding="utf-8")
    )


def test_persona_memory_replay_is_deterministic_across_retests(tmp_path):
    store = BrainStore(path=tmp_path / "source.db")
    session_ids = resolve_full_brain_session_ids(
        runtime_kind="browser",
        client_id="persona-replay",
    )
    store.ensure_default_blocks(load_default_agent_blocks())

    for namespace, value, rendered_text, source_event_id in (
        (
            "interaction.style",
            {"value": "gentle direct collaboration"},
            "User prefers gentle direct collaboration.",
            "evt-style-replay",
        ),
        (
            "teaching.preference.mode",
            {"value": "walkthrough"},
            "Walkthroughs work best.",
            "evt-mode-replay",
        ),
        (
            "teaching.preference.analogy_domain",
            {"value": "physics"},
            "Physics analogies land well.",
            "evt-analogy-replay",
        ),
    ):
        store.remember_fact(
            user_id=session_ids.user_id,
            namespace=namespace,
            subject="interaction",
            value=value,
            rendered_text=rendered_text,
            confidence=0.9,
            singleton=False,
            source_event_id=source_event_id,
            source_episode_id=None,
            provenance={"source": "test"},
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_turn_tool_robot_action",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    first = harness.replay(
        scenario,
        output_store_path=tmp_path / "first.db",
        artifact_path=tmp_path / "first.json",
        presence_scope_key="browser:presence",
    )
    second = harness.replay(
        scenario,
        output_store_path=tmp_path / "second.db",
        artifact_path=tmp_path / "second.json",
        presence_scope_key="browser:presence",
    )

    assert _relationship_state_payload_without_updated_at(
        first.context_surface.relationship_state
    ) == _relationship_state_payload_without_updated_at(second.context_surface.relationship_state)
    for block_kind in (
        BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
        BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
        BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
    ):
        assert block_kind in first.context_surface.core_blocks
        assert block_kind in second.context_surface.core_blocks
        assert (
            first.context_surface.core_blocks[block_kind].content
            == second.context_surface.core_blocks[block_kind].content
        )

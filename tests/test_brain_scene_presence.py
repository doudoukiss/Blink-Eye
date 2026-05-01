from blink.brain.events import BrainEventType
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def test_scene_engagement_relationship_and_body_projections_rebuild_from_events(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    store.upsert_narrative_memory(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        kind="commitment",
        title="给妈妈打电话",
        summary="答应今天给妈妈打电话。",
        details={},
        status="open",
        confidence=0.9,
    )
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
                perception_disabled=False,
            ).as_dict(),
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="perception",
        correlation_id="obs-1",
        payload={
            "presence_scope_key": "browser:presence",
            "frame_seq": 1,
            "camera_connected": True,
            "camera_fresh": True,
            "person_present": "present",
            "attention_to_camera": "toward_camera",
            "engagement_state": "engaged",
            "scene_change": "stable",
            "summary": "One person is facing the camera.",
            "confidence": 0.91,
            "observed_at": "2026-04-17T10:00:00+00:00",
        },
    )

    scene = store.get_scene_state_projection(scope_key="browser:presence")
    engagement = store.get_engagement_state_projection(scope_key="browser:presence")
    relationship = store.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    body = store.get_body_state_projection(scope_key="browser:presence")

    assert scene.camera_connected is True
    assert scene.person_present == "present"
    assert scene.last_visual_summary == "One person is facing the camera."
    assert engagement.engagement_state == "engaged"
    assert engagement.attention_to_camera == "toward_camera"
    assert relationship.user_present is True
    assert relationship.open_commitments == ["答应今天给妈妈打电话。"]
    assert body.vision_connected is True
    assert body.attention_target == "user"
    assert body.engagement_pose == "attentive"

    store.rebuild_brain_projections()

    rebuilt_scene = store.get_scene_state_projection(scope_key="browser:presence")
    rebuilt_engagement = store.get_engagement_state_projection(scope_key="browser:presence")
    rebuilt_relationship = store.get_relationship_state_projection(
        scope_key="browser:presence",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    rebuilt_body = store.get_body_state_projection(scope_key="browser:presence")

    assert rebuilt_scene.person_present == "present"
    assert rebuilt_engagement.engagement_state == "engaged"
    assert rebuilt_relationship.user_present is True
    assert rebuilt_relationship.open_commitments == ["答应今天给妈妈打电话。"]
    assert rebuilt_body.attention_target == "user"

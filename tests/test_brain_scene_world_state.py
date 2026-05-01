import json

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainEngagementStateProjection,
    BrainSceneStateProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)
from blink.brain.scene_world_state import build_scene_world_state_projection


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _event(*, event_id: str, event_type: str, ts: str, payload: dict, seq: int) -> BrainEventRecord:
    return BrainEventRecord(
        id=seq,
        event_id=event_id,
        event_type=event_type,
        ts=ts,
        agent_id="agent-1",
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        source="test",
        correlation_id=None,
        causal_parent_id=None,
        confidence=1.0,
        payload_json=json.dumps(payload, ensure_ascii=False),
        tags_json="[]",
    )


def _scene(**overrides) -> BrainSceneStateProjection:
    payload = {
        "camera_connected": True,
        "camera_track_state": "tracking",
        "person_present": "uncertain",
        "scene_change_state": "stable",
        "last_visual_summary": "Desk camera view.",
        "last_observed_at": _ts(1),
        "last_fresh_frame_at": _ts(1),
        "frame_age_ms": 500,
        "sensor_health_reason": None,
        "confidence": 0.8,
        "updated_at": _ts(1),
    }
    payload.update(overrides)
    return BrainSceneStateProjection(**payload)


def _engagement(**overrides) -> BrainEngagementStateProjection:
    payload = {
        "engagement_state": "idle",
        "attention_to_camera": "unknown",
        "user_present": False,
        "updated_at": _ts(1),
    }
    payload.update(overrides)
    return BrainEngagementStateProjection(**payload)


def _body(**overrides) -> BrainPresenceSnapshot:
    payload = {
        "runtime_kind": "browser",
        "vision_enabled": True,
        "vision_connected": True,
        "camera_track_state": "tracking",
        "sensor_health": "healthy",
        "sensor_health_reason": None,
        "camera_disconnected": False,
        "perception_unreliable": False,
        "last_fresh_frame_at": _ts(1),
        "frame_age_ms": 500,
        "updated_at": _ts(1),
    }
    payload.update(overrides)
    return BrainPresenceSnapshot(**payload)


def test_scene_world_state_record_and_projection_roundtrip():
    entity = BrainSceneWorldEntityRecord(
        entity_id="entity-1",
        entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
        canonical_label="cup",
        summary="A red cup sits on the desk.",
        state=BrainSceneWorldRecordState.ACTIVE.value,
        evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
        zone_id="zone:desk",
        confidence=0.91,
        freshness="current",
        affordance_ids=["aff-1"],
        backing_ids=["entity:cup"],
        source_event_ids=["evt-1"],
        observed_at=_ts(1),
        updated_at=_ts(1),
        expires_at=_ts(20),
        details={"stable_key": "entity:cup"},
    )
    affordance = BrainSceneWorldAffordanceRecord(
        affordance_id="aff-1",
        entity_id="entity-1",
        capability_family="vision.inspect",
        summary="The cup can be visually inspected.",
        availability=BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
        confidence=0.87,
        freshness="current",
        backing_ids=["entity:cup", "vision.inspect"],
        source_event_ids=["evt-1"],
        observed_at=_ts(1),
        updated_at=_ts(1),
        expires_at=_ts(20),
        details={"source": "test"},
    )
    projection = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=[entity],
        affordances=[affordance],
        degraded_mode="limited",
        degraded_reason_codes=["scene_stale"],
        updated_at=_ts(2),
    )

    hydrated = BrainSceneWorldProjection.from_dict(projection.as_dict())

    assert hydrated.scope_type == "presence"
    assert hydrated.entity_counts == {BrainSceneWorldEntityKind.OBJECT.value: 1}
    assert hydrated.affordance_counts == {"vision.inspect": 1}
    assert hydrated.state_counts == {BrainSceneWorldRecordState.ACTIVE.value: 1}
    assert hydrated.active_entity_ids == ["entity-1"]
    assert hydrated.active_affordance_ids == ["aff-1"]
    assert hydrated.degraded_reason_codes == ["scene_stale"]


def test_scene_world_state_builder_is_deterministic_and_caps_structured_entities():
    events = [
        _event(
            event_id="evt-1",
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            ts=_ts(1),
            seq=1,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": True,
                "person_present": "uncertain",
                "summary": "Desk camera view.",
                "observed_at": _ts(1),
                "scene_zones": [
                    {"zone_key": f"zone-{index}", "label": f"Zone {index}"}
                    for index in range(4)
                ],
                "scene_entities": [
                    {
                        "entity_key": f"item-{index}",
                        "kind": "object",
                        "label": f"Item {index}",
                        "summary": f"Item {index} is visible.",
                        "zone_key": f"zone-{index}",
                    }
                    for index in range(5)
                ],
            },
        )
    ]

    first = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(),
        engagement=_engagement(),
        body=_body(),
        recent_events=events,
        reference_ts=_ts(2),
    )
    second = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(),
        engagement=_engagement(),
        body=_body(),
        recent_events=events,
        reference_ts=_ts(2),
    )

    non_zone_entities = [
        record for record in first.entities if record.entity_kind != BrainSceneWorldEntityKind.ZONE.value
    ]
    zone_entities = [
        record for record in first.entities if record.entity_kind == BrainSceneWorldEntityKind.ZONE.value
    ]

    assert first.as_dict() == second.as_dict()
    assert len(non_zone_entities) == 4
    assert len(zone_entities) == 4


def test_scene_world_state_advances_from_active_to_stale_to_expired_without_reactivation():
    events = [
        _event(
            event_id="evt-entity",
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            ts=_ts(1),
            seq=1,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": True,
                "person_present": "uncertain",
                "summary": "Desk camera view.",
                "observed_at": _ts(1),
                "scene_entities": [
                    {
                        "entity_key": "cup",
                        "kind": "object",
                        "label": "cup",
                        "summary": "A cup sits on the desk.",
                        "zone_key": "desk",
                        "fresh_for_secs": 5,
                        "expire_after_secs": 10,
                    }
                ],
            },
        )
    ]

    active = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(),
        engagement=_engagement(),
        body=_body(),
        recent_events=events,
        reference_ts=_ts(4),
    )
    stale = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(),
        engagement=_engagement(),
        body=_body(),
        recent_events=events,
        reference_ts=_ts(7),
    )
    expired = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(),
        engagement=_engagement(),
        body=_body(),
        recent_events=events,
        reference_ts=_ts(12),
    )

    active_entity = next(record for record in active.entities if record.canonical_label == "cup")
    stale_entity = next(record for record in stale.entities if record.canonical_label == "cup")
    expired_entity = next(record for record in expired.entities if record.canonical_label == "cup")

    assert active_entity.state == BrainSceneWorldRecordState.ACTIVE.value
    assert stale_entity.state == BrainSceneWorldRecordState.STALE.value
    assert expired_entity.state == BrainSceneWorldRecordState.EXPIRED.value
    assert active_entity.entity_id not in stale.active_entity_ids
    assert expired_entity.entity_id in expired.expired_entity_ids


def test_scene_world_state_tracks_contradictions_when_entities_move_or_affordances_flip():
    events = [
        _event(
            event_id="evt-1",
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            ts=_ts(1),
            seq=1,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": True,
                "summary": "Cup on the desk.",
                "observed_at": _ts(1),
                "scene_entities": [
                    {
                        "entity_key": "cup",
                        "kind": "object",
                        "label": "cup",
                        "summary": "Cup on the desk.",
                        "zone_key": "desk",
                        "affordances": [
                            {
                                "capability_family": "vision.inspect",
                                "summary": "Inspect the cup.",
                                "availability": "available",
                            }
                        ],
                    }
                ],
            },
        ),
        _event(
            event_id="evt-2",
            event_type=BrainEventType.SCENE_CHANGED,
            ts=_ts(2),
            seq=2,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": True,
                "summary": "Cup moved to the shelf.",
                "observed_at": _ts(2),
                "scene_entities": [
                    {
                        "entity_key": "cup",
                        "kind": "object",
                        "label": "cup",
                        "summary": "Cup on the shelf.",
                        "zone_key": "shelf",
                        "affordances": [
                            {
                                "capability_family": "vision.inspect",
                                "summary": "Inspect the cup.",
                                "availability": "blocked",
                            }
                        ],
                    }
                ],
            },
        ),
    ]

    projection = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(updated_at=_ts(2), last_observed_at=_ts(2), last_fresh_frame_at=_ts(2)),
        engagement=_engagement(updated_at=_ts(2)),
        body=_body(updated_at=_ts(2), last_fresh_frame_at=_ts(2)),
        recent_events=events,
        reference_ts=_ts(2),
    )

    cup_entities = [
        record for record in projection.entities if record.entity_kind == BrainSceneWorldEntityKind.OBJECT.value
    ]
    active_cup = next(record for record in cup_entities if record.state == "active")
    contradicted_cup = next(record for record in cup_entities if record.state == "contradicted")

    assert len(cup_entities) == 2
    assert active_cup.zone_id == "zone:shelf"
    assert "zone_changed" in contradicted_cup.contradiction_codes
    assert "affordance_changed" in contradicted_cup.contradiction_codes
    assert projection.contradiction_counts["zone_changed"] == 1
    assert projection.blocked_affordance_ids
    assert projection.uncertain_affordance_ids


def test_scene_world_state_degraded_mode_suppresses_positive_affordances():
    events = [
        _event(
            event_id="evt-limited",
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            ts=_ts(20),
            seq=1,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": False,
                "camera_track_state": "stalled",
                "person_present": "uncertain",
                "summary": "Package is still on the desk.",
                "observed_at": _ts(20),
                "scene_entities": [
                    {
                        "entity_key": "package",
                        "kind": "object",
                        "label": "package",
                        "summary": "Package on the desk.",
                        "zone_key": "desk",
                        "affordances": [
                            {
                                "capability_family": "vision.inspect",
                                "summary": "Inspect the package.",
                                "availability": "available",
                            }
                        ],
                    }
                ],
            },
        )
    ]

    projection = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(
            camera_track_state="stalled",
            frame_age_ms=20_000,
            sensor_health_reason="stale_frame",
            updated_at=_ts(20),
            last_observed_at=_ts(20),
            last_fresh_frame_at=_ts(0),
        ),
        engagement=_engagement(updated_at=_ts(20)),
        body=_body(
            camera_track_state="stalled",
            perception_unreliable=True,
            frame_age_ms=20_000,
            updated_at=_ts(20),
            last_fresh_frame_at=_ts(0),
        ),
        recent_events=events,
        reference_ts=_ts(20),
    )

    package = next(
        record for record in projection.entities if record.canonical_label == "package"
    )

    assert projection.degraded_mode == "limited"
    assert "scene_stale" in projection.degraded_reason_codes
    assert package.state == BrainSceneWorldRecordState.STALE.value
    assert not projection.active_affordance_ids
    assert projection.uncertain_affordance_ids


def test_scene_world_state_does_not_hallucinate_objects_from_summary_only_perception():
    events = [
        _event(
            event_id="evt-summary",
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            ts=_ts(3),
            seq=1,
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": True,
                "camera_fresh": True,
                "person_present": "uncertain",
                "summary": "A notebook might be on the desk.",
                "observed_at": _ts(3),
            },
        )
    ]

    projection = build_scene_world_state_projection(
        scope_type="presence",
        scope_id="browser:presence",
        scene=_scene(updated_at=_ts(3), last_observed_at=_ts(3), last_fresh_frame_at=_ts(3)),
        engagement=_engagement(updated_at=_ts(3)),
        body=_body(updated_at=_ts(3), last_fresh_frame_at=_ts(3)),
        recent_events=events,
        reference_ts=_ts(4),
    )

    assert projection.entity_counts == {BrainSceneWorldEntityKind.ZONE.value: 1}
    assert all(record.entity_kind == BrainSceneWorldEntityKind.ZONE.value for record in projection.entities)

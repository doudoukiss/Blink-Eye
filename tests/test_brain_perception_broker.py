import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.autonomy import BrainAutonomyDecisionKind, BrainReevaluationConditionKind
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.perception import (
    CameraFeedHealthManager,
    CameraFeedHealthManagerConfig,
    CameraTrackHealthEvent,
    PerceptionBroker,
    PerceptionBrokerConfig,
    PresenceDetectionResult,
    VisionEnrichmentResult,
)
from blink.brain.perception.fusion import PresenceFusionEngine
from blink.brain.perception.health import build_camera_feed_health
from blink.brain.replay import BrainReplayHarness
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.frames.frames import UserImageRawFrame, VisionTextFrame


class StubCameraBuffer:
    def __init__(self, frame: UserImageRawFrame, *, frame_seq: int = 1):
        self.latest_camera_frame = frame
        self.latest_camera_frame_seq = frame_seq
        self.latest_camera_frame_received_monotonic = time.monotonic()
        self.latest_camera_frame_received_at = datetime.now(UTC).isoformat()


class StubVision:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.received_frames = []

    async def run_vision(self, frame):
        self.received_frames.append(frame)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        yield VisionTextFrame(text=output)


class StubDetector:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.available = True
        self.backend = "stub_detector"

    def detect(self, frame):
        self.calls.append(frame)
        output = self.outputs.pop(0)
        if isinstance(output, PresenceDetectionResult):
            return output
        return PresenceDetectionResult(
            state=output["state"],
            confidence=output["confidence"],
            backend=self.backend,
            reason=output.get("reason"),
            face_count=output.get("face_count", 1 if output["state"] == "present" else 0),
        )


class StubTransport:
    def __init__(self):
        self.capture_calls = 0
        self.renegotiation_calls = 0
        self.capture_framerates: list[int | None] = []

    async def capture_participant_video(self, *, video_source="camera", framerate=None):
        self.capture_calls += 1
        self.capture_framerates.append(framerate)

    async def request_renegotiation(self):
        self.renegotiation_calls += 1


class FakePerceptionAdapter:
    def __init__(self):
        self.descriptor = BrainAdapterDescriptor(
            backend_id="fake_perception",
            backend_version="v1",
            capabilities=("presence_detection", "scene_enrichment"),
            degraded_mode_id="unavailable_result",
            default_timeout_ms=50,
        )
        self.presence_detection_backend = "fake_detector"
        self.presence_detection_available = True
        self.scene_enrichment_available = True
        self.detect_calls = []
        self.enrich_calls = []

    def detect_presence(self, frame):
        self.detect_calls.append(frame)
        return PresenceDetectionResult(
            state="present",
            confidence=0.91,
            backend=self.presence_detection_backend,
        )

    async def enrich_scene(self, frame):
        self.enrich_calls.append(frame)
        return VisionEnrichmentResult(
            attention_to_camera="toward_camera",
            engagement_state="engaged",
            summary="A person is centered in frame.",
            confidence=0.9,
        )


def _frame() -> UserImageRawFrame:
    frame = UserImageRawFrame(
        user_id="pc-123",
        image=b"\x00" * 12,
        size=(2, 2),
        format="RGB",
    )
    frame.transport_source = "camera"
    return frame


def _executive(store: BrainStore, session_ids):
    return BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )


def test_camera_feed_health_treats_malformed_frame_timestamp_as_waiting():
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    buffer.latest_camera_frame_received_monotonic = "not-a-monotonic-time"

    health = build_camera_feed_health(
        camera_buffer=buffer,
        camera_connected=True,
        stale_after_secs=1.0,
    )

    assert health.camera_connected is True
    assert health.camera_fresh is False
    assert health.camera_track_state == "waiting_for_frame"
    assert health.frame_age_ms is None
    assert health.sensor_health_reason == "camera_waiting_for_frame"


def test_presence_fusion_clamps_malformed_backend_confidence():
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    health = build_camera_feed_health(
        camera_buffer=buffer,
        camera_connected=True,
        stale_after_secs=1.0,
    )
    engine = PresenceFusionEngine(
        config=PerceptionBrokerConfig(present_required_frames=1)
    )

    observation = engine.fuse(
        health=health,
        detection=PresenceDetectionResult(
            state="present",
            confidence=None,
            backend="test_detector",
        ),
        enrichment=VisionEnrichmentResult(
            attention_to_camera="toward_camera",
            engagement_state="engaged",
            summary="A person is visible.",
            confidence=float("nan"),
        ),
        frame_seq=1,
    )

    assert observation.person_present == "present"
    assert observation.confidence == 0.0
    assert observation.detection_confidence == 0.0


@pytest.mark.asyncio
async def test_perception_broker_emits_symbolic_events_and_deduplicates_unchanged_observations(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    vision = StubVision(
        [
            json.dumps(
                {
                    "person_present": "present",
                    "attention_to_camera": "toward_camera",
                    "engagement_state": "engaged",
                    "scene_change": "stable",
                    "summary": "One person is facing the camera.",
                    "confidence": 0.88,
                }
            ),
            json.dumps(
                {
                    "person_present": "present",
                    "attention_to_camera": "toward_camera",
                    "engagement_state": "engaged",
                    "scene_change": "stable",
                    "summary": "One person is facing the camera.",
                    "confidence": 0.88,
                }
            ),
            json.dumps(
                {
                    "person_present": "present",
                    "attention_to_camera": "toward_camera",
                    "engagement_state": "engaged",
                    "scene_change": "stable",
                    "summary": "One person is facing the camera.",
                    "confidence": 0.88,
                }
            ),
        ]
    )
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(
            enabled=True,
            emit_heartbeat_secs=0.01,
            present_required_frames=1,
        ),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=vision,
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "present", "confidence": 0.88},
                {"state": "present", "confidence": 0.88},
                {"state": "present", "confidence": 0.88},
            ]
        ),
    )

    first = await broker.observe_once()
    assert first is not None
    primary_events = list(
        reversed(
            store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=8,
                event_types=(BrainEventType.PERCEPTION_OBSERVED,),
            )
        )
    )
    assert [event.event_type for event in primary_events] == [BrainEventType.PERCEPTION_OBSERVED]
    all_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
    )
    assert BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED in {
        event.event_type for event in all_events
    }

    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    second = await broker.observe_once()
    assert second is not None
    primary_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
        event_types=(BrainEventType.PERCEPTION_OBSERVED,),
    )
    assert len(primary_events) == 1
    all_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
    )
    assert not any(event.event_type.startswith("tool.") for event in all_events)

    await asyncio.sleep(0.02)
    buffer.latest_camera_frame_seq = 3
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    third = await broker.observe_once()
    assert third is not None
    primary_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
        event_types=(BrainEventType.PERCEPTION_OBSERVED,),
    )
    assert len(primary_events) == 2
    assert vision.received_frames[0].text is not None


@pytest.mark.asyncio
async def test_perception_broker_emits_structured_scene_entities_and_zones(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "A package is on the desk.",
                        "confidence": 0.92,
                        "scene_zones": [
                            {"zone_key": "desk", "label": "Desk", "summary": "Primary desk zone."}
                        ],
                        "scene_entities": [
                            {
                                "entity_key": "package",
                                "kind": "object",
                                "label": "package",
                                "summary": "A package is on the desk.",
                                "zone_key": "desk",
                                "affordances": [
                                    {
                                        "capability_family": "vision.inspect",
                                        "summary": "Inspect the package.",
                                        "availability": "available",
                                    },
                                    {
                                        "capability_family": "arm.reach",
                                        "summary": "Reach the package.",
                                        "availability": "blocked",
                                    },
                                    {
                                        "capability_family": "arm.grasp",
                                        "summary": "Grasp the package.",
                                        "availability": "available",
                                    },
                                ],
                            }
                        ],
                    }
                )
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector([{"state": "present", "confidence": 0.9}]),
    )

    observation = await broker.observe_once()
    event = next(
        item
        for item in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=8,
        )
        if item.event_type == BrainEventType.PERCEPTION_OBSERVED
    )
    payload = event.payload

    assert observation is not None
    assert observation.scene_zones == [
        {"zone_key": "desk", "label": "Desk", "summary": "Primary desk zone."}
    ]
    assert len(observation.scene_entities) == 1
    assert len(observation.scene_entities[0]["affordances"]) == 2
    assert payload["scene_zones"] == observation.scene_zones
    assert payload["scene_entities"] == observation.scene_entities


@pytest.mark.asyncio
async def test_perception_broker_runs_through_adapter_without_direct_detector_or_enrichment(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    adapter = FakePerceptionAdapter()
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=None,
        perception_adapter=adapter,
        camera_connected=lambda: True,
    )

    observation = await broker.observe_once()
    body = store.get_body_state_projection(scope_key="browser:presence")
    primary_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
        event_types=(BrainEventType.PERCEPTION_OBSERVED,),
    )

    assert observation is not None
    assert broker.detector_available is True
    assert broker.enrichment_available is True
    assert len(adapter.detect_calls) == 1
    assert len(adapter.enrich_calls) == 1
    assert body.detection_backend == "fake_detector"
    assert primary_events[0].payload["detection_backend"] == "fake_detector"


@pytest.mark.asyncio
async def test_perception_broker_keeps_presence_when_enrichment_parse_fails(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(["not-json"]),
        camera_connected=lambda: True,
        detector=StubDetector([{"state": "present", "confidence": 0.93}]),
    )

    observation = await broker.observe_once()

    assert observation is not None
    assert observation.person_present == "present"
    assert observation.attention_to_camera == "unknown"
    assert observation.engagement_state == "unknown"
    assert observation.sensor_health_reason == "vision_enrichment_parse_error"
    body = store.get_body_state_projection(scope_key="browser:presence")
    assert body.perception_unreliable is False
    assert body.details["vision_enrichment_available"] is False
    scene = store.get_scene_state_projection(scope_key="browser:presence")
    assert scene.sensor_health_reason == "vision_enrichment_parse_error"
    events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
    )
    assert BrainEventType.PERCEPTION_OBSERVED in {event.event_type for event in events}
    perception_event = next(
        event for event in events if event.event_type == BrainEventType.PERCEPTION_OBSERVED
    )
    assert perception_event.payload["scene_zones"] == []
    assert perception_event.payload["scene_entities"] == []


@pytest.mark.asyncio
async def test_perception_broker_marks_stale_connected_feed_as_uncertain(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "One person is facing the camera.",
                        "confidence": 0.9,
                    }
                )
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector([{"state": "present", "confidence": 0.91}]),
    )

    first = await broker.observe_once()
    assert first is not None
    assert first.person_present == "present"

    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 10.0
    second = await broker.observe_once()

    assert second is not None
    assert second.person_present == "uncertain"
    assert second.camera_track_state == "stalled"
    scene = store.get_scene_state_projection(scope_key="browser:presence")
    assert scene.person_present == "uncertain"
    assert scene.sensor_health_reason == "camera_frame_stale"


@pytest.mark.asyncio
async def test_perception_broker_produces_reentry_candidate_and_replay(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = _executive(store, session_ids)
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "unknown",
                        "engagement_state": "away",
                        "summary": "No person is visible.",
                        "confidence": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "The user returned to frame.",
                        "confidence": 0.88,
                    }
                ),
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "absent", "confidence": 0.85, "face_count": 0},
                {"state": "present", "confidence": 0.88},
            ]
        ),
        candidate_goal_sink=executive.propose_candidate_goal,
    )

    await broker.observe_once()
    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    await broker.observe_once()

    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    goals = [goal for goal in agenda.goals if goal.intent == "autonomy.presence_user_reentered"]
    assert len(goals) == 1

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase6_scene_candidate_action",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario, presence_scope_key="browser:presence")
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert any(event["event_type"] == BrainEventType.GOAL_CANDIDATE_ACCEPTED for event in payload["events"])
    assert result.context_surface.agenda.goal(goals[0].goal_id) is not None
    assert result.autonomy_ledger.current_candidates == []


@pytest.mark.asyncio
async def test_perception_broker_triggers_projection_change_reevaluation_on_meaningful_delta(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    reevaluations = []
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "unknown",
                        "engagement_state": "away",
                        "summary": "No person is visible.",
                        "confidence": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "The user returned to the camera.",
                        "confidence": 0.88,
                    }
                ),
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "absent", "confidence": 0.8, "face_count": 0},
                {"state": "present", "confidence": 0.88},
            ]
        ),
        reevaluation_sink=lambda **kwargs: reevaluations.append(kwargs),
    )

    await broker.observe_once()
    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    await broker.observe_once()

    assert len(reevaluations) == 1
    assert reevaluations[0]["trigger"].kind == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
    assert reevaluations[0]["trigger"].details["person_present"] == "present"


@pytest.mark.asyncio
async def test_perception_broker_produces_attention_return_candidate_when_threshold_met(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = _executive(store, session_ids)
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "away_from_camera",
                        "engagement_state": "engaged",
                        "summary": "The user is in frame but looking away.",
                        "confidence": 0.82,
                    }
                ),
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "The user looks back to the camera.",
                        "confidence": 0.84,
                    }
                ),
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "present", "confidence": 0.83},
                {"state": "present", "confidence": 0.84},
            ]
        ),
        candidate_goal_sink=executive.propose_candidate_goal,
    )

    await broker.observe_once()
    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    await broker.observe_once()

    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    assert any(goal.intent == "autonomy.presence_attention_returned" for goal in agenda.goals)


@pytest.mark.asyncio
async def test_perception_broker_produces_inspect_only_candidate_for_degraded_camera(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = _executive(store, session_ids)
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "A person is visible in frame.",
                        "confidence": 0.9,
                    }
                )
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector([{"state": "present", "confidence": 0.9}]),
        candidate_goal_sink=executive.propose_candidate_goal,
    )

    first = await broker.observe_once()
    assert first is not None
    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 10.0
    degraded = await broker.observe_once()

    assert degraded is not None
    agenda = store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)
    degraded_goals = [goal for goal in agenda.goals if goal.intent == "autonomy.camera_degraded"]
    assert len(degraded_goals) == 1
    assert degraded_goals[0].details["autonomy"]["initiative_class"] == "inspect_only"
    assert not any(goal.intent == "autonomy.presence_brief_reengagement_speech" for goal in agenda.goals)


@pytest.mark.asyncio
async def test_perception_broker_skips_candidates_for_weak_or_stale_presence(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = _executive(store, session_ids)
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "unknown",
                        "engagement_state": "away",
                        "summary": "No person is visible.",
                        "confidence": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "A person might be visible.",
                        "confidence": 0.74,
                    }
                ),
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "absent", "confidence": 0.8, "face_count": 0},
                {"state": "present", "confidence": 0.74},
            ]
        ),
        candidate_goal_sink=executive.propose_candidate_goal,
    )

    await broker.observe_once()
    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    await broker.observe_once()

    autonomy_events = store.recent_autonomy_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=8,
    )
    assert autonomy_events == []


@pytest.mark.asyncio
async def test_perception_broker_spoken_reengagement_candidate_hits_non_action_when_user_turn_open(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = _executive(store, session_ids)
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    broker = PerceptionBroker(
        config=PerceptionBrokerConfig(enabled=True, present_required_frames=1, detect_interval_secs=0.0),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        camera_buffer=buffer,
        vision=StubVision(
            [
                json.dumps(
                    {
                        "attention_to_camera": "unknown",
                        "engagement_state": "away",
                        "summary": "No person is visible.",
                        "confidence": 0.8,
                    }
                ),
                json.dumps(
                    {
                        "attention_to_camera": "toward_camera",
                        "engagement_state": "engaged",
                        "summary": "The user returned and is engaged.",
                        "confidence": 0.9,
                    }
                ),
            ]
        ),
        camera_connected=lambda: True,
        detector=StubDetector(
            [
                {"state": "absent", "confidence": 0.81, "face_count": 0},
                {"state": "present", "confidence": 0.9},
            ]
        ),
        candidate_goal_sink=executive.propose_candidate_goal,
    )

    await broker.observe_once()
    broker._non_present_since = (datetime.now(UTC).replace(microsecond=0) - timedelta(seconds=11)).isoformat()
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    buffer.latest_camera_frame_seq = 2
    buffer.latest_camera_frame_received_monotonic = time.monotonic()
    await broker.observe_once()

    ledger = store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    non_action = next(
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind == BrainAutonomyDecisionKind.NON_ACTION.value
    )
    assert non_action.reason == "user_turn_open"
    assert [candidate.candidate_type for candidate in ledger.current_candidates] == [
        "presence_brief_reengagement_speech"
    ]


@pytest.mark.asyncio
async def test_camera_feed_health_manager_attempts_bounded_recovery(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 1.0
    transport = StubTransport()
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(
            stale_after_secs=0.01,
            auto_recovery_enabled=True,
            renegotiate_after_secs=0.02,
            monitor_interval_secs=0.005,
            max_recovery_attempts=2,
            recovery_window_secs=1.0,
        ),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=transport,
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()
    await manager.start()
    await asyncio.sleep(0.08)
    await manager.close()

    events = {
        event.event_type
        for event in store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=32,
        )
    }
    body = store.get_body_state_projection(scope_key="browser:presence")

    assert transport.capture_calls >= 1
    assert 1 in transport.capture_framerates
    assert transport.renegotiation_calls >= 1
    assert BrainEventType.CAMERA_RECOVERY_ATTEMPTED in events
    assert BrainEventType.CAMERA_RECOVERY_EXHAUSTED in events
    assert body.camera_track_state in {"stalled", "recovering"}
    assert body.recovery_attempts >= 1


@pytest.mark.asyncio
async def test_camera_feed_health_manager_is_diagnostic_only_by_default(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 1.0
    transport = StubTransport()
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(
            stale_after_secs=0.01,
            renegotiate_after_secs=0.02,
            monitor_interval_secs=0.005,
            max_recovery_attempts=2,
            recovery_window_secs=1.0,
        ),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=transport,
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()
    await manager.start()
    await asyncio.sleep(0.04)
    await manager.close()

    body = store.get_body_state_projection(scope_key="browser:presence")
    assert transport.capture_calls == 0
    assert transport.renegotiation_calls == 0
    assert body.camera_track_state == "stalled"
    assert body.sensor_health_reason == "camera_manual_reload_required"
    assert body.recovery_in_progress is False
    assert body.recovery_attempts == 0


@pytest.mark.asyncio
async def test_camera_feed_health_manager_waits_for_first_frame(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=0)
    buffer.latest_camera_frame = None
    buffer.latest_camera_frame_received_monotonic = None
    buffer.latest_camera_frame_received_at = None
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=StubTransport(),
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()

    health = manager.current_health()
    body = store.get_body_state_projection(scope_key="browser:presence")
    assert health.camera_connected is True
    assert health.camera_fresh is False
    assert health.camera_track_state == "waiting_for_frame"
    assert health.sensor_health_reason == "camera_waiting_for_frame"
    assert health.recovery_in_progress is False
    assert health.recovery_attempts == 0
    assert body.vision_connected is True
    assert body.camera_track_state == "waiting_for_frame"
    assert body.sensor_health == "degraded"
    assert body.recovery_in_progress is False
    assert body.recovery_attempts == 0


@pytest.mark.asyncio
async def test_camera_feed_health_manager_forces_disconnect_over_stale_cached_frame(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 30.0
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(stale_after_secs=0.01),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=StubTransport(),
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()
    await manager.handle_client_disconnected()

    health = manager.current_health()
    body = store.get_body_state_projection(scope_key="browser:presence")
    assert health.camera_connected is False
    assert health.camera_track_state == "disconnected"
    assert health.last_fresh_frame_at is None
    assert health.frame_age_ms is None
    assert health.recovery_in_progress is False
    assert health.recovery_attempts == 0
    assert body.vision_connected is False
    assert body.camera_track_state == "disconnected"
    assert body.sensor_health_reason == "camera_disconnected"
    assert body.last_fresh_frame_at is None
    assert body.recovery_in_progress is False
    assert body.recovery_attempts == 0


@pytest.mark.asyncio
async def test_camera_feed_health_manager_clears_stale_detail_metadata_on_disconnect(
    tmp_path,
):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    buffer.latest_camera_frame_received_monotonic = time.monotonic() - 30.0
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(stale_after_secs=0.01),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=StubTransport(),
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()
    await manager.handle_audio_track_stalled(
        CameraTrackHealthEvent(
            source="audio",
            reason="audio_track_stalled",
            consecutive_failures=1,
            last_frame_age_ms=500,
            enabled=True,
        )
    )
    await manager._perform_recovery("capture_camera")
    recovering_body = store.get_body_state_projection(scope_key="browser:presence")
    assert recovering_body.details["last_recovery_action"] == "capture_camera"
    assert recovering_body.details["audio_track_health"]["state"] == "stalled"

    await manager.handle_client_disconnected()

    disconnected_body = store.get_body_state_projection(scope_key="browser:presence")
    assert disconnected_body.camera_track_state == "disconnected"
    assert "last_recovery_action" not in disconnected_body.details
    assert "audio_track_health" not in disconnected_body.details


@pytest.mark.asyncio
async def test_camera_feed_health_manager_audio_stall_is_in_memory_until_next_camera_update(
    tmp_path,
):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    buffer = StubCameraBuffer(_frame(), frame_seq=1)
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(stale_after_secs=0.01),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=buffer,
        transport=StubTransport(),
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    await manager.handle_client_connected()
    before_count = store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    await manager.handle_audio_track_stalled(
        CameraTrackHealthEvent(
            source="audio",
            reason="audio_track_stalled",
            consecutive_failures=1,
            last_frame_age_ms=500,
            enabled=True,
        )
    )
    after_count = store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    body_before_camera_update = store.get_body_state_projection(scope_key="browser:presence")

    assert after_count == before_count
    assert "audio_track_health" not in body_before_camera_update.details

    await manager._perform_recovery("capture_camera")
    body_after_camera_update = store.get_body_state_projection(scope_key="browser:presence")
    assert body_after_camera_update.details["audio_track_health"]["state"] == "stalled"


@pytest.mark.asyncio
async def test_camera_feed_health_manager_close_suppresses_monitor_task_errors(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    manager = CameraFeedHealthManager(
        config=CameraFeedHealthManagerConfig(),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        runtime_kind="browser",
        camera_buffer=StubCameraBuffer(_frame(), frame_seq=0),
        transport=StubTransport(),
        camera_connected=lambda: True,
        vision_enabled=True,
        enrichment_available=False,
        detection_backend="stub_detector",
    )

    async def failed_monitor_task():
        raise json.JSONDecodeError("Expecting value", "", 0)

    manager._task = asyncio.create_task(failed_monitor_task())
    await asyncio.sleep(0)

    await manager.close()

    assert manager._task is None


def test_camera_feed_health_builder_suppresses_recovery_metadata_when_not_actionable():
    buffer = StubCameraBuffer(_frame())
    disconnected = build_camera_feed_health(
        camera_buffer=buffer,
        camera_connected=False,
        stale_after_secs=0.01,
        recovery_in_progress=True,
        recovery_attempts=4,
    )
    assert disconnected.camera_track_state == "disconnected"
    assert disconnected.recovery_in_progress is False
    assert disconnected.recovery_attempts == 0
    assert disconnected.last_fresh_frame_at is None
    assert disconnected.frame_age_ms is None

    buffer.latest_camera_frame = None
    buffer.latest_camera_frame_received_monotonic = None
    buffer.latest_camera_frame_received_at = None
    waiting = build_camera_feed_health(
        camera_buffer=buffer,
        camera_connected=True,
        stale_after_secs=0.01,
        recovery_in_progress=True,
        recovery_attempts=4,
    )
    assert waiting.camera_track_state == "waiting_for_frame"
    assert waiting.recovery_in_progress is False
    assert waiting.recovery_attempts == 0

"""Fused browser perception broker with deterministic presence and optional VLM enrichment."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from loguru import logger

from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.events import BrainEventType
from blink.brain.perception.detector import OnnxFacePresenceDetector, PresenceDetectionResult
from blink.brain.perception.enrichment import VisionEnrichmentEngine, VisionEnrichmentResult
from blink.brain.perception.health import (
    DEFAULT_CAMERA_STALE_FRAME_SECS,
    CameraFeedHealth,
    build_camera_feed_health,
)
from blink.brain.projections import BrainGoalFamily

if TYPE_CHECKING:
    from blink.brain.adapters.perception import PerceptionAdapter


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _expires_in(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _safe_confidence(value: Any, *, default: float = 0.0) -> float:
    """Clamp optional backend confidence values into a deterministic public range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(confidence):
        return default
    return max(0.0, min(1.0, confidence))


@dataclass(frozen=True)
class PerceptionBrokerConfig:
    """Runtime configuration for fused browser visual presence."""

    enabled: bool = False
    interval_secs: float = 3.0
    detect_interval_secs: float = 1.0
    emit_heartbeat_secs: float = 15.0
    max_stale_frame_secs: float = DEFAULT_CAMERA_STALE_FRAME_SECS
    present_confidence_threshold: float = 0.65
    present_required_frames: int = 2
    absent_required_frames: int = 3


@dataclass(frozen=True)
class FusedPresenceState:
    """One fused symbolic visual observation written into the event spine."""

    observation_id: str
    frame_seq: int
    person_present: str
    attention_to_camera: str
    engagement_state: str
    scene_change: str
    summary: str
    confidence: float
    observed_at: str
    camera_track_state: str
    last_fresh_frame_at: str | None
    frame_age_ms: int | None
    detection_backend: str | None
    detection_confidence: float | None
    sensor_health_reason: str | None
    recovery_in_progress: bool
    recovery_attempts: int
    enrichment_available: bool | None
    scene_zones: list[dict[str, Any]] = field(default_factory=list)
    scene_entities: list[dict[str, Any]] = field(default_factory=list)

    def as_event_payload(self, *, camera_connected: bool, camera_fresh: bool) -> dict[str, Any]:
        """Serialize the fused state into the canonical perception payload."""
        return {
            "observation_id": self.observation_id,
            "frame_seq": self.frame_seq,
            "camera_connected": camera_connected,
            "camera_fresh": camera_fresh,
            "camera_track_state": self.camera_track_state,
            "person_present": self.person_present,
            "attention_to_camera": self.attention_to_camera,
            "engagement_state": self.engagement_state,
            "scene_change": self.scene_change,
            "summary": self.summary,
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "last_fresh_frame_at": self.last_fresh_frame_at,
            "frame_age_ms": self.frame_age_ms,
            "detection_backend": self.detection_backend,
            "detection_confidence": self.detection_confidence,
            "sensor_health_reason": self.sensor_health_reason,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_attempts": self.recovery_attempts,
            "enrichment_available": self.enrichment_available,
            "scene_zones": list(self.scene_zones)[:3],
            "scene_entities": [
                {
                    **{
                        key: value
                        for key, value in dict(item).items()
                        if key != "affordances"
                    },
                    **(
                        {
                            "affordances": [
                                dict(affordance)
                                for affordance in list(item.get("affordances", []) or [])[:2]
                                if isinstance(affordance, dict)
                            ]
                        }
                        if isinstance(item, dict) and item.get("affordances")
                        else {}
                    ),
                }
                for item in list(self.scene_entities)[:4]
                if isinstance(item, dict)
            ],
        }


PerceptionObservation = FusedPresenceState


class PresenceFusionEngine:
    """Fuse health, deterministic detection, and optional VLM enrichment."""

    def __init__(self, *, config: PerceptionBrokerConfig):
        """Initialize the hysteresis counters for presence fusion."""
        self._config = config
        self._present_streak = 0
        self._absent_streak = 0
        self._last_person_present = "uncertain"
        self._last_track_state = "disconnected"

    def fuse(
        self,
        *,
        health: CameraFeedHealth,
        detection: PresenceDetectionResult | None,
        enrichment: VisionEnrichmentResult | None,
        frame_seq: int,
        observed_at: str | None = None,
    ) -> FusedPresenceState:
        """Produce one final symbolic observation."""
        if not health.camera_connected:
            person_present = "uncertain"
            attention = "unknown"
            engagement = "unknown"
            summary = "Camera is disconnected; visual presence is uncertain."
            confidence = 0.0
        elif not health.camera_fresh:
            person_present = "uncertain"
            attention = "unknown"
            engagement = "unknown"
            summary = "Camera frame is stale; visual presence is uncertain."
            confidence = 0.0
        elif health.camera_track_state in {"stalled", "recovering"}:
            person_present = "uncertain"
            attention = "unknown"
            engagement = "unknown"
            summary = "Camera feed is recovering; visual presence is uncertain."
            confidence = 0.0
        elif detection is None or not detection.available:
            person_present = "uncertain"
            attention = "unknown"
            engagement = "unknown"
            summary = "Presence detector is unavailable; visual presence is uncertain."
            confidence = 0.0
        elif detection.state == "present":
            self._present_streak += 1
            self._absent_streak = 0
            detection_confidence = _safe_confidence(detection.confidence)
            if self._last_person_present == "present" or (
                self._present_streak >= self._config.present_required_frames
            ):
                person_present = "present"
            else:
                person_present = "uncertain"
            if enrichment is not None and enrichment.available:
                attention = enrichment.attention_to_camera
                engagement = enrichment.engagement_state
                summary = enrichment.summary
                confidence = max(
                    detection_confidence,
                    _safe_confidence(enrichment.confidence),
                )
            else:
                attention = "unknown"
                engagement = "unknown"
                summary = "A person is visible in the current camera frame."
                confidence = detection_confidence
        elif detection.state == "absent":
            self._absent_streak += 1
            self._present_streak = 0
            detection_confidence = _safe_confidence(detection.confidence)
            if self._last_person_present == "absent" or (
                self._absent_streak >= self._config.absent_required_frames
            ):
                person_present = "absent"
            elif self._last_person_present == "present":
                person_present = "present"
            else:
                person_present = "uncertain"
            attention = "unknown"
            engagement = "away" if person_present == "absent" else "unknown"
            summary = (
                "No person is visible in the latest fresh camera frame."
                if person_present == "absent"
                else "Visual presence is still stabilizing."
            )
            confidence = detection_confidence
        else:
            self._present_streak = 0
            self._absent_streak = 0
            person_present = "uncertain"
            attention = "unknown"
            engagement = "unknown"
            summary = "Visual presence is uncertain."
            confidence = (
                _safe_confidence(detection.confidence)
                if detection is not None
                else 0.0
            )

        scene_change = (
            "changed"
            if (
                person_present != self._last_person_present
                or health.camera_track_state != self._last_track_state
            )
            else "stable"
        )

        if person_present != "uncertain":
            self._last_person_present = person_present
        self._last_track_state = health.camera_track_state

        return FusedPresenceState(
            observation_id=uuid4().hex,
            frame_seq=frame_seq,
            person_present=person_present,
            attention_to_camera=attention,
            engagement_state=engagement,
            scene_change=scene_change,
            summary=summary,
            confidence=_safe_confidence(confidence),
            observed_at=observed_at or _utc_now(),
            camera_track_state=health.camera_track_state,
            last_fresh_frame_at=health.last_fresh_frame_at,
            frame_age_ms=health.frame_age_ms,
            detection_backend=(
                detection.backend if detection is not None else health.detection_backend
            ),
            detection_confidence=(
                _safe_confidence(detection.confidence) if detection is not None else None
            ),
            sensor_health_reason=health.sensor_health_reason or (
                detection.reason if detection is not None and detection.reason else None
            ) or (
                enrichment.reason if enrichment is not None and enrichment.reason else None
            ),
            recovery_in_progress=health.recovery_in_progress,
            recovery_attempts=health.recovery_attempts,
            enrichment_available=(
                enrichment.available if enrichment is not None else health.enrichment_available
            ),
            scene_zones=(
                list(enrichment.scene_zones[:3])
                if enrichment is not None and enrichment.available
                else []
            ),
            scene_entities=(
                [
                    {
                        **{
                            key: value
                            for key, value in dict(item).items()
                            if key != "affordances"
                        },
                        **(
                            {
                                "affordances": [
                                    dict(affordance)
                                    for affordance in list(item.get("affordances", []) or [])[:2]
                                    if isinstance(affordance, dict)
                                ]
                            }
                            if isinstance(item, dict) and item.get("affordances")
                            else {}
                        ),
                    }
                    for item in list(enrichment.scene_entities[:4])
                    if isinstance(item, dict)
                ]
                if enrichment is not None and enrichment.available
                else []
            ),
        )


class PerceptionBroker:
    """Low-cadence fused visual presence loop over the latest cached browser frame."""

    def __init__(
        self,
        *,
        config: PerceptionBrokerConfig,
        store,
        session_resolver,
        presence_scope_key: str,
        camera_buffer,
        vision,
        camera_connected=None,
        camera_health_provider=None,
        perception_adapter: "PerceptionAdapter | None" = None,
        detector: OnnxFacePresenceDetector | None = None,
        enrichment: VisionEnrichmentEngine | None = None,
        candidate_goal_sink: Callable[..., Any] | None = None,
        reevaluation_sink: Callable[..., Any] | None = None,
    ):
        """Initialize the fused perception broker."""
        self._config = config
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._camera_buffer = camera_buffer
        self._camera_connected = camera_connected or (lambda: False)
        self._camera_health_provider = camera_health_provider
        self._closed = False
        self._task: asyncio.Task | None = None
        if perception_adapter is None:
            from blink.brain.adapters.perception import LocalPerceptionAdapter

        self._perception_adapter = perception_adapter or LocalPerceptionAdapter(
            vision=vision,
            detector=detector,
            enrichment=enrichment,
        )
        self._candidate_goal_sink = candidate_goal_sink
        self._reevaluation_sink = reevaluation_sink
        self._fusion = PresenceFusionEngine(config=config)
        self._last_emitted_observation: FusedPresenceState | None = None
        self._last_observation_event_id: str | None = None
        self._last_emitted_at_monotonic = 0.0
        self._last_detected_frame_seq = 0
        self._last_detected_at_monotonic = 0.0
        self._last_enriched_frame_seq = 0
        self._last_enriched_at_monotonic = 0.0
        self._last_detection: PresenceDetectionResult | None = None
        self._last_enrichment: VisionEnrichmentResult | None = None
        self._non_present_since: str | None = None

    @property
    def perception_adapter(self) -> "PerceptionAdapter":
        """Expose the backend adapter used for perception."""
        return self._perception_adapter

    @property
    def enabled(self) -> bool:
        """Return whether the fused perception loop is enabled."""
        return bool(self._config.enabled)

    @property
    def detector_available(self) -> bool:
        """Return whether deterministic presence detection is ready."""
        return self._perception_adapter.presence_detection_available

    @property
    def enrichment_available(self) -> bool:
        """Return whether optional VLM enrichment is configured."""
        return self._perception_adapter.scene_enrichment_available

    async def start(self):
        """Start the background observation loop."""
        if not self._config.enabled or self._task is not None or self._closed:
            return
        self._task = asyncio.create_task(self._run_loop(), name="blink-perception-broker")

    async def close(self):
        """Stop the background observation loop."""
        self._closed = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def observe_once(self) -> PerceptionObservation | None:
        """Run one fused perception pass over the latest camera state."""
        if not self._config.enabled:
            return None

        frame = getattr(self._camera_buffer, "latest_camera_frame", None)
        frame_seq = int(getattr(self._camera_buffer, "latest_camera_frame_seq", 0))
        health = self._camera_health_provider() if self._camera_health_provider else build_camera_feed_health(
            camera_buffer=self._camera_buffer,
            camera_connected=bool(self._camera_connected()),
            stale_after_secs=self._config.max_stale_frame_secs,
            detection_backend=self._perception_adapter.presence_detection_backend,
            enrichment_available=self._perception_adapter.scene_enrichment_available,
        )

        if not health.camera_connected and self._last_emitted_observation is None:
            return None

        now = time.monotonic()
        detection = self._last_detection
        if (
            health.camera_fresh
            and frame is not None
            and frame_seq > 0
            and frame_seq != self._last_detected_frame_seq
            and (now - self._last_detected_at_monotonic) >= self._config.detect_interval_secs
        ):
            detection = self._perception_adapter.detect_presence(frame)
            self._last_detection = detection
            self._last_detected_frame_seq = frame_seq
            self._last_detected_at_monotonic = now
        elif not health.camera_fresh:
            detection = None

        observation = self._fusion.fuse(
            health=health,
            detection=detection,
            enrichment=None,
            frame_seq=frame_seq,
        )

        should_run_enrichment = bool(
            health.camera_fresh
            and frame is not None
            and self._perception_adapter.scene_enrichment_available
            and (
                observation.person_present == "present"
                or observation.scene_change == "changed"
            )
            and (
                frame_seq != self._last_enriched_frame_seq
                or (now - self._last_enriched_at_monotonic) >= self._config.interval_secs
            )
        )
        if should_run_enrichment:
            enrichment = await self._perception_adapter.enrich_scene(frame)
            self._last_enrichment = enrichment
            self._last_enriched_frame_seq = frame_seq
            self._last_enriched_at_monotonic = now
            observation = self._fusion.fuse(
                health=health,
                detection=detection,
                enrichment=enrichment,
                frame_seq=frame_seq,
                observed_at=observation.observed_at,
            )

        if not self._should_emit(observation):
            return observation

        session_ids = self._session_resolver()
        payload = {
            **observation.as_event_payload(
                camera_connected=health.camera_connected,
                camera_fresh=health.camera_fresh,
            ),
            "presence_scope_key": self._presence_scope_key,
        }
        observed_event = self._store.append_brain_event(
            event_type=BrainEventType.PERCEPTION_OBSERVED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="perception",
            correlation_id=observation.observation_id,
            causal_parent_id=self._last_observation_event_id,
            confidence=observation.confidence,
            payload=payload,
        )
        self._emit_transition_events(
            session_ids=session_ids,
            observation=observation,
            payload=payload,
        )
        self._maybe_produce_scene_candidate(
            session_ids=session_ids,
            previous=self._last_emitted_observation,
            observation=observation,
            health=health,
            causal_parent_id=observed_event.event_id,
        )
        self._maybe_trigger_projection_reevaluation(
            observation=observation,
            previous=self._last_emitted_observation,
            causal_parent_id=observed_event.event_id,
        )
        self._update_non_present_window(previous=self._last_emitted_observation, observation=observation)
        self._last_observation_event_id = observed_event.event_id
        self._last_emitted_observation = observation
        self._last_emitted_at_monotonic = time.monotonic()
        return observation

    async def _run_loop(self):
        """Run the broker until closed."""
        while not self._closed:
            try:
                await self.observe_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Perception broker loop error: {}: {}",
                    type(exc).__name__,
                    exc,
                )
            await asyncio.sleep(min(self._config.detect_interval_secs, 1.0))

    def _should_emit(self, observation: FusedPresenceState) -> bool:
        if self._last_emitted_observation is None:
            return True
        if self._meaningful_signature(observation) != self._meaningful_signature(
            self._last_emitted_observation
        ):
            return True
        return (time.monotonic() - self._last_emitted_at_monotonic) >= self._config.emit_heartbeat_secs

    def _emit_transition_events(self, *, session_ids, observation: FusedPresenceState, payload: dict[str, Any]):
        previous = self._last_emitted_observation
        if previous is None:
            return
        correlation_id = observation.observation_id
        causal_parent_id = self._last_observation_event_id
        if (
            previous.engagement_state != observation.engagement_state
            or previous.person_present != observation.person_present
        ):
            self._store.append_brain_event(
                event_type=BrainEventType.ENGAGEMENT_CHANGED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="perception",
                correlation_id=correlation_id,
                causal_parent_id=causal_parent_id,
                confidence=observation.confidence,
                payload=payload,
            )
        if previous.attention_to_camera != observation.attention_to_camera:
            self._store.append_brain_event(
                event_type=BrainEventType.ATTENTION_CHANGED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="perception",
                correlation_id=correlation_id,
                causal_parent_id=causal_parent_id,
                confidence=observation.confidence,
                payload=payload,
            )
        if (
            previous.scene_change != observation.scene_change
            or previous.person_present != observation.person_present
            or previous.camera_track_state != observation.camera_track_state
        ):
            self._store.append_brain_event(
                event_type=BrainEventType.SCENE_CHANGED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="perception",
                correlation_id=correlation_id,
                causal_parent_id=causal_parent_id,
                confidence=observation.confidence,
                payload=payload,
            )

    @staticmethod
    def _meaningful_signature(observation: FusedPresenceState) -> tuple[str, str, str, str, str | None]:
        return (
            observation.person_present,
            observation.attention_to_camera,
            observation.engagement_state,
            observation.camera_track_state,
            observation.sensor_health_reason,
        )

    @staticmethod
    def _reevaluation_signature(
        observation: FusedPresenceState,
    ) -> tuple[str, str, str, str, str | None]:
        return (
            observation.person_present,
            observation.attention_to_camera,
            observation.engagement_state,
            observation.camera_track_state,
            observation.sensor_health_reason,
        )

    def _maybe_produce_scene_candidate(
        self,
        *,
        session_ids,
        previous: FusedPresenceState | None,
        observation: FusedPresenceState,
        health: CameraFeedHealth,
        causal_parent_id: str,
    ):
        if self._candidate_goal_sink is None or previous is None:
            return
        candidate = self._build_scene_candidate(
            previous=previous,
            observation=observation,
            health=health,
        )
        if candidate is None:
            return
        self._candidate_goal_sink(
            candidate_goal=candidate,
            source="perception",
            correlation_id=observation.observation_id,
            causal_parent_id=causal_parent_id,
        )

    def _maybe_trigger_projection_reevaluation(
        self,
        *,
        observation: FusedPresenceState,
        previous: FusedPresenceState | None,
        causal_parent_id: str,
    ):
        if self._reevaluation_sink is None or previous is None:
            return
        if self._reevaluation_signature(previous) == self._reevaluation_signature(observation):
            return
        self._reevaluation_sink(
            trigger=BrainReevaluationTrigger(
                kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
                summary="Reevaluate held perception candidates after a meaningful scene-state change.",
                details={
                    "presence_scope_key": self._presence_scope_key,
                    "person_present": observation.person_present,
                    "attention_to_camera": observation.attention_to_camera,
                    "engagement_state": observation.engagement_state,
                    "camera_track_state": observation.camera_track_state,
                    "sensor_health_reason": observation.sensor_health_reason,
                },
                source_event_type=BrainEventType.PERCEPTION_OBSERVED,
                source_event_id=causal_parent_id,
                ts=observation.observed_at,
            )
        )

    def _build_scene_candidate(
        self,
        *,
        previous: FusedPresenceState,
        observation: FusedPresenceState,
        health: CameraFeedHealth,
    ) -> BrainCandidateGoal | None:
        healthy_camera = self._healthy_camera(health=health, observation=observation)
        scope_prefix = f"{self._presence_scope_key}:scene"
        degraded_reason_changed = (
            self._is_degraded_reason(observation.sensor_health_reason)
            and observation.sensor_health_reason != previous.sensor_health_reason
        )
        if (
            previous.camera_track_state == "healthy"
            and observation.camera_track_state in {"stalled", "recovering", "disconnected"}
        ) or degraded_reason_changed:
            return BrainCandidateGoal(
                candidate_goal_id=uuid4().hex,
                candidate_type="camera_degraded",
                source=BrainCandidateGoalSource.PERCEPTION.value,
                summary="Visual presence degraded; inspect camera health before acting.",
                goal_family=BrainGoalFamily.ENVIRONMENT.value,
                urgency=0.8,
                confidence=1.0,
                initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
                cooldown_key=f"{scope_prefix}:camera-degraded",
                dedupe_key=f"{scope_prefix}:camera-degraded",
                policy_tags=["phase6b", "scene", "degraded"],
                requires_user_turn_gap=False,
                expires_at=_expires_in(30),
                payload=self._scene_goal_payload(
                    candidate_type="camera_degraded",
                    observation=observation,
                    transient_only=True,
                ),
            )

        if not healthy_camera or observation.person_present != "present":
            return None

        reentered = previous.person_present != "present" and observation.person_present == "present"
        attention_returned = (
            previous.attention_to_camera != "toward_camera"
            and observation.attention_to_camera == "toward_camera"
            and observation.person_present == "present"
        )
        observation_confidence = _safe_confidence(observation.confidence)
        if (
            reentered
            and observation.attention_to_camera == "toward_camera"
            and observation.engagement_state == "engaged"
            and observation_confidence >= 0.85
            and self._non_present_window_secs(observation=observation) >= 10.0
        ):
            return BrainCandidateGoal(
                candidate_goal_id=uuid4().hex,
                candidate_type="presence_brief_reengagement_speech",
                source=BrainCandidateGoalSource.PERCEPTION.value,
                summary="Offer a brief spoken re-engagement after the user returns.",
                goal_family=BrainGoalFamily.ENVIRONMENT.value,
                urgency=0.95,
                confidence=observation_confidence,
                initiative_class=BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
                cooldown_key=f"{scope_prefix}:spoken-reengagement",
                dedupe_key=f"{scope_prefix}:spoken-reengagement",
                policy_tags=["phase6b", "scene", "spoken"],
                requires_user_turn_gap=True,
                expires_at=_expires_in(12),
                payload=self._scene_goal_payload(
                    candidate_type="presence_brief_reengagement_speech",
                    observation=observation,
                    transient_only=True,
                ),
            )
        if reentered and observation_confidence >= 0.75:
            return BrainCandidateGoal(
                candidate_goal_id=uuid4().hex,
                candidate_type="presence_user_reentered",
                source=BrainCandidateGoalSource.PERCEPTION.value,
                summary="Acknowledge the user's return to the camera frame silently.",
                goal_family=BrainGoalFamily.ENVIRONMENT.value,
                urgency=0.82,
                confidence=observation_confidence,
                initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
                cooldown_key=f"{scope_prefix}:re-entry",
                dedupe_key=f"{scope_prefix}:re-entry",
                policy_tags=["phase6b", "scene", "reentry"],
                requires_user_turn_gap=False,
                expires_at=_expires_in(20),
                payload=self._scene_goal_payload(
                    candidate_type="presence_user_reentered",
                    observation=observation,
                    transient_only=True,
                ),
            )
        if attention_returned and observation_confidence >= 0.8:
            return BrainCandidateGoal(
                candidate_goal_id=uuid4().hex,
                candidate_type="presence_attention_returned",
                source=BrainCandidateGoalSource.PERCEPTION.value,
                summary="Track the user's returned attention silently.",
                goal_family=BrainGoalFamily.ENVIRONMENT.value,
                urgency=0.78,
                confidence=observation_confidence,
                initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
                cooldown_key=f"{scope_prefix}:attention-return",
                dedupe_key=f"{scope_prefix}:attention-return",
                policy_tags=["phase6b", "scene", "attention_return"],
                requires_user_turn_gap=False,
                expires_at=_expires_in(20),
                payload=self._scene_goal_payload(
                    candidate_type="presence_attention_returned",
                    observation=observation,
                    transient_only=True,
                ),
            )
        return None

    @staticmethod
    def _healthy_camera(*, health: CameraFeedHealth, observation: FusedPresenceState) -> bool:
        return bool(
            health.camera_connected
            and health.camera_fresh
            and observation.camera_track_state == "healthy"
            and not observation.recovery_in_progress
        )

    @staticmethod
    def _is_degraded_reason(reason: str | None) -> bool:
        return bool(reason) and reason not in {"vision_enrichment_parse_error"}

    def _non_present_window_secs(self, *, observation: FusedPresenceState) -> float:
        start_at = _parse_ts(self._non_present_since)
        observed_at = _parse_ts(observation.observed_at)
        if start_at is None or observed_at is None:
            return 0.0
        return max(0.0, (observed_at - start_at).total_seconds())

    def _update_non_present_window(
        self,
        *,
        previous: FusedPresenceState | None,
        observation: FusedPresenceState,
    ):
        if observation.person_present == "present":
            self._non_present_since = None
            return
        if previous is None or previous.person_present == "present" or not self._non_present_since:
            self._non_present_since = observation.observed_at

    def _scene_goal_payload(
        self,
        *,
        candidate_type: str,
        observation: FusedPresenceState,
        transient_only: bool,
    ) -> dict[str, Any]:
        return {
            "goal_intent": f"autonomy.{candidate_type}",
            "goal_details": {
                "transient_only": transient_only,
                "scene_candidate": {
                    "presence_scope_key": self._presence_scope_key,
                    "observation_id": observation.observation_id,
                    "person_present": observation.person_present,
                    "attention_to_camera": observation.attention_to_camera,
                    "engagement_state": observation.engagement_state,
                    "camera_track_state": observation.camera_track_state,
                    "sensor_health_reason": observation.sensor_health_reason,
                    "frame_age_ms": observation.frame_age_ms,
                    "detection_confidence": observation.detection_confidence,
                },
            },
        }

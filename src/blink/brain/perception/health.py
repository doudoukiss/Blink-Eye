"""Transport health and bounded camera recovery for browser visual presence."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from loguru import logger

from blink.brain.events import BrainEventType
from blink.brain.presence import BrainPresenceSnapshot, normalize_presence_snapshot

DEFAULT_CAMERA_STALE_FRAME_SECS = 5.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(value: Any) -> float | None:
    """Return a finite float for camera metadata, or None when it is malformed."""
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


@dataclass(frozen=True)
class CameraTrackHealthEvent:
    """One transport-reported media track stall or resume event."""

    source: str
    reason: str
    consecutive_failures: int
    last_frame_age_ms: int | None
    enabled: bool


@dataclass(frozen=True)
class CameraFeedHealth:
    """Current browser camera-feed health summary."""

    camera_connected: bool
    camera_fresh: bool
    camera_track_state: str
    last_fresh_frame_at: str | None
    frame_age_ms: int | None
    sensor_health_reason: str | None
    recovery_in_progress: bool
    recovery_attempts: int
    detection_backend: str | None = None
    enrichment_available: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the current health for event payloads and projections."""
        return {
            "camera_connected": self.camera_connected,
            "camera_fresh": self.camera_fresh,
            "camera_track_state": self.camera_track_state,
            "last_fresh_frame_at": self.last_fresh_frame_at,
            "frame_age_ms": self.frame_age_ms,
            "sensor_health_reason": self.sensor_health_reason,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_attempts": self.recovery_attempts,
            "detection_backend": self.detection_backend,
            "enrichment_available": self.enrichment_available,
        }


def build_camera_feed_health(
    *,
    camera_buffer,
    camera_connected: bool,
    stale_after_secs: float,
    recovery_in_progress: bool = False,
    recovery_attempts: int = 0,
    transport_stalled: bool = False,
    transport_reason: str | None = None,
    detection_backend: str | None = None,
    enrichment_available: bool | None = None,
) -> CameraFeedHealth:
    """Build camera health from the latest cached frame metadata."""
    now = time.monotonic()
    received_at_monotonic = _safe_float(
        getattr(camera_buffer, "latest_camera_frame_received_monotonic", None)
    )
    received_at = getattr(camera_buffer, "latest_camera_frame_received_at", None)
    frame_age_ms = (
        max(0, int((now - received_at_monotonic) * 1000))
        if received_at_monotonic is not None
        else None
    )
    camera_fresh = bool(
        camera_connected
        and received_at_monotonic is not None
        and (now - received_at_monotonic) <= stale_after_secs
    )

    if not camera_connected:
        track_state = "disconnected"
        reason = "camera_disconnected"
    elif received_at_monotonic is None:
        track_state = "waiting_for_frame"
        reason = "camera_waiting_for_frame"
    elif recovery_in_progress and not camera_fresh:
        track_state = "recovering"
        reason = transport_reason or "camera_recovery_in_progress"
    elif transport_stalled:
        track_state = "stalled"
        reason = transport_reason or "camera_track_stalled"
    elif not camera_fresh:
        track_state = "stalled"
        reason = "camera_frame_stale"
    else:
        track_state = "healthy"
        reason = None

    effective_recovery_in_progress = bool(track_state == "recovering")
    effective_recovery_attempts = (
        max(0, int(recovery_attempts))
        if camera_connected and track_state in {"stalled", "recovering"}
        else 0
    )

    return CameraFeedHealth(
        camera_connected=camera_connected,
        camera_fresh=camera_fresh,
        camera_track_state=track_state,
        last_fresh_frame_at=received_at if camera_connected else None,
        frame_age_ms=frame_age_ms if camera_connected else None,
        sensor_health_reason=reason,
        recovery_in_progress=effective_recovery_in_progress,
        recovery_attempts=effective_recovery_attempts,
        detection_backend=detection_backend,
        enrichment_available=enrichment_available,
    )


@dataclass(frozen=True)
class CameraFeedHealthManagerConfig:
    """Runtime policy for camera health monitoring and bounded recovery."""

    stale_after_secs: float = DEFAULT_CAMERA_STALE_FRAME_SECS
    auto_recovery_enabled: bool = False
    renegotiate_after_secs: float = 10.0
    monitor_interval_secs: float = 0.5
    max_recovery_attempts: int = 3
    recovery_window_secs: float = 60.0


class CameraFeedHealthManager:
    """Monitor browser camera freshness and attempt bounded recovery."""

    def __init__(
        self,
        *,
        config: CameraFeedHealthManagerConfig,
        store,
        session_resolver,
        presence_scope_key: str,
        runtime_kind: str,
        camera_buffer,
        transport,
        camera_connected,
        vision_enabled: bool,
        enrichment_available: bool,
        detection_backend: str,
        performance_emit: Callable[..., object] | None = None,
    ):
        """Bind the health manager to one browser runtime."""
        self._config = config
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._runtime_kind = runtime_kind
        self._camera_buffer = camera_buffer
        self._transport = transport
        self._camera_connected = camera_connected
        self._vision_enabled = vision_enabled
        self._enrichment_available = enrichment_available
        self._detection_backend = detection_backend
        self._performance_emit = performance_emit

        self._closed = False
        self._task: asyncio.Task | None = None
        self._transport_stalled = False
        self._transport_reason: str | None = None
        self._current_health = build_camera_feed_health(
            camera_buffer=camera_buffer,
            camera_connected=False,
            stale_after_secs=self._config.stale_after_secs,
            detection_backend=detection_backend,
            enrichment_available=enrichment_available,
        )
        self._stall_started_at_monotonic: float | None = None
        self._recovery_attempts: deque[float] = deque()
        self._recovery_in_progress = False
        self._recovery_exhausted = False
        self._last_recovery_action: str | None = None
        self._last_audio_issue: dict[str, Any] | None = None
        self._force_disconnected = False

    @staticmethod
    def _transition_signature(health: CameraFeedHealth) -> tuple[Any, ...]:
        """Return the fields that should trigger diagnostic persistence/events.

        `frame_age_ms` changes on every poll while no semantic health state has
        changed. Treating that as a transition turns the health monitor into a
        hot SQLite writer and can starve the live browser path when the store is
        locked or closing.
        """
        return (
            health.camera_connected,
            health.camera_fresh,
            health.camera_track_state,
            health.sensor_health_reason,
            health.recovery_in_progress,
            health.recovery_attempts,
            health.detection_backend,
            health.enrichment_available,
        )

    def current_health(self) -> CameraFeedHealth:
        """Return the latest camera-feed health."""
        return self._current_health

    async def start(self):
        """Start the bounded camera-health monitor."""
        if self._task is not None or self._closed:
            return
        await self._refresh_state(force=True)
        self._task = asyncio.create_task(self._run_loop(), name="blink-camera-health")

    async def close(self):
        """Stop the health monitor."""
        self._closed = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive guard for corrupted local state
            logger.warning(
                "Camera health monitor stopped with a suppressed {} during close.",
                type(exc).__name__,
            )
        self._task = None

    async def handle_client_connected(self):
        """Reset recovery state when a browser client reconnects."""
        self._recovery_attempts.clear()
        self._recovery_in_progress = False
        self._recovery_exhausted = False
        self._last_recovery_action = None
        self._transport_stalled = False
        self._transport_reason = None
        self._stall_started_at_monotonic = None
        self._force_disconnected = False
        self._last_audio_issue = None
        await self._refresh_state(force=True)

    async def handle_client_disconnected(self):
        """Record the disconnected camera state immediately."""
        self._recovery_in_progress = False
        self._recovery_attempts.clear()
        self._recovery_exhausted = False
        self._last_recovery_action = None
        self._transport_stalled = False
        self._transport_reason = "camera_disconnected"
        self._stall_started_at_monotonic = None
        self._force_disconnected = True
        self._last_audio_issue = None
        await self._refresh_state(force=True)

    async def handle_video_track_stalled(self, event: CameraTrackHealthEvent):
        """Record a transport-reported video stall."""
        self._transport_stalled = True
        self._transport_reason = event.reason
        if self._stall_started_at_monotonic is None:
            self._stall_started_at_monotonic = time.monotonic()
        await self._refresh_state(
            force=True,
            event_type=BrainEventType.CAMERA_TRACK_STALLED,
            event_payload={
                "source_track": event.source,
                "reason": event.reason,
                "consecutive_failures": event.consecutive_failures,
                "last_frame_age_ms": event.last_frame_age_ms,
                "track_enabled": event.enabled,
            },
        )

    async def handle_video_track_resumed(self, event: CameraTrackHealthEvent):
        """Record a transport-reported video recovery."""
        self._transport_stalled = False
        self._transport_reason = None
        self._recovery_in_progress = False
        self._stall_started_at_monotonic = None
        await self._refresh_state(
            force=True,
            event_type=BrainEventType.CAMERA_TRACK_RESUMED,
            event_payload={
                "source_track": event.source,
                "reason": event.reason,
                "consecutive_failures": event.consecutive_failures,
                "last_frame_age_ms": event.last_frame_age_ms,
                "track_enabled": event.enabled,
            },
        )

    async def handle_audio_track_stalled(self, event: CameraTrackHealthEvent):
        """Store the latest audio stall for operator diagnostics."""
        self._last_audio_issue = {
            "state": "stalled",
            "source": event.source,
            "reason": event.reason,
            "consecutive_failures": event.consecutive_failures,
            "last_frame_age_ms": event.last_frame_age_ms,
        }

    async def handle_audio_track_resumed(self, event: CameraTrackHealthEvent):
        """Clear the latest audio stall for operator diagnostics."""
        self._last_audio_issue = {
            "state": "healthy",
            "source": event.source,
            "reason": event.reason,
            "consecutive_failures": event.consecutive_failures,
            "last_frame_age_ms": event.last_frame_age_ms,
        }

    async def _run_loop(self):
        """Poll for stale browser camera frames and bounded recovery windows."""
        while not self._closed:
            try:
                await self._refresh_state()
                await self._maybe_attempt_recovery()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard for runtime state
                logger.warning(
                    "Camera health monitor suppressed {} during polling.",
                    type(exc).__name__,
                )
            await asyncio.sleep(max(0.01, self._config.monitor_interval_secs))

    async def _refresh_state(
        self,
        *,
        force: bool = False,
        event_type: str | None = None,
        event_payload: dict[str, Any] | None = None,
    ):
        now = time.monotonic()
        new_health = build_camera_feed_health(
            camera_buffer=self._camera_buffer,
            camera_connected=bool(self._camera_connected()) and not self._force_disconnected,
            stale_after_secs=self._config.stale_after_secs,
            recovery_in_progress=self._recovery_in_progress,
            recovery_attempts=len(self._recovery_attempts),
            transport_stalled=self._transport_stalled or self._recovery_exhausted,
            transport_reason=self._transport_reason or ("camera_recovery_exhausted" if self._recovery_exhausted else None),
            detection_backend=self._detection_backend,
            enrichment_available=self._enrichment_available,
        )

        previous = self._current_health
        changed = self._transition_signature(previous) != self._transition_signature(new_health)
        if not changed and not force:
            self._current_health = new_health
            return

        if new_health.camera_track_state in {"stalled", "recovering"}:
            if self._stall_started_at_monotonic is None:
                self._stall_started_at_monotonic = now
        else:
            self._stall_started_at_monotonic = None

        self._current_health = new_health
        try:
            self._persist_presence_snapshot(new_health)
        except Exception as exc:  # pragma: no cover - fail-open realtime guard
            logger.warning(
                "Camera health monitor suppressed {} while persisting state.",
                type(exc).__name__,
            )

        if event_type is not None:
            try:
                self._append_camera_event(
                    event_type=event_type,
                    health=new_health,
                    payload=event_payload or {},
                )
            except Exception as exc:  # pragma: no cover - fail-open realtime guard
                logger.warning(
                    "Camera health monitor suppressed {} while recording event.",
                    type(exc).__name__,
                )
            return

        previous_state = previous.camera_track_state
        new_state = new_health.camera_track_state
        if previous_state in {"healthy", "disconnected"} and new_state in {"stalled", "recovering"}:
            try:
                self._append_camera_event(
                    event_type=BrainEventType.CAMERA_TRACK_STALLED,
                    health=new_health,
                    payload={"reason": new_health.sensor_health_reason},
                )
            except Exception as exc:  # pragma: no cover - fail-open realtime guard
                logger.warning(
                    "Camera health monitor suppressed {} while recording stall.",
                    type(exc).__name__,
                )
            self._emit_public_health_event(new_health)
        elif previous_state in {"stalled", "recovering"} and new_state == "healthy":
            try:
                self._append_camera_event(
                    event_type=BrainEventType.CAMERA_TRACK_RESUMED,
                    health=new_health,
                    payload={"reason": "camera_feed_resumed"},
                )
            except Exception as exc:  # pragma: no cover - fail-open realtime guard
                logger.warning(
                    "Camera health monitor suppressed {} while recording resume.",
                    type(exc).__name__,
                )
            self._emit_public_health_event(new_health, event_type="camera.frame_resumed")

    async def _maybe_attempt_recovery(self):
        health = self._current_health
        if not health.camera_connected or health.camera_fresh:
            self._recovery_in_progress = False
            return
        if health.camera_track_state not in {"stalled", "recovering"}:
            return
        if not self._config.auto_recovery_enabled:
            self._recovery_in_progress = False
            self._last_recovery_action = None
            self._transport_stalled = True
            if self._transport_reason != "camera_manual_reload_required":
                self._transport_reason = "camera_manual_reload_required"
                await self._refresh_state(force=True)
            return

        now = time.monotonic()
        while self._recovery_attempts and now - self._recovery_attempts[0] > self._config.recovery_window_secs:
            self._recovery_attempts.popleft()

        if len(self._recovery_attempts) >= self._config.max_recovery_attempts:
            if not self._recovery_exhausted:
                self._recovery_exhausted = True
                self._transport_reason = "camera_recovery_exhausted"
                await self._refresh_state(
                    force=True,
                    event_type=BrainEventType.CAMERA_RECOVERY_EXHAUSTED,
                    event_payload={"reason": "camera_recovery_exhausted"},
                )
            return

        if not self._recovery_in_progress:
            await self._perform_recovery("capture_camera")
            return

        if (
            self._last_recovery_action != "renegotiate"
            and self._stall_started_at_monotonic is not None
            and now - self._stall_started_at_monotonic >= self._config.renegotiate_after_secs
        ):
            await self._perform_recovery("renegotiate")

    async def _perform_recovery(self, action: str):
        if self._transport is None:
            return
        self._recovery_attempts.append(time.monotonic())
        self._recovery_in_progress = True
        self._last_recovery_action = action
        await self._refresh_state(
            force=True,
            event_type=BrainEventType.CAMERA_RECOVERY_ATTEMPTED,
            event_payload={"action": action},
        )
        try:
            if action == "capture_camera":
                await self._transport.capture_participant_video(video_source="camera", framerate=1)
            else:
                await self._transport.request_renegotiation()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(f"Camera recovery action `{action}` failed: {exc}")
            self._transport_reason = f"camera_recovery_failed:{action}"
            await self._refresh_state(force=True)

    def _persist_presence_snapshot(self, health: CameraFeedHealth):
        session_ids = self._session_resolver()
        snapshot = self._store.get_body_state_projection(scope_key=self._presence_scope_key)
        snapshot.runtime_kind = snapshot.runtime_kind or self._runtime_kind
        snapshot.vision_enabled = self._vision_enabled
        snapshot.vision_connected = health.camera_connected
        snapshot.camera_track_state = health.camera_track_state
        snapshot.last_fresh_frame_at = health.last_fresh_frame_at
        snapshot.frame_age_ms = health.frame_age_ms
        snapshot.sensor_health_reason = health.sensor_health_reason
        snapshot.recovery_in_progress = health.recovery_in_progress
        snapshot.recovery_attempts = health.recovery_attempts
        snapshot.detection_backend = health.detection_backend
        snapshot.camera_disconnected = bool(self._vision_enabled and not health.camera_connected)
        snapshot.perception_disabled = False
        snapshot.perception_unreliable = health.camera_track_state in {
            "stalled",
            "recovering",
            "waiting_for_frame",
        }
        details = dict(snapshot.details)
        details["vision_enrichment_available"] = health.enrichment_available
        if self._last_recovery_action is not None:
            details["last_recovery_action"] = self._last_recovery_action
        else:
            details.pop("last_recovery_action", None)
        if self._last_audio_issue is not None:
            details["audio_track_health"] = dict(self._last_audio_issue)
        else:
            details.pop("audio_track_health", None)
        snapshot.details = details
        snapshot.updated_at = _utc_now()
        normalize_presence_snapshot(snapshot)
        self._store.append_brain_event(
            event_type=BrainEventType.BODY_STATE_UPDATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="visual_health",
            payload={
                "scope_key": self._presence_scope_key,
                "snapshot": snapshot.as_dict(),
            },
        )

    def _append_camera_event(
        self,
        *,
        event_type: str,
        health: CameraFeedHealth,
        payload: dict[str, Any],
    ):
        session_ids = self._session_resolver()
        self._store.append_brain_event(
            event_type=event_type,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="visual_health",
            payload={
                "presence_scope_key": self._presence_scope_key,
                **health.as_dict(),
                **payload,
            },
        )

    def _emit_public_health_event(
        self,
        health: CameraFeedHealth,
        *,
        event_type: str | None = None,
    ) -> None:
        if self._performance_emit is None:
            return
        track_state = health.camera_track_state
        reason = health.sensor_health_reason or track_state
        public_event_type = event_type
        if public_event_type is None:
            public_event_type = (
                "camera.frame_stale"
                if reason in {"camera_frame_stale", "camera_manual_reload_required"}
                else "camera.health_stalled"
            )
        try:
            self._performance_emit(
                event_type=public_event_type,
                source="camera-health",
                mode="error" if track_state in {"stalled", "recovering"} else "connected",
                metadata={
                    "track_state": track_state,
                    "frame_age_ms": health.frame_age_ms,
                    "camera_fresh_state": "fresh" if health.camera_fresh else "not_fresh",
                },
                reason_codes=("camera:health_transition", reason),
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostic sink
            logger.debug(
                "Suppressed public camera health event failure: {}",
                type(exc).__name__,
            )

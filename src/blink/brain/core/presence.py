"""Provider-free runtime presence models for the Blink brain kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BrainPresenceSnapshot:
    """Small structured runtime presence snapshot."""

    runtime_kind: str
    robot_head_enabled: bool = False
    robot_head_mode: str = "none"
    robot_head_armed: bool = False
    robot_head_available: bool = False
    robot_head_last_action: str | None = None
    robot_head_last_accepted_action: str | None = None
    robot_head_last_rejected_action: str | None = None
    robot_head_last_safe_state: str | None = None
    policy_phase: str = "neutral"
    attention_target: str | None = None
    engagement_pose: str | None = None
    vision_enabled: bool = False
    vision_connected: bool = False
    camera_track_state: str = "disconnected"
    sensor_health: str = "unknown"
    sensor_health_reason: str | None = None
    vision_unavailable: bool = False
    camera_disconnected: bool = False
    perception_disabled: bool = False
    perception_unreliable: bool = False
    last_fresh_frame_at: str | None = None
    frame_age_ms: int | None = None
    detection_backend: str | None = None
    detection_confidence: float | None = None
    recovery_in_progress: bool = False
    recovery_attempts: int = 0
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the presence snapshot."""
        return {
            "runtime_kind": self.runtime_kind,
            "robot_head_enabled": self.robot_head_enabled,
            "robot_head_mode": self.robot_head_mode,
            "robot_head_armed": self.robot_head_armed,
            "robot_head_available": self.robot_head_available,
            "robot_head_last_action": self.robot_head_last_action,
            "robot_head_last_accepted_action": self.robot_head_last_accepted_action,
            "robot_head_last_rejected_action": self.robot_head_last_rejected_action,
            "robot_head_last_safe_state": self.robot_head_last_safe_state,
            "policy_phase": self.policy_phase,
            "attention_target": self.attention_target,
            "engagement_pose": self.engagement_pose,
            "vision_enabled": self.vision_enabled,
            "vision_connected": self.vision_connected,
            "camera_track_state": self.camera_track_state,
            "sensor_health": self.sensor_health,
            "sensor_health_reason": self.sensor_health_reason,
            "vision_unavailable": self.vision_unavailable,
            "camera_disconnected": self.camera_disconnected,
            "perception_disabled": self.perception_disabled,
            "perception_unreliable": self.perception_unreliable,
            "last_fresh_frame_at": self.last_fresh_frame_at,
            "frame_age_ms": self.frame_age_ms,
            "detection_backend": self.detection_backend,
            "detection_confidence": self.detection_confidence,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_attempts": self.recovery_attempts,
            "warnings": list(self.warnings),
            "details": dict(self.details),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPresenceSnapshot":
        """Hydrate a presence snapshot from stored JSON."""
        payload = data or {}
        return cls(
            runtime_kind=str(payload.get("runtime_kind", "local")),
            robot_head_enabled=bool(payload.get("robot_head_enabled", False)),
            robot_head_mode=str(payload.get("robot_head_mode", "none")),
            robot_head_armed=bool(payload.get("robot_head_armed", False)),
            robot_head_available=bool(payload.get("robot_head_available", False)),
            robot_head_last_action=payload.get("robot_head_last_action"),
            robot_head_last_accepted_action=payload.get("robot_head_last_accepted_action"),
            robot_head_last_rejected_action=payload.get("robot_head_last_rejected_action"),
            robot_head_last_safe_state=payload.get("robot_head_last_safe_state"),
            policy_phase=str(payload.get("policy_phase", "neutral")),
            attention_target=payload.get("attention_target"),
            engagement_pose=payload.get("engagement_pose"),
            vision_enabled=bool(payload.get("vision_enabled", False)),
            vision_connected=bool(payload.get("vision_connected", False)),
            camera_track_state=str(payload.get("camera_track_state", "disconnected")),
            sensor_health=str(payload.get("sensor_health", "unknown")),
            sensor_health_reason=payload.get("sensor_health_reason"),
            vision_unavailable=bool(payload.get("vision_unavailable", False)),
            camera_disconnected=bool(payload.get("camera_disconnected", False)),
            perception_disabled=bool(payload.get("perception_disabled", False)),
            perception_unreliable=bool(payload.get("perception_unreliable", False)),
            last_fresh_frame_at=payload.get("last_fresh_frame_at"),
            frame_age_ms=(
                int(payload["frame_age_ms"])
                if payload.get("frame_age_ms") is not None
                else None
            ),
            detection_backend=payload.get("detection_backend"),
            detection_confidence=(
                float(payload["detection_confidence"])
                if payload.get("detection_confidence") is not None
                else None
            ),
            recovery_in_progress=bool(payload.get("recovery_in_progress", False)),
            recovery_attempts=int(payload.get("recovery_attempts", 0)),
            warnings=list(payload.get("warnings", [])),
            details=dict(payload.get("details", {})),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


def normalize_presence_snapshot(snapshot: BrainPresenceSnapshot) -> BrainPresenceSnapshot:
    """Normalize derived presence fields in-place and return the snapshot."""
    snapshot.camera_disconnected = bool(snapshot.vision_enabled and not snapshot.vision_connected)
    if snapshot.vision_unavailable:
        snapshot.sensor_health = "unavailable"
        snapshot.sensor_health_reason = snapshot.sensor_health_reason or "vision_unavailable"
    elif snapshot.perception_disabled or not snapshot.vision_enabled:
        snapshot.sensor_health = "disabled"
        snapshot.sensor_health_reason = snapshot.sensor_health_reason or "perception_disabled"
    elif snapshot.camera_disconnected:
        snapshot.sensor_health = "degraded"
        snapshot.sensor_health_reason = snapshot.sensor_health_reason or "camera_disconnected"
    elif snapshot.camera_track_state in {"stalled", "recovering", "waiting_for_frame"}:
        snapshot.sensor_health = "degraded"
        snapshot.sensor_health_reason = snapshot.sensor_health_reason or (
            "camera_waiting_for_frame"
            if snapshot.camera_track_state == "waiting_for_frame"
            else "camera_track_stalled"
        )
    elif snapshot.perception_unreliable:
        snapshot.sensor_health = "degraded"
        snapshot.sensor_health_reason = snapshot.sensor_health_reason or "perception_unreliable"
    else:
        snapshot.sensor_health = "healthy"
        snapshot.sensor_health_reason = None
    return snapshot

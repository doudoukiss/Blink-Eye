"""Public-safe camera presence and browser vision grounding helpers."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping


class BrowserCameraPresenceStatus(str, Enum):
    """Public browser camera presence states."""

    DISABLED = "disabled"
    DISCONNECTED = "disconnected"
    WAITING_FOR_FRAME = "waiting_for_frame"
    AVAILABLE = "available"
    STALE = "stale"
    STALLED = "stalled"
    LOOKING = "looking"
    ERROR = "error"


class CameraSceneStatus(str, Enum):
    """Public browser camera scene states for actor-state diagnostics."""

    DISABLED = "disabled"
    PERMISSION_NEEDED = "permission_needed"
    WAITING_FOR_FRAME = "waiting_for_frame"
    AVAILABLE = "available"
    LOOKING = "looking"
    STALE = "stale"
    STALLED = "stalled"
    ERROR = "error"


SCENE_SOCIAL_STATE_V2_SCHEMA_VERSION = 2
SCENE_SOCIAL_TRANSITIONS = {
    "none",
    "camera_ready",
    "looking_requested",
    "frame_captured",
    "vision_answered",
    "vision_stale",
    "vision_unavailable",
}
SCENE_SOCIAL_CAMERA_HONESTY_STATES = {
    "can_see_now",
    "recent_frame_available",
    "available_not_used",
    "unavailable",
}
_SCENE_SOCIAL_VISIBILITY_HINTS = {
    "unknown",
    "visible",
    "not_visible",
    "partial",
    "not_evaluated",
}
_SCENE_SOCIAL_OBJECT_HINTS = {
    "unknown",
    "visible",
    "not_visible",
    "object_showing",
    "not_evaluated",
}
_RECENT_FRAME_EVIDENCE_MAX_AGE_MS = 5000


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _safe_bool(value: Any) -> bool:
    return value is True


def _safe_float(value: Any, *, default: float = 0.0, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_text(value: Any, *, limit: int = 96) -> str:
    return " ".join(str(value or "").split())[:limit]


def _safe_optional_text(value: Any, *, limit: int = 96) -> str | None:
    text = _safe_text(value, limit=limit)
    return text or None


def _stable_text_hash(value: Any, *, prefix: int = 16) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:prefix]


def _safe_enum(value: Any, allowed: set[str], *, default: str) -> str:
    text = _safe_text(value, limit=80)
    return text if text in allowed else default


def _confidence_bucket(value: float | None) -> str:
    if value is None or value <= 0:
        return "none"
    if value < 0.5:
        return "low"
    if value < 0.75:
        return "medium"
    return "high"


def _safe_reason_codes(values: Any, *, fallback: tuple[str, ...] = ()) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else fallback
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in list(raw_values)[:24]:
        code = _safe_text(raw_value)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= 16:
            break
    return result or list(fallback)


def _safe_frame_age_ms(camera_buffer: Any, fallback: Any = None) -> int | None:
    fallback_age = _safe_int(fallback)
    if fallback_age is not None:
        return fallback_age
    received_at = getattr(camera_buffer, "latest_camera_frame_received_monotonic", None)
    if received_at is None:
        return None
    try:
        parsed = float(received_at)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0, int((time.monotonic() - parsed) * 1000))


def _has_recent_frame_evidence(camera_buffer: Any, frame_age_ms: int | None = None) -> bool:
    """Return true when the browser has delivered a public-safe recent frame."""
    frame_seq = _latest_frame_seq(camera_buffer)
    if frame_seq <= 0:
        return False
    age_ms = frame_age_ms if frame_age_ms is not None else _safe_frame_age_ms(camera_buffer)
    return age_ms is not None and age_ms <= _RECENT_FRAME_EVIDENCE_MAX_AGE_MS


def _camera_health_dict(camera_health: Any) -> dict[str, Any]:
    if camera_health is None:
        return {}
    as_dict = getattr(camera_health, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return dict(camera_health) if isinstance(camera_health, dict) else {}


def _latest_frame_seq(camera_buffer: Any) -> int:
    return _safe_int(getattr(camera_buffer, "latest_camera_frame_seq", None)) or 0


def _latest_frame_timestamp(camera_buffer: Any, health_payload: dict[str, Any]) -> str | None:
    value = getattr(camera_buffer, "latest_camera_frame_received_at", None)
    if value:
        return _safe_text(value, limit=96)
    value = health_payload.get("last_fresh_frame_at")
    return _safe_text(value, limit=96) if value else None


def _camera_scene_status_value(value: Any) -> str:
    text = _safe_text(getattr(value, "value", value), limit=48)
    try:
        return CameraSceneStatus(text).value
    except ValueError:
        return CameraSceneStatus.WAITING_FOR_FRAME.value


def _camera_scene_degradation(
    *,
    status: str,
    active_client_id: str | None,
    active_session_id: str | None,
    reason_codes: list[str],
) -> dict[str, Any]:
    active = bool(active_client_id or active_session_id)
    if status == CameraSceneStatus.ERROR.value:
        state = "error"
        components = ["camera", "vision"]
    elif status in {
        CameraSceneStatus.PERMISSION_NEEDED.value,
        CameraSceneStatus.STALE.value,
        CameraSceneStatus.STALLED.value,
    } or (active and status == CameraSceneStatus.WAITING_FOR_FRAME.value):
        state = "degraded"
        components = ["camera", "vision"]
    else:
        state = "ok"
        components = []
    return {
        "state": state,
        "components": components,
        "reason_codes": _safe_reason_codes(
            [f"camera_scene_degradation:{state}", *reason_codes],
            fallback=(f"camera_scene_degradation:{state}",),
        ),
    }


def infer_scene_social_hints_from_moondream(
    description: Any,
    *,
    language: str = "unknown",
) -> dict[str, Any]:
    """Return public-safe scene-social labels derived from a Moondream answer.

    The raw vision answer is inspected only in memory. The returned payload
    contains bounded labels, counts, and a deterministic hash reference.
    """
    text = " ".join(str(description or "").split())
    lowered = text.lower()
    has_text = bool(text)
    language = _safe_text(language, limit=32) or "unknown"
    absent_person = any(
        token in lowered
        for token in (
            "no person",
            "no people",
            "没有人",
            "未看到人",
            "看不到人",
        )
    )
    face_terms = ("face", "faces", "facial", "脸", "面部")
    hand_terms = ("hand", "hands", "holding", "held", "手", "拿着", "手里", "握着")
    body_terms = ("body", "person", "people", "human", "someone", "身体", "人", "人物")
    object_terms = (
        "object",
        "item",
        "thing",
        "phone",
        "cup",
        "book",
        "paper",
        "device",
        "物体",
        "东西",
        "物品",
        "手机",
        "杯子",
        "书",
        "纸",
    )
    has_face = any(token in lowered or token in text for token in face_terms)
    has_hands = any(token in lowered or token in text for token in hand_terms)
    has_body = any(token in lowered or token in text for token in body_terms)
    has_object = any(token in lowered or token in text for token in object_terms)
    object_showing = has_hands and has_object
    object_showing_likelihood = 0.0
    if object_showing:
        object_showing_likelihood = 0.85
    elif has_object:
        object_showing_likelihood = 0.65
    elif has_hands:
        object_showing_likelihood = 0.55
    confidence = 0.0
    if has_text:
        confidence = max(0.45, object_showing_likelihood, 0.7 if has_body or has_face else 0.0)
    return {
        "user_presence_hint": "absent"
        if absent_person
        else "present"
        if has_body or has_face or has_hands
        else "unknown",
        "face_hint": "visible" if has_face else "not_evaluated",
        "body_hint": "visible" if has_body else "not_evaluated",
        "hands_hint": "visible" if has_hands else "not_evaluated",
        "object_hint": "object_showing"
        if object_showing
        else "visible"
        if has_object
        else "not_evaluated",
        "object_showing_likelihood": object_showing_likelihood,
        "last_grounding_summary": "fresh_object_grounding"
        if object_showing_likelihood >= 0.65
        else "fresh_scene_grounding"
        if has_text
        else None,
        "last_grounding_summary_hash": _stable_text_hash(text),
        "confidence": confidence,
        "confidence_bucket": _confidence_bucket(confidence),
        "last_moondream_result_state": "answered" if has_text else "unavailable",
        "scene_change_reason": "vision_answered" if has_text else "vision_unavailable",
        "scene_transition": "vision_answered" if has_text else "vision_unavailable",
        "reason_codes": [
            "scene_social:v2",
            "scene_social:moondream_grounded",
            f"scene_social_language:{language}",
        ],
    }


def _scene_status_from_inputs(
    *,
    vision_enabled: bool,
    active_client_id: str | None,
    active_session_id: str | None,
    browser_media: dict[str, Any],
    health_payload: dict[str, Any],
    camera_buffer: Any,
    tracker: "BrowserVisionGroundingTracker",
) -> CameraSceneStatus:
    if not vision_enabled:
        return CameraSceneStatus.DISABLED

    tracker_state = str(getattr(tracker.state, "value", tracker.state) or "")
    if tracker_state == BrowserCameraPresenceStatus.LOOKING.value:
        return CameraSceneStatus.LOOKING
    if tracker_state == BrowserCameraPresenceStatus.ERROR.value:
        result_state = _safe_text(tracker.last_result_state, limit=48)
        if result_state in {
            "unavailable",
            "permission_denied",
            "device_not_found",
            "disconnected",
        }:
            return CameraSceneStatus.PERMISSION_NEEDED
        if result_state == "waiting_for_frame":
            return CameraSceneStatus.WAITING_FOR_FRAME
        if result_state == "stale":
            return CameraSceneStatus.STALE
        return CameraSceneStatus.ERROR

    media_camera_state = _safe_text(browser_media.get("camera_state"), limit=48)
    media_mode = _safe_text(browser_media.get("mode"), limit=48)
    frame_age_ms = _safe_frame_age_ms(camera_buffer, health_payload.get("frame_age_ms"))
    recent_frame = _has_recent_frame_evidence(camera_buffer, frame_age_ms)
    if active_client_id and media_camera_state in {"permission_denied", "device_not_found"}:
        return CameraSceneStatus.PERMISSION_NEEDED
    if media_camera_state == "error":
        return CameraSceneStatus.ERROR
    if recent_frame:
        return CameraSceneStatus.AVAILABLE
    if media_camera_state == "stalled":
        return CameraSceneStatus.STALLED
    if media_camera_state == "stale":
        return CameraSceneStatus.STALE
    if active_client_id and (
        media_mode == "audio_only" or media_camera_state == "unavailable"
    ):
        return CameraSceneStatus.PERMISSION_NEEDED

    health_state = _safe_text(health_payload.get("camera_track_state"), limit=64)
    health_reason = _safe_text(health_payload.get("sensor_health_reason"), limit=96)
    if health_state in {"stalled", "recovering"}:
        if health_reason in {"camera_frame_stale", "camera_manual_reload_required"}:
            return CameraSceneStatus.STALE
        return CameraSceneStatus.STALLED
    if _safe_bool(health_payload.get("camera_fresh")):
        return CameraSceneStatus.AVAILABLE
    if health_state == "waiting_for_frame":
        return CameraSceneStatus.WAITING_FOR_FRAME

    frame_seq = _latest_frame_seq(camera_buffer)
    if frame_seq > 0:
        if health_payload and not _safe_bool(health_payload.get("camera_fresh")):
            return CameraSceneStatus.STALE
        return CameraSceneStatus.AVAILABLE
    if not active_client_id and not active_session_id:
        return CameraSceneStatus.WAITING_FOR_FRAME
    if frame_age_ms is not None:
        return CameraSceneStatus.STALE
    return CameraSceneStatus.WAITING_FOR_FRAME


def _status_from_health(
    *,
    vision_enabled: bool,
    active_client_id: str | None,
    browser_media: dict[str, Any],
    health_payload: dict[str, Any],
    camera_buffer: Any,
    tracker_state: str,
) -> BrowserCameraPresenceStatus:
    if not vision_enabled:
        return BrowserCameraPresenceStatus.DISABLED
    if tracker_state == BrowserCameraPresenceStatus.LOOKING.value:
        return BrowserCameraPresenceStatus.LOOKING
    if not active_client_id:
        return BrowserCameraPresenceStatus.DISCONNECTED

    media_camera_state = _safe_text(browser_media.get("camera_state"), limit=48)
    media_mode = _safe_text(browser_media.get("mode"), limit=48)
    frame_age_ms = _safe_frame_age_ms(camera_buffer, health_payload.get("frame_age_ms"))
    recent_frame = _has_recent_frame_evidence(camera_buffer, frame_age_ms)
    if media_camera_state in {"permission_denied", "device_not_found", "error"}:
        return BrowserCameraPresenceStatus.DISCONNECTED
    if recent_frame:
        return BrowserCameraPresenceStatus.AVAILABLE
    if media_camera_state == "stalled":
        return BrowserCameraPresenceStatus.STALLED
    if media_camera_state == "stale":
        return BrowserCameraPresenceStatus.STALE
    if media_mode == "audio_only" and not recent_frame:
        return BrowserCameraPresenceStatus.DISCONNECTED
    if media_camera_state == "unavailable" and not recent_frame:
        return BrowserCameraPresenceStatus.DISCONNECTED

    health_state = _safe_text(health_payload.get("camera_track_state"), limit=64)
    health_reason = _safe_text(health_payload.get("sensor_health_reason"), limit=96)
    if health_state == "disconnected":
        return BrowserCameraPresenceStatus.DISCONNECTED
    if health_state == "waiting_for_frame":
        return BrowserCameraPresenceStatus.WAITING_FOR_FRAME
    if health_state in {"stalled", "recovering"}:
        if health_reason in {"camera_frame_stale", "camera_manual_reload_required"}:
            return BrowserCameraPresenceStatus.STALE
        return BrowserCameraPresenceStatus.STALLED
    if _safe_bool(health_payload.get("camera_fresh")):
        return BrowserCameraPresenceStatus.AVAILABLE
    if _latest_frame_seq(camera_buffer) > 0:
        return BrowserCameraPresenceStatus.AVAILABLE
    return BrowserCameraPresenceStatus.WAITING_FOR_FRAME


@dataclass(frozen=True)
class SceneSocialStateV2:
    """Public-safe scene-social state nested under camera scene state."""

    state_id: str
    profile: str
    language: str
    camera_status: str
    vision_status: str
    camera_honesty_state: str
    frame_freshness: str
    frame_age_ms: int | None = None
    latest_frame_sequence: int = 0
    last_used_frame_sequence: int | None = None
    last_used_frame_age_ms: int | None = None
    user_presence_hint: str = "unknown"
    face_hint: str = "not_evaluated"
    body_hint: str = "not_evaluated"
    hands_hint: str = "not_evaluated"
    object_hint: str = "not_evaluated"
    object_showing_likelihood: float = 0.0
    scene_change_reason: str = "none"
    scene_transition: str = "none"
    last_moondream_result_state: str = "none"
    last_grounding_summary: str | None = None
    last_grounding_summary_hash: str | None = None
    confidence: float = 0.0
    confidence_bucket: str = "none"
    scene_age_ms: int | None = None
    updated_at_ms: int = field(default_factory=_now_ms)
    reason_codes: tuple[str, ...] = ("scene_social:v2",)

    def as_dict(self) -> dict[str, Any]:
        """Return the public API payload."""
        transition = _safe_enum(
            self.scene_transition,
            SCENE_SOCIAL_TRANSITIONS,
            default="none",
        )
        honesty = _safe_enum(
            self.camera_honesty_state,
            SCENE_SOCIAL_CAMERA_HONESTY_STATES,
            default="unavailable",
        )
        confidence = _safe_float(self.confidence)
        reason_codes = _safe_reason_codes(
            (
                "scene_social:v2",
                f"scene_social_transition:{transition}",
                f"camera_honesty:{honesty}",
                *self.reason_codes,
            ),
            fallback=("scene_social:v2", f"scene_social_transition:{transition}"),
        )
        return {
            "schema_version": SCENE_SOCIAL_STATE_V2_SCHEMA_VERSION,
            "state_id": _safe_text(self.state_id, limit=96) or "scene-social-v2",
            "profile": _safe_text(self.profile, limit=96) or "manual",
            "language": _safe_text(self.language, limit=32) or "unknown",
            "camera_status": _safe_enum(
                self.camera_status,
                {"unknown", "permission_needed", "ready", "disabled", "unavailable", "error"},
                default="unknown",
            ),
            "vision_status": _safe_enum(
                self.vision_status,
                {"idle", "looking", "answered", "stale", "unavailable", "degraded"},
                default="idle",
            ),
            "camera_honesty_state": honesty,
            "frame_freshness": _safe_enum(
                self.frame_freshness,
                {"none", "fresh", "recent", "stale", "waiting", "disabled", "unknown"},
                default="unknown",
            ),
            "frame_age_ms": (
                max(0, int(self.frame_age_ms)) if self.frame_age_ms is not None else None
            ),
            "latest_frame_sequence": max(0, int(self.latest_frame_sequence or 0)),
            "last_used_frame_sequence": (
                max(0, int(self.last_used_frame_sequence))
                if self.last_used_frame_sequence is not None
                else None
            ),
            "last_used_frame_age_ms": (
                max(0, int(self.last_used_frame_age_ms))
                if self.last_used_frame_age_ms is not None
                else None
            ),
            "user_presence_hint": _safe_enum(
                self.user_presence_hint,
                {"unknown", "present", "absent", "partial"},
                default="unknown",
            ),
            "face_hint": _safe_enum(
                self.face_hint,
                _SCENE_SOCIAL_VISIBILITY_HINTS,
                default="not_evaluated",
            ),
            "body_hint": _safe_enum(
                self.body_hint,
                _SCENE_SOCIAL_VISIBILITY_HINTS,
                default="not_evaluated",
            ),
            "hands_hint": _safe_enum(
                self.hands_hint,
                _SCENE_SOCIAL_VISIBILITY_HINTS,
                default="not_evaluated",
            ),
            "object_hint": _safe_enum(
                self.object_hint,
                _SCENE_SOCIAL_OBJECT_HINTS,
                default="not_evaluated",
            ),
            "object_showing_likelihood": _safe_float(self.object_showing_likelihood),
            "scene_change_reason": _safe_enum(
                self.scene_change_reason,
                SCENE_SOCIAL_TRANSITIONS,
                default="none",
            ),
            "scene_transition": transition,
            "last_moondream_result_state": _safe_enum(
                self.last_moondream_result_state,
                {"none", "looking", "answered", "unavailable", "error", "stale"},
                default="none",
            ),
            "last_grounding_summary": _safe_optional_text(
                self.last_grounding_summary,
                limit=80,
            ),
            "last_grounding_summary_hash": _safe_optional_text(
                self.last_grounding_summary_hash,
                limit=32,
            ),
            "confidence": confidence,
            "confidence_bucket": _safe_enum(
                self.confidence_bucket or _confidence_bucket(confidence),
                {"none", "low", "medium", "high"},
                default=_confidence_bucket(confidence),
            ),
            "scene_age_ms": (
                max(0, int(self.scene_age_ms)) if self.scene_age_ms is not None else None
            ),
            "updated_at_ms": max(0, int(self.updated_at_ms or 0)),
            "reason_codes": reason_codes,
        }


@dataclass
class BrowserVisionGroundingTracker:
    """Track whether the current browser answer used on-demand vision."""

    state: BrowserCameraPresenceStatus | str = BrowserCameraPresenceStatus.DISCONNECTED
    last_result_state: str = "none"
    current_answer_used_vision: bool = False
    grounding_mode: str = "none"
    last_vision_used_at: str | None = None
    last_vision_frame_seq: int | None = None
    last_vision_frame_age_ms: int | None = None
    scene_transition: str = "none"
    scene_change_reason: str = "none"
    user_presence_hint: str = "unknown"
    face_hint: str = "not_evaluated"
    body_hint: str = "not_evaluated"
    hands_hint: str = "not_evaluated"
    object_hint: str = "not_evaluated"
    object_showing_likelihood: float = 0.0
    last_moondream_result_state: str = "none"
    last_grounding_summary: str | None = None
    last_grounding_summary_hash: str | None = None
    scene_confidence: float = 0.0
    scene_confidence_bucket: str = "none"
    scene_updated_at_ms: int = field(default_factory=_now_ms)
    reason_codes: list[str] = field(default_factory=lambda: ["camera_grounding:idle"])

    def reset_current_answer(self) -> None:
        """Clear current-answer grounding while preserving last result status."""
        self.current_answer_used_vision = False
        self.grounding_mode = "none"
        self.scene_transition = "none"
        self.scene_change_reason = "none"
        self.scene_updated_at_ms = _now_ms()
        self.reason_codes = _safe_reason_codes(
            ("camera_grounding:current_answer_reset",),
            fallback=("camera_grounding:current_answer_reset",),
        )

    def note_scene_transition(
        self,
        transition: str,
        *,
        frame_seq: int | None = None,
        frame_age_ms: int | None = None,
        reason_code: str | None = None,
    ) -> None:
        """Record a public-safe scene-state transition without raw media."""
        transition = _safe_enum(transition, SCENE_SOCIAL_TRANSITIONS, default="none")
        self.scene_transition = transition
        self.scene_change_reason = transition
        if frame_seq is not None:
            self.last_vision_frame_seq = frame_seq
        if frame_age_ms is not None:
            self.last_vision_frame_age_ms = frame_age_ms
        if transition == "vision_stale":
            self.last_moondream_result_state = "stale"
        elif transition == "vision_unavailable":
            self.last_moondream_result_state = "unavailable"
        elif transition == "looking_requested":
            self.last_moondream_result_state = "looking"
        self.scene_updated_at_ms = _now_ms()
        self.reason_codes = _safe_reason_codes(
            (
                reason_code or f"scene_social_transition:{transition}",
                f"scene_social_transition:{transition}",
            ),
            fallback=(f"scene_social_transition:{transition}",),
        )

    def mark_looking(self, *, frame_seq: int | None = None, frame_age_ms: int | None = None) -> None:
        """Mark an active vision lookup."""
        self.state = BrowserCameraPresenceStatus.LOOKING
        self.last_result_state = "looking"
        self.last_vision_frame_seq = frame_seq
        self.last_vision_frame_age_ms = frame_age_ms
        self.scene_transition = "looking_requested"
        self.scene_change_reason = "looking_requested"
        self.last_moondream_result_state = "looking"
        self.scene_updated_at_ms = _now_ms()
        self.reason_codes = ["camera_grounding:looking"]

    def mark_success(
        self,
        *,
        frame_seq: int | None,
        frame_age_ms: int | None,
        scene_social_hints: Mapping[str, Any] | None = None,
    ) -> None:
        """Mark a successful single-frame vision result."""
        hints = dict(scene_social_hints or {})
        self.state = BrowserCameraPresenceStatus.AVAILABLE
        self.last_result_state = "success"
        self.current_answer_used_vision = True
        self.grounding_mode = "single_frame"
        self.last_vision_used_at = _now_iso()
        self.last_vision_frame_seq = frame_seq
        self.last_vision_frame_age_ms = frame_age_ms
        self.scene_transition = "vision_answered"
        self.scene_change_reason = "vision_answered"
        self.user_presence_hint = _safe_enum(
            hints.get("user_presence_hint"),
            {"unknown", "present", "absent", "partial"},
            default=self.user_presence_hint,
        )
        self.face_hint = _safe_enum(
            hints.get("face_hint"),
            _SCENE_SOCIAL_VISIBILITY_HINTS,
            default=self.face_hint,
        )
        self.body_hint = _safe_enum(
            hints.get("body_hint"),
            _SCENE_SOCIAL_VISIBILITY_HINTS,
            default=self.body_hint,
        )
        self.hands_hint = _safe_enum(
            hints.get("hands_hint"),
            _SCENE_SOCIAL_VISIBILITY_HINTS,
            default=self.hands_hint,
        )
        self.object_hint = _safe_enum(
            hints.get("object_hint"),
            _SCENE_SOCIAL_OBJECT_HINTS,
            default=self.object_hint,
        )
        self.object_showing_likelihood = _safe_float(
            hints.get("object_showing_likelihood"),
            default=self.object_showing_likelihood,
        )
        self.last_moondream_result_state = _safe_enum(
            hints.get("last_moondream_result_state"),
            {"none", "looking", "answered", "unavailable", "error", "stale"},
            default="answered",
        )
        self.last_grounding_summary = _safe_optional_text(
            hints.get("last_grounding_summary"),
            limit=80,
        )
        self.last_grounding_summary_hash = _safe_optional_text(
            hints.get("last_grounding_summary_hash"),
            limit=32,
        )
        self.scene_confidence = _safe_float(hints.get("confidence"), default=0.7)
        self.scene_confidence_bucket = _safe_enum(
            hints.get("confidence_bucket"),
            {"none", "low", "medium", "high"},
            default=_confidence_bucket(self.scene_confidence),
        )
        self.scene_updated_at_ms = _now_ms()
        self.reason_codes = _safe_reason_codes(
            [
                "camera_grounding:single_frame",
                "vision:last_result_success",
                "scene_social_transition:vision_answered",
                *_safe_reason_codes(hints.get("reason_codes"), fallback=()),
            ],
            fallback=("camera_grounding:single_frame", "vision:last_result_success"),
        )

    def mark_error(
        self,
        *,
        result_state: str = "error",
        reason_code: str = "vision:last_result_error",
        frame_seq: int | None = None,
        frame_age_ms: int | None = None,
    ) -> None:
        """Mark a failed or unusable vision lookup."""
        self.state = BrowserCameraPresenceStatus.ERROR
        self.last_result_state = _safe_text(result_state, limit=48) or "error"
        self.last_vision_frame_seq = frame_seq
        self.last_vision_frame_age_ms = frame_age_ms
        transition = "vision_stale" if self.last_result_state == "stale" else "vision_unavailable"
        if self.last_result_state in {"error", "vision_result_unavailable"}:
            self.last_moondream_result_state = "error"
        elif self.last_result_state == "stale":
            self.last_moondream_result_state = "stale"
        else:
            self.last_moondream_result_state = "unavailable"
        self.scene_transition = transition
        self.scene_change_reason = transition
        self.scene_confidence = 0.0
        self.scene_confidence_bucket = "none"
        self.scene_updated_at_ms = _now_ms()
        self.reason_codes = _safe_reason_codes(
            (
                reason_code,
                f"vision:last_result:{self.last_result_state}",
                f"scene_social_transition:{transition}",
            ),
            fallback=("vision:last_result_error",),
        )


def _scene_social_camera_status(
    *,
    enabled: bool,
    state: str,
    available: bool,
) -> str:
    if not enabled or state == CameraSceneStatus.DISABLED.value:
        return "disabled"
    if state == CameraSceneStatus.PERMISSION_NEEDED.value:
        return "permission_needed"
    if state == CameraSceneStatus.ERROR.value:
        return "error"
    if available or state in {
        CameraSceneStatus.AVAILABLE.value,
        CameraSceneStatus.LOOKING.value,
        CameraSceneStatus.STALE.value,
        CameraSceneStatus.STALLED.value,
    }:
        return "ready"
    if state == CameraSceneStatus.WAITING_FOR_FRAME.value:
        return "unknown"
    return "unavailable"


def _scene_social_vision_status(state: str, last_result_state: str) -> str:
    if state == CameraSceneStatus.LOOKING.value or last_result_state == "looking":
        return "looking"
    if last_result_state == "success":
        return "answered"
    if state in {CameraSceneStatus.STALE.value, CameraSceneStatus.STALLED.value}:
        return "stale"
    if state in {CameraSceneStatus.DISABLED.value, CameraSceneStatus.PERMISSION_NEEDED.value}:
        return "unavailable"
    if state == CameraSceneStatus.ERROR.value:
        return "degraded"
    return "idle"


def _camera_honesty_state(
    *,
    enabled: bool,
    available: bool,
    camera_fresh: bool,
    current_answer_used_vision: bool,
    state: str,
) -> str:
    if current_answer_used_vision and camera_fresh and state != CameraSceneStatus.ERROR.value:
        return "can_see_now"
    if not enabled or state in {
        CameraSceneStatus.DISABLED.value,
        CameraSceneStatus.PERMISSION_NEEDED.value,
        CameraSceneStatus.ERROR.value,
    }:
        return "unavailable"
    if camera_fresh:
        return "recent_frame_available"
    if available:
        return "available_not_used"
    return "unavailable"


def _scene_social_transition(
    *,
    tracker: BrowserVisionGroundingTracker,
    state: str,
    current_answer_used_vision: bool,
    camera_fresh: bool,
) -> str:
    tracker_transition = _safe_enum(
        tracker.scene_transition,
        SCENE_SOCIAL_TRANSITIONS,
        default="none",
    )
    if tracker_transition != "none":
        return tracker_transition
    if current_answer_used_vision:
        return "vision_answered"
    if state == CameraSceneStatus.LOOKING.value:
        return "looking_requested"
    if state in {CameraSceneStatus.STALE.value, CameraSceneStatus.STALLED.value}:
        return "vision_stale"
    if state in {
        CameraSceneStatus.DISABLED.value,
        CameraSceneStatus.PERMISSION_NEEDED.value,
        CameraSceneStatus.ERROR.value,
    }:
        return "vision_unavailable"
    if camera_fresh:
        return "camera_ready"
    return "none"


def _fallback_scene_social_state_v2(
    *,
    profile: str,
    language: str,
    state: str,
    enabled: bool,
    available: bool,
    camera_fresh: bool,
    freshness_state: str,
    latest_frame_sequence: int,
    latest_frame_age_ms: int | None,
    current_answer_used_vision: bool,
    last_result_state: str,
    reason_codes: list[str],
) -> SceneSocialStateV2:
    tracker = BrowserVisionGroundingTracker()
    transition = _scene_social_transition(
        tracker=tracker,
        state=state,
        current_answer_used_vision=current_answer_used_vision,
        camera_fresh=camera_fresh,
    )
    honesty = _camera_honesty_state(
        enabled=enabled,
        available=available,
        camera_fresh=camera_fresh,
        current_answer_used_vision=current_answer_used_vision,
        state=state,
    )
    state_hash = _stable_text_hash(
        f"{profile}:{language}:{state}:{freshness_state}:{latest_frame_sequence}:{transition}"
    )
    return SceneSocialStateV2(
        state_id=f"scene-social-v2-{state_hash or 'unavailable'}",
        profile=profile,
        language=language,
        camera_status=_scene_social_camera_status(
            enabled=enabled,
            state=state,
            available=available,
        ),
        vision_status=_scene_social_vision_status(state, last_result_state),
        camera_honesty_state=honesty,
        frame_freshness=freshness_state if freshness_state != "unknown" else "none",
        frame_age_ms=latest_frame_age_ms,
        latest_frame_sequence=latest_frame_sequence,
        scene_change_reason=transition,
        scene_transition=transition,
        last_moondream_result_state="answered" if last_result_state == "success" else "none",
        scene_age_ms=latest_frame_age_ms,
        reason_codes=tuple(reason_codes),
    )


def _build_scene_social_state_v2(
    *,
    profile: str,
    language: str,
    enabled: bool,
    available: bool,
    state: str,
    camera_fresh: bool,
    freshness_state: str,
    latest_frame_sequence: int,
    latest_frame_age_ms: int | None,
    current_answer_used_vision: bool,
    tracker: BrowserVisionGroundingTracker,
    reason_codes: list[str],
) -> SceneSocialStateV2:
    transition = _scene_social_transition(
        tracker=tracker,
        state=state,
        current_answer_used_vision=current_answer_used_vision,
        camera_fresh=camera_fresh,
    )
    honesty = _camera_honesty_state(
        enabled=enabled,
        available=available,
        camera_fresh=camera_fresh,
        current_answer_used_vision=current_answer_used_vision,
        state=state,
    )
    last_moondream_result_state = _safe_enum(
        tracker.last_moondream_result_state,
        {"none", "looking", "answered", "unavailable", "error", "stale"},
        default="answered" if tracker.last_result_state == "success" else "none",
    )
    state_hash = _stable_text_hash(
        (
            f"{profile}:{language}:{state}:{freshness_state}:{latest_frame_sequence}:"
            f"{transition}:{honesty}:{last_moondream_result_state}"
        )
    )
    return SceneSocialStateV2(
        state_id=f"scene-social-v2-{state_hash or 'unavailable'}",
        profile=profile,
        language=language,
        camera_status=_scene_social_camera_status(
            enabled=enabled,
            state=state,
            available=available,
        ),
        vision_status=_scene_social_vision_status(state, tracker.last_result_state),
        camera_honesty_state=honesty,
        frame_freshness=freshness_state if freshness_state != "unknown" else "none",
        frame_age_ms=latest_frame_age_ms,
        latest_frame_sequence=latest_frame_sequence,
        last_used_frame_sequence=tracker.last_vision_frame_seq,
        last_used_frame_age_ms=tracker.last_vision_frame_age_ms,
        user_presence_hint=tracker.user_presence_hint,
        face_hint=tracker.face_hint,
        body_hint=tracker.body_hint,
        hands_hint=tracker.hands_hint,
        object_hint=tracker.object_hint,
        object_showing_likelihood=tracker.object_showing_likelihood,
        scene_change_reason=transition,
        scene_transition=transition,
        last_moondream_result_state=last_moondream_result_state,
        last_grounding_summary=tracker.last_grounding_summary,
        last_grounding_summary_hash=tracker.last_grounding_summary_hash,
        confidence=tracker.scene_confidence,
        confidence_bucket=tracker.scene_confidence_bucket,
        scene_age_ms=tracker.last_vision_frame_age_ms or latest_frame_age_ms,
        updated_at_ms=tracker.scene_updated_at_ms,
        reason_codes=tuple(
            _safe_reason_codes(
                [
                    *reason_codes,
                    f"scene_social_transition:{transition}",
                    f"camera_honesty:{honesty}",
                ],
                fallback=("scene_social:v2",),
            )
        ),
    )


@dataclass(frozen=True)
class CameraSceneState:
    """Public-safe camera scene state for browser actor-state and diagnostics."""

    profile: str
    language: str
    enabled: bool
    available: bool
    state: CameraSceneStatus | str
    vision_backend: str = "none"
    continuous_perception_enabled: bool = False
    permission_state: str = "unknown"
    camera_connected: bool = False
    camera_fresh: bool = False
    freshness_state: str = "unknown"
    track_state: str = "unknown"
    latest_frame_sequence: int = 0
    latest_frame_age_ms: int | None = None
    latest_frame_received_at: str | None = None
    on_demand_vision_state: str = "idle"
    current_answer_used_vision: bool = False
    grounding_mode: str = "none"
    last_vision_result_state: str = "none"
    last_used_frame_sequence: int | None = None
    last_used_frame_age_ms: int | None = None
    last_used_frame_at: str | None = None
    degradation: dict[str, Any] | None = None
    scene_social_state_v2: SceneSocialStateV2 | None = None
    reason_codes: tuple[str, ...] = ("camera_scene:v1",)

    def as_dict(self) -> dict[str, Any]:
        """Return the public API payload."""
        state_value = _camera_scene_status_value(self.state)
        reason_codes = _safe_reason_codes(
            (
                "camera_scene:v1",
                f"camera_scene:{state_value}",
                f"camera_grounding:{self.grounding_mode}",
                *self.reason_codes,
            ),
            fallback=("camera_scene:v1", f"camera_scene:{state_value}"),
        )
        degradation = self.degradation or _camera_scene_degradation(
            status=state_value,
            active_client_id=None,
            active_session_id=None,
            reason_codes=reason_codes,
        )
        return {
            "schema_version": 1,
            "profile": _safe_text(self.profile, limit=96) or "manual",
            "language": _safe_text(self.language, limit=32) or "unknown",
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "state": state_value,
            "status": state_value,
            "vision_backend": _safe_text(self.vision_backend, limit=80) or "none",
            "continuous_perception_enabled": bool(self.continuous_perception_enabled),
            "permission_state": _safe_text(self.permission_state, limit=48) or "unknown",
            "camera_connected": bool(self.camera_connected),
            "camera_fresh": bool(self.camera_fresh),
            "freshness_state": _safe_text(self.freshness_state, limit=48) or "unknown",
            "track_state": _safe_text(self.track_state, limit=64) or "unknown",
            "latest_frame_sequence": max(0, int(self.latest_frame_sequence or 0)),
            "latest_frame_age_ms": (
                max(0, int(self.latest_frame_age_ms))
                if self.latest_frame_age_ms is not None
                else None
            ),
            "latest_frame_received_at": self.latest_frame_received_at,
            "on_demand_vision_state": (
                _safe_text(self.on_demand_vision_state, limit=48) or "idle"
            ),
            "current_answer_used_vision": bool(self.current_answer_used_vision),
            "grounding_mode": _safe_text(self.grounding_mode, limit=48) or "none",
            "last_vision_result_state": (
                _safe_text(self.last_vision_result_state, limit=48) or "none"
            ),
            "last_used_frame_sequence": (
                max(0, int(self.last_used_frame_sequence))
                if self.last_used_frame_sequence is not None
                else None
            ),
            "last_used_frame_age_ms": (
                max(0, int(self.last_used_frame_age_ms))
                if self.last_used_frame_age_ms is not None
                else None
            ),
            "last_used_frame_at": self.last_used_frame_at,
            "degradation": {
                "state": _safe_text(degradation.get("state"), limit=48) or "ok",
                "components": _safe_reason_codes(degradation.get("components"), fallback=()),
                "reason_codes": _safe_reason_codes(
                    degradation.get("reason_codes"),
                    fallback=(f"camera_scene_degradation:{state_value}",),
                ),
            },
            "scene_social_state_v2": (
                self.scene_social_state_v2.as_dict()
                if self.scene_social_state_v2 is not None
                else _fallback_scene_social_state_v2(
                    profile=self.profile,
                    language=self.language,
                    state=state_value,
                    enabled=self.enabled,
                    available=self.available,
                    camera_fresh=self.camera_fresh,
                    freshness_state=self.freshness_state,
                    latest_frame_sequence=self.latest_frame_sequence,
                    latest_frame_age_ms=self.latest_frame_age_ms,
                    current_answer_used_vision=self.current_answer_used_vision,
                    last_result_state=self.last_vision_result_state,
                    reason_codes=reason_codes,
                ).as_dict()
            ),
            "reason_codes": reason_codes,
        }


@dataclass(frozen=True)
class BrowserCameraPresenceSnapshot:
    """Public browser camera presence snapshot."""

    enabled: bool
    available: bool
    state: BrowserCameraPresenceStatus | str
    camera_connected: bool = False
    camera_fresh: bool = False
    track_state: str = "unknown"
    latest_frame_seq: int = 0
    latest_frame_age_ms: int | None = None
    last_fresh_frame_at: str | None = None
    last_vision_used_at: str | None = None
    last_vision_result_state: str = "none"
    current_answer_used_vision: bool = False
    grounding_mode: str = "none"
    continuous_perception_enabled: bool = False
    reason_codes: tuple[str, ...] = ("camera_presence:v1",)

    def as_dict(self) -> dict[str, Any]:
        """Return the public API payload."""
        state_value = str(getattr(self.state, "value", self.state) or "disabled")
        reason_codes = _safe_reason_codes(
            (
                "camera_presence:v1",
                f"camera_presence:{state_value}",
                f"camera_grounding:{self.grounding_mode}",
                *self.reason_codes,
            ),
            fallback=("camera_presence:v1", f"camera_presence:{state_value}"),
        )
        return {
            "schema_version": 1,
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "state": state_value,
            "camera_connected": bool(self.camera_connected),
            "camera_fresh": bool(self.camera_fresh),
            "track_state": self.track_state,
            "latest_frame_seq": max(0, int(self.latest_frame_seq or 0)),
            "latest_frame_age_ms": (
                max(0, int(self.latest_frame_age_ms))
                if self.latest_frame_age_ms is not None
                else None
            ),
            "last_fresh_frame_at": self.last_fresh_frame_at,
            "last_vision_used_at": self.last_vision_used_at,
            "last_vision_result_state": self.last_vision_result_state,
            "current_answer_used_vision": bool(self.current_answer_used_vision),
            "grounding_mode": self.grounding_mode,
            "continuous_perception_enabled": bool(self.continuous_perception_enabled),
            "reason_codes": reason_codes,
        }


def build_browser_camera_presence_snapshot(
    *,
    vision_enabled: bool,
    continuous_perception_enabled: bool,
    browser_media: dict[str, Any] | None,
    active_client_id: str | None,
    camera_buffer: Any = None,
    camera_health: Any = None,
    grounding_tracker: BrowserVisionGroundingTracker | None = None,
) -> BrowserCameraPresenceSnapshot:
    """Build a public-safe camera presence snapshot from runtime components."""
    tracker = grounding_tracker or BrowserVisionGroundingTracker()
    media_payload = dict(browser_media or {})
    health_payload = _camera_health_dict(camera_health)
    state = _status_from_health(
        vision_enabled=vision_enabled,
        active_client_id=active_client_id,
        browser_media=media_payload,
        health_payload=health_payload,
        camera_buffer=camera_buffer,
        tracker_state=str(getattr(tracker.state, "value", tracker.state)),
    )
    frame_age_ms = _safe_frame_age_ms(camera_buffer, health_payload.get("frame_age_ms"))
    frame_seq = _latest_frame_seq(camera_buffer)
    recent_frame = _has_recent_frame_evidence(camera_buffer, frame_age_ms)
    camera_connected = _safe_bool(health_payload.get("camera_connected"))
    if vision_enabled and active_client_id and (
        recent_frame or not health_payload
    ):
        camera_connected = media_payload.get("mode") == "camera_and_microphone"
        camera_connected = camera_connected or recent_frame
    camera_fresh = state in {BrowserCameraPresenceStatus.AVAILABLE, BrowserCameraPresenceStatus.LOOKING}
    camera_fresh = camera_fresh and state != BrowserCameraPresenceStatus.STALE
    if health_payload and not recent_frame:
        camera_fresh = _safe_bool(health_payload.get("camera_fresh"))
    grounding_mode = (
        "single_frame"
        if tracker.current_answer_used_vision
        else "continuous_perception"
        if continuous_perception_enabled
        else "none"
    )
    reason_codes = [
        _safe_text(health_payload.get("sensor_health_reason"), limit=96),
        *_safe_reason_codes(getattr(tracker, "reason_codes", ())),
    ]
    return BrowserCameraPresenceSnapshot(
        enabled=vision_enabled,
        available=state in {BrowserCameraPresenceStatus.AVAILABLE, BrowserCameraPresenceStatus.LOOKING},
        state=state,
        camera_connected=camera_connected,
        camera_fresh=camera_fresh,
        track_state=(
            _safe_text(media_payload.get("camera_state"), limit=64)
            if recent_frame
            else _safe_text(health_payload.get("camera_track_state"), limit=64)
            or _safe_text(media_payload.get("camera_state"), limit=64)
            or "unknown"
        )
        or "ready",
        latest_frame_seq=frame_seq,
        latest_frame_age_ms=frame_age_ms,
        last_fresh_frame_at=_latest_frame_timestamp(camera_buffer, health_payload),
        last_vision_used_at=tracker.last_vision_used_at,
        last_vision_result_state=tracker.last_result_state,
        current_answer_used_vision=tracker.current_answer_used_vision,
        grounding_mode=grounding_mode,
        continuous_perception_enabled=continuous_perception_enabled,
        reason_codes=tuple(code for code in reason_codes if code),
    )


def build_camera_scene_state(
    *,
    profile: str,
    language: str,
    vision_enabled: bool,
    continuous_perception_enabled: bool,
    browser_media: dict[str, Any] | None,
    active_client_id: str | None,
    active_session_id: str | None = None,
    camera_buffer: Any = None,
    camera_health: Any = None,
    grounding_tracker: BrowserVisionGroundingTracker | None = None,
    vision_backend: str = "moondream",
) -> CameraSceneState:
    """Build the schema-v1 public camera scene state."""
    tracker = grounding_tracker or BrowserVisionGroundingTracker()
    media_payload = dict(browser_media or {})
    health_payload = _camera_health_dict(camera_health)
    state = _scene_status_from_inputs(
        vision_enabled=vision_enabled,
        active_client_id=active_client_id,
        active_session_id=active_session_id,
        browser_media=media_payload,
        health_payload=health_payload,
        camera_buffer=camera_buffer,
        tracker=tracker,
    )
    state_value = state.value
    frame_seq = _latest_frame_seq(camera_buffer)
    frame_age_ms = _safe_frame_age_ms(camera_buffer, health_payload.get("frame_age_ms"))
    recent_frame = _has_recent_frame_evidence(camera_buffer, frame_age_ms)
    camera_connected = _safe_bool(health_payload.get("camera_connected"))
    if vision_enabled and active_client_id and (
        recent_frame or not health_payload
    ):
        camera_connected = media_payload.get("mode") == "camera_and_microphone"
        camera_connected = camera_connected or recent_frame
    camera_fresh = state in {CameraSceneStatus.AVAILABLE, CameraSceneStatus.LOOKING}
    if health_payload and not recent_frame:
        camera_fresh = _safe_bool(health_payload.get("camera_fresh"))
    permission_state = "granted" if camera_connected else "unknown"
    if state == CameraSceneStatus.DISABLED:
        permission_state = "disabled"
    elif state == CameraSceneStatus.PERMISSION_NEEDED:
        permission_state = "permission_needed"
    freshness_state = (
        "disabled"
        if state == CameraSceneStatus.DISABLED
        else "fresh"
        if camera_fresh
        else "stale"
        if state in {CameraSceneStatus.STALE, CameraSceneStatus.STALLED}
        else "waiting"
        if state == CameraSceneStatus.WAITING_FOR_FRAME
        else "unknown"
    )
    grounding_mode = (
        "single_frame"
        if tracker.current_answer_used_vision
        else "continuous_perception"
        if continuous_perception_enabled
        else "none"
    )
    on_demand_state = "idle"
    if state == CameraSceneStatus.LOOKING:
        on_demand_state = "looking"
    elif tracker.last_result_state == "success":
        on_demand_state = "success"
    elif state == CameraSceneStatus.ERROR or tracker.last_result_state not in {"none", "looking", "success"}:
        on_demand_state = "error"
    reason_codes = _safe_reason_codes(
        [
            _safe_text(health_payload.get("sensor_health_reason"), limit=96),
            *_safe_reason_codes(getattr(tracker, "reason_codes", ())),
        ],
        fallback=("camera_scene:v1", f"camera_scene:{state_value}"),
    )
    degradation = _camera_scene_degradation(
        status=state_value,
        active_client_id=active_client_id,
        active_session_id=active_session_id,
        reason_codes=reason_codes,
    )
    available = state in {CameraSceneStatus.AVAILABLE, CameraSceneStatus.LOOKING}
    scene_social_state = _build_scene_social_state_v2(
        profile=profile,
        language=language,
        enabled=vision_enabled,
        available=available,
        state=state_value,
        camera_fresh=camera_fresh,
        freshness_state=freshness_state,
        latest_frame_sequence=frame_seq,
        latest_frame_age_ms=frame_age_ms,
        current_answer_used_vision=tracker.current_answer_used_vision,
        tracker=tracker,
        reason_codes=reason_codes,
    )
    return CameraSceneState(
        profile=profile,
        language=language,
        enabled=vision_enabled,
        available=available,
        state=state,
        vision_backend=vision_backend if vision_enabled else "none",
        continuous_perception_enabled=continuous_perception_enabled,
        permission_state=permission_state,
        camera_connected=camera_connected,
        camera_fresh=camera_fresh,
        freshness_state=freshness_state,
        track_state=(
            _safe_text(media_payload.get("camera_state"), limit=64)
            if recent_frame
            else _safe_text(health_payload.get("camera_track_state"), limit=64)
            or _safe_text(media_payload.get("camera_state"), limit=64)
            or "unknown"
        )
        or "ready",
        latest_frame_sequence=frame_seq,
        latest_frame_age_ms=frame_age_ms,
        latest_frame_received_at=_latest_frame_timestamp(camera_buffer, health_payload),
        on_demand_vision_state=on_demand_state,
        current_answer_used_vision=tracker.current_answer_used_vision,
        grounding_mode=grounding_mode,
        last_vision_result_state=tracker.last_result_state,
        last_used_frame_sequence=tracker.last_vision_frame_seq,
        last_used_frame_age_ms=tracker.last_vision_frame_age_ms,
        last_used_frame_at=tracker.last_vision_used_at,
        degradation=degradation,
        scene_social_state_v2=scene_social_state,
        reason_codes=tuple(reason_codes),
    )

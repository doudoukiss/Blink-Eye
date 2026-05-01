"""Public-safe browser actor-state snapshot schema v2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from blink.interaction.actor_events import ActorEventV2
from blink.interaction.webrtc_audio_health import WebRTCAudioHealthV2

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_MAX_REASON_CODES = 32
_MAX_MAPPING_ITEMS = 32
_MAX_LIST_ITEMS = 16
_MAX_DEPTH = 5
_MAX_TEXT_CHARS = 180
_UNSAFE_KEY_EXACT = {
    "audio",
    "candidate",
    "content",
    "credentials",
    "hidden_prompt",
    "ice",
    "ice_candidate",
    "image",
    "messages",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "token",
}
_UNSAFE_KEY_FRAGMENTS = {
    "authorization",
    "audio",
    "candidate",
    "credential",
    "hidden",
    "image",
    "message",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "source_ref",
    "token",
    "url",
}
_SAFE_KEY_SUFFIXES = (
    "_available",
    "_chars",
    "_count",
    "_counts",
    "_enabled",
    "_id",
    "_index",
    "_kind",
    "_ms",
    "_state",
)
_SAFE_KEY_EXACT = {
    "audio_level",
    "auto_gain_control",
    "barge_in",
    "barge_in_state",
    "echo_cancellation",
    "echo_risk",
    "echo_risk_state",
    "echo_safe",
    "echo_safe_source",
    "input_track",
    "input_track_state",
    "last_text_kind",
    "noise_suppression",
    "output_playback",
    "output_playback_state",
    "output_route",
    "text_kind",
    "webrtc_audio_health",
}
_UNSAFE_VALUE_TOKENS = (
    "-----begin",
    "a=candidate",
    "authorization:",
    "base64,",
    "bearer ",
    "candidate:",
    "data:audio",
    "data:image",
    "ice-ufrag",
    "m=audio",
    "m=video",
    "o=-",
    "password",
    "prompt",
    "secret",
    "sk-",
    "token",
    "traceback",
    "v=0",
)
_MICROPHONE_ERROR_STATES = {
    "device_not_found",
    "error",
    "permission_denied",
    "unavailable",
}
_MICROPHONE_DEGRADED_STATES = {"stalled"}
_CAMERA_DEGRADED_STATES = {
    "device_not_found",
    "error",
    "permission_denied",
    "stale",
    "stalled",
    "unavailable",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _enum_text(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    raw_value = getattr(value, "value", value)
    text = _TOKEN_RE.sub("_", str(raw_value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_text(value: object, *, default: str = "", limit: int = _MAX_TEXT_CHARS) -> str:
    text = " ".join(str(value or "").split())[:limit]
    if not text:
        return default
    lowered = text.lower()
    if any(token in lowered for token in _UNSAFE_VALUE_TOKENS):
        return default
    return text


def _safe_bool(value: object) -> bool:
    return value is True


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _key_is_safe(key: object) -> bool:
    text = _enum_text(key, default="", limit=80).lower()
    if not text or text in _UNSAFE_KEY_EXACT:
        return False
    if text in _SAFE_KEY_EXACT:
        return True
    if text.endswith(_SAFE_KEY_SUFFIXES):
        return True
    return not any(fragment in text for fragment in _UNSAFE_KEY_FRAGMENTS)


def _dedupe_reason_codes(values: object, *, limit: int = _MAX_REASON_CODES) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        code = _enum_text(raw_value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def _safe_mapping_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = _safe_text(value, limit=_MAX_TEXT_CHARS)
        return text or None
    if depth >= _MAX_DEPTH:
        return _enum_text(type(value).__name__, default="object", limit=64)
    if isinstance(value, (list, tuple, set)):
        result: list[object] = []
        for item in list(value)[:_MAX_LIST_ITEMS]:
            sanitized = _safe_mapping_value(item, depth=depth + 1)
            if sanitized is not None:
                result.append(sanitized)
        return result
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for raw_key, raw_item in list(value.items())[:_MAX_MAPPING_ITEMS]:
            if not _key_is_safe(raw_key):
                continue
            key = _enum_text(raw_key, default="", limit=80)
            if not key:
                continue
            sanitized = _safe_mapping_value(raw_item, depth=depth + 1)
            if sanitized is not None:
                result[key] = sanitized
        return result
    return _enum_text(type(value).__name__, default="object", limit=64)


def _safe_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized = _safe_mapping_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _bounded_live_text(value: object) -> str | None:
    text = _safe_text(value, limit=160)
    return text or None


def _safe_session_id(value: object) -> str | None:
    if value in (None, ""):
        return None
    return _enum_text(value, default="", limit=96) or None


def _dedupe_components(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        component = _enum_text(value, default="", limit=40)
        if not component or component in seen:
            continue
        seen.add(component)
        result.append(component)
        if len(result) >= 8:
            break
    return result


@dataclass
class BrowserActorStateV2:
    """Single public browser actor-state snapshot for UI and diagnostics."""

    profile: str
    language: str
    mode: object
    tts_backend: str
    tts_label: str
    protected_playback: bool
    browser_media: dict[str, Any] = field(default_factory=dict)
    vision_enabled: bool = False
    vision_backend: str = "none"
    continuous_perception_enabled: bool = False
    interruption: dict[str, Any] = field(default_factory=dict)
    active_listening: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Any] = field(default_factory=dict)
    camera_presence: dict[str, Any] = field(default_factory=dict)
    camera_scene: dict[str, Any] = field(default_factory=dict)
    memory_persona: dict[str, Any] = field(default_factory=dict)
    conversation_floor: dict[str, Any] = field(default_factory=dict)
    webrtc_audio_health: dict[str, Any] = field(default_factory=dict)
    active_session_id: str | None = None
    active_client_id: str | None = None
    last_actor_event: ActorEventV2 | None = None
    memory_available: bool = False
    updated_at: str = field(default_factory=_now_iso)
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the state as a public-safe schema-v2 payload."""
        mode = _enum_text(self.mode, default="waiting", limit=40)
        raw_active_listening = self.active_listening if isinstance(self.active_listening, dict) else {}
        raw_speech = self.speech if isinstance(self.speech, dict) else {}
        live_text = {
            "partial_transcript": _bounded_live_text(
                raw_active_listening.get("display_partial_transcript")
            ),
            "final_transcript": _bounded_live_text(
                raw_active_listening.get("display_final_transcript")
            ),
            "assistant_subtitle": _bounded_live_text(raw_speech.get("assistant_subtitle")),
        }
        browser_media = _safe_mapping(self.browser_media)
        active_listening = _safe_mapping(self.active_listening)
        speech = _safe_mapping(self.speech)
        camera_presence = _safe_mapping(self.camera_presence)
        camera_scene = _safe_mapping(self.camera_scene)
        memory_persona = _safe_mapping(self.memory_persona)
        interruption = _safe_mapping(self.interruption)
        conversation_floor = _safe_mapping(self.conversation_floor)
        webrtc_audio_health = _safe_mapping(self.webrtc_audio_health)
        if not conversation_floor:
            conversation_floor = {
                "schema_version": 1,
                "state": "unknown",
                "profile": _enum_text(self.profile, default="manual", limit=96),
                "language": _enum_text(self.language, default="unknown", limit=32),
                "floor_model_version": 3,
                "sub_state": "handoff_complete",
                "user_has_floor": False,
                "assistant_has_floor": False,
                "overlap": False,
                "handoff": False,
                "repair": False,
                "unknown": True,
                "assistant_speaking": False,
                "user_speaking": False,
                "protected_playback": self.protected_playback is True,
                "barge_in_armed": self.protected_playback is not True,
                "last_input_type": "none",
                "last_text_kind": "empty",
                "phrase_class": "empty",
                "phrase_confidence": 1.0,
                "phrase_confidence_bucket": "high",
                "yield_decision": "wait_for_input",
                "echo_risk": "unknown",
                "barge_in_state": "protected"
                if self.protected_playback is True
                else "armed",
                "tts_chunk_role": "unknown",
                "low_confidence_transcript": False,
                "last_transition_at": None,
                "updated_at": self.updated_at,
                "transition_count": 0,
                "counts": {},
                "reason_codes": [
                    "conversation_floor:v1",
                    "conversation_floor:v3",
                    "floor_state:unknown",
                ],
            }
        else:
            conversation_floor.setdefault("last_transition_at", None)
        if not webrtc_audio_health:
            webrtc_audio_health = WebRTCAudioHealthV2(
                profile=self.profile,
                language=self.language,
                protected_playback=self.protected_playback is True,
                explicit_barge_in_armed=self.protected_playback is not True,
            ).as_dict()
        if not camera_scene:
            camera_scene = self._fallback_camera_scene(camera_presence=camera_presence)
        camera_scene = self._camera_scene_with_nullable_defaults(camera_scene)
        camera_state = _enum_text(browser_media.get("camera_state"), default="unknown", limit=40)
        microphone_state = _enum_text(
            browser_media.get("microphone_state"),
            default="unknown",
            limit=40,
        )
        media_mode = _enum_text(browser_media.get("mode"), default="unreported", limit=40)
        active_session_id = _safe_session_id(self.active_session_id)
        active_client_id = _safe_session_id(self.active_client_id)
        last_event = self.last_actor_event.as_dict() if self.last_actor_event is not None else None
        degradation = self._degradation_payload(
            browser_media=browser_media,
            camera_presence=camera_presence,
            camera_scene=camera_scene,
            speech=speech,
            last_actor_event=last_event,
            active_session_id=active_session_id,
            active_client_id=active_client_id,
        )
        reason_codes = _dedupe_reason_codes(
            [
                "browser_actor_state:v2",
                f"mode:{mode}",
                f"profile:{_enum_text(self.profile, default='manual', limit=96)}",
                f"language:{_enum_text(self.language, default='unknown', limit=32)}",
                f"tts_backend:{_enum_text(self.tts_backend, default='unknown', limit=80)}",
                f"vision_backend:{_enum_text(self.vision_backend, default='none', limit=80)}",
                f"degradation:{degradation['state']}",
                *self.reason_codes,
                *interruption.get("reason_codes", []),
                *webrtc_audio_health.get("reason_codes", []),
                *active_listening.get("reason_codes", []),
                *speech.get("reason_codes", []),
                *conversation_floor.get("reason_codes", []),
                *camera_presence.get("reason_codes", []),
                *camera_scene.get("reason_codes", []),
                *memory_persona.get("reason_codes", []),
                *degradation.get("reason_codes", []),
            ]
        )
        return {
            "schema_version": 2,
            "available": True,
            "runtime": "browser",
            "transport": "WebRTC",
            "profile": _enum_text(self.profile, default="manual", limit=96),
            "language": _enum_text(self.language, default="unknown", limit=32),
            "mode": mode,
            "tts": {
                "backend": _enum_text(self.tts_backend, default="unknown", limit=80),
                "label": _safe_text(self.tts_label, default="unknown", limit=96),
            },
            "webrtc": {
                "media_mode": media_mode,
                "session_active": active_session_id is not None,
                "client_active": active_client_id is not None,
                "media": browser_media,
                "reason_codes": _dedupe_reason_codes(browser_media.get("reason_codes", [])),
            },
            "microphone": {
                "state": microphone_state,
                "enabled": True,
                "available": microphone_state in {"ready", "receiving"},
                "reason_codes": _dedupe_reason_codes(
                    [
                        f"browser_microphone:{microphone_state}",
                        *browser_media.get("reason_codes", []),
                    ],
                    limit=12,
                ),
            },
            "camera": {
                "state": camera_state,
                "enabled": self.vision_enabled is True,
                "available": camera_scene.get("available") is True
                or camera_state in {"ready", "receiving"},
                "presence": camera_presence,
                "scene": camera_scene,
                "reason_codes": _dedupe_reason_codes(
                    [
                        f"browser_camera:{camera_state}",
                        *camera_presence.get("reason_codes", []),
                        *camera_scene.get("reason_codes", []),
                    ],
                    limit=12,
                ),
            },
            "vision": {
                "enabled": self.vision_enabled is True,
                "backend": (
                    _enum_text(self.vision_backend, default="none", limit=80)
                    if self.vision_enabled
                    else "none"
                ),
                "continuous_perception_enabled": self.continuous_perception_enabled is True,
                "state": _enum_text(
                    camera_scene.get("state") or camera_presence.get("state"),
                    default="disabled" if not self.vision_enabled else "unknown",
                    limit=40,
                ),
                "available": camera_scene.get("available") is True,
                "grounding_mode": _enum_text(
                    camera_scene.get("grounding_mode") or camera_presence.get("grounding_mode"),
                    default="none",
                    limit=40,
                ),
                "current_answer_used_vision": camera_scene.get(
                    "current_answer_used_vision"
                )
                is True,
                "reason_codes": _dedupe_reason_codes(
                    [
                        *camera_presence.get("reason_codes", []),
                        *camera_scene.get("reason_codes", []),
                    ]
                ),
            },
            "protected_playback": self.protected_playback is True,
            "interruption": interruption,
            "webrtc_audio_health": webrtc_audio_health,
            "active_listening": active_listening,
            "speech": speech,
            "conversation_floor": conversation_floor,
            "camera_scene": camera_scene,
            "memory_persona": memory_persona,
            "memory": {"available": self.memory_available is True},
            "degradation": degradation,
            "live_text": live_text,
            "last_actor_event": last_event,
            "last_actor_event_id": (
                _safe_int(last_event.get("event_id")) if isinstance(last_event, dict) else 0
            ),
            "last_actor_event_at": (
                _safe_text(last_event.get("timestamp"), limit=64)
                if isinstance(last_event, dict)
                else None
            ),
            "active_session_id": active_session_id,
            "active_client_id": active_client_id,
            "updated_at": _safe_text(self.updated_at, default=_now_iso(), limit=64),
            "reason_codes": reason_codes,
        }

    def _degradation_payload(
        self,
        *,
        browser_media: dict[str, Any],
        camera_presence: dict[str, Any],
        camera_scene: dict[str, Any],
        speech: dict[str, Any],
        last_actor_event: dict[str, Any] | None,
        active_session_id: str | None,
        active_client_id: str | None,
    ) -> dict[str, Any]:
        active = active_session_id is not None or active_client_id is not None
        if not active:
            return {
                "state": "ok",
                "components": [],
                "reason_codes": ["runtime:waiting_for_client"],
            }

        components: list[str] = []
        reason_codes: list[str] = []
        state = "ok"
        event_type = _enum_text(
            last_actor_event.get("event_type") if isinstance(last_actor_event, dict) else None,
            default="none",
            limit=40,
        )
        if event_type == "error":
            state = "error"
            components.append("runtime")
            reason_codes.append("actor_event:error")
        elif event_type == "degraded":
            state = "degraded"
            components.append("runtime")
            reason_codes.append("actor_event:degraded")

        media_mode = _enum_text(browser_media.get("mode"), default="unreported", limit=40)
        microphone_state = _enum_text(
            browser_media.get("microphone_state"),
            default="unknown",
            limit=40,
        )
        camera_state = _enum_text(browser_media.get("camera_state"), default="unknown", limit=40)
        if media_mode == "unavailable":
            state = "error"
            components.append("runtime")
            reason_codes.append("browser_media:unavailable")
        if microphone_state in _MICROPHONE_ERROR_STATES:
            state = "error"
            components.append("microphone")
            reason_codes.append(f"browser_microphone:{microphone_state}")
        elif microphone_state in _MICROPHONE_DEGRADED_STATES and state != "error":
            state = "degraded"
            components.append("microphone")
            reason_codes.append(f"browser_microphone:{microphone_state}")

        if self.vision_enabled and camera_state in _CAMERA_DEGRADED_STATES:
            if state != "error":
                state = "degraded"
            components.extend(["camera", "vision"])
            reason_codes.append(f"browser_camera:{camera_state}")
        camera_presence_state = _enum_text(
            camera_presence.get("state"),
            default="unknown",
            limit=40,
        )
        if self.vision_enabled and camera_presence_state in {"stale", "stalled", "error"}:
            if state != "error":
                state = "degraded"
            components.extend(["camera", "vision"])
            reason_codes.append(f"camera_presence:{camera_presence_state}")
        camera_scene_state = _enum_text(
            camera_scene.get("state"),
            default="unknown",
            limit=40,
        )
        if self.vision_enabled and camera_scene_state == "error":
            state = "error"
            components.extend(["camera", "vision"])
            reason_codes.append("camera_scene:error")
        elif self.vision_enabled and camera_scene_state in {
            "permission_needed",
            "stale",
            "stalled",
        }:
            if state != "error":
                state = "degraded"
            components.extend(["camera", "vision"])
            reason_codes.append(f"camera_scene:{camera_scene_state}")
        if _safe_int(speech.get("stale_chunk_drop_count")) > 0 and state != "error":
            state = "degraded"
            components.append("speech")
            reason_codes.append("speech:stale_chunk_drops_observed")
        if state == "ok":
            reason_codes.append("runtime:ok")
        return {
            "state": state,
            "components": _dedupe_components(components),
            "reason_codes": _dedupe_reason_codes(reason_codes, limit=16),
        }

    def _fallback_camera_scene(self, *, camera_presence: dict[str, Any]) -> dict[str, Any]:
        state = _enum_text(
            camera_presence.get("state"),
            default="disabled" if not self.vision_enabled else "waiting_for_frame",
            limit=40,
        )
        return {
            "schema_version": 1,
            "profile": _enum_text(self.profile, default="manual", limit=96),
            "language": _enum_text(self.language, default="unknown", limit=32),
            "enabled": self.vision_enabled is True,
            "available": camera_presence.get("available") is True,
            "state": state,
            "status": state,
            "vision_backend": (
                _enum_text(self.vision_backend, default="none", limit=80)
                if self.vision_enabled
                else "none"
            ),
            "continuous_perception_enabled": self.continuous_perception_enabled is True,
            "permission_state": "unknown",
            "camera_connected": camera_presence.get("camera_connected") is True,
            "camera_fresh": camera_presence.get("camera_fresh") is True,
            "freshness_state": "fresh"
            if camera_presence.get("camera_fresh") is True
            else "unknown",
            "track_state": _enum_text(camera_presence.get("track_state"), default="unknown"),
            "latest_frame_sequence": _safe_int(camera_presence.get("latest_frame_seq")),
            "latest_frame_age_ms": camera_presence.get("latest_frame_age_ms"),
            "latest_frame_received_at": _safe_text(
                camera_presence.get("last_fresh_frame_at"),
                limit=64,
            )
            or None,
            "on_demand_vision_state": "idle",
            "current_answer_used_vision": camera_presence.get("current_answer_used_vision")
            is True,
            "grounding_mode": _enum_text(
                camera_presence.get("grounding_mode"),
                default="none",
            ),
            "last_vision_result_state": _enum_text(
                camera_presence.get("last_vision_result_state"),
                default="none",
            ),
            "last_used_frame_sequence": None,
            "last_used_frame_age_ms": None,
            "last_used_frame_at": _safe_text(
                camera_presence.get("last_vision_used_at"),
                limit=64,
            )
            or None,
            "degradation": {
                "state": "ok",
                "components": [],
                "reason_codes": ["camera_scene_degradation:ok"],
            },
            "scene_social_state_v2": {
                "schema_version": 2,
                "state_id": f"scene-social-v2-{state}",
                "profile": _enum_text(self.profile, default="manual", limit=96),
                "language": _enum_text(self.language, default="unknown", limit=32),
                "camera_status": "disabled" if not self.vision_enabled else "unknown",
                "vision_status": "unavailable" if not self.vision_enabled else "idle",
                "camera_honesty_state": "unavailable",
                "frame_freshness": "disabled" if not self.vision_enabled else "unknown",
                "frame_age_ms": camera_presence.get("latest_frame_age_ms"),
                "latest_frame_sequence": _safe_int(camera_presence.get("latest_frame_seq")),
                "last_used_frame_sequence": None,
                "last_used_frame_age_ms": None,
                "user_presence_hint": "unknown",
                "face_hint": "not_evaluated",
                "body_hint": "not_evaluated",
                "hands_hint": "not_evaluated",
                "object_hint": "not_evaluated",
                "object_showing_likelihood": 0.0,
                "scene_change_reason": "vision_unavailable"
                if not self.vision_enabled
                else "none",
                "scene_transition": "vision_unavailable" if not self.vision_enabled else "none",
                "last_moondream_result_state": "none",
                "last_grounding_summary": None,
                "last_grounding_summary_hash": None,
                "confidence": 0.0,
                "confidence_bucket": "none",
                "scene_age_ms": camera_presence.get("latest_frame_age_ms"),
                "updated_at_ms": 0,
                "reason_codes": ["scene_social:v2", "camera_honesty:unavailable"],
            },
            "reason_codes": _dedupe_reason_codes(
                ["camera_scene:v1", f"camera_scene:{state}"],
                limit=12,
            ),
        }

    @staticmethod
    def _camera_scene_with_nullable_defaults(camera_scene: dict[str, Any]) -> dict[str, Any]:
        camera_scene = dict(camera_scene)
        state = _enum_text(camera_scene.get("state"), default="waiting_for_frame", limit=40)
        camera_scene.setdefault("status", state)
        for key in (
            "latest_frame_age_ms",
            "latest_frame_received_at",
            "last_used_frame_sequence",
            "last_used_frame_age_ms",
            "last_used_frame_at",
        ):
            camera_scene.setdefault(key, None)
        camera_scene.setdefault(
            "scene_social_state_v2",
            {
                "schema_version": 2,
                "state_id": f"scene-social-v2-{state}",
                "profile": _enum_text(camera_scene.get("profile"), default="manual", limit=96),
                "language": _enum_text(
                    camera_scene.get("language"),
                    default="unknown",
                    limit=32,
                ),
                "camera_status": "disabled" if state == "disabled" else "unknown",
                "vision_status": "unavailable" if state == "disabled" else "idle",
                "camera_honesty_state": "unavailable",
                "frame_freshness": _enum_text(
                    camera_scene.get("freshness_state"),
                    default="unknown",
                    limit=40,
                ),
                "frame_age_ms": camera_scene.get("latest_frame_age_ms"),
                "latest_frame_sequence": _safe_int(camera_scene.get("latest_frame_sequence")),
                "last_used_frame_sequence": camera_scene.get("last_used_frame_sequence"),
                "last_used_frame_age_ms": camera_scene.get("last_used_frame_age_ms"),
                "user_presence_hint": "unknown",
                "face_hint": "not_evaluated",
                "body_hint": "not_evaluated",
                "hands_hint": "not_evaluated",
                "object_hint": "not_evaluated",
                "object_showing_likelihood": 0.0,
                "scene_change_reason": "none",
                "scene_transition": "none",
                "last_moondream_result_state": "none",
                "last_grounding_summary": None,
                "last_grounding_summary_hash": None,
                "confidence": 0.0,
                "confidence_bucket": "none",
                "scene_age_ms": camera_scene.get("latest_frame_age_ms"),
                "updated_at_ms": 0,
                "reason_codes": ["scene_social:v2", "camera_honesty:unavailable"],
            },
        )
        scene_social = (
            dict(camera_scene["scene_social_state_v2"])
            if isinstance(camera_scene.get("scene_social_state_v2"), dict)
            else {}
        )
        scene_social.setdefault("frame_age_ms", camera_scene.get("latest_frame_age_ms"))
        scene_social.setdefault(
            "last_used_frame_sequence",
            camera_scene.get("last_used_frame_sequence"),
        )
        scene_social.setdefault(
            "last_used_frame_age_ms",
            camera_scene.get("last_used_frame_age_ms"),
        )
        scene_social.setdefault("last_grounding_summary", None)
        scene_social.setdefault("last_grounding_summary_hash", None)
        scene_social.setdefault("scene_age_ms", camera_scene.get("latest_frame_age_ms"))
        camera_scene["scene_social_state_v2"] = scene_social
        return camera_scene

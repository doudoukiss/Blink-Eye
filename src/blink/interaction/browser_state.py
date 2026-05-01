"""Public-safe browser interaction state snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from blink.interaction.performance_events import (
    BrowserInteractionMode,
    BrowserPerformanceEvent,
)


def _now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _dedupe_reason_codes(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= 24:
            break
    return result


@dataclass
class BrowserInteractionState:
    """Public browser runtime performance state."""

    mode: BrowserInteractionMode = BrowserInteractionMode.WAITING
    profile: str = "manual"
    tts_label: str = "unknown"
    tts_backend: str = "unknown"
    protected_playback: bool = True
    browser_media: dict[str, Any] = field(default_factory=dict)
    vision_enabled: bool = False
    continuous_perception_enabled: bool = False
    memory_available: bool = False
    interruption: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Any] = field(default_factory=dict)
    active_listening: dict[str, Any] = field(default_factory=dict)
    camera_presence: dict[str, Any] = field(default_factory=dict)
    camera_scene: dict[str, Any] = field(default_factory=dict)
    memory_persona: dict[str, Any] = field(default_factory=dict)
    active_session_id: Optional[str] = None
    active_client_id: Optional[str] = None
    last_event: BrowserPerformanceEvent | None = None
    updated_at: str = field(default_factory=_now_iso)
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the state as a public API payload."""
        camera_state = str(self.browser_media.get("camera_state") or "unknown")
        microphone_state = str(self.browser_media.get("microphone_state") or "unknown")
        media_mode = str(self.browser_media.get("mode") or "unreported")
        interruption = dict(self.interruption)
        speech = dict(self.speech)
        active_listening = dict(self.active_listening)
        camera_presence = dict(self.camera_presence)
        camera_scene = dict(self.camera_scene)
        memory_persona = dict(self.memory_persona)
        interruption_state = str(interruption.get("barge_in_state") or "unknown")
        interruption_decision = str(interruption.get("last_decision") or "none")
        speech_mode = str(speech.get("director_mode") or "unavailable")
        active_listening_phase = str(active_listening.get("phase") or "idle")
        camera_presence_state = str(camera_presence.get("state") or "disabled")
        camera_scene_state = str(camera_scene.get("state") or camera_presence_state)
        memory_persona_state = (
            "available" if memory_persona.get("available") is True else "unavailable"
        )
        event_reason_codes = self.last_event.reason_codes if self.last_event is not None else []
        reason_codes = _dedupe_reason_codes(
            (
                "browser_performance_state:v1",
                f"mode:{self.mode.value}",
                f"profile:{self.profile}",
                f"browser_media:{media_mode}",
                f"browser_camera:{camera_state}",
                f"browser_microphone:{microphone_state}",
                "protected_playback:on" if self.protected_playback else "protected_playback:off",
                "vision:enabled" if self.vision_enabled else "vision:disabled",
                (
                    "continuous_perception:enabled"
                    if self.continuous_perception_enabled
                    else "continuous_perception:disabled"
                ),
                "memory:available" if self.memory_available else "memory:unavailable",
                f"interruption:{interruption_state}",
                f"interruption_last_decision:{interruption_decision}",
                f"speech_director:{speech_mode}",
                f"active_listening:{active_listening_phase}",
                f"camera_presence:{camera_presence_state}",
                f"camera_scene:{camera_scene_state}",
                f"memory_persona:{memory_persona_state}",
                *self.reason_codes,
                *tuple(interruption.get("reason_codes") or ()),
                *tuple(speech.get("reason_codes") or ()),
                *tuple(active_listening.get("reason_codes") or ()),
                *tuple(camera_presence.get("reason_codes") or ()),
                *tuple(camera_scene.get("reason_codes") or ()),
                *tuple(memory_persona.get("reason_codes") or ()),
                *event_reason_codes,
            )
        )
        return {
            "schema_version": 1,
            "available": True,
            "runtime": "browser",
            "transport": "WebRTC",
            "mode": self.mode.value,
            "profile": self.profile,
            "tts": self.tts_label,
            "tts_backend": self.tts_backend,
            "protected_playback": self.protected_playback,
            "browser_media": dict(self.browser_media),
            "camera": {
                "state": camera_state,
                "enabled": self.vision_enabled,
            },
            "vision": {
                "enabled": self.vision_enabled,
                "continuous_perception_enabled": self.continuous_perception_enabled,
            },
            "memory": {
                "available": self.memory_available,
            },
            "interruption": interruption,
            "speech": speech,
            "active_listening": active_listening,
            "camera_presence": camera_presence,
            "camera_scene": camera_scene,
            "memory_persona": memory_persona,
            "active_session_id": self.active_session_id,
            "active_client_id": self.active_client_id,
            "last_event_id": self.last_event.event_id if self.last_event is not None else 0,
            "last_event_type": self.last_event.event_type if self.last_event is not None else None,
            "last_event_at": self.last_event.timestamp if self.last_event is not None else None,
            "updated_at": self.updated_at,
            "reason_codes": reason_codes,
        }

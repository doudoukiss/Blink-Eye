"""Authoritative browser runtime session state for V3 stabilization.

This module intentionally owns only compact, public-safe state. Runtime APIs,
actor events, performance episodes, control frames, and workbench payloads can
derive from it, but it does not depend on FastAPI, WebRTC transports, LLM
contexts, raw media, or persistence sinks.
"""

# ruff: noqa: D102

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

_SAFE_MEDIA_MODES = {"unreported", "camera_and_microphone", "audio_only", "unavailable"}
_SAFE_MEDIA_STATES = {
    "unknown",
    "ready",
    "receiving",
    "stalled",
    "stale",
    "unavailable",
    "permission_denied",
    "device_not_found",
    "error",
}
_HARD_CAMERA_UNAVAILABLE = {"permission_denied", "device_not_found", "error"}
_RECENT_FRAME_MAX_AGE_MS = 5000
_MAX_LIVE_TEXT_CHARS = 180


def _safe_text(value: object, *, limit: int = 120) -> str:
    return " ".join(str(value or "").split())[:limit]


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in {"_", "-", ".", ":", "/"} else "_"
        for ch in str(value or "").strip()
    )
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _stable_hash(value: object, *, prefix: int = 16) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:prefix]


def _bounded_live_text(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text[-_MAX_LIVE_TEXT_CHARS:]


@dataclass
class UserTurnTranscriptState:
    """Current browser STT turn state with bounded live text."""

    turn_index: int = 0
    active: bool = False
    partial_seen: bool = False
    partial_transcript_chars: int = 0
    final_transcript_chars: int = 0
    final_fragment_count: int = 0
    live_text: str = ""

    def start_turn(self) -> None:
        """Reset counters for a new VAD turn."""
        self.turn_index += 1
        self.active = True
        self.partial_seen = False
        self.partial_transcript_chars = 0
        self.final_transcript_chars = 0
        self.final_fragment_count = 0
        self.live_text = ""

    def stop_turn(self) -> None:
        self.active = False

    def note_partial(self, text: object) -> dict[str, object]:
        chars = len(str(text or ""))
        self.partial_seen = True
        self.partial_transcript_chars = chars
        self.live_text = _bounded_live_text(text)
        return self.as_public_counts(partial_available=True, final_available=False)

    def note_final(self, text: object) -> dict[str, object]:
        chars = len(str(text or ""))
        self.final_fragment_count += 1
        self.final_transcript_chars += chars
        self.live_text = _bounded_live_text(f"{self.live_text} {str(text or '').strip()}")
        return self.as_public_counts(
            partial_available=self.partial_seen,
            final_available=True,
            final_fragment_chars=chars,
        )

    def as_public_counts(
        self,
        *,
        partial_available: bool | None = None,
        final_available: bool | None = None,
        final_fragment_chars: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "turn_index": self.turn_index,
            "partial_available": self.partial_seen
            if partial_available is None
            else bool(partial_available),
            "final_available": self.final_fragment_count > 0
            if final_available is None
            else bool(final_available),
            "partial_transcript_chars": self.partial_transcript_chars,
            "final_transcript_chars": self.final_transcript_chars,
            "final_transcript_count": self.final_fragment_count,
            "live_text_hash": _stable_hash(self.live_text),
        }
        if final_fragment_chars is not None:
            payload["final_fragment_chars"] = max(0, int(final_fragment_chars))
        return payload

    def as_dict(self, *, include_live_text: bool = False) -> dict[str, object]:
        payload = self.as_public_counts()
        payload.update(
            {
                "schema_version": 3,
                "active": self.active,
                "reason_codes": ["stt_turn_state:v3"],
            }
        )
        if include_live_text:
            payload["live_text"] = self.live_text
        return payload


@dataclass
class CameraTruthState:
    """Single source of truth for browser camera availability and use."""

    vision_enabled: bool = True
    continuous_perception_enabled: bool = False
    client_connected: bool = False
    media_mode: str = "unreported"
    camera_state: str = "unknown"
    microphone_state: str = "unknown"
    frame_seq: int = 0
    frame_age_ms: int | None = None
    frame_received_monotonic: float | None = None
    current_answer_used_vision: bool = False
    last_result_state: str = "not_used"
    scene_transition: str = "none"
    reason_codes: list[str] = field(default_factory=lambda: ["camera_truth:v3"])

    def note_client_connected(self) -> None:
        self.client_connected = True
        self.reason_codes = ["camera_truth:v3", "client:connected"]

    def note_client_disconnected(self) -> None:
        self.client_connected = False
        self.media_mode = "unreported"
        self.camera_state = "unknown"
        self.microphone_state = "unknown"
        self.current_answer_used_vision = False
        self.last_result_state = "not_used"
        self.scene_transition = "none"
        self.reason_codes = ["camera_truth:v3", "client:disconnected"]

    def update_client_media(self, media: Mapping[str, Any]) -> None:
        mode = _safe_token(media.get("mode"), default="unreported", limit=48)
        camera_state = _safe_token(media.get("camera_state"), default="unknown", limit=48)
        microphone_state = _safe_token(media.get("microphone_state"), default="unknown", limit=48)
        self.media_mode = mode if mode in _SAFE_MEDIA_MODES else "unreported"
        self.camera_state = camera_state if camera_state in _SAFE_MEDIA_STATES else "unknown"
        self.microphone_state = (
            microphone_state if microphone_state in _SAFE_MEDIA_STATES else "unknown"
        )
        self.reason_codes = ["camera_truth:v3", *list(media.get("reason_codes") or [])[:8]]
        if self.camera_available_now():
            self.scene_transition = "camera_ready"
            self.last_result_state = "available_not_used"
        elif self.camera_state in _HARD_CAMERA_UNAVAILABLE or self.media_mode in {
            "audio_only",
            "unavailable",
        }:
            self.current_answer_used_vision = False
            self.last_result_state = "unavailable"
            self.scene_transition = "vision_unavailable"

    def note_frame(
        self,
        *,
        frame_seq: object = None,
        frame_age_ms: object = None,
        received_monotonic: float | None = None,
    ) -> None:
        seq = _safe_int(frame_seq, default=self.frame_seq)
        if seq > 0:
            self.frame_seq = seq
        parsed_age = None if frame_age_ms is None else _safe_int(frame_age_ms)
        self.frame_age_ms = parsed_age if parsed_age is not None else self.frame_age_ms
        self.frame_received_monotonic = received_monotonic or time.monotonic()
        self.media_mode = "camera_and_microphone"
        self.camera_state = "receiving"
        self.scene_transition = "frame_captured"
        self.last_result_state = "recent_frame_available"
        self.reason_codes = ["camera_truth:v3", "camera:frame_received"]

    def note_looking_requested(self) -> None:
        self.scene_transition = "looking_requested"
        self.last_result_state = "looking"
        self.reason_codes = ["camera_truth:v3", "vision:looking_requested"]

    def note_vision_success(self, *, frame_seq: object = None, frame_age_ms: object = None) -> None:
        if frame_seq is not None:
            self.frame_seq = _safe_int(frame_seq, default=self.frame_seq)
        if frame_age_ms is not None:
            self.frame_age_ms = _safe_int(frame_age_ms)
        self.current_answer_used_vision = self.has_recent_frame()
        self.last_result_state = "vision_answered" if self.current_answer_used_vision else "stale"
        self.scene_transition = (
            "vision_answered" if self.current_answer_used_vision else "vision_stale"
        )
        self.reason_codes = ["camera_truth:v3", f"scene_social_transition:{self.scene_transition}"]

    def note_vision_error(
        self,
        *,
        result_state: object = "error",
        frame_seq: object = None,
        frame_age_ms: object = None,
    ) -> None:
        if frame_seq is not None:
            self.frame_seq = _safe_int(frame_seq, default=self.frame_seq)
        if frame_age_ms is not None:
            self.frame_age_ms = _safe_int(frame_age_ms)
        state = _safe_token(result_state, default="error", limit=48)
        self.current_answer_used_vision = False
        self.last_result_state = state
        self.scene_transition = "vision_stale" if state == "stale" else "vision_unavailable"
        self.reason_codes = ["camera_truth:v3", f"vision:{state}"]

    def _computed_frame_age_ms(self) -> int | None:
        if self.frame_received_monotonic is None:
            return self.frame_age_ms
        return max(0, int((time.monotonic() - self.frame_received_monotonic) * 1000))

    def has_recent_frame(self, *, max_age_ms: int = _RECENT_FRAME_MAX_AGE_MS) -> bool:
        if self.frame_seq <= 0:
            return False
        age = self._computed_frame_age_ms()
        return age is not None and age <= max_age_ms

    def camera_available_now(self) -> bool:
        return (
            self.vision_enabled
            and self.client_connected
            and self.media_mode == "camera_and_microphone"
            and self.camera_state in {"ready", "receiving"}
        )

    def hard_unavailable(self) -> bool:
        return (
            not self.vision_enabled
            or self.camera_state in _HARD_CAMERA_UNAVAILABLE
            or self.media_mode in {"audio_only", "unavailable"}
        )

    def camera_honesty_state(self) -> str:
        if self.current_answer_used_vision and self.has_recent_frame():
            return "can_see_now"
        if self.has_recent_frame():
            return "recent_frame_available"
        if self.camera_available_now():
            return "available_not_used"
        return "unavailable"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "vision_enabled": bool(self.vision_enabled),
            "continuous_perception_enabled": bool(self.continuous_perception_enabled),
            "client_connected": bool(self.client_connected),
            "media_mode": self.media_mode,
            "camera_state": self.camera_state,
            "microphone_state": self.microphone_state,
            "frame_seq": self.frame_seq,
            "frame_age_ms": self._computed_frame_age_ms(),
            "has_recent_frame": self.has_recent_frame(),
            "current_answer_used_vision": self.current_answer_used_vision,
            "last_result_state": self.last_result_state,
            "scene_transition": self.scene_transition,
            "camera_honesty_state": self.camera_honesty_state(),
            "reason_codes": list(dict.fromkeys(self.reason_codes))[:16],
        }


@dataclass
class SpeechQueueController:
    """Single owner for assistant speech generation queue state."""

    max_speech_chunk_lookahead: int = 2
    max_subtitle_lookahead: int = 2
    generation_id: str = "speech-unavailable"
    turn_id: str = "turn-unavailable"
    outstanding_speech_chunks: int = 0
    outstanding_subtitles: int = 0
    held_speech_chunks: int = 0
    generation_stale: bool = False
    stale_chunk_drops: int = 0
    last_reason_codes: list[str] = field(default_factory=lambda: ["speech_queue:v3"])

    def start_generation(self, *, generation_id: object, turn_id: object) -> None:
        self.generation_id = _safe_token(generation_id, default="speech-unavailable")
        self.turn_id = _safe_token(turn_id, default="turn-unavailable")
        self.outstanding_speech_chunks = 0
        self.outstanding_subtitles = 0
        self.held_speech_chunks = 0
        self.generation_stale = False
        self.last_reason_codes = ["speech_queue:v3", "speech:generation_start"]

    def can_emit(self) -> bool:
        if self.generation_stale:
            return False
        return (
            self.outstanding_speech_chunks < self.max_speech_chunk_lookahead
            and self.outstanding_subtitles < self.max_subtitle_lookahead
        )

    def note_subtitle_ready(self) -> None:
        self.outstanding_speech_chunks += 1
        self.outstanding_subtitles += 1
        self.held_speech_chunks = max(0, self.held_speech_chunks - 1)
        self.last_reason_codes = ["speech_queue:v3", "speech:subtitle_ready"]

    def note_lookahead_held(self, *, count: object = 1) -> None:
        self.held_speech_chunks = max(self.held_speech_chunks, _safe_int(count, default=1))
        self.last_reason_codes = ["speech_queue:v3", "speech:lookahead_held"]

    def note_tts_stopped(self) -> None:
        self.outstanding_speech_chunks = max(0, self.outstanding_speech_chunks - 1)
        self.outstanding_subtitles = max(0, self.outstanding_subtitles - 1)
        self.last_reason_codes = ["speech_queue:v3", "tts:stopped"]

    def note_interruption(self, *, dropped_count: object = 0) -> None:
        self.generation_stale = True
        self.held_speech_chunks = 0
        self.outstanding_speech_chunks = 0
        self.outstanding_subtitles = 0
        self.stale_chunk_drops += _safe_int(dropped_count)
        self.last_reason_codes = ["speech_queue:v3", "interruption:accepted"]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "generation_id": self.generation_id,
            "turn_id": self.turn_id,
            "speech_chunks_outstanding": self.outstanding_speech_chunks,
            "speech_chunks_limit": self.max_speech_chunk_lookahead,
            "subtitles_outstanding": self.outstanding_subtitles,
            "subtitles_limit": self.max_subtitle_lookahead,
            "held_speech_chunks": self.held_speech_chunks,
            "generation_stale": self.generation_stale,
            "stale_chunk_drops": self.stale_chunk_drops,
            "can_emit": self.can_emit(),
            "reason_codes": list(self.last_reason_codes),
        }


@dataclass
class BrowserRuntimeSessionV3:
    """Compact V3 core for live browser runtime truth."""

    profile: str
    language: str
    tts_runtime_label: str
    vision_enabled: bool
    continuous_perception_enabled: bool
    protected_playback: bool
    session_id: str | None = None
    client_id: str | None = None
    transcript: UserTurnTranscriptState = field(default_factory=UserTurnTranscriptState)
    camera: CameraTruthState = field(default_factory=CameraTruthState)
    speech: SpeechQueueController = field(default_factory=SpeechQueueController)

    def __post_init__(self) -> None:
        self.profile = _safe_token(self.profile, default="manual")
        self.language = _safe_token(self.language, default="unknown", limit=32)
        self.tts_runtime_label = _safe_text(self.tts_runtime_label, limit=96) or "unknown"
        self.camera.vision_enabled = bool(self.vision_enabled)
        self.camera.continuous_perception_enabled = bool(self.continuous_perception_enabled)

    def configure(
        self,
        *,
        profile: object | None = None,
        language: object | None = None,
        tts_runtime_label: object | None = None,
        vision_enabled: bool | None = None,
        continuous_perception_enabled: bool | None = None,
        protected_playback: bool | None = None,
    ) -> None:
        if profile is not None:
            self.profile = _safe_token(profile, default=self.profile)
        if language is not None:
            self.language = _safe_token(language, default=self.language, limit=32)
        if tts_runtime_label is not None:
            self.tts_runtime_label = _safe_text(tts_runtime_label, limit=96) or self.tts_runtime_label
        if vision_enabled is not None:
            self.vision_enabled = bool(vision_enabled)
            self.camera.vision_enabled = bool(vision_enabled)
        if continuous_perception_enabled is not None:
            self.continuous_perception_enabled = bool(continuous_perception_enabled)
            self.camera.continuous_perception_enabled = bool(continuous_perception_enabled)
        if protected_playback is not None:
            self.protected_playback = bool(protected_playback)

    def note_client_connected(self, *, session_id: object = None, client_id: object = None) -> None:
        self.session_id = _safe_token(session_id, default="", limit=96) or self.session_id
        self.client_id = _safe_token(client_id, default="", limit=96) or self.client_id
        self.camera.note_client_connected()

    def note_client_disconnected(self) -> None:
        self.session_id = None
        self.client_id = None
        self.camera.note_client_disconnected()
        self.transcript.stop_turn()
        self.speech.note_interruption()

    def note_client_media(self, media: Mapping[str, Any]) -> None:
        self.camera.update_client_media(media)

    def note_camera_frame(
        self,
        *,
        frame_seq: object = None,
        frame_age_ms: object = None,
        received_monotonic: float | None = None,
    ) -> None:
        self.camera.note_frame(
            frame_seq=frame_seq,
            frame_age_ms=frame_age_ms,
            received_monotonic=received_monotonic,
        )

    def note_vision_requested(self) -> None:
        self.camera.note_looking_requested()

    def note_vision_success(self, *, frame_seq: object = None, frame_age_ms: object = None) -> None:
        self.camera.note_vision_success(frame_seq=frame_seq, frame_age_ms=frame_age_ms)

    def note_vision_error(
        self,
        *,
        result_state: object = "error",
        frame_seq: object = None,
        frame_age_ms: object = None,
    ) -> None:
        self.camera.note_vision_error(
            result_state=result_state,
            frame_seq=frame_seq,
            frame_age_ms=frame_age_ms,
        )

    def as_dict(self, *, include_live_text: bool = False) -> dict[str, object]:
        return {
            "schema_version": 3,
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "vision_enabled": bool(self.vision_enabled),
            "continuous_perception_enabled": bool(self.continuous_perception_enabled),
            "protected_playback": bool(self.protected_playback),
            "session_id_hash": _stable_hash(self.session_id),
            "client_id_hash": _stable_hash(self.client_id),
            "stt_turn": self.transcript.as_dict(include_live_text=include_live_text),
            "camera_truth": self.camera.as_dict(),
            "speech_queue": self.speech.as_dict(),
            "reason_codes": ["browser_runtime_session:v3"],
        }

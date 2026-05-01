"""Deterministic ActorControlFrameV3 scheduler for browser actor events."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping

from blink.interaction.actor_events import ActorEventV2, find_actor_trace_safety_violations

ACTOR_CONTROL_FRAME_V3_SCHEMA_VERSION = 3
ACTOR_CONTROL_FRAME_V3_BOUNDARY_TYPES = (
    "vad_boundary",
    "stt_final_boundary",
    "speech_chunk_boundary",
    "tts_queue_boundary",
    "camera_frame_boundary",
    "tool_result_boundary",
    "interruption_boundary",
    "repair_boundary",
)
DEFAULT_SPEECH_CHUNK_LOOKAHEAD = 2
DEFAULT_SUBTITLE_LOOKAHEAD = 2
_SEMANTIC_INTENTS = {
    "question",
    "instruction",
    "correction",
    "object_showing",
    "project_planning",
    "small_talk",
    "unknown",
}
_SEMANTIC_CHIP_IDS = {
    "heard_summary",
    "constraint_detected",
    "question_detected",
    "showing_object",
    "camera_limited",
    "still_listening",
    "ready_to_answer",
}

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
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
    "record",
    "sdp",
    "secret",
    "source_ref",
    "text",
    "token",
    "transcript",
    "url",
}
_UNSAFE_KEY_EXACT = {
    "audio",
    "bytes",
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
    "transcript",
}
_SAFE_KEY_EXACT = {
    "auto_gain_control",
    "barge_in_state",
    "camera_state",
    "context_available",
    "degradation_state",
    "director_mode",
    "echo_risk_state",
    "floor_state",
    "frame_type",
    "freshness_state",
    "grounding_mode",
    "input_type",
    "last_text_kind",
    "media_mode",
    "microphone_state",
    "readiness_state",
    "state",
    "stale_generation_token",
    "text_kind",
    "track_state",
}
_SAFE_KEY_SUFFIXES = (
    "_available",
    "_chars",
    "_count",
    "_counts",
    "_depth",
    "_enabled",
    "_id",
    "_index",
    "_kind",
    "_ms",
    "_seq",
    "_sequence",
    "_state",
)
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


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    raw_value = getattr(value, "value", value)
    text = _TOKEN_RE.sub("_", str(raw_value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object) -> bool:
    return value is True


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(1.0, parsed))


def _safe_string(value: object, *, default: str = "", limit: int = 96) -> str:
    text = " ".join(str(value or "").split())[:limit]
    if not text:
        return default
    lowered = text.lower()
    if any(token in lowered for token in _UNSAFE_VALUE_TOKENS):
        return default
    return text


def _dedupe_reason_codes(values: object, *, limit: int = 32) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        code = _safe_token(raw_value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def _metadata_key_is_safe(key: object) -> bool:
    text = _safe_token(key, default="", limit=80).lower()
    if not text or text in _UNSAFE_KEY_EXACT:
        return False
    if text in _SAFE_KEY_EXACT:
        return True
    if text.endswith(_SAFE_KEY_SUFFIXES):
        return True
    return not any(fragment in text for fragment in _UNSAFE_KEY_FRAGMENTS)


def _safe_metadata_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = _safe_string(value, limit=120)
        return text or None
    if depth >= 3:
        return _safe_token(type(value).__name__, default="object", limit=64)
    if isinstance(value, (list, tuple, set)):
        items: list[object] = []
        for item in list(value)[:12]:
            sanitized = _safe_metadata_value(item, depth=depth + 1)
            if sanitized is not None:
                items.append(sanitized)
        return items
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, raw_item in list(value.items())[:24]:
            if not _metadata_key_is_safe(raw_key):
                continue
            key = _safe_token(raw_key, default="", limit=80)
            if not key:
                continue
            sanitized = _safe_metadata_value(raw_item, depth=depth + 1)
            if sanitized is not None:
                result[key] = sanitized
        return result
    return _safe_token(type(value).__name__, default="object", limit=64)


def _safe_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    sanitized = _safe_metadata_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _speech_capability_labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    labels: list[str] = []
    for key in (
        "supports_chunk_boundaries",
        "supports_interruption_flush",
        "supports_interruption_discard",
        "supports_pause_timing",
        "supports_speech_rate",
        "supports_prosody_emphasis",
        "supports_partial_stream_abort",
        "expression_controls_hardware",
    ):
        state = "supported" if value.get(key) is True else "noop"
        labels.append(f"speech_capability:{key}:{state}")
    backend_label = value.get("backend_label")
    if backend_label not in (None, ""):
        labels.append(f"speech_backend:{_safe_token(backend_label, default='unknown', limit=64)}")
    return tuple(_dedupe_reason_codes(labels, limit=12))


def _contains_unsafe_value(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in _UNSAFE_VALUE_TOKENS)
    if isinstance(value, Mapping):
        return any(_contains_unsafe_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_unsafe_value(item) for item in value)
    return False


def _event_payload(event: ActorEventV2 | Mapping[str, object]) -> dict[str, object]:
    payload = event.as_dict() if isinstance(event, ActorEventV2) else dict(event)
    metadata = _safe_metadata(payload.get("metadata"))
    return {
        "schema_version": _safe_int(payload.get("schema_version"), default=2),
        "event_id": max(1, _safe_int(payload.get("event_id"), default=1)),
        "event_type": _safe_token(payload.get("event_type"), default="waiting", limit=96),
        "mode": _safe_token(payload.get("mode"), default="waiting", limit=40),
        "timestamp": _safe_string(payload.get("timestamp"), limit=64),
        "profile": _safe_token(payload.get("profile"), default="manual", limit=96),
        "language": _safe_token(payload.get("language"), default="unknown", limit=32),
        "tts_backend": _safe_token(payload.get("tts_backend"), default="unknown", limit=80),
        "tts_label": _safe_string(payload.get("tts_label"), default="unknown", limit=80),
        "vision_backend": _safe_token(payload.get("vision_backend"), default="none", limit=80),
        "source": _safe_token(payload.get("source"), default="runtime", limit=64),
        "session_id": _safe_token(payload.get("session_id"), default="", limit=96) or None,
        "client_id": _safe_token(payload.get("client_id"), default="", limit=96) or None,
        "metadata": metadata,
        "reason_codes": _dedupe_reason_codes(payload.get("reason_codes"), limit=32),
    }


def _timestamp_ms(value: object, *, fallback_ms: int) -> int:
    text = str(value or "").strip()
    if not text:
        return fallback_ms
    try:
        normalized = text.replace("Z", "+00:00")
        return max(0, int(datetime.fromisoformat(normalized).timestamp() * 1000))
    except ValueError:
        return fallback_ms


def _digest_payload(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _event_has_reason(payload: Mapping[str, object], *needles: str) -> bool:
    reason_codes = payload.get("reason_codes")
    raw = reason_codes if isinstance(reason_codes, list) else []
    return any(str(code) in needles for code in raw)


def _event_has_reason_prefix(payload: Mapping[str, object], *prefixes: str) -> bool:
    reason_codes = payload.get("reason_codes")
    raw = reason_codes if isinstance(reason_codes, list) else []
    return any(str(code).startswith(prefixes) for code in raw)


def find_actor_control_safety_violations(payload: object) -> list[str]:
    """Return public safety violation labels for actor control inputs."""
    if not isinstance(payload, Mapping):
        return ["actor_control:malformed_payload"]
    violations = list(find_actor_trace_safety_violations(payload))
    for raw_key, raw_value in payload.items():
        key = _safe_token(raw_key, default="", limit=80).lower()
        if key in _UNSAFE_KEY_EXACT or any(fragment in key for fragment in _UNSAFE_KEY_FRAGMENTS):
            violations.append("actor_control:unsafe_key_present")
        if _contains_unsafe_value(raw_value):
            violations.append("actor_control:unsafe_value_present")
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for raw_key, raw_value in metadata.items():
            if not _metadata_key_is_safe(raw_key):
                violations.append("actor_control:unsafe_metadata_key_present")
            if _contains_unsafe_value(raw_value):
                violations.append("actor_control:unsafe_metadata_value_present")
    return _dedupe_reason_codes(violations, limit=32)


def _boundary_for_event(payload: Mapping[str, object]) -> str | None:
    event_type = str(payload.get("event_type") or "waiting")
    mode = str(payload.get("mode") or "waiting")
    source = str(payload.get("source") or "runtime")
    if event_type in {
        "speech_started",
        "listening",
        "partial_heard",
        "listening_started",
        "partial_understanding_updated",
    }:
        return "vad_boundary"
    if event_type in {"final_heard", "final_understanding_ready"}:
        return "stt_final_boundary"
    if event_type == "speaking" and (
        _event_has_reason_prefix(payload, "speech:")
        or _event_has_reason_prefix(payload, "tts:")
        or _safe_metadata(payload.get("metadata")).get("chunk_index") is not None
    ):
        if _event_has_reason(payload, "tts:stopped"):
            return "tts_queue_boundary"
        return "speech_chunk_boundary"
    if event_type == "waiting" and _event_has_reason(payload, "tts:stopped"):
        return "tts_queue_boundary"
    if event_type == "looking" or source in {"camera", "vision", "client-media", "webrtc"} and (
        _event_has_reason_prefix(payload, "camera:")
        or _event_has_reason_prefix(payload, "vision:")
    ):
        return "camera_frame_boundary"
    if event_type in {"memory_used", "persona_plan_compiled"}:
        return "tool_result_boundary"
    if event_type in {"interruption_candidate", "interruption_accepted", "interruption_rejected"}:
        return "interruption_boundary"
    if event_type == "interrupted":
        return "interruption_boundary"
    if event_type in {"output_flushed", "interruption_recovered", "recovered"}:
        return "repair_boundary"
    if event_type == "floor_transition":
        metadata = _safe_metadata(payload.get("metadata"))
        floor_state = _safe_token(metadata.get("floor_state") or metadata.get("state"))
        floor_sub_state = _safe_token(metadata.get("floor_sub_state"))
        if (
            floor_state == "repair"
            or floor_sub_state in {"accepted_interrupt", "repair_requested"}
            or _event_has_reason_prefix(
                payload,
                "floor:interruption_accepted",
                "floor:correction",
                "floor:explicit_interruption",
            )
        ):
            return "repair_boundary"
        if (
            floor_state == "overlap"
            or floor_sub_state == "overlap_candidate"
            or _event_has_reason_prefix(payload, "floor:overlap", "floor:interruption")
        ):
            return "interruption_boundary"
    if event_type in {"error", "degraded"} and (
        source in {"camera", "vision", "client-media"}
        or _event_has_reason_prefix(payload, "camera:", "vision:")
    ):
        return "camera_frame_boundary"
    return None


@dataclass
class ActorControlPersistentStateV3:
    """Persistent control state carried across actor-event boundaries."""

    mode: str = "waiting"
    floor_state: str = "unknown"
    floor_sub_state: str = "handoff_complete"
    floor_yield_decision: str = "wait_for_input"
    active_listener_state: str = "idle"
    active_listener_intent: str = "unknown"
    active_listener_chip_ids: tuple[str, ...] = ()
    speech_generation_id: str = "speech-unavailable"
    speech_turn_id: str = "turn-unavailable"
    speech_generation_stale: bool = False
    outstanding_speech_chunks: int = 0
    outstanding_subtitles: int = 0
    held_speech_chunks: int = 0
    stale_output_dropped_count: int = 0
    camera_state: str = "unknown"
    camera_freshness_state: str = "unknown"
    camera_current_answer_used_vision: bool = False
    camera_scene_transition: str = "none"
    camera_honesty_state: str = "unavailable"
    camera_user_presence_hint: str = "unknown"
    camera_last_moondream_result_state: str = "none"
    camera_object_showing_likelihood: float = 0.0
    interruption_outcome: str = "none"
    memory_effect_count: int = 0
    persona_effect_count: int = 0

    def as_dict(self) -> dict[str, object]:
        """Return public-safe persistent state."""
        return {
            "mode": self.mode,
            "floor_state": self.floor_state,
            "floor_sub_state": self.floor_sub_state,
            "floor_yield_decision": self.floor_yield_decision,
            "active_listener_state": self.active_listener_state,
            "active_listener_intent": self.active_listener_intent,
            "active_listener_chip_ids": list(self.active_listener_chip_ids),
            "speech_generation_id": self.speech_generation_id,
            "speech_turn_id": self.speech_turn_id,
            "speech_generation_stale": self.speech_generation_stale,
            "outstanding_speech_chunks": self.outstanding_speech_chunks,
            "outstanding_subtitles": self.outstanding_subtitles,
            "held_speech_chunks": self.held_speech_chunks,
            "stale_output_dropped_count": self.stale_output_dropped_count,
            "camera_state": self.camera_state,
            "camera_freshness_state": self.camera_freshness_state,
            "camera_current_answer_used_vision": self.camera_current_answer_used_vision,
            "camera_scene_transition": self.camera_scene_transition,
            "camera_honesty_state": self.camera_honesty_state,
            "camera_user_presence_hint": self.camera_user_presence_hint,
            "camera_last_moondream_result_state": self.camera_last_moondream_result_state,
            "camera_object_showing_likelihood": self.camera_object_showing_likelihood,
            "interruption_outcome": self.interruption_outcome,
            "memory_effect_count": self.memory_effect_count,
            "persona_effect_count": self.persona_effect_count,
        }


@dataclass
class ActorControlConditionCacheV3:
    """Refreshable condition cache updated from every actor event."""

    last_event_id: int = 0
    last_event_type: str = "waiting"
    last_mode: str = "waiting"
    active_listener_ready: bool = False
    speech_queue_depth: int = 0
    latest_chunk_index: int = 0
    speech_estimated_duration_ms: int = 0
    speech_subtitle_timing_policy: str = "unknown"
    speech_stale_generation_token_seen: bool = False
    speech_backend_capability_labels: tuple[str, ...] = ()
    latest_frame_sequence: int = 0
    latest_frame_age_ms: int | None = None
    degradation_state: str = "ok"
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return public-safe refreshable cache state."""
        return {
            "last_event_id": self.last_event_id,
            "last_event_type": self.last_event_type,
            "last_mode": self.last_mode,
            "active_listener_ready": self.active_listener_ready,
            "speech_queue_depth": self.speech_queue_depth,
            "latest_chunk_index": self.latest_chunk_index,
            "speech_estimated_duration_ms": self.speech_estimated_duration_ms,
            "speech_subtitle_timing_policy": self.speech_subtitle_timing_policy,
            "speech_stale_generation_token_seen": self.speech_stale_generation_token_seen,
            "speech_backend_capability_labels": list(self.speech_backend_capability_labels),
            "latest_frame_sequence": self.latest_frame_sequence,
            "latest_frame_age_ms": self.latest_frame_age_ms,
            "degradation_state": self.degradation_state,
            "reason_codes": _dedupe_reason_codes(self.reason_codes, limit=16),
        }


@dataclass(frozen=True)
class ActorControlFrameV3:
    """One boundary-aligned public-safe actor control frame."""

    frame_id: str
    sequence: int
    profile: str
    language: str
    tts_backend: str
    tts_runtime_label: str
    boundary: str
    created_at_ms: int
    source_event_ids: tuple[int, ...]
    persistent_state: dict[str, object]
    condition_cache_digest: str
    browser_ui_policy: dict[str, object]
    speech_policy: dict[str, object]
    active_listener_policy: dict[str, object]
    floor_policy: dict[str, object]
    camera_policy: dict[str, object]
    memory_policy: dict[str, object]
    persona_policy: dict[str, object]
    repair_policy: dict[str, object]
    lookahead_counters: dict[str, object]
    reason_trace: tuple[str, ...]
    schema_version: int = ACTOR_CONTROL_FRAME_V3_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe control frame payload."""
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "sequence": self.sequence,
            "profile": self.profile,
            "language": self.language,
            "tts_backend": self.tts_backend,
            "tts_runtime_label": self.tts_runtime_label,
            "boundary": self.boundary,
            "created_at_ms": self.created_at_ms,
            "source_event_ids": list(self.source_event_ids),
            "persistent_state": dict(self.persistent_state),
            "condition_cache_digest": self.condition_cache_digest,
            "browser_ui_policy": dict(self.browser_ui_policy),
            "speech_policy": dict(self.speech_policy),
            "active_listener_policy": dict(self.active_listener_policy),
            "floor_policy": dict(self.floor_policy),
            "camera_policy": dict(self.camera_policy),
            "memory_policy": dict(self.memory_policy),
            "persona_policy": dict(self.persona_policy),
            "repair_policy": dict(self.repair_policy),
            "lookahead_counters": dict(self.lookahead_counters),
            "reason_trace": list(self.reason_trace),
        }


@dataclass(frozen=True)
class ActorControlReplaySummary:
    """Compact deterministic replay summary for control frames."""

    frame_count: int
    profiles: tuple[str, ...]
    languages: tuple[str, ...]
    tts_labels: tuple[str, ...]
    boundary_counts: dict[str, int]
    boundary_timeline: tuple[dict[str, object], ...]
    speech_state: dict[str, object]
    floor_states: tuple[str, ...]
    floor_sub_states: tuple[str, ...]
    camera_state: dict[str, object]
    interruption_outcomes: tuple[str, ...]
    memory_persona_effects: dict[str, object]
    failure_labels: tuple[str, ...]
    performance_plan_summaries: tuple[dict[str, object], ...] = ()
    safety_violations: tuple[dict[str, object], ...] = ()
    schema_version: int = ACTOR_CONTROL_FRAME_V3_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe replay summary."""
        return {
            "schema_version": self.schema_version,
            "frame_count": self.frame_count,
            "profiles": list(self.profiles),
            "languages": list(self.languages),
            "tts_labels": list(self.tts_labels),
            "boundary_counts": dict(self.boundary_counts),
            "boundary_timeline": [dict(item) for item in self.boundary_timeline],
            "speech_state": dict(self.speech_state),
            "floor_states": list(self.floor_states),
            "floor_sub_states": list(self.floor_sub_states),
            "camera_state": dict(self.camera_state),
            "interruption_outcomes": list(self.interruption_outcomes),
            "memory_persona_effects": dict(self.memory_persona_effects),
            "performance_plan_summaries": [
                dict(item) for item in self.performance_plan_summaries
            ],
            "failure_labels": list(self.failure_labels),
            "safety_violations": [dict(item) for item in self.safety_violations],
        }


class ActorControlScheduler:
    """Deterministic boundary scheduler for public-safe actor events."""

    def __init__(
        self,
        *,
        profile: str = "manual",
        language: str = "unknown",
        tts_runtime_label: str = "unknown",
        tts_backend: str = "unknown",
        max_speech_chunk_lookahead: int = DEFAULT_SPEECH_CHUNK_LOOKAHEAD,
        max_subtitle_lookahead: int = DEFAULT_SUBTITLE_LOOKAHEAD,
    ):
        """Initialize the scheduler with fixed lookahead policy limits."""
        self._profile = _safe_token(profile, default="manual", limit=96)
        self._language = _safe_token(language, default="unknown", limit=32)
        self._tts_runtime_label = _safe_string(
            tts_runtime_label,
            default="unknown",
            limit=80,
        )
        self._tts_backend = _safe_token(tts_backend, default="unknown", limit=80)
        self._max_speech_chunk_lookahead = max(1, int(max_speech_chunk_lookahead))
        self._max_subtitle_lookahead = max(1, int(max_subtitle_lookahead))
        self._state = ActorControlPersistentStateV3()
        self._cache = ActorControlConditionCacheV3()
        self._pending_event_ids: list[int] = []
        self._sequence = 0
        self._lock = RLock()

    def configure(
        self,
        *,
        profile: str | None = None,
        language: str | None = None,
        tts_runtime_label: str | None = None,
        tts_backend: str | None = None,
    ) -> None:
        """Update default labels without changing control state."""
        with self._lock:
            if profile not in (None, ""):
                self._profile = _safe_token(profile, default="manual", limit=96)
            if language not in (None, ""):
                self._language = _safe_token(language, default="unknown", limit=32)
            if tts_runtime_label not in (None, ""):
                self._tts_runtime_label = _safe_string(
                    tts_runtime_label,
                    default="unknown",
                    limit=80,
                )
            if tts_backend not in (None, ""):
                self._tts_backend = _safe_token(tts_backend, default="unknown", limit=80)

    def speech_lookahead_can_emit(self) -> bool:
        """Return whether another subtitle/TTS chunk may be queued now."""
        with self._lock:
            return (
                self._state.outstanding_speech_chunks < self._max_speech_chunk_lookahead
                and self._state.outstanding_subtitles < self._max_subtitle_lookahead
                and not self._state.speech_generation_stale
            )

    def observe_actor_event(
        self,
        event: ActorEventV2 | Mapping[str, object],
    ) -> ActorControlFrameV3 | None:
        """Update caches and emit a control frame only at safe boundaries."""
        payload = _event_payload(event)
        with self._lock:
            self._update_from_event(payload)
            boundary = _boundary_for_event(payload)
            if boundary is None:
                return None
            return self._emit_frame_unlocked(payload, boundary)

    def _update_from_event(self, payload: Mapping[str, object]) -> None:
        metadata = _safe_metadata(payload.get("metadata"))
        event_id = _safe_int(payload.get("event_id"), default=self._cache.last_event_id + 1)
        event_type = _safe_token(payload.get("event_type"), default="waiting", limit=96)
        mode = _safe_token(payload.get("mode"), default="waiting", limit=40)
        self._profile = _safe_token(payload.get("profile"), default=self._profile, limit=96)
        self._language = _safe_token(payload.get("language"), default=self._language, limit=32)
        self._tts_backend = _safe_token(
            payload.get("tts_backend"),
            default=self._tts_backend,
            limit=80,
        )
        self._tts_runtime_label = _safe_string(
            payload.get("tts_label"),
            default=self._tts_runtime_label,
            limit=80,
        )
        self._pending_event_ids.append(event_id)
        self._pending_event_ids = self._pending_event_ids[-16:]
        self._cache.last_event_id = event_id
        self._cache.last_event_type = event_type
        self._cache.last_mode = mode
        self._cache.reason_codes = _dedupe_reason_codes(payload.get("reason_codes"), limit=16)
        self._state.mode = mode

        if event_type in {"listening_started", "partial_understanding_updated", "partial_heard"}:
            self._state.active_listener_state = "listening"
            self._cache.active_listener_ready = False
        elif event_type in {"final_understanding_ready", "final_heard"}:
            self._state.active_listener_state = "ready"
            self._cache.active_listener_ready = True
        self._update_active_listener_semantics(metadata)

        if metadata.get("floor_state") is not None:
            self._state.floor_state = _safe_token(metadata.get("floor_state"), limit=64)
        elif event_type == "floor_transition":
            self._state.floor_state = _safe_token(metadata.get("state"), default="unknown")
        if metadata.get("floor_sub_state") is not None:
            self._state.floor_sub_state = _safe_token(
                metadata.get("floor_sub_state"),
                default="handoff_complete",
                limit=64,
            )
        if metadata.get("yield_decision") is not None:
            self._state.floor_yield_decision = _safe_token(
                metadata.get("yield_decision"),
                default="wait_for_input",
                limit=64,
            )

        if event_type == "thinking" and _event_has_reason(payload, "speech:generation_start"):
            self._state.speech_generation_id = _safe_token(
                metadata.get("generation_id"),
                default="speech-unavailable",
                limit=96,
            )
            self._state.speech_turn_id = _safe_token(
                metadata.get("turn_id"),
                default="turn-unavailable",
                limit=96,
            )
            self._state.speech_generation_stale = False
            self._state.held_speech_chunks = 0
            self._state.outstanding_speech_chunks = 0
            self._state.outstanding_subtitles = 0
        if event_type == "speaking" and _event_has_reason(payload, "speech:subtitle_ready"):
            self._state.speech_generation_id = _safe_token(
                metadata.get("generation_id"),
                default=self._state.speech_generation_id,
                limit=96,
            )
            self._state.speech_turn_id = _safe_token(
                metadata.get("turn_id"),
                default=self._state.speech_turn_id,
                limit=96,
            )
            self._state.outstanding_speech_chunks += 1
            self._state.outstanding_subtitles += 1
            self._state.held_speech_chunks = max(0, self._state.held_speech_chunks - 1)
            self._cache.speech_queue_depth = _safe_int(metadata.get("queue_depth"))
            self._cache.latest_chunk_index = _safe_int(metadata.get("chunk_index"))
            self._cache.speech_estimated_duration_ms = _safe_int(
                metadata.get("estimated_duration_ms")
            )
            subtitle_timing = metadata.get("subtitle_timing")
            if isinstance(subtitle_timing, Mapping):
                self._cache.speech_subtitle_timing_policy = _safe_token(
                    subtitle_timing.get("emit_policy"),
                    default="unknown",
                    limit=80,
                )
            self._cache.speech_stale_generation_token_seen = (
                metadata.get("stale_generation_token") not in (None, "")
            )
            self._cache.speech_backend_capability_labels = _speech_capability_labels(
                metadata.get("backend_capabilities")
            )
        if event_type == "speaking" and _event_has_reason(payload, "speech:lookahead_held"):
            self._state.held_speech_chunks = max(
                self._state.held_speech_chunks,
                _safe_int(metadata.get("held_chunk_count"), default=1),
            )
            self._cache.speech_queue_depth = max(
                self._cache.speech_queue_depth,
                _safe_int(metadata.get("speech_queue_depth"), default=0),
            )
        if _event_has_reason(payload, "tts:stopped"):
            self._state.outstanding_speech_chunks = max(
                0,
                self._state.outstanding_speech_chunks - 1,
            )
            self._state.outstanding_subtitles = max(0, self._state.outstanding_subtitles - 1)
        if event_type in {"interrupted", "interruption_accepted"}:
            self._state.interruption_outcome = (
                "accepted" if event_type == "interruption_accepted" else "interrupted"
            )
            self._state.speech_generation_stale = True
            self._state.held_speech_chunks = 0
            self._state.outstanding_speech_chunks = 0
            self._state.outstanding_subtitles = 0
        elif event_type == "interruption_rejected":
            self._state.interruption_outcome = "rejected_protected"
        elif event_type == "interruption_candidate":
            self._state.interruption_outcome = "candidate"
        elif event_type == "output_flushed":
            self._state.interruption_outcome = "output_flushed"
            self._state.stale_output_dropped_count += max(
                1,
                _safe_int(metadata.get("output_flushed_count"), default=1),
            )
        elif event_type == "interruption_recovered":
            self._state.interruption_outcome = "recovered"
            self._state.speech_generation_stale = False

        if event_type in {"looking", "degraded", "error", "recovered"}:
            self._update_camera_state(payload, metadata)
        if event_type == "memory_used":
            self._state.memory_effect_count += max(
                1,
                _safe_int(metadata.get("used_memory_count"), default=1),
            )
        if event_type == "persona_plan_compiled":
            self._state.persona_effect_count += max(
                1,
                _safe_int(metadata.get("persona_reference_count"), default=1),
            )
        if event_type == "degraded":
            self._cache.degradation_state = "degraded"
        elif event_type == "error":
            self._cache.degradation_state = "error"
        elif event_type == "recovered":
            self._cache.degradation_state = "ok"

    def _update_camera_state(
        self,
        payload: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> None:
        if metadata.get("scene_transition") is not None:
            self._state.camera_scene_transition = _safe_token(
                metadata.get("scene_transition"),
                default="none",
                limit=64,
            )
        if metadata.get("camera_honesty_state") is not None:
            self._state.camera_honesty_state = _safe_token(
                metadata.get("camera_honesty_state"),
                default="unavailable",
                limit=64,
            )
        if metadata.get("user_presence_hint") is not None:
            self._state.camera_user_presence_hint = _safe_token(
                metadata.get("user_presence_hint"),
                default="unknown",
                limit=64,
            )
        if metadata.get("last_moondream_result_state") is not None:
            self._state.camera_last_moondream_result_state = _safe_token(
                metadata.get("last_moondream_result_state"),
                default="none",
                limit=64,
            )
        if metadata.get("object_showing_likelihood") is not None:
            self._state.camera_object_showing_likelihood = _safe_float(
                metadata.get("object_showing_likelihood")
            )
        if metadata.get("camera_state") is not None:
            self._state.camera_state = _safe_token(metadata.get("camera_state"), limit=40)
        if metadata.get("frame_seq") is not None:
            self._cache.latest_frame_sequence = _safe_int(metadata.get("frame_seq"))
        if metadata.get("frame_age_ms") is not None:
            self._cache.latest_frame_age_ms = _safe_int(metadata.get("frame_age_ms"))
        if (
            _event_has_reason(
                payload,
                "vision:fetch_user_image_success",
                "vision.fetch_user_image_success",
            )
            or metadata.get("scene_transition") == "vision_answered"
            or metadata.get("on_demand_vision_state") == "success"
        ):
            self._state.camera_state = "fresh_used"
            self._state.camera_freshness_state = "fresh"
            self._state.camera_current_answer_used_vision = True
            self._state.camera_honesty_state = "can_see_now"
            self._state.camera_last_moondream_result_state = "answered"
        elif metadata.get("scene_transition") == "frame_captured":
            if self._state.camera_state not in {"fresh_used", "error", "stale_or_limited"}:
                self._state.camera_state = "available"
            self._state.camera_freshness_state = "fresh"
            if self._state.camera_honesty_state != "can_see_now":
                self._state.camera_honesty_state = "recent_frame_available"
        elif metadata.get("scene_transition") == "camera_ready":
            if self._state.camera_state != "fresh_used":
                self._state.camera_state = "available"
            if self._state.camera_honesty_state != "can_see_now":
                self._state.camera_honesty_state = "available_not_used"
        elif metadata.get("scene_transition") == "looking_requested":
            if self._state.camera_state == "unknown":
                self._state.camera_state = "requested"
            self._state.camera_last_moondream_result_state = "looking"
        elif metadata.get("scene_transition") == "vision_stale":
            if self._state.camera_state != "fresh_used":
                self._state.camera_state = "stale_or_limited"
            self._state.camera_freshness_state = "stale_or_limited"
            self._state.camera_honesty_state = "unavailable"
            self._state.camera_last_moondream_result_state = "stale"
        elif metadata.get("scene_transition") == "vision_unavailable":
            if self._state.camera_state != "fresh_used":
                self._state.camera_state = "error"
            self._state.camera_freshness_state = "error"
            self._state.camera_honesty_state = "unavailable"
            self._state.camera_last_moondream_result_state = "unavailable"
        elif str(payload.get("event_type")) in {"error", "degraded"}:
            self._state.camera_state = "error" if str(payload.get("event_type")) == "error" else "stale_or_limited"
            self._state.camera_freshness_state = (
                "error" if str(payload.get("event_type")) == "error" else "stale_or_limited"
            )
            self._state.camera_honesty_state = "unavailable"

    def _update_active_listener_semantics(self, metadata: Mapping[str, object]) -> None:
        semantic = metadata.get("semantic_listener")
        if not isinstance(semantic, Mapping):
            semantic = metadata.get("semantic_state_v3")
        if not isinstance(semantic, Mapping):
            return
        intent = _safe_token(semantic.get("detected_intent"), default="unknown", limit=64)
        if intent not in _SEMANTIC_INTENTS:
            intent = "unknown"
        if intent:
            self._state.active_listener_intent = intent
        chip_ids: list[str] = []
        raw_chip_ids = semantic.get("listener_chip_ids")
        if isinstance(raw_chip_ids, list):
            chip_ids.extend(
                chip
                for chip in (_safe_token(chip, default="", limit=64) for chip in raw_chip_ids[:8])
                if chip in _SEMANTIC_CHIP_IDS
            )
        raw_chips = semantic.get("listener_chips")
        if isinstance(raw_chips, list):
            for chip in raw_chips[:8]:
                if isinstance(chip, Mapping):
                    chip_id = _safe_token(chip.get("chip_id"), default="", limit=64)
                    if chip_id in _SEMANTIC_CHIP_IDS:
                        chip_ids.append(chip_id)
        if chip_ids:
            self._state.active_listener_chip_ids = tuple(
                _dedupe_reason_codes(chip_ids, limit=8)
            )
        if semantic.get("enough_information_to_answer") is True:
            self._cache.active_listener_ready = True

    def _emit_frame_unlocked(
        self,
        payload: Mapping[str, object],
        boundary: str,
    ) -> ActorControlFrameV3:
        self._sequence += 1
        created_at_ms = _timestamp_ms(
            payload.get("timestamp"),
            fallback_ms=_safe_int(payload.get("event_id")) * 1000,
        )
        source_ids = tuple(dict.fromkeys(self._pending_event_ids))
        self._pending_event_ids.clear()
        state_payload = self._state.as_dict()
        cache_payload = self._cache.as_dict()
        reason_trace = _dedupe_reason_codes(
            [
                "actor_control_frame:v3",
                f"actor_control_boundary:{boundary}",
                f"actor_control_mode:{self._state.mode}",
                *cache_payload.get("reason_codes", []),
            ],
            limit=32,
        )
        frame_id = (
            f"acfv3-{_safe_token(self._profile, default='profile', limit=40)}-"
            f"{self._sequence:06d}-{boundary}"
        )
        return ActorControlFrameV3(
            frame_id=frame_id,
            sequence=self._sequence,
            profile=self._profile,
            language=self._language,
            tts_backend=self._tts_backend,
            tts_runtime_label=self._tts_runtime_label,
            boundary=boundary,
            created_at_ms=created_at_ms,
            source_event_ids=source_ids,
            persistent_state=state_payload,
            condition_cache_digest=_digest_payload(cache_payload),
            browser_ui_policy=self._browser_ui_policy(boundary),
            speech_policy=self._speech_policy(boundary),
            active_listener_policy=self._active_listener_policy(boundary),
            floor_policy=self._floor_policy(boundary),
            camera_policy=self._camera_policy(boundary),
            memory_policy=self._memory_policy(boundary),
            persona_policy=self._persona_policy(boundary),
            repair_policy=self._repair_policy(boundary),
            lookahead_counters=self._lookahead_counters(),
            reason_trace=tuple(reason_trace),
        )

    def _browser_ui_policy(self, boundary: str) -> dict[str, object]:
        return {
            "mode": self._state.mode,
            "boundary": boundary,
            "subtitle_state": "held"
            if self._state.held_speech_chunks
            else "available"
            if self._state.outstanding_subtitles
            else "idle",
            "active_listener_state": self._state.active_listener_state,
            "reason_codes": [f"browser_ui:{self._state.mode}"],
        }

    def _speech_policy(self, boundary: str) -> dict[str, object]:
        if self._state.speech_generation_stale:
            action = "suppress_stale_generation"
        elif self._state.held_speech_chunks:
            action = "hold_lookahead"
        elif self.speech_lookahead_can_emit():
            action = "allow_next_chunk"
        else:
            action = "wait_for_tts_queue_boundary"
        return {
            "action": action,
            "generation_id": self._state.speech_generation_id,
            "turn_id": self._state.speech_turn_id,
            "generation_stale": self._state.speech_generation_stale,
            "estimated_duration_ms": self._cache.speech_estimated_duration_ms,
            "subtitle_timing_policy": self._cache.speech_subtitle_timing_policy,
            "stale_generation_state": "token_seen"
            if self._cache.speech_stale_generation_token_seen
            else "unavailable",
            "backend_capability_labels": list(self._cache.speech_backend_capability_labels),
            "chunk_lookahead_limit": self._max_speech_chunk_lookahead,
            "subtitle_lookahead_limit": self._max_subtitle_lookahead,
            "boundary": boundary,
            "reason_codes": [f"speech_policy:{action}"],
        }

    def _active_listener_policy(self, boundary: str) -> dict[str, object]:
        return {
            "state": self._state.active_listener_state,
            "ready_to_answer": self._cache.active_listener_ready,
            "detected_intent": self._state.active_listener_intent,
            "listener_chip_ids": list(self._state.active_listener_chip_ids),
            "boundary": boundary,
            "reason_codes": [
                f"active_listener:{self._state.active_listener_state}",
                f"active_listener_intent:{self._state.active_listener_intent}",
            ],
        }

    def _floor_policy(self, boundary: str) -> dict[str, object]:
        return {
            "state": self._state.floor_state,
            "sub_state": self._state.floor_sub_state,
            "yield_decision": self._state.floor_yield_decision,
            "protected_playback_default": True,
            "boundary": boundary,
            "reason_codes": [
                f"floor_policy:{self._state.floor_state}",
                f"floor_sub_state:{self._state.floor_sub_state}",
            ],
        }

    def _camera_policy(self, boundary: str) -> dict[str, object]:
        return {
            "state": self._state.camera_state,
            "freshness_state": self._state.camera_freshness_state,
            "fresh_frame_used": self._state.camera_current_answer_used_vision,
            "scene_transition": self._state.camera_scene_transition,
            "camera_honesty_state": self._state.camera_honesty_state,
            "user_presence_hint": self._state.camera_user_presence_hint,
            "last_moondream_result_state": self._state.camera_last_moondream_result_state,
            "object_showing_likelihood": self._state.camera_object_showing_likelihood,
            "latest_frame_sequence": self._cache.latest_frame_sequence,
            "latest_frame_age_ms": self._cache.latest_frame_age_ms,
            "boundary": boundary,
            "reason_codes": [
                f"camera_policy:{self._state.camera_state}",
                f"camera_honesty:{self._state.camera_honesty_state}",
                f"scene_social_transition:{self._state.camera_scene_transition}",
            ],
        }

    def _memory_policy(self, boundary: str) -> dict[str, object]:
        return {
            "used": self._state.memory_effect_count > 0,
            "effect_count": self._state.memory_effect_count,
            "boundary": boundary,
            "reason_codes": [
                "memory_policy:used"
                if self._state.memory_effect_count
                else "memory_policy:unused"
            ],
        }

    def _persona_policy(self, boundary: str) -> dict[str, object]:
        return {
            "plan_available": self._state.persona_effect_count > 0,
            "effect_count": self._state.persona_effect_count,
            "boundary": boundary,
            "reason_codes": [
                "persona_policy:available"
                if self._state.persona_effect_count
                else "persona_policy:unavailable"
            ],
        }

    def _repair_policy(self, boundary: str) -> dict[str, object]:
        return {
            "interruption_outcome": self._state.interruption_outcome,
            "stale_output_action": "dropped_or_suppressed"
            if self._state.stale_output_dropped_count or self._state.speech_generation_stale
            else "none",
            "stale_output_dropped_count": self._state.stale_output_dropped_count,
            "boundary": boundary,
            "reason_codes": [f"repair_policy:{self._state.interruption_outcome}"],
        }

    def _lookahead_counters(self) -> dict[str, object]:
        return {
            "speech_chunks_outstanding": self._state.outstanding_speech_chunks,
            "speech_chunks_limit": self._max_speech_chunk_lookahead,
            "subtitles_outstanding": self._state.outstanding_subtitles,
            "subtitles_limit": self._max_subtitle_lookahead,
            "held_speech_chunks": self._state.held_speech_chunks,
        }


def compile_actor_control_frames_v3(
    events: Iterable[ActorEventV2 | Mapping[str, object]],
    *,
    scheduler: ActorControlScheduler | None = None,
) -> tuple[ActorControlFrameV3, ...]:
    """Compile actor events into deterministic boundary control frames."""
    control_scheduler = scheduler or ActorControlScheduler()
    frames: list[ActorControlFrameV3] = []
    for event in events:
        frame = control_scheduler.observe_actor_event(event)
        if frame is not None:
            frames.append(frame)
    return tuple(frames)


def render_actor_control_frames_v3_jsonl(frames: Iterable[ActorControlFrameV3]) -> str:
    """Render control frames as deterministic JSONL."""
    lines = [
        json.dumps(frame.as_dict(), ensure_ascii=False, sort_keys=True)
        for frame in frames
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def load_actor_events_for_control_v3(path: Path | str) -> tuple[tuple[dict[str, object], ...], list[dict[str, object]]]:
    """Load actor events and report safety violations without echoing values."""
    events: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                violations.append(
                    {"line": line_number, "reason_codes": ["actor_control:malformed_json"]}
                )
                continue
            reason_codes = find_actor_control_safety_violations(payload)
            if reason_codes:
                violations.append({"line": line_number, "reason_codes": reason_codes})
            if isinstance(payload, Mapping):
                events.append(_event_payload(payload))
    return tuple(events), violations


def load_actor_control_frames_v3_jsonl(path: Path | str) -> tuple[tuple[dict[str, object], ...], list[dict[str, object]]]:
    """Load control frame JSONL and report safety violations."""
    frames: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                violations.append(
                    {"line": line_number, "reason_codes": ["actor_control:malformed_json"]}
                )
                continue
            if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
                violations.append(
                    {"line": line_number, "reason_codes": ["actor_control:unsupported_schema"]}
                )
                continue
            reason_codes = find_actor_control_safety_violations(payload)
            if reason_codes:
                violations.append({"line": line_number, "reason_codes": reason_codes})
            frames.append(dict(payload))
    return tuple(frames), violations


def compile_actor_control_frames_v3_from_actor_trace(path: Path | str) -> tuple[ActorControlFrameV3, ...]:
    """Compile an actor trace JSONL file into control frames."""
    events, violations = load_actor_events_for_control_v3(path)
    if violations:
        raise ValueError("actor control input is not public-safe")
    return compile_actor_control_frames_v3(events)


def replay_actor_control_frames_v3_jsonl(path: Path | str) -> ActorControlReplaySummary:
    """Replay control frame JSONL into a compact deterministic summary."""
    frames, violations = load_actor_control_frames_v3_jsonl(path)
    return summarize_actor_control_frames_v3(frames, safety_violations=violations)


def summarize_actor_control_frames_v3(
    frames: Iterable[Mapping[str, object]],
    *,
    safety_violations: Iterable[Mapping[str, object]] = (),
) -> ActorControlReplaySummary:
    """Summarize control frames for offline review."""
    frame_rows = [dict(frame) for frame in frames]
    boundary_counts: Counter[str] = Counter()
    profiles: set[str] = set()
    languages: set[str] = set()
    tts_labels: set[str] = set()
    floor_states: list[str] = []
    floor_sub_states: list[str] = []
    interruption_outcomes: list[str] = []
    failure_labels: set[str] = set()
    timeline: list[dict[str, object]] = []
    speech_state: dict[str, object] = {}
    camera_state: dict[str, object] = {}
    memory_effect_count = 0
    persona_effect_count = 0
    performance_plan_summaries: list[dict[str, object]] = []

    for frame in frame_rows:
        boundary = _safe_token(frame.get("boundary"), default="unknown", limit=64)
        boundary_counts[boundary] += 1
        profiles.add(_safe_token(frame.get("profile"), default="manual", limit=96))
        languages.add(_safe_token(frame.get("language"), default="unknown", limit=32))
        tts_labels.add(_safe_string(frame.get("tts_runtime_label"), default="unknown", limit=80))
        persistent = frame.get("persistent_state") if isinstance(frame.get("persistent_state"), dict) else {}
        speech_policy = frame.get("speech_policy") if isinstance(frame.get("speech_policy"), dict) else {}
        floor_policy = frame.get("floor_policy") if isinstance(frame.get("floor_policy"), dict) else {}
        camera_policy = frame.get("camera_policy") if isinstance(frame.get("camera_policy"), dict) else {}
        memory_policy = frame.get("memory_policy") if isinstance(frame.get("memory_policy"), dict) else {}
        persona_policy = frame.get("persona_policy") if isinstance(frame.get("persona_policy"), dict) else {}
        repair_policy = frame.get("repair_policy") if isinstance(frame.get("repair_policy"), dict) else {}
        floor_state = _safe_token(floor_policy.get("state"), default="unknown", limit=64)
        floor_sub_state = _safe_token(
            floor_policy.get("sub_state") or persistent.get("floor_sub_state"),
            default="handoff_complete",
            limit=64,
        )
        if floor_state not in floor_states:
            floor_states.append(floor_state)
        if floor_sub_state not in floor_sub_states:
            floor_sub_states.append(floor_sub_state)
        outcome = _safe_token(
            repair_policy.get("interruption_outcome")
            or persistent.get("interruption_outcome"),
            default="none",
            limit=64,
        )
        if outcome not in interruption_outcomes:
            interruption_outcomes.append(outcome)
        speech_state = {
            "action": _safe_token(speech_policy.get("action"), default="unknown", limit=64),
            "generation_stale": _safe_bool(speech_policy.get("generation_stale")),
            "estimated_duration_ms": _safe_int(speech_policy.get("estimated_duration_ms")),
            "subtitle_timing_policy": _safe_token(
                speech_policy.get("subtitle_timing_policy"),
                default="unknown",
                limit=80,
            ),
            "stale_generation_state": _safe_token(
                speech_policy.get("stale_generation_state"),
                default="unavailable",
                limit=80,
            ),
            "backend_capability_labels": _dedupe_reason_codes(
                speech_policy.get("backend_capability_labels"),
                limit=12,
            ),
            "outstanding_speech_chunks": _safe_int(
                persistent.get("outstanding_speech_chunks")
            ),
            "held_speech_chunks": _safe_int(persistent.get("held_speech_chunks")),
            "stale_output_dropped_count": _safe_int(
                persistent.get("stale_output_dropped_count")
            ),
        }
        camera_state = {
            "state": _safe_token(camera_policy.get("state"), default="unknown", limit=64),
            "freshness_state": _safe_token(
                camera_policy.get("freshness_state"),
                default="unknown",
                limit=64,
            ),
            "fresh_frame_used": _safe_bool(camera_policy.get("fresh_frame_used")),
            "scene_transition": _safe_token(
                camera_policy.get("scene_transition"),
                default="none",
                limit=64,
            ),
            "camera_honesty_state": _safe_token(
                camera_policy.get("camera_honesty_state"),
                default="unavailable",
                limit=64,
            ),
            "user_presence_hint": _safe_token(
                camera_policy.get("user_presence_hint"),
                default="unknown",
                limit=64,
            ),
            "last_moondream_result_state": _safe_token(
                camera_policy.get("last_moondream_result_state"),
                default="none",
                limit=64,
            ),
            "object_showing_likelihood": _safe_float(
                camera_policy.get("object_showing_likelihood")
            ),
        }
        memory_effect_count = max(memory_effect_count, _safe_int(memory_policy.get("effect_count")))
        persona_effect_count = max(
            persona_effect_count,
            _safe_int(persona_policy.get("effect_count")),
        )
        if camera_state["state"] in {"error", "stale_or_limited"}:
            failure_labels.add(f"camera:{camera_state['state']}")
        if speech_state["generation_stale"]:
            failure_labels.add("speech:generation_stale")
        plan_stance = "attentive_listening"
        plan_shape = "wait_then_answer"
        if outcome in {"accepted", "interrupted", "output_flushed"}:
            plan_stance = "repairing"
            plan_shape = "repair_then_answer"
        elif camera_state["fresh_frame_used"] and camera_state["camera_honesty_state"] == "can_see_now":
            plan_stance = "visually_grounded"
            plan_shape = "visual_grounding"
        elif _safe_bool(memory_policy.get("used")) or _safe_bool(persona_policy.get("plan_available")):
            plan_stance = "grounded_callback"
            plan_shape = "callback_then_answer"
        language = _safe_token(frame.get("language"), default="unknown", limit=32)
        plan_summary = (
            "先按安全边界更新计划，再继续回应。"
            if language == "zh"
            else "Update the plan at the safe boundary, then continue."
        )
        performance_plan_summaries.append(
            {
                "sequence": _safe_int(frame.get("sequence")),
                "boundary": boundary,
                "stance": plan_stance,
                "response_shape": plan_shape,
                "plan_summary": plan_summary,
            }
        )
        timeline.append(
            {
                "sequence": _safe_int(frame.get("sequence")),
                "boundary": boundary,
                "created_at_ms": _safe_int(frame.get("created_at_ms")),
                "speech_action": speech_state["action"],
                "floor_state": floor_state,
                "floor_sub_state": floor_sub_state,
                "camera_state": camera_state["state"],
                "camera_scene_transition": camera_state["scene_transition"],
                "camera_honesty_state": camera_state["camera_honesty_state"],
                "interruption_outcome": outcome,
            }
        )

    for item in safety_violations:
        reason_codes = item.get("reason_codes") if isinstance(item, Mapping) else ()
        for code in reason_codes if isinstance(reason_codes, list) else ():
            failure_labels.add(_safe_token(code, default="actor_control:safety_violation"))
    return ActorControlReplaySummary(
        frame_count=len(frame_rows),
        profiles=tuple(sorted(profiles)),
        languages=tuple(sorted(languages)),
        tts_labels=tuple(sorted(tts_labels)),
        boundary_counts=dict(sorted(boundary_counts.items())),
        boundary_timeline=tuple(timeline),
        speech_state=speech_state,
        floor_states=tuple(floor_states),
        floor_sub_states=tuple(floor_sub_states),
        camera_state=camera_state,
        interruption_outcomes=tuple(interruption_outcomes),
        memory_persona_effects={
            "memory_effect_count": memory_effect_count,
            "persona_effect_count": persona_effect_count,
        },
        failure_labels=tuple(sorted(failure_labels)),
        performance_plan_summaries=tuple(performance_plan_summaries[-12:]),
        safety_violations=tuple(dict(item) for item in safety_violations),
    )


__all__ = [
    "ACTOR_CONTROL_FRAME_V3_BOUNDARY_TYPES",
    "ACTOR_CONTROL_FRAME_V3_SCHEMA_VERSION",
    "DEFAULT_SPEECH_CHUNK_LOOKAHEAD",
    "DEFAULT_SUBTITLE_LOOKAHEAD",
    "ActorControlConditionCacheV3",
    "ActorControlFrameV3",
    "ActorControlPersistentStateV3",
    "ActorControlReplaySummary",
    "ActorControlScheduler",
    "compile_actor_control_frames_v3",
    "compile_actor_control_frames_v3_from_actor_trace",
    "find_actor_control_safety_violations",
    "load_actor_control_frames_v3_jsonl",
    "load_actor_events_for_control_v3",
    "render_actor_control_frames_v3_jsonl",
    "replay_actor_control_frames_v3_jsonl",
    "summarize_actor_control_frames_v3",
]

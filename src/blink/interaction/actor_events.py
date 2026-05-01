"""Public-safe actor event schema v2 and replay helpers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Optional


class ActorEventModeV2(str, Enum):
    """High-level public actor runtime modes."""

    CONNECTED = "connected"
    LISTENING = "listening"
    HEARD = "heard"
    THINKING = "thinking"
    LOOKING = "looking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING = "waiting"
    ERROR = "error"
    DEGRADED = "degraded"
    RECOVERED = "recovered"


class ActorEventTypeV2(str, Enum):
    """Canonical actor event types shared by the bilingual browser paths."""

    CONNECTED = "connected"
    LISTENING = "listening"
    SPEECH_STARTED = "speech_started"
    PARTIAL_HEARD = "partial_heard"
    FINAL_HEARD = "final_heard"
    LISTENING_STARTED = "listening_started"
    PARTIAL_UNDERSTANDING_UPDATED = "partial_understanding_updated"
    FINAL_UNDERSTANDING_READY = "final_understanding_ready"
    LISTENING_DEGRADED = "listening_degraded"
    THINKING = "thinking"
    LOOKING = "looking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING = "waiting"
    ERROR = "error"
    MEMORY_USED = "memory_used"
    PERSONA_PLAN_COMPILED = "persona_plan_compiled"
    FLOOR_TRANSITION = "floor_transition"
    INTERRUPTION_CANDIDATE = "interruption_candidate"
    INTERRUPTION_ACCEPTED = "interruption_accepted"
    INTERRUPTION_REJECTED = "interruption_rejected"
    OUTPUT_FLUSHED = "output_flushed"
    INTERRUPTION_RECOVERED = "interruption_recovered"
    DEGRADED = "degraded"
    RECOVERED = "recovered"


_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_MAX_METADATA_DEPTH = 3
_MAX_METADATA_ITEMS = 24
_MAX_LIST_ITEMS = 16
_MAX_STRING_LENGTH = 180
_MAX_REASON_CODES = 24
_UNSAFE_KEY_EXACT = {
    "audio",
    "bytes",
    "candidate",
    "content",
    "credentials",
    "db_path",
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
_UNSAFE_KEY_FRAGMENTS = {
    "authorization",
    "audio",
    "candidate",
    "credential",
    "example",
    "hidden",
    "image",
    "memory_id",
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
    "discourse_category_labels",
    "memory_conflict_labels",
    "memory_effect_labels",
    "memory_staleness_labels",
    "last_text_kind",
    "selected_memory_ids",
    "stale_generation_token",
    "text_kind",
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
_PERFORMANCE_EVENT_TYPE_MAP: dict[str, ActorEventTypeV2] = {
    "browser_session.created": ActorEventTypeV2.CONNECTED,
    "browser_media.reported": ActorEventTypeV2.CONNECTED,
    "camera.connected": ActorEventTypeV2.CONNECTED,
    "webrtc.client_connected": ActorEventTypeV2.CONNECTED,
    "webrtc.connection_created": ActorEventTypeV2.CONNECTED,
    "voice.speech_started": ActorEventTypeV2.SPEECH_STARTED,
    "voice.speech_continuing": ActorEventTypeV2.LISTENING,
    "voice.speech_stopped": ActorEventTypeV2.WAITING,
    "stt.transcribing": ActorEventTypeV2.LISTENING,
    "stt.partial_transcription": ActorEventTypeV2.PARTIAL_HEARD,
    "stt.transcription": ActorEventTypeV2.FINAL_HEARD,
    "user_turn.summary": ActorEventTypeV2.FINAL_HEARD,
    "active_listening.listening_started": ActorEventTypeV2.LISTENING_STARTED,
    "active_listening.partial_understanding_updated": (
        ActorEventTypeV2.PARTIAL_UNDERSTANDING_UPDATED
    ),
    "active_listening.final_understanding_ready": ActorEventTypeV2.FINAL_UNDERSTANDING_READY,
    "active_listening.listening_degraded": ActorEventTypeV2.LISTENING_DEGRADED,
    "llm.response_start": ActorEventTypeV2.THINKING,
    "llm.response_end": ActorEventTypeV2.WAITING,
    "speech.generation_start": ActorEventTypeV2.THINKING,
    "speech.tts_request_start": ActorEventTypeV2.THINKING,
    "vision.fetch_user_image_requested": ActorEventTypeV2.LOOKING,
    "vision.fetch_user_image_start": ActorEventTypeV2.LOOKING,
    "vision.fetch_user_image_success": ActorEventTypeV2.LOOKING,
    "vision.fetch_user_image_error": ActorEventTypeV2.ERROR,
    "camera.frame_stale": ActorEventTypeV2.DEGRADED,
    "camera.health_stalled": ActorEventTypeV2.DEGRADED,
    "camera.track_stalled": ActorEventTypeV2.DEGRADED,
    "tts.speech_start": ActorEventTypeV2.SPEAKING,
    "speech.audio_start": ActorEventTypeV2.SPEAKING,
    "speech.subtitle_ready": ActorEventTypeV2.SPEAKING,
    "tts.speech_end": ActorEventTypeV2.WAITING,
    "runtime.interrupted": ActorEventTypeV2.INTERRUPTED,
    "interruption.candidate": ActorEventTypeV2.INTERRUPTION_CANDIDATE,
    "interruption.accepted": ActorEventTypeV2.INTERRUPTION_ACCEPTED,
    "interruption.rejected": ActorEventTypeV2.INTERRUPTION_REJECTED,
    "interruption.suppressed": ActorEventTypeV2.INTERRUPTION_REJECTED,
    "interruption.output_flushed": ActorEventTypeV2.OUTPUT_FLUSHED,
    "interruption.output_dropped": ActorEventTypeV2.OUTPUT_FLUSHED,
    "speech.chunk_stale_dropped": ActorEventTypeV2.OUTPUT_FLUSHED,
    "runtime.task_finished": ActorEventTypeV2.WAITING,
    "webrtc.client_disconnected": ActorEventTypeV2.WAITING,
    "webrtc.connection_closed": ActorEventTypeV2.WAITING,
    "memory.action": ActorEventTypeV2.MEMORY_USED,
    "persona.plan_compiled": ActorEventTypeV2.PERSONA_PLAN_COMPILED,
    "memory_persona.performance_plan_committed": ActorEventTypeV2.PERSONA_PLAN_COMPILED,
    "floor.transition": ActorEventTypeV2.FLOOR_TRANSITION,
    "interruption.listening_resumed": ActorEventTypeV2.INTERRUPTION_RECOVERED,
    "camera.track_resumed": ActorEventTypeV2.RECOVERED,
    "microphone.track_resumed": ActorEventTypeV2.RECOVERED,
}
_ACTOR_TYPE_MODE_MAP: dict[ActorEventTypeV2, ActorEventModeV2] = {
    ActorEventTypeV2.CONNECTED: ActorEventModeV2.CONNECTED,
    ActorEventTypeV2.LISTENING: ActorEventModeV2.LISTENING,
    ActorEventTypeV2.SPEECH_STARTED: ActorEventModeV2.LISTENING,
    ActorEventTypeV2.PARTIAL_HEARD: ActorEventModeV2.LISTENING,
    ActorEventTypeV2.FINAL_HEARD: ActorEventModeV2.HEARD,
    ActorEventTypeV2.LISTENING_STARTED: ActorEventModeV2.LISTENING,
    ActorEventTypeV2.PARTIAL_UNDERSTANDING_UPDATED: ActorEventModeV2.LISTENING,
    ActorEventTypeV2.FINAL_UNDERSTANDING_READY: ActorEventModeV2.HEARD,
    ActorEventTypeV2.LISTENING_DEGRADED: ActorEventModeV2.DEGRADED,
    ActorEventTypeV2.THINKING: ActorEventModeV2.THINKING,
    ActorEventTypeV2.LOOKING: ActorEventModeV2.LOOKING,
    ActorEventTypeV2.SPEAKING: ActorEventModeV2.SPEAKING,
    ActorEventTypeV2.INTERRUPTED: ActorEventModeV2.INTERRUPTED,
    ActorEventTypeV2.WAITING: ActorEventModeV2.WAITING,
    ActorEventTypeV2.ERROR: ActorEventModeV2.ERROR,
    ActorEventTypeV2.MEMORY_USED: ActorEventModeV2.THINKING,
    ActorEventTypeV2.PERSONA_PLAN_COMPILED: ActorEventModeV2.THINKING,
    ActorEventTypeV2.FLOOR_TRANSITION: ActorEventModeV2.WAITING,
    ActorEventTypeV2.INTERRUPTION_CANDIDATE: ActorEventModeV2.SPEAKING,
    ActorEventTypeV2.INTERRUPTION_ACCEPTED: ActorEventModeV2.INTERRUPTED,
    ActorEventTypeV2.INTERRUPTION_REJECTED: ActorEventModeV2.SPEAKING,
    ActorEventTypeV2.OUTPUT_FLUSHED: ActorEventModeV2.INTERRUPTED,
    ActorEventTypeV2.INTERRUPTION_RECOVERED: ActorEventModeV2.RECOVERED,
    ActorEventTypeV2.DEGRADED: ActorEventModeV2.DEGRADED,
    ActorEventTypeV2.RECOVERED: ActorEventModeV2.RECOVERED,
}


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = _TOKEN_RE.sub("_", str(value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _safe_optional_token(value: object, *, limit: int = 96) -> str | None:
    if value in (None, ""):
        return None
    return _safe_token(value, limit=limit)


def _safe_public_label(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = " ".join(str(value or "").split())[:limit]
    if not text or _value_has_unsafe_token(text):
        return default
    return text


def _dedupe_reason_codes(values: list[str], *, limit: int = _MAX_REASON_CODES) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        code = _safe_token(raw_value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def _value_has_unsafe_token(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in _UNSAFE_VALUE_TOKENS)


def _metadata_key_is_safe(key: object) -> bool:
    text = _safe_token(key, default="", limit=80).lower()
    if not text or text in _UNSAFE_KEY_EXACT:
        return False
    if text in _SAFE_KEY_EXACT:
        return True
    if text.endswith(_SAFE_KEY_SUFFIXES):
        return True
    return not any(fragment in text for fragment in _UNSAFE_KEY_FRAGMENTS)


def _safe_string_value(value: str) -> str | None:
    text = " ".join(str(value or "").split())[:_MAX_STRING_LENGTH]
    if _value_has_unsafe_token(text):
        return None
    return text


def _sanitize_metadata_value(
    value: object,
    *,
    depth: int,
    violations: list[str],
) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        violations.append("actor_metadata:nonfinite_number_omitted")
        return None
    if isinstance(value, str):
        sanitized = _safe_string_value(value)
        if sanitized is None:
            violations.append("actor_metadata:unsafe_value_omitted")
        return sanitized
    if depth >= _MAX_METADATA_DEPTH:
        violations.append("actor_metadata:max_depth_reached")
        return _safe_token(type(value).__name__, limit=80)
    if isinstance(value, (list, tuple, set)):
        items: list[object] = []
        for item in list(value)[:_MAX_LIST_ITEMS]:
            sanitized = _sanitize_metadata_value(item, depth=depth + 1, violations=violations)
            if sanitized is not None:
                items.append(sanitized)
        if len(value) > _MAX_LIST_ITEMS:  # type: ignore[arg-type]
            violations.append("actor_metadata:list_truncated")
        return items
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        items = list(value.items())
        for raw_key, raw_item in items[:_MAX_METADATA_ITEMS]:
            if not _metadata_key_is_safe(raw_key):
                violations.append("actor_metadata:unsafe_key_omitted")
                continue
            key = _safe_token(raw_key, default="", limit=80)
            if not key:
                violations.append("actor_metadata:unsafe_key_omitted")
                continue
            sanitized = _sanitize_metadata_value(
                raw_item,
                depth=depth + 1,
                violations=violations,
            )
            if sanitized is not None:
                result[key] = sanitized
        if len(items) > _MAX_METADATA_ITEMS:
            violations.append("actor_metadata:object_truncated")
        return result
    violations.append("actor_metadata:unsupported_value_omitted")
    return _safe_token(type(value).__name__, limit=80)


def sanitize_actor_metadata(metadata: object) -> tuple[dict[str, object], list[str]]:
    """Return trace-safe actor metadata and public violation reason codes."""
    violations: list[str] = []
    if not isinstance(metadata, Mapping):
        return {}, []
    sanitized = _sanitize_metadata_value(metadata, depth=0, violations=violations)
    result = sanitized if isinstance(sanitized, dict) else {}
    return result, _dedupe_reason_codes(violations)


def sanitize_actor_reason_codes(values: object) -> tuple[list[str], list[str]]:
    """Return trace-safe reason codes plus sanitizer violation reason codes."""
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    reason_codes: list[str] = []
    violations: list[str] = []
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        raw_text = str(raw_value)
        if _value_has_unsafe_token(raw_text):
            violations.append("actor_reason_codes:unsafe_value_omitted")
            continue
        reason_codes.append(_safe_token(raw_value, default="", limit=96))
    return _dedupe_reason_codes(reason_codes), _dedupe_reason_codes(violations)


@dataclass(frozen=True)
class ActorEventContext:
    """Runtime context attached to every public actor event."""

    profile: str = "manual"
    language: str = "unknown"
    tts_backend: str = "unknown"
    tts_label: str = "unknown"
    vision_backend: str = "none"

    @classmethod
    def from_value(cls, value: object) -> "ActorEventContext":
        """Build actor context from an existing context or mapping."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                profile=_safe_token(value.get("profile"), default="manual", limit=96),
                language=_safe_token(value.get("language"), default="unknown", limit=32),
                tts_backend=_safe_token(value.get("tts_backend"), default="unknown", limit=80),
                tts_label=_safe_public_label(
                    value.get("tts_label"),
                    default="unknown",
                    limit=80,
                ),
                vision_backend=_safe_token(
                    value.get("vision_backend"),
                    default="none",
                    limit=80,
                ),
            )
        return cls()


@dataclass(frozen=True)
class ActorEventV2:
    """One public-safe actor runtime event."""

    event_id: int
    event_type: ActorEventTypeV2 | str
    mode: ActorEventModeV2 | str
    profile: str
    language: str
    tts_backend: str
    tts_label: str
    vision_backend: str
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    schema_version: int = 2

    def as_dict(self) -> dict[str, object]:
        """Return the event as a public-safe API and trace payload."""
        try:
            event_type = ActorEventTypeV2(str(_enum_value(self.event_type))).value
        except ValueError:
            event_type = ActorEventTypeV2.WAITING.value
        try:
            mode = ActorEventModeV2(str(_enum_value(self.mode))).value
        except ValueError:
            mode = _ACTOR_TYPE_MODE_MAP[ActorEventTypeV2(event_type)].value
        metadata, metadata_violations = sanitize_actor_metadata(self.metadata)
        reason_codes, reason_violations = sanitize_actor_reason_codes(self.reason_codes)
        all_reason_codes = _dedupe_reason_codes(
            [*reason_codes, *metadata_violations, *reason_violations],
            limit=_MAX_REASON_CODES,
        )
        return {
            "schema_version": 2,
            "event_id": max(1, int(self.event_id)),
            "event_type": event_type,
            "mode": mode,
            "timestamp": str(self.timestamp),
            "profile": _safe_token(self.profile, default="manual", limit=96),
            "language": _safe_token(self.language, default="unknown", limit=32),
            "tts_backend": _safe_token(self.tts_backend, default="unknown", limit=80),
            "tts_label": _safe_public_label(self.tts_label, default="unknown", limit=80),
            "vision_backend": _safe_token(self.vision_backend, default="none", limit=80),
            "source": _safe_token(self.source, default="runtime", limit=64),
            "session_id": _safe_optional_token(self.session_id, limit=96),
            "client_id": _safe_optional_token(self.client_id, limit=96),
            "metadata": metadata,
            "reason_codes": all_reason_codes,
        }


def actor_event_type_for_performance_event(event_type: object) -> ActorEventTypeV2:
    """Map a browser performance event type to a canonical actor event type."""
    raw_type = str(event_type or "")
    if raw_type in _PERFORMANCE_EVENT_TYPE_MAP:
        return _PERFORMANCE_EVENT_TYPE_MAP[raw_type]
    if raw_type.endswith(".error") or raw_type.endswith(":error"):
        return ActorEventTypeV2.ERROR
    if raw_type.endswith(".degraded") or ".degraded" in raw_type:
        return ActorEventTypeV2.DEGRADED
    if raw_type.endswith(".resumed") or raw_type.endswith(".recovered"):
        return ActorEventTypeV2.RECOVERED
    if raw_type.startswith("vision."):
        return ActorEventTypeV2.LOOKING
    if raw_type.startswith("tts.") or raw_type.startswith("speech."):
        return ActorEventTypeV2.SPEAKING
    if raw_type.startswith("stt.") or raw_type.startswith("voice."):
        return ActorEventTypeV2.LISTENING
    if raw_type.startswith("llm."):
        return ActorEventTypeV2.THINKING
    if raw_type.startswith("memory."):
        return ActorEventTypeV2.MEMORY_USED
    if raw_type.startswith("memory_persona."):
        return ActorEventTypeV2.PERSONA_PLAN_COMPILED
    if raw_type.startswith("persona."):
        return ActorEventTypeV2.PERSONA_PLAN_COMPILED
    if raw_type.startswith("floor."):
        return ActorEventTypeV2.FLOOR_TRANSITION
    if raw_type.startswith("interruption."):
        return ActorEventTypeV2.INTERRUPTED
    return ActorEventTypeV2.WAITING


def actor_mode_for_event_type(event_type: ActorEventTypeV2 | str, fallback: object = None) -> ActorEventModeV2:
    """Return the canonical actor mode for an actor event type."""
    try:
        resolved_type = ActorEventTypeV2(str(event_type))
    except ValueError:
        try:
            resolved_type = ActorEventTypeV2(str(_enum_value(event_type)))
        except ValueError:
            resolved_type = ActorEventTypeV2.WAITING
    if resolved_type in {
        ActorEventTypeV2.ERROR,
        ActorEventTypeV2.DEGRADED,
        ActorEventTypeV2.RECOVERED,
    }:
        return _ACTOR_TYPE_MODE_MAP[resolved_type]
    if resolved_type in {
        ActorEventTypeV2.CONNECTED,
        ActorEventTypeV2.LISTENING,
        ActorEventTypeV2.SPEAKING,
        ActorEventTypeV2.THINKING,
        ActorEventTypeV2.LOOKING,
        ActorEventTypeV2.INTERRUPTED,
        ActorEventTypeV2.INTERRUPTION_CANDIDATE,
        ActorEventTypeV2.INTERRUPTION_ACCEPTED,
        ActorEventTypeV2.INTERRUPTION_REJECTED,
        ActorEventTypeV2.OUTPUT_FLUSHED,
        ActorEventTypeV2.INTERRUPTION_RECOVERED,
    }:
        return _ACTOR_TYPE_MODE_MAP[resolved_type]
    if fallback is not None:
        try:
            return ActorEventModeV2(str(_enum_value(fallback)))
        except ValueError:
            pass
    return _ACTOR_TYPE_MODE_MAP[resolved_type]


def actor_event_from_performance_event(
    performance_event: Any,
    *,
    context: ActorEventContext | Mapping[str, object] | None = None,
) -> ActorEventV2:
    """Build a v2 actor event from a v1 browser performance event."""
    actor_context = ActorEventContext.from_value(context)
    actor_type = actor_event_type_for_performance_event(getattr(performance_event, "event_type", ""))
    mode = actor_mode_for_event_type(actor_type, fallback=getattr(performance_event, "mode", None))
    metadata, metadata_violations = sanitize_actor_metadata(
        getattr(performance_event, "metadata", {})
    )
    reason_codes, reason_violations = sanitize_actor_reason_codes(
        getattr(performance_event, "reason_codes", ())
    )
    return ActorEventV2(
        event_id=int(getattr(performance_event, "event_id", 1)),
        event_type=actor_type,
        mode=mode,
        timestamp=str(getattr(performance_event, "timestamp", datetime.now(UTC).isoformat())),
        profile=actor_context.profile,
        language=actor_context.language,
        tts_backend=actor_context.tts_backend,
        tts_label=actor_context.tts_label,
        vision_backend=actor_context.vision_backend,
        source=str(getattr(performance_event, "source", "runtime")),
        session_id=getattr(performance_event, "session_id", None),
        client_id=getattr(performance_event, "client_id", None),
        metadata=metadata,
        reason_codes=[*reason_codes, *metadata_violations, *reason_violations],
    )


def find_actor_trace_safety_violations(payload: object) -> list[str]:
    """Return public safety violation reason codes for a decoded trace line."""
    violations: list[str] = []
    if not isinstance(payload, Mapping):
        return ["actor_trace:malformed_line"]
    unsafe_top_level = {
        "audio",
        "candidate",
        "content",
        "hidden_prompt",
        "ice",
        "image",
        "messages",
        "prompt",
        "raw",
        "sdp",
        "secret",
        "transcript",
    }
    for key, value in payload.items():
        key_text = _safe_token(key, default="", limit=80).lower()
        if key_text in unsafe_top_level or any(
            fragment in key_text for fragment in _UNSAFE_KEY_FRAGMENTS
        ):
            violations.append("actor_trace:unsafe_key_present")
        if isinstance(value, str) and _value_has_unsafe_token(value):
            violations.append("actor_trace:unsafe_value_present")
    metadata, metadata_violations = sanitize_actor_metadata(payload.get("metadata"))
    if metadata != (payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}):
        violations.append("actor_trace:metadata_not_trace_safe")
    violations.extend(metadata_violations)
    _reason_codes, reason_violations = sanitize_actor_reason_codes(payload.get("reason_codes", ()))
    violations.extend(reason_violations)
    return _dedupe_reason_codes(violations)


class ActorTraceWriter:
    """Bounded JSONL writer for sanitized actor events."""

    def __init__(
        self,
        *,
        trace_dir: Path | str,
        profile: str,
        run_id: str,
        max_events: int = 10_000,
    ):
        """Initialize the writer and create the trace file path."""
        self._trace_dir = Path(trace_dir).expanduser()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_profile = _safe_token(profile, default="manual", limit=80)
        safe_run_id = _safe_token(run_id, default="run", limit=80)
        self._path = self._trace_dir / f"actor-trace-{stamp}-{safe_profile}-{safe_run_id}.jsonl"
        self._max_events = max(1, int(max_events))
        self._written = 0
        self._limit_recorded = False
        self._closed = False
        self._lock = Lock()

    @property
    def path(self) -> Path:
        """Return the trace file path."""
        return self._path

    @property
    def written_count(self) -> int:
        """Return the number of trace lines written."""
        with self._lock:
            return self._written

    def append(self, event: ActorEventV2) -> None:
        """Append one sanitized actor event to the JSONL trace."""
        with self._lock:
            if self._closed:
                return
            if self._written >= self._max_events:
                if not self._limit_recorded:
                    self._write_unlocked(
                        ActorEventV2(
                            event_id=int(event.event_id) + 1,
                            event_type=ActorEventTypeV2.DEGRADED,
                            mode=ActorEventModeV2.DEGRADED,
                            profile=event.profile,
                            language=event.language,
                            tts_backend=event.tts_backend,
                            tts_label=event.tts_label,
                            vision_backend=event.vision_backend,
                            source="actor_trace",
                            session_id=event.session_id,
                            client_id=event.client_id,
                            metadata={"trace_event_limit": self._max_events},
                            reason_codes=["trace.limit_reached"],
                        )
                    )
                    self._limit_recorded = True
                self._closed = True
                return
            self._write_unlocked(event)

    def _write_unlocked(self, event: ActorEventV2) -> None:
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        self._written += 1


def _safe_event_line_summary(index: int, reason: str) -> dict[str, object]:
    return {"line": index, "reason_codes": [reason]}


def replay_actor_trace(path: Path | str) -> dict[str, object]:
    """Replay a saved actor trace into a public summary without live runtime calls."""
    trace_path = Path(path)
    event_type_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    profiles: set[str] = set()
    languages: set[str] = set()
    tts_backends: set[str] = set()
    tts_labels: set[str] = set()
    vision_backends: set[str] = set()
    mode_timeline: list[dict[str, object]] = []
    safety_violations: list[dict[str, object]] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    event_count = 0
    previous_mode: str | None = None

    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                safety_violations.append(
                    _safe_event_line_summary(line_number, "actor_trace:malformed_json")
                )
                continue
            violations = find_actor_trace_safety_violations(payload)
            if violations:
                safety_violations.append({"line": line_number, "reason_codes": violations})
            if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
                safety_violations.append(
                    _safe_event_line_summary(line_number, "actor_trace:unsupported_schema")
                )
                continue

            event_count += 1
            event_type = _safe_token(payload.get("event_type"), default="waiting", limit=96)
            mode = _safe_token(payload.get("mode"), default="waiting", limit=80)
            timestamp = str(payload.get("timestamp") or "")
            event_id = payload.get("event_id")
            event_type_counts[event_type] += 1
            mode_counts[mode] += 1
            if first_timestamp is None and timestamp:
                first_timestamp = timestamp
            if timestamp:
                last_timestamp = timestamp
            profiles.add(_safe_token(payload.get("profile"), default="manual", limit=96))
            languages.add(_safe_token(payload.get("language"), default="unknown", limit=32))
            tts_backends.add(_safe_token(payload.get("tts_backend"), default="unknown", limit=80))
            tts_labels.add(
                _safe_public_label(payload.get("tts_label"), default="unknown", limit=80)
            )
            vision_backends.add(
                _safe_token(payload.get("vision_backend"), default="none", limit=80)
            )
            if mode != previous_mode:
                mode_timeline.append(
                    {
                        "event_id": event_id if isinstance(event_id, int) else event_count,
                        "timestamp": timestamp,
                        "mode": mode,
                        "event_type": event_type,
                    }
                )
                previous_mode = mode

    return {
        "schema_version": 2,
        "event_count": event_count,
        "mode_timeline": mode_timeline,
        "mode_counts": dict(sorted(mode_counts.items())),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "profiles": sorted(profiles),
        "languages": sorted(languages),
        "tts_backends": sorted(tts_backends),
        "tts_labels": sorted(tts_labels),
        "vision_backends": sorted(vision_backends),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "safety_violations": safety_violations,
    }

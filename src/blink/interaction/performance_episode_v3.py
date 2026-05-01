"""Privacy-safe PerformanceEpisodeV3 ledgers built from actor events."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping

from blink.interaction.actor_events import ActorEventV2

PERFORMANCE_EPISODE_V3_SCHEMA_VERSION = 3
PERFORMANCE_EPISODE_V3_SEGMENT_TYPES = (
    "listen_segment",
    "think_segment",
    "look_segment",
    "speak_segment",
    "overlap_segment",
    "repair_segment",
    "idle_segment",
)
PERFORMANCE_EPISODE_V3_PRIVACY_LEVEL = "public_safe"
PERFORMANCE_EPISODE_V3_TERMINAL_EVENTS = {
    "runtime.task_finished",
    "webrtc.client_disconnected",
    "webrtc.connection_closed",
}

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_MAX_REASON_CODES = 48
_MAX_LABELS = 32
_MAX_EVENTS_PER_EPISODE = 10_000
_MAX_MAPPING_ITEMS = 48
_MAX_LIST_ITEMS = 24
_MAX_DEPTH = 8
_HASH_PREFIX_LENGTH = 16
_ALLOWED_ACTOR_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "event_type",
    "mode",
    "timestamp",
    "profile",
    "language",
    "tts_backend",
    "tts_label",
    "vision_backend",
    "source",
    "session_id",
    "client_id",
    "metadata",
    "reason_codes",
}
_ALLOWED_EPISODE_TOP_LEVEL_KEYS = {
    "schema_version",
    "episode_id",
    "profile",
    "language",
    "tts_runtime_label",
    "privacy_level",
    "created_at_ms",
    "session_id_hash",
    "client_id_hash",
    "segments",
    "metrics",
    "failure_labels",
    "sanitizer",
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
    "raw_audio",
    "raw_image",
    "raw_memory",
    "raw_transcript",
    "sdp",
    "secret",
    "system_prompt",
    "token",
    "transcript",
    "url",
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
    "text",
    "token",
    "transcript",
    "url",
}
_SAFE_KEY_EXACT = {
    "active_listening",
    "actor_trace_persistence_opt_in",
    "allow_barge_in_default",
    "assistant_has_floor",
    "blocked_keys",
    "blocked_values",
    "browser_vision_default",
    "camera",
    "camera_state",
    "client_id_hash",
    "continuous_perception_default",
    "created_at_ms",
    "duration_ms",
    "echo_cancellation_state",
    "end_ms",
    "event_ids",
    "event_type",
    "event_type_counts",
    "failure_labels",
    "floor",
    "fresh",
    "interruption",
    "interruption_accepted",
    "interruption_candidate",
    "interruption_recovered",
    "interruption_rejected",
    "language",
    "last_text_kind",
    "memory",
    "memory_used",
    "memory_persona_effect",
    "metadata",
    "metrics",
    "mode",
    "mode_timeline",
    "noise_suppression_state",
    "outcome",
    "overlap",
    "passed",
    "persona",
    "persona_plan_compiled",
    "privacy_level",
    "profile",
    "protected",
    "reason_codes",
    "repair",
    "schema_version",
    "segment_count",
    "segment_id",
    "segment_type",
    "segments",
    "session_id_hash",
    "source",
    "start_ms",
    "state",
    "states",
    "stale_generation_token",
    "connected",
    "degraded",
    "error",
    "final_heard",
    "final_understanding_ready",
    "floor_transition",
    "interrupted",
    "listening",
    "listening_degraded",
    "listening_started",
    "looking",
    "output_flushed",
    "partial_heard",
    "partial_understanding_updated",
    "recovered",
    "speaking",
    "speech_started",
    "thinking",
    "waiting",
    "subturn_index",
    "text_kind",
    "timestamp",
    "tts_label",
    "tts_runtime_label",
    "turn_index",
    "used",
    "user_has_floor",
    "vision_backend",
}
_SAFE_KEY_SUFFIXES = (
    "_available",
    "_chars",
    "_count",
    "_counts",
    "_enabled",
    "_hash",
    "_id",
    "_ids",
    "_index",
    "_kind",
    "_label",
    "_labels",
    "_ms",
    "_state",
    "_states",
)
_UNSAFE_VALUE_TOKENS = (
    "-----begin",
    "[blink_brain_context]",
    "a=candidate",
    "authorization:",
    "base64,",
    "bearer ",
    "candidate:",
    "data:audio",
    "data:image",
    "developer prompt",
    "hidden prompt",
    "ice-ufrag",
    "m=audio",
    "m=video",
    "o=-",
    "password",
    "raw audio",
    "raw image",
    "secret",
    "sk-",
    "system prompt",
    "token",
    "traceback",
    "v=0",
)
_LISTENING_EVENTS = {
    "listening",
    "speech_started",
    "partial_heard",
    "final_heard",
    "listening_started",
    "partial_understanding_updated",
    "final_understanding_ready",
    "listening_degraded",
}
_THINKING_EVENTS = {"thinking", "memory_used", "persona_plan_compiled"}
_LOOKING_EVENTS = {"looking"}
_SPEAKING_EVENTS = {"speaking"}
_OVERLAP_EVENTS = {"interruption_candidate", "interruption_rejected"}
_REPAIR_EVENTS = {
    "interrupted",
    "interruption_accepted",
    "output_flushed",
    "interruption_recovered",
}
_SEGMENT_MODE_MAP = {
    "listen_segment": "listening",
    "think_segment": "thinking",
    "look_segment": "looking",
    "speak_segment": "speaking",
    "overlap_segment": "interrupted",
    "repair_segment": "interrupted",
    "idle_segment": "waiting",
}


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = _TOKEN_RE.sub("_", str(value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_optional_token(value: object, *, limit: int = 96) -> str | None:
    if value in (None, ""):
        return None
    return _safe_token(value, default="", limit=limit) or None


def _safe_sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple, set)) else ()


def _safe_label(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = " ".join(str(value or "").split())[:limit]
    if not text or _value_has_unsafe_token(text):
        return default
    return text


def _hash_ref(value: object) -> str | None:
    if value in (None, ""):
        return None
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:_HASH_PREFIX_LENGTH]


def _dedupe(values: Iterable[object], *, limit: int = _MAX_LABELS) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        text = _safe_token(raw_value, default="", limit=96)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _value_has_unsafe_token(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in _UNSAFE_VALUE_TOKENS)


def _key_is_safe(key: object, *, allowed_keys: set[str] | None = None) -> bool:
    text = _safe_token(key, default="", limit=96).lower()
    if not text:
        return False
    if allowed_keys is not None and text in allowed_keys:
        return True
    if text in _UNSAFE_KEY_EXACT:
        return False
    if text in _SAFE_KEY_EXACT:
        return True
    if text.endswith(_SAFE_KEY_SUFFIXES):
        return True
    return not any(fragment in text for fragment in _UNSAFE_KEY_FRAGMENTS)


@dataclass(frozen=True)
class PerformanceEpisodeSanitizerReport:
    """Public-safe sanitizer report for a performance episode or replay."""

    passed: bool = True
    blocked_keys: tuple[str, ...] = ()
    blocked_values: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def combine(self, other: "PerformanceEpisodeSanitizerReport") -> "PerformanceEpisodeSanitizerReport":
        """Return a merged sanitizer report."""
        return PerformanceEpisodeSanitizerReport(
            passed=self.passed and other.passed,
            blocked_keys=_dedupe([*self.blocked_keys, *other.blocked_keys]),
            blocked_values=_dedupe([*self.blocked_values, *other.blocked_values]),
            reason_codes=_dedupe([*self.reason_codes, *other.reason_codes], limit=96),
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize the sanitizer report."""
        return {
            "passed": self.passed,
            "blocked_keys": list(self.blocked_keys),
            "blocked_values": list(self.blocked_values),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PerformanceEpisodeSegmentV3:
    """One compact turn/subturn segment in a performance episode."""

    segment_id: str
    segment_type: str
    turn_index: int
    subturn_index: int
    start_ms: int
    end_ms: int
    event_ids: tuple[int, ...]
    event_type_counts: Mapping[str, int]
    reason_codes: tuple[str, ...] = ()
    floor: Mapping[str, object] = field(default_factory=dict)
    camera: Mapping[str, object] = field(default_factory=dict)
    interruption: Mapping[str, object] = field(default_factory=dict)
    speech: Mapping[str, object] = field(default_factory=dict)
    memory: Mapping[str, object] = field(default_factory=dict)
    persona: Mapping[str, object] = field(default_factory=dict)
    active_listening: Mapping[str, object] = field(default_factory=dict)
    performance_plan: Mapping[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        """Return the non-negative segment duration."""
        return max(0, int(self.end_ms) - int(self.start_ms))

    def as_dict(self) -> dict[str, object]:
        """Serialize the segment."""
        return {
            "segment_id": self.segment_id,
            "segment_type": self.segment_type,
            "turn_index": self.turn_index,
            "subturn_index": self.subturn_index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "event_ids": list(self.event_ids),
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "reason_codes": list(self.reason_codes),
            "floor": dict(self.floor),
            "camera": dict(self.camera),
            "interruption": dict(self.interruption),
            "speech": dict(self.speech),
            "memory": dict(self.memory),
            "persona": dict(self.persona),
            "active_listening": dict(self.active_listening),
            "performance_plan": dict(self.performance_plan),
        }


@dataclass(frozen=True)
class PerformanceEpisodeV3:
    """A privacy-safe session/turn/subturn performance ledger."""

    episode_id: str
    profile: str
    language: str
    tts_runtime_label: str
    created_at_ms: int
    segments: tuple[PerformanceEpisodeSegmentV3, ...]
    session_id_hash: str | None = None
    client_id_hash: str | None = None
    privacy_level: str = PERFORMANCE_EPISODE_V3_PRIVACY_LEVEL
    metrics: Mapping[str, object] = field(default_factory=dict)
    failure_labels: tuple[str, ...] = ()
    sanitizer: PerformanceEpisodeSanitizerReport = field(
        default_factory=PerformanceEpisodeSanitizerReport
    )

    def as_dict(self) -> dict[str, object]:
        """Serialize the episode."""
        return {
            "schema_version": PERFORMANCE_EPISODE_V3_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "privacy_level": self.privacy_level,
            "created_at_ms": self.created_at_ms,
            "session_id_hash": self.session_id_hash,
            "client_id_hash": self.client_id_hash,
            "segments": [segment.as_dict() for segment in self.segments],
            "metrics": dict(self.metrics),
            "failure_labels": list(self.failure_labels),
            "sanitizer": self.sanitizer.as_dict(),
        }


@dataclass(frozen=True)
class PerformanceEpisodeReplaySummary:
    """Compact offline replay summary for episode JSONL."""

    episode_count: int
    segment_counts: Mapping[str, int]
    profiles: tuple[str, ...]
    languages: tuple[str, ...]
    tts_labels: tuple[str, ...]
    mode_timeline: tuple[Mapping[str, object], ...]
    latency_metrics: Mapping[str, object]
    floor_states: tuple[str, ...]
    floor_sub_states: tuple[str, ...]
    camera_use_states: tuple[str, ...]
    interruption_outcomes: tuple[str, ...]
    memory_persona_effects: tuple[str, ...]
    failure_labels: tuple[str, ...]
    sanitizer: PerformanceEpisodeSanitizerReport

    def as_dict(self) -> dict[str, object]:
        """Serialize the replay summary."""
        return {
            "schema_version": PERFORMANCE_EPISODE_V3_SCHEMA_VERSION,
            "episode_count": self.episode_count,
            "segment_counts": dict(sorted(self.segment_counts.items())),
            "profiles": list(self.profiles),
            "languages": list(self.languages),
            "tts_labels": list(self.tts_labels),
            "mode_timeline": [dict(item) for item in self.mode_timeline],
            "latency_metrics": dict(self.latency_metrics),
            "floor_states": list(self.floor_states),
            "floor_sub_states": list(self.floor_sub_states),
            "camera_use_states": list(self.camera_use_states),
            "interruption_outcomes": list(self.interruption_outcomes),
            "memory_persona_effects": list(self.memory_persona_effects),
            "failure_labels": list(self.failure_labels),
            "sanitizer": self.sanitizer.as_dict(),
        }


def _sanitize_value(
    value: object,
    *,
    path: str,
    allowed_keys: set[str] | None,
    depth: int,
    blocked_keys: list[str],
    blocked_values: list[str],
    reason_codes: list[str],
) -> object:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        blocked_values.append(f"{path}:nonfinite")
        reason_codes.append("performance_episode:nonfinite_number_omitted")
        return None
    if isinstance(value, str):
        if _value_has_unsafe_token(value):
            blocked_values.append(path)
            reason_codes.append("performance_episode:unsafe_value_omitted")
            return None
        return _safe_label(value, default="", limit=180)
    if depth >= _MAX_DEPTH:
        reason_codes.append("performance_episode:max_depth_reached")
        return _safe_token(type(value).__name__, default="object", limit=64)
    if isinstance(value, (list, tuple, set)):
        result: list[object] = []
        values = list(value)
        for index, item in enumerate(values[:_MAX_LIST_ITEMS]):
            sanitized = _sanitize_value(
                item,
                path=f"{path}[{index}]",
                allowed_keys=None,
                depth=depth + 1,
                blocked_keys=blocked_keys,
                blocked_values=blocked_values,
                reason_codes=reason_codes,
            )
            if sanitized is not None:
                result.append(sanitized)
        if len(values) > _MAX_LIST_ITEMS:
            reason_codes.append("performance_episode:list_truncated")
        return result
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        items = list(value.items())
        for raw_key, raw_item in items[:_MAX_MAPPING_ITEMS]:
            key = _safe_token(raw_key, default="", limit=96)
            if not _key_is_safe(key, allowed_keys=allowed_keys if depth == 0 else None):
                blocked_keys.append(f"{path}.{key}" if key else path)
                reason_codes.append("performance_episode:unsafe_key_omitted")
                continue
            sanitized = _sanitize_value(
                raw_item,
                path=f"{path}.{key}",
                allowed_keys=None,
                depth=depth + 1,
                blocked_keys=blocked_keys,
                blocked_values=blocked_values,
                reason_codes=reason_codes,
            )
            if sanitized is not None:
                result[key] = sanitized
        if len(items) > _MAX_MAPPING_ITEMS:
            reason_codes.append("performance_episode:object_truncated")
        return result
    reason_codes.append("performance_episode:unsupported_value_omitted")
    return _safe_token(type(value).__name__, default="object", limit=64)


def sanitize_performance_episode_payload(
    payload: object,
    *,
    allowed_top_level_keys: set[str] | None = None,
) -> tuple[dict[str, object], PerformanceEpisodeSanitizerReport]:
    """Return a sanitized mapping plus fail-closed violation labels."""
    if not isinstance(payload, Mapping):
        return {}, PerformanceEpisodeSanitizerReport(
            passed=False,
            reason_codes=("performance_episode:malformed_payload",),
        )
    blocked_keys: list[str] = []
    blocked_values: list[str] = []
    reason_codes: list[str] = []
    sanitized = _sanitize_value(
        payload,
        path="$",
        allowed_keys=allowed_top_level_keys,
        depth=0,
        blocked_keys=blocked_keys,
        blocked_values=blocked_values,
        reason_codes=reason_codes,
    )
    result = sanitized if isinstance(sanitized, dict) else {}
    report = PerformanceEpisodeSanitizerReport(
        passed=not blocked_keys and not blocked_values and not reason_codes,
        blocked_keys=_dedupe(blocked_keys),
        blocked_values=_dedupe(blocked_values),
        reason_codes=_dedupe(reason_codes, limit=96),
    )
    return result, report


def sanitize_actor_event_for_performance_episode(
    payload: object,
) -> tuple[dict[str, object], PerformanceEpisodeSanitizerReport]:
    """Sanitize an actor-event payload for persistent episode compilation."""
    sanitized, report = sanitize_performance_episode_payload(
        payload,
        allowed_top_level_keys=_ALLOWED_ACTOR_EVENT_KEYS,
    )
    reason_codes = list(report.reason_codes)
    if sanitized.get("schema_version") != 2:
        reason_codes.append("performance_episode:unsupported_actor_event_schema")
    if "event_type" not in sanitized or "mode" not in sanitized:
        reason_codes.append("performance_episode:actor_event_shape_invalid")
    if reason_codes != list(report.reason_codes):
        report = PerformanceEpisodeSanitizerReport(
            passed=False,
            blocked_keys=report.blocked_keys,
            blocked_values=report.blocked_values,
            reason_codes=_dedupe(reason_codes, limit=96),
        )
    return sanitized, report


def find_performance_episode_safety_violations(payload: object) -> tuple[str, ...]:
    """Return public safety violation reason codes for a payload."""
    _sanitized, report = sanitize_performance_episode_payload(payload)
    return report.reason_codes


def _event_payload(event: ActorEventV2 | Mapping[str, object]) -> dict[str, object]:
    if isinstance(event, ActorEventV2):
        return event.as_dict()
    return dict(event)


def _parse_timestamp_ms(value: object) -> int | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _relative_event_times(events: list[dict[str, object]]) -> tuple[list[int], int, tuple[str, ...]]:
    parsed = [_parse_timestamp_ms(event.get("timestamp")) for event in events]
    base = next((value for value in parsed if value is not None), None)
    failure_labels: list[str] = []
    relative: list[int] = []
    previous = 0
    for index, parsed_ms in enumerate(parsed):
        if parsed_ms is None or base is None:
            failure_labels.append("timestamp:unparseable")
            current = previous + (1 if relative else index)
        else:
            current = max(0, parsed_ms - base)
            if relative:
                current = max(current, previous)
        relative.append(current)
        previous = current
    return relative, int(base or 0), _dedupe(failure_labels)


def _combined_event_text(event: Mapping[str, object]) -> str:
    parts = [
        str(event.get("event_type") or ""),
        str(event.get("mode") or ""),
        str(event.get("source") or ""),
    ]
    reason_codes = event.get("reason_codes")
    if isinstance(reason_codes, list):
        parts.extend(str(reason) for reason in reason_codes)
    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}:{value}")
    return " ".join(parts).lower()


def _segment_type_for_event(event: Mapping[str, object]) -> str:
    event_type = _safe_token(event.get("event_type"), default="waiting", limit=80)
    text = _combined_event_text(event)
    if event_type in _LISTENING_EVENTS:
        return "listen_segment"
    if event_type in _THINKING_EVENTS:
        return "think_segment"
    if event_type in _LOOKING_EVENTS or (
        event_type == "error" and ("vision" in text or "camera" in text)
    ):
        return "look_segment"
    if event_type in _SPEAKING_EVENTS:
        return "speak_segment"
    if event_type in _OVERLAP_EVENTS:
        return "overlap_segment"
    if event_type in _REPAIR_EVENTS:
        return "repair_segment"
    if event_type == "floor_transition":
        if "repair" in text:
            return "repair_segment"
        if "overlap" in text:
            return "overlap_segment"
    return "idle_segment"


def _event_id(value: object, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def _event_reason_codes(events: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    values: list[object] = []
    for event in events:
        reason_codes = event.get("reason_codes")
        if isinstance(reason_codes, list):
            values.extend(reason_codes)
    return _dedupe(values, limit=_MAX_REASON_CODES)


def _labels_from_reasons(
    events: Iterable[Mapping[str, object]],
    *,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    labels: list[str] = []
    for event in events:
        reason_codes = event.get("reason_codes")
        if not isinstance(reason_codes, list):
            continue
        for reason in reason_codes:
            text = _safe_token(reason, default="", limit=96)
            if text.startswith(prefixes):
                labels.append(text)
    return _dedupe(labels)


def _speech_capability_labels(value: Mapping[str, object]) -> tuple[str, ...]:
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
    return _dedupe(labels)


def _segment_floor(events: list[Mapping[str, object]]) -> dict[str, object]:
    states: list[str] = []
    sub_states: list[str] = []
    yield_decisions: list[str] = []
    phrase_classes: list[str] = []
    overlap = False
    repair = False
    user_has_floor = False
    assistant_has_floor = False
    for event in events:
        text = _combined_event_text(event)
        metadata = event.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("state", "floor_state"):
                if metadata.get(key) not in (None, ""):
                    states.append(_safe_token(metadata.get(key), default="unknown", limit=64))
            if metadata.get("floor_sub_state") not in (None, ""):
                sub_states.append(
                    _safe_token(metadata.get("floor_sub_state"), default="", limit=64)
                )
            if metadata.get("yield_decision") not in (None, ""):
                yield_decisions.append(
                    _safe_token(metadata.get("yield_decision"), default="", limit=64)
                )
            if metadata.get("phrase_class") not in (None, ""):
                phrase_classes.append(
                    _safe_token(metadata.get("phrase_class"), default="", limit=64)
                )
            user_has_floor = user_has_floor or metadata.get("user_has_floor") is True
            assistant_has_floor = assistant_has_floor or metadata.get("assistant_has_floor") is True
        overlap = overlap or "overlap" in text
        repair = repair or "repair" in text
    return {
        "states": list(_dedupe(states)),
        "sub_states": list(_dedupe(sub_states)),
        "yield_decisions": list(_dedupe(yield_decisions)),
        "phrase_classes": list(_dedupe(phrase_classes)),
        "overlap": overlap,
        "repair": repair,
        "user_has_floor": user_has_floor,
        "assistant_has_floor": assistant_has_floor,
    }


def _segment_camera(events: list[Mapping[str, object]]) -> dict[str, object]:
    state = "not_used"
    used = False
    fresh = False
    transitions: list[str] = []
    honesty_states: list[str] = []
    freshness_states: list[str] = []
    user_presence_hints: list[str] = []
    moondream_result_states: list[str] = []
    object_showing_likelihood_max = 0.0
    for event in events:
        text = _combined_event_text(event)
        camera_related = any(
            token in text for token in ("camera", "frame", "moondream", "vision")
        )
        metadata = event.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        transition = metadata.get("scene_transition")
        if transition not in (None, ""):
            transitions.append(_safe_token(transition, default="", limit=64))
        honesty = metadata.get("camera_honesty_state")
        if honesty not in (None, ""):
            honesty_states.append(_safe_token(honesty, default="", limit=64))
        freshness = metadata.get("freshness_state")
        if freshness not in (None, ""):
            freshness_states.append(_safe_token(freshness, default="", limit=64))
        user_presence = metadata.get("user_presence_hint")
        if user_presence not in (None, ""):
            user_presence_hints.append(_safe_token(user_presence, default="", limit=64))
        result_state = metadata.get("last_moondream_result_state")
        if result_state not in (None, ""):
            moondream_result_states.append(_safe_token(result_state, default="", limit=64))
        try:
            object_showing_likelihood_max = max(
                object_showing_likelihood_max,
                float(metadata.get("object_showing_likelihood") or 0.0),
            )
        except (TypeError, ValueError):
            pass
        if transition == "vision_answered":
            state = "fresh_used"
            used = True
            fresh = True
        elif transition == "vision_stale":
            if state != "fresh_used":
                state = "stale_or_limited"
        elif transition == "vision_unavailable":
            if state != "fresh_used":
                state = "error"
        elif transition in {"looking_requested", "frame_captured", "camera_ready"}:
            if state == "not_used":
                state = "requested" if transition == "looking_requested" else "available"
        if "vision.fetch_user_image_success" in text or "vision_fetch_user_image_success" in text:
            state = "fresh_used"
            used = True
            fresh = True
        elif "vision.fetch_user_image_error" in text or "camera:error" in text:
            if state != "fresh_used":
                state = "error"
        elif camera_related and (
            "stale" in text or "limited" in text or "permission_denied" in text
        ):
            if state not in {"fresh_used", "error"}:
                state = "stale_or_limited"
        elif "vision.fetch_user_image_requested" in text or "vision.fetch_user_image_start" in text:
            if state == "not_used":
                state = "requested"
    return {
        "state": state,
        "used": used,
        "fresh": fresh,
        "scene_transitions": list(_dedupe(transitions)),
        "camera_honesty_states": list(_dedupe(honesty_states)),
        "freshness_states": list(_dedupe(freshness_states)),
        "user_presence_hints": list(_dedupe(user_presence_hints)),
        "last_moondream_result_states": list(_dedupe(moondream_result_states)),
        "object_showing_likelihood_max": round(
            max(0.0, min(1.0, object_showing_likelihood_max)),
            3,
        ),
    }


def _segment_interruption(events: list[Mapping[str, object]]) -> dict[str, object]:
    outcome = "none"
    protected = False
    for event in events:
        event_type = _safe_token(event.get("event_type"), default="", limit=80)
        text = _combined_event_text(event)
        protected = protected or "protected" in text or "suppressed" in text
        if event_type == "interruption_recovered" or "listening_resumed" in text:
            outcome = "recovered"
        elif event_type == "output_flushed" or "output_flushed" in text:
            if outcome != "recovered":
                outcome = "flushed"
        elif event_type == "interruption_accepted" or "accepted" in text:
            if outcome not in {"recovered", "flushed"}:
                outcome = "accepted"
        elif event_type == "interruption_rejected" or "rejected" in text:
            if outcome == "none":
                outcome = "rejected"
    return {"outcome": outcome, "protected": protected}


def _segment_speech(events: list[Mapping[str, object]]) -> dict[str, object]:
    chunk_count = 0
    subtitle_count = 0
    tts_start_count = 0
    estimated_duration_ms_total = 0
    estimated_duration_ms_max = 0
    subtitle_timing_policies: list[str] = []
    backend_capability_labels: list[str] = []
    stale_generation_token_count = 0
    for event in events:
        text = _combined_event_text(event)
        metadata = event.get("metadata")
        if "chunk" in text or (isinstance(metadata, Mapping) and "chunk_index" in metadata):
            chunk_count += 1
        if "subtitle" in text:
            subtitle_count += 1
        if "tts.speech_start" in text or "speech.audio_start" in text:
            tts_start_count += 1
        if not isinstance(metadata, Mapping):
            continue
        estimated_duration_ms = _safe_int(metadata.get("estimated_duration_ms"))
        estimated_duration_ms_total += estimated_duration_ms
        estimated_duration_ms_max = max(estimated_duration_ms_max, estimated_duration_ms)
        subtitle_timing = metadata.get("subtitle_timing")
        if isinstance(subtitle_timing, Mapping):
            subtitle_timing_policies.append(
                _safe_token(subtitle_timing.get("emit_policy"), default="", limit=80)
            )
        if metadata.get("stale_generation_token") not in (None, ""):
            stale_generation_token_count += 1
        backend_capabilities = metadata.get("backend_capabilities")
        if isinstance(backend_capabilities, Mapping):
            backend_capability_labels.extend(_speech_capability_labels(backend_capabilities))
    return {
        "chunk_count": chunk_count,
        "subtitle_count": subtitle_count,
        "tts_start_count": tts_start_count,
        "estimated_duration_ms_total": estimated_duration_ms_total,
        "estimated_duration_ms_max": estimated_duration_ms_max,
        "subtitle_timing_policies": list(_dedupe(subtitle_timing_policies)),
        "stale_generation_token_count": stale_generation_token_count,
        "backend_capability_labels": list(_dedupe(backend_capability_labels)),
    }


def _segment_memory(events: list[Mapping[str, object]]) -> dict[str, object]:
    labels = _labels_from_reasons(events, prefixes=("memory", "memory_persona"))
    used = any(_safe_token(event.get("event_type"), default="") == "memory_used" for event in events)
    return {"used": used or bool(labels), "effect_labels": list(labels)}


def _segment_persona(events: list[Mapping[str, object]]) -> dict[str, object]:
    labels = _labels_from_reasons(events, prefixes=("persona", "memory_persona"))
    planned = any(
        _safe_token(event.get("event_type"), default="") == "persona_plan_compiled"
        for event in events
    )
    return {"planned": planned or bool(labels), "effect_labels": list(labels)}


def _segment_active_listening(events: list[Mapping[str, object]]) -> dict[str, object]:
    states: list[str] = []
    semantic_intents: list[str] = []
    listener_chip_ids: list[str] = []
    summary_hashes: list[str] = []
    camera_reference_states: list[str] = []
    partial_count = 0
    final_count = 0
    constraint_count = 0
    enough_information_to_answer = False
    for event in events:
        event_type = _safe_token(event.get("event_type"), default="", limit=80)
        text = _combined_event_text(event)
        metadata = event.get("metadata")
        if event_type in {"partial_heard", "partial_understanding_updated"}:
            partial_count += 1
        if event_type in {"final_heard", "final_understanding_ready"}:
            final_count += 1
        if "constraint" in text:
            constraint_count += 1
        if isinstance(metadata, Mapping):
            for key in ("phase", "state", "active_listening_state"):
                if metadata.get(key) not in (None, ""):
                    states.append(_safe_token(metadata.get(key), default="unknown", limit=64))
            semantic = metadata.get("semantic_listener")
            if isinstance(semantic, Mapping):
                semantic_intents.append(
                    _safe_token(semantic.get("detected_intent"), default="unknown", limit=64)
                )
                camera_reference_states.append(
                    _safe_token(
                        semantic.get("camera_reference_state"),
                        default="not_used",
                        limit=64,
                    )
                )
                if semantic.get("enough_information_to_answer") is True:
                    enough_information_to_answer = True
                chips = semantic.get("listener_chip_ids")
                if isinstance(chips, list):
                    listener_chip_ids.extend(
                        _safe_token(chip, default="", limit=64) for chip in chips[:8]
                    )
                summary_hash = _safe_optional_token(semantic.get("summary_hash"), limit=32)
                if summary_hash:
                    summary_hashes.append(summary_hash)
            direct_semantic = metadata.get("semantic_state_v3")
            if isinstance(direct_semantic, Mapping):
                semantic_intents.append(
                    _safe_token(
                        direct_semantic.get("detected_intent"),
                        default="unknown",
                        limit=64,
                    )
                )
                camera_reference_states.append(
                    _safe_token(
                        direct_semantic.get("camera_reference_state"),
                        default="not_used",
                        limit=64,
                    )
                )
                if direct_semantic.get("enough_information_to_answer") is True:
                    enough_information_to_answer = True
                safe_summary = direct_semantic.get("safe_live_summary")
                summary_hash = _hash_ref(safe_summary)
                if summary_hash:
                    summary_hashes.append(summary_hash)
                chips = direct_semantic.get("listener_chips")
                if isinstance(chips, list):
                    for chip in chips[:8]:
                        if isinstance(chip, Mapping):
                            listener_chip_ids.append(
                                _safe_token(chip.get("chip_id"), default="", limit=64)
                            )
    return {
        "states": list(_dedupe(states)),
        "partial_count": partial_count,
        "final_count": final_count,
        "constraint_count": constraint_count,
        "semantic_intents": list(_dedupe(semantic_intents)),
        "listener_chip_ids": list(_dedupe(listener_chip_ids)),
        "summary_hashes": list(_dedupe(summary_hashes)),
        "camera_reference_states": list(_dedupe(camera_reference_states)),
        "enough_information_to_answer": enough_information_to_answer,
    }


def _segment_performance_plan(events: list[Mapping[str, object]]) -> dict[str, object]:
    plan_ids: list[str] = []
    stances: list[str] = []
    response_shapes: list[str] = []
    voice_pacing_states: list[str] = []
    speech_budget_states: list[str] = []
    camera_policy_states: list[str] = []
    memory_policy_states: list[str] = []
    memory_ids: list[str] = []
    selected_memory_refs: list[dict[str, object]] = []
    discourse_episode_ids: list[str] = []
    discourse_category_labels: list[str] = []
    memory_effect_labels: list[str] = []
    memory_conflict_labels: list[str] = []
    memory_staleness_labels: list[str] = []
    interruption_policy_states: list[str] = []
    repair_policy_states: list[str] = []
    persona_anchor_ids: list[str] = []
    persona_anchor_situation_keys: list[str] = []
    summary_hashes: list[str] = []
    summaries: list[str] = []
    max_chunk_hard_limit = 0
    max_chunk_target = 0
    for event in events:
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("performance_plan_v3_schema_version") == 3:
            stances.append(_safe_token(metadata.get("stance"), default="unknown", limit=96))
            response_shapes.append(
                _safe_token(metadata.get("response_shape"), default="unknown", limit=96)
            )
            for anchor_id in _safe_sequence(metadata.get("persona_anchor_ids_v3")):
                safe_anchor_id = _safe_optional_token(anchor_id, limit=96)
                if safe_anchor_id:
                    persona_anchor_ids.append(safe_anchor_id)
            for situation_key in _safe_sequence(
                metadata.get("persona_anchor_situation_keys_v3")
            ):
                safe_situation_key = _safe_optional_token(situation_key, limit=80)
                if safe_situation_key:
                    persona_anchor_situation_keys.append(safe_situation_key)
            summary_hash = _safe_optional_token(metadata.get("plan_summary_hash"), limit=32)
            if summary_hash:
                summary_hashes.append(summary_hash)
            for memory_id in _safe_sequence(metadata.get("selected_memory_ids")):
                safe_memory_id = _safe_optional_token(memory_id, limit=160)
                if safe_memory_id:
                    memory_ids.append(safe_memory_id)
            for episode_id in _safe_sequence(metadata.get("discourse_episode_ids")):
                safe_episode_id = _safe_optional_token(episode_id, limit=120)
                if safe_episode_id:
                    discourse_episode_ids.append(safe_episode_id)
            for label in _safe_sequence(metadata.get("discourse_category_labels")):
                safe_label = _safe_optional_token(label, limit=96)
                if safe_label:
                    discourse_category_labels.append(safe_label)
            for label in _safe_sequence(metadata.get("memory_effect_labels")):
                safe_label = _safe_optional_token(label, limit=96)
                if safe_label:
                    memory_effect_labels.append(safe_label)
            for label in _safe_sequence(metadata.get("memory_conflict_labels")):
                safe_label = _safe_optional_token(label, limit=96)
                if safe_label:
                    memory_conflict_labels.append(safe_label)
            for label in _safe_sequence(metadata.get("memory_staleness_labels")):
                safe_label = _safe_optional_token(label, limit=96)
                if safe_label:
                    memory_staleness_labels.append(safe_label)
        plan_payload = metadata.get("performance_plan_v3")
        if not isinstance(plan_payload, Mapping):
            continue
        plan_id = _safe_optional_token(plan_payload.get("plan_id"), limit=96)
        if plan_id:
            plan_ids.append(plan_id)
        stances.append(_safe_token(plan_payload.get("stance"), default="unknown", limit=96))
        response_shapes.append(
            _safe_token(plan_payload.get("response_shape"), default="unknown", limit=96)
        )
        for anchor in _safe_sequence(plan_payload.get("persona_anchor_refs_v3")):
            if not isinstance(anchor, Mapping):
                continue
            anchor_id = _safe_optional_token(anchor.get("anchor_id"), limit=96)
            situation_key = _safe_optional_token(anchor.get("situation_key"), limit=80)
            if anchor_id:
                persona_anchor_ids.append(anchor_id)
            if situation_key:
                persona_anchor_situation_keys.append(situation_key)
        summary = _safe_label(plan_payload.get("plan_summary"), default="", limit=180)
        if summary:
            summaries.append(summary)
            summary_hash = _hash_ref(summary)
            if summary_hash:
                summary_hashes.append(summary_hash)
        for key, output in (
            ("voice_pacing", voice_pacing_states),
            ("speech_chunk_budget", speech_budget_states),
            ("camera_reference_policy", camera_policy_states),
            ("memory_callback_policy", memory_policy_states),
            ("interruption_policy", interruption_policy_states),
            ("repair_policy", repair_policy_states),
        ):
            policy = plan_payload.get(key)
            if isinstance(policy, Mapping):
                output.append(_safe_token(policy.get("state"), default="unknown", limit=96))
                if key == "memory_callback_policy":
                    for item in _safe_sequence(policy.get("selected_memory_refs")):
                        if not isinstance(item, Mapping):
                            continue
                        memory_id = _safe_optional_token(item.get("memory_id"), limit=160)
                        if not memory_id:
                            continue
                        selected_memory_refs.append(
                            {
                                "memory_id": memory_id,
                                "display_kind": _safe_token(
                                    item.get("display_kind"),
                                    default="memory",
                                    limit=64,
                                ),
                                "summary": _safe_label(
                                    item.get("summary"),
                                    default="Memory selected.",
                                    limit=120,
                                ),
                                "source_language": _safe_token(
                                    item.get("source_language"),
                                    default="unknown",
                                    limit=32,
                                ),
                                "cross_language": item.get("cross_language") is True,
                                "effect_labels": [
                                    _safe_token(label, default="", limit=96)
                                    for label in list(_safe_sequence(item.get("effect_labels")))[:8]
                                    if _safe_token(label, default="", limit=96)
                                ],
                                "confidence_bucket": _safe_token(
                                    item.get("confidence_bucket"),
                                    default="medium",
                                    limit=32,
                                ),
                                "reason_codes": [
                                    _safe_token(code, default="", limit=96)
                                    for code in list(_safe_sequence(item.get("reason_codes")))[:12]
                                    if _safe_token(code, default="", limit=96)
                                ],
                            }
                        )
                    for memory_id in _safe_sequence(policy.get("selected_memory_ids")):
                        safe_memory_id = _safe_optional_token(memory_id, limit=160)
                        if safe_memory_id:
                            memory_ids.append(safe_memory_id)
                    for episode_id in _safe_sequence(policy.get("discourse_episode_ids")):
                        safe_episode_id = _safe_optional_token(episode_id, limit=120)
                        if safe_episode_id:
                            discourse_episode_ids.append(safe_episode_id)
                    for label in _safe_sequence(policy.get("discourse_category_labels")):
                        safe_label = _safe_optional_token(label, limit=96)
                        if safe_label:
                            discourse_category_labels.append(safe_label)
                    for label in _safe_sequence(policy.get("effect_labels")):
                        safe_label = _safe_optional_token(label, limit=96)
                        if safe_label:
                            memory_effect_labels.append(safe_label)
                    for label in _safe_sequence(policy.get("conflict_labels")):
                        safe_label = _safe_optional_token(label, limit=96)
                        if safe_label:
                            memory_conflict_labels.append(safe_label)
                    for label in _safe_sequence(policy.get("staleness_labels")):
                        safe_label = _safe_optional_token(label, limit=96)
                        if safe_label:
                            memory_staleness_labels.append(safe_label)
                if key == "speech_chunk_budget":
                    max_chunk_hard_limit = max(
                        max_chunk_hard_limit,
                        _safe_int(policy.get("hard_max_chars")),
                    )
                    max_chunk_target = max(
                        max_chunk_target,
                        _safe_int(policy.get("target_chars")),
                    )
    return {
        "plan_ids": list(_dedupe(plan_ids)),
        "stances": list(_dedupe(stances)),
        "response_shapes": list(_dedupe(response_shapes)),
        "voice_pacing_states": list(_dedupe(voice_pacing_states)),
        "speech_budget_states": list(_dedupe(speech_budget_states)),
        "camera_policy_states": list(_dedupe(camera_policy_states)),
        "memory_policy_states": list(_dedupe(memory_policy_states)),
        "memory_ids": list(_dedupe(memory_ids)),
        "selected_memory_refs": list(
            {str(ref.get("memory_id")): ref for ref in selected_memory_refs}.values()
        )[:8],
        "discourse_episode_ids": list(_dedupe(discourse_episode_ids)),
        "discourse_category_labels": list(_dedupe(discourse_category_labels)),
        "memory_effect_labels": list(_dedupe(memory_effect_labels)),
        "memory_conflict_labels": list(_dedupe(memory_conflict_labels)),
        "memory_staleness_labels": list(_dedupe(memory_staleness_labels)),
        "interruption_policy_states": list(_dedupe(interruption_policy_states)),
        "repair_policy_states": list(_dedupe(repair_policy_states)),
        "persona_anchor_ids_v3": list(_dedupe(persona_anchor_ids)),
        "persona_anchor_situation_keys_v3": list(_dedupe(persona_anchor_situation_keys)),
        "plan_summary_hashes": list(_dedupe(summary_hashes)),
        "plan_summaries": list(dict.fromkeys(summaries[:3])),
        "max_chunk_hard_limit": max_chunk_hard_limit,
        "max_chunk_target": max_chunk_target,
    }


def _segment_failure_labels(events: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    labels: list[str] = []
    for event in events:
        event_type = _safe_token(event.get("event_type"), default="", limit=80)
        text = _combined_event_text(event)
        if event_type == "error":
            labels.append("error")
        if event_type == "degraded":
            labels.append("degraded")
        if "camera" in text and ("stale" in text or "permission_denied" in text):
            labels.append("camera:stale_or_limited")
        if "vision" in text and "error" in text:
            labels.append("vision:error")
        if "self_interruption" in text:
            labels.append("interruption:self_interruption")
        if "unsupported_tts" in text:
            labels.append("tts:unsupported_control_claim")
    return _dedupe(labels)


def _build_segment(
    *,
    segment_index: int,
    segment_type: str,
    turn_index: int,
    subturn_index: int,
    events: list[dict[str, object]],
    relative_times: list[int],
    fallback_event_id: int,
) -> PerformanceEpisodeSegmentV3:
    event_ids = tuple(
        _event_id(event.get("event_id"), fallback_event_id + index)
        for index, event in enumerate(events)
    )
    event_type_counts = Counter(
        _safe_token(event.get("event_type"), default="waiting", limit=80) for event in events
    )
    return PerformanceEpisodeSegmentV3(
        segment_id=f"seg_{segment_index:04d}",
        segment_type=segment_type,
        turn_index=turn_index,
        subturn_index=subturn_index,
        start_ms=min(relative_times) if relative_times else 0,
        end_ms=max(relative_times) if relative_times else 0,
        event_ids=event_ids,
        event_type_counts=dict(event_type_counts),
        reason_codes=_event_reason_codes(events),
        floor=_segment_floor(events),
        camera=_segment_camera(events),
        interruption=_segment_interruption(events),
        speech=_segment_speech(events),
        memory=_segment_memory(events),
        persona=_segment_persona(events),
        active_listening=_segment_active_listening(events),
        performance_plan=_segment_performance_plan(events),
    )


def _compile_segments(
    events: list[dict[str, object]],
    relative_times: list[int],
) -> tuple[PerformanceEpisodeSegmentV3, ...]:
    if not events:
        return ()
    grouped: list[tuple[str, list[dict[str, object]], list[int]]] = []
    current_type = _segment_type_for_event(events[0])
    current_events: list[dict[str, object]] = []
    current_times: list[int] = []
    for event, relative_time in zip(events, relative_times, strict=True):
        segment_type = _segment_type_for_event(event)
        if current_events and segment_type != current_type:
            grouped.append((current_type, current_events, current_times))
            current_events = []
            current_times = []
        current_type = segment_type
        current_events.append(event)
        current_times.append(relative_time)
    if current_events:
        grouped.append((current_type, current_events, current_times))

    segments: list[PerformanceEpisodeSegmentV3] = []
    turn_index = 1
    subturn_index = 0
    seen_listen = False
    previous_type: str | None = None
    for index, (segment_type, segment_events, segment_times) in enumerate(grouped, start=1):
        if segment_type == "listen_segment":
            if seen_listen and previous_type in {
                "idle_segment",
                "speak_segment",
                "repair_segment",
            }:
                turn_index += 1
                subturn_index = 0
            seen_listen = True
        subturn_index += 1
        segment = _build_segment(
            segment_index=index,
            segment_type=segment_type,
            turn_index=turn_index,
            subturn_index=subturn_index,
            events=segment_events,
            relative_times=segment_times,
            fallback_event_id=index,
        )
        segments.append(segment)
        previous_type = segment_type
    return tuple(segments)


def _episode_metrics(
    *,
    events: list[dict[str, object]],
    segments: tuple[PerformanceEpisodeSegmentV3, ...],
    failure_labels: tuple[str, ...],
    sanitizer: PerformanceEpisodeSanitizerReport,
) -> dict[str, object]:
    durations = {segment_type: 0 for segment_type in PERFORMANCE_EPISODE_V3_SEGMENT_TYPES}
    for segment in segments:
        durations[segment.segment_type] = durations.get(segment.segment_type, 0) + segment.duration_ms
    camera_states = _dedupe(segment.camera.get("state") for segment in segments)
    interruption_outcomes = _dedupe(segment.interruption.get("outcome") for segment in segments)
    memory_used = any(segment.memory.get("used") is True for segment in segments)
    persona_planned = any(segment.persona.get("planned") is True for segment in segments)
    return {
        "event_count": len(events),
        "segment_count": len(segments),
        "duration_ms": max((segment.end_ms for segment in segments), default=0),
        "turn_count": max((segment.turn_index for segment in segments), default=0),
        "subturn_count": sum(1 for _segment in segments),
        "listen_ms": durations.get("listen_segment", 0),
        "think_ms": durations.get("think_segment", 0),
        "look_ms": durations.get("look_segment", 0),
        "speak_ms": durations.get("speak_segment", 0),
        "overlap_ms": durations.get("overlap_segment", 0),
        "repair_ms": durations.get("repair_segment", 0),
        "idle_ms": durations.get("idle_segment", 0),
        "camera_use_state": next(iter(camera_states), "not_used"),
        "interruption_outcome": next(iter(interruption_outcomes), "none"),
        "memory_persona_effect": (
            "memory_and_persona"
            if memory_used and persona_planned
            else "memory"
            if memory_used
            else "persona"
            if persona_planned
            else "none"
        ),
        "failure_label_count": len(failure_labels),
        "sanitizer_passed": sanitizer.passed,
    }


def compile_performance_episode_v3(
    actor_events: Iterable[ActorEventV2 | Mapping[str, object]],
    *,
    episode_id: str | None = None,
    privacy_level: str = PERFORMANCE_EPISODE_V3_PRIVACY_LEVEL,
) -> PerformanceEpisodeV3:
    """Compile actor events into a persistent public-safe episode ledger."""
    sanitized_events: list[dict[str, object]] = []
    sanitizer = PerformanceEpisodeSanitizerReport()
    for event in actor_events:
        payload = _event_payload(event)
        sanitized, report = sanitize_actor_event_for_performance_episode(payload)
        sanitizer = sanitizer.combine(report)
        if sanitized.get("schema_version") == 2:
            sanitized_events.append(sanitized)
    relative_times, created_at_ms, timestamp_failures = _relative_event_times(sanitized_events)
    segments = _compile_segments(sanitized_events, relative_times)
    failure_labels = _dedupe(
        [
            *timestamp_failures,
            *(label for event in sanitized_events for label in _segment_failure_labels([event])),
            *(
                "sanitizer:blocked_payload"
                for _reason in sanitizer.reason_codes
                if not sanitizer.passed
            ),
        ],
        limit=96,
    )
    first = sanitized_events[0] if sanitized_events else {}
    profile = _safe_token(first.get("profile"), default="manual", limit=96)
    language = _safe_token(first.get("language"), default="unknown", limit=32)
    tts_label = _safe_label(first.get("tts_label"), default="unknown", limit=80)
    first_event_id = _event_id(first.get("event_id"), 1) if first else 0
    last_event_id = _event_id(sanitized_events[-1].get("event_id"), first_event_id) if first else 0
    session_id_hash = _hash_ref(first.get("session_id"))
    client_id_hash = _hash_ref(first.get("client_id"))
    resolved_episode_id = episode_id or _safe_token(
        "pe_v3_"
        + hashlib.sha256(
            "|".join(
                [
                    profile,
                    language,
                    tts_label,
                    str(session_id_hash or "no_session"),
                    str(first_event_id),
                    str(last_event_id),
                    str(len(sanitized_events)),
                ]
            ).encode("utf-8")
        ).hexdigest()[:20],
        default="pe_v3",
        limit=80,
    )
    metrics = _episode_metrics(
        events=sanitized_events,
        segments=segments,
        failure_labels=failure_labels,
        sanitizer=sanitizer,
    )
    return PerformanceEpisodeV3(
        episode_id=resolved_episode_id,
        profile=profile,
        language=language,
        tts_runtime_label=tts_label,
        privacy_level=privacy_level,
        created_at_ms=created_at_ms,
        session_id_hash=session_id_hash,
        client_id_hash=client_id_hash,
        segments=segments,
        metrics=metrics,
        failure_labels=failure_labels,
        sanitizer=sanitizer,
    )


def render_performance_episode_v3_jsonl(episodes: Iterable[PerformanceEpisodeV3]) -> str:
    """Render episodes as deterministic JSONL."""
    lines = [
        json.dumps(episode.as_dict(), ensure_ascii=False, sort_keys=True)
        for episode in episodes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def load_actor_events_from_jsonl(
    path: Path | str,
) -> tuple[tuple[dict[str, object], ...], PerformanceEpisodeSanitizerReport]:
    """Load and sanitize actor-event JSONL for offline episode conversion."""
    events: list[dict[str, object]] = []
    sanitizer = PerformanceEpisodeSanitizerReport()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                sanitizer = sanitizer.combine(
                    PerformanceEpisodeSanitizerReport(
                        passed=False,
                        reason_codes=(f"performance_episode:malformed_json_line_{line_number}",),
                    )
                )
                continue
            sanitized, report = sanitize_actor_event_for_performance_episode(payload)
            sanitizer = sanitizer.combine(report)
            if sanitized.get("schema_version") == 2:
                events.append(sanitized)
    return tuple(events), sanitizer


def compile_performance_episode_v3_from_actor_trace(path: Path | str) -> PerformanceEpisodeV3:
    """Compile one actor trace JSONL file into one episode."""
    events, sanitizer = load_actor_events_from_jsonl(path)
    episode = compile_performance_episode_v3(events)
    return replace(
        episode,
        sanitizer=episode.sanitizer.combine(sanitizer),
        failure_labels=_dedupe(
            [
                *episode.failure_labels,
                *(
                    "sanitizer:blocked_payload"
                    for _reason in sanitizer.reason_codes
                    if not sanitizer.passed
                ),
            ],
            limit=96,
        ),
    )


def _load_episode_jsonl(
    path: Path | str,
) -> tuple[tuple[dict[str, object], ...], PerformanceEpisodeSanitizerReport]:
    episodes: list[dict[str, object]] = []
    sanitizer = PerformanceEpisodeSanitizerReport()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                sanitizer = sanitizer.combine(
                    PerformanceEpisodeSanitizerReport(
                        passed=False,
                        reason_codes=(f"performance_episode:malformed_json_line_{line_number}",),
                    )
                )
                continue
            sanitized, report = sanitize_performance_episode_payload(
                payload,
                allowed_top_level_keys=_ALLOWED_EPISODE_TOP_LEVEL_KEYS,
            )
            if sanitized.get("schema_version") != PERFORMANCE_EPISODE_V3_SCHEMA_VERSION:
                report = report.combine(
                    PerformanceEpisodeSanitizerReport(
                        passed=False,
                        reason_codes=("performance_episode:unsupported_episode_schema",),
                    )
                )
            sanitizer = sanitizer.combine(report)
            if sanitized.get("schema_version") == PERFORMANCE_EPISODE_V3_SCHEMA_VERSION:
                episodes.append(sanitized)
    return tuple(episodes), sanitizer


def replay_performance_episode_v3_jsonl(path: Path | str) -> PerformanceEpisodeReplaySummary:
    """Replay performance episode JSONL into a public-safe summary."""
    episodes, sanitizer = _load_episode_jsonl(path)
    segment_counts: Counter[str] = Counter()
    profiles: set[str] = set()
    languages: set[str] = set()
    tts_labels: set[str] = set()
    timeline: list[dict[str, object]] = []
    floor_states: list[str] = []
    floor_sub_states: list[str] = []
    camera_states: list[str] = []
    interruption_outcomes: list[str] = []
    memory_persona_effects: list[str] = []
    failure_labels: list[str] = []
    latency_totals = {segment_type: 0 for segment_type in PERFORMANCE_EPISODE_V3_SEGMENT_TYPES}
    duration_ms = 0
    for episode in episodes:
        episode_id = _safe_token(episode.get("episode_id"), default="episode", limit=96)
        profiles.add(_safe_token(episode.get("profile"), default="manual", limit=96))
        languages.add(_safe_token(episode.get("language"), default="unknown", limit=32))
        tts_labels.add(_safe_label(episode.get("tts_runtime_label"), default="unknown", limit=80))
        failure_labels.extend(episode.get("failure_labels", []) if isinstance(episode.get("failure_labels"), list) else [])
        metrics = episode.get("metrics")
        if isinstance(metrics, Mapping):
            duration_ms = max(duration_ms, int(metrics.get("duration_ms") or 0))
            effect = _safe_token(metrics.get("memory_persona_effect"), default="", limit=80)
            if effect:
                memory_persona_effects.append(effect)
        segments = episode.get("segments")
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            segment_type = _safe_token(segment.get("segment_type"), default="idle_segment", limit=80)
            segment_counts[segment_type] += 1
            latency_totals[segment_type] = latency_totals.get(segment_type, 0) + int(
                segment.get("duration_ms") or 0
            )
            floor = segment.get("floor")
            if isinstance(floor, Mapping):
                states = floor.get("states")
                if isinstance(states, list):
                    floor_states.extend(states)
                sub_states = floor.get("sub_states")
                if isinstance(sub_states, list):
                    floor_sub_states.extend(sub_states)
            camera = segment.get("camera")
            if isinstance(camera, Mapping):
                camera_states.append(_safe_token(camera.get("state"), default="not_used"))
            interruption = segment.get("interruption")
            if isinstance(interruption, Mapping):
                interruption_outcomes.append(
                    _safe_token(interruption.get("outcome"), default="none")
                )
            timeline.append(
                {
                    "episode_id": episode_id,
                    "segment_id": _safe_token(segment.get("segment_id"), default="segment"),
                    "mode": _SEGMENT_MODE_MAP.get(segment_type, "waiting"),
                    "segment_type": segment_type,
                    "start_ms": int(segment.get("start_ms") or 0),
                    "end_ms": int(segment.get("end_ms") or 0),
                }
            )
    latency_metrics = {
        "duration_ms": duration_ms,
        **{
            f"{segment_type.replace('_segment', '')}_ms": latency_totals.get(segment_type, 0)
            for segment_type in PERFORMANCE_EPISODE_V3_SEGMENT_TYPES
        },
    }
    return PerformanceEpisodeReplaySummary(
        episode_count=len(episodes),
        segment_counts=dict(segment_counts),
        profiles=tuple(sorted(profiles)),
        languages=tuple(sorted(languages)),
        tts_labels=tuple(sorted(tts_labels)),
        mode_timeline=tuple(timeline),
        latency_metrics=latency_metrics,
        floor_states=_dedupe(floor_states),
        floor_sub_states=_dedupe(floor_sub_states),
        camera_use_states=_dedupe(camera_states),
        interruption_outcomes=_dedupe(interruption_outcomes),
        memory_persona_effects=_dedupe(memory_persona_effects),
        failure_labels=_dedupe(failure_labels, limit=96),
        sanitizer=sanitizer,
    )


class PerformanceEpisodeV3Writer:
    """Opt-in bounded JSONL writer for PerformanceEpisodeV3 ledgers."""

    def __init__(
        self,
        *,
        episode_dir: Path | str,
        profile: str,
        run_id: str,
        max_events: int = _MAX_EVENTS_PER_EPISODE,
    ):
        """Initialize the writer."""
        self._episode_dir = Path(episode_dir).expanduser()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_profile = _safe_token(profile, default="manual", limit=80)
        safe_run_id = _safe_token(run_id, default="run", limit=80)
        self._path = (
            self._episode_dir
            / f"performance-episode-v3-{stamp}-{safe_profile}-{safe_run_id}.jsonl"
        )
        self._max_events = max(1, int(max_events))
        self._events: list[dict[str, object]] = []
        self._written = 0
        self._lock = Lock()

    @property
    def path(self) -> Path:
        """Return the episode JSONL path."""
        return self._path

    @property
    def written_count(self) -> int:
        """Return written episode count."""
        with self._lock:
            return self._written

    def append(
        self,
        event: ActorEventV2 | Mapping[str, object],
        *,
        terminal_event_type: str | None = None,
    ) -> None:
        """Append one actor event and flush on configured terminal boundaries."""
        payload = _event_payload(event)
        with self._lock:
            if len(self._events) < self._max_events:
                self._events.append(payload)
            elif not self._events or self._events[-1].get("reason_codes") != ["episode.limit_reached"]:
                self._events.append(
                    {
                        "schema_version": 2,
                        "event_id": _event_id(payload.get("event_id"), len(self._events) + 1),
                        "event_type": "degraded",
                        "mode": "degraded",
                        "timestamp": payload.get("timestamp"),
                        "profile": payload.get("profile"),
                        "language": payload.get("language"),
                        "tts_backend": payload.get("tts_backend"),
                        "tts_label": payload.get("tts_label"),
                        "vision_backend": payload.get("vision_backend"),
                        "source": "performance_episode_v3",
                        "session_id": payload.get("session_id"),
                        "client_id": payload.get("client_id"),
                        "metadata": {"episode_event_limit": self._max_events},
                        "reason_codes": ["episode.limit_reached"],
                    }
                )
            should_flush = terminal_event_type in PERFORMANCE_EPISODE_V3_TERMINAL_EVENTS
        if should_flush:
            self.flush()

    def flush(self) -> Path | None:
        """Write the current buffered events as one episode JSONL object."""
        with self._lock:
            if not self._events:
                return None
            events = tuple(self._events)
            self._events = []
        episode = compile_performance_episode_v3(events)
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        with self._lock:
            self._written += 1
        return self._path

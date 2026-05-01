"""Public-safe discourse episodes derived from performance episodes."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Iterable, Mapping

DISCOURSE_EPISODE_V3_SCHEMA_VERSION = 3
DISCOURSE_EPISODE_V3_CATEGORY_LABELS = (
    "active_project",
    "user_preference",
    "unresolved_commitment",
    "correction",
    "visual_event",
    "repeated_frustration",
    "success_pattern",
)
DISCOURSE_EPISODE_V3_MEMORY_EFFECT_LABELS = (
    "shorter_explanation",
    "project_constraint_recall",
    "corrected_preference",
    "avoid_repetition",
    "visual_context_acknowledgement",
    "tentative_callback",
    "suppressed_stale_memory",
    "none",
)

_MAX_REFS = 8
_MAX_LABELS = 12
_MAX_REASONS = 24
_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9:_./-]+")
_UNSAFE_MARKERS = (
    "authorization",
    "bearer",
    "credential",
    "developer_message",
    "developer_prompt",
    "hidden_prompt",
    "raw_prompt",
    "secret",
    "system_message",
    "system_prompt",
    "traceback",
    "transcript",
    "api_key",
    "audio_bytes",
    "image_bytes",
    "raw_audio",
    "raw_image",
    "sdp_offer",
    "ice_candidate",
    "full_message",
    "memory_body",
    "model_message",
    "password",
    "token",
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_ZH_RE = re.compile(r"[\u3400-\u9fff]")


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _has_unsafe_marker(value: Any) -> bool:
    text = _normalized_text(value)
    if not text:
        return False
    lowered = text.lower()
    return _URL_RE.search(text) is not None or any(marker in lowered for marker in _UNSAFE_MARKERS)


def _safe_text(value: Any, *, limit: int = 120, fallback: str = "") -> str:
    text = _normalized_text(value)
    if not text:
        return fallback
    if _has_unsafe_marker(text):
        return fallback or "redacted"
    return text[:limit]


def _safe_id(value: Any, *, fallback: str = "", limit: int = 160) -> str:
    text = _normalized_text(value)
    if not text or _has_unsafe_marker(text):
        return fallback
    safe = _SAFE_ID_PATTERN.sub("_", text).strip("_")
    return safe[:limit] or fallback


def _safe_reason(value: Any, *, fallback: str = "unknown", limit: int = 96) -> str:
    text = _normalized_text(value).lower().replace(" ", "_")
    if not text or _has_unsafe_marker(text):
        return fallback
    safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:limit] or fallback


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return value is True


def _dedupe(values: Iterable[Any], *, limit: int = _MAX_LABELS) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        label = _safe_reason(value, fallback="")
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(label)
        if len(result) >= limit:
            break
    return tuple(result)


def _locale(value: Any) -> str:
    text = _safe_reason(value, fallback="unknown")
    if text.startswith("zh") or text.startswith("cmn"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return "unknown"


def _infer_language(text: str, *, fallback: str = "unknown") -> str:
    if _ZH_RE.search(text):
        return "zh"
    lowered = text.lower()
    if any(token in lowered for token in ("english", "kokoro", "browser", "webrtc")):
        return "en"
    if any(token in lowered for token in ("melo", "melotts", "local-http-wav")):
        return "zh"
    return fallback


def _safe_list(value: Any, *, limit: int = _MAX_LABELS) -> tuple[str, ...]:
    raw = value if isinstance(value, (list, tuple, set)) else ()
    return _dedupe(raw, limit=limit)


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_blob(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


@dataclass(frozen=True)
class DiscourseEpisodeMemoryRef:
    """One public-safe memory reference linked to a discourse episode."""

    memory_id: str
    display_kind: str
    summary: str
    source_language: str
    cross_language: bool
    effect_labels: tuple[str, ...]
    confidence_bucket: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize without raw memory bodies."""
        return {
            "memory_id": self.memory_id,
            "display_kind": self.display_kind,
            "summary": self.summary,
            "source_language": self.source_language,
            "cross_language": self.cross_language,
            "effect_labels": list(self.effect_labels),
            "confidence_bucket": self.confidence_bucket,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscourseEpisodeMemoryRef":
        """Hydrate one public-safe memory ref."""
        source_language = _locale(data.get("source_language"))
        return cls(
            memory_id=_safe_id(data.get("memory_id"), fallback="memory:unknown"),
            display_kind=_safe_reason(data.get("display_kind"), fallback="memory"),
            summary=_safe_text(data.get("summary"), limit=120, fallback="Memory selected."),
            source_language=source_language,
            cross_language=data.get("cross_language") is True,
            effect_labels=_valid_effects(data.get("effect_labels")),
            confidence_bucket=_confidence_bucket(data.get("confidence_bucket")),
            reason_codes=_dedupe(data.get("reason_codes") or (), limit=_MAX_REASONS),
        )


@dataclass(frozen=True)
class DiscourseEpisode:
    """Public-safe discourse episode for memory continuity v3."""

    schema_version: int
    discourse_episode_id: str
    source_performance_episode_id: str
    source_event_ids: tuple[int, ...]
    profile: str
    language: str
    tts_runtime_label: str
    created_at_ms: int
    category_labels: tuple[str, ...]
    public_summary: str
    memory_refs: tuple[DiscourseEpisodeMemoryRef, ...]
    effect_labels: tuple[str, ...]
    conflict_labels: tuple[str, ...]
    staleness_labels: tuple[str, ...]
    confidence_bucket: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this discourse episode for public traces and brain events."""
        return {
            "schema_version": self.schema_version,
            "discourse_episode_id": self.discourse_episode_id,
            "source_performance_episode_id": self.source_performance_episode_id,
            "source_event_ids": list(self.source_event_ids),
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "created_at_ms": self.created_at_ms,
            "category_labels": list(self.category_labels),
            "public_summary": self.public_summary,
            "memory_refs": [ref.as_dict() for ref in self.memory_refs],
            "effect_labels": list(self.effect_labels),
            "conflict_labels": list(self.conflict_labels),
            "staleness_labels": list(self.staleness_labels),
            "confidence_bucket": self.confidence_bucket,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscourseEpisode":
        """Hydrate a public-safe discourse episode."""
        memory_refs = tuple(
            DiscourseEpisodeMemoryRef.from_dict(item)
            for item in data.get("memory_refs", ())
            if isinstance(item, Mapping)
        )[:_MAX_REFS]
        categories = _valid_categories(data.get("category_labels"))
        effects = _valid_effects(data.get("effect_labels"))
        return cls(
            schema_version=_safe_int(
                data.get("schema_version"),
                default=DISCOURSE_EPISODE_V3_SCHEMA_VERSION,
            ),
            discourse_episode_id=_safe_id(
                data.get("discourse_episode_id"),
                fallback="discourse-episode-v3:unknown",
                limit=120,
            ),
            source_performance_episode_id=_safe_id(
                data.get("source_performance_episode_id"),
                fallback="episode:unknown",
                limit=120,
            ),
            source_event_ids=tuple(
                _safe_int(item) for item in list(data.get("source_event_ids") or ())[:24]
            ),
            profile=_safe_text(data.get("profile"), limit=96, fallback="manual"),
            language=_locale(data.get("language")),
            tts_runtime_label=_safe_text(data.get("tts_runtime_label"), limit=96, fallback="unknown"),
            created_at_ms=_safe_int(data.get("created_at_ms")),
            category_labels=categories or ("success_pattern",),
            public_summary=_safe_text(
                data.get("public_summary"),
                limit=180,
                fallback="Discourse episode derived from public interaction labels.",
            ),
            memory_refs=memory_refs,
            effect_labels=effects or ("none",),
            conflict_labels=_safe_list(data.get("conflict_labels"), limit=8),
            staleness_labels=_safe_list(data.get("staleness_labels"), limit=8),
            confidence_bucket=_confidence_bucket(data.get("confidence_bucket")),
            reason_codes=_dedupe(data.get("reason_codes") or (), limit=_MAX_REASONS),
        )


def _valid_categories(value: Any) -> tuple[str, ...]:
    allowed = set(DISCOURSE_EPISODE_V3_CATEGORY_LABELS)
    return tuple(label for label in _safe_list(value, limit=8) if label in allowed)


def _valid_effects(value: Any) -> tuple[str, ...]:
    allowed = set(DISCOURSE_EPISODE_V3_MEMORY_EFFECT_LABELS)
    labels = tuple(label for label in _safe_list(value, limit=8) if label in allowed)
    if labels and "none" in labels and len(labels) > 1:
        labels = tuple(label for label in labels if label != "none")
    return labels


def _confidence_bucket(value: Any) -> str:
    label = _safe_reason(value, fallback="")
    if label in {"high", "medium", "low", "uncertain"}:
        return label
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "medium"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    if confidence > 0:
        return "low"
    return "uncertain"


def _episode_payload(episode: Any) -> Mapping[str, Any]:
    if isinstance(episode, DiscourseEpisode):
        return episode.as_dict()
    as_dict = getattr(episode, "as_dict", None)
    if callable(as_dict):
        data = as_dict()
        return data if isinstance(data, Mapping) else {}
    return episode if isinstance(episode, Mapping) else {}


def _segment_labels(segment: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for key in (
        "reason_codes",
        "event_type_counts",
    ):
        value = segment.get(key)
        if isinstance(value, Mapping):
            labels.extend(value.keys())
        elif isinstance(value, (list, tuple, set)):
            labels.extend(value)
    for key in (
        "floor",
        "camera",
        "interruption",
        "speech",
        "memory",
        "persona",
        "active_listening",
        "performance_plan",
    ):
        value = segment.get(key)
        if not isinstance(value, Mapping):
            continue
        labels.extend(_nested_label_values(value))
    return _dedupe(labels, limit=64)


def _nested_label_values(value: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for key, raw in value.items():
        if isinstance(raw, str):
            labels.append(raw)
        elif isinstance(raw, (int, bool)):
            labels.append(f"{key}:{raw}")
        elif isinstance(raw, list):
            labels.extend(str(item) for item in raw[:16] if not isinstance(item, Mapping))
        elif isinstance(raw, Mapping):
            labels.extend(_nested_label_values(raw))
    return labels


def _memory_refs_from_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    target_language: str,
) -> tuple[DiscourseEpisodeMemoryRef, ...]:
    refs: dict[str, DiscourseEpisodeMemoryRef] = {}
    for segment in segments:
        performance_plan = _safe_mapping(segment.get("performance_plan"))
        memory_policy = _safe_mapping(performance_plan.get("memory_callback_policy"))
        selected_refs = memory_policy.get("selected_memory_refs")
        if isinstance(selected_refs, list):
            for item in selected_refs[:_MAX_REFS]:
                if not isinstance(item, Mapping):
                    continue
                ref = _memory_ref_from_policy(item, target_language=target_language)
                if ref.memory_id not in refs or refs[ref.memory_id].display_kind == "memory":
                    refs[ref.memory_id] = ref
        for memory_id in memory_policy.get("selected_memory_ids", ()) or ():
            safe_id = _safe_id(memory_id, fallback="")
            if safe_id and safe_id not in refs:
                refs[safe_id] = DiscourseEpisodeMemoryRef(
                    memory_id=safe_id,
                    display_kind="memory",
                    summary="Memory selected.",
                    source_language="unknown",
                    cross_language=False,
                    effect_labels=_valid_effects(memory_policy.get("effect_labels")) or ("none",),
                    confidence_bucket="medium",
                    reason_codes=("discourse_episode:memory_policy_ref",),
                )
        selected_refs = performance_plan.get("selected_memory_refs")
        if isinstance(selected_refs, list):
            for item in selected_refs[:_MAX_REFS]:
                if not isinstance(item, Mapping):
                    continue
                ref = _memory_ref_from_policy(item, target_language=target_language)
                if ref.memory_id not in refs or refs[ref.memory_id].display_kind == "memory":
                    refs[ref.memory_id] = ref
        summary_effects = _valid_effects(performance_plan.get("memory_effect_labels")) or ("none",)
        for memory_id in performance_plan.get("memory_ids", ()) or ():
            safe_id = _safe_id(memory_id, fallback="")
            if safe_id and safe_id not in refs:
                refs[safe_id] = DiscourseEpisodeMemoryRef(
                    memory_id=safe_id,
                    display_kind="memory",
                    summary="Memory selected.",
                    source_language="unknown",
                    cross_language=False,
                    effect_labels=summary_effects,
                    confidence_bucket="medium",
                    reason_codes=("discourse_episode:performance_plan_memory_ref",),
                )
    return tuple(refs.values())[:_MAX_REFS]


def _memory_ref_from_policy(
    item: Mapping[str, Any],
    *,
    target_language: str,
) -> DiscourseEpisodeMemoryRef:
    source_language = _locale(item.get("source_language"))
    if source_language == "unknown":
        source_language = _infer_language(_normalized_text(item.get("summary")), fallback="unknown")
    target = _locale(target_language)
    return DiscourseEpisodeMemoryRef(
        memory_id=_safe_id(item.get("memory_id"), fallback="memory:unknown"),
        display_kind=_safe_reason(item.get("display_kind"), fallback="memory"),
        summary=_safe_text(item.get("summary"), limit=120, fallback="Memory selected."),
        source_language=source_language,
        cross_language=source_language in {"zh", "en"} and target in {"zh", "en"} and source_language != target,
        effect_labels=_valid_effects(item.get("effect_labels")) or ("none",),
        confidence_bucket=_confidence_bucket(item.get("confidence_bucket")),
        reason_codes=_dedupe(item.get("reason_codes") or ("discourse_episode:memory_ref",)),
    )


def _labels_from_episode(payload: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    labels.extend(payload.get("failure_labels", ()) if isinstance(payload.get("failure_labels"), list) else ())
    metrics = _safe_mapping(payload.get("metrics"))
    labels.extend(f"{key}:{value}" for key, value in metrics.items() if isinstance(value, (str, int, bool)))
    segments = payload.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, Mapping):
                labels.extend(_segment_labels(segment))
    return _dedupe(labels, limit=96)


def _category_labels(
    *,
    labels: tuple[str, ...],
    segments: list[Mapping[str, Any]],
    failure_labels: tuple[str, ...],
) -> tuple[str, ...]:
    blob = _text_blob(*labels, *failure_labels)
    categories: list[str] = []
    if any(token in blob for token in ("project", "planning", "constraint", "plan_steps")):
        categories.append("active_project")
    if "preference" in blob or "display_kind:preference" in blob:
        categories.append("user_preference")
    if any(token in blob for token in ("commitment", "unresolved", "todo", "task")):
        categories.append("unresolved_commitment")
    if any(token in blob for token in ("correction", "repair", "corrected")):
        categories.append("correction")
    if any(token in blob for token in ("fresh_used", "vision_answered", "visual", "can_see_now")):
        categories.append("visual_event")
    if failure_labels or any(
        token in blob
        for token in (
            "frustration",
            "repeated",
            "avoid_repetition",
            "degraded",
            "protected_playback",
            "suppress_stale_callback",
            "suppressed_stale_memory",
        )
    ):
        categories.append("repeated_frustration")
    if not failure_labels and (
        any(
            _safe_reason(segment.get("segment_type"), fallback="") == "speak_segment"
            for segment in segments
        )
        or any(token in blob for token in ("speech.audio_start", "speech:audio_start"))
    ):
        categories.append("success_pattern")
    if not categories:
        categories.append("success_pattern")
    return tuple(label for label in _dedupe(categories, limit=8) if label in DISCOURSE_EPISODE_V3_CATEGORY_LABELS)


def _effect_labels(
    *,
    categories: tuple[str, ...],
    labels: tuple[str, ...],
    memory_refs: tuple[DiscourseEpisodeMemoryRef, ...],
    failure_labels: tuple[str, ...],
) -> tuple[str, ...]:
    blob = _text_blob(*labels, *failure_labels)
    effects: list[str] = [label for ref in memory_refs for label in ref.effect_labels]
    if any(token in blob for token in ("shorter", "concise", "target_chars", "chunk_budget")):
        effects.append("shorter_explanation")
    if "active_project" in categories:
        effects.append("project_constraint_recall")
    if "correction" in categories:
        effects.append("corrected_preference")
    if "repeated_frustration" in categories:
        effects.append("avoid_repetition")
    if "visual_event" in categories:
        effects.append("visual_context_acknowledgement")
    if any(token in blob for token in ("uncertain", "tentative", "stale_limited_context")):
        effects.append("tentative_callback")
    if any(token in blob for token in ("suppressed", "stale", "historical", "contradicted")):
        effects.append("suppressed_stale_memory")
    valid = _valid_effects(effects)
    return valid or ("none",)


def _conflict_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    blob = _text_blob(*labels)
    values: list[str] = []
    if "contradict" in blob:
        values.append("contradicted")
    if "supersed" in blob or "correction" in blob:
        values.append("superseded_by_correction")
    if "low_confidence" in blob or "uncertain" in blob:
        values.append("uncertain")
    return _dedupe(values, limit=8)


def _staleness_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    blob = _text_blob(*labels)
    values: list[str] = []
    if "stale" in blob:
        values.append("stale")
    if "historical" in blob:
        values.append("historical")
    if "suppressed" in blob:
        values.append("suppressed")
    return _dedupe(values, limit=8)


def _public_summary(*, categories: tuple[str, ...], effects: tuple[str, ...], language: str) -> str:
    primary = categories[0] if categories else "success_pattern"
    effect = next((item for item in effects if item != "none"), "none")
    if language == "zh":
        return _safe_text(f"本轮记忆线索：{primary}；行为影响：{effect}。", limit=180)
    return _safe_text(f"Discourse memory cue: {primary}; behavior effect: {effect}.", limit=180)


def _stable_id(payload: Mapping[str, Any], *, categories: tuple[str, ...], effects: tuple[str, ...]) -> str:
    digest = hashlib.sha256(
        repr(
            (
                payload.get("episode_id"),
                payload.get("profile"),
                payload.get("language"),
                payload.get("created_at_ms"),
                tuple(payload.get("failure_labels") or ()),
                categories,
                effects,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"discourse-episode-v3:{digest}"


def compile_discourse_episode_v3(
    performance_episode: Any,
) -> DiscourseEpisode:
    """Compile one public-safe discourse episode from a performance episode."""
    payload = _episode_payload(performance_episode)
    segments = [
        segment for segment in payload.get("segments", []) if isinstance(segment, Mapping)
    ]
    failure_labels = _safe_list(payload.get("failure_labels"), limit=24)
    labels = _labels_from_episode(payload)
    language = _locale(payload.get("language"))
    memory_refs = _memory_refs_from_segments(segments, target_language=language)
    categories = _category_labels(
        labels=labels,
        segments=segments,
        failure_labels=failure_labels,
    )
    effects = _effect_labels(
        categories=categories,
        labels=labels,
        memory_refs=memory_refs,
        failure_labels=failure_labels,
    )
    conflict_labels = _conflict_labels(labels)
    staleness_labels = _staleness_labels(labels)
    confidence = "high"
    if conflict_labels or staleness_labels or "tentative_callback" in effects:
        confidence = "medium"
    if failure_labels:
        confidence = "low"
    if any(_has_unsafe_marker(value) for value in labels):
        confidence = "uncertain"
    reason_codes = _dedupe(
        (
            "discourse_episode:v3",
            f"profile:{_safe_reason(payload.get('profile'), fallback='manual')}",
            f"language:{language}",
            *(f"discourse_category:{label}" for label in categories),
            *(f"memory_effect:{label}" for label in effects),
            *(f"conflict:{label}" for label in conflict_labels),
            *(f"staleness:{label}" for label in staleness_labels),
            *("discourse_episode:unsafe_value_omitted" for value in labels if _has_unsafe_marker(value)),
        ),
        limit=_MAX_REASONS,
    )
    return DiscourseEpisode(
        schema_version=DISCOURSE_EPISODE_V3_SCHEMA_VERSION,
        discourse_episode_id=_stable_id(payload, categories=categories, effects=effects),
        source_performance_episode_id=_safe_id(
            payload.get("episode_id"),
            fallback="episode:unknown",
            limit=120,
        ),
        source_event_ids=tuple(
            sorted(
                {
                    _safe_int(event_id)
                    for segment in segments
                    for event_id in (
                        segment.get("event_ids") if isinstance(segment.get("event_ids"), list) else []
                    )
                }
            )
        )[:24],
        profile=_safe_text(payload.get("profile"), limit=96, fallback="manual"),
        language=language,
        tts_runtime_label=_safe_text(
            payload.get("tts_runtime_label"),
            limit=96,
            fallback="unknown",
        ),
        created_at_ms=_safe_int(payload.get("created_at_ms")),
        category_labels=categories,
        public_summary=_public_summary(categories=categories, effects=effects, language=language),
        memory_refs=memory_refs,
        effect_labels=effects,
        conflict_labels=conflict_labels,
        staleness_labels=staleness_labels,
        confidence_bucket=confidence,
        reason_codes=reason_codes,
    )


def compile_discourse_episode_v3_from_actor_events(
    actor_events: Iterable[Any],
) -> DiscourseEpisode:
    """Compile one discourse episode from public-safe actor events."""
    from blink.interaction.performance_episode_v3 import compile_performance_episode_v3

    performance_episode = compile_performance_episode_v3(actor_events)
    return compile_discourse_episode_v3(performance_episode)


def public_actor_event_metadata_for_discourse_episode(
    episode: DiscourseEpisode | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return actor-event-safe counts and labels for one discourse episode."""
    if isinstance(episode, Mapping):
        episode_obj = DiscourseEpisode.from_dict(episode)
    else:
        episode_obj = episode
    if episode_obj is None:
        return {
            "discourse_episode_count": 0,
            "discourse_category_labels": [],
            "memory_effect_labels": [],
        }
    return {
        "discourse_episode_count": 1,
        "discourse_episode_ids": [episode_obj.discourse_episode_id],
        "discourse_category_labels": list(episode_obj.category_labels),
        "memory_effect_labels": list(episode_obj.effect_labels),
        "conflict_labels": list(episode_obj.conflict_labels),
        "staleness_labels": list(episode_obj.staleness_labels),
    }


class DiscourseEpisodeV3Collector:
    """Runtime collector that persists discourse episodes as brain events."""

    def __init__(
        self,
        *,
        runtime_resolver: Callable[[], Any],
        max_events: int = 400,
    ):
        """Initialize the in-memory collector."""
        self._runtime_resolver = runtime_resolver
        self._max_events = max(1, int(max_events))
        self._events: list[Any] = []
        self._persisted = 0
        self._lock = Lock()

    @property
    def persisted_count(self) -> int:
        """Return the number of persisted discourse episodes."""
        with self._lock:
            return self._persisted

    def append(self, event: Any, *, terminal_event_type: str | None = None) -> None:
        """Append one actor event and persist on terminal boundaries."""
        from blink.interaction.performance_episode_v3 import PERFORMANCE_EPISODE_V3_TERMINAL_EVENTS

        with self._lock:
            if len(self._events) < self._max_events:
                self._events.append(event)
            should_flush = terminal_event_type in PERFORMANCE_EPISODE_V3_TERMINAL_EVENTS
        if should_flush:
            self.flush()

    def flush(self) -> DiscourseEpisode | None:
        """Compile and persist the buffered discourse episode if runtime is active."""
        with self._lock:
            if not self._events:
                return None
            events = tuple(self._events)
            self._events = []
        runtime = self._runtime_resolver()
        if runtime is None:
            return None
        store = getattr(runtime, "store", None)
        session_resolver = getattr(runtime, "session_resolver", None)
        if store is None or not callable(session_resolver):
            return None
        try:
            episode = compile_discourse_episode_v3_from_actor_events(events)
            session_ids = session_resolver()
            writer = getattr(store, "append_discourse_episode", None)
            if not callable(writer):
                return episode
            persisted = writer(
                episode=episode,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="discourse_episode_v3",
            )
            with self._lock:
                self._persisted += 1
            return persisted
        except Exception:
            return None


def discourse_episode_counts_by_category(
    episodes: Iterable[DiscourseEpisode | Mapping[str, Any]],
) -> dict[str, int]:
    """Return compact category counts for public summaries."""
    counter: Counter[str] = Counter()
    for item in episodes:
        episode = item if isinstance(item, DiscourseEpisode) else DiscourseEpisode.from_dict(item)
        for label in episode.category_labels:
            counter[label] += 1
    return dict(sorted(counter.items()))


__all__ = [
    "DISCOURSE_EPISODE_V3_CATEGORY_LABELS",
    "DISCOURSE_EPISODE_V3_MEMORY_EFFECT_LABELS",
    "DISCOURSE_EPISODE_V3_SCHEMA_VERSION",
    "DiscourseEpisode",
    "DiscourseEpisodeMemoryRef",
    "DiscourseEpisodeV3Collector",
    "compile_discourse_episode_v3",
    "compile_discourse_episode_v3_from_actor_events",
    "discourse_episode_counts_by_category",
    "public_actor_event_metadata_for_discourse_episode",
]

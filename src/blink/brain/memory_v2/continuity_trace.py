"""Public-safe memory continuity traces for bilingual Blink replies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from blink.brain.memory_v2.discourse_episode import (
    DISCOURSE_EPISODE_V3_MEMORY_EFFECT_LABELS,
    DiscourseEpisode,
)
from blink.brain.memory_v2.use_trace import BrainMemoryUseTrace, BrainMemoryUseTraceRef

_SCHEMA_VERSION = 1
_MAX_SELECTED_REFS = 8
_MAX_SUPPRESSED_BUCKETS = 8
_MAX_ACTIONS = 8
_MAX_REASON_CODES = 24
_MAX_DISCOURSE_REFS = 8
_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9:_./-]+")
_ZH_RE = re.compile(r"[\u3400-\u9fff]")
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
)
_COMMAND_PHRASES = {
    "remember": {
        "zh": ("记住", "帮我记", "记一下", "记得", "remember"),
        "en": ("remember", "keep in mind", "save this", "make a note"),
    },
    "forget": {
        "zh": ("忘记", "别记", "不要记", "删掉记忆", "forget"),
        "en": ("forget", "delete that memory", "do not remember", "don't remember"),
    },
    "correct": {
        "zh": ("更正", "纠正", "改成", "其实", "不是", "correct"),
        "en": ("correct", "correction", "actually", "not that", "instead"),
    },
    "list_memory": {
        "zh": ("你记得什么", "你现在记得什么", "记忆里有什么", "what do you remember"),
        "en": ("what do you remember", "what do you know about me", "show my memories"),
    },
    "explain_answer": {
        "zh": ("为什么这样回答", "为什么你这么说", "为什么这样说", "why did you answer"),
        "en": ("why did you answer that way", "why did you say that", "why this answer"),
    },
}
_TERM_BRIDGE = (
    ("中文", "Chinese"),
    ("英文", "English"),
    ("浏览器", "browser"),
    ("相机", "camera"),
    ("摄像头", "camera"),
    ("视觉", "vision"),
    ("语音", "voice"),
    ("同等主要", "equal primary"),
    ("同样重要", "equally important"),
    ("主要路径", "primary path"),
    ("第一路径", "primary path"),
    ("本地", "local"),
    ("偏好", "preference"),
    ("项目", "project"),
    ("记忆", "memory"),
    ("修正", "correction"),
    ("更正", "correction"),
    ("梅洛", "Melo"),
    ("柯科罗", "Kokoro"),
    ("月梦", "Moondream"),
    ("local-http-wav", "local-http-wav"),
    ("MeloTTS", "MeloTTS"),
    ("Melo", "Melo"),
    ("Kokoro", "Kokoro"),
    ("Moondream", "Moondream"),
    ("WebRTC", "WebRTC"),
    ("browser-zh-melo", "browser-zh-melo"),
    ("browser-en-kokoro", "browser-en-kokoro"),
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_text(value: Any, *, limit: int = 120, fallback: str = "") -> str:
    text = _normalized_text(value)
    if not text:
        return fallback
    lowered = text.lower()
    if any(marker in lowered for marker in _UNSAFE_MARKERS):
        return fallback or "redacted"
    return text[:limit]


def _safe_id(value: Any, *, fallback: str = "", limit: int = 160) -> str:
    text = _normalized_text(value)
    if not text:
        return fallback
    if any(marker in text.lower() for marker in _UNSAFE_MARKERS):
        return fallback
    safe = _SAFE_ID_PATTERN.sub("_", text).strip("_")
    return safe[:limit] or fallback


def _safe_reason(value: Any, *, fallback: str = "unknown", limit: int = 96) -> str:
    text = _normalized_text(value).lower().replace(" ", "_")
    if not text:
        return fallback
    if any(marker in text for marker in _UNSAFE_MARKERS):
        return fallback
    safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:limit] or fallback


def _dedupe(values: Iterable[Any], *, limit: int = _MAX_REASON_CODES) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        code = _safe_reason(value, fallback="")
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return tuple(result)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _locale(value: Any) -> str:
    text = _safe_reason(getattr(value, "value", value), fallback="unknown")
    if text.startswith("zh") or text.startswith("cmn"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return "unknown"


def _infer_language_from_text(text: str, *, fallback: str = "unknown") -> str:
    if _ZH_RE.search(text):
        return "zh"
    lowered = text.lower()
    if any(word in lowered for word in ("english", "kokoro", "webrtc", "browser", "moondream")):
        return "en"
    if any(word in lowered for word in ("melo", "melotts", "local-http-wav")):
        return "zh"
    return fallback


def _translated_summary(text: str, *, target_language: str) -> str:
    summary = _safe_text(text, limit=96)
    if not summary:
        return "Memory selected."
    if target_language == "en":
        replacements = {zh: en for zh, en in _TERM_BRIDGE}
    elif target_language == "zh":
        replacements = {en.lower(): zh for zh, en in _TERM_BRIDGE}
    else:
        return summary
    result = summary
    for source, target in replacements.items():
        if target_language == "zh":
            result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)
        else:
            result = result.replace(source, target)
    return _safe_text(result, limit=96) or summary


def display_summary_for_memory_ref(
    ref: BrainMemoryUseTraceRef,
    *,
    target_language: str,
    source_language: str = "unknown",
) -> str:
    """Render one bounded user-facing memory summary for the requested locale."""
    target = _locale(target_language)
    source = source_language if source_language in {"zh", "en"} else _infer_language_from_text(
        ref.title,
        fallback=target,
    )
    if source in {"zh", "en"} and target in {"zh", "en"} and source != target:
        translated = _translated_summary(ref.title, target_language=target)
        if translated != ref.title:
            return translated
        if target == "zh":
            return f"已选用跨语言{ref.display_kind}记忆。"
        return f"Cross-language {ref.display_kind} memory selected."
    return _safe_text(ref.title, limit=96, fallback="Memory selected.")


def expand_bilingual_memory_query(text: Any, *, language: Any = None) -> str:
    """Return deterministic zh/en expansion text for continuity retrieval."""
    raw = _safe_text(text, limit=280)
    if not raw:
        return ""
    lowered = raw.lower()
    terms: list[str] = [raw]
    for zh, en in _TERM_BRIDGE:
        if zh in raw or en.lower() in lowered:
            terms.extend([zh, en])
    if language is not None:
        locale = _locale(language)
        if locale == "zh":
            terms.extend(["中文", "英文", "浏览器", "相机", "视觉", "本地"])
        elif locale == "en":
            terms.extend(["Chinese", "English", "browser", "camera", "vision", "local"])
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        normalized = _safe_text(term, limit=64)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return " ".join(result)[:480]


@dataclass(frozen=True)
class MemoryCommandIntent:
    """One bounded bilingual memory-command intent classification."""

    intent: str
    language: str
    confidence: float
    text_chars: int
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize without raw user text."""
        return {
            "intent": self.intent,
            "language": self.language,
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 3),
            "text_chars": max(0, int(self.text_chars)),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MemoryCommandIntent | None":
        """Hydrate a command intent from a public payload."""
        if not isinstance(data, Mapping):
            return None
        intent = _safe_reason(data.get("intent"), fallback="none")
        return cls(
            intent=intent if intent in set(_COMMAND_PHRASES) | {"none"} else "none",
            language=_locale(data.get("language")),
            confidence=max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
            text_chars=_safe_int(data.get("text_chars")),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


def detect_memory_command_intent(text: Any, *, language: Any = None) -> MemoryCommandIntent:
    """Classify bounded zh/en memory command intent without returning raw text."""
    raw = _normalized_text(text)
    lowered = raw.lower()
    locale = _locale(language)
    if locale == "unknown":
        locale = _infer_language_from_text(raw, fallback="en")
    matched_intent = "none"
    matched_locale = locale if locale in {"zh", "en"} else "en"
    for intent, phrase_map in _COMMAND_PHRASES.items():
        for phrase_locale in (matched_locale, "zh" if matched_locale == "en" else "en"):
            for phrase in phrase_map.get(phrase_locale, ()):
                phrase_text = phrase if phrase_locale == "zh" else phrase.lower()
                if (phrase_text in raw) if phrase_locale == "zh" else (phrase_text in lowered):
                    matched_intent = intent
                    matched_locale = phrase_locale
                    break
            if matched_intent != "none":
                break
        if matched_intent != "none":
            break
    confidence = 0.88 if matched_intent != "none" else 0.0
    return MemoryCommandIntent(
        intent=matched_intent,
        language=matched_locale,
        confidence=confidence,
        text_chars=len(raw),
        reason_codes=_dedupe(
            (
                "memory_command_intent:v1",
                f"memory_command_intent:{matched_intent}",
                f"language:{matched_locale}",
            )
        ),
    )


@dataclass(frozen=True)
class MemoryContinuitySelectedRef:
    """One selected memory visible to the user for this reply."""

    memory_id: str
    display_kind: str
    summary: str
    safe_provenance_label: str
    source_language: str
    cross_language: bool
    inspectable: bool
    editable: bool
    effect_labels: tuple[str, ...] = ()
    linked_discourse_episode_ids: tuple[str, ...] = ()
    conflict_labels: tuple[str, ...] = ()
    staleness_labels: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this selected memory reference for public surfaces."""
        return {
            "memory_id": self.memory_id,
            "display_kind": self.display_kind,
            "summary": self.summary,
            "safe_provenance_label": self.safe_provenance_label,
            "source_language": self.source_language,
            "cross_language": self.cross_language,
            "inspectable": self.inspectable,
            "editable": self.editable,
            "effect_labels": list(self.effect_labels),
            "linked_discourse_episode_ids": list(self.linked_discourse_episode_ids),
            "conflict_labels": list(self.conflict_labels),
            "staleness_labels": list(self.staleness_labels),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContinuitySelectedRef":
        """Hydrate one selected ref from a public-safe payload."""
        source_language = _locale(data.get("source_language"))
        return cls(
            memory_id=_safe_id(data.get("memory_id"), fallback="memory:unknown"),
            display_kind=_safe_reason(data.get("display_kind"), fallback="memory"),
            summary=_safe_text(data.get("summary"), limit=120, fallback="Memory selected."),
            safe_provenance_label=_safe_text(
                data.get("safe_provenance_label"),
                limit=120,
                fallback="Derived from prior conversation memory.",
            ),
            source_language=source_language,
            cross_language=data.get("cross_language") is True,
            inspectable=data.get("inspectable") is not False,
            editable=data.get("editable") is not False,
            effect_labels=_memory_effect_labels(data.get("effect_labels")) or ("none",),
            linked_discourse_episode_ids=tuple(
                _safe_id(item, fallback="")
                for item in list(data.get("linked_discourse_episode_ids") or ())[:_MAX_DISCOURSE_REFS]
                if _safe_id(item, fallback="")
            ),
            conflict_labels=_dedupe(data.get("conflict_labels") or (), limit=8),
            staleness_labels=_dedupe(data.get("staleness_labels") or (), limit=8),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )

    @classmethod
    def from_use_ref(
        cls,
        ref: BrainMemoryUseTraceRef,
        *,
        target_language: str,
        effect_labels: Iterable[Any] = (),
        linked_discourse_episode_ids: Iterable[Any] = (),
        conflict_labels: Iterable[Any] = (),
        staleness_labels: Iterable[Any] = (),
    ) -> "MemoryContinuitySelectedRef":
        """Build a continuity selected ref from an existing memory-use ref."""
        target = _locale(target_language)
        source = _infer_language_from_text(ref.title, fallback=target)
        cross_language = source in {"zh", "en"} and target in {"zh", "en"} and source != target
        codes = (
            "memory_continuity:selected",
            f"display_kind:{_safe_reason(ref.display_kind, fallback='memory')}",
            "memory_continuity:cross_language_selected" if cross_language else "memory_continuity:same_language_selected",
            *ref.reason_codes,
            *tuple(
                f"memory_effect:{label}"
                for label in (_memory_effect_labels(effect_labels) or ("none",))
            ),
            *tuple(f"conflict:{label}" for label in _dedupe(conflict_labels, limit=8)),
            *tuple(f"staleness:{label}" for label in _dedupe(staleness_labels, limit=8)),
        )
        return cls(
            memory_id=_safe_id(ref.memory_id, fallback="memory:unknown"),
            display_kind=_safe_reason(ref.display_kind, fallback="memory"),
            summary=display_summary_for_memory_ref(
                ref,
                target_language=target,
                source_language=source,
            ),
            safe_provenance_label=_safe_text(
                ref.safe_provenance_label,
                limit=120,
                fallback="Derived from prior conversation memory.",
            ),
            source_language=source,
            cross_language=cross_language,
            inspectable=True,
            editable=True,
            effect_labels=_memory_effect_labels(effect_labels) or ("none",),
            linked_discourse_episode_ids=tuple(
                item
                for item in (
                    _safe_id(value, fallback="", limit=120)
                    for value in linked_discourse_episode_ids
                )
                if item
            )[:_MAX_DISCOURSE_REFS],
            conflict_labels=_dedupe(conflict_labels, limit=8),
            staleness_labels=_dedupe(staleness_labels, limit=8),
            reason_codes=_dedupe(codes),
        )


@dataclass(frozen=True)
class MemoryContinuitySuppressedReason:
    """One public count bucket explaining why memory was not used."""

    bucket: str
    count: int
    user_visible: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this suppression bucket."""
        return {
            "bucket": self.bucket,
            "count": self.count,
            "user_visible": self.user_visible,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContinuitySuppressedReason":
        """Hydrate one suppression bucket from a public payload."""
        bucket = _safe_reason(data.get("bucket"), fallback="other")
        return cls(
            bucket=bucket,
            count=_safe_int(data.get("count")),
            user_visible=data.get("user_visible") is not False,
            reason_codes=_dedupe(data.get("reason_codes") or (f"memory_suppressed:{bucket}",)),
        )


@dataclass(frozen=True)
class MemoryContinuityTrace:
    """Public-safe per-turn memory continuity trace."""

    schema_version: int
    turn_id: str
    session_id: str
    user_id: str
    agent_id: str
    thread_id: str
    created_at: str
    profile: str
    language: str
    selected_memories: tuple[MemoryContinuitySelectedRef, ...]
    suppressed_memories: tuple[MemoryContinuitySuppressedReason, ...]
    memory_effect: str
    cross_language_count: int
    inspectable: bool
    editable: bool
    user_actions: tuple[str, ...]
    memory_continuity_v3: MemoryContinuityV3 | None = None
    command_intent: MemoryCommandIntent | None = None
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this trace without raw memory bodies or prompts."""
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "profile": self.profile,
            "language": self.language,
            "selected_memory_count": len(self.selected_memories),
            "suppressed_memory_count": sum(item.count for item in self.suppressed_memories),
            "cross_language_count": self.cross_language_count,
            "selected_memories": [ref.as_dict() for ref in self.selected_memories],
            "suppressed_memories": [reason.as_dict() for reason in self.suppressed_memories],
            "memory_effect": self.memory_effect,
            "inspectable": self.inspectable,
            "editable": self.editable,
            "user_actions": list(self.user_actions),
            "memory_continuity_v3": (
                self.memory_continuity_v3.as_dict()
                if self.memory_continuity_v3 is not None
                else _empty_memory_continuity_v3().as_dict()
            ),
            "command_intent": self.command_intent.as_dict() if self.command_intent else None,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContinuityTrace":
        """Hydrate one continuity trace from a public payload."""
        selected = tuple(
            MemoryContinuitySelectedRef.from_dict(item)
            for item in data.get("selected_memories", ())
            if isinstance(item, Mapping)
        )[:_MAX_SELECTED_REFS]
        suppressed = tuple(
            MemoryContinuitySuppressedReason.from_dict(item)
            for item in data.get("suppressed_memories", ())
            if isinstance(item, Mapping)
        )[:_MAX_SUPPRESSED_BUCKETS]
        language = _locale(data.get("language"))
        cross_language_count = _safe_int(
            data.get("cross_language_count"),
            default=sum(1 for item in selected if item.cross_language),
        )
        memory_effect = _safe_reason(data.get("memory_effect"), fallback="")
        if not memory_effect:
            memory_effect = _memory_effect_for_counts(
                selected_count=len(selected),
                suppressed_count=sum(item.count for item in suppressed),
                cross_language_count=cross_language_count,
            )
        return cls(
            schema_version=_safe_int(data.get("schema_version"), default=_SCHEMA_VERSION),
            turn_id=_safe_id(data.get("turn_id"), fallback="turn:auto", limit=120),
            session_id=_safe_id(data.get("session_id"), fallback="", limit=120),
            user_id=_safe_id(data.get("user_id"), fallback="", limit=120),
            agent_id=_safe_id(data.get("agent_id"), fallback="", limit=120),
            thread_id=_safe_id(data.get("thread_id"), fallback="", limit=120),
            created_at=_safe_text(data.get("created_at"), limit=96),
            profile=_safe_text(data.get("profile"), limit=96, fallback="manual"),
            language=language,
            selected_memories=selected,
            suppressed_memories=suppressed,
            memory_effect=memory_effect,
            cross_language_count=cross_language_count,
            inspectable=data.get("inspectable") is not False,
            editable=data.get("editable") is not False,
            user_actions=tuple(_safe_action_list(data.get("user_actions"))),
            memory_continuity_v3=MemoryContinuityV3.from_dict(
                data.get("memory_continuity_v3")
            ),
            command_intent=MemoryCommandIntent.from_dict(data.get("command_intent")),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


def _safe_action_list(value: Any) -> list[str]:
    allowed = {
        "inspect",
        "edit",
        "correct",
        "forget",
        "suppress",
        "pin",
        "mark_stale",
        "list_visible",
        "explain",
    }
    raw_values = value if isinstance(value, (list, tuple, set)) else ()
    result: list[str] = []
    for raw in raw_values:
        action = _safe_reason(raw, fallback="")
        if action in allowed and action not in result:
            result.append(action)
        if len(result) >= _MAX_ACTIONS:
            break
    return result


def _memory_effect_labels(value: Any) -> tuple[str, ...]:
    allowed = set(DISCOURSE_EPISODE_V3_MEMORY_EFFECT_LABELS)
    labels = tuple(label for label in _dedupe(value if isinstance(value, (list, tuple, set)) else ()) if label in allowed)
    if labels and "none" in labels and len(labels) > 1:
        labels = tuple(label for label in labels if label != "none")
    return labels


@dataclass(frozen=True)
class MemoryContinuityDiscourseEpisodeRef:
    """One public-safe discourse episode reference used by continuity v3."""

    discourse_episode_id: str
    category_labels: tuple[str, ...]
    effect_labels: tuple[str, ...]
    confidence_bucket: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this discourse episode reference."""
        return {
            "discourse_episode_id": self.discourse_episode_id,
            "category_labels": list(self.category_labels),
            "effect_labels": list(self.effect_labels),
            "confidence_bucket": self.confidence_bucket,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryContinuityDiscourseEpisodeRef":
        """Hydrate one discourse episode ref from a public payload."""
        return cls(
            discourse_episode_id=_safe_id(
                data.get("discourse_episode_id"),
                fallback="discourse-episode-v3:unknown",
                limit=120,
            ),
            category_labels=_dedupe(data.get("category_labels") or (), limit=8),
            effect_labels=_memory_effect_labels(data.get("effect_labels")) or ("none",),
            confidence_bucket=_safe_reason(data.get("confidence_bucket"), fallback="medium"),
            reason_codes=_dedupe(data.get("reason_codes") or (), limit=12),
        )

    @classmethod
    def from_episode(cls, episode: DiscourseEpisode) -> "MemoryContinuityDiscourseEpisodeRef":
        """Build a continuity ref from one discourse episode."""
        return cls(
            discourse_episode_id=_safe_id(
                episode.discourse_episode_id,
                fallback="discourse-episode-v3:unknown",
                limit=120,
            ),
            category_labels=_dedupe(episode.category_labels, limit=8),
            effect_labels=_memory_effect_labels(episode.effect_labels) or ("none",),
            confidence_bucket=_safe_reason(episode.confidence_bucket, fallback="medium"),
            reason_codes=_dedupe(
                (
                    "memory_continuity_v3:discourse_episode_ref",
                    *episode.reason_codes,
                ),
                limit=12,
            ),
        )


@dataclass(frozen=True)
class MemoryContinuityV3:
    """Additive v3 continuity details nested inside the v1 trace."""

    schema_version: int
    selected_discourse_episodes: tuple[MemoryContinuityDiscourseEpisodeRef, ...]
    effect_labels: tuple[str, ...]
    conflict_labels: tuple[str, ...]
    staleness_labels: tuple[str, ...]
    cross_language_transfer_count: int
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this v3 continuity extension."""
        return {
            "schema_version": self.schema_version,
            "selected_discourse_episodes": [
                ref.as_dict() for ref in self.selected_discourse_episodes
            ],
            "effect_labels": list(self.effect_labels),
            "conflict_labels": list(self.conflict_labels),
            "staleness_labels": list(self.staleness_labels),
            "cross_language_transfer_count": self.cross_language_transfer_count,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MemoryContinuityV3":
        """Hydrate the v3 extension from a public payload."""
        if not isinstance(data, Mapping):
            return _empty_memory_continuity_v3()
        refs = tuple(
            MemoryContinuityDiscourseEpisodeRef.from_dict(item)
            for item in data.get("selected_discourse_episodes", ())
            if isinstance(item, Mapping)
        )[:_MAX_DISCOURSE_REFS]
        return cls(
            schema_version=_safe_int(data.get("schema_version"), default=3),
            selected_discourse_episodes=refs,
            effect_labels=_memory_effect_labels(data.get("effect_labels")) or ("none",),
            conflict_labels=_dedupe(data.get("conflict_labels") or (), limit=8),
            staleness_labels=_dedupe(data.get("staleness_labels") or (), limit=8),
            cross_language_transfer_count=_safe_int(
                data.get("cross_language_transfer_count")
            ),
            reason_codes=_dedupe(data.get("reason_codes") or (), limit=_MAX_REASON_CODES),
        )


def _empty_memory_continuity_v3() -> MemoryContinuityV3:
    return MemoryContinuityV3(
        schema_version=3,
        selected_discourse_episodes=(),
        effect_labels=("none",),
        conflict_labels=(),
        staleness_labels=(),
        cross_language_transfer_count=0,
        reason_codes=("memory_continuity_v3:none",),
    )


def _discourse_episode_objects(value: Iterable[Any] | None) -> tuple[DiscourseEpisode, ...]:
    episodes: list[DiscourseEpisode] = []
    for item in list(value or ())[:_MAX_DISCOURSE_REFS]:
        if isinstance(item, DiscourseEpisode):
            episodes.append(item)
        elif isinstance(item, Mapping):
            try:
                episodes.append(DiscourseEpisode.from_dict(item))
            except (TypeError, ValueError):
                continue
    return tuple(episodes)


def _discourse_effects_for_memory(
    *,
    memory_id: str,
    episodes: tuple[DiscourseEpisode, ...],
) -> tuple[str, ...]:
    effects: list[str] = []
    for episode in episodes:
        for ref in episode.memory_refs:
            if ref.memory_id == memory_id:
                effects.extend(ref.effect_labels)
        if not episode.memory_refs and memory_id:
            effects.extend(episode.effect_labels)
    return _memory_effect_labels(effects)


def _discourse_ids_for_memory(
    *,
    memory_id: str,
    episodes: tuple[DiscourseEpisode, ...],
) -> tuple[str, ...]:
    ids: list[str] = []
    for episode in episodes:
        if not episode.memory_refs or any(ref.memory_id == memory_id for ref in episode.memory_refs):
            ids.append(episode.discourse_episode_id)
    return tuple(
        item
        for item in (_safe_id(value, fallback="", limit=120) for value in ids)
        if item
    )[:_MAX_DISCOURSE_REFS]


def _episode_conflict_labels(episodes: tuple[DiscourseEpisode, ...]) -> tuple[str, ...]:
    return _dedupe(
        (label for episode in episodes for label in episode.conflict_labels),
        limit=8,
    )


def _episode_staleness_labels(episodes: tuple[DiscourseEpisode, ...]) -> tuple[str, ...]:
    return _dedupe(
        (label for episode in episodes for label in episode.staleness_labels),
        limit=8,
    )


def _hidden_conflict_labels(hidden_counts: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(hidden_counts, Mapping):
        return ()
    labels: list[str] = []
    for key, value in hidden_counts.items():
        if _safe_int(value) <= 0:
            continue
        bucket = _safe_reason(key, fallback="")
        if bucket in {"contradicted", "conflict", "superseded"}:
            labels.append(bucket)
        if bucket in {"historical", "stale", "suppressed"}:
            labels.append("memory_not_current")
    return _dedupe(labels, limit=8)


def _hidden_staleness_labels(hidden_counts: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(hidden_counts, Mapping):
        return ()
    labels: list[str] = []
    for key, value in hidden_counts.items():
        if _safe_int(value) <= 0:
            continue
        bucket = _safe_reason(key, fallback="")
        if bucket in {"historical", "stale", "suppressed"}:
            labels.append(bucket)
    return _dedupe(labels, limit=8)


def _build_memory_continuity_v3(
    *,
    selected: tuple[MemoryContinuitySelectedRef, ...],
    discourse_episodes: tuple[DiscourseEpisode, ...],
    hidden_counts: Mapping[str, Any] | None,
    cross_language_count: int,
) -> MemoryContinuityV3:
    discourse_refs = tuple(
        MemoryContinuityDiscourseEpisodeRef.from_episode(episode)
        for episode in discourse_episodes
    )[:_MAX_DISCOURSE_REFS]
    effect_labels = _memory_effect_labels(
        (
            *(label for ref in selected for label in ref.effect_labels),
            *(label for episode in discourse_episodes for label in episode.effect_labels),
        )
    )
    conflict_labels = _dedupe(
        (
            *(label for ref in selected for label in ref.conflict_labels),
            *_episode_conflict_labels(discourse_episodes),
            *_hidden_conflict_labels(hidden_counts),
        ),
        limit=8,
    )
    staleness_labels = _dedupe(
        (
            *(label for ref in selected for label in ref.staleness_labels),
            *_episode_staleness_labels(discourse_episodes),
            *_hidden_staleness_labels(hidden_counts),
        ),
        limit=8,
    )
    cross_language_transfer_count = max(
        cross_language_count,
        sum(1 for ref in selected if ref.cross_language),
        sum(1 for episode in discourse_episodes for ref in episode.memory_refs if ref.cross_language),
    )
    if staleness_labels or conflict_labels:
        effect_labels = _memory_effect_labels(
            (*(() if effect_labels == ("none",) else effect_labels), "suppressed_stale_memory")
        )
    elif not effect_labels:
        if selected or discourse_refs:
            effect_labels = ("tentative_callback",)
        else:
            effect_labels = ("none",)
    reason_codes = _dedupe(
        (
            "memory_continuity_v3:available",
            f"discourse_episode_count:{len(discourse_refs)}",
            f"cross_language_transfer_count:{cross_language_transfer_count}",
            *(f"memory_effect:{label}" for label in effect_labels),
            *(f"conflict:{label}" for label in conflict_labels),
            *(f"staleness:{label}" for label in staleness_labels),
        ),
        limit=_MAX_REASON_CODES,
    )
    return MemoryContinuityV3(
        schema_version=3,
        selected_discourse_episodes=discourse_refs,
        effect_labels=effect_labels,
        conflict_labels=conflict_labels,
        staleness_labels=staleness_labels,
        cross_language_transfer_count=cross_language_transfer_count,
        reason_codes=reason_codes,
    )


def _suppressed_reasons_from_hidden_counts(
    hidden_counts: Mapping[str, Any] | None,
) -> tuple[MemoryContinuitySuppressedReason, ...]:
    if not isinstance(hidden_counts, Mapping):
        return ()
    buckets: list[MemoryContinuitySuppressedReason] = []
    for key, value in list(hidden_counts.items())[:_MAX_SUPPRESSED_BUCKETS]:
        count = _safe_int(value)
        if count <= 0:
            continue
        bucket = _safe_reason(key, fallback="other")
        buckets.append(
            MemoryContinuitySuppressedReason(
                bucket=bucket,
                count=count,
                user_visible=True,
                reason_codes=_dedupe(
                    (
                        "memory_continuity:suppressed",
                        f"memory_suppressed:{bucket}",
                    )
                ),
            )
        )
    return tuple(buckets)


def _memory_effect_for_counts(
    *,
    selected_count: int,
    suppressed_count: int,
    cross_language_count: int,
) -> str:
    if cross_language_count > 0:
        return "cross_language_callback"
    if selected_count > 0:
        return "callback_available"
    if suppressed_count > 0:
        return "repair_or_uncertainty"
    return "none"


def build_memory_continuity_trace(
    *,
    memory_use_trace: BrainMemoryUseTrace | None = None,
    session_id: str = "",
    profile: str = "manual",
    language: Any = "unknown",
    turn_id: str = "",
    created_at: str = "",
    hidden_counts: Mapping[str, Any] | None = None,
    command_intent: MemoryCommandIntent | Mapping[str, Any] | None = None,
    discourse_episodes: Iterable[DiscourseEpisode | Mapping[str, Any]] = (),
    reason_codes: Iterable[Any] = (),
) -> MemoryContinuityTrace:
    """Build one public-safe continuity trace from memory-use and read-model hints."""
    target_language = _locale(language)
    discourse_episode_objects = _discourse_episode_objects(discourse_episodes)
    selected = tuple(
        MemoryContinuitySelectedRef.from_use_ref(
            ref,
            target_language=target_language,
            effect_labels=_discourse_effects_for_memory(
                memory_id=_safe_id(ref.memory_id, fallback="memory:unknown"),
                episodes=discourse_episode_objects,
            ),
            linked_discourse_episode_ids=_discourse_ids_for_memory(
                memory_id=_safe_id(ref.memory_id, fallback="memory:unknown"),
                episodes=discourse_episode_objects,
            ),
            conflict_labels=_episode_conflict_labels(discourse_episode_objects),
            staleness_labels=_episode_staleness_labels(discourse_episode_objects),
        )
        for ref in (memory_use_trace.refs if memory_use_trace is not None else ())
    )[:_MAX_SELECTED_REFS]
    suppressed = _suppressed_reasons_from_hidden_counts(hidden_counts)
    cross_language_count = sum(1 for ref in selected if ref.cross_language)
    suppressed_count = sum(item.count for item in suppressed)
    selected_count = len(selected)
    memory_effect = _memory_effect_for_counts(
        selected_count=selected_count,
        suppressed_count=suppressed_count,
        cross_language_count=cross_language_count,
    )
    resolved_intent: MemoryCommandIntent | None
    if isinstance(command_intent, MemoryCommandIntent):
        resolved_intent = command_intent
    elif isinstance(command_intent, Mapping):
        resolved_intent = MemoryCommandIntent.from_dict(command_intent)
    else:
        resolved_intent = None
    if selected_count > 0:
        actions = ("inspect", "edit", "correct", "forget", "suppress", "pin", "explain")
    else:
        actions = ("list_visible", "explain")
    memory_continuity_v3 = _build_memory_continuity_v3(
        selected=selected,
        discourse_episodes=discourse_episode_objects,
        hidden_counts=hidden_counts,
        cross_language_count=cross_language_count,
    )
    trace_codes = _dedupe(
        (
            "memory_continuity_trace:v1",
            "memory_continuity_trace:v3_nested",
            f"profile:{_safe_reason(profile, fallback='manual')}",
            f"language:{target_language}",
            f"selected_memory_count:{selected_count}",
            f"suppressed_memory_count:{suppressed_count}",
            f"cross_language_count:{cross_language_count}",
            f"memory_effect:{memory_effect}",
            *memory_continuity_v3.reason_codes,
            *(
                memory_use_trace.reason_codes
                if memory_use_trace is not None
                else ("memory_use_trace_empty",)
            ),
            *(
                resolved_intent.reason_codes
                if resolved_intent is not None
                else ("memory_command_intent:none",)
            ),
            *reason_codes,
        )
    )
    return MemoryContinuityTrace(
        schema_version=_SCHEMA_VERSION,
        turn_id=_safe_id(turn_id, fallback="turn:auto", limit=120),
        session_id=_safe_id(session_id, fallback="", limit=120),
        user_id=_safe_id(getattr(memory_use_trace, "user_id", ""), fallback="", limit=120),
        agent_id=_safe_id(getattr(memory_use_trace, "agent_id", ""), fallback="", limit=120),
        thread_id=_safe_id(getattr(memory_use_trace, "thread_id", ""), fallback="", limit=120),
        created_at=_safe_text(created_at or getattr(memory_use_trace, "created_at", ""), limit=96),
        profile=_safe_text(profile, limit=96, fallback="manual"),
        language=target_language,
        selected_memories=selected,
        suppressed_memories=suppressed,
        memory_effect=memory_effect,
        cross_language_count=cross_language_count,
        inspectable=True,
        editable=selected_count > 0,
        user_actions=actions,
        memory_continuity_v3=memory_continuity_v3,
        command_intent=resolved_intent,
        reason_codes=trace_codes,
    )


def stamp_memory_continuity_trace(
    trace: MemoryContinuityTrace,
    *,
    created_at: str,
    session_id: str | None = None,
) -> MemoryContinuityTrace:
    """Return one trace stamped with the persisted event timestamp."""
    return MemoryContinuityTrace(
        schema_version=trace.schema_version,
        turn_id=trace.turn_id,
        session_id=_safe_id(session_id or trace.session_id, fallback="", limit=120),
        user_id=trace.user_id,
        agent_id=trace.agent_id,
        thread_id=trace.thread_id,
        created_at=_safe_text(created_at, limit=96),
        profile=trace.profile,
        language=trace.language,
        selected_memories=trace.selected_memories,
        suppressed_memories=trace.suppressed_memories,
        memory_effect=trace.memory_effect,
        cross_language_count=trace.cross_language_count,
        inspectable=trace.inspectable,
        editable=trace.editable,
        user_actions=trace.user_actions,
        memory_continuity_v3=trace.memory_continuity_v3,
        command_intent=trace.command_intent,
        reason_codes=trace.reason_codes,
    )


def public_actor_event_metadata_for_memory_continuity(
    trace: MemoryContinuityTrace | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return actor-event-safe counts and labels for one continuity trace."""
    if isinstance(trace, Mapping):
        trace_obj = MemoryContinuityTrace.from_dict(trace)
    else:
        trace_obj = trace
    if trace_obj is None:
        return {
            "selected_memory_count": 0,
            "suppressed_memory_count": 0,
            "cross_language_count": 0,
            "memory_effect": "none",
            "discourse_episode_count": 0,
            "memory_effect_labels": [],
        }
    continuity_v3 = trace_obj.memory_continuity_v3 or _empty_memory_continuity_v3()
    return {
        "selected_memory_count": len(trace_obj.selected_memories),
        "suppressed_memory_count": sum(item.count for item in trace_obj.suppressed_memories),
        "cross_language_count": trace_obj.cross_language_count,
        "memory_effect": trace_obj.memory_effect,
        "display_kinds": sorted({ref.display_kind for ref in trace_obj.selected_memories})[:6],
        "discourse_episode_count": len(continuity_v3.selected_discourse_episodes),
        "discourse_episode_ids": [
            ref.discourse_episode_id for ref in continuity_v3.selected_discourse_episodes
        ],
        "memory_effect_labels": list(continuity_v3.effect_labels),
        "conflict_labels": list(continuity_v3.conflict_labels),
        "staleness_labels": list(continuity_v3.staleness_labels),
    }


__all__ = [
    "MemoryCommandIntent",
    "MemoryContinuityDiscourseEpisodeRef",
    "MemoryContinuitySelectedRef",
    "MemoryContinuitySuppressedReason",
    "MemoryContinuityTrace",
    "MemoryContinuityV3",
    "build_memory_continuity_trace",
    "detect_memory_command_intent",
    "display_summary_for_memory_ref",
    "expand_bilingual_memory_query",
    "public_actor_event_metadata_for_memory_continuity",
    "stamp_memory_continuity_trace",
]

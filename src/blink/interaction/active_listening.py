"""Public-safe active-listening state helpers for browser runtimes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class BrowserActiveListeningPhase(str, Enum):
    """Public active-listening phases for the browser status surface."""

    IDLE = "idle"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SPEECH_STOPPED = "speech_stopped"
    TRANSCRIBING = "transcribing"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    ERROR = "error"


class ActiveListenerPhaseV2(str, Enum):
    """Public active-listener v2 phases for actor-state and diagnostics."""

    IDLE = "idle"
    LISTENING_STARTED = "listening_started"
    SPEECH_CONTINUING = "speech_continuing"
    PARTIAL_UNDERSTANDING = "partial_understanding"
    TRANSCRIBING = "transcribing"
    FINAL_UNDERSTANDING = "final_understanding"
    READY_TO_ANSWER = "ready_to_answer"
    DEGRADED = "degraded"
    ERROR = "error"


_PHRASE_SPLIT_RE = re.compile(r"[\n\r。！？!?.；;，,、：:]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:~?/|[A-Za-z]:\\)[^\s]+")
_SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]+|bearer\s+\S+)\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_PROJECT_RE = re.compile(
    r"\b(?:Blink|MeloTTS|Melo|Kokoro|Moondream|WebRTC|Active Listener v2|"
    r"browser-zh-melo|browser-en-kokoro|local-http-wav|actor-state|actor state)\b",
    re.IGNORECASE,
)
_CONSTRAINT_MARKERS = (
    "不要",
    "别",
    "不能",
    "不需要",
    "必须",
    "需要",
    "只能",
    "请",
    "避免",
    "不要用",
    "only",
    "must",
    "need",
    "needs",
    "please",
    "avoid",
    "without",
    "don't",
    "dont",
    "do not",
    "should",
)
_CORRECTION_MARKERS = (
    "更正",
    "纠正",
    "改成",
    "应该是",
    "不是",
    "说错",
    "actually",
    "correction",
    "correct",
    "i meant",
    "instead",
    "should be",
    "change it to",
    "not",
)
_UNCERTAINTY_MARKERS = (
    "可能",
    "不确定",
    "大概",
    "也许",
    "好像",
    "听不清",
    "maybe",
    "probably",
    "not sure",
    "unclear",
    "roughly",
    "possibly",
)
_UNSAFE_HINT_FRAGMENTS = (
    "authorization",
    "bearer ",
    "credential",
    "hidden prompt",
    "memory_id",
    "password",
    "secret",
    "sk-",
    "system prompt",
    "token",
)
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
_SEMANTIC_CAMERA_STATES = {
    "not_used",
    "fresh_supported",
    "fresh_available",
    "stale_or_limited",
    "unsupported",
    "error",
}
_QUESTION_MARKERS = (
    "?",
    "？",
    "吗",
    "什么",
    "怎么",
    "为什么",
    "能不能",
    "是不是",
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "can you",
    "could you",
    "do you",
    "is it",
)
_OBJECT_SHOWING_MARKERS = (
    "看看",
    "看一下",
    "我给你看",
    "这个东西",
    "这个物体",
    "镜头",
    "摄像头",
    "画面",
    "照片",
    "图片",
    "look at this",
    "can you see",
    "what is this",
    "showing",
    "object",
    "camera",
    "image",
    "picture",
)
_PROJECT_PLANNING_MARKERS = (
    "计划",
    "规划",
    "步骤",
    "里程碑",
    "排期",
    "方案",
    "项目",
    "roadmap",
    "plan",
    "planning",
    "milestone",
    "timeline",
    "project",
)
_SMALL_TALK_MARKERS = (
    "你好",
    "谢谢",
    "嗨",
    "早上好",
    "hello",
    "hi",
    "thanks",
    "thank you",
    "good morning",
)
_INSTRUCTION_MARKERS = (
    "帮我",
    "请你",
    "给我",
    "做",
    "写",
    "整理",
    "生成",
    "make",
    "write",
    "create",
    "summarize",
    "help me",
)
_POSITIVE_TONE_MARKERS = ("谢谢", "很好", "不错", "thanks", "great", "good")
_CONCERN_TONE_MARKERS = ("担心", "不确定", "混乱", "confused", "unclear", "worried")
_FRUSTRATED_TONE_MARKERS = ("不是", "不对", "烦", "错了", "wrong", "actually no", "not right")
_ZH_TOPIC_FILLERS = (
    "请",
    "帮我",
    "我想",
    "我希望",
    "一下",
    "这个",
    "那个",
    "然后",
    "就是",
    "关于",
)
_EN_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "can",
    "for",
    "from",
    "have",
    "how",
    "into",
    "make",
    "need",
    "only",
    "please",
    "should",
    "that",
    "the",
    "this",
    "with",
    "without",
    "you",
}

_CHIP_LABELS: dict[str, tuple[str, str]] = {
    "heard_summary": ("I heard...", "我听到..."),
    "constraint_detected": ("constraint detected", "检测到约束"),
    "question_detected": ("question detected", "检测到问题"),
    "showing_object": ("showing object", "正在看物体"),
    "camera_limited": ("camera limited", "视觉受限"),
    "still_listening": ("still listening", "继续听"),
    "ready_to_answer": ("ready to answer", "可以回答"),
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_hint_text(value: object, *, limit: int = 48) -> str:
    text = str(value or "").strip()
    text = _URL_RE.sub(" ", text)
    text = _PATH_RE.sub(" ", text)
    text = _SECRET_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip(" \t\r\n-_:：，,。.!！?？;；")
    return text[:limit]


def _hint_value_is_safe(value: object) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return not any(fragment in text for fragment in _UNSAFE_HINT_FRAGMENTS)


def _dedupe_hints(
    values: list["ActiveListeningHint"],
    *,
    limit: int = 5,
) -> tuple["ActiveListeningHint", ...]:
    seen: set[tuple[str, str]] = set()
    result: list[ActiveListeningHint] = []
    for value in values:
        hint_value = _clean_hint_text(value.value, limit=48)
        if not hint_value or not _hint_value_is_safe(hint_value):
            continue
        kind = value.kind if value.kind in {"topic", "constraint", "correction", "project_reference"} else "topic"
        key = (kind, hint_value.lower())
        if key in seen:
            continue
        seen.add(key)
        confidence = value.confidence if value.confidence in {"heuristic", "observed", "unknown"} else "heuristic"
        source = (
            value.source
            if value.source in {"final_transcript", "partial_transcript", "unknown"}
            else "unknown"
        )
        result.append(
            ActiveListeningHint(
                kind=kind,
                value=hint_value,
                confidence=confidence,
                source=source,
                editable=bool(value.editable),
            )
        )
        if len(result) >= limit:
            break
    return tuple(result)


def _dedupe_strings(values: list[str], *, limit: int = 5) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_hint_text(value, limit=48)
        key = text.lower()
        if not text or not _hint_value_is_safe(text) or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


@dataclass(frozen=True)
class ActiveListeningHint:
    """A bounded, user-inspectable active-listening hint."""

    kind: str
    value: str
    confidence: str = "heuristic"
    source: str = "final_transcript"
    editable: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return a public API payload for the hint."""
        return {
            "kind": self.kind,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "editable": self.editable,
        }


@dataclass(frozen=True)
class ActiveListeningUnderstanding:
    """Deterministic topic and constraint hints derived from a final transcript."""

    topics: tuple[ActiveListeningHint, ...] = ()
    constraints: tuple[ActiveListeningHint, ...] = ()
    corrections: tuple[ActiveListeningHint, ...] = ()
    project_references: tuple[ActiveListeningHint, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ("active_listening_understanding:v1",)

    def as_dict(self) -> dict[str, object]:
        """Return a public-safe understanding payload."""
        return {
            "topics": [hint.as_dict() for hint in self.topics],
            "constraints": [hint.as_dict() for hint in self.constraints],
            "corrections": [hint.as_dict() for hint in self.corrections],
            "project_references": [hint.as_dict() for hint in self.project_references],
            "uncertainty_flags": list(self.uncertainty_flags),
            "topic_count": len(self.topics),
            "constraint_count": len(self.constraints),
            "correction_count": len(self.corrections),
            "project_reference_count": len(self.project_references),
            "uncertainty_flag_count": len(self.uncertainty_flags),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class SemanticListenerChipV3:
    """A compact browser chip for the active-listener semantic state."""

    chip_id: str
    label: str = ""
    localized_label: str = ""
    state: str = "active"
    reason_codes: tuple[str, ...] = ()

    def as_dict(self, *, language: str = "unknown") -> dict[str, object]:
        """Return a public-safe chip payload."""
        chip_id = self.chip_id if self.chip_id in _SEMANTIC_CHIP_IDS else "still_listening"
        label, localized_label = _semantic_chip_labels(chip_id, language=language)
        state = _clean_hint_text(self.state, limit=24) or "active"
        if state not in {"active", "pending", "limited", "ready"}:
            state = "active"
        return {
            "chip_id": chip_id,
            "label": _clean_hint_text(self.label, limit=48) or label,
            "localized_label": _clean_hint_text(self.localized_label, limit=48)
            or localized_label,
            "state": state,
            "reason_codes": list(
                _dedupe_strings(
                    [f"semantic_listener_chip:{chip_id}", *list(self.reason_codes)],
                    limit=8,
                )
            ),
        }


@dataclass(frozen=True)
class SemanticListenerStateV3:
    """Public-safe semantic active-listener state for live actor-state."""

    current_topic: str = "unknown"
    detected_intent: str = "unknown"
    constraints: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    emotional_tone_label: str = "neutral"
    language: str = "unknown"
    enough_information_to_answer: bool = False
    safe_live_summary: str = ""
    listener_chips: tuple[SemanticListenerChipV3, ...] = ()
    camera_reference_state: str = "not_used"
    memory_context_state: str = "unavailable"
    floor_state: str = "unknown"
    reason_codes: tuple[str, ...] = ("semantic_listener:v3",)

    def as_dict(self) -> dict[str, object]:
        """Return the v3 semantic state as a public API payload."""
        language = _semantic_language(self.language)
        intent = self.detected_intent if self.detected_intent in _SEMANTIC_INTENTS else "unknown"
        camera_state = (
            self.camera_reference_state
            if self.camera_reference_state in _SEMANTIC_CAMERA_STATES
            else "not_used"
        )
        memory_context_state = _safe_semantic_state_label(
            self.memory_context_state,
            allowed={"available", "limited", "unavailable"},
            default="unavailable",
        )
        floor_state = _clean_hint_text(self.floor_state, limit=48) or "unknown"
        constraints = _dedupe_strings(list(self.constraints), limit=5)
        uncertainty = _dedupe_strings(list(self.uncertainty), limit=5)
        chips = _dedupe_semantic_chips(self.listener_chips, language=language)
        reason_codes = _dedupe_strings(
            [
                "semantic_listener:v3",
                f"semantic_listener:intent:{intent}",
                f"semantic_listener:camera:{camera_state}",
                *list(self.reason_codes),
            ],
            limit=24,
        )
        return {
            "schema_version": 3,
            "current_topic": _clean_hint_text(self.current_topic, limit=48) or "unknown",
            "detected_intent": intent,
            "constraints": list(constraints),
            "uncertainty": list(uncertainty),
            "emotional_tone_label": _safe_semantic_state_label(
                self.emotional_tone_label,
                allowed={"neutral", "positive", "concerned", "frustrated", "uncertain"},
                default="neutral",
            ),
            "language": language,
            "enough_information_to_answer": bool(self.enough_information_to_answer),
            "safe_live_summary": _safe_live_summary(self.safe_live_summary),
            "listener_chips": chips,
            "listener_chip_count": len(chips),
            "camera_reference_state": camera_state,
            "memory_context_state": memory_context_state,
            "floor_state": floor_state,
            "reason_codes": list(reason_codes),
        }


@dataclass(frozen=True)
class ActiveListenerStateV2:
    """Public-safe active listener v2 state for browser actor-state."""

    profile: str = "manual"
    language: str = "unknown"
    available: bool = False
    phase: ActiveListenerPhaseV2 | str = ActiveListenerPhaseV2.IDLE
    partial_available: bool = False
    final_available: bool = False
    partial_transcript_chars: int = 0
    final_transcript_chars: int = 0
    interim_transcript_count: int = 0
    final_transcript_count: int = 0
    speech_start_count: int = 0
    speech_stop_count: int = 0
    turn_started_at: str | None = None
    turn_stopped_at: str | None = None
    last_update_at: str | None = None
    turn_duration_ms: int | None = None
    topics: tuple[ActiveListeningHint, ...] = ()
    constraints: tuple[ActiveListeningHint, ...] = ()
    corrections: tuple[ActiveListeningHint, ...] = ()
    project_references: tuple[ActiveListeningHint, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()
    ready_to_answer: bool = False
    readiness_state: str = "not_ready"
    degradation: dict[str, Any] | None = None
    semantic_state_v3: SemanticListenerStateV3 | dict[str, Any] | None = None
    reason_codes: tuple[str, ...] = field(default_factory=lambda: ("active_listener:v2",))

    def as_dict(self) -> dict[str, object]:
        """Return the v2 active-listener snapshot as a public API payload."""
        phase = _active_listener_phase_value(self.phase)
        degradation = _safe_degradation_payload(self.degradation, phase=phase)
        topics = _dedupe_hints(list(self.topics))
        constraints = _dedupe_hints(list(self.constraints))
        corrections = _dedupe_hints(list(self.corrections))
        project_references = _dedupe_hints(list(self.project_references))
        uncertainty_flags = _dedupe_strings(list(self.uncertainty_flags))
        reason_codes = _dedupe_strings(
            [
                "active_listener:v2",
                f"active_listener:{phase}",
                f"readiness:{self.readiness_state or 'not_ready'}",
                *list(self.reason_codes),
                *list(degradation.get("reason_codes", [])),
            ],
            limit=24,
        )
        semantic_state = _semantic_listener_state_v3_payload(
            self.semantic_state_v3,
            language=self.language,
            topics=topics,
            constraints=constraints,
            corrections=corrections,
            project_references=project_references,
            uncertainty_flags=uncertainty_flags,
            partial_available=self.partial_available,
            final_available=self.final_available,
            partial_transcript_chars=self.partial_transcript_chars,
            final_transcript_chars=self.final_transcript_chars,
            turn_duration_ms=self.turn_duration_ms,
            ready_to_answer=self.ready_to_answer,
            readiness_state=self.readiness_state,
            floor_state=phase,
            reason_codes=reason_codes,
        )
        return {
            "schema_version": 2,
            "available": bool(self.available),
            "profile": _clean_hint_text(self.profile, limit=96) or "manual",
            "language": _clean_hint_text(self.language, limit=32) or "unknown",
            "phase": phase,
            "partial_available": bool(self.partial_available),
            "final_available": bool(self.final_available),
            "partial_transcript_chars": max(0, int(self.partial_transcript_chars or 0)),
            "final_transcript_chars": max(0, int(self.final_transcript_chars or 0)),
            "interim_transcript_count": max(0, int(self.interim_transcript_count or 0)),
            "final_transcript_count": max(0, int(self.final_transcript_count or 0)),
            "speech_start_count": max(0, int(self.speech_start_count or 0)),
            "speech_stop_count": max(0, int(self.speech_stop_count or 0)),
            "turn_started_at": self.turn_started_at,
            "turn_stopped_at": self.turn_stopped_at,
            "last_update_at": self.last_update_at,
            "turn_duration_ms": (
                max(0, int(self.turn_duration_ms)) if self.turn_duration_ms is not None else None
            ),
            "topics": [hint.as_dict() for hint in topics],
            "constraints": [hint.as_dict() for hint in constraints],
            "corrections": [hint.as_dict() for hint in corrections],
            "project_references": [hint.as_dict() for hint in project_references],
            "topic_count": len(topics),
            "constraint_count": len(constraints),
            "correction_count": len(corrections),
            "project_reference_count": len(project_references),
            "uncertainty_flags": list(uncertainty_flags),
            "uncertainty_flag_count": len(uncertainty_flags),
            "ready_to_answer": bool(self.ready_to_answer),
            "readiness_state": _clean_hint_text(self.readiness_state, limit=40) or "not_ready",
            "degradation": degradation,
            "semantic_state_v3": semantic_state,
            "reason_codes": list(reason_codes),
        }


@dataclass(frozen=True)
class BrowserActiveListeningSnapshot:
    """Public active-listening snapshot for browser runtime state."""

    available: bool = False
    phase: BrowserActiveListeningPhase | str = BrowserActiveListeningPhase.IDLE
    partial_available: bool = False
    partial_transcript_chars: int = 0
    final_transcript_chars: int = 0
    interim_transcript_count: int = 0
    final_transcript_count: int = 0
    speech_start_count: int = 0
    speech_stop_count: int = 0
    turn_started_at: str | None = None
    turn_stopped_at: str | None = None
    last_update_at: str | None = None
    turn_duration_ms: int | None = None
    topics: tuple[ActiveListeningHint, ...] = ()
    constraints: tuple[ActiveListeningHint, ...] = ()
    reason_codes: tuple[str, ...] = field(default_factory=lambda: ("active_listening:v1",))

    def as_dict(self) -> dict[str, object]:
        """Return the snapshot as a public API payload."""
        phase = str(getattr(self.phase, "value", self.phase) or BrowserActiveListeningPhase.IDLE.value)
        return {
            "schema_version": 1,
            "available": bool(self.available),
            "phase": phase,
            "partial_available": bool(self.partial_available),
            "partial_transcript_chars": max(0, int(self.partial_transcript_chars or 0)),
            "final_transcript_chars": max(0, int(self.final_transcript_chars or 0)),
            "interim_transcript_count": max(0, int(self.interim_transcript_count or 0)),
            "final_transcript_count": max(0, int(self.final_transcript_count or 0)),
            "speech_start_count": max(0, int(self.speech_start_count or 0)),
            "speech_stop_count": max(0, int(self.speech_stop_count or 0)),
            "turn_started_at": self.turn_started_at,
            "turn_stopped_at": self.turn_stopped_at,
            "last_update_at": self.last_update_at,
            "turn_duration_ms": (
                max(0, int(self.turn_duration_ms)) if self.turn_duration_ms is not None else None
            ),
            "topics": [hint.as_dict() for hint in self.topics[:5]],
            "constraints": [hint.as_dict() for hint in self.constraints[:5]],
            "topic_count": len(self.topics[:5]),
            "constraint_count": len(self.constraints[:5]),
            "reason_codes": list(self.reason_codes),
        }


def extract_active_listening_understanding(
    text: object,
    *,
    language: object = None,
    source: str = "final_transcript",
) -> ActiveListeningUnderstanding:
    """Extract bounded topic and constraint hints from final transcript text."""
    transcript = str(text or "")
    hint_source = (
        "partial_transcript" if str(source).lower() == "partial_transcript" else "final_transcript"
    )
    phrases = [_clean_hint_text(part) for part in _PHRASE_SPLIT_RE.split(transcript)]
    phrases = [part for part in phrases if len(part) >= 2]
    language_text = str(language or "").lower()
    cjk_present = bool(_CJK_RE.search(transcript)) or language_text.startswith("zh")

    constraints: list[ActiveListeningHint] = []
    corrections: list[ActiveListeningHint] = []
    topics: list[ActiveListeningHint] = []
    uncertainty_flags: list[str] = []
    for phrase in phrases:
        if not _hint_value_is_safe(phrase):
            continue
        phrase_lower = phrase.lower()
        if any(marker in phrase_lower or marker in phrase for marker in _UNCERTAINTY_MARKERS):
            uncertainty_flags.append("uncertain_reference")
        if any(marker in phrase_lower or marker in phrase for marker in _CORRECTION_MARKERS):
            corrections.append(
                ActiveListeningHint(kind="correction", value=phrase, source=hint_source)
            )
            continue
        if any(marker in phrase_lower or marker in phrase for marker in _CONSTRAINT_MARKERS):
            constraints.append(
                ActiveListeningHint(kind="constraint", value=phrase, source=hint_source)
            )
            continue
        if cjk_present:
            topic = phrase
            for filler in _ZH_TOPIC_FILLERS:
                topic = topic.replace(filler, "")
            topic = _clean_hint_text(topic, limit=32)
            if len(topic) >= 2:
                topics.append(ActiveListeningHint(kind="topic", value=topic, source=hint_source))
            continue
        words = [
            word.lower()
            for word in _EN_WORD_RE.findall(phrase)
            if word.lower() not in _EN_STOPWORDS
        ]
        if words:
            topics.append(
                ActiveListeningHint(
                    kind="topic",
                    value=" ".join(words[:4])[:32],
                    source=hint_source,
                )
            )

    if not topics and cjk_present:
        compact = _clean_hint_text(transcript, limit=32)
        for filler in _ZH_TOPIC_FILLERS:
            compact = compact.replace(filler, "")
        compact = _clean_hint_text(compact, limit=32)
        if len(compact) >= 2 and not any(marker in compact for marker in _CONSTRAINT_MARKERS):
            topics.append(ActiveListeningHint(kind="topic", value=compact, source=hint_source))

    project_references = [
        ActiveListeningHint(
            kind="project_reference",
            value=_clean_project_reference(match.group(0)),
            source=hint_source,
        )
        for match in _PROJECT_RE.finditer(transcript)
    ]

    deduped_topics = _dedupe_hints(topics)
    deduped_constraints = _dedupe_hints(constraints)
    deduped_corrections = _dedupe_hints(corrections)
    deduped_project_references = _dedupe_hints(project_references)
    deduped_uncertainty_flags = _dedupe_strings(uncertainty_flags)
    reason_codes = ["active_listening_understanding:v1"]
    if deduped_topics:
        reason_codes.append("active_listening:topics_detected")
    if deduped_constraints:
        reason_codes.append("active_listening:constraints_detected")
    if deduped_corrections:
        reason_codes.append("active_listening:corrections_detected")
    if deduped_project_references:
        reason_codes.append("active_listening:project_references_detected")
    if deduped_uncertainty_flags:
        reason_codes.append("active_listening:uncertainty_detected")
    if not any(
        (
            deduped_topics,
            deduped_constraints,
            deduped_corrections,
            deduped_project_references,
            deduped_uncertainty_flags,
        )
    ):
        reason_codes.append("active_listening:no_hints")
    return ActiveListeningUnderstanding(
        topics=deduped_topics,
        constraints=deduped_constraints,
        corrections=deduped_corrections,
        project_references=deduped_project_references,
        uncertainty_flags=deduped_uncertainty_flags,
        reason_codes=tuple(reason_codes),
    )


def build_semantic_listener_state_v3(
    *,
    language: object = "unknown",
    understanding: ActiveListeningUnderstanding | dict[str, Any] | None = None,
    topics: object = None,
    constraints: object = None,
    corrections: object = None,
    project_references: object = None,
    uncertainty_flags: object = None,
    source_text: object = None,
    safe_live_summary: object = None,
    partial_available: bool = False,
    final_available: bool = False,
    partial_transcript_chars: int = 0,
    final_transcript_chars: int = 0,
    turn_duration_ms: int | None = None,
    ready_to_answer: bool = False,
    readiness_state: object = "not_ready",
    camera_scene: object = None,
    memory_context: object = None,
    floor_state: object = None,
    reason_codes: tuple[str, ...] | list[str] = (),
) -> SemanticListenerStateV3:
    """Build public-safe semantic listener state from bounded live signals."""
    normalized_language = _semantic_language(language)
    topic_values = _hint_values(
        _understanding_field(understanding, "topics", topics),
        kind="topic",
    )
    constraint_values = _hint_values(
        _understanding_field(understanding, "constraints", constraints),
        kind="constraint",
    )
    correction_values = _hint_values(
        _understanding_field(understanding, "corrections", corrections),
        kind="correction",
    )
    project_values = _hint_values(
        _understanding_field(understanding, "project_references", project_references),
        kind="project_reference",
    )
    uncertainty = _dedupe_strings(
        [
            str(item)
            for item in (
                _understanding_field(understanding, "uncertainty_flags", uncertainty_flags)
                or []
            )
        ],
        limit=5,
    )
    source = _safe_live_summary(source_text, limit=128)
    detection_text = " ".join(
        part
        for part in (
            source,
            " ".join(topic_values),
            " ".join(constraint_values),
            " ".join(correction_values),
            " ".join(project_values),
        )
        if part
    )
    intent = _detect_semantic_intent(
        detection_text,
        language=normalized_language,
        constraints=constraint_values,
        corrections=correction_values,
        project_references=project_values,
    )
    camera_state = _semantic_camera_reference_state(
        camera_scene,
        detection_text=detection_text,
        intent=intent,
    )
    memory_state = _semantic_memory_context_state(memory_context)
    floor = _semantic_floor_state(floor_state)
    enough_information = bool(ready_to_answer or final_available)
    current_topic = (
        topic_values[0]
        if topic_values
        else project_values[0]
        if project_values
        else "unknown"
    )
    live_summary = _safe_live_summary(safe_live_summary)
    if not live_summary:
        live_summary = _default_semantic_summary(
            language=normalized_language,
            current_topic=current_topic,
            intent=intent,
        )
    chips = _build_semantic_chips(
        language=normalized_language,
        current_topic=current_topic,
        constraints=constraint_values,
        intent=intent,
        camera_reference_state=camera_state,
        enough_information_to_answer=enough_information,
    )
    semantic_reasons = [
        "semantic_listener:v3",
        f"semantic_listener:intent:{intent}",
        f"semantic_listener:camera:{camera_state}",
        f"semantic_listener:memory:{memory_state}",
        f"semantic_listener:floor:{floor}",
        f"semantic_listener:readiness:{_clean_hint_text(readiness_state, limit=32) or 'not_ready'}",
        *list(reason_codes or ()),
    ]
    if partial_available:
        semantic_reasons.append("semantic_listener:partial_available")
    if final_available:
        semantic_reasons.append("semantic_listener:final_available")
    if turn_duration_ms is not None and int(turn_duration_ms or 0) >= 2500:
        semantic_reasons.append("semantic_listener:long_turn")
    if partial_transcript_chars or final_transcript_chars:
        semantic_reasons.append("semantic_listener:text_observed")
    return SemanticListenerStateV3(
        current_topic=current_topic,
        detected_intent=intent,
        constraints=constraint_values,
        uncertainty=uncertainty,
        emotional_tone_label=_semantic_tone_label(detection_text, uncertainty=uncertainty),
        language=normalized_language,
        enough_information_to_answer=enough_information,
        safe_live_summary=live_summary,
        listener_chips=chips,
        camera_reference_state=camera_state,
        memory_context_state=memory_state,
        floor_state=floor,
        reason_codes=tuple(_dedupe_strings(semantic_reasons, limit=24)),
    )


def unavailable_active_listening_snapshot(*reason_codes: str) -> dict[str, Any]:
    """Return the public unavailable active-listening payload."""
    codes = ("active_listening:v1", *(reason_codes or ("active_listening:unavailable",)))
    return BrowserActiveListeningSnapshot(
        available=False,
        phase=BrowserActiveListeningPhase.IDLE,
        reason_codes=codes,
    ).as_dict()


def unavailable_active_listener_state_v2(
    *,
    profile: str = "manual",
    language: str = "unknown",
    reason_codes: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Return a public unavailable active-listener v2 payload."""
    return ActiveListenerStateV2(
        profile=profile,
        language=language,
        available=False,
        phase=ActiveListenerPhaseV2.IDLE,
        reason_codes=("active_listener:v2", *(reason_codes or ("active_listener:unavailable",))),
    ).as_dict()


def _active_listener_phase_value(value: object) -> str:
    raw = str(getattr(value, "value", value) or ActiveListenerPhaseV2.IDLE.value)
    try:
        return ActiveListenerPhaseV2(raw).value
    except ValueError:
        return ActiveListenerPhaseV2.IDLE.value


def _clean_project_reference(value: object) -> str:
    text = _clean_hint_text(value, limit=48)
    aliases = {
        "active listener v2": "Active Listener v2",
        "actor state": "actor-state",
        "blink": "Blink",
        "kokoro": "Kokoro",
        "local-http-wav": "local-http-wav",
        "melo": "Melo",
        "melotts": "MeloTTS",
        "moondream": "Moondream",
        "webrtc": "WebRTC",
    }
    return aliases.get(text.lower(), text)


def _safe_degradation_payload(value: object, *, phase: str) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    state = _clean_hint_text(raw.get("state"), limit=32) or (
        "error" if phase == ActiveListenerPhaseV2.ERROR.value else "ok"
    )
    if state not in {"ok", "degraded", "error"}:
        state = "ok"
    components = [
        component
        for component in _dedupe_strings(
            [
                str(item)
                for item in (
                    raw.get("components") if isinstance(raw.get("components"), list) else []
                )
            ],
            limit=5,
        )
        if component in {"microphone", "stt", "runtime", "media"}
    ]
    reason_codes = _dedupe_strings(
        [
            str(item)
            for item in (
                raw.get("reason_codes") if isinstance(raw.get("reason_codes"), list) else []
            )
        ]
        or [f"active_listener_degradation:{state}"],
        limit=8,
    )
    return {
        "state": state,
        "components": components,
        "reason_codes": list(reason_codes),
    }


def _semantic_language(value: object) -> str:
    raw = _clean_hint_text(value, limit=16).lower()
    if raw.startswith("zh") or raw in {"cn", "chinese"}:
        return "zh"
    if raw.startswith("en") or raw == "english":
        return "en"
    return "unknown"


def _semantic_chip_labels(chip_id: str, *, language: str = "unknown") -> tuple[str, str]:
    english, chinese = _CHIP_LABELS.get(chip_id, _CHIP_LABELS["still_listening"])
    if _semantic_language(language) == "zh":
        return chinese, chinese
    return english, chinese


def _dedupe_semantic_chips(
    chips: tuple[SemanticListenerChipV3, ...] | list[SemanticListenerChipV3],
    *,
    language: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for chip in chips:
        if isinstance(chip, dict):
            chip_id = _clean_hint_text(chip.get("chip_id"), limit=48)
            label = _clean_hint_text(chip.get("label"), limit=48)
            localized_label = _clean_hint_text(chip.get("localized_label"), limit=48)
            state = _clean_hint_text(chip.get("state"), limit=24) or "active"
            reason_codes = tuple(str(item) for item in chip.get("reason_codes", [])[:8])
            chip = SemanticListenerChipV3(
                chip_id=chip_id,
                label=label,
                localized_label=localized_label,
                state=state,
                reason_codes=reason_codes,
            )
        if not isinstance(chip, SemanticListenerChipV3):
            continue
        chip_id = chip.chip_id if chip.chip_id in _SEMANTIC_CHIP_IDS else "still_listening"
        if chip_id in seen:
            continue
        seen.add(chip_id)
        result.append(chip.as_dict(language=language))
        if len(result) >= 7:
            break
    return result


def _safe_semantic_state_label(value: object, *, allowed: set[str], default: str) -> str:
    text = _clean_hint_text(value, limit=48).lower()
    return text if text in allowed else default


def _safe_live_summary(value: object, *, limit: int = 96) -> str:
    text = _clean_hint_text(value, limit=limit)
    lowered = text.lower()
    if any(fragment in lowered for fragment in _UNSAFE_HINT_FRAGMENTS):
        return ""
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return ""
    return text[:limit]


def _semantic_listener_state_v3_payload(
    payload: SemanticListenerStateV3 | dict[str, Any] | None,
    *,
    language: object,
    topics: tuple[ActiveListeningHint, ...],
    constraints: tuple[ActiveListeningHint, ...],
    corrections: tuple[ActiveListeningHint, ...],
    project_references: tuple[ActiveListeningHint, ...],
    uncertainty_flags: tuple[str, ...],
    partial_available: bool,
    final_available: bool,
    partial_transcript_chars: int,
    final_transcript_chars: int,
    turn_duration_ms: int | None,
    ready_to_answer: bool,
    readiness_state: object,
    floor_state: object,
    reason_codes: tuple[str, ...],
) -> dict[str, object]:
    if isinstance(payload, SemanticListenerStateV3):
        return payload.as_dict()
    if isinstance(payload, dict):
        chips = tuple(
            SemanticListenerChipV3(
                chip_id=_clean_hint_text(item.get("chip_id"), limit=48),
                label=_clean_hint_text(item.get("label"), limit=48),
                localized_label=_clean_hint_text(item.get("localized_label"), limit=48),
                state=_clean_hint_text(item.get("state"), limit=24) or "active",
                reason_codes=tuple(str(code) for code in item.get("reason_codes", [])[:8])
                if isinstance(item, dict)
                else (),
            )
            for item in (payload.get("listener_chips") if isinstance(payload.get("listener_chips"), list) else [])
            if isinstance(item, dict)
        )
        return SemanticListenerStateV3(
            current_topic=_clean_hint_text(payload.get("current_topic"), limit=48) or "unknown",
            detected_intent=_safe_semantic_state_label(
                payload.get("detected_intent"),
                allowed=_SEMANTIC_INTENTS,
                default="unknown",
            ),
            constraints=tuple(
                _dedupe_strings(
                    [str(item) for item in payload.get("constraints", [])]
                    if isinstance(payload.get("constraints"), list)
                    else [],
                    limit=5,
                )
            ),
            uncertainty=tuple(
                _dedupe_strings(
                    [str(item) for item in payload.get("uncertainty", [])]
                    if isinstance(payload.get("uncertainty"), list)
                    else [],
                    limit=5,
                )
            ),
            emotional_tone_label=_safe_semantic_state_label(
                payload.get("emotional_tone_label"),
                allowed={"neutral", "positive", "concerned", "frustrated", "uncertain"},
                default="neutral",
            ),
            language=_semantic_language(payload.get("language") or language),
            enough_information_to_answer=bool(payload.get("enough_information_to_answer")),
            safe_live_summary=_safe_live_summary(payload.get("safe_live_summary")),
            listener_chips=chips,
            camera_reference_state=_safe_semantic_state_label(
                payload.get("camera_reference_state"),
                allowed=_SEMANTIC_CAMERA_STATES,
                default="not_used",
            ),
            memory_context_state=_safe_semantic_state_label(
                payload.get("memory_context_state"),
                allowed={"available", "limited", "unavailable"},
                default="unavailable",
            ),
            floor_state=_clean_hint_text(payload.get("floor_state"), limit=48) or "unknown",
            reason_codes=tuple(
                _dedupe_strings(
                    [str(item) for item in payload.get("reason_codes", [])]
                    if isinstance(payload.get("reason_codes"), list)
                    else [],
                    limit=24,
                )
            ),
        ).as_dict()
    return build_semantic_listener_state_v3(
        language=language,
        topics=topics,
        constraints=constraints,
        corrections=corrections,
        project_references=project_references,
        uncertainty_flags=uncertainty_flags,
        partial_available=partial_available,
        final_available=final_available,
        partial_transcript_chars=partial_transcript_chars,
        final_transcript_chars=final_transcript_chars,
        turn_duration_ms=turn_duration_ms,
        ready_to_answer=ready_to_answer,
        readiness_state=readiness_state,
        floor_state=floor_state,
        reason_codes=reason_codes,
    ).as_dict()


def _understanding_field(
    understanding: ActiveListeningUnderstanding | dict[str, Any] | None,
    field_name: str,
    fallback: object,
) -> object:
    if isinstance(understanding, ActiveListeningUnderstanding):
        return getattr(understanding, field_name)
    if isinstance(understanding, dict) and field_name in understanding:
        return understanding.get(field_name)
    return fallback


def _hint_values(value: object, *, kind: str) -> tuple[str, ...]:
    raw_values = value if isinstance(value, (list, tuple)) else []
    values: list[str] = []
    for item in raw_values:
        if isinstance(item, ActiveListeningHint):
            if item.kind != kind:
                continue
            values.append(item.value)
        elif isinstance(item, dict):
            if str(item.get("kind") or kind) != kind:
                continue
            values.append(str(item.get("value") or ""))
        else:
            values.append(str(item))
    return _dedupe_strings(values, limit=5)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered or marker in text for marker in markers)


def _detect_semantic_intent(
    text: str,
    *,
    language: str,
    constraints: tuple[str, ...],
    corrections: tuple[str, ...],
    project_references: tuple[str, ...],
) -> str:
    if corrections or _contains_marker(text, _CORRECTION_MARKERS):
        return "correction"
    if _contains_marker(text, _OBJECT_SHOWING_MARKERS):
        return "object_showing"
    if project_references or _contains_marker(text, _PROJECT_PLANNING_MARKERS):
        return "project_planning"
    if _contains_marker(text, _QUESTION_MARKERS):
        return "question"
    if constraints or _contains_marker(text, _INSTRUCTION_MARKERS):
        return "instruction"
    compact = _SPACE_RE.sub("", text.lower())
    if _contains_marker(text, _SMALL_TALK_MARKERS) and len(compact) <= (10 if language == "zh" else 24):
        return "small_talk"
    return "unknown"


def _semantic_tone_label(text: str, *, uncertainty: tuple[str, ...]) -> str:
    if _contains_marker(text, _FRUSTRATED_TONE_MARKERS):
        return "frustrated"
    if uncertainty or _contains_marker(text, _CONCERN_TONE_MARKERS):
        return "uncertain"
    if _contains_marker(text, _POSITIVE_TONE_MARKERS):
        return "positive"
    return "neutral"


def _semantic_camera_reference_state(
    camera_scene: object,
    *,
    detection_text: str,
    intent: str,
) -> str:
    scene = camera_scene if isinstance(camera_scene, dict) else {}
    raw_state = str(scene.get("state") or scene.get("status") or "").lower()
    freshness = str(
        scene.get("freshness_state")
        or scene.get("freshness")
        or scene.get("frame_state")
        or ""
    ).lower()
    grounding_mode = str(scene.get("grounding_mode") or scene.get("grounding") or "").lower()
    reason_codes = " ".join(str(code).lower() for code in scene.get("reason_codes", [])[:8]) if isinstance(scene.get("reason_codes"), list) else ""
    current_answer_used_vision = bool(scene.get("current_answer_used_vision"))
    stale_or_limited = (
        raw_state in {"disabled", "limited", "stale", "unavailable", "unknown"}
        or freshness in {"stale", "limited", "expired"}
        or grounding_mode in {"disabled", "unavailable", "stale"}
        or "stale" in reason_codes
        or "limited" in reason_codes
    )
    fresh = (
        not stale_or_limited
        and (
            current_answer_used_vision
            or freshness in {"fresh", "current", "fresh_used"}
            or raw_state in {"fresh", "ready", "receiving"}
            or grounding_mode not in {"", "none", "disabled", "unavailable", "stale"}
        )
    )
    error = raw_state == "error" or "error" in reason_codes
    object_referenced = intent == "object_showing" or _contains_marker(
        detection_text,
        _OBJECT_SHOWING_MARKERS,
    )
    if error:
        return "error"
    if object_referenced:
        return "fresh_supported" if fresh else "stale_or_limited"
    if fresh:
        return "fresh_available"
    if stale_or_limited:
        return "stale_or_limited"
    return "not_used"


def _semantic_memory_context_state(memory_context: object) -> str:
    if not isinstance(memory_context, dict):
        return "unavailable"
    if memory_context.get("available") is True or memory_context.get("state") in {
        "available",
        "ready",
        "used",
    }:
        return "available"
    if memory_context.get("state") in {"limited", "degraded"}:
        return "limited"
    return "unavailable"


def _semantic_floor_state(floor_state: object) -> str:
    if isinstance(floor_state, dict):
        for key in ("floor_sub_state", "status", "state", "current_state"):
            value = _clean_hint_text(floor_state.get(key), limit=48)
            if value:
                return value
    return _clean_hint_text(floor_state, limit=48) or "unknown"


def _default_semantic_summary(*, language: str, current_topic: str, intent: str) -> str:
    if current_topic and current_topic != "unknown":
        if language == "zh":
            return _safe_live_summary(f"我听到：{current_topic}")
        return _safe_live_summary(f"I heard: {current_topic}")
    if intent != "unknown":
        if language == "zh":
            return _safe_live_summary(f"识别到：{intent}")
        return _safe_live_summary(f"Detected: {intent}")
    return "继续听" if language == "zh" else "still listening"


def _build_semantic_chips(
    *,
    language: str,
    current_topic: str,
    constraints: tuple[str, ...],
    intent: str,
    camera_reference_state: str,
    enough_information_to_answer: bool,
) -> tuple[SemanticListenerChipV3, ...]:
    chip_ids: list[str] = []
    if current_topic and current_topic != "unknown":
        chip_ids.append("heard_summary")
    if constraints:
        chip_ids.append("constraint_detected")
    if intent == "question":
        chip_ids.append("question_detected")
    if intent == "object_showing" and camera_reference_state == "fresh_supported":
        chip_ids.append("showing_object")
    elif intent == "object_showing":
        chip_ids.append("camera_limited")
    chip_ids.append("ready_to_answer" if enough_information_to_answer else "still_listening")
    chips: list[SemanticListenerChipV3] = []
    for chip_id in chip_ids:
        label, localized_label = _semantic_chip_labels(chip_id, language=language)
        state = "ready" if chip_id == "ready_to_answer" else "limited" if chip_id == "camera_limited" else "active"
        chips.append(
            SemanticListenerChipV3(
                chip_id=chip_id,
                label=label,
                localized_label=localized_label,
                state=state,
            )
        )
    return tuple(chips)

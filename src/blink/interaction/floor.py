"""Deterministic conversation floor state for browser turn-taking."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import RLock
from typing import Any

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_TEXT_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)
_MAX_REASON_CODES = 24
_FLOOR_MODEL_VERSION = 3
_LOW_CONFIDENCE_THRESHOLD = 0.35


class ConversationFloorStatus(str, Enum):
    """Public turn-taking state for the current conversation floor."""

    USER_HAS_FLOOR = "user_has_floor"
    ASSISTANT_HAS_FLOOR = "assistant_has_floor"
    OVERLAP = "overlap"
    HANDOFF = "handoff"
    REPAIR = "repair"
    UNKNOWN = "unknown"


class ConversationFloorInputType(str, Enum):
    """Symbolic low-level input consumed by the floor controller."""

    VAD_USER_STARTED = "vad_user_started"
    VAD_USER_CONTINUING = "vad_user_continuing"
    VAD_USER_STOPPED = "vad_user_stopped"
    STT_INTERIM = "stt_interim"
    STT_FINAL = "stt_final"
    LLM_STARTED = "llm_started"
    LLM_ENDED = "llm_ended"
    TTS_STARTED = "tts_started"
    TTS_STOPPED = "tts_stopped"
    INTERRUPTION_CANDIDATE = "interruption_candidate"
    INTERRUPTION_ACCEPTED = "interruption_accepted"
    INTERRUPTION_REJECTED = "interruption_rejected"
    INTERRUPTION_SUPPRESSED = "interruption_suppressed"
    INTERRUPTION_RESUMED = "interruption_resumed"


class ConversationFloorTextKind(str, Enum):
    """Safe text classification labels used by the floor controller."""

    EMPTY = "empty"
    BACKCHANNEL = "backchannel"
    CONFIRMATION = "confirmation"
    HESITATION = "hesitation"
    CORRECTION = "correction"
    EXPLICIT_INTERRUPTION = "explicit_interruption"
    MEANINGFUL = "meaningful"


class ConversationFloorSubStateV3(str, Enum):
    """Detailed v3 turn-taking sub-state carried inside the v1 payload."""

    USER_HOLDING_FLOOR = "user_holding_floor"
    ASSISTANT_HOLDING_FLOOR = "assistant_holding_floor"
    OVERLAP_CANDIDATE = "overlap_candidate"
    ACCEPTED_INTERRUPT = "accepted_interrupt"
    IGNORED_BACKCHANNEL = "ignored_backchannel"
    REPAIR_REQUESTED = "repair_requested"
    HANDOFF_PENDING = "handoff_pending"
    HANDOFF_COMPLETE = "handoff_complete"


_SHORT_BACKCHANNELS = {
    "嗯",
    "嗯嗯",
    "对",
    "对的",
    "好",
    "好的",
    "行",
    "可以",
    "是",
    "是的",
    "ok",
    "okay",
    "yes",
    "yeah",
    "yep",
    "right",
    "sure",
    "mm",
    "mhm",
    "uhhuh",
}
_CONTINUER_PHRASES = {
    "继续",
    "接着说",
    "说下去",
    "goon",
    "goahead",
    "continue",
    "keepgoing",
}
_HESITATION_PHRASES = {
    "呃",
    "额",
    "那个",
    "我想想",
    "让我想想",
    "稍等我想想",
    "uh",
    "um",
    "umm",
    "hmm",
    "letmethink",
}
_CORRECTION_PHRASES = {
    "不是",
    "不对",
    "错了",
    "我不是说",
    "应该是",
    "纠正",
    "actually",
    "imean",
    "notthat",
    "thatsnot",
    "no",
    "nope",
    "correction",
}
_EXPLICIT_INTERRUPTION_PHRASES = {
    "等一下",
    "等等",
    "停一下",
    "先别说",
    "打断一下",
    "我打断一下",
    "暂停",
    "wait",
    "waitasecond",
    "holdon",
    "hangon",
    "stop",
    "pause",
    "letmeinterrupt",
    "interrupting",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return max(0.0, min(1.0, parsed))


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < _LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


def _dedupe_reason_codes(values: object, *, limit: int = _MAX_REASON_CODES) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        code = _safe_token(raw_value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def normalized_floor_text(value: str) -> str:
    """Normalize user-visible text for deterministic floor classification."""
    return _TEXT_TOKEN_RE.sub("", str(value or "").strip().lower())[:96]


def _has_phrase(normalized_text: str, phrases: set[str]) -> bool:
    return any(phrase and phrase in normalized_text for phrase in phrases)


@dataclass(frozen=True)
class ConversationFloorPhraseClassification:
    """Public-safe bounded phrase classification for floor policy."""

    text_kind: ConversationFloorTextKind | str
    phrase_class: str
    phrase_confidence: float | None = None
    confidence_bucket: str = "unknown"
    low_confidence_transcript: bool = False
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the classification without raw text or matched phrase."""
        text_kind = _safe_token(self.text_kind, default="empty", limit=64)
        confidence = _safe_float(self.phrase_confidence)
        return {
            "text_kind": text_kind,
            "phrase_class": _safe_token(self.phrase_class, default=text_kind, limit=64),
            "phrase_confidence": round(confidence, 3) if confidence is not None else None,
            "phrase_confidence_bucket": _safe_token(
                self.confidence_bucket,
                default="unknown",
                limit=32,
            ),
            "low_confidence_transcript": self.low_confidence_transcript is True,
            "reason_codes": _dedupe_reason_codes(self.reason_codes, limit=8),
        }


def classify_floor_phrase(
    value: str,
    *,
    transcript_confidence: float | None = None,
) -> ConversationFloorPhraseClassification:
    """Classify bounded in-memory text into public-safe floor labels."""
    normalized = normalized_floor_text(value)
    stt_confidence = _safe_float(transcript_confidence)
    low_confidence = (
        stt_confidence is not None and stt_confidence < _LOW_CONFIDENCE_THRESHOLD
    )
    match_confidence = 1.0
    text_kind = ConversationFloorTextKind.EMPTY
    phrase_class = "empty"
    reasons: list[str] = ["floor_phrase:empty"]
    if not normalized:
        pass
    elif normalized in _SHORT_BACKCHANNELS:
        match_confidence = 0.95
        if normalized in {"ok", "okay", "yes", "yeah", "yep", "right", "sure", "是", "是的"}:
            text_kind = ConversationFloorTextKind.CONFIRMATION
            phrase_class = "confirmation"
        else:
            text_kind = ConversationFloorTextKind.BACKCHANNEL
            phrase_class = "backchannel"
        reasons = ["floor_phrase:backchannel", "short_backchannel"]
    elif normalized in _CONTINUER_PHRASES:
        match_confidence = 0.95
        text_kind = ConversationFloorTextKind.CONFIRMATION
        phrase_class = "continuer"
        reasons = ["floor_phrase:continuer", "user_continuing"]
    elif _has_phrase(normalized, _EXPLICIT_INTERRUPTION_PHRASES):
        match_confidence = 0.88
        text_kind = ConversationFloorTextKind.EXPLICIT_INTERRUPTION
        phrase_class = "explicit_interrupt"
        reasons = ["floor_phrase:explicit_interrupt", "explicit_interrupt"]
    elif _has_phrase(normalized, _CORRECTION_PHRASES):
        match_confidence = 0.88
        text_kind = ConversationFloorTextKind.CORRECTION
        phrase_class = "correction"
        reasons = ["floor_phrase:correction", "correction"]
    elif normalized in _HESITATION_PHRASES or _has_phrase(normalized, _HESITATION_PHRASES):
        match_confidence = 0.82
        text_kind = ConversationFloorTextKind.HESITATION
        phrase_class = "hesitation"
        reasons = ["floor_phrase:hesitation"]
    else:
        match_confidence = 0.6
        text_kind = ConversationFloorTextKind.MEANINGFUL
        phrase_class = "meaningful"
        reasons = ["floor_phrase:meaningful"]
    confidence = min(match_confidence, stt_confidence) if stt_confidence is not None else match_confidence
    if low_confidence:
        reasons.append("low_confidence_transcript")
    return ConversationFloorPhraseClassification(
        text_kind=text_kind,
        phrase_class=phrase_class,
        phrase_confidence=confidence,
        confidence_bucket=_confidence_bucket(confidence),
        low_confidence_transcript=low_confidence,
        reason_codes=tuple(reasons),
    )


def classify_floor_text(value: str) -> ConversationFloorTextKind:
    """Classify bounded in-memory text without returning the text itself."""
    classification = classify_floor_phrase(value)
    try:
        if isinstance(classification.text_kind, ConversationFloorTextKind):
            return classification.text_kind
        return ConversationFloorTextKind(str(classification.text_kind))
    except ValueError:
        return ConversationFloorTextKind.MEANINGFUL


def is_short_backchannel(value: str) -> bool:
    """Return whether text is a short continue/backchannel signal."""
    classification = classify_floor_phrase(value)
    return classification.phrase_class in {"backchannel", "confirmation", "continuer"}


def is_explicit_interruption_phrase(value: str) -> bool:
    """Return whether text is an explicit interruption phrase."""
    return classify_floor_text(value) == ConversationFloorTextKind.EXPLICIT_INTERRUPTION


@dataclass(frozen=True)
class ConversationFloorInput:
    """One deterministic input signal for the conversation floor controller."""

    input_type: ConversationFloorInputType | str
    profile: str | None = None
    language: str | None = None
    timestamp: str = field(default_factory=_now_iso)
    text: str | None = None
    text_chars: int = 0
    assistant_speaking: bool | None = None
    user_speaking: bool | None = None
    protected_playback: bool | None = None
    barge_in_armed: bool | None = None
    echo_safe: bool | None = None
    transcript_confidence: float | None = None
    tts_chunk_role: str | None = None
    echo_risk: str | None = None
    barge_in_state: str | None = None
    speech_age_ms: int = 0
    turn_duration_ms: int = 0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationFloorState:
    """Public-safe snapshot of explicit turn ownership."""

    state: ConversationFloorStatus | str
    profile: str
    language: str
    assistant_speaking: bool = False
    user_speaking: bool = False
    protected_playback: bool = True
    barge_in_armed: bool = False
    floor_model_version: int = _FLOOR_MODEL_VERSION
    sub_state: ConversationFloorSubStateV3 | str = ConversationFloorSubStateV3.HANDOFF_COMPLETE
    last_input_type: str = "none"
    last_text_kind: str = "empty"
    phrase_class: str = "empty"
    phrase_confidence: float | None = 1.0
    phrase_confidence_bucket: str = "high"
    yield_decision: str = "wait_for_input"
    echo_risk: str = "unknown"
    barge_in_state: str = "protected"
    tts_chunk_role: str = "unknown"
    low_confidence_transcript: bool = False
    last_transition_at: str | None = None
    updated_at: str = field(default_factory=_now_iso)
    transition_count: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    schema_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return a public-safe floor state payload."""
        state = _coerce_status(self.state).value
        reason_codes = _dedupe_reason_codes(
            [
                "conversation_floor:v1",
                "conversation_floor:v3",
                f"floor_state:{state}",
                f"floor_sub_state:{self.sub_state}",
                f"floor_input:{self.last_input_type}",
                f"floor_text_kind:{self.last_text_kind}",
                f"floor_phrase_class:{self.phrase_class}",
                f"floor_yield:{self.yield_decision}",
                *self.reason_codes,
            ]
        )
        phrase_confidence = _safe_float(self.phrase_confidence)
        if phrase_confidence is None:
            phrase_confidence = 1.0 if self.last_text_kind == "empty" else 0.0
        return {
            "schema_version": 1,
            "state": state,
            "profile": _safe_token(self.profile, default="manual", limit=96),
            "language": _safe_token(self.language, default="unknown", limit=32),
            "floor_model_version": _FLOOR_MODEL_VERSION,
            "sub_state": _safe_token(self.sub_state, default="handoff_complete", limit=64),
            "user_has_floor": state == ConversationFloorStatus.USER_HAS_FLOOR.value,
            "assistant_has_floor": state == ConversationFloorStatus.ASSISTANT_HAS_FLOOR.value,
            "overlap": state == ConversationFloorStatus.OVERLAP.value,
            "handoff": state == ConversationFloorStatus.HANDOFF.value,
            "repair": state == ConversationFloorStatus.REPAIR.value,
            "unknown": state == ConversationFloorStatus.UNKNOWN.value,
            "assistant_speaking": self.assistant_speaking is True,
            "user_speaking": self.user_speaking is True,
            "protected_playback": self.protected_playback is True,
            "barge_in_armed": self.barge_in_armed is True,
            "last_input_type": _safe_token(self.last_input_type, default="none", limit=64),
            "last_text_kind": _safe_token(self.last_text_kind, default="empty", limit=64),
            "phrase_class": _safe_token(self.phrase_class, default="empty", limit=64),
            "phrase_confidence": round(phrase_confidence, 3),
            "phrase_confidence_bucket": _safe_token(
                self.phrase_confidence_bucket,
                default="unknown",
                limit=32,
            ),
            "yield_decision": _safe_token(
                self.yield_decision,
                default="wait_for_input",
                limit=64,
            ),
            "echo_risk": _safe_token(self.echo_risk, default="unknown", limit=32),
            "barge_in_state": _safe_token(self.barge_in_state, default="protected", limit=32),
            "tts_chunk_role": _safe_token(self.tts_chunk_role, default="unknown", limit=64),
            "low_confidence_transcript": self.low_confidence_transcript is True,
            "last_transition_at": self.last_transition_at,
            "updated_at": self.updated_at,
            "transition_count": _safe_int(self.transition_count),
            "counts": {
                _safe_token(key, default="unknown", limit=64): _safe_int(value)
                for key, value in sorted(self.counts.items())
            },
            "reason_codes": reason_codes,
        }


@dataclass(frozen=True)
class ConversationFloorUpdate:
    """Result of applying one floor input."""

    previous_state: ConversationFloorStatus
    state: ConversationFloorState
    changed: bool

    def event_metadata(self) -> dict[str, object]:
        """Return public-safe metadata for a floor transition event."""
        payload = self.state.as_dict()
        return {
            "floor_state": payload["state"],
            "floor_sub_state": payload["sub_state"],
            "floor_model_version": payload["floor_model_version"],
            "previous_floor_state": self.previous_state.value,
            "input_type": payload["last_input_type"],
            "text_kind": payload["last_text_kind"],
            "phrase_class": payload["phrase_class"],
            "phrase_confidence": payload["phrase_confidence"],
            "phrase_confidence_bucket": payload["phrase_confidence_bucket"],
            "yield_decision": payload["yield_decision"],
            "transition_count": payload["transition_count"],
            "assistant_speaking": payload["assistant_speaking"],
            "user_speaking": payload["user_speaking"],
            "barge_in_armed": payload["barge_in_armed"],
            "protected_playback": payload["protected_playback"],
            "echo_risk": payload["echo_risk"],
            "barge_in_state": payload["barge_in_state"],
            "low_confidence_transcript": payload["low_confidence_transcript"],
        }


def _coerce_status(value: ConversationFloorStatus | str) -> ConversationFloorStatus:
    try:
        if isinstance(value, ConversationFloorStatus):
            return value
        return ConversationFloorStatus(str(value))
    except ValueError:
        return ConversationFloorStatus.UNKNOWN


def _coerce_input_type(value: ConversationFloorInputType | str) -> ConversationFloorInputType:
    try:
        if isinstance(value, ConversationFloorInputType):
            return value
        return ConversationFloorInputType(str(value))
    except ValueError:
        return ConversationFloorInputType.STT_INTERIM


class ConversationFloorController:
    """Deterministic controller for browser conversation floor transitions."""

    def __init__(
        self,
        *,
        profile: str = "manual",
        language: str = "unknown",
        protected_playback: bool = True,
        barge_in_armed: bool = False,
        echo_safe: bool = False,
    ):
        """Initialize the controller."""
        self._profile = _safe_token(profile, default="manual", limit=96)
        self._language = _safe_token(language, default="unknown", limit=32)
        self._protected_playback = bool(protected_playback)
        self._barge_in_armed = bool(barge_in_armed)
        self._echo_safe = bool(echo_safe)
        self._echo_risk = "low" if echo_safe else "unknown"
        self._barge_in_state = "armed" if barge_in_armed or not protected_playback else "protected"
        self._state = ConversationFloorStatus.UNKNOWN
        self._sub_state = ConversationFloorSubStateV3.HANDOFF_COMPLETE
        self._assistant_speaking = False
        self._user_speaking = False
        self._last_input_type = "none"
        self._last_text_kind = ConversationFloorTextKind.EMPTY.value
        self._last_phrase_class = "empty"
        self._last_phrase_confidence: float | None = 1.0
        self._last_phrase_confidence_bucket = "high"
        self._yield_decision = "wait_for_input"
        self._tts_chunk_role = "unknown"
        self._low_confidence_transcript = False
        self._last_transition_at: str | None = None
        self._updated_at = _now_iso()
        self._transition_count = 0
        self._counts: Counter[str] = Counter()
        self._reason_codes = ["floor:initialized"]
        self._lock = RLock()

    def configure(
        self,
        *,
        profile: str | None = None,
        language: str | None = None,
        protected_playback: bool | None = None,
        barge_in_armed: bool | None = None,
        echo_safe: bool | None = None,
        echo_risk: str | None = None,
        barge_in_state: str | None = None,
    ) -> None:
        """Update runtime policy labels without changing floor ownership."""
        with self._lock:
            if profile not in (None, ""):
                self._profile = _safe_token(profile, default="manual", limit=96)
            if language not in (None, ""):
                self._language = _safe_token(language, default="unknown", limit=32)
            if protected_playback is not None:
                self._protected_playback = bool(protected_playback)
            if barge_in_armed is not None:
                self._barge_in_armed = bool(barge_in_armed)
            if echo_safe is not None:
                self._echo_safe = bool(echo_safe)
                if self._echo_safe and echo_risk in (None, ""):
                    self._echo_risk = "low"
            if echo_risk not in (None, ""):
                self._echo_risk = _safe_token(echo_risk, default="unknown", limit=32).lower()
            if barge_in_state not in (None, ""):
                state = _safe_token(barge_in_state, default="protected", limit=32).lower()
                self._barge_in_state = (
                    state if state in {"protected", "armed", "adaptive"} else "protected"
                )
            elif protected_playback is not None or barge_in_armed is not None:
                self._barge_in_state = (
                    "armed"
                    if self._barge_in_armed or not self._protected_playback
                    else "protected"
                )

    def snapshot(self) -> ConversationFloorState:
        """Return the current public-safe floor state."""
        with self._lock:
            return self._snapshot_unlocked()

    def apply(self, floor_input: ConversationFloorInput) -> ConversationFloorUpdate:
        """Apply one deterministic input and return the resulting state."""
        with self._lock:
            input_type = _coerce_input_type(floor_input.input_type)
            self.configure(
                profile=floor_input.profile,
                language=floor_input.language,
                protected_playback=floor_input.protected_playback,
                barge_in_armed=floor_input.barge_in_armed,
                echo_safe=floor_input.echo_safe,
                echo_risk=floor_input.echo_risk,
                barge_in_state=floor_input.barge_in_state,
            )
            if floor_input.tts_chunk_role not in (None, ""):
                self._tts_chunk_role = _safe_token(
                    floor_input.tts_chunk_role,
                    default="unknown",
                    limit=64,
                )
            self._update_speaking_flags(input_type, floor_input)
            classification = classify_floor_phrase(
                floor_input.text or "",
                transcript_confidence=floor_input.transcript_confidence,
            )
            text_kind = (
                classification.text_kind
                if isinstance(classification.text_kind, ConversationFloorTextKind)
                else ConversationFloorTextKind(str(classification.text_kind))
            )
            previous_state = self._state
            previous_sub_state = self._sub_state
            next_state, next_sub_state, yield_decision, reasons = self._decide(
                input_type,
                classification,
            )
            self._counts[input_type.value] += 1
            self._last_input_type = input_type.value
            self._last_text_kind = text_kind.value
            self._last_phrase_class = _safe_token(
                classification.phrase_class,
                default=text_kind.value,
                limit=64,
            )
            self._last_phrase_confidence = _safe_float(classification.phrase_confidence)
            self._last_phrase_confidence_bucket = _safe_token(
                classification.confidence_bucket,
                default="unknown",
                limit=32,
            )
            self._yield_decision = _safe_token(
                yield_decision,
                default="wait_for_input",
                limit=64,
            )
            self._low_confidence_transcript = classification.low_confidence_transcript
            self._updated_at = str(floor_input.timestamp or _now_iso())
            changed = next_state != previous_state or next_sub_state != previous_sub_state
            if changed:
                self._state = next_state
                self._sub_state = next_sub_state
                self._last_transition_at = self._updated_at
                self._transition_count += 1
            self._reason_codes = _dedupe_reason_codes(
                [
                    *classification.reason_codes,
                    *reasons,
                    *floor_input.reason_codes,
                    f"floor_policy:{self._barge_in_state}",
                    "floor:state_changed" if changed else "floor:state_unchanged",
                ]
            )
            return ConversationFloorUpdate(
                previous_state=previous_state,
                state=self._snapshot_unlocked(),
                changed=changed,
            )

    def _update_speaking_flags(
        self,
        input_type: ConversationFloorInputType,
        floor_input: ConversationFloorInput,
    ) -> None:
        if floor_input.assistant_speaking is not None:
            self._assistant_speaking = bool(floor_input.assistant_speaking)
        if floor_input.user_speaking is not None:
            self._user_speaking = bool(floor_input.user_speaking)
        if input_type in {
            ConversationFloorInputType.VAD_USER_STARTED,
            ConversationFloorInputType.VAD_USER_CONTINUING,
        }:
            self._user_speaking = True
        elif input_type == ConversationFloorInputType.VAD_USER_STOPPED:
            self._user_speaking = False
        elif input_type == ConversationFloorInputType.TTS_STARTED:
            self._assistant_speaking = True
        elif input_type == ConversationFloorInputType.TTS_STOPPED:
            self._assistant_speaking = False

    def _can_yield_to_user(self) -> bool:
        return (
            self._barge_in_armed
            or self._echo_safe
            or not self._protected_playback
            or self._barge_in_state == "armed"
            or (self._barge_in_state == "adaptive" and self._echo_risk == "low")
        )

    def _policy_reasons(self, *, can_yield: bool) -> list[str]:
        if can_yield:
            return ["floor:yield_available"]
        reasons = ["floor:protected_continue"]
        if self._protected_playback or self._barge_in_state == "protected":
            reasons.append("protected_playback")
        if self._assistant_speaking and self._echo_risk != "low":
            reasons.append("echo_risk")
            reasons.append(f"floor_echo_risk:{self._echo_risk}")
        return reasons

    @staticmethod
    def _decision(
        status: ConversationFloorStatus,
        sub_state: ConversationFloorSubStateV3,
        yield_decision: str,
        reasons: list[str],
    ) -> tuple[
        ConversationFloorStatus,
        ConversationFloorSubStateV3,
        str,
        list[str],
    ]:
        return status, sub_state, yield_decision, _dedupe_reason_codes(reasons)

    def _decide(
        self,
        input_type: ConversationFloorInputType,
        classification: ConversationFloorPhraseClassification,
    ) -> tuple[
        ConversationFloorStatus,
        ConversationFloorSubStateV3,
        str,
        list[str],
    ]:
        can_yield = self._can_yield_to_user()
        if input_type == ConversationFloorInputType.VAD_USER_STARTED:
            if self._assistant_speaking:
                yield_decision = "yield_to_user" if can_yield else "protected_continue"
                return self._decision(
                    ConversationFloorStatus.OVERLAP,
                    ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                    yield_decision,
                    ["floor:overlap_candidate", *self._policy_reasons(can_yield=can_yield)],
                )
            return self._decision(
                ConversationFloorStatus.USER_HAS_FLOOR,
                ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
                "hold_user",
                ["floor:user_turn_started"],
            )
        if input_type == ConversationFloorInputType.VAD_USER_CONTINUING:
            if self._assistant_speaking:
                yield_decision = "yield_to_user" if can_yield else "protected_continue"
                return self._decision(
                    ConversationFloorStatus.OVERLAP,
                    ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                    yield_decision,
                    [
                        "floor:overlap_continues",
                        "user_continuing",
                        *self._policy_reasons(can_yield=can_yield),
                    ],
                )
            return self._decision(
                ConversationFloorStatus.USER_HAS_FLOOR,
                ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
                "hold_user",
                ["floor:user_continuing", "user_continuing"],
            )
        if input_type == ConversationFloorInputType.VAD_USER_STOPPED:
            return self._decision(
                ConversationFloorStatus.HANDOFF,
                ConversationFloorSubStateV3.HANDOFF_PENDING,
                "wait_for_assistant",
                ["floor:user_short_pause", "floor:waiting"],
            )

        if input_type in {
            ConversationFloorInputType.STT_INTERIM,
            ConversationFloorInputType.STT_FINAL,
        }:
            return self._decide_from_transcription(input_type, classification, can_yield)

        if input_type == ConversationFloorInputType.LLM_STARTED:
            return self._decision(
                ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR,
                "continue_assistant",
                ["floor:assistant_thinking"],
            )
        if input_type == ConversationFloorInputType.LLM_ENDED:
            return self._decision(
                ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR,
                "continue_assistant",
                ["floor:assistant_response_ready"],
            )
        if input_type == ConversationFloorInputType.TTS_STARTED:
            return self._decision(
                ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR,
                "continue_assistant",
                ["floor:assistant_speaking"],
            )
        if input_type == ConversationFloorInputType.TTS_STOPPED:
            return self._decision(
                ConversationFloorStatus.HANDOFF,
                ConversationFloorSubStateV3.HANDOFF_COMPLETE,
                "handoff_complete",
                ["floor:assistant_finished", "floor:waiting", "assistant_pause"],
            )
        if input_type == ConversationFloorInputType.INTERRUPTION_CANDIDATE:
            yield_decision = "yield_to_user" if can_yield else "protected_continue"
            return self._decision(
                ConversationFloorStatus.OVERLAP,
                ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                yield_decision,
                ["floor:interruption_candidate", *self._policy_reasons(can_yield=can_yield)],
            )
        if input_type == ConversationFloorInputType.INTERRUPTION_ACCEPTED:
            return self._decision(
                ConversationFloorStatus.REPAIR,
                ConversationFloorSubStateV3.ACCEPTED_INTERRUPT,
                "yield_to_user",
                ["floor:interruption_accepted", "floor:yielded"],
            )
        if input_type == ConversationFloorInputType.INTERRUPTION_REJECTED:
            return self._decision(
                ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR,
                "continue_assistant",
                ["floor:interruption_rejected", "floor:continued"],
            )
        if input_type == ConversationFloorInputType.INTERRUPTION_SUPPRESSED:
            return self._decision(
                ConversationFloorStatus.OVERLAP,
                ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                "protected_continue",
                [
                    "floor:interruption_suppressed",
                    "floor:protected_continue",
                    "protected_playback",
                ],
            )
        if input_type == ConversationFloorInputType.INTERRUPTION_RESUMED:
            return self._decision(
                ConversationFloorStatus.HANDOFF,
                ConversationFloorSubStateV3.HANDOFF_COMPLETE,
                "handoff_complete",
                ["floor:interruption_resumed", "floor:waiting"],
            )
        return self._decision(
            ConversationFloorStatus.UNKNOWN,
            ConversationFloorSubStateV3.HANDOFF_COMPLETE,
            "wait_for_input",
            ["floor:unknown_input"],
        )

    def _decide_from_transcription(
        self,
        input_type: ConversationFloorInputType,
        classification: ConversationFloorPhraseClassification,
        can_yield: bool,
    ) -> tuple[
        ConversationFloorStatus,
        ConversationFloorSubStateV3,
        str,
        list[str],
    ]:
        final = input_type == ConversationFloorInputType.STT_FINAL
        text_kind = (
            classification.text_kind
            if isinstance(classification.text_kind, ConversationFloorTextKind)
            else ConversationFloorTextKind(str(classification.text_kind))
        )
        phrase_class = _safe_token(classification.phrase_class, default=text_kind.value, limit=64)
        if text_kind == ConversationFloorTextKind.EMPTY:
            return self._decision(
                self._state,
                self._sub_state,
                self._yield_decision,
                ["floor:empty_transcript"],
            )
        if classification.low_confidence_transcript:
            reasons = [
                "floor:low_confidence_transcript",
                "low_confidence_transcript",
                *self._policy_reasons(can_yield=False),
            ]
            if self._assistant_speaking:
                return self._decision(
                    ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                    ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR,
                    "continue_assistant",
                    reasons,
                )
            return self._decision(
                ConversationFloorStatus.HANDOFF
                if final
                else ConversationFloorStatus.USER_HAS_FLOOR,
                ConversationFloorSubStateV3.HANDOFF_PENDING
                if final
                else ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
                "wait_for_assistant" if final else "hold_user",
                reasons,
            )
        if phrase_class in {"backchannel", "confirmation", "continuer"} or text_kind in {
            ConversationFloorTextKind.BACKCHANNEL,
            ConversationFloorTextKind.CONFIRMATION,
        }:
            if self._assistant_speaking:
                sub_state = (
                    ConversationFloorSubStateV3.ASSISTANT_HOLDING_FLOOR
                    if phrase_class == "continuer"
                    else ConversationFloorSubStateV3.IGNORED_BACKCHANNEL
                )
                reasons = ["floor:backchannel_continue", "floor:continued"]
                if phrase_class == "continuer":
                    reasons.append("user_continuing")
                else:
                    reasons.append("short_backchannel")
                return self._decision(
                    ConversationFloorStatus.ASSISTANT_HAS_FLOOR,
                    sub_state,
                    "continue_assistant",
                    reasons,
                )
            return self._decision(
                ConversationFloorStatus.HANDOFF
                if final
                else ConversationFloorStatus.USER_HAS_FLOOR,
                ConversationFloorSubStateV3.HANDOFF_PENDING
                if final
                else ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
                "wait_for_assistant" if final else "hold_user",
                [
                    "floor:short_confirmation",
                    "user_continuing" if phrase_class == "continuer" else "short_backchannel",
                    "floor:waiting" if final else "floor:user_holding",
                ],
            )
        if text_kind == ConversationFloorTextKind.HESITATION:
            if self._assistant_speaking:
                yield_decision = "yield_to_user" if can_yield else "protected_continue"
                return self._decision(
                    ConversationFloorStatus.OVERLAP,
                    ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                    yield_decision,
                    [
                        "floor:hesitation_overlap",
                        *self._policy_reasons(can_yield=can_yield),
                    ],
                )
            return self._decision(
                ConversationFloorStatus.USER_HAS_FLOOR,
                ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
                "hold_user",
                ["floor:hesitation_user_holding", "floor:waited"],
            )
        if text_kind == ConversationFloorTextKind.CORRECTION:
            reasons = ["floor:correction", "floor:repaired", "correction"]
            if self._assistant_speaking:
                reasons.extend(
                    ["floor:yielded"] if can_yield else self._policy_reasons(can_yield=False)
                )
            return self._decision(
                ConversationFloorStatus.REPAIR,
                ConversationFloorSubStateV3.REPAIR_REQUESTED,
                "repair_requested" if can_yield or not self._assistant_speaking else "protected_continue",
                reasons,
            )
        if text_kind == ConversationFloorTextKind.EXPLICIT_INTERRUPTION:
            if self._assistant_speaking:
                if can_yield:
                    return self._decision(
                        ConversationFloorStatus.REPAIR,
                        ConversationFloorSubStateV3.REPAIR_REQUESTED,
                        "yield_to_user",
                        [
                            "floor:explicit_interruption",
                            "floor:yielded",
                            "explicit_interrupt",
                        ],
                    )
                return self._decision(
                    ConversationFloorStatus.OVERLAP,
                    ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                    "protected_continue",
                    [
                        "floor:explicit_interruption",
                        "explicit_interrupt",
                        *self._policy_reasons(can_yield=False),
                    ],
                )
            return self._decision(
                ConversationFloorStatus.REPAIR,
                ConversationFloorSubStateV3.REPAIR_REQUESTED,
                "repair_requested",
                ["floor:explicit_interruption", "floor:repaired", "explicit_interrupt"],
            )
        if self._assistant_speaking:
            yield_decision = "yield_to_user" if can_yield else "protected_continue"
            return self._decision(
                ConversationFloorStatus.OVERLAP,
                ConversationFloorSubStateV3.OVERLAP_CANDIDATE,
                yield_decision,
                ["floor:meaningful_overlap", *self._policy_reasons(can_yield=can_yield)],
            )
        if final:
            return self._decision(
                ConversationFloorStatus.HANDOFF,
                ConversationFloorSubStateV3.HANDOFF_PENDING,
                "wait_for_assistant",
                ["floor:user_final", "floor:handoff"],
            )
        return self._decision(
            ConversationFloorStatus.USER_HAS_FLOOR,
            ConversationFloorSubStateV3.USER_HOLDING_FLOOR,
            "hold_user",
            ["floor:meaningful_user_speech"],
        )

    def _snapshot_unlocked(self) -> ConversationFloorState:
        return ConversationFloorState(
            state=self._state,
            profile=self._profile,
            language=self._language,
            assistant_speaking=self._assistant_speaking,
            user_speaking=self._user_speaking,
            protected_playback=self._protected_playback,
            barge_in_armed=self._barge_in_armed,
            sub_state=self._sub_state,
            last_input_type=self._last_input_type,
            last_text_kind=self._last_text_kind,
            phrase_class=self._last_phrase_class,
            phrase_confidence=self._last_phrase_confidence,
            phrase_confidence_bucket=self._last_phrase_confidence_bucket,
            yield_decision=self._yield_decision,
            echo_risk=self._echo_risk,
            barge_in_state=self._barge_in_state,
            tts_chunk_role=self._tts_chunk_role,
            low_confidence_transcript=self._low_confidence_transcript,
            last_transition_at=self._last_transition_at,
            updated_at=self._updated_at,
            transition_count=self._transition_count,
            counts=dict(self._counts),
            reason_codes=list(self._reason_codes),
        )


__all__ = [
    "ConversationFloorController",
    "ConversationFloorInput",
    "ConversationFloorInputType",
    "ConversationFloorPhraseClassification",
    "ConversationFloorState",
    "ConversationFloorStatus",
    "ConversationFloorSubStateV3",
    "ConversationFloorTextKind",
    "ConversationFloorUpdate",
    "classify_floor_phrase",
    "classify_floor_text",
    "is_explicit_interruption_phrase",
    "is_short_backchannel",
    "normalized_floor_text",
]

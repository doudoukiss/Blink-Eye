"""Speech chunk planning for browser voice runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from blink.brain.persona.voice_backend_registry import (
    normalize_voice_backend_label,
    resolve_voice_backend_capabilities,
)

SpeechDirectorMode = Literal[
    "melo_chunked",
    "kokoro_chunked",
    "kokoro_passthrough",
    "unavailable",
]

MELO_STRONG_BOUNDARIES = ".!?。！？；;\n"
MELO_SOFT_BOUNDARIES = "，、：:,"
MELO_TARGET_CHARS = 160
MELO_MIN_CHARS = 80
MELO_HARD_MAX_CHARS = 220
MELO_FIRST_SUBTITLE_CHARS = 120
KOKORO_STRONG_BOUNDARIES = ".!?;\n"
KOKORO_SOFT_BOUNDARIES = ",:"
KOKORO_TARGET_CHARS = 120
KOKORO_MIN_CHARS = 40
KOKORO_HARD_MAX_CHARS = 180
SPEECH_DIRECTOR_V3_VERSION = 3
SUBTITLE_TIMING_POLICY_V3 = "before_or_at_playback_start"
_LOCAL_HTTP_WAV_BACKENDS = {"local-http-wav", "local_http_wav"}


@dataclass(frozen=True)
class SpeechChunkBudgetV3:
    """Bounded chunk budget used by the dual TTS speech director."""

    schema_version: int
    director_mode: str
    tts_backend: str
    min_chars: int
    target_chars: int
    hard_max_chars: int
    max_chunks_per_flush: int
    reason_codes: tuple[str, ...]

    @classmethod
    def from_plan(
        cls,
        *,
        director_mode: str,
        tts_backend: str | None,
        actuation_chunk_limit: int,
        plan_budget: Mapping[str, Any] | None = None,
        local_http_wav_turn_chunk_limit: int = MELO_HARD_MAX_CHARS,
    ) -> "SpeechChunkBudgetV3":
        """Resolve a deterministic speech budget from backend defaults and plan hints."""
        mode = str(director_mode or "unavailable").strip() or "unavailable"
        backend = normalize_voice_backend_label(tts_backend)
        plan = plan_budget if isinstance(plan_budget, Mapping) else {}
        reasons: list[str] = ["speech_chunk_budget_v3:resolved"]

        if mode == "melo_chunked" and backend in _LOCAL_HTTP_WAV_BACKENDS:
            min_chars = MELO_MIN_CHARS
            target_chars = MELO_TARGET_CHARS
            hard_max_chars = MELO_HARD_MAX_CHARS
            reasons.append("speech_chunk_budget_v3:melo_defaults")
        elif mode == "kokoro_chunked" and backend == "kokoro":
            min_chars = KOKORO_MIN_CHARS
            target_chars = KOKORO_TARGET_CHARS
            hard_max_chars = KOKORO_HARD_MAX_CHARS
            reasons.append("speech_chunk_budget_v3:kokoro_defaults")
        else:
            hard_max_chars = max(1, int(actuation_chunk_limit or 1))
            if backend in _LOCAL_HTTP_WAV_BACKENDS:
                hard_max_chars = max(hard_max_chars, int(local_http_wav_turn_chunk_limit))
                reasons.append("speech_chunk_budget_v3:local_http_wav_legacy")
            min_chars = min(180, max(96, hard_max_chars // 2)) if backend in _LOCAL_HTTP_WAV_BACKENDS else 0
            target_chars = hard_max_chars
            reasons.append("speech_chunk_budget_v3:legacy_defaults")

        plan_hard_max = _safe_positive_int(plan.get("hard_max_chars"))
        if plan_hard_max > 0:
            hard_max_chars = max(1, min(hard_max_chars, plan_hard_max))
            reasons.append("speech_chunk_budget_v3:plan_hard_max_clamped")

        plan_target = _safe_positive_int(plan.get("target_chars"))
        if plan_target > 0:
            target_chars = max(1, min(hard_max_chars, plan_target))
            min_chars = min(hard_max_chars, max(24, target_chars // 2))
            reasons.append("speech_chunk_budget_v3:plan_target")
        else:
            target_chars = max(1, min(hard_max_chars, target_chars))
            min_chars = min(hard_max_chars, min_chars)

        max_chunks_per_flush = _safe_positive_int(plan.get("max_chunks_per_flush"))
        if max_chunks_per_flush > 0:
            max_chunks_per_flush = min(max_chunks_per_flush, 12)
            reasons.append("speech_chunk_budget_v3:plan_flush_cap")

        plan_reasons = plan.get("reason_codes")
        if isinstance(plan_reasons, (list, tuple)):
            reasons.extend(str(reason) for reason in plan_reasons)

        return cls(
            schema_version=3,
            director_mode=mode,
            tts_backend=backend,
            min_chars=max(0, int(min_chars)),
            target_chars=max(1, int(target_chars)),
            hard_max_chars=max(1, int(hard_max_chars)),
            max_chunks_per_flush=max(0, int(max_chunks_per_flush)),
            reason_codes=_dedupe_reason_codes(tuple(reasons)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return public-safe budget metadata."""
        return {
            "schema_version": self.schema_version,
            "director_mode": self.director_mode,
            "tts_backend": self.tts_backend,
            "min_chars": self.min_chars,
            "target_chars": self.target_chars,
            "hard_max_chars": self.hard_max_chars,
            "max_chunks_per_flush": self.max_chunks_per_flush,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, init=False)
class SpeechPerformanceChunk:
    """Live speech chunk contract shared by Melo and Kokoro browser paths."""

    schema_version: int
    speech_director_version: int
    chunk_id: str
    role: str
    text: str
    language: str
    tts_backend: str
    display_text: str
    pause_after_ms: int
    interruptible: bool
    context_id: str | None
    generation_token: str
    turn_id: str
    chunk_index: int
    estimated_duration_ms: int
    subtitle_timing: Mapping[str, object]
    stale_generation_token: str
    backend_capabilities: Mapping[str, object]
    created_at_monotonic: float
    reason_codes: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        role: str,
        text: str | None = None,
        language: str = "unknown",
        tts_backend: str | None = None,
        display_text: str | None = None,
        pause_after_ms: int = 0,
        interruptible: bool = True,
        context_id: str | None = None,
        generation_token: str | None = None,
        turn_id: str = "turn-unavailable",
        chunk_index: int = 0,
        chunk_id: str | None = None,
        schema_version: int = 1,
        speech_director_version: int = SPEECH_DIRECTOR_V3_VERSION,
        estimated_duration_ms: int | None = None,
        subtitle_timing: Mapping[str, object] | None = None,
        stale_generation_token: str | None = None,
        backend_capabilities: Mapping[str, object] | None = None,
        created_at_monotonic: float | None = None,
        reason_codes: tuple[str, ...] | list[str] = (),
        speak_text: str | None = None,
        generation_id: str | None = None,
    ):
        """Initialize a speech chunk while accepting legacy field names."""
        chunk_text = text if text is not None else speak_text
        chunk_text = str(chunk_text if chunk_text is not None else display_text or "")
        display = str(display_text if display_text is not None else chunk_text)
        backend = str(tts_backend or "unknown").strip() or "unknown"
        token = str(generation_token or generation_id or "speech-unavailable").strip()
        token = token or "speech-unavailable"
        safe_turn_id = str(turn_id or "turn-unavailable").strip() or "turn-unavailable"
        safe_chunk_index = max(0, int(chunk_index))
        safe_chunk_id = str(
            chunk_id or f"{token}:{safe_turn_id}:{safe_chunk_index}"
        ).strip()
        safe_pause_after_ms = max(0, int(pause_after_ms))
        safe_duration_ms = (
            max(0, int(estimated_duration_ms))
            if estimated_duration_ms is not None
            else estimate_speech_duration_ms(
                chunk_text,
                language=language,
                tts_backend=backend,
                pause_after_ms=safe_pause_after_ms,
            )
        )
        safe_stale_token = str(stale_generation_token or token).strip() or token
        object.__setattr__(self, "schema_version", int(schema_version))
        object.__setattr__(self, "speech_director_version", int(speech_director_version))
        object.__setattr__(self, "chunk_id", safe_chunk_id or f"{token}:{safe_chunk_index}")
        object.__setattr__(self, "role", str(role or "assistant").strip() or "assistant")
        object.__setattr__(self, "text", chunk_text)
        object.__setattr__(self, "language", str(language or "unknown").strip() or "unknown")
        object.__setattr__(self, "tts_backend", backend)
        object.__setattr__(self, "display_text", display)
        object.__setattr__(self, "pause_after_ms", safe_pause_after_ms)
        object.__setattr__(self, "interruptible", bool(interruptible))
        object.__setattr__(self, "context_id", str(context_id).strip() if context_id else None)
        object.__setattr__(self, "generation_token", token)
        object.__setattr__(self, "turn_id", safe_turn_id)
        object.__setattr__(self, "chunk_index", safe_chunk_index)
        object.__setattr__(self, "estimated_duration_ms", safe_duration_ms)
        object.__setattr__(
            self,
            "subtitle_timing",
            _normalize_subtitle_timing(subtitle_timing),
        )
        object.__setattr__(self, "stale_generation_token", safe_stale_token)
        object.__setattr__(
            self,
            "backend_capabilities",
            _normalize_backend_capabilities_v3(backend, backend_capabilities),
        )
        object.__setattr__(
            self,
            "created_at_monotonic",
            float(created_at_monotonic if created_at_monotonic is not None else time.monotonic()),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _dedupe_reason_codes(
                (
                    "speech_director:v3",
                    "speech_duration:estimated_from_chars",
                    "subtitle_timing:before_or_at_playback_start",
                    *tuple(reason_codes),
                )
            ),
        )

    @property
    def speak_text(self) -> str:
        """Return the TTS-bound text for compatibility with older callers."""
        return self.text

    @property
    def generation_id(self) -> str:
        """Return the generation identifier used by existing runtime metadata."""
        return self.generation_token

    @property
    def display_chars(self) -> int:
        """Return the display-text length without exposing the text itself."""
        return len(self.display_text)

    @property
    def speak_chars(self) -> int:
        """Return the TTS-bound text length without exposing the text itself."""
        return len(self.text)

    def as_dict(self) -> dict[str, object]:
        """Return the live chunk payload, including bounded display/speech text."""
        return {
            "schema_version": self.schema_version,
            "speech_director_version": self.speech_director_version,
            "chunk_id": self.chunk_id,
            "role": self.role,
            "text": self.text,
            "language": self.language,
            "tts_backend": self.tts_backend,
            "display_text": self.display_text,
            "pause_after_ms": self.pause_after_ms,
            "interruptible": self.interruptible,
            "context_id": self.context_id,
            "generation_token": self.generation_token,
            "turn_id": self.turn_id,
            "chunk_index": self.chunk_index,
            "estimated_duration_ms": self.estimated_duration_ms,
            "subtitle_timing": dict(self.subtitle_timing),
            "stale_generation_token": self.stale_generation_token,
            "backend_capabilities": dict(self.backend_capabilities),
            "reason_codes": list(self.reason_codes),
        }

    def public_metadata(self, *, queue_depth: int = 0) -> dict[str, object]:
        """Return text-free public metadata for events and persistent traces."""
        return {
            "speech_director_version": self.speech_director_version,
            "role": self.role,
            "chunk_id": self.chunk_id,
            "language": self.language,
            "tts_backend": self.tts_backend,
            "generation_id": self.generation_id,
            "stale_generation_token": self.stale_generation_token,
            "turn_id": self.turn_id,
            "chunk_index": self.chunk_index,
            "display_chars": self.display_chars,
            "speak_chars": self.speak_chars,
            "estimated_duration_ms": self.estimated_duration_ms,
            "subtitle_timing": dict(self.subtitle_timing),
            "backend_capabilities": dict(self.backend_capabilities),
            "interruptible": self.interruptible,
            "pause_after_ms": self.pause_after_ms,
            "queue_depth": max(0, int(queue_depth)),
            "context_available": self.context_id is not None,
        }


BrainSpeechChunk = SpeechPerformanceChunk


def build_speech_chunk_frame_metadata(frame: object, chunk: SpeechPerformanceChunk) -> None:
    """Attach browser speech metadata to a TTS-bound frame."""
    setattr(frame, "blink_speech_chunk", chunk)
    setattr(frame, "blink_display_text", chunk.display_text)
    setattr(frame, "blink_generation_id", chunk.generation_id)
    setattr(frame, "blink_generation_token", chunk.generation_token)
    setattr(frame, "blink_stale_generation_token", chunk.stale_generation_token)
    setattr(frame, "blink_turn_id", chunk.turn_id)
    setattr(frame, "blink_speech_chunk_id", chunk.chunk_id)
    setattr(frame, "blink_speech_language", chunk.language)
    setattr(frame, "blink_tts_backend", chunk.tts_backend)
    setattr(frame, "blink_pause_after_ms", chunk.pause_after_ms)
    setattr(frame, "blink_estimated_duration_ms", chunk.estimated_duration_ms)
    setattr(frame, "blink_subtitle_timing", dict(chunk.subtitle_timing))
    setattr(frame, "blink_backend_capabilities", dict(chunk.backend_capabilities))
    setattr(frame, "blink_speech_director_version", chunk.speech_director_version)
    setattr(frame, "blink_subtitle_immediate", True)


def next_melo_speech_chunk(
    text: str,
    *,
    force: bool = False,
    min_chars: int = MELO_MIN_CHARS,
    target_chars: int = MELO_TARGET_CHARS,
    hard_max_chars: int = MELO_HARD_MAX_CHARS,
) -> tuple[str, str, int, tuple[str, ...]] | tuple[None, str, int, tuple[str, ...]]:
    """Return the next bounded Melo speech chunk from buffered assistant text."""
    normalized = str(text or "")
    if not normalized.strip():
        return None, "", 0, ("speech_director:empty",)

    safe_hard_max = max(1, int(hard_max_chars))
    safe_target = max(1, min(int(target_chars), safe_hard_max))
    safe_min = 0 if force else max(0, min(int(min_chars), safe_hard_max))

    if len(normalized) <= safe_hard_max and not force:
        strong = _last_boundary_before(
            normalized,
            markers=MELO_STRONG_BOUNDARIES,
            limit=len(normalized),
        )
        if strong is None:
            return None, normalized, 0, ("speech_director:waiting_for_boundary",)
        candidate = normalized[:strong].strip()
        if len(candidate) < safe_min and len(normalized) < MELO_FIRST_SUBTITLE_CHARS:
            return None, normalized, 0, ("speech_director:waiting_for_minimum",)
        chunk, remaining = _split_at(normalized, strong)
        return chunk, remaining, _pause_after_ms(chunk), ("speech_director:strong_boundary",)

    boundary = _last_boundary_before(
        normalized,
        markers=MELO_STRONG_BOUNDARIES,
        limit=min(len(normalized), safe_hard_max),
    )
    reason = "speech_director:strong_boundary"
    if boundary is None or boundary < max(24, safe_min):
        soft_limit = min(len(normalized), safe_target)
        boundary = _last_boundary_before(
            normalized,
            markers=MELO_SOFT_BOUNDARIES,
            limit=soft_limit,
        )
        reason = "speech_director:soft_boundary"
    if boundary is None or boundary < max(24, safe_min):
        boundary = _last_space_before(normalized, limit=min(len(normalized), safe_hard_max))
        reason = "speech_director:space_boundary"
    if boundary is None or boundary < max(24, safe_min):
        boundary = min(len(normalized), safe_hard_max)
        reason = "speech_director:hard_boundary"

    chunk, remaining = _split_at(normalized, boundary)
    if not chunk:
        return None, normalized, 0, ("speech_director:waiting",)
    return chunk, remaining, _pause_after_ms(chunk), (reason,)


def next_kokoro_speech_chunk(
    text: str,
    *,
    force: bool = False,
    min_chars: int = KOKORO_MIN_CHARS,
    target_chars: int = KOKORO_TARGET_CHARS,
    hard_max_chars: int = KOKORO_HARD_MAX_CHARS,
) -> tuple[str, str, int, tuple[str, ...]] | tuple[None, str, int, tuple[str, ...]]:
    """Return the next balanced English Kokoro speech chunk."""
    normalized = str(text or "")
    if not normalized.strip():
        return None, "", 0, ("speech_director:empty",)

    safe_hard_max = max(1, int(hard_max_chars))
    safe_target = max(1, min(int(target_chars), safe_hard_max))
    safe_min = 0 if force else max(0, min(int(min_chars), safe_hard_max))

    if len(normalized) <= safe_hard_max:
        if force:
            chunk = normalized.strip()
            return chunk, "", _pause_after_ms(chunk), ("speech_director:kokoro_short_reply",)
        boundary = _last_boundary_before(
            normalized,
            markers=KOKORO_STRONG_BOUNDARIES,
            limit=len(normalized),
        )
        if boundary is None:
            return None, normalized, 0, ("speech_director:waiting_for_boundary",)
        candidate = normalized[:boundary].strip()
        if len(candidate) < safe_min:
            return None, normalized, 0, ("speech_director:waiting_for_minimum",)
        chunk, remaining = _split_at(normalized, boundary)
        return chunk, remaining, _pause_after_ms(chunk), ("speech_director:kokoro_sentence",)

    boundary = _last_boundary_before(
        normalized,
        markers=KOKORO_STRONG_BOUNDARIES,
        limit=min(len(normalized), safe_target),
    )
    reason = "speech_director:kokoro_sentence"
    if boundary is None or boundary < max(24, safe_min):
        boundary = _last_boundary_before(
            normalized,
            markers=KOKORO_STRONG_BOUNDARIES,
            limit=min(len(normalized), safe_hard_max),
        )
        reason = "speech_director:kokoro_sentence_hard_limit"
    if boundary is None or boundary < max(24, safe_min):
        boundary = _last_boundary_before(
            normalized,
            markers=KOKORO_SOFT_BOUNDARIES,
            limit=min(len(normalized), safe_target),
        )
        reason = "speech_director:kokoro_soft_boundary"
    if boundary is None or boundary < max(24, safe_min):
        boundary = _last_space_before(normalized, limit=min(len(normalized), safe_hard_max))
        reason = "speech_director:kokoro_space_boundary"
    if boundary is None or boundary < max(24, safe_min):
        boundary = min(len(normalized), safe_hard_max)
        reason = "speech_director:kokoro_hard_boundary"

    chunk, remaining = _split_at(normalized, boundary)
    if not chunk:
        return None, normalized, 0, ("speech_director:waiting",)
    return chunk, remaining, _pause_after_ms(chunk), (reason,)


def _last_boundary_before(text: str, *, markers: str, limit: int) -> int | None:
    bounded = text[: max(0, int(limit))]
    indexes = [bounded.rfind(marker) for marker in markers]
    index = max(indexes)
    return index + 1 if index >= 0 else None


def _last_space_before(text: str, *, limit: int) -> int | None:
    index = text[: max(0, int(limit))].rfind(" ")
    return index + 1 if index >= 0 else None


def _split_at(text: str, index: int) -> tuple[str, str]:
    chunk = text[:index].strip()
    remaining = text[index:].lstrip()
    return chunk, remaining


def _pause_after_ms(chunk: str) -> int:
    text = str(chunk or "").rstrip()
    if not text:
        return 0
    last = text[-1]
    if last in ".。！？!?":
        return 180
    if last in "；;：:":
        return 120
    if last in "，、,":
        return 80
    return 0


def estimate_speech_duration_ms(
    text: str,
    *,
    language: str | None,
    tts_backend: str | None,
    pause_after_ms: int = 0,
) -> int:
    """Return a deterministic text-length duration estimate for replay and planning."""
    normalized = "".join(str(text or "").split())
    if not normalized:
        return 0
    backend = normalize_voice_backend_label(tts_backend)
    language_label = str(language or "").strip().lower()
    if language_label == "zh" or backend in _LOCAL_HTTP_WAV_BACKENDS:
        estimated = len(normalized) * 170
    else:
        estimated = len(normalized) * 55
    return max(250, min(300_000, int(estimated) + max(0, int(pause_after_ms))))


def _normalize_subtitle_timing(value: Mapping[str, object] | None) -> dict[str, object]:
    payload = dict(value or {})
    emit_policy = str(payload.get("emit_policy") or SUBTITLE_TIMING_POLICY_V3).strip()
    if emit_policy != SUBTITLE_TIMING_POLICY_V3:
        emit_policy = SUBTITLE_TIMING_POLICY_V3
    return {
        "emit_policy": emit_policy,
        "ready_at_ms": _safe_nonnegative_int(payload.get("ready_at_ms")),
        "playback_start_aligned": payload.get("playback_start_aligned") is not False,
        "timing_source": "speech_director_v3",
    }


def _normalize_backend_capabilities_v3(
    tts_backend: str | None,
    value: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(value, Mapping):
        payload = dict(value)
        return {
            "backend_label": normalize_voice_backend_label(
                payload.get("backend_label") or tts_backend
            ),
            "supports_chunk_boundaries": bool(payload.get("supports_chunk_boundaries")),
            "supports_interruption_flush": bool(payload.get("supports_interruption_flush")),
            "supports_interruption_discard": bool(payload.get("supports_interruption_discard")),
            "supports_pause_timing": bool(payload.get("supports_pause_timing")),
            "supports_speech_rate": bool(payload.get("supports_speech_rate")),
            "supports_prosody_emphasis": bool(payload.get("supports_prosody_emphasis")),
            "supports_partial_stream_abort": bool(payload.get("supports_partial_stream_abort")),
            "expression_controls_hardware": bool(payload.get("expression_controls_hardware")),
            "reason_codes": list(_dedupe_reason_codes(tuple(payload.get("reason_codes") or ()))),
        }
    resolution = resolve_voice_backend_capabilities(tts_backend)
    capabilities = resolution.capabilities
    return {
        "backend_label": normalize_voice_backend_label(capabilities.backend_label),
        "supports_chunk_boundaries": capabilities.supports_chunk_boundaries,
        "supports_interruption_flush": capabilities.supports_interruption_flush,
        "supports_interruption_discard": capabilities.supports_interruption_discard,
        "supports_pause_timing": capabilities.supports_pause_timing,
        "supports_speech_rate": capabilities.supports_speech_rate,
        "supports_prosody_emphasis": capabilities.supports_prosody_emphasis,
        "supports_partial_stream_abort": capabilities.supports_partial_stream_abort,
        "expression_controls_hardware": capabilities.expression_controls_hardware,
        "reason_codes": list(_dedupe_reason_codes((*resolution.reason_codes, *capabilities.reason_codes))),
    }


def _safe_positive_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _dedupe_reason_codes(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


__all__ = [
    "BrainSpeechChunk",
    "KOKORO_HARD_MAX_CHARS",
    "KOKORO_MIN_CHARS",
    "KOKORO_SOFT_BOUNDARIES",
    "KOKORO_STRONG_BOUNDARIES",
    "KOKORO_TARGET_CHARS",
    "MELO_FIRST_SUBTITLE_CHARS",
    "MELO_HARD_MAX_CHARS",
    "MELO_MIN_CHARS",
    "MELO_SOFT_BOUNDARIES",
    "MELO_STRONG_BOUNDARIES",
    "MELO_TARGET_CHARS",
    "SPEECH_DIRECTOR_V3_VERSION",
    "SUBTITLE_TIMING_POLICY_V3",
    "SpeechDirectorMode",
    "SpeechChunkBudgetV3",
    "SpeechPerformanceChunk",
    "build_speech_chunk_frame_metadata",
    "estimate_speech_duration_ms",
    "next_kokoro_speech_chunk",
    "next_melo_speech_chunk",
]

"""Provider-neutral expression-to-voice policy mapping for Blink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blink.brain.persona.policy import BrainPersonaModality, resolve_persona_modality

_VOICE_LIKE_MODALITIES = {
    BrainPersonaModality.VOICE,
    BrainPersonaModality.BROWSER,
    BrainPersonaModality.EMBODIED,
}
_BALANCED_CHUNK_CHARS = 220
_CONCISE_CHUNK_CHARS = 132
_SAFETY_CHUNK_CHARS = 96


def _dedupe_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _percentile_int(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    sorted_values = sorted(max(0, int(value)) for value in values)
    index = max(
        0, min(len(sorted_values) - 1, round((percentile / 100) * (len(sorted_values) - 1)))
    )
    return sorted_values[index]


@dataclass(frozen=True)
class BrainExpressionVoicePolicy:
    """High-level voice behavior policy derived from an expression frame."""

    available: bool
    modality: str
    concise_chunking_active: bool
    chunking_mode: str
    max_spoken_chunk_chars: int
    interruption_strategy_label: str
    pause_yield_hint: str
    active_hints: tuple[str, ...]
    unsupported_hints: tuple[str, ...]
    noop_reason_codes: tuple[str, ...]
    expression_controls_hardware: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the voice policy."""
        return {
            "available": self.available,
            "modality": self.modality,
            "concise_chunking_active": self.concise_chunking_active,
            "chunking_mode": self.chunking_mode,
            "max_spoken_chunk_chars": self.max_spoken_chunk_chars,
            "interruption_strategy_label": self.interruption_strategy_label,
            "pause_yield_hint": self.pause_yield_hint,
            "active_hints": list(self.active_hints),
            "unsupported_hints": list(self.unsupported_hints),
            "noop_reason_codes": list(self.noop_reason_codes),
            "expression_controls_hardware": self.expression_controls_hardware,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainExpressionVoiceMetricsSnapshot:
    """Provider-neutral voice policy metrics for runtime and browser surfaces."""

    available: bool
    response_count: int
    concise_chunking_activation_count: int
    chunk_count: int
    max_chunk_chars: int
    average_chunk_chars: float
    interruption_frame_count: int
    buffer_flush_count: int
    buffer_discard_count: int
    last_chunking_mode: str
    last_max_spoken_chunk_chars: int
    expression_controls_hardware: bool
    reason_codes: tuple[str, ...]
    first_audio_latency_ms: float = 0.0
    first_audio_latency_sample_count: int = 0
    resumed_latency_after_interrupt_ms: float = 0.0
    resumed_latency_sample_count: int = 0
    interruption_accept_count: int = 0
    partial_stream_abort_count: int = 0
    average_chunks_per_response: float = 0.0
    p50_chunk_chars: int = 0
    p95_chunk_chars: int = 0
    first_subtitle_latency_ms: float = 0.0
    first_subtitle_latency_sample_count: int = 0
    speech_chunk_latency_ms: float = 0.0
    speech_chunk_latency_sample_count: int = 0
    speech_queue_depth_current: int = 0
    speech_queue_depth_max: int = 0
    stale_chunk_drop_count: int = 0
    speech_director_mode: str = "unavailable"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the voice metrics snapshot."""
        return {
            "available": self.available,
            "response_count": self.response_count,
            "concise_chunking_activation_count": self.concise_chunking_activation_count,
            "chunk_count": self.chunk_count,
            "max_chunk_chars": self.max_chunk_chars,
            "average_chunk_chars": self.average_chunk_chars,
            "interruption_frame_count": self.interruption_frame_count,
            "buffer_flush_count": self.buffer_flush_count,
            "buffer_discard_count": self.buffer_discard_count,
            "last_chunking_mode": self.last_chunking_mode,
            "last_max_spoken_chunk_chars": self.last_max_spoken_chunk_chars,
            "first_audio_latency_ms": self.first_audio_latency_ms,
            "first_audio_latency_sample_count": self.first_audio_latency_sample_count,
            "resumed_latency_after_interrupt_ms": self.resumed_latency_after_interrupt_ms,
            "resumed_latency_sample_count": self.resumed_latency_sample_count,
            "interruption_accept_count": self.interruption_accept_count,
            "partial_stream_abort_count": self.partial_stream_abort_count,
            "average_chunks_per_response": self.average_chunks_per_response,
            "p50_chunk_chars": self.p50_chunk_chars,
            "p95_chunk_chars": self.p95_chunk_chars,
            "first_subtitle_latency_ms": self.first_subtitle_latency_ms,
            "first_subtitle_latency_sample_count": self.first_subtitle_latency_sample_count,
            "speech_chunk_latency_ms": self.speech_chunk_latency_ms,
            "speech_chunk_latency_sample_count": self.speech_chunk_latency_sample_count,
            "speech_queue_depth_current": self.speech_queue_depth_current,
            "speech_queue_depth_max": self.speech_queue_depth_max,
            "stale_chunk_drop_count": self.stale_chunk_drop_count,
            "speech_director_mode": self.speech_director_mode,
            "expression_controls_hardware": self.expression_controls_hardware,
            "reason_codes": list(self.reason_codes),
        }


class BrainExpressionVoiceMetricsRecorder:
    """In-process recorder for concrete expression voice policy behavior."""

    def __init__(self):
        """Initialize an empty metrics recorder."""
        self.reset()

    def reset(self):
        """Reset all voice metrics counters."""
        self._response_count = 0
        self._concise_chunking_activation_count = 0
        self._chunk_count = 0
        self._total_chunk_chars = 0
        self._max_chunk_chars = 0
        self._chunk_sizes: list[int] = []
        self._interruption_frame_count = 0
        self._interruption_accept_count = 0
        self._buffer_flush_count = 0
        self._buffer_discard_count = 0
        self._partial_stream_abort_count = 0
        self._first_audio_latency_total_ms = 0.0
        self._first_audio_latency_sample_count = 0
        self._resumed_latency_total_ms = 0.0
        self._resumed_latency_sample_count = 0
        self._first_subtitle_latency_total_ms = 0.0
        self._first_subtitle_latency_sample_count = 0
        self._speech_chunk_latency_total_ms = 0.0
        self._speech_chunk_latency_sample_count = 0
        self._speech_queue_depth_current = 0
        self._speech_queue_depth_max = 0
        self._stale_chunk_drop_count = 0
        self._speech_director_mode = "unavailable"
        self._last_chunking_mode = "none"
        self._last_max_spoken_chunk_chars = 0

    def record_response_start(self, policy: Any | None, *, speech_director_mode: str | None = None):
        """Record the start of one assistant response through the voice policy layer."""
        self._response_count += 1
        if speech_director_mode:
            self._speech_director_mode = str(speech_director_mode)
        self._speech_queue_depth_current = 0
        if policy is None:
            self._last_chunking_mode = "unavailable"
            self._last_max_spoken_chunk_chars = 0
            return
        self._last_chunking_mode = policy.chunking_mode
        self._last_max_spoken_chunk_chars = max(0, int(policy.max_spoken_chunk_chars))
        if (
            policy.available
            and bool(
                getattr(
                    policy,
                    "chunk_boundaries_enabled",
                    getattr(policy, "concise_chunking_active", False),
                )
            )
            and policy.max_spoken_chunk_chars > 0
        ):
            self._concise_chunking_activation_count += 1

    def record_chunk(self, text: str):
        """Record one emitted TTS-bound text chunk."""
        length = len(str(text or ""))
        self._chunk_count += 1
        self._total_chunk_chars += length
        self._max_chunk_chars = max(self._max_chunk_chars, length)
        self._chunk_sizes.append(length)

    def record_buffer_flush(self, *, emitted_chunk_count: int):
        """Record a buffer flush event when at least one chunk was emitted."""
        if emitted_chunk_count > 0:
            self._buffer_flush_count += 1
            self.record_speech_queue_depth(emitted_chunk_count)

    def record_interruption(self, *, discarded_buffer: bool, accepted: bool = True):
        """Record one interruption and whether it discarded buffered text."""
        self._interruption_frame_count += 1
        if accepted:
            self._interruption_accept_count += 1
        if discarded_buffer:
            self._buffer_discard_count += 1

    def record_partial_stream_abort(self):
        """Record one partial stream abort event when a backend reports it."""
        self._partial_stream_abort_count += 1

    def record_first_subtitle_latency(self, latency_ms: float):
        """Record one first-subtitle latency sample in milliseconds."""
        value = max(0.0, float(latency_ms))
        self._first_subtitle_latency_total_ms += value
        self._first_subtitle_latency_sample_count += 1

    def record_speech_chunk_latency(self, latency_ms: float):
        """Record one speech chunk latency sample in milliseconds."""
        value = max(0.0, float(latency_ms))
        self._speech_chunk_latency_total_ms += value
        self._speech_chunk_latency_sample_count += 1

    def record_speech_queue_depth(self, current: int):
        """Record the current speech queue depth approximation."""
        value = max(0, int(current))
        self._speech_queue_depth_current = value
        self._speech_queue_depth_max = max(self._speech_queue_depth_max, value)

    def record_stale_chunk_drop(self, count: int = 1):
        """Record stale speech chunks dropped before TTS playback."""
        self._stale_chunk_drop_count += max(0, int(count))

    def record_first_audio_latency(self, latency_ms: float):
        """Record one first-audio latency sample in milliseconds."""
        value = max(0.0, float(latency_ms))
        self._first_audio_latency_total_ms += value
        self._first_audio_latency_sample_count += 1

    def record_resumed_latency_after_interrupt(self, latency_ms: float):
        """Record one resumed-audio latency sample after an interruption."""
        value = max(0.0, float(latency_ms))
        self._resumed_latency_total_ms += value
        self._resumed_latency_sample_count += 1

    def record_audio_started(
        self,
        *,
        first_audio_latency_ms: float | None = None,
        resumed_after_interrupt_latency_ms: float | None = None,
    ):
        """Record safe audio-start latency samples when an observer can provide them."""
        if first_audio_latency_ms is not None:
            self.record_first_audio_latency(first_audio_latency_ms)
        if resumed_after_interrupt_latency_ms is not None:
            self.record_resumed_latency_after_interrupt(resumed_after_interrupt_latency_ms)

    def snapshot(self) -> BrainExpressionVoiceMetricsSnapshot:
        """Return a stable snapshot of the current metrics counters."""
        average = (
            round(self._total_chunk_chars / self._chunk_count, 4) if self._chunk_count else 0.0
        )
        first_audio_latency = (
            round(
                self._first_audio_latency_total_ms / self._first_audio_latency_sample_count,
                4,
            )
            if self._first_audio_latency_sample_count
            else 0.0
        )
        resumed_latency = (
            round(self._resumed_latency_total_ms / self._resumed_latency_sample_count, 4)
            if self._resumed_latency_sample_count
            else 0.0
        )
        first_subtitle_latency = (
            round(
                self._first_subtitle_latency_total_ms
                / self._first_subtitle_latency_sample_count,
                4,
            )
            if self._first_subtitle_latency_sample_count
            else 0.0
        )
        speech_chunk_latency = (
            round(
                self._speech_chunk_latency_total_ms / self._speech_chunk_latency_sample_count,
                4,
            )
            if self._speech_chunk_latency_sample_count
            else 0.0
        )
        chunks_per_response = (
            round(self._chunk_count / self._response_count, 4) if self._response_count else 0.0
        )
        reason_codes = [
            "voice_metrics:v1",
            "voice_metrics:available",
            f"voice_metrics_responses:{self._response_count}",
        ]
        if self._concise_chunking_activation_count:
            reason_codes.append("voice_metrics_concise_chunking_observed")
        if self._interruption_frame_count:
            reason_codes.append("voice_metrics_interruption_observed")
        if self._buffer_discard_count:
            reason_codes.append("voice_metrics_buffer_discard_observed")
        if self._first_audio_latency_sample_count:
            reason_codes.append("voice_metrics_first_audio_latency_observed")
        if self._resumed_latency_sample_count:
            reason_codes.append("voice_metrics_resumed_latency_observed")
        if self._partial_stream_abort_count:
            reason_codes.append("voice_metrics_partial_stream_abort_observed")
        if self._first_subtitle_latency_sample_count:
            reason_codes.append("voice_metrics_first_subtitle_latency_observed")
        if self._speech_chunk_latency_sample_count:
            reason_codes.append("voice_metrics_speech_chunk_latency_observed")
        if self._stale_chunk_drop_count:
            reason_codes.append("voice_metrics_stale_chunk_drop_observed")
        return BrainExpressionVoiceMetricsSnapshot(
            available=True,
            response_count=self._response_count,
            concise_chunking_activation_count=self._concise_chunking_activation_count,
            chunk_count=self._chunk_count,
            max_chunk_chars=self._max_chunk_chars,
            average_chunk_chars=average,
            interruption_frame_count=self._interruption_frame_count,
            buffer_flush_count=self._buffer_flush_count,
            buffer_discard_count=self._buffer_discard_count,
            last_chunking_mode=self._last_chunking_mode,
            last_max_spoken_chunk_chars=self._last_max_spoken_chunk_chars,
            expression_controls_hardware=False,
            reason_codes=_dedupe_preserve_order(tuple(reason_codes)),
            first_audio_latency_ms=first_audio_latency,
            first_audio_latency_sample_count=self._first_audio_latency_sample_count,
            resumed_latency_after_interrupt_ms=resumed_latency,
            resumed_latency_sample_count=self._resumed_latency_sample_count,
            interruption_accept_count=self._interruption_accept_count,
            partial_stream_abort_count=self._partial_stream_abort_count,
            average_chunks_per_response=chunks_per_response,
            p50_chunk_chars=_percentile_int(self._chunk_sizes, 50),
            p95_chunk_chars=_percentile_int(self._chunk_sizes, 95),
            first_subtitle_latency_ms=first_subtitle_latency,
            first_subtitle_latency_sample_count=self._first_subtitle_latency_sample_count,
            speech_chunk_latency_ms=speech_chunk_latency,
            speech_chunk_latency_sample_count=self._speech_chunk_latency_sample_count,
            speech_queue_depth_current=self._speech_queue_depth_current,
            speech_queue_depth_max=self._speech_queue_depth_max,
            stale_chunk_drop_count=self._stale_chunk_drop_count,
            speech_director_mode=self._speech_director_mode,
        )


def unavailable_expression_voice_metrics_snapshot(
    *reason_codes: str,
) -> BrainExpressionVoiceMetricsSnapshot:
    """Return a safe unavailable voice metrics snapshot."""
    return BrainExpressionVoiceMetricsSnapshot(
        available=False,
        response_count=0,
        concise_chunking_activation_count=0,
        chunk_count=0,
        max_chunk_chars=0,
        average_chunk_chars=0.0,
        interruption_frame_count=0,
        buffer_flush_count=0,
        buffer_discard_count=0,
        last_chunking_mode="unavailable",
        last_max_spoken_chunk_chars=0,
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "voice_metrics:v1",
                "voice_metrics:unavailable",
                *reason_codes,
                "voice_policy_noop:hardware_control_forbidden",
            )
        ),
        first_audio_latency_ms=0.0,
        first_audio_latency_sample_count=0,
        resumed_latency_after_interrupt_ms=0.0,
        resumed_latency_sample_count=0,
        interruption_accept_count=0,
        partial_stream_abort_count=0,
        average_chunks_per_response=0.0,
        p50_chunk_chars=0,
        p95_chunk_chars=0,
        first_subtitle_latency_ms=0.0,
        first_subtitle_latency_sample_count=0,
        speech_chunk_latency_ms=0.0,
        speech_chunk_latency_sample_count=0,
        speech_queue_depth_current=0,
        speech_queue_depth_max=0,
        stale_chunk_drop_count=0,
        speech_director_mode="unavailable",
    )


def compile_expression_voice_policy(
    frame,
    *,
    modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
    tts_backend: str | None = None,
) -> BrainExpressionVoicePolicy:
    """Map one expression frame into provider-neutral voice behavior policy."""
    resolved_modality = resolve_persona_modality(modality)
    backend_label = str(tts_backend or "provider-neutral").strip() or "provider-neutral"
    unavailable_reason = "voice_policy_unavailable"
    if frame is None:
        return _unavailable_policy(
            modality=resolved_modality,
            reason_codes=(unavailable_reason, "voice_policy_frame_missing"),
        )
    if resolved_modality not in _VOICE_LIKE_MODALITIES:
        return _unavailable_policy(
            modality=resolved_modality,
            reason_codes=(unavailable_reason, f"voice_policy_modality:{resolved_modality.value}"),
        )

    voice_hints = getattr(frame, "voice_hints", None)
    if voice_hints is None:
        return _unavailable_policy(
            modality=resolved_modality,
            reason_codes=(unavailable_reason, "voice_policy_hints_missing"),
        )

    reason_codes = list(getattr(frame, "reason_codes", ()))
    is_safety = "seriousness:safety" in reason_codes
    concise = bool(getattr(voice_hints, "concise_chunking", False)) or (
        getattr(frame, "response_length", "") == "concise"
    )
    chunk_limit = _BALANCED_CHUNK_CHARS
    chunking_mode = "off"
    active_hints: list[str] = []
    if concise:
        chunk_limit = _SAFETY_CHUNK_CHARS if is_safety else _CONCISE_CHUNK_CHARS
        chunking_mode = "safety_concise" if is_safety else "concise"
        active_hints.append("concise_chunking")

    pause_yield_hint = (
        f"pause={float(getattr(voice_hints, 'pause_density', 0.0)):.2f}; "
        f"yield={str(getattr(voice_hints, 'interruption_strategy', '')).strip() or 'not active'}"
    )
    active_hints.extend(("interruption_strategy_label", "pause_yield_metadata"))
    unsupported_hints = (
        "speech_rate",
        "prosody_emphasis",
        "pause_timing",
        "hardware_control",
    )
    noop_reason_codes = (
        f"voice_policy_noop:speech_rate:{backend_label}",
        f"voice_policy_noop:prosody:{backend_label}",
        "voice_policy_noop:pause_timing_metadata_only",
        "voice_policy_noop:hardware_control_forbidden",
    )
    return BrainExpressionVoicePolicy(
        available=True,
        modality=resolved_modality.value,
        concise_chunking_active=concise,
        chunking_mode=chunking_mode,
        max_spoken_chunk_chars=chunk_limit,
        interruption_strategy_label=(
            str(getattr(voice_hints, "interruption_strategy", "")).strip() or "not active"
        ),
        pause_yield_hint=pause_yield_hint,
        active_hints=_dedupe_preserve_order(tuple(active_hints)),
        unsupported_hints=unsupported_hints,
        noop_reason_codes=noop_reason_codes,
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "voice_policy:available",
                f"voice_policy_modality:{resolved_modality.value}",
                f"voice_policy_chunking:{chunking_mode}",
                *noop_reason_codes,
            )
        ),
    )


def _unavailable_policy(
    *,
    modality: BrainPersonaModality,
    reason_codes: tuple[str, ...],
) -> BrainExpressionVoicePolicy:
    return BrainExpressionVoicePolicy(
        available=False,
        modality=modality.value,
        concise_chunking_active=False,
        chunking_mode="unavailable",
        max_spoken_chunk_chars=0,
        interruption_strategy_label="unavailable",
        pause_yield_hint="unavailable",
        active_hints=(),
        unsupported_hints=(),
        noop_reason_codes=("voice_policy_noop:hardware_control_forbidden",),
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "voice_policy:unavailable",
                *reason_codes,
                "voice_policy_noop:hardware_control_forbidden",
            )
        ),
    )


__all__ = [
    "BrainExpressionVoicePolicy",
    "BrainExpressionVoiceMetricsRecorder",
    "BrainExpressionVoiceMetricsSnapshot",
    "compile_expression_voice_policy",
    "unavailable_expression_voice_metrics_snapshot",
]

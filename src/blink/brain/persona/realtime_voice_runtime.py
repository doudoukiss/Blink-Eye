"""Capability-aware realtime voice actuation plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blink.brain.persona.voice_backend_adapters import resolve_voice_backend_adapter
from blink.brain.persona.voice_backend_registry import BrainVoiceBackendCapabilityRegistry
from blink.brain.persona.voice_capabilities import BrainVoiceBackendCapabilities
from blink.brain.persona.voice_policy import BrainExpressionVoicePolicy


def _dedupe_preserve_order(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class BrainRealtimeVoiceActuationPlan:
    """Concrete provider-neutral voice actuation plan for one runtime turn."""

    available: bool
    backend_label: str
    modality: str
    chunk_boundaries_enabled: bool
    interruption_flush_enabled: bool
    interruption_discard_enabled: bool
    pause_timing_enabled: bool
    speech_rate_enabled: bool
    prosody_emphasis_enabled: bool
    partial_stream_abort_enabled: bool
    expression_controls_hardware: bool
    chunking_mode: str
    max_spoken_chunk_chars: int
    interruption_strategy_label: str
    pause_yield_hint: str
    requested_hints: tuple[str, ...]
    applied_hints: tuple[str, ...]
    active_hints: tuple[str, ...]
    unsupported_hints: tuple[str, ...]
    noop_reason_codes: tuple[str, ...]
    capability_reason_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the realtime voice actuation plan."""
        return {
            "available": self.available,
            "backend_label": self.backend_label,
            "modality": self.modality,
            "chunk_boundaries_enabled": self.chunk_boundaries_enabled,
            "interruption_flush_enabled": self.interruption_flush_enabled,
            "interruption_discard_enabled": self.interruption_discard_enabled,
            "pause_timing_enabled": self.pause_timing_enabled,
            "speech_rate_enabled": self.speech_rate_enabled,
            "prosody_emphasis_enabled": self.prosody_emphasis_enabled,
            "partial_stream_abort_enabled": self.partial_stream_abort_enabled,
            "expression_controls_hardware": self.expression_controls_hardware,
            "chunking_mode": self.chunking_mode,
            "max_spoken_chunk_chars": self.max_spoken_chunk_chars,
            "interruption_strategy_label": self.interruption_strategy_label,
            "pause_yield_hint": self.pause_yield_hint,
            "requested_hints": list(self.requested_hints),
            "applied_hints": list(self.applied_hints),
            "active_hints": list(self.active_hints),
            "unsupported_hints": list(self.unsupported_hints),
            "noop_reason_codes": list(self.noop_reason_codes),
            "capability_reason_codes": list(self.capability_reason_codes),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainRealtimeVoiceActuationPlan":
        """Hydrate an actuation plan from a dict."""
        payload = dict(data or {})
        return cls(
            available=bool(payload.get("available", False)),
            backend_label=str(payload.get("backend_label") or "provider-neutral"),
            modality=str(payload.get("modality") or "unavailable"),
            chunk_boundaries_enabled=bool(payload.get("chunk_boundaries_enabled", False)),
            interruption_flush_enabled=bool(payload.get("interruption_flush_enabled", False)),
            interruption_discard_enabled=bool(payload.get("interruption_discard_enabled", False)),
            pause_timing_enabled=bool(payload.get("pause_timing_enabled", False)),
            speech_rate_enabled=bool(payload.get("speech_rate_enabled", False)),
            prosody_emphasis_enabled=bool(payload.get("prosody_emphasis_enabled", False)),
            partial_stream_abort_enabled=bool(payload.get("partial_stream_abort_enabled", False)),
            expression_controls_hardware=False,
            chunking_mode=str(payload.get("chunking_mode") or "unavailable"),
            max_spoken_chunk_chars=max(0, int(payload.get("max_spoken_chunk_chars") or 0)),
            interruption_strategy_label=str(
                payload.get("interruption_strategy_label") or "unavailable"
            ),
            pause_yield_hint=str(payload.get("pause_yield_hint") or "unavailable"),
            requested_hints=_dedupe_preserve_order(
                tuple(
                    payload.get("requested_hints")
                    or (
                        list(payload.get("active_hints") or ())
                        + list(payload.get("unsupported_hints") or ())
                    )
                )
            ),
            applied_hints=_dedupe_preserve_order(
                tuple(payload.get("applied_hints") or payload.get("active_hints") or ())
            ),
            active_hints=_dedupe_preserve_order(tuple(payload.get("active_hints") or ())),
            unsupported_hints=_dedupe_preserve_order(tuple(payload.get("unsupported_hints") or ())),
            noop_reason_codes=_dedupe_preserve_order(tuple(payload.get("noop_reason_codes") or ())),
            capability_reason_codes=_dedupe_preserve_order(
                tuple(payload.get("capability_reason_codes") or ())
            ),
            reason_codes=_dedupe_preserve_order(tuple(payload.get("reason_codes") or ())),
        )


def unavailable_realtime_voice_actuation_plan(
    *reason_codes: str,
    backend_label: str | None = None,
    modality: str = "unavailable",
) -> BrainRealtimeVoiceActuationPlan:
    """Return a safe unavailable voice actuation plan."""
    label = str(backend_label or "provider-neutral").strip() or "provider-neutral"
    noop_reason_codes = (
        "voice_actuation_noop:chunk_boundaries_unavailable",
        "voice_actuation_noop:interruption_flush_unavailable",
        "voice_actuation_noop:interruption_discard_unavailable",
        "voice_actuation_noop:pause_timing_unsupported",
        "voice_actuation_noop:speech_rate_unsupported",
        "voice_actuation_noop:prosody_emphasis_unsupported",
        "voice_actuation_noop:partial_stream_abort_unsupported",
        "voice_actuation_noop:hardware_control_forbidden",
    )
    return BrainRealtimeVoiceActuationPlan(
        available=False,
        backend_label=label,
        modality=modality,
        chunk_boundaries_enabled=False,
        interruption_flush_enabled=False,
        interruption_discard_enabled=False,
        pause_timing_enabled=False,
        speech_rate_enabled=False,
        prosody_emphasis_enabled=False,
        partial_stream_abort_enabled=False,
        expression_controls_hardware=False,
        chunking_mode="unavailable",
        max_spoken_chunk_chars=0,
        interruption_strategy_label="unavailable",
        pause_yield_hint="unavailable",
        requested_hints=(),
        applied_hints=(),
        active_hints=(),
        unsupported_hints=(
            "chunk_boundaries",
            "interruption_flush",
            "interruption_discard",
            "pause_timing",
            "speech_rate",
            "prosody_emphasis",
            "partial_stream_abort",
            "hardware_control",
        ),
        noop_reason_codes=noop_reason_codes,
        capability_reason_codes=(),
        reason_codes=_dedupe_preserve_order(
            (
                "voice_actuation:unavailable",
                *reason_codes,
                *noop_reason_codes,
            )
        ),
    )


def compile_realtime_voice_actuation_plan(
    policy: BrainExpressionVoicePolicy | None,
    *,
    capabilities: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
    tts_backend: str | None = None,
    capability_registry: BrainVoiceBackendCapabilityRegistry | None = None,
) -> BrainRealtimeVoiceActuationPlan:
    """Compile a concrete voice actuation plan from policy and backend capabilities."""
    adapter = resolve_voice_backend_adapter(
        tts_backend,
        capabilities_override=capabilities,
        registry=capability_registry,
    )
    caps = adapter.capabilities
    if policy is None or not policy.available:
        return unavailable_realtime_voice_actuation_plan(
            "voice_actuation_policy_unavailable",
            backend_label=adapter.backend_label,
            modality=getattr(policy, "modality", "unavailable")
            if policy is not None
            else "unavailable",
        )
    chunk_boundaries_enabled = bool(
        policy.concise_chunking_active
        and policy.max_spoken_chunk_chars > 0
        and caps.supports_chunk_boundaries
    )
    interruption_flush_enabled = bool(caps.supports_interruption_flush)
    interruption_discard_enabled = bool(caps.supports_interruption_discard)
    capability_noops: list[str] = []
    unsupported_hints = list(policy.unsupported_hints)
    requested_hints = list(policy.active_hints)
    if policy.concise_chunking_active:
        requested_hints.append("chunk_boundaries")
    requested_hints.extend(
        (
            "interruption_flush",
            "speech_rate",
            "prosody_emphasis",
            "pause_timing",
            "partial_stream_abort",
        )
    )
    applied_hints: list[str] = []

    if policy.concise_chunking_active and not caps.supports_chunk_boundaries:
        capability_noops.append("voice_actuation_noop:chunk_boundaries_unsupported")
        unsupported_hints.append("chunk_boundaries")
    elif chunk_boundaries_enabled:
        applied_hints.append("chunk_boundaries")

    if interruption_flush_enabled:
        applied_hints.append("interruption_flush")

    capability_pairs = (
        ("pause_timing", caps.supports_pause_timing),
        ("speech_rate", caps.supports_speech_rate),
        ("prosody_emphasis", caps.supports_prosody_emphasis),
        ("partial_stream_abort", caps.supports_partial_stream_abort),
    )
    for hint, supported in capability_pairs:
        if supported:
            applied_hints.append(hint)
        else:
            unsupported_hints.append(hint)
            capability_noops.append(f"voice_actuation_noop:{hint}_unsupported")
    if not caps.supports_interruption_flush:
        unsupported_hints.append("interruption_flush")
        capability_noops.append("voice_actuation_noop:interruption_flush_unsupported")
    if not caps.supports_interruption_discard:
        unsupported_hints.append("interruption_discard")
        capability_noops.append("voice_actuation_noop:interruption_discard_unsupported")
    else:
        applied_hints.append("interruption_discard")
    unsupported_hints.append("hardware_control")
    capability_noops.append("voice_actuation_noop:hardware_control_forbidden")

    requested_hints = list(_dedupe_preserve_order(tuple(requested_hints)))
    applied_hints = list(_dedupe_preserve_order(tuple(applied_hints)))
    unsupported_hints = [
        hint
        for hint in _dedupe_preserve_order(tuple(unsupported_hints))
        if hint not in applied_hints
    ]
    dropped_hints = [
        hint for hint in requested_hints if hint not in applied_hints and hint in unsupported_hints
    ]
    reason_codes = _dedupe_preserve_order(
        (
            "voice_actuation:available",
            f"voice_actuation_backend:{adapter.backend_label}",
            f"voice_actuation_chunking:{policy.chunking_mode if chunk_boundaries_enabled else 'off'}",
            f"voice_actuation_requested_hints:{len(requested_hints)}",
            f"voice_actuation_applied_hints:{len(applied_hints)}",
            f"voice_actuation_dropped_hints:{len(dropped_hints)}",
            *adapter.reason_codes,
            *caps.reason_codes,
            *capability_noops,
        )
    )
    return BrainRealtimeVoiceActuationPlan(
        available=True,
        backend_label=adapter.backend_label,
        modality=policy.modality,
        chunk_boundaries_enabled=chunk_boundaries_enabled,
        interruption_flush_enabled=interruption_flush_enabled,
        interruption_discard_enabled=interruption_discard_enabled,
        pause_timing_enabled=bool(caps.supports_pause_timing),
        speech_rate_enabled=bool(caps.supports_speech_rate),
        prosody_emphasis_enabled=bool(caps.supports_prosody_emphasis),
        partial_stream_abort_enabled=bool(caps.supports_partial_stream_abort),
        expression_controls_hardware=False,
        chunking_mode=policy.chunking_mode if chunk_boundaries_enabled else "off",
        max_spoken_chunk_chars=policy.max_spoken_chunk_chars if chunk_boundaries_enabled else 0,
        interruption_strategy_label=policy.interruption_strategy_label,
        pause_yield_hint=policy.pause_yield_hint,
        requested_hints=tuple(requested_hints),
        applied_hints=tuple(applied_hints),
        active_hints=tuple(applied_hints),
        unsupported_hints=tuple(unsupported_hints),
        noop_reason_codes=_dedupe_preserve_order((*policy.noop_reason_codes, *capability_noops)),
        capability_reason_codes=_dedupe_preserve_order((*adapter.reason_codes, *caps.reason_codes)),
        reason_codes=reason_codes,
    )


__all__ = [
    "BrainRealtimeVoiceActuationPlan",
    "compile_realtime_voice_actuation_plan",
    "unavailable_realtime_voice_actuation_plan",
]

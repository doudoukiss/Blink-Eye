"""Provider-neutral voice backend capability accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class BrainVoiceBackendCapabilities:
    """Capability matrix for one voice runtime/backend seam."""

    backend_label: str
    supports_chunk_boundaries: bool = True
    supports_interruption_flush: bool = True
    supports_interruption_discard: bool = False
    supports_pause_timing: bool = False
    supports_speech_rate: bool = False
    supports_prosody_emphasis: bool = False
    supports_partial_stream_abort: bool = False
    expression_controls_hardware: bool = False
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the voice backend capability matrix."""
        return {
            "backend_label": self.backend_label,
            "supports_chunk_boundaries": self.supports_chunk_boundaries,
            "supports_interruption_flush": self.supports_interruption_flush,
            "supports_interruption_discard": self.supports_interruption_discard,
            "supports_pause_timing": self.supports_pause_timing,
            "supports_speech_rate": self.supports_speech_rate,
            "supports_prosody_emphasis": self.supports_prosody_emphasis,
            "supports_partial_stream_abort": self.supports_partial_stream_abort,
            "expression_controls_hardware": self.expression_controls_hardware,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainVoiceBackendCapabilities":
        """Hydrate a capability matrix from a dict."""
        payload = dict(data or {})
        backend_label = str(payload.get("backend_label") or "provider-neutral").strip()
        return cls(
            backend_label=backend_label or "provider-neutral",
            supports_chunk_boundaries=bool(payload.get("supports_chunk_boundaries", True)),
            supports_interruption_flush=bool(payload.get("supports_interruption_flush", True)),
            supports_interruption_discard=bool(payload.get("supports_interruption_discard", False)),
            supports_pause_timing=bool(payload.get("supports_pause_timing", False)),
            supports_speech_rate=bool(payload.get("supports_speech_rate", False)),
            supports_prosody_emphasis=bool(payload.get("supports_prosody_emphasis", False)),
            supports_partial_stream_abort=bool(payload.get("supports_partial_stream_abort", False)),
            expression_controls_hardware=False,
            reason_codes=_dedupe_preserve_order(tuple(payload.get("reason_codes") or ())),
        )


def provider_neutral_voice_capabilities(
    *,
    backend_label: str | None = None,
) -> BrainVoiceBackendCapabilities:
    """Return the conservative default capability matrix for Blink voice runtimes."""
    label = str(backend_label or "provider-neutral").strip() or "provider-neutral"
    return BrainVoiceBackendCapabilities(
        backend_label=label,
        supports_chunk_boundaries=True,
        supports_interruption_flush=True,
        supports_interruption_discard=False,
        supports_pause_timing=False,
        supports_speech_rate=False,
        supports_prosody_emphasis=False,
        supports_partial_stream_abort=False,
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "voice_capabilities:v1",
                f"voice_capabilities_backend:{label}",
                "voice_capability_supported:chunk_boundaries",
                "voice_capability_supported:interruption_flush",
                "voice_capability_noop:interruption_discard",
                "voice_capability_noop:pause_timing",
                "voice_capability_noop:speech_rate",
                "voice_capability_noop:prosody_emphasis",
                "voice_capability_noop:partial_stream_abort",
                "voice_capability_noop:hardware_control_forbidden",
            )
        ),
    )


__all__ = [
    "BrainVoiceBackendCapabilities",
    "provider_neutral_voice_capabilities",
]

"""Backend-specific voice capability registry for Blink runtime surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from blink.brain.persona.voice_capabilities import (
    BrainVoiceBackendCapabilities,
    provider_neutral_voice_capabilities,
)

_KNOWN_BACKEND_LABELS = (
    "kokoro",
    "local-http-wav",
    "piper",
    "xtts",
)


def normalize_voice_backend_label(tts_backend: str | None) -> str:
    """Normalize a TTS backend label for capability registry lookups."""
    label = str(tts_backend or "provider-neutral").strip().lower()
    return label or "provider-neutral"


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


def _known_text_chunk_backend_capabilities(
    backend_label: str,
) -> BrainVoiceBackendCapabilities:
    """Return conservative capabilities for local text-to-WAV style backends."""
    backend_specific_reason = (
        "voice_backend_registry:kokoro_chunked"
        if backend_label == "kokoro"
        else "voice_backend_registry:local_http_wav_melo"
        if backend_label == "local-http-wav"
        else f"voice_backend_registry:{backend_label}"
    )
    return BrainVoiceBackendCapabilities(
        backend_label=backend_label,
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
                f"voice_capabilities_backend:{backend_label}",
                "voice_backend_registry:known_backend",
                backend_specific_reason,
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


def default_voice_backend_capability_entries() -> dict[str, BrainVoiceBackendCapabilities]:
    """Return the built-in conservative voice backend capability profiles."""
    return {
        label: _known_text_chunk_backend_capabilities(label)
        for label in _KNOWN_BACKEND_LABELS
    }


@dataclass(frozen=True)
class BrainVoiceBackendCapabilityResolution:
    """Resolved backend capability profile and lookup accounting."""

    requested_backend_label: str
    resolved_backend_label: str
    capabilities: BrainVoiceBackendCapabilities
    fallback_used: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the capability resolution."""
        return {
            "requested_backend_label": self.requested_backend_label,
            "resolved_backend_label": self.resolved_backend_label,
            "capabilities": self.capabilities.as_dict(),
            "fallback_used": self.fallback_used,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class BrainVoiceBackendCapabilityRegistry:
    """Deterministic registry of backend-specific voice capability profiles."""

    entries: dict[str, BrainVoiceBackendCapabilities] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize registry keys and capability labels."""
        normalized: dict[str, BrainVoiceBackendCapabilities] = {}
        for key, capabilities in dict(self.entries).items():
            label = normalize_voice_backend_label(key)
            normalized[label] = capabilities
        self.entries = normalized

    @classmethod
    def default(cls) -> "BrainVoiceBackendCapabilityRegistry":
        """Return the default registry for local Blink voice backends."""
        return cls(entries=default_voice_backend_capability_entries())

    def resolve(
        self,
        tts_backend: str | None = None,
        *,
        capabilities_override: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
    ) -> BrainVoiceBackendCapabilityResolution:
        """Resolve backend capabilities with explicit override and fallback accounting."""
        requested_label = normalize_voice_backend_label(tts_backend)
        if isinstance(capabilities_override, BrainVoiceBackendCapabilities):
            capabilities = capabilities_override
            resolved_label = normalize_voice_backend_label(capabilities.backend_label)
            return BrainVoiceBackendCapabilityResolution(
                requested_backend_label=requested_label,
                resolved_backend_label=resolved_label,
                capabilities=capabilities,
                fallback_used=False,
                reason_codes=_dedupe_preserve_order(
                    (
                        "voice_backend_registry:v1",
                        f"voice_backend_registry_requested:{requested_label}",
                        f"voice_backend_registry_resolved:{resolved_label}",
                        "voice_backend_registry:override",
                    )
                ),
            )
        if isinstance(capabilities_override, dict):
            capabilities = BrainVoiceBackendCapabilities.from_dict(capabilities_override)
            resolved_label = normalize_voice_backend_label(capabilities.backend_label)
            return BrainVoiceBackendCapabilityResolution(
                requested_backend_label=requested_label,
                resolved_backend_label=resolved_label,
                capabilities=capabilities,
                fallback_used=False,
                reason_codes=_dedupe_preserve_order(
                    (
                        "voice_backend_registry:v1",
                        f"voice_backend_registry_requested:{requested_label}",
                        f"voice_backend_registry_resolved:{resolved_label}",
                        "voice_backend_registry:override_dict",
                    )
                ),
            )

        capabilities = self.entries.get(requested_label)
        if capabilities is not None:
            resolved_label = normalize_voice_backend_label(capabilities.backend_label)
            return BrainVoiceBackendCapabilityResolution(
                requested_backend_label=requested_label,
                resolved_backend_label=resolved_label,
                capabilities=capabilities,
                fallback_used=False,
                reason_codes=_dedupe_preserve_order(
                    (
                        "voice_backend_registry:v1",
                        f"voice_backend_registry_requested:{requested_label}",
                        f"voice_backend_registry_resolved:{resolved_label}",
                        "voice_backend_registry:known_backend",
                    )
                ),
            )

        capabilities = provider_neutral_voice_capabilities(backend_label=requested_label)
        return BrainVoiceBackendCapabilityResolution(
            requested_backend_label=requested_label,
            resolved_backend_label=normalize_voice_backend_label(capabilities.backend_label),
            capabilities=capabilities,
            fallback_used=True,
            reason_codes=_dedupe_preserve_order(
                (
                    "voice_backend_registry:v1",
                    f"voice_backend_registry_requested:{requested_label}",
                    f"voice_backend_registry_resolved:{capabilities.backend_label}",
                    "voice_backend_registry:fallback_provider_neutral",
                )
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the registry without runtime-only state."""
        return {
            "backend_labels": sorted(self.entries.keys()),
            "capabilities": {
                key: value.as_dict() for key, value in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_mapping(
        cls,
        entries: Mapping[str, BrainVoiceBackendCapabilities | dict[str, Any]] | None,
    ) -> "BrainVoiceBackendCapabilityRegistry":
        """Hydrate a registry from capability entries."""
        rows: dict[str, BrainVoiceBackendCapabilities] = {}
        for key, value in dict(entries or {}).items():
            label = normalize_voice_backend_label(key)
            if isinstance(value, BrainVoiceBackendCapabilities):
                rows[label] = value
            elif isinstance(value, dict):
                rows[label] = BrainVoiceBackendCapabilities.from_dict(value)
        return cls(entries=rows)


def resolve_voice_backend_capabilities(
    tts_backend: str | None = None,
    *,
    registry: BrainVoiceBackendCapabilityRegistry | None = None,
    capabilities_override: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
) -> BrainVoiceBackendCapabilityResolution:
    """Resolve capabilities through the supplied or default backend registry."""
    selected_registry = registry or BrainVoiceBackendCapabilityRegistry.default()
    return selected_registry.resolve(
        tts_backend,
        capabilities_override=capabilities_override,
    )


__all__ = [
    "BrainVoiceBackendCapabilityRegistry",
    "BrainVoiceBackendCapabilityResolution",
    "default_voice_backend_capability_entries",
    "normalize_voice_backend_label",
    "resolve_voice_backend_capabilities",
]

"""Compatibility adapter wrapper for backend voice capability resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blink.brain.persona.voice_backend_registry import (
    BrainVoiceBackendCapabilityRegistry,
    normalize_voice_backend_label,
    resolve_voice_backend_capabilities,
)
from blink.brain.persona.voice_capabilities import (
    BrainVoiceBackendCapabilities,
)


@dataclass(frozen=True)
class BrainVoiceBackendAdapter:
    """Resolved voice backend adapter metadata and capability surface."""

    backend_label: str
    capabilities: BrainVoiceBackendCapabilities
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the adapter metadata."""
        return {
            "backend_label": self.backend_label,
            "capabilities": self.capabilities.as_dict(),
            "reason_codes": list(self.reason_codes),
        }


def resolve_voice_backend_adapter(
    tts_backend: str | None = None,
    *,
    capabilities_override: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
    registry: BrainVoiceBackendCapabilityRegistry | None = None,
) -> BrainVoiceBackendAdapter:
    """Resolve one voice backend adapter through the capability registry."""
    resolution = resolve_voice_backend_capabilities(
        tts_backend,
        registry=registry,
        capabilities_override=capabilities_override,
    )
    return BrainVoiceBackendAdapter(
        backend_label=resolution.resolved_backend_label,
        capabilities=resolution.capabilities,
        reason_codes=(
            "voice_backend_adapter:v1",
            f"voice_backend:{resolution.requested_backend_label}",
            *resolution.reason_codes,
        ),
    )


__all__ = [
    "BrainVoiceBackendAdapter",
    "normalize_voice_backend_label",
    "resolve_voice_backend_adapter",
]

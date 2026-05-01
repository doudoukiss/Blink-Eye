"""Dynamic runtime state for Blink's persona kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(frozen=True)
class BrainPersonaState:
    """Compact current-state surface for persona compilation."""

    current_mode: str = "reply"
    interaction_energy: float = 0.5
    social_distance: float = 0.4
    expressivity_boost: float = 0.0
    current_arc_summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the persona state."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPersonaState | None":
        """Hydrate one persona state from JSON-like data."""
        if not isinstance(data, dict):
            return None
        current_mode = _normalized_text(data.get("current_mode")) or "reply"
        return cls(
            current_mode=current_mode,
            interaction_energy=float(data.get("interaction_energy") or 0.5),
            social_distance=float(data.get("social_distance") or 0.4),
            expressivity_boost=float(data.get("expressivity_boost") or 0.0),
            current_arc_summary=_normalized_text(data.get("current_arc_summary")),
        )

    @classmethod
    def neutral(cls, *, current_mode: str = "reply") -> "BrainPersonaState":
        """Return a neutral persona state for the requested mode."""
        return cls(
            current_mode=_normalized_text(current_mode) or "reply",
        )


__all__ = ["BrainPersonaState"]

"""Policy helpers and invariants for Blink's persona kernel."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from blink.project_identity import PROJECT_IDENTITY

PERSONA_CANONICAL_NAME = PROJECT_IDENTITY.display_name
PERSONA_ALLOWED_ONTOLOGICAL_STATUSES = (
    "local_ai_system",
    "local_cognitive_runtime",
    "local_software_brain",
)
PERSONA_INVARIANT_GUARDRAILS = (
    "canonical_name_blink",
    "local_non_human_identity",
    "human_identity_claims_disabled",
    "no_fake_human_autobiography",
)
PERSONA_REQUIRED_RELATIONSHIP_POSTURE = (
    "warm",
    "intelligent",
    "collaborative",
    "non-romantic",
    "non-sexual",
    "non-exclusive",
)
PERSONA_REQUIRED_DEPENDENCY_GUARDRAILS = (
    "avoid guilt language",
    "avoid exclusivity",
    "encourage human support when appropriate",
)
PERSONA_RELATIONSHIP_STYLE_NAMESPACES = (
    "interaction.style",
    "interaction.preference",
    "interaction.misfire",
)
PERSONA_TEACHING_PROFILE_NAMESPACES = (
    "teaching.preference.mode",
    "teaching.preference.analogy_domain",
    "teaching.history.helpful_pattern",
)
PERSONA_RELATIONSHIP_MEMORY_NAMESPACES = (
    *PERSONA_RELATIONSHIP_STYLE_NAMESPACES,
    *PERSONA_TEACHING_PROFILE_NAMESPACES,
)


class BrainPersonaTaskMode(str, Enum):
    """Supported high-level task modes for persona compilation."""

    REPLY = "reply"
    PLANNING = "planning"
    AUDIT = "audit"
    REFLECTION = "reflection"


class BrainPersonaModality(str, Enum):
    """Supported output modalities for persona compilation."""

    TEXT = "text"
    VOICE = "voice"
    BROWSER = "browser"
    EMBODIED = "embodied"


def resolve_persona_task_mode(value: BrainPersonaTaskMode | str) -> BrainPersonaTaskMode:
    """Return a validated persona task mode."""
    if isinstance(value, BrainPersonaTaskMode):
        return value
    normalized = str(value or "").strip()
    try:
        return BrainPersonaTaskMode(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported persona task mode: {value!r}") from exc


def resolve_persona_modality(value: BrainPersonaModality | str) -> BrainPersonaModality:
    """Return a validated persona modality."""
    if isinstance(value, BrainPersonaModality):
        return value
    normalized = str(value or "").strip()
    try:
        return BrainPersonaModality(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported persona modality: {value!r}") from exc


def assert_persona_identity_invariants(
    *,
    canonical_name: str,
    ontological_status: str,
    human_identity_claims_allowed: bool,
) -> None:
    """Validate the non-negotiable Blink persona invariants."""
    if canonical_name != PERSONA_CANONICAL_NAME:
        raise ValueError(
            f"Persona canonical_name must remain {PERSONA_CANONICAL_NAME!r}, got {canonical_name!r}."
        )
    if ontological_status not in PERSONA_ALLOWED_ONTOLOGICAL_STATUSES:
        raise ValueError(
            "Persona ontological_status must remain local and non-human, "
            f"got {ontological_status!r}."
        )
    if human_identity_claims_allowed:
        raise ValueError("Persona human_identity_claims_allowed must remain false.")


def assert_relationship_style_invariants(
    *,
    default_posture: Iterable[str],
    dependency_guardrails: Iterable[str],
    intimacy_ceiling: str,
) -> None:
    """Validate the non-negotiable relationship-style invariants."""
    normalized_posture = {str(item or "").strip().lower() for item in default_posture if item}
    missing_posture = [
        item for item in PERSONA_REQUIRED_RELATIONSHIP_POSTURE if item not in normalized_posture
    ]
    if missing_posture:
        raise ValueError(
            "Relationship posture must remain warm, collaborative, and explicitly non-exclusive; "
            f"missing {missing_posture!r}."
        )

    normalized_guardrails = {
        str(item or "").strip().lower() for item in dependency_guardrails if item
    }
    missing_guardrails = [
        item
        for item in PERSONA_REQUIRED_DEPENDENCY_GUARDRAILS
        if item not in normalized_guardrails
    ]
    if missing_guardrails:
        raise ValueError(
            "Relationship dependency guardrails are incomplete; "
            f"missing {missing_guardrails!r}."
        )

    normalized_ceiling = str(intimacy_ceiling or "").strip().lower()
    banned_fragments = ("romantic", "sexual", "exclusive")
    if any(
        fragment in normalized_ceiling and f"non-{fragment}" not in normalized_ceiling
        for fragment in banned_fragments
    ):
        raise ValueError(
            "Relationship intimacy_ceiling must not permit romantic, sexual, or exclusive framing."
        )


__all__ = [
    "BrainPersonaModality",
    "BrainPersonaTaskMode",
    "PERSONA_ALLOWED_ONTOLOGICAL_STATUSES",
    "PERSONA_CANONICAL_NAME",
    "PERSONA_INVARIANT_GUARDRAILS",
    "PERSONA_RELATIONSHIP_MEMORY_NAMESPACES",
    "PERSONA_RELATIONSHIP_STYLE_NAMESPACES",
    "PERSONA_REQUIRED_DEPENDENCY_GUARDRAILS",
    "PERSONA_REQUIRED_RELATIONSHIP_POSTURE",
    "PERSONA_TEACHING_PROFILE_NAMESPACES",
    "assert_persona_identity_invariants",
    "assert_relationship_style_invariants",
    "resolve_persona_modality",
    "resolve_persona_task_mode",
]

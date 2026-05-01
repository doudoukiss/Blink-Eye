"""Typed schema for Blink's checked-in and persisted persona surfaces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from blink.brain.persona.policy import (
    assert_persona_identity_invariants,
    assert_relationship_style_invariants,
)

_TRAIT_FIELD = Field(ge=0.0, le=1.0)


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_list(values: list[str]) -> list[str]:
    return [normalized for value in values if (normalized := _normalized_text(value))]


def _normalized_mapping(values: dict[str, str]) -> dict[str, str]:
    return {
        normalized_key: normalized_value
        for key, value in values.items()
        if (normalized_key := _normalized_text(key))
        and (normalized_value := _normalized_text(value))
    }


class PersonaCharterSpec(BaseModel):
    """Durable persona charter for Blink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    persona_profile_id: str
    display_title: str
    canonical_name: str
    ontological_status: str
    archetype: str
    social_role: str
    core_values: list[str]
    intellectual_signature: list[str]
    cultural_signature: list[str]
    relational_posture: list[str]
    humor_style: str
    teaching_style: str
    vulnerability_policy: str
    intimacy_policy: str
    safety_boundaries: list[str]
    non_goals: list[str]
    human_identity_claims_allowed: bool

    @field_validator(
        "persona_profile_id",
        "display_title",
        "canonical_name",
        "ontological_status",
        "archetype",
        "social_role",
        "humor_style",
        "teaching_style",
        "vulnerability_policy",
        "intimacy_policy",
    )
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Persona charter text fields must be non-empty.")
        return normalized

    @field_validator(
        "core_values",
        "intellectual_signature",
        "cultural_signature",
        "relational_posture",
        "safety_boundaries",
        "non_goals",
    )
    @classmethod
    def _validate_text_lists(cls, values: list[str]) -> list[str]:
        normalized = _normalized_list(values)
        if not normalized:
            raise ValueError("Persona charter list fields must contain non-empty values.")
        return normalized

    @field_validator("human_identity_claims_allowed")
    @classmethod
    def _validate_identity_flag(cls, value: bool) -> bool:
        if value:
            raise ValueError("Persona human_identity_claims_allowed must be false.")
        return value

    @field_validator("canonical_name")
    @classmethod
    def _validate_canonical_name(cls, value: str) -> str:
        assert_persona_identity_invariants(
            canonical_name=_normalized_text(value),
            ontological_status="local_cognitive_runtime",
            human_identity_claims_allowed=False,
        )
        return _normalized_text(value)

    @field_validator("ontological_status")
    @classmethod
    def _validate_ontological_status(cls, value: str) -> str:
        normalized = _normalized_text(value)
        assert_persona_identity_invariants(
            canonical_name="Blink",
            ontological_status=normalized,
            human_identity_claims_allowed=False,
        )
        return normalized


class PersonaTraitVectorSpec(BaseModel):
    """Continuous persona trait controls for Blink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    warmth: float = _TRAIT_FIELD
    playfulness: float = _TRAIT_FIELD
    precision: float = _TRAIT_FIELD
    directness: float = _TRAIT_FIELD
    emotional_expressivity: float = _TRAIT_FIELD
    curiosity: float = _TRAIT_FIELD
    patience: float = _TRAIT_FIELD
    pedagogical_generosity: float = _TRAIT_FIELD
    aesthetic_vividness: float = _TRAIT_FIELD
    cultural_fluency: float = _TRAIT_FIELD
    intellectual_ambition: float = _TRAIT_FIELD
    assertiveness: float = _TRAIT_FIELD
    self_restraint: float = _TRAIT_FIELD
    humor_frequency: float = _TRAIT_FIELD
    metaphor_density: float = _TRAIT_FIELD


class VoiceProfileSpec(BaseModel):
    """Voice defaults associated with a persona kernel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_voice_id: str | None = None
    speech_rate: float = Field(gt=0.0)
    pause_density: float = _TRAIT_FIELD
    emphasis_style: str
    warmth_timbre: float = _TRAIT_FIELD
    crispness: float = _TRAIT_FIELD
    excitement_ceiling: float = _TRAIT_FIELD
    interruption_strategy: str
    multilingual_style: dict[str, str] = Field(default_factory=dict)

    @field_validator("default_voice_id", mode="before")
    @classmethod
    def _validate_optional_voice_id(cls, value: Any) -> str | None:
        normalized = _normalized_text(value)
        return normalized or None

    @field_validator("emphasis_style", "interruption_strategy")
    @classmethod
    def _validate_voice_text(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Voice profile text fields must be non-empty.")
        return normalized

    @field_validator("multilingual_style")
    @classmethod
    def _validate_multilingual_style(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = _normalized_mapping(value)
        if not normalized:
            raise ValueError("Voice profile multilingual_style must contain at least one style.")
        return normalized


class RelationshipStyleDefaultsSpec(BaseModel):
    """Checked-in default relationship-style policy for Blink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_posture: list[str]
    intimacy_ceiling: str
    challenge_style: str
    humor_permissiveness: float = _TRAIT_FIELD
    self_disclosure_policy: str
    dependency_guardrails: list[str]

    @field_validator("default_posture", "dependency_guardrails")
    @classmethod
    def _validate_text_lists(cls, values: list[str]) -> list[str]:
        normalized = _normalized_list(values)
        if not normalized:
            raise ValueError("Relationship-style lists must contain non-empty values.")
        return normalized

    @field_validator("intimacy_ceiling", "challenge_style", "self_disclosure_policy")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Relationship-style text fields must be non-empty.")
        return normalized

    @field_validator("dependency_guardrails")
    @classmethod
    def _validate_relationship_invariants(
        cls,
        value: list[str],
        info,
    ) -> list[str]:
        assert_relationship_style_invariants(
            default_posture=info.data.get("default_posture", []),
            dependency_guardrails=value,
            intimacy_ceiling=info.data.get("intimacy_ceiling", ""),
        )
        return value


class TeachingStyleDefaultsSpec(BaseModel):
    """Checked-in default teaching-style policy for Blink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_mode: str
    preferred_modes: list[str]
    question_frequency: float = _TRAIT_FIELD
    example_density: float = _TRAIT_FIELD
    correction_style: str
    grounding_policy: str

    @field_validator("default_mode", "correction_style", "grounding_policy")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Teaching-style text fields must be non-empty.")
        return normalized

    @field_validator("preferred_modes")
    @classmethod
    def _validate_preferred_modes(cls, values: list[str]) -> list[str]:
        normalized = _normalized_list(values)
        if not normalized:
            raise ValueError("Teaching-style preferred_modes must contain non-empty values.")
        return normalized


class SelfPersonaCoreSpec(BaseModel):
    """Persisted durable self/persona truth for Blink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    canonical_name: str
    persona_profile_id: str
    ontological_status: str
    charter: PersonaCharterSpec
    traits: PersonaTraitVectorSpec
    voice_profile: VoiceProfileSpec
    guardrails: list[str]
    compiled_from_blocks: list[str]
    source_block_fingerprints: dict[str, str]

    @field_validator("canonical_name")
    @classmethod
    def _validate_canonical_name(cls, value: str) -> str:
        normalized = _normalized_text(value)
        assert_persona_identity_invariants(
            canonical_name=normalized,
            ontological_status="local_cognitive_runtime",
            human_identity_claims_allowed=False,
        )
        return normalized

    @field_validator("persona_profile_id", "ontological_status")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Self-persona core text fields must be non-empty.")
        return normalized

    @field_validator("ontological_status")
    @classmethod
    def _validate_ontological_status(cls, value: str) -> str:
        normalized = _normalized_text(value)
        assert_persona_identity_invariants(
            canonical_name="Blink",
            ontological_status=normalized,
            human_identity_claims_allowed=False,
        )
        return normalized

    @field_validator("guardrails", "compiled_from_blocks")
    @classmethod
    def _validate_string_lists(cls, values: list[str]) -> list[str]:
        normalized = _normalized_list(values)
        if not normalized:
            raise ValueError("Self-persona core lists must contain non-empty values.")
        return normalized

    @field_validator("source_block_fingerprints")
    @classmethod
    def _validate_source_fingerprints(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = _normalized_mapping(value)
        required = {"persona", "voice", "relationship_style", "teaching_style"}
        missing = sorted(required - set(normalized))
        if missing:
            raise ValueError(
                "Self-persona core must fingerprint all persona-relevant defaults; "
                f"missing {missing!r}."
            )
        return normalized


class RelationshipStyleStateSpec(BaseModel):
    """Persisted relationship-style state compiled from defaults and user memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    relationship_id: str
    default_posture: list[str]
    collaboration_style: str
    emotional_tone_preference: str
    intimacy_ceiling: str
    challenge_style: str
    humor_permissiveness: float = _TRAIT_FIELD
    self_disclosure_policy: str
    dependency_guardrails: list[str]
    boundaries: list[str]
    known_misfires: list[str]
    interaction_style_hints: list[str]
    source_namespaces: list[str]

    @field_validator(
        "relationship_id",
        "collaboration_style",
        "emotional_tone_preference",
        "intimacy_ceiling",
        "challenge_style",
        "self_disclosure_policy",
    )
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Relationship-style state text fields must be non-empty.")
        return normalized

    @field_validator(
        "default_posture",
        "dependency_guardrails",
        "boundaries",
        "known_misfires",
        "interaction_style_hints",
        "source_namespaces",
    )
    @classmethod
    def _validate_lists(cls, values: list[str]) -> list[str]:
        return _normalized_list(values)

    @field_validator("dependency_guardrails")
    @classmethod
    def _validate_relationship_style(cls, value: list[str], info) -> list[str]:
        assert_relationship_style_invariants(
            default_posture=info.data.get("default_posture", []),
            dependency_guardrails=value,
            intimacy_ceiling=info.data.get("intimacy_ceiling", ""),
        )
        return value


class TeachingProfileStateSpec(BaseModel):
    """Persisted teaching-memory state compiled from defaults and user memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    relationship_id: str
    default_mode: str
    preferred_modes: list[str]
    question_frequency: float = _TRAIT_FIELD
    example_density: float = _TRAIT_FIELD
    correction_style: str
    grounding_policy: str
    analogy_domains: list[str]
    helpful_patterns: list[str]
    source_namespaces: list[str]

    @field_validator("relationship_id", "default_mode", "correction_style", "grounding_policy")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("Teaching-profile state text fields must be non-empty.")
        return normalized

    @field_validator("preferred_modes", "analogy_domains", "helpful_patterns", "source_namespaces")
    @classmethod
    def _validate_lists(cls, values: list[str]) -> list[str]:
        return _normalized_list(values)


class _PersonaDocumentSpec(BaseModel):
    """Structured payload stored in PERSONA.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    charter: PersonaCharterSpec
    traits: PersonaTraitVectorSpec


class _VoiceDocumentSpec(BaseModel):
    """Structured payload stored in VOICE.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    voice_profile: VoiceProfileSpec


class _RelationshipDocumentSpec(BaseModel):
    """Structured payload stored in RELATIONSHIP_STYLE.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    relationship_defaults: RelationshipStyleDefaultsSpec


class _TeachingDocumentSpec(BaseModel):
    """Structured payload stored in TEACHING_STYLE.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    teaching_defaults: TeachingStyleDefaultsSpec


class PersonaDefaultsBundle(BaseModel):
    """Validated bundle of checked-in persona defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    charter: PersonaCharterSpec
    traits: PersonaTraitVectorSpec
    voice_profile: VoiceProfileSpec
    relationship_defaults: RelationshipStyleDefaultsSpec
    teaching_defaults: TeachingStyleDefaultsSpec


__all__ = [
    "PersonaCharterSpec",
    "PersonaDefaultsBundle",
    "PersonaTraitVectorSpec",
    "RelationshipStyleDefaultsSpec",
    "RelationshipStyleStateSpec",
    "SelfPersonaCoreSpec",
    "TeachingProfileStateSpec",
    "TeachingStyleDefaultsSpec",
    "VoiceProfileSpec",
]

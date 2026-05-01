"""Typed schemas for Blink's opt-in teaching knowledge scaffold."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_lower(value: Any) -> str:
    return _normalized_text(value).lower()


def _normalized_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    raw_values = (values,) if isinstance(values, str) else values
    return tuple(normalized for value in raw_values if (normalized := _normalized_text(value)))


def _normalized_lower_tuple(values: Any) -> tuple[str, ...]:
    return tuple(value.lower() for value in _normalized_tuple(values))


def _validate_required_text(value: str) -> str:
    normalized = _normalized_text(value)
    if not normalized:
        raise ValueError("Knowledge record text fields must be non-empty.")
    return normalized


def _validate_required_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one non-empty value.")
    return values


class _KnowledgeRecordBase(BaseModel):
    """Common immutable fields for curated knowledge records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source: str
    provenance: dict[str, str]

    @field_validator("source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        return _validate_required_text(value)

    @field_validator("provenance", mode="before")
    @classmethod
    def _validate_provenance(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError("provenance must be a non-empty mapping.")
        normalized = {
            key_text: value_text
            for key, raw_value in value.items()
            if (key_text := _normalized_text(key)) and (value_text := _normalized_text(raw_value))
        }
        if not normalized:
            raise ValueError("provenance must contain at least one non-empty field.")
        return normalized


class KnowledgeReserveEntry(_KnowledgeRecordBase):
    """Curated teaching or factual anchor available for explicit selection."""

    entry_id: str
    title: str
    summary: str
    body: str
    tags: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en",)
    task_modes: tuple[str, ...] = ("reply",)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("entry_id", "title", "summary", "body")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_required_text(value)

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> tuple[str, ...]:
        return _normalized_lower_tuple(value)

    @field_validator("languages", "task_modes", mode="before")
    @classmethod
    def _validate_modes(cls, value: Any) -> tuple[str, ...]:
        return _normalized_lower_tuple(value)

    @field_validator("languages", "task_modes")
    @classmethod
    def _validate_non_empty_modes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_required_tuple(value, field_name="languages/task_modes")


class ConceptMapEntry(_KnowledgeRecordBase):
    """Small concept map node that can support future teaching selection."""

    concept_id: str
    label: str
    summary: str
    prerequisites: tuple[str, ...] = ()
    related_concepts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @field_validator("concept_id", "label", "summary")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_required_text(value)

    @field_validator("prerequisites", "related_concepts", "tags", mode="before")
    @classmethod
    def _validate_text_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_lower_tuple(value)


class ExplanationExemplar(_KnowledgeRecordBase):
    """Opt-in explanation example selected by task, language, mode, and query."""

    exemplar_id: str
    title: str
    query_terms: tuple[str, ...]
    teaching_modes: tuple[str, ...] = ()
    task_modes: tuple[str, ...] = ("reply",)
    language: str = "en"
    prompt_pattern: str
    explanation: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("exemplar_id", "title", "prompt_pattern", "explanation")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_required_text(value)

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> str:
        return _normalized_lower(value) or "en"

    @field_validator("query_terms", "teaching_modes", "task_modes", mode="before")
    @classmethod
    def _validate_text_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_lower_tuple(value)

    @field_validator("query_terms", "task_modes")
    @classmethod
    def _validate_required_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_required_tuple(value, field_name="query_terms/task_modes")


class TeachingSequence(_KnowledgeRecordBase):
    """Ordered teaching flow selected explicitly for a task and query."""

    sequence_id: str
    title: str
    query_terms: tuple[str, ...]
    teaching_modes: tuple[str, ...] = ()
    task_modes: tuple[str, ...] = ("reply",)
    language: str = "en"
    steps: tuple[str, ...]
    priority: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("sequence_id", "title")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _validate_required_text(value)

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> str:
        return _normalized_lower(value) or "en"

    @field_validator("query_terms", "teaching_modes", "task_modes", mode="before")
    @classmethod
    def _validate_lower_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_lower_tuple(value)

    @field_validator("steps", mode="before")
    @classmethod
    def _validate_step_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalized_tuple(value)

    @field_validator("query_terms", "task_modes", "steps")
    @classmethod
    def _validate_required_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_required_tuple(value, field_name="query_terms/task_modes/steps")


__all__ = [
    "ConceptMapEntry",
    "ExplanationExemplar",
    "KnowledgeReserveEntry",
    "TeachingSequence",
]

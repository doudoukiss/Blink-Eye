"""Immutable registry for curated Blink teaching knowledge records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from blink.brain.knowledge.schema import (
    ConceptMapEntry,
    ExplanationExemplar,
    KnowledgeReserveEntry,
    TeachingSequence,
)


@dataclass(frozen=True)
class KnowledgeReserveRegistry:
    """Small immutable container for opt-in teaching knowledge records."""

    entries: tuple[KnowledgeReserveEntry, ...] = ()
    concept_maps: tuple[ConceptMapEntry, ...] = ()
    exemplars: tuple[ExplanationExemplar, ...] = ()
    teaching_sequences: tuple[TeachingSequence, ...] = ()

    @classmethod
    def empty(cls) -> "KnowledgeReserveRegistry":
        """Return an empty registry."""
        return cls()

    @classmethod
    def from_entries(
        cls,
        *,
        entries: Iterable[KnowledgeReserveEntry] = (),
        concept_maps: Iterable[ConceptMapEntry] = (),
        exemplars: Iterable[ExplanationExemplar] = (),
        teaching_sequences: Iterable[TeachingSequence] = (),
    ) -> "KnowledgeReserveRegistry":
        """Return a registry with stable id ordering."""
        return cls(
            entries=tuple(sorted(entries, key=lambda item: item.entry_id)),
            concept_maps=tuple(sorted(concept_maps, key=lambda item: item.concept_id)),
            exemplars=tuple(sorted(exemplars, key=lambda item: item.exemplar_id)),
            teaching_sequences=tuple(sorted(teaching_sequences, key=lambda item: item.sequence_id)),
        )

    @property
    def is_empty(self) -> bool:
        """Return whether the registry has no records."""
        return not (self.entries or self.concept_maps or self.exemplars or self.teaching_sequences)


__all__ = ["KnowledgeReserveRegistry"]

"""Typed retrieval interfaces for Blink's layered memory system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class BrainMemoryEmbeddingProvider(Protocol):
    """Optional embedding provider interface for retrieval reranking."""

    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass(frozen=True)
class BrainMemoryQuery:
    """Typed retrieval request across one or more memory layers."""

    user_id: str
    text: str = ""
    thread_id: str | None = None
    layers: tuple[str, ...] = ("semantic", "narrative", "episodic")
    namespaces: tuple[str, ...] = ()
    narrative_kinds: tuple[str, ...] = ()
    episodic_kinds: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ("active", "open")
    limit: int = 8
    include_stale: bool = False


@dataclass(frozen=True)
class BrainMemorySearchResult:
    """One retrieval hit from the layered memory system."""

    layer: str
    record_id: int
    summary: str
    score: float
    confidence: float
    stale: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class BrainMemoryRetriever:
    """Typed retrieval orchestrator across semantic, narrative, and episodic layers."""

    def __init__(self, *, store, embedding_provider: BrainMemoryEmbeddingProvider | None = None):
        """Initialize the retriever.

        Args:
            store: Canonical brain store.
            embedding_provider: Optional embedding provider for future reranking.
        """
        self._store = store
        self._embedding_provider = embedding_provider

    def retrieve(self, query: BrainMemoryQuery) -> list[BrainMemorySearchResult]:
        """Return typed retrieval hits across the requested memory layers."""
        results: list[BrainMemorySearchResult] = []

        if "semantic" in query.layers:
            results.extend(
                self._store.search_semantic_memories(
                    user_id=query.user_id,
                    text=query.text,
                    namespaces=query.namespaces or None,
                    limit=query.limit,
                    include_stale=query.include_stale,
                )
            )

        if "narrative" in query.layers:
            results.extend(
                self._store.search_narrative_memories(
                    user_id=query.user_id,
                    thread_id=query.thread_id,
                    text=query.text,
                    kinds=query.narrative_kinds or None,
                    limit=query.limit,
                    include_stale=query.include_stale,
                )
            )

        if "episodic" in query.layers:
            results.extend(
                self._store.search_episodic_memories(
                    user_id=query.user_id,
                    thread_id=query.thread_id,
                    text=query.text,
                    kinds=query.episodic_kinds or None,
                    limit=query.limit,
                    include_stale=query.include_stale,
                )
            )

        ranked = sorted(results, key=lambda item: (item.score, item.confidence), reverse=True)
        return ranked[: query.limit]

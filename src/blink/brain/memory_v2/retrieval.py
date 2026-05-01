"""Current-truth-first retrieval over Blink continuity memory."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from blink.brain.memory_layers.retrieval import BrainMemorySearchResult
from blink.brain.memory_v2.claims import (
    BrainClaimRecord,
    BrainClaimTemporalMode,
    ClaimLedger,
    render_claim_summary,
)
from blink.brain.memory_v2.continuity_trace import expand_bilingual_memory_query
from blink.brain.memory_v2.governance import claim_matches_temporal_mode


@dataclass(frozen=True)
class BrainContinuityQuery:
    """Typed continuity-memory retrieval request."""

    text: str = ""
    temporal_mode: BrainClaimTemporalMode | str = BrainClaimTemporalMode.CURRENT
    subject_entity_id: str | None = None
    predicate: str | None = None
    entity_type: str | None = None
    scope_type: str | None = None
    scope_id: str | None = None
    currentness_states: tuple[str, ...] | None = None
    review_states: tuple[str, ...] | None = None
    retention_classes: tuple[str, ...] | None = None
    limit: int = 8


@dataclass(frozen=True)
class BrainContinuitySearchResult:
    """One continuity retrieval hit."""

    claim: BrainClaimRecord
    summary: str
    score: float
    current: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuityRetriever:
    """Current-truth-first continuity retriever with SQLite FTS baseline."""

    def __init__(self, *, store):
        """Bind the retriever to one canonical store."""
        self._store = store
        self._ledger = ClaimLedger(store=store)

    def retrieve(self, query: BrainContinuityQuery) -> list[BrainContinuitySearchResult]:
        """Return current-truth-first continuity hits."""
        if query.text.strip() and getattr(self._store, "_fts_enabled", False):
            try:
                fts_results = self._search_fts(query)
            except sqlite3.OperationalError:
                fts_results = []
            if fts_results:
                return fts_results[: query.limit]
        return self._search_lexical(query)[: query.limit]

    def retrieve_as_memory_results(
        self, query: BrainContinuityQuery
    ) -> list[BrainMemorySearchResult]:
        """Return continuity hits shaped like legacy layered retrieval results."""
        return [
            BrainMemorySearchResult(
                layer="continuity",
                record_id=index + 1,
                summary=result.summary,
                score=result.score,
                confidence=result.claim.confidence,
                stale=result.claim.is_stale,
                metadata={
                    **result.metadata,
                    "claim_id": result.claim.claim_id,
                    "predicate": result.claim.predicate,
                    "current": result.current,
                },
            )
            for index, result in enumerate(self.retrieve(query))
        ]

    def _search_fts(self, query: BrainContinuityQuery) -> list[BrainContinuitySearchResult]:
        clauses = []
        params: list[Any] = []
        if query.subject_entity_id is not None:
            clauses.append("c.subject_entity_id = ?")
            params.append(query.subject_entity_id)
        if query.predicate is not None:
            clauses.append("c.predicate = ?")
            params.append(query.predicate)
        if query.scope_type is not None:
            clauses.append("c.scope_type = ?")
            params.append(query.scope_type)
        if query.scope_id is not None:
            clauses.append("c.scope_id = ?")
            params.append(query.scope_id)
        fts_query = _normalize_fts_query(expand_bilingual_memory_query(query.text))
        if not fts_query:
            return []
        clauses.append("memory_claims_fts MATCH ?")
        where = f"WHERE {' AND '.join(clauses)}"
        rows = self._store._conn.execute(
            f"""
            SELECT c.*, s.canonical_name AS subject_name, o.canonical_name AS object_name,
                   bm25(memory_claims_fts) AS fts_rank
            FROM memory_claims_fts
            JOIN claims c ON c.claim_id = memory_claims_fts.claim_id
            LEFT JOIN entities s ON s.entity_id = c.subject_entity_id
            LEFT JOIN entities o ON o.entity_id = c.object_entity_id
            {where}
            ORDER BY fts_rank ASC, c.confidence DESC, c.updated_at DESC
            LIMIT ?
            """,
            (*params, fts_query, max(query.limit * 8, query.limit)),
        ).fetchall()
        results: list[BrainContinuitySearchResult] = []
        now = self._store._utc_now_for_memory_v2()
        currentness_states = (
            {str(getattr(item, "value", item)) for item in query.currentness_states}
            if query.currentness_states is not None
            else None
        )
        review_states = (
            {str(getattr(item, "value", item)) for item in query.review_states}
            if query.review_states is not None
            else None
        )
        retention_classes = (
            {str(getattr(item, "value", item)) for item in query.retention_classes}
            if query.retention_classes is not None
            else None
        )
        for row in rows:
            claim = self._ledger.get_claim(str(row["claim_id"]))
            if claim is None:
                continue
            if not claim_matches_temporal_mode(
                temporal_mode=str(query.temporal_mode),
                currentness_status=claim.currentness_status,
                truth_status=claim.status,
                valid_from=claim.valid_from,
                valid_to=claim.valid_to,
                stale_after_seconds=claim.stale_after_seconds,
                now=now,
            ):
                continue
            if (
                currentness_states is not None
                and claim.effective_currentness_status not in currentness_states
            ):
                continue
            if review_states is not None and claim.effective_review_state not in review_states:
                continue
            if (
                retention_classes is not None
                and claim.effective_retention_class not in retention_classes
            ):
                continue
            rank = float(row["fts_rank"]) if row["fts_rank"] is not None else 1.0
            score = 1.0 / (1.0 + max(rank, 0.0))
            results.append(
                BrainContinuitySearchResult(
                    claim=claim,
                    summary=render_claim_summary(
                        claim,
                        subject_name=str(row["subject_name"])
                        if row["subject_name"] is not None
                        else None,
                        object_name=str(row["object_name"])
                        if row["object_name"] is not None
                        else None,
                    ),
                    score=score,
                    current=claim.is_current,
                    metadata={
                        "subject_name": str(row["subject_name"])
                        if row["subject_name"] is not None
                        else "",
                        "object_name": str(row["object_name"])
                        if row["object_name"] is not None
                        else "",
                    },
                )
            )
        return results

    def _search_lexical(self, query: BrainContinuityQuery) -> list[BrainContinuitySearchResult]:
        claims = self._ledger.query_claims(
            temporal_mode=query.temporal_mode,
            subject_entity_id=query.subject_entity_id,
            predicate=query.predicate,
            entity_type=query.entity_type,
            scope_type=query.scope_type,
            scope_id=query.scope_id,
            currentness_states=query.currentness_states,
            review_states=query.review_states,
            retention_classes=query.retention_classes,
            limit=max(query.limit * 4, query.limit),
        )
        expanded_text = expand_bilingual_memory_query(query.text)
        query_tokens = _lexical_query_tokens(expanded_text)
        results: list[BrainContinuitySearchResult] = []
        for claim in claims:
            summary = render_claim_summary(claim)
            haystack = " ".join([summary, claim.predicate]).lower()
            if query_tokens:
                matched = [token for token in query_tokens if token in haystack]
                if not matched:
                    continue
                score = 1.0 + len(matched) + sum(haystack.count(token) for token in matched) * 0.1
            else:
                score = 1.0
            if claim.is_current:
                score += 1.5
            if claim.is_stale:
                score -= 0.25
            results.append(
                BrainContinuitySearchResult(
                    claim=claim,
                    summary=summary,
                    score=score,
                    current=claim.is_current,
                )
            )
        return sorted(
            results,
            key=lambda item: (item.score, item.claim.confidence, item.claim.updated_at),
            reverse=True,
        )


def _normalize_fts_query(text: str) -> str:
    """Return a conservative FTS5-safe query string."""
    tokens = [token for token in re.findall(r"[\w-]+", (text or "").lower()) if token]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def _lexical_query_tokens(text: str) -> tuple[str, ...]:
    """Return conservative token fragments for bilingual lexical fallback."""
    raw = (text or "").lower()
    tokens = re.findall(r"[\w-]+|[\u3400-\u9fff]{2,}", raw)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result[:24])

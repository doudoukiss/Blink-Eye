"""Pure deterministic selectors for opt-in teaching knowledge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from blink.brain.context.budgets import approximate_token_count
from blink.brain.knowledge.registry import KnowledgeReserveRegistry
from blink.brain.knowledge.schema import (
    ExplanationExemplar,
    KnowledgeReserveEntry,
    TeachingSequence,
)

_LANGUAGE_WILDCARDS = {"*", "any", "all"}
_TASK_WILDCARDS = {"*", "any", "all"}
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff]+")


class KnowledgeSelectionRequest(BaseModel):
    """One explicit teaching-knowledge selection request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_text: str
    task_mode: str
    language: str = "en"
    teaching_mode: str | None = None
    max_items: int = Field(default=3, ge=0, le=12)
    max_tokens: int = Field(default=160, ge=0, le=1000)

    @field_validator("query_text", mode="before")
    @classmethod
    def _validate_query_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("task_mode", "language", mode="before")
    @classmethod
    def _validate_lower_text(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator("task_mode")
    @classmethod
    def _validate_task_mode(cls, value: str) -> str:
        if not value:
            raise ValueError("task_mode must be non-empty.")
        return value

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        return value or "en"

    @field_validator("teaching_mode", mode="before")
    @classmethod
    def _validate_teaching_mode(cls, value: Any) -> str | None:
        normalized = str(value or "").strip().lower()
        return normalized or None


class KnowledgeSelectionResult(BaseModel):
    """Selected teaching knowledge plus compact rendered text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_entries: tuple[KnowledgeReserveEntry, ...] = ()
    selected_exemplars: tuple[ExplanationExemplar, ...] = ()
    selected_sequences: tuple[TeachingSequence, ...] = ()
    reason_codes: tuple[str, ...] = ()
    rendered_text: str = ""
    estimated_tokens: int = 0


@dataclass(frozen=True)
class BrainKnowledgeRoutingItem:
    """One public-safe selected teaching knowledge item."""

    item_kind: str
    item_id: str
    title: str
    source_label: str
    provenance_kind: str = ""
    provenance_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the public selected item."""
        return {
            "item_kind": self.item_kind,
            "item_id": self.item_id,
            "title": self.title,
            "source_label": self.source_label,
            "provenance_kind": self.provenance_kind,
            "provenance_version": self.provenance_version,
        }


@dataclass(frozen=True)
class BrainKnowledgeRoutingDecision:
    """Compact public-safe teaching knowledge routing decision."""

    schema_version: int
    available: bool
    selection_kind: str
    task_mode: str
    language: str
    teaching_mode: str
    selected_items: tuple[BrainKnowledgeRoutingItem, ...] = ()
    estimated_tokens: int = 0
    reason_codes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        """Return a compact operator-facing summary."""
        if not self.available:
            return "Teaching knowledge routing unavailable."
        if not self.selected_items:
            return "No teaching knowledge selected."
        counts: dict[str, int] = {}
        for item in self.selected_items:
            counts[item.item_kind] = counts.get(item.item_kind, 0) + 1
        count_summary = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        return f"{len(self.selected_items)} teaching knowledge items selected: {count_summary}"

    def as_dict(self) -> dict[str, Any]:
        """Serialize without raw rendered teaching text."""
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "selection_kind": self.selection_kind,
            "task_mode": self.task_mode,
            "language": self.language,
            "teaching_mode": self.teaching_mode,
            "summary": self.summary,
            "selected_items": [item.as_dict() for item in self.selected_items],
            "estimated_tokens": self.estimated_tokens,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class _ScoredKnowledgeCandidate:
    category: str
    identifier: str
    score: float
    item: KnowledgeReserveEntry | ExplanationExemplar | TeachingSequence

    @property
    def sort_key(self) -> tuple[float, int, str]:
        category_rank = {"entry": 0, "exemplar": 1, "sequence": 2}[self.category]
        return (-self.score, category_rank, self.identifier)


def select_teaching_knowledge(
    registry: KnowledgeReserveRegistry,
    request: KnowledgeSelectionRequest,
) -> KnowledgeSelectionResult:
    """Select compact teaching knowledge deterministically for one explicit request."""
    tokens = _query_tokens(request.query_text)
    if not tokens:
        return _empty_result("knowledge_selection_empty", "query_empty")
    if request.max_items <= 0 or request.max_tokens <= 0:
        return _empty_result("knowledge_selection_empty", "selection_budget_empty")
    if registry.is_empty:
        return _empty_result("knowledge_selection_empty", "registry_empty")

    candidates: list[_ScoredKnowledgeCandidate] = []
    for entry in registry.entries:
        candidate = _score_entry(entry, request=request, tokens=tokens)
        if candidate is not None:
            candidates.append(candidate)
    for exemplar in registry.exemplars:
        candidate = _score_exemplar(exemplar, request=request, tokens=tokens)
        if candidate is not None:
            candidates.append(candidate)
    for sequence in registry.teaching_sequences:
        candidate = _score_sequence(sequence, request=request, tokens=tokens)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return _empty_result("knowledge_selection_empty", "no_query_match")

    selected_candidates: list[_ScoredKnowledgeCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.sort_key):
        if len(selected_candidates) >= request.max_items:
            break
        trial_candidates = [*selected_candidates, candidate]
        rendered = _render_candidates(trial_candidates)
        if approximate_token_count(rendered) <= request.max_tokens:
            selected_candidates.append(candidate)

    if not selected_candidates:
        return _empty_result("knowledge_selection_empty", "selection_budget_exhausted")

    rendered_text = _render_candidates(selected_candidates)
    entries = tuple(
        candidate.item for candidate in selected_candidates if candidate.category == "entry"
    )
    exemplars = tuple(
        candidate.item for candidate in selected_candidates if candidate.category == "exemplar"
    )
    sequences = tuple(
        candidate.item for candidate in selected_candidates if candidate.category == "sequence"
    )
    return KnowledgeSelectionResult(
        selected_entries=entries,
        selected_exemplars=exemplars,
        selected_sequences=sequences,
        reason_codes=(
            "knowledge_selection_candidate",
            "knowledge_selection_selected",
            f"selected_items:{len(selected_candidates)}",
        ),
        rendered_text=rendered_text,
        estimated_tokens=approximate_token_count(rendered_text),
    )


def render_selected_teaching_knowledge(result: KnowledgeSelectionResult) -> str:
    """Return compact selected teaching knowledge text."""
    return result.rendered_text.strip()


def knowledge_routing_decision_from_selection(
    *,
    result: KnowledgeSelectionResult,
    request: KnowledgeSelectionRequest,
    selection_kind: str = "auto",
    reason_codes: tuple[str, ...] = (),
) -> BrainKnowledgeRoutingDecision:
    """Return a compact public-safe routing decision for one selection result."""
    selected_items = (
        *(
            _routing_item(
                item_kind="reserve",
                item_id=entry.entry_id,
                title=entry.title,
                source=entry.source,
                provenance=entry.provenance,
            )
            for entry in result.selected_entries
        ),
        *(
            _routing_item(
                item_kind="exemplar",
                item_id=exemplar.exemplar_id,
                title=exemplar.title,
                source=exemplar.source,
                provenance=exemplar.provenance,
            )
            for exemplar in result.selected_exemplars
        ),
        *(
            _routing_item(
                item_kind="sequence",
                item_id=sequence.sequence_id,
                title=sequence.title,
                source=sequence.source,
                provenance=sequence.provenance,
            )
            for sequence in result.selected_sequences
        ),
    )
    return BrainKnowledgeRoutingDecision(
        schema_version=1,
        available=bool(result.rendered_text.strip() or selected_items),
        selection_kind=_one_line(selection_kind, 48) or "auto",
        task_mode=_one_line(request.task_mode, 48) or "reply",
        language=_one_line(request.language, 32) or "en",
        teaching_mode=_one_line(request.teaching_mode or "none", 48),
        selected_items=selected_items,
        estimated_tokens=max(0, int(result.estimated_tokens or 0)),
        reason_codes=_dedupe_reason_codes(
            (
                "knowledge_routing_decision:v1",
                f"knowledge_selection_kind:{selection_kind or 'auto'}",
                *result.reason_codes,
                *reason_codes,
                *(f"knowledge_route:{item.item_kind}:{item.item_id}" for item in selected_items),
            )
        ),
    )


def explicit_knowledge_routing_decision(
    *,
    task_mode: str,
    language: str,
    teaching_mode: str | None,
    estimated_tokens: int,
    reason_codes: tuple[str, ...],
) -> BrainKnowledgeRoutingDecision:
    """Return a compact decision for caller-provided teaching context."""
    return BrainKnowledgeRoutingDecision(
        schema_version=1,
        available=True,
        selection_kind="explicit_context",
        task_mode=_one_line(task_mode, 48) or "reply",
        language=_one_line(language, 32) or "en",
        teaching_mode=_one_line(teaching_mode or "none", 48),
        selected_items=(),
        estimated_tokens=max(0, int(estimated_tokens or 0)),
        reason_codes=_dedupe_reason_codes(
            (
                "knowledge_routing_decision:v1",
                "knowledge_selection_kind:explicit_context",
                *reason_codes,
            )
        ),
    )


def unavailable_knowledge_routing_decision(*reason_codes: str) -> BrainKnowledgeRoutingDecision:
    """Return a public-safe unavailable routing decision."""
    return BrainKnowledgeRoutingDecision(
        schema_version=1,
        available=False,
        selection_kind="unavailable",
        task_mode="unavailable",
        language="unavailable",
        teaching_mode="unavailable",
        reason_codes=_dedupe_reason_codes(
            (
                "knowledge_routing_decision:v1",
                "knowledge_routing:unavailable",
                *reason_codes,
            )
        ),
    )


def _empty_result(*reason_codes: str) -> KnowledgeSelectionResult:
    return KnowledgeSelectionResult(reason_codes=tuple(reason_codes))


def _routing_item(
    *,
    item_kind: str,
    item_id: str,
    title: str,
    source: str,
    provenance: dict[str, str],
) -> BrainKnowledgeRoutingItem:
    return BrainKnowledgeRoutingItem(
        item_kind=_one_line(item_kind, 24),
        item_id=_one_line(item_id, 96),
        title=_one_line(title, 96),
        source_label=_safe_source_label(source),
        provenance_kind=_one_line(provenance.get("kind", ""), 48),
        provenance_version=_one_line(provenance.get("version", ""), 32),
    )


def _safe_source_label(source: str) -> str:
    normalized = _one_line(source, 96)
    if not normalized:
        return "curated"
    if normalized == "blink-default-teaching-canon":
        return normalized
    if "://" in normalized:
        return normalized.split("://", 1)[0] or "curated"
    if any(marker in normalized for marker in ("/tmp", ".db", "Traceback", "RuntimeError")):
        return "curated"
    return normalized


def _dedupe_reason_codes(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _one_line(value, 128)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _score_entry(
    entry: KnowledgeReserveEntry,
    *,
    request: KnowledgeSelectionRequest,
    tokens: tuple[str, ...],
) -> _ScoredKnowledgeCandidate | None:
    if not _task_matches(entry.task_modes, request.task_mode):
        return None
    if not _language_matches(entry.languages, request.language):
        return None
    overlap = _overlap_count(
        tokens,
        (entry.title, entry.summary, entry.body, *entry.tags),
    )
    if overlap <= 0:
        return None
    return _ScoredKnowledgeCandidate(
        category="entry",
        identifier=entry.entry_id,
        score=(10.0 * overlap) + entry.priority,
        item=entry,
    )


def _score_exemplar(
    exemplar: ExplanationExemplar,
    *,
    request: KnowledgeSelectionRequest,
    tokens: tuple[str, ...],
) -> _ScoredKnowledgeCandidate | None:
    if not _task_matches(exemplar.task_modes, request.task_mode):
        return None
    if not _single_language_matches(exemplar.language, request.language):
        return None
    if not _teaching_mode_matches(exemplar.teaching_modes, request.teaching_mode):
        return None
    overlap = _overlap_count(
        tokens,
        (
            exemplar.title,
            exemplar.prompt_pattern,
            exemplar.explanation,
            *exemplar.query_terms,
        ),
    )
    if overlap <= 0:
        return None
    return _ScoredKnowledgeCandidate(
        category="exemplar",
        identifier=exemplar.exemplar_id,
        score=(10.0 * overlap) + exemplar.priority + 0.25,
        item=exemplar,
    )


def _score_sequence(
    sequence: TeachingSequence,
    *,
    request: KnowledgeSelectionRequest,
    tokens: tuple[str, ...],
) -> _ScoredKnowledgeCandidate | None:
    if not _task_matches(sequence.task_modes, request.task_mode):
        return None
    if not _single_language_matches(sequence.language, request.language):
        return None
    if not _teaching_mode_matches(sequence.teaching_modes, request.teaching_mode):
        return None
    overlap = _overlap_count(
        tokens,
        (
            sequence.title,
            *sequence.query_terms,
            *sequence.steps,
        ),
    )
    if overlap <= 0:
        return None
    return _ScoredKnowledgeCandidate(
        category="sequence",
        identifier=sequence.sequence_id,
        score=(10.0 * overlap) + sequence.priority + 0.15,
        item=sequence,
    )


def _task_matches(task_modes: tuple[str, ...], task_mode: str) -> bool:
    normalized = {value.lower() for value in task_modes}
    return bool(normalized.intersection(_TASK_WILDCARDS)) or task_mode in normalized


def _language_matches(languages: tuple[str, ...], language: str) -> bool:
    normalized = {value.lower() for value in languages}
    variants = _language_variants(language)
    return bool(normalized.intersection(_LANGUAGE_WILDCARDS)) or bool(
        normalized.intersection(variants)
    )


def _single_language_matches(record_language: str, request_language: str) -> bool:
    normalized = record_language.lower()
    return normalized in _LANGUAGE_WILDCARDS or normalized in _language_variants(request_language)


def _teaching_mode_matches(
    teaching_modes: tuple[str, ...],
    request_teaching_mode: str | None,
) -> bool:
    if request_teaching_mode is None:
        return True
    normalized = {value.lower() for value in teaching_modes}
    return not normalized or request_teaching_mode in normalized


def _overlap_count(tokens: tuple[str, ...], parts: tuple[str, ...]) -> int:
    haystack = " ".join(str(part or "") for part in parts).lower()
    return sum(1 for token in tokens if token in haystack)


def _query_tokens(text: str) -> tuple[str, ...]:
    normalized = text.lower()
    tokens = {token for token in re.findall(r"[\w-]+", normalized) if token}
    for sequence in _CJK_SEQUENCE_RE.findall(normalized):
        tokens.update(_bounded_ngrams(sequence, sizes=(2, 3)))
    return tuple(sorted(tokens))


def _bounded_ngrams(text: str, *, sizes: tuple[int, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for size in sizes:
        if len(text) < size:
            continue
        tokens.extend(text[index : index + size] for index in range(0, len(text) - size + 1))
    return tuple(tokens)


def _language_variants(language: str) -> set[str]:
    normalized = (language or "en").lower()
    variants = {normalized}
    if "-" in normalized:
        variants.add(normalized.split("-", 1)[0])
    return variants


def _render_candidates(candidates: list[_ScoredKnowledgeCandidate]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        item = candidate.item
        if isinstance(item, KnowledgeReserveEntry):
            lines.append(
                f"- reserve:{item.entry_id}: {_one_line(item.title, 64)}; "
                f"{_one_line(item.summary, 120)} | source={_one_line(item.source, 48)}; "
                f"provenance={_render_provenance(item.provenance)}"
            )
        elif isinstance(item, ExplanationExemplar):
            lines.append(
                f"- exemplar:{item.exemplar_id}: {_one_line(item.title, 64)}; "
                f"pattern={_one_line(item.prompt_pattern, 80)}; "
                f"example={_one_line(item.explanation, 120)} | "
                f"source={_one_line(item.source, 48)}; "
                f"provenance={_render_provenance(item.provenance)}"
            )
        elif isinstance(item, TeachingSequence):
            steps = " > ".join(_one_line(step, 56) for step in item.steps[:4])
            lines.append(
                f"- sequence:{item.sequence_id}: {_one_line(item.title, 64)}; "
                f"steps={steps} | source={_one_line(item.source, 48)}; "
                f"provenance={_render_provenance(item.provenance)}"
            )
    return "\n".join(lines)


def _render_provenance(provenance: dict[str, str]) -> str:
    parts = [
        f"{_one_line(key, 18)}:{_one_line(value, 24)}"
        for key, value in sorted(provenance.items())
        if str(key).strip() and str(value).strip()
    ]
    return ",".join(parts) if parts else "unknown"


def _one_line(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


__all__ = [
    "BrainKnowledgeRoutingDecision",
    "BrainKnowledgeRoutingItem",
    "KnowledgeSelectionRequest",
    "KnowledgeSelectionResult",
    "explicit_knowledge_routing_decision",
    "knowledge_routing_decision_from_selection",
    "render_selected_teaching_knowledge",
    "select_teaching_knowledge",
    "unavailable_knowledge_routing_decision",
]

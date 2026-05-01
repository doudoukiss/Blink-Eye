"""Safe user-facing memory-use traces for Blink replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

_SCHEMA_VERSION = 1
_DISPLAY_ORDER = {
    "profile": 0,
    "preference": 1,
    "task": 2,
    "relationship_style": 3,
    "teaching_profile": 4,
    "claim": 5,
}


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _safe_reason(value: Any, *, fallback: str = "selected_for_reply_context") -> str:
    text = _normalized_text(value).replace(" ", "_").lower()
    if not text:
        return fallback
    allowed = []
    for char in text:
        if char.isalnum() or char in {"_", "-", ":"}:
            allowed.append(char)
    return "".join(allowed) or fallback


def display_kind_for_claim_predicate(predicate: str) -> str:
    """Return the Memory Palace display kind for one claim predicate."""
    normalized = _normalized_text(predicate)
    if normalized.startswith("profile."):
        return "profile"
    if normalized.startswith("preference."):
        return "preference"
    return "claim"


def render_safe_memory_provenance_label(
    *,
    display_kind: str,
    currentness_status: str | None = None,
) -> str:
    """Render a public-safe provenance label for one Memory Palace record."""
    kind = _normalized_text(display_kind)
    currentness = _normalized_text(currentness_status)
    if kind == "profile":
        return "Remembered from your profile memory."
    if kind == "preference":
        return "Remembered from your explicit preference."
    if kind == "task":
        return "Task you asked Blink to track."
    if kind == "relationship_style":
        return "Part of your relationship-style settings."
    if kind == "teaching_profile":
        return "Part of your teaching-profile settings."
    if currentness and currentness not in {"current", "none"}:
        return "Derived from a prior conversation and not recently confirmed."
    return "Derived from prior conversation memory."


@dataclass(frozen=True)
class BrainMemoryUseTraceRef:
    """One user-facing memory reference that influenced a compiled reply packet."""

    memory_id: str
    display_kind: str
    title: str
    section_key: str
    used_reason: str
    safe_provenance_label: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the trace ref without raw internal provenance."""
        return {
            "memory_id": self.memory_id,
            "display_kind": self.display_kind,
            "title": self.title,
            "section_key": self.section_key,
            "used_reason": self.used_reason,
            "safe_provenance_label": self.safe_provenance_label,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainMemoryUseTraceRef":
        """Hydrate one trace ref from serialized data."""
        return cls(
            memory_id=_normalized_text(data.get("memory_id")),
            display_kind=_normalized_text(data.get("display_kind")) or "memory",
            title=_normalized_text(data.get("title")) or "Memory",
            section_key=_normalized_text(data.get("section_key")),
            used_reason=_safe_reason(data.get("used_reason")),
            safe_provenance_label=(
                _normalized_text(data.get("safe_provenance_label"))
                or render_safe_memory_provenance_label(
                    display_kind=_normalized_text(data.get("display_kind")) or "memory",
                )
            ),
            reason_codes=_dedupe_preserve_order(data.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class BrainMemoryUseTrace:
    """Compact trace of user-facing memories selected for one reply packet."""

    schema_version: int
    user_id: str
    agent_id: str
    thread_id: str
    created_at: str
    task: str
    selected_section_names: tuple[str, ...]
    refs: tuple[BrainMemoryUseTraceRef, ...]
    summary: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the trace without raw internal provenance."""
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "task": self.task,
            "selected_section_names": list(self.selected_section_names),
            "refs": [ref.as_dict() for ref in self.refs],
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainMemoryUseTrace":
        """Hydrate one use trace from serialized data."""
        refs = tuple(
            _sorted_refs(
                BrainMemoryUseTraceRef.from_dict(item)
                for item in data.get("refs", [])
                if isinstance(item, dict)
            )
        )
        return cls(
            schema_version=int(data.get("schema_version") or _SCHEMA_VERSION),
            user_id=_normalized_text(data.get("user_id")),
            agent_id=_normalized_text(data.get("agent_id")),
            thread_id=_normalized_text(data.get("thread_id")),
            created_at=_normalized_text(data.get("created_at")),
            task=_normalized_text(data.get("task")) or "reply",
            selected_section_names=tuple(
                sorted(_dedupe_preserve_order(data.get("selected_section_names") or ()))
            ),
            refs=refs,
            summary=_normalized_text(data.get("summary")) or _trace_summary(refs),
            reason_codes=_dedupe_preserve_order(data.get("reason_codes") or ()),
        )


def _ref_sort_key(ref: BrainMemoryUseTraceRef) -> tuple[int, str, str, str]:
    return (
        _DISPLAY_ORDER.get(ref.display_kind, 99),
        ref.title.lower(),
        ref.section_key,
        ref.memory_id,
    )


def _sorted_refs(refs: Iterable[BrainMemoryUseTraceRef]) -> tuple[BrainMemoryUseTraceRef, ...]:
    unique: dict[str, BrainMemoryUseTraceRef] = {}
    for ref in refs:
        if not ref.memory_id:
            continue
        existing = unique.get(ref.memory_id)
        if existing is None or _ref_sort_key(ref) < _ref_sort_key(existing):
            unique[ref.memory_id] = ref
    return tuple(sorted(unique.values(), key=_ref_sort_key))


def _trace_summary(refs: tuple[BrainMemoryUseTraceRef, ...]) -> str:
    if not refs:
        return "No user-visible memories influenced this packet."
    counts: dict[str, int] = {}
    for ref in refs:
        counts[ref.display_kind] = counts.get(ref.display_kind, 0) + 1
    parts = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    return f"{len(refs)} user-visible memories influenced this packet: {parts}."


def build_memory_use_trace(
    *,
    user_id: str,
    agent_id: str,
    thread_id: str,
    task: str,
    selected_section_names: Iterable[Any] = (),
    refs: Iterable[BrainMemoryUseTraceRef] = (),
    created_at: str = "",
    reason_codes: Iterable[Any] = (),
) -> BrainMemoryUseTrace:
    """Build one deterministic compact memory-use trace."""
    sorted_refs = _sorted_refs(refs)
    base_codes = (
        "memory_use_trace:v1",
        f"task:{_normalized_text(task) or 'reply'}",
        "memory_use_trace_empty" if not sorted_refs else "memory_use_trace_selected",
        f"memory_use_ref_count:{len(sorted_refs)}",
    )
    return BrainMemoryUseTrace(
        schema_version=_SCHEMA_VERSION,
        user_id=_normalized_text(user_id),
        agent_id=_normalized_text(agent_id),
        thread_id=_normalized_text(thread_id),
        created_at=_normalized_text(created_at),
        task=_normalized_text(task) or "reply",
        selected_section_names=tuple(sorted(_dedupe_preserve_order(selected_section_names))),
        refs=sorted_refs,
        summary=_trace_summary(sorted_refs),
        reason_codes=_dedupe_preserve_order((*base_codes, *reason_codes)),
    )


def stamp_memory_use_trace(trace: BrainMemoryUseTrace, *, created_at: str) -> BrainMemoryUseTrace:
    """Return one trace with a persisted timestamp while preserving stable refs."""
    return build_memory_use_trace(
        user_id=trace.user_id,
        agent_id=trace.agent_id,
        thread_id=trace.thread_id,
        task=trace.task,
        selected_section_names=trace.selected_section_names,
        refs=trace.refs,
        created_at=created_at,
        reason_codes=trace.reason_codes,
    )


__all__ = [
    "BrainMemoryUseTrace",
    "BrainMemoryUseTraceRef",
    "build_memory_use_trace",
    "display_kind_for_claim_predicate",
    "render_safe_memory_provenance_label",
    "stamp_memory_use_trace",
]

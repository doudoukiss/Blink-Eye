"""Narrative memory records and commitment extraction helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BrainTaskCandidate:
    """One explicit task or reminder candidate."""

    title: str
    details: dict[str, str]
    status: str = "open"


@dataclass(frozen=True)
class BrainNarrativeMemoryRecord:
    """One canonical narrative memory row."""

    id: int
    user_id: str
    thread_id: str
    kind: str
    title: str
    summary: str
    details_json: str
    status: str
    confidence: float
    source_event_id: str | None
    provenance_json: str
    contradiction_key: str | None
    supersedes_memory_id: int | None
    observed_at: str
    updated_at: str
    stale_after_seconds: int | None

    @property
    def details(self) -> dict[str, Any]:
        """Return the decoded narrative payload."""
        return json.loads(self.details_json)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return the decoded provenance metadata."""
        return json.loads(self.provenance_json)

    @property
    def is_stale(self) -> bool:
        """Return whether the memory is older than its freshness window."""
        if self.stale_after_seconds in (None, 0):
            return False
        observed = datetime.fromisoformat(self.observed_at)
        return datetime.now(UTC) > observed + timedelta(seconds=int(self.stale_after_seconds))


def narrative_default_staleness(kind: str) -> int | None:
    """Return the default freshness horizon for one narrative memory kind."""
    if kind == "commitment":
        return 180 * 24 * 60 * 60
    if kind == "session_summary":
        return 7 * 24 * 60 * 60
    if kind == "daily_summary":
        return 30 * 24 * 60 * 60
    return 90 * 24 * 60 * 60


def extract_task_candidates(text: str) -> list[BrainTaskCandidate]:
    """Extract explicit reminder or todo statements into typed task candidates."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []

    patterns = (
        r"提醒我([^，。！？,.!?]{1,48})",
        r"待办[:：]\s*([^，。！？]{1,48})",
        r"\bremind me to ([^,.!?]{1,60})",
        r"\bremember to ([^,.!?]{1,60})",
        r"\btodo[: ]+([^,.!?]{1,60})",
    )

    candidates: list[BrainTaskCandidate] = []
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        title = re.sub(r"\s+", " ", match.group(1).strip(" ，。！？,.!?"))
        if not title:
            continue
        candidates.append(
            BrainTaskCandidate(
                title=title,
                details={"source_text": normalized},
            )
        )
        break
    return candidates


def build_thread_summary_text(
    *,
    recent_user_turns: list[str],
    recent_assistant_turns: list[str],
    open_commitments: list[str],
) -> str:
    """Build a concise narrative summary from recent turns and commitments."""
    summary_parts: list[str] = []
    for user_text in recent_user_turns[-3:]:
        if user_text:
            summary_parts.append(f"U: {user_text}")
    for assistant_text in recent_assistant_turns[-3:]:
        if assistant_text:
            summary_parts.append(f"A: {assistant_text}")
    if open_commitments:
        summary_parts.append(f"Tasks: {', '.join(open_commitments[:3])}")
    return " | ".join(summary_parts)

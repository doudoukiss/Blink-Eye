"""Episodic memory records and event-to-episode extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from blink.brain.events import BrainEventRecord, BrainEventType


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BrainEpisodicMemoryCandidate:
    """One episodic memory candidate derived from an event."""

    kind: str
    summary: str
    payload: dict[str, Any]
    confidence: float
    stale_after_seconds: int | None


@dataclass(frozen=True)
class BrainEpisodicMemoryRecord:
    """One canonical episodic memory row."""

    id: int
    agent_id: str
    user_id: str
    session_id: str
    thread_id: str
    kind: str
    summary: str
    payload_json: str
    confidence: float
    source_event_id: str | None
    provenance_json: str
    observed_at: str
    updated_at: str
    stale_after_seconds: int | None
    status: str

    @property
    def payload(self) -> dict[str, Any]:
        """Return the decoded episodic payload."""
        return json.loads(self.payload_json)

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


def _tool_result_summary(result: dict[str, Any]) -> str:
    for key in ("description", "answer", "summary", "content", "text"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def build_episodic_candidates_from_event(event: BrainEventRecord) -> list[BrainEpisodicMemoryCandidate]:
    """Project one brain event into zero or more episodic memory candidates."""
    payload = event.payload
    if event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
        text = str(payload.get("text", "")).strip()
        if text:
            return [
                BrainEpisodicMemoryCandidate(
                    kind="user_turn",
                    summary=text,
                    payload={"text": text},
                    confidence=event.confidence,
                    stale_after_seconds=7 * 24 * 60 * 60,
                )
            ]
        return []

    if event.event_type == BrainEventType.ASSISTANT_TURN_ENDED:
        text = str(payload.get("text", "")).strip()
        if text:
            return [
                BrainEpisodicMemoryCandidate(
                    kind="assistant_turn",
                    summary=text,
                    payload={"text": text},
                    confidence=event.confidence,
                    stale_after_seconds=7 * 24 * 60 * 60,
                )
            ]
        return []

    if event.event_type == BrainEventType.TOOL_COMPLETED:
        function_name = str(payload.get("function_name", "")).strip()
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        summary = _tool_result_summary(result)
        if not summary:
            return []
        kind = "vision_observation" if function_name == "fetch_user_image" else "tool_result"
        return [
            BrainEpisodicMemoryCandidate(
                kind=kind,
                summary=summary,
                payload={
                    "function_name": function_name,
                    "result": result,
                },
                confidence=event.confidence,
                stale_after_seconds=6 * 60 * 60 if kind == "vision_observation" else 3 * 24 * 60 * 60,
            )
        ]

    return []

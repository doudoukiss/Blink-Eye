"""Working-memory snapshots composed from projections and recent durable memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from blink.brain.projections import (
    BrainAgendaProjection,
    BrainHeartbeatProjection,
    BrainWorkingContextProjection,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BrainWorkingMemorySnapshot:
    """Composite working-memory view for the current thread."""

    thread_id: str
    context: BrainWorkingContextProjection
    agenda: BrainAgendaProjection
    heartbeat: BrainHeartbeatProjection
    recent_episodic: list[dict[str, Any]] = field(default_factory=list)
    open_commitments: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the working-memory snapshot."""
        return {
            "thread_id": self.thread_id,
            "context": self.context.as_dict(),
            "agenda": self.agenda.as_dict(),
            "heartbeat": self.heartbeat.as_dict(),
            "recent_episodic": list(self.recent_episodic),
            "open_commitments": list(self.open_commitments),
            "generated_at": self.generated_at,
        }


def build_working_memory_snapshot(*, store, user_id: str, thread_id: str) -> BrainWorkingMemorySnapshot:
    """Compose the current working-memory snapshot from projections and recent memory."""
    episodic = [
        {
            "kind": record.kind,
            "summary": record.summary,
            "observed_at": record.observed_at,
        }
        for record in store.episodic_memories(user_id=user_id, thread_id=thread_id, limit=4)
    ]
    commitments = [task["title"] for task in store.active_tasks(user_id=user_id, limit=6)]
    return BrainWorkingMemorySnapshot(
        thread_id=thread_id,
        context=store.get_working_context_projection(scope_key=thread_id),
        agenda=store.get_agenda_projection(scope_key=thread_id, user_id=user_id),
        heartbeat=store.get_heartbeat_projection(scope_key=thread_id),
        recent_episodic=episodic,
        open_commitments=commitments,
    )

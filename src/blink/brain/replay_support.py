"""Shared helpers for replay-safe event materialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.identity import load_default_agent_blocks
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore

_EXECUTIVE_REPLAY_EVENT_TYPES = {
    BrainEventType.GOAL_CREATED,
    BrainEventType.GOAL_UPDATED,
    BrainEventType.GOAL_DEFERRED,
    BrainEventType.GOAL_RESUMED,
    BrainEventType.GOAL_CANCELLED,
    BrainEventType.GOAL_REPAIRED,
    BrainEventType.GOAL_COMPLETED,
    BrainEventType.GOAL_FAILED,
}


def append_replay_event_payloads(
    *,
    store: BrainStore,
    session_ids: BrainSessionIds,
    payloads: Iterable[dict[str, Any]],
) -> tuple[BrainEventRecord, ...]:
    """Append replay/eval event payloads to the store in order."""
    appended: list[BrainEventRecord] = []
    for payload in payloads:
        event_type = str(payload.get("event_type", "")).strip()
        if not event_type:
            raise ValueError("Replay/eval event payload must include event_type.")

        event_id = str(payload.get("event_id", "")).strip()
        ts = str(payload.get("ts", "")).strip()
        if event_id and ts:
            event = BrainEventRecord(
                id=0,
                event_id=event_id,
                event_type=event_type,
                ts=ts,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=str(payload.get("source", "eval")),
                correlation_id=payload.get("correlation_id"),
                causal_parent_id=payload.get("causal_parent_id"),
                confidence=float(payload.get("confidence", 1.0)),
                payload_json=json.dumps(payload.get("payload", {}), ensure_ascii=False, sort_keys=True),
                tags_json=json.dumps(payload.get("tags", []), ensure_ascii=False, sort_keys=True),
            )
            store.import_brain_event(event)
            appended.append(event)
            continue

        appended.append(
            store.append_brain_event(
                event_type=event_type,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=str(payload.get("source", "eval")),
                correlation_id=payload.get("correlation_id"),
                causal_parent_id=payload.get("causal_parent_id"),
                payload=dict(payload.get("payload", {})),
            )
        )
    return tuple(appended)


def materialize_replayed_events(
    *,
    store: BrainStore,
    session_ids: BrainSessionIds,
    events: Iterable[BrainEventRecord],
    agent_blocks: dict[str, str] | None = None,
) -> None:
    """Apply derived memory, executive, and procedural state for replayed events."""
    events = tuple(events)
    store.ensure_default_blocks(agent_blocks or store.get_agent_blocks() or load_default_agent_blocks())
    for event in events:
        if (
            event.event_type.startswith("memory.")
            or event.event_type.startswith("autobiography.entry.")
            or event.event_type.startswith("reflection.cycle.")
        ):
            store.apply_memory_event(event)
        if event.event_type in _EXECUTIVE_REPLAY_EVENT_TYPES:
            store.apply_executive_event(event)
    store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    latest_event = max(
        events,
        key=lambda item: (int(getattr(item, "id", 0)), item.ts, item.event_id),
        default=None,
    )
    store.refresh_private_working_memory_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        source_event_id=(latest_event.event_id if latest_event is not None else None),
        updated_at=(latest_event.ts if latest_event is not None else None),
        agent_id=session_ids.agent_id,
        commit=True,
    )
    store.refresh_active_situation_model_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        source_event_id=(latest_event.event_id if latest_event is not None else None),
        updated_at=(latest_event.ts if latest_event is not None else None),
        agent_id=session_ids.agent_id,
        commit=True,
    )


__all__ = [
    "append_replay_event_payloads",
    "materialize_replayed_events",
]

"""Stable session and user identity helpers for the Blink brain kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrainSessionIds:
    """Resolved runtime identifiers for one conversation thread."""

    agent_id: str
    user_id: str
    session_id: str
    thread_id: str


def resolve_brain_session_ids(
    *,
    runtime_kind: str,
    client_id: str | None = None,
    agent_id: str = "blink/main",
    default_user_id: str = "local_primary",
) -> BrainSessionIds:
    """Resolve stable Blink brain IDs for the active runtime."""
    user_id = str(client_id or default_user_id).strip() or default_user_id
    session_prefix = {
        "browser": "browser",
        "text": "text",
    }.get(runtime_kind, "voice")
    session_id = f"{session_prefix}:{user_id}"
    thread_id = f"{session_prefix}:{user_id}"
    return BrainSessionIds(
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        thread_id=thread_id,
    )

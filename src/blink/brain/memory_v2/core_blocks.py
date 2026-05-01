"""Versioned core memory blocks for Blink continuity state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventType


def _stable_id(prefix: str, *parts: object) -> str:
    """Return a deterministic text id for continuity rows."""
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class BrainCoreMemoryBlockKind(str, Enum):
    """Well-known continuity block kinds."""

    SELF_CORE = "self_core"
    SELF_PERSONA_CORE = "self_persona_core"
    SELF_CURRENT_ARC = "self_current_arc"
    USER_CORE = "user_core"
    RELATIONSHIP_CORE = "relationship_core"
    ACTIVE_COMMITMENTS = "active_commitments"
    RELATIONSHIP_STYLE = "relationship_style"
    TEACHING_PROFILE = "teaching_profile"
    BEHAVIOR_CONTROL_PROFILE = "behavior_control_profile"


@dataclass(frozen=True)
class BrainCoreMemoryBlockRecord:
    """One versioned core-memory block row."""

    block_id: str
    block_kind: str
    scope_type: str
    scope_id: str
    version: int
    content_json: str
    status: str
    source_event_id: str | None
    supersedes_block_id: str | None
    created_at: str
    updated_at: str

    @property
    def content(self) -> dict[str, Any]:
        """Return the decoded block content."""
        return json.loads(self.content_json)


class CoreMemoryBlockService:
    """Versioned core-memory block persistence on top of BrainStore."""

    def __init__(self, *, store):
        """Bind the service to one canonical store."""
        self._store = store

    def get_current_block(
        self,
        block_kind: str,
        scope_type: str,
        scope_id: str,
    ) -> BrainCoreMemoryBlockRecord | None:
        """Return the current block for one `(kind, scope_type, scope_id)` tuple."""
        row = self._store._conn.execute(
            """
            SELECT * FROM core_memory_blocks
            WHERE block_kind = ? AND scope_type = ? AND scope_id = ? AND status = 'current'
            ORDER BY version DESC
            LIMIT 1
            """,
            (block_kind, scope_type, scope_id),
        ).fetchone()
        return self._row_to_block(row)

    def list_current_blocks(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        block_kinds: tuple[str, ...] | None = None,
        limit: int = 16,
    ) -> list[BrainCoreMemoryBlockRecord]:
        """Return current blocks for one optional scope filter."""
        clauses = ["status = 'current'"]
        params: list[Any] = []
        if scope_type is not None:
            clauses.append("scope_type = ?")
            params.append(scope_type)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if block_kinds:
            clauses.append(f"block_kind IN ({','.join('?' for _ in block_kinds)})")
            params.extend(block_kinds)
        rows = self._store._conn.execute(
            f"""
            SELECT * FROM core_memory_blocks
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, version DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def list_block_versions(
        self,
        block_kind: str,
        scope_type: str,
        scope_id: str,
    ) -> list[BrainCoreMemoryBlockRecord]:
        """Return all persisted versions for one block identity."""
        rows = self._store._conn.execute(
            """
            SELECT * FROM core_memory_blocks
            WHERE block_kind = ? AND scope_type = ? AND scope_id = ?
            ORDER BY version DESC
            """,
            (block_kind, scope_type, scope_id),
        ).fetchall()
        return [self._row_to_block(row) for row in rows]

    def upsert_block(
        self,
        block_kind: str,
        scope_type: str,
        scope_id: str,
        content: dict[str, Any],
        source_event_id: str | None,
        expected_version: int | None = None,
        *,
        event_context: dict[str, Any] | None = None,
    ) -> BrainCoreMemoryBlockRecord:
        """Append a new block version and supersede the prior current version."""
        now = self._store._utc_now_for_memory_v2()
        normalized_content_json = _json(content)
        current = self.get_current_block(block_kind, scope_type, scope_id)
        if current is not None:
            if expected_version is not None and current.version != expected_version:
                raise ValueError(
                    f"Core block version mismatch for {block_kind}:{scope_type}:{scope_id}."
                )
            if current.content_json == normalized_content_json:
                return current

        next_version = 1 if current is None else current.version + 1
        block_id = _stable_id("block", block_kind, scope_type, scope_id, next_version)
        if current is not None:
            self._store._conn.execute(
                """
                UPDATE core_memory_blocks
                SET status = 'superseded', updated_at = ?
                WHERE block_id = ?
                """,
                (now, current.block_id),
            )
        try:
            self._store._conn.execute(
                """
                INSERT INTO core_memory_blocks (
                    block_id, block_kind, scope_type, scope_id, version, content_json, status,
                    source_event_id, supersedes_block_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'current', ?, ?, ?, ?)
                """,
                (
                    block_id,
                    block_kind,
                    scope_type,
                    scope_id,
                    next_version,
                    normalized_content_json,
                    source_event_id,
                    current.block_id if current is not None else None,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            self._store._conn.rollback()
            raced_current = self.get_current_block(block_kind, scope_type, scope_id)
            if raced_current is not None and raced_current.content_json == normalized_content_json:
                return raced_current
            raise RuntimeError(
                f"Concurrent core block update detected for {block_kind}:{scope_type}:{scope_id}."
            ) from exc
        self._store._conn.commit()
        record = self.get_current_block(block_kind, scope_type, scope_id)
        if record is None:
            raise RuntimeError("Failed to persist core memory block.")

        if event_context is not None:
            self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_BLOCK_UPSERTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "block_id": record.block_id,
                    "block_kind": record.block_kind,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "version": record.version,
                    "status": record.status,
                    "source_event_id": record.source_event_id,
                    "supersedes_block_id": record.supersedes_block_id,
                    "content": record.content,
                },
            )
        return record

    def _row_to_block(self, row) -> BrainCoreMemoryBlockRecord | None:
        if row is None:
            return None
        return BrainCoreMemoryBlockRecord(
            block_id=str(row["block_id"]),
            block_kind=str(row["block_kind"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            version=int(row["version"]),
            content_json=str(row["content_json"]),
            status=str(row["status"]),
            source_event_id=str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None,
            supersedes_block_id=(
                str(row["supersedes_block_id"]) if row["supersedes_block_id"] is not None else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

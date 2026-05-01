"""Provider-free SQLite-backed Blink brain kernel store."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from blink.brain.bounded_json import (
    MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,
    dumps_bounded_json,
)
from blink.brain.core.autonomy import (
    BrainAutonomyLedgerEntry,
    BrainAutonomyLedgerProjection,
    BrainCandidateGoal,
    BrainReevaluationTrigger,
    autonomy_decision_kind_for_event_type,
)
from blink.brain.core.events import BrainEventRecord, BrainEventType
from blink.brain.core.presence import BrainPresenceSnapshot, normalize_presence_snapshot
from blink.brain.core.projections import (
    AGENDA_PROJECTION,
    AUTONOMY_LEDGER_PROJECTION,
    BODY_STATE_PROJECTION,
    ENGAGEMENT_STATE_PROJECTION,
    HEARTBEAT_PROJECTION,
    RELATIONSHIP_STATE_PROJECTION,
    SCENE_STATE_PROJECTION,
    WORKING_CONTEXT_PROJECTION,
    BrainAgendaProjection,
    BrainCommitmentRecord,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainEngagementStateProjection,
    BrainGoal,
    BrainGoalStatus,
    BrainHeartbeatProjection,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainRelationshipStateProjection,
    BrainSceneStateProjection,
    BrainWakeCondition,
    BrainWorkingContextProjection,
)
from blink.project_identity import cache_dir

DEFAULT_BRAIN_DB_PATH = cache_dir("brain", "brain.db")
_SCHEMA_VERSION = 4


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class BrainCoreStore:
    """Provider-free SQLite state kernel for Blink."""

    SQLITE_BUSY_TIMEOUT_MS = 30_000

    def __init__(self, *, path: str | Path | None = None):
        """Initialize the store and ensure the schema exists."""
        self.path = Path(path) if path else DEFAULT_BRAIN_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=self.SQLITE_BUSY_TIMEOUT_MS / 1000,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_sqlite_connection()
        self._fts_enabled = False
        self._initialize()

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()

    def _configure_sqlite_connection(self):
        """Configure SQLite for local multi-surface runtime access."""
        self._conn.execute(f"PRAGMA busy_timeout = {self.SQLITE_BUSY_TIMEOUT_MS}")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass

    def _initialize(self):
        """Create kernel tables and schema metadata if they do not already exist."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_blocks (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS presence_snapshots (
                scope_key TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                ts TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                source TEXT NOT NULL,
                correlation_id TEXT,
                causal_parent_id TEXT,
                confidence REAL NOT NULL,
                payload_json TEXT NOT NULL,
                tags_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_core_brain_events_user_thread_id
            ON brain_events (user_id, thread_id, id DESC)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_core_brain_events_user_thread_type_id
            ON brain_events (user_id, thread_id, event_type, id DESC)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_projections (
                projection_name TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                projection_json TEXT NOT NULL,
                source_event_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (projection_name, scope_key)
            )
            """
        )
        self._initialize_extended_schema(cursor)
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('schema_version', ?)
            """,
            (str(_SCHEMA_VERSION),),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ('memory:fts_enabled', ?)
            """,
            ("1" if self._fts_enabled else "0",),
        )
        self._conn.commit()

    def _initialize_extended_schema(self, cursor: sqlite3.Cursor):
        """Allow compatibility stores to register additional tables."""

    def ensure_default_blocks(self, blocks: dict[str, str]):
        """Insert default pinned agent blocks on first use."""
        now = _utc_now()
        for name, content in blocks.items():
            self._conn.execute(
                """
                INSERT INTO agent_blocks (name, content, source, updated_at)
                VALUES (?, ?, 'default', ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, content, now),
            )
        self._conn.commit()

    def get_agent_blocks(self) -> dict[str, str]:
        """Return the current pinned agent blocks."""
        rows = self._conn.execute("SELECT name, content FROM agent_blocks ORDER BY name").fetchall()
        return {str(row["name"]): str(row["content"]) for row in rows}

    def get_metadata(self, key: str) -> str | None:
        """Return one metadata value if present."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_metadata(self, key: str, value: str):
        """Upsert one metadata value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def set_presence_snapshot(self, *, scope_key: str, snapshot: dict[str, Any]):
        """Persist one compatibility presence snapshot directly."""
        normalized = normalize_presence_snapshot(BrainPresenceSnapshot.from_dict(snapshot))
        self._persist_body_state(
            scope_key=scope_key,
            snapshot=normalized,
            source_event_id=None,
            updated_at=normalized.updated_at,
            commit=True,
        )

    def get_presence_snapshot(self, *, scope_key: str) -> dict[str, Any] | None:
        """Return one compatibility presence snapshot if present."""
        row = self._conn.execute(
            """
            SELECT snapshot_json FROM presence_snapshots
            WHERE scope_key = ?
            """,
            (scope_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["snapshot_json"]))

    def append_brain_event(
        self,
        *,
        event_type: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one typed event and update projections."""
        event_record = BrainEventRecord(
            id=0,
            event_id=str(uuid4()),
            event_type=event_type,
            ts=ts or _utc_now(),
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            payload_json=dumps_bounded_json(
                payload or {},
                max_chars=MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,
                overflow_kind="brain_event_payload_too_large",
            ),
            tags_json=json.dumps(tags or [], ensure_ascii=False),
        )
        cursor = self._conn.execute(
            """
            INSERT INTO brain_events (
                event_id, event_type, ts, agent_id, user_id, session_id,
                thread_id, source, correlation_id, causal_parent_id, confidence,
                payload_json, tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_record.event_id,
                event_record.event_type,
                event_record.ts,
                event_record.agent_id,
                event_record.user_id,
                event_record.session_id,
                event_record.thread_id,
                event_record.source,
                event_record.correlation_id,
                event_record.causal_parent_id,
                event_record.confidence,
                event_record.payload_json,
                event_record.tags_json,
            ),
        )
        stored = BrainEventRecord(
            id=int(cursor.lastrowid),
            event_id=event_record.event_id,
            event_type=event_record.event_type,
            ts=event_record.ts,
            agent_id=event_record.agent_id,
            user_id=event_record.user_id,
            session_id=event_record.session_id,
            thread_id=event_record.thread_id,
            source=event_record.source,
            correlation_id=event_record.correlation_id,
            causal_parent_id=event_record.causal_parent_id,
            confidence=event_record.confidence,
            payload_json=event_record.payload_json,
            tags_json=event_record.tags_json,
        )
        self._apply_event_to_projections(stored, commit=True)
        return stored

    def recent_brain_events(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 64,
        event_types: tuple[str, ...] | None = None,
    ) -> list[BrainEventRecord]:
        """Return recent append-only events for one user thread."""
        clauses = ["user_id = ?", "thread_id = ?"]
        params: list[Any] = [user_id, thread_id]
        if event_types:
            clauses.append(f"event_type IN ({','.join('?' for _ in event_types)})")
            params.extend(event_types)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT * FROM brain_events
            WHERE {" AND ".join(clauses)}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [self._brain_event_from_row(row) for row in rows]

    def recent_autonomy_events(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int = 64,
    ) -> list[BrainEventRecord]:
        """Return recent candidate-lifecycle and director-decision events."""
        rows = self._conn.execute(
            """
            SELECT * FROM brain_events
            WHERE user_id = ? AND thread_id = ? AND event_type IN (?, ?, ?, ?, ?, ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                user_id,
                thread_id,
                BrainEventType.GOAL_CANDIDATE_CREATED,
                BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
                BrainEventType.GOAL_CANDIDATE_MERGED,
                BrainEventType.GOAL_CANDIDATE_ACCEPTED,
                BrainEventType.GOAL_CANDIDATE_EXPIRED,
                BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
                limit,
            ),
        ).fetchall()
        return [self._brain_event_from_row(row) for row in rows]

    def latest_brain_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        event_type: str,
    ) -> BrainEventRecord | None:
        """Return the most recent matching event for one user thread."""
        row = self._conn.execute(
            """
            SELECT * FROM brain_events
            WHERE user_id = ? AND thread_id = ? AND event_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, thread_id, event_type),
        ).fetchone()
        return self._brain_event_from_row(row)

    def brain_events_since(
        self,
        *,
        user_id: str,
        thread_id: str,
        since_event_id: int,
    ) -> list[BrainEventRecord]:
        """Return all events after one stored row id for a user thread."""
        rows = self._conn.execute(
            """
            SELECT * FROM brain_events
            WHERE user_id = ? AND thread_id = ? AND id > ?
            ORDER BY id ASC
            """,
            (user_id, thread_id, since_event_id),
        ).fetchall()
        return [self._brain_event_from_row(row) for row in rows]

    def import_brain_event(self, event: BrainEventRecord):
        """Import one event into this store and update projections."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO brain_events (
                event_id, event_type, ts, agent_id, user_id, session_id,
                thread_id, source, correlation_id, causal_parent_id, confidence,
                payload_json, tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.ts,
                event.agent_id,
                event.user_id,
                event.session_id,
                event.thread_id,
                event.source,
                event.correlation_id,
                event.causal_parent_id,
                event.confidence,
                event.payload_json,
                event.tags_json,
            ),
        )
        row = self._conn.execute(
            "SELECT * FROM brain_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if row is not None:
            self._apply_event_to_projections(self._brain_event_from_row(row), commit=True)

    def rebuild_brain_projections(self):
        """Rebuild projection tables by replaying the append-only event log."""
        self._conn.execute("DELETE FROM brain_projections")
        rows = self._conn.execute("SELECT * FROM brain_events ORDER BY id ASC").fetchall()
        for row in rows:
            self._apply_event_to_projections(self._brain_event_from_row(row), commit=False)
        self._conn.commit()

    def get_body_state_projection(self, *, scope_key: str) -> BrainPresenceSnapshot:
        """Return the current body-state projection for one runtime scope."""
        return self._get_body_state_projection(scope_key=scope_key)

    def _get_body_state_projection(self, *, scope_key: str) -> BrainPresenceSnapshot:
        projection = self._get_projection_dict(
            projection_name=BODY_STATE_PROJECTION,
            scope_key=scope_key,
        )
        if projection is not None:
            return normalize_presence_snapshot(BrainPresenceSnapshot.from_dict(projection))
        return normalize_presence_snapshot(
            BrainPresenceSnapshot.from_dict(self.get_presence_snapshot(scope_key=scope_key))
        )

    def get_scene_state_projection(self, *, scope_key: str) -> BrainSceneStateProjection:
        """Return the current symbolic scene projection for one runtime scope."""
        return self._get_scene_state_projection(scope_key=scope_key)

    def _get_scene_state_projection(self, *, scope_key: str) -> BrainSceneStateProjection:
        return BrainSceneStateProjection.from_dict(
            self._get_projection_dict(
                projection_name=SCENE_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_engagement_state_projection(self, *, scope_key: str) -> BrainEngagementStateProjection:
        """Return the current symbolic engagement projection for one runtime scope."""
        return self._get_engagement_state_projection(scope_key=scope_key)

    def _get_engagement_state_projection(self, *, scope_key: str) -> BrainEngagementStateProjection:
        return BrainEngagementStateProjection.from_dict(
            self._get_projection_dict(
                projection_name=ENGAGEMENT_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_relationship_state_projection(self, *, scope_key: str) -> BrainRelationshipStateProjection:
        """Return the current projected relationship continuity surface."""
        return self._get_relationship_state_projection(scope_key=scope_key)

    def _get_relationship_state_projection(
        self,
        *,
        scope_key: str,
    ) -> BrainRelationshipStateProjection:
        return BrainRelationshipStateProjection.from_dict(
            self._get_projection_dict(
                projection_name=RELATIONSHIP_STATE_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_working_context_projection(self, *, scope_key: str) -> BrainWorkingContextProjection:
        """Return the current working-context projection for one thread."""
        return self._get_working_context_projection(scope_key=scope_key)

    def _get_working_context_projection(self, *, scope_key: str) -> BrainWorkingContextProjection:
        return BrainWorkingContextProjection.from_dict(
            self._get_projection_dict(
                projection_name=WORKING_CONTEXT_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_agenda_projection(self, *, scope_key: str) -> BrainAgendaProjection:
        """Return the current agenda projection."""
        return self._get_agenda_projection(scope_key=scope_key)

    def _get_agenda_projection(self, *, scope_key: str) -> BrainAgendaProjection:
        return BrainAgendaProjection.from_dict(
            self._get_projection_dict(
                projection_name=AGENDA_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_heartbeat_projection(self, *, scope_key: str) -> BrainHeartbeatProjection:
        """Return the current heartbeat projection for one thread."""
        return self._get_heartbeat_projection(scope_key=scope_key)

    def _get_heartbeat_projection(self, *, scope_key: str) -> BrainHeartbeatProjection:
        return BrainHeartbeatProjection.from_dict(
            self._get_projection_dict(
                projection_name=HEARTBEAT_PROJECTION,
                scope_key=scope_key,
            )
        )

    def get_autonomy_ledger_projection(self, *, scope_key: str) -> BrainAutonomyLedgerProjection:
        """Return the current autonomy-ledger projection for one thread."""
        return BrainAutonomyLedgerProjection.from_dict(
            self._get_projection_dict(
                projection_name=AUTONOMY_LEDGER_PROJECTION,
                scope_key=scope_key,
            )
        )

    def append_candidate_goal_created(
        self,
        *,
        candidate_goal: BrainCandidateGoal,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one candidate-goal creation event."""
        return self.append_brain_event(
            event_type=BrainEventType.GOAL_CANDIDATE_CREATED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={"candidate_goal": candidate_goal.as_dict()},
            correlation_id=correlation_id or candidate_goal.candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_candidate_goal_suppressed(
        self,
        *,
        candidate_goal_id: str,
        reason: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        reason_details: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        executive_policy: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one candidate-goal suppression event."""
        return self.append_brain_event(
            event_type=BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "candidate_goal_id": candidate_goal_id,
                "reason": reason,
                "reason_details": dict(reason_details or {}),
                "reason_codes": list(reason_codes or []),
                "executive_policy": (
                    dict(executive_policy) if executive_policy is not None else None
                ),
            },
            correlation_id=correlation_id or candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_candidate_goal_merged(
        self,
        *,
        candidate_goal_id: str,
        merged_into_candidate_goal_id: str,
        reason: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        reason_details: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        executive_policy: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one candidate-goal merge event."""
        return self.append_brain_event(
            event_type=BrainEventType.GOAL_CANDIDATE_MERGED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "candidate_goal_id": candidate_goal_id,
                "merged_into_candidate_goal_id": merged_into_candidate_goal_id,
                "reason": reason,
                "reason_details": dict(reason_details or {}),
                "reason_codes": list(reason_codes or []),
                "executive_policy": (
                    dict(executive_policy) if executive_policy is not None else None
                ),
            },
            correlation_id=correlation_id or merged_into_candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_candidate_goal_accepted(
        self,
        *,
        candidate_goal_id: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        goal_id: str | None = None,
        commitment_id: str | None = None,
        reason: str | None = None,
        reason_details: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        executive_policy: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one candidate-goal acceptance event."""
        return self.append_brain_event(
            event_type=BrainEventType.GOAL_CANDIDATE_ACCEPTED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "candidate_goal_id": candidate_goal_id,
                "goal_id": goal_id,
                "commitment_id": commitment_id,
                "reason": reason,
                "reason_details": dict(reason_details or {}),
                "reason_codes": list(reason_codes or []),
                "executive_policy": (
                    dict(executive_policy) if executive_policy is not None else None
                ),
            },
            correlation_id=correlation_id or candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_candidate_goal_expired(
        self,
        *,
        candidate_goal_id: str,
        reason: str,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        reason_details: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        executive_policy: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one candidate-goal expiry event."""
        return self.append_brain_event(
            event_type=BrainEventType.GOAL_CANDIDATE_EXPIRED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "candidate_goal_id": candidate_goal_id,
                "reason": reason,
                "reason_details": dict(reason_details or {}),
                "reason_codes": list(reason_codes or []),
                "executive_policy": (
                    dict(executive_policy) if executive_policy is not None else None
                ),
            },
            correlation_id=correlation_id or candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_director_non_action(
        self,
        *,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        candidate_goal_id: str | None = None,
        reason: str,
        reason_details: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        executive_policy: dict[str, Any] | None = None,
        expected_reevaluation_condition: str | None = None,
        expected_reevaluation_condition_kind: str | None = None,
        expected_reevaluation_condition_details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one explicit director non-action event."""
        return self.append_brain_event(
            event_type=BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "candidate_goal_id": candidate_goal_id,
                "reason": reason,
                "reason_details": dict(reason_details or {}),
                "reason_codes": list(reason_codes or []),
                "executive_policy": (
                    dict(executive_policy) if executive_policy is not None else None
                ),
                "expected_reevaluation_condition": expected_reevaluation_condition,
                "expected_reevaluation_condition_kind": expected_reevaluation_condition_kind,
                "expected_reevaluation_condition_details": dict(
                    expected_reevaluation_condition_details or {}
                ),
            },
            correlation_id=correlation_id or candidate_goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts,
        )

    def append_director_reevaluation_triggered(
        self,
        *,
        trigger: BrainReevaluationTrigger,
        candidate_goal_ids: list[str],
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one explicit reevaluation-trigger event."""
        return self.append_brain_event(
            event_type=BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "trigger": trigger.as_dict(),
                "candidate_goal_ids": list(candidate_goal_ids),
            },
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts or trigger.ts,
        )

    def append_commitment_wake_triggered(
        self,
        *,
        commitment: BrainCommitmentRecord,
        wake_condition: BrainWakeCondition,
        trigger: BrainCommitmentWakeTrigger,
        routing: BrainCommitmentWakeRoutingDecision,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one explicit commitment-wake trigger event."""
        return self.append_brain_event(
            event_type=BrainEventType.COMMITMENT_WAKE_TRIGGERED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "commitment": commitment.as_dict(),
                "wake_condition": wake_condition.as_dict(),
                "trigger": trigger.as_dict(),
                "routing": routing.as_dict(),
            },
            correlation_id=correlation_id or commitment.commitment_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts or trigger.ts,
        )

    def append_planning_proposed(
        self,
        *,
        proposal: BrainPlanProposal,
        decision: BrainPlanProposalDecision | None = None,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one planning proposal event."""
        return self.append_brain_event(
            event_type=BrainEventType.PLANNING_PROPOSED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "goal_id": proposal.goal_id,
                "commitment_id": proposal.commitment_id,
                "proposal": proposal.as_dict(),
                "decision": decision.as_dict() if decision is not None else None,
            },
            correlation_id=correlation_id or proposal.goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts or proposal.created_at,
        )

    def append_planning_adopted(
        self,
        *,
        proposal: BrainPlanProposal,
        decision: BrainPlanProposalDecision,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one planning adoption event."""
        return self.append_brain_event(
            event_type=BrainEventType.PLANNING_ADOPTED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "goal_id": proposal.goal_id,
                "commitment_id": proposal.commitment_id,
                "proposal": proposal.as_dict(),
                "decision": decision.as_dict(),
            },
            correlation_id=correlation_id or proposal.goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts or proposal.created_at,
        )

    def append_planning_rejected(
        self,
        *,
        proposal: BrainPlanProposal,
        decision: BrainPlanProposalDecision,
        agent_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        source: str,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        ts: str | None = None,
    ) -> BrainEventRecord:
        """Append one planning rejection event."""
        return self.append_brain_event(
            event_type=BrainEventType.PLANNING_REJECTED,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            source=source,
            payload={
                "goal_id": proposal.goal_id,
                "commitment_id": proposal.commitment_id,
                "proposal": proposal.as_dict(),
                "decision": decision.as_dict(),
            },
            correlation_id=correlation_id or proposal.goal_id,
            causal_parent_id=causal_parent_id,
            confidence=confidence,
            tags=tags,
            ts=ts or proposal.created_at,
        )

    def _brain_event_from_row(self, row: sqlite3.Row | None) -> BrainEventRecord | None:
        """Hydrate one stored event row."""
        if row is None:
            return None
        return BrainEventRecord(
            id=int(row["id"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            ts=str(row["ts"]),
            agent_id=str(row["agent_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            thread_id=str(row["thread_id"]),
            source=str(row["source"]),
            correlation_id=row["correlation_id"],
            causal_parent_id=row["causal_parent_id"],
            confidence=float(row["confidence"]),
            payload_json=str(row["payload_json"]),
            tags_json=str(row["tags_json"]),
        )

    def _get_projection_dict(self, *, projection_name: str, scope_key: str) -> dict[str, Any] | None:
        """Return one raw projection JSON blob."""
        row = self._conn.execute(
            """
            SELECT projection_json FROM brain_projections
            WHERE projection_name = ? AND scope_key = ?
            """,
            (projection_name, scope_key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["projection_json"]))

    def _upsert_projection(
        self,
        *,
        projection_name: str,
        scope_key: str,
        projection: dict[str, Any],
        source_event_id: str | None,
        updated_at: str,
        commit: bool,
    ):
        """Persist one raw projection JSON blob."""
        self._conn.execute(
            """
            INSERT INTO brain_projections (
                projection_name, scope_key, projection_json, source_event_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(projection_name, scope_key) DO UPDATE SET
                projection_json = excluded.projection_json,
                source_event_id = excluded.source_event_id,
                updated_at = excluded.updated_at
            """,
            (
                projection_name,
                scope_key,
                json.dumps(projection, ensure_ascii=False, sort_keys=True),
                source_event_id,
                updated_at,
            ),
        )
        if commit:
            self._conn.commit()

    def _persist_body_state(
        self,
        *,
        scope_key: str,
        snapshot: BrainPresenceSnapshot,
        source_event_id: str | None,
        updated_at: str,
        commit: bool,
    ):
        """Persist the current body-state projection and compatibility snapshot."""
        snapshot = normalize_presence_snapshot(snapshot)
        self._conn.execute(
            """
            INSERT INTO presence_snapshots (scope_key, snapshot_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at = excluded.updated_at
            """,
            (
                scope_key,
                json.dumps(snapshot.as_dict(), ensure_ascii=False, sort_keys=True),
                updated_at,
            ),
        )
        self._upsert_projection(
            projection_name=BODY_STATE_PROJECTION,
            scope_key=scope_key,
            projection=snapshot.as_dict(),
            source_event_id=source_event_id,
            updated_at=updated_at,
            commit=False,
        )
        if commit:
            self._conn.commit()

    def _apply_event_to_projections(self, event: BrainEventRecord, *, commit: bool):
        """Update projection tables from one append-only event."""
        payload = event.payload
        updated_at = event.ts
        self._apply_body_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_scene_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_engagement_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_relationship_state_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_working_context_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_agenda_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_autonomy_event(event=event, payload=payload, updated_at=updated_at)
        self._apply_heartbeat_event(event=event, payload=payload, updated_at=updated_at)
        if commit:
            self._conn.commit()

    def _apply_body_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            scope_key = str(payload.get("scope_key", "local:presence"))
            self._persist_body_state(
                scope_key=scope_key,
                snapshot=BrainPresenceSnapshot.from_dict(payload.get("snapshot")),
                source_event_id=event.event_id,
                updated_at=updated_at,
                commit=False,
            )
            return

        if event.event_type not in {
            BrainEventType.ROBOT_ACTION_OUTCOME,
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            return

        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        snapshot = self._get_body_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.ROBOT_ACTION_OUTCOME:
            status = payload.get("status", {})
            snapshot.robot_head_mode = str(status.get("mode", snapshot.robot_head_mode))
            snapshot.robot_head_armed = bool(status.get("armed", snapshot.robot_head_armed))
            snapshot.robot_head_available = bool(status.get("available", snapshot.robot_head_available))
            snapshot.robot_head_last_action = payload.get("action_id")
            if payload.get("accepted", False):
                snapshot.robot_head_last_accepted_action = payload.get("action_id")
            else:
                snapshot.robot_head_last_rejected_action = payload.get("action_id")
            if payload.get("action_id") in {"cmd_return_neutral", "auto_safe_idle"}:
                snapshot.robot_head_last_safe_state = "neutral"
            snapshot.warnings = list(status.get("warnings", snapshot.warnings))
            snapshot.details = dict(status.get("details", snapshot.details))
        else:
            snapshot.vision_enabled = True
            snapshot.vision_connected = bool(payload.get("camera_connected", snapshot.vision_connected))
            snapshot.camera_disconnected = bool(
                snapshot.vision_enabled and not payload.get("camera_connected", snapshot.vision_connected)
            )
            snapshot.perception_disabled = False
            snapshot.perception_unreliable = not bool(payload.get("camera_fresh", True))
            snapshot.vision_unavailable = False
            if payload.get("person_present") == "present":
                snapshot.attention_target = "user"
            elif payload.get("person_present") == "absent":
                snapshot.attention_target = None
            if payload.get("engagement_state") in {"engaged", "listening", "speaking"}:
                snapshot.engagement_pose = "attentive"
            elif payload.get("engagement_state") in {"away", "idle"}:
                snapshot.engagement_pose = "neutral"
            details = dict(snapshot.details)
            if payload.get("summary"):
                details["last_visual_summary"] = payload.get("summary")
            details["last_perception_event_type"] = event.event_type
            snapshot.details = details
        snapshot.updated_at = updated_at
        self._persist_body_state(
            scope_key=scope_key,
            snapshot=snapshot,
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_scene_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self._get_scene_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.camera_connected = snapshot.vision_connected
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.SCENE_CHANGED,
            BrainEventType.ENGAGEMENT_CHANGED,
        }:
            projection.camera_connected = bool(payload.get("camera_connected", projection.camera_connected))
            projection.person_present = str(payload.get("person_present", projection.person_present))
            projection.scene_change_state = str(
                payload.get("scene_change", projection.scene_change_state)
            )
            projection.last_visual_summary = payload.get("summary") or projection.last_visual_summary
            projection.last_observed_at = payload.get("observed_at") or projection.last_observed_at
            confidence = payload.get("confidence")
            projection.confidence = float(confidence) if confidence is not None else projection.confidence
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=SCENE_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_engagement_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self._get_engagement_state_projection(scope_key=scope_key)
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.user_present = bool(snapshot.vision_connected and not snapshot.camera_disconnected)
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            projection.engagement_state = str(
                payload.get("engagement_state", projection.engagement_state)
            )
            projection.attention_to_camera = str(
                payload.get("attention_to_camera", projection.attention_to_camera)
            )
            projection.user_present = payload.get("person_present") == "present"
            if projection.user_present and projection.engagement_state in {
                "engaged",
                "listening",
                "speaking",
            }:
                projection.last_engaged_at = payload.get("observed_at") or updated_at
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=ENGAGEMENT_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_relationship_state_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        scope_key = str(
            payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
        )
        projection = self._get_relationship_state_projection(scope_key=scope_key)
        if event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            user_present = payload.get("person_present") == "present"
            projection.user_present = user_present
            if user_present:
                projection.last_seen_at = payload.get("observed_at") or updated_at
            projection.engagement_state = str(
                payload.get("engagement_state", projection.engagement_state)
            )
            projection.attention_to_camera = str(
                payload.get("attention_to_camera", projection.attention_to_camera)
            )
            projection.source = event.source
            projection.updated_at = updated_at
        elif event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.user_present = bool(snapshot.vision_connected and not snapshot.camera_disconnected)
            projection.source = event.source
            projection.updated_at = updated_at
        else:
            return
        self._upsert_projection(
            projection_name=RELATIONSHIP_STATE_PROJECTION,
            scope_key=scope_key,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_working_context_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self._get_working_context_projection(scope_key=event.thread_id)
        if event.event_type == BrainEventType.USER_TURN_STARTED:
            projection.user_turn_open = True
        elif event.event_type == BrainEventType.USER_TURN_ENDED:
            projection.user_turn_open = False
        elif event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
            projection.last_user_text = str(payload.get("text", "")).strip() or None
        elif event.event_type == BrainEventType.ASSISTANT_TURN_STARTED:
            projection.assistant_turn_open = True
        elif event.event_type == BrainEventType.ASSISTANT_TURN_ENDED:
            projection.assistant_turn_open = False
            projection.last_assistant_text = str(payload.get("text", "")).strip() or None
        elif event.event_type == BrainEventType.TOOL_CALLED:
            projection.last_tool_name = payload.get("function_name")
        elif event.event_type == BrainEventType.TOOL_COMPLETED:
            projection.last_tool_name = payload.get("function_name") or projection.last_tool_name
            projection.last_tool_result = payload.get("result")
        else:
            return
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=WORKING_CONTEXT_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_agenda_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self._get_agenda_projection(scope_key=event.thread_id)
        if event.event_type == BrainEventType.USER_TURN_TRANSCRIBED:
            projection.agenda_seed = str(payload.get("text", "")).strip() or projection.agenda_seed
        elif event.event_type == BrainEventType.GOAL_CREATED:
            if isinstance(payload.get("goal"), dict):
                goal = BrainGoal.from_dict(payload.get("goal"))
                if goal.goal_id and projection.goal(goal.goal_id) is None:
                    projection.goals.append(goal)
            else:
                title = str(payload.get("title", "")).strip()
                if title and title not in projection.open_goals:
                    projection.open_goals.append(title)
                if title in projection.completed_goals:
                    projection.completed_goals = [
                        goal_title for goal_title in projection.completed_goals if goal_title != title
                    ]
        elif event.event_type == BrainEventType.PLANNING_REQUESTED:
            goal_id = str(payload.get("goal_id", "")).strip()
            goal = projection.goal(goal_id)
            if goal is None:
                return
            goal.status = BrainGoalStatus.PLANNING.value
            goal.planning_requested = True
            goal.updated_at = updated_at
        elif event.event_type in {
            BrainEventType.GOAL_UPDATED,
            BrainEventType.GOAL_DEFERRED,
            BrainEventType.GOAL_RESUMED,
            BrainEventType.GOAL_CANCELLED,
            BrainEventType.GOAL_REPAIRED,
            BrainEventType.GOAL_FAILED,
            BrainEventType.GOAL_COMPLETED,
        } and isinstance(payload.get("goal"), dict):
            goal = BrainGoal.from_dict(payload.get("goal"))
            existing = projection.goal(goal.goal_id)
            if existing is None:
                projection.goals.append(goal)
            else:
                projection.goals = [
                    goal if item.goal_id == goal.goal_id else item for item in projection.goals
                ]
        elif event.event_type == BrainEventType.GOAL_COMPLETED:
            title = str(payload.get("title", "")).strip()
            projection.open_goals = [goal for goal in projection.open_goals if goal != title]
            if title and title not in projection.completed_goals:
                projection.completed_goals.append(title)
        elif event.event_type == BrainEventType.GOAL_CANCELLED:
            title = str(payload.get("title", "")).strip()
            projection.open_goals = [goal for goal in projection.open_goals if goal != title]
            if title and title not in projection.cancelled_goals:
                projection.cancelled_goals.append(title)
        else:
            return
        if projection.goals:
            projection.sync_lists()
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=AGENDA_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_autonomy_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        decision_kind = autonomy_decision_kind_for_event_type(event.event_type)
        if decision_kind is None:
            return

        projection = self.get_autonomy_ledger_projection(scope_key=event.thread_id)
        candidate_goal_id = _optional_text(payload.get("candidate_goal_id"))
        reason = _optional_text(payload.get("reason"))
        reason_details = dict(payload.get("reason_details") or {})
        reason_codes = sorted(
            {
                str(item).strip()
                for item in payload.get("reason_codes", [])
                if str(item).strip()
            }
        )
        executive_policy = (
            dict(payload.get("executive_policy", {}))
            if isinstance(payload.get("executive_policy"), dict)
            else None
        )
        expected_reevaluation_condition = _optional_text(
            payload.get("expected_reevaluation_condition")
        )
        expected_reevaluation_condition_kind = _optional_text(
            payload.get("expected_reevaluation_condition_kind")
        )
        expected_reevaluation_condition_details = dict(
            payload.get("expected_reevaluation_condition_details") or {}
        )
        existing_candidate = projection.candidate(candidate_goal_id)
        summary: str | None = existing_candidate.summary if existing_candidate is not None else None

        if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED:
            if not isinstance(payload.get("candidate_goal"), dict):
                return
            candidate = BrainCandidateGoal.from_dict(payload.get("candidate_goal"))
            if not candidate.candidate_goal_id:
                return
            projection.current_candidates = [
                item
                for item in projection.current_candidates
                if item.candidate_goal_id != candidate.candidate_goal_id
            ]
            projection.current_candidates.append(candidate)
            candidate_goal_id = candidate.candidate_goal_id
            summary = candidate.summary
        elif event.event_type in {
            BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
            BrainEventType.GOAL_CANDIDATE_MERGED,
            BrainEventType.GOAL_CANDIDATE_ACCEPTED,
            BrainEventType.GOAL_CANDIDATE_EXPIRED,
        }:
            if candidate_goal_id:
                projection.current_candidates = [
                    item
                    for item in projection.current_candidates
                    if item.candidate_goal_id != candidate_goal_id
                ]
        elif event.event_type == BrainEventType.DIRECTOR_NON_ACTION_RECORDED:
            if summary is None and reason is not None:
                summary = reason
            if existing_candidate is not None:
                existing_candidate.expected_reevaluation_condition = expected_reevaluation_condition
                existing_candidate.expected_reevaluation_condition_kind = (
                    expected_reevaluation_condition_kind
                )
                existing_candidate.expected_reevaluation_condition_details = dict(
                    expected_reevaluation_condition_details
                )

        entry = BrainAutonomyLedgerEntry(
            event_id=event.event_id,
            event_type=event.event_type,
            decision_kind=decision_kind,
            candidate_goal_id=candidate_goal_id,
            summary=summary,
            reason=reason,
            reason_details=reason_details,
            reason_codes=reason_codes,
            executive_policy=executive_policy,
            merged_into_candidate_goal_id=_optional_text(payload.get("merged_into_candidate_goal_id")),
            accepted_goal_id=_optional_text(payload.get("goal_id")),
            accepted_commitment_id=_optional_text(payload.get("commitment_id")),
            expected_reevaluation_condition=expected_reevaluation_condition,
            expected_reevaluation_condition_kind=expected_reevaluation_condition_kind,
            expected_reevaluation_condition_details=expected_reevaluation_condition_details,
            ts=event.ts,
        )
        projection.recent_entries.append(entry)
        projection.trim_recent_entries()
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=AUTONOMY_LEDGER_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

    def _apply_heartbeat_event(
        self,
        *,
        event: BrainEventRecord,
        payload: dict[str, Any],
        updated_at: str,
    ):
        projection = self._get_heartbeat_projection(scope_key=event.thread_id)
        projection.last_event_type = event.event_type
        projection.last_event_at = event.ts
        if event.event_type in {BrainEventType.TOOL_CALLED, BrainEventType.TOOL_COMPLETED}:
            projection.last_tool_name = payload.get("function_name")
        if event.event_type in {
            BrainEventType.CAPABILITY_REQUESTED,
            BrainEventType.CAPABILITY_COMPLETED,
            BrainEventType.CAPABILITY_FAILED,
        }:
            projection.last_tool_name = payload.get("capability_id") or projection.last_tool_name
        if event.event_type == BrainEventType.ROBOT_ACTION_OUTCOME:
            projection.last_robot_action = payload.get("action_id")
            projection.warnings = list(payload.get("status", {}).get("warnings", []))
        if event.event_type == BrainEventType.BODY_STATE_UPDATED:
            snapshot = BrainPresenceSnapshot.from_dict(payload.get("snapshot"))
            projection.warnings = list(snapshot.warnings)
        if event.event_type in {
            BrainEventType.PERCEPTION_OBSERVED,
            BrainEventType.ENGAGEMENT_CHANGED,
            BrainEventType.ATTENTION_CHANGED,
            BrainEventType.SCENE_CHANGED,
        }:
            body_state = self._get_body_state_projection(
                scope_key=str(
                    payload.get("presence_scope_key") or payload.get("scope_key") or "local:presence"
                )
            )
            projection.warnings = list(body_state.warnings)
        projection.updated_at = updated_at
        self._upsert_projection(
            projection_name=HEARTBEAT_PROJECTION,
            scope_key=event.thread_id,
            projection=projection.as_dict(),
            source_event_id=event.event_id,
            updated_at=updated_at,
            commit=False,
        )

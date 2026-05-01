"""Structured autobiographical entries for Blink continuity memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventType
from blink.brain.memory_v2.governance import normalize_reason_codes
from blink.brain.projections import BrainClaimRetentionClass, BrainClaimReviewState


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


class BrainAutobiographyEntryKind(str, Enum):
    """Well-known autobiographical entry kinds."""

    SHARED_HISTORY_SUMMARY = "shared_history_summary"
    RELATIONSHIP_ARC = "relationship_arc"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    PROJECT_ARC = "project_arc"
    SCENE_EPISODE = "scene_episode"


@dataclass(frozen=True)
class BrainAutobiographicalEntryRecord:
    """One inspectable autobiographical entry row."""

    entry_id: str
    scope_type: str
    scope_id: str
    entry_kind: str
    rendered_summary: str
    content_json: str
    status: str
    salience: float
    source_episode_ids_json: str
    source_claim_ids_json: str
    source_event_ids_json: str
    modality: str | None
    review_state: str | None
    retention_class: str | None
    privacy_class: str | None
    governance_reason_codes_json: str
    last_governance_event_id: str | None
    source_presence_scope_key: str | None
    source_scene_entity_ids_json: str
    source_scene_affordance_ids_json: str
    redacted_at: str | None
    supersedes_entry_id: str | None
    valid_from: str
    valid_to: str | None
    created_at: str
    updated_at: str

    @property
    def content(self) -> dict[str, Any]:
        """Return the decoded content payload."""
        return dict(json.loads(self.content_json))

    @property
    def source_episode_ids(self) -> list[int]:
        """Return source episode ids."""
        return [int(value) for value in json.loads(self.source_episode_ids_json)]

    @property
    def source_claim_ids(self) -> list[str]:
        """Return source claim ids."""
        return [str(value) for value in json.loads(self.source_claim_ids_json)]

    @property
    def source_event_ids(self) -> list[str]:
        """Return source event ids."""
        return [str(value) for value in json.loads(self.source_event_ids_json)]

    @property
    def governance_reason_codes(self) -> list[str]:
        """Return the normalized governance reason codes."""
        return [str(value) for value in json.loads(self.governance_reason_codes_json or "[]")]

    @property
    def source_scene_entity_ids(self) -> list[str]:
        """Return source scene entity ids."""
        return [str(value) for value in json.loads(self.source_scene_entity_ids_json or "[]")]

    @property
    def source_scene_affordance_ids(self) -> list[str]:
        """Return source scene affordance ids."""
        return [str(value) for value in json.loads(self.source_scene_affordance_ids_json or "[]")]


class AutobiographyService:
    """Typed autobiographical-entry persistence on top of ``BrainStore``."""

    _VERSIONED_KINDS = {
        BrainAutobiographyEntryKind.SHARED_HISTORY_SUMMARY.value,
        BrainAutobiographyEntryKind.RELATIONSHIP_ARC.value,
    }

    def __init__(self, *, store):
        """Bind the service to one canonical store."""
        self._store = store

    def list_entries(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        entry_kinds: tuple[str, ...] | None = None,
        statuses: tuple[str, ...] | None = ("current",),
        review_states: tuple[str, ...] | None = None,
        retention_classes: tuple[str, ...] | None = None,
        privacy_classes: tuple[str, ...] | None = None,
        modalities: tuple[str, ...] | None = None,
        limit: int = 16,
    ) -> list[BrainAutobiographicalEntryRecord]:
        """Return autobiographical entries for an optional scope/kind filter."""
        clauses: list[str] = []
        params: list[Any] = []
        if scope_type is not None:
            clauses.append("scope_type = ?")
            params.append(scope_type)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if entry_kinds:
            clauses.append(f"entry_kind IN ({','.join('?' for _ in entry_kinds)})")
            params.extend(entry_kinds)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        if review_states:
            clauses.append(f"review_state IN ({','.join('?' for _ in review_states)})")
            params.extend(review_states)
        if retention_classes:
            clauses.append(f"retention_class IN ({','.join('?' for _ in retention_classes)})")
            params.extend(retention_classes)
        if privacy_classes:
            clauses.append(f"privacy_class IN ({','.join('?' for _ in privacy_classes)})")
            params.extend(privacy_classes)
        if modalities:
            clauses.append(f"modality IN ({','.join('?' for _ in modalities)})")
            params.extend(modalities)
        query = "SELECT * FROM autobiographical_entries"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        rows = self._store._conn.execute(query, (*params, limit)).fetchall()
        return [self._row_to_entry(row) for row in rows if row is not None]

    def latest_entry(
        self,
        *,
        scope_type: str,
        scope_id: str,
        entry_kind: str,
    ) -> BrainAutobiographicalEntryRecord | None:
        """Return the latest current entry for one scope/kind."""
        entries = self.list_entries(
            scope_type=scope_type,
            scope_id=scope_id,
            entry_kinds=(entry_kind,),
            statuses=("current",),
            limit=1,
        )
        return entries[0] if entries else None

    def upsert_entry(
        self,
        *,
        scope_type: str,
        scope_id: str,
        entry_kind: str,
        rendered_summary: str,
        content: dict[str, Any],
        salience: float,
        source_episode_ids: list[int] | None = None,
        source_claim_ids: list[str] | None = None,
        source_event_ids: list[str] | None = None,
        source_event_id: str | None = None,
        valid_from: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        append_only: bool = False,
        identity_key: str | None = None,
        modality: str | None = None,
        review_state: str | None = None,
        retention_class: str | None = None,
        privacy_class: str | None = None,
        governance_reason_codes: list[str] | None = None,
        last_governance_event_id: str | None = None,
        source_presence_scope_key: str | None = None,
        source_scene_entity_ids: list[str] | None = None,
        source_scene_affordance_ids: list[str] | None = None,
        redacted_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainAutobiographicalEntryRecord:
        """Insert a new autobiographical entry and supersede prior current state when needed."""
        now = self._store._utc_now_for_memory_v2()
        record_created_at = created_at or updated_at or valid_from or now
        record_updated_at = updated_at or created_at or valid_from or now
        summary = " ".join((rendered_summary or "").split()).strip()
        if not summary:
            raise ValueError("Autobiographical summary must not be empty.")

        normalized_content = dict(content or {})
        source_episode_ids = sorted({int(value) for value in (source_episode_ids or [])})
        source_claim_ids = sorted({str(value) for value in (source_claim_ids or []) if str(value).strip()})
        source_event_ids = sorted({str(value) for value in (source_event_ids or []) if str(value).strip()})
        source_scene_entity_ids = sorted(
            {str(value) for value in (source_scene_entity_ids or []) if str(value).strip()}
        )
        source_scene_affordance_ids = sorted(
            {str(value) for value in (source_scene_affordance_ids or []) if str(value).strip()}
        )
        governance_reason_codes = sorted(
            {str(value) for value in (governance_reason_codes or []) if str(value).strip()}
        )
        normalized_content_json = json.dumps(normalized_content, ensure_ascii=False, sort_keys=True)

        current = None
        if append_only:
            identity = identity_key or str(normalized_content.get("project_key") or summary)
            for candidate in self.list_entries(
                scope_type=scope_type,
                scope_id=scope_id,
                entry_kinds=(entry_kind,),
                statuses=("current",),
                limit=32,
            ):
                candidate_identity = str(
                    candidate.content.get("identity_key")
                    or candidate.content.get("project_key")
                    or candidate.rendered_summary
                )
                if candidate_identity == identity:
                    current = candidate
                    break
        else:
            current = self.latest_entry(scope_type=scope_type, scope_id=scope_id, entry_kind=entry_kind)

        if current is not None:
            same_content = (
                current.rendered_summary == summary
                and current.content_json == normalized_content_json
                and round(float(current.salience), 4) == round(float(salience), 4)
            )
            if same_content:
                return current

        if current is not None and (
            not append_only or entry_kind in self._VERSIONED_KINDS or identity_key is not None
        ):
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET status = 'superseded',
                    valid_to = COALESCE(valid_to, ?),
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (record_updated_at, record_updated_at, current.entry_id),
            )

        identity = identity_key or normalized_content.get("identity_key") or summary
        entry_id = _stable_id(
            "autobio",
            scope_type,
            scope_id,
            entry_kind,
            identity,
            json.dumps(source_episode_ids, ensure_ascii=False),
            json.dumps(source_claim_ids, ensure_ascii=False),
            json.dumps(source_event_ids, ensure_ascii=False),
            source_event_id or "",
        )
        self._store._conn.execute(
            """
            INSERT OR REPLACE INTO autobiographical_entries (
                entry_id, scope_type, scope_id, entry_kind, rendered_summary, content_json,
                status, salience, source_episode_ids_json, source_claim_ids_json, source_event_ids_json,
                modality, review_state, retention_class, privacy_class, governance_reason_codes_json,
                last_governance_event_id, source_presence_scope_key, source_scene_entity_ids_json,
                source_scene_affordance_ids_json, redacted_at, supersedes_entry_id, valid_from, valid_to,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                scope_type,
                scope_id,
                entry_kind,
                summary,
                normalized_content_json,
                "current",
                max(0.0, min(10.0, float(salience))),
                json.dumps(source_episode_ids, ensure_ascii=False, sort_keys=True),
                json.dumps(source_claim_ids, ensure_ascii=False, sort_keys=True),
                json.dumps(source_event_ids, ensure_ascii=False, sort_keys=True),
                modality,
                review_state,
                retention_class,
                privacy_class,
                json.dumps(governance_reason_codes, ensure_ascii=False, sort_keys=True),
                last_governance_event_id,
                source_presence_scope_key,
                json.dumps(source_scene_entity_ids, ensure_ascii=False, sort_keys=True),
                json.dumps(source_scene_affordance_ids, ensure_ascii=False, sort_keys=True),
                redacted_at,
                current.entry_id if current is not None else None,
                valid_from or record_updated_at,
                None,
                record_created_at,
                record_updated_at,
            ),
        )
        self._store._conn.commit()
        record = self._get_entry(entry_id)
        if event_context is not None:
            self._store.append_brain_event(
                event_type=BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "entry_id": record.entry_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "entry_kind": record.entry_kind,
                    "rendered_summary": record.rendered_summary,
                    "content": record.content,
                    "status": record.status,
                    "salience": record.salience,
                    "source_episode_ids": record.source_episode_ids,
                    "source_claim_ids": record.source_claim_ids,
                    "source_event_ids": record.source_event_ids,
                    "modality": record.modality,
                    "review_state": record.review_state,
                    "retention_class": record.retention_class,
                    "privacy_class": record.privacy_class,
                    "governance_reason_codes": record.governance_reason_codes,
                    "last_governance_event_id": record.last_governance_event_id,
                    "source_presence_scope_key": record.source_presence_scope_key,
                    "source_scene_entity_ids": record.source_scene_entity_ids,
                    "source_scene_affordance_ids": record.source_scene_affordance_ids,
                    "redacted_at": record.redacted_at,
                    "supersedes_entry_id": record.supersedes_entry_id,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                },
            )
        return record

    def request_entry_review(
        self,
        entry_id: str,
        *,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainAutobiographicalEntryRecord:
        """Move one autobiographical entry into explicit requested review."""
        record = self._get_entry(entry_id)
        normalized_reason_codes = list(normalize_reason_codes(reason_codes or ()))
        now = updated_at or self._store._utc_now_for_memory_v2()
        self._store._conn.execute(
            """
            UPDATE autobiographical_entries
            SET review_state = ?,
                governance_reason_codes_json = ?,
                updated_at = ?
            WHERE entry_id = ?
            """,
            (
                BrainClaimReviewState.REQUESTED.value,
                json.dumps(normalized_reason_codes, ensure_ascii=False, sort_keys=True),
                now,
                entry_id,
            ),
        )
        if event_context is None and source_event_id:
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (source_event_id, now, entry_id),
            )
        self._store._conn.commit()
        refreshed = self._get_entry(entry_id)
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.AUTOBIOGRAPHY_ENTRY_REVIEW_REQUESTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "entry_id": refreshed.entry_id,
                    "scope_type": refreshed.scope_type,
                    "scope_id": refreshed.scope_id,
                    "reason_codes": normalized_reason_codes,
                    "summary": summary,
                    "notes": notes,
                    "review_state": BrainClaimReviewState.REQUESTED.value,
                },
            )
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (event.event_id, event.ts, entry_id),
            )
            self._store._conn.commit()
            refreshed = self._get_entry(entry_id)
        return refreshed

    def reclassify_entry_retention(
        self,
        entry_id: str,
        *,
        retention_class: str,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainAutobiographicalEntryRecord:
        """Update explicit autobiographical retention without bypassing the event spine."""
        normalized_reason_codes = list(normalize_reason_codes(reason_codes or ()))
        now = updated_at or self._store._utc_now_for_memory_v2()
        resolved_retention_class = BrainClaimRetentionClass(
            str(getattr(retention_class, "value", retention_class))
        ).value
        self._store._conn.execute(
            """
            UPDATE autobiographical_entries
            SET retention_class = ?,
                governance_reason_codes_json = ?,
                updated_at = ?
            WHERE entry_id = ?
            """,
            (
                resolved_retention_class,
                json.dumps(normalized_reason_codes, ensure_ascii=False, sort_keys=True),
                now,
                entry_id,
            ),
        )
        if event_context is None and source_event_id:
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (source_event_id, now, entry_id),
            )
        self._store._conn.commit()
        refreshed = self._get_entry(entry_id)
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.AUTOBIOGRAPHY_ENTRY_RETENTION_RECLASSIFIED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "entry_id": refreshed.entry_id,
                    "scope_type": refreshed.scope_type,
                    "scope_id": refreshed.scope_id,
                    "reason_codes": normalized_reason_codes,
                    "summary": summary,
                    "notes": notes,
                    "retention_class": resolved_retention_class,
                },
            )
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (event.event_id, event.ts, entry_id),
            )
            self._store._conn.commit()
            refreshed = self._get_entry(entry_id)
        return refreshed

    def redact_entry(
        self,
        entry_id: str,
        *,
        redacted_summary: str,
        source_event_id: str | None,
        reason_codes: list[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainAutobiographicalEntryRecord:
        """Redact rendered autobiographical content without deleting the entry row."""
        record = self._get_entry(entry_id)
        normalized_summary = " ".join((redacted_summary or "").split()).strip()
        if not normalized_summary:
            raise ValueError("redacted_summary must not be empty")
        normalized_reason_codes = list(normalize_reason_codes(reason_codes or ()))
        content = dict(record.content)
        content.update(
            {
                "summary": normalized_summary,
                "redacted": True,
                "redacted_summary": normalized_summary,
            }
        )
        now = updated_at or self._store._utc_now_for_memory_v2()
        self._store._conn.execute(
            """
            UPDATE autobiographical_entries
            SET rendered_summary = ?,
                content_json = ?,
                privacy_class = ?,
                review_state = ?,
                governance_reason_codes_json = ?,
                redacted_at = ?,
                updated_at = ?
            WHERE entry_id = ?
            """,
            (
                normalized_summary,
                json.dumps(content, ensure_ascii=False, sort_keys=True),
                "redacted",
                BrainClaimReviewState.RESOLVED.value,
                json.dumps(normalized_reason_codes, ensure_ascii=False, sort_keys=True),
                now,
                now,
                entry_id,
            ),
        )
        if event_context is None and source_event_id:
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (source_event_id, now, entry_id),
            )
        self._store._conn.commit()
        refreshed = self._get_entry(entry_id)
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.AUTOBIOGRAPHY_ENTRY_REDACTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "entry_id": refreshed.entry_id,
                    "scope_type": refreshed.scope_type,
                    "scope_id": refreshed.scope_id,
                    "reason_codes": normalized_reason_codes,
                    "summary": summary,
                    "notes": notes,
                    "redacted_summary": normalized_summary,
                },
            )
            self._store._conn.execute(
                """
                UPDATE autobiographical_entries
                SET last_governance_event_id = ?,
                    redacted_at = ?,
                    updated_at = ?
                WHERE entry_id = ?
                """,
                (event.event_id, event.ts, event.ts, entry_id),
            )
            self._store._conn.commit()
            refreshed = self._get_entry(entry_id)
        return refreshed

    def _get_entry(self, entry_id: str) -> BrainAutobiographicalEntryRecord:
        row = self._store._conn.execute(
            "SELECT * FROM autobiographical_entries WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Missing autobiographical entry id {entry_id}")
        return self._row_to_entry(row)

    def _row_to_entry(self, row) -> BrainAutobiographicalEntryRecord:
        return BrainAutobiographicalEntryRecord(
            entry_id=str(row["entry_id"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            entry_kind=str(row["entry_kind"]),
            rendered_summary=str(row["rendered_summary"]),
            content_json=str(row["content_json"]),
            status=str(row["status"]),
            salience=float(row["salience"]),
            source_episode_ids_json=str(row["source_episode_ids_json"]),
            source_claim_ids_json=str(row["source_claim_ids_json"]),
            source_event_ids_json=str(row["source_event_ids_json"]),
            modality=str(row["modality"]) if row["modality"] is not None else None,
            review_state=str(row["review_state"]) if row["review_state"] is not None else None,
            retention_class=(
                str(row["retention_class"]) if row["retention_class"] is not None else None
            ),
            privacy_class=str(row["privacy_class"]) if row["privacy_class"] is not None else None,
            governance_reason_codes_json=str(row["governance_reason_codes_json"] or "[]"),
            last_governance_event_id=(
                str(row["last_governance_event_id"])
                if row["last_governance_event_id"] is not None
                else None
            ),
            source_presence_scope_key=(
                str(row["source_presence_scope_key"])
                if row["source_presence_scope_key"] is not None
                else None
            ),
            source_scene_entity_ids_json=str(row["source_scene_entity_ids_json"] or "[]"),
            source_scene_affordance_ids_json=str(row["source_scene_affordance_ids_json"] or "[]"),
            redacted_at=str(row["redacted_at"]) if row["redacted_at"] is not None else None,
            supersedes_entry_id=(
                str(row["supersedes_entry_id"]) if row["supersedes_entry_id"] is not None else None
            ),
            valid_from=str(row["valid_from"]),
            valid_to=str(row["valid_to"]) if row["valid_to"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

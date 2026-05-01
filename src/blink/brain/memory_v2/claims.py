"""Temporal claim ledger for Blink continuity memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.bounded_json import (
    MAX_CLAIM_OBJECT_JSON_CHARS,
    dumps_bounded_json,
    safe_json_dict,
)
from blink.brain.events import BrainEventType
from blink.brain.memory_layers.semantic import render_preference_fact, render_profile_fact
from blink.brain.memory_v2.governance import (
    claim_matches_temporal_mode,
    effective_claim_currentness_status,
    effective_claim_retention_class,
    effective_claim_review_state,
    normalize_reason_codes,
    seed_claim_governance,
    transition_claim_governance,
)
from blink.brain.projections import (
    BrainClaimCurrentnessStatus,
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainGovernanceReasonCode,
)


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class BrainClaimTemporalMode(str, Enum):
    """Retrieval mode for continuity claims."""

    CURRENT = "current"
    HISTORICAL = "historical"
    ALL = "all"


@dataclass(frozen=True)
class BrainClaimRecord:
    """One versioned temporal claim row."""

    claim_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str | None
    object_json: str
    status: str
    confidence: float
    valid_from: str
    valid_to: str | None
    source_event_id: str | None
    scope_type: str | None
    scope_id: str | None
    claim_key: str | None
    stale_after_seconds: int | None
    currentness_status: str | None
    review_state: str | None
    retention_class: str | None
    governance_reason_codes: tuple[str, ...]
    last_governance_event_id: str | None
    governance_updated_at: str | None
    created_at: str
    updated_at: str

    @property
    def object(self) -> dict[str, Any]:
        """Return the decoded claim object payload."""
        return safe_json_dict(
            self.object_json,
            max_chars=MAX_CLAIM_OBJECT_JSON_CHARS,
            overflow_kind="claim_object_too_large",
        )

    @property
    def is_current(self) -> bool:
        """Return whether this claim should be treated as current truth."""
        return self.effective_currentness_status == BrainClaimCurrentnessStatus.CURRENT.value

    @property
    def is_stale(self) -> bool:
        """Return whether the claim is past its freshness window."""
        return self.effective_currentness_status == BrainClaimCurrentnessStatus.STALE.value

    @property
    def is_historical(self) -> bool:
        """Return whether this claim is explicitly historical."""
        return self.effective_currentness_status == BrainClaimCurrentnessStatus.HISTORICAL.value

    @property
    def is_held(self) -> bool:
        """Return whether this claim is explicitly held for review."""
        return self.effective_currentness_status == BrainClaimCurrentnessStatus.HELD.value

    @property
    def effective_currentness_status(self) -> str:
        """Return the explicit or compatibility currentness state."""
        return effective_claim_currentness_status(
            self.currentness_status,
            truth_status=self.status,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            stale_after_seconds=self.stale_after_seconds,
        )

    @property
    def effective_review_state(self) -> str:
        """Return the explicit or compatibility review state."""
        return effective_claim_review_state(self.review_state)

    @property
    def effective_retention_class(self) -> str:
        """Return the explicit or compatibility retention class."""
        return effective_claim_retention_class(
            self.retention_class,
            scope_type=self.scope_type,
            predicate=self.predicate,
        )


@dataclass(frozen=True)
class BrainClaimEvidenceRecord:
    """One evidence row supporting a claim."""

    evidence_id: str
    claim_id: str
    source_event_id: str | None
    source_episode_id: int | None
    evidence_summary: str
    evidence_json: str
    created_at: str

    @property
    def evidence(self) -> dict[str, Any]:
        """Return decoded evidence metadata."""
        return dict(json.loads(self.evidence_json))


@dataclass(frozen=True)
class BrainClaimSupersessionRecord:
    """One correction link from prior claim to replacement claim."""

    supersession_id: str
    prior_claim_id: str
    new_claim_id: str
    reason: str
    source_event_id: str | None
    created_at: str


def render_claim_summary(
    claim: BrainClaimRecord,
    *,
    subject_name: str | None = None,
    object_name: str | None = None,
) -> str:
    """Render a claim into compact human-readable text."""
    object_value = (
        object_name or str(claim.object.get("value") or claim.object.get("summary") or "").strip()
    )
    if claim.predicate in {"profile.name", "profile.role", "profile.origin"} and object_value:
        return render_profile_fact(claim.predicate, object_value)
    if claim.predicate in {"preference.like", "preference.dislike"} and object_value:
        return render_preference_fact(claim.predicate, object_value)
    if claim.predicate == "session.summary":
        return object_value
    subject = subject_name or claim.subject_entity_id
    if object_value:
        return f"{subject} {claim.predicate} {object_value}"
    return f"{subject} {claim.predicate}"


class ClaimLedger:
    """Append-mostly continuity claim ledger."""

    def __init__(self, *, store):
        """Bind the ledger to one canonical store."""
        self._store = store

    def _refresh_claim_governance_projection_for_scope(
        self,
        *,
        scope_type: str | None,
        scope_id: str | None,
        source_event_id: str | None,
        updated_at: str | None,
        commit: bool,
    ):
        if not scope_type or not scope_id:
            return
        self._store.build_claim_governance_projection(
            scope_type=scope_type,
            scope_id=scope_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            commit=commit,
        )

    def _refresh_claim_governance_projection_for_claim(
        self,
        *,
        claim_id: str,
        source_event_id: str | None,
        updated_at: str | None,
        commit: bool,
    ):
        claim = self.get_claim(claim_id)
        if claim is None:
            return
        self._refresh_claim_governance_projection_for_scope(
            scope_type=claim.scope_type,
            scope_id=claim.scope_id,
            source_event_id=source_event_id,
            updated_at=updated_at,
            commit=commit,
        )

    def _apply_claim_governance_update(
        self,
        *,
        claim_id: str,
        truth_status: str | None = None,
        valid_to: str | None = None,
        confidence: float | None = None,
        governance_transition: str,
        source_event_id: str | None,
        updated_at: str | None = None,
        reason_codes: Iterable[str] | None = None,
        explicit_review_state: str | None = None,
        explicit_retention_class: str | None = None,
    ) -> int:
        record = self.get_claim(claim_id)
        if record is None:
            raise KeyError(f"Missing claim id {claim_id}")
        transition_at = updated_at or _utc_now()
        next_truth_status = truth_status or record.status
        next_valid_to = record.valid_to if valid_to is None else valid_to
        governance_state = transition_claim_governance(
            truth_status=next_truth_status,
            currentness_status=record.currentness_status,
            review_state=record.review_state,
            retention_class=record.retention_class,
            existing_reason_codes=record.governance_reason_codes,
            scope_type=record.scope_type,
            predicate=record.predicate,
            transition=governance_transition,
            updated_at=transition_at,
            last_governance_event_id=source_event_id,
            reason_codes=reason_codes,
            explicit_review_state=explicit_review_state,
            explicit_retention_class=explicit_retention_class,
        )
        cursor = self._store._conn.execute(
            """
            UPDATE claims
            SET status = ?, valid_to = ?, confidence = ?, currentness_status = ?,
                review_state = ?, retention_class = ?, governance_reason_codes_json = ?,
                last_governance_event_id = ?, governance_updated_at = ?, updated_at = ?
            WHERE claim_id = ?
            """,
            (
                next_truth_status,
                next_valid_to,
                (
                    max(0.0, min(1.0, float(confidence)))
                    if confidence is not None
                    else record.confidence
                ),
                governance_state.currentness_status,
                governance_state.review_state,
                governance_state.retention_class,
                json.dumps(list(governance_state.reason_codes), ensure_ascii=False, sort_keys=True),
                governance_state.last_governance_event_id,
                governance_state.governance_updated_at,
                transition_at,
                claim_id,
            ),
        )
        self._upsert_claim_fts(claim_id)
        self._refresh_claim_governance_projection_for_scope(
            scope_type=record.scope_type,
            scope_id=record.scope_id,
            source_event_id=source_event_id,
            updated_at=transition_at,
            commit=False,
        )
        return int(cursor.rowcount)

    def _set_last_governance_event_id(
        self,
        *,
        claim_id: str,
        event_id: str,
        updated_at: str,
    ):
        record = self.get_claim(claim_id)
        if record is None:
            return
        self._store._conn.execute(
            """
            UPDATE claims
            SET last_governance_event_id = ?, governance_updated_at = ?, updated_at = ?
            WHERE claim_id = ?
            """,
            (event_id, updated_at, updated_at, claim_id),
        )
        self._refresh_claim_governance_projection_for_scope(
            scope_type=record.scope_type,
            scope_id=record.scope_id,
            source_event_id=event_id,
            updated_at=updated_at,
            commit=False,
        )

    def record_claim(
        self,
        *,
        subject_entity_id: str,
        predicate: str,
        object_entity_id: str | None = None,
        object_value: str | None = None,
        object_data: dict[str, Any] | None = None,
        status: str = "active",
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source_event_id: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        claim_key: str | None = None,
        stale_after_seconds: int | None = None,
        evidence_summary: str | None = None,
        evidence_json: dict[str, Any] | None = None,
        source_episode_id: int | None = None,
        currentness_status: str | None = None,
        review_state: str | None = None,
        retention_class: str | None = None,
        reason_codes: Iterable[str] | None = None,
        last_governance_event_id: str | None = None,
        governance_updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Insert one claim row if it does not already exist."""
        now = _utc_now()
        normalized_object = dict(object_data or {})
        if object_value is not None and "value" not in normalized_object:
            normalized_object["value"] = object_value
        object_json = dumps_bounded_json(
            normalized_object,
            max_chars=MAX_CLAIM_OBJECT_JSON_CHARS,
            overflow_kind="claim_object_too_large",
            max_depth=4,
            max_items=12,
            max_string_chars=512,
        )
        claim_key = (
            claim_key or f"{scope_type}:{scope_id}:{predicate}"
            if scope_type and scope_id
            else claim_key
        )
        effective_valid_from = valid_from or now
        claim_id = _stable_id(
            "claim",
            subject_entity_id,
            predicate,
            object_entity_id or "",
            object_json,
            status,
            effective_valid_from,
            source_event_id or "",
            claim_key or "",
        )
        existing = self.get_claim(claim_id)
        if existing is not None:
            return existing
        governance_state = seed_claim_governance(
            scope_type=scope_type,
            predicate=predicate,
            truth_status=status,
            updated_at=governance_updated_at or now,
            currentness_status=currentness_status,
            review_state=review_state,
            retention_class=retention_class,
            reason_codes=reason_codes,
            last_governance_event_id=last_governance_event_id,
        )

        self._store._conn.execute(
            """
            INSERT INTO claims (
                claim_id, subject_entity_id, predicate, object_entity_id, object_json, status,
                confidence, valid_from, valid_to, source_event_id, scope_type, scope_id,
                claim_key, stale_after_seconds, currentness_status, review_state,
                retention_class, governance_reason_codes_json, last_governance_event_id,
                governance_updated_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id,
                subject_entity_id,
                predicate,
                object_entity_id,
                object_json,
                status,
                max(0.0, min(1.0, float(confidence))),
                effective_valid_from,
                valid_to,
                source_event_id,
                scope_type,
                scope_id,
                claim_key,
                stale_after_seconds,
                governance_state.currentness_status,
                governance_state.review_state,
                governance_state.retention_class,
                json.dumps(list(governance_state.reason_codes), ensure_ascii=False, sort_keys=True),
                governance_state.last_governance_event_id,
                governance_state.governance_updated_at,
                now,
                now,
            ),
        )
        if any(
            value is not None
            for value in (source_event_id, source_episode_id, evidence_summary, evidence_json)
        ):
            evidence_id = _stable_id(
                "evidence",
                claim_id,
                source_event_id or "",
                source_episode_id or "",
                evidence_summary or "",
            )
            self._store._conn.execute(
                """
                INSERT OR IGNORE INTO claim_evidence (
                    evidence_id, claim_id, source_event_id, source_episode_id,
                    evidence_summary, evidence_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    claim_id,
                    source_event_id,
                    source_episode_id,
                    evidence_summary or "",
                    json.dumps(evidence_json or {}, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        self._upsert_claim_fts(claim_id)
        self._refresh_claim_governance_projection_for_scope(
            scope_type=scope_type,
            scope_id=scope_id,
            source_event_id=governance_state.last_governance_event_id or source_event_id,
            updated_at=governance_state.governance_updated_at,
            commit=False,
        )
        self._store._conn.commit()
        record = self.get_claim(claim_id)
        if record is None:
            raise RuntimeError("Failed to persist claim.")
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_RECORDED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": record.claim_id,
                    "subject_entity_id": record.subject_entity_id,
                    "predicate": record.predicate,
                    "object_entity_id": record.object_entity_id,
                    "object": record.object,
                    "status": record.status,
                    "confidence": record.confidence,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "claim_key": record.claim_key,
                    "currentness_status": record.effective_currentness_status,
                    "review_state": record.effective_review_state,
                    "retention_class": record.effective_retention_class,
                    "reason_codes": list(record.governance_reason_codes),
                    "last_governance_event_id": record.last_governance_event_id,
                    "governance_updated_at": record.governance_updated_at,
                    "stale_after_seconds": record.stale_after_seconds,
                },
            )
            self._set_last_governance_event_id(
                claim_id=record.claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
            refreshed = self.get_claim(record.claim_id)
            if refreshed is not None:
                record = refreshed
        return record

    def record_evidence(
        self,
        *,
        claim_id: str,
        source_event_id: str | None = None,
        source_episode_id: int | None = None,
        evidence_summary: str,
        evidence_json: dict[str, Any] | None = None,
    ) -> BrainClaimEvidenceRecord:
        """Append one evidence row to an existing claim without creating a new claim."""
        claim = self.get_claim(claim_id)
        if claim is None:
            raise KeyError(f"Missing claim id {claim_id}")
        now = _utc_now()
        evidence_id = _stable_id(
            "evidence",
            claim_id,
            source_event_id or "",
            source_episode_id or "",
            evidence_summary,
            json.dumps(evidence_json or {}, ensure_ascii=False, sort_keys=True),
        )
        self._store._conn.execute(
            """
            INSERT OR IGNORE INTO claim_evidence (
                evidence_id, claim_id, source_event_id, source_episode_id,
                evidence_summary, evidence_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                claim_id,
                source_event_id,
                source_episode_id,
                evidence_summary,
                json.dumps(evidence_json or {}, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        self._store._conn.commit()
        evidence = self.claim_evidence(claim_id)
        if not evidence:
            raise RuntimeError("Failed to persist claim evidence.")
        return evidence[0]

    def supersede_claim(
        self,
        prior_claim_id: str,
        replacement_claim: dict[str, Any],
        reason: str,
        source_event_id: str | None,
        *,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Supersede a prior claim by appending a new replacement row."""
        prior = self.get_claim(prior_claim_id)
        if prior is None:
            raise KeyError(f"Missing prior claim id {prior_claim_id}")
        now = _utc_now()
        self._apply_claim_governance_update(
            claim_id=prior_claim_id,
            truth_status="superseded",
            valid_to=now,
            governance_transition="superseded",
            source_event_id=source_event_id,
            updated_at=now,
        )
        replacement = self.record_claim(
            source_event_id=source_event_id,
            valid_from=now,
            event_context=event_context,
            **replacement_claim,
        )
        supersession_id = _stable_id("supersession", prior_claim_id, replacement.claim_id, reason)
        self._store._conn.execute(
            """
            INSERT OR IGNORE INTO claim_supersessions (
                supersession_id, prior_claim_id, new_claim_id, reason, source_event_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                supersession_id,
                prior_claim_id,
                replacement.claim_id,
                reason,
                source_event_id,
                now,
            ),
        )
        self._upsert_claim_fts(prior_claim_id)
        self._upsert_claim_fts(replacement.claim_id)
        self._store._conn.commit()
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_SUPERSEDED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "prior_claim_id": prior_claim_id,
                    "new_claim_id": replacement.claim_id,
                    "reason": reason,
                    "reason_codes": [BrainGovernanceReasonCode.SUPERSEDED.value],
                },
            )
            self._set_last_governance_event_id(
                claim_id=prior_claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        return replacement

    def supersede_with_existing(
        self,
        *,
        prior_claim_id: str,
        new_claim_id: str,
        reason: str,
        source_event_id: str | None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimSupersessionRecord:
        """Supersede a prior claim by linking it to an already-persisted replacement."""
        prior = self.get_claim(prior_claim_id)
        replacement = self.get_claim(new_claim_id)
        if prior is None:
            raise KeyError(f"Missing prior claim id {prior_claim_id}")
        if replacement is None:
            raise KeyError(f"Missing replacement claim id {new_claim_id}")
        now = _utc_now()
        self._apply_claim_governance_update(
            claim_id=prior_claim_id,
            truth_status="superseded",
            valid_to=prior.valid_to or now,
            governance_transition="superseded",
            source_event_id=source_event_id,
            updated_at=now,
        )
        supersession_id = _stable_id("supersession", prior_claim_id, new_claim_id, reason)
        self._store._conn.execute(
            """
            INSERT OR IGNORE INTO claim_supersessions (
                supersession_id, prior_claim_id, new_claim_id, reason, source_event_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                supersession_id,
                prior_claim_id,
                new_claim_id,
                reason,
                source_event_id,
                now,
            ),
        )
        self._upsert_claim_fts(prior_claim_id)
        self._upsert_claim_fts(new_claim_id)
        self._store._conn.commit()
        if event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_SUPERSEDED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "prior_claim_id": prior_claim_id,
                    "new_claim_id": new_claim_id,
                    "reason": reason,
                    "reason_codes": [BrainGovernanceReasonCode.SUPERSEDED.value],
                },
            )
            self._set_last_governance_event_id(
                claim_id=prior_claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        records = self.claim_supersessions(prior_claim_id)
        if not records:
            raise RuntimeError("Failed to persist claim supersession.")
        return records[0]

    def revoke_claim(
        self,
        claim_id: str,
        *,
        reason: str,
        source_event_id: str | None,
        reason_codes: Iterable[str] | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> int:
        """Mark one claim revoked without deleting history."""
        now = updated_at or _utc_now()
        rowcount = self._apply_claim_governance_update(
            claim_id=claim_id,
            truth_status="revoked",
            valid_to=now,
            governance_transition="revoked",
            source_event_id=source_event_id,
            updated_at=now,
            reason_codes=reason_codes,
        )
        self._upsert_claim_fts(claim_id)
        self._store._conn.commit()
        if rowcount and event_context is not None:
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_REVOKED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": claim_id,
                    "reason": reason,
                    "reason_codes": list(
                        normalize_reason_codes(
                            reason_codes or (BrainGovernanceReasonCode.CONTRADICTION.value,)
                        )
                    ),
                },
            )
            self._set_last_governance_event_id(
                claim_id=claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        return int(rowcount)

    def request_claim_review(
        self,
        claim_id: str,
        *,
        source_event_id: str | None,
        reason_codes: Iterable[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Move one live claim into explicit held/requested review."""
        now = updated_at or _utc_now()
        rowcount = self._apply_claim_governance_update(
            claim_id=claim_id,
            governance_transition="review_requested",
            source_event_id=source_event_id,
            updated_at=now,
            reason_codes=reason_codes,
        )
        self._store._conn.commit()
        if rowcount and event_context is not None:
            refreshed = self.get_claim(claim_id)
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_REVIEW_REQUESTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": claim_id,
                    "scope_type": refreshed.scope_type if refreshed is not None else None,
                    "scope_id": refreshed.scope_id if refreshed is not None else None,
                    "reason_codes": list(
                        normalize_reason_codes(
                            reason_codes or (BrainGovernanceReasonCode.OPERATOR_HOLD.value,)
                        )
                    ),
                    "review_state": BrainClaimReviewState.REQUESTED.value,
                    "summary": summary,
                    "notes": notes,
                },
            )
            self._set_last_governance_event_id(
                claim_id=claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        record = self.get_claim(claim_id)
        if record is None:
            raise RuntimeError("Failed to update claim review state.")
        return record

    def revalidate_claim(
        self,
        claim_id: str,
        *,
        source_event_id: str | None,
        confidence: float | None = None,
        reason_codes: Iterable[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Return one stale or held claim to explicit current status."""
        record = self.get_claim(claim_id)
        if record is None:
            raise KeyError(f"Missing claim id {claim_id}")
        if record.status in {"revoked", "superseded"}:
            raise ValueError("Cannot revalidate a terminal historical claim.")
        now = updated_at or _utc_now()
        rowcount = self._apply_claim_governance_update(
            claim_id=claim_id,
            confidence=confidence,
            governance_transition="revalidated",
            source_event_id=source_event_id,
            updated_at=now,
            reason_codes=reason_codes,
        )
        self._store._conn.commit()
        if rowcount and event_context is not None:
            refreshed = self.get_claim(claim_id)
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_REVALIDATED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": claim_id,
                    "scope_type": refreshed.scope_type if refreshed is not None else None,
                    "scope_id": refreshed.scope_id if refreshed is not None else None,
                    "reason_codes": list(
                        normalize_reason_codes(reason_codes) if reason_codes is not None else []
                    ),
                    "summary": summary,
                    "notes": notes,
                    "confidence": confidence,
                },
            )
            self._set_last_governance_event_id(
                claim_id=claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        refreshed = self.get_claim(claim_id)
        if refreshed is None:
            raise RuntimeError("Failed to revalidate claim.")
        return refreshed

    def expire_claim(
        self,
        claim_id: str,
        *,
        source_event_id: str | None,
        reason_codes: Iterable[str] | None = None,
        review_state: str | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Move one live claim into explicit stale state without deleting it."""
        now = updated_at or _utc_now()
        rowcount = self._apply_claim_governance_update(
            claim_id=claim_id,
            governance_transition="expired",
            source_event_id=source_event_id,
            updated_at=now,
            reason_codes=reason_codes,
            explicit_review_state=review_state,
        )
        self._store._conn.commit()
        if rowcount and event_context is not None:
            refreshed = self.get_claim(claim_id)
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_EXPIRED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": claim_id,
                    "scope_type": refreshed.scope_type if refreshed is not None else None,
                    "scope_id": refreshed.scope_id if refreshed is not None else None,
                    "reason_codes": list(
                        normalize_reason_codes(
                            reason_codes or (BrainGovernanceReasonCode.EXPIRED_BY_POLICY.value,)
                        )
                    ),
                    "summary": summary,
                    "notes": notes,
                },
            )
            self._set_last_governance_event_id(
                claim_id=claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        refreshed = self.get_claim(claim_id)
        if refreshed is None:
            raise RuntimeError("Failed to expire claim.")
        return refreshed

    def reclassify_claim_retention(
        self,
        claim_id: str,
        *,
        retention_class: str,
        source_event_id: str | None,
        reason_codes: Iterable[str] | None = None,
        summary: str | None = None,
        notes: str | None = None,
        updated_at: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainClaimRecord:
        """Reclassify explicit retention without bypassing the event spine."""
        now = updated_at or _utc_now()
        rowcount = self._apply_claim_governance_update(
            claim_id=claim_id,
            governance_transition="retention_reclassified",
            source_event_id=source_event_id,
            updated_at=now,
            reason_codes=reason_codes,
            explicit_retention_class=retention_class,
        )
        self._store._conn.commit()
        if rowcount and event_context is not None:
            refreshed = self.get_claim(claim_id)
            event = self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_CLAIM_RETENTION_RECLASSIFIED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "claim_id": claim_id,
                    "scope_type": refreshed.scope_type if refreshed is not None else None,
                    "scope_id": refreshed.scope_id if refreshed is not None else None,
                    "reason_codes": list(
                        normalize_reason_codes(reason_codes) if reason_codes is not None else []
                    ),
                    "summary": summary,
                    "notes": notes,
                    "retention_class": BrainClaimRetentionClass(
                        str(getattr(retention_class, "value", retention_class))
                    ).value,
                },
            )
            self._set_last_governance_event_id(
                claim_id=claim_id,
                event_id=event.event_id,
                updated_at=event.ts,
            )
            self._store._conn.commit()
        refreshed = self.get_claim(claim_id)
        if refreshed is None:
            raise RuntimeError("Failed to reclassify claim retention.")
        return refreshed

    def get_claim(self, claim_id: str) -> BrainClaimRecord | None:
        """Return one claim by id if present."""
        row = self._store._conn.execute(
            "SELECT * FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        return self._row_to_claim(row)

    def query_claims(
        self,
        *,
        temporal_mode: BrainClaimTemporalMode | str = BrainClaimTemporalMode.CURRENT,
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        entity_type: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        currentness_states: Iterable[str] | None = None,
        review_states: Iterable[str] | None = None,
        retention_classes: Iterable[str] | None = None,
        limit: int | None = 24,
    ) -> list[BrainClaimRecord]:
        """Return claims filtered by temporal mode and optional typed fields."""
        def _enum_value(item: Any) -> str:
            return str(getattr(item, "value", item))

        normalized_temporal_mode = _enum_value(temporal_mode)
        clauses = []
        params: list[Any] = []
        if subject_entity_id is not None:
            clauses.append("c.subject_entity_id = ?")
            params.append(subject_entity_id)
        if predicate is not None:
            clauses.append("c.predicate = ?")
            params.append(predicate)
        if scope_type is not None:
            clauses.append("c.scope_type = ?")
            params.append(scope_type)
        if scope_id is not None:
            clauses.append("c.scope_id = ?")
            params.append(scope_id)
        if entity_type is not None:
            clauses.append("s.entity_type = ?")
            params.append(entity_type)
        query = """
            SELECT c.*
            FROM claims c
            LEFT JOIN entities s ON s.entity_id = c.subject_entity_id
        """
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY c.updated_at DESC, c.created_at DESC"
        if (
            limit is not None
            and normalized_temporal_mode == BrainClaimTemporalMode.ALL.value
            and currentness_states is None
            and review_states is None
            and retention_classes is None
        ):
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        rows = self._store._conn.execute(query, tuple(params)).fetchall()

        normalized_currentness = (
            {BrainClaimCurrentnessStatus(_enum_value(value)).value for value in currentness_states}
            if currentness_states is not None
            else None
        )
        normalized_review = (
            {BrainClaimReviewState(_enum_value(value)).value for value in review_states}
            if review_states is not None
            else None
        )
        normalized_retention = (
            {BrainClaimRetentionClass(_enum_value(value)).value for value in retention_classes}
            if retention_classes is not None
            else None
        )
        filtered: list[BrainClaimRecord] = []
        now = _utc_now()
        for row in rows:
            claim = self._row_to_claim(row)
            if claim is None:
                continue
            if not claim_matches_temporal_mode(
                temporal_mode=normalized_temporal_mode,
                currentness_status=claim.currentness_status,
                truth_status=claim.status,
                valid_from=claim.valid_from,
                valid_to=claim.valid_to,
                stale_after_seconds=claim.stale_after_seconds,
                now=now,
            ):
                continue
            if (
                normalized_currentness is not None
                and claim.effective_currentness_status not in normalized_currentness
            ):
                continue
            if (
                normalized_review is not None
                and claim.effective_review_state not in normalized_review
            ):
                continue
            if (
                normalized_retention is not None
                and claim.effective_retention_class not in normalized_retention
            ):
                continue
            filtered.append(claim)
            if limit is not None and len(filtered) >= limit:
                break
        return filtered

    def claim_history(
        self,
        subject_entity_id: str,
        predicate: str | None = None,
        *,
        limit: int = 24,
    ) -> list[BrainClaimRecord]:
        """Return current and historical claims for one subject."""
        return self.query_claims(
            subject_entity_id=subject_entity_id,
            predicate=predicate,
            temporal_mode=BrainClaimTemporalMode.ALL,
            limit=limit,
        )

    def claim_evidence(self, claim_id: str) -> list[BrainClaimEvidenceRecord]:
        """Return evidence rows linked to one claim."""
        rows = self._store._conn.execute(
            """
            SELECT * FROM claim_evidence
            WHERE claim_id = ?
            ORDER BY created_at DESC
            """,
            (claim_id,),
        ).fetchall()
        return [
            BrainClaimEvidenceRecord(
                evidence_id=str(row["evidence_id"]),
                claim_id=str(row["claim_id"]),
                source_event_id=str(row["source_event_id"])
                if row["source_event_id"] is not None
                else None,
                source_episode_id=(
                    int(row["source_episode_id"]) if row["source_episode_id"] is not None else None
                ),
                evidence_summary=str(row["evidence_summary"]),
                evidence_json=str(row["evidence_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def claim_supersessions(
        self, claim_id: str | None = None
    ) -> list[BrainClaimSupersessionRecord]:
        """Return supersession links for all claims or one claim id."""
        clauses = []
        params: list[Any] = []
        if claim_id is not None:
            clauses.append("(prior_claim_id = ? OR new_claim_id = ?)")
            params.extend([claim_id, claim_id])
        query = "SELECT * FROM claim_supersessions"
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
        query += " ORDER BY created_at DESC"
        rows = self._store._conn.execute(query, tuple(params)).fetchall()
        return [
            BrainClaimSupersessionRecord(
                supersession_id=str(row["supersession_id"]),
                prior_claim_id=str(row["prior_claim_id"]),
                new_claim_id=str(row["new_claim_id"]),
                reason=str(row["reason"]),
                source_event_id=str(row["source_event_id"])
                if row["source_event_id"] is not None
                else None,
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def _row_to_claim(self, row) -> BrainClaimRecord | None:
        if row is None:
            return None
        reason_codes_json = row["governance_reason_codes_json"]
        return BrainClaimRecord(
            claim_id=str(row["claim_id"]),
            subject_entity_id=str(row["subject_entity_id"]),
            predicate=str(row["predicate"]),
            object_entity_id=str(row["object_entity_id"])
            if row["object_entity_id"] is not None
            else None,
            object_json=str(row["object_json"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            valid_from=str(row["valid_from"]),
            valid_to=str(row["valid_to"]) if row["valid_to"] is not None else None,
            source_event_id=str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None,
            scope_type=str(row["scope_type"]) if row["scope_type"] is not None else None,
            scope_id=str(row["scope_id"]) if row["scope_id"] is not None else None,
            claim_key=str(row["claim_key"]) if row["claim_key"] is not None else None,
            stale_after_seconds=(
                int(row["stale_after_seconds"]) if row["stale_after_seconds"] is not None else None
            ),
            currentness_status=(
                str(row["currentness_status"]) if row["currentness_status"] is not None else None
            ),
            review_state=str(row["review_state"]) if row["review_state"] is not None else None,
            retention_class=(
                str(row["retention_class"]) if row["retention_class"] is not None else None
            ),
            governance_reason_codes=normalize_reason_codes(
                json.loads(str(reason_codes_json)) if reason_codes_json not in (None, "") else []
            ),
            last_governance_event_id=(
                str(row["last_governance_event_id"])
                if row["last_governance_event_id"] is not None
                else None
            ),
            governance_updated_at=(
                str(row["governance_updated_at"])
                if row["governance_updated_at"] is not None
                else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _upsert_claim_fts(self, claim_id: str):
        if not getattr(self._store, "_fts_enabled", False):
            return
        self._store._conn.execute("DELETE FROM memory_claims_fts WHERE claim_id = ?", (claim_id,))
        row = self._store._conn.execute(
            """
            SELECT c.*, s.canonical_name AS subject_name, o.canonical_name AS object_name
            FROM claims c
            LEFT JOIN entities s ON s.entity_id = c.subject_entity_id
            LEFT JOIN entities o ON o.entity_id = c.object_entity_id
            WHERE c.claim_id = ?
            """,
            (claim_id,),
        ).fetchone()
        if row is None:
            return
        claim = self._row_to_claim(row)
        if claim is None:
            return
        summary = render_claim_summary(
            claim,
            subject_name=str(row["subject_name"]) if row["subject_name"] is not None else None,
            object_name=str(row["object_name"]) if row["object_name"] is not None else None,
        )
        object_text = (
            str(row["object_name"])
            if row["object_name"] is not None
            else str(claim.object.get("value", ""))
        )
        self._store._conn.execute(
            """
            INSERT INTO memory_claims_fts (
                claim_id, status, scope_type, scope_id, rendered_text, predicate, subject_name, object_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.claim_id,
                claim.status,
                claim.scope_type or "",
                claim.scope_id or "",
                summary,
                claim.predicate,
                str(row["subject_name"]) if row["subject_name"] is not None else "",
                object_text,
            ),
        )

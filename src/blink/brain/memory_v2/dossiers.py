"""Derived compiled continuity dossiers for Blink memory v2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventRecord
from blink.brain.memory_v2.autobiography import BrainAutobiographicalEntryRecord
from blink.brain.memory_v2.claims import (
    BrainClaimRecord,
    BrainClaimSupersessionRecord,
    render_claim_summary,
)
from blink.brain.memory_v2.core_blocks import BrainCoreMemoryBlockRecord
from blink.brain.memory_v2.graph import (
    BrainContinuityGraphEdgeKind,
    BrainContinuityGraphNodeKind,
    BrainContinuityGraphProjection,
)
from blink.brain.memory_v2.multimodal_autobiography import parse_multimodal_autobiography_record
from blink.brain.memory_v2.skills import BrainProceduralSkillProjection
from blink.brain.projections import (
    BrainAgendaProjection,
    BrainCommitmentProjection,
    BrainSceneWorldProjection,
)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _sorted_unique_texts(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _sorted_unique_episode_ids(values: Iterable[int | str | None]) -> list[int]:
    episode_ids: set[int] = set()
    for value in values:
        if value in (None, ""):
            continue
        try:
            episode_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(episode_ids)


def _event_sort_key(event: BrainEventRecord) -> tuple[int, datetime, str]:
    return (
        int(getattr(event, "id", 0)),
        _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
        event.event_id,
    )


class BrainContinuityDossierKind(str, Enum):
    """Supported compiled dossier kinds."""

    RELATIONSHIP = "relationship"
    PROJECT = "project"
    SELF_POLICY = "self_policy"
    USER = "user"
    COMMITMENT = "commitment"
    PLAN = "plan"
    PROCEDURAL = "procedural"
    SCENE_WORLD = "scene_world"


class BrainContinuityDossierFreshness(str, Enum):
    """Freshness states for a compiled dossier."""

    FRESH = "fresh"
    STALE = "stale"
    NEEDS_REFRESH = "needs_refresh"


class BrainContinuityDossierContradiction(str, Enum):
    """Contradiction states for a compiled dossier."""

    CLEAR = "clear"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"


class BrainContinuityDossierAvailability(str, Enum):
    """Availability states for one dossier under a specific task mode."""

    AVAILABLE = "available"
    ANNOTATED = "annotated"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class BrainContinuityDossierTaskAvailability:
    """Availability and reasons for one dossier/task pairing."""

    task: str
    availability: str
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the task-availability record."""
        return {
            "task": self.task,
            "availability": self.availability,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None
    ) -> "BrainContinuityDossierTaskAvailability | None":
        """Hydrate one task-availability record from JSON."""
        if not isinstance(data, dict):
            return None
        task = str(data.get("task", "")).strip()
        availability = str(data.get("availability", "")).strip()
        if not task or not availability:
            return None
        return cls(
            task=task,
            availability=availability,
            reason_codes=_sorted_unique_texts(data.get("reason_codes", [])),
        )


@dataclass(frozen=True)
class BrainContinuityDossierGovernanceRecord:
    """Explicit governance state for one compiled dossier."""

    supporting_claim_currentness_counts: dict[str, int] = field(default_factory=dict)
    supporting_claim_review_state_counts: dict[str, int] = field(default_factory=dict)
    supporting_claim_reason_codes: list[str] = field(default_factory=list)
    review_debt_count: int = 0
    last_refresh_cause: str = "no_fresh_support"
    task_availability: list[BrainContinuityDossierTaskAvailability] = field(default_factory=list)

    def availability_for_task(
        self, task: str
    ) -> BrainContinuityDossierTaskAvailability | None:
        """Return the availability record for one task if present."""
        normalized_task = str(task).strip()
        for record in self.task_availability:
            if record.task == normalized_task:
                return record
        return None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the dossier-governance record."""
        return {
            "supporting_claim_currentness_counts": dict(
                self.supporting_claim_currentness_counts
            ),
            "supporting_claim_review_state_counts": dict(
                self.supporting_claim_review_state_counts
            ),
            "supporting_claim_reason_codes": list(self.supporting_claim_reason_codes),
            "review_debt_count": int(self.review_debt_count),
            "last_refresh_cause": self.last_refresh_cause,
            "task_availability": [record.as_dict() for record in self.task_availability],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None
    ) -> "BrainContinuityDossierGovernanceRecord":
        """Hydrate one dossier-governance record from JSON."""
        payload = data or {}
        return cls(
            supporting_claim_currentness_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("supporting_claim_currentness_counts", {})
                ).items()
            },
            supporting_claim_review_state_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("supporting_claim_review_state_counts", {})
                ).items()
            },
            supporting_claim_reason_codes=_sorted_unique_texts(
                payload.get("supporting_claim_reason_codes", [])
            ),
            review_debt_count=int(payload.get("review_debt_count") or 0),
            last_refresh_cause=str(payload.get("last_refresh_cause") or "no_fresh_support"),
            task_availability=[
                record
                for item in payload.get("task_availability", [])
                if (record := BrainContinuityDossierTaskAvailability.from_dict(item)) is not None
            ],
        )


@dataclass(frozen=True)
class BrainContinuityDossierEvidenceRef:
    """Explicit evidence pointers supporting one dossier artifact."""

    claim_ids: list[str] = field(default_factory=list)
    entry_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[int] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)
    graph_edge_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence reference."""
        return {
            "claim_ids": list(self.claim_ids),
            "entry_ids": list(self.entry_ids),
            "source_event_ids": list(self.source_event_ids),
            "source_episode_ids": list(self.source_episode_ids),
            "graph_node_ids": list(self.graph_node_ids),
            "graph_edge_ids": list(self.graph_edge_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityDossierEvidenceRef":
        """Hydrate one evidence reference from JSON."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            claim_ids=_sorted_unique_texts(data.get("claim_ids", [])),
            entry_ids=_sorted_unique_texts(data.get("entry_ids", [])),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            source_episode_ids=_sorted_unique_episode_ids(data.get("source_episode_ids", [])),
            graph_node_ids=_sorted_unique_texts(data.get("graph_node_ids", [])),
            graph_edge_ids=_sorted_unique_texts(data.get("graph_edge_ids", [])),
        )


@dataclass(frozen=True)
class BrainContinuityDossierFactRecord:
    """One current fact or recent change rendered for a dossier."""

    fact_id: str
    summary: str
    status: str
    valid_from: str | None = None
    valid_to: str | None = None
    evidence: BrainContinuityDossierEvidenceRef = field(
        default_factory=BrainContinuityDossierEvidenceRef
    )
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the fact record."""
        return {
            "fact_id": self.fact_id,
            "summary": self.summary,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "evidence": self.evidence.as_dict(),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityDossierFactRecord | None":
        """Hydrate one fact record from JSON."""
        if not isinstance(data, dict):
            return None
        fact_id = str(data.get("fact_id", "")).strip()
        summary = str(data.get("summary", "")).strip()
        status = str(data.get("status", "")).strip()
        if not fact_id or not summary or not status:
            return None
        return cls(
            fact_id=fact_id,
            summary=summary,
            status=status,
            valid_from=_optional_text(data.get("valid_from")),
            valid_to=_optional_text(data.get("valid_to")),
            evidence=BrainContinuityDossierEvidenceRef.from_dict(data.get("evidence")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainContinuityDossierIssueRecord:
    """One open dossier issue such as staleness or contradiction."""

    issue_id: str
    kind: str
    summary: str
    status: str
    evidence: BrainContinuityDossierEvidenceRef = field(
        default_factory=BrainContinuityDossierEvidenceRef
    )
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the issue record."""
        return {
            "issue_id": self.issue_id,
            "kind": self.kind,
            "summary": self.summary,
            "status": self.status,
            "evidence": self.evidence.as_dict(),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityDossierIssueRecord | None":
        """Hydrate one issue record from JSON."""
        if not isinstance(data, dict):
            return None
        issue_id = str(data.get("issue_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        status = str(data.get("status", "")).strip()
        if not issue_id or not kind or not summary or not status:
            return None
        return cls(
            issue_id=issue_id,
            kind=kind,
            summary=summary,
            status=status,
            evidence=BrainContinuityDossierEvidenceRef.from_dict(data.get("evidence")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainContinuityDossierRecord:
    """One deterministic compiled continuity dossier."""

    dossier_id: str
    kind: str
    scope_type: str
    scope_id: str
    title: str
    summary: str
    status: str
    freshness: str
    contradiction: str
    support_strength: float
    summary_evidence: BrainContinuityDossierEvidenceRef = field(
        default_factory=BrainContinuityDossierEvidenceRef
    )
    key_current_facts: list[BrainContinuityDossierFactRecord] = field(default_factory=list)
    recent_changes: list[BrainContinuityDossierFactRecord] = field(default_factory=list)
    open_issues: list[BrainContinuityDossierIssueRecord] = field(default_factory=list)
    source_entry_ids: list[str] = field(default_factory=list)
    source_claim_ids: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    source_commitment_ids: list[str] = field(default_factory=list)
    source_plan_proposal_ids: list[str] = field(default_factory=list)
    source_skill_ids: list[str] = field(default_factory=list)
    source_scene_entity_ids: list[str] = field(default_factory=list)
    source_scene_affordance_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    source_episode_ids: list[int] = field(default_factory=list)
    governance: BrainContinuityDossierGovernanceRecord = field(
        default_factory=BrainContinuityDossierGovernanceRecord
    )
    details: dict[str, Any] = field(default_factory=dict)
    project_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the dossier record."""
        return {
            "dossier_id": self.dossier_id,
            "kind": self.kind,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "project_key": self.project_key,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "freshness": self.freshness,
            "contradiction": self.contradiction,
            "support_strength": self.support_strength,
            "summary_evidence": self.summary_evidence.as_dict(),
            "key_current_facts": [record.as_dict() for record in self.key_current_facts],
            "recent_changes": [record.as_dict() for record in self.recent_changes],
            "open_issues": [record.as_dict() for record in self.open_issues],
            "source_entry_ids": list(self.source_entry_ids),
            "source_claim_ids": list(self.source_claim_ids),
            "source_block_ids": list(self.source_block_ids),
            "source_commitment_ids": list(self.source_commitment_ids),
            "source_plan_proposal_ids": list(self.source_plan_proposal_ids),
            "source_skill_ids": list(self.source_skill_ids),
            "source_scene_entity_ids": list(self.source_scene_entity_ids),
            "source_scene_affordance_ids": list(self.source_scene_affordance_ids),
            "source_event_ids": list(self.source_event_ids),
            "source_episode_ids": list(self.source_episode_ids),
            "governance": self.governance.as_dict(),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityDossierRecord | None":
        """Hydrate one dossier record from JSON."""
        if not isinstance(data, dict):
            return None
        dossier_id = str(data.get("dossier_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        title = str(data.get("title", "")).strip()
        summary = str(data.get("summary", "")).strip()
        status = str(data.get("status", "")).strip()
        freshness = str(data.get("freshness", "")).strip()
        contradiction = str(data.get("contradiction", "")).strip()
        if not dossier_id or not kind or not scope_type or not scope_id:
            return None
        return cls(
            dossier_id=dossier_id,
            kind=kind,
            scope_type=scope_type,
            scope_id=scope_id,
            project_key=_optional_text(data.get("project_key")),
            title=title,
            summary=summary,
            status=status,
            freshness=freshness,
            contradiction=contradiction,
            support_strength=float(data.get("support_strength", 0.0)),
            summary_evidence=BrainContinuityDossierEvidenceRef.from_dict(
                data.get("summary_evidence")
            ),
            key_current_facts=[
                record
                for item in data.get("key_current_facts", [])
                if (record := BrainContinuityDossierFactRecord.from_dict(item)) is not None
            ],
            recent_changes=[
                record
                for item in data.get("recent_changes", [])
                if (record := BrainContinuityDossierFactRecord.from_dict(item)) is not None
            ],
            open_issues=[
                record
                for item in data.get("open_issues", [])
                if (record := BrainContinuityDossierIssueRecord.from_dict(item)) is not None
            ],
            source_entry_ids=_sorted_unique_texts(data.get("source_entry_ids", [])),
            source_claim_ids=_sorted_unique_texts(data.get("source_claim_ids", [])),
            source_block_ids=_sorted_unique_texts(data.get("source_block_ids", [])),
            source_commitment_ids=_sorted_unique_texts(data.get("source_commitment_ids", [])),
            source_plan_proposal_ids=_sorted_unique_texts(
                data.get("source_plan_proposal_ids", [])
            ),
            source_skill_ids=_sorted_unique_texts(data.get("source_skill_ids", [])),
            source_scene_entity_ids=_sorted_unique_texts(
                data.get("source_scene_entity_ids", [])
            ),
            source_scene_affordance_ids=_sorted_unique_texts(
                data.get("source_scene_affordance_ids", [])
            ),
            source_event_ids=_sorted_unique_texts(data.get("source_event_ids", [])),
            source_episode_ids=_sorted_unique_episode_ids(data.get("source_episode_ids", [])),
            governance=BrainContinuityDossierGovernanceRecord.from_dict(data.get("governance")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainContinuityDossierProjection:
    """Compiled dossier view above raw continuity evidence."""

    scope_type: str
    scope_id: str
    dossiers: list[BrainContinuityDossierRecord] = field(default_factory=list)
    dossier_counts: dict[str, int] = field(default_factory=dict)
    freshness_counts: dict[str, int] = field(default_factory=dict)
    contradiction_counts: dict[str, int] = field(default_factory=dict)
    current_dossier_ids: list[str] = field(default_factory=list)
    stale_dossier_ids: list[str] = field(default_factory=list)
    needs_refresh_dossier_ids: list[str] = field(default_factory=list)
    uncertain_dossier_ids: list[str] = field(default_factory=list)
    contradicted_dossier_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the dossier projection."""
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "dossiers": [record.as_dict() for record in self.dossiers],
            "dossier_counts": dict(self.dossier_counts),
            "freshness_counts": dict(self.freshness_counts),
            "contradiction_counts": dict(self.contradiction_counts),
            "current_dossier_ids": list(self.current_dossier_ids),
            "stale_dossier_ids": list(self.stale_dossier_ids),
            "needs_refresh_dossier_ids": list(self.needs_refresh_dossier_ids),
            "uncertain_dossier_ids": list(self.uncertain_dossier_ids),
            "contradicted_dossier_ids": list(self.contradicted_dossier_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainContinuityDossierProjection | None":
        """Hydrate one dossier projection from JSON."""
        if not isinstance(data, dict):
            return None
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        if not scope_type or not scope_id:
            return None
        dossiers = [
            record
            for item in data.get("dossiers", [])
            if (record := BrainContinuityDossierRecord.from_dict(item)) is not None
        ]
        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            dossiers=sorted(dossiers, key=_dossier_sort_key),
            dossier_counts={
                str(key): int(value) for key, value in dict(data.get("dossier_counts", {})).items()
            },
            freshness_counts={
                str(key): int(value)
                for key, value in dict(data.get("freshness_counts", {})).items()
            },
            contradiction_counts={
                str(key): int(value)
                for key, value in dict(data.get("contradiction_counts", {})).items()
            },
            current_dossier_ids=_sorted_unique_texts(data.get("current_dossier_ids", [])),
            stale_dossier_ids=_sorted_unique_texts(data.get("stale_dossier_ids", [])),
            needs_refresh_dossier_ids=_sorted_unique_texts(
                data.get("needs_refresh_dossier_ids", [])
            ),
            uncertain_dossier_ids=_sorted_unique_texts(data.get("uncertain_dossier_ids", [])),
            contradicted_dossier_ids=_sorted_unique_texts(data.get("contradicted_dossier_ids", [])),
        )


def _entry_sort_key(entry: BrainAutobiographicalEntryRecord) -> tuple[datetime, str]:
    return (_parse_ts(entry.updated_at) or datetime.min.replace(tzinfo=UTC), entry.entry_id)


def _claim_sort_key(claim: BrainClaimRecord) -> tuple[datetime, str]:
    return (_parse_ts(claim.updated_at) or datetime.min.replace(tzinfo=UTC), claim.claim_id)


def _fact_sort_key(record: BrainContinuityDossierFactRecord) -> tuple[datetime, str]:
    return (_parse_ts(record.valid_from) or datetime.min.replace(tzinfo=UTC), record.fact_id)


def _issue_sort_key(record: BrainContinuityDossierIssueRecord) -> tuple[str, str]:
    return (record.kind, record.issue_id)


def _dossier_sort_key(record: BrainContinuityDossierRecord) -> tuple[str, str, str]:
    return (record.kind, record.project_key or "", record.dossier_id)


_DOSSIER_TASKS = ("reply", "planning", "recall", "reflection", "critique")
_LAST_REFRESH_CAUSE_BY_FRESHNESS = {
    BrainContinuityDossierFreshness.FRESH.value: "fresh_current_support",
    BrainContinuityDossierFreshness.NEEDS_REFRESH.value: "newer_support_exists",
    BrainContinuityDossierFreshness.STALE.value: "no_fresh_support",
}


def _count_texts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _dossier_governance_reason_codes(
    *,
    dossier: BrainContinuityDossierRecord | None = None,
    freshness: str,
    contradiction: str,
    review_debt_count: int,
    held_claim_ids: Iterable[str],
    empty_summary: bool = False,
) -> list[str]:
    reason_codes: list[str] = []
    if freshness == BrainContinuityDossierFreshness.STALE.value:
        reason_codes.append("stale_support")
    elif freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
        reason_codes.append("needs_refresh")
    if contradiction == BrainContinuityDossierContradiction.UNCERTAIN.value:
        reason_codes.append("uncertain_support")
    elif contradiction == BrainContinuityDossierContradiction.CONTRADICTED.value:
        reason_codes.append("contradicted_support")
    if review_debt_count > 0:
        reason_codes.append("review_debt")
    if list(held_claim_ids):
        reason_codes.append("held_support")
    if empty_summary:
        reason_codes.append("empty_dossier")
    return _sorted_unique_texts(reason_codes)


def _dossier_task_availability(
    *,
    task: str,
    freshness: str,
    contradiction: str,
    review_debt_count: int,
    held_claim_ids: Iterable[str],
    empty_summary: bool,
) -> BrainContinuityDossierTaskAvailability:
    reason_codes = _dossier_governance_reason_codes(
        freshness=freshness,
        contradiction=contradiction,
        review_debt_count=review_debt_count,
        held_claim_ids=held_claim_ids,
        empty_summary=empty_summary,
    )
    if task == "reply":
        availability = (
            BrainContinuityDossierAvailability.AVAILABLE.value
            if (
                freshness == BrainContinuityDossierFreshness.FRESH.value
                and contradiction == BrainContinuityDossierContradiction.CLEAR.value
                and review_debt_count == 0
            )
            else BrainContinuityDossierAvailability.SUPPRESSED.value
        )
    elif task in {"planning", "critique"}:
        if (
            contradiction == BrainContinuityDossierContradiction.CONTRADICTED.value
            or review_debt_count > 0
            or list(held_claim_ids)
        ):
            availability = BrainContinuityDossierAvailability.SUPPRESSED.value
        elif freshness != BrainContinuityDossierFreshness.FRESH.value or (
            contradiction == BrainContinuityDossierContradiction.UNCERTAIN.value
        ):
            availability = BrainContinuityDossierAvailability.ANNOTATED.value
        else:
            availability = BrainContinuityDossierAvailability.AVAILABLE.value
    else:
        if empty_summary:
            availability = BrainContinuityDossierAvailability.SUPPRESSED.value
        elif (
            freshness == BrainContinuityDossierFreshness.FRESH.value
            and contradiction == BrainContinuityDossierContradiction.CLEAR.value
            and review_debt_count == 0
        ):
            availability = BrainContinuityDossierAvailability.AVAILABLE.value
        else:
            availability = BrainContinuityDossierAvailability.ANNOTATED.value
    return BrainContinuityDossierTaskAvailability(
        task=task,
        availability=availability,
        reason_codes=reason_codes,
    )


def _compile_dossier_governance(
    *,
    supporting_claims: Iterable[BrainClaimRecord],
    freshness: str,
    contradiction: str,
    summary: str,
) -> BrainContinuityDossierGovernanceRecord:
    supporting_claim_list = list(supporting_claims)
    held_claim_ids = _sorted_unique_texts(
        claim.claim_id for claim in supporting_claim_list if claim.is_held
    )
    review_debt_claim_ids = _sorted_unique_texts(
        claim.claim_id
        for claim in supporting_claim_list
        if claim.is_held or claim.effective_review_state == "requested"
    )
    return BrainContinuityDossierGovernanceRecord(
        supporting_claim_currentness_counts=_count_texts(
            claim.effective_currentness_status for claim in supporting_claim_list
        ),
        supporting_claim_review_state_counts=_count_texts(
            claim.effective_review_state for claim in supporting_claim_list
        ),
        supporting_claim_reason_codes=_sorted_unique_texts(
            reason_code
            for claim in supporting_claim_list
            for reason_code in claim.governance_reason_codes
        ),
        review_debt_count=len(review_debt_claim_ids),
        last_refresh_cause=_LAST_REFRESH_CAUSE_BY_FRESHNESS.get(
            freshness, "no_fresh_support"
        ),
        task_availability=[
            _dossier_task_availability(
                task=task,
                freshness=freshness,
                contradiction=contradiction,
                review_debt_count=len(review_debt_claim_ids),
                held_claim_ids=held_claim_ids,
                empty_summary=not bool(str(summary).strip()),
            )
            for task in _DOSSIER_TASKS
        ],
    )


@dataclass(frozen=True)
class _GraphIndexes:
    node_by_id: dict[str, dict[str, Any]]
    node_id_by_kind_backing: dict[tuple[str, str], str]
    edge_by_id: dict[str, dict[str, Any]]
    edge_ids_by_node_id: dict[str, set[str]]


def _build_graph_indexes(continuity_graph: BrainContinuityGraphProjection) -> _GraphIndexes:
    node_by_id: dict[str, dict[str, Any]] = {}
    node_id_by_kind_backing: dict[tuple[str, str], str] = {}
    edge_by_id: dict[str, dict[str, Any]] = {}
    edge_ids_by_node_id: dict[str, set[str]] = {}
    graph_payload = continuity_graph.as_dict()
    for node in graph_payload.get("nodes", []):
        node_id = str(node.get("node_id", "")).strip()
        kind = str(node.get("kind", "")).strip()
        backing_record_id = str(node.get("backing_record_id", "")).strip()
        if not node_id or not kind or not backing_record_id:
            continue
        node_by_id[node_id] = node
        node_id_by_kind_backing[(kind, backing_record_id)] = node_id
    for edge in graph_payload.get("edges", []):
        edge_id = str(edge.get("edge_id", "")).strip()
        from_node_id = str(edge.get("from_node_id", "")).strip()
        to_node_id = str(edge.get("to_node_id", "")).strip()
        if not edge_id or not from_node_id or not to_node_id:
            continue
        edge_by_id[edge_id] = edge
        edge_ids_by_node_id.setdefault(from_node_id, set()).add(edge_id)
        edge_ids_by_node_id.setdefault(to_node_id, set()).add(edge_id)
    return _GraphIndexes(
        node_by_id=node_by_id,
        node_id_by_kind_backing=node_id_by_kind_backing,
        edge_by_id=edge_by_id,
        edge_ids_by_node_id=edge_ids_by_node_id,
    )


def _collect_claim_source_refs(
    claim_id: str,
    *,
    graph_indexes: _GraphIndexes,
) -> tuple[list[str], list[int]]:
    event_ids: set[str] = set()
    episode_ids: set[int] = set()
    node_id = graph_indexes.node_id_by_kind_backing.get(
        (BrainContinuityGraphNodeKind.CLAIM.value, claim_id)
    )
    if node_id is None:
        return [], []
    node = graph_indexes.node_by_id.get(node_id, {})
    event_ids.update(_sorted_unique_texts(node.get("source_event_ids", [])))
    episode_ids.update(_sorted_unique_episode_ids(node.get("source_episode_ids", [])))
    for edge_id in graph_indexes.edge_ids_by_node_id.get(node_id, set()):
        edge = graph_indexes.edge_by_id.get(edge_id, {})
        event_ids.update(_sorted_unique_texts(edge.get("source_event_ids", [])))
        episode_ids.update(_sorted_unique_episode_ids(edge.get("source_episode_ids", [])))
    return sorted(event_ids), sorted(episode_ids)


def _resolve_reference_dt(
    *,
    recent_events: Iterable[BrainEventRecord],
    current_claims: Iterable[BrainClaimRecord],
    historical_claims: Iterable[BrainClaimRecord],
    autobiography: Iterable[BrainAutobiographicalEntryRecord],
    core_blocks: Iterable[BrainCoreMemoryBlockRecord],
    commitment_projection: BrainCommitmentProjection | None,
    procedural_skills: BrainProceduralSkillProjection | None,
    scene_world_state: BrainSceneWorldProjection | None,
) -> datetime | None:
    candidates: list[datetime] = []
    for event in recent_events:
        if (parsed := _parse_ts(event.ts)) is not None:
            candidates.append(parsed)
    for claim in (*current_claims, *historical_claims):
        for value in (claim.updated_at, claim.valid_from, claim.valid_to):
            if (parsed := _parse_ts(value)) is not None:
                candidates.append(parsed)
    for entry in autobiography:
        for value in (entry.updated_at, entry.valid_from, entry.valid_to):
            if (parsed := _parse_ts(value)) is not None:
                candidates.append(parsed)
    for block in core_blocks:
        for value in (block.updated_at, block.created_at):
            if (parsed := _parse_ts(value)) is not None:
                candidates.append(parsed)
    if commitment_projection is not None:
        for record in (
            list(commitment_projection.active_commitments)
            + list(commitment_projection.deferred_commitments)
            + list(commitment_projection.blocked_commitments)
            + list(commitment_projection.recent_terminal_commitments)
        ):
            for value in (record.updated_at, record.created_at, record.completed_at):
                if (parsed := _parse_ts(value)) is not None:
                    candidates.append(parsed)
    if procedural_skills is not None:
        for skill in procedural_skills.skills:
            for value in (skill.updated_at, skill.created_at, skill.retired_at):
                if (parsed := _parse_ts(value)) is not None:
                    candidates.append(parsed)
    if scene_world_state is not None:
        if (parsed := _parse_ts(scene_world_state.updated_at)) is not None:
            candidates.append(parsed)
        for entity in scene_world_state.entities:
            for value in (entity.updated_at, entity.observed_at, entity.expires_at):
                if (parsed := _parse_ts(value)) is not None:
                    candidates.append(parsed)
        for affordance in scene_world_state.affordances:
            for value in (affordance.updated_at, affordance.observed_at, affordance.expires_at):
                if (parsed := _parse_ts(value)) is not None:
                    candidates.append(parsed)
    return max(candidates) if candidates else None


def _graph_node_id(
    graph_indexes: _GraphIndexes,
    *,
    kind: str,
    backing_record_id: str,
) -> str | None:
    return graph_indexes.node_id_by_kind_backing.get((kind, str(backing_record_id).strip()))


def _graph_node(
    graph_indexes: _GraphIndexes,
    *,
    kind: str,
    backing_record_id: str,
) -> dict[str, Any] | None:
    node_id = _graph_node_id(
        graph_indexes,
        kind=kind,
        backing_record_id=backing_record_id,
    )
    if node_id is None:
        return None
    return graph_indexes.node_by_id.get(node_id)


def _graph_edge_ids(
    graph_indexes: _GraphIndexes,
    *,
    node_id: str,
    kinds: Iterable[str] | None = None,
    other_node_id: str | None = None,
) -> list[str]:
    allowed = set(kinds or [])
    edge_ids: list[str] = []
    for edge_id in sorted(graph_indexes.edge_ids_by_node_id.get(node_id, set())):
        edge = graph_indexes.edge_by_id.get(edge_id, {})
        if allowed and str(edge.get("kind", "")).strip() not in allowed:
            continue
        if other_node_id is not None and {
            str(edge.get("from_node_id", "")).strip(),
            str(edge.get("to_node_id", "")).strip(),
        } != {node_id, other_node_id}:
            continue
        edge_ids.append(edge_id)
    return edge_ids


def _graph_node_source_refs(
    node_id: str,
    *,
    graph_indexes: _GraphIndexes,
) -> tuple[list[str], list[int]]:
    node = graph_indexes.node_by_id.get(node_id, {})
    event_ids = set(_sorted_unique_texts(node.get("source_event_ids", [])))
    episode_ids = set(_sorted_unique_episode_ids(node.get("source_episode_ids", [])))
    for edge_id in graph_indexes.edge_ids_by_node_id.get(node_id, set()):
        edge = graph_indexes.edge_by_id.get(edge_id, {})
        event_ids.update(_sorted_unique_texts(edge.get("source_event_ids", [])))
        episode_ids.update(_sorted_unique_episode_ids(edge.get("source_episode_ids", [])))
    return sorted(event_ids), sorted(episode_ids)


def _graph_node_anchor_at(
    node_id: str,
    *,
    graph_indexes: _GraphIndexes,
    event_ts_by_id: dict[str, datetime],
) -> datetime | None:
    node = graph_indexes.node_by_id.get(node_id, {})
    event_timestamps = [
        event_ts_by_id[event_id]
        for event_id in _sorted_unique_texts(node.get("source_event_ids", []))
        if event_id in event_ts_by_id
    ]
    if event_timestamps:
        return max(event_timestamps)
    return _parse_ts(node.get("valid_from")) or _parse_ts(node.get("valid_to"))


def _build_evidence_ref(
    *,
    graph_indexes: _GraphIndexes,
    claim_ids: Iterable[str] = (),
    entry_ids: Iterable[str] = (),
    source_event_ids: Iterable[str] = (),
    source_episode_ids: Iterable[int | str] = (),
    extra_graph_node_ids: Iterable[str] = (),
    extra_graph_edge_ids: Iterable[str] = (),
) -> BrainContinuityDossierEvidenceRef:
    claim_id_set = set(_sorted_unique_texts(claim_ids))
    entry_id_set = set(_sorted_unique_texts(entry_ids))
    event_id_set = set(_sorted_unique_texts(source_event_ids))
    episode_id_set = set(_sorted_unique_episode_ids(source_episode_ids))
    node_id_set = set(_sorted_unique_texts(extra_graph_node_ids))
    edge_id_set = set(_sorted_unique_texts(extra_graph_edge_ids))

    for claim_id in claim_id_set:
        node_id = graph_indexes.node_id_by_kind_backing.get(
            (BrainContinuityGraphNodeKind.CLAIM.value, claim_id)
        )
        if node_id is not None:
            node_id_set.add(node_id)
    for entry_id in entry_id_set:
        node_id = graph_indexes.node_id_by_kind_backing.get(
            (BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value, entry_id)
        )
        if node_id is not None:
            node_id_set.add(node_id)
    for event_id in event_id_set:
        node_id = graph_indexes.node_id_by_kind_backing.get(
            (BrainContinuityGraphNodeKind.EVENT_ANCHOR.value, event_id)
        )
        if node_id is not None:
            node_id_set.add(node_id)
    for episode_id in episode_id_set:
        node_id = graph_indexes.node_id_by_kind_backing.get(
            (BrainContinuityGraphNodeKind.EPISODE_ANCHOR.value, str(int(episode_id)))
        )
        if node_id is not None:
            node_id_set.add(node_id)

    for node_id in list(node_id_set):
        node = graph_indexes.node_by_id.get(node_id, {})
        event_id_set.update(_sorted_unique_texts(node.get("source_event_ids", [])))
        episode_id_set.update(_sorted_unique_episode_ids(node.get("source_episode_ids", [])))
        claim_id_set.update(_sorted_unique_texts(node.get("supporting_claim_ids", [])))
        for edge_id in graph_indexes.edge_ids_by_node_id.get(node_id, set()):
            edge_id_set.add(edge_id)

    for edge_id in list(edge_id_set):
        edge = graph_indexes.edge_by_id.get(edge_id, {})
        event_id_set.update(_sorted_unique_texts(edge.get("source_event_ids", [])))
        episode_id_set.update(_sorted_unique_episode_ids(edge.get("source_episode_ids", [])))
        claim_id_set.update(_sorted_unique_texts(edge.get("supporting_claim_ids", [])))

    return BrainContinuityDossierEvidenceRef(
        claim_ids=sorted(claim_id_set),
        entry_ids=sorted(entry_id_set),
        source_event_ids=sorted(event_id_set),
        source_episode_ids=sorted(episode_id_set),
        graph_node_ids=sorted(node_id_set),
        graph_edge_ids=sorted(edge_id_set),
    )


def _fact_from_graph_node(
    node: dict[str, Any],
    *,
    dossier_id: str,
    graph_indexes: _GraphIndexes,
    event_ts_by_id: dict[str, datetime],
    status: str | None = None,
    extra_graph_node_ids: Iterable[str] = (),
    extra_graph_edge_ids: Iterable[str] = (),
    details: dict[str, Any] | None = None,
) -> BrainContinuityDossierFactRecord:
    node_id = _optional_text(node.get("node_id")) or ""
    backing_record_id = _optional_text(node.get("backing_record_id")) or ""
    node_kind = _optional_text(node.get("kind")) or ""
    source_event_ids, source_episode_ids = _graph_node_source_refs(
        node_id,
        graph_indexes=graph_indexes,
    )
    anchored_event_times = [
        event_ts_by_id[event_id]
        for event_id in _sorted_unique_texts(node.get("source_event_ids", []))
        if event_id in event_ts_by_id
    ]
    anchor_at = max(anchored_event_times) if anchored_event_times else None
    return BrainContinuityDossierFactRecord(
        fact_id=_stable_id("dossier_fact", dossier_id, node_kind, backing_record_id),
        summary=_optional_text(node.get("summary")) or backing_record_id,
        status=status or _optional_text(node.get("status")) or "current",
        valid_from=anchor_at.isoformat() if anchor_at is not None else None,
        valid_to=_optional_text(node.get("valid_to")) if anchor_at is not None else None,
        evidence=_build_evidence_ref(
            graph_indexes=graph_indexes,
            source_event_ids=source_event_ids,
            source_episode_ids=source_episode_ids,
            extra_graph_node_ids=[node_id, *extra_graph_node_ids],
            extra_graph_edge_ids=extra_graph_edge_ids,
        ),
        details={
            "kind": node_kind,
            **dict(node.get("details", {})),
            **(details or {}),
        },
    )


def _fact_from_claim(
    claim: BrainClaimRecord,
    *,
    dossier_id: str,
    graph_indexes: _GraphIndexes,
    event_ts_by_id: dict[str, datetime],
    status: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainContinuityDossierFactRecord:
    source_event_ids, source_episode_ids = _collect_claim_source_refs(
        claim.claim_id,
        graph_indexes=graph_indexes,
    )
    anchor_at = _claim_anchor_at(claim, event_ts_by_id=event_ts_by_id)
    display_status = status
    if display_status is None:
        if claim.currentness_status is not None:
            display_status = claim.effective_currentness_status
        else:
            display_status = "stale" if claim.is_stale else claim.status
    return BrainContinuityDossierFactRecord(
        fact_id=_stable_id("dossier_fact", dossier_id, claim.claim_id),
        summary=render_claim_summary(claim),
        status=display_status,
        valid_from=anchor_at.isoformat() if anchor_at is not None else claim.valid_from,
        valid_to=claim.valid_to,
        evidence=_build_evidence_ref(
            graph_indexes=graph_indexes,
            claim_ids=[claim.claim_id],
            source_event_ids=source_event_ids,
            source_episode_ids=source_episode_ids,
        ),
        details={
            "kind": "claim",
            "predicate": claim.predicate,
            "claim_key": claim.claim_key,
            "confidence": claim.confidence,
            "object": claim.object,
            "stale": claim.is_stale,
            "truth_status": claim.status,
            "currentness_status": claim.effective_currentness_status,
            "review_state": claim.effective_review_state,
            "retention_class": claim.effective_retention_class,
            "reason_codes": list(claim.governance_reason_codes),
            **(details or {}),
        },
    )


def _fact_from_entry(
    entry: BrainAutobiographicalEntryRecord,
    *,
    dossier_id: str,
    graph_indexes: _GraphIndexes,
    event_ts_by_id: dict[str, datetime],
    valid_to_override: datetime | None = None,
    status: str | None = None,
    details: dict[str, Any] | None = None,
    extra_graph_node_ids: Iterable[str] = (),
    extra_graph_edge_ids: Iterable[str] = (),
) -> BrainContinuityDossierFactRecord:
    anchor_at = _entry_anchor_at(entry, event_ts_by_id=event_ts_by_id)
    return BrainContinuityDossierFactRecord(
        fact_id=_stable_id("dossier_fact", dossier_id, entry.entry_id),
        summary=entry.rendered_summary,
        status=status or entry.status,
        valid_from=anchor_at.isoformat() if anchor_at is not None else entry.valid_from,
        valid_to=valid_to_override.isoformat() if valid_to_override is not None else entry.valid_to,
        evidence=_build_evidence_ref(
            graph_indexes=graph_indexes,
            claim_ids=entry.source_claim_ids,
            entry_ids=[entry.entry_id],
            source_event_ids=entry.source_event_ids,
            source_episode_ids=entry.source_episode_ids,
            extra_graph_node_ids=extra_graph_node_ids,
            extra_graph_edge_ids=extra_graph_edge_ids,
        ),
        details={
            "kind": "autobiography",
            "entry_kind": entry.entry_kind,
            "salience": entry.salience,
            "content": entry.content,
            **(details or {}),
        },
    )


def _claim_slot_key(claim: BrainClaimRecord) -> str:
    if claim.claim_key:
        return claim.claim_key
    return "|".join(
        [
            claim.subject_entity_id,
            claim.predicate,
            claim.object_entity_id or "",
        ]
    )


def _claim_value_key(claim: BrainClaimRecord) -> str:
    return json.dumps(claim.object, ensure_ascii=False, sort_keys=True)


def _is_current_entry(
    entry: BrainAutobiographicalEntryRecord,
    *,
    reference_dt: datetime | None,
) -> bool:
    if entry.status != "current":
        return False
    valid_to = _parse_ts(entry.valid_to)
    if valid_to is None:
        return True
    if reference_dt is None:
        return False
    return valid_to > reference_dt


def _entry_anchor_at(
    entry: BrainAutobiographicalEntryRecord,
    *,
    event_ts_by_id: dict[str, datetime],
) -> datetime | None:
    event_timestamps = [
        event_ts_by_id[event_id]
        for event_id in entry.source_event_ids
        if event_id in event_ts_by_id
    ]
    if event_timestamps:
        return max(event_timestamps)
    return _parse_ts(entry.valid_from) or _parse_ts(entry.updated_at)


def _entry_recency(
    entry: BrainAutobiographicalEntryRecord,
    *,
    event_ts_by_id: dict[str, datetime],
) -> datetime:
    return _entry_anchor_at(entry, event_ts_by_id=event_ts_by_id) or datetime.min.replace(
        tzinfo=UTC
    )


def _claim_anchor_at(
    claim: BrainClaimRecord,
    *,
    event_ts_by_id: dict[str, datetime],
) -> datetime | None:
    if claim.source_event_id and claim.source_event_id in event_ts_by_id:
        return event_ts_by_id[claim.source_event_id]
    return _parse_ts(claim.valid_from) or _parse_ts(claim.updated_at)


def _claim_recency(
    claim: BrainClaimRecord,
    *,
    event_ts_by_id: dict[str, datetime],
) -> datetime:
    return _claim_anchor_at(claim, event_ts_by_id=event_ts_by_id) or datetime.min.replace(
        tzinfo=UTC
    )


def _support_strength(
    summary_evidence: BrainContinuityDossierEvidenceRef,
    *,
    key_current_facts: Iterable[BrainContinuityDossierFactRecord],
) -> float:
    count = (
        len(summary_evidence.claim_ids)
        + len(summary_evidence.entry_ids)
        + len(summary_evidence.source_event_ids)
        + len(summary_evidence.source_episode_ids)
        + sum(
            len(record.evidence.claim_ids) + len(record.evidence.entry_ids)
            for record in key_current_facts
        )
    )
    return round(min(1.0, count / 8.0), 3)


def _compile_freshness(
    *,
    summary_anchor_at: datetime | None,
    summary_is_current: bool,
    supporting_claims: list[BrainClaimRecord],
    supporting_entries: list[BrainAutobiographicalEntryRecord],
    event_ts_by_id: dict[str, datetime],
    reference_dt: datetime | None,
) -> str:
    current_claims = [claim for claim in supporting_claims if claim.is_current]
    fresh_claims = [claim for claim in current_claims if not claim.is_stale]
    current_entries = [
        entry for entry in supporting_entries if _is_current_entry(entry, reference_dt=reference_dt)
    ]
    has_current_support = bool(current_entries or current_claims)
    if summary_anchor_at is not None:
        newer_claim = any(
            (_claim_anchor_at(claim, event_ts_by_id=event_ts_by_id) or summary_anchor_at)
            > summary_anchor_at
            for claim in current_claims
        )
        newer_entry = any(
            (_entry_anchor_at(entry, event_ts_by_id=event_ts_by_id) or summary_anchor_at)
            > summary_anchor_at
            for entry in current_entries
        )
        if newer_claim or newer_entry or (not summary_is_current and has_current_support):
            return BrainContinuityDossierFreshness.NEEDS_REFRESH.value
    if current_entries or fresh_claims:
        return BrainContinuityDossierFreshness.FRESH.value
    return BrainContinuityDossierFreshness.STALE.value


def _compile_contradiction(
    *,
    current_claims: list[BrainClaimRecord],
) -> tuple[str, dict[str, list[str]]]:
    conflicts: dict[str, list[str]] = {}
    slots: dict[str, set[str]] = {}
    slot_claim_ids: dict[str, list[str]] = {}
    for claim in current_claims:
        slot = _claim_slot_key(claim)
        slots.setdefault(slot, set()).add(_claim_value_key(claim))
        slot_claim_ids.setdefault(slot, []).append(claim.claim_id)
    for slot, values in slots.items():
        if len(values) > 1:
            conflicts[slot] = sorted(slot_claim_ids.get(slot, []))
    if conflicts:
        return BrainContinuityDossierContradiction.CONTRADICTED.value, conflicts
    if any(claim.status == "uncertain" for claim in current_claims):
        return BrainContinuityDossierContradiction.UNCERTAIN.value, {}
    return BrainContinuityDossierContradiction.CLEAR.value, {}


def _render_claim_summary_fallback(claims: list[BrainClaimRecord]) -> str:
    if not claims:
        return "No continuity summary available."
    summaries = [render_claim_summary(claim) for claim in claims[:3]]
    return (
        "; ".join(summary for summary in summaries if summary) or "No continuity summary available."
    )


def _sort_recent_change_records(
    records: list[tuple[datetime, BrainContinuityDossierFactRecord]],
    *,
    limit: int,
) -> list[BrainContinuityDossierFactRecord]:
    ordered = [
        record
        for _, record in sorted(
            records,
            key=lambda item: (item[0], item[1].fact_id),
            reverse=True,
        )
    ]
    return ordered[:limit]


def _iter_dossier_evidence_refs(
    *,
    summary_evidence: BrainContinuityDossierEvidenceRef,
    key_current_facts: Iterable[BrainContinuityDossierFactRecord],
    recent_changes: Iterable[BrainContinuityDossierFactRecord],
    open_issues: Iterable[BrainContinuityDossierIssueRecord],
) -> Iterable[BrainContinuityDossierEvidenceRef]:
    yield summary_evidence
    for record in key_current_facts:
        yield record.evidence
    for record in recent_changes:
        yield record.evidence
    for record in open_issues:
        yield record.evidence


def _collect_source_event_ids(
    *,
    summary_evidence: BrainContinuityDossierEvidenceRef,
    key_current_facts: Iterable[BrainContinuityDossierFactRecord],
    recent_changes: Iterable[BrainContinuityDossierFactRecord],
    open_issues: Iterable[BrainContinuityDossierIssueRecord],
) -> list[str]:
    return _sorted_unique_texts(
        event_id
        for evidence in _iter_dossier_evidence_refs(
            summary_evidence=summary_evidence,
            key_current_facts=key_current_facts,
            recent_changes=recent_changes,
            open_issues=open_issues,
        )
        for event_id in evidence.source_event_ids
    )


def _collect_source_episode_ids(
    *,
    summary_evidence: BrainContinuityDossierEvidenceRef,
    key_current_facts: Iterable[BrainContinuityDossierFactRecord],
    recent_changes: Iterable[BrainContinuityDossierFactRecord],
    open_issues: Iterable[BrainContinuityDossierIssueRecord],
) -> list[int]:
    return _sorted_unique_episode_ids(
        episode_id
        for evidence in _iter_dossier_evidence_refs(
            summary_evidence=summary_evidence,
            key_current_facts=key_current_facts,
            recent_changes=recent_changes,
            open_issues=open_issues,
        )
        for episode_id in evidence.source_episode_ids
    )


def build_continuity_dossier_projection(
    *,
    scope_type: str,
    scope_id: str,
    thread_id: str | None = None,
    current_claims: Iterable[BrainClaimRecord],
    historical_claims: Iterable[BrainClaimRecord],
    claim_supersessions: Iterable[BrainClaimSupersessionRecord],
    autobiography: Iterable[BrainAutobiographicalEntryRecord],
    continuity_graph: BrainContinuityGraphProjection,
    recent_events: Iterable[BrainEventRecord],
    core_blocks: Iterable[BrainCoreMemoryBlockRecord] = (),
    commitment_projection: BrainCommitmentProjection | None = None,
    agenda: BrainAgendaProjection | None = None,
    procedural_skills: BrainProceduralSkillProjection | None = None,
    scene_world_state: BrainSceneWorldProjection | None = None,
) -> BrainContinuityDossierProjection:
    """Compile deterministic continuity dossiers across durable runtime slices."""
    graph_indexes = _build_graph_indexes(continuity_graph)
    recent_event_list = sorted(list(recent_events), key=_event_sort_key)
    event_ts_by_id = {
        event.event_id: (_parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC))
        for event in recent_event_list
        if event.event_id
    }
    current_claim_list = sorted(list(current_claims), key=_claim_sort_key, reverse=True)
    historical_claim_list = sorted(list(historical_claims), key=_claim_sort_key, reverse=True)
    autobiography_list = sorted(list(autobiography), key=_entry_sort_key, reverse=True)
    core_block_list = sorted(
        list(core_blocks),
        key=lambda record: (
            record.block_kind,
            record.scope_type,
            record.scope_id,
            -int(record.version),
        ),
    )
    reference_dt = _resolve_reference_dt(
        recent_events=recent_event_list,
        current_claims=current_claim_list,
        historical_claims=historical_claim_list,
        autobiography=autobiography_list,
        core_blocks=core_block_list,
        commitment_projection=commitment_projection,
        procedural_skills=procedural_skills,
        scene_world_state=scene_world_state,
    )
    supersession_by_prior = {
        record.prior_claim_id: record for record in claim_supersessions if record.prior_claim_id
    }
    superseding_entry_anchor_by_prior = {
        entry.supersedes_entry_id: anchor_at
        for entry in autobiography_list
        if entry.supersedes_entry_id
        and (anchor_at := _entry_anchor_at(entry, event_ts_by_id=event_ts_by_id)) is not None
    }

    relationship_scope_ids = {
        entry.scope_id
        for entry in autobiography_list
        if entry.scope_type == "relationship" and entry.scope_id
    } | {
        block.scope_id
        for block in core_block_list
        if block.scope_type == "relationship" and block.scope_id
    }
    relationship_scope_id = (
        sorted(relationship_scope_ids)[0] if relationship_scope_ids else scope_id
    )

    relationship_current_entries = [
        entry
        for entry in autobiography_list
        if entry.scope_type == "relationship"
        and entry.scope_id == relationship_scope_id
        and entry.entry_kind
        in {
            "relationship_arc",
            "shared_history_summary",
        }
        and entry.status == "current"
    ]
    relationship_recent_change_entries = [
        entry
        for entry in autobiography_list
        if entry.scope_type == "relationship"
        and entry.scope_id == relationship_scope_id
        and (
            entry.entry_kind == "relationship_milestone"
            or (
                entry.entry_kind in {"relationship_arc", "shared_history_summary"}
                and entry.status == "superseded"
            )
        )
    ]
    relationship_summary_entry = next(
        (entry for entry in relationship_current_entries if entry.entry_kind == "relationship_arc"),
        None,
    ) or next(
        (
            entry
            for entry in relationship_current_entries
            if entry.entry_kind == "shared_history_summary"
        ),
        None,
    )
    relationship_summary = (
        relationship_summary_entry.rendered_summary
        if relationship_summary_entry is not None
        else _render_claim_summary_fallback(current_claim_list)
    )
    relationship_summary_evidence = (
        _build_evidence_ref(
            graph_indexes=graph_indexes,
            claim_ids=relationship_summary_entry.source_claim_ids,
            entry_ids=[relationship_summary_entry.entry_id],
            source_event_ids=relationship_summary_entry.source_event_ids,
            source_episode_ids=relationship_summary_entry.source_episode_ids,
        )
        if relationship_summary_entry is not None
        else _build_evidence_ref(
            graph_indexes=graph_indexes,
            claim_ids=[claim.claim_id for claim in current_claim_list[:3]],
        )
    )
    relationship_dossier_id = _stable_id(
        "dossier",
        BrainContinuityDossierKind.RELATIONSHIP.value,
        relationship_scope_id,
    )
    relationship_persona_blocks = [
        block
        for block in core_block_list
        if block.scope_type == "relationship"
        and block.scope_id == relationship_scope_id
        and block.status == "current"
        and block.block_kind in {"relationship_style", "teaching_profile"}
    ]
    relationship_current_facts = [
        _fact_from_entry(
            entry,
            dossier_id=relationship_dossier_id,
            graph_indexes=graph_indexes,
            event_ts_by_id=event_ts_by_id,
        )
        for entry in relationship_current_entries
    ] + [
        _fact_from_claim(
            claim,
            dossier_id=relationship_dossier_id,
            graph_indexes=graph_indexes,
            event_ts_by_id=event_ts_by_id,
        )
        for claim in current_claim_list
    ]
    relationship_current_facts.extend(
        _fact_from_graph_node(
            block_node,
            dossier_id=relationship_dossier_id,
            graph_indexes=graph_indexes,
            event_ts_by_id=event_ts_by_id,
            details={"recent_change_kind": "current_block"},
        )
        for block_node in (
            _graph_node(
                graph_indexes,
                kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                backing_record_id=block.block_id,
            )
            for block in relationship_persona_blocks
        )
        if block_node is not None
    )
    relationship_current_facts = sorted(
        relationship_current_facts,
        key=_fact_sort_key,
        reverse=True,
    )[:8]
    relationship_recent_changes: list[tuple[datetime, BrainContinuityDossierFactRecord]] = []
    for claim in historical_claim_list:
        supersession = supersession_by_prior.get(claim.claim_id)
        if supersession is None:
            continue
        relationship_recent_changes.append(
            (
                _parse_ts(supersession.created_at)
                or _parse_ts(claim.updated_at)
                or datetime.min.replace(tzinfo=UTC),
                _fact_from_claim(
                    claim,
                    dossier_id=relationship_dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                    status="historical",
                    details={
                        "recent_change_kind": "superseded_claim",
                        "supersession_reason": supersession.reason,
                        "replacement_claim_id": supersession.new_claim_id,
                    },
                ),
            )
        )
    for entry in relationship_recent_change_entries:
        relationship_recent_changes.append(
            (
                _entry_recency(entry, event_ts_by_id=event_ts_by_id),
                _fact_from_entry(
                    entry,
                    dossier_id=relationship_dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                    valid_to_override=superseding_entry_anchor_by_prior.get(entry.entry_id),
                    status="historical" if entry.status == "superseded" else "current",
                    details={"recent_change_kind": entry.entry_kind},
                ),
            )
        )
    relationship_recent_change_records = _sort_recent_change_records(
        relationship_recent_changes,
        limit=6,
    )
    relationship_contradiction, relationship_conflicts = _compile_contradiction(
        current_claims=current_claim_list,
    )
    relationship_freshness = _compile_freshness(
        summary_anchor_at=(
            _entry_anchor_at(relationship_summary_entry, event_ts_by_id=event_ts_by_id)
            if relationship_summary_entry is not None
            else max(
                (
                    _claim_anchor_at(claim, event_ts_by_id=event_ts_by_id)
                    for claim in current_claim_list[:1]
                ),
                default=None,
            )
        ),
        summary_is_current=relationship_summary_entry is None
        or relationship_summary_entry.status == "current",
        supporting_claims=current_claim_list,
        supporting_entries=relationship_current_entries,
        event_ts_by_id=event_ts_by_id,
        reference_dt=reference_dt,
    )
    relationship_supporting_claims = [*current_claim_list, *historical_claim_list]
    relationship_review_debt_claim_ids = _sorted_unique_texts(
        claim.claim_id
        for claim in relationship_supporting_claims
        if claim.is_held or claim.effective_review_state == "requested"
    )
    relationship_held_claim_ids = _sorted_unique_texts(
        claim.claim_id for claim in relationship_supporting_claims if claim.is_held
    )
    relationship_governance = _compile_dossier_governance(
        supporting_claims=relationship_supporting_claims,
        freshness=relationship_freshness,
        contradiction=relationship_contradiction,
        summary=relationship_summary,
    )
    relationship_issues: list[BrainContinuityDossierIssueRecord] = []
    if relationship_freshness == BrainContinuityDossierFreshness.STALE.value:
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id("dossier_issue", relationship_dossier_id, "stale_support"),
                kind="stale_support",
                summary="Current relationship support is stale or missing a fresh summary anchor.",
                status="open",
                evidence=relationship_summary_evidence,
                details={},
            )
        )
    if relationship_freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id("dossier_issue", relationship_dossier_id, "needs_refresh"),
                kind="needs_refresh",
                summary="Newer relationship evidence exists after the current summary anchor.",
                status="open",
                evidence=relationship_summary_evidence,
                details={},
            )
        )
    uncertain_claim_ids = [
        claim.claim_id for claim in current_claim_list if claim.status == "uncertain"
    ]
    if (
        uncertain_claim_ids
        and relationship_contradiction != BrainContinuityDossierContradiction.CONTRADICTED.value
    ):
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id("dossier_issue", relationship_dossier_id, "uncertain_claim"),
                kind="uncertain_claim",
                summary="Current relationship support includes uncertain claims.",
                status="open",
                evidence=_build_evidence_ref(
                    graph_indexes=graph_indexes,
                    claim_ids=uncertain_claim_ids,
                ),
                details={"claim_ids": uncertain_claim_ids},
            )
        )
    if relationship_conflicts:
        conflict_claim_ids = sorted(
            {claim_id for ids in relationship_conflicts.values() for claim_id in ids}
        )
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id(
                    "dossier_issue",
                    relationship_dossier_id,
                    "conflicting_current_claims",
                ),
                kind="conflicting_current_claims",
                summary="Multiple current claims conflict within the same relationship fact slot.",
                status="open",
                evidence=_build_evidence_ref(
                    graph_indexes=graph_indexes,
                    claim_ids=conflict_claim_ids,
                ),
                details={"conflicting_slots": relationship_conflicts},
            )
        )
    if relationship_review_debt_claim_ids:
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id("dossier_issue", relationship_dossier_id, "review_debt"),
                kind="review_debt",
                summary="Current relationship support requires review before it should be treated as clean continuity.",
                status="open",
                evidence=_build_evidence_ref(
                    graph_indexes=graph_indexes,
                    claim_ids=relationship_review_debt_claim_ids,
                ),
                details={
                    "claim_ids": relationship_review_debt_claim_ids,
                    "review_debt_count": len(relationship_review_debt_claim_ids),
                },
            )
        )
    if relationship_held_claim_ids:
        relationship_issues.append(
            BrainContinuityDossierIssueRecord(
                issue_id=_stable_id("dossier_issue", relationship_dossier_id, "held_support"),
                kind="held_support",
                summary="Current relationship support includes claims that are held for review.",
                status="open",
                evidence=_build_evidence_ref(
                    graph_indexes=graph_indexes,
                    claim_ids=relationship_held_claim_ids,
                ),
                details={"claim_ids": relationship_held_claim_ids},
            )
        )
    relationship_record = BrainContinuityDossierRecord(
        dossier_id=relationship_dossier_id,
        kind=BrainContinuityDossierKind.RELATIONSHIP.value,
        scope_type="relationship",
        scope_id=relationship_scope_id,
        title=f"Relationship with {relationship_scope_id.split(':')[-1]}",
        summary=relationship_summary,
        status="current" if relationship_current_entries or current_claim_list else "historical",
        freshness=relationship_freshness,
        contradiction=relationship_contradiction,
        support_strength=_support_strength(
            relationship_summary_evidence,
            key_current_facts=relationship_current_facts,
        ),
        summary_evidence=relationship_summary_evidence,
        key_current_facts=relationship_current_facts,
        recent_changes=relationship_recent_change_records,
        open_issues=sorted(relationship_issues, key=_issue_sort_key),
        source_entry_ids=_sorted_unique_texts(entry.entry_id for entry in autobiography_list),
        source_claim_ids=_sorted_unique_texts(
            claim.claim_id for claim in current_claim_list + historical_claim_list
        ),
        source_event_ids=_sorted_unique_texts(
            event_id
            for event_id in (
                *relationship_summary_evidence.source_event_ids,
                *(
                    event_id
                    for record in relationship_current_facts + relationship_recent_change_records
                    for event_id in record.evidence.source_event_ids
                ),
            )
        ),
        source_episode_ids=_sorted_unique_episode_ids(
            episode_id
            for episode_id in (
                *relationship_summary_evidence.source_episode_ids,
                *(
                    episode_id
                    for record in relationship_current_facts + relationship_recent_change_records
                    for episode_id in record.evidence.source_episode_ids
                ),
            )
        ),
        governance=relationship_governance,
        details={
            "summary_source_entry_id": (
                relationship_summary_entry.entry_id
                if relationship_summary_entry is not None
                else None
            ),
            "current_entry_kinds": sorted(
                {entry.entry_kind for entry in relationship_current_entries}
            ),
            "current_block_kinds": sorted(
                {block.block_kind for block in relationship_persona_blocks}
            ),
            "recent_change_count": len(relationship_recent_change_records),
        },
    )

    project_entries_by_key: dict[str, list[BrainAutobiographicalEntryRecord]] = {}
    for entry in autobiography_list:
        if entry.entry_kind != "project_arc":
            continue
        project_key = str(entry.content.get("project_key", "")).strip()
        if not project_key:
            continue
        project_entries_by_key.setdefault(project_key, []).append(entry)

    dossier_records = [relationship_record]
    for project_key in sorted(project_entries_by_key):
        project_entries = project_entries_by_key[project_key]
        current_project_entries = [entry for entry in project_entries if entry.status == "current"]
        summary_entry = (
            current_project_entries[0] if current_project_entries else project_entries[0]
        )
        project_dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.PROJECT.value,
            relationship_scope_id,
            project_key,
        )
        related_current_claims: list[BrainClaimRecord] = []
        related_historical_claims: list[BrainClaimRecord] = []
        entry_event_ids = set(summary_entry.source_event_ids)
        entry_episode_ids = set(summary_entry.source_episode_ids)
        for claim in current_claim_list:
            claim_event_ids, claim_episode_ids = _collect_claim_source_refs(
                claim.claim_id,
                graph_indexes=graph_indexes,
            )
            if entry_event_ids.intersection(claim_event_ids) or entry_episode_ids.intersection(
                claim_episode_ids
            ):
                related_current_claims.append(claim)
        for claim in historical_claim_list:
            claim_event_ids, claim_episode_ids = _collect_claim_source_refs(
                claim.claim_id,
                graph_indexes=graph_indexes,
            )
            if entry_event_ids.intersection(claim_event_ids) or entry_episode_ids.intersection(
                claim_episode_ids
            ):
                related_historical_claims.append(claim)
        project_summary_evidence = _build_evidence_ref(
            graph_indexes=graph_indexes,
            claim_ids=summary_entry.source_claim_ids,
            entry_ids=[summary_entry.entry_id],
            source_event_ids=summary_entry.source_event_ids,
            source_episode_ids=summary_entry.source_episode_ids,
        )
        project_current_facts = [
            _fact_from_entry(
                entry,
                dossier_id=project_dossier_id,
                graph_indexes=graph_indexes,
                event_ts_by_id=event_ts_by_id,
            )
            for entry in current_project_entries
        ] + [
            _fact_from_claim(
                claim,
                dossier_id=project_dossier_id,
                graph_indexes=graph_indexes,
                event_ts_by_id=event_ts_by_id,
            )
            for claim in related_current_claims
        ]
        project_current_facts = sorted(
            project_current_facts,
            key=_fact_sort_key,
            reverse=True,
        )[:6]
        project_recent_changes: list[tuple[datetime, BrainContinuityDossierFactRecord]] = []
        for claim in related_historical_claims:
            supersession = supersession_by_prior.get(claim.claim_id)
            if supersession is None:
                continue
            project_recent_changes.append(
                (
                    _parse_ts(supersession.created_at)
                    or _parse_ts(claim.updated_at)
                    or datetime.min.replace(tzinfo=UTC),
                    _fact_from_claim(
                        claim,
                        dossier_id=project_dossier_id,
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                        status="historical",
                        details={
                            "recent_change_kind": "superseded_claim",
                            "supersession_reason": supersession.reason,
                            "replacement_claim_id": supersession.new_claim_id,
                        },
                    ),
                )
            )
        project_recent_changes.extend(
            [
                (
                    _entry_recency(entry, event_ts_by_id=event_ts_by_id),
                    _fact_from_entry(
                        entry,
                        dossier_id=project_dossier_id,
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                        valid_to_override=superseding_entry_anchor_by_prior.get(entry.entry_id),
                        status="historical" if entry.status == "superseded" else entry.status,
                        details={"recent_change_kind": "project_arc"},
                    ),
                )
                for entry in project_entries
                if entry.status == "superseded"
            ]
        )
        project_recent_changes = _sort_recent_change_records(project_recent_changes, limit=4)
        project_contradiction, project_conflicts = _compile_contradiction(
            current_claims=related_current_claims,
        )
        project_freshness = _compile_freshness(
            summary_anchor_at=_entry_anchor_at(summary_entry, event_ts_by_id=event_ts_by_id),
            summary_is_current=summary_entry.status == "current",
            supporting_claims=related_current_claims,
            supporting_entries=current_project_entries,
            event_ts_by_id=event_ts_by_id,
            reference_dt=reference_dt,
        )
        project_supporting_claims = [*related_current_claims, *related_historical_claims]
        project_review_debt_claim_ids = _sorted_unique_texts(
            claim.claim_id
            for claim in project_supporting_claims
            if claim.is_held or claim.effective_review_state == "requested"
        )
        project_held_claim_ids = _sorted_unique_texts(
            claim.claim_id for claim in project_supporting_claims if claim.is_held
        )
        project_governance = _compile_dossier_governance(
            supporting_claims=project_supporting_claims,
            freshness=project_freshness,
            contradiction=project_contradiction,
            summary=summary_entry.rendered_summary,
        )
        project_issues: list[BrainContinuityDossierIssueRecord] = []
        if project_freshness == BrainContinuityDossierFreshness.STALE.value:
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id("dossier_issue", project_dossier_id, "stale_support"),
                    kind="stale_support",
                    summary="Current project support is stale or only historical.",
                    status="open",
                    evidence=project_summary_evidence,
                    details={},
                )
            )
        if project_freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id("dossier_issue", project_dossier_id, "needs_refresh"),
                    kind="needs_refresh",
                    summary="Newer project-linked evidence exists after the current project summary.",
                    status="open",
                    evidence=project_summary_evidence,
                    details={},
                )
            )
        project_uncertain_claim_ids = [
            claim.claim_id for claim in related_current_claims if claim.status == "uncertain"
        ]
        if (
            project_uncertain_claim_ids
            and project_contradiction != BrainContinuityDossierContradiction.CONTRADICTED.value
        ):
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id("dossier_issue", project_dossier_id, "uncertain_claim"),
                    kind="uncertain_claim",
                    summary="Current project-linked claims are uncertain.",
                    status="open",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=project_uncertain_claim_ids,
                    ),
                    details={"claim_ids": project_uncertain_claim_ids},
                )
            )
        if project_conflicts:
            project_conflict_claim_ids = sorted(
                {claim_id for ids in project_conflicts.values() for claim_id in ids}
            )
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id(
                        "dossier_issue",
                        project_dossier_id,
                        "conflicting_current_claims",
                    ),
                    kind="conflicting_current_claims",
                    summary="Current project-linked claims conflict within the same fact slot.",
                    status="open",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=project_conflict_claim_ids,
                    ),
                    details={"conflicting_slots": project_conflicts},
                )
            )
        if project_review_debt_claim_ids:
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id("dossier_issue", project_dossier_id, "review_debt"),
                    kind="review_debt",
                    summary="Current project support requires review before it should be treated as clean continuity.",
                    status="open",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=project_review_debt_claim_ids,
                    ),
                    details={
                        "claim_ids": project_review_debt_claim_ids,
                        "review_debt_count": len(project_review_debt_claim_ids),
                    },
                )
            )
        if project_held_claim_ids:
            project_issues.append(
                BrainContinuityDossierIssueRecord(
                    issue_id=_stable_id("dossier_issue", project_dossier_id, "held_support"),
                    kind="held_support",
                    summary="Current project support includes claims that are held for review.",
                    status="open",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=project_held_claim_ids,
                    ),
                    details={"claim_ids": project_held_claim_ids},
                )
            )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=project_dossier_id,
                kind=BrainContinuityDossierKind.PROJECT.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
                project_key=project_key,
                title=f"Project {project_key}",
                summary=summary_entry.rendered_summary,
                status="current"
                if current_project_entries or related_current_claims
                else "historical",
                freshness=project_freshness,
                contradiction=project_contradiction,
                support_strength=_support_strength(
                    project_summary_evidence,
                    key_current_facts=project_current_facts,
                ),
                summary_evidence=project_summary_evidence,
                key_current_facts=project_current_facts,
                recent_changes=project_recent_changes,
                open_issues=sorted(project_issues, key=_issue_sort_key),
                source_entry_ids=_sorted_unique_texts(entry.entry_id for entry in project_entries),
                source_claim_ids=_sorted_unique_texts(
                    claim.claim_id for claim in related_current_claims + related_historical_claims
                ),
                source_event_ids=_sorted_unique_texts(
                    event_id
                    for event_id in (
                        *project_summary_evidence.source_event_ids,
                        *(
                            event_id
                            for record in project_current_facts + project_recent_changes
                            for event_id in record.evidence.source_event_ids
                        ),
                    )
                ),
                source_episode_ids=_sorted_unique_episode_ids(
                    episode_id
                    for episode_id in (
                        *project_summary_evidence.source_episode_ids,
                        *(
                            episode_id
                            for record in project_current_facts + project_recent_changes
                            for episode_id in record.evidence.source_episode_ids
                        ),
                    )
                ),
                governance=project_governance,
                details={
                    "summary_source_entry_id": summary_entry.entry_id,
                    "project_entry_count": len(project_entries),
                    "current_project_entry_count": len(current_project_entries),
                },
            )
        )

    def _freshness_from_support(
        *,
        summary_anchor_at: datetime | None,
        current_support_anchors: Iterable[datetime | None],
        has_current_support: bool,
    ) -> str:
        anchors = [anchor for anchor in current_support_anchors if anchor is not None]
        if summary_anchor_at is not None and any(anchor > summary_anchor_at for anchor in anchors):
            return BrainContinuityDossierFreshness.NEEDS_REFRESH.value
        if has_current_support:
            return BrainContinuityDossierFreshness.FRESH.value
        return BrainContinuityDossierFreshness.STALE.value

    def _support_issue(
        *,
        dossier_id: str,
        kind: str,
        summary: str,
        evidence: BrainContinuityDossierEvidenceRef,
        details: dict[str, Any] | None = None,
    ) -> BrainContinuityDossierIssueRecord:
        return BrainContinuityDossierIssueRecord(
            issue_id=_stable_id("dossier_issue", dossier_id, kind),
            kind=kind,
            summary=summary,
            status="open",
            evidence=evidence,
            details=dict(details or {}),
        )

    core_blocks_by_scope_kind: dict[tuple[str, str, str], list[BrainCoreMemoryBlockRecord]] = {}
    for block in core_block_list:
        core_blocks_by_scope_kind.setdefault(
            (block.scope_type, block.scope_id, block.block_kind),
            [],
        ).append(block)
    for block_list in core_blocks_by_scope_kind.values():
        block_list.sort(key=lambda record: (-int(record.version), record.block_id))

    live_commitment_ids = {
        record.commitment_id
        for record in (
            (
                list(commitment_projection.active_commitments)
                + list(commitment_projection.deferred_commitments)
                + list(commitment_projection.blocked_commitments)
            )
            if commitment_projection is not None
            else []
        )
        if record.commitment_id
    }
    goal_ids = {
        goal.goal_id
        for goal in (agenda.goals if agenda is not None else [])
        if goal.goal_id
    }

    self_policy_block_lists = {
        scope_id_value: sorted(
            [
                block
                for (block_scope_type, scope_id_value_candidate, _), blocks in core_blocks_by_scope_kind.items()
                if block_scope_type == "agent"
                and scope_id_value_candidate == scope_id_value
                for block in blocks
                if block.block_kind in {"self_core", "self_persona_core", "self_current_arc"}
            ],
            key=lambda record: (
                0 if record.block_kind == "self_current_arc" else 1,
                0 if record.block_kind == "self_persona_core" else 1,
                0 if record.status == "current" else 1,
                -int(record.version),
                record.block_id,
            ),
        )
        for _, scope_id_value, _ in core_blocks_by_scope_kind
        if _ in {"self_core", "self_persona_core", "self_current_arc"}
    }
    seen_self_policy_scope_ids: set[str] = set()
    for agent_scope_id, block_versions in list(self_policy_block_lists.items()):
        if agent_scope_id in seen_self_policy_scope_ids:
            continue
        seen_self_policy_scope_ids.add(agent_scope_id)
        if not block_versions:
            continue
        current_blocks = [block for block in block_versions if block.status == "current"]
        historical_blocks = [block for block in block_versions if block.status != "current"]
        summary_block = next(
            (block for block in current_blocks if block.block_kind == "self_current_arc"),
            None,
        ) or next(
            (block for block in current_blocks if block.block_kind == "self_persona_core"),
            None,
        ) or next(
            (block for block in current_blocks if block.block_kind == "self_core"),
            None,
        ) or block_versions[0]
        summary_block_node = _graph_node(
            graph_indexes,
            kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
            backing_record_id=summary_block.block_id,
        )
        review_plan_nodes = sorted(
            [
                node
                for node in graph_indexes.node_by_id.values()
                if node.get("kind") == BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value
                and node.get("status") in {"proposed", "adopted"}
                and str(node.get("details", {}).get("review_policy", "")).strip()
                not in {"", "auto_adopt_ok"}
            ],
            key=lambda node: (
                _graph_node_anchor_at(
                    str(node.get("node_id", "")).strip(),
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                )
                or datetime.min.replace(tzinfo=UTC),
                str(node.get("backing_record_id", "")),
            ),
            reverse=True,
        )
        summary_text = (
            str(summary_block_node.get("summary", "")).strip()
            if summary_block_node is not None
            else ""
        )
        if not summary_text and review_plan_nodes:
            policy = str(review_plan_nodes[0].get("details", {}).get("review_policy", "")).strip()
            summary_text = f"{len(review_plan_nodes)} plans currently require {policy}"
        summary_evidence = (
            _build_evidence_ref(
                graph_indexes=graph_indexes,
                extra_graph_node_ids=[str(summary_block_node.get("node_id"))],
            )
            if summary_block_node is not None
            else _build_evidence_ref(
                graph_indexes=graph_indexes,
                extra_graph_node_ids=[
                    str(node.get("node_id", "")).strip() for node in review_plan_nodes[:2]
                ],
            )
        )
        dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.SELF_POLICY.value,
            agent_scope_id,
        )
        key_current_facts = [
            _fact_from_graph_node(
                summary_block_node,
                dossier_id=dossier_id,
                graph_indexes=graph_indexes,
                event_ts_by_id=event_ts_by_id,
                details={"recent_change_kind": "current_block"},
            )
            for summary_block_node in (
                _graph_node(
                    graph_indexes,
                    kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                    backing_record_id=block.block_id,
                )
                for block in current_blocks[:3]
            )
            if summary_block_node is not None
        ]
        key_current_facts.extend(
            _fact_from_graph_node(
                node,
                dossier_id=dossier_id,
                graph_indexes=graph_indexes,
                event_ts_by_id=event_ts_by_id,
                details={"review_policy_fact": True},
            )
            for node in review_plan_nodes[:4]
        )
        key_current_facts = sorted(key_current_facts, key=_fact_sort_key, reverse=True)[:6]
        recent_changes = _sort_recent_change_records(
            [
                (
                    _graph_node_anchor_at(
                        str(node.get("node_id", "")).strip(),
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                    )
                    or datetime.min.replace(tzinfo=UTC),
                    _fact_from_graph_node(
                        node,
                        dossier_id=dossier_id,
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                        status="historical",
                        details={"recent_change_kind": "superseded_block"},
                    ),
                )
                for node in (
                    _graph_node(
                        graph_indexes,
                        kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                        backing_record_id=block.block_id,
                    )
                    for block in historical_blocks
                )
                if node is not None
            ],
            limit=4,
        )
        freshness = _freshness_from_support(
            summary_anchor_at=(
                _graph_node_anchor_at(
                    str(summary_block_node.get("node_id", "")).strip(),
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                )
                if summary_block_node is not None
                else None
            ),
            current_support_anchors=[
                *(
                    _graph_node_anchor_at(
                        str(node.get("node_id", "")).strip(),
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                    )
                    for node in (
                        _graph_node(
                            graph_indexes,
                            kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                            backing_record_id=block.block_id,
                        )
                        for block in current_blocks
                    )
                    if node is not None
                ),
                *(
                    _graph_node_anchor_at(
                        str(node.get("node_id", "")).strip(),
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                    )
                    for node in review_plan_nodes
                ),
            ],
            has_current_support=bool(current_blocks or review_plan_nodes),
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if freshness == BrainContinuityDossierFreshness.STALE.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="stale_support",
                    summary="Self-policy continuity only has historical block support.",
                    evidence=summary_evidence,
                )
            )
        elif freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="needs_refresh",
                    summary="Newer self-policy support exists after the current summary anchor.",
                    evidence=summary_evidence,
                )
            )
        if review_plan_nodes:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="review_policy_required",
                    summary="Current planning state includes proposals that require explicit review.",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        extra_graph_node_ids=[
                            str(node.get("node_id", "")).strip() for node in review_plan_nodes[:3]
                        ],
                    ),
                    details={
                        "plan_proposal_ids": [
                            str(node.get("backing_record_id", "")).strip()
                            for node in review_plan_nodes[:3]
                        ]
                    },
                )
            )
        governance = _compile_dossier_governance(
            supporting_claims=[],
            freshness=freshness,
            contradiction=BrainContinuityDossierContradiction.CLEAR.value,
            summary=summary_text,
        )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.SELF_POLICY.value,
                scope_type="agent",
                scope_id=agent_scope_id,
                title="Self Policy",
                summary=summary_text or "No self-policy summary available.",
                status="current" if current_blocks or review_plan_nodes else "historical",
                freshness=freshness,
                contradiction=BrainContinuityDossierContradiction.CLEAR.value,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_block_ids=_sorted_unique_texts(block.block_id for block in block_versions),
                source_plan_proposal_ids=_sorted_unique_texts(
                    str(node.get("backing_record_id", "")).strip() for node in review_plan_nodes
                ),
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "summary_block_id": summary_block.block_id if summary_block is not None else None,
                    "current_block_kinds": sorted({block.block_kind for block in current_blocks}),
                    "review_plan_count": len(review_plan_nodes),
                },
            )
        )

    user_claims_current = [
        claim
        for claim in current_claim_list
        if claim.predicate.startswith("profile.") or claim.predicate.startswith("preference.")
    ]
    user_claims_historical = [
        claim
        for claim in historical_claim_list
        if claim.predicate.startswith("profile.") or claim.predicate.startswith("preference.")
    ]
    user_block_versions = core_blocks_by_scope_kind.get(("user", scope_id, "user_core"), [])
    if user_block_versions or user_claims_current or user_claims_historical:
        current_user_blocks = [block for block in user_block_versions if block.status == "current"]
        historical_user_blocks = [block for block in user_block_versions if block.status != "current"]
        summary_block = current_user_blocks[0] if current_user_blocks else (
            user_block_versions[0] if user_block_versions else None
        )
        summary_node = (
            _graph_node(
                graph_indexes,
                kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                backing_record_id=summary_block.block_id,
            )
            if summary_block is not None
            else None
        )
        summary_text = (
            str(summary_node.get("summary", "")).strip()
            if summary_node is not None
            else _render_claim_summary_fallback(user_claims_current)
        )
        summary_evidence = (
            _build_evidence_ref(
                graph_indexes=graph_indexes,
                extra_graph_node_ids=[str(summary_node.get("node_id", ""))],
            )
            if summary_node is not None
            else _build_evidence_ref(
                graph_indexes=graph_indexes,
                claim_ids=[claim.claim_id for claim in user_claims_current[:3]],
            )
        )
        dossier_id = _stable_id("dossier", BrainContinuityDossierKind.USER.value, scope_id)
        key_current_facts = [
            *[
                _fact_from_graph_node(
                    node,
                    dossier_id=dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                    details={"recent_change_kind": "current_block"},
                )
                for node in (
                    _graph_node(
                        graph_indexes,
                        kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                        backing_record_id=block.block_id,
                    )
                    for block in current_user_blocks[:1]
                )
                if node is not None
            ],
            *[
                _fact_from_claim(
                    claim,
                    dossier_id=dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                )
                for claim in user_claims_current
            ],
        ]
        key_current_facts = sorted(key_current_facts, key=_fact_sort_key, reverse=True)[:6]
        recent_change_records = [
            (
                _graph_node_anchor_at(
                    str(node.get("node_id", "")).strip(),
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                )
                or datetime.min.replace(tzinfo=UTC),
                _fact_from_graph_node(
                    node,
                    dossier_id=dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                    status="historical",
                    details={"recent_change_kind": "superseded_block"},
                ),
            )
            for node in (
                _graph_node(
                    graph_indexes,
                    kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                    backing_record_id=block.block_id,
                )
                for block in historical_user_blocks
            )
            if node is not None
        ]
        for claim in user_claims_historical:
            supersession = supersession_by_prior.get(claim.claim_id)
            recent_change_records.append(
                (
                    _parse_ts(supersession.created_at)
                    if supersession is not None
                    else _claim_recency(claim, event_ts_by_id=event_ts_by_id),
                    _fact_from_claim(
                        claim,
                        dossier_id=dossier_id,
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                        status="historical",
                        details={"recent_change_kind": "historical_claim"},
                    ),
                )
            )
        recent_changes = _sort_recent_change_records(recent_change_records, limit=4)
        contradiction, conflicts = _compile_contradiction(current_claims=user_claims_current)
        freshness = _freshness_from_support(
            summary_anchor_at=(
                _graph_node_anchor_at(
                    str(summary_node.get("node_id", "")).strip(),
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                )
                if summary_node is not None
                else max(
                    (
                        _claim_anchor_at(claim, event_ts_by_id=event_ts_by_id)
                        for claim in user_claims_current[:1]
                    ),
                    default=None,
                )
            ),
            current_support_anchors=[
                *(
                    _graph_node_anchor_at(
                        str(node.get("node_id", "")).strip(),
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                    )
                    for node in (
                        _graph_node(
                            graph_indexes,
                            kind=BrainContinuityGraphNodeKind.CORE_MEMORY_BLOCK.value,
                            backing_record_id=block.block_id,
                        )
                        for block in current_user_blocks
                    )
                    if node is not None
                ),
                *(
                    _claim_anchor_at(claim, event_ts_by_id=event_ts_by_id)
                    for claim in user_claims_current
                ),
            ],
            has_current_support=bool(current_user_blocks or user_claims_current),
        )
        governance = _compile_dossier_governance(
            supporting_claims=[*user_claims_current, *user_claims_historical],
            freshness=freshness,
            contradiction=contradiction,
            summary=summary_text,
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if freshness == BrainContinuityDossierFreshness.STALE.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="stale_support",
                    summary="User continuity only has historical support.",
                    evidence=summary_evidence,
                )
            )
        elif freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="needs_refresh",
                    summary="Newer user-support evidence exists after the current summary anchor.",
                    evidence=summary_evidence,
                )
            )
        if conflicts:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="conflicting_current_claims",
                    summary="Current user-profile facts conflict within the same slot.",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=[claim_id for ids in conflicts.values() for claim_id in ids],
                    ),
                    details={"conflicting_slots": conflicts},
                )
            )
        uncertain_user_claim_ids = [
            claim.claim_id for claim in user_claims_current if claim.status == "uncertain"
        ]
        if uncertain_user_claim_ids and contradiction != BrainContinuityDossierContradiction.CONTRADICTED.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="uncertain_claim",
                    summary="Current user-profile support includes uncertain claims.",
                    evidence=_build_evidence_ref(
                        graph_indexes=graph_indexes,
                        claim_ids=uncertain_user_claim_ids,
                    ),
                    details={"claim_ids": uncertain_user_claim_ids},
                )
            )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.USER.value,
                scope_type="user",
                scope_id=scope_id,
                title="User Profile",
                summary=summary_text or "No user profile summary available.",
                status="current" if current_user_blocks or user_claims_current else "historical",
                freshness=freshness,
                contradiction=contradiction,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_claim_ids=_sorted_unique_texts(
                    claim.claim_id for claim in (*user_claims_current, *user_claims_historical)
                ),
                source_block_ids=_sorted_unique_texts(block.block_id for block in user_block_versions),
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "summary_block_id": summary_block.block_id if summary_block is not None else None,
                    "current_claim_count": len(user_claims_current),
                },
            )
        )

    commitment_records = sorted(
        (
            (
                list(commitment_projection.active_commitments)
                + list(commitment_projection.deferred_commitments)
                + list(commitment_projection.blocked_commitments)
                + list(commitment_projection.recent_terminal_commitments)
            )
            if commitment_projection is not None
            else []
        ),
        key=lambda record: (
            record.status,
            record.updated_at,
            record.commitment_id,
        ),
    )
    commitment_node_ids = {
        record.commitment_id: _graph_node_id(
            graph_indexes,
            kind=BrainContinuityGraphNodeKind.COMMITMENT.value,
            backing_record_id=record.commitment_id,
        )
        for record in commitment_records
    }
    plan_node_by_id = {
        (_optional_text(node.get("backing_record_id")) or ""): node
        for node in graph_indexes.node_by_id.values()
        if node.get("kind") == BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value
    }
    plan_node_ids = {
        proposal_id: str(node.get("node_id", "")).strip()
        for proposal_id, node in plan_node_by_id.items()
        if proposal_id
    }
    plan_superseded_ids = {
        plan_id
        for plan_id, node_id in plan_node_ids.items()
        if node_id in continuity_graph.superseded_node_ids
    }

    def _node_anchor_for_payload(node: dict[str, Any] | None) -> datetime | None:
        if node is None:
            return None
        return _graph_node_anchor_at(
            str(node.get("node_id", "")).strip(),
            graph_indexes=graph_indexes,
            event_ts_by_id=event_ts_by_id,
        )

    def _node_fact(
        node: dict[str, Any],
        *,
        dossier_id: str,
        status: str | None = None,
        extra_graph_node_ids: Iterable[str] = (),
        extra_graph_edge_ids: Iterable[str] = (),
        details: dict[str, Any] | None = None,
    ) -> BrainContinuityDossierFactRecord:
        return _fact_from_graph_node(
            node,
            dossier_id=dossier_id,
            graph_indexes=graph_indexes,
            event_ts_by_id=event_ts_by_id,
            status=status,
            extra_graph_node_ids=extra_graph_node_ids,
            extra_graph_edge_ids=extra_graph_edge_ids,
            details=details,
        )

    for commitment in commitment_records:
        commitment_node_id = commitment_node_ids.get(commitment.commitment_id)
        if commitment_node_id is None:
            continue
        commitment_node = graph_indexes.node_by_id.get(commitment_node_id)
        if commitment_node is None:
            continue
        dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.COMMITMENT.value,
            commitment.commitment_id,
        )
        linked_plan_ids = _sorted_unique_texts(
            [
                *(
                    _optional_text(commitment.details.get(key))
                    for key in ("current_plan_proposal_id", "pending_plan_proposal_id")
                ),
                *(
                    (
                        graph_indexes.node_by_id.get(str(edge.get("to_node_id", "")).strip(), {}).get(
                            "backing_record_id"
                        )
                        if str(edge.get("from_node_id", "")).strip() == commitment_node_id
                        else graph_indexes.node_by_id.get(
                            str(edge.get("from_node_id", "")).strip(), {}
                        ).get("backing_record_id")
                    )
                    for edge_id in graph_indexes.edge_ids_by_node_id.get(commitment_node_id, set())
                    if (
                        edge := graph_indexes.edge_by_id.get(edge_id, {})
                    ).get("kind")
                    in {
                        BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                        BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                    }
                ),
            ]
        )
        linked_plan_nodes = [
            plan_node_by_id[plan_id]
            for plan_id in linked_plan_ids
            if plan_id in plan_node_by_id
        ]
        current_plan_nodes = [
            node
            for node in linked_plan_nodes
            if str(node.get("status", "")).strip() in {"proposed", "adopted"}
            and (_optional_text(node.get("backing_record_id")) or "") not in plan_superseded_ids
        ]
        summary_plan_node = next(
            (
                node
                for node in current_plan_nodes
                if str(node.get("status", "")).strip() == "adopted"
            ),
            None,
        ) or next(iter(current_plan_nodes), None)
        summary_text = (
            str(summary_plan_node.get("summary", "")).strip()
            if summary_plan_node is not None
            else str(commitment_node.get("summary", "")).strip()
        ) or commitment.title
        summary_edge_ids = (
            _graph_edge_ids(
                graph_indexes,
                node_id=commitment_node_id,
                kinds=(
                    BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                    BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                ),
                other_node_id=str(summary_plan_node.get("node_id", "")).strip(),
            )
            if summary_plan_node is not None
            else []
        )
        summary_evidence = _build_evidence_ref(
            graph_indexes=graph_indexes,
            extra_graph_node_ids=[
                commitment_node_id,
                *(
                    [str(summary_plan_node.get("node_id", "")).strip()]
                    if summary_plan_node is not None
                    else []
                ),
            ],
            extra_graph_edge_ids=summary_edge_ids,
        )
        key_current_facts = [
            _node_fact(
                commitment_node,
                dossier_id=dossier_id,
                details={"record_kind": "commitment"},
            ),
            *[
                _node_fact(
                    node,
                    dossier_id=dossier_id,
                    extra_graph_node_ids=[commitment_node_id],
                    extra_graph_edge_ids=_graph_edge_ids(
                        graph_indexes,
                        node_id=commitment_node_id,
                        kinds=(
                            BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                            BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                        ),
                        other_node_id=str(node.get("node_id", "")).strip(),
                    ),
                    details={"record_kind": "linked_plan"},
                )
                for node in current_plan_nodes[:5]
            ],
        ]
        key_current_facts = sorted(key_current_facts, key=_fact_sort_key, reverse=True)[:6]
        recent_change_records: list[tuple[datetime, BrainContinuityDossierFactRecord]] = []
        for node in linked_plan_nodes:
            plan_id = str(node.get("backing_record_id", "")).strip()
            if (
                plan_id not in plan_superseded_ids
                and str(node.get("status", "")).strip() != "rejected"
            ):
                continue
            recent_change_records.append(
                (
                    _node_anchor_for_payload(node) or datetime.min.replace(tzinfo=UTC),
                    _node_fact(
                        node,
                        dossier_id=dossier_id,
                        status="historical",
                        extra_graph_node_ids=[commitment_node_id],
                        extra_graph_edge_ids=_graph_edge_ids(
                            graph_indexes,
                            node_id=commitment_node_id,
                            kinds=(
                                BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                                BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                            ),
                            other_node_id=str(node.get("node_id", "")).strip(),
                        ),
                        details={"recent_change_kind": "plan_transition"},
                    ),
                )
            )
        if commitment.status in {"completed", "cancelled", "failed"}:
            recent_change_records.append(
                (
                    _node_anchor_for_payload(commitment_node)
                    or _parse_ts(commitment.completed_at)
                    or datetime.min.replace(tzinfo=UTC),
                    _node_fact(
                        commitment_node,
                        dossier_id=dossier_id,
                        status="historical",
                        details={"recent_change_kind": "terminal_commitment"},
                    ),
                )
            )
        recent_changes = _sort_recent_change_records(recent_change_records, limit=4)
        has_current_support = commitment.status not in {"completed", "cancelled", "failed"}
        freshness = _freshness_from_support(
            summary_anchor_at=_node_anchor_for_payload(summary_plan_node or commitment_node),
            current_support_anchors=[
                _node_anchor_for_payload(commitment_node),
                *(_node_anchor_for_payload(node) for node in current_plan_nodes),
            ],
            has_current_support=has_current_support,
        )
        contradiction = (
            BrainContinuityDossierContradiction.UNCERTAIN.value
            if commitment.status == "blocked"
            else BrainContinuityDossierContradiction.CLEAR.value
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if freshness == BrainContinuityDossierFreshness.STALE.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="stale_support",
                    summary="Commitment continuity only has terminal or historical support.",
                    evidence=summary_evidence,
                )
            )
        elif freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="needs_refresh",
                    summary="A newer linked plan exists after the current commitment summary anchor.",
                    evidence=summary_evidence,
                )
            )
        if commitment.status == "blocked":
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="blocked_commitment",
                    summary="This commitment is currently blocked.",
                    evidence=summary_evidence,
                    details={
                        "blocked_reason": (
                            commitment.blocked_reason.as_dict()
                            if commitment.blocked_reason is not None
                            else None
                        )
                    },
                )
            )
        governance = _compile_dossier_governance(
            supporting_claims=[],
            freshness=freshness,
            contradiction=contradiction,
            summary=summary_text,
        )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.COMMITMENT.value,
                scope_type=commitment.scope_type,
                scope_id=commitment.scope_id,
                title=commitment.title or commitment.commitment_id,
                summary=summary_text or commitment.title or commitment.commitment_id,
                status="current" if has_current_support else "historical",
                freshness=freshness,
                contradiction=contradiction,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_commitment_ids=[commitment.commitment_id],
                source_plan_proposal_ids=linked_plan_ids,
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "goal_family": commitment.goal_family,
                    "intent": commitment.intent,
                    "current_goal_id": commitment.current_goal_id,
                    "plan_revision": commitment.plan_revision,
                    "linked_plan_count": len(linked_plan_ids),
                },
            )
        )

    relevant_plan_ids = {
        proposal_id
        for proposal_id, node in plan_node_by_id.items()
        if str(node.get("status", "")).strip() == "proposed"
    }
    for commitment in (
        list(commitment_projection.active_commitments)
        + list(commitment_projection.deferred_commitments)
        + list(commitment_projection.blocked_commitments)
        if commitment_projection is not None
        else []
    ):
        for key in ("current_plan_proposal_id", "pending_plan_proposal_id"):
            proposal_id = _optional_text(commitment.details.get(key)) or ""
            if proposal_id:
                relevant_plan_ids.add(proposal_id)
    for proposal_id, node in plan_node_by_id.items():
        if (_optional_text(node.get("details", {}).get("commitment_id")) or "") in live_commitment_ids:
            relevant_plan_ids.add(proposal_id)
            continue
        if (_optional_text(node.get("details", {}).get("goal_id")) or "") in goal_ids:
            relevant_plan_ids.add(proposal_id)
    pending_plan_ids = set(relevant_plan_ids)
    while pending_plan_ids:
        proposal_id = pending_plan_ids.pop()
        prior_id = _optional_text(
            plan_node_by_id.get(proposal_id, {}).get("details", {}).get("supersedes_plan_proposal_id")
        ) or ""
        if prior_id and prior_id not in relevant_plan_ids and prior_id in plan_node_by_id:
            relevant_plan_ids.add(prior_id)
            pending_plan_ids.add(prior_id)

    resolved_thread_scope_id = thread_id or scope_id
    for proposal_id in sorted(relevant_plan_ids):
        proposal_node = plan_node_by_id.get(proposal_id)
        if proposal_node is None:
            continue
        proposal_node_id = str(proposal_node.get("node_id", "")).strip()
        dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.PLAN.value,
            resolved_thread_scope_id,
            proposal_id,
        )
        linked_commitment_id = _optional_text(proposal_node.get("details", {}).get("commitment_id")) or ""
        linked_commitment_node_id = (
            commitment_node_ids.get(linked_commitment_id) if linked_commitment_id else None
        )
        linked_commitment_node = (
            graph_indexes.node_by_id.get(linked_commitment_node_id)
            if linked_commitment_node_id is not None
            else None
        )
        supersedes_id = _optional_text(
            proposal_node.get("details", {}).get("supersedes_plan_proposal_id")
        ) or ""
        supersedes_node = plan_node_by_id.get(supersedes_id) if supersedes_id else None
        summary_evidence = _build_evidence_ref(
            graph_indexes=graph_indexes,
            extra_graph_node_ids=[
                proposal_node_id,
                *([linked_commitment_node_id] if linked_commitment_node_id is not None else []),
            ],
            extra_graph_edge_ids=_graph_edge_ids(
                graph_indexes,
                node_id=proposal_node_id,
                kinds=(
                    BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                    BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                ),
                other_node_id=linked_commitment_node_id,
            ),
        )
        key_current_facts = [
            _node_fact(
                proposal_node,
                dossier_id=dossier_id,
                details={"record_kind": "plan_proposal"},
            ),
            *(
                [
                    _node_fact(
                        linked_commitment_node,
                        dossier_id=dossier_id,
                        extra_graph_node_ids=[proposal_node_id],
                        extra_graph_edge_ids=_graph_edge_ids(
                            graph_indexes,
                            node_id=proposal_node_id,
                            kinds=(
                                BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
                                BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
                            ),
                            other_node_id=linked_commitment_node_id,
                        ),
                        details={"record_kind": "linked_commitment"},
                    )
                ]
                if linked_commitment_node is not None
                else []
            ),
        ]
        key_current_facts = sorted(key_current_facts, key=_fact_sort_key, reverse=True)[:6]
        recent_change_records: list[tuple[datetime, BrainContinuityDossierFactRecord]] = []
        if supersedes_node is not None:
            recent_change_records.append(
                (
                    _node_anchor_for_payload(supersedes_node) or datetime.min.replace(tzinfo=UTC),
                    _node_fact(
                        supersedes_node,
                        dossier_id=dossier_id,
                        status="historical",
                        extra_graph_node_ids=[proposal_node_id],
                        extra_graph_edge_ids=_graph_edge_ids(
                            graph_indexes,
                            node_id=proposal_node_id,
                            kinds=(BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_SUPERSEDES.value,),
                            other_node_id=str(supersedes_node.get("node_id", "")).strip(),
                        ),
                        details={"recent_change_kind": "superseded_plan"},
                    ),
                )
            )
        recent_changes = _sort_recent_change_records(recent_change_records, limit=4)
        proposal_status = str(proposal_node.get("status", "")).strip()
        freshness = (
            BrainContinuityDossierFreshness.STALE.value
            if proposal_status == "rejected" or proposal_id in plan_superseded_ids
            else BrainContinuityDossierFreshness.FRESH.value
        )
        review_policy = _optional_text(proposal_node.get("details", {}).get("review_policy")) or ""
        contradiction = (
            BrainContinuityDossierContradiction.UNCERTAIN.value
            if proposal_status == "rejected" or review_policy not in {"", "auto_adopt_ok"}
            else BrainContinuityDossierContradiction.CLEAR.value
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if proposal_status == "rejected":
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="rejected_plan",
                    summary="This plan proposal was rejected and is retained for traceability.",
                    evidence=summary_evidence,
                )
            )
        if proposal_id in plan_superseded_ids:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="superseded_plan",
                    summary="A newer proposal superseded this plan.",
                    evidence=summary_evidence,
                )
            )
        if review_policy not in {"", "auto_adopt_ok"}:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="review_policy_required",
                    summary="This plan proposal requires explicit review.",
                    evidence=summary_evidence,
                    details={"review_policy": review_policy},
                )
            )
        governance = _compile_dossier_governance(
            supporting_claims=[],
            freshness=freshness,
            contradiction=contradiction,
            summary=str(proposal_node.get("summary", "")).strip(),
        )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.PLAN.value,
                scope_type="thread",
                scope_id=resolved_thread_scope_id,
                title=f"Plan {proposal_id}",
                summary=str(proposal_node.get("summary", "")).strip() or proposal_id,
                status=(
                    "current"
                    if proposal_status in {"proposed", "adopted"} and proposal_id not in plan_superseded_ids
                    else "historical"
                ),
                freshness=freshness,
                contradiction=contradiction,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_commitment_ids=_sorted_unique_texts([linked_commitment_id]),
                source_plan_proposal_ids=_sorted_unique_texts([proposal_id, supersedes_id]),
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "goal_id": proposal_node.get("details", {}).get("goal_id"),
                    "commitment_id": linked_commitment_id or None,
                    "review_policy": review_policy or None,
                },
            )
        )

    procedural_family_records: dict[str, list[dict[str, Any]]] = {}
    for node in graph_indexes.node_by_id.values():
        if node.get("kind") != BrainContinuityGraphNodeKind.PROCEDURAL_SKILL.value:
            continue
        family_key = str(node.get("details", {}).get("skill_family_key", "")).strip()
        if not family_key:
            continue
        procedural_family_records.setdefault(family_key, []).append(node)

    procedural_status_order = {"active": 0, "candidate": 1, "superseded": 2, "retired": 3}
    for family_key in sorted(procedural_family_records):
        family_nodes = sorted(
            procedural_family_records[family_key],
            key=lambda node: (
                procedural_status_order.get(str(node.get("status", "")).strip(), 99),
                -float(node.get("details", {}).get("confidence") or 0.0),
                _node_anchor_for_payload(node) or datetime.min.replace(tzinfo=UTC),
                str(node.get("backing_record_id", "")),
            ),
        )
        leader_node = family_nodes[0]
        dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.PROCEDURAL.value,
            resolved_thread_scope_id,
            family_key,
        )
        summary_evidence = _build_evidence_ref(
            graph_indexes=graph_indexes,
            extra_graph_node_ids=[str(leader_node.get("node_id", "")).strip()],
        )
        current_family_nodes = [
            node
            for node in family_nodes
            if str(node.get("status", "")).strip() in {"active", "candidate"}
        ]
        key_current_facts = [
            _node_fact(
                node,
                dossier_id=dossier_id,
                details={"record_kind": "procedural_skill"},
            )
            for node in current_family_nodes[:6]
        ]
        recent_changes = _sort_recent_change_records(
            [
                (
                    _node_anchor_for_payload(node) or datetime.min.replace(tzinfo=UTC),
                    _node_fact(
                        node,
                        dossier_id=dossier_id,
                        status="historical",
                        details={"recent_change_kind": "historical_skill"},
                    ),
                )
                for node in family_nodes
                if str(node.get("status", "")).strip() in {"superseded", "retired"}
            ],
            limit=4,
        )
        leader_status = str(leader_node.get("status", "")).strip()
        freshness = (
            BrainContinuityDossierFreshness.FRESH.value
            if leader_status in {"active", "candidate"}
            else BrainContinuityDossierFreshness.STALE.value
        )
        contradiction = (
            BrainContinuityDossierContradiction.UNCERTAIN.value
            if leader_status == "candidate"
            else BrainContinuityDossierContradiction.CLEAR.value
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if leader_status == "candidate":
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="candidate_skill",
                    summary="This procedural family is still a candidate rather than an active skill.",
                    evidence=summary_evidence,
                )
            )
        if leader_status in {"superseded", "retired"}:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="historical_skill",
                    summary="This procedural family currently only has historical skill support.",
                    evidence=summary_evidence,
                )
            )
        governance = _compile_dossier_governance(
            supporting_claims=[],
            freshness=freshness,
            contradiction=contradiction,
            summary=str(leader_node.get("summary", "")).strip(),
        )
        source_skill_ids = _sorted_unique_texts(
            str(node.get("backing_record_id", "")).strip() for node in family_nodes
        )
        source_commitment_ids = _sorted_unique_texts(
            commitment_id
            for node in family_nodes
            for commitment_id in node.get("details", {}).get("supporting_commitment_ids", [])
        )
        source_plan_ids = _sorted_unique_texts(
            proposal_id
            for node in family_nodes
            for proposal_id in node.get("details", {}).get("supporting_plan_proposal_ids", [])
        )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.PROCEDURAL.value,
                scope_type="thread",
                scope_id=resolved_thread_scope_id,
                title=f"Procedure {family_key}",
                summary=str(leader_node.get("summary", "")).strip() or family_key,
                status="current" if leader_status in {"active", "candidate"} else "historical",
                freshness=freshness,
                contradiction=contradiction,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_skill_ids=source_skill_ids,
                source_commitment_ids=source_commitment_ids,
                source_plan_proposal_ids=source_plan_ids,
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "skill_family_key": family_key,
                    "leader_skill_id": str(leader_node.get("backing_record_id", "")).strip(),
                    "confidence": leader_node.get("details", {}).get("confidence"),
                },
            )
        )

    if scene_world_state is not None and (
        scene_world_state.entities
        or scene_world_state.affordances
        or scene_world_state.degraded_mode != "healthy"
    ):
        dossier_id = _stable_id(
            "dossier",
            BrainContinuityDossierKind.SCENE_WORLD.value,
            scene_world_state.scope_id,
        )
        entity_nodes = [
            _graph_node(
                graph_indexes,
                kind=BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
                backing_record_id=entity.entity_id,
            )
            for entity in scene_world_state.entities
        ]
        entity_nodes = [node for node in entity_nodes if node is not None]
        affordance_nodes = [
            _graph_node(
                graph_indexes,
                kind=BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
                backing_record_id=record.affordance_id,
            )
            for record in scene_world_state.affordances
        ]
        affordance_nodes = [node for node in affordance_nodes if node is not None]
        active_entity_nodes = [
            node for node in entity_nodes if str(node.get("status", "")).strip() == "active"
        ]
        available_affordance_nodes = [
            node for node in affordance_nodes if str(node.get("status", "")).strip() == "available"
        ]
        scene_episode_entries = [
            entry
            for entry in autobiography_list
            if entry.entry_kind == "scene_episode"
            and entry.scope_type == "presence"
            and entry.scope_id == scene_world_state.scope_id
        ]
        current_scene_episode = next(
            (
                entry
                for entry in scene_episode_entries
                if entry.status == "current"
                and (
                    (parsed := parse_multimodal_autobiography_record(entry)) is not None
                    and parsed.privacy_class != "redacted"
                )
            ),
            None,
        )
        summary_bits = [scene_world_state.degraded_mode]
        summary_bits.extend(str(node.get("summary", "")).strip() for node in active_entity_nodes[:2])
        if available_affordance_nodes:
            summary_bits.append(
                "affordances: "
                + ", ".join(
                    str(node.get("details", {}).get("capability_family", "")).strip()
                    for node in available_affordance_nodes[:2]
                    if str(node.get("details", {}).get("capability_family", "")).strip()
                )
            )
        summary_text = "; ".join(bit for bit in summary_bits if bit)
        summary_anchor_nodes = [
            *(active_entity_nodes[:2]),
            *(available_affordance_nodes[:2]),
        ] or entity_nodes[:1] or affordance_nodes[:1]
        summary_evidence = _build_evidence_ref(
            graph_indexes=graph_indexes,
            extra_graph_node_ids=[
                *(str(node.get("node_id", "")).strip() for node in summary_anchor_nodes),
            ],
        )
        key_current_facts = [
            *[
                _node_fact(
                    node,
                    dossier_id=dossier_id,
                    details={"record_kind": "scene_entity"},
                )
                for node in active_entity_nodes[:4]
            ],
            *[
                _node_fact(
                    node,
                    dossier_id=dossier_id,
                    details={"record_kind": "scene_affordance"},
                )
                for node in available_affordance_nodes[:2]
            ],
        ]
        if current_scene_episode is not None:
            key_current_facts.insert(
                0,
                _fact_from_entry(
                    current_scene_episode,
                    dossier_id=dossier_id,
                    graph_indexes=graph_indexes,
                    event_ts_by_id=event_ts_by_id,
                    details={
                        "record_kind": "scene_episode",
                        "privacy_class": current_scene_episode.privacy_class,
                        "review_state": current_scene_episode.review_state,
                        "retention_class": current_scene_episode.retention_class,
                    },
                    extra_graph_node_ids=[
                        *(
                            _graph_node_id(
                                graph_indexes,
                                kind=BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
                                backing_record_id=entity_id,
                            )
                            for entity_id in current_scene_episode.source_scene_entity_ids
                        ),
                        *(
                            _graph_node_id(
                                graph_indexes,
                                kind=BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
                                backing_record_id=affordance_id,
                            )
                            for affordance_id in current_scene_episode.source_scene_affordance_ids
                        ),
                    ],
                ),
            )
        key_current_facts = key_current_facts[:6]
        recent_changes = _sort_recent_change_records(
            [
                (
                    _node_anchor_for_payload(node) or datetime.min.replace(tzinfo=UTC),
                    _node_fact(
                        node,
                        dossier_id=dossier_id,
                        status="historical",
                        details={"recent_change_kind": "scene_state_change"},
                    ),
                )
                for node in (
                    [
                        node
                        for node in entity_nodes
                        if str(node.get("status", "")).strip()
                        in {"stale", "contradicted", "expired"}
                    ]
                    + [
                        node
                        for node in affordance_nodes
                        if str(node.get("status", "")).strip()
                        in {"blocked", "uncertain", "stale"}
                    ]
                )
            ]
            + [
                (
                    _entry_recency(entry, event_ts_by_id=event_ts_by_id),
                    _fact_from_entry(
                        entry,
                        dossier_id=dossier_id,
                        graph_indexes=graph_indexes,
                        event_ts_by_id=event_ts_by_id,
                        status="historical",
                        details={
                            "recent_change_kind": (
                                "redacted_scene_episode"
                                if (
                                    (parsed := parse_multimodal_autobiography_record(entry)) is not None
                                    and parsed.privacy_class == "redacted"
                                )
                                else "scene_episode_superseded"
                            ),
                            "privacy_class": entry.privacy_class,
                            "review_state": entry.review_state,
                        },
                        extra_graph_node_ids=[
                            *(
                                _graph_node_id(
                                    graph_indexes,
                                    kind=BrainContinuityGraphNodeKind.SCENE_WORLD_ENTITY.value,
                                    backing_record_id=entity_id,
                                )
                                for entity_id in entry.source_scene_entity_ids
                            ),
                            *(
                                _graph_node_id(
                                    graph_indexes,
                                    kind=BrainContinuityGraphNodeKind.SCENE_WORLD_AFFORDANCE.value,
                                    backing_record_id=affordance_id,
                                )
                                for affordance_id in entry.source_scene_affordance_ids
                            ),
                        ],
                    ),
                )
                for entry in scene_episode_entries
                if entry.entry_id != getattr(current_scene_episode, "entry_id", None)
                and (
                    entry.status == "superseded"
                    or (
                        (parsed := parse_multimodal_autobiography_record(entry)) is not None
                        and parsed.privacy_class == "redacted"
                    )
                )
            ],
            limit=4,
        )
        freshness = _freshness_from_support(
            summary_anchor_at=_parse_ts(scene_world_state.updated_at),
            current_support_anchors=[
                *(_node_anchor_for_payload(node) for node in active_entity_nodes),
                *(_node_anchor_for_payload(node) for node in available_affordance_nodes),
            ],
            has_current_support=bool(active_entity_nodes or available_affordance_nodes),
        )
        contradiction = (
            BrainContinuityDossierContradiction.CONTRADICTED.value
            if scene_world_state.contradiction_counts
            else (
                BrainContinuityDossierContradiction.UNCERTAIN.value
                if scene_world_state.degraded_mode != "healthy"
                or scene_world_state.blocked_affordance_ids
                or scene_world_state.uncertain_affordance_ids
                else BrainContinuityDossierContradiction.CLEAR.value
            )
        )
        issues: list[BrainContinuityDossierIssueRecord] = []
        if scene_world_state.degraded_mode != "healthy":
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="degraded_mode",
                    summary=f"Scene-world state is running in {scene_world_state.degraded_mode} mode.",
                    evidence=summary_evidence,
                    details={"degraded_reason_codes": list(scene_world_state.degraded_reason_codes)},
                )
            )
        if scene_world_state.contradiction_counts:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="contradicted_scene_support",
                    summary="Scene-world state contains contradicted entities.",
                    evidence=summary_evidence,
                    details={"contradiction_counts": dict(scene_world_state.contradiction_counts)},
                )
            )
        if scene_world_state.blocked_affordance_ids or scene_world_state.uncertain_affordance_ids:
            issues.append(
                _support_issue(
                    dossier_id=dossier_id,
                    kind="uncertain_affordances",
                    summary="Some scene affordances are blocked or uncertain.",
                    evidence=summary_evidence,
                    details={
                        "blocked_affordance_ids": list(scene_world_state.blocked_affordance_ids),
                        "uncertain_affordance_ids": list(scene_world_state.uncertain_affordance_ids),
                    },
                )
            )
        governance = _compile_dossier_governance(
            supporting_claims=[],
            freshness=freshness,
            contradiction=contradiction,
            summary=summary_text,
        )
        dossier_records.append(
            BrainContinuityDossierRecord(
                dossier_id=dossier_id,
                kind=BrainContinuityDossierKind.SCENE_WORLD.value,
                scope_type=scene_world_state.scope_type,
                scope_id=scene_world_state.scope_id,
                title="Scene World",
                summary=summary_text or "No scene-world summary available.",
                status="current" if active_entity_nodes or available_affordance_nodes else "historical",
                freshness=freshness,
                contradiction=contradiction,
                support_strength=_support_strength(summary_evidence, key_current_facts=key_current_facts),
                summary_evidence=summary_evidence,
                key_current_facts=key_current_facts,
                recent_changes=recent_changes,
                open_issues=sorted(issues, key=_issue_sort_key),
                source_entry_ids=_sorted_unique_texts(entry.entry_id for entry in scene_episode_entries),
                source_scene_entity_ids=_sorted_unique_texts(
                    entity.entity_id for entity in scene_world_state.entities
                ),
                source_scene_affordance_ids=_sorted_unique_texts(
                    record.affordance_id for record in scene_world_state.affordances
                ),
                source_event_ids=_collect_source_event_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                source_episode_ids=_collect_source_episode_ids(
                    summary_evidence=summary_evidence,
                    key_current_facts=key_current_facts,
                    recent_changes=recent_changes,
                    open_issues=issues,
                ),
                governance=governance,
                details={
                    "degraded_mode": scene_world_state.degraded_mode,
                    "entity_count": len(scene_world_state.entities),
                    "affordance_count": len(scene_world_state.affordances),
                },
            )
        )

    dossier_counts: dict[str, int] = {}
    freshness_counts: dict[str, int] = {}
    contradiction_counts: dict[str, int] = {}
    current_dossier_ids: list[str] = []
    stale_dossier_ids: list[str] = []
    needs_refresh_dossier_ids: list[str] = []
    uncertain_dossier_ids: list[str] = []
    contradicted_dossier_ids: list[str] = []
    sorted_dossiers = sorted(dossier_records, key=_dossier_sort_key)
    for dossier in sorted_dossiers:
        dossier_counts[dossier.kind] = dossier_counts.get(dossier.kind, 0) + 1
        freshness_counts[dossier.freshness] = freshness_counts.get(dossier.freshness, 0) + 1
        contradiction_counts[dossier.contradiction] = (
            contradiction_counts.get(dossier.contradiction, 0) + 1
        )
        if dossier.status == "current":
            current_dossier_ids.append(dossier.dossier_id)
        if dossier.freshness == BrainContinuityDossierFreshness.STALE.value:
            stale_dossier_ids.append(dossier.dossier_id)
        if dossier.freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
            needs_refresh_dossier_ids.append(dossier.dossier_id)
        if dossier.contradiction == BrainContinuityDossierContradiction.UNCERTAIN.value:
            uncertain_dossier_ids.append(dossier.dossier_id)
        if dossier.contradiction == BrainContinuityDossierContradiction.CONTRADICTED.value:
            contradicted_dossier_ids.append(dossier.dossier_id)

    return BrainContinuityDossierProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        dossiers=sorted_dossiers,
        dossier_counts=dossier_counts,
        freshness_counts=freshness_counts,
        contradiction_counts=contradiction_counts,
        current_dossier_ids=_sorted_unique_texts(current_dossier_ids),
        stale_dossier_ids=_sorted_unique_texts(stale_dossier_ids),
        needs_refresh_dossier_ids=_sorted_unique_texts(needs_refresh_dossier_ids),
        uncertain_dossier_ids=_sorted_unique_texts(uncertain_dossier_ids),
        contradicted_dossier_ids=_sorted_unique_texts(contradicted_dossier_ids),
    )


__all__ = [
    "BrainContinuityDossierAvailability",
    "BrainContinuityDossierContradiction",
    "BrainContinuityDossierEvidenceRef",
    "BrainContinuityDossierFactRecord",
    "BrainContinuityDossierFreshness",
    "BrainContinuityDossierGovernanceRecord",
    "BrainContinuityDossierIssueRecord",
    "BrainContinuityDossierKind",
    "BrainContinuityDossierProjection",
    "BrainContinuityDossierRecord",
    "BrainContinuityDossierTaskAvailability",
    "build_continuity_dossier_projection",
]

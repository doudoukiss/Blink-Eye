"""Deterministic reflection, autobiography, and reconciliation for Blink."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_layers.semantic import extract_memory_candidates, semantic_default_staleness
from blink.brain.memory_v2.autobiography import (
    AutobiographyService,
    BrainAutobiographyEntryKind,
)
from blink.brain.memory_v2.core_blocks import BrainCoreMemoryBlockKind
from blink.brain.memory_v2.health import MemoryHealthService
from blink.brain.projections import BrainGoalFamily

_PROJECT_PATTERNS = (
    r"\bproject\b",
    r"\broadmap\b",
    r"\brelease\b",
    r"\blaunch\b",
    r"项目",
    r"路线图",
    r"发布",
    r"上线",
)

_CORRECTION_PATTERNS = (
    r"\bactually\b",
    r"\binstead\b",
    r"\bcorrection\b",
    r"\bnot\b",
    r"\bdon't\b",
    r"其实",
    r"纠正",
    r"不是",
    r"改成",
    r"更正",
)

_VISION_EVENT_TYPES = {
    BrainEventType.PERCEPTION_OBSERVED,
    BrainEventType.SCENE_CHANGED,
    BrainEventType.ENGAGEMENT_CHANGED,
    BrainEventType.ATTENTION_CHANGED,
    BrainEventType.BODY_STATE_UPDATED,
}

_COMMITMENT_EVENT_TYPES = {
    BrainEventType.GOAL_CREATED,
    BrainEventType.GOAL_UPDATED,
    BrainEventType.GOAL_DEFERRED,
    BrainEventType.GOAL_RESUMED,
    BrainEventType.GOAL_CANCELLED,
    BrainEventType.GOAL_REPAIRED,
    BrainEventType.GOAL_COMPLETED,
    BrainEventType.GOAL_FAILED,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    return datetime.fromisoformat(value)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "reflection"


def _draft_artifact_name(draft: "BrainReflectionDraft") -> str:
    trigger_slug = _safe_slug(draft.trigger)
    thread_slug = _safe_slug(draft.thread_id)
    return (
        f"{thread_slug}_{trigger_slug}_"
        f"episodes_{draft.input_episode_cursor}_{draft.terminal_episode_cursor}_"
        f"events_{draft.input_event_cursor}_{draft.terminal_event_cursor}.json"
    )


def _summarize_text(text: str, *, limit: int = 96) -> str:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = (text or "").strip()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _extract_project_keys(text: str) -> tuple[str, ...]:
    normalized = " ".join((text or "").split())
    project_names = {
        match.strip()
        for match in re.findall(
            r"([A-Za-z][A-Za-z0-9_-]{1,31})\s+project", normalized, flags=re.IGNORECASE
        )
        if match.strip()
    }
    if "项目" in normalized and not project_names:
        project_names.add("default_project")
    return tuple(sorted(project_names))


def _claim_value(claim: Any) -> str:
    return str(claim.object.get("value", "")).strip().lower()


def _compact_claim_object(claim: Any) -> dict[str, Any]:
    """Return a compact claim object suitable for uncertain-conflict evidence."""
    value = str(claim.object.get("value") or "").strip()
    compact: dict[str, Any] = {}
    if value:
        compact["value"] = _summarize_text(value, limit=80)
    if getattr(claim, "claim_id", None):
        compact["claim_id"] = str(claim.claim_id)
    if getattr(claim, "predicate", None):
        compact["predicate"] = str(claim.predicate)
    return compact


@dataclass(frozen=True)
class BrainReflectionEngineConfig:
    """Deterministic configuration for one reflection cycle."""

    max_new_episodes: int = 24
    merge_window_secs: int = 900
    salience_threshold: float = 1.75


@dataclass(frozen=True)
class BrainReflectionSchedulerConfig:
    """Periodic scheduler configuration for background maintenance."""

    wakeup_interval_secs: float = 30.0
    idle_grace_secs: float = 45.0
    engine: BrainReflectionEngineConfig = field(default_factory=BrainReflectionEngineConfig)


@dataclass(frozen=True)
class BrainReflectionCycleRecord:
    """One durable reflection-cycle row."""

    cycle_id: str
    user_id: str
    thread_id: str
    trigger: str
    status: str
    input_episode_cursor: int
    input_event_cursor: int
    terminal_episode_cursor: int
    terminal_event_cursor: int
    draft_artifact_path: str | None
    result_stats_json: str
    skip_reason: str | None
    error_json: str | None
    started_at: str
    completed_at: str | None

    @property
    def result_stats(self) -> dict[str, Any]:
        """Return decoded result statistics."""
        return dict(json.loads(self.result_stats_json or "{}"))

    @property
    def error_payload(self) -> dict[str, Any] | None:
        """Return decoded error payload."""
        if not self.error_json:
            return None
        return dict(json.loads(self.error_json))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the cycle record."""
        return {
            "cycle_id": self.cycle_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "trigger": self.trigger,
            "status": self.status,
            "input_episode_cursor": self.input_episode_cursor,
            "input_event_cursor": self.input_event_cursor,
            "terminal_episode_cursor": self.terminal_episode_cursor,
            "terminal_event_cursor": self.terminal_event_cursor,
            "draft_artifact_path": self.draft_artifact_path,
            "result_stats": self.result_stats,
            "skip_reason": self.skip_reason,
            "error_payload": self.error_payload,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class BrainReflectionSegment:
    """One deterministically segmented reflection window."""

    segment_id: str
    session_id: str
    start_at: str
    end_at: str
    episode_ids: tuple[int, ...]
    source_event_ids: tuple[str, ...]
    summary: str
    text: str
    salience: float
    reasons: tuple[str, ...] = ()
    project_keys: tuple[str, ...] = ()
    correction_signal: bool = False


@dataclass(frozen=True)
class BrainReflectionClaimCandidate:
    """One structured claim candidate extracted from reflected segments."""

    candidate_id: str
    segment_id: str
    predicate: str
    subject: str
    object_value: str
    confidence: float
    singleton: bool
    claim_key: str
    correction_signal: bool
    source_episode_ids: tuple[int, ...] = ()
    source_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainReflectionReconciliationDecision:
    """One explicit reconciliation action proposed by reflection."""

    action: str
    reason: str
    candidate_id: str
    claim_key: str
    predicate: str
    prior_claim_id: str | None = None
    target_claim_id: str | None = None
    replacement_claim: dict[str, Any] = field(default_factory=dict)
    evidence_summary: str | None = None
    evidence_json: dict[str, Any] = field(default_factory=dict)
    source_episode_ids: tuple[int, ...] = ()
    source_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainReflectionBlockUpdate:
    """One explicit core-block update proposed by reflection."""

    block_kind: str
    scope_type: str
    scope_id: str
    content: dict[str, Any]
    salience: float
    reason: str


@dataclass(frozen=True)
class BrainReflectionAutobiographyDraft:
    """One explicit autobiographical update proposed by reflection."""

    entry_kind: str
    scope_type: str
    scope_id: str
    rendered_summary: str
    content: dict[str, Any]
    salience: float
    source_episode_ids: tuple[int, ...] = ()
    source_claim_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    append_only: bool = False
    identity_key: str | None = None


@dataclass(frozen=True)
class BrainReflectionHealthFinding:
    """One inspectable reflection health finding."""

    code: str
    severity: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrainReflectionDraft:
    """Structured reflection draft persisted before any mutation is applied."""

    cycle_id: str
    user_id: str
    thread_id: str
    trigger: str
    generated_at: str
    input_episode_cursor: int
    input_event_cursor: int
    terminal_episode_cursor: int
    terminal_event_cursor: int
    source_episode_ids: tuple[int, ...]
    source_event_ids: tuple[str, ...]
    segments: tuple[BrainReflectionSegment, ...]
    claim_candidates: tuple[BrainReflectionClaimCandidate, ...]
    reconciliation_decisions: tuple[BrainReflectionReconciliationDecision, ...]
    block_updates: tuple[BrainReflectionBlockUpdate, ...]
    autobiography_updates: tuple[BrainReflectionAutobiographyDraft, ...]
    health_findings: tuple[BrainReflectionHealthFinding, ...]
    stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the draft to JSON-safe data."""
        return {
            "cycle_id": self.cycle_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "trigger": self.trigger,
            "generated_at": self.generated_at,
            "input_episode_cursor": self.input_episode_cursor,
            "input_event_cursor": self.input_event_cursor,
            "terminal_episode_cursor": self.terminal_episode_cursor,
            "terminal_event_cursor": self.terminal_event_cursor,
            "source_episode_ids": list(self.source_episode_ids),
            "source_event_ids": list(self.source_event_ids),
            "segments": [asdict(item) for item in self.segments],
            "claim_candidates": [asdict(item) for item in self.claim_candidates],
            "reconciliation_decisions": [asdict(item) for item in self.reconciliation_decisions],
            "block_updates": [asdict(item) for item in self.block_updates],
            "autobiography_updates": [asdict(item) for item in self.autobiography_updates],
            "health_findings": [asdict(item) for item in self.health_findings],
            "stats": dict(self.stats),
        }


@dataclass(frozen=True)
class BrainReflectionRunResult:
    """Applied reflection result with inspectable artifact paths and cycle state."""

    cycle_id: str
    status: str
    trigger: str
    source_episode_ids: tuple[int, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    draft_artifact_path: Path | None = None
    applied_claim_actions: tuple[dict[str, Any], ...] = ()
    applied_block_actions: tuple[dict[str, Any], ...] = ()
    applied_autobiography_entries: tuple[str, ...] = ()
    health_report_id: str | None = None
    skip_reason: str | None = None
    cycle_record: BrainReflectionCycleRecord | None = None


class BrainReflectionEngine:
    """Run one deterministic reflection cycle against the canonical store."""

    def __init__(self, *, store, config: BrainReflectionEngineConfig | None = None):
        """Bind the engine to one canonical store."""
        self._store = store
        self._config = config or BrainReflectionEngineConfig()
        self._autobiography = AutobiographyService(store=store)
        self._health = MemoryHealthService(store=store)
        self._claims = store._claims()
        self._entities = store._entities()

    def run_once(
        self,
        user_id: str,
        thread_id: str,
        *,
        session_ids=None,
        trigger: str = "manual",
    ) -> BrainReflectionRunResult:
        """Run one deterministic reflection cycle for a user/thread."""
        resolved_session_ids = self._resolve_session_ids(
            user_id=user_id,
            thread_id=thread_id,
            session_ids=session_ids,
        )
        cycle_id = f"reflection_{uuid4().hex}"
        input_episode_cursor, input_event_cursor = self._store.latest_reflection_cursors(
            user_id=user_id,
            thread_id=thread_id,
        )
        cycle_record = self._store.start_reflection_cycle(
            cycle_id=cycle_id,
            user_id=user_id,
            thread_id=thread_id,
            trigger=trigger,
            input_episode_cursor=input_episode_cursor,
            input_event_cursor=input_event_cursor,
        )
        try:
            inputs = self._collect_inputs(
                user_id=user_id,
                thread_id=thread_id,
                session_ids=resolved_session_ids,
                input_episode_cursor=input_episode_cursor,
                input_event_cursor=input_event_cursor,
            )
            draft = self._build_draft(
                cycle_id=cycle_id,
                user_id=user_id,
                thread_id=thread_id,
                trigger=trigger,
                session_ids=resolved_session_ids,
                **inputs,
            )
            artifact_path = self._persist_draft(draft)
            event_context = {
                "agent_id": resolved_session_ids.agent_id,
                "user_id": resolved_session_ids.user_id,
                "session_id": resolved_session_ids.session_id,
                "thread_id": resolved_session_ids.thread_id,
                "source": "reflection",
                "correlation_id": cycle_id,
            }
            start_event = self._store.append_brain_event(
                event_type=BrainEventType.REFLECTION_CYCLE_STARTED,
                agent_id=resolved_session_ids.agent_id,
                user_id=resolved_session_ids.user_id,
                session_id=resolved_session_ids.session_id,
                thread_id=resolved_session_ids.thread_id,
                source="reflection",
                correlation_id=cycle_id,
                payload={
                    "cycle_id": cycle_id,
                    "trigger": trigger,
                    "input_episode_cursor": draft.input_episode_cursor,
                    "input_event_cursor": draft.input_event_cursor,
                    "source_episode_ids": list(draft.source_episode_ids),
                    "source_event_ids": list(draft.source_event_ids),
                    "draft_artifact_path": str(artifact_path),
                },
            )
            applied_claim_actions = self._apply_reconciliation(
                decisions=draft.reconciliation_decisions,
                source_event_id=start_event.event_id,
                event_context=event_context,
            )
            applied_block_actions = self._apply_block_updates(
                updates=draft.block_updates,
                source_event_id=start_event.event_id,
                event_context=event_context,
            )
            applied_autobiography_entries = self._apply_autobiography_updates(
                updates=draft.autobiography_updates,
                source_event_id=start_event.event_id,
                event_context=event_context,
            )
            relationship_scope_id = f"{resolved_session_ids.agent_id}:{user_id}"
            health_report = self._health.record_report(
                scope_type="relationship",
                scope_id=relationship_scope_id,
                cycle_id=cycle_id,
                score=float(draft.stats.get("health_score", 1.0)),
                status=str(draft.stats.get("health_status", "healthy")),
                findings=[asdict(item) for item in draft.health_findings],
                stats=dict(draft.stats),
                artifact_path=str(artifact_path),
                source_event_id=start_event.event_id,
                event_context=event_context,
            )
            cycle_record = self._store.complete_reflection_cycle(
                cycle_id=cycle_id,
                terminal_episode_cursor=draft.terminal_episode_cursor,
                terminal_event_cursor=draft.terminal_event_cursor,
                draft_artifact_path=str(artifact_path),
                result_stats={
                    **draft.stats,
                    "source_episode_ids": list(draft.source_episode_ids),
                    "source_event_ids": list(draft.source_event_ids),
                },
            )
            self._store.append_brain_event(
                event_type=BrainEventType.REFLECTION_CYCLE_COMPLETED,
                agent_id=resolved_session_ids.agent_id,
                user_id=resolved_session_ids.user_id,
                session_id=resolved_session_ids.session_id,
                thread_id=resolved_session_ids.thread_id,
                source="reflection",
                correlation_id=cycle_id,
                causal_parent_id=start_event.event_id,
                payload={
                    "cycle_id": cycle_id,
                    "trigger": trigger,
                    "input_episode_cursor": draft.input_episode_cursor,
                    "input_event_cursor": draft.input_event_cursor,
                    "terminal_episode_cursor": draft.terminal_episode_cursor,
                    "terminal_event_cursor": draft.terminal_event_cursor,
                    "draft_artifact_path": str(artifact_path),
                    "result_stats": cycle_record.result_stats,
                    "applied_claim_actions": applied_claim_actions,
                    "applied_block_actions": applied_block_actions,
                    "applied_autobiography_entries": list(applied_autobiography_entries),
                    "health_report_id": health_report.report_id,
                },
            )
            self._store.record_memory_export(
                user_id=user_id,
                thread_id=thread_id,
                export_kind="reflection_draft",
                path=artifact_path,
                payload=draft.as_dict(),
            )
            return BrainReflectionRunResult(
                cycle_id=cycle_id,
                status="completed",
                trigger=trigger,
                source_episode_ids=draft.source_episode_ids,
                source_event_ids=draft.source_event_ids,
                draft_artifact_path=artifact_path,
                applied_claim_actions=tuple(applied_claim_actions),
                applied_block_actions=tuple(applied_block_actions),
                applied_autobiography_entries=tuple(applied_autobiography_entries),
                health_report_id=health_report.report_id,
                cycle_record=cycle_record,
            )
        except Exception as exc:
            error_payload = {"type": type(exc).__name__, "message": str(exc)}
            cycle_record = self._store.fail_reflection_cycle(
                cycle_id=cycle_id,
                error_payload=error_payload,
            )
            self._store.append_brain_event(
                event_type=BrainEventType.REFLECTION_CYCLE_FAILED,
                agent_id=resolved_session_ids.agent_id,
                user_id=resolved_session_ids.user_id,
                session_id=resolved_session_ids.session_id,
                thread_id=resolved_session_ids.thread_id,
                source="reflection",
                correlation_id=cycle_id,
                payload={
                    "cycle_id": cycle_id,
                    "trigger": trigger,
                    "input_episode_cursor": input_episode_cursor,
                    "input_event_cursor": input_event_cursor,
                    "error": error_payload,
                },
            )
            raise

    def _resolve_session_ids(self, *, user_id: str, thread_id: str, session_ids=None):
        if session_ids is not None:
            return session_ids
        latest_event = self._store.recent_brain_events(
            user_id=user_id, thread_id=thread_id, limit=1
        )
        if latest_event:
            event = latest_event[0]
            return type(
                "BrainSessionIdsProxy",
                (),
                {
                    "agent_id": event.agent_id,
                    "user_id": event.user_id,
                    "session_id": event.session_id,
                    "thread_id": event.thread_id,
                },
            )()
        return type(
            "BrainSessionIdsProxy",
            (),
            {
                "agent_id": "blink/main",
                "user_id": user_id,
                "session_id": f"reflection:{thread_id}",
                "thread_id": thread_id,
            },
        )()

    def _collect_inputs(
        self,
        *,
        user_id: str,
        thread_id: str,
        session_ids,
        input_episode_cursor: int,
        input_event_cursor: int,
    ) -> dict[str, Any]:
        events = self._store.brain_events_since(
            user_id=user_id,
            thread_id=thread_id,
            after_id=input_event_cursor,
        )
        episodes = self._store.episodes_since(
            user_id=user_id,
            thread_id=thread_id,
            after_id=input_episode_cursor,
            limit=self._config.max_new_episodes,
        )
        if not episodes:
            new_event_ids = {event.event_id for event in events}
            episodes = [
                record
                for record in self._store.episodic_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    limit=self._config.max_new_episodes,
                )
                if record.source_event_id and record.source_event_id in new_event_ids
            ]
        terminal_episode_cursor = max(
            [input_episode_cursor, *[int(getattr(record, "id", 0)) for record in episodes]]
        )
        terminal_event_cursor = max([input_event_cursor, *[int(event.id) for event in events]])
        current_claims = self._store.query_claims(
            temporal_mode="all",
            scope_type="user",
            scope_id=user_id,
            limit=128,
        )
        core_blocks = self._store.list_current_core_memory_blocks(limit=24)
        relationship_scope_id = f"{session_ids.agent_id}:{user_id}"
        commitments = self._store.list_executive_commitments(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            statuses=("active", "deferred", "blocked"),
            limit=32,
        )
        return {
            "events": events,
            "episodes": episodes,
            "current_claims": current_claims,
            "core_blocks": core_blocks,
            "commitments": commitments,
            "input_episode_cursor": input_episode_cursor,
            "input_event_cursor": input_event_cursor,
            "terminal_episode_cursor": terminal_episode_cursor,
            "terminal_event_cursor": terminal_event_cursor,
        }

    def _build_draft(
        self,
        *,
        cycle_id: str,
        user_id: str,
        thread_id: str,
        trigger: str,
        session_ids,
        events: list[BrainEventRecord],
        episodes: list[Any],
        current_claims: list[Any],
        core_blocks: list[Any],
        commitments: list[Any],
        input_episode_cursor: int,
        input_event_cursor: int,
        terminal_episode_cursor: int,
        terminal_event_cursor: int,
    ) -> BrainReflectionDraft:
        segments = self._segment_sources(episodes=episodes, events=events)
        claim_candidates = self._extract_claim_candidates(
            user_id=user_id,
            segments=segments,
        )
        reconciliation_decisions, decision_findings = self._build_reconciliation_decisions(
            user_id=user_id,
            claim_candidates=claim_candidates,
            current_claims=current_claims,
        )
        block_updates = self._build_block_updates(
            cycle_id=cycle_id,
            user_id=user_id,
            session_ids=session_ids,
            segments=segments,
            commitments=commitments,
            core_blocks=core_blocks,
        )
        autobiography_updates = self._build_autobiography_updates(
            cycle_id=cycle_id,
            user_id=user_id,
            session_ids=session_ids,
            segments=segments,
            decisions=reconciliation_decisions,
        )
        health_findings = decision_findings + self._build_health_findings(
            user_id=user_id,
            thread_id=thread_id,
            segments=segments,
            current_claims=current_claims,
            decisions=reconciliation_decisions,
            commitments=commitments,
        )
        health_score = max(
            0.0,
            min(
                1.0,
                1.0
                - (0.12 * sum(1 for item in health_findings if item.severity == "warning"))
                - (0.22 * sum(1 for item in health_findings if item.severity == "critical")),
            ),
        )
        stats = {
            "segment_count": len(segments),
            "salient_segment_count": sum(
                1 for segment in segments if segment.salience >= self._config.salience_threshold
            ),
            "claim_candidate_count": len(claim_candidates),
            "reconciliation_decision_count": len(reconciliation_decisions),
            "autobiography_update_count": len(autobiography_updates),
            "block_update_count": len(block_updates),
            "open_commitment_count": len(commitments),
            "health_finding_count": len(health_findings),
            "health_score": round(health_score, 3),
            "health_status": (
                "needs_attention"
                if health_score < 0.6
                else "watch"
                if health_score < 0.8
                else "healthy"
            ),
        }
        return BrainReflectionDraft(
            cycle_id=cycle_id,
            user_id=user_id,
            thread_id=thread_id,
            trigger=trigger,
            generated_at=_utc_now(),
            input_episode_cursor=input_episode_cursor,
            input_event_cursor=input_event_cursor,
            terminal_episode_cursor=terminal_episode_cursor,
            terminal_event_cursor=terminal_event_cursor,
            source_episode_ids=tuple(
                int(getattr(record, "id"))
                for record in episodes
                if getattr(record, "id", None) is not None
            ),
            source_event_ids=tuple(event.event_id for event in events),
            segments=tuple(segments),
            claim_candidates=tuple(claim_candidates),
            reconciliation_decisions=tuple(reconciliation_decisions),
            block_updates=tuple(block_updates),
            autobiography_updates=tuple(autobiography_updates),
            health_findings=tuple(health_findings),
            stats=stats,
        )

    def _segment_sources(
        self,
        *,
        episodes: list[Any],
        events: list[BrainEventRecord],
    ) -> list[BrainReflectionSegment]:
        if not episodes:
            return []
        sorted_records = sorted(
            episodes,
            key=lambda record: _parse_ts(
                getattr(record, "created_at", None) or getattr(record, "observed_at", None)
            ),
        )
        segments: list[BrainReflectionSegment] = []
        current_bucket: list[Any] = []
        for record in sorted_records:
            current_ts = _parse_ts(
                getattr(record, "created_at", None) or getattr(record, "observed_at", None)
            )
            if not current_bucket:
                current_bucket.append(record)
                continue
            previous = current_bucket[-1]
            previous_ts = _parse_ts(
                getattr(previous, "created_at", None) or getattr(previous, "observed_at", None)
            )
            same_session = getattr(record, "session_id", "") == getattr(previous, "session_id", "")
            if (
                same_session
                and (current_ts - previous_ts).total_seconds() <= self._config.merge_window_secs
            ):
                current_bucket.append(record)
                continue
            segments.append(self._build_segment(current_bucket, events=events))
            current_bucket = [record]
        if current_bucket:
            segments.append(self._build_segment(current_bucket, events=events))

        project_sessions: dict[str, set[str]] = {}
        for segment in segments:
            for project_key in segment.project_keys:
                project_sessions.setdefault(project_key, set()).add(segment.session_id)
        enriched: list[BrainReflectionSegment] = []
        for segment in segments:
            boost = 0.0
            reasons = list(segment.reasons)
            if any(
                len(project_sessions.get(project_key, set())) > 1
                for project_key in segment.project_keys
            ):
                boost += 0.6
                reasons.append("cross_session_project")
            enriched.append(
                replace(
                    segment,
                    salience=round(segment.salience + boost, 3),
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        return enriched

    def _build_segment(
        self, bucket: list[Any], *, events: list[BrainEventRecord]
    ) -> BrainReflectionSegment:
        first = bucket[0]
        last = bucket[-1]
        start_at = (
            getattr(first, "created_at", None) or getattr(first, "observed_at", None) or _utc_now()
        )
        end_at = getattr(last, "created_at", None) or getattr(last, "observed_at", None) or start_at
        start_dt = _parse_ts(str(start_at))
        end_dt = _parse_ts(str(end_at))
        text_parts: list[str] = []
        episode_ids: list[int] = []
        source_event_ids: set[str] = set()
        for record in bucket:
            if hasattr(record, "user_text"):
                text_parts.extend(
                    [
                        str(getattr(record, "user_text", "")).strip(),
                        str(getattr(record, "assistant_summary", "")).strip(),
                    ]
                )
                episode_ids.append(int(record.id))
            else:
                text_parts.append(str(getattr(record, "summary", "")).strip())
            if getattr(record, "source_event_id", None):
                source_event_ids.add(str(record.source_event_id))
        window_events = [
            event
            for event in events
            if event.event_id in source_event_ids or (start_dt <= _parse_ts(event.ts) <= end_dt)
        ]
        text = " ".join(part for part in text_parts if part)
        correction_signal = _contains_pattern(text, _CORRECTION_PATTERNS)
        project_keys = _extract_project_keys(text)
        reasons: list[str] = []
        salience = 0.7 + (0.35 * max(len(bucket) - 1, 0))
        if correction_signal:
            salience += 1.5
            reasons.append("correction")
        if project_keys or _contains_pattern(text, _PROJECT_PATTERNS):
            salience += 1.1
            reasons.append("project_signal")
        if any(event.event_type.startswith("tool.") for event in window_events) or any(
            event.event_type in _VISION_EVENT_TYPES for event in window_events
        ):
            salience += 0.35
            reasons.append("tool_or_vision_signal")
        if any(event.event_type.startswith("memory.claim.") for event in window_events):
            salience += 0.45
            reasons.append("claim_change")
        if any(event.event_type in _COMMITMENT_EVENT_TYPES for event in window_events):
            salience += 0.55
            reasons.append("commitment_change")
        summary = _summarize_text(text or getattr(first, "summary", "") or "reflection segment")
        return BrainReflectionSegment(
            segment_id=f"segment_{uuid4().hex[:12]}",
            session_id=str(getattr(first, "session_id", "unknown")),
            start_at=str(start_at),
            end_at=str(end_at),
            episode_ids=tuple(episode_ids),
            source_event_ids=tuple(
                sorted(source_event_ids or {event.event_id for event in window_events})
            ),
            summary=summary,
            text=text or summary,
            salience=round(salience, 3),
            reasons=tuple(reasons),
            project_keys=tuple(project_keys),
            correction_signal=correction_signal,
        )

    def _extract_claim_candidates(
        self,
        *,
        user_id: str,
        segments: list[BrainReflectionSegment],
    ) -> list[BrainReflectionClaimCandidate]:
        candidates: list[BrainReflectionClaimCandidate] = []
        for segment in segments:
            if "claim_change" in segment.reasons:
                continue
            for fact in extract_memory_candidates(segment.text):
                object_value = str(fact.value.get("value", "")).strip()
                if not object_value:
                    continue
                claim_key = (
                    f"{fact.namespace}:{user_id}"
                    if fact.singleton
                    else (
                        f"preference:{object_value.lower()}"
                        if fact.namespace.startswith("preference.")
                        else f"{fact.namespace}:{fact.subject.lower()}"
                    )
                )
                candidates.append(
                    BrainReflectionClaimCandidate(
                        candidate_id=f"candidate_{uuid4().hex[:12]}",
                        segment_id=segment.segment_id,
                        predicate=fact.namespace,
                        subject=fact.subject,
                        object_value=object_value,
                        confidence=min(
                            1.0, float(fact.confidence) + max(segment.salience - 1.0, 0.0) * 0.05
                        ),
                        singleton=fact.singleton,
                        claim_key=claim_key,
                        correction_signal=segment.correction_signal,
                        source_episode_ids=segment.episode_ids,
                        source_event_ids=segment.source_event_ids,
                    )
                )
        deduped: dict[tuple[str, str, str], BrainReflectionClaimCandidate] = {}
        for candidate in candidates:
            key = (candidate.segment_id, candidate.predicate, candidate.object_value.lower())
            existing = deduped.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                deduped[key] = candidate
        return list(deduped.values())

    def _build_reconciliation_decisions(
        self,
        *,
        user_id: str,
        claim_candidates: list[BrainReflectionClaimCandidate],
        current_claims: list[Any],
    ) -> tuple[list[BrainReflectionReconciliationDecision], list[BrainReflectionHealthFinding]]:
        user_entity = self._entities.ensure_entity(
            entity_type="user",
            canonical_name=user_id,
            aliases=[user_id],
            attributes={"user_id": user_id},
        )
        active_claims = [
            claim for claim in current_claims if claim.status in {"active", "uncertain"}
        ]
        decisions: list[BrainReflectionReconciliationDecision] = []
        findings: list[BrainReflectionHealthFinding] = []
        decided_claim_keys: set[str] = set()

        for candidate in claim_candidates:
            same_predicate = [
                claim for claim in active_claims if claim.predicate == candidate.predicate
            ]
            same_value = [
                claim
                for claim in same_predicate
                if _claim_value(claim) == candidate.object_value.lower()
            ]
            if same_value:
                target = sorted(
                    same_value,
                    key=lambda claim: (float(claim.confidence), claim.updated_at),
                    reverse=True,
                )[0]
                decisions.append(
                    BrainReflectionReconciliationDecision(
                        action="reinforce_existing",
                        reason="same_value_reinforcement",
                        candidate_id=candidate.candidate_id,
                        claim_key=candidate.claim_key,
                        predicate=candidate.predicate,
                        target_claim_id=target.claim_id,
                        evidence_summary=f"Reflection reinforced {candidate.predicate}={candidate.object_value}",
                        evidence_json={
                            "candidate_id": candidate.candidate_id,
                            "source": "reflection",
                            "source_event_ids": list(candidate.source_event_ids),
                        },
                        source_episode_ids=candidate.source_episode_ids,
                        source_event_ids=candidate.source_event_ids,
                    )
                )
                decided_claim_keys.add(candidate.claim_key)
                continue

            replacement_claim = self._replacement_claim_payload(
                user_entity_id=user_entity.entity_id,
                user_id=user_id,
                candidate=candidate,
                status="active",
            )

            prior_claim: Any | None = None
            if candidate.singleton:
                prior_claim = same_predicate[0] if same_predicate else None
            else:
                conflicting_predicate = (
                    "preference.dislike"
                    if candidate.predicate == "preference.like"
                    else "preference.like"
                    if candidate.predicate == "preference.dislike"
                    else None
                )
                if conflicting_predicate is not None:
                    for claim in active_claims:
                        if claim.predicate != conflicting_predicate:
                            continue
                        if _claim_value(claim) == candidate.object_value.lower():
                            prior_claim = claim
                            break

            if prior_claim is None:
                decisions.append(
                    BrainReflectionReconciliationDecision(
                        action="record_new",
                        reason="new_reflection_candidate",
                        candidate_id=candidate.candidate_id,
                        claim_key=candidate.claim_key,
                        predicate=candidate.predicate,
                        replacement_claim=replacement_claim,
                        evidence_summary=f"Reflection extracted {candidate.predicate}={candidate.object_value}",
                        evidence_json={
                            "candidate_id": candidate.candidate_id,
                            "source": "reflection",
                            "source_event_ids": list(candidate.source_event_ids),
                        },
                        source_episode_ids=candidate.source_episode_ids,
                        source_event_ids=candidate.source_event_ids,
                    )
                )
                decided_claim_keys.add(candidate.claim_key)
                continue

            if candidate.correction_signal:
                decisions.append(
                    BrainReflectionReconciliationDecision(
                        action="supersede_existing",
                        reason="explicit_correction",
                        candidate_id=candidate.candidate_id,
                        claim_key=candidate.claim_key,
                        predicate=candidate.predicate,
                        prior_claim_id=prior_claim.claim_id,
                        replacement_claim=replacement_claim,
                        evidence_summary=f"Reflection extracted an explicit correction for {candidate.predicate}",
                        evidence_json={
                            "candidate_id": candidate.candidate_id,
                            "source": "reflection",
                            "source_event_ids": list(candidate.source_event_ids),
                        },
                        source_episode_ids=candidate.source_episode_ids,
                        source_event_ids=candidate.source_event_ids,
                    )
                )
                decided_claim_keys.add(candidate.claim_key)
                continue

            stronger = float(candidate.confidence) >= float(prior_claim.confidence) + 0.12
            if stronger and candidate.singleton:
                decisions.append(
                    BrainReflectionReconciliationDecision(
                        action="supersede_existing",
                        reason="newer_stronger_evidence",
                        candidate_id=candidate.candidate_id,
                        claim_key=candidate.claim_key,
                        predicate=candidate.predicate,
                        prior_claim_id=prior_claim.claim_id,
                        replacement_claim=replacement_claim,
                        evidence_summary=f"Reflection found stronger evidence for {candidate.predicate}",
                        evidence_json={
                            "candidate_id": candidate.candidate_id,
                            "source": "reflection",
                            "source_event_ids": list(candidate.source_event_ids),
                        },
                        source_episode_ids=candidate.source_episode_ids,
                        source_event_ids=candidate.source_event_ids,
                    )
                )
                decided_claim_keys.add(candidate.claim_key)
                continue

            uncertain_claim = self._replacement_claim_payload(
                user_entity_id=user_entity.entity_id,
                user_id=user_id,
                candidate=candidate,
                status="uncertain",
            )
            uncertain_claim["claim_key"] = f"{candidate.claim_key}:uncertain"
            uncertain_claim["confidence"] = min(float(candidate.confidence), 0.55)
            uncertain_claim["object_data"] = {
                "value": candidate.object_value,
                "candidates": [
                    _compact_claim_object(prior_claim),
                    {"value": candidate.object_value},
                ],
            }
            decisions.append(
                BrainReflectionReconciliationDecision(
                    action="record_uncertain",
                    reason="ambiguous_conflict",
                    candidate_id=candidate.candidate_id,
                    claim_key=candidate.claim_key,
                    predicate=candidate.predicate,
                    prior_claim_id=prior_claim.claim_id,
                    replacement_claim=uncertain_claim,
                    evidence_summary=f"Reflection found an ambiguous contradiction for {candidate.predicate}",
                    evidence_json={
                        "candidate_id": candidate.candidate_id,
                        "source": "reflection",
                        "source_event_ids": list(candidate.source_event_ids),
                        "conflicting_claim_id": prior_claim.claim_id,
                    },
                    source_episode_ids=candidate.source_episode_ids,
                    source_event_ids=candidate.source_event_ids,
                )
            )
            decided_claim_keys.add(candidate.claim_key)
            findings.append(
                BrainReflectionHealthFinding(
                    code="ambiguous_claim_conflict",
                    severity="warning",
                    summary=f"Reflection left {candidate.predicate} uncertain due to ambiguous evidence.",
                    details={
                        "candidate_id": candidate.candidate_id,
                        "prior_claim_id": prior_claim.claim_id,
                        "claim_key": candidate.claim_key,
                    },
                )
            )
        grouped_existing: dict[str, list[Any]] = {}
        for claim in active_claims:
            claim_key = str(
                claim.claim_key or f"{claim.scope_type}:{claim.scope_id}:{claim.predicate}"
            )
            grouped_existing.setdefault(claim_key, []).append(claim)
        for claim_key, claims in grouped_existing.items():
            if claim_key in decided_claim_keys:
                continue
            distinct_values = {
                json.dumps(claim.object, ensure_ascii=False, sort_keys=True): claim
                for claim in claims
            }
            if len(distinct_values) <= 1:
                continue
            ordered = sorted(
                claims,
                key=lambda claim: (float(claim.confidence), claim.updated_at),
                reverse=True,
            )
            leader = ordered[0]
            uncertain_claim = {
                "subject_entity_id": leader.subject_entity_id,
                "predicate": leader.predicate,
                "object_entity_id": None,
                "object_data": {
                    "value": leader.object.get("value"),
                    "candidates": [_compact_claim_object(claim) for claim in ordered[:3]],
                },
                "status": "uncertain",
                "confidence": min(float(leader.confidence), 0.55),
                "scope_type": leader.scope_type,
                "scope_id": leader.scope_id,
                "claim_key": f"{claim_key}:uncertain",
                "stale_after_seconds": leader.stale_after_seconds,
            }
            decisions.append(
                BrainReflectionReconciliationDecision(
                    action="record_uncertain",
                    reason="ambiguous_existing_conflict",
                    candidate_id=f"existing_conflict_{claim_key}",
                    claim_key=claim_key,
                    predicate=leader.predicate,
                    prior_claim_id=leader.claim_id,
                    replacement_claim=uncertain_claim,
                    evidence_summary=f"Reflection found ambiguous existing conflict for {claim_key}",
                    evidence_json={
                        "source": "reflection",
                        "conflicting_claim_ids": [claim.claim_id for claim in ordered],
                    },
                )
            )
            findings.append(
                BrainReflectionHealthFinding(
                    code="ambiguous_claim_conflict",
                    severity="warning",
                    summary=f"Reflection left {claim_key} uncertain due to conflicting current claims.",
                    details={"claim_ids": [claim.claim_id for claim in ordered]},
                )
            )
        return decisions, findings

    def _replacement_claim_payload(
        self,
        *,
        user_entity_id: str,
        user_id: str,
        candidate: BrainReflectionClaimCandidate,
        status: str,
    ) -> dict[str, Any]:
        object_entity_id: str | None = None
        if candidate.predicate.startswith("preference."):
            topic = self._entities.ensure_entity(
                entity_type="topic",
                canonical_name=candidate.object_value,
                aliases=[candidate.subject],
                attributes={"normalized_subject": candidate.subject},
            )
            object_entity_id = topic.entity_id
        return {
            "subject_entity_id": user_entity_id,
            "predicate": candidate.predicate,
            "object_entity_id": object_entity_id,
            "object_value": candidate.object_value,
            "object_data": {"value": candidate.object_value},
            "status": status,
            "confidence": candidate.confidence,
            "scope_type": "user",
            "scope_id": user_id,
            "claim_key": candidate.claim_key,
            "stale_after_seconds": semantic_default_staleness(candidate.predicate),
        }

    def _build_block_updates(
        self,
        *,
        cycle_id: str,
        user_id: str,
        session_ids,
        segments: list[BrainReflectionSegment],
        commitments: list[Any],
        core_blocks: list[Any],
    ) -> list[BrainReflectionBlockUpdate]:
        salient = [
            segment for segment in segments if segment.salience >= self._config.salience_threshold
        ]
        if not salient:
            return []
        current_arc = next(
            (
                record
                for record in core_blocks
                if record.block_kind == BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value
            ),
            None,
        )
        summary = " | ".join(segment.summary for segment in salient[:3])
        content = {
            "summary": summary,
            "cycle_id": cycle_id,
            "user_id": user_id,
            "salient_segment_ids": [segment.segment_id for segment in salient[:6]],
            "source_episode_ids": [
                episode_id for segment in salient for episode_id in segment.episode_ids
            ],
            "open_commitment_titles": [record.title for record in commitments[:6]],
        }
        if current_arc is not None and current_arc.content == content:
            return []
        return [
            BrainReflectionBlockUpdate(
                block_kind=BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
                scope_type="agent",
                scope_id=session_ids.agent_id,
                content=content,
                salience=max(segment.salience for segment in salient),
                reason="reflection_salient_episode_window",
            )
        ]

    def _build_autobiography_updates(
        self,
        *,
        cycle_id: str,
        user_id: str,
        session_ids,
        segments: list[BrainReflectionSegment],
        decisions: list[BrainReflectionReconciliationDecision],
    ) -> list[BrainReflectionAutobiographyDraft]:
        relationship_scope_id = f"{session_ids.agent_id}:{user_id}"
        salient = [
            segment for segment in segments if segment.salience >= self._config.salience_threshold
        ]
        if not salient:
            return []
        updates: list[BrainReflectionAutobiographyDraft] = []
        relationship_summary = "；".join(segment.summary for segment in salient[:3])
        updates.append(
            BrainReflectionAutobiographyDraft(
                entry_kind=BrainAutobiographyEntryKind.RELATIONSHIP_ARC.value,
                scope_type="relationship",
                scope_id=relationship_scope_id,
                rendered_summary=relationship_summary,
                content={
                    "cycle_id": cycle_id,
                    "summary": relationship_summary,
                    "salient_segment_ids": [segment.segment_id for segment in salient[:6]],
                    "interaction_shift": any(segment.correction_signal for segment in salient),
                },
                salience=max(segment.salience for segment in salient),
                source_episode_ids=tuple(
                    episode_id for segment in salient for episode_id in segment.episode_ids
                ),
                source_event_ids=tuple(
                    event_id for segment in salient for event_id in segment.source_event_ids
                ),
            )
        )
        if len(salient) >= 2:
            updates.append(
                BrainReflectionAutobiographyDraft(
                    entry_kind=BrainAutobiographyEntryKind.SHARED_HISTORY_SUMMARY.value,
                    scope_type="relationship",
                    scope_id=relationship_scope_id,
                    rendered_summary="；".join(segment.summary for segment in salient[:2]),
                    content={
                        "cycle_id": cycle_id,
                        "summary_points": [segment.summary for segment in salient[:4]],
                    },
                    salience=max(segment.salience for segment in salient[:2]),
                    source_episode_ids=tuple(
                        episode_id for segment in salient[:2] for episode_id in segment.episode_ids
                    ),
                    source_event_ids=tuple(
                        event_id for segment in salient[:2] for event_id in segment.source_event_ids
                    ),
                )
            )
        if decisions or any(segment.correction_signal for segment in salient):
            updates.append(
                BrainReflectionAutobiographyDraft(
                    entry_kind=BrainAutobiographyEntryKind.RELATIONSHIP_MILESTONE.value,
                    scope_type="relationship",
                    scope_id=relationship_scope_id,
                    rendered_summary="Memory continuity changed after a correction or reconciliation.",
                    content={
                        "cycle_id": cycle_id,
                        "reconciliation_actions": [asdict(item) for item in decisions[:4]],
                    },
                    salience=2.0,
                    source_claim_ids=tuple(
                        claim_id
                        for item in decisions
                        for claim_id in [item.prior_claim_id, item.target_claim_id]
                        if claim_id is not None
                    ),
                    append_only=True,
                    identity_key=cycle_id,
                )
            )
        for segment in salient:
            for project_key in segment.project_keys:
                updates.append(
                    BrainReflectionAutobiographyDraft(
                        entry_kind=BrainAutobiographyEntryKind.PROJECT_ARC.value,
                        scope_type="relationship",
                        scope_id=relationship_scope_id,
                        rendered_summary=segment.summary,
                        content={
                            "cycle_id": cycle_id,
                            "project_key": project_key,
                            "summary": segment.summary,
                            "segment_id": segment.segment_id,
                        },
                        salience=segment.salience,
                        source_episode_ids=segment.episode_ids,
                        source_event_ids=segment.source_event_ids,
                        append_only=True,
                        identity_key=project_key,
                    )
                )
        return updates

    def _build_health_findings(
        self,
        *,
        user_id: str,
        thread_id: str,
        segments: list[BrainReflectionSegment],
        current_claims: list[Any],
        decisions: list[BrainReflectionReconciliationDecision],
        commitments: list[Any],
    ) -> list[BrainReflectionHealthFinding]:
        findings: list[BrainReflectionHealthFinding] = []
        stale_claims = [
            claim
            for claim in current_claims
            if not getattr(claim, "is_historical", False) and claim.is_stale
        ]
        if stale_claims:
            findings.append(
                BrainReflectionHealthFinding(
                    code="stale_current_claims",
                    severity="warning",
                    summary="Some current claims are stale and may need reconfirmation.",
                    details={"claim_ids": [claim.claim_id for claim in stale_claims[:8]]},
                )
            )
        low_salience = [
            segment for segment in segments if segment.salience < self._config.salience_threshold
        ]
        if low_salience and not decisions:
            findings.append(
                BrainReflectionHealthFinding(
                    code="low_salience_noop_run",
                    severity="info",
                    summary="Recent context was low-salience and produced no durable updates.",
                    details={"segment_ids": [segment.segment_id for segment in low_salience[:8]]},
                )
            )
        uncertain_claims = [
            claim
            for claim in current_claims
            if claim.status == "uncertain" and not getattr(claim, "is_historical", False)
        ]
        if uncertain_claims:
            findings.append(
                BrainReflectionHealthFinding(
                    code="unresolved_claim_conflicts",
                    severity="warning",
                    summary="Some claims remain uncertain and may require operator review.",
                    details={"claim_ids": [claim.claim_id for claim in uncertain_claims[:8]]},
                )
            )
        recent_skips = self._store.list_reflection_cycles(
            user_id=user_id,
            thread_id=thread_id,
            statuses=("skipped",),
            limit=8,
        )
        busy_skips = [
            record
            for record in recent_skips
            if record.skip_reason in {"thread_active", "idle_grace_not_elapsed"}
        ]
        if len(busy_skips) >= 3:
            findings.append(
                BrainReflectionHealthFinding(
                    code="repeated_busy_skips",
                    severity="info",
                    summary="Reflection has been skipped repeatedly because the thread stayed active.",
                    details={"cycle_ids": [record.cycle_id for record in busy_skips[:8]]},
                )
            )
        failed_cycles = self._store.list_reflection_cycles(
            user_id=user_id,
            thread_id=thread_id,
            statuses=("failed",),
            limit=4,
        )
        if failed_cycles:
            findings.append(
                BrainReflectionHealthFinding(
                    code="recent_scheduler_failures",
                    severity="warning",
                    summary="Recent reflection cycles failed and may require operator attention.",
                    details={"cycle_ids": [record.cycle_id for record in failed_cycles]},
                )
            )
        if not segments and not commitments:
            findings.append(
                BrainReflectionHealthFinding(
                    code="no_new_segments",
                    severity="info",
                    summary="No new episodes or open commitments required reflection.",
                    details={},
                )
            )
        return findings

    def _apply_reconciliation(
        self,
        *,
        decisions: tuple[BrainReflectionReconciliationDecision, ...],
        source_event_id: str,
        event_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for decision in decisions:
            if decision.action == "reinforce_existing" and decision.target_claim_id:
                evidence = self._claims.record_evidence(
                    claim_id=decision.target_claim_id,
                    source_event_id=source_event_id,
                    evidence_summary=decision.evidence_summary or decision.reason,
                    evidence_json=decision.evidence_json,
                    source_episode_id=(
                        decision.source_episode_ids[0] if decision.source_episode_ids else None
                    ),
                )
                applied.append(
                    {
                        "action": decision.action,
                        "claim_id": decision.target_claim_id,
                        "evidence_id": evidence.evidence_id,
                        "reason": decision.reason,
                    }
                )
                continue
            if decision.action == "record_new":
                record = self._claims.record_claim(
                    source_event_id=source_event_id,
                    evidence_summary=decision.evidence_summary,
                    evidence_json=decision.evidence_json,
                    source_episode_id=(
                        decision.source_episode_ids[0] if decision.source_episode_ids else None
                    ),
                    event_context=event_context,
                    **decision.replacement_claim,
                )
                applied.append(
                    {
                        "action": decision.action,
                        "claim_id": record.claim_id,
                        "reason": decision.reason,
                    }
                )
                continue
            if decision.action == "supersede_existing" and decision.prior_claim_id:
                record = self._claims.supersede_claim(
                    decision.prior_claim_id,
                    decision.replacement_claim,
                    decision.reason,
                    source_event_id,
                    event_context=event_context,
                )
                applied.append(
                    {
                        "action": decision.action,
                        "prior_claim_id": decision.prior_claim_id,
                        "new_claim_id": record.claim_id,
                        "reason": decision.reason,
                    }
                )
                continue
            if decision.action == "record_uncertain":
                record = self._claims.record_claim(
                    source_event_id=source_event_id,
                    evidence_summary=decision.evidence_summary,
                    evidence_json=decision.evidence_json,
                    source_episode_id=(
                        decision.source_episode_ids[0] if decision.source_episode_ids else None
                    ),
                    event_context=event_context,
                    **decision.replacement_claim,
                )
                applied.append(
                    {
                        "action": decision.action,
                        "claim_id": record.claim_id,
                        "reason": decision.reason,
                    }
                )

        if any(item.get("action") in {"record_new", "supersede_existing"} for item in applied):
            self._store._refresh_user_core_block(
                user_id=str(event_context["user_id"]),
                agent_id=str(event_context["agent_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source_event_id=source_event_id,
            )
        return applied

    def _apply_block_updates(
        self,
        *,
        updates: tuple[BrainReflectionBlockUpdate, ...],
        source_event_id: str,
        event_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for update in updates:
            record = self._store.upsert_core_memory_block(
                block_kind=update.block_kind,
                scope_type=update.scope_type,
                scope_id=update.scope_id,
                content=update.content,
                source_event_id=source_event_id,
                event_context=event_context,
            )
            applied.append(
                {
                    "block_id": record.block_id,
                    "block_kind": record.block_kind,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "version": record.version,
                }
            )
        return applied

    def _apply_autobiography_updates(
        self,
        *,
        updates: tuple[BrainReflectionAutobiographyDraft, ...],
        source_event_id: str,
        event_context: dict[str, Any],
    ) -> list[str]:
        applied: list[str] = []
        for update in updates:
            record = self._autobiography.upsert_entry(
                scope_type=update.scope_type,
                scope_id=update.scope_id,
                entry_kind=update.entry_kind,
                rendered_summary=update.rendered_summary,
                content=update.content,
                salience=update.salience,
                source_episode_ids=list(update.source_episode_ids),
                source_claim_ids=list(update.source_claim_ids),
                source_event_ids=list(update.source_event_ids),
                source_event_id=source_event_id,
                append_only=update.append_only,
                identity_key=update.identity_key,
                event_context=event_context,
            )
            applied.append(record.entry_id)
        return applied

    def _persist_draft(self, draft: BrainReflectionDraft) -> Path:
        path = self._store.path.parent / "exports" / "reflection" / _draft_artifact_name(draft)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(draft.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


class BrainReflectionScheduler:
    """Periodic idle-gated reflection scheduler for one local runtime thread."""

    def __init__(
        self,
        *,
        store_path: str | Path,
        session_resolver,
        config: BrainReflectionSchedulerConfig | None = None,
        candidate_goal_sink=None,
        reevaluation_sink=None,
    ):
        """Bind the scheduler to one runtime store path and session resolver."""
        self._store_path = Path(store_path)
        self._session_resolver = session_resolver
        self._config = config or BrainReflectionSchedulerConfig()
        self._candidate_goal_sink = candidate_goal_sink
        self._reevaluation_sink = reevaluation_sink
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the periodic loop is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the periodic background loop."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="blink-reflection-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, final_catchup: bool = False) -> BrainReflectionRunResult | None:
        """Stop the periodic loop and optionally run one final catch-up cycle."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(self._config.wakeup_interval_secs + 1.0, 2.0))
        self._thread = None
        if final_catchup:
            return self.run_cycle(trigger="runtime_close", force=True)
        return None

    def set_candidate_goal_sink(self, sink):
        """Update the optional timer-maintenance candidate sink."""
        self._candidate_goal_sink = sink

    def set_reevaluation_sink(self, sink):
        """Update the optional timer-driven reevaluation sink."""
        self._reevaluation_sink = sink

    def run_cycle(
        self,
        *,
        trigger: str = "timer",
        force: bool = False,
    ) -> BrainReflectionRunResult:
        """Run one scheduled reflection attempt with idle gating and skip recording."""
        if not self._run_lock.acquire(blocking=False):
            return self._record_skip(trigger=trigger, skip_reason="single_flight_in_progress")
        try:
            from blink.brain.store import BrainStore

            store = BrainStore(path=self._store_path)
            try:
                session_ids = self._session_resolver()
                skip_reason = self._skip_reason(store=store, session_ids=session_ids, force=force)
                if skip_reason is not None:
                    result = self._record_skip(
                        trigger=trigger,
                        skip_reason=skip_reason,
                        store=store,
                        session_ids=session_ids,
                    )
                    self._maybe_emit_timer_candidate_for_skip(
                        store=store,
                        session_ids=session_ids,
                        trigger=trigger,
                        result=result,
                    )
                    self._maybe_trigger_timer_reevaluation(
                        store=store,
                        session_ids=session_ids,
                        trigger=trigger,
                        result=result,
                    )
                    return result
                result = BrainReflectionEngine(
                    store=store,
                    config=self._config.engine,
                ).run_once(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    session_ids=session_ids,
                    trigger=trigger,
                )
                self._maybe_emit_timer_candidate_for_completed_cycle(
                    store=store,
                    session_ids=session_ids,
                    trigger=trigger,
                    result=result,
                )
                self._maybe_trigger_timer_reevaluation(
                    store=store,
                    session_ids=session_ids,
                    trigger=trigger,
                    result=result,
                )
                return result
            finally:
                store.close()
        finally:
            self._run_lock.release()

    def _run_loop(self):
        while not self._stop_event.wait(self._config.wakeup_interval_secs):
            try:
                self.run_cycle(trigger="timer")
            except Exception:
                continue

    def _skip_reason(self, *, store, session_ids, force: bool) -> str | None:
        if not store.has_pending_reflection_work(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        ):
            return "no_new_work"
        if force:
            return None
        working = store.get_working_context_projection(scope_key=session_ids.thread_id)
        if working.user_turn_open or working.assistant_turn_open:
            return "thread_active"
        idle_age_secs = (datetime.now(UTC) - _parse_ts(working.updated_at)).total_seconds()
        if idle_age_secs < self._config.idle_grace_secs:
            return "idle_grace_not_elapsed"
        return None

    def _record_skip(
        self,
        *,
        trigger: str,
        skip_reason: str,
        store=None,
        session_ids=None,
    ) -> BrainReflectionRunResult:
        from blink.brain.store import BrainStore

        owns_store = store is None
        working_store = store or BrainStore(path=self._store_path)
        try:
            resolved_session_ids = session_ids or self._session_resolver()
            cycle_id = f"reflection_{uuid4().hex}"
            input_episode_cursor, input_event_cursor = working_store.latest_reflection_cursors(
                user_id=resolved_session_ids.user_id,
                thread_id=resolved_session_ids.thread_id,
            )
            cycle_record = working_store.skip_reflection_cycle(
                cycle_id=cycle_id,
                user_id=resolved_session_ids.user_id,
                thread_id=resolved_session_ids.thread_id,
                trigger=trigger,
                input_episode_cursor=input_episode_cursor,
                input_event_cursor=input_event_cursor,
                terminal_episode_cursor=input_episode_cursor,
                terminal_event_cursor=input_event_cursor,
                skip_reason=skip_reason,
                result_stats={"skip_reason": skip_reason},
            )
            working_store.append_brain_event(
                event_type=BrainEventType.REFLECTION_CYCLE_SKIPPED,
                agent_id=resolved_session_ids.agent_id,
                user_id=resolved_session_ids.user_id,
                session_id=resolved_session_ids.session_id,
                thread_id=resolved_session_ids.thread_id,
                source="reflection",
                correlation_id=cycle_id,
                payload={
                    "cycle_id": cycle_id,
                    "trigger": trigger,
                    "input_episode_cursor": input_episode_cursor,
                    "input_event_cursor": input_event_cursor,
                    "terminal_episode_cursor": input_episode_cursor,
                    "terminal_event_cursor": input_event_cursor,
                    "skip_reason": skip_reason,
                    "result_stats": {"skip_reason": skip_reason},
                },
            )
            return BrainReflectionRunResult(
                cycle_id=cycle_id,
                status="skipped",
                trigger=trigger,
                skip_reason=skip_reason,
                cycle_record=cycle_record,
            )
        finally:
            if owns_store:
                working_store.close()

    def _maybe_emit_timer_candidate_for_completed_cycle(
        self,
        *,
        store,
        session_ids,
        trigger: str,
        result: BrainReflectionRunResult,
    ):
        if self._candidate_goal_sink is None or trigger != "timer" or result.status != "completed":
            return
        relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
        report = store.latest_memory_health_report(
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if report is None:
            return
        has_actionable_finding = report.status == "needs_attention" or any(
            str(finding.get("severity", "")).strip() in {"warning", "critical"}
            for finding in report.findings
        )
        if not has_actionable_finding:
            return
        candidate = BrainCandidateGoal(
            candidate_goal_id=uuid4().hex,
            candidate_type="maintenance_review_findings",
            source=BrainCandidateGoalSource.TIMER.value,
            summary="Memory maintenance review found issues that need follow-up.",
            goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
            urgency=0.72,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
            cooldown_key=f"{session_ids.thread_id}:maintenance_review_findings",
            dedupe_key=f"{session_ids.thread_id}:maintenance_review_findings",
            policy_tags=["phase6b", "maintenance", "reflection_findings"],
            requires_user_turn_gap=True,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            payload={
                "goal_intent": "autonomy.maintenance_review_findings",
                "goal_details": {
                    "transient_only": True,
                    "maintenance": {
                        "report_id": report.report_id,
                        "cycle_id": report.cycle_id,
                        "status": report.status,
                        "findings": report.findings,
                    },
                },
            },
        )
        self._candidate_goal_sink(
            candidate_goal=candidate,
            source="reflection",
            correlation_id=result.cycle_id,
            store=store,
            session_ids=session_ids,
        )

    def _maybe_emit_timer_candidate_for_skip(
        self,
        *,
        store,
        session_ids,
        trigger: str,
        result: BrainReflectionRunResult,
    ):
        if (
            self._candidate_goal_sink is None
            or trigger != "timer"
            or result.status != "skipped"
            or result.skip_reason != "thread_active"
        ):
            return
        cutoff = datetime.now(UTC) - timedelta(minutes=10)
        recent_thread_active_skips = [
            record
            for record in store.list_reflection_cycles(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                statuses=("skipped",),
                limit=16,
            )
            if record.trigger == "timer"
            and record.skip_reason == "thread_active"
            and _parse_ts(record.started_at) >= cutoff
        ]
        if len(recent_thread_active_skips) < 3:
            return
        candidate = BrainCandidateGoal(
            candidate_goal_id=uuid4().hex,
            candidate_type="maintenance_thread_active_backpressure",
            source=BrainCandidateGoalSource.TIMER.value,
            summary="Reflection has been skipped repeatedly because the thread stayed active.",
            goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
            urgency=0.66,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
            cooldown_key=f"{session_ids.thread_id}:maintenance_thread_active_backpressure",
            dedupe_key=f"{session_ids.thread_id}:maintenance_thread_active_backpressure",
            policy_tags=["phase6b", "maintenance", "backpressure"],
            requires_user_turn_gap=True,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            payload={
                "goal_intent": "autonomy.maintenance_thread_active_backpressure",
                "goal_details": {
                    "transient_only": True,
                    "maintenance": {
                        "skip_reason": "thread_active",
                        "recent_skip_count": len(recent_thread_active_skips),
                    },
                },
            },
        )
        self._candidate_goal_sink(
            candidate_goal=candidate,
            source="reflection",
            correlation_id=result.cycle_id,
            store=store,
            session_ids=session_ids,
        )

    def _maybe_trigger_timer_reevaluation(
        self,
        *,
        store,
        session_ids,
        trigger: str,
        result: BrainReflectionRunResult,
    ):
        if self._reevaluation_sink is None or trigger != "timer":
            return
        source_event_type = (
            BrainEventType.REFLECTION_CYCLE_COMPLETED
            if result.status == "completed"
            else BrainEventType.REFLECTION_CYCLE_SKIPPED
        )
        source_event = store.latest_brain_event(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            event_types=(source_event_type,),
        )
        for condition_kind, summary in (
            (
                BrainReevaluationConditionKind.TIME_REACHED.value,
                "Reevaluate held candidates on the timer wake for time-based deadlines.",
            ),
            (
                BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
                "Reevaluate held maintenance candidates when the maintenance window may open.",
            ),
        ):
            self._reevaluation_sink(
                trigger=BrainReevaluationTrigger(
                    kind=condition_kind,
                    summary=summary,
                    details={
                        "reflection_trigger": trigger,
                        "cycle_id": result.cycle_id,
                        "reflection_status": result.status,
                    },
                    source_event_type=source_event.event_type if source_event is not None else None,
                    source_event_id=source_event.event_id if source_event is not None else None,
                    ts=source_event.ts
                    if source_event is not None
                    else datetime.now(UTC).isoformat(),
                ),
                store=store,
                session_ids=session_ids,
            )


# Compatibility aliases kept while call sites move to the engine/scheduler names.
BrainReflectionWorkerConfig = BrainReflectionEngineConfig
BrainReflectionClaimDecision = BrainReflectionReconciliationDecision
BrainReflectionAutobiographyUpdate = BrainReflectionAutobiographyDraft
BrainReflectionCycleResult = BrainReflectionRunResult
BrainReflectionWorker = BrainReflectionEngine

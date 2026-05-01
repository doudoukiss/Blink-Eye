"""Read-only user-facing memory-palace snapshot for Blink."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from blink.brain.memory_v2.claims import BrainClaimRecord, render_claim_summary
from blink.brain.memory_v2.core_blocks import (
    BrainCoreMemoryBlockKind,
    BrainCoreMemoryBlockRecord,
)
from blink.brain.memory_v2.use_trace import (
    BrainMemoryUseTrace,
    render_safe_memory_provenance_label,
)
from blink.brain.persona.schema import RelationshipStyleStateSpec, TeachingProfileStateSpec
from blink.brain.projections import (
    BrainClaimCurrentnessStatus,
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainGovernanceReasonCode,
)

_SCHEMA_VERSION = 1
_DISPLAY_ORDER = {
    "profile": 0,
    "preference": 1,
    "task": 2,
    "relationship_style": 3,
    "teaching_profile": 4,
    "claim": 5,
}
_HIDDEN_REASON_CODES = frozenset(
    {
        BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value,
        BrainGovernanceReasonCode.OPERATOR_HOLD.value,
    }
)
USER_PINNED_REASON_CODE = BrainGovernanceReasonCode.USER_PINNED.value


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _optional_text(value: Any) -> str | None:
    text = _normalized_text(value)
    return text or None


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(_normalized_text(part) for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _compact_join(values: Iterable[Any], *, fallback: str = "") -> str:
    items = _dedupe_preserve_order(values)
    return "; ".join(items) if items else fallback


@dataclass(frozen=True)
class BrainMemoryPalaceRecord:
    """One compact user-facing memory-palace record."""

    memory_id: str
    display_kind: str
    scope_type: str
    scope_id: str
    title: str
    summary: str
    status: str
    currentness_status: str | None
    review_state: str | None
    retention_class: str | None
    confidence: float | None
    pinned: bool
    suppressed: bool
    editable: bool
    forgettable: bool
    source_refs: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    last_used_at: str | None
    last_used_reason: str | None
    used_in_current_turn: bool
    safe_provenance_label: str
    user_actions: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the memory-palace record."""
        return {
            "memory_id": self.memory_id,
            "display_kind": self.display_kind,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "currentness_status": self.currentness_status,
            "review_state": self.review_state,
            "retention_class": self.retention_class,
            "confidence": self.confidence,
            "pinned": self.pinned,
            "suppressed": self.suppressed,
            "editable": self.editable,
            "forgettable": self.forgettable,
            "source_refs": list(self.source_refs),
            "source_event_ids": list(self.source_event_ids),
            "last_used_at": self.last_used_at,
            "last_used_reason": self.last_used_reason,
            "used_in_current_turn": self.used_in_current_turn,
            "safe_provenance_label": self.safe_provenance_label,
            "user_actions": list(self.user_actions),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainMemoryPalaceSnapshot:
    """Read-only user-facing memory-palace snapshot."""

    schema_version: int
    user_id: str
    agent_id: str
    generated_at: str
    records: tuple[BrainMemoryPalaceRecord, ...]
    hidden_counts: dict[str, int]
    health_summary: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the memory-palace snapshot."""
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "generated_at": self.generated_at,
            "records": [record.as_dict() for record in self.records],
            "hidden_counts": dict(self.hidden_counts),
            "health_summary": self.health_summary,
            "reason_codes": list(self.reason_codes),
        }


def _display_kind_for_claim(claim: BrainClaimRecord) -> str:
    predicate = _normalized_text(claim.predicate)
    if predicate.startswith("profile."):
        return "profile"
    if predicate.startswith("preference."):
        return "preference"
    return "claim"


def _claim_is_suppressed(claim: BrainClaimRecord) -> bool:
    if claim.effective_currentness_status == BrainClaimCurrentnessStatus.HELD.value:
        return True
    if claim.effective_review_state == BrainClaimReviewState.REQUESTED.value:
        return True
    return bool(set(claim.governance_reason_codes) & _HIDDEN_REASON_CODES)


def _claim_title(claim: BrainClaimRecord, display_kind: str) -> str:
    value = _optional_text(claim.object.get("value") or claim.object.get("summary"))
    predicate = _normalized_text(claim.predicate)
    if display_kind == "profile":
        return value or predicate.replace(".", " ").title()
    if display_kind == "preference":
        subject = _optional_text(claim.subject_entity_id)
        if value:
            return value
        return subject or predicate.replace(".", " ").title()
    return value or predicate.replace(".", " ").title()


def _claim_actions(
    *,
    suppressed: bool,
    currentness_status: str,
    pinned: bool,
) -> tuple[str, ...]:
    if suppressed:
        return ("review", "export")
    actions = ["review", "suppress", "correct", "forget", "export"]
    if not pinned:
        actions.insert(1, "pin")
    if currentness_status != BrainClaimCurrentnessStatus.STALE.value:
        actions.insert(len(actions) - 1, "mark_stale")
    return tuple(actions)


def _claim_record(claim: BrainClaimRecord) -> BrainMemoryPalaceRecord:
    display_kind = _display_kind_for_claim(claim)
    currentness = claim.effective_currentness_status
    review_state = claim.effective_review_state
    retention_class = claim.effective_retention_class
    suppressed = _claim_is_suppressed(claim)
    pinned = (
        retention_class == BrainClaimRetentionClass.DURABLE.value
        and USER_PINNED_REASON_CODE in claim.governance_reason_codes
    )
    pin_source = (
        "user"
        if pinned
        else "default_policy"
        if retention_class == BrainClaimRetentionClass.DURABLE.value
        else "none"
    )
    summary = render_claim_summary(claim)
    return BrainMemoryPalaceRecord(
        memory_id=f"memory_claim:{claim.scope_type or 'unknown'}:{claim.scope_id or 'unknown'}:{claim.claim_id}",
        display_kind=display_kind,
        scope_type=claim.scope_type or "unknown",
        scope_id=claim.scope_id or "unknown",
        title=_claim_title(claim, display_kind),
        summary=summary,
        status=claim.status,
        currentness_status=currentness,
        review_state=review_state,
        retention_class=retention_class,
        confidence=round(float(claim.confidence), 4),
        pinned=pinned,
        suppressed=suppressed,
        editable=not suppressed,
        forgettable=not suppressed,
        source_refs=(claim.claim_id,),
        source_event_ids=_dedupe_preserve_order([claim.source_event_id]),
        last_used_at=None,
        last_used_reason=None,
        used_in_current_turn=False,
        safe_provenance_label=render_safe_memory_provenance_label(
            display_kind=display_kind,
            currentness_status=currentness,
        ),
        user_actions=_claim_actions(
            suppressed=suppressed,
            currentness_status=currentness,
            pinned=pinned,
        ),
        reason_codes=_dedupe_preserve_order(
            (
                "source:claim",
                f"display_kind:{display_kind}",
                f"currentness:{currentness}",
                f"review:{review_state}",
                f"retention:{retention_class}",
                f"pin_source:{pin_source}",
                "visibility:suppressed" if suppressed else "visibility:visible",
                *claim.governance_reason_codes,
            )
        ),
    )


def _task_record(*, task: dict[str, Any], user_id: str) -> BrainMemoryPalaceRecord | None:
    title = _normalized_text(task.get("title"))
    if not title:
        return None
    status = _normalized_text(task.get("status")) or "open"
    commitment_id = _optional_text(task.get("commitment_id"))
    stable_ref = commitment_id or _stable_id(
        "task",
        user_id,
        title,
        status,
        task.get("updated_at") or task.get("created_at") or "",
    )
    details = task.get("details") if isinstance(task.get("details"), dict) else {}
    summary = _optional_text(details.get("summary")) or title
    source_refs = [commitment_id, task.get("current_goal_id")]
    updated_at = _optional_text(task.get("updated_at"))
    user_actions = (
        ("review", "mark_done", "cancel", "export") if commitment_id else ("review", "export")
    )
    return BrainMemoryPalaceRecord(
        memory_id=f"memory_task:user:{user_id}:{stable_ref}",
        display_kind="task",
        scope_type="user",
        scope_id=user_id,
        title=title,
        summary=summary,
        status=status,
        currentness_status=None,
        review_state=None,
        retention_class=None,
        confidence=None,
        pinned=False,
        suppressed=False,
        editable=True,
        forgettable=False,
        source_refs=_dedupe_preserve_order(source_refs),
        source_event_ids=(),
        last_used_at=updated_at,
        last_used_reason="task_updated" if updated_at else None,
        used_in_current_turn=False,
        safe_provenance_label=render_safe_memory_provenance_label(display_kind="task"),
        user_actions=user_actions,
        reason_codes=_dedupe_preserve_order(
            (
                "source:task",
                "task_ref:commitment" if commitment_id else "task_ref:stable_hash",
            )
        ),
    )


def _relationship_summary(content: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    try:
        style = RelationshipStyleStateSpec.model_validate(content)
    except (ValidationError, ValueError):
        collaboration = _optional_text(content.get("collaboration_style"))
        boundaries = (
            content.get("boundaries") if isinstance(content.get("boundaries"), list) else []
        )
        summary = _compact_join(
            (
                f"collaboration={collaboration}" if collaboration else "",
                f"boundaries={_compact_join(boundaries, fallback='bounded')}",
            ),
            fallback="Relationship style block is current.",
        )
        return summary, ("block_validation:unvalidated",)
    summary = _compact_join(
        (
            f"collaboration={style.collaboration_style}",
            f"challenge={style.challenge_style}",
            f"boundaries={_compact_join(style.boundaries, fallback='bounded')}",
        )
    )
    return summary, ("block_validation:validated",)


def _teaching_summary(content: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    try:
        profile = TeachingProfileStateSpec.model_validate(content)
    except (ValidationError, ValueError):
        mode = _optional_text(content.get("default_mode"))
        correction = _optional_text(content.get("correction_style"))
        summary = _compact_join(
            (
                f"mode={mode}" if mode else "",
                f"correction={correction}" if correction else "",
            ),
            fallback="Teaching profile block is current.",
        )
        return summary, ("block_validation:unvalidated",)
    mode = profile.preferred_modes[0] if profile.preferred_modes else profile.default_mode
    summary = _compact_join(
        (
            f"mode={mode}",
            f"examples={profile.example_density:.2f}",
            f"questions={profile.question_frequency:.2f}",
            f"correction={profile.correction_style}",
        )
    )
    return summary, ("block_validation:validated",)


def _block_record(
    *,
    block: BrainCoreMemoryBlockRecord,
    display_kind: str,
) -> BrainMemoryPalaceRecord:
    if display_kind == "relationship_style":
        title = "Relationship style"
        summary, validation_codes = _relationship_summary(block.content)
    else:
        title = "Teaching profile"
        summary, validation_codes = _teaching_summary(block.content)
    return BrainMemoryPalaceRecord(
        memory_id=f"memory_block:{block.scope_type}:{block.scope_id}:{block.block_kind}:{block.version}",
        display_kind=display_kind,
        scope_type=block.scope_type,
        scope_id=block.scope_id,
        title=title,
        summary=summary,
        status=block.status,
        currentness_status=BrainClaimCurrentnessStatus.CURRENT.value,
        review_state=None,
        retention_class=BrainClaimRetentionClass.DURABLE.value,
        confidence=None,
        pinned=False,
        suppressed=False,
        editable=False,
        forgettable=False,
        source_refs=(block.block_id,),
        source_event_ids=_dedupe_preserve_order([block.source_event_id]),
        last_used_at=None,
        last_used_reason=None,
        used_in_current_turn=False,
        safe_provenance_label=render_safe_memory_provenance_label(display_kind=display_kind),
        user_actions=("review", "export"),
        reason_codes=_dedupe_preserve_order(
            (
                "source:core_block",
                f"display_kind:{display_kind}",
                *validation_codes,
            )
        ),
    )


def render_public_memory_provenance(record: BrainMemoryPalaceRecord) -> str:
    """Return a public-safe provenance label for one Memory Palace record."""
    return render_safe_memory_provenance_label(
        display_kind=record.display_kind,
        currentness_status=record.currentness_status,
    )


def _health_summary(report: Any | None) -> tuple[str, tuple[str, ...], str | None]:
    if report is None:
        return "Memory health unavailable.", ("memory_health:missing",), None
    finding_count = len(getattr(report, "findings", []) or [])
    stats = getattr(report, "stats", {}) or {}
    stats_summary = _compact_join(
        (f"{key}={stats[key]}" for key in sorted(stats)[:3]),
        fallback="no_stats",
    )
    return (
        (
            f"Memory health {report.status}; score={float(report.score):.2f}; "
            f"findings={finding_count}; {stats_summary}"
        ),
        ("memory_health:available", f"memory_health_status:{report.status}"),
        _optional_text(getattr(report, "created_at", None)),
    )


def _record_sort_prefix(record: BrainMemoryPalaceRecord) -> tuple[int, str, str, str]:
    return (
        _DISPLAY_ORDER.get(record.display_kind, 99),
        record.title.lower(),
        record.status,
        record.currentness_status or "",
    )


def _record_source_ts(
    record: BrainMemoryPalaceRecord,
    source_timestamps_by_record: dict[str, list[str]],
) -> str:
    timestamps = source_timestamps_by_record.get(record.memory_id, [])
    return max(timestamps) if timestamps else ""


def _trace_sort_key(trace: BrainMemoryUseTrace) -> tuple[str, str]:
    return (trace.created_at or "", trace.summary)


def _trace_scope_matches(
    *,
    trace: BrainMemoryUseTrace,
    user_id: str,
    agent_id: str,
    thread_id: str,
) -> bool:
    if trace.user_id and trace.user_id != user_id:
        return False
    if trace.agent_id and trace.agent_id != agent_id:
        return False
    if trace.thread_id and thread_id and trace.thread_id != thread_id:
        return False
    return True


def _resolve_use_traces(
    *,
    store,
    user_id: str,
    agent_id: str,
    thread_id: str,
    current_turn_trace: BrainMemoryUseTrace | None,
    recent_use_traces: Iterable[BrainMemoryUseTrace] | None,
) -> tuple[BrainMemoryUseTrace | None, tuple[BrainMemoryUseTrace, ...], tuple[str, ...]]:
    reason_codes: list[str] = []
    if current_turn_trace is not None and not _trace_scope_matches(
        trace=current_turn_trace,
        user_id=user_id,
        agent_id=agent_id,
        thread_id=thread_id,
    ):
        current_turn_trace = None
        reason_codes.append("memory_use_trace_current_scope_mismatch")
    if recent_use_traces is None:
        reader = getattr(store, "recent_memory_use_traces", None)
        if callable(reader) and thread_id:
            recent_use_traces = reader(user_id=user_id, thread_id=thread_id, limit=8)
            reason_codes.append("memory_use_traces:store")
        else:
            recent_use_traces = ()
            reason_codes.append("memory_use_traces:unavailable")
    filtered = tuple(
        trace
        for trace in recent_use_traces
        if _trace_scope_matches(
            trace=trace,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
        )
    )
    if current_turn_trace is not None:
        reason_codes.append("memory_use_trace_current:available")
    return (
        current_turn_trace,
        tuple(sorted(filtered, key=_trace_sort_key, reverse=True)),
        tuple(reason_codes),
    )


def _annotate_records_with_use_traces(
    *,
    records: tuple[BrainMemoryPalaceRecord, ...],
    current_turn_trace: BrainMemoryUseTrace | None,
    recent_use_traces: tuple[BrainMemoryUseTrace, ...],
    source_timestamps_by_record: dict[str, list[str]],
) -> tuple[BrainMemoryPalaceRecord, ...]:
    current_ids = {
        ref.memory_id for ref in (current_turn_trace.refs if current_turn_trace is not None else ())
    }
    latest_ref_by_memory_id: dict[str, tuple[str, str]] = {}
    for trace in recent_use_traces:
        for ref in trace.refs:
            latest_ref_by_memory_id.setdefault(
                ref.memory_id,
                (trace.created_at, ref.used_reason),
            )
    if current_turn_trace is not None and current_turn_trace.created_at:
        for ref in current_turn_trace.refs:
            latest_ref_by_memory_id.setdefault(
                ref.memory_id,
                (current_turn_trace.created_at, ref.used_reason),
            )
    annotated: list[BrainMemoryPalaceRecord] = []
    for record in records:
        latest = latest_ref_by_memory_id.get(record.memory_id)
        last_used_at = record.last_used_at
        last_used_reason = record.last_used_reason
        safe_provenance_label = record.safe_provenance_label or render_public_memory_provenance(
            record
        )
        if latest is not None:
            trace_ts, trace_reason = latest
            last_used_at = trace_ts or last_used_at
            last_used_reason = trace_reason or last_used_reason
            if trace_ts:
                source_timestamps_by_record.setdefault(record.memory_id, []).append(trace_ts)
        annotated.append(
            replace(
                record,
                last_used_at=last_used_at,
                last_used_reason=last_used_reason,
                used_in_current_turn=record.memory_id in current_ids,
                safe_provenance_label=safe_provenance_label,
            )
        )
    return tuple(annotated)


def build_memory_palace_snapshot(
    *,
    store,
    session_ids,
    include_suppressed: bool = False,
    include_historical: bool = False,
    current_turn_trace: BrainMemoryUseTrace | None = None,
    recent_use_traces: Iterable[BrainMemoryUseTrace] | None = None,
    limit: int = 80,
    claim_scan_limit: int | None = None,
) -> BrainMemoryPalaceSnapshot:
    """Build a pure read-only memory-palace snapshot from existing store surfaces."""
    user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    agent_id = _normalized_text(getattr(session_ids, "agent_id", "")) or "blink/main"
    thread_id = _normalized_text(getattr(session_ids, "thread_id", ""))
    relationship_scope_id = f"{agent_id}:{user_id}"
    hidden_counts = {"suppressed": 0, "historical": 0, "limit": 0}
    reason_codes: list[str] = ["memory_palace:v1", "read_model:store"]
    records: list[BrainMemoryPalaceRecord] = []
    source_timestamps_by_record: dict[str, list[str]] = {}
    current_turn_trace, resolved_use_traces, use_trace_reason_codes = _resolve_use_traces(
        store=store,
        user_id=user_id,
        agent_id=agent_id,
        thread_id=thread_id,
        current_turn_trace=current_turn_trace,
        recent_use_traces=recent_use_traces,
    )
    reason_codes.extend(use_trace_reason_codes)

    bounded_claim_scan_limit: int | None = None
    if claim_scan_limit is not None:
        bounded_claim_scan_limit = max(0, int(claim_scan_limit))
        reason_codes.extend(
            ("claims_scan:bounded", f"claims_scan_limit:{bounded_claim_scan_limit}")
        )

    claims = store.query_claims(
        temporal_mode="all",
        scope_type="user",
        scope_id=user_id,
        limit=bounded_claim_scan_limit,
    )
    for claim in claims:
        historical = (
            claim.effective_currentness_status == BrainClaimCurrentnessStatus.HISTORICAL.value
        )
        if historical and not include_historical:
            hidden_counts["historical"] += 1
            continue
        suppressed = _claim_is_suppressed(claim)
        if suppressed and not include_suppressed:
            hidden_counts["suppressed"] += 1
            continue
        record = _claim_record(claim)
        records.append(record)
        source_timestamps_by_record[record.memory_id] = [
            timestamp
            for timestamp in (claim.updated_at, claim.created_at, claim.governance_updated_at)
            if timestamp
        ]

    active_tasks = getattr(store, "active_tasks", None)
    if callable(active_tasks):
        for task in active_tasks(user_id=user_id, limit=max(0, int(limit or 0)) or 80):
            record = _task_record(task=dict(task), user_id=user_id)
            if record is not None:
                records.append(record)
                source_timestamps_by_record[record.memory_id] = [
                    timestamp
                    for timestamp in (task.get("updated_at"), task.get("created_at"))
                    if timestamp
                ]
        reason_codes.append("tasks:available")
    else:
        reason_codes.append("tasks:unavailable")

    for block_kind, display_kind in (
        (BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value, "relationship_style"),
        (BrainCoreMemoryBlockKind.TEACHING_PROFILE.value, "teaching_profile"),
    ):
        block = store.get_current_core_memory_block(
            block_kind=block_kind,
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        if block is None:
            reason_codes.append(f"{display_kind}:missing")
            continue
        record = _block_record(block=block, display_kind=display_kind)
        records.append(record)
        source_timestamps_by_record[record.memory_id] = [
            timestamp for timestamp in (block.updated_at, block.created_at) if timestamp
        ]
        reason_codes.append(f"{display_kind}:available")

    latest_health = getattr(store, "latest_memory_health_report", None)
    health_report = (
        latest_health(scope_type="user", scope_id=user_id) if callable(latest_health) else None
    )
    health_summary, health_reason_codes, health_generated_at = _health_summary(health_report)
    reason_codes.extend(health_reason_codes)

    sorted_records = tuple(
        sorted(
            records,
            key=lambda record: (
                *_record_sort_prefix(record),
                _record_source_ts(record, source_timestamps_by_record),
                record.memory_id,
            ),
        )
    )
    sorted_records = _annotate_records_with_use_traces(
        records=sorted_records,
        current_turn_trace=current_turn_trace,
        recent_use_traces=resolved_use_traces,
        source_timestamps_by_record=source_timestamps_by_record,
    )
    bounded_limit = max(0, int(limit))
    if bounded_limit and len(sorted_records) > bounded_limit:
        hidden_counts["limit"] = len(sorted_records) - bounded_limit
        sorted_records = sorted_records[:bounded_limit]
    elif bounded_limit == 0 and sorted_records:
        hidden_counts["limit"] = len(sorted_records)
        sorted_records = ()

    generated_timestamps = [
        timestamp
        for record in sorted_records
        for timestamp in source_timestamps_by_record.get(record.memory_id, [])
    ]
    if health_generated_at:
        generated_timestamps.append(health_generated_at)
    generated_at = max(generated_timestamps) if generated_timestamps else ""
    return BrainMemoryPalaceSnapshot(
        schema_version=_SCHEMA_VERSION,
        user_id=user_id,
        agent_id=agent_id,
        generated_at=generated_at,
        records=sorted_records,
        hidden_counts=hidden_counts,
        health_summary=health_summary,
        reason_codes=_dedupe_preserve_order(reason_codes),
    )


__all__ = [
    "BrainMemoryPalaceRecord",
    "BrainMemoryPalaceSnapshot",
    "build_memory_palace_snapshot",
    "render_public_memory_provenance",
]

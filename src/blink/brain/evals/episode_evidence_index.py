"""Public-safe recent episode evidence index for operator surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any, Iterable
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from blink.brain.evals.episode_export import BrainEpisodeRecord
from blink.brain.evals.practice_episode_export import build_episodes_from_practice_plan_payload

_SCHEMA_VERSION = 1
_MAX_ROWS = 24
_MAX_LINKS = 10
_MAX_ARTIFACT_REFS = 8
_MAX_REASON_CODES = 16
_REDACTION_MARKERS = (
    "Traceback",
    "RuntimeError",
    "/tmp",
    ".db",
    "raw_json",
    "source_event",
    "prompt_text",
    "private_working_memory",
    "artifact_paths",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _public_text(value: Any, *, limit: int = 160) -> str:
    text = _text(value)
    if not text:
        return ""
    if any(marker in text for marker in _REDACTION_MARKERS):
        return "redacted"
    return text[:limit]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dedupe(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _public_text(value, limit=96)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(_text(part) for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _as_dict(value: Any) -> dict[str, Any]:
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        payload = serializer()
        return dict(payload) if isinstance(payload, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _sorted_count_map(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = _public_text(value, limit=96)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _reason_categories(reason_codes: Iterable[Any]) -> tuple[str, ...]:
    return _dedupe(str(code).split(":", 1)[0] for code in reason_codes)


def _artifact_uri_kind(uri: Any) -> str:
    text = _text(uri)
    if not text:
        return "unavailable"
    parsed = urlparse(text)
    if parsed.scheme:
        return f"{parsed.scheme}_uri"
    suffix = PurePath(text).suffix.lower().lstrip(".")
    return f"{suffix}_file" if suffix else "path"


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _origin_sort_key(row: "BrainEpisodeEvidenceRow") -> tuple[str, str, str, str]:
    recent = row.generated_at or row.ended_at or row.started_at
    return (recent, row.source, row.scenario_family, row.evidence_id)


@dataclass(frozen=True)
class BrainEpisodeEvidenceArtifactRef:
    """One public-safe artifact reference without raw artifact contents or paths."""

    artifact_id: str
    artifact_kind: str
    uri_kind: str
    redacted_uri: bool = True
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the artifact reference."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "uri_kind": self.uri_kind,
            "redacted_uri": self.redacted_uri,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainEpisodeEvidenceLink:
    """One typed link from evidence to a rollout, practice, or benchmark object."""

    link_kind: str
    link_id: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the link."""
        return {
            "link_kind": self.link_kind,
            "link_id": self.link_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainEpisodeEvidenceRow:
    """One compact public-safe evidence row for recent live, practice, or eval artifacts."""

    evidence_id: str
    episode_id: str
    source: str
    scenario_id: str
    scenario_family: str
    scenario_version: str
    summary: str
    source_run_id: str | None = None
    execution_backend: str | None = None
    candidate_backend_id: str | None = None
    candidate_backend_version: str | None = None
    outcome_label: str = "unknown"
    task_success: bool | None = None
    safety_success: bool | None = None
    preview_only: bool = False
    scenario_count: int = 1
    artifact_refs: tuple[BrainEpisodeEvidenceArtifactRef, ...] = ()
    links: tuple[BrainEpisodeEvidenceLink, ...] = ()
    started_at: str | None = None
    ended_at: str | None = None
    generated_at: str | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def reason_code_categories(self) -> tuple[str, ...]:
        """Return stable reason-code categories."""
        return _reason_categories(self.reason_codes)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence row without raw artifact internals."""
        return {
            "evidence_id": self.evidence_id,
            "episode_id": self.episode_id,
            "source": self.source,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "scenario_version": self.scenario_version,
            "summary": self.summary,
            "source_run_id": self.source_run_id,
            "execution_backend": self.execution_backend,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "outcome_label": self.outcome_label,
            "task_success": self.task_success,
            "safety_success": self.safety_success,
            "preview_only": self.preview_only,
            "scenario_count": self.scenario_count,
            "artifact_refs": [record.as_dict() for record in self.artifact_refs],
            "links": [record.as_dict() for record in self.links],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "generated_at": self.generated_at,
            "reason_codes": list(self.reason_codes),
            "reason_code_categories": list(self.reason_code_categories),
        }


@dataclass(frozen=True)
class BrainEpisodeEvidenceSnapshot:
    """Public-safe recent evidence index snapshot."""

    schema_version: int
    available: bool
    generated_at: str
    rows: tuple[BrainEpisodeEvidenceRow, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def source_counts(self) -> dict[str, int]:
        """Return row counts by evidence source."""
        return _sorted_count_map(row.source for row in self.rows)

    @property
    def reason_code_counts(self) -> dict[str, int]:
        """Return row reason-code counts."""
        return _sorted_count_map(code for row in self.rows for code in row.reason_codes)

    @property
    def summary(self) -> str:
        """Return a compact operator-facing summary."""
        if not self.available:
            return "Episode evidence unavailable."
        if not self.rows:
            return "No recent episode evidence."
        counts = ", ".join(f"{source}={count}" for source, count in self.source_counts.items())
        return f"{len(self.rows)} recent evidence rows" + (f": {counts}" if counts else ".")

    def as_dict(self) -> dict[str, Any]:
        """Serialize the snapshot."""
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "episode_count": len(self.rows),
            "source_counts": self.source_counts,
            "reason_code_counts": self.reason_code_counts,
            "rows": [record.as_dict() for record in self.rows],
            "reason_codes": list(self.reason_codes),
        }


def _links_for_ids(
    *,
    link_kind: str,
    ids: Iterable[Any],
    reason_code: str,
) -> tuple[BrainEpisodeEvidenceLink, ...]:
    return tuple(
        BrainEpisodeEvidenceLink(
            link_kind=link_kind,
            link_id=link_id,
            reason_codes=(reason_code,),
        )
        for link_id in _dedupe(ids)
    )[:_MAX_LINKS]


def _links_by_report_id(
    *,
    adapter_projection: Any,
    rollout_controller: Any | None,
) -> dict[str, list[BrainEpisodeEvidenceLink]]:
    links: dict[str, list[BrainEpisodeEvidenceLink]] = {}
    for decision in list(getattr(adapter_projection, "recent_decisions", []) or []):
        payload = _as_dict(decision)
        report_id = _public_text(payload.get("report_id"), limit=96)
        decision_id = _public_text(payload.get("decision_id"), limit=96)
        if not report_id or not decision_id:
            continue
        links.setdefault(report_id, []).append(
            BrainEpisodeEvidenceLink(
                link_kind="adapter_promotion_decision",
                link_id=decision_id,
                reason_codes=("evidence_link:adapter_promotion_decision",),
            )
        )
    for plan in list(getattr(rollout_controller, "plans", []) or []):
        payload = _as_dict(plan)
        plan_id = _public_text(payload.get("plan_id"), limit=96)
        if not plan_id:
            continue
        for report_id in _dedupe(payload.get("sim_to_real_report_ids") or ()):
            links.setdefault(report_id, []).append(
                BrainEpisodeEvidenceLink(
                    link_kind="rollout_plan",
                    link_id=plan_id,
                    reason_codes=("evidence_link:rollout_plan_report",),
                )
            )
    return {
        key: sorted(value, key=lambda item: (item.link_kind, item.link_id))[:_MAX_LINKS]
        for key, value in links.items()
    }


def _artifact_refs_from_episode(
    episode: BrainEpisodeRecord,
) -> tuple[BrainEpisodeEvidenceArtifactRef, ...]:
    refs: list[BrainEpisodeEvidenceArtifactRef] = []
    for artifact in episode.artifact_refs[:_MAX_ARTIFACT_REFS]:
        refs.append(
            BrainEpisodeEvidenceArtifactRef(
                artifact_id=_public_text(artifact.artifact_id, limit=96),
                artifact_kind=_public_text(artifact.artifact_kind, limit=64),
                uri_kind=_artifact_uri_kind(artifact.uri),
                reason_codes=("artifact_uri:redacted",),
            )
        )
    return tuple(ref for ref in refs if ref.artifact_id and ref.artifact_kind)


def _outcome_label(episode: BrainEpisodeRecord) -> str:
    if episode.outcome_summary.task_success is True:
        return "task_success"
    if episode.outcome_summary.task_success is False:
        return "task_failed"
    for candidate in (
        episode.outcome_summary.goal_status,
        episode.outcome_summary.planning_outcome,
        episode.action_summary.trace_status,
    ):
        text = _public_text(candidate, limit=64)
        if text:
            return text
    return "unknown"


def _episode_reason_codes(episode: BrainEpisodeRecord, *, source: str) -> tuple[str, ...]:
    reason_codes: list[str] = [
        "episode_evidence_row:v1",
        f"episode_source:{source}",
        f"scenario_family:{_public_text(episode.scenario_family, limit=64) or 'unknown'}",
    ]
    if episode.action_summary.preview_only:
        reason_codes.append("preview_only:true")
    if episode.outcome_summary.task_success is True:
        reason_codes.append("task_success:true")
    elif episode.outcome_summary.task_success is False:
        reason_codes.append("task_success:false")
    if episode.safety_summary.safety_success is True:
        reason_codes.append("safety_success:true")
    elif episode.safety_summary.safety_success is False:
        reason_codes.append("safety_success:false")
    reason_codes.extend(episode.safety_summary.risk_codes[:_MAX_REASON_CODES])
    reason_codes.extend(episode.safety_summary.mismatch_codes[:_MAX_REASON_CODES])
    reason_codes.extend(episode.safety_summary.repair_codes[:_MAX_REASON_CODES])
    return _dedupe(reason_codes)[:_MAX_REASON_CODES]


def _row_from_canonical_episode(
    episode: BrainEpisodeRecord,
    *,
    source: str,
    links: Iterable[BrainEpisodeEvidenceLink] = (),
) -> BrainEpisodeEvidenceRow:
    scenario_family = _public_text(episode.scenario_family, limit=96) or "unknown"
    scenario_id = _public_text(episode.scenario_id, limit=96) or scenario_family
    row_links = list(links)
    row_links.extend(
        _links_for_ids(
            link_kind="practice_plan",
            ids=(episode.provenance_ids.get("practice_plan_id"),),
            reason_code="evidence_link:practice_plan",
        )
    )
    evidence_id = _stable_id("episode_evidence", source, episode.episode_id)
    summary = " | ".join(
        value
        for value in (
            scenario_family,
            _public_text(episode.execution_backend, limit=64),
            _outcome_label(episode),
        )
        if value
    )
    return BrainEpisodeEvidenceRow(
        evidence_id=evidence_id,
        episode_id=_public_text(episode.episode_id, limit=128) or evidence_id,
        source=source,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version=_public_text(episode.scenario_version, limit=64) or "v1",
        summary=summary or "episode evidence",
        source_run_id=_public_text(episode.source_run_id, limit=96) or None,
        execution_backend=_public_text(episode.execution_backend, limit=96) or None,
        outcome_label=_outcome_label(episode),
        task_success=episode.outcome_summary.task_success,
        safety_success=episode.safety_summary.safety_success,
        preview_only=episode.action_summary.preview_only,
        artifact_refs=_artifact_refs_from_episode(episode),
        links=tuple(sorted(row_links, key=lambda item: (item.link_kind, item.link_id)))[:_MAX_LINKS],
        started_at=_public_text(episode.started_at, limit=96) or None,
        ended_at=_public_text(episode.ended_at, limit=96) or None,
        generated_at=_public_text(episode.generated_at, limit=96) or None,
        reason_codes=_episode_reason_codes(episode, source=source),
    )


def _row_from_live_turn(record: Any) -> BrainEpisodeEvidenceRow | None:
    episode_db_id = _safe_int(getattr(record, "id", 0))
    if episode_db_id <= 0:
        return None
    created_at = _public_text(getattr(record, "created_at", ""), limit=96)
    assistant_summary = _public_text(getattr(record, "assistant_summary", ""), limit=160)
    episode_id = f"live_turn:{episode_db_id}"
    evidence_id = _stable_id("episode_evidence", "live", episode_id)
    return BrainEpisodeEvidenceRow(
        evidence_id=evidence_id,
        episode_id=episode_id,
        source="live",
        scenario_id="live_runtime_turn",
        scenario_family="live_runtime_turn",
        scenario_version="store.turn.v1",
        summary=assistant_summary or "live turn recorded",
        source_run_id=_public_text(getattr(record, "session_id", ""), limit=96) or None,
        execution_backend="runtime_store",
        outcome_label="recorded",
        scenario_count=1,
        started_at=created_at or None,
        ended_at=created_at or None,
        generated_at=created_at or None,
        reason_codes=(
            "episode_evidence_row:v1",
            "episode_source:live",
            "live_turn:stored",
        ),
    )


def _rows_from_recent_live_turns(
    *,
    store: Any,
    session_ids: Any,
    recent_limit: int,
) -> tuple[BrainEpisodeEvidenceRow, ...]:
    reader = getattr(store, "recent_episodes", None)
    if not callable(reader):
        return ()
    rows: list[BrainEpisodeEvidenceRow] = []
    for record in reader(
        user_id=getattr(session_ids, "user_id", ""),
        thread_id=getattr(session_ids, "thread_id", ""),
        limit=recent_limit,
    ):
        row = _row_from_live_turn(record)
        if row is not None:
            rows.append(row)
    return tuple(rows)


def _fallback_practice_plan_row(record: Any) -> BrainEpisodeEvidenceRow | None:
    payload = _as_dict(record)
    plan_id = _public_text(payload.get("plan_id"), limit=96)
    if not plan_id:
        return None
    reason_counts = dict(payload.get("reason_code_counts") or {})
    reason_codes = _dedupe(
        (
            "episode_evidence_row:v1",
            "episode_source:practice",
            "practice_plan:fallback",
            *reason_counts.keys(),
        )
    )[:_MAX_REASON_CODES]
    evidence_id = _stable_id("episode_evidence", "practice", plan_id)
    return BrainEpisodeEvidenceRow(
        evidence_id=evidence_id,
        episode_id=f"practice_plan:{plan_id}",
        source="practice",
        scenario_id="practice_plan",
        scenario_family="practice_plan",
        scenario_version="practice.plan.v1",
        summary=_public_text(payload.get("summary"), limit=160) or "practice plan",
        source_run_id=plan_id,
        execution_backend="simulation",
        outcome_label="planned",
        preview_only=True,
        scenario_count=max(1, len(_list(payload.get("targets")))),
        links=(
            BrainEpisodeEvidenceLink(
                link_kind="practice_plan",
                link_id=plan_id,
                reason_codes=("evidence_link:practice_plan",),
            ),
        ),
        generated_at=_public_text(payload.get("updated_at"), limit=96) or None,
        reason_codes=reason_codes,
    )


def _rows_from_practice_projection(
    *,
    practice_projection: Any,
    recent_limit: int,
) -> tuple[BrainEpisodeEvidenceRow, ...]:
    rows: list[BrainEpisodeEvidenceRow] = []
    for plan in list(getattr(practice_projection, "recent_plans", []) or [])[:recent_limit]:
        payload = _as_dict(plan)
        try:
            episodes = build_episodes_from_practice_plan_payload(payload)
        except (TypeError, ValueError, AttributeError):
            fallback = _fallback_practice_plan_row(plan)
            if fallback is not None:
                rows.append(fallback)
            continue
        rows.extend(
            _row_from_canonical_episode(episode, source="practice") for episode in episodes
        )
    return tuple(rows[:recent_limit])


def _rows_from_adapter_projection(
    *,
    adapter_projection: Any,
    rollout_controller: Any | None,
    recent_limit: int,
) -> tuple[BrainEpisodeEvidenceRow, ...]:
    links_by_report = _links_by_report_id(
        adapter_projection=adapter_projection,
        rollout_controller=rollout_controller,
    )
    rows: list[BrainEpisodeEvidenceRow] = []
    for report in list(getattr(adapter_projection, "recent_reports", []) or [])[:recent_limit]:
        payload = _as_dict(report)
        report_id = _public_text(payload.get("report_id"), limit=96)
        if not report_id:
            continue
        target_families = _dedupe(payload.get("target_families") or ())
        scenario_family = target_families[0] if len(target_families) == 1 else (
            _public_text(payload.get("adapter_family"), limit=64) or "adapter_benchmark"
        )
        reason_codes = _dedupe(
            (
                "episode_evidence_row:v1",
                "episode_source:eval",
                f"adapter_family:{payload.get('adapter_family')}",
                "benchmark_passed:true"
                if payload.get("benchmark_passed") is True
                else "benchmark_passed:false"
                if payload.get("benchmark_passed") is False
                else "benchmark_passed:unknown",
                *list(payload.get("blocked_reason_codes") or ()),
            )
        )[:_MAX_REASON_CODES]
        links = [
            BrainEpisodeEvidenceLink(
                link_kind="benchmark_report",
                link_id=report_id,
                reason_codes=("evidence_link:benchmark_report",),
            ),
            *links_by_report.get(report_id, []),
        ]
        episode_id = f"eval_report:{report_id}"
        rows.append(
            BrainEpisodeEvidenceRow(
                evidence_id=_stable_id("episode_evidence", "eval", report_id),
                episode_id=episode_id,
                source="eval",
                scenario_id=report_id,
                scenario_family=scenario_family,
                scenario_version="adapter.benchmark.v1",
                summary=_public_text(payload.get("summary"), limit=160)
                or (
                    f"{_public_text(payload.get('adapter_family'), limit=64) or 'adapter'} "
                    "benchmark report"
                ),
                execution_backend="benchmark",
                candidate_backend_id=_public_text(payload.get("candidate_backend_id"), limit=96)
                or None,
                candidate_backend_version=_public_text(
                    payload.get("candidate_backend_version"), limit=64
                )
                or None,
                outcome_label=(
                    "benchmark_passed"
                    if payload.get("benchmark_passed") is True
                    else "benchmark_blocked"
                    if payload.get("benchmark_passed") is False
                    else "benchmark_recorded"
                ),
                task_success=_bool_or_none(payload.get("benchmark_passed")),
                safety_success=_bool_or_none(payload.get("smoke_suite_green")),
                scenario_count=max(0, _safe_int(payload.get("scenario_count"))),
                links=tuple(sorted(links, key=lambda item: (item.link_kind, item.link_id)))[
                    :_MAX_LINKS
                ],
                generated_at=_public_text(payload.get("updated_at"), limit=96) or None,
                reason_codes=reason_codes,
            )
        )
    return tuple(rows)


def _build_store_rows(
    *,
    store: Any,
    session_ids: Any,
    presence_scope_key: str,
    rollout_controller: Any | None,
    recent_limit: int,
) -> tuple[BrainEpisodeEvidenceRow, ...]:
    rows: list[BrainEpisodeEvidenceRow] = []
    rows.extend(
        _rows_from_recent_live_turns(
            store=store,
            session_ids=session_ids,
            recent_limit=recent_limit,
        )
    )

    practice_builder = getattr(store, "build_practice_director_projection", None)
    if callable(practice_builder):
        practice_projection = practice_builder(
            user_id=getattr(session_ids, "user_id", ""),
            thread_id=getattr(session_ids, "thread_id", ""),
            agent_id=getattr(session_ids, "agent_id", ""),
            presence_scope_key=presence_scope_key,
        )
        rows.extend(
            _rows_from_practice_projection(
                practice_projection=practice_projection,
                recent_limit=recent_limit,
            )
        )

    adapter_builder = getattr(store, "build_adapter_governance_projection", None)
    if callable(adapter_builder):
        adapter_projection = adapter_builder(
            user_id=getattr(session_ids, "user_id", ""),
            thread_id=getattr(session_ids, "thread_id", ""),
            agent_id=getattr(session_ids, "agent_id", ""),
        )
        rows.extend(
            _rows_from_adapter_projection(
                adapter_projection=adapter_projection,
                rollout_controller=rollout_controller,
                recent_limit=recent_limit,
            )
        )

    deduped: dict[str, BrainEpisodeEvidenceRow] = {}
    for row in sorted(rows, key=_origin_sort_key, reverse=True):
        deduped[row.evidence_id] = row
    return tuple(sorted(deduped.values(), key=_origin_sort_key, reverse=True))[:recent_limit]


def build_episode_evidence_index(
    *,
    store: Any,
    session_ids: Any,
    presence_scope_key: str = "",
    rollout_controller: Any | None = None,
    recent_limit: int = 8,
    generated_at: str = "",
) -> BrainEpisodeEvidenceSnapshot:
    """Build a deterministic public-safe evidence index from recent runtime projections."""
    limit = max(0, min(_MAX_ROWS, int(recent_limit)))
    if store is None or session_ids is None:
        return BrainEpisodeEvidenceSnapshot(
            schema_version=_SCHEMA_VERSION,
            available=False,
            generated_at=_public_text(generated_at, limit=96) or _utc_now(),
            rows=(),
            reason_codes=("episode_evidence:unavailable", "runtime_evidence_surface_missing"),
        )
    rows = _build_store_rows(
        store=store,
        session_ids=session_ids,
        presence_scope_key=_public_text(presence_scope_key, limit=96) or "local:presence",
        rollout_controller=rollout_controller,
        recent_limit=limit,
    )
    return BrainEpisodeEvidenceSnapshot(
        schema_version=_SCHEMA_VERSION,
        available=True,
        generated_at=_public_text(generated_at, limit=96) or _utc_now(),
        rows=rows,
        reason_codes=_dedupe(
            (
                "episode_evidence:v1",
                "episode_evidence:available",
                f"episode_evidence_count:{len(rows)}",
                *(f"episode_source:{source}" for source in _sorted_count_map(row.source for row in rows)),
            )
        ),
    )


__all__ = [
    "BrainEpisodeEvidenceArtifactRef",
    "BrainEpisodeEvidenceLink",
    "BrainEpisodeEvidenceRow",
    "BrainEpisodeEvidenceSnapshot",
    "build_episode_evidence_index",
]

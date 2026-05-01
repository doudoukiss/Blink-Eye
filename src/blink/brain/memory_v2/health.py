"""Inspectable memory health reports for Blink continuity memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventType


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


@dataclass(frozen=True)
class BrainMemoryHealthReportRecord:
    """One inspectable memory-health report row."""

    report_id: str
    scope_type: str
    scope_id: str
    cycle_id: str
    score: float
    status: str
    findings_json: str
    stats_json: str
    artifact_path: str | None
    created_at: str

    @property
    def findings(self) -> list[dict[str, Any]]:
        """Return decoded health findings."""
        return list(json.loads(self.findings_json))

    @property
    def stats(self) -> dict[str, Any]:
        """Return decoded health stats."""
        return dict(json.loads(self.stats_json))


class MemoryHealthService:
    """Typed health-report persistence on top of ``BrainStore``."""

    def __init__(self, *, store):
        """Bind the service to one canonical store."""
        self._store = store

    def record_report(
        self,
        *,
        scope_type: str,
        scope_id: str,
        cycle_id: str,
        score: float,
        status: str,
        findings: list[dict[str, Any]],
        stats: dict[str, Any],
        artifact_path: str | None = None,
        source_event_id: str | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> BrainMemoryHealthReportRecord:
        """Persist one inspectable memory-health report."""
        now = self._store._utc_now_for_memory_v2()
        report_id = _stable_id("memory_health", scope_type, scope_id, cycle_id)
        self._store._conn.execute(
            """
            INSERT OR REPLACE INTO memory_health_reports (
                report_id, scope_type, scope_id, cycle_id, score, status, findings_json,
                stats_json, artifact_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                scope_type,
                scope_id,
                cycle_id,
                max(0.0, min(1.0, float(score))),
                status,
                json.dumps(findings, ensure_ascii=False, sort_keys=True),
                json.dumps(stats, ensure_ascii=False, sort_keys=True),
                artifact_path,
                now,
            ),
        )
        self._store._conn.commit()
        record = self.latest_report(scope_type=scope_type, scope_id=scope_id)
        if record is None:
            raise RuntimeError("Failed to persist memory health report.")
        if event_context is not None:
            self._store.append_brain_event(
                event_type=BrainEventType.MEMORY_HEALTH_REPORTED,
                agent_id=str(event_context["agent_id"]),
                user_id=str(event_context["user_id"]),
                session_id=str(event_context["session_id"]),
                thread_id=str(event_context["thread_id"]),
                source=str(event_context.get("source", "memory_v2")),
                correlation_id=event_context.get("correlation_id"),
                causal_parent_id=source_event_id,
                payload={
                    "report_id": record.report_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "cycle_id": record.cycle_id,
                    "score": record.score,
                    "status": record.status,
                    "findings": record.findings,
                    "stats": record.stats,
                    "artifact_path": record.artifact_path,
                },
            )
        return record

    def latest_report(
        self,
        *,
        scope_type: str,
        scope_id: str,
    ) -> BrainMemoryHealthReportRecord | None:
        """Return the latest health report for one scope."""
        row = self._store._conn.execute(
            """
            SELECT * FROM memory_health_reports
            WHERE scope_type = ? AND scope_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (scope_type, scope_id),
        ).fetchone()
        if row is None:
            return None
        return BrainMemoryHealthReportRecord(
            report_id=str(row["report_id"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]),
            cycle_id=str(row["cycle_id"]),
            score=float(row["score"]),
            status=str(row["status"]),
            findings_json=str(row["findings_json"]),
            stats_json=str(row["stats_json"]),
            artifact_path=str(row["artifact_path"]) if row["artifact_path"] is not None else None,
            created_at=str(row["created_at"]),
        )

"""Inspectable export artifacts for Blink's layered memory state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blink.brain.autonomy_digest import build_autonomy_digest
from blink.brain.events import BrainEventType
from blink.brain.memory_layers.working import build_working_memory_snapshot
from blink.brain.memory_v2.core_blocks import BrainCoreMemoryBlockKind
from blink.brain.planning_digest import build_planning_digest
from blink.brain.reevaluation_digest import build_reevaluation_digest
from blink.brain.session import BrainSessionIds
from blink.brain.wake_digest import build_wake_digest
from blink.transcriptions.language import Language


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BrainMemoryExportArtifact:
    """One inspectable memory export artifact."""

    user_id: str
    thread_id: str
    generated_at: str
    payload: dict[str, Any]
    path: Path | None = None


class BrainMemoryExporter:
    """Write inspectable layered-memory digests to local artifacts."""

    def __init__(self, *, store):
        """Initialize the exporter.

        Args:
            store: Canonical brain store.
        """
        self._store = store

    def export_thread_digest(
        self,
        *,
        user_id: str,
        thread_id: str,
        output_path: Path | None = None,
    ) -> BrainMemoryExportArtifact:
        """Export a thread-scoped layered-memory digest as JSON."""
        working = build_working_memory_snapshot(store=self._store, user_id=user_id, thread_id=thread_id)
        heartbeat = self._store.get_heartbeat_projection(scope_key=thread_id)
        relationship_scope_id = f"blink/main:{user_id}"
        current_blocks = self._store.list_current_core_memory_blocks(
            scope_id=relationship_scope_id,
            scope_type="relationship",
            block_kinds=(
                BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
                BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
            ),
            limit=8,
        ) + self._store.list_current_core_memory_blocks(
            scope_id=user_id,
            scope_type="user",
            block_kinds=(BrainCoreMemoryBlockKind.USER_CORE.value,),
            limit=4,
        ) + self._store.list_current_core_memory_blocks(
            scope_type="agent",
            block_kinds=(
                BrainCoreMemoryBlockKind.SELF_CORE.value,
                BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
            ),
            limit=4,
        )
        block_versions = {
            record.block_kind: [
                {
                    "block_id": version.block_id,
                    "scope_type": version.scope_type,
                    "scope_id": version.scope_id,
                    "version": version.version,
                    "status": version.status,
                    "source_event_id": version.source_event_id,
                    "supersedes_block_id": version.supersedes_block_id,
                    "content": version.content,
                }
                for version in self._store.list_core_memory_block_versions(
                    block_kind=record.block_kind,
                    scope_type=record.scope_type,
                    scope_id=record.scope_id,
                )
            ]
            for record in current_blocks
        }
        autobiography = self._store.autobiographical_entries(
            scope_type="relationship",
            scope_id=relationship_scope_id,
            statuses=("current", "superseded"),
            limit=24,
        )
        health_report = self._store.latest_memory_health_report(
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        reflection_cycles = self._store.list_reflection_cycles(
            user_id=user_id,
            thread_id=thread_id,
            limit=12,
        )
        autonomy_ledger = self._store.get_autonomy_ledger_projection(scope_key=thread_id)
        commitment_projection = self._store.get_session_commitment_projection(
            agent_id=None,
            user_id=user_id,
            thread_id=thread_id,
        )
        recent_events = self._store.recent_brain_events(
            user_id=user_id,
            thread_id=thread_id,
            limit=192,
        )
        continuity_graph = self._store.build_continuity_graph(
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
        )
        continuity_dossiers = self._store.build_continuity_dossiers(
            user_id=user_id,
            thread_id=thread_id,
            scope_type="user",
            scope_id=user_id,
            continuity_graph=continuity_graph,
        )
        from blink.brain.evals.memory_state import build_continuity_state

        latest_event = recent_events[0] if recent_events else None
        session_ids = BrainSessionIds(
            agent_id=(
                str(getattr(latest_event, "agent_id", "")).strip() or "blink/main"
            ),
            user_id=user_id,
            session_id=(
                str(getattr(latest_event, "session_id", "")).strip() or thread_id
            ),
            thread_id=thread_id,
        )
        presence_event = self._store.latest_brain_event(
            user_id=user_id,
            thread_id=thread_id,
            event_types=(BrainEventType.BODY_STATE_UPDATED,),
        )
        presence_scope_key = str(
            ((presence_event.payload or {}) if presence_event is not None else {}).get("scope_key")
            or "local:presence"
        )
        continuity_state = build_continuity_state(
            store=self._store,
            session_ids=session_ids,
            presence_scope_key=presence_scope_key,
            language=Language.EN,
        )
        payload = {
            "generated_at": _utc_now(),
            "user_id": user_id,
            "thread_id": thread_id,
            "heartbeat": {
                "thread_id": thread_id,
                **heartbeat.as_dict(),
            },
            "working_memory": working.as_dict(),
            "autonomy_ledger": autonomy_ledger.as_dict(),
            "autonomy_digest": build_autonomy_digest(
                autonomy_ledger=autonomy_ledger,
                agenda=working.agenda,
            ),
            "reevaluation_digest": build_reevaluation_digest(
                autonomy_ledger=autonomy_ledger,
                recent_events=recent_events,
            ),
            "wake_digest": build_wake_digest(
                commitment_projection=commitment_projection,
                recent_events=recent_events,
            ),
            "planning_digest": build_planning_digest(
                agenda=working.agenda,
                commitment_projection=commitment_projection,
                recent_events=recent_events,
            ),
            "continuity_graph": continuity_graph.as_dict(),
            "continuity_graph_digest": continuity_state.get("continuity_graph_digest", {}),
            "continuity_dossiers": continuity_dossiers.as_dict(),
            "continuity_governance_report": continuity_state.get(
                "continuity_governance_report",
                {},
            ),
            "procedural_traces": continuity_state.get("procedural_traces", {}),
            "procedural_skills": continuity_state.get("procedural_skills", {}),
            "procedural_skill_digest": continuity_state.get("procedural_skill_digest", {}),
            "procedural_skill_governance_report": continuity_state.get(
                "procedural_skill_governance_report",
                {},
            ),
            "procedural_qa_report": continuity_state.get("procedural_qa_report", {}),
            "scene_world_state": continuity_state.get("scene_world_state", {}),
            "scene_world_state_digest": continuity_state.get("scene_world_state_digest", {}),
            "private_working_memory": continuity_state.get("private_working_memory", {}),
            "private_working_memory_digest": continuity_state.get(
                "private_working_memory_digest",
                {},
            ),
            "active_situation_model": continuity_state.get("active_situation_model", {}),
            "active_situation_model_digest": continuity_state.get(
                "active_situation_model_digest",
                {},
            ),
            "packet_traces": continuity_state.get("packet_traces", {}),
            "context_packet_digest": continuity_state.get("context_packet_digest", {}),
            "core_blocks": [
                {
                    "block_id": record.block_id,
                    "block_kind": record.block_kind,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "version": record.version,
                    "status": record.status,
                    "source_event_id": record.source_event_id,
                    "supersedes_block_id": record.supersedes_block_id,
                    "content": record.content,
                }
                for record in current_blocks
            ],
            "core_block_versions": block_versions,
            "current_claims": [
                {
                    "claim_id": record.claim_id,
                    "predicate": record.predicate,
                    "object": record.object,
                    "status": record.status,
                    "confidence": record.confidence,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                    "source_event_id": record.source_event_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "claim_key": record.claim_key,
                }
                for record in self._store.query_claims(
                    temporal_mode="current",
                    scope_type="user",
                    scope_id=user_id,
                    limit=32,
                )
            ],
            "historical_claims": [
                {
                    "claim_id": record.claim_id,
                    "predicate": record.predicate,
                    "object": record.object,
                    "status": record.status,
                    "confidence": record.confidence,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                    "source_event_id": record.source_event_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "claim_key": record.claim_key,
                }
                for record in self._store.query_claims(
                    temporal_mode="historical",
                    scope_type="user",
                    scope_id=user_id,
                    limit=32,
                )
            ],
            "claim_supersessions": [
                {
                    "supersession_id": record.supersession_id,
                    "prior_claim_id": record.prior_claim_id,
                    "new_claim_id": record.new_claim_id,
                    "reason": record.reason,
                    "source_event_id": record.source_event_id,
                    "created_at": record.created_at,
                }
                for record in self._store.claim_supersessions()
            ],
            "autobiography": [
                {
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
                    "supersedes_entry_id": record.supersedes_entry_id,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                }
                for record in autobiography
            ],
            "memory_health_report": (
                {
                    "report_id": health_report.report_id,
                    "scope_type": health_report.scope_type,
                    "scope_id": health_report.scope_id,
                    "cycle_id": health_report.cycle_id,
                    "score": health_report.score,
                    "status": health_report.status,
                    "findings": health_report.findings,
                    "stats": health_report.stats,
                    "artifact_path": health_report.artifact_path,
                    "created_at": health_report.created_at,
                }
                if health_report is not None
                else None
            ),
            "reflection_cycles": [record.as_dict() for record in reflection_cycles],
            "latest_reflection_draft_path": next(
                (
                    record.draft_artifact_path
                    for record in reflection_cycles
                    if record.draft_artifact_path
                ),
                None,
            ),
            "semantic": [
                {
                    "id": record.id,
                    "namespace": record.namespace,
                    "subject": record.subject,
                    "rendered_text": record.rendered_text,
                    "confidence": record.confidence,
                    "status": record.status,
                    "observed_at": record.observed_at,
                }
                for record in self._store.semantic_memories(user_id=user_id, limit=24)
            ],
            "narrative": [
                {
                    "id": record.id,
                    "kind": record.kind,
                    "title": record.title,
                    "summary": record.summary,
                    "status": record.status,
                    "confidence": record.confidence,
                    "observed_at": record.observed_at,
                }
                for record in self._store.narrative_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    limit=24,
                )
            ],
            "episodic": [
                {
                    "id": record.id,
                    "kind": record.kind,
                    "summary": record.summary,
                    "confidence": record.confidence,
                    "observed_at": record.observed_at,
                }
                for record in self._store.episodic_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    limit=24,
                )
            ],
        }

        path = output_path or (
            self._store.path.parent
            / "exports"
            / f"{user_id}_{thread_id.replace(':', '_').replace('/', '_')}_memory.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self._store.record_memory_export(
            user_id=user_id,
            thread_id=thread_id,
            export_kind="thread_digest",
            path=path,
            payload=payload,
        )
        return BrainMemoryExportArtifact(
            user_id=user_id,
            thread_id=thread_id,
            generated_at=str(payload["generated_at"]),
            payload=payload,
            path=path,
        )

"""Replay and evaluation helpers for append-only Blink brain events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from blink.brain.autonomy import BrainAutonomyLedgerProjection
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, BrainContextSurfaceSnapshot
from blink.brain.core.replay import (
    BrainCoreReplayHarness,
    BrainCoreReplayResult,
    BrainCoreReplayScenario,
)
from blink.brain.identity import load_default_agent_blocks
from blink.brain.replay_support import materialize_replayed_events
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

BrainReplayScenario = BrainCoreReplayScenario


@dataclass(frozen=True)
class BrainReplayResult:
    """Replay result plus the rebuilt prompt-facing state surface."""

    scenario: BrainReplayScenario
    replay_store_path: Path
    artifact_path: Path
    context_surface: BrainContextSurfaceSnapshot
    autonomy_ledger: BrainAutonomyLedgerProjection


class BrainReplayHarness:
    """Capture and replay inspectable higher-layer brain scenarios."""

    def __init__(self, *, store: BrainStore):
        """Initialize the harness against a source store."""
        self._store = store
        self._core = BrainCoreReplayHarness(store=store)

    def capture_builtin_scenario(self, **kwargs) -> BrainReplayScenario:
        """Capture one built-in replay scenario from the append-only event log."""
        return self._core.capture_builtin_scenario(**kwargs)

    def replay(
        self,
        scenario: BrainReplayScenario,
        *,
        output_store_path: Path | None = None,
        artifact_path: Path | None = None,
        presence_scope_key: str | None = None,
        language: Language = Language.ZH,
        capability_registry=None,
        context_queries: dict[str, str] | None = None,
    ) -> BrainReplayResult:
        """Replay one scenario and rebuild the prompt-facing context surface."""
        core_result: BrainCoreReplayResult = self._core.replay(
            scenario,
            output_store_path=output_store_path,
            artifact_path=artifact_path,
            presence_scope_key=presence_scope_key,
        )
        replay_store = BrainStore(path=core_result.replay_store_path)
        try:
            materialize_replayed_events(
                store=replay_store,
                session_ids=scenario.session_ids,
                events=scenario.events,
                agent_blocks=self._store.get_agent_blocks() or load_default_agent_blocks(),
            )
            # Core replay imports rebuild only the core projection slice. Rebuild the
            # higher-layer projection table here so replay artifacts reflect stored
            # predictive and rehearsal lifecycle events as well.
            replay_store.rebuild_brain_projections()
            resolved_presence_scope_key = presence_scope_key or next(
                (
                    str(event.payload.get("scope_key"))
                    for event in scenario.events
                    if event.event_type == "body.state.updated"
                    and str(event.payload.get("scope_key", "")).strip()
                ),
                "local:presence",
            )
            surface_builder = BrainContextSurfaceBuilder(
                store=replay_store,
                session_resolver=lambda: scenario.session_ids,
                presence_scope_key=resolved_presence_scope_key,
                language=language,
                capability_registry=capability_registry,
            )
            context_surface = surface_builder.build(latest_user_text="", include_historical_claims=True)

            payload = json.loads(core_result.artifact_path.read_text(encoding="utf-8"))
            payload["core_blocks"] = {
                name: {
                    "block_id": record.block_id,
                    "scope_type": record.scope_type,
                    "scope_id": record.scope_id,
                    "version": record.version,
                    "status": record.status,
                    "content": record.content,
                }
                for name, record in context_surface.core_blocks.items()
            }
            payload["core_block_versions"] = {
                name: [
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
                    for version in replay_store.list_core_memory_block_versions(
                        block_kind=record.block_kind,
                        scope_type=record.scope_type,
                        scope_id=record.scope_id,
                    )
                ]
                for name, record in context_surface.core_blocks.items()
            }
            payload["current_claims"] = [
                {
                    "claim_id": record.claim_id,
                    "predicate": record.predicate,
                    "object": record.object,
                    "status": record.status,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                }
                for record in context_surface.current_claims
            ]
            payload["historical_claims"] = [
                {
                    "claim_id": record.claim_id,
                    "predicate": record.predicate,
                    "object": record.object,
                    "status": record.status,
                    "valid_from": record.valid_from,
                    "valid_to": record.valid_to,
                }
                for record in context_surface.historical_claims
            ]
            payload["autobiography"] = [
                {
                    "entry_id": record.entry_id,
                    "entry_kind": record.entry_kind,
                    "rendered_summary": record.rendered_summary,
                    "status": record.status,
                    "salience": record.salience,
                    "content": record.content,
                }
                for record in context_surface.autobiography
            ]
            payload["memory_health_summary"] = (
                {
                    "report_id": context_surface.health_summary.report_id,
                    "status": context_surface.health_summary.status,
                    "score": context_surface.health_summary.score,
                    "findings": context_surface.health_summary.findings,
                    "stats": context_surface.health_summary.stats,
                }
                if context_surface.health_summary is not None
                else None
            )
            payload["reflection_cycles"] = [
                record.as_dict()
                for record in replay_store.list_reflection_cycles(
                    user_id=scenario.session_ids.user_id,
                    thread_id=scenario.session_ids.thread_id,
                    limit=16,
                )
            ]
            payload["latest_reflection_draft_path"] = next(
                (
                    record["draft_artifact_path"]
                    for record in payload["reflection_cycles"]
                    if record.get("draft_artifact_path")
                ),
                None,
            )
            payload["commitment_projection"] = context_surface.commitment_projection.as_dict()
            payload["autonomy_ledger"] = context_surface.autonomy_ledger.as_dict()
            from blink.brain.evals.memory_state import build_continuity_state

            payload["continuity_state"] = build_continuity_state(
                store=replay_store,
                session_ids=scenario.session_ids,
                presence_scope_key=resolved_presence_scope_key,
                language=language,
                context_queries=context_queries,
            )
            payload["continuity_graph"] = payload["continuity_state"].get("continuity_graph", {})
            payload["continuity_dossiers"] = payload["continuity_state"].get(
                "continuity_dossiers",
                {},
            )
            payload["procedural_traces"] = payload["continuity_state"].get(
                "procedural_traces",
                {},
            )
            payload["procedural_skills"] = payload["continuity_state"].get(
                "procedural_skills",
                {},
            )
            payload["continuity_graph_digest"] = payload["continuity_state"].get(
                "continuity_graph_digest",
                {},
            )
            payload["context_packet_digest"] = payload["continuity_state"].get(
                "context_packet_digest",
                {},
            )
            payload["continuity_governance_report"] = payload["continuity_state"].get(
                "continuity_governance_report",
                {},
            )
            payload["procedural_skill_digest"] = payload["continuity_state"].get(
                "procedural_skill_digest",
                {},
            )
            payload["procedural_skill_governance_report"] = payload["continuity_state"].get(
                "procedural_skill_governance_report",
                {},
            )
            payload["procedural_qa_report"] = payload["continuity_state"].get(
                "procedural_qa_report",
                {},
            )
            payload["scene_world_state"] = payload["continuity_state"].get(
                "scene_world_state",
                {},
            )
            payload["scene_world_state_digest"] = payload["continuity_state"].get(
                "scene_world_state_digest",
                {},
            )
            payload["multimodal_autobiography"] = payload["continuity_state"].get(
                "multimodal_autobiography",
                [],
            )
            payload["multimodal_autobiography_digest"] = payload["continuity_state"].get(
                "multimodal_autobiography_digest",
                {},
            )
            payload["private_working_memory"] = payload["continuity_state"].get(
                "private_working_memory",
                {},
            )
            payload["private_working_memory_digest"] = payload["continuity_state"].get(
                "private_working_memory_digest",
                {},
            )
            payload["active_situation_model"] = payload["continuity_state"].get(
                "active_situation_model",
                {},
            )
            payload["active_situation_model_digest"] = payload["continuity_state"].get(
                "active_situation_model_digest",
                {},
            )
            payload["autonomy_digest"] = payload["continuity_state"].get("autonomy_digest", {})
            payload["reevaluation_digest"] = payload["continuity_state"].get(
                "reevaluation_digest",
                {},
            )
            payload["wake_digest"] = payload["continuity_state"].get("wake_digest", {})
            payload["planning_digest"] = payload["continuity_state"].get("planning_digest", {})
            payload["executive_policy_audit"] = payload["continuity_state"].get(
                "executive_policy_audit",
                {},
            )
            payload["packet_traces"] = payload["continuity_state"].get("packet_traces", {})
            payload["context_queries"] = dict(context_queries or {})
            payload["recent_memory"] = [
                {
                    "layer": record.layer,
                    "record_id": record.record_id,
                    "summary": record.summary,
                    "score": record.score,
                    "confidence": record.confidence,
                    "stale": record.stale,
                }
                for record in context_surface.recent_memory
            ]
            core_result.artifact_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if hasattr(replay_store, "record_memory_export"):
                replay_store.record_memory_export(
                    user_id=scenario.session_ids.user_id,
                    thread_id=scenario.session_ids.thread_id,
                    export_kind="brain_replay",
                    path=core_result.artifact_path,
                    payload=payload,
                )
        finally:
            replay_store.close()
        return BrainReplayResult(
            scenario=scenario,
            replay_store_path=core_result.replay_store_path,
            artifact_path=core_result.artifact_path,
            context_surface=context_surface,
            autonomy_ledger=context_surface.autonomy_ledger,
        )


__all__ = ["BrainReplayHarness", "BrainReplayResult", "BrainReplayScenario"]

"""Canonical context surfaces composed from projections and continuity memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from blink.brain.autonomy import BrainAutonomyLedgerProjection
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.capability_manifests import (
    render_capability_manifest,
    render_internal_capability_manifest,
)
from blink.brain.context.policy import (
    BrainContextSurfaceQueryStrategy,
    BrainContextTask,
    get_brain_context_mode_policy,
)
from blink.brain.memory_layers.retrieval import (
    BrainMemoryQuery,
    BrainMemoryRetriever,
    BrainMemorySearchResult,
)
from blink.brain.memory_layers.semantic import render_profile_fact
from blink.brain.memory_layers.working import build_working_memory_snapshot
from blink.brain.memory_v2 import (
    BrainAutobiographicalEntryRecord,
    BrainAutobiographyEntryKind,
    BrainClaimGovernanceProjection,
    BrainClaimRecord,
    BrainClaimSupersessionRecord,
    BrainContinuityDossierProjection,
    BrainContinuityGraphProjection,
    BrainContinuityQuery,
    BrainCoreMemoryBlockKind,
    BrainCoreMemoryBlockRecord,
    BrainMemoryHealthReportRecord,
    BrainProceduralSkillProjection,
    ContinuityRetriever,
    render_claim_summary,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainAgendaProjection,
    BrainCommitmentProjection,
    BrainEmbodiedExecutiveProjection,
    BrainEngagementStateProjection,
    BrainHeartbeatProjection,
    BrainPredictiveWorldModelProjection,
    BrainPrivateWorkingMemoryProjection,
    BrainRelationshipStateProjection,
    BrainSceneStateProjection,
    BrainSceneWorldProjection,
    BrainWorkingContextProjection,
)
from blink.transcriptions.language import Language


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class BrainContextMode(str, Enum):
    """Compatibility shim for older reply/planning surface selection."""

    REPLY = "reply"
    PLANNING = "planning"


@dataclass(frozen=True)
class BrainContextSurfaceSnapshot:
    """Immutable prompt-facing state composed from projections and continuity memory."""

    agent_blocks: dict[str, str]
    body: BrainPresenceSnapshot
    scene: BrainSceneStateProjection
    engagement: BrainEngagementStateProjection
    relationship_state: BrainRelationshipStateProjection
    working_context: BrainWorkingContextProjection
    scene_world_state: BrainSceneWorldProjection
    predictive_world_model: BrainPredictiveWorldModelProjection
    embodied_executive: BrainEmbodiedExecutiveProjection
    private_working_memory: BrainPrivateWorkingMemoryProjection
    active_situation_model: BrainActiveSituationProjection
    agenda: BrainAgendaProjection
    commitment_projection: BrainCommitmentProjection
    heartbeat: BrainHeartbeatProjection
    autonomy_ledger: BrainAutonomyLedgerProjection = field(
        default_factory=BrainAutonomyLedgerProjection
    )
    core_blocks: dict[str, BrainCoreMemoryBlockRecord] = field(default_factory=dict)
    current_claims: tuple[BrainClaimRecord, ...] = ()
    historical_claims: tuple[BrainClaimRecord, ...] = ()
    claim_governance: BrainClaimGovernanceProjection | None = None
    claim_supersessions: tuple[BrainClaimSupersessionRecord, ...] = ()
    autobiography: tuple[BrainAutobiographicalEntryRecord, ...] = ()
    scene_episodes: tuple[BrainAutobiographicalEntryRecord, ...] = ()
    health_summary: BrainMemoryHealthReportRecord | None = None
    recent_memory: tuple[BrainMemorySearchResult, ...] = ()
    historical_recent_memory: tuple[BrainMemorySearchResult, ...] = ()
    episodic_fallback: tuple[BrainMemorySearchResult, ...] = ()
    continuity_graph: BrainContinuityGraphProjection | None = None
    continuity_dossiers: BrainContinuityDossierProjection | None = None
    procedural_skills: BrainProceduralSkillProjection | None = None
    capability_manifest: str = ""
    internal_capability_manifest: str = ""
    generated_at: str = field(default_factory=_utc_now)


@dataclass(frozen=True)
class BrainContextSurfaceBuilder:
    """Build the canonical compiler input from projections and continuity retrieval."""

    store: Any
    session_resolver: Any
    presence_scope_key: str
    language: Language
    capability_registry: CapabilityRegistry | None = None

    def build(
        self,
        *,
        latest_user_text: str,
        task: BrainContextTask | None = None,
        mode: BrainContextMode | None = None,
        include_historical_claims: bool | None = None,
    ) -> BrainContextSurfaceSnapshot:
        """Compose the current brain context surface for the active session."""
        resolved_task = task or _task_from_mode(mode)
        policy = get_brain_context_mode_policy(resolved_task)
        include_historical = (
            policy.include_historical_claims
            if include_historical_claims is None
            else include_historical_claims
        )
        session_ids = self.session_resolver()
        agent_blocks = self.store.get_agent_blocks()
        working_memory = build_working_memory_snapshot(
            store=self.store,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
        core_blocks: dict[str, BrainCoreMemoryBlockRecord] = {}
        for record in self.store.list_current_core_memory_blocks(
            scope_id=session_ids.user_id,
            scope_type="user",
            block_kinds=(BrainCoreMemoryBlockKind.USER_CORE.value,),
        ):
            core_blocks[record.block_kind] = record
        for record in self.store.list_current_core_memory_blocks(
            scope_id=relationship_scope_id,
            scope_type="relationship",
            block_kinds=(
                BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
                BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
                BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
                BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
                BrainCoreMemoryBlockKind.BEHAVIOR_CONTROL_PROFILE.value,
            ),
        ):
            core_blocks[record.block_kind] = record
        for record in self.store.list_current_core_memory_blocks(
            scope_id=session_ids.agent_id,
            scope_type="agent",
            block_kinds=(
                BrainCoreMemoryBlockKind.SELF_CORE.value,
                BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value,
                BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
            ),
        ):
            core_blocks[record.block_kind] = record

        current_claims = tuple(
            self.store.query_claims(
                temporal_mode="current",
                scope_type="user",
                scope_id=session_ids.user_id,
                limit=24,
            )
        )
        historical_claims = (
            tuple(
                self.store.query_claims(
                    temporal_mode="historical",
                    scope_type="user",
                    scope_id=session_ids.user_id,
                    limit=12,
                )
            )
            if include_historical
            else ()
        )
        claim_supersessions = tuple(self.store.claim_supersessions())
        claim_governance = self.store.preview_claim_governance_projection(
            scope_type="user",
            scope_id=session_ids.user_id,
        )

        autobiography = tuple(
            self.store.autobiographical_entries(
                scope_type="relationship",
                scope_id=relationship_scope_id,
                entry_kinds=(
                    BrainAutobiographyEntryKind.SHARED_HISTORY_SUMMARY.value,
                    BrainAutobiographyEntryKind.RELATIONSHIP_ARC.value,
                    BrainAutobiographyEntryKind.RELATIONSHIP_MILESTONE.value,
                    BrainAutobiographyEntryKind.PROJECT_ARC.value,
                ),
                statuses=("current",),
                limit=12,
            )
        )
        scene_episodes = tuple(
            self.store.autobiographical_entries(
                scope_type="presence",
                scope_id=self.presence_scope_key,
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                modalities=("scene_world",),
                limit=8,
            )
        )
        health_summary = self.store.latest_memory_health_report(
            scope_type="relationship",
            scope_id=relationship_scope_id,
        )
        commitment_projection = self.store.get_session_commitment_projection(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        continuity_core_blocks = tuple(
            {
                version.block_id: version
                for record in core_blocks.values()
                for version in self.store.list_core_memory_block_versions(
                    block_kind=record.block_kind,
                    scope_type=record.scope_type,
                    scope_id=record.scope_id,
                )
            }.values()
        )
        procedural_skills = self.store.build_procedural_skill_projection(
            scope_type="thread",
            scope_id=session_ids.thread_id,
        )
        scene_world_state = self.store.build_scene_world_state_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self.presence_scope_key,
        )
        continuity_graph = self.store.build_continuity_graph(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            scope_type="user",
            scope_id=session_ids.user_id,
            core_blocks=continuity_core_blocks,
            commitment_projection=commitment_projection,
            agenda=working_memory.agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=self.presence_scope_key,
            recent_event_limit=96,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        continuity_dossiers = self.store.build_continuity_dossiers(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            scope_type="user",
            scope_id=session_ids.user_id,
            continuity_graph=continuity_graph,
            core_blocks=continuity_core_blocks,
            commitment_projection=commitment_projection,
            agenda=working_memory.agenda,
            procedural_skills=procedural_skills,
            scene_world_state=scene_world_state,
            presence_scope_key=self.presence_scope_key,
            recent_event_limit=96,
            current_claim_limit=32,
            historical_claim_limit=16,
            autobiography_limit=24,
        )
        autonomy_ledger = self.store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        retrieval_text = _resolve_surface_query_text(
            latest_user_text=latest_user_text,
            task=resolved_task,
            query_strategy=policy.surface_query_strategy,
            working_memory=working_memory,
            commitment_projection=commitment_projection,
            autonomy_ledger=autonomy_ledger,
            heartbeat=working_memory.heartbeat,
            continuity_dossiers=continuity_dossiers,
            claim_governance=claim_governance,
        )
        retriever = ContinuityRetriever(store=self.store)
        recent_memory = tuple(
            retriever.retrieve_as_memory_results(
                BrainContinuityQuery(
                    text=retrieval_text,
                    scope_type="user",
                    scope_id=session_ids.user_id,
                    temporal_mode="current",
                    limit=12,
                )
            )
        )
        historical_recent_memory = tuple(
            retriever.retrieve_as_memory_results(
                BrainContinuityQuery(
                    text=retrieval_text,
                    scope_type="user",
                    scope_id=session_ids.user_id,
                    temporal_mode="historical",
                    limit=12,
                )
            )
        )
        legacy_retriever = BrainMemoryRetriever(store=self.store)
        episodic_fallback = tuple(
            legacy_retriever.retrieve(
                BrainMemoryQuery(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    text=retrieval_text,
                    layers=("episodic",),
                    limit=4,
                    include_stale=False,
                )
            )
        )
        private_working_memory = self.store.build_private_working_memory_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self.presence_scope_key,
        )
        predictive_world_model = self.store.build_predictive_world_model_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self.presence_scope_key,
        )
        embodied_executive = self.store.build_embodied_executive_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self.presence_scope_key,
        )
        active_situation_model = self.store.build_active_situation_model_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self.presence_scope_key,
        )
        return BrainContextSurfaceSnapshot(
            agent_blocks=agent_blocks,
            body=self.store.get_body_state_projection(scope_key=self.presence_scope_key),
            scene=self.store.get_scene_state_projection(scope_key=self.presence_scope_key),
            engagement=self.store.get_engagement_state_projection(
                scope_key=self.presence_scope_key
            ),
            relationship_state=self.store.get_relationship_state_projection(
                scope_key=self.presence_scope_key,
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
            ),
            working_context=working_memory.context,
            scene_world_state=scene_world_state,
            predictive_world_model=predictive_world_model,
            embodied_executive=embodied_executive,
            private_working_memory=private_working_memory,
            active_situation_model=active_situation_model,
            agenda=working_memory.agenda,
            commitment_projection=commitment_projection,
            autonomy_ledger=autonomy_ledger,
            heartbeat=working_memory.heartbeat,
            core_blocks=core_blocks,
            current_claims=current_claims,
            historical_claims=historical_claims,
            claim_governance=claim_governance,
            claim_supersessions=claim_supersessions,
            autobiography=autobiography,
            scene_episodes=scene_episodes,
            health_summary=health_summary,
            recent_memory=recent_memory,
            historical_recent_memory=historical_recent_memory,
            episodic_fallback=episodic_fallback,
            continuity_graph=continuity_graph,
            continuity_dossiers=continuity_dossiers,
            procedural_skills=procedural_skills,
            capability_manifest=render_capability_manifest(
                language=self.language,
                registry=self.capability_registry,
                fallback_text=agent_blocks.get("action_library", ""),
            ),
            internal_capability_manifest=render_internal_capability_manifest(
                language=self.language,
                registry=self.capability_registry,
            ),
        )


def _task_from_mode(mode: BrainContextMode | None) -> BrainContextTask:
    if mode == BrainContextMode.PLANNING:
        return BrainContextTask.PLANNING
    return BrainContextTask.REPLY


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _join_query_parts(*parts: str | None) -> str:
    return " | ".join(part for part in parts if _optional_text(part) is not None)


def _wake_query_text(commitment_projection, heartbeat) -> str:
    waiting = list(commitment_projection.blocked_commitments) + list(
        commitment_projection.deferred_commitments
    )
    first_waiting = next((record.title for record in waiting if _optional_text(record.title)), None)
    second_waiting = next(
        (
            record.title
            for record in waiting[1:]
            if _optional_text(record.title) and record.title != first_waiting
        ),
        None,
    )
    heartbeat_summary = _join_query_parts(
        _optional_text(heartbeat.last_event_type),
        _optional_text(heartbeat.last_robot_action),
        _optional_text(heartbeat.last_tool_name),
        heartbeat.warnings[0] if heartbeat.warnings else None,
    )
    return _join_query_parts(first_waiting or heartbeat_summary, heartbeat_summary, second_waiting)


def _reevaluation_query_text(autonomy_ledger, commitment_projection, agenda) -> str:
    held_candidate = next(
        (
            candidate.summary
            for candidate in autonomy_ledger.current_candidates
            if _optional_text(candidate.summary)
        ),
        None,
    )
    blocked_commitment = next(
        (
            record.title
            for record in commitment_projection.blocked_commitments
            if _optional_text(record.title)
        ),
        None,
    )
    return _join_query_parts(
        held_candidate or blocked_commitment,
        blocked_commitment if held_candidate is not None else None,
        _optional_text(agenda.active_goal_summary),
    )


def _audit_query_text(continuity_dossiers, claim_governance, agenda, latest_user_text: str) -> str:
    review_debt = next(
        (
            f"{record.title}: review debt"
            for record in continuity_dossiers.dossiers
            if int(getattr(record.governance, "review_debt_count", 0) or 0) > 0
        ),
        None,
    )
    governance_issue = next(
        (
            f"{record.title}: {issue.summary}"
            for record in continuity_dossiers.dossiers
            for issue in record.open_issues
            if _optional_text(issue.summary)
        ),
        None,
    )
    held_claim = next(
        (
            f"{record.claim_id}: {record.currentness_status}"
            for record in getattr(claim_governance, "records", ())
            if record.currentness_status in {"held", "stale", "historical"}
        ),
        None,
    )
    return _join_query_parts(
        review_debt or governance_issue or held_claim,
        _optional_text(agenda.active_goal_summary),
        _optional_text(latest_user_text),
    )


def _resolve_surface_query_text(
    *,
    latest_user_text: str,
    task: BrainContextTask,
    query_strategy: BrainContextSurfaceQueryStrategy,
    working_memory,
    commitment_projection,
    autonomy_ledger,
    heartbeat,
    continuity_dossiers,
    claim_governance,
) -> str:
    latest = _optional_text(latest_user_text)
    last_user = _optional_text(working_memory.context.last_user_text)
    agenda_summary = _optional_text(working_memory.agenda.active_goal_summary)
    if query_strategy == BrainContextSurfaceQueryStrategy.REPLY:
        return latest or last_user or ""
    if query_strategy == BrainContextSurfaceQueryStrategy.PLANNING:
        pending_summary = next(
            (
                record.title
                for record in (
                    list(commitment_projection.active_commitments)
                    + list(commitment_projection.blocked_commitments)
                    + list(commitment_projection.deferred_commitments)
                )
                if _optional_text(record.title)
            ),
            None,
        )
        return latest or agenda_summary or pending_summary or last_user or ""
    if query_strategy == BrainContextSurfaceQueryStrategy.WAKE:
        return latest or _wake_query_text(commitment_projection, heartbeat) or agenda_summary or ""
    if query_strategy == BrainContextSurfaceQueryStrategy.REEVALUATION:
        return (
            latest
            or _reevaluation_query_text(
                autonomy_ledger, commitment_projection, working_memory.agenda
            )
            or agenda_summary
            or last_user
            or ""
        )
    if query_strategy == BrainContextSurfaceQueryStrategy.AUDIT:
        return (
            latest
            or _audit_query_text(
                continuity_dossiers,
                claim_governance,
                working_memory.agenda,
                latest_user_text,
            )
            or agenda_summary
            or last_user
            or ""
        )
    raise AssertionError(f"Unhandled query strategy for task={task.value}: {query_strategy.value}")


def _current_claim_map(snapshot: BrainContextSurfaceSnapshot) -> dict[str, list[BrainClaimRecord]]:
    grouped: dict[str, list[BrainClaimRecord]] = {}
    for claim in snapshot.current_claims:
        grouped.setdefault(claim.predicate, []).append(claim)
    return grouped


def render_relationship_continuity_summary(
    snapshot: BrainContextSurfaceSnapshot,
    language: Language,
) -> str:
    """Render relationship continuity from core blocks, not legacy narrative rows."""
    relationship_arc = next(
        (
            entry
            for entry in snapshot.autobiography
            if entry.entry_kind == BrainAutobiographyEntryKind.RELATIONSHIP_ARC.value
        ),
        None,
    )
    relationship_core = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value)
    active_commitments = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value)
    parts: list[str] = []
    if relationship_arc is not None:
        prefix = (
            "关系弧线" if language.value.lower().startswith(("zh", "cmn")) else "Relationship arc"
        )
        parts.append(f"{prefix}: {relationship_arc.rendered_summary}")
    if relationship_core is not None:
        summary = str(relationship_core.content.get("last_session_summary", "")).strip()
        if summary:
            prefix = "连续性" if language.value.lower().startswith(("zh", "cmn")) else "Continuity"
            parts.append(f"{prefix}: {summary}")
    if active_commitments is not None:
        commitments = list(active_commitments.content.get("commitments", []))
        for item in commitments[:4]:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            prefix = "承诺" if language.value.lower().startswith(("zh", "cmn")) else "Commitment"
            parts.append(f"{prefix}: {title}")
    if not parts:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    return "\n".join(parts)


def render_autobiography_summary(snapshot: BrainContextSurfaceSnapshot, language: Language) -> str:
    """Render the strongest autobiographical continuity artifacts."""
    parts: list[str] = []
    self_current_arc = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value)
    if self_current_arc is not None:
        summary = str(self_current_arc.content.get("summary", "")).strip()
        if summary:
            prefix = (
                "当前弧线" if language.value.lower().startswith(("zh", "cmn")) else "Current arc"
            )
            parts.append(f"{prefix}: {summary}")
    for entry in snapshot.autobiography[:3]:
        label = entry.entry_kind.replace("_", " ")
        parts.append(f"- {label}: {entry.rendered_summary}")
    if not parts:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    return "\n".join(parts)


def render_health_summary(snapshot: BrainContextSurfaceSnapshot, language: Language) -> str:
    """Render the latest memory-health report into compact prompt text."""
    report = snapshot.health_summary
    if report is None:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    warning_codes = [
        finding.get("code", "")
        for finding in report.findings
        if isinstance(finding, dict) and finding.get("severity") in {"warning", "critical"}
    ]
    if language.value.lower().startswith(("zh", "cmn")):
        return (
            f"状态: {report.status}\n"
            f"健康分: {report.score:.2f}\n"
            f"重点问题: {'；'.join(warning_codes) if warning_codes else '无'}"
        )
    return (
        f"Status: {report.status}\n"
        f"Health score: {report.score:.2f}\n"
        f"Key issues: {'; '.join(warning_codes) if warning_codes else 'None'}"
    )


def render_recent_memory_summary(
    results: tuple[BrainMemorySearchResult, ...],
    language: Language,
) -> str:
    """Render current-truth retrieval hits into prompt-safe text."""
    if not results:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"

    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            f"- [{result.layer}] {result.summary}" + (" (可能已过时)" if result.stale else "")
            for result in results
        )

    return "\n".join(
        f"- [{result.layer}] {result.summary}" + (" (possibly stale)" if result.stale else "")
        for result in results
    )


def render_user_profile_summary(snapshot: BrainContextSurfaceSnapshot, language: Language) -> str:
    """Render the continuity-backed user profile slice of a context surface."""
    user_core = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.USER_CORE.value)
    claim_map = _current_claim_map(snapshot)
    parts: list[str] = []
    if user_core is not None:
        content = user_core.content
        for key, predicate in (
            ("name", "profile.name"),
            ("role", "profile.role"),
            ("origin", "profile.origin"),
        ):
            value = str(content.get(key, "")).strip()
            if value:
                parts.append(render_profile_fact(predicate, value))

    for predicate in ("profile.name", "profile.role", "profile.origin"):
        if any(predicate == claim.predicate for claim in snapshot.current_claims):
            rendered = render_claim_summary(_current_claim_map(snapshot)[predicate][0])
            if rendered not in parts:
                parts.append(rendered)
    for predicate in ("preference.like", "preference.dislike"):
        for claim in claim_map.get(predicate, [])[:4]:
            rendered = render_claim_summary(claim)
            if rendered not in parts:
                parts.append(rendered)

    if not parts:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    return (
        "；".join(parts) if language.value.lower().startswith(("zh", "cmn")) else "; ".join(parts)
    )

"""Structured continuity-state eval helpers for Blink."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blink.brain.active_situation_model_digest import build_active_situation_model_digest
from blink.brain.autonomy_digest import build_autonomy_digest
from blink.brain.context import (
    BrainContextSelector,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context.policy import all_brain_context_tasks
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.context_surfaces import (
    BrainContextSurfaceBuilder,
    BrainContextSurfaceSnapshot,
)
from blink.brain.continuity_governance_report import build_continuity_governance_report
from blink.brain.continuity_graph_digest import build_continuity_graph_digest
from blink.brain.counterfactual_rehearsal_digest import build_counterfactual_rehearsal_digest
from blink.brain.embodied_executive_digest import build_embodied_executive_digest
from blink.brain.evals.adapter_promotion import build_adapter_governance_inspection
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest
from blink.brain.events import BrainEventType
from blink.brain.executive_policy_audit import build_executive_policy_audit
from blink.brain.memory import BrainMemoryConsolidator
from blink.brain.memory_v2 import (
    BrainProceduralSkillProjection,
    BrainProceduralTraceProjection,
    BrainReflectionEngine,
    build_continuity_dossier_projection,
    build_continuity_graph_projection,
    build_multimodal_autobiography_digest,
    parse_multimodal_autobiography_record,
)
from blink.brain.memory_v2.skill_evidence import build_skill_evidence_inspection
from blink.brain.planning_digest import build_planning_digest
from blink.brain.practice_director import build_practice_inspection
from blink.brain.private_working_memory_digest import build_private_working_memory_digest
from blink.brain.procedural_qa_report import build_procedural_qa_report
from blink.brain.procedural_skill_digest import build_procedural_skill_digest
from blink.brain.procedural_skill_governance_report import (
    build_procedural_skill_governance_report,
)
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainCounterfactualRehearsalProjection,
    BrainEmbodiedExecutiveProjection,
    BrainPredictiveWorldModelProjection,
    BrainPrivateWorkingMemoryProjection,
    BrainSceneWorldProjection,
)
from blink.brain.reevaluation_digest import build_reevaluation_digest
from blink.brain.replay_support import (
    append_replay_event_payloads,
    materialize_replayed_events,
)
from blink.brain.runtime_shell_digest import build_runtime_shell_digest
from blink.brain.scene_world_state_digest import build_scene_world_state_digest
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore
from blink.brain.wake_digest import build_wake_digest
from blink.brain.world_model_digest import build_world_model_digest
from blink.transcriptions.language import Language


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _combined_surface_autobiography(
    snapshot: BrainContextSurfaceSnapshot,
):
    return tuple(snapshot.autobiography) + tuple(snapshot.scene_episodes)


def _recent_action_event_rows(
    *,
    store: BrainStore,
    session_ids: BrainSessionIds,
    limit: int = 12,
) -> list[dict[str, Any]]:
    return [
        {
            "action_id": record.action_id,
            "source": record.source,
            "accepted": record.accepted,
            "preview_only": record.preview_only,
            "summary": record.summary,
            "metadata": dict(record.metadata),
            "created_at": record.created_at,
        }
        for record in store.recent_action_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=limit,
        )
    ]


_CONTEXT_TASKS = tuple(task.value for task in all_brain_context_tasks())


def _normalize_context_queries(queries: dict[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for task in _CONTEXT_TASKS:
        text = _optional_text((queries or {}).get(task))
        if text is not None:
            normalized[task] = text
    return normalized


def _proposal_summary_for_ids(
    *,
    proposal_ids: list[str],
    recent_events: list[Any] | None = None,
) -> str:
    wanted_ids = {proposal_id for proposal_id in proposal_ids if proposal_id}
    if not wanted_ids:
        return ""
    for event in recent_events or []:
        if getattr(event, "event_type", "") not in {
            BrainEventType.PLANNING_PROPOSED,
            BrainEventType.PLANNING_ADOPTED,
            BrainEventType.PLANNING_REJECTED,
        }:
            continue
        proposal = dict((getattr(event, "payload", None) or {}).get("proposal") or {})
        if str(proposal.get("plan_proposal_id", "")).strip() not in wanted_ids:
            continue
        summary = _optional_text(proposal.get("summary"))
        if summary is not None:
            return summary
    return ""


def _derive_default_reply_query(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    recent_events: list[Any] | None = None,
    recent_episodes: list[Any] | None = None,
) -> str:
    if _optional_text(snapshot.working_context.last_user_text) is not None:
        return _optional_text(snapshot.working_context.last_user_text) or ""
    for event in recent_events or []:
        if getattr(event, "event_type", "") != BrainEventType.USER_TURN_TRANSCRIBED:
            continue
        text = _optional_text((getattr(event, "payload", None) or {}).get("text"))
        if text is not None:
            return text
    for episode in recent_episodes or []:
        text = _optional_text(getattr(episode, "user_text", None))
        if text is not None:
            return text
    return ""


def _derive_default_planning_query(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    recent_events: list[Any] | None = None,
) -> str:
    if snapshot.agenda.active_goal_summary:
        return snapshot.agenda.active_goal_summary
    for field_name in ("blocked_commitments", "deferred_commitments", "active_commitments"):
        records = getattr(snapshot.commitment_projection, field_name, ())
        for record in records:
            title = _optional_text(getattr(record, "title", None))
            if title is not None:
                return title
    for event in recent_events or []:
        if getattr(event, "event_type", "") != BrainEventType.PLANNING_PROPOSED:
            continue
        proposal = dict((getattr(event, "payload", None) or {}).get("proposal") or {})
        summary = _optional_text(proposal.get("summary"))
        if summary is not None:
            return summary
    proposal_ids: list[str] = []
    for goal in snapshot.agenda.goals:
        details = getattr(goal, "details", {})
        for key in ("pending_plan_proposal_id", "current_plan_proposal_id"):
            proposal_id = _optional_text(details.get(key))
            if proposal_id is not None and proposal_id not in proposal_ids:
                proposal_ids.append(proposal_id)
    for field_name in ("blocked_commitments", "deferred_commitments", "active_commitments"):
        for record in getattr(snapshot.commitment_projection, field_name, ()):
            details = getattr(record, "details", {})
            for key in ("pending_plan_proposal_id", "current_plan_proposal_id"):
                proposal_id = _optional_text(details.get(key))
                if proposal_id is not None and proposal_id not in proposal_ids:
                    proposal_ids.append(proposal_id)
    proposal_summary = _proposal_summary_for_ids(
        proposal_ids=proposal_ids,
        recent_events=recent_events,
    )
    if proposal_summary:
        return proposal_summary
    return ""


def _resolve_context_queries(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    recent_events: list[Any] | None = None,
    recent_episodes: list[Any] | None = None,
    context_queries: dict[str, Any] | None = None,
) -> dict[str, str]:
    normalized = _normalize_context_queries(context_queries)
    reply_query = normalized.get("reply") or _derive_default_reply_query(
        snapshot=snapshot,
        recent_events=recent_events,
        recent_episodes=recent_episodes,
    )
    planning_query = (
        normalized.get("planning")
        or _derive_default_planning_query(
            snapshot=snapshot,
            recent_events=recent_events,
        )
        or ""
    )
    return {
        "reply": reply_query,
        "planning": planning_query,
        "recall": normalized.get("recall") or reply_query,
        "reflection": normalized.get("reflection") or planning_query or reply_query,
        "critique": normalized.get("critique") or planning_query or reply_query,
        "wake": normalized.get("wake")
        or next(
            (
                record.title
                for record in (
                    list(snapshot.commitment_projection.blocked_commitments)
                    + list(snapshot.commitment_projection.deferred_commitments)
                )
                if _optional_text(record.title) is not None
            ),
            None,
        )
        or snapshot.heartbeat.last_event_type
        or planning_query
        or reply_query,
        "reevaluation": normalized.get("reevaluation")
        or next(
            (
                candidate.summary
                for candidate in snapshot.autonomy_ledger.current_candidates
                if _optional_text(candidate.summary) is not None
            ),
            None,
        )
        or next(
            (
                record.title
                for record in snapshot.commitment_projection.blocked_commitments
                if _optional_text(record.title) is not None
            ),
            None,
        )
        or planning_query
        or reply_query,
        "operator_audit": normalized.get("operator_audit")
        or next(
            (
                f"{dossier.title}: {issue.summary}"
                for dossier in (snapshot.continuity_dossiers.dossiers if snapshot.continuity_dossiers else ())
                for issue in dossier.open_issues
                if _optional_text(issue.summary) is not None
            ),
            None,
        )
        or planning_query
        or reply_query,
        "governance_review": normalized.get("governance_review")
        or next(
            (
                f"{dossier.title}: review debt"
                for dossier in (snapshot.continuity_dossiers.dossiers if snapshot.continuity_dossiers else ())
                if int(getattr(dossier.governance, "review_debt_count", 0) or 0) > 0
            ),
            None,
        )
        or normalized.get("operator_audit")
        or planning_query
        or reply_query,
    }


@dataclass(frozen=True)
class BrainConversationStep:
    """One scripted conversation or mutation step for continuity evals."""

    user_text: str
    assistant_text: str
    assistant_summary: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    remember_facts: tuple[dict[str, Any], ...] = ()
    upsert_tasks: tuple[dict[str, Any], ...] = ()
    task_status_updates: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    session_ids: BrainSessionIds | None = None


@dataclass(frozen=True)
class BrainContinuityExpectedState:
    """Structured expected continuity state for one eval or replay case."""

    current_claims: tuple[dict[str, Any], ...] = ()
    historical_claims: tuple[dict[str, Any], ...] = ()
    core_blocks_present: tuple[str, ...] = ()
    commitment_titles_by_status: dict[str, tuple[str, ...]] = field(default_factory=dict)
    agenda_contains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    autobiography_entry_kinds: tuple[str, ...] = ()
    health_finding_codes: tuple[str, ...] = ()
    health_score_min: float | None = None
    health_score_max: float | None = None
    relationship_arc_contains: str | None = None
    autonomy_current_candidate_ids: tuple[str, ...] = ()
    autonomy_current_candidate_summaries: tuple[str, ...] = ()
    autonomy_recent_decision_kinds: tuple[str, ...] = ()
    autonomy_reason_codes: tuple[str, ...] = ()
    autonomy_reevaluation_conditions: tuple[str, ...] = ()
    reevaluation_current_hold_ids: tuple[str, ...] = ()
    reevaluation_trigger_kinds: tuple[str, ...] = ()
    reevaluation_transition_candidate_ids: tuple[str, ...] = ()
    reevaluation_transition_outcome_kinds: tuple[str, ...] = ()
    wake_current_waiting_commitment_ids: tuple[str, ...] = ()
    wake_recent_wake_kinds: tuple[str, ...] = ()
    wake_recent_route_kinds: tuple[str, ...] = ()
    wake_reason_codes: tuple[str, ...] = ()
    planning_current_pending_proposal_ids: tuple[str, ...] = ()
    planning_current_pending_goal_ids: tuple[str, ...] = ()
    planning_recent_review_policies: tuple[str, ...] = ()
    planning_recent_outcome_kinds: tuple[str, ...] = ()
    planning_reason_codes: tuple[str, ...] = ()
    planning_recent_revision_goal_ids: tuple[str, ...] = ()
    continuity_graph_current_backing_ids: tuple[str, ...] = ()
    continuity_graph_historical_backing_ids: tuple[str, ...] = ()
    continuity_graph_edge_kinds: tuple[str, ...] = ()
    continuity_graph_superseded_backing_ids: tuple[str, ...] = ()
    continuity_graph_stale_backing_ids: tuple[str, ...] = ()
    relationship_dossier_summary_contains: str | None = None
    relationship_dossier_freshness: str | None = None
    relationship_dossier_contradiction: str | None = None
    project_dossier_keys: tuple[str, ...] = ()
    dossier_stale_ids: tuple[str, ...] = ()
    dossier_needs_refresh_ids: tuple[str, ...] = ()
    dossier_uncertain_ids: tuple[str, ...] = ()
    dossier_contradicted_ids: tuple[str, ...] = ()
    reply_packet_temporal_mode: str | None = None
    planning_packet_temporal_mode: str | None = None
    reply_selected_anchor_types: tuple[str, ...] = ()
    planning_selected_anchor_types: tuple[str, ...] = ()
    reply_selected_temporal_kinds: tuple[str, ...] = ()
    planning_selected_temporal_kinds: tuple[str, ...] = ()
    reply_selected_backing_ids: tuple[str, ...] = ()
    planning_selected_backing_ids: tuple[str, ...] = ()
    reply_drop_reason_codes: tuple[str, ...] = ()
    planning_drop_reason_codes: tuple[str, ...] = ()
    packet_temporal_modes: dict[str, str] = field(default_factory=dict)
    packet_selected_anchor_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    packet_selected_item_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    packet_selected_temporal_kinds: dict[str, tuple[str, ...]] = field(default_factory=dict)
    packet_selected_backing_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    packet_selected_provenance_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    packet_drop_reason_codes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    graph_digest_current_commitment_ids: tuple[str, ...] = ()
    graph_digest_current_plan_proposal_ids: tuple[str, ...] = ()
    governance_open_issue_kinds: tuple[str, ...] = ()
    governance_stale_graph_backing_ids: tuple[str, ...] = ()
    governance_superseded_graph_backing_ids: tuple[str, ...] = ()
    procedural_active_skill_ids: tuple[str, ...] = ()
    procedural_candidate_skill_ids: tuple[str, ...] = ()
    procedural_retired_skill_ids: tuple[str, ...] = ()
    procedural_superseded_skill_ids: tuple[str, ...] = ()
    procedural_failure_signature_codes: tuple[str, ...] = ()
    procedural_low_confidence_skill_ids: tuple[str, ...] = ()
    procedural_retirement_reason_codes: tuple[str, ...] = ()
    planning_procedural_origins: tuple[str, ...] = ()
    planning_selected_skill_ids: tuple[str, ...] = ()
    planning_skill_rejection_reason_codes: tuple[str, ...] = ()
    planning_delta_operation_counts: tuple[str, ...] = ()
    procedural_high_risk_failure_signature_codes: tuple[str, ...] = ()
    procedural_follow_up_trace_ids: tuple[str, ...] = ()
    procedural_negative_transfer_reason_codes: tuple[str, ...] = ()
    private_working_memory_active_record_ids: tuple[str, ...] = ()
    private_working_memory_stale_record_ids: tuple[str, ...] = ()
    private_working_memory_resolved_record_ids: tuple[str, ...] = ()
    private_working_memory_unresolved_record_ids: tuple[str, ...] = ()
    private_working_memory_buffer_counts: dict[str, int] = field(default_factory=dict)
    private_working_memory_state_counts: dict[str, int] = field(default_factory=dict)
    scene_world_state_active_entity_ids: tuple[str, ...] = ()
    scene_world_state_stale_entity_ids: tuple[str, ...] = ()
    scene_world_state_contradicted_entity_ids: tuple[str, ...] = ()
    scene_world_state_expired_entity_ids: tuple[str, ...] = ()
    scene_world_state_active_affordance_ids: tuple[str, ...] = ()
    scene_world_state_uncertain_affordance_ids: tuple[str, ...] = ()
    scene_world_state_degraded_mode: str | None = None
    scene_world_state_degraded_reason_codes: tuple[str, ...] = ()
    scene_world_state_contradiction_codes: tuple[str, ...] = ()
    active_situation_active_record_ids: tuple[str, ...] = ()
    active_situation_stale_record_ids: tuple[str, ...] = ()
    active_situation_unresolved_record_ids: tuple[str, ...] = ()
    active_situation_kind_counts: dict[str, int] = field(default_factory=dict)
    active_situation_state_counts: dict[str, int] = field(default_factory=dict)
    active_situation_uncertainty_codes: tuple[str, ...] = ()
    active_situation_linked_commitment_ids: tuple[str, ...] = ()
    active_situation_linked_plan_proposal_ids: tuple[str, ...] = ()
    active_situation_linked_skill_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainContinuityEvalCase:
    """One scripted state-based continuity eval."""

    name: str
    description: str
    session_ids: BrainSessionIds
    steps: tuple[BrainConversationStep, ...]
    expected_state: BrainContinuityExpectedState
    presence_scope_key: str = "browser:presence"
    language: Language = Language.EN
    context_queries: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BrainContinuityEvalResult:
    """Result from one continuity eval run."""

    case: BrainContinuityEvalCase
    matched: bool
    mismatches: tuple[dict[str, Any], ...]
    actual_state: dict[str, Any]
    artifact_path: Path | None = None


class BrainContinuityEvalHarness:
    """Run deterministic scripted continuity evals against the local store."""

    def __init__(self, *, store: BrainStore):
        """Bind the harness to one canonical store."""
        self._store = store
        self._consolidator = BrainMemoryConsolidator(store=store)
        self._reflection = BrainReflectionEngine(store=store)

    def run_case(
        self,
        case: BrainContinuityEvalCase,
        *,
        output_path: Path | None = None,
    ) -> BrainContinuityEvalResult:
        """Run one scripted continuity eval and write an inspectable artifact."""
        replayed_events_by_session: dict[BrainSessionIds, list[Any]] = {}
        for step in case.steps:
            session_ids = step.session_ids or case.session_ids
            self._store.add_episode(
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                user_text=step.user_text,
                assistant_text=step.assistant_text,
                assistant_summary=step.assistant_summary,
                tool_calls=list(step.tool_calls),
            )
            for payload in step.remember_facts:
                self._store.remember_fact(
                    user_id=session_ids.user_id,
                    agent_id=session_ids.agent_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    **dict(payload),
                )
            for payload in step.upsert_tasks:
                self._store.upsert_task(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    agent_id=session_ids.agent_id,
                    session_id=session_ids.session_id,
                    **dict(payload),
                )
            for payload in step.task_status_updates:
                self._store.update_task_status(
                    user_id=session_ids.user_id,
                    agent_id=session_ids.agent_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    **dict(payload),
                )
            if step.events:
                replayed_events_by_session.setdefault(session_ids, []).extend(
                    append_replay_event_payloads(
                        store=self._store,
                        session_ids=session_ids,
                        payloads=step.events,
                    )
                )

        for session_ids, events in replayed_events_by_session.items():
            materialize_replayed_events(
                store=self._store,
                session_ids=session_ids,
                events=events,
            )

        self._consolidator.run_once(
            user_id=case.session_ids.user_id,
            thread_id=case.session_ids.thread_id,
        )
        self._reflection.run_once(
            user_id=case.session_ids.user_id,
            thread_id=case.session_ids.thread_id,
            session_ids=case.session_ids,
            trigger="manual",
        )
        actual_state = build_continuity_state(
            store=self._store,
            session_ids=case.session_ids,
            presence_scope_key=case.presence_scope_key,
            language=case.language,
            context_queries=case.context_queries,
        )
        mismatches = compare_continuity_state(actual_state, case.expected_state)
        artifact_path = output_path
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                json.dumps(
                    {
                        "case": {
                            "name": case.name,
                            "description": case.description,
                            "generated_at": _utc_now(),
                        },
                        "actual_state": actual_state,
                        "expected_state": _expected_state_as_dict(case.expected_state),
                        "matched": not mismatches,
                        "mismatches": mismatches,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return BrainContinuityEvalResult(
            case=case,
            matched=not mismatches,
            mismatches=tuple(mismatches),
            actual_state=actual_state,
            artifact_path=artifact_path,
        )


def build_continuity_state(
    *,
    store: BrainStore,
    session_ids: BrainSessionIds,
    presence_scope_key: str,
    language: Language,
    context_queries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured continuity-state payload for evals and audits."""
    surface_builder = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key=presence_scope_key,
        language=language,
    )
    bootstrap_snapshot = surface_builder.build(
        latest_user_text=_normalize_context_queries(context_queries).get("reply", ""),
        include_historical_claims=True,
    )
    recent_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=192,
    )
    recent_episodes = store.recent_episodes(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=1,
    )
    resolved_context_queries = _resolve_context_queries(
        snapshot=bootstrap_snapshot,
        recent_events=recent_events,
        recent_episodes=recent_episodes,
        context_queries=context_queries,
    )
    snapshot = surface_builder.build(
        latest_user_text=resolved_context_queries["reply"],
        include_historical_claims=True,
        task=BrainContextTask.REPLY,
    )
    task_surfaces = {
        task: surface_builder.build(
            latest_user_text=resolved_context_queries[task.value],
            task=task,
            include_historical_claims=True,
        )
        for task in all_brain_context_tasks()
    }
    state = build_continuity_state_from_surface(
        snapshot=snapshot,
        language=language,
        context_queries=resolved_context_queries,
        task_surfaces=task_surfaces,
    )
    continuity_graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    state["continuity_graph"] = continuity_graph.as_dict()
    state["continuity_dossiers"] = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=continuity_graph,
    ).as_dict()
    state["procedural_traces"] = store.build_procedural_trace_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    ).as_dict()
    state["procedural_skills"] = store.build_procedural_skill_projection(
        scope_type="thread",
        scope_id=session_ids.thread_id,
    ).as_dict()
    state["scene_world_state"] = store.build_scene_world_state_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    multimodal_autobiography_entries = store.autobiographical_entries(
        scope_type="presence",
        scope_id=presence_scope_key,
        entry_kinds=("scene_episode",),
        statuses=("current", "superseded"),
        modalities=("scene_world",),
        limit=24,
    )
    state["multimodal_autobiography"] = [
        record.as_dict()
        for entry in multimodal_autobiography_entries
        if (record := parse_multimodal_autobiography_record(entry)) is not None
    ]
    state["multimodal_autobiography_digest"] = build_multimodal_autobiography_digest(
        multimodal_autobiography_entries
    )
    state["private_working_memory"] = store.build_private_working_memory_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["predictive_world_model"] = store.build_predictive_world_model_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["counterfactual_rehearsal"] = store.build_counterfactual_rehearsal_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["embodied_executive"] = store.build_embodied_executive_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["practice_director"] = store.build_practice_director_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["skill_evidence_ledger"] = store.build_skill_evidence_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    ).as_dict()
    state["skill_governance"] = store.build_skill_governance_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    ).as_dict()
    state["adapter_governance"] = store.build_adapter_governance_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    ).as_dict()
    state["active_situation_model"] = store.build_active_situation_model_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    ).as_dict()
    state["reevaluation_digest"] = build_reevaluation_digest(
        autonomy_ledger=snapshot.autonomy_ledger,
        recent_events=recent_events,
    )
    state["wake_digest"] = build_wake_digest(
        commitment_projection=snapshot.commitment_projection,
        recent_events=recent_events,
    )
    state["planning_digest"] = build_planning_digest(
        agenda=snapshot.agenda,
        commitment_projection=snapshot.commitment_projection,
        recent_events=recent_events,
    )
    state["continuity_graph_digest"] = build_continuity_graph_digest(
        continuity_graph=state["continuity_graph"]
    )
    state["context_packet_digest"] = build_context_packet_digest(
        packet_traces=state.get("packet_traces")
    )
    state["claim_governance"] = (
        snapshot.claim_governance.as_dict() if snapshot.claim_governance is not None else {}
    )
    state["continuity_governance_report"] = build_continuity_governance_report(
        continuity_dossiers=state["continuity_dossiers"],
        continuity_graph=state["continuity_graph"],
        claim_governance=state.get("claim_governance"),
        packet_traces=state.get("packet_traces"),
        embodied_executive=state.get("embodied_executive"),
    )
    state["practice_digest"] = build_practice_inspection(
        practice_projection=store.build_practice_director_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=presence_scope_key,
        ),
    )
    state["skill_evidence_digest"] = build_skill_evidence_inspection(
        skill_evidence_ledger=store.build_skill_evidence_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
        skill_governance=store.build_skill_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
    )
    state["adapter_governance_digest"] = build_adapter_governance_inspection(
        adapter_governance=store.build_adapter_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
    )
    state["sim_to_real_digest"] = build_sim_to_real_digest(
        adapter_governance=store.build_adapter_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
    )
    state["procedural_skill_digest"] = build_procedural_skill_digest(
        procedural_skills=state["procedural_skills"],
    )
    state["procedural_skill_governance_report"] = build_procedural_skill_governance_report(
        procedural_skills=state["procedural_skills"],
        procedural_traces=state["procedural_traces"],
        planning_digest=state["planning_digest"],
    )
    state["executive_policy_audit"] = build_executive_policy_audit(
        autonomy_digest=state["autonomy_digest"],
        reevaluation_digest=state["reevaluation_digest"],
        wake_digest=state["wake_digest"],
        planning_digest=state["planning_digest"],
        procedural_skill_governance_report=state["procedural_skill_governance_report"],
    )
    recent_action_events = _recent_action_event_rows(
        store=store,
        session_ids=session_ids,
        limit=12,
    )
    state["runtime_shell_digest"] = build_runtime_shell_digest(
        recent_events=recent_events,
        reflection_cycles=[
            record.as_dict()
            for record in store.list_reflection_cycles(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=12,
            )
        ],
        memory_exports=store.list_memory_exports(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        ),
        counterfactual_rehearsal=state["counterfactual_rehearsal"],
        predictive_world_model=state["predictive_world_model"],
        embodied_executive=state["embodied_executive"],
        practice_director=store.build_practice_director_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=presence_scope_key,
        ),
        skill_evidence_ledger=store.build_skill_evidence_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
        skill_governance=store.build_skill_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
        adapter_governance=store.build_adapter_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        ),
        recent_action_events=recent_action_events,
        multimodal_autobiography=list(state["multimodal_autobiography"]),
        packet_traces=state.get("packet_traces"),
    )
    state["rehearsal_digest"] = build_counterfactual_rehearsal_digest(
        counterfactual_rehearsal=state["counterfactual_rehearsal"]
    )
    state["predictive_digest"] = build_world_model_digest(
        predictive_world_model=state["predictive_world_model"]
    )
    state["embodied_digest"] = build_embodied_executive_digest(
        embodied_executive=state["embodied_executive"],
        recent_action_events=recent_action_events,
    )
    state["private_working_memory_digest"] = build_private_working_memory_digest(
        private_working_memory=state["private_working_memory"],
    )
    state["scene_world_state_digest"] = build_scene_world_state_digest(
        scene_world_state=state["scene_world_state"],
    )
    state["active_situation_model_digest"] = build_active_situation_model_digest(
        active_situation_model=state["active_situation_model"],
    )
    state["procedural_qa_report"] = build_procedural_qa_report(
        procedural_skill_digest=state["procedural_skill_digest"],
        procedural_skill_governance_report=state["procedural_skill_governance_report"],
        planning_digest=state["planning_digest"],
    )
    state["reflection_cycles"] = [
        record.as_dict()
        for record in store.list_reflection_cycles(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        )
    ]
    state["latest_reflection_draft_path"] = next(
        (
            record["draft_artifact_path"]
            for record in state["reflection_cycles"]
            if record.get("draft_artifact_path")
        ),
        None,
    )
    state["core_block_versions"] = {
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
            for version in store.list_core_memory_block_versions(
                block_kind=record.block_kind,
                scope_type=record.scope_type,
                scope_id=record.scope_id,
            )
        ]
        for name, record in snapshot.core_blocks.items()
    }
    state["context_queries"] = dict(resolved_context_queries)
    return state


def build_continuity_state_from_surface(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    planning_snapshot: BrainContextSurfaceSnapshot | None = None,
    task_surfaces: dict[BrainContextTask, BrainContextSurfaceSnapshot] | None = None,
    language: Language,
    scope_type: str = "surface",
    scope_id: str = "surface",
    context_queries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one context surface into a structured continuity-state payload."""
    planning_surface = planning_snapshot or snapshot
    resolved_context_queries = _resolve_context_queries(
        snapshot=snapshot,
        recent_events=[],
        recent_episodes=[],
        context_queries=context_queries,
    )
    selector = BrainContextSelector()
    resolved_task_surfaces = task_surfaces or {
        BrainContextTask.REPLY: snapshot,
        BrainContextTask.PLANNING: planning_surface,
        BrainContextTask.RECALL: snapshot,
        BrainContextTask.REFLECTION: planning_surface,
        BrainContextTask.CRITIQUE: planning_surface,
        BrainContextTask.WAKE: planning_surface,
        BrainContextTask.REEVALUATION: planning_surface,
        BrainContextTask.OPERATOR_AUDIT: planning_surface,
        BrainContextTask.GOVERNANCE_REVIEW: planning_surface,
    }
    selected_contexts = {
        task.value: selector.select(
            snapshot=task_snapshot,
            task=task,
            language=language,
            static_sections={},
        )
        for task, task_snapshot in resolved_task_surfaces.items()
    }
    compiled_packets = {
        task.value: compile_context_packet_from_surface(
            snapshot=task_snapshot,
            latest_user_text=resolved_context_queries[task.value],
            task=task,
            language=language,
        )
        for task, task_snapshot in resolved_task_surfaces.items()
    }
    relationship_arc = next(
        (entry for entry in snapshot.autobiography if entry.entry_kind == "relationship_arc"),
        None,
    )
    health_findings = (
        list(snapshot.health_summary.findings) if snapshot.health_summary is not None else []
    )
    health_findings.extend(_derive_visual_health_findings(snapshot))
    deduped_findings: list[dict[str, Any]] = []
    seen_finding_keys: set[tuple[str, str]] = set()
    for finding in health_findings:
        if not isinstance(finding, dict):
            continue
        key = (str(finding.get("code", "")), str(finding.get("severity", "")))
        if key in seen_finding_keys:
            continue
        seen_finding_keys.add(key)
        deduped_findings.append(finding)
    placeholder_graph = snapshot.continuity_graph or build_continuity_graph_projection(
        scope_type=scope_type,
        scope_id=scope_id,
        current_claims=snapshot.current_claims,
        historical_claims=snapshot.historical_claims,
        claim_supersessions=snapshot.claim_supersessions,
        autobiography=_combined_surface_autobiography(snapshot),
        commitment_projection=snapshot.commitment_projection,
        core_blocks=tuple(snapshot.core_blocks.values()),
        procedural_skills=snapshot.procedural_skills,
        scene_world_state=snapshot.scene_world_state,
        recent_events=[],
    )
    placeholder_dossiers = snapshot.continuity_dossiers or build_continuity_dossier_projection(
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=scope_id,
        current_claims=snapshot.current_claims,
        historical_claims=snapshot.historical_claims,
        claim_supersessions=snapshot.claim_supersessions,
        autobiography=_combined_surface_autobiography(snapshot),
        continuity_graph=placeholder_graph,
        core_blocks=tuple(snapshot.core_blocks.values()),
        commitment_projection=snapshot.commitment_projection,
        agenda=snapshot.agenda,
        procedural_skills=snapshot.procedural_skills,
        scene_world_state=snapshot.scene_world_state,
        recent_events=[],
    )
    packet_traces = {
        task: (packet.packet_trace.as_dict() if packet.packet_trace is not None else None)
        for task, packet in compiled_packets.items()
    }
    continuity_graph_dict = placeholder_graph.as_dict()
    continuity_dossiers_dict = placeholder_dossiers.as_dict()
    claim_governance_dict = (
        snapshot.claim_governance.as_dict() if snapshot.claim_governance is not None else {}
    )
    planning_digest = build_planning_digest(
        agenda=snapshot.agenda,
        commitment_projection=snapshot.commitment_projection,
        recent_events=[],
    )
    procedural_trace_projection = BrainProceduralTraceProjection(
        scope_type=scope_type,
        scope_id=scope_id,
    )
    procedural_skill_projection = snapshot.procedural_skills or BrainProceduralSkillProjection(
        scope_type=scope_type,
        scope_id=scope_id,
    )
    scene_world_state_projection = (
        snapshot.scene_world_state
        if snapshot.scene_world_state is not None
        else BrainSceneWorldProjection(scope_type="presence", scope_id="surface")
    )
    private_working_memory_projection = (
        snapshot.private_working_memory
        if snapshot.private_working_memory is not None
        else BrainPrivateWorkingMemoryProjection(scope_type=scope_type, scope_id=scope_id)
    )
    predictive_world_model_projection = (
        snapshot.predictive_world_model
        if snapshot.predictive_world_model is not None
        else BrainPredictiveWorldModelProjection(
            scope_key=scope_id,
            presence_scope_key="surface:presence",
        )
    )
    counterfactual_rehearsal_projection = BrainCounterfactualRehearsalProjection(
        scope_key=scope_id,
        presence_scope_key="surface:presence",
        updated_at="",
    )
    embodied_executive_projection = (
        snapshot.embodied_executive
        if snapshot.embodied_executive is not None
        else BrainEmbodiedExecutiveProjection(
            scope_key=scope_id,
            presence_scope_key="surface:presence",
        )
    )
    active_situation_projection = (
        snapshot.active_situation_model
        if snapshot.active_situation_model is not None
        else BrainActiveSituationProjection(scope_type=scope_type, scope_id=scope_id)
    )
    procedural_trace_dict = procedural_trace_projection.as_dict()
    procedural_skill_dict = procedural_skill_projection.as_dict()
    scene_world_state_dict = scene_world_state_projection.as_dict()
    private_working_memory_dict = private_working_memory_projection.as_dict()
    predictive_world_model_dict = predictive_world_model_projection.as_dict()
    counterfactual_rehearsal_dict = counterfactual_rehearsal_projection.as_dict()
    embodied_executive_dict = embodied_executive_projection.as_dict()
    active_situation_dict = active_situation_projection.as_dict()
    procedural_skill_digest = build_procedural_skill_digest(
        procedural_skills=procedural_skill_dict,
    )
    procedural_skill_governance_report = build_procedural_skill_governance_report(
        procedural_skills=procedural_skill_dict,
        procedural_traces=procedural_trace_dict,
        planning_digest=planning_digest,
    )
    multimodal_autobiography = [
        record.as_dict()
        for entry in snapshot.scene_episodes
        if (record := parse_multimodal_autobiography_record(entry)) is not None
    ]
    return {
        "generated_at": _utc_now(),
        "core_blocks": {
            name: {
                "block_id": record.block_id,
                "version": record.version,
                "status": record.status,
                "content": record.content,
            }
            for name, record in snapshot.core_blocks.items()
        },
        "current_claims": [
            {
                "claim_id": record.claim_id,
                "predicate": record.predicate,
                "value": record.object.get("value"),
                "status": record.status,
                "source_event_id": record.source_event_id,
            }
            for record in snapshot.current_claims
        ],
        "historical_claims": [
            {
                "claim_id": record.claim_id,
                "predicate": record.predicate,
                "value": record.object.get("value"),
                "status": record.status,
                "source_event_id": record.source_event_id,
            }
            for record in snapshot.historical_claims
        ],
        "claim_supersessions": [
            {
                "prior_claim_id": record.prior_claim_id,
                "new_claim_id": record.new_claim_id,
                "reason": record.reason,
            }
            for record in snapshot.claim_supersessions
        ],
        "agenda": snapshot.agenda.as_dict(),
        "commitment_projection": snapshot.commitment_projection.as_dict(),
        "autonomy_ledger": snapshot.autonomy_ledger.as_dict(),
        "autonomy_digest": build_autonomy_digest(
            autonomy_ledger=snapshot.autonomy_ledger,
            agenda=snapshot.agenda,
        ),
        "reevaluation_digest": build_reevaluation_digest(
            autonomy_ledger=snapshot.autonomy_ledger,
            recent_events=[],
        ),
        "wake_digest": build_wake_digest(
            commitment_projection=snapshot.commitment_projection,
            recent_events=[],
        ),
        "planning_digest": planning_digest,
        "continuity_graph": continuity_graph_dict,
        "continuity_dossiers": continuity_dossiers_dict,
        "claim_governance": claim_governance_dict,
        "procedural_traces": procedural_trace_dict,
        "procedural_skills": procedural_skill_dict,
        "scene_world_state": scene_world_state_dict,
        "private_working_memory": private_working_memory_dict,
        "predictive_world_model": predictive_world_model_dict,
        "counterfactual_rehearsal": counterfactual_rehearsal_dict,
        "embodied_executive": embodied_executive_dict,
        "practice_director": {},
        "skill_evidence_ledger": {},
        "skill_governance": {},
        "adapter_governance": {},
        "active_situation_model": active_situation_dict,
        "continuity_graph_digest": build_continuity_graph_digest(
            continuity_graph=continuity_graph_dict
        ),
        "continuity_governance_report": build_continuity_governance_report(
            continuity_dossiers=continuity_dossiers_dict,
            continuity_graph=continuity_graph_dict,
            claim_governance=claim_governance_dict,
            packet_traces=packet_traces,
            embodied_executive=embodied_executive_dict,
        ),
        "practice_digest": {},
        "skill_evidence_digest": {},
        "adapter_governance_digest": {},
        "sim_to_real_digest": {},
        "procedural_skill_digest": procedural_skill_digest,
        "procedural_skill_governance_report": procedural_skill_governance_report,
        "multimodal_autobiography": multimodal_autobiography,
        "multimodal_autobiography_digest": build_multimodal_autobiography_digest(
            snapshot.scene_episodes
        ),
        "executive_policy_audit": build_executive_policy_audit(
            autonomy_digest=build_autonomy_digest(
                autonomy_ledger=snapshot.autonomy_ledger,
                agenda=snapshot.agenda,
            ),
            reevaluation_digest=build_reevaluation_digest(
                autonomy_ledger=snapshot.autonomy_ledger,
                recent_events=[],
            ),
            wake_digest=build_wake_digest(
                commitment_projection=snapshot.commitment_projection,
                recent_events=[],
            ),
            planning_digest=planning_digest,
            procedural_skill_governance_report=procedural_skill_governance_report,
        ),
        "runtime_shell_digest": build_runtime_shell_digest(
            recent_events=[],
            reflection_cycles=[],
            memory_exports=[],
            counterfactual_rehearsal=counterfactual_rehearsal_dict,
            predictive_world_model=predictive_world_model_dict,
            embodied_executive=embodied_executive_dict,
            practice_director=None,
            skill_evidence_ledger=None,
            skill_governance=None,
            adapter_governance=None,
            recent_action_events=[],
            multimodal_autobiography=multimodal_autobiography,
            packet_traces=packet_traces,
        ),
        "rehearsal_digest": build_counterfactual_rehearsal_digest(
            counterfactual_rehearsal=counterfactual_rehearsal_dict,
        ),
        "predictive_digest": build_world_model_digest(
            predictive_world_model=predictive_world_model_dict,
        ),
        "embodied_digest": build_embodied_executive_digest(
            embodied_executive=embodied_executive_dict,
            recent_action_events=[],
        ),
        "private_working_memory_digest": build_private_working_memory_digest(
            private_working_memory=private_working_memory_dict,
        ),
        "scene_world_state_digest": build_scene_world_state_digest(
            scene_world_state=scene_world_state_dict,
        ),
        "active_situation_model_digest": build_active_situation_model_digest(
            active_situation_model=active_situation_dict,
        ),
        "procedural_qa_report": build_procedural_qa_report(
            procedural_skill_digest=procedural_skill_digest,
            procedural_skill_governance_report=procedural_skill_governance_report,
            planning_digest=planning_digest,
        ),
        "autobiography": [
            {
                "entry_id": entry.entry_id,
                "entry_kind": entry.entry_kind,
                "rendered_summary": entry.rendered_summary,
                "status": entry.status,
                "modality": entry.modality,
                "review_state": entry.review_state,
                "retention_class": entry.retention_class,
                "privacy_class": entry.privacy_class,
                "governance_reason_codes": list(entry.governance_reason_codes),
                "source_presence_scope_key": entry.source_presence_scope_key,
                "source_scene_entity_ids": list(entry.source_scene_entity_ids),
                "source_scene_affordance_ids": list(entry.source_scene_affordance_ids),
                "redacted_at": entry.redacted_at,
            }
            for entry in snapshot.autobiography
        ],
        "relationship_arc_summary": relationship_arc.rendered_summary
        if relationship_arc is not None
        else None,
        "memory_health": (
            {
                "report_id": snapshot.health_summary.report_id,
                "status": snapshot.health_summary.status,
                "score": snapshot.health_summary.score,
                "findings": deduped_findings,
            }
            if snapshot.health_summary is not None
            else {
                "report_id": None,
                "status": "runtime_only",
                "score": None,
                "findings": deduped_findings,
            }
        ),
        "visual_health": {
            "camera_connected": snapshot.scene.camera_connected,
            "camera_track_state": snapshot.scene.camera_track_state,
            "person_present": snapshot.scene.person_present,
            "last_fresh_frame_at": snapshot.scene.last_fresh_frame_at,
            "frame_age_ms": snapshot.scene.frame_age_ms,
            "detection_backend": snapshot.scene.detection_backend,
            "detection_confidence": snapshot.scene.detection_confidence,
            "sensor_health_reason": snapshot.scene.sensor_health_reason
            or snapshot.body.sensor_health_reason,
            "recovery_in_progress": snapshot.scene.recovery_in_progress
            or snapshot.body.recovery_in_progress,
            "recovery_attempts": max(
                snapshot.scene.recovery_attempts, snapshot.body.recovery_attempts
            ),
            "vision_enrichment_available": snapshot.scene.enrichment_available,
        },
        "selection_traces": {
            task: selected_context.selection_trace.as_dict()
            for task, selected_context in selected_contexts.items()
        },
        "packet_traces": {
            **packet_traces,
        },
        "context_packet_digest": build_context_packet_digest(packet_traces=packet_traces),
        "selection_sections": {
            task: [section.key for section in selected_context.sections]
            for task, selected_context in selected_contexts.items()
        },
        "context_queries": dict(resolved_context_queries),
        "claim_counts": {
            "current": len(snapshot.current_claims),
            "historical": len(snapshot.historical_claims),
            "uncertain_current": sum(
                1 for record in snapshot.current_claims if record.status == "uncertain"
            ),
            "supersession_links": len(snapshot.claim_supersessions),
        },
    }


def compare_continuity_state(
    actual_state: dict[str, Any],
    expected_state: BrainContinuityExpectedState,
) -> list[dict[str, Any]]:
    """Compare actual continuity state against the structured expected state."""
    mismatches: list[dict[str, Any]] = []
    current_claims = list(actual_state.get("current_claims", []))
    historical_claims = list(actual_state.get("historical_claims", []))
    core_blocks = dict(actual_state.get("core_blocks", {}))
    commitment_projection = dict(actual_state.get("commitment_projection", {}))
    agenda = dict(actual_state.get("agenda", {}))
    autonomy_digest = dict(actual_state.get("autonomy_digest", {}))
    reevaluation_digest = dict(actual_state.get("reevaluation_digest", {}))
    wake_digest = dict(actual_state.get("wake_digest", {}))
    planning_digest = dict(actual_state.get("planning_digest", {}))
    continuity_graph = dict(actual_state.get("continuity_graph", {}))
    continuity_dossiers = dict(actual_state.get("continuity_dossiers", {}))
    continuity_graph_digest = dict(actual_state.get("continuity_graph_digest", {}))
    context_packet_digest = dict(actual_state.get("context_packet_digest", {}))
    continuity_governance_report = dict(actual_state.get("continuity_governance_report", {}))
    procedural_skills = dict(actual_state.get("procedural_skills", {}))
    procedural_skill_digest = dict(actual_state.get("procedural_skill_digest", {}))
    procedural_skill_governance_report = dict(
        actual_state.get("procedural_skill_governance_report", {})
    )
    scene_world_state = dict(actual_state.get("scene_world_state", {}))
    scene_world_state_digest = dict(actual_state.get("scene_world_state_digest", {}))
    private_working_memory = dict(actual_state.get("private_working_memory", {}))
    private_working_memory_digest = dict(actual_state.get("private_working_memory_digest", {}))
    active_situation_model = dict(actual_state.get("active_situation_model", {}))
    active_situation_model_digest = dict(actual_state.get("active_situation_model_digest", {}))
    autobiography = list(actual_state.get("autobiography", []))
    health = actual_state.get("memory_health") or {}

    for expected in expected_state.current_claims:
        if not any(_claim_matches(claim, expected) for claim in current_claims):
            mismatches.append(
                {
                    "path": "current_claims",
                    "expected": expected,
                    "actual": current_claims,
                }
            )

    for expected in expected_state.historical_claims:
        if not any(_claim_matches(claim, expected) for claim in historical_claims):
            mismatches.append(
                {
                    "path": "historical_claims",
                    "expected": expected,
                    "actual": historical_claims,
                }
            )

    for block_kind in expected_state.core_blocks_present:
        if block_kind not in core_blocks:
            mismatches.append(
                {
                    "path": "core_blocks",
                    "expected": block_kind,
                    "actual": list(core_blocks),
                }
            )

    for status, titles in expected_state.commitment_titles_by_status.items():
        actual_titles = [
            record.get("title") for record in commitment_projection.get(f"{status}_commitments", [])
        ]
        missing = [title for title in titles if title not in actual_titles]
        if missing:
            mismatches.append(
                {
                    "path": f"commitment_projection.{status}_commitments",
                    "expected": list(titles),
                    "actual": actual_titles,
                }
            )

    for field_name, values in expected_state.agenda_contains.items():
        actual_values = list(agenda.get(field_name, []))
        missing = [value for value in values if value not in actual_values]
        if missing:
            mismatches.append(
                {
                    "path": f"agenda.{field_name}",
                    "expected": list(values),
                    "actual": actual_values,
                }
            )

    if expected_state.autobiography_entry_kinds:
        actual_kinds = [entry.get("entry_kind") for entry in autobiography]
        missing = [
            kind for kind in expected_state.autobiography_entry_kinds if kind not in actual_kinds
        ]
        if missing:
            mismatches.append(
                {
                    "path": "autobiography.entry_kind",
                    "expected": list(expected_state.autobiography_entry_kinds),
                    "actual": actual_kinds,
                }
            )

    actual_codes = [
        finding.get("code")
        for finding in health.get("findings", [])
        if isinstance(finding, dict) and finding.get("code")
    ]
    for code in expected_state.health_finding_codes:
        if code not in actual_codes:
            mismatches.append(
                {
                    "path": "memory_health.findings",
                    "expected": list(expected_state.health_finding_codes),
                    "actual": actual_codes,
                }
            )

    actual_score = health.get("score")
    if expected_state.health_score_min is not None and (
        actual_score is None or float(actual_score) < expected_state.health_score_min
    ):
        mismatches.append(
            {
                "path": "memory_health.score",
                "expected": {"min": expected_state.health_score_min},
                "actual": actual_score,
            }
        )
    if expected_state.health_score_max is not None and (
        actual_score is None or float(actual_score) > expected_state.health_score_max
    ):
        mismatches.append(
            {
                "path": "memory_health.score",
                "expected": {"max": expected_state.health_score_max},
                "actual": actual_score,
            }
        )

    if expected_state.relationship_arc_contains:
        summary = str(actual_state.get("relationship_arc_summary") or "")
        if expected_state.relationship_arc_contains not in summary:
            mismatches.append(
                {
                    "path": "relationship_arc_summary",
                    "expected": expected_state.relationship_arc_contains,
                    "actual": summary,
                }
            )

    actual_current_candidate_ids = [
        item.get("candidate_goal_id")
        for item in autonomy_digest.get("current_candidates", [])
        if item.get("candidate_goal_id")
    ]
    missing_candidate_ids = [
        candidate_goal_id
        for candidate_goal_id in expected_state.autonomy_current_candidate_ids
        if candidate_goal_id not in actual_current_candidate_ids
    ]
    if missing_candidate_ids:
        mismatches.append(
            {
                "path": "autonomy_digest.current_candidates.candidate_goal_id",
                "expected": list(expected_state.autonomy_current_candidate_ids),
                "actual": actual_current_candidate_ids,
            }
        )

    actual_current_candidate_summaries = [
        item.get("summary")
        for item in autonomy_digest.get("current_candidates", [])
        if item.get("summary")
    ]
    missing_candidate_summaries = [
        summary
        for summary in expected_state.autonomy_current_candidate_summaries
        if summary not in actual_current_candidate_summaries
    ]
    if missing_candidate_summaries:
        mismatches.append(
            {
                "path": "autonomy_digest.current_candidates.summary",
                "expected": list(expected_state.autonomy_current_candidate_summaries),
                "actual": actual_current_candidate_summaries,
            }
        )

    actual_recent_decision_kinds = [
        item.get("decision_kind")
        for item in autonomy_digest.get("recent_decisions", [])
        if item.get("decision_kind")
    ]
    missing_recent_decision_kinds = [
        decision_kind
        for decision_kind in expected_state.autonomy_recent_decision_kinds
        if decision_kind not in actual_recent_decision_kinds
    ]
    if missing_recent_decision_kinds:
        mismatches.append(
            {
                "path": "autonomy_digest.recent_decisions.decision_kind",
                "expected": list(expected_state.autonomy_recent_decision_kinds),
                "actual": actual_recent_decision_kinds,
            }
        )

    actual_reason_codes = list(dict(autonomy_digest.get("reason_counts", {})))
    missing_reason_codes = [
        code for code in expected_state.autonomy_reason_codes if code not in actual_reason_codes
    ]
    if missing_reason_codes:
        mismatches.append(
            {
                "path": "autonomy_digest.reason_counts",
                "expected": list(expected_state.autonomy_reason_codes),
                "actual": actual_reason_codes,
            }
        )

    actual_reevaluation_conditions = [
        item.get("expected_reevaluation_condition")
        for item in autonomy_digest.get("recent_non_actions", [])
        if item.get("expected_reevaluation_condition")
    ]
    missing_reevaluation_conditions = [
        condition
        for condition in expected_state.autonomy_reevaluation_conditions
        if condition not in actual_reevaluation_conditions
    ]
    if missing_reevaluation_conditions:
        mismatches.append(
            {
                "path": "autonomy_digest.recent_non_actions.expected_reevaluation_condition",
                "expected": list(expected_state.autonomy_reevaluation_conditions),
                "actual": actual_reevaluation_conditions,
            }
        )

    actual_current_hold_ids = [
        item.get("candidate_goal_id")
        for item in reevaluation_digest.get("current_holds", [])
        if item.get("candidate_goal_id")
    ]
    missing_current_hold_ids = [
        candidate_goal_id
        for candidate_goal_id in expected_state.reevaluation_current_hold_ids
        if candidate_goal_id not in actual_current_hold_ids
    ]
    if missing_current_hold_ids:
        mismatches.append(
            {
                "path": "reevaluation_digest.current_holds.candidate_goal_id",
                "expected": list(expected_state.reevaluation_current_hold_ids),
                "actual": actual_current_hold_ids,
            }
        )

    actual_trigger_kinds = [
        item.get("kind")
        for item in reevaluation_digest.get("recent_triggers", [])
        if item.get("kind")
    ]
    missing_trigger_kinds = [
        kind
        for kind in expected_state.reevaluation_trigger_kinds
        if kind not in actual_trigger_kinds
    ]
    if missing_trigger_kinds:
        mismatches.append(
            {
                "path": "reevaluation_digest.recent_triggers.kind",
                "expected": list(expected_state.reevaluation_trigger_kinds),
                "actual": actual_trigger_kinds,
            }
        )

    actual_transition_candidate_ids = [
        item.get("candidate_goal_id")
        for item in reevaluation_digest.get("recent_transitions", [])
        if item.get("candidate_goal_id")
    ]
    missing_transition_candidate_ids = [
        candidate_goal_id
        for candidate_goal_id in expected_state.reevaluation_transition_candidate_ids
        if candidate_goal_id not in actual_transition_candidate_ids
    ]
    if missing_transition_candidate_ids:
        mismatches.append(
            {
                "path": "reevaluation_digest.recent_transitions.candidate_goal_id",
                "expected": list(expected_state.reevaluation_transition_candidate_ids),
                "actual": actual_transition_candidate_ids,
            }
        )

    actual_transition_outcomes = [
        item.get("outcome_decision_kind")
        for item in reevaluation_digest.get("recent_transitions", [])
        if item.get("outcome_decision_kind")
    ]
    missing_transition_outcomes = [
        decision_kind
        for decision_kind in expected_state.reevaluation_transition_outcome_kinds
        if decision_kind not in actual_transition_outcomes
    ]
    if missing_transition_outcomes:
        mismatches.append(
            {
                "path": "reevaluation_digest.recent_transitions.outcome_decision_kind",
                "expected": list(expected_state.reevaluation_transition_outcome_kinds),
                "actual": actual_transition_outcomes,
            }
        )

    actual_waiting_commitment_ids = [
        item.get("commitment_id")
        for item in wake_digest.get("current_waiting_commitments", [])
        if item.get("commitment_id")
    ]
    missing_waiting_commitment_ids = [
        commitment_id
        for commitment_id in expected_state.wake_current_waiting_commitment_ids
        if commitment_id not in actual_waiting_commitment_ids
    ]
    if missing_waiting_commitment_ids:
        mismatches.append(
            {
                "path": "wake_digest.current_waiting_commitments.commitment_id",
                "expected": list(expected_state.wake_current_waiting_commitment_ids),
                "actual": actual_waiting_commitment_ids,
            }
        )

    actual_wake_kinds = [
        item.get("wake_kind")
        for item in wake_digest.get("recent_triggers", [])
        if item.get("wake_kind")
    ]
    missing_wake_kinds = [
        wake_kind
        for wake_kind in expected_state.wake_recent_wake_kinds
        if wake_kind not in actual_wake_kinds
    ]
    if missing_wake_kinds:
        mismatches.append(
            {
                "path": "wake_digest.recent_triggers.wake_kind",
                "expected": list(expected_state.wake_recent_wake_kinds),
                "actual": actual_wake_kinds,
            }
        )

    actual_route_kinds = [
        item.get("route_kind")
        for item in wake_digest.get("recent_triggers", [])
        if item.get("route_kind")
    ]
    missing_route_kinds = [
        route_kind
        for route_kind in expected_state.wake_recent_route_kinds
        if route_kind not in actual_route_kinds
    ]
    if missing_route_kinds:
        mismatches.append(
            {
                "path": "wake_digest.recent_triggers.route_kind",
                "expected": list(expected_state.wake_recent_route_kinds),
                "actual": actual_route_kinds,
            }
        )

    actual_wake_reason_codes = list(dict(wake_digest.get("reason_counts", {})))
    missing_wake_reason_codes = [
        reason
        for reason in expected_state.wake_reason_codes
        if reason not in actual_wake_reason_codes
    ]
    if missing_wake_reason_codes:
        mismatches.append(
            {
                "path": "wake_digest.reason_counts",
                "expected": list(expected_state.wake_reason_codes),
                "actual": actual_wake_reason_codes,
            }
        )

    actual_pending_proposal_ids = [
        item.get("plan_proposal_id")
        for item in planning_digest.get("current_pending_proposals", [])
        if item.get("plan_proposal_id")
    ]
    missing_pending_proposal_ids = [
        proposal_id
        for proposal_id in expected_state.planning_current_pending_proposal_ids
        if proposal_id not in actual_pending_proposal_ids
    ]
    if missing_pending_proposal_ids:
        mismatches.append(
            {
                "path": "planning_digest.current_pending_proposals.plan_proposal_id",
                "expected": list(expected_state.planning_current_pending_proposal_ids),
                "actual": actual_pending_proposal_ids,
            }
        )

    actual_pending_goal_ids = [
        item.get("goal_id")
        for item in planning_digest.get("current_pending_proposals", [])
        if item.get("goal_id")
    ]
    missing_pending_goal_ids = [
        goal_id
        for goal_id in expected_state.planning_current_pending_goal_ids
        if goal_id not in actual_pending_goal_ids
    ]
    if missing_pending_goal_ids:
        mismatches.append(
            {
                "path": "planning_digest.current_pending_proposals.goal_id",
                "expected": list(expected_state.planning_current_pending_goal_ids),
                "actual": actual_pending_goal_ids,
            }
        )

    actual_review_policies = [
        item.get("review_policy")
        for item in planning_digest.get("recent_proposals", [])
        if item.get("review_policy")
    ]
    missing_review_policies = [
        review_policy
        for review_policy in expected_state.planning_recent_review_policies
        if review_policy not in actual_review_policies
    ]
    if missing_review_policies:
        mismatches.append(
            {
                "path": "planning_digest.recent_proposals.review_policy",
                "expected": list(expected_state.planning_recent_review_policies),
                "actual": actual_review_policies,
            }
        )

    actual_outcome_kinds = (
        ["adopted" for _ in planning_digest.get("recent_adoptions", [])]
        + ["rejected" for _ in planning_digest.get("recent_rejections", [])]
        + [
            (
                "pending_user_review"
                if item.get("review_policy") == "needs_user_review"
                else "pending_operator_review"
                if item.get("review_policy") == "needs_operator_review"
                else None
            )
            for item in planning_digest.get("current_pending_proposals", [])
            if item.get("review_policy")
        ]
        + [
            item.get("outcome_kind")
            for item in planning_digest.get("recent_revision_flows", [])
            if item.get("outcome_kind")
        ]
    )
    missing_outcome_kinds = [
        outcome_kind
        for outcome_kind in expected_state.planning_recent_outcome_kinds
        if outcome_kind not in actual_outcome_kinds
    ]
    if missing_outcome_kinds:
        mismatches.append(
            {
                "path": "planning_digest.outcomes",
                "expected": list(expected_state.planning_recent_outcome_kinds),
                "actual": actual_outcome_kinds,
            }
        )

    actual_planning_reason_codes = list(dict(planning_digest.get("reason_counts", {})))
    missing_planning_reason_codes = [
        reason
        for reason in expected_state.planning_reason_codes
        if reason not in actual_planning_reason_codes
    ]
    if missing_planning_reason_codes:
        mismatches.append(
            {
                "path": "planning_digest.reason_counts",
                "expected": list(expected_state.planning_reason_codes),
                "actual": actual_planning_reason_codes,
            }
        )

    actual_revision_goal_ids = [
        item.get("goal_id")
        for item in planning_digest.get("recent_revision_flows", [])
        if item.get("goal_id")
    ]
    missing_revision_goal_ids = [
        goal_id
        for goal_id in expected_state.planning_recent_revision_goal_ids
        if goal_id not in actual_revision_goal_ids
    ]
    if missing_revision_goal_ids:
        mismatches.append(
            {
                "path": "planning_digest.recent_revision_flows.goal_id",
                "expected": list(expected_state.planning_recent_revision_goal_ids),
                "actual": actual_revision_goal_ids,
            }
        )

    graph_nodes = list(continuity_graph.get("nodes", []))
    graph_edges = list(continuity_graph.get("edges", []))
    graph_current_ids = set(continuity_graph.get("current_node_ids", []))
    graph_historical_ids = set(continuity_graph.get("historical_node_ids", []))
    graph_stale_ids = set(continuity_graph.get("stale_node_ids", []))
    graph_superseded_ids = set(continuity_graph.get("superseded_node_ids", []))
    backing_record_by_node_id = {
        str(node.get("node_id")): str(node.get("backing_record_id"))
        for node in graph_nodes
        if node.get("node_id") and node.get("backing_record_id")
    }
    actual_graph_current_backing_ids = sorted(
        {
            backing_record_by_node_id[node_id]
            for node_id in graph_current_ids
            if node_id in backing_record_by_node_id
        }
    )
    missing_graph_current_backing_ids = [
        backing_id
        for backing_id in expected_state.continuity_graph_current_backing_ids
        if backing_id not in actual_graph_current_backing_ids
    ]
    if missing_graph_current_backing_ids:
        mismatches.append(
            {
                "path": "continuity_graph.current_node_ids",
                "expected": list(expected_state.continuity_graph_current_backing_ids),
                "actual": actual_graph_current_backing_ids,
            }
        )

    actual_graph_historical_backing_ids = sorted(
        {
            backing_record_by_node_id[node_id]
            for node_id in graph_historical_ids
            if node_id in backing_record_by_node_id
        }
    )
    missing_graph_historical_backing_ids = [
        backing_id
        for backing_id in expected_state.continuity_graph_historical_backing_ids
        if backing_id not in actual_graph_historical_backing_ids
    ]
    if missing_graph_historical_backing_ids:
        mismatches.append(
            {
                "path": "continuity_graph.historical_node_ids",
                "expected": list(expected_state.continuity_graph_historical_backing_ids),
                "actual": actual_graph_historical_backing_ids,
            }
        )

    actual_graph_edge_kinds = sorted(
        {str(edge.get("kind")) for edge in graph_edges if str(edge.get("kind", "")).strip()}
    )
    missing_graph_edge_kinds = [
        edge_kind
        for edge_kind in expected_state.continuity_graph_edge_kinds
        if edge_kind not in actual_graph_edge_kinds
    ]
    if missing_graph_edge_kinds:
        mismatches.append(
            {
                "path": "continuity_graph.edges.kind",
                "expected": list(expected_state.continuity_graph_edge_kinds),
                "actual": actual_graph_edge_kinds,
            }
        )

    actual_graph_superseded_backing_ids = sorted(
        {
            backing_record_by_node_id[node_id]
            for node_id in graph_superseded_ids
            if node_id in backing_record_by_node_id
        }
    )
    missing_graph_superseded_backing_ids = [
        backing_id
        for backing_id in expected_state.continuity_graph_superseded_backing_ids
        if backing_id not in actual_graph_superseded_backing_ids
    ]
    if missing_graph_superseded_backing_ids:
        mismatches.append(
            {
                "path": "continuity_graph.superseded_node_ids",
                "expected": list(expected_state.continuity_graph_superseded_backing_ids),
                "actual": actual_graph_superseded_backing_ids,
            }
        )

    actual_graph_stale_backing_ids = sorted(
        {
            backing_record_by_node_id[node_id]
            for node_id in graph_stale_ids
            if node_id in backing_record_by_node_id
        }
    )
    missing_graph_stale_backing_ids = [
        backing_id
        for backing_id in expected_state.continuity_graph_stale_backing_ids
        if backing_id not in actual_graph_stale_backing_ids
    ]
    if missing_graph_stale_backing_ids:
        mismatches.append(
            {
                "path": "continuity_graph.stale_node_ids",
                "expected": list(expected_state.continuity_graph_stale_backing_ids),
                "actual": actual_graph_stale_backing_ids,
            }
        )

    dossier_records = list(continuity_dossiers.get("dossiers", []))
    relationship_dossier = next(
        (record for record in dossier_records if record.get("kind") == "relationship"),
        None,
    )
    if expected_state.relationship_dossier_summary_contains:
        summary = ""
        if isinstance(relationship_dossier, dict):
            summary = str(relationship_dossier.get("summary", ""))
        if expected_state.relationship_dossier_summary_contains not in summary:
            mismatches.append(
                {
                    "path": "continuity_dossiers.relationship.summary",
                    "expected": expected_state.relationship_dossier_summary_contains,
                    "actual": summary,
                }
            )

    if expected_state.relationship_dossier_freshness:
        actual_freshness = (
            str(relationship_dossier.get("freshness", ""))
            if isinstance(relationship_dossier, dict)
            else ""
        )
        if actual_freshness != expected_state.relationship_dossier_freshness:
            mismatches.append(
                {
                    "path": "continuity_dossiers.relationship.freshness",
                    "expected": expected_state.relationship_dossier_freshness,
                    "actual": actual_freshness,
                }
            )

    if expected_state.relationship_dossier_contradiction:
        actual_contradiction = (
            str(relationship_dossier.get("contradiction", ""))
            if isinstance(relationship_dossier, dict)
            else ""
        )
        if actual_contradiction != expected_state.relationship_dossier_contradiction:
            mismatches.append(
                {
                    "path": "continuity_dossiers.relationship.contradiction",
                    "expected": expected_state.relationship_dossier_contradiction,
                    "actual": actual_contradiction,
                }
            )

    actual_project_keys = [
        record.get("project_key")
        for record in dossier_records
        if record.get("kind") == "project" and record.get("project_key")
    ]
    missing_project_keys = [
        project_key
        for project_key in expected_state.project_dossier_keys
        if project_key not in actual_project_keys
    ]
    if missing_project_keys:
        mismatches.append(
            {
                "path": "continuity_dossiers.project.project_key",
                "expected": list(expected_state.project_dossier_keys),
                "actual": actual_project_keys,
            }
        )

    actual_stale_dossier_ids = list(continuity_dossiers.get("stale_dossier_ids", []))
    missing_stale_dossier_ids = [
        dossier_id
        for dossier_id in expected_state.dossier_stale_ids
        if dossier_id not in actual_stale_dossier_ids
    ]
    if missing_stale_dossier_ids:
        mismatches.append(
            {
                "path": "continuity_dossiers.stale_dossier_ids",
                "expected": list(expected_state.dossier_stale_ids),
                "actual": actual_stale_dossier_ids,
            }
        )

    actual_needs_refresh_dossier_ids = list(
        continuity_dossiers.get("needs_refresh_dossier_ids", [])
    )
    missing_needs_refresh_dossier_ids = [
        dossier_id
        for dossier_id in expected_state.dossier_needs_refresh_ids
        if dossier_id not in actual_needs_refresh_dossier_ids
    ]
    if missing_needs_refresh_dossier_ids:
        mismatches.append(
            {
                "path": "continuity_dossiers.needs_refresh_dossier_ids",
                "expected": list(expected_state.dossier_needs_refresh_ids),
                "actual": actual_needs_refresh_dossier_ids,
            }
        )

    actual_uncertain_dossier_ids = list(continuity_dossiers.get("uncertain_dossier_ids", []))
    missing_uncertain_dossier_ids = [
        dossier_id
        for dossier_id in expected_state.dossier_uncertain_ids
        if dossier_id not in actual_uncertain_dossier_ids
    ]
    if missing_uncertain_dossier_ids:
        mismatches.append(
            {
                "path": "continuity_dossiers.uncertain_dossier_ids",
                "expected": list(expected_state.dossier_uncertain_ids),
                "actual": actual_uncertain_dossier_ids,
            }
        )

    actual_contradicted_dossier_ids = list(continuity_dossiers.get("contradicted_dossier_ids", []))
    missing_contradicted_dossier_ids = [
        dossier_id
        for dossier_id in expected_state.dossier_contradicted_ids
        if dossier_id not in actual_contradicted_dossier_ids
    ]
    if missing_contradicted_dossier_ids:
        mismatches.append(
            {
                "path": "continuity_dossiers.contradicted_dossier_ids",
                "expected": list(expected_state.dossier_contradicted_ids),
                "actual": actual_contradicted_dossier_ids,
            }
        )

    for task, expected_mode in (
        ("reply", expected_state.reply_packet_temporal_mode),
        ("planning", expected_state.planning_packet_temporal_mode),
    ):
        if expected_mode is None:
            continue
        actual_mode = str(context_packet_digest.get(task, {}).get("temporal_mode", ""))
        if actual_mode != expected_mode:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.temporal_mode",
                    "expected": expected_mode,
                    "actual": actual_mode,
                }
            )

    for task, expected_anchor_types in (
        ("reply", expected_state.reply_selected_anchor_types),
        ("planning", expected_state.planning_selected_anchor_types),
    ):
        actual_anchor_types = list(
            context_packet_digest.get(task, {}).get("selected_anchor_types", [])
        )
        missing_anchor_types = [
            item for item in expected_anchor_types if item not in actual_anchor_types
        ]
        if missing_anchor_types:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_anchor_types",
                    "expected": list(expected_anchor_types),
                    "actual": actual_anchor_types,
                }
            )

    for task, expected_temporal_kinds in (
        ("reply", expected_state.reply_selected_temporal_kinds),
        ("planning", expected_state.planning_selected_temporal_kinds),
    ):
        actual_temporal_kinds = list(
            context_packet_digest.get(task, {}).get("selected_temporal_kinds", [])
        )
        missing_temporal_kinds = [
            item for item in expected_temporal_kinds if item not in actual_temporal_kinds
        ]
        if missing_temporal_kinds:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_temporal_kinds",
                    "expected": list(expected_temporal_kinds),
                    "actual": actual_temporal_kinds,
                }
            )

    for task, expected_backing_ids in (
        ("reply", expected_state.reply_selected_backing_ids),
        ("planning", expected_state.planning_selected_backing_ids),
    ):
        actual_backing_ids = list(
            context_packet_digest.get(task, {}).get("selected_backing_ids", [])
        )
        missing_backing_ids = [
            item for item in expected_backing_ids if item not in actual_backing_ids
        ]
        if missing_backing_ids:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_backing_ids",
                    "expected": list(expected_backing_ids),
                    "actual": actual_backing_ids,
                }
            )

    for task, expected_drop_reason_codes in (
        ("reply", expected_state.reply_drop_reason_codes),
        ("planning", expected_state.planning_drop_reason_codes),
    ):
        actual_drop_reason_codes = list(
            dict(context_packet_digest.get(task, {}).get("drop_reason_counts", {}))
        )
        missing_drop_reason_codes = [
            item for item in expected_drop_reason_codes if item not in actual_drop_reason_codes
        ]
        if missing_drop_reason_codes:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.drop_reason_counts",
                    "expected": list(expected_drop_reason_codes),
                    "actual": actual_drop_reason_codes,
                }
            )

    for task, expected_mode in expected_state.packet_temporal_modes.items():
        actual_mode = str(context_packet_digest.get(task, {}).get("temporal_mode", ""))
        if actual_mode != expected_mode:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.temporal_mode",
                    "expected": expected_mode,
                    "actual": actual_mode,
                }
            )

    for task, expected_anchor_types in expected_state.packet_selected_anchor_types.items():
        actual_anchor_types = list(
            context_packet_digest.get(task, {}).get("selected_anchor_types", [])
        )
        missing_anchor_types = [
            item for item in expected_anchor_types if item not in actual_anchor_types
        ]
        if missing_anchor_types:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_anchor_types",
                    "expected": list(expected_anchor_types),
                    "actual": actual_anchor_types,
                }
            )

    for task, expected_item_types in expected_state.packet_selected_item_types.items():
        actual_item_types = list(
            dict(context_packet_digest.get(task, {}).get("selected_item_counts", {}))
        )
        missing_item_types = [item for item in expected_item_types if item not in actual_item_types]
        if missing_item_types:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_item_counts",
                    "expected": list(expected_item_types),
                    "actual": actual_item_types,
                }
            )

    for task, expected_temporal_kinds in expected_state.packet_selected_temporal_kinds.items():
        actual_temporal_kinds = list(
            context_packet_digest.get(task, {}).get("selected_temporal_kinds", [])
        )
        missing_temporal_kinds = [
            item for item in expected_temporal_kinds if item not in actual_temporal_kinds
        ]
        if missing_temporal_kinds:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_temporal_kinds",
                    "expected": list(expected_temporal_kinds),
                    "actual": actual_temporal_kinds,
                }
            )

    for task, expected_backing_ids in expected_state.packet_selected_backing_ids.items():
        actual_backing_ids = list(
            context_packet_digest.get(task, {}).get("selected_backing_ids", [])
        )
        missing_backing_ids = [
            item for item in expected_backing_ids if item not in actual_backing_ids
        ]
        if missing_backing_ids:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_backing_ids",
                    "expected": list(expected_backing_ids),
                    "actual": actual_backing_ids,
                }
            )

    for task, expected_provenance_ids in expected_state.packet_selected_provenance_ids.items():
        actual_provenance_ids = list(
            context_packet_digest.get(task, {}).get("selected_provenance_ids", [])
        )
        missing_provenance_ids = [
            item for item in expected_provenance_ids if item not in actual_provenance_ids
        ]
        if missing_provenance_ids:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.selected_provenance_ids",
                    "expected": list(expected_provenance_ids),
                    "actual": actual_provenance_ids,
                }
            )

    for task, expected_drop_reasons in expected_state.packet_drop_reason_codes.items():
        actual_drop_reason_codes = list(
            dict(context_packet_digest.get(task, {}).get("drop_reason_counts", {}))
        )
        missing_drop_reason_codes = [
            item for item in expected_drop_reasons if item not in actual_drop_reason_codes
        ]
        if missing_drop_reason_codes:
            mismatches.append(
                {
                    "path": f"context_packet_digest.{task}.drop_reason_counts",
                    "expected": list(expected_drop_reasons),
                    "actual": actual_drop_reason_codes,
                }
            )

    actual_current_commitment_ids = [
        item.get("commitment_id")
        for item in continuity_graph_digest.get("current_commitment_plan_links", [])
        if item.get("commitment_id")
    ]
    missing_current_commitment_ids = [
        item
        for item in expected_state.graph_digest_current_commitment_ids
        if item not in actual_current_commitment_ids
    ]
    if missing_current_commitment_ids:
        mismatches.append(
            {
                "path": "continuity_graph_digest.current_commitment_plan_links.commitment_id",
                "expected": list(expected_state.graph_digest_current_commitment_ids),
                "actual": actual_current_commitment_ids,
            }
        )

    actual_current_plan_proposal_ids = [
        item.get("plan_proposal_id")
        for item in continuity_graph_digest.get("current_commitment_plan_links", [])
        if item.get("plan_proposal_id")
    ]
    missing_current_plan_proposal_ids = [
        item
        for item in expected_state.graph_digest_current_plan_proposal_ids
        if item not in actual_current_plan_proposal_ids
    ]
    if missing_current_plan_proposal_ids:
        mismatches.append(
            {
                "path": "continuity_graph_digest.current_commitment_plan_links.plan_proposal_id",
                "expected": list(expected_state.graph_digest_current_plan_proposal_ids),
                "actual": actual_current_plan_proposal_ids,
            }
        )

    actual_governance_issue_kinds = list(
        dict(continuity_governance_report.get("open_issue_counts", {}))
    )
    missing_governance_issue_kinds = [
        item
        for item in expected_state.governance_open_issue_kinds
        if item not in actual_governance_issue_kinds
    ]
    if missing_governance_issue_kinds:
        mismatches.append(
            {
                "path": "continuity_governance_report.open_issue_counts",
                "expected": list(expected_state.governance_open_issue_kinds),
                "actual": actual_governance_issue_kinds,
            }
        )

    actual_governance_stale_graph_backing_ids = list(
        continuity_governance_report.get("stale_graph_backing_ids", [])
    )
    missing_governance_stale_graph_backing_ids = [
        item
        for item in expected_state.governance_stale_graph_backing_ids
        if item not in actual_governance_stale_graph_backing_ids
    ]
    if missing_governance_stale_graph_backing_ids:
        mismatches.append(
            {
                "path": "continuity_governance_report.stale_graph_backing_ids",
                "expected": list(expected_state.governance_stale_graph_backing_ids),
                "actual": actual_governance_stale_graph_backing_ids,
            }
        )

    actual_governance_superseded_graph_backing_ids = list(
        continuity_governance_report.get("superseded_graph_backing_ids", [])
    )
    missing_governance_superseded_graph_backing_ids = [
        item
        for item in expected_state.governance_superseded_graph_backing_ids
        if item not in actual_governance_superseded_graph_backing_ids
    ]
    if missing_governance_superseded_graph_backing_ids:
        mismatches.append(
            {
                "path": "continuity_governance_report.superseded_graph_backing_ids",
                "expected": list(expected_state.governance_superseded_graph_backing_ids),
                "actual": actual_governance_superseded_graph_backing_ids,
            }
        )

    for path, expected_ids, actual_ids in (
        (
            "procedural_skills.active_skill_ids",
            expected_state.procedural_active_skill_ids,
            list(procedural_skills.get("active_skill_ids", [])),
        ),
        (
            "procedural_skills.candidate_skill_ids",
            expected_state.procedural_candidate_skill_ids,
            list(procedural_skills.get("candidate_skill_ids", [])),
        ),
        (
            "procedural_skills.retired_skill_ids",
            expected_state.procedural_retired_skill_ids,
            list(procedural_skills.get("retired_skill_ids", [])),
        ),
        (
            "procedural_skills.superseded_skill_ids",
            expected_state.procedural_superseded_skill_ids,
            list(procedural_skills.get("superseded_skill_ids", [])),
        ),
    ):
        missing_ids = [item for item in expected_ids if item not in actual_ids]
        if missing_ids:
            mismatches.append(
                {
                    "path": path,
                    "expected": list(expected_ids),
                    "actual": actual_ids,
                }
            )

    actual_failure_signature_codes = _sorted_unique(
        [
            str(item.get("reason_code") or "")
            for item in procedural_skill_digest.get("top_failure_signatures", [])
            if str(item.get("reason_code", "")).strip()
        ]
    )
    missing_failure_signature_codes = [
        item
        for item in expected_state.procedural_failure_signature_codes
        if item not in actual_failure_signature_codes
    ]
    if missing_failure_signature_codes:
        mismatches.append(
            {
                "path": "procedural_skill_digest.top_failure_signatures.reason_code",
                "expected": list(expected_state.procedural_failure_signature_codes),
                "actual": actual_failure_signature_codes,
            }
        )

    actual_low_confidence_skill_ids = list(
        procedural_skill_governance_report.get("low_confidence_skill_ids", [])
    )
    missing_low_confidence_skill_ids = [
        item
        for item in expected_state.procedural_low_confidence_skill_ids
        if item not in actual_low_confidence_skill_ids
    ]
    if missing_low_confidence_skill_ids:
        mismatches.append(
            {
                "path": "procedural_skill_governance_report.low_confidence_skill_ids",
                "expected": list(expected_state.procedural_low_confidence_skill_ids),
                "actual": actual_low_confidence_skill_ids,
            }
        )

    actual_retirement_reason_codes = list(
        dict(procedural_skill_governance_report.get("retirement_reason_counts", {}))
    )
    missing_retirement_reason_codes = [
        item
        for item in expected_state.procedural_retirement_reason_codes
        if item not in actual_retirement_reason_codes
    ]
    if missing_retirement_reason_codes:
        mismatches.append(
            {
                "path": "procedural_skill_governance_report.retirement_reason_counts",
                "expected": list(expected_state.procedural_retirement_reason_codes),
                "actual": actual_retirement_reason_codes,
            }
        )

    actual_planning_procedural_origins = list(
        dict(planning_digest.get("procedural_origin_counts", {}))
    )
    missing_planning_procedural_origins = [
        item
        for item in expected_state.planning_procedural_origins
        if item not in actual_planning_procedural_origins
    ]
    if missing_planning_procedural_origins:
        mismatches.append(
            {
                "path": "planning_digest.procedural_origin_counts",
                "expected": list(expected_state.planning_procedural_origins),
                "actual": actual_planning_procedural_origins,
            }
        )

    actual_planning_selected_skill_ids = list(planning_digest.get("recent_selected_skill_ids", []))
    missing_planning_selected_skill_ids = [
        item
        for item in expected_state.planning_selected_skill_ids
        if item not in actual_planning_selected_skill_ids
    ]
    if missing_planning_selected_skill_ids:
        mismatches.append(
            {
                "path": "planning_digest.recent_selected_skill_ids",
                "expected": list(expected_state.planning_selected_skill_ids),
                "actual": actual_planning_selected_skill_ids,
            }
        )

    actual_planning_skill_rejection_reason_codes = list(
        dict(planning_digest.get("skill_rejection_reason_counts", {}))
    )
    missing_planning_skill_rejection_reason_codes = [
        item
        for item in expected_state.planning_skill_rejection_reason_codes
        if item not in actual_planning_skill_rejection_reason_codes
    ]
    if missing_planning_skill_rejection_reason_codes:
        mismatches.append(
            {
                "path": "planning_digest.skill_rejection_reason_counts",
                "expected": list(expected_state.planning_skill_rejection_reason_codes),
                "actual": actual_planning_skill_rejection_reason_codes,
            }
        )

    actual_planning_delta_operation_counts = list(
        dict(planning_digest.get("delta_operation_counts", {}))
    )
    missing_planning_delta_operation_counts = [
        item
        for item in expected_state.planning_delta_operation_counts
        if item not in actual_planning_delta_operation_counts
    ]
    if missing_planning_delta_operation_counts:
        mismatches.append(
            {
                "path": "planning_digest.delta_operation_counts",
                "expected": list(expected_state.planning_delta_operation_counts),
                "actual": actual_planning_delta_operation_counts,
            }
        )

    actual_high_risk_failure_signature_codes = _sorted_unique(
        [
            str(item.get("reason_code") or "")
            for item in procedural_skill_governance_report.get("high_risk_failure_signatures", [])
            if str(item.get("reason_code", "")).strip()
        ]
    )
    missing_high_risk_failure_signature_codes = [
        item
        for item in expected_state.procedural_high_risk_failure_signature_codes
        if item not in actual_high_risk_failure_signature_codes
    ]
    if missing_high_risk_failure_signature_codes:
        mismatches.append(
            {
                "path": "procedural_skill_governance_report.high_risk_failure_signatures.reason_code",
                "expected": list(expected_state.procedural_high_risk_failure_signature_codes),
                "actual": actual_high_risk_failure_signature_codes,
            }
        )

    actual_follow_up_trace_ids = list(
        procedural_skill_governance_report.get("follow_up_trace_ids", [])
    )
    missing_follow_up_trace_ids = [
        item
        for item in expected_state.procedural_follow_up_trace_ids
        if item not in actual_follow_up_trace_ids
    ]
    if missing_follow_up_trace_ids:
        mismatches.append(
            {
                "path": "procedural_skill_governance_report.follow_up_trace_ids",
                "expected": list(expected_state.procedural_follow_up_trace_ids),
                "actual": actual_follow_up_trace_ids,
            }
        )

    actual_negative_transfer_reason_codes = list(
        dict(procedural_skill_governance_report.get("negative_transfer_reason_counts", {}))
    )
    missing_negative_transfer_reason_codes = [
        item
        for item in expected_state.procedural_negative_transfer_reason_codes
        if item not in actual_negative_transfer_reason_codes
    ]
    if missing_negative_transfer_reason_codes:
        mismatches.append(
            {
                "path": "procedural_skill_governance_report.negative_transfer_reason_counts",
                "expected": list(expected_state.procedural_negative_transfer_reason_codes),
                "actual": actual_negative_transfer_reason_codes,
            }
        )

    for path, expected_ids, actual_ids in (
        (
            "private_working_memory.active_record_ids",
            expected_state.private_working_memory_active_record_ids,
            list(private_working_memory.get("active_record_ids", [])),
        ),
        (
            "private_working_memory.stale_record_ids",
            expected_state.private_working_memory_stale_record_ids,
            list(private_working_memory.get("stale_record_ids", [])),
        ),
        (
            "private_working_memory.resolved_record_ids",
            expected_state.private_working_memory_resolved_record_ids,
            list(private_working_memory.get("resolved_record_ids", [])),
        ),
        (
            "private_working_memory_digest.unresolved_record_ids",
            expected_state.private_working_memory_unresolved_record_ids,
            list(private_working_memory_digest.get("unresolved_record_ids", [])),
        ),
    ):
        missing_ids = [item for item in expected_ids if item not in actual_ids]
        if missing_ids:
            mismatches.append(
                {
                    "path": path,
                    "expected": list(expected_ids),
                    "actual": actual_ids,
                }
            )

    for path, expected_counts, actual_counts in (
        (
            "private_working_memory.buffer_counts",
            expected_state.private_working_memory_buffer_counts,
            dict(private_working_memory.get("buffer_counts", {})),
        ),
        (
            "private_working_memory.state_counts",
            expected_state.private_working_memory_state_counts,
            dict(private_working_memory.get("state_counts", {})),
        ),
    ):
        missing_counts = {
            key: value for key, value in expected_counts.items() if actual_counts.get(key) != value
        }
        if missing_counts:
            mismatches.append(
                {
                    "path": path,
                    "expected": dict(expected_counts),
                    "actual": actual_counts,
                }
            )

    for path, expected_ids, actual_ids in (
        (
            "scene_world_state.active_entity_ids",
            expected_state.scene_world_state_active_entity_ids,
            list(scene_world_state.get("active_entity_ids", [])),
        ),
        (
            "scene_world_state.stale_entity_ids",
            expected_state.scene_world_state_stale_entity_ids,
            list(scene_world_state.get("stale_entity_ids", [])),
        ),
        (
            "scene_world_state.contradicted_entity_ids",
            expected_state.scene_world_state_contradicted_entity_ids,
            list(scene_world_state.get("contradicted_entity_ids", [])),
        ),
        (
            "scene_world_state.expired_entity_ids",
            expected_state.scene_world_state_expired_entity_ids,
            list(scene_world_state.get("expired_entity_ids", [])),
        ),
        (
            "scene_world_state.active_affordance_ids",
            expected_state.scene_world_state_active_affordance_ids,
            list(scene_world_state.get("active_affordance_ids", [])),
        ),
        (
            "scene_world_state_digest.uncertain_affordance_ids",
            expected_state.scene_world_state_uncertain_affordance_ids,
            list(scene_world_state_digest.get("uncertain_affordance_ids", [])),
        ),
    ):
        missing_ids = [item for item in expected_ids if item not in actual_ids]
        if missing_ids:
            mismatches.append(
                {
                    "path": path,
                    "expected": list(expected_ids),
                    "actual": actual_ids,
                }
            )

    if (
        expected_state.scene_world_state_degraded_mode is not None
        and scene_world_state.get("degraded_mode") != expected_state.scene_world_state_degraded_mode
    ):
        mismatches.append(
            {
                "path": "scene_world_state.degraded_mode",
                "expected": expected_state.scene_world_state_degraded_mode,
                "actual": scene_world_state.get("degraded_mode"),
            }
        )

    actual_scene_degraded_reason_codes = list(scene_world_state.get("degraded_reason_codes", []))
    missing_scene_degraded_reason_codes = [
        item
        for item in expected_state.scene_world_state_degraded_reason_codes
        if item not in actual_scene_degraded_reason_codes
    ]
    if missing_scene_degraded_reason_codes:
        mismatches.append(
            {
                "path": "scene_world_state.degraded_reason_codes",
                "expected": list(expected_state.scene_world_state_degraded_reason_codes),
                "actual": actual_scene_degraded_reason_codes,
            }
        )

    actual_scene_contradiction_codes = list(dict(scene_world_state.get("contradiction_counts", {})))
    missing_scene_contradiction_codes = [
        item
        for item in expected_state.scene_world_state_contradiction_codes
        if item not in actual_scene_contradiction_codes
    ]
    if missing_scene_contradiction_codes:
        mismatches.append(
            {
                "path": "scene_world_state.contradiction_counts",
                "expected": list(expected_state.scene_world_state_contradiction_codes),
                "actual": actual_scene_contradiction_codes,
            }
        )
    for path, expected_ids, actual_ids in (
        (
            "active_situation_model.active_record_ids",
            expected_state.active_situation_active_record_ids,
            list(active_situation_model.get("active_record_ids", [])),
        ),
        (
            "active_situation_model.stale_record_ids",
            expected_state.active_situation_stale_record_ids,
            list(active_situation_model.get("stale_record_ids", [])),
        ),
        (
            "active_situation_model.unresolved_record_ids",
            expected_state.active_situation_unresolved_record_ids,
            list(active_situation_model.get("unresolved_record_ids", [])),
        ),
        (
            "active_situation_model.linked_commitment_ids",
            expected_state.active_situation_linked_commitment_ids,
            list(active_situation_model_digest.get("linked_commitment_ids", [])),
        ),
        (
            "active_situation_model.linked_plan_proposal_ids",
            expected_state.active_situation_linked_plan_proposal_ids,
            list(active_situation_model_digest.get("linked_plan_proposal_ids", [])),
        ),
        (
            "active_situation_model.linked_skill_ids",
            expected_state.active_situation_linked_skill_ids,
            list(active_situation_model_digest.get("linked_skill_ids", [])),
        ),
    ):
        missing_ids = [item for item in expected_ids if item not in actual_ids]
        if missing_ids:
            mismatches.append(
                {
                    "path": path,
                    "expected": list(expected_ids),
                    "actual": actual_ids,
                }
            )

    for path, expected_counts, actual_counts in (
        (
            "active_situation_model.kind_counts",
            expected_state.active_situation_kind_counts,
            dict(active_situation_model.get("kind_counts", {})),
        ),
        (
            "active_situation_model.state_counts",
            expected_state.active_situation_state_counts,
            dict(active_situation_model.get("state_counts", {})),
        ),
    ):
        missing_counts = {
            key: value for key, value in expected_counts.items() if actual_counts.get(key) != value
        }
        if missing_counts:
            mismatches.append(
                {
                    "path": path,
                    "expected": dict(expected_counts),
                    "actual": actual_counts,
                }
            )

    actual_uncertainty_codes = list(dict(active_situation_model.get("uncertainty_code_counts", {})))
    missing_uncertainty_codes = [
        item
        for item in expected_state.active_situation_uncertainty_codes
        if item not in actual_uncertainty_codes
    ]
    if missing_uncertainty_codes:
        mismatches.append(
            {
                "path": "active_situation_model.uncertainty_code_counts",
                "expected": list(expected_state.active_situation_uncertainty_codes),
                "actual": actual_uncertainty_codes,
            }
        )
    return mismatches


def _claim_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _derive_visual_health_findings(snapshot: BrainContextSurfaceSnapshot) -> list[dict[str, Any]]:
    """Derive current visual-health findings from the latest runtime surfaces."""
    findings: list[dict[str, Any]] = []
    scene = snapshot.scene
    body = snapshot.body

    if scene.camera_track_state in {"stalled", "recovering"}:
        findings.append(
            {
                "code": "camera_track_stalled",
                "severity": "warning",
                "summary": "The browser camera track is stalled or recovering.",
                "details": {"camera_track_state": scene.camera_track_state},
            }
        )
    if (scene.frame_age_ms is not None and scene.frame_age_ms > 5000) or (
        scene.sensor_health_reason == "camera_frame_stale"
    ):
        findings.append(
            {
                "code": "camera_frame_stale",
                "severity": "warning",
                "summary": "The latest cached browser camera frame is stale.",
                "details": {"frame_age_ms": scene.frame_age_ms},
            }
        )
    if body.vision_unavailable or scene.sensor_health_reason == "presence_detector_unavailable":
        findings.append(
            {
                "code": "presence_detector_unavailable",
                "severity": "critical",
                "summary": "Deterministic browser presence detection is unavailable.",
                "details": {"detection_backend": scene.detection_backend or body.detection_backend},
            }
        )
    elif scene.detection_confidence is not None and scene.detection_confidence < 0.65:
        findings.append(
            {
                "code": "presence_detector_low_confidence",
                "severity": "warning",
                "summary": "The deterministic presence detector is running with low confidence.",
                "details": {"detection_confidence": scene.detection_confidence},
            }
        )
    if scene.enrichment_available is False:
        findings.append(
            {
                "code": "vision_enrichment_unavailable",
                "severity": "warning",
                "summary": "Optional VLM enrichment is unavailable.",
                "details": {},
            }
        )
    if scene.sensor_health_reason == "vision_enrichment_parse_error":
        findings.append(
            {
                "code": "vision_enrichment_parse_error",
                "severity": "warning",
                "summary": "The latest VLM enrichment response could not be parsed.",
                "details": {},
            }
        )
    return findings


def _expected_state_as_dict(expected_state: BrainContinuityExpectedState) -> dict[str, Any]:
    return {
        "current_claims": list(expected_state.current_claims),
        "historical_claims": list(expected_state.historical_claims),
        "core_blocks_present": list(expected_state.core_blocks_present),
        "commitment_titles_by_status": {
            key: list(value) for key, value in expected_state.commitment_titles_by_status.items()
        },
        "agenda_contains": {
            key: list(value) for key, value in expected_state.agenda_contains.items()
        },
        "autobiography_entry_kinds": list(expected_state.autobiography_entry_kinds),
        "health_finding_codes": list(expected_state.health_finding_codes),
        "health_score_min": expected_state.health_score_min,
        "health_score_max": expected_state.health_score_max,
        "relationship_arc_contains": expected_state.relationship_arc_contains,
        "autonomy_current_candidate_ids": list(expected_state.autonomy_current_candidate_ids),
        "autonomy_current_candidate_summaries": list(
            expected_state.autonomy_current_candidate_summaries
        ),
        "autonomy_recent_decision_kinds": list(expected_state.autonomy_recent_decision_kinds),
        "autonomy_reason_codes": list(expected_state.autonomy_reason_codes),
        "autonomy_reevaluation_conditions": list(expected_state.autonomy_reevaluation_conditions),
        "reevaluation_current_hold_ids": list(expected_state.reevaluation_current_hold_ids),
        "reevaluation_trigger_kinds": list(expected_state.reevaluation_trigger_kinds),
        "reevaluation_transition_candidate_ids": list(
            expected_state.reevaluation_transition_candidate_ids
        ),
        "reevaluation_transition_outcome_kinds": list(
            expected_state.reevaluation_transition_outcome_kinds
        ),
        "wake_current_waiting_commitment_ids": list(
            expected_state.wake_current_waiting_commitment_ids
        ),
        "wake_recent_wake_kinds": list(expected_state.wake_recent_wake_kinds),
        "wake_recent_route_kinds": list(expected_state.wake_recent_route_kinds),
        "wake_reason_codes": list(expected_state.wake_reason_codes),
        "planning_current_pending_proposal_ids": list(
            expected_state.planning_current_pending_proposal_ids
        ),
        "planning_current_pending_goal_ids": list(expected_state.planning_current_pending_goal_ids),
        "planning_recent_review_policies": list(expected_state.planning_recent_review_policies),
        "planning_recent_outcome_kinds": list(expected_state.planning_recent_outcome_kinds),
        "planning_reason_codes": list(expected_state.planning_reason_codes),
        "planning_recent_revision_goal_ids": list(expected_state.planning_recent_revision_goal_ids),
        "continuity_graph_current_backing_ids": list(
            expected_state.continuity_graph_current_backing_ids
        ),
        "continuity_graph_historical_backing_ids": list(
            expected_state.continuity_graph_historical_backing_ids
        ),
        "continuity_graph_edge_kinds": list(expected_state.continuity_graph_edge_kinds),
        "continuity_graph_superseded_backing_ids": list(
            expected_state.continuity_graph_superseded_backing_ids
        ),
        "continuity_graph_stale_backing_ids": list(
            expected_state.continuity_graph_stale_backing_ids
        ),
        "relationship_dossier_summary_contains": expected_state.relationship_dossier_summary_contains,
        "relationship_dossier_freshness": expected_state.relationship_dossier_freshness,
        "relationship_dossier_contradiction": expected_state.relationship_dossier_contradiction,
        "project_dossier_keys": list(expected_state.project_dossier_keys),
        "dossier_stale_ids": list(expected_state.dossier_stale_ids),
        "dossier_needs_refresh_ids": list(expected_state.dossier_needs_refresh_ids),
        "dossier_uncertain_ids": list(expected_state.dossier_uncertain_ids),
        "dossier_contradicted_ids": list(expected_state.dossier_contradicted_ids),
        "reply_packet_temporal_mode": expected_state.reply_packet_temporal_mode,
        "planning_packet_temporal_mode": expected_state.planning_packet_temporal_mode,
        "reply_selected_anchor_types": list(expected_state.reply_selected_anchor_types),
        "planning_selected_anchor_types": list(expected_state.planning_selected_anchor_types),
        "reply_selected_temporal_kinds": list(expected_state.reply_selected_temporal_kinds),
        "planning_selected_temporal_kinds": list(expected_state.planning_selected_temporal_kinds),
        "reply_selected_backing_ids": list(expected_state.reply_selected_backing_ids),
        "planning_selected_backing_ids": list(expected_state.planning_selected_backing_ids),
        "reply_drop_reason_codes": list(expected_state.reply_drop_reason_codes),
        "planning_drop_reason_codes": list(expected_state.planning_drop_reason_codes),
        "packet_temporal_modes": dict(expected_state.packet_temporal_modes),
        "packet_selected_anchor_types": {
            key: list(value) for key, value in expected_state.packet_selected_anchor_types.items()
        },
        "packet_selected_item_types": {
            key: list(value) for key, value in expected_state.packet_selected_item_types.items()
        },
        "packet_selected_temporal_kinds": {
            key: list(value) for key, value in expected_state.packet_selected_temporal_kinds.items()
        },
        "packet_selected_backing_ids": {
            key: list(value) for key, value in expected_state.packet_selected_backing_ids.items()
        },
        "packet_selected_provenance_ids": {
            key: list(value) for key, value in expected_state.packet_selected_provenance_ids.items()
        },
        "packet_drop_reason_codes": {
            key: list(value) for key, value in expected_state.packet_drop_reason_codes.items()
        },
        "graph_digest_current_commitment_ids": list(
            expected_state.graph_digest_current_commitment_ids
        ),
        "graph_digest_current_plan_proposal_ids": list(
            expected_state.graph_digest_current_plan_proposal_ids
        ),
        "governance_open_issue_kinds": list(expected_state.governance_open_issue_kinds),
        "governance_stale_graph_backing_ids": list(
            expected_state.governance_stale_graph_backing_ids
        ),
        "governance_superseded_graph_backing_ids": list(
            expected_state.governance_superseded_graph_backing_ids
        ),
        "procedural_active_skill_ids": list(expected_state.procedural_active_skill_ids),
        "procedural_candidate_skill_ids": list(expected_state.procedural_candidate_skill_ids),
        "procedural_retired_skill_ids": list(expected_state.procedural_retired_skill_ids),
        "procedural_superseded_skill_ids": list(expected_state.procedural_superseded_skill_ids),
        "procedural_failure_signature_codes": list(
            expected_state.procedural_failure_signature_codes
        ),
        "procedural_low_confidence_skill_ids": list(
            expected_state.procedural_low_confidence_skill_ids
        ),
        "procedural_retirement_reason_codes": list(
            expected_state.procedural_retirement_reason_codes
        ),
        "planning_procedural_origins": list(expected_state.planning_procedural_origins),
        "planning_selected_skill_ids": list(expected_state.planning_selected_skill_ids),
        "planning_skill_rejection_reason_codes": list(
            expected_state.planning_skill_rejection_reason_codes
        ),
        "planning_delta_operation_counts": list(expected_state.planning_delta_operation_counts),
        "procedural_high_risk_failure_signature_codes": list(
            expected_state.procedural_high_risk_failure_signature_codes
        ),
        "procedural_follow_up_trace_ids": list(expected_state.procedural_follow_up_trace_ids),
        "procedural_negative_transfer_reason_codes": list(
            expected_state.procedural_negative_transfer_reason_codes
        ),
    }


__all__ = [
    "BrainContinuityEvalCase",
    "BrainContinuityEvalHarness",
    "BrainContinuityEvalResult",
    "BrainContinuityExpectedState",
    "BrainConversationStep",
    "build_continuity_state",
    "build_continuity_state_from_surface",
    "compare_continuity_state",
]

"""Active context compiler for task-aware Blink packets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError

from blink.brain.context.budgets import BrainContextBudgetProfile, approximate_token_count
from blink.brain.context.policy import (
    BrainContextEdgeProfile,
    BrainContextModePolicy,
    BrainContextPredictionPolicy,
    BrainContextSceneEpisodePolicy,
    BrainContextTask,
    get_brain_context_mode_policy,
)
from blink.brain.context.selectors import (
    BrainContextSelectionDecision,
    BrainContextSelectionTrace,
    BrainContextSelector,
    BrainSelectedContext,
    BrainSelectedSection,
)
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, BrainContextSurfaceSnapshot
from blink.brain.identity import BRAIN_CONTEXT_HEADER
from blink.brain.knowledge import (
    BrainKnowledgeRoutingDecision,
    KnowledgeReserveRegistry,
    KnowledgeSelectionRequest,
    build_default_teaching_canon,
    explicit_knowledge_routing_decision,
    knowledge_routing_decision_from_selection,
    select_teaching_knowledge,
    unavailable_knowledge_routing_decision,
)
from blink.brain.memory_layers.retrieval import BrainMemorySearchResult
from blink.brain.memory_v2 import (
    BrainAutobiographyEntryKind,
    BrainContinuityDossierAvailability,
    BrainContinuityDossierContradiction,
    BrainContinuityDossierFreshness,
    BrainContinuityGraphEdgeKind,
    BrainContinuityGraphNodeKind,
    BrainCoreMemoryBlockKind,
    BrainMemoryUseTrace,
    BrainMemoryUseTraceRef,
    build_memory_use_trace,
    display_kind_for_claim_predicate,
    render_safe_memory_provenance_label,
)
from blink.brain.persona import (
    BrainBehaviorControlProfile,
    BrainExpressionFrame,
    BrainPersonaModality,
    BrainPersonaTaskMode,
    RelationshipStyleStateSpec,
    TeachingProfileStateSpec,
    compile_expression_frame,
    compile_persona_frame,
    load_behavior_control_profile,
    render_persona_expression_summary,
)
from blink.brain.procedural_planning import match_planning_skills, planning_completed_prefix
from blink.brain.projections import (
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldRecordState,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

_ACTIVE_SECTION_TITLES = {
    "active_state": "Active State",
    "active_continuity": "Active Continuity",
    "unresolved_state": "Unresolved State",
    "recent_changes": "Recent Changes",
    "planning_anchors": "Planning Anchors",
    "relevant_continuity": "Relevant Continuity",
}
_REPLY_ALLOWED_EDGE_KINDS = {
    BrainContinuityGraphEdgeKind.CLAIM_SUBJECT.value,
    BrainContinuityGraphEdgeKind.CLAIM_OBJECT.value,
    BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value,
    BrainContinuityGraphEdgeKind.SUPPORTED_BY_EPISODE.value,
    BrainContinuityGraphEdgeKind.SUPERSEDES.value,
    BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_CLAIM.value,
    BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_EVENT.value,
}
_PLANNING_ALLOWED_EDGE_KINDS = _REPLY_ALLOWED_EDGE_KINDS | {
    BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value,
    BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_SUPERSEDES.value,
    BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value,
}
_CURRENT_FIRST_CUES = (
    "before",
    "earlier",
    "previously",
    "used to",
    "history",
    "changed",
    "之前",
    "以前",
    "曾经",
    "原来",
    "历史",
    "变化",
)
_STATE_PRIORITY = {
    BrainActiveSituationRecordState.ACTIVE.value: 9.0,
    BrainActiveSituationRecordState.UNRESOLVED.value: 8.0,
    BrainActiveSituationRecordState.STALE.value: 6.0,
    BrainSceneWorldRecordState.ACTIVE.value: 8.0,
    BrainSceneWorldRecordState.STALE.value: 6.5,
    BrainSceneWorldRecordState.CONTRADICTED.value: 8.0,
    BrainSceneWorldRecordState.EXPIRED.value: 4.0,
    BrainPrivateWorkingMemoryRecordState.ACTIVE.value: 8.0,
    BrainPrivateWorkingMemoryRecordState.STALE.value: 6.0,
    BrainSceneWorldAffordanceAvailability.AVAILABLE.value: 7.5,
    BrainSceneWorldAffordanceAvailability.BLOCKED.value: 7.0,
    BrainSceneWorldAffordanceAvailability.UNCERTAIN.value: 7.5,
    BrainSceneWorldAffordanceAvailability.STALE.value: 6.0,
}
_PERSONA_DEFAULT_BLOCKS = ("persona", "voice", "relationship_style", "teaching_style")


class BrainContextTemporalMode(str, Enum):
    """Temporal stance for active context compilation."""

    CURRENT_FIRST = "current_first"
    HISTORICAL_FOCUS = "historical_focus"


@dataclass(frozen=True)
class BrainActiveContextAnchorCandidate:
    """One anchor candidate considered by the active compiler."""

    anchor_id: str
    anchor_type: str
    label: str
    score: float
    seed_node_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the anchor candidate."""
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "label": self.label,
            "score": self.score,
            "seed_node_ids": list(self.seed_node_ids),
            "provenance": dict(self.provenance),
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BrainActiveContextExpansionRecord:
    """One inspected graph expansion edge."""

    from_node_id: str
    to_node_id: str
    edge_id: str
    edge_kind: str
    hop_distance: int
    accepted: bool
    reason: str
    reverse: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize the expansion record."""
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "edge_id": self.edge_id,
            "edge_kind": self.edge_kind,
            "hop_distance": self.hop_distance,
            "accepted": self.accepted,
            "reason": self.reason,
            "reverse": self.reverse,
        }


@dataclass(frozen=True)
class BrainActiveContextPacketItemRecord:
    """One selected or dropped dynamic packet item."""

    item_id: str
    item_type: str
    section_key: str
    title: str
    content: str
    estimated_tokens: int
    temporal_kind: str
    availability_state: str = "available"
    governance_reason_codes: tuple[str, ...] = ()
    decision_reason_codes: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    hop_distance: int | None = None
    selected: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the packet item record."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "section_key": self.section_key,
            "title": self.title,
            "content": self.content,
            "estimated_tokens": self.estimated_tokens,
            "temporal_kind": self.temporal_kind,
            "availability_state": self.availability_state,
            "governance_reason_codes": list(self.governance_reason_codes),
            "decision_reason_codes": list(self.decision_reason_codes),
            "provenance": dict(self.provenance),
            "hop_distance": self.hop_distance,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BrainActiveContextSectionDecision:
    """Per-section budget and selection outcome for one dynamic packet section."""

    section_key: str
    title: str
    selected: bool
    estimated_tokens: int
    selected_item_count: int
    dropped_item_count: int
    reason: str
    decision_reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the section decision."""
        return {
            "section_key": self.section_key,
            "title": self.title,
            "selected": self.selected,
            "estimated_tokens": self.estimated_tokens,
            "selected_item_count": self.selected_item_count,
            "dropped_item_count": self.dropped_item_count,
            "reason": self.reason,
            "decision_reason_codes": list(self.decision_reason_codes),
        }


@dataclass(frozen=True)
class BrainActiveContextTrace:
    """Inspectable trace for graph-aware active context compilation."""

    task: str
    query_text: str
    temporal_mode: str
    mode_policy: BrainContextModePolicy
    static_token_usage: int
    dynamic_token_budget: int
    anchor_candidates: tuple[BrainActiveContextAnchorCandidate, ...]
    selected_anchors: tuple[BrainActiveContextAnchorCandidate, ...]
    expansions: tuple[BrainActiveContextExpansionRecord, ...]
    section_decisions: tuple[BrainActiveContextSectionDecision, ...]
    selected_items: tuple[BrainActiveContextPacketItemRecord, ...]
    dropped_items: tuple[BrainActiveContextPacketItemRecord, ...]
    final_selected_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize the active context trace."""
        return {
            "task": self.task,
            "query_text": self.query_text,
            "temporal_mode": self.temporal_mode,
            "mode_policy": self.mode_policy.as_dict(),
            "static_token_usage": self.static_token_usage,
            "dynamic_token_budget": self.dynamic_token_budget,
            "anchor_candidates": [item.as_dict() for item in self.anchor_candidates],
            "selected_anchors": [item.as_dict() for item in self.selected_anchors],
            "expansions": [item.as_dict() for item in self.expansions],
            "section_decisions": [item.as_dict() for item in self.section_decisions],
            "selected_items": [item.as_dict() for item in self.selected_items],
            "dropped_items": [item.as_dict() for item in self.dropped_items],
            "final_selected_tokens": self.final_selected_tokens,
        }


@dataclass(frozen=True)
class BrainActiveContextPacket:
    """Dynamic sections and trace produced by the active compiler."""

    sections: tuple[BrainSelectedSection, ...]
    trace: BrainActiveContextTrace

    @property
    def estimated_tokens(self) -> int:
        """Return the selected dynamic-token count."""
        return sum(section.estimated_tokens for section in self.sections)


@dataclass(frozen=True)
class BrainCompiledContextPacket:
    """Structured output from context compilation."""

    task: BrainContextTask
    prompt: str
    selected_context: BrainSelectedContext
    packet_trace: BrainActiveContextTrace | None = None
    memory_use_trace: BrainMemoryUseTrace | None = None
    teaching_knowledge_decision: BrainKnowledgeRoutingDecision | None = None


@dataclass(frozen=True)
class _GraphIndex:
    node_by_id: dict[str, Any]
    edge_by_id: dict[str, Any]
    node_id_by_backing: dict[tuple[str, str], str]
    neighbors_by_node_id: dict[str, tuple[tuple[Any, str, bool], ...]]
    current_node_ids: set[str]
    historical_node_ids: set[str]
    stale_node_ids: set[str]
    superseded_node_ids: set[str]


@dataclass(frozen=True)
class _PacketItemCandidate:
    item_id: str
    item_type: str
    section_key: str
    title: str
    content: str
    temporal_kind: str
    provenance: dict[str, Any]
    score: float
    availability_state: str = "available"
    governance_reason_codes: tuple[str, ...] = ()
    decision_reason_codes: tuple[str, ...] = ()
    hop_distance: int | None = None
    is_dossier: bool = False
    is_graph_node: bool = False
    forced_reason: str | None = None

    @property
    def estimated_tokens(self) -> int:
        return approximate_token_count(self.content)


class BrainActiveContextCompiler:
    """Anchor-first, hop-bounded active context compiler."""

    def compile(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        latest_user_text: str,
        task: BrainContextTask,
        dynamic_token_budget: int,
        static_token_usage: int,
        language: Language,
    ) -> BrainActiveContextPacket:
        """Compile bounded dynamic packet sections for one task."""
        policy = get_brain_context_mode_policy(task)
        config = dict(policy.anchor_caps)
        temporal_mode = _infer_temporal_mode(latest_user_text)
        graph = snapshot.continuity_graph
        if dynamic_token_budget <= 0:
            trace = BrainActiveContextTrace(
                task=task.value,
                query_text=latest_user_text,
                temporal_mode=temporal_mode.value,
                mode_policy=policy,
                static_token_usage=static_token_usage,
                dynamic_token_budget=max(dynamic_token_budget, 0),
                anchor_candidates=(),
                selected_anchors=(),
                expansions=(),
                section_decisions=(),
                selected_items=(),
                dropped_items=(),
                final_selected_tokens=0,
            )
            return BrainActiveContextPacket(sections=(), trace=trace)

        graph_index = _build_graph_index(graph) if graph is not None else _empty_graph_index()
        anchor_candidates = self._build_anchor_candidates(
            snapshot=snapshot,
            graph_index=graph_index,
            latest_user_text=latest_user_text,
            task=task,
            temporal_mode=temporal_mode,
        )
        selected_anchors, dropped_anchor_candidates = _select_anchor_candidates(
            anchor_candidates,
            max_anchors=config["max_anchors"],
        )
        if graph is not None:
            visited_nodes, expansions = _expand_selected_anchors(
                selected_anchors=selected_anchors,
                graph_index=graph_index,
                allowed_edge_kinds=_allowed_edge_kinds(task),
                max_hops=config["max_hops"],
                max_graph_nodes=config["max_graph_nodes"],
            )
        else:
            visited_nodes, expansions = {}, ()
        item_candidates = self._build_packet_item_candidates(
            snapshot=snapshot,
            graph_index=graph_index,
            selected_anchors=selected_anchors,
            visited_nodes=visited_nodes,
            task=task,
            temporal_mode=temporal_mode,
            language=language,
            latest_user_text=latest_user_text,
        )
        selected_items, dropped_items = _select_packet_items(
            item_candidates=item_candidates,
            dynamic_token_budget=dynamic_token_budget,
            max_dossiers=config["max_dossiers"],
            max_current_items=config["max_current_items"],
            max_history_items=config["max_history_items"],
            task=task,
        )
        selected_sections = _render_dynamic_sections(
            task=task,
            items=selected_items,
        )
        section_decisions = _build_active_section_decisions(
            task=task,
            selected_sections=selected_sections,
            selected_items=selected_items,
            dropped_items=dropped_items,
        )
        trace = BrainActiveContextTrace(
            task=task.value,
            query_text=latest_user_text,
            temporal_mode=temporal_mode.value,
            mode_policy=policy,
            static_token_usage=static_token_usage,
            dynamic_token_budget=dynamic_token_budget,
            anchor_candidates=tuple(list(selected_anchors) + list(dropped_anchor_candidates)),
            selected_anchors=selected_anchors,
            expansions=expansions,
            section_decisions=section_decisions,
            selected_items=selected_items,
            dropped_items=dropped_items,
            final_selected_tokens=sum(section.estimated_tokens for section in selected_sections),
        )
        return BrainActiveContextPacket(
            sections=selected_sections,
            trace=trace,
        )

    def _build_anchor_candidates(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        graph_index: _GraphIndex,
        latest_user_text: str,
        task: BrainContextTask,
        temporal_mode: BrainContextTemporalMode,
    ) -> list[BrainActiveContextAnchorCandidate]:
        query_text = (latest_user_text or "").strip()
        tokens = _query_tokens(query_text)
        candidates: dict[str, BrainActiveContextAnchorCandidate] = {}

        def add(candidate: BrainActiveContextAnchorCandidate):
            existing = candidates.get(candidate.anchor_id)
            if existing is None or candidate.score > existing.score:
                candidates[candidate.anchor_id] = candidate

        memory_hits = (
            snapshot.historical_recent_memory
            if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS
            else snapshot.recent_memory
        )
        historical_hits = snapshot.historical_recent_memory
        for hit in memory_hits[:8]:
            candidate = _anchor_from_memory_result(
                hit,
                graph_index=graph_index,
                score_bonus=5.0 if task == BrainContextTask.REPLY else 2.5,
            )
            if candidate is not None:
                add(candidate)
        if temporal_mode == BrainContextTemporalMode.CURRENT_FIRST:
            for hit in historical_hits[:4]:
                candidate = _anchor_from_memory_result(
                    hit,
                    graph_index=graph_index,
                    score_bonus=1.5 if task == BrainContextTask.REPLY else 1.0,
                )
                if candidate is not None:
                    add(candidate)

        if _task_uses_planning_anchors(task):
            planning_seed_node_ids: set[str] = set()
            for record in (
                list(snapshot.commitment_projection.active_commitments)
                + list(snapshot.commitment_projection.blocked_commitments)
                + list(snapshot.commitment_projection.deferred_commitments)
            ):
                node_id = graph_index.node_id_by_backing.get(
                    (BrainContinuityGraphNodeKind.COMMITMENT.value, record.commitment_id)
                )
                if node_id is None:
                    continue
                planning_seed_node_ids.add(node_id)
                add(
                    BrainActiveContextAnchorCandidate(
                        anchor_id=f"commitment:{record.commitment_id}",
                        anchor_type="commitment",
                        label=record.title,
                        score=_planning_commitment_anchor_score(record.status),
                        seed_node_ids=(node_id,),
                        provenance={
                            "commitment_id": record.commitment_id,
                            "goal_id": record.current_goal_id,
                        },
                    )
                )
                for proposal_key in ("current_plan_proposal_id", "pending_plan_proposal_id"):
                    proposal_id = str(record.details.get(proposal_key, "")).strip()
                    if not proposal_id:
                        continue
                    proposal_node_id = graph_index.node_id_by_backing.get(
                        (BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value, proposal_id)
                    )
                    if proposal_node_id is None:
                        continue
                    planning_seed_node_ids.add(proposal_node_id)
                    add(
                        BrainActiveContextAnchorCandidate(
                            anchor_id=f"plan_proposal:{proposal_id}",
                            anchor_type="plan_proposal",
                            label=record.title,
                            score=7.25 if proposal_key == "pending_plan_proposal_id" else 7.5,
                            seed_node_ids=(proposal_node_id,),
                            provenance={
                                "commitment_id": record.commitment_id,
                                "plan_proposal_id": proposal_id,
                            },
                        )
                    )
            for skill_candidate in _planning_skill_anchor_candidates(
                snapshot=snapshot,
                query_text=query_text,
            ):
                add(skill_candidate)
            neighborhood = _planning_neighborhood(
                graph_index=graph_index,
                seed_node_ids=planning_seed_node_ids,
            )
            for dossier in (
                snapshot.continuity_dossiers.dossiers if snapshot.continuity_dossiers else ()
            ):
                if not _task_includes_continuity_items(task):
                    continue
                evidence_node_ids = set(dossier.summary_evidence.graph_node_ids)
                if not evidence_node_ids.intersection(neighborhood):
                    continue
                add(
                    BrainActiveContextAnchorCandidate(
                        anchor_id=f"dossier:{dossier.dossier_id}",
                        anchor_type="dossier",
                        label=dossier.title,
                        score=4.5 + _freshness_bonus(dossier.freshness),
                        seed_node_ids=tuple(sorted(evidence_node_ids)),
                        provenance={
                            "dossier_id": dossier.dossier_id,
                            "project_key": dossier.project_key,
                        },
                    )
                )
        else:
            for dossier in (
                snapshot.continuity_dossiers.dossiers if snapshot.continuity_dossiers else ()
            ):
                match_score = _dossier_match_score(dossier, query_text=query_text, tokens=tokens)
                if dossier.kind == "relationship":
                    match_score = max(match_score, 0.25)
                if match_score <= 0:
                    continue
                add(
                    BrainActiveContextAnchorCandidate(
                        anchor_id=f"dossier:{dossier.dossier_id}",
                        anchor_type="dossier",
                        label=dossier.title,
                        score=4.0 + match_score + _freshness_bonus(dossier.freshness),
                        seed_node_ids=tuple(sorted(dossier.summary_evidence.graph_node_ids)),
                        provenance={
                            "dossier_id": dossier.dossier_id,
                            "project_key": dossier.project_key,
                        },
                    )
                )
        for entry in snapshot.scene_episodes:
            candidate = _anchor_from_scene_episode(
                entry=entry,
                graph_index=graph_index,
                query_text=query_text,
                tokens=tokens,
                task=task,
                temporal_mode=temporal_mode,
            )
            if candidate is not None:
                add(candidate)

        return sorted(candidates.values(), key=_anchor_sort_key)

    def _build_packet_item_candidates(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        graph_index: _GraphIndex,
        selected_anchors: tuple[BrainActiveContextAnchorCandidate, ...],
        visited_nodes: dict[str, int],
        task: BrainContextTask,
        temporal_mode: BrainContextTemporalMode,
        language: Language,
        latest_user_text: str,
    ) -> list[_PacketItemCandidate]:
        items: dict[str, _PacketItemCandidate] = {}
        selected_anchor_ids = {anchor.anchor_id for anchor in selected_anchors}
        blocked_commitment_ids = {
            record.commitment_id for record in snapshot.commitment_projection.blocked_commitments
        }
        active_links = _current_linkage(snapshot)
        tokens = _query_tokens(latest_user_text)

        def add(item: _PacketItemCandidate):
            existing = items.get(item.item_id)
            if existing is None or item.score > existing.score:
                items[item.item_id] = item

        for anchor in selected_anchors:
            if anchor.anchor_type == "dossier":
                dossier_id = str(anchor.provenance.get("dossier_id", "")).strip()
                dossier = next(
                    (
                        record
                        for record in snapshot.continuity_dossiers.dossiers
                        if record.dossier_id == dossier_id
                    ),
                    None,
                )
                if dossier is None:
                    continue
                if not _task_includes_continuity_items(task):
                    continue
                if not _allow_dossier_current(
                    dossier,
                    task=task,
                    temporal_mode=temporal_mode,
                    blocked_commitment_ids=blocked_commitment_ids,
                ):
                    availability_state, _ = _dossier_task_availability(
                        dossier=dossier,
                        task=task,
                    )
                    if availability_state == BrainContinuityDossierAvailability.SUPPRESSED.value:
                        suppressed_item = _packet_item_from_dossier(
                            dossier,
                            task=task,
                            language=language,
                            forced_reason="governance_suppressed",
                        )
                        if suppressed_item is not None:
                            add(suppressed_item)
                    if not _allow_dossier_change_context(
                        dossier,
                        task=task,
                        temporal_mode=temporal_mode,
                        blocked_commitment_ids=blocked_commitment_ids,
                    ):
                        continue
                    add(
                        _packet_item_from_dossier_recent_change(
                            dossier,
                            task=task,
                            language=language,
                            temporal_mode=temporal_mode,
                        )
                    )
                    continue
                dossier_item = _packet_item_from_dossier(dossier, task=task, language=language)
                if dossier_item is not None:
                    add(dossier_item)
            elif anchor.anchor_type == "scene_episode" and _task_includes_continuity_items(task):
                entry_id = str(anchor.provenance.get("entry_id", "")).strip()
                node_id = graph_index.node_id_by_backing.get(
                    (BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value, entry_id)
                )
                node = graph_index.node_by_id.get(node_id) if node_id is not None else None
                if node is None or not _is_scene_episode_node(node):
                    continue
                temporal_kind = (
                    "current"
                    if node_id in graph_index.current_node_ids
                    and node_id not in graph_index.historical_node_ids
                    else "historical"
                )
                if (
                    temporal_kind == "historical"
                    and temporal_mode == BrainContextTemporalMode.CURRENT_FIRST
                    and not _allow_historical_node(node=node, graph_index=graph_index, task=task)
                ):
                    continue
                add(
                    _packet_item_from_scene_episode_node(
                        node=node,
                        task=task,
                        current_section_key=_continuity_candidate_section_key(
                            task=task,
                            temporal_kind=temporal_kind,
                        ),
                        temporal_kind=temporal_kind,
                        score=max(
                            anchor.score,
                            _node_item_score(
                                node_kind=node.kind,
                                temporal_kind=temporal_kind,
                                hop_distance=0,
                                temporal_mode=temporal_mode,
                                stale=node_id in graph_index.stale_node_ids,
                            )
                            + 0.75,
                        ),
                        hop_distance=0,
                    )
                )
            elif anchor.anchor_type == "commitment" and _task_has_section(task, "planning_anchors"):
                commitment_id = str(anchor.provenance.get("commitment_id", "")).strip()
                record = _commitment_by_id(snapshot, commitment_id)
                if record is None:
                    continue
                add(_packet_item_from_commitment(record))
            elif anchor.anchor_type == "procedural_skill" and _task_has_section(
                task, "planning_anchors"
            ):
                add(_packet_item_from_procedural_skill(anchor))

        for record in snapshot.active_situation_model.records:
            candidate = _packet_item_from_active_situation_record(
                record,
                task=task,
                active_links=active_links,
                tokens=tokens,
            )
            if candidate is not None:
                add(candidate)
        for record in snapshot.private_working_memory.records:
            candidate = _packet_item_from_private_working_memory_record(
                record,
                task=task,
                active_links=active_links,
                tokens=tokens,
            )
            if candidate is not None:
                add(candidate)
        for record in snapshot.scene_world_state.entities:
            candidate = _packet_item_from_scene_world_entity(
                record,
                task=task,
                tokens=tokens,
            )
            if candidate is not None:
                add(candidate)
        for record in snapshot.scene_world_state.affordances:
            candidate = _packet_item_from_scene_world_affordance(
                record,
                task=task,
                tokens=tokens,
            )
            if candidate is not None:
                add(candidate)
        scene_uncertainty = _scene_world_uncertainty_item(snapshot=snapshot, task=task)
        if scene_uncertainty is not None:
            add(scene_uncertainty)

        for node_id, hop_distance in visited_nodes.items():
            node = graph_index.node_by_id.get(node_id)
            if node is None:
                continue
            node_kind = node.kind
            is_current = (
                node_id in graph_index.current_node_ids
                and node_id not in graph_index.historical_node_ids
            )
            temporal_kind = "current" if is_current else "historical"
            if (
                temporal_kind == "historical"
                and temporal_mode == BrainContextTemporalMode.CURRENT_FIRST
            ):
                if not _allow_historical_node(node=node, graph_index=graph_index, task=task):
                    continue
            if (
                node_kind == BrainContinuityGraphNodeKind.CLAIM.value
                and _task_includes_continuity_items(task)
            ):
                availability_state, _ = _claim_node_availability(node)
                base_section_key = _continuity_candidate_section_key(
                    task=task,
                    temporal_kind=temporal_kind,
                )
                add(
                    _packet_item_from_graph_node(
                        node=node,
                        section_key=_annotated_section_key(
                            task=task,
                            current_section_key=base_section_key,
                            availability_state=availability_state,
                        ),
                        item_type="claim",
                        score=_node_item_score(
                            node_kind=node_kind,
                            temporal_kind=temporal_kind,
                            hop_distance=hop_distance,
                            temporal_mode=temporal_mode,
                            stale=node_id in graph_index.stale_node_ids,
                        ),
                        hop_distance=hop_distance,
                    )
                )
            elif (
                node_kind == BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value
                and _task_includes_continuity_items(task)
            ):
                base_section_key = _continuity_candidate_section_key(
                    task=task,
                    temporal_kind=temporal_kind,
                )
                score = _node_item_score(
                    node_kind=node_kind,
                    temporal_kind=temporal_kind,
                    hop_distance=hop_distance,
                    temporal_mode=temporal_mode,
                    stale=node_id in graph_index.stale_node_ids,
                )
                if _is_scene_episode_node(node):
                    add(
                        _packet_item_from_scene_episode_node(
                            node=node,
                            task=task,
                            current_section_key=base_section_key,
                            temporal_kind=temporal_kind,
                            score=score + 0.75,
                            hop_distance=hop_distance,
                        )
                    )
                else:
                    add(
                        _packet_item_from_graph_node(
                            node=node,
                            section_key=base_section_key,
                            item_type="autobiography",
                            score=score,
                            hop_distance=hop_distance,
                        )
                    )
            elif (
                node_kind == BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value
                and _task_uses_planning_anchors(task)
                and _task_has_section(task, "planning_anchors")
            ):
                section_key = "planning_anchors" if temporal_kind == "current" else "recent_changes"
                score = _node_item_score(
                    node_kind=node_kind,
                    temporal_kind=temporal_kind,
                    hop_distance=hop_distance,
                    temporal_mode=temporal_mode,
                    stale=node_id in graph_index.stale_node_ids,
                )
                if f"plan_proposal:{node.backing_record_id}" in selected_anchor_ids:
                    score += 6.0
                add(
                    _packet_item_from_graph_node(
                        node=node,
                        section_key=section_key,
                        item_type="plan_proposal",
                        score=score,
                        hop_distance=hop_distance,
                    )
                )
            elif (
                node_kind == BrainContinuityGraphNodeKind.COMMITMENT.value
                and _task_uses_planning_anchors(task)
                and _task_has_section(task, "planning_anchors")
            ):
                if f"commitment:{node.backing_record_id}" in selected_anchor_ids:
                    continue
                add(
                    _packet_item_from_graph_node(
                        node=node,
                        section_key="planning_anchors",
                        item_type="commitment",
                        score=12.0 - (hop_distance * 0.5),
                        hop_distance=hop_distance,
                    )
                )

        return sorted(
            items.values(),
            key=lambda candidate: _item_sort_key(
                candidate,
                task=task,
                temporal_mode=temporal_mode,
            ),
        )


@dataclass(frozen=True)
class BrainContextCompiler:
    """Compile Blink context packets from a projection-backed surface."""

    store: BrainStore
    session_resolver: callable[[], BrainSessionIds]
    language: Language
    base_prompt: str
    context_surface_builder: BrainContextSurfaceBuilder
    context_selector: BrainContextSelector = field(default_factory=BrainContextSelector)
    teaching_knowledge_registry: KnowledgeReserveRegistry = field(
        default_factory=build_default_teaching_canon
    )

    def compile(
        self,
        *,
        latest_user_text: str,
        persona_modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
        persona_seriousness: str = "normal",
    ) -> str:
        """Compile the current Blink reply context."""
        return self.compile_packet(
            latest_user_text=latest_user_text,
            task=BrainContextTask.REPLY,
            persona_modality=persona_modality,
            persona_seriousness=persona_seriousness,
        ).prompt

    def compile_persona_expression_frame(
        self,
        *,
        latest_user_text: str = "",
        task: BrainContextTask = BrainContextTask.REPLY,
        enable_persona_expression: bool = True,
        persona_modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
        persona_seriousness: str = "normal",
    ) -> tuple[BrainExpressionFrame | None, tuple[str, ...], dict[str, str]]:
        """Compile the current persona expression frame without rendering a packet."""
        surface = self.context_surface_builder.build(
            latest_user_text=latest_user_text,
            task=task,
        )
        behavior_controls = self._behavior_controls()
        return _compile_persona_expression_frame_from_surface(
            snapshot=surface,
            task=task,
            language=self.language,
            enable_persona_expression=enable_persona_expression,
            persona_modality=persona_modality,
            persona_seriousness=persona_seriousness,
            behavior_controls=behavior_controls,
        )

    def compile_packet(
        self,
        *,
        latest_user_text: str,
        task: BrainContextTask = BrainContextTask.REPLY,
        enable_persona_expression: bool = True,
        persona_modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
        persona_seriousness: str = "normal",
        teaching_knowledge_context: str | None = None,
    ) -> BrainCompiledContextPacket:
        """Compile one task-aware context packet."""
        surface = self.context_surface_builder.build(
            latest_user_text=latest_user_text,
            task=task,
        )
        behavior_controls = self._behavior_controls()
        teaching_reason_codes: tuple[str, ...] | None = None
        teaching_decision: BrainKnowledgeRoutingDecision | None = None
        if teaching_knowledge_context is None:
            teaching_knowledge_context, teaching_reason_codes, teaching_decision = (
                self._auto_teaching_knowledge_context(
                    snapshot=surface,
                    latest_user_text=latest_user_text,
                    task=task,
                    enable_persona_expression=enable_persona_expression,
                    persona_modality=persona_modality,
                    persona_seriousness=persona_seriousness,
                    behavior_controls=behavior_controls,
                )
            )
        elif task == BrainContextTask.REPLY:
            teaching_decision = explicit_knowledge_routing_decision(
                task_mode=_persona_task_mode_for_context(task).value,
                language=_language_value(self.language),
                teaching_mode=None,
                estimated_tokens=approximate_token_count(teaching_knowledge_context),
                reason_codes=("teaching_knowledge_candidate", "teaching_knowledge_selected"),
            )
        return compile_context_packet_from_surface(
            snapshot=surface,
            latest_user_text=latest_user_text,
            task=task,
            language=self.language,
            session_ids=self.session_resolver(),
            base_prompt=self.base_prompt,
            budget_profile=BrainContextBudgetProfile.for_task(task.value),
            context_selector=self.context_selector,
            enable_persona_expression=enable_persona_expression,
            persona_modality=persona_modality,
            persona_seriousness=persona_seriousness,
            behavior_controls=behavior_controls,
            teaching_knowledge_context=teaching_knowledge_context,
            teaching_knowledge_reason_codes=teaching_reason_codes,
            teaching_knowledge_decision=teaching_decision,
        )

    def _behavior_controls(self) -> BrainBehaviorControlProfile | None:
        """Load behavior controls from the replayable relationship-scoped block."""
        try:
            return load_behavior_control_profile(
                store=self.store,
                session_ids=self.session_resolver(),
            )
        except Exception:
            return None

    def _auto_teaching_knowledge_context(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        latest_user_text: str,
        task: BrainContextTask,
        enable_persona_expression: bool,
        persona_modality: BrainPersonaModality | str,
        persona_seriousness: str,
        behavior_controls: BrainBehaviorControlProfile | None,
    ) -> tuple[str | None, tuple[str, ...] | None, BrainKnowledgeRoutingDecision]:
        """Select default teaching knowledge for reply packets when useful."""
        if task != BrainContextTask.REPLY:
            return (
                None,
                None,
                unavailable_knowledge_routing_decision("teaching_knowledge_reply_only"),
            )
        if self.teaching_knowledge_registry.is_empty:
            return (
                None,
                (
                    "teaching_knowledge_excluded",
                    "teaching_knowledge_auto_empty",
                    "registry_empty",
                    "teaching_mode:none",
                ),
                unavailable_knowledge_routing_decision(
                    "teaching_knowledge_auto_empty",
                    "registry_empty",
                ),
            )

        expression_frame, _persona_reason_codes, _status = (
            _compile_persona_expression_frame_from_surface(
                snapshot=snapshot,
                task=task,
                language=self.language,
                enable_persona_expression=enable_persona_expression,
                persona_modality=persona_modality,
                persona_seriousness=persona_seriousness,
                behavior_controls=behavior_controls,
            )
        )
        teaching_mode = expression_frame.teaching_mode if expression_frame is not None else None
        request = KnowledgeSelectionRequest(
            query_text=latest_user_text,
            task_mode=_persona_task_mode_for_context(task).value,
            language=_language_value(self.language),
            teaching_mode=teaching_mode,
            max_items=2,
            max_tokens=96,
        )
        result = select_teaching_knowledge(
            self.teaching_knowledge_registry,
            request,
        )
        teaching_mode_code = f"teaching_mode:{teaching_mode or 'none'}"
        decision = knowledge_routing_decision_from_selection(
            result=result,
            request=request,
            selection_kind="auto",
            reason_codes=(teaching_mode_code,),
        )
        if result.rendered_text.strip():
            return (
                result.rendered_text,
                (
                    "teaching_knowledge_candidate",
                    "teaching_knowledge_auto_selected",
                    *result.reason_codes,
                    teaching_mode_code,
                ),
                decision,
            )
        return (
            None,
            (
                "teaching_knowledge_excluded",
                "teaching_knowledge_auto_empty",
                *result.reason_codes,
                teaching_mode_code,
            ),
            decision,
        )


def compile_context_packet_from_surface(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    latest_user_text: str,
    task: BrainContextTask,
    language: Language,
    session_ids: BrainSessionIds | None = None,
    base_prompt: str = "",
    budget_profile: BrainContextBudgetProfile | None = None,
    context_selector: BrainContextSelector | None = None,
    enable_persona_expression: bool = True,
    persona_modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
    persona_seriousness: str = "normal",
    behavior_controls: BrainBehaviorControlProfile | None = None,
    teaching_knowledge_context: str | None = None,
    teaching_knowledge_reason_codes: tuple[str, ...] | None = None,
    teaching_knowledge_decision: BrainKnowledgeRoutingDecision | None = None,
) -> BrainCompiledContextPacket:
    """Compile one packet directly from a prebuilt surface snapshot."""
    policy = get_brain_context_mode_policy(task)
    budget = (budget_profile or BrainContextBudgetProfile.for_task(task.value)).resolved(task)
    selector = context_selector or BrainContextSelector()
    static_sections, static_section_reason_codes = _static_sections_from_surface(
        snapshot=snapshot,
        base_prompt=base_prompt,
        task=task,
        language=language,
        enable_persona_expression=enable_persona_expression,
        persona_modality=persona_modality,
        persona_seriousness=persona_seriousness,
        behavior_controls=behavior_controls,
        teaching_knowledge_context=teaching_knowledge_context,
        teaching_knowledge_reason_codes=teaching_knowledge_reason_codes,
    )
    if (
        teaching_knowledge_decision is None
        and task == BrainContextTask.REPLY
        and str(teaching_knowledge_context or "").strip()
    ):
        teaching_knowledge_decision = explicit_knowledge_routing_decision(
            task_mode=_persona_task_mode_for_context(task).value,
            language=_language_value(language),
            teaching_mode=None,
            estimated_tokens=approximate_token_count(teaching_knowledge_context or ""),
            reason_codes=teaching_knowledge_reason_codes
            or ("teaching_knowledge_candidate", "teaching_knowledge_selected"),
        )
    static_candidates = selector.build_candidates(
        snapshot=snapshot,
        language=language,
        task=task,
        static_sections=static_sections,
        static_section_reason_codes=static_section_reason_codes,
        budget_profile=budget,
    )
    static_keys = policy.static_section_keys
    dynamic_budget_reserve = min(
        budget.dynamic_token_reserve or 0,
        budget.max_tokens,
    )
    static_budget_cap = max(0, budget.max_tokens - dynamic_budget_reserve)
    static_sections_selected: list[BrainSelectedSection] = []
    decisions: list[BrainContextSelectionDecision] = []
    static_selected_tokens = 0
    for key in static_keys:
        candidate = static_candidates.get(key)
        if candidate is None:
            decisions.append(
                BrainContextSelectionDecision(
                    section_key=key,
                    title=_section_title(key),
                    selected=False,
                    estimated_tokens=0,
                    reason="not_available",
                    decision_reason_codes=("section_not_available",),
                )
            )
            continue
        if not candidate.content.strip():
            decisions.append(
                BrainContextSelectionDecision(
                    section_key=key,
                    title=candidate.title,
                    selected=False,
                    estimated_tokens=0,
                    reason="empty",
                    decision_reason_codes=(
                        "section_empty",
                        *candidate.decision_reason_codes,
                    ),
                )
            )
            continue
        if static_selected_tokens + candidate.estimated_tokens > static_budget_cap:
            decisions.append(
                BrainContextSelectionDecision(
                    section_key=key,
                    title=candidate.title,
                    selected=False,
                    estimated_tokens=candidate.estimated_tokens,
                    reason="budget_exceeded",
                    decision_reason_codes=(
                        "section_budget_exceeded",
                        *candidate.decision_reason_codes,
                    ),
                )
            )
            continue
        static_sections_selected.append(
            BrainSelectedSection(
                key=candidate.key,
                title=candidate.title,
                content=candidate.content,
                estimated_tokens=candidate.estimated_tokens,
                source=candidate.source,
            )
        )
        static_selected_tokens += candidate.estimated_tokens
        decisions.append(
            BrainContextSelectionDecision(
                section_key=candidate.key,
                title=candidate.title,
                selected=True,
                estimated_tokens=candidate.estimated_tokens,
                reason="selected",
                decision_reason_codes=(
                    "section_selected",
                    *candidate.decision_reason_codes,
                ),
            )
        )

    active_packet = BrainActiveContextCompiler().compile(
        snapshot=snapshot,
        latest_user_text=latest_user_text,
        task=task,
        dynamic_token_budget=max(
            dynamic_budget_reserve,
            budget.max_tokens - static_selected_tokens,
        ),
        static_token_usage=static_selected_tokens,
        language=language,
    )
    dynamic_section_map = {section.key: section for section in active_packet.sections}
    for key in policy.dynamic_section_keys:
        section = dynamic_section_map.get(key)
        if section is None:
            decisions.append(
                BrainContextSelectionDecision(
                    section_key=key,
                    title=_section_title(key),
                    selected=False,
                    estimated_tokens=0,
                    reason="empty",
                    decision_reason_codes=("dynamic_section_empty",),
                )
            )
            continue
        decisions.append(
            BrainContextSelectionDecision(
                section_key=key,
                title=section.title,
                selected=True,
                estimated_tokens=section.estimated_tokens,
                reason="selected",
                decision_reason_codes=("dynamic_section_selected",),
            )
        )

    selected = BrainSelectedContext(
        task=task,
        budget_profile=budget,
        sections=tuple(static_sections_selected + list(active_packet.sections)),
        selection_trace=BrainContextSelectionTrace(
            task=task.value,
            budget_profile=budget,
            total_candidate_tokens=sum(
                static_candidates[key].estimated_tokens
                for key in static_keys
                if key in static_candidates
            )
            + active_packet.estimated_tokens,
            total_selected_tokens=static_selected_tokens + active_packet.estimated_tokens,
            decisions=tuple(decisions),
        ),
    )
    memory_use_trace = _memory_use_trace_from_context_selection(
        snapshot=snapshot,
        task=task,
        selected_context=selected,
        active_trace=active_packet.trace,
        session_ids=session_ids,
    )
    return BrainCompiledContextPacket(
        task=task,
        prompt=selected.render_prompt(header=BRAIN_CONTEXT_HEADER),
        selected_context=selected,
        packet_trace=active_packet.trace,
        memory_use_trace=memory_use_trace,
        teaching_knowledge_decision=teaching_knowledge_decision,
    )


def _build_graph_index(graph) -> _GraphIndex:
    node_by_id = {record.node_id: record for record in graph.nodes}
    edge_by_id = {record.edge_id: record for record in graph.edges}
    node_id_by_backing = {
        (record.kind, record.backing_record_id): record.node_id for record in graph.nodes
    }
    neighbors: dict[str, list[tuple[Any, str, bool]]] = {}
    for edge in graph.edges:
        neighbors.setdefault(edge.from_node_id, []).append((edge, edge.to_node_id, False))
        neighbors.setdefault(edge.to_node_id, []).append((edge, edge.from_node_id, True))
    return _GraphIndex(
        node_by_id=node_by_id,
        edge_by_id=edge_by_id,
        node_id_by_backing=node_id_by_backing,
        neighbors_by_node_id={key: tuple(value) for key, value in neighbors.items()},
        current_node_ids=set(graph.current_node_ids),
        historical_node_ids=set(graph.historical_node_ids),
        stale_node_ids=set(graph.stale_node_ids),
        superseded_node_ids=set(graph.superseded_node_ids),
    )


def _empty_graph_index() -> _GraphIndex:
    return _GraphIndex(
        node_by_id={},
        edge_by_id={},
        node_id_by_backing={},
        neighbors_by_node_id={},
        current_node_ids=set(),
        historical_node_ids=set(),
        stale_node_ids=set(),
        superseded_node_ids=set(),
    )


def _infer_temporal_mode(text: str) -> BrainContextTemporalMode:
    normalized = (text or "").lower()
    if any(cue in normalized for cue in _CURRENT_FIRST_CUES):
        return BrainContextTemporalMode.HISTORICAL_FOCUS
    return BrainContextTemporalMode.CURRENT_FIRST


def _query_tokens(text: str) -> tuple[str, ...]:
    return tuple(sorted({token for token in re.findall(r"[\w-]+", (text or "").lower()) if token}))


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _memory_use_trace_from_context_selection(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    task: BrainContextTask,
    selected_context: BrainSelectedContext,
    active_trace: BrainActiveContextTrace | None,
    session_ids: BrainSessionIds | None,
) -> BrainMemoryUseTrace:
    user_id, agent_id, thread_id = _memory_trace_scope(snapshot=snapshot, session_ids=session_ids)
    selected_section_names = tuple(section.key for section in selected_context.sections)
    refs: list[BrainMemoryUseTraceRef] = []
    claim_by_id = {
        claim.claim_id: claim
        for claim in (*snapshot.current_claims, *snapshot.historical_claims)
        if _normalized_text(getattr(claim, "claim_id", ""))
    }
    if active_trace is not None:
        for item in active_trace.selected_items:
            ref = _memory_use_ref_from_packet_item(
                item=item,
                user_id=user_id,
                claim_by_id=claim_by_id,
            )
            if ref is not None:
                refs.append(ref)
    if selected_context.section("user_profile") is not None:
        refs.extend(_user_profile_memory_use_refs(snapshot.current_claims, user_id=user_id))
    if selected_context.section("persona_expression") is not None:
        refs.extend(_persona_expression_memory_use_refs(snapshot.core_blocks))
    return build_memory_use_trace(
        user_id=user_id,
        agent_id=agent_id,
        thread_id=thread_id,
        task=task.value,
        selected_section_names=selected_section_names,
        refs=refs,
        reason_codes=(
            "context_selection_sidecar",
            "context_selection_refs_empty" if not refs else "context_selection_refs_selected",
        ),
    )


def _memory_trace_scope(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    session_ids: BrainSessionIds | None,
) -> tuple[str, str, str]:
    if session_ids is not None:
        return (
            _normalized_text(session_ids.user_id),
            _normalized_text(session_ids.agent_id),
            _normalized_text(session_ids.thread_id),
        )
    user_id = next(
        (
            _normalized_text(claim.scope_id)
            for claim in (*snapshot.current_claims, *snapshot.historical_claims)
            if _normalized_text(getattr(claim, "scope_type", "")) == "user"
        ),
        "",
    )
    relationship_scope = next(
        (
            _normalized_text(block.scope_id)
            for block in snapshot.core_blocks.values()
            if _normalized_text(getattr(block, "scope_type", "")) == "relationship"
        ),
        "",
    )
    agent_id = ""
    if relationship_scope and ":" in relationship_scope:
        agent_id, possible_user_id = relationship_scope.split(":", 1)
        user_id = user_id or possible_user_id
    if not agent_id:
        agent_id = next(
            (
                _normalized_text(block.scope_id)
                for block in snapshot.core_blocks.values()
                if _normalized_text(getattr(block, "scope_type", "")) == "agent"
            ),
            "",
        )
    return user_id, agent_id, ""


def _claim_trace_title(claim: Any) -> str:
    obj = getattr(claim, "object", None)
    if isinstance(obj, dict):
        for key in ("value", "summary", "text"):
            value = _normalized_text(obj.get(key))
            if value:
                return value
    return _normalized_text(getattr(claim, "predicate", "")) or "Memory"


def _user_profile_memory_use_refs(
    claims: tuple[Any, ...],
    *,
    user_id: str,
) -> tuple[BrainMemoryUseTraceRef, ...]:
    by_predicate: dict[str, list[Any]] = {}
    for claim in claims:
        predicate = _normalized_text(getattr(claim, "predicate", ""))
        if predicate.startswith(("profile.", "preference.")):
            by_predicate.setdefault(predicate, []).append(claim)

    selected_claims: list[Any] = []
    for predicate in ("profile.name", "profile.role", "profile.origin"):
        selected_claims.extend(by_predicate.get(predicate, ())[:1])
    for predicate in ("preference.like", "preference.dislike"):
        selected_claims.extend(by_predicate.get(predicate, ())[:4])

    refs: list[BrainMemoryUseTraceRef] = []
    for claim in selected_claims:
        claim_id = _normalized_text(getattr(claim, "claim_id", ""))
        if not claim_id:
            continue
        display_kind = display_kind_for_claim_predicate(
            _normalized_text(getattr(claim, "predicate", ""))
        )
        currentness = _normalized_text(getattr(claim, "effective_currentness_status", ""))
        refs.append(
            BrainMemoryUseTraceRef(
                memory_id=f"memory_claim:user:{user_id}:{claim_id}",
                display_kind=display_kind,
                title=_claim_trace_title(claim),
                section_key="user_profile",
                used_reason="selected_for_user_profile",
                safe_provenance_label=render_safe_memory_provenance_label(
                    display_kind=display_kind,
                    currentness_status=currentness,
                ),
                reason_codes=(
                    "source:context_selection",
                    "source_item:claim",
                    "section:user_profile",
                ),
            )
        )
    return tuple(refs)


def _memory_use_ref_from_packet_item(
    *,
    item: BrainActiveContextPacketItemRecord,
    user_id: str,
    claim_by_id: dict[str, Any],
) -> BrainMemoryUseTraceRef | None:
    if item.item_type == "claim":
        claim_id = _normalized_text(
            item.provenance.get("backing_record_id") or item.provenance.get("claim_id")
        )
        if not claim_id:
            return None
        claim = claim_by_id.get(claim_id)
        display_kind = (
            display_kind_for_claim_predicate(_normalized_text(getattr(claim, "predicate", "")))
            if claim is not None
            else "claim"
        )
        currentness = (
            _normalized_text(getattr(claim, "effective_currentness_status", ""))
            if claim is not None
            else None
        )
        return BrainMemoryUseTraceRef(
            memory_id=f"memory_claim:user:{user_id}:{claim_id}",
            display_kind=display_kind,
            title=_normalized_text(item.title) or "Memory",
            section_key=item.section_key,
            used_reason=_used_reason_for_section(item.section_key),
            safe_provenance_label=render_safe_memory_provenance_label(
                display_kind=display_kind,
                currentness_status=currentness,
            ),
            reason_codes=(
                "source:context_selection",
                "source_item:claim",
                f"section:{item.section_key}",
            ),
        )
    if item.item_type == "commitment":
        commitment_id = _normalized_text(
            item.provenance.get("commitment_id") or item.provenance.get("backing_record_id")
        )
        if not commitment_id:
            return None
        return BrainMemoryUseTraceRef(
            memory_id=f"memory_task:user:{user_id}:{commitment_id}",
            display_kind="task",
            title=_normalized_text(item.title) or "Task",
            section_key=item.section_key,
            used_reason=_used_reason_for_section(item.section_key),
            safe_provenance_label=render_safe_memory_provenance_label(display_kind="task"),
            reason_codes=(
                "source:context_selection",
                "source_item:commitment",
                f"section:{item.section_key}",
            ),
        )
    return None


def _persona_expression_memory_use_refs(
    core_blocks: dict[str, Any],
) -> tuple[BrainMemoryUseTraceRef, ...]:
    refs: list[BrainMemoryUseTraceRef] = []
    for block_kind, display_kind, title in (
        (
            BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value,
            "relationship_style",
            "Relationship style",
        ),
        (
            BrainCoreMemoryBlockKind.TEACHING_PROFILE.value,
            "teaching_profile",
            "Teaching profile",
        ),
    ):
        block = core_blocks.get(block_kind)
        if block is None:
            continue
        refs.append(
            BrainMemoryUseTraceRef(
                memory_id=(
                    f"memory_block:{block.scope_type}:{block.scope_id}:"
                    f"{block.block_kind}:{block.version}"
                ),
                display_kind=display_kind,
                title=title,
                section_key="persona_expression",
                used_reason="selected_for_persona_expression",
                safe_provenance_label=render_safe_memory_provenance_label(
                    display_kind=display_kind
                ),
                reason_codes=(
                    "source:context_selection",
                    "source_item:core_block",
                    "section:persona_expression",
                ),
            )
        )
    return tuple(refs)


def _used_reason_for_section(section_key: str) -> str:
    return {
        "persona_expression": "selected_for_persona_expression",
        "active_continuity": "selected_for_active_continuity",
        "relevant_continuity": "selected_for_relevant_continuity",
        "recent_changes": "selected_for_recent_context",
        "unresolved_state": "selected_for_unresolved_context",
        "planning_anchors": "selected_for_planning_context",
        "user_profile": "selected_for_user_profile",
    }.get(section_key, "selected_for_reply_context")


def _text_match_score(*, tokens: tuple[str, ...], parts: Any) -> float:
    haystack = " ".join(str(part or "") for part in parts).lower()
    if not haystack:
        return 0.0
    return float(sum(1 for token in tokens if token in haystack))


def _mode_policy(task: BrainContextTask) -> BrainContextModePolicy:
    return get_brain_context_mode_policy(task)


def _allowed_edge_kinds(task: BrainContextTask) -> set[str]:
    policy = _mode_policy(task)
    if policy.allowed_edge_profile == BrainContextEdgeProfile.PLANNING:
        return _PLANNING_ALLOWED_EDGE_KINDS
    return _REPLY_ALLOWED_EDGE_KINDS


def _task_has_section(task: BrainContextTask, section_key: str) -> bool:
    return section_key in _mode_policy(task).dynamic_section_keys


def _task_uses_planning_anchors(task: BrainContextTask) -> bool:
    return _mode_policy(task).uses_planning_anchors


def _task_includes_continuity_items(task: BrainContextTask) -> bool:
    return _mode_policy(task).includes_continuity_items


def _task_surface_label(task: BrainContextTask) -> str:
    return task.value


def _anchor_from_memory_result(
    result: BrainMemorySearchResult,
    *,
    graph_index: _GraphIndex,
    score_bonus: float,
) -> BrainActiveContextAnchorCandidate | None:
    claim_id = str(result.metadata.get("claim_id", "")).strip()
    if not claim_id:
        return None
    node_id = graph_index.node_id_by_backing.get(
        (BrainContinuityGraphNodeKind.CLAIM.value, claim_id)
    )
    if node_id is None:
        return None
    return BrainActiveContextAnchorCandidate(
        anchor_id=f"claim:{claim_id}",
        anchor_type="claim",
        label=result.summary,
        score=float(result.score) + score_bonus + (1.5 if not result.stale else 0.0),
        seed_node_ids=(node_id,),
        provenance={"claim_id": claim_id},
    )


def _anchor_from_scene_episode(
    *,
    entry,
    graph_index: _GraphIndex,
    query_text: str,
    tokens: tuple[str, ...],
    task: BrainContextTask,
    temporal_mode: BrainContextTemporalMode,
) -> BrainActiveContextAnchorCandidate | None:
    node_id = graph_index.node_id_by_backing.get(
        (BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value, entry.entry_id)
    )
    if node_id is None:
        return None
    match_score = _text_match_score(
        tokens=tokens,
        parts=(query_text, entry.rendered_summary, entry.entry_kind, entry.content),
    )
    current_bonus = (
        1.5
        if entry.status == "current"
        else (1.0 if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS else 0.25)
    )
    task_bonus = (
        1.0
        if _mode_policy(task).scene_episode_policy == BrainContextSceneEpisodePolicy.ALLOWED
        else 0.5
    )
    return BrainActiveContextAnchorCandidate(
        anchor_id=f"scene_episode:{entry.entry_id}",
        anchor_type="scene_episode",
        label=entry.rendered_summary,
        score=6.0 + match_score + current_bonus + task_bonus,
        seed_node_ids=(node_id,),
        provenance={
            "entry_id": entry.entry_id,
            "status": entry.status,
            "privacy_class": entry.privacy_class,
            "review_state": entry.review_state,
            "source_presence_scope_key": entry.source_presence_scope_key,
        },
    )


def _planning_commitment_anchor_score(status: str) -> float:
    if status == "active":
        return 9.0
    if status == "blocked":
        return 8.0
    if status == "deferred":
        return 7.0
    return 6.0


def _planning_skill_anchor_candidates(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    query_text: str,
) -> tuple[BrainActiveContextAnchorCandidate, ...]:
    if snapshot.procedural_skills is None:
        return ()
    candidates: dict[str, BrainActiveContextAnchorCandidate] = {}
    for record in (
        list(snapshot.commitment_projection.active_commitments)
        + list(snapshot.commitment_projection.blocked_commitments)
        + list(snapshot.commitment_projection.deferred_commitments)
    ):
        if not record.current_goal_id:
            continue
        goal = snapshot.agenda.goal(record.current_goal_id)
        if goal is None:
            continue
        completed_prefix = planning_completed_prefix(goal)
        for candidate in match_planning_skills(
            procedural_skills=snapshot.procedural_skills,
            goal=goal,
            commitment=record,
            completed_prefix=completed_prefix,
            query_text=query_text or goal.title,
            max_candidates=4,
        ):
            score = _procedural_skill_anchor_score(candidate)
            anchor = BrainActiveContextAnchorCandidate(
                anchor_id=f"procedural_skill:{candidate.skill_id}",
                anchor_type="procedural_skill",
                label=candidate.title or candidate.skill_id,
                score=score,
                seed_node_ids=(),
                provenance={
                    "skill_id": candidate.skill_id,
                    "goal_id": goal.goal_id,
                    "commitment_id": record.commitment_id,
                    "skill_status": candidate.skill_status,
                    "eligibility": candidate.eligibility,
                    "confidence": candidate.confidence,
                    "required_capability_ids": list(candidate.required_capability_ids),
                    "support_trace_ids": list(candidate.support_trace_ids),
                    "support_plan_proposal_ids": list(candidate.support_plan_proposal_ids),
                    "review_policy": candidate.review_policy,
                    "reason": candidate.reason,
                },
            )
            existing = candidates.get(anchor.anchor_id)
            if existing is None or anchor.score > existing.score:
                candidates[anchor.anchor_id] = anchor
    return tuple(sorted(candidates.values(), key=_anchor_sort_key))


def _procedural_skill_anchor_score(candidate) -> float:
    base = {
        "reusable": 8.5,
        "advisory": 6.0,
        "rejected": 5.0,
    }.get(candidate.eligibility, 4.0)
    return base + float(candidate.confidence)


def _planning_neighborhood(
    *,
    graph_index: _GraphIndex,
    seed_node_ids: set[str],
) -> set[str]:
    neighborhood = set(seed_node_ids)
    for node_id in list(seed_node_ids):
        for edge, neighbor_id, _reverse in graph_index.neighbors_by_node_id.get(node_id, ()):
            if edge.kind not in _PLANNING_ALLOWED_EDGE_KINDS:
                continue
            neighborhood.add(neighbor_id)
    return neighborhood


def _freshness_bonus(freshness: str) -> float:
    if freshness == BrainContinuityDossierFreshness.FRESH.value:
        return 1.0
    if freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value:
        return -0.5
    return -1.0


def _dossier_match_score(dossier, *, query_text: str, tokens: tuple[str, ...]) -> float:
    haystack = " ".join(
        part for part in (dossier.title, dossier.summary, dossier.project_key or "") if part
    ).lower()
    score = 0.0
    if query_text and query_text.lower() in haystack:
        score += 2.0
    for token in tokens:
        if token in haystack:
            score += 0.75
    return score


def _select_anchor_candidates(
    anchor_candidates: list[BrainActiveContextAnchorCandidate],
    *,
    max_anchors: int,
) -> tuple[
    tuple[BrainActiveContextAnchorCandidate, ...], tuple[BrainActiveContextAnchorCandidate, ...]
]:
    selected: list[BrainActiveContextAnchorCandidate] = []
    dropped: list[BrainActiveContextAnchorCandidate] = []
    for index, candidate in enumerate(anchor_candidates):
        if index < max_anchors:
            selected.append(
                BrainActiveContextAnchorCandidate(
                    **{**candidate.as_dict(), "selected": True, "reason": "selected"}
                )
            )
        else:
            dropped.append(
                BrainActiveContextAnchorCandidate(
                    **{**candidate.as_dict(), "selected": False, "reason": "max_anchors"}
                )
            )
    return tuple(selected), tuple(dropped)


def _expand_selected_anchors(
    *,
    selected_anchors: tuple[BrainActiveContextAnchorCandidate, ...],
    graph_index: _GraphIndex,
    allowed_edge_kinds: set[str],
    max_hops: int,
    max_graph_nodes: int,
) -> tuple[dict[str, int], tuple[BrainActiveContextExpansionRecord, ...]]:
    visited: dict[str, int] = {}
    queue: list[tuple[str, int]] = []
    for anchor in selected_anchors:
        for node_id in anchor.seed_node_ids:
            if node_id not in visited:
                visited[node_id] = 0
                queue.append((node_id, 0))
    expansions: list[BrainActiveContextExpansionRecord] = []
    index = 0
    while index < len(queue):
        node_id, hop = queue[index]
        index += 1
        if hop >= max_hops:
            for edge, neighbor_id, reverse in graph_index.neighbors_by_node_id.get(node_id, ()):
                expansions.append(
                    BrainActiveContextExpansionRecord(
                        from_node_id=node_id,
                        to_node_id=neighbor_id,
                        edge_id=edge.edge_id,
                        edge_kind=edge.kind,
                        hop_distance=hop + 1,
                        accepted=False,
                        reason="hop_limit",
                        reverse=reverse,
                    )
                )
            continue
        node = graph_index.node_by_id.get(node_id)
        if node is None:
            continue
        if node.kind in {
            BrainContinuityGraphNodeKind.EVENT_ANCHOR.value,
            BrainContinuityGraphNodeKind.EPISODE_ANCHOR.value,
        }:
            continue
        for edge, neighbor_id, reverse in graph_index.neighbors_by_node_id.get(node_id, ()):
            if edge.kind not in allowed_edge_kinds:
                expansions.append(
                    BrainActiveContextExpansionRecord(
                        from_node_id=node_id,
                        to_node_id=neighbor_id,
                        edge_id=edge.edge_id,
                        edge_kind=edge.kind,
                        hop_distance=hop + 1,
                        accepted=False,
                        reason="unsupported_edge_kind",
                        reverse=reverse,
                    )
                )
                continue
            if neighbor_id in visited:
                expansions.append(
                    BrainActiveContextExpansionRecord(
                        from_node_id=node_id,
                        to_node_id=neighbor_id,
                        edge_id=edge.edge_id,
                        edge_kind=edge.kind,
                        hop_distance=hop + 1,
                        accepted=False,
                        reason="visited",
                        reverse=reverse,
                    )
                )
                continue
            if len(visited) >= max_graph_nodes:
                expansions.append(
                    BrainActiveContextExpansionRecord(
                        from_node_id=node_id,
                        to_node_id=neighbor_id,
                        edge_id=edge.edge_id,
                        edge_kind=edge.kind,
                        hop_distance=hop + 1,
                        accepted=False,
                        reason="graph_node_cap_exceeded",
                        reverse=reverse,
                    )
                )
                continue
            visited[neighbor_id] = hop + 1
            queue.append((neighbor_id, hop + 1))
            expansions.append(
                BrainActiveContextExpansionRecord(
                    from_node_id=node_id,
                    to_node_id=neighbor_id,
                    edge_id=edge.edge_id,
                    edge_kind=edge.kind,
                    hop_distance=hop + 1,
                    accepted=True,
                    reason="expanded",
                    reverse=reverse,
                )
            )
    return visited, tuple(expansions)


def _allow_dossier_current(
    dossier,
    *,
    task: BrainContextTask,
    temporal_mode: BrainContextTemporalMode,
    blocked_commitment_ids: set[str],
) -> bool:
    if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS and dossier.recent_changes:
        return False
    availability_state, _ = _dossier_task_availability(dossier=dossier, task=task)
    return availability_state == BrainContinuityDossierAvailability.AVAILABLE.value


def _allow_dossier_change_context(
    dossier,
    *,
    task: BrainContextTask,
    temporal_mode: BrainContextTemporalMode,
    blocked_commitment_ids: set[str],
) -> bool:
    if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS:
        return True
    availability_state, _ = _dossier_task_availability(dossier=dossier, task=task)
    if availability_state != BrainContinuityDossierAvailability.AVAILABLE.value:
        return True
    if _task_uses_planning_anchors(task) and blocked_commitment_ids:
        return True
    return False


def _allow_historical_node(*, node, graph_index: _GraphIndex, task: BrainContextTask) -> bool:
    if (
        node.kind == BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value
        and _task_uses_planning_anchors(task)
    ):
        return True
    if (
        _task_uses_planning_anchors(task)
        and node.kind == BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value
        and node.status == "superseded"
    ):
        return True
    if (
        _is_scene_episode_node(node)
        and _mode_policy(task).scene_episode_policy == BrainContextSceneEpisodePolicy.ALLOWED
    ):
        return True
    return False


def _node_item_score(
    *,
    node_kind: str,
    temporal_kind: str,
    hop_distance: int,
    temporal_mode: BrainContextTemporalMode,
    stale: bool,
) -> float:
    base = {
        BrainContinuityGraphNodeKind.CLAIM.value: 8.0,
        BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value: 7.0,
        BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value: 8.5,
        BrainContinuityGraphNodeKind.COMMITMENT.value: 9.0,
    }.get(node_kind, 5.0)
    if temporal_kind == "historical":
        base -= 3.0
        if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS:
            base += 2.0
    if stale:
        base -= 1.0
    return base - (hop_distance * 0.75)


def _continuity_section_key(task: BrainContextTask) -> str | None:
    return _mode_policy(task).continuity_section_key


def _continuity_candidate_section_key(
    *,
    task: BrainContextTask,
    temporal_kind: str,
) -> str:
    if temporal_kind == "historical":
        return "recent_changes"
    return _continuity_section_key(task) or (
        "unresolved_state" if _task_has_section(task, "unresolved_state") else "recent_changes"
    )


def _dossier_task_availability(
    dossier,
    *,
    task: BrainContextTask,
) -> tuple[str, tuple[str, ...]]:
    governance = getattr(dossier, "governance", None)
    if governance is not None:
        record = governance.availability_for_task(task.value)
        if record is not None:
            return record.availability, tuple(record.reason_codes)
    fallback_reason_codes: list[str] = []
    if dossier.freshness != BrainContinuityDossierFreshness.FRESH.value:
        fallback_reason_codes.append(dossier.freshness)
    if dossier.contradiction != BrainContinuityDossierContradiction.CLEAR.value:
        fallback_reason_codes.append(dossier.contradiction)
    availability_state = (
        BrainContinuityDossierAvailability.AVAILABLE.value
        if not fallback_reason_codes
        else BrainContinuityDossierAvailability.ANNOTATED.value
    )
    return availability_state, tuple(sorted(set(fallback_reason_codes)))


def _claim_node_availability(node) -> tuple[str, tuple[str, ...]]:
    if node.kind != BrainContinuityGraphNodeKind.CLAIM.value:
        return BrainContinuityDossierAvailability.AVAILABLE.value, ()
    details = dict(node.details or {})
    currentness = _normalized_text(details.get("currentness_status"))
    review_state = _normalized_text(details.get("review_state"))
    truth_status = _normalized_text(details.get("truth_status"))
    reason_codes = [
        str(code).strip() for code in details.get("reason_codes", []) if str(code).strip()
    ]
    if currentness == "stale":
        reason_codes.append("stale_support")
    elif currentness == "held":
        reason_codes.append("held_support")
    if review_state == "requested":
        reason_codes.append("review_debt")
    if truth_status == "uncertain":
        reason_codes.append("uncertain_support")
    availability_state = (
        BrainContinuityDossierAvailability.AVAILABLE.value
        if currentness in {"", "current"}
        and review_state in {"", "none"}
        and truth_status in {"", "active"}
        else BrainContinuityDossierAvailability.ANNOTATED.value
    )
    return availability_state, tuple(sorted({code for code in reason_codes if code}))


def _is_scene_episode_node(node) -> bool:
    return (
        node.kind == BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value
        and _normalized_text(dict(node.details or {}).get("entry_kind"))
        == BrainAutobiographyEntryKind.SCENE_EPISODE.value
    )


def _scene_episode_node_availability(node) -> tuple[str, tuple[str, ...]]:
    details = dict(node.details or {})
    privacy_class = _normalized_text(details.get("privacy_class"))
    review_state = _normalized_text(details.get("review_state"))
    redacted_at = _normalized_text(details.get("redacted_at"))
    reason_codes = [
        str(code).strip()
        for code in details.get("governance_reason_codes", [])
        if str(code).strip()
    ]
    if privacy_class == "redacted" or redacted_at:
        reason_codes.append("redacted")
    elif privacy_class == "sensitive":
        reason_codes.append("sensitive_support")
    if review_state == "requested":
        reason_codes.append("review_debt")
    availability_state = (
        BrainContinuityDossierAvailability.AVAILABLE.value
        if privacy_class in {"", "standard"} and review_state in {"", "none"} and not redacted_at
        else BrainContinuityDossierAvailability.ANNOTATED.value
    )
    return availability_state, tuple(sorted({code for code in reason_codes if code}))


def _scene_episode_forced_reason(
    *,
    task: BrainContextTask,
    node,
    temporal_kind: str,
) -> str | None:
    policy = _mode_policy(task)
    details = dict(node.details or {})
    privacy_class = _normalized_text(details.get("privacy_class"))
    review_state = _normalized_text(details.get("review_state"))
    redacted_at = _normalized_text(details.get("redacted_at"))
    if policy.scene_episode_policy == BrainContextSceneEpisodePolicy.DISABLED:
        return "scene_episode_task_ineligible"
    if policy.scene_episode_policy != BrainContextSceneEpisodePolicy.CONSERVATIVE:
        return None
    if privacy_class == "redacted" or redacted_at:
        return "scene_episode_redacted"
    if temporal_kind != "current" or node.status != "current":
        return "scene_episode_not_current"
    if privacy_class not in {"", "standard"}:
        return "scene_episode_privacy_gated"
    if review_state not in {"", "none"}:
        return "scene_episode_review_gated"
    return None


def _prediction_forced_reason(
    *,
    task: BrainContextTask,
    record,
) -> str | None:
    policy = _mode_policy(task)
    confidence_band = _normalized_text(dict(record.details or {}).get("confidence_band"))
    resolution_kind = _normalized_text(dict(record.details or {}).get("resolution_kind"))
    if policy.prediction_policy == BrainContextPredictionPolicy.DISABLED:
        return "prediction_task_ineligible"
    if policy.prediction_policy != BrainContextPredictionPolicy.CONSERVATIVE:
        return None
    if resolution_kind:
        return "prediction_resolution_ineligible"
    if confidence_band not in {"medium", "high"}:
        return "prediction_low_confidence"
    return None


def _annotated_section_key(
    *,
    task: BrainContextTask,
    current_section_key: str,
    availability_state: str,
) -> str:
    policy = _mode_policy(task)
    if (
        availability_state != BrainContinuityDossierAvailability.AVAILABLE.value
        and policy.prefer_unresolved_for_nonclean_continuity
        and _task_has_section(task, "unresolved_state")
    ):
        return "unresolved_state"
    return current_section_key


def _governance_tags(
    *,
    availability_state: str,
    reason_codes: tuple[str, ...],
) -> list[str]:
    tags: list[str] = []
    if availability_state != BrainContinuityDossierAvailability.AVAILABLE.value:
        tags.append(availability_state)
    if reason_codes:
        tags.append("reason=" + ",".join(reason_codes[:2]))
    return tags


def _packet_item_from_dossier(
    dossier,
    *,
    task: BrainContextTask,
    language: Language,
    forced_reason: str | None = None,
) -> _PacketItemCandidate | None:
    section_key = _continuity_section_key(task)
    if section_key is None:
        return None
    availability_state, reason_codes = _dossier_task_availability(dossier=dossier, task=task)
    summary = f"{dossier.title}: {dossier.summary}"
    if dossier.freshness != BrainContinuityDossierFreshness.FRESH.value:
        summary += f" [{dossier.freshness}]"
    if dossier.contradiction != BrainContinuityDossierContradiction.CLEAR.value:
        summary += f" [{dossier.contradiction}]"
    for tag in _governance_tags(
        availability_state=availability_state,
        reason_codes=reason_codes,
    ):
        summary += f" [{tag}]"
    return _PacketItemCandidate(
        item_id=f"dossier:{dossier.dossier_id}",
        item_type="dossier",
        section_key=section_key,
        title=dossier.title,
        content=f"- {summary}",
        temporal_kind="current",
        availability_state=availability_state,
        governance_reason_codes=reason_codes,
        provenance={
            "dossier_id": dossier.dossier_id,
            "claim_ids": list(dossier.summary_evidence.claim_ids),
            "entry_ids": list(dossier.summary_evidence.entry_ids),
            "source_event_ids": list(dossier.summary_evidence.source_event_ids),
            "graph_node_ids": list(dossier.summary_evidence.graph_node_ids),
            "graph_edge_ids": list(dossier.summary_evidence.graph_edge_ids),
        },
        score=9.0 + _freshness_bonus(dossier.freshness),
        decision_reason_codes=(
            "dossier_candidate",
            f"availability_{availability_state}",
            "rendered_as_current_context",
        ),
        is_dossier=True,
        forced_reason=forced_reason,
    )


def _packet_item_from_dossier_recent_change(
    dossier,
    *,
    task: BrainContextTask,
    language: Language,
    temporal_mode: BrainContextTemporalMode,
) -> _PacketItemCandidate:
    availability_state, reason_codes = _dossier_task_availability(dossier=dossier, task=task)
    change = None
    for candidate in dossier.recent_changes:
        summary = str(candidate.summary or "").strip()
        if summary and not summary.startswith("Memory continuity changed after"):
            change = candidate
            break
    if change is None:
        change = dossier.recent_changes[0] if dossier.recent_changes else None
    content = f"- {dossier.title}: {dossier.summary}"
    if change is not None:
        content = f"- {dossier.title}: {change.summary}"
    tags = _governance_tags(
        availability_state=availability_state,
        reason_codes=reason_codes,
    )
    if tags:
        content = f"- {' '.join(f'[{tag}]' for tag in tags)} {content[2:]}"
    return _PacketItemCandidate(
        item_id=f"dossier_change:{dossier.dossier_id}",
        item_type="dossier_change",
        section_key=(
            "recent_changes"
            if temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS
            else "unresolved_state"
            if _task_has_section(task, "unresolved_state")
            and _mode_policy(task).prefer_unresolved_for_nonclean_continuity
            else "recent_changes"
        ),
        title=dossier.title,
        content=content,
        temporal_kind="historical",
        availability_state=availability_state,
        governance_reason_codes=reason_codes,
        provenance={
            "dossier_id": dossier.dossier_id,
            "claim_ids": list(dossier.summary_evidence.claim_ids),
            "entry_ids": list(dossier.summary_evidence.entry_ids),
            "source_event_ids": list(dossier.summary_evidence.source_event_ids),
            "graph_node_ids": list(dossier.summary_evidence.graph_node_ids),
            "graph_edge_ids": list(dossier.summary_evidence.graph_edge_ids),
        },
        score=5.0,
        decision_reason_codes=(
            "dossier_candidate",
            f"availability_{availability_state}",
            "rendered_as_change_context",
        ),
        is_dossier=True,
    )


def _packet_item_from_commitment(record) -> _PacketItemCandidate:
    parts = [
        f"{record.title} [{record.status}]",
        f"family={record.goal_family}",
        f"goal={record.current_goal_id or 'None'}",
        f"rev={record.plan_revision}",
    ]
    if record.blocked_reason is not None:
        parts.append(f"blocker={record.blocked_reason.summary}")
    current_plan_proposal_id = str(record.details.get("current_plan_proposal_id", "")).strip()
    pending_plan_proposal_id = str(record.details.get("pending_plan_proposal_id", "")).strip()
    if current_plan_proposal_id:
        parts.append(f"current_plan={current_plan_proposal_id}")
    if pending_plan_proposal_id:
        parts.append(f"pending_plan={pending_plan_proposal_id}")
    return _PacketItemCandidate(
        item_id=f"commitment:{record.commitment_id}",
        item_type="commitment",
        section_key="planning_anchors",
        title=record.title,
        content="- " + ", ".join(parts),
        temporal_kind="current",
        provenance={
            "commitment_id": record.commitment_id,
            "goal_id": record.current_goal_id,
            "current_plan_proposal_id": current_plan_proposal_id or None,
            "pending_plan_proposal_id": pending_plan_proposal_id or None,
        },
        score=12.0,
        decision_reason_codes=("planning_anchor_candidate", "rendered_as_planning_anchor"),
    )


def _packet_item_from_procedural_skill(
    anchor: BrainActiveContextAnchorCandidate,
) -> _PacketItemCandidate:
    provenance = dict(anchor.provenance)
    eligibility = str(provenance.get("eligibility", "")).strip() or "advisory"
    skill_id = str(provenance.get("skill_id", "")).strip()
    required_capability_ids = [
        str(item).strip()
        for item in provenance.get("required_capability_ids", [])
        if str(item).strip()
    ]
    review_policy = str(provenance.get("review_policy", "")).strip()
    summary_parts = [anchor.label or skill_id, f"[{eligibility}]"]
    if review_policy:
        summary_parts.append(f"review={review_policy}")
    if required_capability_ids:
        summary_parts.append("caps=" + " -> ".join(required_capability_ids))
    reason = str(provenance.get("reason", "")).strip()
    if reason:
        summary_parts.append(f"reason={reason}")
    section_key = "planning_anchors" if eligibility != "rejected" else "recent_changes"
    temporal_kind = "current" if eligibility != "rejected" else "historical"
    return _PacketItemCandidate(
        item_id=f"procedural_skill:{skill_id}",
        item_type="procedural_skill",
        section_key=section_key,
        title=anchor.label,
        content="- " + ", ".join(part for part in summary_parts if part),
        temporal_kind=temporal_kind,
        provenance={
            "skill_id": skill_id,
            "goal_id": provenance.get("goal_id"),
            "commitment_id": provenance.get("commitment_id"),
            "support_trace_ids": list(provenance.get("support_trace_ids") or []),
            "support_plan_proposal_ids": list(provenance.get("support_plan_proposal_ids") or []),
            "eligibility": eligibility,
            "reason": reason or None,
        },
        score=anchor.score,
        decision_reason_codes=(
            "planning_anchor_candidate",
            f"eligibility_{eligibility}",
            "rendered_as_planning_anchor"
            if section_key == "planning_anchors"
            else "rendered_as_recent_change_context",
        ),
    )


def _packet_item_from_graph_node(
    *,
    node,
    section_key: str,
    item_type: str,
    score: float,
    hop_distance: int,
) -> _PacketItemCandidate:
    content = node.summary
    availability_state = BrainContinuityDossierAvailability.AVAILABLE.value
    governance_reason_codes: tuple[str, ...] = ()
    if item_type == "plan_proposal":
        details = node.details
        review = _normalized_text(details.get("review_policy"))
        current_revision = details.get("current_plan_revision")
        plan_revision = details.get("plan_revision")
        procedural = dict(details.get("procedural", {}))
        rendered = [node.summary, f"status={node.status}"]
        if review:
            rendered.append(f"review={review}")
        if current_revision is not None and plan_revision is not None:
            rendered.append(f"rev={current_revision}->{plan_revision}")
        origin = str(procedural.get("origin", "")).strip()
        if origin:
            rendered.append(f"origin={origin}")
        selected_skill_id = str(procedural.get("selected_skill_id", "")).strip()
        if selected_skill_id:
            rendered.append(f"skill={selected_skill_id}")
        delta = procedural.get("delta", {})
        if isinstance(delta, dict) and int(delta.get("operation_count") or 0) > 0:
            rendered.append(f"delta_ops={int(delta.get('operation_count') or 0)}")
        rejected = [
            str(item.get("reason", "")).strip()
            for item in procedural.get("rejected_skills", [])
            if isinstance(item, dict) and str(item.get("reason", "")).strip()
        ]
        if rejected:
            rendered.append("rejections=" + ",".join(sorted(set(rejected))))
        content = ", ".join(rendered)
    elif item_type == "autobiography":
        content = f"{node.details.get('entry_kind', 'entry')}: {node.summary}"
    elif item_type == "commitment":
        content = f"{node.summary} [{node.status}]"
    elif item_type == "claim":
        availability_state, governance_reason_codes = _claim_node_availability(node)
        tags: list[str] = []
        currentness = str(node.details.get("currentness_status", "")).strip()
        review_state = str(node.details.get("review_state", "")).strip()
        truth_status = str(node.details.get("truth_status", "")).strip()
        if currentness and currentness != "current":
            tags.append(currentness)
        if review_state and review_state != "none":
            tags.append(f"review={review_state}")
        if truth_status and truth_status != "active":
            tags.append(f"truth={truth_status}")
        tags.extend(
            _governance_tags(
                availability_state=availability_state,
                reason_codes=governance_reason_codes,
            )
        )
        content = _format_tagged_content(tags=tags, summary=node.summary)[2:]
    return _PacketItemCandidate(
        item_id=f"node:{node.node_id}",
        item_type=item_type,
        section_key=section_key,
        title=node.summary,
        content=f"- {content}",
        temporal_kind="historical" if section_key == "recent_changes" else "current",
        availability_state=availability_state,
        governance_reason_codes=governance_reason_codes,
        provenance={
            "node_id": node.node_id,
            "backing_record_id": node.backing_record_id,
            "entry_id": (
                node.backing_record_id if item_type in {"autobiography", "scene_episode"} else None
            ),
            "source_event_ids": list(node.source_event_ids),
            "source_episode_ids": list(node.source_episode_ids),
            "supporting_claim_ids": list(node.supporting_claim_ids),
            "entry_kind": _normalized_text(dict(node.details or {}).get("entry_kind")) or None,
            "modality": _normalized_text(dict(node.details or {}).get("modality")) or None,
            "privacy_class": _normalized_text(dict(node.details or {}).get("privacy_class"))
            or None,
            "review_state": _normalized_text(dict(node.details or {}).get("review_state")) or None,
            "retention_class": _normalized_text(dict(node.details or {}).get("retention_class"))
            or None,
            "redacted_at": _normalized_text(dict(node.details or {}).get("redacted_at")) or None,
            "source_presence_scope_key": _normalized_text(
                dict(node.details or {}).get("source_presence_scope_key")
            )
            or None,
            "source_scene_entity_ids": list(
                dict(node.details or {}).get("source_scene_entity_ids", [])
            ),
            "source_scene_affordance_ids": list(
                dict(node.details or {}).get("source_scene_affordance_ids", [])
            ),
            "skill_id": (
                str(dict(node.details.get("procedural", {})).get("selected_skill_id", "")).strip()
                or None
            ),
            "support_trace_ids": list(
                dict(node.details.get("procedural", {})).get("selected_skill_support_trace_ids", [])
            ),
            "support_plan_proposal_ids": list(
                dict(node.details.get("procedural", {})).get(
                    "selected_skill_support_plan_proposal_ids", []
                )
            ),
        },
        score=score,
        decision_reason_codes=(
            "graph_node_candidate",
            f"availability_{availability_state}",
            "rendered_as_recent_change_context"
            if section_key == "recent_changes"
            else "rendered_as_unresolved_context"
            if section_key == "unresolved_state"
            else "rendered_as_current_context",
        ),
        hop_distance=hop_distance,
        is_graph_node=True,
    )


def _packet_item_from_scene_episode_node(
    *,
    node,
    task: BrainContextTask,
    current_section_key: str,
    temporal_kind: str,
    score: float,
    hop_distance: int,
) -> _PacketItemCandidate:
    availability_state, governance_reason_codes = _scene_episode_node_availability(node)
    section_key = _annotated_section_key(
        task=task,
        current_section_key=current_section_key,
        availability_state=availability_state,
    )
    details = dict(node.details or {})
    privacy_class = _normalized_text(details.get("privacy_class")) or "standard"
    review_state = _normalized_text(details.get("review_state")) or "none"
    retention_class = _normalized_text(details.get("retention_class")) or "session"
    redacted_at = _normalized_text(details.get("redacted_at"))
    tags: list[str] = []
    if privacy_class != "standard":
        tags.append(f"privacy={privacy_class}")
    if review_state != "none":
        tags.append(f"review={review_state}")
    if retention_class:
        tags.append(f"retention={retention_class}")
    if redacted_at:
        tags.append("redacted")
    tags.extend(
        _governance_tags(
            availability_state=availability_state,
            reason_codes=governance_reason_codes,
        )
    )
    return _PacketItemCandidate(
        item_id=f"node:{node.node_id}",
        item_type="scene_episode",
        section_key=section_key,
        title=node.summary,
        content=_format_tagged_content(tags=tags, summary=node.summary),
        temporal_kind=temporal_kind,
        availability_state=availability_state,
        governance_reason_codes=governance_reason_codes,
        provenance={
            "node_id": node.node_id,
            "backing_record_id": node.backing_record_id,
            "entry_id": node.backing_record_id,
            "entry_kind": _normalized_text(details.get("entry_kind")) or None,
            "modality": _normalized_text(details.get("modality")) or None,
            "privacy_class": privacy_class,
            "review_state": review_state,
            "retention_class": retention_class,
            "redacted_at": redacted_at or None,
            "source_presence_scope_key": _normalized_text(details.get("source_presence_scope_key"))
            or None,
            "source_scene_entity_ids": list(details.get("source_scene_entity_ids", [])),
            "source_scene_affordance_ids": list(details.get("source_scene_affordance_ids", [])),
            "source_event_ids": list(node.source_event_ids),
            "source_episode_ids": list(node.source_episode_ids),
            "supporting_claim_ids": list(node.supporting_claim_ids),
        },
        score=score,
        decision_reason_codes=(
            "graph_node_candidate",
            "scene_episode_candidate",
            f"scene_episode_policy_{_mode_policy(task).scene_episode_policy.value}",
            f"scene_episode_privacy_{privacy_class}",
            f"scene_episode_review_{review_state}",
            f"availability_{availability_state}",
            "rendered_as_recent_change_context"
            if section_key == "recent_changes"
            else "rendered_as_unresolved_context"
            if section_key == "unresolved_state"
            else "rendered_as_current_context",
        ),
        hop_distance=hop_distance,
        is_graph_node=True,
        forced_reason=_scene_episode_forced_reason(
            task=task,
            node=node,
            temporal_kind=temporal_kind,
        ),
    )


def _current_linkage(snapshot: BrainContextSurfaceSnapshot) -> dict[str, set[str]]:
    active_goal_ids = {
        record.goal_id
        for record in snapshot.agenda.goals
        if record.goal_id and record.status not in {"completed", "cancelled", "failed"}
    }
    active_commitment_ids = {
        record.commitment_id
        for record in (
            list(snapshot.commitment_projection.active_commitments)
            + list(snapshot.commitment_projection.blocked_commitments)
            + list(snapshot.commitment_projection.deferred_commitments)
        )
        if record.commitment_id
    }
    active_plan_proposal_ids: set[str] = set()
    for record in (
        list(snapshot.commitment_projection.active_commitments)
        + list(snapshot.commitment_projection.blocked_commitments)
        + list(snapshot.commitment_projection.deferred_commitments)
    ):
        for key in ("current_plan_proposal_id", "pending_plan_proposal_id"):
            proposal_id = str(record.details.get(key, "")).strip()
            if proposal_id:
                active_plan_proposal_ids.add(proposal_id)
    active_plan_proposal_ids.update(snapshot.active_situation_model.linked_plan_proposal_ids)
    active_skill_ids = set(snapshot.active_situation_model.linked_skill_ids)
    if snapshot.procedural_skills is not None:
        active_skill_ids.update(snapshot.procedural_skills.active_skill_ids)
        active_skill_ids.update(snapshot.procedural_skills.candidate_skill_ids)
    return {
        "goal_ids": active_goal_ids,
        "commitment_ids": active_commitment_ids,
        "plan_proposal_ids": active_plan_proposal_ids,
        "skill_ids": active_skill_ids,
    }


def _linkage_bonus(
    *,
    goal_id: str | None,
    commitment_id: str | None,
    plan_proposal_id: str | None,
    skill_id: str | None,
    active_links: dict[str, set[str]],
) -> float:
    bonus = 0.0
    if goal_id and goal_id in active_links["goal_ids"]:
        bonus += 2.0
    if commitment_id and commitment_id in active_links["commitment_ids"]:
        bonus += 2.0
    if plan_proposal_id and plan_proposal_id in active_links["plan_proposal_ids"]:
        bonus += 2.5
    if skill_id and skill_id in active_links["skill_ids"]:
        bonus += 2.0
    return bonus


def _format_tagged_content(*, tags: list[str], summary: str) -> str:
    rendered_tags = "".join(f"[{tag}]" for tag in tags if str(tag).strip())
    return f"- {rendered_tags} {summary}".rstrip()


def _active_state_section_for_record(
    *,
    task: BrainContextTask,
    unresolved: bool,
) -> str | None:
    if unresolved:
        return "unresolved_state" if _task_has_section(task, "unresolved_state") else None
    return "active_state" if _task_has_section(task, "active_state") else None


def _item_section_priority(
    *,
    section_key: str,
    task: BrainContextTask,
    temporal_mode: BrainContextTemporalMode,
) -> int:
    ordered_keys = list(_mode_policy(task).dynamic_section_keys)
    if (
        temporal_mode == BrainContextTemporalMode.HISTORICAL_FOCUS
        and "recent_changes" in ordered_keys
    ):
        ordered_keys = ["recent_changes"] + [key for key in ordered_keys if key != "recent_changes"]
    order = {key: index for index, key in enumerate(ordered_keys)}
    return order.get(section_key, 9)


def _packet_item_from_active_situation_record(
    record,
    *,
    task: BrainContextTask,
    active_links: dict[str, set[str]],
    tokens: tuple[str, ...],
) -> _PacketItemCandidate | None:
    is_prediction = record.record_kind == BrainActiveSituationRecordKind.PREDICTION_STATE.value
    unresolved = (
        record.record_kind == BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value
        or record.state != BrainActiveSituationRecordState.ACTIVE.value
    )
    section_key = _active_state_section_for_record(task=task, unresolved=unresolved)
    if section_key is None:
        return None
    item_type = (
        "uncertainty_record"
        if record.record_kind == BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value
        else "prediction"
        if is_prediction
        else "active_situation_record"
    )
    tags = [record.state, record.evidence_kind, record.record_kind]
    if is_prediction:
        prediction_kind = _normalized_text(dict(record.details or {}).get("prediction_kind"))
        prediction_role = _normalized_text(dict(record.details or {}).get("prediction_role"))
        confidence_band = _normalized_text(dict(record.details or {}).get("confidence_band"))
        if prediction_kind is not None:
            tags.append(prediction_kind)
        if prediction_role is not None:
            tags.append(prediction_role)
        if confidence_band is not None:
            tags.append(f"confidence={confidence_band}")
    if record.freshness:
        tags.append(record.freshness)
    if record.uncertainty_codes:
        tags.append("uncertainty=" + ",".join(record.uncertainty_codes[:2]))
    score = (
        _STATE_PRIORITY.get(record.state, 5.0)
        + _linkage_bonus(
            goal_id=record.goal_id,
            commitment_id=record.commitment_id,
            plan_proposal_id=record.plan_proposal_id,
            skill_id=record.skill_id,
            active_links=active_links,
        )
        + _text_match_score(
            tokens=tokens,
            parts=(
                record.summary,
                record.record_kind,
                record.goal_id,
                record.commitment_id,
                record.plan_proposal_id,
                record.skill_id,
                " ".join(record.backing_ids),
            ),
        )
    )
    provenance = {
        "record_id": record.record_id,
        "private_record_ids": list(record.private_record_ids),
        "backing_ids": list(record.backing_ids),
        "source_event_ids": list(record.source_event_ids),
        "goal_id": record.goal_id,
        "commitment_id": record.commitment_id,
        "plan_proposal_id": record.plan_proposal_id,
        "skill_id": record.skill_id,
    }
    if is_prediction:
        provenance.update(
            {
                "prediction_id": dict(record.details or {}).get("prediction_id"),
                "prediction_kind": dict(record.details or {}).get("prediction_kind"),
                "prediction_role": dict(record.details or {}).get("prediction_role"),
                "subject_kind": dict(record.details or {}).get("subject_kind"),
                "subject_id": dict(record.details or {}).get("subject_id"),
                "confidence": record.confidence,
                "confidence_band": dict(record.details or {}).get("confidence_band"),
                "risk_codes": list(dict(record.details or {}).get("risk_codes", [])),
                "supporting_event_ids": list(record.source_event_ids),
                "valid_to": record.expires_at,
                "resolution_kind": dict(record.details or {}).get("resolution_kind"),
                "resolution_summary": dict(record.details or {}).get("resolution_summary"),
                "resolution_event_ids": list(
                    dict(record.details or {}).get("resolution_event_ids", [])
                ),
            }
        )
    return _PacketItemCandidate(
        item_id=f"active_situation:{record.record_id}",
        item_type=item_type,
        section_key=section_key,
        title=record.summary,
        content=_format_tagged_content(tags=tags, summary=record.summary),
        temporal_kind="current",
        provenance=provenance,
        score=score,
        decision_reason_codes=(
            "prediction_candidate" if is_prediction else "active_state_candidate",
            "rendered_as_unresolved_context"
            if section_key == "unresolved_state"
            else "rendered_as_active_state",
        ),
        forced_reason=_prediction_forced_reason(task=task, record=record)
        if is_prediction
        else None,
    )


def _packet_item_from_private_working_memory_record(
    record,
    *,
    task: BrainContextTask,
    active_links: dict[str, set[str]],
    tokens: tuple[str, ...],
) -> _PacketItemCandidate | None:
    if record.state == BrainPrivateWorkingMemoryRecordState.RESOLVED.value:
        return None
    if (
        record.buffer_kind not in {"plan_assumption", "unresolved_uncertainty"}
        and record.state == BrainPrivateWorkingMemoryRecordState.ACTIVE.value
    ):
        return None
    unresolved = True
    section_key = _active_state_section_for_record(task=task, unresolved=unresolved)
    if section_key is None:
        return None
    item_type = (
        "uncertainty_record"
        if record.buffer_kind == "unresolved_uncertainty"
        else "private_working_memory_record"
    )
    tags = [record.state, record.evidence_kind, record.buffer_kind]
    if record.buffer_kind == "unresolved_uncertainty":
        tags.append("unresolved")
    score = (
        _STATE_PRIORITY.get(record.state, 5.5)
        + 1.5
        + _linkage_bonus(
            goal_id=record.goal_id,
            commitment_id=record.commitment_id,
            plan_proposal_id=record.plan_proposal_id,
            skill_id=record.skill_id,
            active_links=active_links,
        )
        + _text_match_score(
            tokens=tokens,
            parts=(
                record.summary,
                record.buffer_kind,
                record.goal_id,
                record.commitment_id,
                record.plan_proposal_id,
                record.skill_id,
                " ".join(record.backing_ids),
            ),
        )
    )
    return _PacketItemCandidate(
        item_id=f"private_working_memory:{record.record_id}",
        item_type=item_type,
        section_key=section_key,
        title=record.summary,
        content=_format_tagged_content(tags=tags, summary=record.summary),
        temporal_kind="current",
        provenance={
            "record_id": record.record_id,
            "backing_ids": list(record.backing_ids),
            "source_event_ids": list(record.source_event_ids),
            "goal_id": record.goal_id,
            "commitment_id": record.commitment_id,
            "plan_proposal_id": record.plan_proposal_id,
            "skill_id": record.skill_id,
        },
        score=score,
        decision_reason_codes=(
            "active_state_candidate",
            "rendered_as_unresolved_context",
        ),
    )


def _packet_item_from_scene_world_entity(
    record,
    *,
    task: BrainContextTask,
    tokens: tuple[str, ...],
) -> _PacketItemCandidate | None:
    unresolved = record.state != BrainSceneWorldRecordState.ACTIVE.value
    section_key = _active_state_section_for_record(task=task, unresolved=unresolved)
    if section_key is None:
        return None
    tags = [record.state, record.entity_kind]
    if record.zone_id:
        tags.append(f"zone={record.zone_id}")
    if record.freshness:
        tags.append(record.freshness)
    if record.contradiction_codes:
        tags.append("contradiction=" + ",".join(record.contradiction_codes[:2]))
    score = _STATE_PRIORITY.get(record.state, 5.0) + _text_match_score(
        tokens=tokens,
        parts=(
            record.summary,
            record.canonical_label,
            record.entity_kind,
            record.zone_id,
            " ".join(record.backing_ids),
        ),
    )
    return _PacketItemCandidate(
        item_id=f"scene_world_entity:{record.entity_id}",
        item_type="scene_world_entity",
        section_key=section_key,
        title=record.canonical_label,
        content=_format_tagged_content(tags=tags, summary=record.summary),
        temporal_kind="current",
        provenance={
            "entity_id": record.entity_id,
            "zone_id": record.zone_id,
            "backing_ids": list(record.backing_ids),
            "source_event_ids": list(record.source_event_ids),
        },
        score=score,
        decision_reason_codes=(
            "active_state_candidate",
            "rendered_as_unresolved_context"
            if section_key == "unresolved_state"
            else "rendered_as_active_state",
        ),
    )


def _packet_item_from_scene_world_affordance(
    record,
    *,
    task: BrainContextTask,
    tokens: tuple[str, ...],
) -> _PacketItemCandidate | None:
    unresolved = record.availability != BrainSceneWorldAffordanceAvailability.AVAILABLE.value
    section_key = _active_state_section_for_record(task=task, unresolved=unresolved)
    if section_key is None:
        return None
    tags = [record.availability, record.capability_family]
    if record.freshness:
        tags.append(record.freshness)
    if record.reason_codes:
        tags.append("reason=" + ",".join(record.reason_codes[:2]))
    score = _STATE_PRIORITY.get(record.availability, 5.0) + _text_match_score(
        tokens=tokens,
        parts=(
            record.summary,
            record.capability_family,
            record.entity_id,
            " ".join(record.backing_ids),
        ),
    )
    return _PacketItemCandidate(
        item_id=f"scene_world_affordance:{record.affordance_id}",
        item_type="scene_world_affordance",
        section_key=section_key,
        title=record.capability_family,
        content=_format_tagged_content(tags=tags, summary=record.summary),
        temporal_kind="current",
        provenance={
            "affordance_id": record.affordance_id,
            "entity_id": record.entity_id,
            "backing_ids": list(record.backing_ids),
            "source_event_ids": list(record.source_event_ids),
        },
        score=score,
        decision_reason_codes=(
            "active_state_candidate",
            "rendered_as_unresolved_context"
            if section_key == "unresolved_state"
            else "rendered_as_active_state",
        ),
    )


def _scene_world_uncertainty_item(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    task: BrainContextTask,
) -> _PacketItemCandidate | None:
    projection = snapshot.scene_world_state
    if projection.degraded_mode == "healthy" and not projection.contradiction_counts:
        return None
    section_key = _active_state_section_for_record(task=task, unresolved=True)
    if section_key is None:
        return None
    reasons = list(projection.degraded_reason_codes[:3])
    contradiction_codes = sorted(projection.contradiction_counts)[:3]
    tags = ["unresolved", "scene_world_state", projection.degraded_mode]
    if reasons:
        tags.append("reason=" + ",".join(reasons))
    if contradiction_codes:
        tags.append("contradiction=" + ",".join(contradiction_codes))
    summary = "Scene perception is degraded or contradicted; treat world-state evidence cautiously."
    return _PacketItemCandidate(
        item_id=f"scene_world_uncertainty:{projection.scope_id}",
        item_type="uncertainty_record",
        section_key=section_key,
        title="Scene world-state uncertainty",
        content=_format_tagged_content(tags=tags, summary=summary),
        temporal_kind="current",
        provenance={
            "record_id": f"scene_world_uncertainty:{projection.scope_id}",
            "backing_ids": list(projection.active_entity_ids[:3]),
            "source_event_ids": [],
        },
        score=8.5,
        decision_reason_codes=("active_state_candidate", "rendered_as_unresolved_context"),
    )


def _select_packet_items(
    *,
    item_candidates: list[_PacketItemCandidate],
    dynamic_token_budget: int,
    max_dossiers: int,
    max_current_items: int,
    max_history_items: int,
    task: BrainContextTask,
) -> tuple[
    tuple[BrainActiveContextPacketItemRecord, ...], tuple[BrainActiveContextPacketItemRecord, ...]
]:
    selected: list[BrainActiveContextPacketItemRecord] = []
    dropped: list[BrainActiveContextPacketItemRecord] = []
    used_tokens = 0
    counts = {
        "dossier": 0,
        "current": 0,
        "historical": 0,
        "scene_episode": 0,
        "prediction": 0,
        "prediction_active": 0,
        "prediction_unresolved": 0,
    }
    section_seen: set[str] = set()
    scene_episode_cap = max(_mode_policy(task).scene_episode_cap, 0)
    prediction_cap = max(_mode_policy(task).prediction_cap, 0)

    for candidate in item_candidates:
        section_overhead = (
            approximate_token_count(f"## {_ACTIVE_SECTION_TITLES[candidate.section_key]}")
            if candidate.section_key not in section_seen
            else 0
        )
        reason = candidate.forced_reason or "selected"
        if reason != "selected":
            pass
        elif candidate.is_dossier and counts["dossier"] >= max_dossiers:
            reason = "type_cap_exceeded"
        elif (
            candidate.item_type == "scene_episode"
            and scene_episode_cap > 0
            and counts["scene_episode"] >= scene_episode_cap
        ):
            reason = "scene_episode_cap_exceeded"
        elif (
            candidate.item_type == "prediction"
            and _mode_policy(task).prediction_policy == BrainContextPredictionPolicy.CONSERVATIVE
            and candidate.section_key == "unresolved_state"
            and counts["prediction_unresolved"] >= 1
        ):
            reason = "prediction_role_cap_exceeded"
        elif (
            candidate.item_type == "prediction"
            and _mode_policy(task).prediction_policy == BrainContextPredictionPolicy.CONSERVATIVE
            and candidate.section_key == "active_state"
            and counts["prediction_active"] >= 1
        ):
            reason = "prediction_role_cap_exceeded"
        elif (
            candidate.item_type == "prediction"
            and prediction_cap > 0
            and counts["prediction"] >= prediction_cap
        ):
            reason = "prediction_cap_exceeded"
        elif (
            candidate.item_type != "scene_episode"
            and candidate.temporal_kind == "current"
            and counts["current"] >= max_current_items
        ):
            reason = "type_cap_exceeded"
        elif (
            candidate.item_type != "scene_episode"
            and candidate.temporal_kind == "historical"
            and counts["historical"] >= max_history_items
        ):
            reason = "type_cap_exceeded"
        elif used_tokens + candidate.estimated_tokens + section_overhead > dynamic_token_budget:
            reason = "dynamic_budget_exceeded"

        record = BrainActiveContextPacketItemRecord(
            item_id=candidate.item_id,
            item_type=candidate.item_type,
            section_key=candidate.section_key,
            title=candidate.title,
            content=candidate.content,
            estimated_tokens=candidate.estimated_tokens,
            temporal_kind=candidate.temporal_kind,
            availability_state=candidate.availability_state,
            governance_reason_codes=tuple(candidate.governance_reason_codes),
            decision_reason_codes=tuple(candidate.decision_reason_codes)
            + _decision_reason_codes_for_packet_reason(reason),
            provenance=dict(candidate.provenance),
            hop_distance=candidate.hop_distance,
            selected=reason == "selected",
            reason=reason,
        )
        if reason == "selected":
            selected.append(record)
            used_tokens += candidate.estimated_tokens + section_overhead
            section_seen.add(candidate.section_key)
            if candidate.is_dossier:
                counts["dossier"] += 1
            if candidate.item_type == "scene_episode":
                counts["scene_episode"] += 1
            elif candidate.item_type == "prediction":
                counts["prediction"] += 1
                if candidate.section_key == "unresolved_state":
                    counts["prediction_unresolved"] += 1
                else:
                    counts["prediction_active"] += 1
                counts["current"] += 1
            elif candidate.temporal_kind == "current":
                counts["current"] += 1
            else:
                counts["historical"] += 1
        else:
            dropped.append(record)

    return tuple(selected), tuple(dropped)


def _decision_reason_codes_for_packet_reason(reason: str) -> tuple[str, ...]:
    return {
        "selected": ("item_selected",),
        "governance_suppressed": ("dropped_for_governance",),
        "scene_episode_task_ineligible": (
            "dropped_for_scene_episode_policy",
            "scene_episode_task_ineligible",
        ),
        "scene_episode_cap_exceeded": ("dropped_for_scene_episode_cap",),
        "scene_episode_not_current": (
            "dropped_for_scene_episode_policy",
            "scene_episode_requires_current",
        ),
        "scene_episode_privacy_gated": (
            "dropped_for_scene_episode_policy",
            "scene_episode_requires_standard_privacy",
        ),
        "scene_episode_review_gated": (
            "dropped_for_scene_episode_policy",
            "scene_episode_requires_no_review",
        ),
        "scene_episode_redacted": (
            "dropped_for_scene_episode_policy",
            "scene_episode_redacted",
        ),
        "prediction_task_ineligible": (
            "dropped_for_prediction_policy",
            "prediction_task_ineligible",
        ),
        "prediction_low_confidence": (
            "dropped_for_prediction_policy",
            "prediction_requires_medium_or_high_confidence",
        ),
        "prediction_resolution_ineligible": (
            "dropped_for_prediction_policy",
            "prediction_resolution_requires_allowed_task",
        ),
        "prediction_cap_exceeded": ("dropped_for_prediction_cap",),
        "prediction_role_cap_exceeded": ("dropped_for_prediction_role_cap",),
        "type_cap_exceeded": ("dropped_for_type_cap",),
        "dynamic_budget_exceeded": ("dropped_for_dynamic_budget",),
    }.get(reason, ("item_decision_recorded",))


def _build_active_section_decisions(
    *,
    task: BrainContextTask,
    selected_sections: tuple[BrainSelectedSection, ...],
    selected_items: tuple[BrainActiveContextPacketItemRecord, ...],
    dropped_items: tuple[BrainActiveContextPacketItemRecord, ...],
) -> tuple[BrainActiveContextSectionDecision, ...]:
    selected_by_section = {section.key: section for section in selected_sections}
    selected_item_counts: dict[str, int] = {}
    dropped_item_counts: dict[str, int] = {}
    for item in selected_items:
        selected_item_counts[item.section_key] = selected_item_counts.get(item.section_key, 0) + 1
    for item in dropped_items:
        dropped_item_counts[item.section_key] = dropped_item_counts.get(item.section_key, 0) + 1
    decisions: list[BrainActiveContextSectionDecision] = []
    for key in _mode_policy(task).dynamic_section_keys:
        section = selected_by_section.get(key)
        selected_count = selected_item_counts.get(key, 0)
        dropped_count = dropped_item_counts.get(key, 0)
        if section is None:
            decisions.append(
                BrainActiveContextSectionDecision(
                    section_key=key,
                    title=_section_title(key),
                    selected=False,
                    estimated_tokens=0,
                    selected_item_count=0,
                    dropped_item_count=dropped_count,
                    reason="empty",
                    decision_reason_codes=("dynamic_section_empty",),
                )
            )
            continue
        decisions.append(
            BrainActiveContextSectionDecision(
                section_key=key,
                title=section.title,
                selected=True,
                estimated_tokens=section.estimated_tokens,
                selected_item_count=selected_count,
                dropped_item_count=dropped_count,
                reason="selected",
                decision_reason_codes=("dynamic_section_selected",),
            )
        )
    return tuple(decisions)


def _render_dynamic_sections(
    *,
    task: BrainContextTask,
    items: tuple[BrainActiveContextPacketItemRecord, ...],
) -> tuple[BrainSelectedSection, ...]:
    grouped: dict[str, list[BrainActiveContextPacketItemRecord]] = {}
    for item in items:
        grouped.setdefault(item.section_key, []).append(item)
    sections: list[BrainSelectedSection] = []
    for key in _mode_policy(task).dynamic_section_keys:
        group = grouped.get(key, [])
        if not group:
            continue
        content = "\n".join(item.content for item in group)
        sections.append(
            BrainSelectedSection(
                key=key,
                title=_ACTIVE_SECTION_TITLES[key],
                content=content,
                estimated_tokens=approximate_token_count(content),
                source="active_context",
            )
        )
    return tuple(sections)


def _anchor_sort_key(candidate: BrainActiveContextAnchorCandidate) -> tuple[float, str, str]:
    type_order = {
        "commitment": 0,
        "procedural_skill": 1,
        "plan_proposal": 2,
        "claim": 3,
        "scene_episode": 4,
        "dossier": 5,
    }
    return (-candidate.score, str(type_order.get(candidate.anchor_type, 9)), candidate.anchor_id)


def _item_sort_key(
    candidate: _PacketItemCandidate,
    *,
    task: BrainContextTask,
    temporal_mode: BrainContextTemporalMode,
) -> tuple[int, int, float, str]:
    prediction_priority = 1
    if (
        task == BrainContextTask.PLANNING
        and candidate.item_type == "prediction"
        and candidate.temporal_kind == "current"
        and candidate.forced_reason is None
    ):
        prediction_priority = 0
    scene_episode_priority = 1
    if (
        task == BrainContextTask.PLANNING
        and candidate.item_type == "scene_episode"
        and candidate.temporal_kind == "current"
        and candidate.forced_reason is None
    ):
        # Reserve ordering space for one policy-clean current scene episode so the
        # conservative planning surface is not starved by lower-signal unresolved items.
        scene_episode_priority = 0
    return (
        prediction_priority,
        scene_episode_priority,
        _item_section_priority(
            section_key=candidate.section_key,
            task=task,
            temporal_mode=temporal_mode,
        ),
        -candidate.score,
        candidate.item_id,
    )


def _static_sections_from_surface(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    base_prompt: str,
    task: BrainContextTask,
    language: Language,
    enable_persona_expression: bool,
    persona_modality: BrainPersonaModality | str,
    persona_seriousness: str,
    behavior_controls: BrainBehaviorControlProfile | None,
    teaching_knowledge_context: str | None,
    teaching_knowledge_reason_codes: tuple[str, ...] | None,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    sections = {
        "policy": "\n".join(
            part.strip()
            for part in (base_prompt.strip(), snapshot.agent_blocks.get("policy", ""))
            if part.strip()
        ),
        "identity": snapshot.agent_blocks.get("identity", ""),
        "style": snapshot.agent_blocks.get("style", ""),
        "capabilities": snapshot.capability_manifest,
        "internal_capabilities": snapshot.internal_capability_manifest,
    }
    reason_codes: dict[str, tuple[str, ...]] = {}
    if task == BrainContextTask.REPLY:
        persona_content, persona_reason_codes = _persona_expression_static_section(
            snapshot=snapshot,
            task=task,
            language=language,
            enable_persona_expression=enable_persona_expression,
            persona_modality=persona_modality,
            persona_seriousness=persona_seriousness,
            behavior_controls=behavior_controls,
        )
        sections["persona_expression"] = persona_content
        reason_codes["persona_expression"] = persona_reason_codes
        teaching_content, teaching_reason_codes = _teaching_knowledge_static_section(
            teaching_knowledge_context=teaching_knowledge_context,
            teaching_knowledge_reason_codes=teaching_knowledge_reason_codes,
        )
        sections["teaching_knowledge"] = teaching_content
        reason_codes["teaching_knowledge"] = teaching_reason_codes
    return sections, reason_codes


def _teaching_knowledge_static_section(
    *,
    teaching_knowledge_context: str | None,
    teaching_knowledge_reason_codes: tuple[str, ...] | None = None,
) -> tuple[str, tuple[str, ...]]:
    normalized = (teaching_knowledge_context or "").strip()
    if not normalized:
        return "", teaching_knowledge_reason_codes or (
            "teaching_knowledge_excluded",
            "teaching_knowledge_empty",
        )
    return normalized, teaching_knowledge_reason_codes or (
        "teaching_knowledge_candidate",
        "teaching_knowledge_selected",
    )


def _persona_expression_static_section(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    task: BrainContextTask,
    language: Language,
    enable_persona_expression: bool,
    persona_modality: BrainPersonaModality | str,
    persona_seriousness: str,
    behavior_controls: BrainBehaviorControlProfile | None,
) -> tuple[str, tuple[str, ...]]:
    expression_frame, reason_codes, _status = _compile_persona_expression_frame_from_surface(
        snapshot=snapshot,
        task=task,
        language=language,
        enable_persona_expression=enable_persona_expression,
        persona_modality=persona_modality,
        persona_seriousness=persona_seriousness,
        behavior_controls=behavior_controls,
    )
    if expression_frame is None:
        return "", reason_codes

    rendered = render_persona_expression_summary(expression_frame).strip()
    if not rendered:
        return "", ("persona_expression_excluded", "persona_expression_empty")

    return rendered, reason_codes


def _compile_persona_expression_frame_from_surface(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    task: BrainContextTask,
    language: Language,
    enable_persona_expression: bool,
    persona_modality: BrainPersonaModality | str,
    persona_seriousness: str,
    behavior_controls: BrainBehaviorControlProfile | None = None,
) -> tuple[BrainExpressionFrame | None, tuple[str, ...], dict[str, str]]:
    status = {
        "persona_expression": "excluded",
        "persona_defaults": "unknown",
        "relationship_style": "unknown",
        "teaching_profile": "unknown",
    }
    if not enable_persona_expression:
        status["persona_expression"] = "disabled"
        return None, ("persona_expression_excluded", "persona_expression_disabled"), status

    missing_blocks = [
        block_name
        for block_name in _PERSONA_DEFAULT_BLOCKS
        if not str(snapshot.agent_blocks.get(block_name, "")).strip()
    ]
    if missing_blocks:
        status["persona_defaults"] = "missing"
        return None, ("persona_expression_excluded", "persona_defaults_missing"), status

    try:
        task_mode = _persona_task_mode_for_context(task)
        persona_frame = compile_persona_frame(
            agent_blocks=snapshot.agent_blocks,
            task_mode=task_mode,
            modality=persona_modality,
        )
    except (ValidationError, ValueError):
        status["persona_defaults"] = "invalid"
        return None, ("persona_expression_excluded", "persona_defaults_invalid"), status

    status["persona_defaults"] = "valid"
    relationship_state, relationship_reason_codes = _relationship_style_state_from_surface(snapshot)
    teaching_profile, teaching_reason_codes = _teaching_profile_state_from_surface(snapshot)
    status["relationship_style"] = (
        "invalid"
        if "relationship_style_invalid" in relationship_reason_codes
        else "available"
        if relationship_state is not None
        else "missing"
    )
    status["teaching_profile"] = (
        "invalid"
        if "teaching_profile_invalid" in teaching_reason_codes
        else "available"
        if teaching_profile is not None
        else "missing"
    )
    try:
        expression_frame = compile_expression_frame(
            persona_frame=persona_frame,
            relationship_style=relationship_state,
            teaching_profile=teaching_profile,
            behavior_controls=behavior_controls,
            task_mode=task_mode,
            modality=persona_modality,
            language=language,
            seriousness=persona_seriousness,
            recent_misfires=tuple(snapshot.relationship_state.known_misfires),
        )
    except (ValidationError, ValueError):
        status["persona_expression"] = "invalid"
        return None, ("persona_expression_excluded", "persona_expression_invalid"), status

    status["persona_expression"] = "available"
    return (
        expression_frame,
        (
            "persona_expression_candidate",
            "persona_expression_compiled",
            *expression_frame.reason_codes,
            *relationship_reason_codes,
            *teaching_reason_codes,
        ),
        status,
    )


def _persona_task_mode_for_context(task: BrainContextTask) -> BrainPersonaTaskMode:
    if task == BrainContextTask.PLANNING:
        return BrainPersonaTaskMode.PLANNING
    if task == BrainContextTask.REFLECTION:
        return BrainPersonaTaskMode.REFLECTION
    if task in {BrainContextTask.OPERATOR_AUDIT, BrainContextTask.GOVERNANCE_REVIEW}:
        return BrainPersonaTaskMode.AUDIT
    return BrainPersonaTaskMode.REPLY


def _language_value(language: Language | str) -> str:
    if isinstance(language, Language):
        return language.value.lower()
    return str(language or "en").strip().lower() or "en"


def _relationship_style_state_from_surface(
    snapshot: BrainContextSurfaceSnapshot,
) -> tuple[RelationshipStyleStateSpec | None, tuple[str, ...]]:
    record = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.RELATIONSHIP_STYLE.value)
    if record is None:
        return None, ()
    try:
        return RelationshipStyleStateSpec.model_validate(record.content), ()
    except ValidationError:
        return None, ("relationship_style_invalid",)


def _teaching_profile_state_from_surface(
    snapshot: BrainContextSurfaceSnapshot,
) -> tuple[TeachingProfileStateSpec | None, tuple[str, ...]]:
    record = snapshot.core_blocks.get(BrainCoreMemoryBlockKind.TEACHING_PROFILE.value)
    if record is None:
        return None, ()
    try:
        return TeachingProfileStateSpec.model_validate(record.content), ()
    except ValidationError:
        return None, ("teaching_profile_invalid",)


def _section_title(key: str) -> str:
    if key in _ACTIVE_SECTION_TITLES:
        return _ACTIVE_SECTION_TITLES[key]
    return {
        "policy": "Policy",
        "identity": "Identity",
        "style": "Style",
        "persona_expression": "Persona Expression",
        "teaching_knowledge": "Teaching Knowledge",
        "capabilities": "Capabilities",
        "internal_capabilities": "Internal Capabilities",
        "presence": "Presence",
        "scene": "Scene",
        "engagement": "Engagement",
        "working_context": "Working Context",
        "private_working_memory": "Private Working Memory",
        "agenda": "Agenda",
        "heartbeat": "Heartbeat",
        "relationship_state": "Relationship State",
        "core_blocks": "Core Blocks",
        "commitment_projection": "Commitment Ledger",
    }.get(key, key.replace("_", " ").title())


def _commitment_by_id(snapshot: BrainContextSurfaceSnapshot, commitment_id: str):
    for record in (
        list(snapshot.commitment_projection.active_commitments)
        + list(snapshot.commitment_projection.blocked_commitments)
        + list(snapshot.commitment_projection.deferred_commitments)
    ):
        if record.commitment_id == commitment_id:
            return record
    return None


__all__ = [
    "BrainActiveContextAnchorCandidate",
    "BrainActiveContextCompiler",
    "BrainActiveContextExpansionRecord",
    "BrainActiveContextPacket",
    "BrainActiveContextPacketItemRecord",
    "BrainActiveContextSectionDecision",
    "BrainActiveContextTrace",
    "BrainCompiledContextPacket",
    "BrainContextCompiler",
    "BrainContextTemporalMode",
    "compile_context_packet_from_surface",
]

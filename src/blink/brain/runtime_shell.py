"""Narrow shell facade around Blink cognition runtime surfaces."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from blink.brain.actions import build_brain_capability_registry
from blink.brain.autonomy_digest import build_autonomy_digest
from blink.brain.context import BrainContextTask, compile_context_packet_from_surface
from blink.brain.context.policy import all_brain_context_tasks
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, BrainContextSurfaceSnapshot
from blink.brain.counterfactual_rehearsal_digest import build_counterfactual_rehearsal_digest
from blink.brain.embodied_executive_digest import build_embodied_executive_digest
from blink.brain.evals.adapter_promotion import build_adapter_governance_inspection
from blink.brain.evals.continuity_metrics import (
    BrainContinuityAuditExporter,
    BrainContinuityAuditReport,
)
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.executive_policy_audit import build_executive_policy_audit
from blink.brain.identity import base_brain_system_prompt
from blink.brain.memory_v2 import (
    BrainReflectionEngine,
    BrainReflectionRunResult,
    build_multimodal_autobiography_digest,
    parse_multimodal_autobiography_record,
)
from blink.brain.memory_v2.skill_evidence import build_skill_evidence_inspection
from blink.brain.persona import (
    BrainPersonaModality,
    runtime_expression_state_from_frame,
    unavailable_runtime_expression_state,
)
from blink.brain.planning_digest import build_planning_digest
from blink.brain.practice_director import build_practice_inspection
from blink.brain.procedural_skill_governance_report import (
    build_procedural_skill_governance_report,
)
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentRecord,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.reevaluation_digest import build_reevaluation_digest
from blink.brain.runtime_shell_digest import build_runtime_shell_digest
from blink.brain.session import BrainSessionIds, resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.brain.wake_digest import build_wake_digest
from blink.brain.world_model_digest import build_world_model_digest
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.brain.runtime import BrainRuntime


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        return _json_ready(payload)
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolved_session_ids(
    *,
    runtime_kind: str,
    client_id: str | None,
    user_id: str | None,
    thread_id: str | None,
) -> BrainSessionIds:
    session_ids = resolve_brain_session_ids(runtime_kind=runtime_kind, client_id=client_id)
    return BrainSessionIds(
        agent_id=session_ids.agent_id,
        user_id=user_id or session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=thread_id or session_ids.thread_id,
    )


def _persona_modality_for_runtime_kind(runtime_kind: str) -> BrainPersonaModality:
    if runtime_kind == "voice":
        return BrainPersonaModality.VOICE
    if runtime_kind == "browser":
        return BrainPersonaModality.BROWSER
    return BrainPersonaModality.TEXT


@dataclass(frozen=True)
class BrainRuntimeShellSnapshot:
    """Inspectable runtime-shell snapshot."""

    session_ids: BrainSessionIds
    runtime_kind: str
    presence_scope_key: str
    generated_at: str
    surface: BrainContextSurfaceSnapshot
    expression_state: dict[str, Any]
    recent_scene_episodes: tuple[dict[str, Any], ...]
    multimodal_autobiography_digest: dict[str, Any]
    latest_source_presence_scope_key: str | None
    predictive_digest: dict[str, Any]
    recent_active_predictions: tuple[dict[str, Any], ...]
    recent_prediction_resolutions: tuple[dict[str, Any], ...]
    embodied_digest: dict[str, Any]
    current_embodied_intent: dict[str, Any]
    recent_embodied_execution_traces: tuple[dict[str, Any], ...]
    recent_embodied_recoveries: tuple[dict[str, Any], ...]
    recent_low_level_embodied_actions: tuple[dict[str, Any], ...]
    practice_digest: dict[str, Any]
    recent_practice_plans: tuple[dict[str, Any], ...]
    skill_evidence_digest: dict[str, Any]
    recent_skill_governance_proposals: tuple[dict[str, Any], ...]
    adapter_governance_digest: dict[str, Any]
    recent_adapter_cards: tuple[dict[str, Any], ...]
    recent_adapter_promotion_decisions: tuple[dict[str, Any], ...]
    sim_to_real_digest: dict[str, Any]
    rehearsal_digest: dict[str, Any]
    recent_rehearsals: tuple[dict[str, Any], ...]
    recent_rehearsal_comparisons: tuple[dict[str, Any], ...]
    autonomy_digest: dict[str, Any]
    reevaluation_digest: dict[str, Any]
    wake_digest: dict[str, Any]
    planning_digest: dict[str, Any]
    executive_policy_audit: dict[str, Any]
    background_maintenance_running: bool | None = None
    next_reevaluation_wake: dict[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the shell snapshot into JSON-friendly data."""
        return _json_ready(self)


@dataclass(frozen=True)
class BrainRuntimePacketInspection:
    """Inspectable task-specific packet compilation result."""

    task: BrainContextTask
    query_text: str
    surface: BrainContextSurfaceSnapshot
    compiled_packet: Any
    selected_context: Any
    packet_trace: dict[str, Any] | None
    packet_digest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the packet inspection into JSON-friendly data."""
        return _json_ready(self)


@dataclass(frozen=True)
class BrainRuntimePendingWakeInspection:
    """Inspectable waiting-commitment summary for the shell."""

    session_ids: BrainSessionIds
    generated_at: str
    current_waiting_commitments: list[dict[str, Any]]
    current_wait_kind_counts: dict[str, int]
    recent_wait_rows: list[dict[str, Any]]
    wake_digest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the wake inspection into JSON-friendly data."""
        return _json_ready(self)


@dataclass(frozen=True)
class BrainRuntimeCommitmentControlResult:
    """Result from one bounded shell commitment control."""

    control_kind: str
    commitment_id: str
    applied: bool
    status_before: str | None
    status_after: str | None
    reason_summary: str | None = None
    event_id: str | None = None
    source: str = "runtime_shell"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the control result into JSON-friendly data."""
        return _json_ready(self)


class BrainRuntimeShell:
    """Thin shell facade above the cognition core."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver: Callable[[], BrainSessionIds],
        runtime_kind: str,
        presence_scope_key: str,
        language: Language,
        surface_builder: BrainContextSurfaceBuilder,
        compiler: Any,
        executive: BrainExecutive,
        reflection_engine: BrainReflectionEngine,
        base_prompt: str,
        persona_modality: BrainPersonaModality | str | None = None,
        owns_store: bool = False,
        maintenance_running_resolver: Callable[[], bool | None] | None = None,
        next_reevaluation_wake_resolver: Callable[[], tuple[str, Any] | None] | None = None,
    ):
        """Bind the shell to one store-backed cognition runtime surface."""
        self._store = store
        self._session_resolver = session_resolver
        self._runtime_kind = runtime_kind
        self._presence_scope_key = presence_scope_key
        self._language = language
        self._surface_builder = surface_builder
        self._compiler = compiler
        self._executive = executive
        self._reflection_engine = reflection_engine
        self._base_prompt = base_prompt
        self._persona_modality = persona_modality or _persona_modality_for_runtime_kind(
            runtime_kind
        )
        self._owns_store = owns_store
        self._maintenance_running_resolver = maintenance_running_resolver
        self._next_reevaluation_wake_resolver = next_reevaluation_wake_resolver

    @classmethod
    def from_runtime(cls, runtime: "BrainRuntime") -> "BrainRuntimeShell":
        """Bind one shell facade to a live runtime instance."""
        return cls(
            store=runtime.store,
            session_resolver=runtime.session_resolver,
            runtime_kind=runtime.runtime_kind,
            presence_scope_key=runtime.presence_scope_key,
            language=runtime.language,
            surface_builder=runtime.compiler.context_surface_builder,
            compiler=runtime.compiler,
            executive=runtime.executive,
            reflection_engine=runtime.reflection_engine,
            base_prompt=runtime.static_system_prompt,
            persona_modality=runtime.persona_modality,
            owns_store=False,
            maintenance_running_resolver=lambda: runtime.reflection_scheduler.is_running,
            next_reevaluation_wake_resolver=runtime._next_reevaluation_wake,
        )

    @classmethod
    def open(
        cls,
        *,
        brain_db_path: str | Path | None = None,
        runtime_kind: str = "browser",
        client_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        presence_scope_key: str | None = None,
        language: Language = Language.EN,
        base_prompt: str | None = None,
    ) -> "BrainRuntimeShell":
        """Open one offline operator shell against an existing SQLite store."""
        resolved_session_ids = _resolved_session_ids(
            runtime_kind=runtime_kind,
            client_id=client_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        store = BrainStore(path=Path(brain_db_path) if brain_db_path else None)
        capability_registry = build_brain_capability_registry(language=language)
        resolved_presence_scope_key = presence_scope_key or f"{runtime_kind}:presence"
        surface_builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            presence_scope_key=resolved_presence_scope_key,
            language=language,
            capability_registry=capability_registry,
        )
        from blink.brain.context import BrainContextCompiler

        compiler = BrainContextCompiler(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            language=language,
            base_prompt=base_prompt or base_brain_system_prompt(language),
            context_surface_builder=surface_builder,
        )
        executive = BrainExecutive(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            capability_registry=capability_registry,
            presence_scope_key=resolved_presence_scope_key,
            context_surface_builder=compiler.context_surface_builder,
            context_selector=compiler.context_selector,
            planning_callback=None,
        )
        return cls(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            runtime_kind=runtime_kind,
            presence_scope_key=resolved_presence_scope_key,
            language=language,
            surface_builder=surface_builder,
            compiler=compiler,
            executive=executive,
            reflection_engine=BrainReflectionEngine(store=store),
            base_prompt=base_prompt or base_brain_system_prompt(language),
            persona_modality=_persona_modality_for_runtime_kind(runtime_kind),
            owns_store=True,
        )

    def close(self):
        """Close any shell-owned store resources."""
        if self._owns_store:
            self._store.close()

    def _current_expression_state(self):
        try:
            frame, reason_codes, status = self._compiler.compile_persona_expression_frame(
                latest_user_text="",
                task=BrainContextTask.REPLY,
                persona_modality=self._persona_modality,
            )
        except Exception as exc:
            return unavailable_runtime_expression_state(
                modality=self._persona_modality,
                reason_codes=(f"runtime_expression_error:{type(exc).__name__}",),
                memory_persona_section_status={
                    "persona_expression": "error",
                    "persona_defaults": "unknown",
                },
            )
        return runtime_expression_state_from_frame(
            frame,
            modality=self._persona_modality,
            reason_codes=reason_codes,
            memory_persona_section_status=status,
        )

    def snapshot(self) -> BrainRuntimeShellSnapshot:
        """Return one non-mutating runtime snapshot."""
        session_ids = self._session_resolver()
        surface = self._surface_builder.build(
            latest_user_text="",
            task=BrainContextTask.REPLY,
            include_historical_claims=True,
        )
        recent_events = self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=192,
        )
        recent_scene_episodes = tuple(
            record.as_dict()
            for entry in surface.scene_episodes[:4]
            if (record := parse_multimodal_autobiography_record(entry)) is not None
        )
        multimodal_autobiography_digest = build_multimodal_autobiography_digest(
            surface.scene_episodes
        )
        predictive_digest = build_world_model_digest(
            predictive_world_model=surface.predictive_world_model.as_dict(),
        )
        embodied_projection = self._store.build_embodied_executive_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self._presence_scope_key,
        )
        recent_action_events = [
            {
                "action_id": record.action_id,
                "source": record.source,
                "accepted": record.accepted,
                "preview_only": record.preview_only,
                "summary": record.summary,
                "metadata": dict(record.metadata),
                "created_at": record.created_at,
            }
            for record in self._store.recent_action_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=8,
            )
        ]
        embodied_digest = build_embodied_executive_digest(
            embodied_executive=embodied_projection.as_dict(),
            recent_action_events=recent_action_events,
        )
        practice_projection = self._store.build_practice_director_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self._presence_scope_key,
        )
        skill_evidence_projection = self._store.build_skill_evidence_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        )
        skill_governance_projection = self._store.build_skill_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        )
        adapter_governance_projection = self._store.build_adapter_governance_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
        )
        practice_digest = build_practice_inspection(practice_projection, recent_limit=6)
        skill_evidence_digest = build_skill_evidence_inspection(
            skill_evidence_ledger=skill_evidence_projection,
            skill_governance=skill_governance_projection,
            recent_limit=6,
        )
        adapter_governance_digest = build_adapter_governance_inspection(
            adapter_governance=adapter_governance_projection,
            recent_limit=6,
        )
        sim_to_real_digest = build_sim_to_real_digest(
            adapter_governance=adapter_governance_projection,
            recent_limit=6,
        )
        counterfactual_rehearsal = self._store.build_counterfactual_rehearsal_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=self._presence_scope_key,
        )
        rehearsal_digest = build_counterfactual_rehearsal_digest(
            counterfactual_rehearsal=counterfactual_rehearsal.as_dict(),
        )
        latest_source_presence_scope_key = (
            next(
                (
                    value
                    for record in recent_scene_episodes
                    if (value := _optional_text(record.get("source_presence_scope_key")))
                    is not None
                ),
                None,
            )
            or self._presence_scope_key
        )
        planning_digest = build_planning_digest(
            agenda=surface.agenda,
            commitment_projection=surface.commitment_projection,
            recent_events=recent_events,
        )
        procedural_skill_governance_report = build_procedural_skill_governance_report(
            procedural_skills=surface.procedural_skills.as_dict()
            if surface.procedural_skills is not None
            else {},
            procedural_traces={},
            planning_digest=planning_digest,
        )
        autonomy_digest = build_autonomy_digest(
            autonomy_ledger=surface.autonomy_ledger,
            agenda=surface.agenda,
        )
        reevaluation_digest = build_reevaluation_digest(
            autonomy_ledger=surface.autonomy_ledger,
            recent_events=recent_events,
        )
        wake_digest = build_wake_digest(
            commitment_projection=surface.commitment_projection,
            recent_events=recent_events,
        )
        next_wake = (
            self._next_reevaluation_wake_resolver()
            if self._next_reevaluation_wake_resolver is not None
            else None
        )
        return BrainRuntimeShellSnapshot(
            session_ids=session_ids,
            runtime_kind=self._runtime_kind,
            presence_scope_key=self._presence_scope_key,
            generated_at=surface.generated_at,
            surface=surface,
            expression_state=self._current_expression_state().as_dict(),
            recent_scene_episodes=recent_scene_episodes,
            multimodal_autobiography_digest=multimodal_autobiography_digest,
            latest_source_presence_scope_key=latest_source_presence_scope_key,
            predictive_digest=predictive_digest,
            recent_active_predictions=tuple(
                {
                    "prediction_id": record.prediction_id,
                    "prediction_kind": record.prediction_kind,
                    "subject_id": record.subject_id,
                    "summary": record.summary,
                    "confidence_band": record.confidence_band,
                    "risk_codes": list(record.risk_codes),
                    "valid_to": record.valid_to,
                }
                for record in surface.predictive_world_model.active_predictions[:4]
            ),
            recent_prediction_resolutions=tuple(
                {
                    "prediction_id": record.prediction_id,
                    "prediction_kind": record.prediction_kind,
                    "subject_id": record.subject_id,
                    "resolution_kind": record.resolution_kind,
                    "resolution_summary": record.resolution_summary,
                    "resolved_at": record.resolved_at,
                }
                for record in surface.predictive_world_model.recent_resolutions[:4]
            ),
            embodied_digest=embodied_digest,
            current_embodied_intent=dict(embodied_digest.get("current_intent") or {}),
            recent_embodied_execution_traces=tuple(
                dict(record) for record in embodied_digest.get("recent_execution_traces", [])[:4]
            ),
            recent_embodied_recoveries=tuple(
                dict(record) for record in embodied_digest.get("recent_recoveries", [])[:4]
            ),
            recent_low_level_embodied_actions=tuple(
                dict(record)
                for record in embodied_digest.get("recent_low_level_embodied_actions", [])[:4]
            ),
            practice_digest=practice_digest,
            recent_practice_plans=tuple(
                {
                    "plan_id": record.get("plan_id"),
                    "dataset_manifest_id": record.get("dataset_manifest_id"),
                    "target_count": record.get("target_count"),
                    "reason_code_counts": dict(record.get("reason_code_counts") or {}),
                    "summary": record.get("summary"),
                    "artifact_paths": dict(record.get("artifact_paths") or {}),
                    "updated_at": record.get("updated_at"),
                }
                for record in practice_digest.get("recent_plans", [])[:4]
            ),
            skill_evidence_digest=skill_evidence_digest,
            recent_skill_governance_proposals=tuple(
                {
                    "proposal_id": record.get("proposal_id"),
                    "skill_id": record.get("skill_id"),
                    "status": record.get("status"),
                    "kind": record.get("kind"),
                    "reason_codes": list(record.get("reason_codes") or []),
                    "updated_at": record.get("updated_at"),
                }
                for record in skill_evidence_digest.get("recent_governance_proposals", [])[:4]
            ),
            adapter_governance_digest=adapter_governance_digest,
            recent_adapter_cards=tuple(
                {
                    "card_id": record.get("card_id"),
                    "adapter_family": record.get("adapter_family"),
                    "backend_id": record.get("backend_id"),
                    "backend_version": record.get("backend_version"),
                    "promotion_state": record.get("promotion_state"),
                    "approved_target_families": list(
                        record.get("approved_target_families", []) or []
                    ),
                    "updated_at": record.get("updated_at"),
                }
                for record in adapter_governance_digest.get("recent_cards", [])[:4]
            ),
            recent_adapter_promotion_decisions=tuple(
                {
                    "decision_id": record.get("decision_id"),
                    "adapter_family": record.get("adapter_family"),
                    "backend_id": record.get("backend_id"),
                    "decision_outcome": record.get("decision_outcome"),
                    "from_state": record.get("from_state"),
                    "to_state": record.get("to_state"),
                    "blocked_reason_codes": list(record.get("blocked_reason_codes", []) or []),
                    "updated_at": record.get("updated_at"),
                }
                for record in adapter_governance_digest.get("recent_promotion_decisions", [])[:4]
            ),
            sim_to_real_digest=sim_to_real_digest,
            rehearsal_digest=rehearsal_digest,
            recent_rehearsals=tuple(
                {
                    "rehearsal_id": record.rehearsal_id,
                    "plan_proposal_id": record.plan_proposal_id,
                    "step_index": record.step_index,
                    "candidate_action_id": record.candidate_action_id,
                    "decision_recommendation": record.decision_recommendation,
                    "predicted_success_probability": record.predicted_success_probability,
                    "confidence_band": record.confidence_band,
                    "risk_codes": list(record.risk_codes),
                    "summary": record.summary,
                    "completed_at": record.completed_at,
                    "skipped": record.skipped,
                }
                for record in counterfactual_rehearsal.recent_rehearsals[:4]
            ),
            recent_rehearsal_comparisons=tuple(
                {
                    "comparison_id": record.comparison_id,
                    "rehearsal_id": record.rehearsal_id,
                    "candidate_action_id": record.candidate_action_id,
                    "observed_outcome_kind": record.observed_outcome_kind,
                    "calibration_bucket": record.calibration_bucket,
                    "comparison_summary": record.comparison_summary,
                    "compared_at": record.compared_at,
                }
                for record in counterfactual_rehearsal.recent_comparisons[:4]
            ),
            autonomy_digest=autonomy_digest,
            reevaluation_digest=reevaluation_digest,
            wake_digest=wake_digest,
            planning_digest=planning_digest,
            executive_policy_audit=build_executive_policy_audit(
                autonomy_digest=autonomy_digest,
                reevaluation_digest=reevaluation_digest,
                wake_digest=wake_digest,
                planning_digest=planning_digest,
                procedural_skill_governance_report=procedural_skill_governance_report,
            ),
            background_maintenance_running=(
                self._maintenance_running_resolver()
                if self._maintenance_running_resolver is not None
                else None
            ),
            next_reevaluation_wake=(
                {
                    "kind": str(next_wake[0]),
                    "deadline": getattr(next_wake[1], "isoformat", lambda: str(next_wake[1]))(),
                }
                if next_wake is not None
                else None
            ),
        )

    def inspect_packet(
        self,
        task: BrainContextTask,
        query_text: str,
        include_historical_claims: bool | None = None,
    ) -> BrainRuntimePacketInspection:
        """Inspect one compiled packet without mutating state."""
        surface = self._surface_builder.build(
            latest_user_text=query_text,
            task=task,
            include_historical_claims=include_historical_claims,
        )
        packet = compile_context_packet_from_surface(
            snapshot=surface,
            latest_user_text=query_text,
            task=task,
            language=self._language,
            base_prompt=self._base_prompt,
            context_selector=self._compiler.context_selector,
            persona_modality=self._persona_modality,
        )
        packet_trace = packet.packet_trace.as_dict() if packet.packet_trace is not None else None
        packet_digest = build_context_packet_digest(
            packet_traces={task.value: packet_trace},
        ).get(task.value, {})
        return BrainRuntimePacketInspection(
            task=task,
            query_text=query_text,
            surface=surface,
            compiled_packet=packet,
            selected_context=packet.selected_context,
            packet_trace=packet_trace,
            packet_digest=packet_digest,
        )

    def inspect_pending_wakes(self) -> BrainRuntimePendingWakeInspection:
        """Inspect current waiting commitments and recent wake-routing rows."""
        session_ids = self._session_resolver()
        surface = self._surface_builder.build(
            latest_user_text="",
            task=BrainContextTask.WAKE,
            include_historical_claims=True,
        )
        recent_events = self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=192,
        )
        wake_digest = build_wake_digest(
            commitment_projection=surface.commitment_projection,
            recent_events=recent_events,
        )
        return BrainRuntimePendingWakeInspection(
            session_ids=session_ids,
            generated_at=surface.generated_at,
            current_waiting_commitments=list(wake_digest.get("current_waiting_commitments", [])),
            current_wait_kind_counts=dict(wake_digest.get("current_wait_kind_counts", {})),
            recent_wait_rows=list(wake_digest.get("recent_keep_waiting", [])),
            wake_digest=wake_digest,
        )

    def interrupt_commitment(
        self,
        commitment_id: str,
        reason_summary: str,
        wake_conditions: list[BrainWakeCondition] | None = None,
    ) -> BrainRuntimeCommitmentControlResult:
        """Interrupt one active commitment through the executive defer path."""
        commitment = self._require_commitment(commitment_id)
        if commitment.status != "active":
            return BrainRuntimeCommitmentControlResult(
                control_kind="interrupt",
                commitment_id=commitment_id,
                applied=False,
                status_before=commitment.status,
                status_after=commitment.status,
                reason_summary=reason_summary,
            )
        updated = self._executive.defer_commitment(
            commitment_id=commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.EXPLICIT_DEFER.value,
                summary=reason_summary,
                details={
                    "runtime_shell_control": {
                        "control_kind": "interrupt",
                        "reason_summary": reason_summary,
                    }
                },
            ),
            wake_conditions=(
                list(wake_conditions)
                if wake_conditions is not None
                else [
                    BrainWakeCondition(
                        kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                        summary="Resume explicitly when ready.",
                        details={},
                    )
                ]
            ),
            source="runtime_shell",
            event_details={
                "runtime_shell_control": {
                    "control_kind": "interrupt",
                    "status_before": commitment.status,
                    "status_after": "deferred",
                    "reason_summary": reason_summary,
                }
            },
        )
        event = self._latest_shell_event(
            event_type=BrainEventType.GOAL_DEFERRED,
            commitment_id=commitment_id,
        )
        return BrainRuntimeCommitmentControlResult(
            control_kind="interrupt",
            commitment_id=commitment_id,
            applied=True,
            status_before=commitment.status,
            status_after=updated.status,
            reason_summary=reason_summary,
            event_id=event.event_id if event is not None else None,
        )

    def suppress_commitment(
        self,
        commitment_id: str,
        reason_summary: str,
    ) -> BrainRuntimeCommitmentControlResult:
        """Move one non-terminal commitment into explicit-resume-only holding state."""
        commitment = self._require_commitment(commitment_id)
        if commitment.status in {"completed", "cancelled", "failed"}:
            return BrainRuntimeCommitmentControlResult(
                control_kind="suppress",
                commitment_id=commitment_id,
                applied=False,
                status_before=commitment.status,
                status_after=commitment.status,
                reason_summary=reason_summary,
            )
        updated = self._executive.defer_commitment(
            commitment_id=commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.EXPLICIT_DEFER.value,
                summary=reason_summary,
                details={
                    "runtime_shell_control": {
                        "control_kind": "suppress",
                        "reason_summary": reason_summary,
                    }
                },
            ),
            wake_conditions=[
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume explicitly when ready.",
                    details={},
                )
            ],
            source="runtime_shell",
            event_details={
                "runtime_shell_control": {
                    "control_kind": "suppress",
                    "status_before": commitment.status,
                    "status_after": "deferred",
                    "reason_summary": reason_summary,
                }
            },
        )
        event = self._latest_shell_event(
            event_type=BrainEventType.GOAL_DEFERRED,
            commitment_id=commitment_id,
        )
        return BrainRuntimeCommitmentControlResult(
            control_kind="suppress",
            commitment_id=commitment_id,
            applied=True,
            status_before=commitment.status,
            status_after=updated.status,
            reason_summary=reason_summary,
            event_id=event.event_id if event is not None else None,
        )

    def resume_commitment(
        self,
        commitment_id: str,
        reason_summary: str | None = None,
    ) -> BrainRuntimeCommitmentControlResult:
        """Resume one deferred or blocked commitment through the executive."""
        commitment = self._require_commitment(commitment_id)
        if commitment.status not in {"deferred", "blocked"}:
            return BrainRuntimeCommitmentControlResult(
                control_kind="resume",
                commitment_id=commitment_id,
                applied=False,
                status_before=commitment.status,
                status_after=commitment.status,
                reason_summary=reason_summary,
            )
        updated = self._executive.resume_commitment(
            commitment_id=commitment_id,
            source="runtime_shell",
            event_details={
                "runtime_shell_control": {
                    "control_kind": "resume",
                    "status_before": commitment.status,
                    "status_after": "active",
                    "reason_summary": reason_summary,
                }
            },
        )
        event = self._latest_shell_event(
            event_type=BrainEventType.GOAL_RESUMED,
            commitment_id=commitment_id,
        )
        return BrainRuntimeCommitmentControlResult(
            control_kind="resume",
            commitment_id=commitment_id,
            applied=True,
            status_before=commitment.status,
            status_after=updated.status,
            reason_summary=reason_summary,
            event_id=event.event_id if event is not None else None,
        )

    def run_reflection_once(self, *, trigger: str = "manual") -> BrainReflectionRunResult:
        """Run one bounded shell-triggered reflection pass."""
        session_ids = self._session_resolver()
        resolved_trigger = (
            trigger if trigger.startswith("runtime_shell:") else f"runtime_shell:{trigger}"
        )
        return self._reflection_engine.run_once(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            session_ids=session_ids,
            trigger=resolved_trigger,
        )

    def export_audit(
        self,
        *,
        output_dir: Path | None = None,
        replay_cases_dir: Path | None = None,
        context_queries: dict[str, str] | None = None,
    ) -> BrainContinuityAuditReport:
        """Export one shell-triggered continuity audit artifact."""
        session_ids = self._session_resolver()
        return BrainContinuityAuditExporter(store=self._store).export(
            session_ids=session_ids,
            presence_scope_key=self._presence_scope_key,
            language=self._language,
            output_dir=output_dir,
            replay_cases_dir=replay_cases_dir,
            context_queries=context_queries,
            export_metadata={
                "source": "runtime_shell",
                "action_kind": "audit_export",
            },
        )

    def runtime_shell_digest(self) -> dict[str, Any]:
        """Return the derived shell-control digest."""
        session_ids = self._session_resolver()
        surface = self._surface_builder.build(
            latest_user_text="",
            task=BrainContextTask.REPLY,
            include_historical_claims=True,
        )
        packet_traces = self._packet_traces_for_digest(reply_surface=surface)
        return build_runtime_shell_digest(
            recent_events=self._store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=192,
            ),
            reflection_cycles=[
                record.as_dict()
                for record in self._store.list_reflection_cycles(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=12,
                )
            ],
            memory_exports=self._store.list_memory_exports(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=12,
            ),
            counterfactual_rehearsal=self._store.build_counterfactual_rehearsal_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                presence_scope_key=self._presence_scope_key,
            ).as_dict(),
            embodied_executive=self._store.build_embodied_executive_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                presence_scope_key=self._presence_scope_key,
            ).as_dict(),
            practice_director=self._store.build_practice_director_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                presence_scope_key=self._presence_scope_key,
            ),
            skill_evidence_ledger=self._store.build_skill_evidence_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
            ),
            skill_governance=self._store.build_skill_governance_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
            ),
            adapter_governance=self._store.build_adapter_governance_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
            ),
            recent_action_events=[
                {
                    "action_id": record.action_id,
                    "source": record.source,
                    "accepted": record.accepted,
                    "preview_only": record.preview_only,
                    "summary": record.summary,
                    "metadata": dict(record.metadata),
                    "created_at": record.created_at,
                }
                for record in self._store.recent_action_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=12,
                )
            ],
            predictive_world_model=surface.predictive_world_model.as_dict(),
            multimodal_autobiography=[
                record.as_dict()
                for entry in surface.scene_episodes
                if (record := parse_multimodal_autobiography_record(entry)) is not None
            ],
            packet_traces=packet_traces,
        )

    def _packet_traces_for_digest(
        self,
        *,
        reply_surface: BrainContextSurfaceSnapshot,
    ) -> dict[str, Any]:
        default_query = (
            _optional_text(reply_surface.working_context.last_user_text)
            or _optional_text(reply_surface.agenda.active_goal_summary)
            or ""
        )
        packet_traces: dict[str, Any] = {}
        for task in all_brain_context_tasks():
            surface = (
                reply_surface
                if task == BrainContextTask.REPLY
                else self._surface_builder.build(
                    latest_user_text=default_query,
                    task=task,
                    include_historical_claims=True,
                )
            )
            packet = compile_context_packet_from_surface(
                snapshot=surface,
                latest_user_text=default_query,
                task=task,
                language=self._language,
                base_prompt=self._base_prompt,
                context_selector=self._compiler.context_selector,
            )
            packet_traces[task.value] = (
                packet.packet_trace.as_dict() if packet.packet_trace is not None else None
            )
        return packet_traces

    def _require_commitment(self, commitment_id: str) -> BrainCommitmentRecord:
        commitment = self._store.get_executive_commitment(commitment_id=commitment_id)
        if commitment is None:
            raise KeyError(f"Missing commitment '{commitment_id}'.")
        return commitment

    def _latest_shell_event(
        self,
        *,
        event_type: str,
        commitment_id: str,
    ):
        session_ids = self._session_resolver()
        for event in self._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=32,
        ):
            if event.event_type != event_type:
                continue
            payload = dict(event.payload or {})
            commitment = dict(payload.get("commitment") or {})
            goal = dict(payload.get("goal") or {})
            if str(event.source or "").strip() != "runtime_shell":
                continue
            if (
                str(commitment.get("commitment_id", "")).strip() == commitment_id
                or str(goal.get("commitment_id", "")).strip() == commitment_id
            ):
                return event
        return None


__all__ = [
    "BrainRuntimeCommitmentControlResult",
    "BrainRuntimePacketInspection",
    "BrainRuntimePendingWakeInspection",
    "BrainRuntimeShell",
    "BrainRuntimeShellSnapshot",
]

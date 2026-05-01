"""Runtime assembly for Blink brain-enabled local flows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.brain._executive import BrainPlanningDraft, BrainPlanningRequest
from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    EmbodiedCapabilityDispatcher,
    EmbodiedCommandInterpreter,
    embodied_action_tool_prompt,
    register_embodied_action_tools,
)
from blink.brain.adapters.embodied_policy import LocalRobotHeadEmbodiedPolicyAdapter
from blink.brain.adapters.world_model import LocalDeterministicWorldModelAdapter
from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import CapabilitySideEffectSink
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.context import BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.counterfactuals import BrainCounterfactualRehearsalEngine
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.knowledge import (
    BrainKnowledgeRoutingDecision,
    unavailable_knowledge_routing_decision,
)
from blink.brain.memory import (
    BrainMemoryConsolidator,
    memory_tool_prompt,
    register_memory_tools,
)
from blink.brain.memory_v2 import (
    BrainReflectionEngine,
    BrainReflectionScheduler,
    MemoryCommandIntent,
    MemoryContinuityTrace,
    build_memory_continuity_trace,
    detect_memory_command_intent,
    public_actor_event_metadata_for_memory_continuity,
)
from blink.brain.persona import (
    BrainBehaviorControlProfile,
    BrainBehaviorControlUpdateResult,
    BrainExpressionFrame,
    BrainExpressionVoiceMetricsRecorder,
    BrainExpressionVoiceMetricsSnapshot,
    BrainExpressionVoicePolicy,
    BrainMemoryPersonaPerformancePlan,
    BrainPersonaModality,
    BrainRealtimeVoiceActuationPlan,
    BrainRuntimeExpressionState,
    BrainVoiceBackendCapabilities,
    BrainVoiceBackendCapabilityRegistry,
    apply_behavior_control_update,
    compile_expression_voice_policy,
    compile_memory_persona_performance_plan,
    compile_realtime_voice_actuation_plan,
    load_behavior_control_profile,
    resolve_voice_backend_capabilities,
    runtime_expression_state_from_frame,
    unavailable_expression_voice_metrics_snapshot,
    unavailable_memory_persona_performance_plan,
    unavailable_runtime_expression_state,
)
from blink.brain.presence import BrainPresenceSnapshot, normalize_presence_snapshot
from blink.brain.procedural_planning import split_planning_skill_candidates
from blink.brain.runtime_shell import BrainRuntimeShell
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.tools import register_robot_head_tools, robot_head_tool_prompt
from blink.interaction.active_listening import (
    ActiveListenerPhaseV2,
    ActiveListenerStateV2,
    ActiveListeningHint,
    ActiveListeningUnderstanding,
    BrowserActiveListeningPhase,
    BrowserActiveListeningSnapshot,
    extract_active_listening_understanding,
)
from blink.project_identity import local_env_name
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.brain.context import BrainCompiledContextPacket
    from blink.brain.memory_v2 import BrainMemoryUseTrace
    from blink.processors.aggregators.llm_context import LLMContext


def _parse_ts(value: str | None) -> datetime | None:
    """Parse one stored ISO timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _public_reason_fragment(value: object) -> str:
    """Normalize one diagnostic reason into a compact public-safe fragment."""
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in str(value or ""))
    return "_".join(part for part in text.split("_") if part)[:80] or "unknown"


def _dedupe_reason_codes(values: tuple[str, ...]) -> tuple[str, ...]:
    """Dedupe compact reason codes while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _merge_active_listening_hints(
    existing: tuple[ActiveListeningHint, ...],
    new: tuple[ActiveListeningHint, ...],
    *,
    limit: int = 5,
) -> tuple[ActiveListeningHint, ...]:
    """Merge public-safe active-listening hints without keeping raw transcripts."""
    seen: set[tuple[str, str]] = set()
    result: list[ActiveListeningHint] = []
    for hint in (*existing, *new):
        key = (str(hint.kind).lower(), str(hint.value).lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(hint)
        if len(result) >= limit:
            break
    return tuple(result)


_VOICE_INPUT_STT_WAIT_TOO_LONG_MS = 5000


def _persona_modality_for_runtime(
    *,
    runtime_kind: str,
    robot_head_controller: RobotHeadController | None,
) -> BrainPersonaModality:
    """Resolve one high-level expression modality from runtime configuration."""
    if robot_head_controller is not None:
        return BrainPersonaModality.EMBODIED
    if runtime_kind == "voice":
        return BrainPersonaModality.VOICE
    if runtime_kind == "browser":
        return BrainPersonaModality.BROWSER
    return BrainPersonaModality.TEXT


class BrainRuntime:
    """Own the browser/voice brain runtime assembly."""

    def __init__(
        self,
        *,
        base_prompt: str,
        language: Language,
        runtime_kind: str,
        session_resolver,
        llm,
        robot_head_controller: RobotHeadController | None = None,
        robot_head_operator_mode: bool = False,
        vision_enabled: bool = False,
        continuous_perception_enabled: bool = False,
        store: BrainStore | None = None,
        brain_db_path: str | Path | None = None,
        context: LLMContext | None = None,
        tts_backend: str | None = None,
        voice_backend_registry: BrainVoiceBackendCapabilityRegistry | None = None,
        performance_event_sink: Callable[..., object] | None = None,
    ):
        """Initialize a local Blink brain runtime.

        Args:
            base_prompt: Baseline system prompt before dynamic brain context.
            language: Active runtime language.
            runtime_kind: Runtime identifier such as `browser` or `voice`.
            session_resolver: Callable returning stable session ids.
            llm: Active LLM service for tool registration.
            robot_head_controller: Optional robot-head controller.
            robot_head_operator_mode: Whether raw diagnostic tools remain exposed.
            vision_enabled: Whether this runtime has camera vision enabled.
            continuous_perception_enabled: Whether symbolic browser perception should run.
            store: Optional prebuilt brain store.
            brain_db_path: Optional SQLite database override path.
            context: Optional already-created shared LLM context.
            tts_backend: Optional runtime TTS backend label for policy audit surfaces.
            voice_backend_registry: Optional backend-specific voice capability registry.
            performance_event_sink: Optional public-safe browser performance event sink.
        """
        from blink.brain.processors import (
            BrainContextCompiler,
            BrainContextCompilerProcessor,
            BrainEventRecorderProcessor,
            BrainExecutiveStartupProcessor,
            BrainTextUserTurnProcessor,
            EmbodiedCommandProcessor,
            HotPathMemoryExtractor,
        )

        self.language = language
        self.runtime_kind = runtime_kind
        self.session_resolver = session_resolver
        self.context = context
        self.llm = llm
        self.robot_head_controller = robot_head_controller
        self.robot_head_operator_mode = robot_head_operator_mode
        self.vision_enabled = vision_enabled
        self.continuous_perception_enabled = continuous_perception_enabled and vision_enabled
        self.tts_backend = tts_backend
        self.voice_backend_registry = (
            voice_backend_registry or BrainVoiceBackendCapabilityRegistry.default()
        )
        self.performance_event_sink = performance_event_sink
        self.persona_modality = _persona_modality_for_runtime(
            runtime_kind=runtime_kind,
            robot_head_controller=robot_head_controller,
        )
        self._owns_store = store is None
        resolved_brain_db_path = brain_db_path or os.getenv(local_env_name("BRAIN_DB_PATH"))
        self.store = store or BrainStore(
            path=resolved_brain_db_path,
            world_model_adapter=LocalDeterministicWorldModelAdapter(),
        )
        self.store.ensure_default_blocks(load_default_agent_blocks())
        self._migrate_legacy_memory()
        self.presence_scope_key = f"{runtime_kind}:presence"
        resolved_session_ids = self.session_resolver()
        self.store.ensure_default_adapter_cards(
            agent_id=resolved_session_ids.agent_id,
            user_id=resolved_session_ids.user_id,
            session_id=resolved_session_ids.session_id,
            thread_id=resolved_session_ids.thread_id,
            source="brain_runtime",
        )
        self._capability_side_effect_sink: CapabilitySideEffectSink | None = None
        self._update_presence(
            BrainPresenceSnapshot(
                runtime_kind=runtime_kind,
                robot_head_enabled=robot_head_controller is not None,
                robot_head_mode=(
                    robot_head_controller.driver_mode if robot_head_controller else "none"
                ),
                robot_head_available=robot_head_controller is not None,
                robot_head_armed=False,
                vision_enabled=vision_enabled,
                vision_connected=False,
                perception_disabled=not self.continuous_perception_enabled,
            )
        )
        self.action_engine: EmbodiedActionEngine | None = None
        self.action_dispatcher: EmbodiedCapabilityDispatcher | None = None
        if robot_head_controller is not None:
            self.action_engine = EmbodiedActionEngine(
                library=EmbodiedActionLibrary.build_default(),
                controller=robot_head_controller,
                policy_adapter=LocalRobotHeadEmbodiedPolicyAdapter(
                    controller=robot_head_controller
                ),
                store=self.store,
                session_resolver=self.session_resolver,
                presence_scope_key=self.presence_scope_key,
            )
        self.capability_registry = build_brain_capability_registry(
            language=language,
            action_engine=self.action_engine,
        )
        context_surface_builder = BrainContextSurfaceBuilder(
            store=self.store,
            session_resolver=self.session_resolver,
            presence_scope_key=self.presence_scope_key,
            language=language,
            capability_registry=self.capability_registry,
        )
        self.compiler = BrainContextCompiler(
            store=self.store,
            session_resolver=self.session_resolver,
            language=language,
            base_prompt=base_prompt,
            context_surface_builder=context_surface_builder,
        )
        self.voice_metrics_recorder = BrainExpressionVoiceMetricsRecorder()
        self._voice_input_connected = False
        self._voice_input_microphone_state = "disconnected"
        self._voice_input_stt_state = "idle"
        self._voice_input_audio_frame_count = 0
        self._voice_input_speech_start_count = 0
        self._voice_input_speech_stop_count = 0
        self._voice_input_interim_transcription_count = 0
        self._voice_input_transcription_count = 0
        self._voice_input_stt_error_count = 0
        self._voice_input_last_audio_frame_at: str | None = None
        self._voice_input_last_audio_frame_monotonic: float | None = None
        self._voice_input_last_stt_event_at: str | None = None
        self._voice_input_waiting_since_at: str | None = None
        self._voice_input_waiting_since_monotonic: float | None = None
        self._voice_input_active_listening_phase = BrowserActiveListeningPhase.IDLE.value
        self._voice_input_active_listening_last_update_at: str | None = None
        self._voice_input_turn_started_at: str | None = None
        self._voice_input_turn_started_monotonic: float | None = None
        self._voice_input_turn_stopped_at: str | None = None
        self._voice_input_last_turn_duration_ms: int | None = None
        self._voice_input_partial_available = False
        self._voice_input_last_partial_at: str | None = None
        self._voice_input_last_partial_chars = 0
        self._voice_input_last_transcription_at: str | None = None
        self._voice_input_last_transcription_chars = 0
        self._voice_input_turn_final_transcription_chars = 0
        self._voice_input_turn_final_transcription_count = 0
        self._voice_input_active_topics: tuple[ActiveListeningHint, ...] = ()
        self._voice_input_active_constraints: tuple[ActiveListeningHint, ...] = ()
        self._voice_input_active_corrections: tuple[ActiveListeningHint, ...] = ()
        self._voice_input_active_project_references: tuple[ActiveListeningHint, ...] = ()
        self._voice_input_active_uncertainty_flags: tuple[str, ...] = ()
        self._voice_input_memory_command_intent: MemoryCommandIntent | None = None
        self._voice_input_track_reason: str | None = None
        self._voice_input_track_enabled: bool | None = None
        self._latest_compiled_memory_use_trace: BrainMemoryUseTrace | None = None
        self._current_turn_memory_use_trace: BrainMemoryUseTrace | None = None
        self._latest_compiled_memory_continuity_trace: MemoryContinuityTrace | None = None
        self._current_memory_continuity_trace: MemoryContinuityTrace | None = None
        self._current_memory_persona_performance_plan: BrainMemoryPersonaPerformancePlan | None = None
        self._current_teaching_knowledge_decision: BrainKnowledgeRoutingDecision | None = None
        self._recent_teaching_knowledge_decisions: list[BrainKnowledgeRoutingDecision] = []
        counterfactual_rehearsal_engine = (
            BrainCounterfactualRehearsalEngine(
                store=self.store,
                session_resolver=self.session_resolver,
                presence_scope_key=self.presence_scope_key,
                action_engine=self.action_engine,
                context_surface_builder=self.compiler.context_surface_builder,
                context_selector=self.compiler.context_selector,
                language=language,
            )
            if self.action_engine is not None
            else None
        )
        self.consolidator = BrainMemoryConsolidator(store=self.store)
        self.reflection_engine = BrainReflectionEngine(store=self.store)
        self.reflection_scheduler = BrainReflectionScheduler(
            store_path=self.store.path,
            session_resolver=self.session_resolver,
            candidate_goal_sink=self.propose_candidate_goal,
            reevaluation_sink=self.run_presence_director_reevaluation,
        )
        self.executive = BrainExecutive(
            store=self.store,
            session_resolver=self.session_resolver,
            capability_registry=self.capability_registry,
            presence_scope_key=self.presence_scope_key,
            context_surface_builder=self.compiler.context_surface_builder,
            context_selector=self.compiler.context_selector,
            capability_side_effect_sink=self._capability_side_effect_sink,
            counterfactual_rehearsal_engine=counterfactual_rehearsal_engine,
            autonomy_state_changed_callback=self._refresh_reevaluation_alarm,
            planning_callback=self._request_bounded_plan_proposal,
        )
        self.shell = BrainRuntimeShell.from_runtime(self)
        self._reevaluation_alarm_task: asyncio.Task | None = None
        self._reevaluation_alarm_key: tuple[str, str] | None = None
        self.pre_llm_processors = [BrainExecutiveStartupProcessor(executive=self.executive)]
        if runtime_kind == "text":
            self.pre_llm_processors.append(
                BrainTextUserTurnProcessor(
                    store=self.store,
                    session_resolver=self.session_resolver,
                    language=language,
                    executive=self.executive,
                )
            )
        else:
            self.pre_llm_processors.extend(
                [
                    BrainEventRecorderProcessor(
                        store=self.store,
                        session_resolver=self.session_resolver,
                        executive=self.executive,
                    ),
                    HotPathMemoryExtractor(
                        store=self.store,
                        session_resolver=self.session_resolver,
                        language=language,
                    ),
                ]
            )
        self.post_context_processors = []
        self.post_aggregation_processors = []
        if self.action_engine is not None:
            self.action_dispatcher = EmbodiedCapabilityDispatcher(
                action_engine=self.action_engine,
                capability_registry=self.capability_registry,
            )
            if not robot_head_operator_mode:
                self.pre_llm_processors.append(
                    EmbodiedCommandProcessor(
                        interpreter=EmbodiedCommandInterpreter(),
                        action_dispatcher=self.action_dispatcher,
                        executive=self.executive,
                        session_resolver=self.session_resolver,
                        store=self.store,
                        presence_scope_key=self.presence_scope_key,
                        language=language,
                    )
                )

        self.pre_llm_processors.append(
            BrainContextCompilerProcessor(
                compiler=self.compiler,
                persona_modality=self.persona_modality,
                packet_callback=self._record_compiled_context_packet,
            )
        )
        if context is not None:
            self.bind_context(context)

    @property
    def static_system_prompt(self) -> str:
        """Return the thin static system prompt used when creating the LLM."""
        return base_brain_system_prompt(self.language)

    @property
    def prompt_suffix(self) -> str:
        """Return the runtime tool prompt suffix."""
        suffixes = [memory_tool_prompt(self.language)]
        if self.robot_head_controller is not None:
            suffixes.append(
                robot_head_tool_prompt(self.language)
                if self.robot_head_operator_mode
                else embodied_action_tool_prompt(self.language)
            )
        return " ".join(part for part in suffixes if part)

    def _record_compiled_context_packet(self, packet: "BrainCompiledContextPacket"):
        """Remember the latest compiled memory-use trace until the reply commits."""
        self._latest_compiled_memory_use_trace = packet.memory_use_trace
        self._current_turn_memory_use_trace = packet.memory_use_trace
        continuity_trace = self._build_memory_continuity_trace(packet.memory_use_trace)
        self._latest_compiled_memory_continuity_trace = continuity_trace
        self._current_memory_continuity_trace = continuity_trace
        self._current_memory_persona_performance_plan = self.current_memory_persona_performance_plan(
            current_turn_state="thinking",
            memory_use_trace=packet.memory_use_trace,
            memory_continuity_trace=continuity_trace,
        )
        if packet.teaching_knowledge_decision is not None:
            self._current_teaching_knowledge_decision = packet.teaching_knowledge_decision
            self._recent_teaching_knowledge_decisions.insert(0, packet.teaching_knowledge_decision)
            del self._recent_teaching_knowledge_decisions[8:]

    def _pending_memory_use_trace(self) -> "BrainMemoryUseTrace | None":
        """Return the pending trace for the reply currently being committed."""
        return self._latest_compiled_memory_use_trace

    def _build_memory_continuity_trace(
        self,
        memory_use_trace: "BrainMemoryUseTrace | None",
        *,
        created_at: str = "",
        hidden_counts: dict[str, Any] | None = None,
        turn_id: str = "",
        reason_codes: tuple[str, ...] = ("memory_continuity:runtime_compiled",),
    ) -> MemoryContinuityTrace:
        """Build a public-safe continuity trace for current runtime scope."""
        session_ids = self.session_resolver()
        recent_discourse_episodes = ()
        try:
            reader = getattr(self.store, "recent_discourse_episodes", None)
            if callable(reader):
                recent_discourse_episodes = reader(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=8,
                )
        except Exception:
            recent_discourse_episodes = ()
        return build_memory_continuity_trace(
            memory_use_trace=memory_use_trace,
            session_id=session_ids.session_id,
            profile=self.runtime_kind,
            language=self.language.value,
            turn_id=turn_id
            or (
                f"turn:{self.runtime_kind}:{self.language.value}:"
                f"{len(memory_use_trace.refs) if memory_use_trace is not None else 0}"
            ),
            created_at=created_at,
            hidden_counts=hidden_counts,
            command_intent=self._voice_input_memory_command_intent,
            discourse_episodes=recent_discourse_episodes,
            reason_codes=reason_codes,
        )

    def _mark_memory_use_trace_committed(self, trace: "BrainMemoryUseTrace"):
        """Keep the timestamped trace visible as the current-turn trace."""
        self._current_turn_memory_use_trace = trace
        continuity_trace = self._build_memory_continuity_trace(
            trace,
            created_at=trace.created_at,
            reason_codes=(),
        )
        try:
            session_ids = self.session_resolver()
            continuity_trace = self.store.append_memory_continuity_trace(
                trace=continuity_trace,
                session_id=session_ids.session_id,
                source="memory_continuity_trace",
                ts=trace.created_at or None,
            )
        except Exception:
            pass
        self._current_memory_continuity_trace = continuity_trace
        self._latest_compiled_memory_continuity_trace = continuity_trace
        plan = self.current_memory_persona_performance_plan(
            current_turn_state="committed",
            memory_use_trace=trace,
            memory_continuity_trace=continuity_trace,
        )
        self._current_memory_persona_performance_plan = plan
        continuity_metadata = public_actor_event_metadata_for_memory_continuity(continuity_trace)
        if self.performance_event_sink is not None:
            try:
                self.performance_event_sink(
                    event_type="memory.use_trace_committed",
                    source="memory",
                    mode="thinking",
                    metadata={
                        "available": True,
                        "trace_committed": True,
                        **continuity_metadata,
                    },
                    reason_codes=("memory:use_trace_committed",),
                )
                self.performance_event_sink(
                    event_type="memory_persona.performance_plan_committed",
                    source="memory_persona",
                    mode="thinking",
                    metadata={
                        "available": plan.available,
                        "performance_plan_schema_version": (
                            plan.performance_plan_v3.schema_version
                            if plan.performance_plan_v3 is not None
                            else plan.performance_plan_v2.schema_version
                            if plan.performance_plan_v2 is not None
                            else 0
                        ),
                        "performance_plan_v2_schema_version": (
                            plan.performance_plan_v2.schema_version
                            if plan.performance_plan_v2 is not None
                            else 0
                        ),
                        "performance_plan_v3_schema_version": (
                            plan.performance_plan_v3.schema_version
                            if plan.performance_plan_v3 is not None
                            else 0
                        ),
                        "selected_memory_count": plan.selected_memory_count,
                        "suppressed_memory_count": plan.suppressed_memory_count,
                        "cross_language_count": continuity_trace.cross_language_count,
                        "memory_effect": continuity_trace.memory_effect,
                        "memory_effect_labels": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "effect_labels",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "discourse_episode_ids": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "discourse_episode_ids",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "discourse_category_labels": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "discourse_category_labels",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "selected_memory_ids": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "selected_memory_ids",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "memory_conflict_labels": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "conflict_labels",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "memory_staleness_labels": (
                            list(
                                plan.performance_plan_v3.memory_callback_policy.get(
                                    "staleness_labels",
                                    [],
                                )
                            )
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "behavior_effect_count": len(plan.behavior_effects),
                        "persona_reference_count": len(
                            [ref for ref in plan.persona_references if ref.applies]
                        ),
                        "persona_reference_count_v2": (
                            len(plan.performance_plan_v2.persona_references_used)
                            if plan.performance_plan_v2 is not None
                            else 0
                        ),
                        "persona_anchor_count_v3": (
                            len(plan.performance_plan_v3.persona_anchor_refs_v3)
                            if plan.performance_plan_v3 is not None
                            else 0
                        ),
                        "persona_anchor_ids_v3": (
                            [
                                anchor.anchor_id
                                for anchor in plan.performance_plan_v3.persona_anchor_refs_v3
                            ]
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "persona_anchor_situation_keys_v3": (
                            [
                                anchor.situation_key
                                for anchor in plan.performance_plan_v3.persona_anchor_refs_v3
                            ]
                            if plan.performance_plan_v3 is not None
                            else []
                        ),
                        "style_summary_chars": (
                            len(plan.performance_plan_v2.style_summary)
                            if plan.performance_plan_v2 is not None
                            else 0
                        ),
                        "plan_summary_chars": (
                            len(plan.performance_plan_v3.plan_summary)
                            if plan.performance_plan_v3 is not None
                            else 0
                        ),
                        "plan_summary_hash": (
                            hashlib.sha256(
                                plan.performance_plan_v3.plan_summary.encode("utf-8")
                            ).hexdigest()[:16]
                            if plan.performance_plan_v3 is not None
                            else ""
                        ),
                        "stance": (
                            plan.performance_plan_v3.stance
                            if plan.performance_plan_v3 is not None
                            else plan.performance_plan_v2.stance
                            if plan.performance_plan_v2 is not None
                            else "unavailable"
                        ),
                        "response_shape": (
                            plan.performance_plan_v3.response_shape
                            if plan.performance_plan_v3 is not None
                            else plan.performance_plan_v2.response_shape
                            if plan.performance_plan_v2 is not None
                            else "unavailable"
                        ),
                    },
                    reason_codes=(
                        "memory_persona:performance_plan_committed",
                        *plan.reason_codes,
                    ),
                )
            except Exception:
                self.performance_event_sink = None

    def current_memory_use_trace(self) -> "BrainMemoryUseTrace | None":
        """Return the current or most recently compiled memory-use trace."""
        return self._current_turn_memory_use_trace

    def cached_memory_persona_performance_plan(
        self,
    ) -> BrainMemoryPersonaPerformancePlan | None:
        """Return the latest compiled performance plan without touching the store."""
        return self._current_memory_persona_performance_plan

    def current_memory_continuity_trace(self) -> MemoryContinuityTrace | None:
        """Return the current or most recently committed memory-continuity trace."""
        return self._current_memory_continuity_trace

    def recent_memory_use_traces(self, *, limit: int = 8) -> tuple["BrainMemoryUseTrace", ...]:
        """Return recent persisted memory-use traces for this runtime scope."""
        session_ids = self.session_resolver()
        return self.store.recent_memory_use_traces(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=limit,
        )

    def recent_memory_continuity_traces(
        self,
        *,
        limit: int = 8,
    ) -> tuple[MemoryContinuityTrace, ...]:
        """Return recent persisted memory-continuity traces for this runtime scope."""
        session_ids = self.session_resolver()
        return self.store.recent_memory_continuity_traces(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=limit,
        )

    def current_memory_persona_performance_plan(
        self,
        *,
        profile: str | None = None,
        tts_label: str | None = None,
        protected_playback: bool = True,
        camera_state: str | None = None,
        continuous_perception_enabled: bool | None = None,
        current_turn_state: str = "unknown",
        memory_use_trace: "BrainMemoryUseTrace | None" = None,
        memory_continuity_trace: MemoryContinuityTrace | None = None,
        suppressed_memory_count: int = 0,
        active_listening: dict[str, Any] | None = None,
        camera_scene: dict[str, Any] | None = None,
        floor_state: dict[str, Any] | str | None = None,
        user_affect: dict[str, Any] | str | None = None,
        user_intent: dict[str, Any] | str | None = None,
        actor_control_frame: dict[str, Any] | Any | None = None,
        voice_capabilities: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
        voice_actuation_plan: BrainRealtimeVoiceActuationPlan | dict[str, Any] | None = None,
        turn_id: str = "",
    ) -> BrainMemoryPersonaPerformancePlan:
        """Return the current public-safe memory/persona performance plan."""
        try:
            behavior_profile = self.current_behavior_control_profile()
            trace = memory_use_trace if memory_use_trace is not None else self.current_memory_use_trace()
            continuity_trace = (
                memory_continuity_trace
                if memory_continuity_trace is not None
                else self.current_memory_continuity_trace()
            )
            resolved_voice_capabilities = (
                voice_capabilities
                if voice_capabilities is not None
                else self.current_voice_capabilities()
            )
            resolved_voice_actuation_plan = (
                voice_actuation_plan
                if voice_actuation_plan is not None
                else self.current_voice_actuation_plan()
            )
            return compile_memory_persona_performance_plan(
                profile=profile or self.runtime_kind,
                modality=getattr(self.persona_modality, "value", str(self.persona_modality)),
                language=self.language.value,
                tts_label=tts_label or self.tts_backend or "unknown",
                protected_playback=protected_playback,
                camera_state=(
                    camera_state
                    or ("available" if self.vision_enabled else "disabled")
                ),
                continuous_perception_enabled=(
                    self.continuous_perception_enabled
                    if continuous_perception_enabled is None
                    else continuous_perception_enabled
                ),
                current_turn_state=current_turn_state,
                behavior_profile=behavior_profile,
                memory_use_trace=trace,
                memory_continuity_trace=continuity_trace,
                suppressed_memory_count=suppressed_memory_count,
                active_listening=active_listening,
                camera_scene=camera_scene,
                floor_state=floor_state,
                user_affect=user_affect,
                user_intent=user_intent,
                actor_control_frame=actor_control_frame,
                voice_capabilities=resolved_voice_capabilities,
                voice_actuation_plan=resolved_voice_actuation_plan,
                turn_id=turn_id,
            )
        except Exception as exc:  # pragma: no cover - defensive browser/operator surface
            return unavailable_memory_persona_performance_plan(
                f"memory_persona_performance_error:{type(exc).__name__}",
                profile=profile or self.runtime_kind,
            )

    def current_teaching_knowledge_routing(self) -> BrainKnowledgeRoutingDecision:
        """Return the current compact teaching knowledge routing decision."""
        return self._current_teaching_knowledge_decision or unavailable_knowledge_routing_decision(
            "teaching_knowledge_not_compiled"
        )

    def recent_teaching_knowledge_routing(
        self,
        *,
        limit: int = 6,
    ) -> tuple[BrainKnowledgeRoutingDecision, ...]:
        """Return recent compact process-local teaching knowledge routing decisions."""
        safe_limit = max(0, min(24, int(limit or 0)))
        return tuple(self._recent_teaching_knowledge_decisions[:safe_limit])

    def current_expression_frame(
        self,
        *,
        latest_user_text: str = "",
        persona_seriousness: str = "normal",
    ) -> BrainExpressionFrame | None:
        """Return the current compact persona expression frame, if it compiles safely."""
        try:
            frame, _reason_codes, _status = self.compiler.compile_persona_expression_frame(
                latest_user_text=latest_user_text,
                task=BrainContextTask.REPLY,
                persona_modality=self.persona_modality,
                persona_seriousness=persona_seriousness,
            )
        except Exception:
            return None
        return frame

    def current_behavior_control_profile(self) -> BrainBehaviorControlProfile:
        """Return the current relationship-scoped behavior-control profile."""
        return load_behavior_control_profile(
            store=self.store,
            session_ids=self.session_resolver(),
        )

    def update_behavior_control_profile(
        self,
        updates: dict[str, object],
        *,
        source: str = "runtime",
    ) -> BrainBehaviorControlUpdateResult:
        """Apply one bounded behavior-control update."""
        return apply_behavior_control_update(
            store=self.store,
            session_ids=self.session_resolver(),
            updates=dict(updates),
            source=source,
        )

    def current_expression_state(
        self,
        *,
        latest_user_text: str = "",
        persona_seriousness: str = "normal",
    ) -> BrainRuntimeExpressionState:
        """Return compact high-level expression state for audit/browser surfaces."""
        try:
            frame, reason_codes, status = self.compiler.compile_persona_expression_frame(
                latest_user_text=latest_user_text,
                task=BrainContextTask.REPLY,
                persona_modality=self.persona_modality,
                persona_seriousness=persona_seriousness,
            )
        except Exception as exc:
            return unavailable_runtime_expression_state(
                modality=self.persona_modality,
                reason_codes=(f"runtime_expression_error:{type(exc).__name__}",),
                memory_persona_section_status={
                    "persona_expression": "error",
                    "persona_defaults": "unknown",
                },
                tts_backend=self.tts_backend,
                voice_backend_registry=self.voice_backend_registry,
            )
        return runtime_expression_state_from_frame(
            frame,
            modality=self.persona_modality,
            reason_codes=reason_codes,
            memory_persona_section_status=status,
            tts_backend=self.tts_backend,
            voice_backend_registry=self.voice_backend_registry,
        )

    def current_voice_policy(
        self,
        *,
        latest_user_text: str = "",
        persona_seriousness: str = "normal",
    ) -> BrainExpressionVoicePolicy:
        """Return the current provider-neutral voice policy."""
        frame = self.current_expression_frame(
            latest_user_text=latest_user_text,
            persona_seriousness=persona_seriousness,
        )
        return compile_expression_voice_policy(
            frame,
            modality=self.persona_modality,
            tts_backend=self.tts_backend,
        )

    def current_voice_capabilities(self) -> BrainVoiceBackendCapabilities:
        """Return the current backend-specific voice capability matrix."""
        return resolve_voice_backend_capabilities(
            self.tts_backend,
            registry=self.voice_backend_registry,
        ).capabilities

    def current_voice_actuation_plan(
        self,
        *,
        latest_user_text: str = "",
        persona_seriousness: str = "normal",
    ) -> BrainRealtimeVoiceActuationPlan:
        """Return the current capability-aware voice actuation plan."""
        policy = self.current_voice_policy(
            latest_user_text=latest_user_text,
            persona_seriousness=persona_seriousness,
        )
        return compile_realtime_voice_actuation_plan(
            policy,
            capabilities=self.current_voice_capabilities(),
            tts_backend=self.tts_backend,
        )

    def current_voice_metrics(self) -> BrainExpressionVoiceMetricsSnapshot:
        """Return current provider-neutral voice policy metrics."""
        try:
            return self.voice_metrics_recorder.snapshot()
        except Exception as exc:
            return unavailable_expression_voice_metrics_snapshot(
                f"runtime_voice_metrics_error:{type(exc).__name__}",
            )

    def _mark_voice_input_event(self) -> str:
        timestamp = datetime.now(UTC).isoformat()
        self._voice_input_last_stt_event_at = timestamp
        return timestamp

    def _mark_active_listening_phase(self, phase: BrowserActiveListeningPhase | str) -> str:
        """Record the current public active-listening phase."""
        timestamp = datetime.now(UTC).isoformat()
        try:
            raw_phase = phase.value if isinstance(phase, BrowserActiveListeningPhase) else str(phase)
            resolved_phase = BrowserActiveListeningPhase(raw_phase).value
        except ValueError:
            resolved_phase = BrowserActiveListeningPhase.IDLE.value
        self._voice_input_active_listening_phase = resolved_phase
        self._voice_input_active_listening_last_update_at = timestamp
        return timestamp

    def note_voice_input_connected(self, connected: bool):
        """Record whether the browser voice input transport is connected."""
        self._voice_input_connected = bool(connected)
        self._voice_input_track_reason = None if connected else "browser_disconnected"
        self._voice_input_track_enabled = bool(connected)
        self._voice_input_microphone_state = "connected" if connected else "disconnected"
        if not connected:
            self._voice_input_stt_state = "idle"
            self._voice_input_waiting_since_at = None
            self._voice_input_waiting_since_monotonic = None
            self._voice_input_turn_started_at = None
            self._voice_input_turn_started_monotonic = None
            self._voice_input_turn_stopped_at = None
            self._voice_input_last_turn_duration_ms = None
            self._voice_input_partial_available = False
            self._voice_input_turn_final_transcription_chars = 0
            self._voice_input_turn_final_transcription_count = 0
            self._mark_active_listening_phase(BrowserActiveListeningPhase.IDLE)
        self._mark_voice_input_event()

    def note_voice_input_audio_frame(self):
        """Record one received input audio frame without storing audio payloads."""
        self._voice_input_audio_frame_count += 1
        self._voice_input_last_audio_frame_at = datetime.now(UTC).isoformat()
        self._voice_input_last_audio_frame_monotonic = time.monotonic()
        self._voice_input_track_reason = None
        self._voice_input_track_enabled = True
        if self._voice_input_microphone_state != "stalled":
            self._voice_input_microphone_state = "receiving"

    def note_voice_input_speech_started(self):
        """Record that VAD observed a user speech start."""
        self._voice_input_speech_start_count += 1
        self._voice_input_stt_state = "speech_detected"
        self._voice_input_waiting_since_at = None
        self._voice_input_waiting_since_monotonic = None
        self._voice_input_turn_started_at = self._mark_voice_input_event()
        self._voice_input_turn_started_monotonic = time.monotonic()
        self._voice_input_turn_stopped_at = None
        self._voice_input_last_turn_duration_ms = None
        self._voice_input_partial_available = False
        self._voice_input_last_partial_chars = 0
        self._voice_input_last_transcription_chars = 0
        self._voice_input_turn_final_transcription_chars = 0
        self._voice_input_turn_final_transcription_count = 0
        self._voice_input_active_topics = ()
        self._voice_input_active_constraints = ()
        self._voice_input_active_corrections = ()
        self._voice_input_active_project_references = ()
        self._voice_input_active_uncertainty_flags = ()
        self._mark_active_listening_phase(BrowserActiveListeningPhase.SPEECH_STARTED)
        return

    def note_voice_input_interim_transcription(self, text: object = ""):
        """Record one interim STT transcript without storing transcript text."""
        self._voice_input_interim_transcription_count += 1
        self._voice_input_last_partial_chars = len(str(text or ""))
        self._voice_input_last_partial_at = self._mark_voice_input_event()
        self._voice_input_partial_available = True
        self._voice_input_stt_state = "transcribing"
        understanding = extract_active_listening_understanding(
            text,
            language=self.language,
            source="partial_transcript",
        )
        self._apply_active_listening_understanding(understanding)
        intent = detect_memory_command_intent(text, language=self.language)
        if intent.intent != "none":
            self._voice_input_memory_command_intent = intent
        self._mark_active_listening_phase(BrowserActiveListeningPhase.PARTIAL_TRANSCRIPT)

    def _current_voice_turn_duration_ms(self) -> int | None:
        if self._voice_input_turn_started_monotonic is None:
            return self._voice_input_last_turn_duration_ms
        if (
            self._voice_input_turn_stopped_at is not None
            and self._voice_input_waiting_since_monotonic is None
        ):
            return self._voice_input_last_turn_duration_ms
        end = (
            self._voice_input_waiting_since_monotonic
            if self._voice_input_turn_stopped_at is not None
            else time.monotonic()
        )
        if end is None:
            return None
        return max(0, int((end - self._voice_input_turn_started_monotonic) * 1000))

    def note_voice_input_speech_stopped(self):
        """Record that VAD ended a user speech segment and STT is expected."""
        self._voice_input_speech_stop_count += 1
        self._voice_input_stt_state = "waiting"
        self._voice_input_waiting_since_at = self._mark_voice_input_event()
        self._voice_input_waiting_since_monotonic = time.monotonic()
        self._voice_input_turn_stopped_at = self._voice_input_waiting_since_at
        self._voice_input_last_turn_duration_ms = self._current_voice_turn_duration_ms()
        self._mark_active_listening_phase(BrowserActiveListeningPhase.TRANSCRIBING)

    def note_voice_input_transcription(self, text: object = ""):
        """Record one STT transcript without storing the transcript text."""
        turn_duration_ms = self._current_voice_turn_duration_ms()
        transcript_chars = len(str(text or ""))
        self._voice_input_transcription_count += 1
        self._voice_input_last_transcription_chars = transcript_chars
        self._voice_input_turn_final_transcription_chars += transcript_chars
        self._voice_input_turn_final_transcription_count += 1
        self._voice_input_last_transcription_at = self._mark_voice_input_event()
        self._voice_input_stt_state = "transcribed"
        self._voice_input_waiting_since_at = None
        self._voice_input_waiting_since_monotonic = None
        self._voice_input_last_turn_duration_ms = turn_duration_ms
        if (
            self._voice_input_turn_stopped_at is None
            and self._voice_input_turn_started_monotonic is not None
        ):
            self._voice_input_turn_stopped_at = self._voice_input_last_transcription_at
        understanding = extract_active_listening_understanding(text, language=self.language)
        self._merge_active_listening_understanding(understanding)
        self._voice_input_memory_command_intent = detect_memory_command_intent(
            text,
            language=self.language,
        )
        self._mark_active_listening_phase(BrowserActiveListeningPhase.FINAL_TRANSCRIPT)

    def _apply_active_listening_understanding(
        self,
        understanding: ActiveListeningUnderstanding,
    ):
        """Record bounded active-listening hints without storing transcript text."""
        self._voice_input_active_topics = understanding.topics
        self._voice_input_active_constraints = understanding.constraints
        self._voice_input_active_corrections = understanding.corrections
        self._voice_input_active_project_references = understanding.project_references
        self._voice_input_active_uncertainty_flags = understanding.uncertainty_flags

    def _merge_active_listening_understanding(
        self,
        understanding: ActiveListeningUnderstanding,
    ):
        """Merge final-fragment understanding for the current spoken turn."""
        self._voice_input_active_topics = _merge_active_listening_hints(
            self._voice_input_active_topics,
            understanding.topics,
        )
        self._voice_input_active_constraints = _merge_active_listening_hints(
            self._voice_input_active_constraints,
            understanding.constraints,
        )
        self._voice_input_active_corrections = _merge_active_listening_hints(
            self._voice_input_active_corrections,
            understanding.corrections,
        )
        self._voice_input_active_project_references = _merge_active_listening_hints(
            self._voice_input_active_project_references,
            understanding.project_references,
        )
        self._voice_input_active_uncertainty_flags = tuple(
            dict.fromkeys(
                (
                    *self._voice_input_active_uncertainty_flags,
                    *understanding.uncertainty_flags,
                )
            )
        )[:5]

    def note_voice_input_stt_error(self, error_type: object = "ErrorFrame"):
        """Record one bounded STT error category without storing raw error text."""
        self._voice_input_stt_error_count += 1
        self._voice_input_stt_state = "error"
        self._voice_input_track_reason = _public_reason_fragment(error_type)
        self._voice_input_waiting_since_at = None
        self._voice_input_waiting_since_monotonic = None
        self._mark_active_listening_phase(BrowserActiveListeningPhase.ERROR)
        self._mark_voice_input_event()

    def note_voice_input_track_stalled(self, event: object):
        """Record a transport-reported microphone track stall."""
        self._voice_input_microphone_state = "stalled"
        self._voice_input_track_reason = _public_reason_fragment(getattr(event, "reason", None))
        enabled = getattr(event, "enabled", None)
        self._voice_input_track_enabled = bool(enabled) if enabled is not None else None
        self._mark_voice_input_event()

    def note_voice_input_track_resumed(self, event: object):
        """Record microphone track recovery."""
        self._voice_input_microphone_state = "receiving"
        self._voice_input_track_reason = _public_reason_fragment(getattr(event, "reason", None))
        enabled = getattr(event, "enabled", None)
        self._voice_input_track_enabled = bool(enabled) if enabled is not None else True
        self._mark_voice_input_event()

    def current_voice_input_health(self) -> dict[str, object]:
        """Return public-safe browser voice input and STT health."""
        now = time.monotonic()
        audio_age_ms = (
            max(0, int((now - self._voice_input_last_audio_frame_monotonic) * 1000))
            if self._voice_input_last_audio_frame_monotonic is not None
            else None
        )
        stt_wait_age_ms = (
            max(0, int((now - self._voice_input_waiting_since_monotonic) * 1000))
            if self._voice_input_waiting_since_monotonic is not None
            and self._voice_input_stt_state == "waiting"
            else None
        )
        stt_waiting_too_long = bool(
            stt_wait_age_ms is not None
            and stt_wait_age_ms >= _VOICE_INPUT_STT_WAIT_TOO_LONG_MS
        )
        microphone_state = self._voice_input_microphone_state
        if not self._voice_input_connected:
            microphone_state = "disconnected"
        elif microphone_state != "stalled" and self._voice_input_last_audio_frame_at is None:
            microphone_state = "waiting_for_audio"
        elif (
            microphone_state != "stalled"
            and audio_age_ms is not None
            and audio_age_ms > 3000
            and self._voice_input_stt_state != "waiting"
        ):
            microphone_state = "no_audio_frames"

        reason_codes = [
            "voice_input_health:v1",
            "voice_input:available"
            if self.runtime_kind in {"browser", "voice"}
            else "voice_input:unavailable",
            f"microphone:{microphone_state}",
            f"stt:{self._voice_input_stt_state}",
        ]
        if self._voice_input_transcription_count:
            reason_codes.append("stt:transcribed")
        if self._voice_input_interim_transcription_count:
            reason_codes.append("stt:partial_transcript")
        if self._voice_input_stt_error_count:
            reason_codes.append("stt:error_observed")
        if self._voice_input_stt_state == "waiting" and microphone_state == "receiving":
            reason_codes.append("voice_input:mic_receiving_but_stt_waiting")
        if stt_waiting_too_long:
            reason_codes.append("stt:waiting_too_long")
        if self._voice_input_track_reason:
            reason_codes.append(f"voice_input_reason:{self._voice_input_track_reason}")

        return {
            "schema_version": 1,
            "available": self.runtime_kind in {"browser", "voice"},
            "microphone_state": microphone_state,
            "stt_state": self._voice_input_stt_state,
            "audio_frame_count": self._voice_input_audio_frame_count,
            "speech_start_count": self._voice_input_speech_start_count,
            "speech_stop_count": self._voice_input_speech_stop_count,
            "interim_transcription_count": self._voice_input_interim_transcription_count,
            "last_partial_transcription_chars": self._voice_input_last_partial_chars,
            "last_partial_transcription_at": self._voice_input_last_partial_at,
            "partial_transcript_available": self._voice_input_partial_available,
            "transcription_count": self._voice_input_transcription_count,
            "stt_error_count": self._voice_input_stt_error_count,
            "last_audio_frame_at": self._voice_input_last_audio_frame_at,
            "last_audio_frame_age_ms": audio_age_ms,
            "last_stt_event_at": self._voice_input_last_stt_event_at,
            "stt_waiting_since_at": self._voice_input_waiting_since_at,
            "stt_wait_age_ms": stt_wait_age_ms,
            "stt_waiting_too_long": stt_waiting_too_long,
            "last_transcription_at": self._voice_input_last_transcription_at,
            "last_transcription_chars": self._voice_input_last_transcription_chars,
            "track_enabled": self._voice_input_track_enabled,
            "track_reason": self._voice_input_track_reason,
            "reason_codes": _dedupe_reason_codes(tuple(reason_codes)),
        }

    def current_active_listening_state(self) -> dict[str, Any]:
        """Return public-safe active-listening state for the browser UI."""
        phase = self._voice_input_active_listening_phase
        turn_duration_ms = self._current_voice_turn_duration_ms()
        if (
            phase == BrowserActiveListeningPhase.SPEECH_STARTED.value
            and turn_duration_ms is not None
            and turn_duration_ms >= 700
        ):
            phase = BrowserActiveListeningPhase.SPEECH_CONTINUING.value
        reason_codes = [
            "active_listening:v1",
            "active_listening:available"
            if self.runtime_kind in {"browser", "voice"}
            else "active_listening:unavailable",
            f"active_listening:{phase}",
        ]
        if self._voice_input_partial_available:
            reason_codes.append("active_listening:partial_available")
        else:
            reason_codes.append("active_listening:partial_unavailable")
        if self._voice_input_active_topics:
            reason_codes.append("active_listening:topics_detected")
        if self._voice_input_active_constraints:
            reason_codes.append("active_listening:constraints_detected")
        return BrowserActiveListeningSnapshot(
            available=self.runtime_kind in {"browser", "voice"},
            phase=phase,
            partial_available=self._voice_input_partial_available,
            partial_transcript_chars=self._voice_input_last_partial_chars,
            final_transcript_chars=(
                self._voice_input_turn_final_transcription_chars
                or self._voice_input_last_transcription_chars
            ),
            interim_transcript_count=self._voice_input_interim_transcription_count,
            final_transcript_count=self._voice_input_transcription_count,
            speech_start_count=self._voice_input_speech_start_count,
            speech_stop_count=self._voice_input_speech_stop_count,
            turn_started_at=self._voice_input_turn_started_at,
            turn_stopped_at=self._voice_input_turn_stopped_at,
            last_update_at=self._voice_input_active_listening_last_update_at,
            turn_duration_ms=turn_duration_ms,
            topics=self._voice_input_active_topics,
            constraints=self._voice_input_active_constraints,
            reason_codes=_dedupe_reason_codes(tuple(reason_codes)),
        ).as_dict()

    def current_active_listener_state_v2(self, *, profile: str | None = None) -> dict[str, Any]:
        """Return public-safe active-listener v2 state for browser actor-state."""
        phase = self._active_listener_phase_v2()
        health = self.current_voice_input_health()
        degradation_state = "ok"
        degradation_components: list[str] = []
        degradation_reasons: list[str] = ["active_listener_degradation:ok"]
        microphone_state = str(health.get("microphone_state") or "unknown")
        if self._voice_input_stt_state == "error":
            degradation_state = "error"
            degradation_components.append("stt")
            degradation_reasons = ["active_listener_degradation:error", "stt:error_observed"]
            phase = ActiveListenerPhaseV2.ERROR.value
        elif health.get("stt_waiting_too_long") is True:
            degradation_state = "degraded"
            degradation_components.append("stt")
            degradation_reasons = [
                "active_listener_degradation:degraded",
                "stt:waiting_too_long",
            ]
            phase = ActiveListenerPhaseV2.DEGRADED.value
        elif microphone_state in {"stalled", "no_audio_frames"}:
            degradation_state = "degraded"
            degradation_components.append("microphone")
            degradation_reasons = [
                "active_listener_degradation:degraded",
                f"microphone:{microphone_state}",
            ]
            phase = ActiveListenerPhaseV2.DEGRADED.value

        ready_to_answer = (
            phase == ActiveListenerPhaseV2.FINAL_UNDERSTANDING.value
            and self._voice_input_transcription_count > 0
        )
        if ready_to_answer:
            readiness_state = "ready"
        elif phase == ActiveListenerPhaseV2.PARTIAL_UNDERSTANDING.value:
            readiness_state = "partial"
        elif phase == ActiveListenerPhaseV2.TRANSCRIBING.value:
            readiness_state = "transcribing"
        elif phase in {
            ActiveListenerPhaseV2.LISTENING_STARTED.value,
            ActiveListenerPhaseV2.SPEECH_CONTINUING.value,
        }:
            readiness_state = "listening"
        elif degradation_state != "ok":
            readiness_state = "degraded"
        else:
            readiness_state = "not_ready"

        reason_codes = [
            "active_listener:v2",
            "active_listener:available"
            if self.runtime_kind in {"browser", "voice"}
            else "active_listener:unavailable",
            f"active_listener:{phase}",
            f"readiness:{readiness_state}",
        ]
        if self._voice_input_partial_available:
            reason_codes.append("active_listener:partial_available")
        if self._voice_input_transcription_count:
            reason_codes.append("active_listener:final_available")
        if self._voice_input_active_topics:
            reason_codes.append("active_listener:topics_detected")
        if self._voice_input_active_constraints:
            reason_codes.append("active_listener:constraints_detected")
        if self._voice_input_active_corrections:
            reason_codes.append("active_listener:corrections_detected")
        if self._voice_input_active_project_references:
            reason_codes.append("active_listener:project_references_detected")
        if self._voice_input_active_uncertainty_flags:
            reason_codes.append("active_listener:uncertainty_detected")
        return ActiveListenerStateV2(
            profile=profile or self.runtime_kind,
            language=self.language.value,
            available=self.runtime_kind in {"browser", "voice"},
            phase=phase,
            partial_available=self._voice_input_partial_available,
            final_available=self._voice_input_transcription_count > 0,
            partial_transcript_chars=self._voice_input_last_partial_chars,
            final_transcript_chars=(
                self._voice_input_turn_final_transcription_chars
                or self._voice_input_last_transcription_chars
            ),
            interim_transcript_count=self._voice_input_interim_transcription_count,
            final_transcript_count=self._voice_input_transcription_count,
            speech_start_count=self._voice_input_speech_start_count,
            speech_stop_count=self._voice_input_speech_stop_count,
            turn_started_at=self._voice_input_turn_started_at,
            turn_stopped_at=self._voice_input_turn_stopped_at,
            last_update_at=self._voice_input_active_listening_last_update_at,
            turn_duration_ms=self._current_voice_turn_duration_ms(),
            topics=self._voice_input_active_topics,
            constraints=self._voice_input_active_constraints,
            corrections=self._voice_input_active_corrections,
            project_references=self._voice_input_active_project_references,
            uncertainty_flags=self._voice_input_active_uncertainty_flags,
            ready_to_answer=ready_to_answer,
            readiness_state=readiness_state,
            degradation={
                "state": degradation_state,
                "components": degradation_components,
                "reason_codes": degradation_reasons,
            },
            reason_codes=_dedupe_reason_codes(tuple(reason_codes)),
        ).as_dict()

    def _active_listener_phase_v2(self) -> str:
        phase = self._voice_input_active_listening_phase
        turn_duration_ms = self._current_voice_turn_duration_ms()
        if (
            phase == BrowserActiveListeningPhase.SPEECH_STARTED.value
            and turn_duration_ms is not None
            and turn_duration_ms >= 700
        ):
            return ActiveListenerPhaseV2.SPEECH_CONTINUING.value
        mapping = {
            BrowserActiveListeningPhase.IDLE.value: ActiveListenerPhaseV2.IDLE.value,
            BrowserActiveListeningPhase.SPEECH_STARTED.value: (
                ActiveListenerPhaseV2.LISTENING_STARTED.value
            ),
            BrowserActiveListeningPhase.SPEECH_CONTINUING.value: (
                ActiveListenerPhaseV2.SPEECH_CONTINUING.value
            ),
            BrowserActiveListeningPhase.SPEECH_STOPPED.value: ActiveListenerPhaseV2.TRANSCRIBING.value,
            BrowserActiveListeningPhase.TRANSCRIBING.value: ActiveListenerPhaseV2.TRANSCRIBING.value,
            BrowserActiveListeningPhase.PARTIAL_TRANSCRIPT.value: (
                ActiveListenerPhaseV2.PARTIAL_UNDERSTANDING.value
            ),
            BrowserActiveListeningPhase.FINAL_TRANSCRIPT.value: (
                ActiveListenerPhaseV2.FINAL_UNDERSTANDING.value
            ),
            BrowserActiveListeningPhase.ERROR.value: ActiveListenerPhaseV2.ERROR.value,
        }
        return mapping.get(str(phase), ActiveListenerPhaseV2.IDLE.value)

    def register_memory_tools(self) -> ToolsSchema:
        """Register the bounded explicit memory tools for this runtime."""
        return register_memory_tools(
            llm=self.llm,
            store=self.store,
            session_resolver=self.session_resolver,
            language=self.language,
        )

    def register_robot_head_tools(self):
        """Register the correct robot-head tool surface for this runtime."""
        if self.robot_head_controller is None:
            return None
        if self.robot_head_operator_mode:
            return register_robot_head_tools(
                llm=self.llm,
                controller=self.robot_head_controller,
                catalog=self.robot_head_controller.catalog,
                language=self.language,
            )
        return register_embodied_action_tools(
            llm=self.llm,
            dispatcher=self.action_dispatcher,
            language=self.language,
        )

    def register_daily_tools(self) -> ToolsSchema:
        """Register and return the merged daily-use brain tool surface."""
        standard_tools = list(self.register_memory_tools().standard_tools)
        robot_head_tools = self.register_robot_head_tools()
        if robot_head_tools is not None:
            standard_tools.extend(robot_head_tools.standard_tools)
        return ToolsSchema(standard_tools=standard_tools)

    async def _request_bounded_plan_proposal(
        self,
        request: BrainPlanningRequest,
    ) -> BrainPlanningDraft | None:
        """Run one bounded out-of-band planning inference through the runtime-owned LLM."""
        if not hasattr(self.llm, "run_inference"):
            return None
        packet = self.compiler.compile_packet(
            latest_user_text=request.goal.title,
            task=BrainContextTask.PLANNING,
        )
        prompt_payload = self._planning_request_payload(request)
        planning_instruction = (
            "You are Blink's bounded planner. Return strict JSON only with keys "
            "`summary`, `remaining_steps`, `assumptions`, `missing_inputs`, "
            "`review_policy`, `procedural_origin`, `selected_skill_id`, "
            "`selected_skill_support_trace_ids`, `selected_skill_support_plan_proposal_ids`, "
            "`rejected_skills`, `delta`, and `details`. Only use registered capability ids from these families: "
            "observation, reporting, maintenance, dialogue, robot_head. "
            "For `initial_plan`, `remaining_steps` is the full plan. "
            "For `revise_tail`, `remaining_steps` must replace only the unfinished tail "
            "and must not repeat the completed prefix. "
            "If a reusable skill matches exactly, set `procedural_origin` to `skill_reuse` and "
            "set `selected_skill_id`. If you adapt one reusable skill with at most two step-level "
            "changes, set `procedural_origin` to `skill_delta`, set `selected_skill_id`, and include "
            "a bounded `delta`. Otherwise set `procedural_origin` to `fresh_draft`. "
            "Do not select advisory or rejected skills. Do not weaken a selected skill's review policy."
        )
        system_instruction = f"{packet.prompt}\n\n{planning_instruction}"
        try:
            from blink.processors.aggregators.llm_context import LLMContext
        except ModuleNotFoundError:
            return None
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True),
                }
            ]
        )
        response = await self.llm.run_inference(
            context,
            system_instruction=system_instruction,
        )
        return self._parse_planning_response(response)

    def _planning_request_payload(self, request: BrainPlanningRequest) -> dict[str, object]:
        """Render one bounded planning request into strict JSON-friendly input."""
        goal = request.goal
        commitment = request.commitment
        current_steps = [step.as_dict() for step in goal.steps]
        return {
            "request_kind": request.request_kind,
            "goal": {
                "goal_id": goal.goal_id,
                "title": goal.title,
                "intent": goal.intent,
                "goal_family": goal.goal_family,
                "details": goal.details,
                "plan_revision": goal.plan_revision,
            },
            "commitment": (
                {
                    "commitment_id": commitment.commitment_id,
                    "title": commitment.title,
                    "status": commitment.status,
                    "goal_family": commitment.goal_family,
                    "intent": commitment.intent,
                    "plan_revision": commitment.plan_revision,
                    "details": commitment.details,
                }
                if commitment is not None
                else None
            ),
            "completed_prefix": [step.as_dict() for step in request.completed_prefix],
            "current_steps": current_steps,
            "remaining_tail": current_steps[len(request.completed_prefix) :],
            "supersedes_plan_proposal_id": request.supersedes_plan_proposal_id,
            "procedural_skills": split_planning_skill_candidates(request.skill_candidates),
        }

    @staticmethod
    def _parse_planning_response(response: str | None) -> BrainPlanningDraft | None:
        """Parse one strict JSON planning response into a typed draft."""
        if not response:
            return None
        raw = response.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return BrainPlanningDraft.from_dict(payload)

    def bind_context(self, context: LLMContext):
        """Bind the shared runtime context after the final tool schema is known."""
        from blink.brain.processors import (
            BrainTextAssistantTurnFinalizer,
            BrainTextAssistantTurnStartRecorder,
            TurnRecorderProcessor,
        )

        self.context = context
        self.post_context_processors = []
        self.post_aggregation_processors = []

        if self.runtime_kind == "text":
            self.post_context_processors = [
                BrainTextAssistantTurnStartRecorder(
                    store=self.store,
                    session_resolver=self.session_resolver,
                )
            ]
            self.post_aggregation_processors = [
                BrainTextAssistantTurnFinalizer(
                    store=self.store,
                    session_resolver=self.session_resolver,
                    context=context,
                    consolidator=self.consolidator,
                    executive=self.executive,
                    memory_use_trace_resolver=self._pending_memory_use_trace,
                    memory_use_trace_committed_callback=self._mark_memory_use_trace_committed,
                )
            ]
            return

        self.post_context_processors = [
            TurnRecorderProcessor(
                store=self.store,
                session_resolver=self.session_resolver,
                context=context,
                consolidator=self.consolidator,
                executive=self.executive,
                memory_use_trace_resolver=self._pending_memory_use_trace,
                memory_use_trace_committed_callback=self._mark_memory_use_trace_committed,
            )
        ]

    def set_capability_side_effect_sink(self, sink: CapabilitySideEffectSink | None):
        """Set the narrow side-effect sink used by bounded internal capabilities."""
        self._capability_side_effect_sink = sink
        self.executive.set_capability_side_effect_sink(sink)

    def start_background_maintenance(self):
        """Start periodic idle-gated background reflection for this runtime."""
        self.reflection_scheduler.start()
        self._refresh_reevaluation_alarm()

    def stop_background_maintenance(self, *, final_catchup: bool = False):
        """Stop background reflection and optionally run one final catch-up cycle."""
        self._cancel_reevaluation_alarm()
        return self.reflection_scheduler.stop(final_catchup=final_catchup)

    def run_presence_director_pass(self):
        """Run one explicit bounded PresenceDirector pass for this runtime."""
        return self.executive.run_presence_director_pass()

    async def run_commitment_wake_router(
        self,
        *,
        boundary_kind: str,
        source_event=None,
        store: BrainStore | None = None,
        session_ids=None,
    ):
        """Run one explicit bounded commitment wake-router pass for this runtime."""
        if store is None:
            return await self.executive.run_commitment_wake_router(
                boundary_kind=boundary_kind,
                source_event=source_event,
            )
        resolved_session_ids = session_ids or self.session_resolver()
        temporary_executive = BrainExecutive(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            capability_registry=self.capability_registry,
            presence_scope_key=self.presence_scope_key,
            context_surface_builder=self.compiler.context_surface_builder,
            context_selector=self.compiler.context_selector,
            capability_side_effect_sink=self._capability_side_effect_sink,
            planning_callback=self._request_bounded_plan_proposal,
        )
        return await temporary_executive.run_commitment_wake_router(
            boundary_kind=boundary_kind,
            source_event=source_event,
        )

    def run_presence_director_reevaluation(
        self,
        *,
        trigger: BrainReevaluationTrigger,
        store: BrainStore | None = None,
        session_ids=None,
    ):
        """Run one explicit bounded reevaluation pass for this runtime."""
        if trigger.kind == BrainReevaluationConditionKind.TIME_REACHED.value:
            return self.run_presence_director_expiry_cleanup(
                trigger=trigger,
                store=store,
                session_ids=session_ids,
            )
        if store is None:
            return self.executive.run_presence_director_reevaluation(trigger)
        resolved_session_ids = session_ids or self.session_resolver()
        temporary_executive = BrainExecutive(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            capability_registry=self.capability_registry,
            presence_scope_key=self.presence_scope_key,
            context_surface_builder=self.compiler.context_surface_builder,
            context_selector=self.compiler.context_selector,
            capability_side_effect_sink=self._capability_side_effect_sink,
            planning_callback=self._request_bounded_plan_proposal,
        )
        return temporary_executive.run_presence_director_reevaluation(trigger)

    def run_presence_director_expiry_cleanup(
        self,
        *,
        trigger: BrainReevaluationTrigger,
        store: BrainStore | None = None,
        session_ids=None,
    ):
        """Run one explicit bounded expiry-cleanup pass for this runtime."""
        if store is None:
            return self.executive.run_presence_director_expiry_cleanup(trigger)
        resolved_session_ids = session_ids or self.session_resolver()
        temporary_executive = BrainExecutive(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            capability_registry=self.capability_registry,
            presence_scope_key=self.presence_scope_key,
            context_surface_builder=self.compiler.context_surface_builder,
            context_selector=self.compiler.context_selector,
            capability_side_effect_sink=self._capability_side_effect_sink,
            planning_callback=self._request_bounded_plan_proposal,
        )
        return temporary_executive.run_presence_director_expiry_cleanup(trigger)

    def propose_candidate_goal(
        self,
        *,
        candidate_goal: BrainCandidateGoal,
        source: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        store: BrainStore | None = None,
        session_ids=None,
    ):
        """Append one candidate goal and run the bounded director pass.

        When ``store`` is provided, this method uses a temporary executive bound to
        that store so background scheduler threads never touch the runtime-owned
        SQLite connection directly.
        """
        if store is None:
            return self.executive.propose_candidate_goal(
                candidate_goal=candidate_goal,
                source=source,
                correlation_id=correlation_id,
                causal_parent_id=causal_parent_id,
            )
        resolved_session_ids = session_ids or self.session_resolver()
        temporary_executive = BrainExecutive(
            store=store,
            session_resolver=lambda: resolved_session_ids,
            capability_registry=self.capability_registry,
            presence_scope_key=self.presence_scope_key,
            context_surface_builder=self.compiler.context_surface_builder,
            context_selector=self.compiler.context_selector,
            capability_side_effect_sink=self._capability_side_effect_sink,
            planning_callback=self._request_bounded_plan_proposal,
        )
        return temporary_executive.propose_candidate_goal(
            candidate_goal=candidate_goal,
            source=source,
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
        )

    def _cancel_reevaluation_alarm(self):
        """Cancel the one-shot reevaluation alarm, if one is armed."""
        if self._reevaluation_alarm_task is not None:
            self._reevaluation_alarm_task.cancel()
        self._reevaluation_alarm_task = None
        self._reevaluation_alarm_key = None

    def _refresh_reevaluation_alarm(self):
        """Arm one nearest-deadline reevaluation wake without polling aggressively."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        wake = self._next_reevaluation_wake()
        if wake is None:
            self._cancel_reevaluation_alarm()
            return
        kind, deadline = wake
        key = (kind, deadline.isoformat())
        if (
            self._reevaluation_alarm_task is not None
            and not self._reevaluation_alarm_task.done()
            and self._reevaluation_alarm_key == key
        ):
            return
        self._cancel_reevaluation_alarm()
        self._reevaluation_alarm_key = key
        self._reevaluation_alarm_task = loop.create_task(
            self._run_reevaluation_alarm(kind=kind, deadline=deadline),
            name=f"blink-reevaluation-alarm:{kind}",
        )

    def _next_reevaluation_wake(self) -> tuple[str, datetime] | None:
        """Return the nearest candidate reevaluation wake currently pending."""
        session_ids = self.session_resolver()
        ledger = self.store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        deadlines: list[tuple[str, datetime]] = []
        for candidate in ledger.current_candidates:
            expires_at = _parse_ts(candidate.expires_at)
            if expires_at is not None:
                deadlines.append((BrainReevaluationConditionKind.TIME_REACHED.value, expires_at))
            if (
                candidate.expected_reevaluation_condition_kind
                == BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value
            ):
                details = dict(candidate.expected_reevaluation_condition_details or {})
                not_before = _parse_ts(str(details.get("not_before", "")).strip() or None)
                if not_before is not None:
                    deadlines.append(
                        (BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value, not_before)
                    )
        if not deadlines:
            return None
        return min(deadlines, key=lambda item: item[1])

    async def _run_reevaluation_alarm(self, *, kind: str, deadline: datetime):
        """Sleep until the nearest reevaluation wake and then fire one bounded pass."""
        current_task = asyncio.current_task()
        try:
            delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
            if delay > 0:
                await asyncio.sleep(delay)
            self._reevaluation_alarm_task = None
            self._reevaluation_alarm_key = None
            details = {"deadline": deadline.isoformat()}
            if kind == BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value:
                details["not_before"] = deadline.isoformat()
            trigger = BrainReevaluationTrigger(
                kind=kind,
                summary="Reevaluate held candidates when the next bounded wake opens.",
                details=details,
                source_event_type="runtime.reevaluation_alarm",
                ts=deadline.isoformat(),
            )
            if kind == BrainReevaluationConditionKind.TIME_REACHED.value:
                self.executive.run_presence_director_expiry_cleanup(trigger)
            else:
                self.executive.run_presence_director_reevaluation(trigger)
        except asyncio.CancelledError:
            raise
        finally:
            if self._reevaluation_alarm_task is current_task:
                self._reevaluation_alarm_task = None
                self._reevaluation_alarm_key = None

    def note_vision_connected(self, connected: bool):
        """Update the current vision presence."""
        snapshot = self.store.get_body_state_projection(scope_key=self.presence_scope_key)
        snapshot.runtime_kind = self.runtime_kind
        snapshot.robot_head_enabled = self.robot_head_controller is not None
        snapshot.robot_head_mode = (
            self.robot_head_controller.driver_mode if self.robot_head_controller else "none"
        )
        snapshot.robot_head_available = self.robot_head_controller is not None
        snapshot.vision_enabled = self.vision_enabled
        snapshot.vision_connected = connected
        snapshot.camera_track_state = "waiting_for_frame" if connected else "disconnected"
        snapshot.sensor_health_reason = (
            "camera_waiting_for_frame" if connected else "camera_disconnected"
        )
        snapshot.perception_disabled = not self.continuous_perception_enabled
        snapshot.perception_unreliable = bool(connected)
        snapshot.last_fresh_frame_at = None
        snapshot.frame_age_ms = None
        snapshot.camera_disconnected = bool(self.vision_enabled and not connected)
        self._update_presence(snapshot)
        session_ids = self.session_resolver()
        self.store.append_brain_event(
            event_type=BrainEventType.SCENE_CHANGED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="runtime",
            payload={
                "presence_scope_key": self.presence_scope_key,
                "camera_connected": connected,
                "camera_fresh": False,
                "camera_track_state": "waiting_for_frame" if connected else "disconnected",
                "sensor_health_reason": (
                    "camera_waiting_for_frame" if connected else "camera_disconnected"
                ),
                "person_present": "uncertain" if connected else "absent",
                "attention_to_camera": "unknown",
                "engagement_state": "unknown" if connected else "away",
                "scene_change": "changed",
                "summary": "camera connected" if connected else "camera disconnected",
                "confidence": 1.0,
                "observed_at": datetime.now(UTC).isoformat(),
            },
        )

    def note_perception_availability(
        self,
        *,
        enabled: bool,
        unreliable: bool = False,
        unavailable: bool = False,
        detection_backend: str | None = None,
        enrichment_available: bool | None = None,
    ):
        """Update perception-related body-state availability flags."""
        snapshot = self.store.get_body_state_projection(scope_key=self.presence_scope_key)
        snapshot.vision_enabled = self.vision_enabled
        snapshot.vision_unavailable = unavailable
        snapshot.perception_unreliable = unreliable
        snapshot.perception_disabled = not enabled
        if detection_backend is not None:
            snapshot.detection_backend = detection_backend
        details = dict(snapshot.details)
        if enrichment_available is not None:
            details["vision_enrichment_available"] = enrichment_available
        snapshot.details = details
        self._update_presence(snapshot)

    def close(self):
        """Close runtime-owned resources that are safe to tear down locally."""
        session_ids = self.session_resolver()
        self.consolidator.run_once(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        result = self.stop_background_maintenance(final_catchup=True)
        if result is None:
            self.run_reflection_once(trigger="runtime_close")
        if self._owns_store:
            self.store.close()

    def run_reflection_once(self, *, trigger: str = "manual"):
        """Run one bounded reflection pass after hot-path consolidation."""
        session_ids = self.session_resolver()
        return self.reflection_engine.run_once(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            session_ids=session_ids,
            trigger=trigger,
        )

    def _update_presence(self, snapshot: BrainPresenceSnapshot):
        """Persist the latest presence snapshot for this runtime."""
        session_ids = self.session_resolver()
        snapshot = normalize_presence_snapshot(snapshot)
        self.store.append_brain_event(
            event_type=BrainEventType.BODY_STATE_UPDATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="runtime",
            payload={
                "scope_key": self.presence_scope_key,
                "snapshot": snapshot.as_dict(),
            },
        )

    def _migrate_legacy_memory(self):
        """Import legacy JSON fact memory into the canonical SQLite store once."""
        from blink.cli.local_brain import LocalBrainMemoryStore

        default_legacy_path = Path.home() / ".cache" / "blink" / "local_brain" / "memory.json"
        legacy_path = Path(
            __import__("os").getenv(local_env_name("BRAIN_MEMORY_PATH"), str(default_legacy_path))
        )
        if not legacy_path.exists():
            return
        facts = LocalBrainMemoryStore(path=legacy_path).facts()
        if not facts:
            return
        legacy_pairs = []
        for fact in facts:
            if fact.category == "user_name":
                namespace = "profile.name"
            elif fact.category == "user_role":
                namespace = "profile.role"
            elif fact.category == "user_origin":
                namespace = "profile.origin"
            elif fact.category == "user_like":
                namespace = "preference.like"
            else:
                namespace = "preference.dislike"
            legacy_pairs.append((namespace, fact.statement))
        self.store.migrate_legacy_local_facts(
            user_id=self.session_resolver().user_id,
            facts=legacy_pairs,
        )


def build_session_resolver(
    *,
    runtime_kind: str,
    active_client: Optional[dict[str, Optional[str]]] = None,
    user_id: str | None = None,
    use_active_client_id: bool = False,
):
    """Build a session resolver closure for one local runtime."""
    active_client = active_client or {"id": None}
    configured_user_id = (
        user_id or os.getenv(local_env_name("BRAIN_USER_ID")) or "local_primary"
    ).strip() or "local_primary"

    def resolver():
        client_id = configured_user_id
        if use_active_client_id:
            client_id = active_client.get("id") or configured_user_id
        return resolve_brain_session_ids(
            runtime_kind=runtime_kind,
            client_id=client_id,
            default_user_id=configured_user_id,
        )

    return resolver

"""Provider-light capability registry assembly for planning and executive seams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from blink.brain.autonomy import BrainInitiativeClass
from blink.brain.capabilities import (
    BaseCapabilityInput,
    CapabilityAssistantUtterance,
    CapabilityDefinition,
    CapabilityExecutionContext,
    CapabilityExecutionResult,
    CapabilityFamily,
    CapabilityInitiativePolicy,
    CapabilityRegistry,
    CapabilityUserTurnPolicy,
)
from blink.brain.projections import BrainGoalFamily
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.brain.actions import EmbodiedActionEngine


class ObservationInspectPresenceInput(BaseCapabilityInput):
    """Structured input for symbolic presence inspection."""

    presence_scope_key: str
    scene_candidate: dict[str, Any] = Field(default_factory=dict)


class ObservationInspectCameraHealthInput(BaseCapabilityInput):
    """Structured input for camera-health inspection."""

    presence_scope_key: str
    scene_candidate: dict[str, Any] = Field(default_factory=dict)


class DialogueEmitBriefReengagementInput(BaseCapabilityInput):
    """Structured input for the bounded spoken re-engagement path."""

    presence_scope_key: str
    scene_candidate: dict[str, Any] = Field(default_factory=dict)


class MaintenanceReviewMemoryHealthInput(BaseCapabilityInput):
    """Structured input for memory-health review capabilities."""

    maintenance: dict[str, Any] = Field(default_factory=dict)


class MaintenanceReviewSchedulerBackpressureInput(BaseCapabilityInput):
    """Structured input for reflection-backpressure review capabilities."""

    maintenance: dict[str, Any] = Field(default_factory=dict)


class ReportingPresenceEventInput(BaseCapabilityInput):
    """Structured input for scene-event reporting."""

    scene_candidate: dict[str, Any] = Field(default_factory=dict)


class ReportingCommitmentWakeInput(BaseCapabilityInput):
    """Structured input for commitment-wake reporting."""

    commitment_wake: dict[str, Any] = Field(default_factory=dict)


class ReportingMaintenanceNoteInput(BaseCapabilityInput):
    """Structured input for maintenance-note reporting."""

    maintenance: dict[str, Any] = Field(default_factory=dict)


def register_internal_capability_families(
    *,
    registry: CapabilityRegistry,
    language: Language,
):
    """Register bounded internal capability families on an existing registry."""

    def _require_store(context: CapabilityExecutionContext, capability_id: str) -> BrainStore | None:
        if context.store is None:
            return None
        return context.store

    async def inspect_presence_executor(
        inputs: ObservationInspectPresenceInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        store = _require_store(context, "observation.inspect_presence_state")
        if store is None:
            return CapabilityExecutionResult.failed(
                capability_id="observation.inspect_presence_state",
                summary="Presence state is unavailable without a bound brain store.",
                error_code="store_unavailable",
            )
        body = store.get_body_state_projection(scope_key=inputs.presence_scope_key)
        scene = store.get_scene_state_projection(scope_key=inputs.presence_scope_key)
        engagement = store.get_engagement_state_projection(scope_key=inputs.presence_scope_key)
        output = {
            "runtime_kind": body.runtime_kind,
            "camera_track_state": body.camera_track_state,
            "frame_age_ms": body.frame_age_ms,
            "detection_backend": body.detection_backend,
            "detection_confidence": body.detection_confidence,
            "sensor_health_reason": body.sensor_health_reason,
            "person_present": scene.person_present,
            "attention_to_camera": engagement.attention_to_camera,
            "engagement_state": engagement.engagement_state,
            "scene_candidate": dict(inputs.scene_candidate),
        }
        if language.value.lower().startswith(("zh", "cmn")):
            summary = (
                f"存在检测={scene.person_present}，注视={engagement.attention_to_camera}，"
                f"参与度={engagement.engagement_state}。"
            )
        else:
            summary = (
                f"Presence={scene.person_present}, attention={engagement.attention_to_camera}, "
                f"engagement={engagement.engagement_state}."
            )
        return CapabilityExecutionResult.success(
            capability_id="observation.inspect_presence_state",
            summary=summary,
            output=output,
            metadata={"operator_visible": True},
        )

    async def inspect_camera_health_executor(
        inputs: ObservationInspectCameraHealthInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        store = _require_store(context, "observation.inspect_camera_health")
        if store is None:
            return CapabilityExecutionResult.failed(
                capability_id="observation.inspect_camera_health",
                summary="Camera health is unavailable without a bound brain store.",
                error_code="store_unavailable",
            )
        body = store.get_body_state_projection(scope_key=inputs.presence_scope_key)
        output = {
            "runtime_kind": body.runtime_kind,
            "camera_track_state": body.camera_track_state,
            "sensor_health": body.sensor_health,
            "sensor_health_reason": body.sensor_health_reason,
            "frame_age_ms": body.frame_age_ms,
            "detection_backend": body.detection_backend,
            "detection_confidence": body.detection_confidence,
            "recovery_in_progress": body.recovery_in_progress,
            "recovery_attempts": body.recovery_attempts,
            "scene_candidate": dict(inputs.scene_candidate),
        }
        if language.value.lower().startswith(("zh", "cmn")):
            summary = (
                f"摄像头轨道={body.camera_track_state}，健康状态={body.sensor_health}，"
                f"原因={body.sensor_health_reason or '无'}。"
            )
        else:
            summary = (
                f"Camera track={body.camera_track_state}, sensor_health={body.sensor_health}, "
                f"reason={body.sensor_health_reason or 'none'}."
            )
        return CapabilityExecutionResult.success(
            capability_id="observation.inspect_camera_health",
            summary=summary,
            output=output,
            metadata={"operator_visible": True},
        )

    async def brief_reengagement_executor(
        inputs: DialogueEmitBriefReengagementInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        if context.side_effect_sink is None:
            return CapabilityExecutionResult.blocked(
                capability_id="dialogue.emit_brief_reengagement",
                summary="Brief re-engagement is unavailable without a dialogue sink.",
                error_code="dialogue_sink_unavailable",
                retryable=True,
            )
        text = "欢迎回来。" if language.value.lower().startswith(("zh", "cmn")) else "Welcome back."
        await context.side_effect_sink.emit_assistant_utterance(
            CapabilityAssistantUtterance(
                text=text,
                metadata={
                    "goal_intent": context.goal_intent,
                    "initiative_class": context.initiative_class,
                    "scene_candidate": dict(inputs.scene_candidate),
                },
            )
        )
        return CapabilityExecutionResult.success(
            capability_id="dialogue.emit_brief_reengagement",
            summary=text,
            output={"utterance": text, "scene_candidate": dict(inputs.scene_candidate)},
            metadata={"operator_visible": True, "proactive_dialogue": True},
        )

    async def review_memory_health_executor(
        inputs: MaintenanceReviewMemoryHealthInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        maintenance = dict(inputs.maintenance)
        if context.store is not None and not any(
            key in maintenance for key in ("report_id", "cycle_id", "status", "findings")
        ):
            relationship_scope_id = f"{context.session_ids.agent_id}:{context.session_ids.user_id}"
            report = context.store.latest_memory_health_report(
                scope_type="relationship",
                scope_id=relationship_scope_id,
            )
            if report is not None:
                maintenance = {
                    "report_id": report.report_id,
                    "cycle_id": report.cycle_id,
                    "status": report.status,
                    "score": report.score,
                    "findings": report.findings,
                }
        findings = list(maintenance.get("findings", []))
        status = str(maintenance.get("status", "unknown")).strip() or "unknown"
        if language.value.lower().startswith(("zh", "cmn")):
            summary = f"记忆健康状态={status}，发现 {len(findings)} 个关注项。"
        else:
            summary = f"Memory health status={status} with {len(findings)} findings."
        return CapabilityExecutionResult.success(
            capability_id="maintenance.review_memory_health",
            summary=summary,
            output={
                "report_id": maintenance.get("report_id"),
                "cycle_id": maintenance.get("cycle_id"),
                "status": status,
                "findings": findings,
            },
            metadata={"operator_visible": True},
        )

    async def review_scheduler_backpressure_executor(
        inputs: MaintenanceReviewSchedulerBackpressureInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        maintenance = dict(inputs.maintenance)
        recent_skip_count = int(maintenance.get("recent_skip_count", 0) or 0)
        if context.store is not None and recent_skip_count <= 0:
            cycles = context.store.list_reflection_cycles(
                user_id=context.session_ids.user_id,
                thread_id=context.session_ids.thread_id,
                statuses=("skipped",),
                limit=16,
            )
            recent_skip_count = max(
                recent_skip_count,
                sum(1 for record in cycles if record.trigger == "timer" and record.skip_reason == "thread_active"),
            )
        if language.value.lower().startswith(("zh", "cmn")):
            summary = f"后台反思因线程活跃被跳过 {recent_skip_count} 次。"
        else:
            summary = f"Background reflection was skipped {recent_skip_count} times because the thread stayed active."
        return CapabilityExecutionResult.success(
            capability_id="maintenance.review_scheduler_backpressure",
            summary=summary,
            output={
                "skip_reason": maintenance.get("skip_reason", "thread_active"),
                "recent_skip_count": recent_skip_count,
            },
            metadata={"operator_visible": True},
        )

    async def record_presence_event_executor(
        inputs: ReportingPresenceEventInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        candidate_type = str(context.goal_intent or "").removeprefix("autonomy.") or "presence_event"
        return CapabilityExecutionResult.success(
            capability_id="reporting.record_presence_event",
            summary=f"Recorded presence event for {candidate_type}.",
            output={"candidate_type": candidate_type, "scene_candidate": dict(inputs.scene_candidate)},
            metadata={"operator_visible": True},
        )

    async def record_commitment_wake_executor(
        inputs: ReportingCommitmentWakeInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        wake = dict(inputs.commitment_wake)
        wake_kind = str(wake.get("wake_kind", "unknown")).strip() or "unknown"
        summary = (
            f"记录承诺唤醒：{wake_kind}。"
            if language.value.lower().startswith(("zh", "cmn"))
            else f"Recorded commitment wake: {wake_kind}."
        )
        return CapabilityExecutionResult.success(
            capability_id="reporting.record_commitment_wake",
            summary=summary,
            output={"commitment_wake": wake},
            metadata={"operator_visible": True},
        )

    async def record_maintenance_note_executor(
        inputs: ReportingMaintenanceNoteInput,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        maintenance = dict(inputs.maintenance)
        note_kind = str(context.goal_intent or "").removeprefix("autonomy.") or "maintenance_note"
        summary = (
            f"记录维护说明：{note_kind}。"
            if language.value.lower().startswith(("zh", "cmn"))
            else f"Recorded maintenance note: {note_kind}."
        )
        return CapabilityExecutionResult.success(
            capability_id="reporting.record_maintenance_note",
            summary=summary,
            output={"note_kind": note_kind, "maintenance": maintenance},
            metadata={"operator_visible": True},
        )

    registry.register(
        CapabilityDefinition(
            capability_id="observation.inspect_presence_state",
            family=CapabilityFamily.OBSERVATION.value,
            description="Read current symbolic presence, attention, and engagement state.",
            input_model=ObservationInspectPresenceInput,
            sensitivity="safe",
            executor=inspect_presence_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                allowed_initiative_classes=(
                    BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
                    BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
                ),
                user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="observation.inspect_camera_health",
            family=CapabilityFamily.OBSERVATION.value,
            description="Read current camera-track and sensor-health state.",
            input_model=ObservationInspectCameraHealthInput,
            sensitivity="safe",
            executor=inspect_camera_health_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                allowed_initiative_classes=(BrainInitiativeClass.INSPECT_ONLY.value,),
                user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="dialogue.emit_brief_reengagement",
            family=CapabilityFamily.DIALOGUE.value,
            description="Emit one deterministic brief re-engagement utterance.",
            input_model=DialogueEmitBriefReengagementInput,
            sensitivity="safe",
            executor=brief_reengagement_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                allowed_initiative_classes=(BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,),
                user_turn_policy=CapabilityUserTurnPolicy.REQUIRES_GAP.value,
                operator_visible=True,
                proactive_dialogue=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="maintenance.review_memory_health",
            family=CapabilityFamily.MAINTENANCE.value,
            description="Review the latest memory-health findings.",
            input_model=MaintenanceReviewMemoryHealthInput,
            sensitivity="safe",
            executor=review_memory_health_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.MEMORY_MAINTENANCE.value,),
                allowed_initiative_classes=(BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,),
                user_turn_policy=CapabilityUserTurnPolicy.REQUIRES_GAP.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="maintenance.review_scheduler_backpressure",
            family=CapabilityFamily.MAINTENANCE.value,
            description="Review reflection backpressure caused by repeated timer skips.",
            input_model=MaintenanceReviewSchedulerBackpressureInput,
            sensitivity="safe",
            executor=review_scheduler_backpressure_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.MEMORY_MAINTENANCE.value,),
                allowed_initiative_classes=(BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,),
                user_turn_policy=CapabilityUserTurnPolicy.REQUIRES_GAP.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="reporting.record_presence_event",
            family=CapabilityFamily.REPORTING.value,
            description="Emit structured operator-facing output for one presence event.",
            input_model=ReportingPresenceEventInput,
            sensitivity="safe",
            executor=record_presence_event_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                allowed_initiative_classes=(
                    BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
                    BrainInitiativeClass.INSPECT_ONLY.value,
                    BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
                ),
                user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="reporting.record_commitment_wake",
            family=CapabilityFamily.REPORTING.value,
            description="Emit structured operator-facing output for one commitment wake.",
            input_model=ReportingCommitmentWakeInput,
            sensitivity="safe",
            executor=record_commitment_wake_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(
                    BrainGoalFamily.CONVERSATION.value,
                    BrainGoalFamily.MEMORY_MAINTENANCE.value,
                ),
                allowed_initiative_classes=(
                    BrainInitiativeClass.DEFER_UNTIL_USER_TURN.value,
                    BrainInitiativeClass.INSPECT_ONLY.value,
                    BrainInitiativeClass.MAINTENANCE_INTERNAL.value,
                ),
                user_turn_policy=CapabilityUserTurnPolicy.REQUIRES_GAP.value,
                operator_visible=True,
            ),
        )
    )
    registry.register(
        CapabilityDefinition(
            capability_id="reporting.record_maintenance_note",
            family=CapabilityFamily.REPORTING.value,
            description="Emit structured operator-facing output for one maintenance review.",
            input_model=ReportingMaintenanceNoteInput,
            sensitivity="safe",
            executor=record_maintenance_note_executor,
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.MEMORY_MAINTENANCE.value,),
                allowed_initiative_classes=(BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,),
                user_turn_policy=CapabilityUserTurnPolicy.REQUIRES_GAP.value,
                operator_visible=True,
            ),
        )
    )


def build_brain_capability_registry(
    *,
    language: Language,
    action_engine: EmbodiedActionEngine | None = None,
) -> CapabilityRegistry:
    """Build the bounded Blink capability registry for runtime execution."""
    if action_engine is not None:
        from blink.brain.actions import build_embodied_capability_registry

        registry = build_embodied_capability_registry(action_engine=action_engine)
    else:
        registry = CapabilityRegistry()
    register_internal_capability_families(registry=registry, language=language)
    return registry

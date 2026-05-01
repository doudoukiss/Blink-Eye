"""Typed projection surfaces for Blink's brain package."""

from __future__ import annotations

import json
from typing import Any

from blink.brain.core.projections import (
    ACTIVE_SITUATION_MODEL_PROJECTION,
    ADAPTER_GOVERNANCE_PROJECTION,
    AGENDA_PROJECTION,
    AUTONOMY_LEDGER_PROJECTION,
    BODY_STATE_PROJECTION,
    CLAIM_GOVERNANCE_PROJECTION,
    COMMITMENT_PROJECTION,
    COUNTERFACTUAL_REHEARSAL_PROJECTION,
    EMBODIED_EXECUTIVE_PROJECTION,
    ENGAGEMENT_STATE_PROJECTION,
    HEARTBEAT_PROJECTION,
    PRACTICE_DIRECTOR_PROJECTION,
    PREDICTIVE_WORLD_MODEL_PROJECTION,
    PRIVATE_WORKING_MEMORY_PROJECTION,
    RELATIONSHIP_STATE_PROJECTION,
    SCENE_STATE_PROJECTION,
    SCENE_WORLD_STATE_PROJECTION,
    SKILL_EVIDENCE_PROJECTION,
    SKILL_GOVERNANCE_PROJECTION,
    WORKING_CONTEXT_PROJECTION,
    BrainActionOutcomeComparisonRecord,
    BrainActionRehearsalRequest,
    BrainActionRehearsalResult,
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainAgendaProjection,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCalibrationBucket,
    BrainClaimCurrentnessStatus,
    BrainClaimGovernanceProjection,
    BrainClaimGovernanceRecord,
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainCommitmentProjection,
    BrainCommitmentRecord,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainCounterfactualCalibrationSummary,
    BrainCounterfactualEvaluationRecord,
    BrainCounterfactualRehearsalKind,
    BrainCounterfactualRehearsalProjection,
    BrainEmbodiedActionEnvelope,
    BrainEmbodiedDispatchDisposition,
    BrainEmbodiedExecutionTrace,
    BrainEmbodiedExecutiveProjection,
    BrainEmbodiedExecutorKind,
    BrainEmbodiedIntent,
    BrainEmbodiedIntentKind,
    BrainEmbodiedRecoveryRecord,
    BrainEmbodiedTraceStatus,
    BrainEngagementStateProjection,
    BrainGoal,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainGoalStep,
    BrainGovernanceReasonCode,
    BrainHeartbeatProjection,
    BrainObservedActionOutcomeKind,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainPredictionCalibrationSummary,
    BrainPredictionConfidenceBand,
    BrainPredictionKind,
    BrainPredictionRecord,
    BrainPredictionResolutionKind,
    BrainPredictionSubjectKind,
    BrainPredictiveWorldModelProjection,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainRehearsalDecisionRecommendation,
    BrainRelationshipStateProjection,
    BrainSceneStateProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
    BrainWakeCondition,
    BrainWakeConditionKind,
    BrainWorkingContextProjection,
)
from blink.transcriptions.language import Language


def _render_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_working_context_summary(
    projection: BrainWorkingContextProjection,
    language: Language,
) -> str:
    """Render the working-context projection into prompt-safe text."""
    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            [
                f"最近用户输入: {projection.last_user_text or '无'}",
                f"最近助手回复: {projection.last_assistant_text or '无'}",
                f"最近工具: {projection.last_tool_name or '无'}",
                f"最近工具结果: {_render_value(projection.last_tool_result) or '无'}",
            ]
        )

    return "\n".join(
        [
            f"Latest user text: {projection.last_user_text or 'None'}",
            f"Latest assistant text: {projection.last_assistant_text or 'None'}",
            f"Latest tool: {projection.last_tool_name or 'None'}",
            f"Latest tool result: {_render_value(projection.last_tool_result) or 'None'}",
        ]
    )


def render_private_working_memory_summary(
    projection: BrainPrivateWorkingMemoryProjection,
    language: Language,
    *,
    max_records: int = 10,
) -> str:
    """Render bounded private working-memory records into prompt-safe text."""
    records = [
        record
        for record in projection.records
        if record.state != BrainPrivateWorkingMemoryRecordState.RESOLVED.value
    ][:max_records]
    if not records:
        if language.value.lower().startswith(("zh", "cmn")):
            return "无活动私有工作记忆。"
        return "No active private working-memory records."

    def _state_tag(state: str) -> str:
        if language.value.lower().startswith(("zh", "cmn")):
            return {
                BrainPrivateWorkingMemoryRecordState.ACTIVE.value: "active",
                BrainPrivateWorkingMemoryRecordState.STALE.value: "stale",
                BrainPrivateWorkingMemoryRecordState.RESOLVED.value: "resolved",
            }.get(state, state)
        return state

    def _evidence_tag(kind: str) -> str:
        if language.value.lower().startswith(("zh", "cmn")):
            return {
                BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value: "observed",
                BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value: "derived",
                BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value: "hypothesized",
            }.get(kind, kind)
        return kind

    def _buffer_tag(kind: str) -> str:
        return kind

    lines = []
    for record in records:
        unresolved = (
            record.buffer_kind == "unresolved_uncertainty"
            and record.state != BrainPrivateWorkingMemoryRecordState.RESOLVED.value
        )
        tags = [
            _state_tag(record.state),
            _evidence_tag(record.evidence_kind),
        ]
        if unresolved:
            tags.append("unresolved")
        lines.append(
            f"- [{']['.join(tag for tag in tags if tag)}][{_buffer_tag(record.buffer_kind)}] {record.summary}"
        )
    return "\n".join(lines)


def render_scene_summary(projection: BrainSceneStateProjection, language: Language) -> str:
    """Render the scene-state projection into prompt-safe text."""
    if language.value.lower().startswith(("zh", "cmn")):
        lines = [
            f"摄像头连接: {'是' if projection.camera_connected else '否'}",
            f"轨道状态: {projection.camera_track_state}",
            f"人物在场: {projection.person_present}",
            f"场景变化: {projection.scene_change_state}",
            f"视觉摘要: {projection.last_visual_summary or '无'}",
        ]
        if projection.last_fresh_frame_at:
            lines.append(f"最近新鲜帧: {projection.last_fresh_frame_at}")
        if projection.frame_age_ms is not None:
            lines.append(f"帧年龄: {projection.frame_age_ms}ms")
        if projection.detection_backend:
            confidence = (
                f"{projection.detection_confidence:.2f}"
                if projection.detection_confidence is not None
                else "n/a"
            )
            lines.append(f"检测器: {projection.detection_backend} ({confidence})")
        if projection.sensor_health_reason:
            lines.append(f"视觉降级原因: {projection.sensor_health_reason}")
        return "\n".join(lines)

    lines = [
        f"Camera connected: {projection.camera_connected}",
        f"Track state: {projection.camera_track_state}",
        f"Person present: {projection.person_present}",
        f"Scene change: {projection.scene_change_state}",
        f"Visual summary: {projection.last_visual_summary or 'None'}",
    ]
    if projection.last_fresh_frame_at:
        lines.append(f"Last fresh frame: {projection.last_fresh_frame_at}")
    if projection.frame_age_ms is not None:
        lines.append(f"Frame age: {projection.frame_age_ms} ms")
    if projection.detection_backend:
        confidence = (
            f"{projection.detection_confidence:.2f}"
            if projection.detection_confidence is not None
            else "n/a"
        )
        lines.append(f"Detector: {projection.detection_backend} ({confidence})")
    if projection.sensor_health_reason:
        lines.append(f"Visual degraded reason: {projection.sensor_health_reason}")
    return "\n".join(lines)


def render_scene_world_state_summary(
    projection: BrainSceneWorldProjection,
    language: Language,
    *,
    max_entities: int = 6,
) -> str:
    """Render the symbolic scene-world projection into prompt-safe text."""
    entities = projection.entities[:max_entities]
    if not entities:
        if language.value.lower().startswith(("zh", "cmn")):
            return f"世界状态: 无实体；降级模式={projection.degraded_mode}"
        return f"World state: no entities; degraded_mode={projection.degraded_mode}"

    lines = []
    if language.value.lower().startswith(("zh", "cmn")):
        lines.append(f"世界状态降级模式: {projection.degraded_mode}")
        if projection.degraded_reason_codes:
            lines.append(f"降级原因: {'；'.join(projection.degraded_reason_codes)}")
        for record in entities:
            tags = [record.state, record.entity_kind]
            if record.zone_id:
                tags.append(f"zone={record.zone_id}")
            lines.append(f"- [{'|'.join(tags)}] {record.summary}")
        return "\n".join(lines)

    lines.append(f"World-state degraded mode: {projection.degraded_mode}")
    if projection.degraded_reason_codes:
        lines.append(f"Degraded reasons: {'; '.join(projection.degraded_reason_codes)}")
    for record in entities:
        tags = [record.state, record.entity_kind]
        if record.zone_id:
            tags.append(f"zone={record.zone_id}")
        lines.append(f"- [{'|'.join(tags)}] {record.summary}")
    return "\n".join(lines)


def render_engagement_summary(
    projection: BrainEngagementStateProjection,
    language: Language,
) -> str:
    """Render the engagement-state projection into prompt-safe text."""
    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            [
                f"参与状态: {projection.engagement_state}",
                f"朝向摄像头: {projection.attention_to_camera}",
                f"用户在场: {'是' if projection.user_present else '否'}",
            ]
        )

    return "\n".join(
        [
            f"Engagement state: {projection.engagement_state}",
            f"Attention to camera: {projection.attention_to_camera}",
            f"User present: {projection.user_present}",
        ]
    )


def render_relationship_state_summary(
    projection: BrainRelationshipStateProjection,
    language: Language,
) -> str:
    """Render the relationship-state projection into prompt-safe text."""
    commitments = projection.open_commitments or []
    hints = projection.interaction_style_hints or []
    boundaries = projection.boundaries or []
    teaching_modes = projection.preferred_teaching_modes or []
    analogy_domains = projection.analogy_domains or []
    misfires = projection.known_misfires or []
    if language.value.lower().startswith(("zh", "cmn")):
        lines = [
            f"最近看见用户: {projection.last_seen_at or '无'}",
            f"关系连续性: {projection.continuity_summary or '无'}",
            f"开放承诺: {'；'.join(commitments) if commitments else '无'}",
            f"互动提示: {'；'.join(hints) if hints else '无'}",
        ]
        if projection.collaboration_style or boundaries or misfires:
            boundary_text = "；".join(boundaries[:3]) if boundaries else "无"
            misfire_text = "；".join(misfires[:2]) if misfires else "无"
            lines.append(
                "关系风格: "
                f"{projection.collaboration_style or '无'}"
                f"；边界: {boundary_text}"
                f"；避免失误: {misfire_text}"
            )
        if teaching_modes or analogy_domains:
            teaching_text = "；".join(teaching_modes[:3]) if teaching_modes else "无"
            analogy_text = "；".join(analogy_domains[:2]) if analogy_domains else "无"
            lines.append(f"教学匹配: 模式 {teaching_text}；类比域 {analogy_text}")
        return "\n".join(lines)

    lines = [
        f"Last seen user: {projection.last_seen_at or 'None'}",
        f"Relationship continuity: {projection.continuity_summary or 'None'}",
        f"Open commitments: {'; '.join(commitments) if commitments else 'None'}",
        f"Interaction hints: {'; '.join(hints) if hints else 'None'}",
    ]
    if projection.collaboration_style or boundaries or misfires:
        boundary_text = "; ".join(boundaries[:3]) if boundaries else "None"
        misfire_text = "; ".join(misfires[:2]) if misfires else "None"
        lines.append(
            "Relationship style: "
            f"{projection.collaboration_style or 'None'}; "
            f"boundaries: {boundary_text}; "
            f"avoid: {misfire_text}"
        )
    if teaching_modes or analogy_domains:
        teaching_text = "; ".join(teaching_modes[:3]) if teaching_modes else "None"
        analogy_text = "; ".join(analogy_domains[:2]) if analogy_domains else "None"
        lines.append(f"Teaching fit: modes {teaching_text}; analogy domains {analogy_text}")
    return "\n".join(lines)


def render_agenda_summary(projection: BrainAgendaProjection, language: Language) -> str:
    """Render the agenda projection into prompt-safe text."""
    open_goals = projection.open_goals or []
    blocked_goals = projection.blocked_goals or []
    completed_goals = projection.completed_goals or []
    deferred_goals = projection.deferred_goals or []
    cancelled_goals = projection.cancelled_goals or []
    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            [
                f"议程种子: {projection.agenda_seed or '无'}",
                f"当前目标: {projection.active_goal_summary or (projection.goal(projection.active_goal_id).title if projection.active_goal_id and projection.goal(projection.active_goal_id) else '无')}",
                f"开放目标: {'；'.join(open_goals) if open_goals else '无'}",
                f"延期目标: {'；'.join(deferred_goals) if deferred_goals else '无'}",
                f"阻塞目标: {'；'.join(blocked_goals) if blocked_goals else '无'}",
                f"已完成目标: {'；'.join(completed_goals) if completed_goals else '无'}",
                f"已取消目标: {'；'.join(cancelled_goals) if cancelled_goals else '无'}",
            ]
        )

    return "\n".join(
        [
            f"Agenda seed: {projection.agenda_seed or 'None'}",
            f"Active goal: {projection.active_goal_summary or (projection.goal(projection.active_goal_id).title if projection.active_goal_id and projection.goal(projection.active_goal_id) else 'None')}",
            f"Open goals: {'; '.join(open_goals) if open_goals else 'None'}",
            f"Deferred goals: {'; '.join(deferred_goals) if deferred_goals else 'None'}",
            f"Blocked goals: {'; '.join(blocked_goals) if blocked_goals else 'None'}",
            f"Completed goals: {'; '.join(completed_goals) if completed_goals else 'None'}",
            f"Cancelled goals: {'; '.join(cancelled_goals) if cancelled_goals else 'None'}",
        ]
    )


def render_commitment_projection_summary(
    projection: BrainCommitmentProjection,
    language: Language,
) -> str:
    """Render the durable commitment projection into prompt-safe text."""

    def _titles(records: list[BrainCommitmentRecord]) -> str:
        return "；".join(_label(record) for record in records if record.title)

    def _blocked(records: list[BrainCommitmentRecord]) -> str:
        parts: list[str] = []
        for record in records:
            reason = record.blocked_reason.summary if record.blocked_reason else ""
            label = _label(record)
            parts.append(f"{label}({reason})" if reason else label)
        return "；".join(part for part in parts if part)

    def _label(record: BrainCommitmentRecord) -> str:
        return f"{record.title}[{record.goal_family}/{record.scope_type},rev={record.plan_revision},resume={record.resume_count}]"

    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            [
                f"当前承诺摘要: {projection.current_active_summary or '无'}",
                f"活跃承诺: {_titles(projection.active_commitments) or '无'}",
                f"延期承诺: {_titles(projection.deferred_commitments) or '无'}",
                f"阻塞承诺: {_blocked(projection.blocked_commitments) or '无'}",
                f"最近终止承诺: {_titles(projection.recent_terminal_commitments) or '无'}",
            ]
        )

    return "\n".join(
        [
            f"Current commitment summary: {projection.current_active_summary or 'None'}",
            f"Active commitments: {_titles(projection.active_commitments) or 'None'}",
            f"Deferred commitments: {_titles(projection.deferred_commitments) or 'None'}",
            f"Blocked commitments: {_blocked(projection.blocked_commitments) or 'None'}",
            f"Recent terminal commitments: {_titles(projection.recent_terminal_commitments) or 'None'}",
        ]
    )


def render_heartbeat_summary(projection: BrainHeartbeatProjection, language: Language) -> str:
    """Render the heartbeat projection into prompt-safe text."""
    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            [
                f"最近事件: {projection.last_event_type or '无'}",
                f"最近工具: {projection.last_tool_name or '无'}",
                f"最近机器人动作: {projection.last_robot_action or '无'}",
                f"告警: {'；'.join(projection.warnings) if projection.warnings else '无'}",
            ]
        )

    return "\n".join(
        [
            f"Latest event: {projection.last_event_type or 'None'}",
            f"Latest tool: {projection.last_tool_name or 'None'}",
            f"Latest robot action: {projection.last_robot_action or 'None'}",
            f"Warnings: {'; '.join(projection.warnings) if projection.warnings else 'None'}",
        ]
    )


__all__ = [
    "ACTIVE_SITUATION_MODEL_PROJECTION",
    "ADAPTER_GOVERNANCE_PROJECTION",
    "AGENDA_PROJECTION",
    "AUTONOMY_LEDGER_PROJECTION",
    "BODY_STATE_PROJECTION",
    "COMMITMENT_PROJECTION",
    "COUNTERFACTUAL_REHEARSAL_PROJECTION",
    "ENGAGEMENT_STATE_PROJECTION",
    "HEARTBEAT_PROJECTION",
    "PREDICTIVE_WORLD_MODEL_PROJECTION",
    "PRIVATE_WORKING_MEMORY_PROJECTION",
    "RELATIONSHIP_STATE_PROJECTION",
    "SCENE_STATE_PROJECTION",
    "SCENE_WORLD_STATE_PROJECTION",
    "WORKING_CONTEXT_PROJECTION",
    "BrainActiveSituationEvidenceKind",
    "BrainActiveSituationProjection",
    "BrainActiveSituationRecord",
    "BrainActiveSituationRecordKind",
    "BrainActiveSituationRecordState",
    "BrainAgendaProjection",
    "BrainBlockedReason",
    "BrainBlockedReasonKind",
    "BrainCommitmentProjection",
    "BrainCommitmentRecord",
    "BrainCommitmentScopeType",
    "BrainCommitmentStatus",
    "BrainCounterfactualCalibrationSummary",
    "BrainCounterfactualEvaluationRecord",
    "BrainCounterfactualRehearsalKind",
    "BrainCounterfactualRehearsalProjection",
    "BrainCalibrationBucket",
    "BrainObservedActionOutcomeKind",
    "BrainActionOutcomeComparisonRecord",
    "BrainActionRehearsalRequest",
    "BrainActionRehearsalResult",
    "BrainRehearsalDecisionRecommendation",
    "BrainEngagementStateProjection",
    "BrainGoalFamily",
    "BrainGoal",
    "BrainGoalStatus",
    "BrainGoalStep",
    "BrainHeartbeatProjection",
    "BrainPlanProposal",
    "BrainPlanProposalDecision",
    "BrainPlanProposalSource",
    "BrainPlanReviewPolicy",
    "BrainPredictionCalibrationSummary",
    "BrainPredictionConfidenceBand",
    "BrainPredictionKind",
    "BrainPredictionRecord",
    "BrainPredictionResolutionKind",
    "BrainPredictionSubjectKind",
    "BrainPredictiveWorldModelProjection",
    "BrainPrivateWorkingMemoryBufferKind",
    "BrainPrivateWorkingMemoryEvidenceKind",
    "BrainPrivateWorkingMemoryProjection",
    "BrainPrivateWorkingMemoryRecord",
    "BrainPrivateWorkingMemoryRecordState",
    "BrainRelationshipStateProjection",
    "BrainSceneStateProjection",
    "BrainSceneWorldAffordanceAvailability",
    "BrainSceneWorldAffordanceRecord",
    "BrainSceneWorldEntityKind",
    "BrainSceneWorldEntityRecord",
    "BrainSceneWorldEvidenceKind",
    "BrainSceneWorldProjection",
    "BrainSceneWorldRecordState",
    "BrainWakeConditionKind",
    "BrainWakeCondition",
    "BrainWorkingContextProjection",
    "render_agenda_summary",
    "render_commitment_projection_summary",
    "render_engagement_summary",
    "render_heartbeat_summary",
    "render_private_working_memory_summary",
    "render_relationship_state_summary",
    "render_scene_summary",
    "render_scene_world_state_summary",
    "render_working_context_summary",
]

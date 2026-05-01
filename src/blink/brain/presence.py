"""Presence snapshots for Blink brain runtimes."""

from __future__ import annotations

from blink.brain.core.presence import BrainPresenceSnapshot, normalize_presence_snapshot
from blink.transcriptions.language import Language


def render_presence_summary(snapshot: BrainPresenceSnapshot, language: Language) -> str:
    """Render the runtime presence snapshot into prompt-safe text."""
    snapshot = normalize_presence_snapshot(snapshot)
    if language.value.lower().startswith(("zh", "cmn")):
        robot_line = (
            f"机器人头: {'已启用' if snapshot.robot_head_enabled else '未启用'}"
            f"，模式={snapshot.robot_head_mode}"
            f"，可用={snapshot.robot_head_available}"
            f"，已上锁臂={snapshot.robot_head_armed}"
            f"，策略阶段={snapshot.policy_phase}"
        )
        if snapshot.robot_head_last_action:
            robot_line += f"，最近动作={snapshot.robot_head_last_action}"
        if snapshot.robot_head_last_accepted_action:
            robot_line += f"，最近接受动作={snapshot.robot_head_last_accepted_action}"
        if snapshot.robot_head_last_rejected_action:
            robot_line += f"，最近拒绝动作={snapshot.robot_head_last_rejected_action}"
        if snapshot.robot_head_last_safe_state:
            robot_line += f"，最近安全状态={snapshot.robot_head_last_safe_state}"
        posture_line = (
            f"姿态: 注意目标={snapshot.attention_target or '无'}"
            f"，参与姿态={snapshot.engagement_pose or '无'}"
        )
        vision_line = (
            f"视觉: {'已启用' if snapshot.vision_enabled else '未启用'}"
            f"，摄像头连接={'是' if snapshot.vision_connected else '否'}"
            f"，轨道状态={snapshot.camera_track_state}"
            f"，传感器健康={snapshot.sensor_health}"
        )
        if snapshot.last_fresh_frame_at:
            vision_line += f"，最近新鲜帧={snapshot.last_fresh_frame_at}"
        if snapshot.frame_age_ms is not None:
            vision_line += f"，帧年龄={snapshot.frame_age_ms}ms"
        if snapshot.detection_backend:
            confidence = (
                f"{snapshot.detection_confidence:.2f}"
                if snapshot.detection_confidence is not None
                else "n/a"
            )
            vision_line += f"，检测器={snapshot.detection_backend}({confidence})"
        degraded_flags = [
            label
            for enabled, label in [
                (snapshot.vision_unavailable, "vision_unavailable"),
                (snapshot.camera_disconnected, "camera_disconnected"),
                (snapshot.perception_disabled, "perception_disabled"),
                (snapshot.perception_unreliable, "perception_unreliable"),
            ]
            if enabled
        ]
        degradation_line = (
            f"降级标志: {'；'.join(degraded_flags)}" if degraded_flags else "降级标志: 无"
        )
        if snapshot.sensor_health_reason:
            degradation_line += f"，原因={snapshot.sensor_health_reason}"
        if snapshot.recovery_in_progress or snapshot.recovery_attempts:
            degradation_line += (
                f"，恢复中={'是' if snapshot.recovery_in_progress else '否'}"
                f"，恢复尝试={snapshot.recovery_attempts}"
            )
        warning_line = (
            f"运行警告: {'；'.join(snapshot.warnings)}"
            if snapshot.warnings
            else "运行警告: 无"
        )
        return "\n".join([robot_line, posture_line, vision_line, degradation_line, warning_line])

    robot_line = (
        f"Robot head: enabled={snapshot.robot_head_enabled}, mode={snapshot.robot_head_mode}, "
        f"available={snapshot.robot_head_available}, armed={snapshot.robot_head_armed}, "
        f"policy_phase={snapshot.policy_phase}"
    )
    if snapshot.robot_head_last_action:
        robot_line += f", last_action={snapshot.robot_head_last_action}"
    if snapshot.robot_head_last_accepted_action:
        robot_line += f", last_accepted_action={snapshot.robot_head_last_accepted_action}"
    if snapshot.robot_head_last_rejected_action:
        robot_line += f", last_rejected_action={snapshot.robot_head_last_rejected_action}"
    if snapshot.robot_head_last_safe_state:
        robot_line += f", last_safe_state={snapshot.robot_head_last_safe_state}"
    posture_line = (
        f"Embodiment: attention_target={snapshot.attention_target or 'None'}, "
        f"engagement_pose={snapshot.engagement_pose or 'None'}"
    )
    vision_line = (
        f"Vision: enabled={snapshot.vision_enabled}, camera_connected={snapshot.vision_connected}, "
        f"track_state={snapshot.camera_track_state}, sensor_health={snapshot.sensor_health}"
    )
    if snapshot.last_fresh_frame_at:
        vision_line += f", last_fresh_frame_at={snapshot.last_fresh_frame_at}"
    if snapshot.frame_age_ms is not None:
        vision_line += f", frame_age_ms={snapshot.frame_age_ms}"
    if snapshot.detection_backend:
        confidence = (
            f"{snapshot.detection_confidence:.2f}"
            if snapshot.detection_confidence is not None
            else "n/a"
        )
        vision_line += f", detector={snapshot.detection_backend}({confidence})"
    degraded_flags = [
        label
        for enabled, label in [
            (snapshot.vision_unavailable, "vision_unavailable"),
            (snapshot.camera_disconnected, "camera_disconnected"),
            (snapshot.perception_disabled, "perception_disabled"),
            (snapshot.perception_unreliable, "perception_unreliable"),
        ]
        if enabled
    ]
    degradation_line = (
        f"Degraded flags: {', '.join(degraded_flags)}"
        if degraded_flags
        else "Degraded flags: none"
    )
    if snapshot.sensor_health_reason:
        degradation_line += f", reason={snapshot.sensor_health_reason}"
    if snapshot.recovery_in_progress or snapshot.recovery_attempts:
        degradation_line += (
            f", recovery_in_progress={snapshot.recovery_in_progress}, "
            f"recovery_attempts={snapshot.recovery_attempts}"
        )
    warning_line = (
        f"Warnings: {'; '.join(snapshot.warnings)}" if snapshot.warnings else "Warnings: none"
    )
    return "\n".join([robot_line, posture_line, vision_line, degradation_line, warning_line])


__all__ = [
    "BrainPresenceSnapshot",
    "normalize_presence_snapshot",
    "render_presence_summary",
]

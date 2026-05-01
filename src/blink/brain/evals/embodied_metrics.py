"""Bounded metrics for Phase 21A embodied eval runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _sorted_unique(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _count_map(values: list[str] | tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return counts


@dataclass(frozen=True)
class BrainEmbodiedEvalMetricRow:
    """One bounded metric row for a scenario/profile run."""

    run_id: str
    scenario_id: str
    scenario_family: str
    scenario_version: str
    profile_id: str
    matrix_index: int
    execution_backend: str
    perception_backend_id: str | None
    world_model_backend_id: str
    embodied_policy_backend_id: str
    task_success: bool
    safety_success: bool
    preview_only: bool
    operator_intervention_count: int
    recovery_count: int
    step_count: int
    review_floor_count: int
    skill_reuse_detected: bool
    trace_status: str | None = None
    mismatch_codes: tuple[str, ...] = ()
    repair_codes: tuple[str, ...] = ()
    prediction_resolution_counts: dict[str, int] = field(default_factory=dict)
    rehearsal_recommendation_counts: dict[str, int] = field(default_factory=dict)
    artifact_paths: dict[str, str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable metric-row payload."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "scenario_version": self.scenario_version,
            "profile_id": self.profile_id,
            "matrix_index": self.matrix_index,
            "execution_backend": self.execution_backend,
            "perception_backend_id": self.perception_backend_id,
            "world_model_backend_id": self.world_model_backend_id,
            "embodied_policy_backend_id": self.embodied_policy_backend_id,
            "task_success": self.task_success,
            "safety_success": self.safety_success,
            "preview_only": self.preview_only,
            "operator_intervention_count": self.operator_intervention_count,
            "recovery_count": self.recovery_count,
            "step_count": self.step_count,
            "review_floor_count": self.review_floor_count,
            "skill_reuse_detected": self.skill_reuse_detected,
            "trace_status": self.trace_status,
            "mismatch_codes": list(self.mismatch_codes),
            "repair_codes": list(self.repair_codes),
            "prediction_resolution_counts": dict(self.prediction_resolution_counts),
            "rehearsal_recommendation_counts": dict(self.rehearsal_recommendation_counts),
            "artifact_paths": dict(self.artifact_paths),
        }


def build_embodied_eval_metric_row(
    *,
    run_id: str,
    scenario_id: str,
    scenario_family: str,
    scenario_version: str,
    profile_id: str,
    matrix_index: int,
    execution_backend: str,
    perception_backend_id: str | None,
    world_model_backend_id: str,
    embodied_policy_backend_id: str,
    goal_status: str,
    planning_outcome: str | None,
    step_count: int,
    recent_action_events: list[dict[str, Any]],
    predictive_inspection: dict[str, Any],
    rehearsal_inspection: dict[str, Any],
    recent_execution_trace: dict[str, Any] | None,
    recent_recoveries: list[dict[str, Any]],
    artifact_paths: dict[str, str | None],
) -> BrainEmbodiedEvalMetricRow:
    """Build one bounded metric row from existing shell and action surfaces."""
    mismatch_codes = _sorted_unique((recent_execution_trace or {}).get("mismatch_codes", []) or [])
    repair_codes = _sorted_unique((recent_execution_trace or {}).get("repair_codes", []) or [])
    trace_status = str((recent_execution_trace or {}).get("status") or "").strip() or None
    preview_only = any(bool(event.get("preview_only")) for event in recent_action_events)
    review_floor_count = (
        1
        if str(planning_outcome or "").strip()
        in {"needs_operator_review", "operator_review_required"}
        else 0
    )
    critical_mismatch_codes = {"unsafe", "robot_head_unarmed", "robot_head_status_unavailable"}
    safety_success = not any(code in critical_mismatch_codes for code in mismatch_codes) and (
        trace_status != "aborted"
    )
    task_success = str(goal_status).strip() == "completed"
    recommendation_counts = _count_map(
        list((rehearsal_inspection or {}).get("recommendation_counts", {}).keys())
    )
    if not recommendation_counts:
        recommendation_counts = {
            str(key): int(value)
            for key, value in dict(
                (rehearsal_inspection or {}).get("recommendation_counts", {})
            ).items()
            if str(key).strip()
        }
    prediction_resolution_counts = {
        str(key): int(value)
        for key, value in dict((predictive_inspection or {}).get("recent_resolution_counts", {})).items()
        if str(key).strip()
    }
    skill_reuse_detected = any(
        bool(dict(event.get("metadata") or {}).get("skill_id"))
        or bool(dict(event.get("metadata") or {}).get("skill_ids"))
        for event in recent_action_events
    )
    return BrainEmbodiedEvalMetricRow(
        run_id=run_id,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version=scenario_version,
        profile_id=profile_id,
        matrix_index=matrix_index,
        execution_backend=execution_backend,
        perception_backend_id=perception_backend_id,
        world_model_backend_id=world_model_backend_id,
        embodied_policy_backend_id=embodied_policy_backend_id,
        task_success=task_success,
        safety_success=safety_success,
        preview_only=preview_only,
        operator_intervention_count=0,
        recovery_count=len(recent_recoveries),
        step_count=int(step_count),
        review_floor_count=review_floor_count,
        skill_reuse_detected=skill_reuse_detected,
        trace_status=trace_status,
        mismatch_codes=mismatch_codes,
        repair_codes=repair_codes,
        prediction_resolution_counts=prediction_resolution_counts,
        rehearsal_recommendation_counts=recommendation_counts,
        artifact_paths={str(key): value for key, value in artifact_paths.items()},
    )


__all__ = ["BrainEmbodiedEvalMetricRow", "build_embodied_eval_metric_row"]

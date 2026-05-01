"""Bounded canonical episode export for practice-plan artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blink.brain.evals.episode_export import (
    _EPISODE_SCHEMA_VERSION,
    _MAX_TRACKED_IDS,
    BrainEpisodeOrigin,
    BrainEpisodeOutcomeSummary,
    BrainEpisodeRecord,
    BrainEpisodeSafetySummary,
    _build_action_summary,
    _build_artifact_refs,
    _optional_text,
    _sorted_unique,
    _stable_episode_id,
)
from blink.brain.practice_director import BrainPracticePlan, BrainPracticeTarget

_RAW_ARTIFACT_KIND_PARTS = frozenset(
    {
        "audio",
        "bytes",
        "content",
        "db",
        "frame",
        "image",
        "media",
        "mp3",
        "mp4",
        "pcm",
        "raw",
        "sqlite",
        "video",
        "wav",
    }
)


def _target_sort_key(record: BrainPracticeTarget) -> tuple[float, str, str, str]:
    return (-float(record.score), record.scenario_family, record.scenario_id, record.target_id)


def _safe_artifact_paths(
    artifact_paths: dict[str, Any] | None,
    *,
    source_path: Path | None,
) -> dict[str, str]:
    rows: dict[str, str] = {}
    if source_path is not None and source_path.suffix.lower() == ".json":
        rows["practice_plan_json"] = str(source_path)
    for key, value in sorted(dict(artifact_paths or {}).items()):
        artifact_kind = str(key).strip()
        uri = _optional_text(value)
        if not artifact_kind or uri is None:
            continue
        lowered = artifact_kind.lower()
        suffix = Path(uri).suffix.lower().lstrip(".")
        if (
            any(part in lowered for part in _RAW_ARTIFACT_KIND_PARTS)
            or suffix in _RAW_ARTIFACT_KIND_PARTS
        ):
            continue
        rows[artifact_kind] = uri
        if len(rows) >= _MAX_TRACKED_IDS:
            break
    return rows


def _plan_from_payload(payload: dict[str, Any]) -> BrainPracticePlan:
    candidate = payload.get("practice_plan")
    if not isinstance(candidate, dict):
        candidate = payload.get("plan")
    if not isinstance(candidate, dict):
        candidate = payload
    plan = BrainPracticePlan.from_dict(dict(candidate))
    if plan is None:
        raise ValueError("Unsupported practice artifact payload.")
    return plan


def _practice_provenance_ids(
    *,
    plan: BrainPracticePlan,
    target: BrainPracticeTarget,
) -> dict[str, str]:
    rows = {
        "dataset_manifest_id": plan.dataset_manifest_id,
        "practice_plan_id": plan.plan_id,
        "practice_target_id": target.target_id,
        "selected_failure_cluster_id": next(iter(target.failure_cluster_ids), None),
        "selected_supporting_episode_id": next(iter(target.supporting_episode_ids), None),
        "supporting_episode_count": str(len(target.supporting_episode_ids)),
    }
    return {key: value for key, value in sorted(rows.items()) if _optional_text(value) is not None}


def _build_episode_from_target(
    *,
    plan: BrainPracticePlan,
    target: BrainPracticeTarget,
    source_path: Path | None,
) -> BrainEpisodeRecord:
    artifact_refs = _build_artifact_refs(
        _safe_artifact_paths(plan.artifact_paths, source_path=source_path),
        include_input_path=None,
    )
    action_summary = _build_action_summary(
        execution_backend=_optional_text(target.execution_backend),
        preview_only=True,
        mismatch_codes=[],
        repair_codes=[],
        recent_low_level_actions=[],
        recent_execution_traces=[],
        recent_recoveries=[],
    )
    provenance_ids = _practice_provenance_ids(plan=plan, target=target)
    episode_id = _stable_episode_id(
        origin=BrainEpisodeOrigin.PRACTICE.value,
        scenario_id=target.scenario_id,
        scenario_family=target.scenario_family,
        scenario_version=target.scenario_version or "practice.v1",
        source_event_ids=(),
        goal_id=None,
        commitment_id=None,
        plan_proposal_id=None,
        rehearsal_ids=(),
        trace_ids=(),
        comparison_ids=(),
        backend_ids={
            "execution_backend": _optional_text(target.execution_backend),
            "practice_plan_id": plan.plan_id,
            "practice_target_id": target.target_id,
            "selected_profile_id": target.selected_profile_id,
        },
    )
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version=_EPISODE_SCHEMA_VERSION,
        origin=BrainEpisodeOrigin.PRACTICE.value,
        scenario_id=target.scenario_id,
        scenario_family=target.scenario_family,
        scenario_version=target.scenario_version or "practice.v1",
        source_run_id=plan.plan_id,
        skill_ids=_sorted_unique(target.related_skill_ids),
        execution_backend=_optional_text(target.execution_backend),
        embodied_policy_backend_id=_optional_text(target.selected_profile_id),
        artifact_refs=artifact_refs,
        action_summary=action_summary,
        outcome_summary=BrainEpisodeOutcomeSummary(
            goal_status="planned",
            planning_outcome="practice_target_selected",
            task_success=None,
            trace_status=None,
            operator_review_floored=False,
        ),
        safety_summary=BrainEpisodeSafetySummary(safety_success=None),
        started_at=plan.created_at,
        ended_at=None,
        generated_at=plan.updated_at or plan.created_at,
        provenance_ids=provenance_ids,
    )


def build_episodes_from_practice_plan_payload(
    practice_payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> tuple[BrainEpisodeRecord, ...]:
    """Build canonical planned-practice episodes from one practice-plan artifact."""
    plan = _plan_from_payload(dict(practice_payload))
    return tuple(
        _build_episode_from_target(plan=plan, target=target, source_path=source_path)
        for target in sorted(plan.targets, key=_target_sort_key)
    )


__all__ = ["build_episodes_from_practice_plan_payload"]

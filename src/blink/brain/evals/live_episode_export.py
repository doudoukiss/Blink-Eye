"""Bounded canonical episode export for live runtime artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blink.brain.evals.episode_export import (
    _EPISODE_SCHEMA_VERSION,
    _MAX_CODE_LIST,
    _MAX_TRACKED_IDS,
    BrainEpisodeOrigin,
    BrainEpisodeOutcomeSummary,
    BrainEpisodeRecord,
    BrainEpisodeSafetySummary,
    _build_action_summary,
    _build_artifact_refs,
    _build_observation_refs,
    _build_prediction_summary,
    _build_rehearsal_summary,
    _first_text_from_payloads,
    _optional_text,
    _provenance_ids,
    _skill_ids_from_payloads,
    _sorted_mapping,
    _sorted_unique,
    _stable_episode_id,
    _timestamps_from_events,
)

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


def _payload_body(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("live_episode")
    if not isinstance(nested, dict):
        nested = payload.get("episode")
    if not isinstance(nested, dict):
        return dict(payload)
    merged = dict(nested)
    merged.update(
        {key: value for key, value in payload.items() if key not in {"episode", "live_episode"}}
    )
    return merged


def _payload_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("events")
    if not isinstance(rows, list):
        rows = payload.get("event_slice")
    return [dict(item) for item in rows or [] if isinstance(item, dict)]


def _source_event_ids(events: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(event.get("event_id", "")).strip()
        for event in sorted(
            events,
            key=lambda item: (
                str(item.get("ts") or ""),
                str(item.get("event_type") or ""),
                str(item.get("event_id") or ""),
            ),
        )
        if str(event.get("event_id", "")).strip()
    )[:_MAX_TRACKED_IDS]


def _safe_artifact_paths(
    artifact_paths: dict[str, Any] | None,
    *,
    source_path: Path | None,
) -> dict[str, str]:
    rows: dict[str, str] = {}
    if source_path is not None and source_path.suffix.lower() == ".json":
        rows["live_runtime_artifact_json"] = str(source_path)
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


def _runtime_lineage(payload: dict[str, Any], runtime: dict[str, Any]) -> dict[str, str]:
    rows = {
        "context_packet_id": _optional_text(payload.get("context_packet_id")),
        "live_artifact_id": _optional_text(payload.get("artifact_id")),
        "memory_use_trace_id": _optional_text(payload.get("memory_use_trace_id")),
        "runtime_artifact_id": _optional_text(payload.get("runtime_artifact_id")),
        "runtime_episode_id": (
            _optional_text(payload.get("source_runtime_episode_id"))
            or _optional_text(payload.get("runtime_episode_id"))
            or _optional_text(payload.get("episode_id"))
        ),
        "runtime_kind": _optional_text(payload.get("runtime_kind"))
        or _optional_text(runtime.get("runtime_kind")),
        "turn_id": _optional_text(payload.get("turn_id")),
    }
    return {key: value for key, value in sorted(rows.items()) if value is not None}


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def build_episode_from_live_runtime_payload(
    live_payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> BrainEpisodeRecord:
    """Build one canonical episode from one bounded live runtime artifact payload."""
    payload = _payload_body(dict(live_payload))
    runtime = dict(payload.get("runtime", {})) if isinstance(payload.get("runtime"), dict) else {}
    metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {}
    continuity_state = (
        dict(payload.get("continuity_state", {}))
        if isinstance(payload.get("continuity_state"), dict)
        else {}
    )
    shell_digest = (
        dict(payload.get("shell_digest", {}))
        if isinstance(payload.get("shell_digest"), dict)
        else {}
    )
    events = _payload_events(payload)
    predictive_digest = dict(
        payload.get("predictive_digest")
        or continuity_state.get("predictive_digest")
        or shell_digest.get("predictive_inspection")
        or {}
    )
    predictive_world_model = dict(continuity_state.get("predictive_world_model", {}))
    rehearsal_digest = dict(
        payload.get("rehearsal_digest")
        or continuity_state.get("rehearsal_digest")
        or shell_digest.get("rehearsal_inspection")
        or {}
    )
    counterfactual_rehearsal = dict(continuity_state.get("counterfactual_rehearsal", {}))
    embodied_digest = dict(
        payload.get("embodied_digest")
        or continuity_state.get("embodied_digest")
        or shell_digest.get("embodied_inspection")
        or {}
    )
    recent_traces = [
        dict(item)
        for item in embodied_digest.get("recent_execution_traces", [])
        if isinstance(item, dict)
    ]
    recent_recoveries = [
        dict(item)
        for item in embodied_digest.get("recent_recoveries", [])
        if isinstance(item, dict)
    ]
    recent_actions = [
        dict(item)
        for item in embodied_digest.get("recent_low_level_embodied_actions", [])
        if isinstance(item, dict)
    ]
    scenario_id = _optional_text(payload.get("scenario_id")) or "live_runtime_turn"
    scenario_family = _optional_text(payload.get("scenario_family")) or "live_runtime_turn"
    scenario_version = _optional_text(payload.get("scenario_version")) or "live.v1"
    execution_backend = (
        _optional_text(metrics.get("execution_backend"))
        or _optional_text(payload.get("execution_backend"))
        or _optional_text(payload.get("runtime_kind"))
        or _optional_text(runtime.get("runtime_kind"))
    )
    prediction_summary = _build_prediction_summary(predictive_digest, predictive_world_model)
    rehearsal_summary = _build_rehearsal_summary(rehearsal_digest, counterfactual_rehearsal)
    action_summary = _build_action_summary(
        execution_backend=execution_backend,
        preview_only=bool(metrics.get("preview_only", False)),
        mismatch_codes=list(metrics.get("mismatch_codes", []))
        or list(dict(embodied_digest.get("mismatch_code_counts", {})).keys()),
        repair_codes=list(metrics.get("repair_codes", []))
        or list(dict(embodied_digest.get("repair_code_counts", {})).keys()),
        recent_low_level_actions=recent_actions,
        recent_execution_traces=recent_traces,
        recent_recoveries=recent_recoveries,
    )
    planning_digest = dict(continuity_state.get("planning_digest", {}))
    goal_id = _first_text_from_payloads(
        payload,
        planning_digest,
        embodied_digest,
        events,
        key="goal_id",
        prefer_last=True,
    )
    commitment_id = _first_text_from_payloads(
        payload,
        planning_digest,
        embodied_digest,
        events,
        key="commitment_id",
        prefer_last=True,
    )
    plan_proposal_id = _first_text_from_payloads(
        payload,
        planning_digest,
        embodied_digest,
        counterfactual_rehearsal,
        events,
        key="plan_proposal_id",
        prefer_last=True,
    )
    outcome_summary = BrainEpisodeOutcomeSummary(
        goal_status=_first_text_from_payloads(
            planning_digest, events, key="status", prefer_last=True
        ),
        planning_outcome=_optional_text(metrics.get("planning_outcome"))
        or _optional_text(payload.get("planning_outcome")),
        task_success=_bool_or_none(metrics.get("task_success")),
        trace_status=_optional_text(metrics.get("trace_status")) or action_summary.trace_status,
        operator_review_floored=int(metrics.get("review_floor_count", 0)) > 0,
        observed_outcome_counts=_sorted_mapping(rehearsal_summary.observed_outcome_counts),
        calibration_bucket_counts=_sorted_mapping(rehearsal_summary.calibration_bucket_counts),
    )
    safety_summary = BrainEpisodeSafetySummary(
        safety_success=_bool_or_none(metrics.get("safety_success")),
        review_floor_count=int(metrics.get("review_floor_count", 0)),
        operator_intervention_count=int(metrics.get("operator_intervention_count", 0)),
        recovery_count=action_summary.recovery_count,
        risk_codes=_sorted_unique(
            list(metrics.get("risk_codes", []))
            or list(rehearsal_summary.risk_code_counts.keys())[:_MAX_CODE_LIST]
        ),
        mismatch_codes=action_summary.mismatch_codes,
        repair_codes=action_summary.repair_codes,
    )
    source_event_ids = _source_event_ids(events)
    started_at, ended_at = _timestamps_from_events(events)
    artifact_refs = _build_artifact_refs(
        _safe_artifact_paths(payload.get("artifact_paths"), source_path=source_path),
        include_input_path=None,
    )
    source_run_id = (
        _optional_text(payload.get("source_run_id"))
        or _optional_text(payload.get("runtime_run_id"))
        or _optional_text(payload.get("run_id"))
        or _optional_text(payload.get("turn_id"))
    )
    provenance_ids = {
        **_provenance_ids(
            prediction_summary=prediction_summary,
            rehearsal_summary=rehearsal_summary,
            action_summary=action_summary,
        ),
        **_runtime_lineage(payload, runtime),
    }
    episode_id = _stable_episode_id(
        origin=BrainEpisodeOrigin.LIVE.value,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version=scenario_version,
        source_event_ids=source_event_ids,
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        rehearsal_ids=rehearsal_summary.rehearsal_ids,
        trace_ids=action_summary.trace_ids,
        comparison_ids=rehearsal_summary.comparison_ids,
        backend_ids={
            "execution_backend": execution_backend,
            "runtime_episode_id": provenance_ids.get("runtime_episode_id"),
            "source_run_id": source_run_id,
            "turn_id": provenance_ids.get("turn_id"),
        },
    )
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version=_EPISODE_SCHEMA_VERSION,
        origin=BrainEpisodeOrigin.LIVE.value,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version=scenario_version,
        source_run_id=source_run_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_ids=_skill_ids_from_payloads(events, recent_actions, recent_traces, planning_digest),
        execution_backend=execution_backend,
        source_event_ids=source_event_ids,
        observation_refs=_build_observation_refs(events),
        artifact_refs=artifact_refs,
        prediction_summary=prediction_summary,
        rehearsal_summary=rehearsal_summary,
        action_summary=action_summary,
        outcome_summary=outcome_summary,
        safety_summary=safety_summary,
        started_at=started_at,
        ended_at=ended_at,
        generated_at=ended_at or started_at,
        provenance_ids=dict(sorted(provenance_ids.items())),
    )


__all__ = ["build_episode_from_live_runtime_payload"]

"""Canonical bounded episode export records for Phase 22."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

_EPISODE_SCHEMA_VERSION = "brain_episode/v1"
_OBSERVATION_EVENT_TYPES = frozenset(
    {
        "body.state.updated",
        "perception.observed",
        "engagement.changed",
        "attention.changed",
        "scene.changed",
    }
)
_MAX_OBSERVATION_REFS = 12
_MAX_ARTIFACT_REFS = 12
_MAX_TRACKED_IDS = 12
_MAX_CODE_LIST = 8


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({text for value in values if (text := _optional_text(value)) is not None}))


def _sorted_mapping(values: dict[str, Any] | None) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(dict(values or {}).items())
        if _optional_text(key) is not None
    }


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, json.dumps(payload, ensure_ascii=False, sort_keys=True)).hex}"


def _event_sort_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("ts") or ""),
        str(event.get("event_type") or ""),
        str(event.get("event_id") or ""),
    )


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, dict) else {}


def _bounded_path(path: str | None) -> str | None:
    return _optional_text(path)


def _ordered_values_for_key(value: Any, key: str) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                if isinstance(item_value, list):
                    values.extend(
                        text for element in item_value if (text := _optional_text(element)) is not None
                    )
                elif (text := _optional_text(item_value)) is not None:
                    values.append(text)
            values.extend(_ordered_values_for_key(item_value, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_ordered_values_for_key(item, key))
    return values


def _first_text_from_payloads(*payloads: Any, key: str, prefer_last: bool = False) -> str | None:
    for payload in payloads:
        values = _ordered_values_for_key(payload, key)
        if values:
            return values[-1] if prefer_last else values[0]
    return None


def _skill_ids_from_payloads(*payloads: Any) -> tuple[str, ...]:
    values: list[str] = []
    for payload in payloads:
        values.extend(_ordered_values_for_key(payload, "skill_id"))
        values.extend(_ordered_values_for_key(payload, "skill_ids"))
    return _sorted_unique(values)


def _timestamps_from_events(events: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    timestamps = [str(event.get("ts") or "").strip() for event in events if str(event.get("ts") or "").strip()]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _observation_summary(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "").strip()
    payload = _event_payload(event)
    if event_type == "scene.changed":
        return _optional_text(payload.get("summary")) or "Scene changed."
    if event_type == "perception.observed":
        return _optional_text(payload.get("summary")) or "Perception observed."
    if event_type == "body.state.updated":
        snapshot = dict(payload.get("snapshot") or {})
        sensor_health = _optional_text(snapshot.get("sensor_health")) or "unknown"
        return f"Body state updated ({sensor_health})."
    if event_type == "engagement.changed":
        return _optional_text(payload.get("summary")) or "Engagement changed."
    if event_type == "attention.changed":
        return _optional_text(payload.get("summary")) or "Attention changed."
    return event_type or "Observation recorded."


class BrainEpisodeOrigin(str, Enum):
    """Canonical source kinds for exported episodes."""

    SIMULATION = "simulation"
    REPLAY = "replay"
    LIVE = "live"
    PRACTICE = "practice"


@dataclass(frozen=True)
class BrainEpisodeArtifactRef:
    """One bounded reference to an external artifact or trace directory."""

    artifact_id: str
    artifact_kind: str
    uri: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the artifact reference."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "uri": self.uri,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeArtifactRef | None":
        """Hydrate one artifact reference from JSON."""
        if not isinstance(data, dict):
            return None
        artifact_id = _optional_text(data.get("artifact_id"))
        artifact_kind = _optional_text(data.get("artifact_kind"))
        uri = _optional_text(data.get("uri"))
        if artifact_id is None or artifact_kind is None or uri is None:
            return None
        return cls(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            uri=uri,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainEpisodeObservationRef:
    """One bounded reference to an observed source event."""

    observation_id: str
    event_id: str | None
    event_type: str
    observed_at: str | None
    summary: str
    presence_scope_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the observation reference."""
        return {
            "observation_id": self.observation_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "observed_at": self.observed_at,
            "summary": self.summary,
            "presence_scope_key": self.presence_scope_key,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeObservationRef | None":
        """Hydrate one observation reference from JSON."""
        if not isinstance(data, dict):
            return None
        observation_id = _optional_text(data.get("observation_id"))
        event_type = _optional_text(data.get("event_type"))
        summary = _optional_text(data.get("summary"))
        if observation_id is None or event_type is None or summary is None:
            return None
        return cls(
            observation_id=observation_id,
            event_id=_optional_text(data.get("event_id")),
            event_type=event_type,
            observed_at=_optional_text(data.get("observed_at")),
            summary=summary,
            presence_scope_key=_optional_text(data.get("presence_scope_key")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainEpisodePredictionSummary:
    """Compact predictive-state summary attached to one episode."""

    active_prediction_ids: tuple[str, ...] = ()
    recent_resolution_ids: tuple[str, ...] = ()
    active_kind_counts: dict[str, int] = field(default_factory=dict)
    active_confidence_band_counts: dict[str, int] = field(default_factory=dict)
    resolution_kind_counts: dict[str, int] = field(default_factory=dict)
    highest_risk_prediction_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the predictive summary."""
        return {
            "active_prediction_ids": list(self.active_prediction_ids),
            "recent_resolution_ids": list(self.recent_resolution_ids),
            "active_kind_counts": dict(self.active_kind_counts),
            "active_confidence_band_counts": dict(self.active_confidence_band_counts),
            "resolution_kind_counts": dict(self.resolution_kind_counts),
            "highest_risk_prediction_ids": list(self.highest_risk_prediction_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodePredictionSummary":
        """Hydrate the predictive summary from JSON."""
        payload = dict(data or {})
        return cls(
            active_prediction_ids=_sorted_unique(list(payload.get("active_prediction_ids", []))),
            recent_resolution_ids=_sorted_unique(list(payload.get("recent_resolution_ids", []))),
            active_kind_counts=_sorted_mapping(payload.get("active_kind_counts")),
            active_confidence_band_counts=_sorted_mapping(
                payload.get("active_confidence_band_counts")
            ),
            resolution_kind_counts=_sorted_mapping(payload.get("resolution_kind_counts")),
            highest_risk_prediction_ids=_sorted_unique(
                list(payload.get("highest_risk_prediction_ids", []))
            ),
        )


@dataclass(frozen=True)
class BrainEpisodeRehearsalSummary:
    """Compact rehearsal and calibration summary for one episode."""

    rehearsal_ids: tuple[str, ...] = ()
    comparison_ids: tuple[str, ...] = ()
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    observed_outcome_counts: dict[str, int] = field(default_factory=dict)
    risk_code_counts: dict[str, int] = field(default_factory=dict)
    recurrent_mismatch_patterns: dict[str, int] = field(default_factory=dict)
    selected_rehearsal_id: str | None = None
    selected_comparison_id: str | None = None
    simulated_backend: str | None = None
    decision_recommendation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the rehearsal summary."""
        return {
            "rehearsal_ids": list(self.rehearsal_ids),
            "comparison_ids": list(self.comparison_ids),
            "recommendation_counts": dict(self.recommendation_counts),
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "observed_outcome_counts": dict(self.observed_outcome_counts),
            "risk_code_counts": dict(self.risk_code_counts),
            "recurrent_mismatch_patterns": dict(self.recurrent_mismatch_patterns),
            "selected_rehearsal_id": self.selected_rehearsal_id,
            "selected_comparison_id": self.selected_comparison_id,
            "simulated_backend": self.simulated_backend,
            "decision_recommendation": self.decision_recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeRehearsalSummary":
        """Hydrate the rehearsal summary from JSON."""
        payload = dict(data or {})
        return cls(
            rehearsal_ids=_sorted_unique(list(payload.get("rehearsal_ids", []))),
            comparison_ids=_sorted_unique(list(payload.get("comparison_ids", []))),
            recommendation_counts=_sorted_mapping(payload.get("recommendation_counts")),
            calibration_bucket_counts=_sorted_mapping(payload.get("calibration_bucket_counts")),
            observed_outcome_counts=_sorted_mapping(payload.get("observed_outcome_counts")),
            risk_code_counts=_sorted_mapping(payload.get("risk_code_counts")),
            recurrent_mismatch_patterns=_sorted_mapping(payload.get("recurrent_mismatch_patterns")),
            selected_rehearsal_id=_optional_text(payload.get("selected_rehearsal_id")),
            selected_comparison_id=_optional_text(payload.get("selected_comparison_id")),
            simulated_backend=_optional_text(payload.get("simulated_backend")),
            decision_recommendation=_optional_text(payload.get("decision_recommendation")),
        )


@dataclass(frozen=True)
class BrainEpisodeActionSummary:
    """Compact action, execution, and recovery summary."""

    action_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    recovery_ids: tuple[str, ...] = ()
    trace_status: str | None = None
    mismatch_codes: tuple[str, ...] = ()
    repair_codes: tuple[str, ...] = ()
    execution_backend: str | None = None
    preview_only: bool = False
    recovery_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize the action summary."""
        return {
            "action_ids": list(self.action_ids),
            "trace_ids": list(self.trace_ids),
            "recovery_ids": list(self.recovery_ids),
            "trace_status": self.trace_status,
            "mismatch_codes": list(self.mismatch_codes),
            "repair_codes": list(self.repair_codes),
            "execution_backend": self.execution_backend,
            "preview_only": self.preview_only,
            "recovery_count": self.recovery_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeActionSummary":
        """Hydrate the action summary from JSON."""
        payload = dict(data or {})
        return cls(
            action_ids=_sorted_unique(list(payload.get("action_ids", []))),
            trace_ids=_sorted_unique(list(payload.get("trace_ids", []))),
            recovery_ids=_sorted_unique(list(payload.get("recovery_ids", []))),
            trace_status=_optional_text(payload.get("trace_status")),
            mismatch_codes=_sorted_unique(list(payload.get("mismatch_codes", []))),
            repair_codes=_sorted_unique(list(payload.get("repair_codes", []))),
            execution_backend=_optional_text(payload.get("execution_backend")),
            preview_only=bool(payload.get("preview_only", False)),
            recovery_count=int(payload.get("recovery_count", 0)),
        )


@dataclass(frozen=True)
class BrainEpisodeOutcomeSummary:
    """Compact high-level outcome summary."""

    goal_status: str | None = None
    planning_outcome: str | None = None
    task_success: bool | None = None
    trace_status: str | None = None
    operator_review_floored: bool = False
    observed_outcome_counts: dict[str, int] = field(default_factory=dict)
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the high-level outcome summary."""
        return {
            "goal_status": self.goal_status,
            "planning_outcome": self.planning_outcome,
            "task_success": self.task_success,
            "trace_status": self.trace_status,
            "operator_review_floored": self.operator_review_floored,
            "observed_outcome_counts": dict(self.observed_outcome_counts),
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeOutcomeSummary":
        """Hydrate the high-level outcome summary from JSON."""
        payload = dict(data or {})
        task_success = payload.get("task_success")
        return cls(
            goal_status=_optional_text(payload.get("goal_status")),
            planning_outcome=_optional_text(payload.get("planning_outcome")),
            task_success=bool(task_success) if isinstance(task_success, bool) else None,
            trace_status=_optional_text(payload.get("trace_status")),
            operator_review_floored=bool(payload.get("operator_review_floored", False)),
            observed_outcome_counts=_sorted_mapping(payload.get("observed_outcome_counts")),
            calibration_bucket_counts=_sorted_mapping(payload.get("calibration_bucket_counts")),
        )


@dataclass(frozen=True)
class BrainEpisodeSafetySummary:
    """Compact safety and review summary."""

    safety_success: bool | None = None
    review_floor_count: int = 0
    operator_intervention_count: int = 0
    recovery_count: int = 0
    risk_codes: tuple[str, ...] = ()
    mismatch_codes: tuple[str, ...] = ()
    repair_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the safety summary."""
        return {
            "safety_success": self.safety_success,
            "review_floor_count": self.review_floor_count,
            "operator_intervention_count": self.operator_intervention_count,
            "recovery_count": self.recovery_count,
            "risk_codes": list(self.risk_codes),
            "mismatch_codes": list(self.mismatch_codes),
            "repair_codes": list(self.repair_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeSafetySummary":
        """Hydrate the safety summary from JSON."""
        payload = dict(data or {})
        safety_success = payload.get("safety_success")
        return cls(
            safety_success=bool(safety_success) if isinstance(safety_success, bool) else None,
            review_floor_count=int(payload.get("review_floor_count", 0)),
            operator_intervention_count=int(payload.get("operator_intervention_count", 0)),
            recovery_count=int(payload.get("recovery_count", 0)),
            risk_codes=_sorted_unique(list(payload.get("risk_codes", []))),
            mismatch_codes=_sorted_unique(list(payload.get("mismatch_codes", []))),
            repair_codes=_sorted_unique(list(payload.get("repair_codes", []))),
        )


@dataclass(frozen=True)
class BrainEpisodeRecord:
    """One canonical bounded episode record exported from eval or replay surfaces."""

    episode_id: str
    schema_version: str
    origin: str
    scenario_id: str
    scenario_family: str
    scenario_version: str
    source_run_id: str | None = None
    source_replay_name: str | None = None
    goal_id: str | None = None
    commitment_id: str | None = None
    plan_proposal_id: str | None = None
    skill_ids: tuple[str, ...] = ()
    execution_backend: str | None = None
    perception_backend_id: str | None = None
    world_model_backend_id: str | None = None
    embodied_policy_backend_id: str | None = None
    backend_versions: dict[str, str] = field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()
    observation_refs: tuple[BrainEpisodeObservationRef, ...] = ()
    artifact_refs: tuple[BrainEpisodeArtifactRef, ...] = ()
    prediction_summary: BrainEpisodePredictionSummary = field(
        default_factory=BrainEpisodePredictionSummary
    )
    rehearsal_summary: BrainEpisodeRehearsalSummary = field(
        default_factory=BrainEpisodeRehearsalSummary
    )
    action_summary: BrainEpisodeActionSummary = field(default_factory=BrainEpisodeActionSummary)
    outcome_summary: BrainEpisodeOutcomeSummary = field(default_factory=BrainEpisodeOutcomeSummary)
    safety_summary: BrainEpisodeSafetySummary = field(default_factory=BrainEpisodeSafetySummary)
    started_at: str | None = None
    ended_at: str | None = None
    generated_at: str | None = None
    provenance_ids: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the full episode record."""
        return {
            "episode_id": self.episode_id,
            "schema_version": self.schema_version,
            "origin": self.origin,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "scenario_version": self.scenario_version,
            "source_run_id": self.source_run_id,
            "source_replay_name": self.source_replay_name,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "skill_ids": list(self.skill_ids),
            "execution_backend": self.execution_backend,
            "perception_backend_id": self.perception_backend_id,
            "world_model_backend_id": self.world_model_backend_id,
            "embodied_policy_backend_id": self.embodied_policy_backend_id,
            "backend_versions": dict(self.backend_versions),
            "source_event_ids": list(self.source_event_ids),
            "observation_refs": [record.as_dict() for record in self.observation_refs],
            "artifact_refs": [record.as_dict() for record in self.artifact_refs],
            "prediction_summary": self.prediction_summary.as_dict(),
            "rehearsal_summary": self.rehearsal_summary.as_dict(),
            "action_summary": self.action_summary.as_dict(),
            "outcome_summary": self.outcome_summary.as_dict(),
            "safety_summary": self.safety_summary.as_dict(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "generated_at": self.generated_at,
            "provenance_ids": dict(self.provenance_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEpisodeRecord | None":
        """Hydrate the full episode record from JSON."""
        if not isinstance(data, dict):
            return None
        episode_id = _optional_text(data.get("episode_id"))
        schema_version = _optional_text(data.get("schema_version"))
        origin = _optional_text(data.get("origin"))
        scenario_id = _optional_text(data.get("scenario_id"))
        scenario_family = _optional_text(data.get("scenario_family"))
        scenario_version = _optional_text(data.get("scenario_version"))
        if not all((episode_id, schema_version, origin, scenario_id, scenario_family, scenario_version)):
            return None
        return cls(
            episode_id=episode_id,
            schema_version=schema_version,
            origin=origin,
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            scenario_version=scenario_version,
            source_run_id=_optional_text(data.get("source_run_id")),
            source_replay_name=_optional_text(data.get("source_replay_name")),
            goal_id=_optional_text(data.get("goal_id")),
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            skill_ids=_sorted_unique(list(data.get("skill_ids", []))),
            execution_backend=_optional_text(data.get("execution_backend")),
            perception_backend_id=_optional_text(data.get("perception_backend_id")),
            world_model_backend_id=_optional_text(data.get("world_model_backend_id")),
            embodied_policy_backend_id=_optional_text(data.get("embodied_policy_backend_id")),
            backend_versions={
                str(key): str(value)
                for key, value in sorted(dict(data.get("backend_versions", {})).items())
                if _optional_text(key) is not None and _optional_text(value) is not None
            },
            source_event_ids=_sorted_unique(list(data.get("source_event_ids", []))),
            observation_refs=tuple(
                record
                for item in data.get("observation_refs", [])
                if (record := BrainEpisodeObservationRef.from_dict(item)) is not None
            ),
            artifact_refs=tuple(
                record
                for item in data.get("artifact_refs", [])
                if (record := BrainEpisodeArtifactRef.from_dict(item)) is not None
            ),
            prediction_summary=BrainEpisodePredictionSummary.from_dict(data.get("prediction_summary")),
            rehearsal_summary=BrainEpisodeRehearsalSummary.from_dict(data.get("rehearsal_summary")),
            action_summary=BrainEpisodeActionSummary.from_dict(data.get("action_summary")),
            outcome_summary=BrainEpisodeOutcomeSummary.from_dict(data.get("outcome_summary")),
            safety_summary=BrainEpisodeSafetySummary.from_dict(data.get("safety_summary")),
            started_at=_optional_text(data.get("started_at")),
            ended_at=_optional_text(data.get("ended_at")),
            generated_at=_optional_text(data.get("generated_at")),
            provenance_ids={
                str(key): str(value)
                for key, value in sorted(dict(data.get("provenance_ids", {})).items())
                if _optional_text(key) is not None and _optional_text(value) is not None
            },
        )


def _build_observation_refs(events: list[dict[str, Any]]) -> tuple[BrainEpisodeObservationRef, ...]:
    refs: list[BrainEpisodeObservationRef] = []
    for event in sorted(events, key=_event_sort_key):
        event_type = str(event.get("event_type") or "").strip()
        if event_type not in _OBSERVATION_EVENT_TYPES:
            continue
        payload = _event_payload(event)
        event_id = _optional_text(event.get("event_id"))
        summary = _observation_summary(event)
        observation_id = _stable_id(
            "observation",
            {
                "event_id": event_id,
                "event_type": event_type,
                "summary": summary,
            },
        )
        refs.append(
            BrainEpisodeObservationRef(
                observation_id=observation_id,
                event_id=event_id,
                event_type=event_type,
                observed_at=_optional_text(event.get("ts")),
                summary=summary,
                presence_scope_key=(
                    _optional_text(payload.get("presence_scope_key"))
                    or _optional_text(payload.get("scope_key"))
                ),
                details={
                    key: value
                    for key, value in {
                        "source": _optional_text(event.get("source")),
                        "correlation_id": _optional_text(event.get("correlation_id")),
                        "causal_parent_id": _optional_text(event.get("causal_parent_id")),
                    }.items()
                    if value is not None
                },
            )
        )
        if len(refs) >= _MAX_OBSERVATION_REFS:
            break
    return tuple(refs)


def _build_artifact_refs(
    artifact_paths: dict[str, Any] | None,
    *,
    run_source_path: Path | None = None,
    include_input_path: Path | None = None,
) -> tuple[BrainEpisodeArtifactRef, ...]:
    refs: list[BrainEpisodeArtifactRef] = []
    path_rows = {
        str(key): _bounded_path(value)
        for key, value in dict(artifact_paths or {}).items()
        if _bounded_path(value) is not None
    }
    inferred_trace_paths: dict[str, str] = {}
    candidate_run_path = Path(path_rows.get("run_json")) if path_rows.get("run_json") else run_source_path
    if candidate_run_path is not None:
        for trace_name in ("simulation", "preview"):
            trace_path = candidate_run_path.parent / trace_name
            if trace_path.exists():
                inferred_trace_paths[f"{trace_name}_trace_dir"] = str(trace_path)
    if include_input_path is not None:
        inferred_trace_paths.setdefault("source_input", str(include_input_path))
    merged = {**path_rows, **inferred_trace_paths}
    for artifact_kind, uri in sorted(merged.items()):
        refs.append(
            BrainEpisodeArtifactRef(
                artifact_id=_stable_id(
                    "artifact",
                    {"artifact_kind": artifact_kind, "uri": uri},
                ),
                artifact_kind=artifact_kind,
                uri=uri,
                details={"is_directory": str(Path(uri).is_dir()).lower()},
            )
        )
        if len(refs) >= _MAX_ARTIFACT_REFS:
            break
    return tuple(refs)


def _build_prediction_summary(
    predictive_digest: dict[str, Any] | None,
    predictive_world_model: dict[str, Any] | None = None,
) -> BrainEpisodePredictionSummary:
    digest = dict(predictive_digest or {})
    projection = dict(predictive_world_model or {})
    active_prediction_ids = [
        str(item.get("prediction_id", "")).strip()
        for item in projection.get("active_predictions", [])
        if isinstance(item, dict)
    ][: _MAX_TRACKED_IDS]
    recent_resolution_ids = [
        str(item.get("prediction_id", "")).strip()
        for item in projection.get("recent_resolutions", [])
        if isinstance(item, dict)
    ][: _MAX_TRACKED_IDS]
    return BrainEpisodePredictionSummary(
        active_prediction_ids=_sorted_unique(active_prediction_ids),
        recent_resolution_ids=_sorted_unique(recent_resolution_ids),
        active_kind_counts=_sorted_mapping(digest.get("active_kind_counts") or projection.get("active_kind_counts")),
        active_confidence_band_counts=_sorted_mapping(
            digest.get("active_confidence_band_counts") or projection.get("active_confidence_band_counts")
        ),
        resolution_kind_counts=_sorted_mapping(
            digest.get("resolution_kind_counts") or projection.get("resolution_kind_counts")
        ),
        highest_risk_prediction_ids=_sorted_unique(
            list(digest.get("highest_risk_prediction_ids", []))[:_MAX_TRACKED_IDS]
        ),
    )


def _build_rehearsal_summary(
    rehearsal_digest: dict[str, Any] | None,
    counterfactual_rehearsal: dict[str, Any] | None = None,
) -> BrainEpisodeRehearsalSummary:
    digest = dict(rehearsal_digest or {})
    projection = dict(counterfactual_rehearsal or {})
    recent_rehearsals = [
        dict(item)
        for item in projection.get("recent_rehearsals", [])
        if isinstance(item, dict)
    ]
    recent_comparisons = [
        dict(item)
        for item in projection.get("recent_comparisons", [])
        if isinstance(item, dict)
    ]
    selected_rehearsal = recent_rehearsals[0] if recent_rehearsals else {}
    selected_comparison = recent_comparisons[0] if recent_comparisons else {}
    return BrainEpisodeRehearsalSummary(
        rehearsal_ids=_sorted_unique(
            [
                str(item.get("rehearsal_id", "")).strip()
                for item in recent_rehearsals[:_MAX_TRACKED_IDS]
            ]
            + list(digest.get("open_rehearsal_ids", []))
        ),
        comparison_ids=_sorted_unique(
            [
                str(item.get("comparison_id", "")).strip()
                for item in recent_comparisons[:_MAX_TRACKED_IDS]
            ]
        ),
        recommendation_counts=_sorted_mapping(digest.get("recommendation_counts")),
        calibration_bucket_counts=_sorted_mapping(digest.get("calibration_bucket_counts")),
        observed_outcome_counts=_sorted_mapping(digest.get("observed_outcome_counts")),
        risk_code_counts=_sorted_mapping(digest.get("risk_code_counts")),
        recurrent_mismatch_patterns=_sorted_mapping(digest.get("recurrent_mismatch_patterns")),
        selected_rehearsal_id=_optional_text(selected_rehearsal.get("rehearsal_id")),
        selected_comparison_id=_optional_text(selected_comparison.get("comparison_id")),
        simulated_backend=(
            _optional_text(selected_rehearsal.get("simulated_backend"))
            or _optional_text(projection.get("presence_scope_key"))
        ),
        decision_recommendation=_optional_text(selected_rehearsal.get("decision_recommendation")),
    )


def _build_action_summary(
    *,
    execution_backend: str | None,
    preview_only: bool,
    mismatch_codes: list[str] | tuple[str, ...],
    repair_codes: list[str] | tuple[str, ...],
    recent_low_level_actions: list[dict[str, Any]] | None,
    recent_execution_traces: list[dict[str, Any]] | None,
    recent_recoveries: list[dict[str, Any]] | None,
) -> BrainEpisodeActionSummary:
    actions = [dict(item) for item in recent_low_level_actions or [] if isinstance(item, dict)]
    traces = [dict(item) for item in recent_execution_traces or [] if isinstance(item, dict)]
    recoveries = [dict(item) for item in recent_recoveries or [] if isinstance(item, dict)]
    latest_trace = traces[0] if traces else {}
    return BrainEpisodeActionSummary(
        action_ids=_sorted_unique(
            [
                str(item.get("action_id", "")).strip()
                for item in actions[:_MAX_TRACKED_IDS]
            ]
        ),
        trace_ids=_sorted_unique(
            [
                str(item.get("trace_id", "")).strip()
                for item in traces[:_MAX_TRACKED_IDS]
            ]
        ),
        recovery_ids=_sorted_unique(
            [
                str(item.get("recovery_id", "")).strip()
                for item in recoveries[:_MAX_TRACKED_IDS]
            ]
        ),
        trace_status=_optional_text(latest_trace.get("status")),
        mismatch_codes=_sorted_unique(list(mismatch_codes)[:_MAX_CODE_LIST]),
        repair_codes=_sorted_unique(list(repair_codes)[:_MAX_CODE_LIST]),
        execution_backend=_optional_text(execution_backend),
        preview_only=bool(preview_only),
        recovery_count=len(recoveries),
    )


def _derive_replay_task_success(events: list[dict[str, Any]]) -> bool | None:
    latest_status: bool | None = None
    for event in sorted(events, key=_event_sort_key):
        event_type = str(event.get("event_type") or "").strip()
        if event_type == "goal.completed":
            latest_status = True
        elif event_type in {"goal.failed", "goal.cancelled", "goal.blocked"}:
            latest_status = False
    return latest_status


def _provenance_ids(
    *,
    prediction_summary: BrainEpisodePredictionSummary,
    rehearsal_summary: BrainEpisodeRehearsalSummary,
    action_summary: BrainEpisodeActionSummary,
) -> dict[str, str]:
    rows = {
        "selected_prediction_id": next(iter(prediction_summary.highest_risk_prediction_ids), None)
        or next(iter(prediction_summary.active_prediction_ids), None),
        "selected_rehearsal_id": rehearsal_summary.selected_rehearsal_id,
        "selected_comparison_id": rehearsal_summary.selected_comparison_id,
        "selected_trace_id": next(iter(action_summary.trace_ids), None),
        "selected_recovery_id": next(iter(action_summary.recovery_ids), None),
    }
    return {key: value for key, value in sorted(rows.items()) if value is not None}


def _stable_episode_id(
    *,
    origin: str,
    scenario_id: str,
    scenario_family: str,
    scenario_version: str,
    source_event_ids: tuple[str, ...],
    goal_id: str | None,
    commitment_id: str | None,
    plan_proposal_id: str | None,
    rehearsal_ids: tuple[str, ...],
    trace_ids: tuple[str, ...],
    comparison_ids: tuple[str, ...],
    backend_ids: dict[str, str | None],
) -> str:
    return _stable_id(
        "episode",
        {
            "schema_version": _EPISODE_SCHEMA_VERSION,
            "origin": origin,
            "scenario_id": scenario_id,
            "scenario_family": scenario_family,
            "scenario_version": scenario_version,
            "source_event_ids": list(source_event_ids),
            "goal_id": goal_id,
            "commitment_id": commitment_id,
            "plan_proposal_id": plan_proposal_id,
            "rehearsal_ids": list(rehearsal_ids),
            "trace_ids": list(trace_ids),
            "comparison_ids": list(comparison_ids),
            "backend_ids": {
                key: value
                for key, value in sorted(backend_ids.items())
                if value is not None
            },
        },
    )


def build_episode_from_embodied_eval_run_payload(
    run_payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> BrainEpisodeRecord:
    """Build one canonical episode from one eval-run payload."""
    metrics = dict(run_payload.get("metrics", {}))
    shell_snapshot = dict(run_payload.get("shell_snapshot", {}))
    shell_digest = dict(run_payload.get("shell_digest", {}))
    events = [
        dict(item)
        for item in run_payload.get("event_slice", [])
        if isinstance(item, dict)
    ]
    embodied_inspection = dict(shell_digest.get("embodied_inspection", {}))
    predictive_inspection = dict(shell_digest.get("predictive_inspection", {}))
    rehearsal_inspection = dict(shell_digest.get("rehearsal_inspection", {}))
    current_intent = dict(embodied_inspection.get("current_intent", {}))
    last_envelope = dict(embodied_inspection.get("last_action_envelope", {}))
    recent_traces = [
        dict(item)
        for item in shell_snapshot.get("recent_embodied_execution_traces", [])
        if isinstance(item, dict)
    ] or [
        dict(item)
        for item in embodied_inspection.get("recent_execution_traces", [])
        if isinstance(item, dict)
    ]
    recent_recoveries = [
        dict(item)
        for item in shell_snapshot.get("recent_embodied_recoveries", [])
        if isinstance(item, dict)
    ] or [
        dict(item)
        for item in embodied_inspection.get("recent_recoveries", [])
        if isinstance(item, dict)
    ]
    recent_actions = [
        dict(item)
        for item in embodied_inspection.get("recent_low_level_embodied_actions", [])
        if isinstance(item, dict)
    ]
    goal_id = _first_text_from_payloads(current_intent, last_envelope, events, key="goal_id", prefer_last=True)
    commitment_id = _first_text_from_payloads(
        current_intent,
        events,
        key="commitment_id",
        prefer_last=True,
    )
    plan_proposal_id = _first_text_from_payloads(
        current_intent,
        last_envelope,
        rehearsal_inspection.get("recent_rehearsals", []),
        events,
        key="plan_proposal_id",
        prefer_last=True,
    )
    prediction_summary = _build_prediction_summary(predictive_inspection)
    rehearsal_summary = _build_rehearsal_summary(rehearsal_inspection)
    action_summary = _build_action_summary(
        execution_backend=_optional_text(metrics.get("execution_backend")),
        preview_only=bool(metrics.get("preview_only", False)),
        mismatch_codes=list(metrics.get("mismatch_codes", [])),
        repair_codes=list(metrics.get("repair_codes", [])),
        recent_low_level_actions=recent_actions,
        recent_execution_traces=recent_traces,
        recent_recoveries=recent_recoveries,
    )
    outcome_summary = BrainEpisodeOutcomeSummary(
        goal_status=_optional_text(run_payload.get("goal_status")),
        planning_outcome=_optional_text(run_payload.get("planning_outcome")),
        task_success=bool(metrics.get("task_success", False)),
        trace_status=_optional_text(metrics.get("trace_status")),
        operator_review_floored=int(metrics.get("review_floor_count", 0)) > 0,
        observed_outcome_counts=_sorted_mapping(rehearsal_summary.observed_outcome_counts),
        calibration_bucket_counts=_sorted_mapping(rehearsal_summary.calibration_bucket_counts),
    )
    safety_summary = BrainEpisodeSafetySummary(
        safety_success=bool(metrics.get("safety_success", False)),
        review_floor_count=int(metrics.get("review_floor_count", 0)),
        operator_intervention_count=int(metrics.get("operator_intervention_count", 0)),
        recovery_count=int(metrics.get("recovery_count", 0)),
        risk_codes=_sorted_unique(
            list(rehearsal_summary.risk_code_counts.keys())[:_MAX_CODE_LIST]
        ),
        mismatch_codes=_sorted_unique(list(metrics.get("mismatch_codes", []))[:_MAX_CODE_LIST]),
        repair_codes=_sorted_unique(list(metrics.get("repair_codes", []))[:_MAX_CODE_LIST]),
    )
    source_event_ids = tuple(
        str(event.get("event_id", "")).strip()
        for event in sorted(events, key=_event_sort_key)
        if str(event.get("event_id", "")).strip()
    )
    skill_ids = _skill_ids_from_payloads(events, recent_actions, recent_traces, current_intent)
    started_at, ended_at = _timestamps_from_events(events)
    artifact_refs = _build_artifact_refs(
        run_payload.get("artifact_paths"),
        run_source_path=(Path(run_payload["artifact_paths"]["run_json"]) if dict(run_payload.get("artifact_paths", {})).get("run_json") else source_path),
    )
    episode_id = _stable_episode_id(
        origin=BrainEpisodeOrigin.SIMULATION.value,
        scenario_id=str(run_payload.get("scenario_id", "")).strip(),
        scenario_family=str(run_payload.get("scenario_family", "")).strip(),
        scenario_version=str(
            metrics.get("scenario_version") or run_payload.get("scenario_version") or "v1"
        ).strip(),
        source_event_ids=source_event_ids,
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        rehearsal_ids=rehearsal_summary.rehearsal_ids,
        trace_ids=action_summary.trace_ids,
        comparison_ids=rehearsal_summary.comparison_ids,
        backend_ids={
            "execution_backend": _optional_text(metrics.get("execution_backend")),
            "perception_backend_id": _optional_text(metrics.get("perception_backend_id")),
            "world_model_backend_id": _optional_text(metrics.get("world_model_backend_id")),
            "embodied_policy_backend_id": _optional_text(metrics.get("embodied_policy_backend_id")),
        },
    )
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version=_EPISODE_SCHEMA_VERSION,
        origin=BrainEpisodeOrigin.SIMULATION.value,
        scenario_id=str(run_payload.get("scenario_id", "")).strip(),
        scenario_family=str(run_payload.get("scenario_family", "")).strip(),
        scenario_version=str(
            metrics.get("scenario_version") or run_payload.get("scenario_version") or "v1"
        ).strip(),
        source_run_id=_optional_text(run_payload.get("run_id")),
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_ids=skill_ids,
        execution_backend=_optional_text(metrics.get("execution_backend")),
        perception_backend_id=_optional_text(metrics.get("perception_backend_id")),
        world_model_backend_id=_optional_text(metrics.get("world_model_backend_id")),
        embodied_policy_backend_id=_optional_text(metrics.get("embodied_policy_backend_id")),
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
        provenance_ids=_provenance_ids(
            prediction_summary=prediction_summary,
            rehearsal_summary=rehearsal_summary,
            action_summary=action_summary,
        ),
    )


def build_episode_from_replay_artifact_payload(
    artifact_payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> BrainEpisodeRecord:
    """Build one canonical episode from one replay artifact payload."""
    scenario = dict(artifact_payload.get("scenario", {}))
    continuity_state = dict(artifact_payload.get("continuity_state", {}))
    events = [
        dict(item)
        for item in artifact_payload.get("events", [])
        if isinstance(item, dict)
    ]
    predictive_world_model = dict(continuity_state.get("predictive_world_model", {}))
    counterfactual_rehearsal = dict(continuity_state.get("counterfactual_rehearsal", {}))
    embodied_executive = dict(continuity_state.get("embodied_executive", {}))
    predictive_digest = dict(continuity_state.get("predictive_digest", {}))
    rehearsal_digest = dict(continuity_state.get("rehearsal_digest", {}))
    embodied_digest = dict(continuity_state.get("embodied_digest", {}))
    current_intent = dict(embodied_digest.get("current_intent", {}))
    last_envelope = dict(embodied_digest.get("last_action_envelope", {}))
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
    planning_digest = dict(continuity_state.get("planning_digest", {}))
    goal_id = _first_text_from_payloads(
        current_intent,
        last_envelope,
        planning_digest,
        events,
        key="goal_id",
        prefer_last=True,
    )
    commitment_id = _first_text_from_payloads(
        current_intent,
        planning_digest,
        events,
        key="commitment_id",
        prefer_last=True,
    )
    plan_proposal_id = _first_text_from_payloads(
        current_intent,
        last_envelope,
        planning_digest,
        counterfactual_rehearsal,
        events,
        key="plan_proposal_id",
        prefer_last=True,
    )
    prediction_summary = _build_prediction_summary(predictive_digest, predictive_world_model)
    rehearsal_summary = _build_rehearsal_summary(rehearsal_digest, counterfactual_rehearsal)
    action_summary = _build_action_summary(
        execution_backend=_optional_text(embodied_digest.get("current_low_level_executor")),
        preview_only=False,
        mismatch_codes=list(embodied_digest.get("mismatch_code_counts", {}).keys()),
        repair_codes=list(embodied_digest.get("repair_code_counts", {}).keys()),
        recent_low_level_actions=recent_actions,
        recent_execution_traces=recent_traces,
        recent_recoveries=recent_recoveries,
    )
    review_floor_count = 1 if _optional_text(planning_digest.get("outcome")) in {
        "needs_operator_review",
        "operator_review_required",
    } else 0
    safety_success = False if action_summary.trace_status in {"failed", "aborted"} else None
    outcome_summary = BrainEpisodeOutcomeSummary(
        goal_status=_first_text_from_payloads(planning_digest, events, key="status", prefer_last=True),
        planning_outcome=_optional_text(planning_digest.get("outcome")),
        task_success=_derive_replay_task_success(events),
        trace_status=action_summary.trace_status,
        operator_review_floored=review_floor_count > 0,
        observed_outcome_counts=_sorted_mapping(rehearsal_summary.observed_outcome_counts),
        calibration_bucket_counts=_sorted_mapping(rehearsal_summary.calibration_bucket_counts),
    )
    safety_summary = BrainEpisodeSafetySummary(
        safety_success=safety_success,
        review_floor_count=review_floor_count,
        operator_intervention_count=0,
        recovery_count=action_summary.recovery_count,
        risk_codes=_sorted_unique(
            list(rehearsal_summary.risk_code_counts.keys())[:_MAX_CODE_LIST]
        ),
        mismatch_codes=action_summary.mismatch_codes,
        repair_codes=action_summary.repair_codes,
    )
    source_event_ids = tuple(
        str(event.get("event_id", "")).strip()
        for event in sorted(events, key=_event_sort_key)
        if str(event.get("event_id", "")).strip()
    )
    skill_ids = _skill_ids_from_payloads(
        continuity_state.get("procedural_skills", {}),
        planning_digest,
        events,
        recent_actions,
    )
    started_at, ended_at = _timestamps_from_events(events)
    artifact_paths: dict[str, str] = {}
    if source_path is not None:
        artifact_paths["replay_artifact_json"] = str(source_path)
    latest_reflection_draft_path = _optional_text(artifact_payload.get("latest_reflection_draft_path"))
    if latest_reflection_draft_path is not None:
        artifact_paths["latest_reflection_draft"] = latest_reflection_draft_path
    artifact_refs = _build_artifact_refs(
        artifact_paths,
        include_input_path=source_path,
    )
    scenario_name = str(scenario.get("name", "")).strip()
    scenario_family = _optional_text(scenario.get("family")) or scenario_name or "replay"
    scenario_version = _optional_text(scenario.get("version")) or "replay.v1"
    episode_id = _stable_episode_id(
        origin=BrainEpisodeOrigin.REPLAY.value,
        scenario_id=scenario_name or "replay",
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
            "execution_backend": _optional_text(embodied_digest.get("current_low_level_executor")),
            "world_model_backend_id": _optional_text(
                predictive_world_model.get("world_model_backend_id")
            ),
            "embodied_policy_backend_id": _optional_text(
                _ordered_values_for_key(events, "embodied_policy_backend_id")[-1]
                if _ordered_values_for_key(events, "embodied_policy_backend_id")
                else None
            ),
        },
    )
    return BrainEpisodeRecord(
        episode_id=episode_id,
        schema_version=_EPISODE_SCHEMA_VERSION,
        origin=BrainEpisodeOrigin.REPLAY.value,
        scenario_id=scenario_name or "replay",
        scenario_family=scenario_family,
        scenario_version=scenario_version,
        source_replay_name=scenario_name or "replay",
        goal_id=goal_id,
        commitment_id=commitment_id,
        plan_proposal_id=plan_proposal_id,
        skill_ids=skill_ids,
        execution_backend=_optional_text(embodied_digest.get("current_low_level_executor")),
        perception_backend_id=None,
        world_model_backend_id=None,
        embodied_policy_backend_id=None,
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
        provenance_ids=_provenance_ids(
            prediction_summary=prediction_summary,
            rehearsal_summary=rehearsal_summary,
            action_summary=action_summary,
        ),
    )


__all__ = [
    "BrainEpisodeActionSummary",
    "BrainEpisodeArtifactRef",
    "BrainEpisodeObservationRef",
    "BrainEpisodeOrigin",
    "BrainEpisodeOutcomeSummary",
    "BrainEpisodePredictionSummary",
    "BrainEpisodeRecord",
    "BrainEpisodeRehearsalSummary",
    "BrainEpisodeSafetySummary",
    "build_episode_from_embodied_eval_run_payload",
    "build_episode_from_replay_artifact_payload",
]

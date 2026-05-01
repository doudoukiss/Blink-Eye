"""Deterministic failure-cluster summaries for exported episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.evals.episode_export import BrainEpisodeRecord


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


@dataclass(frozen=True)
class BrainEpisodeFailureClusterRow:
    """One deterministic signature-based failure cluster."""

    cluster_id: str
    signature: str
    scenario_family: str
    origin: str
    execution_backend: str | None
    embodied_policy_backend_id: str | None
    trace_status: str | None
    mismatch_codes: tuple[str, ...] = ()
    repair_codes: tuple[str, ...] = ()
    risk_codes: tuple[str, ...] = ()
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    episode_count: int = 0
    episode_ids: tuple[str, ...] = ()
    task_failure_count: int = 0
    safety_degraded_count: int = 0
    review_floor_count: int = 0
    recovery_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize the failure-cluster row."""
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
            "scenario_family": self.scenario_family,
            "origin": self.origin,
            "execution_backend": self.execution_backend,
            "embodied_policy_backend_id": self.embodied_policy_backend_id,
            "trace_status": self.trace_status,
            "mismatch_codes": list(self.mismatch_codes),
            "repair_codes": list(self.repair_codes),
            "risk_codes": list(self.risk_codes),
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "episode_count": self.episode_count,
            "episode_ids": list(self.episode_ids),
            "task_failure_count": self.task_failure_count,
            "safety_degraded_count": self.safety_degraded_count,
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
        }


def episode_requires_failure_cluster(record: BrainEpisodeRecord) -> bool:
    """Return whether one episode should participate in failure clustering."""
    return (
        record.outcome_summary.task_success is False
        or record.safety_summary.safety_success is False
        or record.safety_summary.review_floor_count > 0
        or record.safety_summary.recovery_count > 0
    )


def failure_cluster_signature(record: BrainEpisodeRecord) -> str:
    """Return the deterministic signature key for one episode."""
    signature_payload = {
        "scenario_family": record.scenario_family,
        "origin": record.origin,
        "execution_backend": record.execution_backend,
        "embodied_policy_backend_id": record.embodied_policy_backend_id,
        "trace_status": record.action_summary.trace_status or record.outcome_summary.trace_status,
        "mismatch_codes": list(record.action_summary.mismatch_codes),
        "repair_codes": list(record.action_summary.repair_codes),
        "risk_codes": list(record.safety_summary.risk_codes),
        "calibration_bucket_counts": dict(record.outcome_summary.calibration_bucket_counts),
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def build_failure_clusters(
    episodes: tuple[BrainEpisodeRecord, ...] | list[BrainEpisodeRecord],
) -> tuple[BrainEpisodeFailureClusterRow, ...]:
    """Build deterministic failure clusters over one episode set."""
    relevant = [
        record
        for record in sorted(
            episodes,
            key=lambda item: (
                item.scenario_family,
                item.origin,
                item.execution_backend or "",
                item.embodied_policy_backend_id or "",
                item.episode_id,
            ),
        )
        if episode_requires_failure_cluster(record)
    ]
    grouped: dict[str, list[BrainEpisodeRecord]] = {}
    for record in relevant:
        grouped.setdefault(failure_cluster_signature(record), []).append(record)
    rows: list[BrainEpisodeFailureClusterRow] = []
    for signature, grouped_records in sorted(grouped.items()):
        exemplar = grouped_records[0]
        calibration_bucket_counts: dict[str, int] = {}
        task_failure_count = 0
        safety_degraded_count = 0
        review_floor_count = 0
        recovery_count = 0
        for record in grouped_records:
            if record.outcome_summary.task_success is False:
                task_failure_count += 1
            if record.safety_summary.safety_success is False:
                safety_degraded_count += 1
            review_floor_count += record.safety_summary.review_floor_count
            recovery_count += record.safety_summary.recovery_count
            for key, value in record.outcome_summary.calibration_bucket_counts.items():
                calibration_bucket_counts[key] = calibration_bucket_counts.get(key, 0) + int(value)
        rows.append(
            BrainEpisodeFailureClusterRow(
                cluster_id=_stable_id(
                    "episode_cluster",
                    {"signature": signature, "episode_ids": [item.episode_id for item in grouped_records]},
                ),
                signature=signature,
                scenario_family=exemplar.scenario_family,
                origin=exemplar.origin,
                execution_backend=exemplar.execution_backend,
                embodied_policy_backend_id=exemplar.embodied_policy_backend_id,
                trace_status=exemplar.action_summary.trace_status or exemplar.outcome_summary.trace_status,
                mismatch_codes=_sorted_unique(list(exemplar.action_summary.mismatch_codes)),
                repair_codes=_sorted_unique(list(exemplar.action_summary.repair_codes)),
                risk_codes=_sorted_unique(list(exemplar.safety_summary.risk_codes)),
                calibration_bucket_counts=_sorted_mapping(calibration_bucket_counts),
                episode_count=len(grouped_records),
                episode_ids=tuple(sorted(record.episode_id for record in grouped_records)),
                task_failure_count=task_failure_count,
                safety_degraded_count=safety_degraded_count,
                review_floor_count=review_floor_count,
                recovery_count=recovery_count,
            )
        )
    return tuple(rows)


__all__ = [
    "BrainEpisodeFailureClusterRow",
    "build_failure_clusters",
    "episode_requires_failure_cluster",
    "failure_cluster_signature",
]

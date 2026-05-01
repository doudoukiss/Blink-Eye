"""Deterministic dataset manifests for exported brain episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.evals.episode_export import BrainEpisodeRecord
from blink.brain.evals.failure_clusters import (
    BrainEpisodeFailureClusterRow,
    build_failure_clusters,
)

_DATASET_SCHEMA_VERSION = "brain_episode_dataset/v1"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_mapping(values: dict[str, Any] | None) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(dict(values or {}).items())
        if _optional_text(key) is not None
    }


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, json.dumps(payload, ensure_ascii=False, sort_keys=True)).hex}"


@dataclass(frozen=True)
class BrainEpisodeFamilyCoverageRow:
    """One deterministic family-coverage row over exported episodes."""

    scenario_family: str
    origin: str
    execution_backend: str | None
    embodied_policy_backend_id: str | None
    episode_count: int
    task_success_count: int
    safety_success_count: int
    review_floor_count: int
    recovery_count: int
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the family-coverage row."""
        return {
            "scenario_family": self.scenario_family,
            "origin": self.origin,
            "execution_backend": self.execution_backend,
            "embodied_policy_backend_id": self.embodied_policy_backend_id,
            "episode_count": self.episode_count,
            "task_success_count": self.task_success_count,
            "safety_success_count": self.safety_success_count,
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
        }


@dataclass(frozen=True)
class BrainEpisodeDatasetManifest:
    """One deterministic manifest over a batch of exported episodes."""

    manifest_id: str
    schema_version: str
    episode_count: int
    episode_ids: tuple[str, ...]
    origins: tuple[str, ...]
    scenario_family_counts: dict[str, int] = field(default_factory=dict)
    family_coverage: tuple[BrainEpisodeFamilyCoverageRow, ...] = ()
    failure_clusters: tuple[BrainEpisodeFailureClusterRow, ...] = ()
    generated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the dataset manifest."""
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "episode_count": self.episode_count,
            "episode_ids": list(self.episode_ids),
            "origins": list(self.origins),
            "scenario_family_counts": dict(self.scenario_family_counts),
            "family_coverage": [row.as_dict() for row in self.family_coverage],
            "failure_clusters": [row.as_dict() for row in self.failure_clusters],
            "generated_at": self.generated_at,
        }

    def render_markdown(self) -> str:
        """Render the manifest as compact operator-readable markdown."""
        lines = [
            "# Brain Episode Dataset Manifest",
            "",
            f"- manifest_id: {self.manifest_id}",
            f"- episode_count: {self.episode_count}",
            f"- origins: {', '.join(self.origins) or 'none'}",
            f"- scenario_families: {', '.join(sorted(self.scenario_family_counts)) or 'none'}",
            "",
            "## Family Coverage",
            "",
            "| family | origin | backend | policy backend | episodes | task success | safety success | review floors | recoveries |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.family_coverage:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.scenario_family,
                        row.origin,
                        row.execution_backend or "unknown",
                        row.embodied_policy_backend_id or "unknown",
                        str(row.episode_count),
                        str(row.task_success_count),
                        str(row.safety_success_count),
                        str(row.review_floor_count),
                        str(row.recovery_count),
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Failure Clusters", ""])
        if not self.failure_clusters:
            lines.append("No failure clusters were produced.")
            return "\n".join(lines)
        lines.extend(
            [
                "| cluster_id | family | origin | backend | episodes | trace_status | mismatch_codes | repair_codes | risk_codes |",
                "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in self.failure_clusters:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.cluster_id,
                        row.scenario_family,
                        row.origin,
                        row.execution_backend or "unknown",
                        str(row.episode_count),
                        row.trace_status or "none",
                        ",".join(row.mismatch_codes) or "none",
                        ",".join(row.repair_codes) or "none",
                        ",".join(row.risk_codes) or "none",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


def build_family_coverage(
    episodes: tuple[BrainEpisodeRecord, ...] | list[BrainEpisodeRecord],
) -> tuple[BrainEpisodeFamilyCoverageRow, ...]:
    """Build deterministic family-coverage rows."""
    grouped: dict[tuple[str, str, str | None, str | None], list[BrainEpisodeRecord]] = {}
    for record in sorted(
        episodes,
        key=lambda item: (
            item.scenario_family,
            item.origin,
            item.execution_backend or "",
            item.embodied_policy_backend_id or "",
            item.episode_id,
        ),
    ):
        grouped.setdefault(
            (
                record.scenario_family,
                record.origin,
                record.execution_backend,
                record.embodied_policy_backend_id,
            ),
            [],
        ).append(record)
    rows: list[BrainEpisodeFamilyCoverageRow] = []
    for key, grouped_records in sorted(grouped.items()):
        calibration_bucket_counts: dict[str, int] = {}
        task_success_count = 0
        safety_success_count = 0
        review_floor_count = 0
        recovery_count = 0
        for record in grouped_records:
            if record.outcome_summary.task_success is True:
                task_success_count += 1
            if record.safety_summary.safety_success is True:
                safety_success_count += 1
            review_floor_count += record.safety_summary.review_floor_count
            recovery_count += record.safety_summary.recovery_count
            for bucket, count in record.outcome_summary.calibration_bucket_counts.items():
                calibration_bucket_counts[bucket] = calibration_bucket_counts.get(bucket, 0) + int(count)
        rows.append(
            BrainEpisodeFamilyCoverageRow(
                scenario_family=key[0],
                origin=key[1],
                execution_backend=key[2],
                embodied_policy_backend_id=key[3],
                episode_count=len(grouped_records),
                task_success_count=task_success_count,
                safety_success_count=safety_success_count,
                review_floor_count=review_floor_count,
                recovery_count=recovery_count,
                calibration_bucket_counts=_sorted_mapping(calibration_bucket_counts),
            )
        )
    return tuple(rows)


def build_episode_dataset_manifest(
    episodes: tuple[BrainEpisodeRecord, ...] | list[BrainEpisodeRecord],
) -> BrainEpisodeDatasetManifest:
    """Build one deterministic manifest from exported episode records."""
    ordered = tuple(
        sorted(
            episodes,
            key=lambda item: (
                item.episode_id,
                item.scenario_family,
                item.origin,
            ),
        )
    )
    episode_ids = tuple(record.episode_id for record in ordered)
    scenario_family_counts: dict[str, int] = {}
    timestamps = [record.generated_at for record in ordered if record.generated_at]
    for record in ordered:
        scenario_family_counts[record.scenario_family] = (
            scenario_family_counts.get(record.scenario_family, 0) + 1
        )
    family_coverage = build_family_coverage(ordered)
    failure_clusters = build_failure_clusters(ordered)
    return BrainEpisodeDatasetManifest(
        manifest_id=_stable_id(
            "episode_dataset",
            {
                "schema_version": _DATASET_SCHEMA_VERSION,
                "episode_ids": list(episode_ids),
            },
        ),
        schema_version=_DATASET_SCHEMA_VERSION,
        episode_count=len(ordered),
        episode_ids=episode_ids,
        origins=tuple(sorted({record.origin for record in ordered})),
        scenario_family_counts=_sorted_mapping(scenario_family_counts),
        family_coverage=family_coverage,
        failure_clusters=failure_clusters,
        generated_at=max(timestamps) if timestamps else None,
    )


__all__ = [
    "BrainEpisodeDatasetManifest",
    "BrainEpisodeFamilyCoverageRow",
    "build_episode_dataset_manifest",
    "build_family_coverage",
]

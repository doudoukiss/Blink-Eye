"""Deterministic simulation-first practice planning above Phase 21A/22 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.evals.dataset_manifest import BrainEpisodeDatasetManifest
from blink.brain.evals.embodied_scenarios import (
    BrainEmbodiedEvalScenario,
    BrainEmbodiedEvalSuite,
    build_benchmark_embodied_eval_suite,
    build_smoke_embodied_eval_suite,
)
from blink.brain.evals.episode_export import BrainEpisodeRecord
from blink.brain.events import BrainEventType

if TYPE_CHECKING:
    from blink.brain.store import BrainStore


_PRACTICE_EVENT_TYPES = frozenset({BrainEventType.PRACTICE_PLAN_CREATED})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _sorted_mapping(values: dict[str, Any] | None) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(dict(values or {}).items())
        if _optional_text(key) is not None
    }


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _target_sort_key(record: "BrainPracticeTarget") -> tuple[float, str, str, str]:
    return (-float(record.score), record.scenario_family, record.scenario_id, record.target_id)


def _plan_sort_key(record: "BrainPracticePlan") -> tuple[str, str]:
    return (record.updated_at, record.plan_id)


def practice_event_types() -> frozenset[str]:
    """Return explicit practice-director lifecycle events."""
    return _PRACTICE_EVENT_TYPES


class BrainPracticeReasonCode(str, Enum):
    """Deterministic reason codes for simulation-first practice selection."""

    FAILURE_CLUSTER_PRESSURE = "failure_cluster_pressure"
    LOW_FAMILY_COVERAGE = "low_family_coverage"
    OVERCONFIDENT_CALIBRATION = "overconfident_calibration"
    REVIEW_FLOOR_PRESSURE = "review_floor_pressure"
    RECOVERY_PRESSURE = "recovery_pressure"
    LOW_CONFIDENCE_SKILL_LINK = "low_confidence_skill_link"
    RETIRED_SKILL_LINK = "retired_skill_link"


@dataclass(frozen=True)
class BrainPracticeTarget:
    """One bounded simulation-first practice target."""

    target_id: str
    scenario_family: str
    scenario_id: str
    scenario_version: str
    suite_id: str
    selected_profile_id: str
    execution_backend: str
    score: float
    reason_codes: list[str] = field(default_factory=list)
    supporting_episode_ids: list[str] = field(default_factory=list)
    failure_cluster_ids: list[str] = field(default_factory=list)
    related_skill_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the practice target."""
        return {
            "target_id": self.target_id,
            "scenario_family": self.scenario_family,
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "suite_id": self.suite_id,
            "selected_profile_id": self.selected_profile_id,
            "execution_backend": self.execution_backend,
            "score": self.score,
            "reason_codes": list(self.reason_codes),
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "failure_cluster_ids": list(self.failure_cluster_ids),
            "related_skill_ids": list(self.related_skill_ids),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPracticeTarget | None":
        """Hydrate one practice target from JSON."""
        if not isinstance(data, dict):
            return None
        target_id = str(data.get("target_id", "")).strip()
        scenario_family = str(data.get("scenario_family", "")).strip()
        scenario_id = str(data.get("scenario_id", "")).strip()
        suite_id = str(data.get("suite_id", "")).strip()
        selected_profile_id = str(data.get("selected_profile_id", "")).strip()
        execution_backend = str(data.get("execution_backend", "")).strip()
        scenario_version = str(data.get("scenario_version", "")).strip()
        if not all(
            (
                target_id,
                scenario_family,
                scenario_id,
                suite_id,
                selected_profile_id,
                execution_backend,
                scenario_version,
            )
        ):
            return None
        return cls(
            target_id=target_id,
            scenario_family=scenario_family,
            scenario_id=scenario_id,
            scenario_version=scenario_version,
            suite_id=suite_id,
            selected_profile_id=selected_profile_id,
            execution_backend=execution_backend,
            score=float(data.get("score", 0.0)),
            reason_codes=_sorted_unique(data.get("reason_codes", [])),
            supporting_episode_ids=_sorted_unique(data.get("supporting_episode_ids", [])),
            failure_cluster_ids=_sorted_unique(data.get("failure_cluster_ids", [])),
            related_skill_ids=_sorted_unique(data.get("related_skill_ids", [])),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainPracticePlan:
    """One bounded deterministic practice plan."""

    plan_id: str
    scope_key: str
    presence_scope_key: str
    dataset_manifest_id: str | None = None
    targets: list[BrainPracticeTarget] = field(default_factory=list)
    reason_code_counts: dict[str, int] = field(default_factory=dict)
    supporting_episode_ids: list[str] = field(default_factory=list)
    failure_cluster_ids: list[str] = field(default_factory=list)
    related_skill_ids: list[str] = field(default_factory=list)
    summary: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the practice plan."""
        return {
            "plan_id": self.plan_id,
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "dataset_manifest_id": self.dataset_manifest_id,
            "targets": [record.as_dict() for record in sorted(self.targets, key=_target_sort_key)],
            "reason_code_counts": dict(self.reason_code_counts),
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "failure_cluster_ids": list(self.failure_cluster_ids),
            "related_skill_ids": list(self.related_skill_ids),
            "summary": self.summary,
            "artifact_paths": dict(self.artifact_paths),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPracticePlan | None":
        """Hydrate one practice plan from JSON."""
        if not isinstance(data, dict):
            return None
        plan_id = str(data.get("plan_id", "")).strip()
        scope_key = str(data.get("scope_key", "")).strip()
        presence_scope_key = str(data.get("presence_scope_key", "")).strip()
        if not plan_id or not scope_key or not presence_scope_key:
            return None
        return cls(
            plan_id=plan_id,
            scope_key=scope_key,
            presence_scope_key=presence_scope_key,
            dataset_manifest_id=_optional_text(data.get("dataset_manifest_id")),
            targets=[
                record
                for item in data.get("targets", [])
                if (record := BrainPracticeTarget.from_dict(item)) is not None
            ],
            reason_code_counts=_sorted_mapping(data.get("reason_code_counts")),
            supporting_episode_ids=_sorted_unique(data.get("supporting_episode_ids", [])),
            failure_cluster_ids=_sorted_unique(data.get("failure_cluster_ids", [])),
            related_skill_ids=_sorted_unique(data.get("related_skill_ids", [])),
            summary=str(data.get("summary", "")).strip(),
            artifact_paths={
                str(key): str(value)
                for key, value in sorted(dict(data.get("artifact_paths", {})).items())
                if _optional_text(key) is not None and _optional_text(value) is not None
            },
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )

    def render_markdown(self) -> str:
        """Render a compact operator-readable practice plan."""
        lines = [
            f"# Brain Practice Plan — {self.plan_id}",
            "",
            f"- summary: {self.summary or 'none'}",
            f"- target_count: {len(self.targets)}",
            f"- reason_codes: {', '.join(f'{key}={value}' for key, value in self.reason_code_counts.items()) or 'none'}",
            "",
        ]
        for target in sorted(self.targets, key=_target_sort_key):
            lines.extend(
                [
                    f"## {target.scenario_id}",
                    "",
                    f"- family: {target.scenario_family}",
                    f"- suite: {target.suite_id}",
                    f"- profile: {target.selected_profile_id}",
                    f"- backend: {target.execution_backend}",
                    f"- score: {target.score:.2f}",
                    f"- reasons: {', '.join(target.reason_codes) or 'none'}",
                    f"- supporting episodes: {', '.join(target.supporting_episode_ids) or 'none'}",
                    f"- related skills: {', '.join(target.related_skill_ids) or 'none'}",
                    "",
                ]
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class BrainPracticeDirectorSummary:
    """Compact operator-facing summary for recent practice planning."""

    recent_plan_ids: tuple[str, ...] = ()
    scenario_family_counts: dict[str, int] = field(default_factory=dict)
    reason_code_counts: dict[str, int] = field(default_factory=dict)
    recent_targets: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the practice summary."""
        return {
            "recent_plan_ids": list(self.recent_plan_ids),
            "scenario_family_counts": dict(self.scenario_family_counts),
            "reason_code_counts": dict(self.reason_code_counts),
            "recent_targets": [dict(item) for item in self.recent_targets],
        }


@dataclass
class BrainPracticeDirectorProjection:
    """Replay-safe recent practice planning projection."""

    scope_key: str
    presence_scope_key: str
    recent_plans: list[BrainPracticePlan] = field(default_factory=list)
    recent_plan_ids: list[str] = field(default_factory=list)
    scenario_family_counts: dict[str, int] = field(default_factory=dict)
    reason_code_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = ""

    def sync_lists(self):
        """Refresh derived ids and counters."""
        self.recent_plans = sorted(self.recent_plans, key=_plan_sort_key, reverse=True)[:12]
        scenario_family_counts: dict[str, int] = {}
        reason_code_counts: dict[str, int] = {}
        for plan in self.recent_plans:
            for target in plan.targets:
                scenario_family_counts[target.scenario_family] = (
                    scenario_family_counts.get(target.scenario_family, 0) + 1
                )
                for reason_code in target.reason_codes:
                    reason_code_counts[reason_code] = reason_code_counts.get(reason_code, 0) + 1
        self.recent_plan_ids = [record.plan_id for record in self.recent_plans]
        self.scenario_family_counts = dict(sorted(scenario_family_counts.items()))
        self.reason_code_counts = dict(sorted(reason_code_counts.items()))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the practice projection."""
        self.sync_lists()
        return {
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "recent_plans": [record.as_dict() for record in self.recent_plans],
            "recent_plan_ids": list(self.recent_plan_ids),
            "scenario_family_counts": dict(self.scenario_family_counts),
            "reason_code_counts": dict(self.reason_code_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPracticeDirectorProjection":
        """Hydrate the practice projection from JSON."""
        payload = dict(data or {})
        projection = cls(
            scope_key=str(payload.get("scope_key", "")).strip(),
            presence_scope_key=str(payload.get("presence_scope_key", "")).strip()
            or "local:presence",
            recent_plans=[
                record
                for item in payload.get("recent_plans", [])
                if (record := BrainPracticePlan.from_dict(item)) is not None
            ],
            recent_plan_ids=_sorted_unique(payload.get("recent_plan_ids", [])),
            scenario_family_counts=_sorted_mapping(payload.get("scenario_family_counts")),
            reason_code_counts=_sorted_mapping(payload.get("reason_code_counts")),
            updated_at=str(payload.get("updated_at") or ""),
        )
        projection.sync_lists()
        return projection


def append_practice_plan(
    projection: BrainPracticeDirectorProjection,
    plan: BrainPracticePlan,
) -> None:
    """Append one practice plan into the replay-safe projection."""
    projection.recent_plans = [
        record for record in projection.recent_plans if record.plan_id != plan.plan_id
    ]
    projection.recent_plans.append(plan)
    projection.updated_at = max(projection.updated_at, plan.updated_at)
    projection.sync_lists()


def _default_practice_suites() -> tuple[BrainEmbodiedEvalSuite, ...]:
    return (
        build_smoke_embodied_eval_suite(),
        build_benchmark_embodied_eval_suite(),
    )


def _simulation_catalog(
    suites: Iterable[BrainEmbodiedEvalSuite],
) -> dict[str, tuple[str, BrainEmbodiedEvalScenario, str]]:
    catalog: dict[str, tuple[str, BrainEmbodiedEvalScenario, str]] = {}
    for suite in sorted(suites, key=lambda item: item.suite_id):
        for scenario in sorted(suite.scenarios, key=lambda item: item.scenario_id):
            simulation_profile_id = next(
                (
                    entry.profile.profile_id
                    for entry in scenario.adapter_matrix.entries
                    if entry.profile.execution_backend == "simulation"
                ),
                None,
            )
            if simulation_profile_id is None:
                continue
            existing = catalog.get(scenario.family)
            candidate = (suite.suite_id, scenario, simulation_profile_id)
            if existing is None or (scenario.scenario_id, suite.suite_id) < (
                existing[1].scenario_id,
                existing[0],
            ):
                catalog[scenario.family] = candidate
    return catalog


def build_practice_plan(
    *,
    episodes: Iterable[BrainEpisodeRecord],
    dataset_manifest: BrainEpisodeDatasetManifest,
    procedural_skill_governance_report: dict[str, Any] | None,
    scope_key: str,
    presence_scope_key: str,
    suites: Iterable[BrainEmbodiedEvalSuite] | None = None,
    max_targets: int = 3,
) -> BrainPracticePlan:
    """Build one deterministic bounded practice plan from landed evidence surfaces."""
    suites = tuple(suites or _default_practice_suites())
    catalog = _simulation_catalog(suites)
    low_confidence_skill_ids = set(
        _sorted_unique(
            (procedural_skill_governance_report or {}).get("low_confidence_skill_ids", [])
        )
    )
    retired_skill_ids = set(
        _sorted_unique((procedural_skill_governance_report or {}).get("retired_skill_ids", []))
    )
    episodes_by_family: dict[str, list[BrainEpisodeRecord]] = {}
    for episode in sorted(list(episodes), key=lambda item: (item.scenario_family, item.episode_id)):
        episodes_by_family.setdefault(episode.scenario_family, []).append(episode)
    coverage_by_family: dict[str, list[Any]] = {}
    for row in dataset_manifest.family_coverage:
        coverage_by_family.setdefault(row.scenario_family, []).append(row)
    clusters_by_family: dict[str, list[Any]] = {}
    for row in dataset_manifest.failure_clusters:
        clusters_by_family.setdefault(row.scenario_family, []).append(row)
    scored_targets: list[BrainPracticeTarget] = []
    for scenario_family, candidate in sorted(catalog.items()):
        suite_id, scenario, simulation_profile_id = candidate
        family_episodes = episodes_by_family.get(scenario_family, [])
        family_cover_rows = coverage_by_family.get(scenario_family, [])
        family_clusters = clusters_by_family.get(scenario_family, [])
        episode_count = sum(int(getattr(row, "episode_count", 0)) for row in family_cover_rows)
        review_floor_count = sum(
            int(getattr(row, "review_floor_count", 0)) for row in family_cover_rows
        )
        recovery_count = sum(int(getattr(row, "recovery_count", 0)) for row in family_cover_rows)
        overconfident_count = sum(
            int(dict(getattr(row, "calibration_bucket_counts", {}) or {}).get("overconfident", 0))
            for row in family_cover_rows
        )
        failure_cluster_pressure = sum(int(getattr(row, "episode_count", 0)) for row in family_clusters)
        linked_skill_ids = _sorted_unique(
            skill_id
            for episode in family_episodes
            for skill_id in episode.skill_ids
        )
        low_confidence_linked_skill_ids = sorted(low_confidence_skill_ids.intersection(linked_skill_ids))
        retired_linked_skill_ids = sorted(retired_skill_ids.intersection(linked_skill_ids))
        score = 0.0
        reason_codes: list[str] = []
        if failure_cluster_pressure > 0:
            score += 4.0 + failure_cluster_pressure
            reason_codes.append(BrainPracticeReasonCode.FAILURE_CLUSTER_PRESSURE.value)
        if episode_count < 2:
            score += 3.0
            reason_codes.append(BrainPracticeReasonCode.LOW_FAMILY_COVERAGE.value)
        if overconfident_count > 0:
            score += 2.0 + overconfident_count
            reason_codes.append(BrainPracticeReasonCode.OVERCONFIDENT_CALIBRATION.value)
        if review_floor_count > 0:
            score += float(review_floor_count)
            reason_codes.append(BrainPracticeReasonCode.REVIEW_FLOOR_PRESSURE.value)
        if recovery_count > 0:
            score += float(recovery_count)
            reason_codes.append(BrainPracticeReasonCode.RECOVERY_PRESSURE.value)
        if low_confidence_linked_skill_ids:
            score += 2.0
            reason_codes.append(BrainPracticeReasonCode.LOW_CONFIDENCE_SKILL_LINK.value)
        if retired_linked_skill_ids:
            score += 2.0
            reason_codes.append(BrainPracticeReasonCode.RETIRED_SKILL_LINK.value)
        if score <= 0.0:
            continue
        supporting_episode_ids = _sorted_unique(episode.episode_id for episode in family_episodes)
        failure_cluster_ids = _sorted_unique(row.cluster_id for row in family_clusters)
        related_skill_ids = _sorted_unique([*low_confidence_linked_skill_ids, *retired_linked_skill_ids])
        target_id = _stable_id(
            "practice_target",
            scope_key,
            scenario_family,
            scenario.scenario_id,
            simulation_profile_id,
        )
        scored_targets.append(
            BrainPracticeTarget(
                target_id=target_id,
                scenario_family=scenario_family,
                scenario_id=scenario.scenario_id,
                scenario_version=scenario.version,
                suite_id=suite_id,
                selected_profile_id=simulation_profile_id,
                execution_backend="simulation",
                score=score,
                reason_codes=_sorted_unique(reason_codes),
                supporting_episode_ids=supporting_episode_ids,
                failure_cluster_ids=failure_cluster_ids,
                related_skill_ids=related_skill_ids,
                details={
                    "episode_count": episode_count,
                    "review_floor_count": review_floor_count,
                    "recovery_count": recovery_count,
                    "overconfident_count": overconfident_count,
                    "failure_cluster_pressure": failure_cluster_pressure,
                },
            )
        )
    targets = sorted(scored_targets, key=_target_sort_key)[:max_targets]
    reason_code_counts: dict[str, int] = {}
    supporting_episode_ids: list[str] = []
    failure_cluster_ids: list[str] = []
    related_skill_ids: list[str] = []
    for target in targets:
        supporting_episode_ids.extend(target.supporting_episode_ids)
        failure_cluster_ids.extend(target.failure_cluster_ids)
        related_skill_ids.extend(target.related_skill_ids)
        for reason_code in target.reason_codes:
            reason_code_counts[reason_code] = reason_code_counts.get(reason_code, 0) + 1
    summary = (
        f"Selected {len(targets)} simulation-first practice targets across "
        f"{len({target.scenario_family for target in targets})} scenario families."
        if targets
        else "No bounded simulation-first practice targets were selected."
    )
    updated_at_candidates = [
        _optional_text(getattr(episode, "generated_at", None))
        or _optional_text(getattr(episode, "ended_at", None))
        or _optional_text(getattr(episode, "started_at", None))
        for target in targets
        for episode in episodes_by_family.get(target.scenario_family, [])
    ]
    updated_at = max(
        [
            value
            for value in [
                *updated_at_candidates,
                _optional_text(dataset_manifest.generated_at),
            ]
            if value is not None
        ]
        or [""]
    )
    return BrainPracticePlan(
        plan_id=_stable_id(
            "practice_plan",
            scope_key,
            dataset_manifest.manifest_id,
            *(target.target_id for target in targets),
        ),
        scope_key=scope_key,
        presence_scope_key=presence_scope_key,
        dataset_manifest_id=dataset_manifest.manifest_id,
        targets=targets,
        reason_code_counts=dict(sorted(reason_code_counts.items())),
        supporting_episode_ids=_sorted_unique(supporting_episode_ids),
        failure_cluster_ids=_sorted_unique(failure_cluster_ids),
        related_skill_ids=_sorted_unique(related_skill_ids),
        summary=summary,
        created_at=updated_at,
        updated_at=updated_at,
        details={
            "scenario_family_counts": dict(dataset_manifest.scenario_family_counts),
            "source_episode_count": int(dataset_manifest.episode_count),
        },
    )


def build_practice_director_summary(
    projection: BrainPracticeDirectorProjection,
    *,
    recent_limit: int = 6,
) -> BrainPracticeDirectorSummary:
    """Build a bounded operator-facing summary from recent practice plans."""
    projection.sync_lists()
    recent_targets = [
        {
            "plan_id": plan.plan_id,
            "scenario_id": target.scenario_id,
            "scenario_family": target.scenario_family,
            "selected_profile_id": target.selected_profile_id,
            "execution_backend": target.execution_backend,
            "score": target.score,
            "reason_codes": list(target.reason_codes),
            "related_skill_ids": list(target.related_skill_ids),
        }
        for plan in projection.recent_plans[:recent_limit]
        for target in sorted(plan.targets, key=_target_sort_key)
    ][:recent_limit]
    return BrainPracticeDirectorSummary(
        recent_plan_ids=tuple(projection.recent_plan_ids[:recent_limit]),
        scenario_family_counts=dict(projection.scenario_family_counts),
        reason_code_counts=dict(projection.reason_code_counts),
        recent_targets=tuple(recent_targets),
    )


def build_practice_inspection(
    practice_projection: BrainPracticeDirectorProjection,
    *,
    recent_limit: int = 6,
) -> dict[str, Any]:
    """Build a compact practice inspection payload for runtime shell and audits."""
    summary = build_practice_director_summary(practice_projection, recent_limit=recent_limit)
    recent_plans = [
        {
            "plan_id": plan.plan_id,
            "dataset_manifest_id": plan.dataset_manifest_id,
            "target_count": len(plan.targets),
            "reason_code_counts": dict(plan.reason_code_counts),
            "summary": plan.summary,
            "artifact_paths": dict(plan.artifact_paths),
            "updated_at": plan.updated_at,
        }
        for plan in practice_projection.recent_plans[:recent_limit]
    ]
    return {
        **summary.as_dict(),
        "recent_plans": recent_plans,
    }


def write_practice_plan_artifacts(
    *,
    plan: BrainPracticePlan,
    output_dir: Path,
) -> BrainPracticePlan:
    """Write one file-first practice plan artifact set and return an updated plan."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{plan.plan_id}.json"
    markdown_path = output_dir / f"{plan.plan_id}.md"
    json_path.write_text(
        json.dumps(plan.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(plan.render_markdown(), encoding="utf-8")
    payload = plan.as_dict()
    payload["artifact_paths"] = {
        **dict(plan.artifact_paths),
        "practice_plan_json": str(json_path),
        "practice_plan_markdown": str(markdown_path),
    }
    return BrainPracticePlan.from_dict(payload) or plan


class BrainPracticeDirector:
    """Build and optionally record deterministic simulation-first practice plans."""

    def __init__(
        self,
        *,
        store: BrainStore | None = None,
        session_resolver: Any | None = None,
        presence_scope_key: str = "local:presence",
    ):
        """Initialize the optional store-backed practice director."""
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key

    def create_plan(
        self,
        *,
        episodes: Iterable[BrainEpisodeRecord],
        dataset_manifest: BrainEpisodeDatasetManifest,
        procedural_skill_governance_report: dict[str, Any] | None,
        scope_key: str,
        suites: Iterable[BrainEmbodiedEvalSuite] | None = None,
        output_dir: Path | None = None,
    ) -> BrainPracticePlan:
        """Build and optionally persist one practice plan."""
        plan = build_practice_plan(
            episodes=episodes,
            dataset_manifest=dataset_manifest,
            procedural_skill_governance_report=procedural_skill_governance_report,
            scope_key=scope_key,
            presence_scope_key=self._presence_scope_key,
            suites=suites,
        )
        if output_dir is not None:
            plan = write_practice_plan_artifacts(plan=plan, output_dir=output_dir)
        if self._store is not None and self._session_resolver is not None:
            session_ids = self._session_resolver()
            self._store.append_brain_event(
                event_type=BrainEventType.PRACTICE_PLAN_CREATED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="practice_director",
                payload={
                    "scope_key": scope_key,
                    "presence_scope_key": self._presence_scope_key,
                    "practice_plan": plan.as_dict(),
                },
            )
        return plan


__all__ = [
    "BrainPracticeDirector",
    "BrainPracticeDirectorProjection",
    "BrainPracticeDirectorSummary",
    "BrainPracticePlan",
    "BrainPracticeReasonCode",
    "BrainPracticeTarget",
    "append_practice_plan",
    "build_practice_director_summary",
    "build_practice_inspection",
    "build_practice_plan",
    "practice_event_types",
    "write_practice_plan_artifacts",
]

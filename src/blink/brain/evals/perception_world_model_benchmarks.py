"""Deterministic perception and world-model benchmark evidence reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters.cards import BrainAdapterFamily, BrainAdapterWeakFamilySummary
from blink.brain.evals.adapter_promotion import (
    BrainAdapterBenchmarkComparisonReport,
    BrainAdapterComparisonFamilyRow,
)

PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS = (
    "spatial_reasoning",
    "affordance_prediction",
    "temporal_consistency",
    "abstention_under_insufficient_evidence",
    "success_detection",
    "mismatch_recovery",
)

_SUPPORTED_ADAPTER_FAMILIES = frozenset(
    {
        BrainAdapterFamily.PERCEPTION.value,
        BrainAdapterFamily.WORLD_MODEL.value,
    }
)
_CRITICAL_MISMATCH_CODES = frozenset(
    {
        "failed_to_abstain",
        "false_affordance_claim",
        "hallucinated_visible_object",
        "missed_success_detection",
        "temporal_contradiction",
        "unsafe",
    }
)


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


def _metric_row_sort_key(record: "BrainPerceptionWorldModelMetricRow") -> tuple[str, ...]:
    return (
        record.scenario_family,
        record.scenario_id,
        str(record.matrix_index),
        record.profile_id,
        record.run_id,
    )


def _family_sort_key(record: BrainAdapterComparisonFamilyRow) -> tuple[str, str]:
    return (record.scenario_family, record.family_key)


def _normalize_adapter_family(value: BrainAdapterFamily | str) -> str:
    text = value.value if isinstance(value, BrainAdapterFamily) else str(value).strip()
    if text not in _SUPPORTED_ADAPTER_FAMILIES:
        raise ValueError(f"Unsupported perception/world-model adapter family: {value}")
    return text


def _round_points(value: float) -> float:
    return round(float(value), 3)


def _rate(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (float(successes) / float(total)) * 100.0


def _count_codes(
    rows: Iterable["BrainPerceptionWorldModelMetricRow"], field_name: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for code in getattr(row, field_name):
            counts[code] = counts.get(code, 0) + 1
    return _sorted_mapping(counts)


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return _round_points(sum(numbers) / len(numbers))


@dataclass(frozen=True)
class BrainPerceptionWorldModelMetricRow:
    """One bounded metric row for perception or world-model evidence."""

    run_id: str
    scenario_id: str
    scenario_family: str
    scenario_version: str
    profile_id: str
    matrix_index: int
    evaluation_backend: str
    adapter_family: str
    backend_id: str
    backend_version: str
    spatial_reasoning_success: bool
    affordance_prediction_success: bool
    temporal_consistency_success: bool
    abstention_success: bool
    success_detection_success: bool
    mismatch_recovery_success: bool
    occlusion_present: bool = False
    insufficient_evidence_present: bool = False
    mismatch_codes: tuple[str, ...] = ()
    recovery_codes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    updated_at: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the metric row."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "scenario_version": self.scenario_version,
            "profile_id": self.profile_id,
            "matrix_index": self.matrix_index,
            "evaluation_backend": self.evaluation_backend,
            "adapter_family": self.adapter_family,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "spatial_reasoning_success": self.spatial_reasoning_success,
            "affordance_prediction_success": self.affordance_prediction_success,
            "temporal_consistency_success": self.temporal_consistency_success,
            "abstention_success": self.abstention_success,
            "success_detection_success": self.success_detection_success,
            "mismatch_recovery_success": self.mismatch_recovery_success,
            "occlusion_present": self.occlusion_present,
            "insufficient_evidence_present": self.insufficient_evidence_present,
            "mismatch_codes": _sorted_unique(self.mismatch_codes),
            "recovery_codes": _sorted_unique(self.recovery_codes),
            "reason_codes": _sorted_unique(self.reason_codes),
            "updated_at": self.updated_at,
            "details": {str(key): value for key, value in sorted(self.details.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPerceptionWorldModelMetricRow | None":
        """Hydrate one metric row from JSON."""
        if not isinstance(data, dict):
            return None
        run_id = str(data.get("run_id", "")).strip()
        scenario_id = str(data.get("scenario_id", "")).strip()
        scenario_family = str(data.get("scenario_family", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        backend_id = str(data.get("backend_id", "")).strip()
        if not all((run_id, scenario_id, scenario_family, adapter_family, backend_id)):
            return None
        return cls(
            run_id=run_id,
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            scenario_version=str(data.get("scenario_version", "")).strip(),
            profile_id=str(data.get("profile_id", "")).strip(),
            matrix_index=int(data.get("matrix_index", 0)),
            evaluation_backend=str(data.get("evaluation_backend", "")).strip(),
            adapter_family=adapter_family,
            backend_id=backend_id,
            backend_version=str(data.get("backend_version", "")).strip(),
            spatial_reasoning_success=bool(data.get("spatial_reasoning_success", False)),
            affordance_prediction_success=bool(data.get("affordance_prediction_success", False)),
            temporal_consistency_success=bool(data.get("temporal_consistency_success", False)),
            abstention_success=bool(data.get("abstention_success", False)),
            success_detection_success=bool(data.get("success_detection_success", False)),
            mismatch_recovery_success=bool(data.get("mismatch_recovery_success", False)),
            occlusion_present=bool(data.get("occlusion_present", False)),
            insufficient_evidence_present=bool(data.get("insufficient_evidence_present", False)),
            mismatch_codes=tuple(_sorted_unique(data.get("mismatch_codes", []))),
            recovery_codes=tuple(_sorted_unique(data.get("recovery_codes", []))),
            reason_codes=tuple(_sorted_unique(data.get("reason_codes", []))),
            updated_at=str(data.get("updated_at") or ""),
            details=dict(data.get("details", {})),
        )


def _metric_delta(
    pairs: list[tuple[BrainPerceptionWorldModelMetricRow, BrainPerceptionWorldModelMetricRow]],
    field_name: str,
    *,
    abstention_only: bool = False,
) -> tuple[float, float, float, int]:
    denominator_pairs = [
        pair
        for pair in pairs
        if not abstention_only
        or pair[0].occlusion_present
        or pair[0].insufficient_evidence_present
        or pair[1].occlusion_present
        or pair[1].insufficient_evidence_present
    ]
    total = len(denominator_pairs)
    incumbent_success = sum(
        int(bool(getattr(incumbent, field_name))) for incumbent, _ in denominator_pairs
    )
    candidate_success = sum(
        int(bool(getattr(candidate, field_name))) for _, candidate in denominator_pairs
    )
    incumbent_rate = _rate(incumbent_success, total)
    candidate_rate = _rate(candidate_success, total)
    return (
        _round_points(candidate_rate - incumbent_rate),
        _round_points(incumbent_rate),
        _round_points(candidate_rate),
        total,
    )


def _family_evidence_details(
    *,
    scenario_family: str,
    pairs: list[tuple[BrainPerceptionWorldModelMetricRow, BrainPerceptionWorldModelMetricRow]],
    metric_deltas: dict[str, float],
    incumbent_rates: dict[str, float],
    candidate_rates: dict[str, float],
    abstention_case_count: int,
    candidate_mismatch_counts: dict[str, int],
    candidate_recovery_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "scenario_family": scenario_family,
        "scenario_count": len(pairs),
        "abstention_case_count": abstention_case_count,
        "candidate_mismatch_count": sum(candidate_mismatch_counts.values()),
        "candidate_recovery_count": sum(candidate_recovery_counts.values()),
        "candidate_mismatch_code_counts": dict(candidate_mismatch_counts),
        "candidate_recovery_code_counts": dict(candidate_recovery_counts),
        "incumbent_rates": dict(sorted(incumbent_rates.items())),
        "candidate_rates": dict(sorted(candidate_rates.items())),
        "metric_delta_points": dict(sorted(metric_deltas.items())),
    }


def build_perception_world_model_benchmark_comparison_report(
    metric_rows: tuple[BrainPerceptionWorldModelMetricRow, ...]
    | list[BrainPerceptionWorldModelMetricRow],
    *,
    adapter_family: BrainAdapterFamily | str,
    incumbent_backend_id: str,
    candidate_backend_id: str,
    incumbent_backend_version: str = "v1",
    candidate_backend_version: str = "v1",
    target_families: Iterable[str | None] = (),
    smoke_suite_green: bool = True,
    updated_at: str | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> BrainAdapterBenchmarkComparisonReport:
    """Build a deterministic perception/world-model candidate comparison report."""
    resolved_family = _normalize_adapter_family(adapter_family)
    rows = sorted(
        [
            row
            for row in metric_rows
            if row.adapter_family == resolved_family
            and row.backend_id in {incumbent_backend_id, candidate_backend_id}
        ],
        key=_metric_row_sort_key,
    )
    target_family_list = _sorted_unique(target_families)
    grouped_by_scenario: dict[str, list[BrainPerceptionWorldModelMetricRow]] = {}
    for row in rows:
        grouped_by_scenario.setdefault(row.scenario_id, []).append(row)

    pairs_by_family: dict[
        str, list[tuple[BrainPerceptionWorldModelMetricRow, BrainPerceptionWorldModelMetricRow]]
    ] = {}
    scenario_ids_by_family: dict[str, set[str]] = {}
    for scenario_id, scenario_rows in sorted(grouped_by_scenario.items()):
        incumbent = next(
            (
                row
                for row in sorted(scenario_rows, key=_metric_row_sort_key)
                if row.backend_id == incumbent_backend_id
            ),
            None,
        )
        candidate = next(
            (
                row
                for row in sorted(scenario_rows, key=_metric_row_sort_key)
                if row.backend_id == candidate_backend_id
            ),
            None,
        )
        if incumbent is None or candidate is None:
            continue
        pairs_by_family.setdefault(incumbent.scenario_family, []).append((incumbent, candidate))
        scenario_ids_by_family.setdefault(incumbent.scenario_family, set()).add(scenario_id)

    family_rows: list[BrainAdapterComparisonFamilyRow] = []
    weak_families: list[BrainAdapterWeakFamilySummary] = []
    blocking_families: list[str] = []
    blocked_reason_codes: set[str] = set()
    family_evidence: list[dict[str, Any]] = []
    shared_scenarios = sum(len(values) for values in pairs_by_family.values())
    report_mismatch_counts: dict[str, int] = {}
    report_recovery_counts: dict[str, int] = {}
    report_abstention_case_count = 0

    if not pairs_by_family:
        blocked_reason_codes.add("missing_shared_family_evidence")

    for scenario_family, pairs in sorted(pairs_by_family.items()):
        metric_deltas: dict[str, float] = {}
        incumbent_rates: dict[str, float] = {}
        candidate_rates: dict[str, float] = {}
        for metric_name, field_name, abstention_only in (
            ("spatial_reasoning", "spatial_reasoning_success", False),
            ("affordance_prediction", "affordance_prediction_success", False),
            ("temporal_consistency", "temporal_consistency_success", False),
            ("abstention_under_insufficient_evidence", "abstention_success", True),
            ("success_detection", "success_detection_success", False),
            ("mismatch_recovery", "mismatch_recovery_success", False),
        ):
            delta, incumbent_rate, candidate_rate, denominator = _metric_delta(
                pairs, field_name, abstention_only=abstention_only
            )
            metric_deltas[f"{metric_name}_delta_points"] = delta
            incumbent_rates[metric_name] = incumbent_rate
            candidate_rates[metric_name] = candidate_rate
            if metric_name == "abstention_under_insufficient_evidence":
                report_abstention_case_count += denominator
                abstention_case_count = denominator

        incumbent_rows = [incumbent for incumbent, _ in pairs]
        candidate_rows = [candidate for _, candidate in pairs]
        incumbent_mismatch_counts = _count_codes(incumbent_rows, "mismatch_codes")
        candidate_mismatch_counts = _count_codes(candidate_rows, "mismatch_codes")
        candidate_recovery_counts = _count_codes(candidate_rows, "recovery_codes")
        for code, count in candidate_mismatch_counts.items():
            report_mismatch_counts[code] = report_mismatch_counts.get(code, 0) + count
        for code, count in candidate_recovery_counts.items():
            report_recovery_counts[code] = report_recovery_counts.get(code, 0) + count

        candidate_mismatch_total = sum(candidate_mismatch_counts.values())
        incumbent_mismatch_total = sum(incumbent_mismatch_counts.values())
        critical_signature = any(
            code in _CRITICAL_MISMATCH_CODES for code in candidate_mismatch_counts
        )
        family_blockers: list[str] = []
        regression_checks = (
            ("spatial_reasoning_regressed", "spatial_reasoning_delta_points"),
            ("affordance_prediction_regressed", "affordance_prediction_delta_points"),
            ("temporal_consistency_regressed", "temporal_consistency_delta_points"),
            (
                "abstention_regressed",
                "abstention_under_insufficient_evidence_delta_points",
            ),
            ("success_detection_regressed", "success_detection_delta_points"),
            ("mismatch_recovery_regressed", "mismatch_recovery_delta_points"),
        )
        for reason_code, delta_key in regression_checks:
            if metric_deltas[delta_key] < 0:
                family_blockers.append(reason_code)
        if critical_signature:
            family_blockers.append("new_critical_failure_signature")
        if (
            "failed_to_abstain" in candidate_mismatch_counts
            and abstention_case_count > 0
            and "abstention_regressed" not in family_blockers
        ):
            family_blockers.append("abstention_regressed")
        if candidate_mismatch_total > incumbent_mismatch_total:
            family_blockers.append("mismatch_count_regressed")

        task_delta = _mean(
            (
                metric_deltas["spatial_reasoning_delta_points"],
                metric_deltas["affordance_prediction_delta_points"],
                metric_deltas["temporal_consistency_delta_points"],
            )
        )
        safety_delta = _mean(
            (
                metric_deltas["abstention_under_insufficient_evidence_delta_points"],
                metric_deltas["success_detection_delta_points"],
                metric_deltas["mismatch_recovery_delta_points"],
            )
        )
        operator_burden_delta = float(candidate_mismatch_total - incumbent_mismatch_total)
        recovery_delta = metric_deltas["mismatch_recovery_delta_points"]
        weak_family = bool(family_blockers)
        family_reason_codes = _sorted_unique(family_blockers)
        if family_reason_codes:
            blocking_families.append(scenario_family)
            blocked_reason_codes.update(family_reason_codes)

        evidence_details = _family_evidence_details(
            scenario_family=scenario_family,
            pairs=pairs,
            metric_deltas=metric_deltas,
            incumbent_rates=incumbent_rates,
            candidate_rates=candidate_rates,
            abstention_case_count=abstention_case_count,
            candidate_mismatch_counts=candidate_mismatch_counts,
            candidate_recovery_counts=candidate_recovery_counts,
        )
        family_evidence.append(evidence_details)
        family_rows.append(
            BrainAdapterComparisonFamilyRow(
                family_key=_stable_id(
                    "adapter_family_row",
                    resolved_family,
                    scenario_family,
                    incumbent_backend_id,
                    candidate_backend_id,
                ),
                scenario_family=scenario_family,
                scenario_ids=sorted(scenario_ids_by_family.get(scenario_family, set())),
                scenario_count=len(pairs),
                incumbent_backend_id=incumbent_backend_id,
                candidate_backend_id=candidate_backend_id,
                task_success_delta_points=task_delta,
                safety_success_delta_points=safety_delta,
                operator_burden_delta_points=operator_burden_delta,
                recovery_delta_points=recovery_delta,
                preview_only_delta_count=0,
                mismatch_code_counts=dict(candidate_mismatch_counts),
                repair_code_counts=dict(candidate_recovery_counts),
                weak_family=weak_family,
                blocking_reason_codes=family_reason_codes,
                details=evidence_details,
            )
        )
        if weak_family:
            weak_families.append(
                BrainAdapterWeakFamilySummary(
                    family_key=_stable_id(
                        "adapter_weak_family",
                        resolved_family,
                        scenario_family,
                        incumbent_backend_id,
                        candidate_backend_id,
                    ),
                    scenario_family=scenario_family,
                    issue_count=len(family_reason_codes) or 1,
                    blocking=bool(family_reason_codes),
                    reason_codes=family_reason_codes,
                    task_success_delta_points=task_delta,
                    safety_success_delta_points=safety_delta,
                    operator_burden_delta_points=operator_burden_delta,
                    recovery_delta_points=recovery_delta,
                    details={
                        "scenario_ids": sorted(scenario_ids_by_family.get(scenario_family, set())),
                        "evidence_metric_names": list(PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS),
                    },
                )
            )

    if not smoke_suite_green:
        blocked_reason_codes.add("smoke_suite_regressed")
    primary_target_family = next(iter(target_family_list), None) or next(
        iter(sorted(pairs_by_family)), None
    )
    primary_target_row = next(
        (row for row in family_rows if row.scenario_family == primary_target_family),
        None,
    )
    if primary_target_row is None:
        blocked_reason_codes.add("missing_primary_target_family")
    else:
        primary_metric_deltas = dict(primary_target_row.details.get("metric_delta_points", {}))
        primary_improved = (
            any(float(value) >= 3.0 for value in primary_metric_deltas.values())
            or primary_target_row.operator_burden_delta_points < 0
        )
        if not primary_improved:
            blocked_reason_codes.add("no_primary_family_improvement")

    benchmark_passed = not blocked_reason_codes
    summary = (
        f"Candidate {candidate_backend_id}@{candidate_backend_version} "
        f"{'passed' if benchmark_passed else 'did not pass'} {resolved_family} "
        f"evidence against {incumbent_backend_id}@{incumbent_backend_version} "
        f"across {len(family_rows)} compared families."
    )
    scenario_ids = sorted(
        scenario_id for ids in scenario_ids_by_family.values() for scenario_id in ids
    )
    return BrainAdapterBenchmarkComparisonReport(
        report_id=_stable_id(
            "adapter_benchmark_report",
            resolved_family,
            incumbent_backend_id,
            incumbent_backend_version,
            candidate_backend_id,
            candidate_backend_version,
            *scenario_ids,
        ),
        adapter_family=resolved_family,
        incumbent_backend_id=incumbent_backend_id,
        incumbent_backend_version=incumbent_backend_version,
        candidate_backend_id=candidate_backend_id,
        candidate_backend_version=candidate_backend_version,
        scenario_count=shared_scenarios,
        compared_family_count=len(family_rows),
        target_families=target_family_list,
        family_rows=sorted(family_rows, key=_family_sort_key),
        weak_families=sorted(weak_families, key=lambda row: (row.scenario_family, row.family_key)),
        blocking_families=_sorted_unique(blocking_families),
        blocked_reason_codes=sorted(blocked_reason_codes),
        smoke_suite_green=bool(smoke_suite_green),
        benchmark_passed=benchmark_passed,
        summary=summary,
        artifact_paths={
            str(key): str(value)
            for key, value in sorted(dict(artifact_paths or {}).items())
            if _optional_text(key) is not None and _optional_text(value) is not None
        },
        updated_at=str(updated_at or _utc_now()),
        details={
            "abstention_case_count": report_abstention_case_count,
            "evidence_metric_names": list(PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS),
            "family_evidence": family_evidence,
            "mismatch_code_counts": _sorted_mapping(report_mismatch_counts),
            "primary_target_family": primary_target_family,
            "recovery_code_counts": _sorted_mapping(report_recovery_counts),
            "shadow_and_canary_governance_only": True,
            "weak_family_count": len(weak_families),
        },
    )


__all__ = [
    "BrainPerceptionWorldModelMetricRow",
    "PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS",
    "build_perception_world_model_benchmark_comparison_report",
]

"""Governance-first adapter benchmark reports and promotion projections."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters.cards import (
    BrainAdapterBenchmarkSummary,
    BrainAdapterCard,
    BrainAdapterFamily,
    BrainAdapterPromotionState,
    BrainAdapterWeakFamilySummary,
)
from blink.brain.evals.embodied_metrics import BrainEmbodiedEvalMetricRow

_CRITICAL_MISMATCH_CODES = frozenset(
    {"unsafe", "robot_head_unarmed", "robot_head_status_unavailable"}
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


def _report_sort_key(record: "BrainAdapterBenchmarkComparisonReport") -> tuple[str, str]:
    return (record.updated_at, record.report_id)


def _decision_sort_key(record: "BrainAdapterPromotionDecision") -> tuple[str, str]:
    return (record.updated_at, record.decision_id)


def _card_sort_key(record: BrainAdapterCard) -> tuple[str, str, str]:
    return (record.adapter_family, record.backend_id, record.backend_version)


def _family_sort_key(record: "BrainAdapterComparisonFamilyRow") -> tuple[str, str]:
    return (record.scenario_family, record.family_key)


def _family_is_safety_critical(scenario_family: str) -> bool:
    family = scenario_family.lower()
    return any(token in family for token in ("safety", "unsafe", "critical"))


def _state_rank(state: str) -> int:
    return {
        BrainAdapterPromotionState.EXPERIMENTAL.value: 0,
        BrainAdapterPromotionState.SHADOW.value: 1,
        BrainAdapterPromotionState.CANARY.value: 2,
        BrainAdapterPromotionState.DEFAULT.value: 3,
        BrainAdapterPromotionState.ROLLED_BACK.value: 4,
    }.get(str(state).strip(), -1)


@dataclass(frozen=True)
class BrainAdapterComparisonFamilyRow:
    """One deterministic family-level incumbent-vs-candidate comparison row."""

    family_key: str
    scenario_family: str
    scenario_ids: list[str]
    scenario_count: int
    incumbent_backend_id: str
    candidate_backend_id: str
    task_success_delta_points: float
    safety_success_delta_points: float
    operator_burden_delta_points: float
    recovery_delta_points: float
    preview_only_delta_count: int
    mismatch_code_counts: dict[str, int] = field(default_factory=dict)
    repair_code_counts: dict[str, int] = field(default_factory=dict)
    weak_family: bool = False
    blocking_reason_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the family comparison row."""
        return {
            "family_key": self.family_key,
            "scenario_family": self.scenario_family,
            "scenario_ids": list(self.scenario_ids),
            "scenario_count": self.scenario_count,
            "incumbent_backend_id": self.incumbent_backend_id,
            "candidate_backend_id": self.candidate_backend_id,
            "task_success_delta_points": self.task_success_delta_points,
            "safety_success_delta_points": self.safety_success_delta_points,
            "operator_burden_delta_points": self.operator_burden_delta_points,
            "recovery_delta_points": self.recovery_delta_points,
            "preview_only_delta_count": self.preview_only_delta_count,
            "mismatch_code_counts": dict(self.mismatch_code_counts),
            "repair_code_counts": dict(self.repair_code_counts),
            "weak_family": self.weak_family,
            "blocking_reason_codes": list(self.blocking_reason_codes),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterComparisonFamilyRow | None":
        """Hydrate one family comparison row from JSON."""
        if not isinstance(data, dict):
            return None
        family_key = str(data.get("family_key", "")).strip()
        scenario_family = str(data.get("scenario_family", "")).strip()
        incumbent_backend_id = str(data.get("incumbent_backend_id", "")).strip()
        candidate_backend_id = str(data.get("candidate_backend_id", "")).strip()
        if (
            not family_key
            or not scenario_family
            or not incumbent_backend_id
            or not candidate_backend_id
        ):
            return None
        return cls(
            family_key=family_key,
            scenario_family=scenario_family,
            scenario_ids=_sorted_unique(data.get("scenario_ids", [])),
            scenario_count=int(data.get("scenario_count", 0)),
            incumbent_backend_id=incumbent_backend_id,
            candidate_backend_id=candidate_backend_id,
            task_success_delta_points=float(data.get("task_success_delta_points", 0.0)),
            safety_success_delta_points=float(data.get("safety_success_delta_points", 0.0)),
            operator_burden_delta_points=float(data.get("operator_burden_delta_points", 0.0)),
            recovery_delta_points=float(data.get("recovery_delta_points", 0.0)),
            preview_only_delta_count=int(data.get("preview_only_delta_count", 0)),
            mismatch_code_counts=_sorted_mapping(data.get("mismatch_code_counts")),
            repair_code_counts=_sorted_mapping(data.get("repair_code_counts")),
            weak_family=bool(data.get("weak_family", False)),
            blocking_reason_codes=_sorted_unique(data.get("blocking_reason_codes", [])),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainAdapterBenchmarkComparisonReport:
    """One deterministic comparison report for adapter governance."""

    report_id: str
    adapter_family: str
    incumbent_backend_id: str
    incumbent_backend_version: str
    candidate_backend_id: str
    candidate_backend_version: str
    scenario_count: int
    compared_family_count: int
    target_families: list[str] = field(default_factory=list)
    family_rows: list[BrainAdapterComparisonFamilyRow] = field(default_factory=list)
    weak_families: list[BrainAdapterWeakFamilySummary] = field(default_factory=list)
    blocking_families: list[str] = field(default_factory=list)
    blocked_reason_codes: list[str] = field(default_factory=list)
    smoke_suite_green: bool | None = None
    benchmark_passed: bool | None = None
    summary: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the comparison report."""
        return {
            "report_id": self.report_id,
            "adapter_family": self.adapter_family,
            "incumbent_backend_id": self.incumbent_backend_id,
            "incumbent_backend_version": self.incumbent_backend_version,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "scenario_count": self.scenario_count,
            "compared_family_count": self.compared_family_count,
            "target_families": list(self.target_families),
            "family_rows": [
                record.as_dict() for record in sorted(self.family_rows, key=_family_sort_key)
            ],
            "weak_families": [record.as_dict() for record in self.weak_families],
            "blocking_families": list(self.blocking_families),
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "smoke_suite_green": self.smoke_suite_green,
            "benchmark_passed": self.benchmark_passed,
            "summary": self.summary,
            "artifact_paths": dict(self.artifact_paths),
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None
    ) -> "BrainAdapterBenchmarkComparisonReport | None":
        """Hydrate one comparison report from JSON."""
        if not isinstance(data, dict):
            return None
        report_id = str(data.get("report_id", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        incumbent_backend_id = str(data.get("incumbent_backend_id", "")).strip()
        candidate_backend_id = str(data.get("candidate_backend_id", "")).strip()
        if (
            not report_id
            or not adapter_family
            or not incumbent_backend_id
            or not candidate_backend_id
        ):
            return None
        return cls(
            report_id=report_id,
            adapter_family=adapter_family,
            incumbent_backend_id=incumbent_backend_id,
            incumbent_backend_version=str(data.get("incumbent_backend_version", "")).strip(),
            candidate_backend_id=candidate_backend_id,
            candidate_backend_version=str(data.get("candidate_backend_version", "")).strip(),
            scenario_count=int(data.get("scenario_count", 0)),
            compared_family_count=int(data.get("compared_family_count", 0)),
            target_families=_sorted_unique(data.get("target_families", [])),
            family_rows=[
                record
                for item in data.get("family_rows", [])
                if (record := BrainAdapterComparisonFamilyRow.from_dict(item)) is not None
            ],
            weak_families=[
                record
                for item in data.get("weak_families", [])
                if (record := BrainAdapterWeakFamilySummary.from_dict(item)) is not None
            ],
            blocking_families=_sorted_unique(data.get("blocking_families", [])),
            blocked_reason_codes=_sorted_unique(data.get("blocked_reason_codes", [])),
            smoke_suite_green=(
                bool(data["smoke_suite_green"])
                if isinstance(data.get("smoke_suite_green"), bool)
                else None
            ),
            benchmark_passed=(
                bool(data["benchmark_passed"])
                if isinstance(data.get("benchmark_passed"), bool)
                else None
            ),
            summary=str(data.get("summary", "")).strip(),
            artifact_paths={
                str(key): str(value)
                for key, value in sorted(dict(data.get("artifact_paths", {})).items())
                if _optional_text(key) is not None and _optional_text(value) is not None
            },
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )

    def render_markdown(self) -> str:
        """Render a bounded operator-readable markdown report."""
        lines = [
            "# Adapter Benchmark Comparison Report",
            "",
            f"- adapter_family: {self.adapter_family}",
            f"- incumbent: {self.incumbent_backend_id}@{self.incumbent_backend_version}",
            f"- candidate: {self.candidate_backend_id}@{self.candidate_backend_version}",
            f"- scenario_count: {self.scenario_count}",
            f"- compared_family_count: {self.compared_family_count}",
            f"- target_families: {', '.join(self.target_families) or 'none'}",
            f"- smoke_suite_green: {self.smoke_suite_green}",
            f"- benchmark_passed: {self.benchmark_passed}",
            f"- blocked_reason_codes: {', '.join(self.blocked_reason_codes) or 'none'}",
            "",
            "| family | scenarios | task Δpts | safety Δpts | review Δpts | recovery Δpts | weak | blocked |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in sorted(self.family_rows, key=_family_sort_key):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.scenario_family,
                        str(row.scenario_count),
                        f"{row.task_success_delta_points:.1f}",
                        f"{row.safety_success_delta_points:.1f}",
                        f"{row.operator_burden_delta_points:.1f}",
                        f"{row.recovery_delta_points:.1f}",
                        "yes" if row.weak_family else "no",
                        ",".join(row.blocking_reason_codes) or "none",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class BrainAdapterPromotionDecision:
    """One explicit, replay-safe adapter promotion or rollback decision."""

    decision_id: str
    adapter_family: str
    backend_id: str
    backend_version: str
    decision_outcome: str
    from_state: str
    to_state: str
    report_id: str | None = None
    approved_target_families: list[str] = field(default_factory=list)
    blocked_reason_codes: list[str] = field(default_factory=list)
    weak_families: list[str] = field(default_factory=list)
    smoke_suite_green: bool | None = None
    benchmark_passed: bool | None = None
    decided_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the promotion decision."""
        return {
            "decision_id": self.decision_id,
            "adapter_family": self.adapter_family,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "decision_outcome": self.decision_outcome,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "report_id": self.report_id,
            "approved_target_families": list(self.approved_target_families),
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "weak_families": list(self.weak_families),
            "smoke_suite_green": self.smoke_suite_green,
            "benchmark_passed": self.benchmark_passed,
            "decided_at": self.decided_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterPromotionDecision | None":
        """Hydrate one decision from JSON."""
        if not isinstance(data, dict):
            return None
        decision_id = str(data.get("decision_id", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        backend_id = str(data.get("backend_id", "")).strip()
        backend_version = str(data.get("backend_version", "")).strip()
        decision_outcome = str(data.get("decision_outcome", "")).strip()
        from_state = str(data.get("from_state", "")).strip()
        to_state = str(data.get("to_state", "")).strip()
        if not all(
            (
                decision_id,
                adapter_family,
                backend_id,
                backend_version,
                decision_outcome,
                from_state,
                to_state,
            )
        ):
            return None
        return cls(
            decision_id=decision_id,
            adapter_family=adapter_family,
            backend_id=backend_id,
            backend_version=backend_version,
            decision_outcome=decision_outcome,
            from_state=from_state,
            to_state=to_state,
            report_id=_optional_text(data.get("report_id")),
            approved_target_families=_sorted_unique(data.get("approved_target_families", [])),
            blocked_reason_codes=_sorted_unique(data.get("blocked_reason_codes", [])),
            weak_families=_sorted_unique(data.get("weak_families", [])),
            smoke_suite_green=(
                bool(data["smoke_suite_green"])
                if isinstance(data.get("smoke_suite_green"), bool)
                else None
            ),
            benchmark_passed=(
                bool(data["benchmark_passed"])
                if isinstance(data.get("benchmark_passed"), bool)
                else None
            ),
            decided_at=str(data.get("decided_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("decided_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainAdapterGovernanceProjection:
    """Replay-safe bounded adapter-governance projection."""

    scope_key: str
    adapter_cards: list[BrainAdapterCard] = field(default_factory=list)
    recent_reports: list[BrainAdapterBenchmarkComparisonReport] = field(default_factory=list)
    recent_decisions: list[BrainAdapterPromotionDecision] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = ""

    def sync_lists(self):
        """Refresh deduped lists and counters."""
        cards_by_key: dict[tuple[str, str, str], BrainAdapterCard] = {}
        for record in sorted(self.adapter_cards, key=_card_sort_key):
            cards_by_key[(record.adapter_family, record.backend_id, record.backend_version)] = (
                record
            )
        reports_by_id: dict[str, BrainAdapterBenchmarkComparisonReport] = {}
        for record in sorted(self.recent_reports, key=_report_sort_key):
            reports_by_id[record.report_id] = record
        decisions_by_id: dict[str, BrainAdapterPromotionDecision] = {}
        for record in sorted(self.recent_decisions, key=_decision_sort_key):
            decisions_by_id[record.decision_id] = record
        self.adapter_cards = sorted(cards_by_key.values(), key=_card_sort_key)
        self.recent_reports = sorted(reports_by_id.values(), key=_report_sort_key, reverse=True)[
            :24
        ]
        self.recent_decisions = sorted(
            decisions_by_id.values(), key=_decision_sort_key, reverse=True
        )[:24]
        state_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for card in self.adapter_cards:
            state_counts[card.promotion_state] = state_counts.get(card.promotion_state, 0) + 1
            family_counts[card.adapter_family] = family_counts.get(card.adapter_family, 0) + 1
        self.state_counts = dict(sorted(state_counts.items()))
        self.family_counts = dict(sorted(family_counts.items()))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the adapter-governance projection."""
        self.sync_lists()
        return {
            "scope_key": self.scope_key,
            "adapter_cards": [record.as_dict() for record in self.adapter_cards],
            "recent_reports": [record.as_dict() for record in self.recent_reports],
            "recent_decisions": [record.as_dict() for record in self.recent_decisions],
            "state_counts": dict(self.state_counts),
            "family_counts": dict(self.family_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterGovernanceProjection":
        """Hydrate the adapter-governance projection from JSON."""
        payload = dict(data or {})
        projection = cls(
            scope_key=str(payload.get("scope_key", "")).strip(),
            adapter_cards=[
                record
                for item in payload.get("adapter_cards", [])
                if (record := BrainAdapterCard.from_dict(item)) is not None
            ],
            recent_reports=[
                record
                for item in payload.get("recent_reports", [])
                if (record := BrainAdapterBenchmarkComparisonReport.from_dict(item)) is not None
            ],
            recent_decisions=[
                record
                for item in payload.get("recent_decisions", [])
                if (record := BrainAdapterPromotionDecision.from_dict(item)) is not None
            ],
            state_counts=_sorted_mapping(payload.get("state_counts")),
            family_counts=_sorted_mapping(payload.get("family_counts")),
            updated_at=str(payload.get("updated_at") or ""),
        )
        projection.sync_lists()
        return projection


def append_adapter_card(
    projection: BrainAdapterGovernanceProjection,
    card: BrainAdapterCard,
) -> None:
    """Append or replace one adapter card by stable family/backend key."""
    projection.adapter_cards = [
        record
        for record in projection.adapter_cards
        if (
            record.adapter_family,
            record.backend_id,
            record.backend_version,
        )
        != (card.adapter_family, card.backend_id, card.backend_version)
    ]
    projection.adapter_cards.append(card)
    projection.updated_at = max(projection.updated_at, card.updated_at)
    projection.sync_lists()


def append_adapter_benchmark_report(
    projection: BrainAdapterGovernanceProjection,
    report: BrainAdapterBenchmarkComparisonReport,
) -> None:
    """Append or replace one benchmark comparison report."""
    projection.recent_reports = [
        record for record in projection.recent_reports if record.report_id != report.report_id
    ]
    projection.recent_reports.append(report)
    projection.updated_at = max(projection.updated_at, report.updated_at)
    projection.sync_lists()


def append_adapter_promotion_decision(
    projection: BrainAdapterGovernanceProjection,
    decision: BrainAdapterPromotionDecision,
) -> None:
    """Append or replace one promotion decision."""
    projection.recent_decisions = [
        record
        for record in projection.recent_decisions
        if record.decision_id != decision.decision_id
    ]
    projection.recent_decisions.append(decision)
    projection.updated_at = max(projection.updated_at, decision.updated_at)
    projection.sync_lists()


def build_benchmark_summary_from_report(
    report: BrainAdapterBenchmarkComparisonReport,
) -> BrainAdapterBenchmarkSummary:
    """Build the compact summary stored on adapter cards from one report."""
    compact_evidence_keys = (
        "abstention_case_count",
        "evidence_metric_names",
        "family_evidence",
        "mismatch_code_counts",
        "recovery_code_counts",
        "weak_family_count",
    )
    details = {
        "incumbent_backend_id": report.incumbent_backend_id,
        "candidate_backend_id": report.candidate_backend_id,
    }
    for key in compact_evidence_keys:
        if key in report.details:
            details[key] = report.details[key]
    return BrainAdapterBenchmarkSummary(
        report_id=report.report_id,
        adapter_family=report.adapter_family,
        scenario_count=report.scenario_count,
        compared_family_count=report.compared_family_count,
        benchmark_passed=report.benchmark_passed,
        smoke_suite_green=report.smoke_suite_green,
        target_families=list(report.target_families),
        blocked_reason_codes=list(report.blocked_reason_codes),
        weak_families=list(report.weak_families),
        updated_at=report.updated_at,
        details=details,
    )


def with_card_benchmark_summary(
    card: BrainAdapterCard,
    report: BrainAdapterBenchmarkComparisonReport,
) -> BrainAdapterCard:
    """Return one adapter card updated with the latest report summary."""
    return replace(
        card,
        latest_benchmark_summary=build_benchmark_summary_from_report(report),
        updated_at=max(card.updated_at, report.updated_at),
    )


def apply_promotion_decision_to_card(
    card: BrainAdapterCard,
    decision: BrainAdapterPromotionDecision,
) -> BrainAdapterCard:
    """Apply one explicit promotion or rollback decision to a card snapshot."""
    merged_details = dict(card.details)
    merged_details["last_decision_id"] = decision.decision_id
    if decision.decision_outcome == "rollback":
        merged_details["rollback_reason_codes"] = list(decision.blocked_reason_codes)
    elif decision.decision_outcome == "hold":
        merged_details["blocked_reason_codes"] = list(decision.blocked_reason_codes)
    return replace(
        card,
        promotion_state=decision.to_state,
        approved_target_families=list(decision.approved_target_families),
        updated_at=max(card.updated_at, decision.updated_at),
        details=merged_details,
    )


def _rate(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (float(successes) / float(total)) * 100.0


def build_embodied_policy_benchmark_comparison_report(
    metric_rows: tuple[BrainEmbodiedEvalMetricRow, ...] | list[BrainEmbodiedEvalMetricRow],
    *,
    incumbent_backend_id: str,
    candidate_backend_id: str,
    incumbent_backend_version: str = "v1",
    candidate_backend_version: str = "v1",
    target_families: Iterable[str | None] = (),
    smoke_suite_green: bool = True,
    updated_at: str | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> BrainAdapterBenchmarkComparisonReport:
    """Build a deterministic embodied-policy candidate-vs-incumbent comparison report."""
    rows = sorted(
        list(metric_rows),
        key=lambda row: (
            row.scenario_family,
            row.scenario_id,
            row.matrix_index,
            row.profile_id,
            row.run_id,
        ),
    )
    target_family_list = _sorted_unique(target_families)
    pairs_by_family: dict[
        str, list[tuple[BrainEmbodiedEvalMetricRow, BrainEmbodiedEvalMetricRow]]
    ] = {}
    scenario_ids_by_family: dict[str, set[str]] = {}
    all_target_families = set(target_family_list)
    grouped_by_scenario: dict[str, list[BrainEmbodiedEvalMetricRow]] = {}
    for row in rows:
        if row.embodied_policy_backend_id not in {incumbent_backend_id, candidate_backend_id}:
            continue
        grouped_by_scenario.setdefault(row.scenario_id, []).append(row)
        all_target_families.add(row.scenario_family)
    for scenario_id, scenario_rows in sorted(grouped_by_scenario.items()):
        incumbent = next(
            (
                row
                for row in sorted(
                    scenario_rows,
                    key=lambda item: (item.matrix_index, item.profile_id, item.run_id),
                )
                if row.embodied_policy_backend_id == incumbent_backend_id
            ),
            None,
        )
        candidate = next(
            (
                row
                for row in sorted(
                    scenario_rows,
                    key=lambda item: (item.matrix_index, item.profile_id, item.run_id),
                )
                if row.embodied_policy_backend_id == candidate_backend_id
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
    shared_scenarios = sum(len(values) for values in pairs_by_family.values())
    if not pairs_by_family:
        blocked_reason_codes.add("missing_shared_family_evidence")
    for scenario_family, pairs in sorted(pairs_by_family.items()):
        incumbent_success = sum(int(incumbent.task_success) for incumbent, _ in pairs)
        candidate_success = sum(int(candidate.task_success) for _, candidate in pairs)
        incumbent_safety = sum(int(incumbent.safety_success) for incumbent, _ in pairs)
        candidate_safety = sum(int(candidate.safety_success) for _, candidate in pairs)
        incumbent_review = sum(incumbent.review_floor_count for incumbent, _ in pairs)
        candidate_review = sum(candidate.review_floor_count for _, candidate in pairs)
        incumbent_interventions = sum(
            incumbent.operator_intervention_count for incumbent, _ in pairs
        )
        candidate_interventions = sum(
            candidate.operator_intervention_count for _, candidate in pairs
        )
        incumbent_operator_burden = incumbent_review + incumbent_interventions
        candidate_operator_burden = candidate_review + candidate_interventions
        incumbent_recovery = sum(incumbent.recovery_count for incumbent, _ in pairs)
        candidate_recovery = sum(candidate.recovery_count for _, candidate in pairs)
        preview_only_delta = sum(
            int(candidate.preview_only) - int(incumbent.preview_only)
            for incumbent, candidate in pairs
        )
        mismatch_counts: dict[str, int] = {}
        repair_counts: dict[str, int] = {}
        critical_signature = False
        for _, candidate in pairs:
            for code in candidate.mismatch_codes:
                mismatch_counts[code] = mismatch_counts.get(code, 0) + 1
                if code in _CRITICAL_MISMATCH_CODES:
                    critical_signature = True
            for code in candidate.repair_codes:
                repair_counts[code] = repair_counts.get(code, 0) + 1
        task_delta = _rate(candidate_success, len(pairs)) - _rate(incumbent_success, len(pairs))
        safety_delta = _rate(candidate_safety, len(pairs)) - _rate(incumbent_safety, len(pairs))
        operator_burden_delta = _rate(candidate_operator_burden, len(pairs)) - _rate(
            incumbent_operator_burden, len(pairs)
        )
        recovery_delta = _rate(candidate_recovery, len(pairs)) - _rate(
            incumbent_recovery, len(pairs)
        )
        family_blockers: list[str] = []
        if _family_is_safety_critical(scenario_family) and safety_delta < 0:
            family_blockers.append("safety_critical_regression")
        if critical_signature:
            family_blockers.append("new_critical_failure_signature")
        weak_family = bool(
            task_delta < 0
            or safety_delta < 0
            or operator_burden_delta > 0
            or recovery_delta > 0
            or preview_only_delta > 0
            or family_blockers
        )
        if family_blockers:
            blocking_families.append(scenario_family)
            blocked_reason_codes.update(family_blockers)
        family_rows.append(
            BrainAdapterComparisonFamilyRow(
                family_key=_stable_id(
                    "adapter_family_row",
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
                preview_only_delta_count=preview_only_delta,
                mismatch_code_counts=_sorted_mapping(mismatch_counts),
                repair_code_counts=_sorted_mapping(repair_counts),
                weak_family=weak_family,
                blocking_reason_codes=_sorted_unique(family_blockers),
                details={
                    "critical_signature_detected": critical_signature,
                    "incumbent_operator_burden_total": incumbent_operator_burden,
                    "candidate_operator_burden_total": candidate_operator_burden,
                },
            )
        )
        if weak_family:
            weak_families.append(
                BrainAdapterWeakFamilySummary(
                    family_key=_stable_id(
                        "adapter_weak_family",
                        scenario_family,
                        incumbent_backend_id,
                        candidate_backend_id,
                    ),
                    scenario_family=scenario_family,
                    issue_count=len(family_blockers) or 1,
                    blocking=bool(family_blockers),
                    reason_codes=_sorted_unique(
                        [
                            *family_blockers,
                            *(["task_success_regressed"] if task_delta < 0 else []),
                            *(["operator_burden_regressed"] if operator_burden_delta > 0 else []),
                            *(["recovery_burden_regressed"] if recovery_delta > 0 else []),
                            *(["preview_only_regressed"] if preview_only_delta > 0 else []),
                        ]
                    ),
                    task_success_delta_points=task_delta,
                    safety_success_delta_points=safety_delta,
                    operator_burden_delta_points=operator_burden_delta,
                    recovery_delta_points=recovery_delta,
                    details={
                        "scenario_ids": sorted(scenario_ids_by_family.get(scenario_family, set()))
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
        primary_improved = (
            primary_target_row.task_success_delta_points >= 3.0
            or primary_target_row.operator_burden_delta_points <= -10.0
        )
        if not primary_improved:
            blocked_reason_codes.add("no_primary_family_improvement")
    benchmark_passed = not blocked_reason_codes
    summary = (
        f"Candidate {candidate_backend_id}@{candidate_backend_version} "
        f"{'passed' if benchmark_passed else 'did not pass'} against "
        f"{incumbent_backend_id}@{incumbent_backend_version} "
        f"across {len(family_rows)} compared families."
    )
    return BrainAdapterBenchmarkComparisonReport(
        report_id=_stable_id(
            "adapter_benchmark_report",
            BrainAdapterFamily.EMBODIED_POLICY.value,
            incumbent_backend_id,
            incumbent_backend_version,
            candidate_backend_id,
            candidate_backend_version,
            *sorted(scenario_id for ids in scenario_ids_by_family.values() for scenario_id in ids),
        ),
        adapter_family=BrainAdapterFamily.EMBODIED_POLICY.value,
        incumbent_backend_id=incumbent_backend_id,
        incumbent_backend_version=incumbent_backend_version,
        candidate_backend_id=candidate_backend_id,
        candidate_backend_version=candidate_backend_version,
        scenario_count=shared_scenarios,
        compared_family_count=len(family_rows),
        target_families=target_family_list,
        family_rows=family_rows,
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
            "primary_target_family": primary_target_family,
            "shadow_and_canary_governance_only": True,
        },
    )


def build_adapter_promotion_decision(
    *,
    card: BrainAdapterCard,
    outcome: str,
    report: BrainAdapterBenchmarkComparisonReport | None = None,
    approved_target_families: Iterable[str | None] = (),
    blocked_reason_codes: Iterable[str | None] = (),
    updated_at: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainAdapterPromotionDecision:
    """Build one explicit adapter promotion or rollback decision."""
    resolved_outcome = str(outcome).strip()
    if resolved_outcome not in {"hold", "promote", "rollback"}:
        raise ValueError(f"Unsupported adapter promotion outcome: {outcome}")
    current_state = str(card.promotion_state).strip()
    if resolved_outcome == "rollback":
        next_state = BrainAdapterPromotionState.ROLLED_BACK.value
    elif resolved_outcome == "promote":
        next_state = {
            BrainAdapterPromotionState.EXPERIMENTAL.value: BrainAdapterPromotionState.SHADOW.value,
            BrainAdapterPromotionState.SHADOW.value: BrainAdapterPromotionState.CANARY.value,
            BrainAdapterPromotionState.CANARY.value: BrainAdapterPromotionState.DEFAULT.value,
            BrainAdapterPromotionState.DEFAULT.value: BrainAdapterPromotionState.DEFAULT.value,
            BrainAdapterPromotionState.ROLLED_BACK.value: BrainAdapterPromotionState.ROLLED_BACK.value,
        }.get(current_state, current_state)
    else:
        next_state = current_state
    resolved_updated_at = str(updated_at or _utc_now())
    approved_families = _sorted_unique(approved_target_families) or (
        list(report.target_families) if report is not None else []
    )
    blocked_codes = _sorted_unique(blocked_reason_codes) or (
        list(report.blocked_reason_codes) if report is not None else []
    )
    weak_families = sorted(
        record.scenario_family for record in (report.weak_families if report is not None else [])
    )
    return BrainAdapterPromotionDecision(
        decision_id=_stable_id(
            "adapter_promotion_decision",
            card.adapter_family,
            card.backend_id,
            card.backend_version,
            resolved_outcome,
            current_state,
            next_state,
            report.report_id if report is not None else "",
            resolved_updated_at,
        ),
        adapter_family=card.adapter_family,
        backend_id=card.backend_id,
        backend_version=card.backend_version,
        decision_outcome=resolved_outcome,
        from_state=current_state,
        to_state=next_state,
        report_id=report.report_id if report is not None else None,
        approved_target_families=approved_families,
        blocked_reason_codes=blocked_codes,
        weak_families=weak_families,
        smoke_suite_green=report.smoke_suite_green if report is not None else None,
        benchmark_passed=report.benchmark_passed if report is not None else None,
        decided_at=resolved_updated_at,
        updated_at=resolved_updated_at,
        details=dict(details or {}),
    )


def build_adapter_governance_inspection(
    *,
    adapter_governance: BrainAdapterGovernanceProjection | dict[str, Any] | None,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build bounded operator-facing inspection for adapter governance."""
    projection = (
        adapter_governance
        if isinstance(adapter_governance, BrainAdapterGovernanceProjection)
        else BrainAdapterGovernanceProjection.from_dict(adapter_governance)
    )
    projection.sync_lists()
    current_default_cards = [
        {
            "adapter_family": record.adapter_family,
            "backend_id": record.backend_id,
            "backend_version": record.backend_version,
            "promotion_state": record.promotion_state,
        }
        for record in projection.adapter_cards
        if record.promotion_state == BrainAdapterPromotionState.DEFAULT.value
    ]
    pending_or_blocked = [
        record
        for record in projection.recent_decisions
        if record.decision_outcome == "hold"
        or record.to_state != BrainAdapterPromotionState.DEFAULT.value
    ]
    rollback_reason_counts: dict[str, int] = {}
    for decision in projection.recent_decisions:
        if decision.to_state != BrainAdapterPromotionState.ROLLED_BACK.value:
            continue
        for code in decision.blocked_reason_codes:
            rollback_reason_counts[code] = rollback_reason_counts.get(code, 0) + 1
    return {
        "state_counts": dict(projection.state_counts),
        "family_counts": dict(projection.family_counts),
        "current_default_cards": current_default_cards,
        "recent_cards": [record.as_dict() for record in projection.adapter_cards[:recent_limit]],
        "recent_reports": [record.as_dict() for record in projection.recent_reports[:recent_limit]],
        "recent_promotion_decisions": [
            record.as_dict() for record in projection.recent_decisions[:recent_limit]
        ],
        "pending_or_blocked_decisions": [
            record.as_dict() for record in pending_or_blocked[:recent_limit]
        ],
        "rollback_reason_counts": dict(sorted(rollback_reason_counts.items())),
    }


def write_adapter_benchmark_report(
    *,
    report: BrainAdapterBenchmarkComparisonReport,
    output_dir: Path,
) -> dict[str, str]:
    """Write one file-first adapter benchmark report pair."""
    resolved_output_dir = output_dir / report.adapter_family
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / f"{report.report_id}.json"
    markdown_path = resolved_output_dir / f"{report.report_id}.md"
    json_path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(report.render_markdown(), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


__all__ = [
    "BrainAdapterBenchmarkComparisonReport",
    "BrainAdapterComparisonFamilyRow",
    "BrainAdapterGovernanceProjection",
    "BrainAdapterPromotionDecision",
    "append_adapter_benchmark_report",
    "append_adapter_card",
    "append_adapter_promotion_decision",
    "apply_promotion_decision_to_card",
    "build_adapter_governance_inspection",
    "build_adapter_promotion_decision",
    "build_benchmark_summary_from_report",
    "build_embodied_policy_benchmark_comparison_report",
    "with_card_benchmark_summary",
    "write_adapter_benchmark_report",
]

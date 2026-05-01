"""Derived sim-to-real readiness summaries for adapter governance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters.cards import BrainAdapterCard, BrainAdapterPromotionState
from blink.brain.evals.adapter_promotion import BrainAdapterGovernanceProjection


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _compact_evidence_details(report: dict[str, Any]) -> dict[str, Any]:
    details = dict(report.get("details", {})) if isinstance(report.get("details"), dict) else {}
    compact_keys = (
        "abstention_case_count",
        "evidence_metric_names",
        "family_evidence",
        "mismatch_code_counts",
        "recovery_code_counts",
        "weak_family_count",
    )
    return {key: details[key] for key in compact_keys if key in details}


@dataclass(frozen=True)
class BrainSimToRealReadinessReport:
    """One bounded, governance-only sim-to-real readiness summary."""

    report_id: str
    adapter_family: str
    backend_id: str
    backend_version: str
    promotion_state: str
    benchmark_passed: bool | None = None
    smoke_suite_green: bool | None = None
    shadow_ready: bool = False
    canary_ready: bool = False
    default_ready: bool = False
    rollback_required: bool = False
    governance_only: bool = True
    weak_families: list[str] = field(default_factory=list)
    blocked_reason_codes: list[str] = field(default_factory=list)
    parity_summary: str = ""
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the readiness report."""
        return {
            "report_id": self.report_id,
            "adapter_family": self.adapter_family,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "promotion_state": self.promotion_state,
            "benchmark_passed": self.benchmark_passed,
            "smoke_suite_green": self.smoke_suite_green,
            "shadow_ready": self.shadow_ready,
            "canary_ready": self.canary_ready,
            "default_ready": self.default_ready,
            "rollback_required": self.rollback_required,
            "governance_only": self.governance_only,
            "weak_families": list(self.weak_families),
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "parity_summary": self.parity_summary,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSimToRealReadinessReport | None":
        """Hydrate one readiness report from JSON."""
        if not isinstance(data, dict):
            return None
        report_id = str(data.get("report_id", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        backend_id = str(data.get("backend_id", "")).strip()
        backend_version = str(data.get("backend_version", "")).strip()
        promotion_state = str(data.get("promotion_state", "")).strip()
        if not all((report_id, adapter_family, backend_id, backend_version, promotion_state)):
            return None
        return cls(
            report_id=report_id,
            adapter_family=adapter_family,
            backend_id=backend_id,
            backend_version=backend_version,
            promotion_state=promotion_state,
            benchmark_passed=(
                bool(data["benchmark_passed"])
                if isinstance(data.get("benchmark_passed"), bool)
                else None
            ),
            smoke_suite_green=(
                bool(data["smoke_suite_green"])
                if isinstance(data.get("smoke_suite_green"), bool)
                else None
            ),
            shadow_ready=bool(data.get("shadow_ready", False)),
            canary_ready=bool(data.get("canary_ready", False)),
            default_ready=bool(data.get("default_ready", False)),
            rollback_required=bool(data.get("rollback_required", False)),
            governance_only=bool(data.get("governance_only", True)),
            weak_families=sorted(
                {
                    str(value).strip()
                    for value in data.get("weak_families", [])
                    if str(value).strip()
                }
            ),
            blocked_reason_codes=sorted(
                {
                    str(value).strip()
                    for value in data.get("blocked_reason_codes", [])
                    if str(value).strip()
                }
            ),
            parity_summary=str(data.get("parity_summary", "")).strip(),
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


def _latest_report_for_card(
    *,
    governance: BrainAdapterGovernanceProjection,
    card: BrainAdapterCard,
) -> dict[str, Any]:
    for report in governance.recent_reports:
        if (
            report.adapter_family == card.adapter_family
            and report.candidate_backend_id == card.backend_id
            and report.candidate_backend_version == card.backend_version
        ):
            return report.as_dict()
        if (
            report.adapter_family == card.adapter_family
            and report.incumbent_backend_id == card.backend_id
            and report.incumbent_backend_version == card.backend_version
        ):
            return report.as_dict()
    summary = card.latest_benchmark_summary
    return summary.as_dict() if summary is not None else {}


def build_sim_to_real_readiness_reports(
    *,
    adapter_governance: BrainAdapterGovernanceProjection | dict[str, Any] | None,
) -> tuple[BrainSimToRealReadinessReport, ...]:
    """Build bounded sim-to-real readiness reports from adapter governance."""
    governance = (
        adapter_governance
        if isinstance(adapter_governance, BrainAdapterGovernanceProjection)
        else BrainAdapterGovernanceProjection.from_dict(adapter_governance)
    )
    governance.sync_lists()
    reports: list[BrainSimToRealReadinessReport] = []
    for card in governance.adapter_cards:
        latest_report = _latest_report_for_card(governance=governance, card=card)
        blocked_reason_codes = sorted(
            {
                *(
                    str(code).strip()
                    for code in latest_report.get("blocked_reason_codes", [])
                    if str(code).strip()
                ),
                *(
                    str(code).strip()
                    for code in card.details.get("blocked_reason_codes", [])
                    if str(code).strip()
                ),
                *(
                    str(code).strip()
                    for code in card.details.get("rollback_reason_codes", [])
                    if str(code).strip()
                ),
            }
        )
        weak_families = sorted(
            {
                *(
                    str(record.get("scenario_family", "")).strip()
                    for record in latest_report.get("weak_families", [])
                    if isinstance(record, dict)
                ),
                *(
                    str(value).strip()
                    for value in card.approved_target_families
                    if str(value).strip()
                ),
            }
        )
        benchmark_passed = latest_report.get("benchmark_passed")
        smoke_suite_green = latest_report.get("smoke_suite_green")
        promotion_state = str(card.promotion_state).strip()
        shadow_ready = bool(benchmark_passed) and promotion_state in {
            BrainAdapterPromotionState.SHADOW.value,
            BrainAdapterPromotionState.CANARY.value,
            BrainAdapterPromotionState.DEFAULT.value,
        }
        canary_ready = bool(benchmark_passed) and promotion_state in {
            BrainAdapterPromotionState.CANARY.value,
            BrainAdapterPromotionState.DEFAULT.value,
        }
        default_ready = (
            bool(benchmark_passed) and promotion_state == BrainAdapterPromotionState.DEFAULT.value
        )
        rollback_required = (
            promotion_state == BrainAdapterPromotionState.ROLLED_BACK.value
            or "safety_critical_regression" in blocked_reason_codes
            or "new_critical_failure_signature" in blocked_reason_codes
            or "smoke_suite_regressed" in blocked_reason_codes
        )
        parity_summary = (
            "governance-only; no live routing change"
            if promotion_state
            in {
                BrainAdapterPromotionState.EXPERIMENTAL.value,
                BrainAdapterPromotionState.SHADOW.value,
                BrainAdapterPromotionState.CANARY.value,
            }
            else "current baseline retained"
            if promotion_state == BrainAdapterPromotionState.DEFAULT.value
            else "candidate rolled back"
        )
        reports.append(
            BrainSimToRealReadinessReport(
                report_id=_stable_id(
                    "sim_to_real",
                    card.adapter_family,
                    card.backend_id,
                    card.backend_version,
                    promotion_state,
                ),
                adapter_family=card.adapter_family,
                backend_id=card.backend_id,
                backend_version=card.backend_version,
                promotion_state=promotion_state,
                benchmark_passed=(
                    bool(benchmark_passed) if isinstance(benchmark_passed, bool) else None
                ),
                smoke_suite_green=(
                    bool(smoke_suite_green) if isinstance(smoke_suite_green, bool) else None
                ),
                shadow_ready=shadow_ready,
                canary_ready=canary_ready,
                default_ready=default_ready,
                rollback_required=rollback_required,
                governance_only=True,
                weak_families=weak_families,
                blocked_reason_codes=blocked_reason_codes,
                parity_summary=parity_summary,
                updated_at=max(str(card.updated_at), str(latest_report.get("updated_at") or "")),
                details={
                    "approved_target_families": list(card.approved_target_families),
                    "latest_report_id": _optional_text(latest_report.get("report_id")),
                    **_compact_evidence_details(latest_report),
                },
            )
        )
    return tuple(
        sorted(
            reports,
            key=lambda record: (
                record.adapter_family,
                record.backend_id,
                record.backend_version,
            ),
        )
    )


def build_sim_to_real_digest(
    *,
    adapter_governance: BrainAdapterGovernanceProjection | dict[str, Any] | None,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build bounded operator-facing sim-to-real inspection."""
    reports = list(build_sim_to_real_readiness_reports(adapter_governance=adapter_governance))
    readiness_counts = {
        "shadow_ready": sum(int(report.shadow_ready) for report in reports),
        "canary_ready": sum(int(report.canary_ready) for report in reports),
        "default_ready": sum(int(report.default_ready) for report in reports),
        "rollback_required": sum(int(report.rollback_required) for report in reports),
    }
    promotion_state_counts: dict[str, int] = {}
    blocked_reason_counts: dict[str, int] = {}
    for report in reports:
        promotion_state_counts[report.promotion_state] = (
            promotion_state_counts.get(report.promotion_state, 0) + 1
        )
        for code in report.blocked_reason_codes:
            blocked_reason_counts[code] = blocked_reason_counts.get(code, 0) + 1
    return {
        "readiness_counts": dict(sorted(readiness_counts.items())),
        "promotion_state_counts": dict(sorted(promotion_state_counts.items())),
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
        "readiness_reports": [report.as_dict() for report in reports[:recent_limit]],
    }


__all__ = [
    "BrainSimToRealReadinessReport",
    "build_sim_to_real_digest",
    "build_sim_to_real_readiness_reports",
]

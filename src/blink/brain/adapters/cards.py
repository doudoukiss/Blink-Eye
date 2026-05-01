"""Typed adapter governance cards layered above narrow adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters import (
    LOCAL_EMBODIED_POLICY_DESCRIPTOR,
    LOCAL_PERCEPTION_DESCRIPTOR,
    LOCAL_WORLD_MODEL_DESCRIPTOR,
    BrainAdapterDescriptor,
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


def _summary_sort_key(record: "BrainAdapterWeakFamilySummary") -> tuple[str, str]:
    return (record.scenario_family, record.family_key)


def _card_sort_key(record: "BrainAdapterCard") -> tuple[str, str, str]:
    return (record.adapter_family, record.backend_id, record.backend_version)


class BrainAdapterFamily(str, Enum):
    """Supported adapter families under bounded governance."""

    PERCEPTION = "perception"
    WORLD_MODEL = "world_model"
    EMBODIED_POLICY = "embodied_policy"


class BrainAdapterPromotionState(str, Enum):
    """Explicit governance-only promotion states for one adapter backend."""

    EXPERIMENTAL = "experimental"
    SHADOW = "shadow"
    CANARY = "canary"
    DEFAULT = "default"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class BrainAdapterWeakFamilySummary:
    """One weak or blocking scenario-family summary for adapter governance."""

    family_key: str
    scenario_family: str
    issue_count: int
    blocking: bool
    reason_codes: list[str] = field(default_factory=list)
    task_success_delta_points: float = 0.0
    safety_success_delta_points: float = 0.0
    operator_burden_delta_points: float = 0.0
    recovery_delta_points: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the weak-family summary."""
        return {
            "family_key": self.family_key,
            "scenario_family": self.scenario_family,
            "issue_count": self.issue_count,
            "blocking": self.blocking,
            "reason_codes": list(self.reason_codes),
            "task_success_delta_points": self.task_success_delta_points,
            "safety_success_delta_points": self.safety_success_delta_points,
            "operator_burden_delta_points": self.operator_burden_delta_points,
            "recovery_delta_points": self.recovery_delta_points,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterWeakFamilySummary | None":
        """Hydrate one weak-family summary from JSON."""
        if not isinstance(data, dict):
            return None
        family_key = str(data.get("family_key", "")).strip()
        scenario_family = str(data.get("scenario_family", "")).strip()
        if not family_key or not scenario_family:
            return None
        return cls(
            family_key=family_key,
            scenario_family=scenario_family,
            issue_count=int(data.get("issue_count", 0)),
            blocking=bool(data.get("blocking", False)),
            reason_codes=_sorted_unique(data.get("reason_codes", [])),
            task_success_delta_points=float(data.get("task_success_delta_points", 0.0)),
            safety_success_delta_points=float(data.get("safety_success_delta_points", 0.0)),
            operator_burden_delta_points=float(data.get("operator_burden_delta_points", 0.0)),
            recovery_delta_points=float(data.get("recovery_delta_points", 0.0)),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainAdapterBenchmarkSummary:
    """Compact latest benchmark evidence summary stored on one adapter card."""

    report_id: str
    adapter_family: str
    scenario_count: int
    compared_family_count: int
    benchmark_passed: bool | None = None
    smoke_suite_green: bool | None = None
    target_families: list[str] = field(default_factory=list)
    blocked_reason_codes: list[str] = field(default_factory=list)
    weak_families: list[BrainAdapterWeakFamilySummary] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the benchmark summary."""
        return {
            "report_id": self.report_id,
            "adapter_family": self.adapter_family,
            "scenario_count": self.scenario_count,
            "compared_family_count": self.compared_family_count,
            "benchmark_passed": self.benchmark_passed,
            "smoke_suite_green": self.smoke_suite_green,
            "target_families": list(self.target_families),
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "weak_families": [
                record.as_dict() for record in sorted(self.weak_families, key=_summary_sort_key)
            ],
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterBenchmarkSummary | None":
        """Hydrate one benchmark summary from JSON."""
        if not isinstance(data, dict):
            return None
        report_id = str(data.get("report_id", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        if not report_id or not adapter_family:
            return None
        return cls(
            report_id=report_id,
            adapter_family=adapter_family,
            scenario_count=int(data.get("scenario_count", 0)),
            compared_family_count=int(data.get("compared_family_count", 0)),
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
            target_families=_sorted_unique(data.get("target_families", [])),
            blocked_reason_codes=_sorted_unique(data.get("blocked_reason_codes", [])),
            weak_families=[
                record
                for item in data.get("weak_families", [])
                if (record := BrainAdapterWeakFamilySummary.from_dict(item)) is not None
            ],
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainAdapterCard:
    """One inspectable adapter governance card."""

    card_id: str
    adapter_family: str
    backend_id: str
    backend_version: str
    capabilities: list[str] = field(default_factory=list)
    degraded_mode_id: str | None = None
    default_timeout_ms: int | None = None
    promotion_state: str = BrainAdapterPromotionState.EXPERIMENTAL.value
    supported_task_families: list[str] = field(default_factory=list)
    approved_target_families: list[str] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    latest_benchmark_summary: BrainAdapterBenchmarkSummary | None = None
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the adapter card."""
        return {
            "card_id": self.card_id,
            "adapter_family": self.adapter_family,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "capabilities": list(self.capabilities),
            "degraded_mode_id": self.degraded_mode_id,
            "default_timeout_ms": self.default_timeout_ms,
            "promotion_state": self.promotion_state,
            "supported_task_families": list(self.supported_task_families),
            "approved_target_families": list(self.approved_target_families),
            "safety_constraints": list(self.safety_constraints),
            "latest_benchmark_summary": (
                self.latest_benchmark_summary.as_dict()
                if self.latest_benchmark_summary is not None
                else None
            ),
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAdapterCard | None":
        """Hydrate one adapter card from JSON."""
        if not isinstance(data, dict):
            return None
        card_id = str(data.get("card_id", "")).strip()
        adapter_family = str(data.get("adapter_family", "")).strip()
        backend_id = str(data.get("backend_id", "")).strip()
        backend_version = str(data.get("backend_version", "")).strip()
        if not card_id or not adapter_family or not backend_id or not backend_version:
            return None
        return cls(
            card_id=card_id,
            adapter_family=adapter_family,
            backend_id=backend_id,
            backend_version=backend_version,
            capabilities=_sorted_unique(data.get("capabilities", [])),
            degraded_mode_id=_optional_text(data.get("degraded_mode_id")),
            default_timeout_ms=(
                int(data["default_timeout_ms"])
                if data.get("default_timeout_ms") is not None
                else None
            ),
            promotion_state=str(
                data.get("promotion_state", BrainAdapterPromotionState.EXPERIMENTAL.value)
            ).strip(),
            supported_task_families=_sorted_unique(data.get("supported_task_families", [])),
            approved_target_families=_sorted_unique(data.get("approved_target_families", [])),
            safety_constraints=_sorted_unique(data.get("safety_constraints", [])),
            latest_benchmark_summary=BrainAdapterBenchmarkSummary.from_dict(
                data.get("latest_benchmark_summary")
            ),
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


def build_adapter_card(
    *,
    adapter_family: str,
    descriptor: BrainAdapterDescriptor,
    promotion_state: str = BrainAdapterPromotionState.EXPERIMENTAL.value,
    supported_task_families: Iterable[str | None] = (),
    approved_target_families: Iterable[str | None] = (),
    safety_constraints: Iterable[str | None] = (),
    latest_benchmark_summary: BrainAdapterBenchmarkSummary | None = None,
    updated_at: str | None = None,
    details: dict[str, Any] | None = None,
) -> BrainAdapterCard:
    """Build one deterministic adapter card from a descriptor and governance metadata."""
    return BrainAdapterCard(
        card_id=_stable_id(
            "adapter_card",
            adapter_family,
            descriptor.backend_id,
            descriptor.backend_version,
        ),
        adapter_family=adapter_family,
        backend_id=descriptor.backend_id,
        backend_version=descriptor.backend_version,
        capabilities=_sorted_unique(descriptor.capabilities),
        degraded_mode_id=descriptor.degraded_mode_id,
        default_timeout_ms=descriptor.default_timeout_ms,
        promotion_state=promotion_state,
        supported_task_families=_sorted_unique(supported_task_families),
        approved_target_families=_sorted_unique(approved_target_families),
        safety_constraints=_sorted_unique(safety_constraints),
        latest_benchmark_summary=latest_benchmark_summary,
        updated_at=str(updated_at or _utc_now()),
        details=dict(details or {}),
    )


def build_default_adapter_cards(
    *,
    updated_at: str | None = None,
) -> tuple[BrainAdapterCard, ...]:
    """Build the deterministic baseline governance cards for the local adapters."""
    resolved_updated_at = str(updated_at or _utc_now())
    return tuple(
        sorted(
            (
                build_adapter_card(
                    adapter_family=BrainAdapterFamily.PERCEPTION.value,
                    descriptor=LOCAL_PERCEPTION_DESCRIPTOR,
                    promotion_state=BrainAdapterPromotionState.DEFAULT.value,
                    supported_task_families=("presence_detection", "scene_enrichment"),
                    safety_constraints=("deterministic_local_baseline",),
                    updated_at=resolved_updated_at,
                    details={"governance_only": False, "baseline_seeded": True},
                ),
                build_adapter_card(
                    adapter_family=BrainAdapterFamily.WORLD_MODEL.value,
                    descriptor=LOCAL_WORLD_MODEL_DESCRIPTOR,
                    promotion_state=BrainAdapterPromotionState.DEFAULT.value,
                    supported_task_families=("predictive_world_model",),
                    safety_constraints=("deterministic_local_baseline",),
                    updated_at=resolved_updated_at,
                    details={"governance_only": False, "baseline_seeded": True},
                ),
                build_adapter_card(
                    adapter_family=BrainAdapterFamily.EMBODIED_POLICY.value,
                    descriptor=LOCAL_EMBODIED_POLICY_DESCRIPTOR,
                    promotion_state=BrainAdapterPromotionState.DEFAULT.value,
                    supported_task_families=("robot_head_embodied_execution",),
                    safety_constraints=(
                        "deterministic_local_baseline",
                        "simulation_backed_dispatch_first",
                    ),
                    updated_at=resolved_updated_at,
                    details={"governance_only": False, "baseline_seeded": True},
                ),
            ),
            key=_card_sort_key,
        )
    )


__all__ = [
    "BrainAdapterBenchmarkSummary",
    "BrainAdapterCard",
    "BrainAdapterFamily",
    "BrainAdapterPromotionState",
    "BrainAdapterWeakFamilySummary",
    "build_adapter_card",
    "build_default_adapter_cards",
]

"""Bounded rollout budgets for adapter live-routing plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

_SCHEMA_VERSION = 1
_DEFAULT_ROLLBACK_TRIGGERS = (
    "safety_critical_regression",
    "new_critical_failure_signature",
    "smoke_suite_regressed",
    "operator_requested_rollback",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(_text(part) for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _clamp_fraction(value: Any) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = 0.0
    return max(0.0, min(1.0, round(fraction, 4)))


@dataclass(frozen=True)
class RolloutBudget:
    """Explicit guardrails for one adapter rollout family."""

    schema_version: int
    budget_id: str
    adapter_family: str
    max_traffic_fraction: float
    max_duration_seconds: int
    eligible_scopes: tuple[str, ...]
    allow_preview_only: bool
    allow_embodied_live: bool
    require_operator_ack: bool
    require_recovery_floor: bool
    require_benchmark_passed: bool
    require_smoke_green: bool
    minimum_scenario_count: int
    minimum_compared_family_count: int
    rollback_trigger_codes: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the budget in stable key order."""
        return {
            "schema_version": self.schema_version,
            "budget_id": self.budget_id,
            "adapter_family": self.adapter_family,
            "max_traffic_fraction": self.max_traffic_fraction,
            "max_duration_seconds": self.max_duration_seconds,
            "eligible_scopes": list(self.eligible_scopes),
            "allow_preview_only": self.allow_preview_only,
            "allow_embodied_live": self.allow_embodied_live,
            "require_operator_ack": self.require_operator_ack,
            "require_recovery_floor": self.require_recovery_floor,
            "require_benchmark_passed": self.require_benchmark_passed,
            "require_smoke_green": self.require_smoke_green,
            "minimum_scenario_count": self.minimum_scenario_count,
            "minimum_compared_family_count": self.minimum_compared_family_count,
            "rollback_trigger_codes": list(self.rollback_trigger_codes),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RolloutBudget | None":
        """Hydrate a rollout budget from JSON-like data."""
        if not isinstance(data, dict):
            return None
        adapter_family = _text(data.get("adapter_family"))
        if not adapter_family:
            return None
        max_duration = max(0, int(data.get("max_duration_seconds") or 0))
        return cls(
            schema_version=int(data.get("schema_version") or _SCHEMA_VERSION),
            budget_id=_text(data.get("budget_id"))
            or _stable_id("rollout_budget", adapter_family, max_duration),
            adapter_family=adapter_family,
            max_traffic_fraction=_clamp_fraction(data.get("max_traffic_fraction")),
            max_duration_seconds=max_duration,
            eligible_scopes=_dedupe(data.get("eligible_scopes") or ("local",)),
            allow_preview_only=bool(data.get("allow_preview_only", False)),
            allow_embodied_live=bool(data.get("allow_embodied_live", False)),
            require_operator_ack=bool(data.get("require_operator_ack", True)),
            require_recovery_floor=bool(data.get("require_recovery_floor", True)),
            require_benchmark_passed=bool(data.get("require_benchmark_passed", True)),
            require_smoke_green=bool(data.get("require_smoke_green", True)),
            minimum_scenario_count=max(0, int(data.get("minimum_scenario_count") or 0)),
            minimum_compared_family_count=max(
                0, int(data.get("minimum_compared_family_count") or 0)
            ),
            rollback_trigger_codes=_dedupe(
                data.get("rollback_trigger_codes") or _DEFAULT_ROLLBACK_TRIGGERS
            ),
            reason_codes=_dedupe(data.get("reason_codes") or ("rollout_budget:v1",)),
        )


def build_rollout_budget(
    *,
    adapter_family: str,
    budget_id: str = "",
    max_traffic_fraction: float = 0.05,
    max_duration_seconds: int = 86_400,
    eligible_scopes: Iterable[Any] = ("local",),
    allow_preview_only: bool = False,
    allow_embodied_live: bool = False,
    require_operator_ack: bool = True,
    require_recovery_floor: bool = True,
    require_benchmark_passed: bool = True,
    require_smoke_green: bool = True,
    minimum_scenario_count: int = 1,
    minimum_compared_family_count: int = 1,
    rollback_trigger_codes: Iterable[Any] = _DEFAULT_ROLLBACK_TRIGGERS,
    reason_codes: Iterable[Any] = ("rollout_budget:v1", "rollout_budget:bounded"),
) -> RolloutBudget:
    """Build a deterministic rollout budget."""
    family = _text(adapter_family)
    return RolloutBudget(
        schema_version=_SCHEMA_VERSION,
        budget_id=_text(budget_id)
        or _stable_id("rollout_budget", family, max_traffic_fraction, max_duration_seconds),
        adapter_family=family,
        max_traffic_fraction=_clamp_fraction(max_traffic_fraction),
        max_duration_seconds=max(0, int(max_duration_seconds)),
        eligible_scopes=_dedupe(eligible_scopes),
        allow_preview_only=bool(allow_preview_only),
        allow_embodied_live=bool(allow_embodied_live),
        require_operator_ack=bool(require_operator_ack),
        require_recovery_floor=bool(require_recovery_floor),
        require_benchmark_passed=bool(require_benchmark_passed),
        require_smoke_green=bool(require_smoke_green),
        minimum_scenario_count=max(0, int(minimum_scenario_count)),
        minimum_compared_family_count=max(0, int(minimum_compared_family_count)),
        rollback_trigger_codes=_dedupe(rollback_trigger_codes),
        reason_codes=_dedupe(reason_codes),
    )


__all__ = [
    "RolloutBudget",
    "build_rollout_budget",
]

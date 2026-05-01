"""Explicit adapter live-routing rollout plans and decisions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters.cards import BrainAdapterCard, BrainAdapterPromotionState
from blink.brain.adapters.rollout_budget import RolloutBudget

_SCHEMA_VERSION = 1
_TERMINAL_STATES = frozenset({"rolled_back", "expired", "rejected"})
_EMBODIED_FAMILY = "embodied_policy"
_ACTION_TARGETS = {
    "approve": "approved",
    "activate": "active_limited",
    "pause": "paused",
    "resume": "active_limited",
    "rollback": "rolled_back",
    "expire": "expired",
    "reject": "rejected",
    "promote_default_candidate": "default_candidate",
    "make_default": "default",
    "narrow_traffic": None,
}
_ALLOWED_TRANSITIONS = {
    "proposed": {"approved", "rejected", "expired", "rolled_back"},
    "approved": {"active_limited", "paused", "expired", "rolled_back"},
    "active_limited": {"paused", "default_candidate", "expired", "rolled_back"},
    "paused": {"active_limited", "expired", "rolled_back"},
    "default_candidate": {"default", "paused", "expired", "rolled_back"},
    "default": {"paused", "rolled_back"},
    "rolled_back": set(),
    "expired": set(),
    "rejected": set(),
}


class AdapterRoutingState(str, Enum):
    """Explicit live-routing states above adapter promotion governance."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE_LIMITED = "active_limited"
    PAUSED = "paused"
    DEFAULT_CANDIDATE = "default_candidate"
    DEFAULT = "default"
    ROLLED_BACK = "rolled_back"
    EXPIRED = "expired"
    REJECTED = "rejected"


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


def _parse_time(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_expired(plan: "AdapterRoutingPlan", *, now: str = "") -> bool:
    now_dt = _parse_time(now)
    expires_dt = _parse_time(plan.expires_at)
    if now_dt is None or expires_dt is None:
        return False
    return now_dt >= expires_dt


@dataclass(frozen=True)
class AdapterRoutingPlan:
    """One explicit plan that can route bounded traffic to a candidate adapter."""

    schema_version: int
    plan_id: str
    adapter_family: str
    incumbent_backend_id: str
    incumbent_backend_version: str
    candidate_backend_id: str
    candidate_backend_version: str
    routing_state: str
    promotion_state: str
    traffic_fraction: float
    scope_key: str
    eligible_scopes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    sim_to_real_report_ids: tuple[str, ...]
    approval_source: str
    created_at: str
    updated_at: str
    expires_at: str
    rollback_triggers: tuple[str, ...]
    budget_id: str
    embodied_live: bool
    operator_acknowledged: bool
    benchmark_passed: bool | None
    smoke_suite_green: bool | None
    scenario_count: int
    compared_family_count: int
    recovery_floor_passed: bool
    reason_codes: tuple[str, ...]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the routing plan in stable key order."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "adapter_family": self.adapter_family,
            "incumbent_backend_id": self.incumbent_backend_id,
            "incumbent_backend_version": self.incumbent_backend_version,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "routing_state": self.routing_state,
            "promotion_state": self.promotion_state,
            "traffic_fraction": self.traffic_fraction,
            "scope_key": self.scope_key,
            "eligible_scopes": list(self.eligible_scopes),
            "evidence_ids": list(self.evidence_ids),
            "sim_to_real_report_ids": list(self.sim_to_real_report_ids),
            "approval_source": self.approval_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "rollback_triggers": list(self.rollback_triggers),
            "budget_id": self.budget_id,
            "embodied_live": self.embodied_live,
            "operator_acknowledged": self.operator_acknowledged,
            "benchmark_passed": self.benchmark_passed,
            "smoke_suite_green": self.smoke_suite_green,
            "scenario_count": self.scenario_count,
            "compared_family_count": self.compared_family_count,
            "recovery_floor_passed": self.recovery_floor_passed,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdapterRoutingPlan | None":
        """Hydrate a routing plan from JSON-like data."""
        if not isinstance(data, dict):
            return None
        required = (
            "plan_id",
            "adapter_family",
            "incumbent_backend_id",
            "candidate_backend_id",
            "candidate_backend_version",
            "routing_state",
        )
        if any(not _text(data.get(key)) for key in required):
            return None
        return cls(
            schema_version=int(data.get("schema_version") or _SCHEMA_VERSION),
            plan_id=_text(data.get("plan_id")),
            adapter_family=_text(data.get("adapter_family")),
            incumbent_backend_id=_text(data.get("incumbent_backend_id")),
            incumbent_backend_version=_text(data.get("incumbent_backend_version")) or "v1",
            candidate_backend_id=_text(data.get("candidate_backend_id")),
            candidate_backend_version=_text(data.get("candidate_backend_version")),
            routing_state=_text(data.get("routing_state")),
            promotion_state=_text(data.get("promotion_state"))
            or BrainAdapterPromotionState.EXPERIMENTAL.value,
            traffic_fraction=_clamp_fraction(data.get("traffic_fraction")),
            scope_key=_text(data.get("scope_key")) or "local",
            eligible_scopes=_dedupe(data.get("eligible_scopes") or ("local",)),
            evidence_ids=_dedupe(data.get("evidence_ids") or ()),
            sim_to_real_report_ids=_dedupe(data.get("sim_to_real_report_ids") or ()),
            approval_source=_text(data.get("approval_source")),
            created_at=_text(data.get("created_at")),
            updated_at=_text(data.get("updated_at")),
            expires_at=_text(data.get("expires_at")),
            rollback_triggers=_dedupe(data.get("rollback_triggers") or ()),
            budget_id=_text(data.get("budget_id")),
            embodied_live=bool(data.get("embodied_live", False)),
            operator_acknowledged=bool(data.get("operator_acknowledged", False)),
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
            scenario_count=max(0, int(data.get("scenario_count") or 0)),
            compared_family_count=max(0, int(data.get("compared_family_count") or 0)),
            recovery_floor_passed=bool(data.get("recovery_floor_passed", True)),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class RolloutDecisionRecord:
    """One replay-safe live-routing rollout decision."""

    schema_version: int
    decision_id: str
    plan_id: str
    adapter_family: str
    candidate_backend_id: str
    candidate_backend_version: str
    action: str
    accepted: bool
    from_state: str
    to_state: str
    traffic_fraction: float
    budget_id: str
    regression_codes: tuple[str, ...]
    decided_at: str
    updated_at: str
    reason_codes: tuple[str, ...]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the rollout decision in stable key order."""
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "plan_id": self.plan_id,
            "adapter_family": self.adapter_family,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "action": self.action,
            "accepted": self.accepted,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "traffic_fraction": self.traffic_fraction,
            "budget_id": self.budget_id,
            "regression_codes": list(self.regression_codes),
            "decided_at": self.decided_at,
            "updated_at": self.updated_at,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RolloutDecisionRecord | None":
        """Hydrate a rollout decision from JSON-like data."""
        if not isinstance(data, dict):
            return None
        required = ("decision_id", "plan_id", "adapter_family", "action", "from_state", "to_state")
        if any(not _text(data.get(key)) for key in required):
            return None
        return cls(
            schema_version=int(data.get("schema_version") or _SCHEMA_VERSION),
            decision_id=_text(data.get("decision_id")),
            plan_id=_text(data.get("plan_id")),
            adapter_family=_text(data.get("adapter_family")),
            candidate_backend_id=_text(data.get("candidate_backend_id")),
            candidate_backend_version=_text(data.get("candidate_backend_version")),
            action=_text(data.get("action")),
            accepted=bool(data.get("accepted", False)),
            from_state=_text(data.get("from_state")),
            to_state=_text(data.get("to_state")),
            traffic_fraction=_clamp_fraction(data.get("traffic_fraction")),
            budget_id=_text(data.get("budget_id")),
            regression_codes=_dedupe(data.get("regression_codes") or ()),
            decided_at=_text(data.get("decided_at")),
            updated_at=_text(data.get("updated_at")),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
            details=dict(data.get("details") or {}),
        )


def build_adapter_routing_plan(
    *,
    card: BrainAdapterCard,
    budget: RolloutBudget,
    incumbent_backend_id: str,
    incumbent_backend_version: str = "v1",
    scope_key: str = "local",
    traffic_fraction: float = 0.0,
    evidence_ids: Iterable[Any] = (),
    sim_to_real_report_ids: Iterable[Any] = (),
    approval_source: str = "",
    created_at: str = "",
    expires_at: str = "",
    embodied_live: bool = False,
    operator_acknowledged: bool = False,
    recovery_floor_passed: bool = True,
    details: dict[str, Any] | None = None,
) -> AdapterRoutingPlan:
    """Build a proposed live-routing plan from an adapter governance card."""
    benchmark = card.latest_benchmark_summary
    evidence = _dedupe(
        (
            *(evidence_ids or ()),
            benchmark.report_id if benchmark is not None else "",
        )
    )
    return AdapterRoutingPlan(
        schema_version=_SCHEMA_VERSION,
        plan_id=_stable_id(
            "adapter_routing_plan",
            card.adapter_family,
            incumbent_backend_id,
            card.backend_id,
            card.backend_version,
            scope_key,
        ),
        adapter_family=card.adapter_family,
        incumbent_backend_id=_text(incumbent_backend_id),
        incumbent_backend_version=_text(incumbent_backend_version) or "v1",
        candidate_backend_id=card.backend_id,
        candidate_backend_version=card.backend_version,
        routing_state=AdapterRoutingState.PROPOSED.value,
        promotion_state=card.promotion_state,
        traffic_fraction=_clamp_fraction(traffic_fraction),
        scope_key=_text(scope_key) or "local",
        eligible_scopes=budget.eligible_scopes,
        evidence_ids=evidence,
        sim_to_real_report_ids=_dedupe(sim_to_real_report_ids),
        approval_source=_text(approval_source),
        created_at=_text(created_at),
        updated_at=_text(created_at),
        expires_at=_text(expires_at),
        rollback_triggers=budget.rollback_trigger_codes,
        budget_id=budget.budget_id,
        embodied_live=bool(embodied_live),
        operator_acknowledged=bool(operator_acknowledged),
        benchmark_passed=benchmark.benchmark_passed if benchmark is not None else None,
        smoke_suite_green=benchmark.smoke_suite_green if benchmark is not None else None,
        scenario_count=benchmark.scenario_count if benchmark is not None else 0,
        compared_family_count=benchmark.compared_family_count if benchmark is not None else 0,
        recovery_floor_passed=bool(recovery_floor_passed),
        reason_codes=_dedupe(
            (
                "adapter_routing_plan:v1",
                "adapter_routing_plan:proposed",
                f"promotion_state:{card.promotion_state}",
                "routing_no_live_change",
            )
        ),
        details=dict(details or {}),
    )


def _budget_rejections(
    plan: AdapterRoutingPlan,
    budget: RolloutBudget,
    *,
    traffic_fraction: float,
    operator_acknowledged: bool,
) -> tuple[str, ...]:
    rejected: list[str] = []
    if plan.adapter_family != budget.adapter_family:
        rejected.append("budget_adapter_family_mismatch")
    if plan.scope_key not in budget.eligible_scopes:
        rejected.append("budget_scope_not_eligible")
    if traffic_fraction > budget.max_traffic_fraction:
        rejected.append("budget_traffic_fraction_exceeded")
    if plan.embodied_live and not budget.allow_embodied_live:
        rejected.append("budget_embodied_live_not_allowed")
    if budget.require_operator_ack and not operator_acknowledged:
        rejected.append("budget_operator_ack_required")
    if budget.require_benchmark_passed and plan.benchmark_passed is not True:
        rejected.append("budget_benchmark_required")
    if budget.require_smoke_green and plan.smoke_suite_green is not True:
        rejected.append("budget_smoke_suite_required")
    if plan.scenario_count < budget.minimum_scenario_count:
        rejected.append("budget_scenario_count_below_minimum")
    if plan.compared_family_count < budget.minimum_compared_family_count:
        rejected.append("budget_family_count_below_minimum")
    if budget.require_recovery_floor and not plan.recovery_floor_passed:
        rejected.append("budget_recovery_floor_required")
    return tuple(rejected)


def _regression_triggers(
    plan: AdapterRoutingPlan,
    budget: RolloutBudget | None,
    regression_codes: Iterable[Any],
) -> tuple[str, ...]:
    triggers = set(budget.rollback_trigger_codes if budget is not None else plan.rollback_triggers)
    return tuple(code for code in _dedupe(regression_codes) if code in triggers)


def build_rollout_decision(
    *,
    plan: AdapterRoutingPlan,
    action: str,
    budget: RolloutBudget | None = None,
    requested_traffic_fraction: float | None = None,
    operator_acknowledged: bool | None = None,
    regression_codes: Iterable[Any] = (),
    decided_at: str = "",
    details: dict[str, Any] | None = None,
) -> RolloutDecisionRecord:
    """Build an explicit rollout transition decision without mutating the plan."""
    normalized_action = _text(action).lower().replace("-", "_")
    current_state = plan.routing_state
    current_fraction = plan.traffic_fraction
    requested_fraction = (
        current_fraction
        if requested_traffic_fraction is None
        else _clamp_fraction(requested_traffic_fraction)
    )
    ack = (
        plan.operator_acknowledged if operator_acknowledged is None else bool(operator_acknowledged)
    )
    budget_id = budget.budget_id if budget is not None else plan.budget_id
    reasons = ["rollout_decision:v1", f"rollout_action:{normalized_action}"]
    accepted = False
    target_state = current_state
    target_fraction = current_fraction
    trigger_codes = _regression_triggers(plan, budget, regression_codes)

    if trigger_codes:
        accepted = True
        target_state = AdapterRoutingState.ROLLED_BACK.value
        target_fraction = 0.0
        reasons.extend(("rollout_rollback_triggered", *trigger_codes))
    elif _is_expired(plan, now=decided_at):
        accepted = True
        target_state = AdapterRoutingState.EXPIRED.value
        target_fraction = 0.0
        reasons.append("rollout_plan_expired")
    elif current_state in _TERMINAL_STATES:
        reasons.append("rollout_state_terminal")
    elif normalized_action not in _ACTION_TARGETS:
        reasons.append("rollout_action_unsupported")
    else:
        target_state = _ACTION_TARGETS[normalized_action] or current_state
        target_fraction = requested_fraction
        if (
            target_state not in _ALLOWED_TRANSITIONS.get(current_state, set())
            and target_state != current_state
        ):
            reasons.append("rollout_transition_invalid")
        elif normalized_action in {"pause", "rollback", "expire", "reject"}:
            accepted = True
            if normalized_action != "pause":
                target_fraction = 0.0
            reasons.append(f"rollout_{normalized_action}_accepted")
        elif budget is None:
            reasons.append("rollout_budget_missing")
        else:
            budget_rejections = _budget_rejections(
                plan,
                budget,
                traffic_fraction=target_fraction,
                operator_acknowledged=ack,
            )
            if budget_rejections:
                reasons.extend(budget_rejections)
            else:
                accepted = True
                reasons.append("rollout_budget_satisfied")
                if normalized_action == "narrow_traffic" and requested_fraction > current_fraction:
                    accepted = False
                    reasons.append("rollout_narrow_requires_lower_fraction")
                elif normalized_action == "activate" and target_fraction <= 0.0:
                    accepted = False
                    reasons.append("rollout_activation_requires_positive_fraction")
                elif normalized_action == "make_default" and target_fraction < 1.0:
                    accepted = False
                    reasons.append("rollout_default_requires_full_fraction")

    if not accepted:
        target_state = current_state
        target_fraction = current_fraction
        reasons.append("rollout_decision_rejected")
    else:
        reasons.append("rollout_decision_accepted")

    decision_time = _text(decided_at)
    return RolloutDecisionRecord(
        schema_version=_SCHEMA_VERSION,
        decision_id=_stable_id(
            "rollout_decision",
            plan.plan_id,
            normalized_action,
            current_state,
            target_state,
            target_fraction,
            decision_time,
            ",".join(_dedupe(regression_codes)),
        ),
        plan_id=plan.plan_id,
        adapter_family=plan.adapter_family,
        candidate_backend_id=plan.candidate_backend_id,
        candidate_backend_version=plan.candidate_backend_version,
        action=normalized_action,
        accepted=accepted,
        from_state=current_state,
        to_state=target_state,
        traffic_fraction=target_fraction,
        budget_id=budget_id,
        regression_codes=_dedupe(regression_codes),
        decided_at=decision_time,
        updated_at=decision_time,
        reason_codes=_dedupe(reasons),
        details=dict(details or {}),
    )


def apply_rollout_decision(
    plan: AdapterRoutingPlan,
    decision: RolloutDecisionRecord,
) -> AdapterRoutingPlan:
    """Apply an accepted decision to a routing plan snapshot."""
    if decision.plan_id != plan.plan_id:
        raise ValueError("Rollout decision does not match routing plan.")
    if decision.adapter_family != plan.adapter_family:
        raise ValueError("Rollout decision adapter family mismatch.")
    if not decision.accepted:
        return plan
    details = dict(plan.details)
    details["last_rollout_decision_id"] = decision.decision_id
    if decision.regression_codes:
        details["last_regression_codes"] = list(decision.regression_codes)
    return replace(
        plan,
        routing_state=decision.to_state,
        traffic_fraction=decision.traffic_fraction,
        operator_acknowledged=(
            plan.operator_acknowledged or "rollout_budget_satisfied" in decision.reason_codes
        ),
        updated_at=max(plan.updated_at, decision.updated_at),
        reason_codes=_dedupe(
            (
                *plan.reason_codes,
                f"routing_state:{decision.to_state}",
                f"last_rollout_action:{decision.action}",
            )
        ),
        details=details,
    )


def active_routing_plan_for_family(
    plans: Iterable[AdapterRoutingPlan],
    *,
    adapter_family: str,
    scope_key: str = "",
) -> AdapterRoutingPlan | None:
    """Return the highest-precedence active plan for one family and optional scope."""
    family = _text(adapter_family)
    scope = _text(scope_key)
    candidates = [
        plan
        for plan in plans
        if plan.adapter_family == family
        and plan.routing_state
        in {
            AdapterRoutingState.ACTIVE_LIMITED.value,
            AdapterRoutingState.DEFAULT_CANDIDATE.value,
            AdapterRoutingState.DEFAULT.value,
        }
        and (not scope or plan.scope_key == scope)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda plan: (
            plan.routing_state != AdapterRoutingState.DEFAULT.value,
            -plan.traffic_fraction,
            plan.updated_at,
            plan.plan_id,
        ),
    )[0]


__all__ = [
    "AdapterRoutingPlan",
    "AdapterRoutingState",
    "RolloutDecisionRecord",
    "active_routing_plan_for_family",
    "apply_rollout_decision",
    "build_adapter_routing_plan",
    "build_rollout_decision",
]

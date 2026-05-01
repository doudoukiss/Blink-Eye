"""Replay-safe controller for bounded adapter live-routing rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from blink.brain.adapters.live_routing import (
    AdapterRoutingPlan,
    AdapterRoutingState,
    RolloutDecisionRecord,
    apply_rollout_decision,
    build_rollout_decision,
)
from blink.brain.adapters.rollout_budget import RolloutBudget

_SCHEMA_VERSION = 1
_ACTIVE_STATES = frozenset(
    {
        AdapterRoutingState.ACTIVE_LIMITED.value,
        AdapterRoutingState.DEFAULT_CANDIDATE.value,
        AdapterRoutingState.DEFAULT.value,
    }
)
_TERMINAL_STATES = frozenset(
    {
        AdapterRoutingState.ROLLED_BACK.value,
        AdapterRoutingState.EXPIRED.value,
        AdapterRoutingState.REJECTED.value,
    }
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


def _clamp_recent_limit(value: int) -> int:
    return max(0, min(32, int(value)))


def _parse_time(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _plan_is_expired(plan: AdapterRoutingPlan, *, now: str) -> bool:
    now_dt = _parse_time(now)
    expires_dt = _parse_time(plan.expires_at)
    if now_dt is None or expires_dt is None:
        return False
    return now_dt >= expires_dt


@dataclass(frozen=True)
class LiveRoutingControllerResult:
    """Outcome of one explicit live-routing controller operation."""

    schema_version: int
    accepted: bool
    applied: bool
    action: str
    plan_id: str
    from_state: str
    to_state: str
    traffic_fraction: float
    decision: RolloutDecisionRecord | None
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the controller result without raw plan details."""
        payload = {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "applied": self.applied,
            "action": self.action,
            "plan_id": self.plan_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "traffic_fraction": self.traffic_fraction,
            "reason_codes": list(self.reason_codes),
        }
        if self.decision is not None:
            payload["decision"] = self.decision.as_dict()
        return payload


@dataclass(frozen=True)
class LiveRoutingPlanStatus:
    """Public-safe status for one live-routing plan."""

    plan_id: str
    adapter_family: str
    candidate_backend_id: str
    candidate_backend_version: str
    routing_state: str
    promotion_state: str
    traffic_fraction: float
    scope_key: str
    expires_at: str
    embodied_live: bool
    budget_id: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the public-safe plan status."""
        return {
            "plan_id": self.plan_id,
            "adapter_family": self.adapter_family,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "routing_state": self.routing_state,
            "promotion_state": self.promotion_state,
            "traffic_fraction": self.traffic_fraction,
            "scope_key": self.scope_key,
            "expires_at": self.expires_at,
            "embodied_live": self.embodied_live,
            "budget_id": self.budget_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class LiveRoutingDecisionStatus:
    """Public-safe status for one recent controller decision."""

    plan_id: str
    adapter_family: str
    action: str
    accepted: bool
    from_state: str
    to_state: str
    traffic_fraction: float
    regression_count: int
    updated_at: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the public-safe decision status."""
        return {
            "plan_id": self.plan_id,
            "adapter_family": self.adapter_family,
            "action": self.action,
            "accepted": self.accepted,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "traffic_fraction": self.traffic_fraction,
            "regression_count": self.regression_count,
            "updated_at": self.updated_at,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class LiveRoutingControllerStatus:
    """Public-safe aggregate controller status for operator surfaces."""

    schema_version: int
    available: bool
    generated_at: str
    plan_count: int
    active_plan_count: int
    paused_plan_count: int
    rolled_back_plan_count: int
    expired_plan_count: int
    live_routing_active: bool
    controlled_rollout_supported: bool
    governance_only: bool
    state_counts: dict[str, int]
    family_counts: dict[str, int]
    plan_summaries: tuple[LiveRoutingPlanStatus, ...]
    recent_decisions: tuple[LiveRoutingDecisionStatus, ...]
    reason_codes: tuple[str, ...]

    @property
    def summary(self) -> str:
        """Return a compact operator-facing status summary."""
        return (
            f"{self.plan_count} rollout plans; {self.active_plan_count} active; "
            f"{self.paused_plan_count} paused; {self.rolled_back_plan_count} rolled back."
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the public-safe controller status."""
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "plan_count": self.plan_count,
            "active_plan_count": self.active_plan_count,
            "paused_plan_count": self.paused_plan_count,
            "rolled_back_plan_count": self.rolled_back_plan_count,
            "expired_plan_count": self.expired_plan_count,
            "live_routing_active": self.live_routing_active,
            "controlled_rollout_supported": self.controlled_rollout_supported,
            "governance_only": self.governance_only,
            "state_counts": dict(self.state_counts),
            "family_counts": dict(self.family_counts),
            "plan_summaries": [record.as_dict() for record in self.plan_summaries],
            "recent_decisions": [record.as_dict() for record in self.recent_decisions],
            "reason_codes": list(self.reason_codes),
        }


def _plan_status(plan: AdapterRoutingPlan) -> LiveRoutingPlanStatus:
    return LiveRoutingPlanStatus(
        plan_id=plan.plan_id,
        adapter_family=plan.adapter_family,
        candidate_backend_id=plan.candidate_backend_id,
        candidate_backend_version=plan.candidate_backend_version,
        routing_state=plan.routing_state,
        promotion_state=plan.promotion_state,
        traffic_fraction=plan.traffic_fraction,
        scope_key=plan.scope_key,
        expires_at=plan.expires_at,
        embodied_live=plan.embodied_live,
        budget_id=plan.budget_id,
        reason_codes=plan.reason_codes,
    )


def _decision_status(decision: RolloutDecisionRecord) -> LiveRoutingDecisionStatus:
    return LiveRoutingDecisionStatus(
        plan_id=decision.plan_id,
        adapter_family=decision.adapter_family,
        action=decision.action,
        accepted=decision.accepted,
        from_state=decision.from_state,
        to_state=decision.to_state,
        traffic_fraction=decision.traffic_fraction,
        regression_count=len(decision.regression_codes),
        updated_at=decision.updated_at,
        reason_codes=decision.reason_codes,
    )


class LiveRoutingController:
    """Explicit bounded controller above live-routing plans, budgets, and decisions."""

    def __init__(
        self,
        *,
        plans: Iterable[AdapterRoutingPlan | dict[str, Any]] = (),
        budgets: Iterable[RolloutBudget | dict[str, Any]] = (),
        decisions: Iterable[RolloutDecisionRecord | dict[str, Any]] = (),
    ):
        """Initialize the controller with replayable plans, budgets, and decisions.

        Args:
            plans: Existing routing plans to manage.
            budgets: Explicit rollout budgets used for gated transitions.
            decisions: Historical decision records retained for status and replay.
        """
        self._plans: dict[str, AdapterRoutingPlan] = {}
        self._budgets: dict[str, RolloutBudget] = {}
        self._decisions: list[RolloutDecisionRecord] = []
        for budget in budgets:
            self.register_budget(budget)
        for plan in plans:
            self.register_plan(plan)
        for decision in decisions:
            hydrated = (
                decision
                if isinstance(decision, RolloutDecisionRecord)
                else RolloutDecisionRecord.from_dict(decision)
            )
            if hydrated is not None:
                self._decisions.append(hydrated)

    @property
    def plans(self) -> tuple[AdapterRoutingPlan, ...]:
        """Return managed plans sorted by stable id."""
        return tuple(self._plans[key] for key in sorted(self._plans))

    @property
    def budgets(self) -> tuple[RolloutBudget, ...]:
        """Return registered budgets sorted by stable id."""
        return tuple(self._budgets[key] for key in sorted(self._budgets))

    @property
    def decisions(self) -> tuple[RolloutDecisionRecord, ...]:
        """Return recorded decisions in insertion order."""
        return tuple(self._decisions)

    def register_plan(self, plan: AdapterRoutingPlan | dict[str, Any]) -> AdapterRoutingPlan:
        """Register or replace a managed routing plan."""
        hydrated = plan if isinstance(plan, AdapterRoutingPlan) else AdapterRoutingPlan.from_dict(plan)
        if hydrated is None:
            raise ValueError("Invalid live-routing plan.")
        self._plans[hydrated.plan_id] = hydrated
        return hydrated

    def register_budget(self, budget: RolloutBudget | dict[str, Any]) -> RolloutBudget:
        """Register or replace a rollout budget."""
        hydrated = budget if isinstance(budget, RolloutBudget) else RolloutBudget.from_dict(budget)
        if hydrated is None:
            raise ValueError("Invalid rollout budget.")
        self._budgets[hydrated.budget_id] = hydrated
        return hydrated

    def plan(self, plan_id: str) -> AdapterRoutingPlan | None:
        """Return one managed plan by id."""
        return self._plans.get(_text(plan_id))

    def _budget_for_plan(self, plan: AdapterRoutingPlan) -> RolloutBudget | None:
        if plan.budget_id in self._budgets:
            return self._budgets[plan.budget_id]
        family_matches = [
            budget for budget in self._budgets.values() if budget.adapter_family == plan.adapter_family
        ]
        return family_matches[0] if len(family_matches) == 1 else None

    def _rejected_result(
        self,
        *,
        action: str,
        plan_id: str,
        reason_codes: Iterable[Any],
    ) -> LiveRoutingControllerResult:
        return LiveRoutingControllerResult(
            schema_version=_SCHEMA_VERSION,
            accepted=False,
            applied=False,
            action=_text(action),
            plan_id=_text(plan_id),
            from_state="unavailable",
            to_state="unavailable",
            traffic_fraction=0.0,
            decision=None,
            reason_codes=_dedupe(
                (
                    "live_routing_controller:v1",
                    "live_routing_controller_rejected",
                    *reason_codes,
                )
            ),
        )

    def _decide(
        self,
        *,
        plan_id: str,
        action: str,
        requested_traffic_fraction: float | None = None,
        operator_acknowledged: bool | None = None,
        regression_codes: Iterable[Any] = (),
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        normalized_plan_id = _text(plan_id)
        normalized_action = _text(action).lower().replace("-", "_")
        plan = self._plans.get(normalized_plan_id)
        if plan is None:
            return self._rejected_result(
                action=normalized_action,
                plan_id=normalized_plan_id,
                reason_codes=("live_routing_plan_not_found",),
            )
        budget = self._budget_for_plan(plan)
        decision = build_rollout_decision(
            plan=plan,
            action=normalized_action,
            budget=budget,
            requested_traffic_fraction=requested_traffic_fraction,
            operator_acknowledged=operator_acknowledged,
            regression_codes=regression_codes,
            decided_at=decided_at,
            details=details,
        )
        updated = apply_rollout_decision(plan, decision)
        if decision.accepted:
            self._plans[normalized_plan_id] = updated
        self._decisions.append(decision)
        applied = decision.accepted and updated != plan
        return LiveRoutingControllerResult(
            schema_version=_SCHEMA_VERSION,
            accepted=decision.accepted,
            applied=applied,
            action=normalized_action,
            plan_id=normalized_plan_id,
            from_state=decision.from_state,
            to_state=decision.to_state,
            traffic_fraction=decision.traffic_fraction,
            decision=decision,
            reason_codes=_dedupe(
                (
                    "live_routing_controller:v1",
                    "live_routing_controller_accepted"
                    if decision.accepted
                    else "live_routing_controller_rejected",
                    *decision.reason_codes,
                )
            ),
        )

    def evaluate_plan(
        self,
        plan_id: str,
        *,
        operator_acknowledged: bool | None = None,
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        """Evaluate and approve a proposed plan when its explicit budget allows it."""
        return self._decide(
            plan_id=plan_id,
            action="approve",
            operator_acknowledged=operator_acknowledged,
            decided_at=decided_at,
            details=details,
        )

    def activate_plan(
        self,
        plan_id: str,
        *,
        traffic_fraction: float,
        operator_acknowledged: bool | None = None,
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        """Activate an already approved plan within its rollout budget."""
        return self._decide(
            plan_id=plan_id,
            action="activate",
            requested_traffic_fraction=traffic_fraction,
            operator_acknowledged=operator_acknowledged,
            decided_at=decided_at,
            details=details,
        )

    def pause_plan(
        self,
        plan_id: str,
        *,
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        """Pause an active plan without changing model weights or hardware policy."""
        return self._decide(
            plan_id=plan_id,
            action="pause",
            decided_at=decided_at,
            details=details,
        )

    def resume_plan(
        self,
        plan_id: str,
        *,
        traffic_fraction: float,
        operator_acknowledged: bool | None = None,
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        """Resume a paused plan within its rollout budget."""
        return self._decide(
            plan_id=plan_id,
            action="resume",
            requested_traffic_fraction=traffic_fraction,
            operator_acknowledged=operator_acknowledged,
            decided_at=decided_at,
            details=details,
        )

    def rollback_plan(
        self,
        plan_id: str,
        *,
        regression_codes: Iterable[Any] = (),
        decided_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> LiveRoutingControllerResult:
        """Roll back a plan explicitly or because rollback trigger codes were observed."""
        return self._decide(
            plan_id=plan_id,
            action="rollback",
            regression_codes=regression_codes,
            decided_at=decided_at,
            details=details,
        )

    def expire_stale_plans(self, *, now: str) -> tuple[LiveRoutingControllerResult, ...]:
        """Explicitly expire non-terminal plans whose expiry timestamp has passed."""
        results: list[LiveRoutingControllerResult] = []
        for plan in self.plans:
            if plan.routing_state in _TERMINAL_STATES:
                continue
            if not _plan_is_expired(plan, now=now):
                continue
            results.append(
                self._decide(
                    plan_id=plan.plan_id,
                    action="expire",
                    decided_at=now,
                )
            )
        return tuple(results)

    def current_status(
        self,
        *,
        generated_at: str = "",
        recent_limit: int = 8,
    ) -> LiveRoutingControllerStatus:
        """Build a public-safe status snapshot from managed plans and decisions."""
        plans = self.plans
        decisions = sorted(
            self._decisions,
            key=lambda decision: (decision.updated_at, decision.decision_id),
            reverse=True,
        )[: _clamp_recent_limit(recent_limit)]
        state_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for plan in plans:
            state_counts[plan.routing_state] = state_counts.get(plan.routing_state, 0) + 1
            family_counts[plan.adapter_family] = family_counts.get(plan.adapter_family, 0) + 1
        active_count = sum(1 for plan in plans if plan.routing_state in _ACTIVE_STATES)
        paused_count = state_counts.get(AdapterRoutingState.PAUSED.value, 0)
        rolled_back_count = state_counts.get(AdapterRoutingState.ROLLED_BACK.value, 0)
        expired_count = state_counts.get(AdapterRoutingState.EXPIRED.value, 0)
        return LiveRoutingControllerStatus(
            schema_version=_SCHEMA_VERSION,
            available=True,
            generated_at=_text(generated_at),
            plan_count=len(plans),
            active_plan_count=active_count,
            paused_plan_count=paused_count,
            rolled_back_plan_count=rolled_back_count,
            expired_plan_count=expired_count,
            live_routing_active=active_count > 0,
            controlled_rollout_supported=True,
            governance_only=False,
            state_counts=dict(sorted(state_counts.items())),
            family_counts=dict(sorted(family_counts.items())),
            plan_summaries=tuple(_plan_status(plan) for plan in plans),
            recent_decisions=tuple(_decision_status(decision) for decision in decisions),
            reason_codes=_dedupe(
                (
                    "live_routing_controller:v1",
                    "live_routing_controller:available",
                    f"live_routing_plan_count:{len(plans)}",
                    f"live_routing_active_count:{active_count}",
                )
            ),
        )


__all__ = [
    "LiveRoutingController",
    "LiveRoutingControllerResult",
    "LiveRoutingControllerStatus",
    "LiveRoutingDecisionStatus",
    "LiveRoutingPlanStatus",
]

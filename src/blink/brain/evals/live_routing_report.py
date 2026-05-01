"""Deterministic live-routing rollout inspection reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from blink.brain.adapters.live_routing import AdapterRoutingPlan, RolloutDecisionRecord

_SCHEMA_VERSION = 1


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


def _plan_sort_key(plan: AdapterRoutingPlan) -> tuple[str, str, str, str]:
    return (plan.adapter_family, plan.scope_key, plan.candidate_backend_id, plan.plan_id)


def _decision_sort_key(decision: RolloutDecisionRecord) -> tuple[str, str]:
    return (decision.updated_at, decision.decision_id)


@dataclass(frozen=True)
class LiveRoutingPlanSummary:
    """One compact public-safe rollout plan summary."""

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
    rollback_trigger_count: int
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the plan summary."""
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
            "rollback_trigger_count": self.rollback_trigger_count,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class LiveRoutingDecisionSummary:
    """One compact public-safe rollout decision summary."""

    decision_id: str
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
        """Serialize the decision summary."""
        return {
            "decision_id": self.decision_id,
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
class LiveRoutingReport:
    """Bounded replay-safe live-routing rollout report."""

    schema_version: int
    generated_at: str
    plan_count: int
    active_plan_count: int
    rollback_required_count: int
    state_counts: dict[str, int]
    family_counts: dict[str, int]
    plan_summaries: tuple[LiveRoutingPlanSummary, ...]
    recent_decisions: tuple[LiveRoutingDecisionSummary, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the live-routing report."""
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "plan_count": self.plan_count,
            "active_plan_count": self.active_plan_count,
            "rollback_required_count": self.rollback_required_count,
            "state_counts": dict(self.state_counts),
            "family_counts": dict(self.family_counts),
            "plan_summaries": [record.as_dict() for record in self.plan_summaries],
            "recent_decisions": [record.as_dict() for record in self.recent_decisions],
            "reason_codes": list(self.reason_codes),
        }


def _summarize_plan(plan: AdapterRoutingPlan) -> LiveRoutingPlanSummary:
    return LiveRoutingPlanSummary(
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
        rollback_trigger_count=len(plan.rollback_triggers),
        reason_codes=plan.reason_codes,
    )


def _summarize_decision(decision: RolloutDecisionRecord) -> LiveRoutingDecisionSummary:
    return LiveRoutingDecisionSummary(
        decision_id=decision.decision_id,
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


def build_live_routing_report(
    *,
    plans: Iterable[AdapterRoutingPlan | dict[str, Any]],
    decisions: Iterable[RolloutDecisionRecord | dict[str, Any]] = (),
    recent_limit: int = 8,
    generated_at: str = "",
) -> LiveRoutingReport:
    """Build a deterministic live-routing inspection report."""
    hydrated_plans = [
        plan
        for item in plans
        if (
            plan := item
            if isinstance(item, AdapterRoutingPlan)
            else AdapterRoutingPlan.from_dict(item)
        )
        is not None
    ]
    hydrated_decisions = [
        decision
        for item in decisions
        if (
            decision := item
            if isinstance(item, RolloutDecisionRecord)
            else RolloutDecisionRecord.from_dict(item)
        )
        is not None
    ]
    sorted_plans = tuple(sorted(hydrated_plans, key=_plan_sort_key))
    sorted_decisions = tuple(
        sorted(hydrated_decisions, key=_decision_sort_key, reverse=True)[:recent_limit]
    )
    state_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    active_states = {"active_limited", "default_candidate", "default"}
    rollback_states = {"rolled_back"}
    for plan in sorted_plans:
        state_counts[plan.routing_state] = state_counts.get(plan.routing_state, 0) + 1
        family_counts[plan.adapter_family] = family_counts.get(plan.adapter_family, 0) + 1
    active_count = sum(1 for plan in sorted_plans if plan.routing_state in active_states)
    rollback_count = sum(1 for plan in sorted_plans if plan.routing_state in rollback_states)
    reason_codes = _dedupe(
        (
            "live_routing_report:v1",
            "live_routing_governance_only",
            f"live_routing_plan_count:{len(sorted_plans)}",
            f"live_routing_active_count:{active_count}",
            f"live_routing_rollback_count:{rollback_count}",
        )
    )
    return LiveRoutingReport(
        schema_version=_SCHEMA_VERSION,
        generated_at=_text(generated_at),
        plan_count=len(sorted_plans),
        active_plan_count=active_count,
        rollback_required_count=rollback_count,
        state_counts=dict(sorted(state_counts.items())),
        family_counts=dict(sorted(family_counts.items())),
        plan_summaries=tuple(_summarize_plan(plan) for plan in sorted_plans),
        recent_decisions=tuple(_summarize_decision(decision) for decision in sorted_decisions),
        reason_codes=reason_codes,
    )


def render_live_routing_report_markdown(report: LiveRoutingReport) -> str:
    """Render a compact markdown rollout report."""
    lines = [
        "# Live Routing Report",
        "",
        f"- plan_count: {report.plan_count}",
        f"- active_plan_count: {report.active_plan_count}",
        f"- rollback_required_count: {report.rollback_required_count}",
        f"- state_counts: {report.state_counts}",
        "",
        "| family | candidate | state | traffic | scope | embodied |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for plan in report.plan_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    plan.adapter_family,
                    f"{plan.candidate_backend_id}@{plan.candidate_backend_version}",
                    plan.routing_state,
                    f"{plan.traffic_fraction:.4f}",
                    plan.scope_key,
                    "yes" if plan.embodied_live else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


__all__ = [
    "LiveRoutingDecisionSummary",
    "LiveRoutingPlanSummary",
    "LiveRoutingReport",
    "build_live_routing_report",
    "render_live_routing_report_markdown",
]

"""Unified deterministic release-gate reports for rollout decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from blink.brain.adapters.live_routing import AdapterRoutingPlan, AdapterRoutingState
from blink.brain.adapters.rollout_budget import RolloutBudget
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest
from blink.brain.persona.voice_backend_registry import (
    BrainVoiceBackendCapabilityRegistry,
    resolve_voice_backend_capabilities,
)
from blink.brain.persona.voice_capabilities import BrainVoiceBackendCapabilities

RELEASE_GATE_SCHEMA_VERSION = 1
RELEASE_GATE_SUITE_ID = "release_gate/v1"
RELEASE_GATE_ARTIFACT_DIR = Path("artifacts/brain_evals/release_gate")
_PASSING_OUTCOME = "passed"
_BLOCKED_OUTCOME = "blocked"
_INSPECTION_OUTCOME = "inspection_only"
_TERMINAL_ROLLOUT_STATES = {
    AdapterRoutingState.ROLLED_BACK.value,
    AdapterRoutingState.EXPIRED.value,
    AdapterRoutingState.REJECTED.value,
}
_BANNED_REPORT_TOKENS = (
    "source_event_ids",
    "source_refs",
    "event_id",
    "raw_json",
    "brain.db",
    ".db",
    "/tmp",
    "Traceback",
    "RuntimeError",
    "prompt_text",
    "private_working_memory",
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


def _as_dict(value: Any) -> dict[str, Any]:
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        payload = serializer()
        return dict(payload) if isinstance(payload, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_time(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _report_leak_codes(payload: dict[str, Any]) -> tuple[str, ...]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return tuple(f"release_gate_payload_leak:{token}" for token in _BANNED_REPORT_TOKENS if token in encoded)


@dataclass(frozen=True)
class BrainReleaseGateEvidenceRef:
    """One compact evidence reference consumed by a release-gate report."""

    evidence_kind: str
    evidence_id: str
    passed: bool | None
    summary: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence reference."""
        return {
            "evidence_kind": self.evidence_kind,
            "evidence_id": self.evidence_id,
            "passed": self.passed,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainReleaseGateCheck:
    """One deterministic release-gate check result."""

    check_id: str
    passed: bool
    summary: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the check."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "summary": self.summary,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainReleaseGateReport:
    """Unified public-safe release gate report for one rollout decision."""

    schema_version: int
    suite_id: str
    report_id: str
    generated_at: str
    outcome: str
    rollout_plan_id: str | None = None
    adapter_family: str | None = None
    candidate_backend_id: str | None = None
    candidate_backend_version: str | None = None
    budget_id: str | None = None
    checks: tuple[BrainReleaseGateCheck, ...] = ()
    evidence_refs: tuple[BrainReleaseGateEvidenceRef, ...] = ()
    blocking_reason_codes: tuple[str, ...] = ()
    warning_reason_codes: tuple[str, ...] = ()
    artifact_links: dict[str, str] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every gate check passed."""
        return self.outcome == _PASSING_OUTCOME

    def check_results(self) -> dict[str, bool]:
        """Return a stable check pass/fail map."""
        return {check.check_id: check.passed for check in sorted(self.checks, key=lambda item: item.check_id)}

    def rollout_reference(self) -> dict[str, Any]:
        """Return the compact reference a rollout plan or decision can store."""
        return {
            "gate_report_id": self.report_id,
            "gate_outcome": self.outcome,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes[:12]),
        }

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report with stable key order."""
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "outcome": self.outcome,
            "passed": self.passed,
            "rollout_plan_id": self.rollout_plan_id,
            "adapter_family": self.adapter_family,
            "candidate_backend_id": self.candidate_backend_id,
            "candidate_backend_version": self.candidate_backend_version,
            "budget_id": self.budget_id,
            "checks": [check.as_dict() for check in sorted(self.checks, key=lambda item: item.check_id)],
            "check_results": self.check_results(),
            "evidence_refs": [
                ref.as_dict()
                for ref in sorted(self.evidence_refs, key=lambda item: (item.evidence_kind, item.evidence_id))
            ],
            "blocking_reason_codes": list(self.blocking_reason_codes),
            "warning_reason_codes": list(self.warning_reason_codes),
            "artifact_links": dict(sorted(self.artifact_links.items())),
            "rollout_reference": self.rollout_reference(),
            "reason_codes": list(self.reason_codes),
        }


def _frontier_behavior_check(report: Any | None) -> tuple[BrainReleaseGateCheck, BrainReleaseGateEvidenceRef]:
    payload = _as_dict(report)
    suite_id = _text(payload.get("suite_id")) or "frontier_behavior_workbench"
    passed = payload.get("passed") is True
    gates = dict(payload.get("gates") or {})
    failed_gates = tuple(key for key, value in sorted(gates.items()) if value is not True)
    reason_codes = _dedupe(
        (
            "release_gate_source:frontier_behavior_workbench",
            "frontier_behavior:passed" if passed else "frontier_behavior:blocked",
            *(f"frontier_behavior_gate_failed:{gate}" for gate in failed_gates),
            "frontier_behavior_missing" if not payload else "",
        )
    )
    check = BrainReleaseGateCheck(
        check_id="frontier_behavior_workbench",
        passed=passed,
        summary="Frontier behavior workbench passed." if passed else "Frontier behavior workbench blocked.",
        reason_codes=reason_codes,
    )
    ref = BrainReleaseGateEvidenceRef(
        evidence_kind="frontier_behavior_workbench",
        evidence_id=_stable_id("frontier_behavior", suite_id, passed, failed_gates),
        passed=passed,
        summary=suite_id,
        reason_codes=reason_codes,
    )
    return check, ref


def _autonomy_benchmark_check(report: Any | None) -> tuple[BrainReleaseGateCheck, BrainReleaseGateEvidenceRef]:
    payload = _as_dict(report)
    suite_id = _text(payload.get("suite_id")) or "autonomy_benchmark_program"
    passed = payload.get("passed") is True
    failures = _dedupe(payload.get("gating_failures") or ())
    reason_codes = _dedupe(
        (
            "release_gate_source:autonomy_benchmark_program",
            "autonomy_benchmark:passed" if passed else "autonomy_benchmark:blocked",
            *(f"autonomy_benchmark_failure:{failure}" for failure in failures[:10]),
            "autonomy_benchmark_missing" if not payload else "",
        )
    )
    check = BrainReleaseGateCheck(
        check_id="autonomy_benchmark_program",
        passed=passed,
        summary="Autonomy benchmark program passed." if passed else "Autonomy benchmark program blocked.",
        reason_codes=reason_codes,
    )
    ref = BrainReleaseGateEvidenceRef(
        evidence_kind="autonomy_benchmark_program",
        evidence_id=_stable_id("autonomy_benchmark", suite_id, passed, failures),
        passed=passed,
        summary=suite_id,
        reason_codes=reason_codes,
    )
    return check, ref


def _sim_digest(
    *,
    sim_to_real_digest: dict[str, Any] | None,
    adapter_governance: Any | None,
) -> dict[str, Any]:
    if isinstance(sim_to_real_digest, dict):
        return dict(sim_to_real_digest)
    if adapter_governance is not None:
        return build_sim_to_real_digest(adapter_governance=adapter_governance)
    return {"readiness_reports": [], "readiness_counts": {}, "blocked_reason_counts": {}}


def _matching_readiness_report(
    digest: dict[str, Any],
    plan: AdapterRoutingPlan | None,
) -> dict[str, Any] | None:
    reports = [dict(item) for item in _list(digest.get("readiness_reports")) if isinstance(item, dict)]
    if plan is None:
        return reports[0] if len(reports) == 1 else None
    for report in reports:
        if (
            _text(report.get("adapter_family")) == plan.adapter_family
            and _text(report.get("backend_id")) == plan.candidate_backend_id
            and _text(report.get("backend_version")) == plan.candidate_backend_version
        ):
            return report
    return None


def _requires_canary_ready(plan: AdapterRoutingPlan | None) -> bool:
    if plan is None:
        return False
    return plan.traffic_fraction > 0.0 or plan.routing_state in {
        AdapterRoutingState.ACTIVE_LIMITED.value,
        AdapterRoutingState.DEFAULT_CANDIDATE.value,
        AdapterRoutingState.DEFAULT.value,
    }


def _sim_to_real_check(
    *,
    digest: dict[str, Any],
    plan: AdapterRoutingPlan | None,
) -> tuple[BrainReleaseGateCheck, BrainReleaseGateEvidenceRef | None]:
    readiness = _matching_readiness_report(digest, plan)
    if readiness is None:
        reason_codes = (
            "release_gate_source:sim_to_real_readiness",
            "sim_to_real_readiness_missing",
        )
        return (
            BrainReleaseGateCheck(
                check_id="sim_to_real_readiness",
                passed=False,
                summary="No matching sim-to-real readiness report.",
                reason_codes=reason_codes,
            ),
            None,
        )
    blocked_codes = _dedupe(readiness.get("blocked_reason_codes") or ())
    failures: list[str] = []
    if readiness.get("rollback_required") is True:
        failures.append("sim_to_real_rollback_required")
    if readiness.get("shadow_ready") is not True:
        failures.append("sim_to_real_shadow_not_ready")
    if _requires_canary_ready(plan) and readiness.get("canary_ready") is not True:
        failures.append("sim_to_real_canary_not_ready")
    if plan is not None and plan.routing_state == AdapterRoutingState.DEFAULT.value and readiness.get("default_ready") is not True:
        failures.append("sim_to_real_default_not_ready")
    failures.extend(f"sim_to_real_blocked:{code}" for code in blocked_codes)
    passed = not failures
    reason_codes = _dedupe(
        (
            "release_gate_source:sim_to_real_readiness",
            "sim_to_real:passed" if passed else "sim_to_real:blocked",
            *failures,
        )
    )
    ref = BrainReleaseGateEvidenceRef(
        evidence_kind="sim_to_real_readiness",
        evidence_id=_text(readiness.get("report_id"))
        or _stable_id("sim_to_real", readiness.get("adapter_family"), readiness.get("backend_id")),
        passed=passed,
        summary=" | ".join(
            value
            for value in (
                _text(readiness.get("adapter_family")),
                _text(readiness.get("backend_id")),
                _text(readiness.get("promotion_state")),
            )
            if value
        )
        or "sim-to-real readiness",
        reason_codes=reason_codes,
    )
    return (
        BrainReleaseGateCheck(
            check_id="sim_to_real_readiness",
            passed=passed,
            summary="Sim-to-real readiness passed." if passed else "Sim-to-real readiness blocked.",
            reason_codes=reason_codes,
        ),
        ref,
    )


def _voice_check(
    *,
    tts_backend: str | None,
    voice_capabilities: BrainVoiceBackendCapabilities | dict[str, Any] | None,
    voice_registry: BrainVoiceBackendCapabilityRegistry | None,
) -> tuple[BrainReleaseGateCheck, BrainReleaseGateEvidenceRef]:
    resolution = resolve_voice_backend_capabilities(
        tts_backend,
        registry=voice_registry,
        capabilities_override=voice_capabilities,
    )
    capabilities = resolution.capabilities
    failures: list[str] = []
    if not capabilities.supports_chunk_boundaries:
        failures.append("voice_chunk_boundaries_unsupported")
    if not capabilities.supports_interruption_flush:
        failures.append("voice_interruption_flush_unsupported")
    if capabilities.expression_controls_hardware:
        failures.append("voice_hardware_control_forbidden")
    passed = not failures
    reason_codes = _dedupe(
        (
            "release_gate_source:voice_backend_support",
            f"voice_backend:{resolution.resolved_backend_label}",
            "voice_backend:fallback_provider_neutral" if resolution.fallback_used else "",
            "voice_backend:passed" if passed else "voice_backend:blocked",
            *failures,
            *capabilities.reason_codes,
        )
    )
    check = BrainReleaseGateCheck(
        check_id="voice_backend_support",
        passed=passed,
        summary=f"Voice backend {resolution.resolved_backend_label} capability check.",
        reason_codes=reason_codes,
    )
    ref = BrainReleaseGateEvidenceRef(
        evidence_kind="voice_backend_support",
        evidence_id=_stable_id(
            "voice_backend",
            resolution.requested_backend_label,
            resolution.resolved_backend_label,
            resolution.fallback_used,
        ),
        passed=passed,
        summary=resolution.resolved_backend_label,
        reason_codes=reason_codes,
    )
    return check, ref


def _hydrate_plan(value: AdapterRoutingPlan | dict[str, Any] | None) -> AdapterRoutingPlan | None:
    if isinstance(value, AdapterRoutingPlan):
        return value
    return AdapterRoutingPlan.from_dict(value) if isinstance(value, dict) else None


def _hydrate_budget(value: RolloutBudget | dict[str, Any] | None) -> RolloutBudget | None:
    if isinstance(value, RolloutBudget):
        return value
    return RolloutBudget.from_dict(value) if isinstance(value, dict) else None


def _budget_check(
    *,
    plan: AdapterRoutingPlan | None,
    budget: RolloutBudget | None,
    generated_at: str,
) -> tuple[BrainReleaseGateCheck, tuple[BrainReleaseGateEvidenceRef, ...]]:
    evidence_refs: list[BrainReleaseGateEvidenceRef] = []
    failures: list[str] = []
    warnings: list[str] = []
    if plan is None:
        failures.append("rollout_plan_missing")
    if budget is None:
        failures.append("rollout_budget_missing")
    if plan is not None:
        evidence_refs.append(
            BrainReleaseGateEvidenceRef(
                evidence_kind="rollout_plan",
                evidence_id=plan.plan_id,
                passed=None,
                summary=f"{plan.adapter_family} | {plan.candidate_backend_id} | {plan.routing_state}",
                reason_codes=("release_gate_source:rollout_plan",),
            )
        )
    if budget is not None:
        evidence_refs.append(
            BrainReleaseGateEvidenceRef(
                evidence_kind="rollout_budget",
                evidence_id=budget.budget_id,
                passed=None,
                summary=f"{budget.adapter_family} budget max {budget.max_traffic_fraction:.4f}",
                reason_codes=("release_gate_source:rollout_budget",),
            )
        )
    if plan is not None and budget is not None:
        if plan.budget_id and plan.budget_id != budget.budget_id:
            failures.append("rollout_budget_id_mismatch")
        if plan.adapter_family != budget.adapter_family:
            failures.append("rollout_budget_family_mismatch")
        if plan.routing_state in _TERMINAL_ROLLOUT_STATES:
            failures.append("rollout_plan_terminal")
        if budget.require_operator_ack and not plan.operator_acknowledged:
            failures.append("operator_ack_required")
        if plan.traffic_fraction > budget.max_traffic_fraction:
            failures.append("rollout_traffic_exceeds_budget")
        if plan.scope_key not in budget.eligible_scopes:
            failures.append("rollout_scope_not_eligible")
        if plan.embodied_live and not budget.allow_embodied_live:
            failures.append("embodied_live_not_allowed")
        if budget.require_benchmark_passed and plan.benchmark_passed is not True:
            failures.append("benchmark_pass_required")
        if budget.require_smoke_green and plan.smoke_suite_green is not True:
            failures.append("smoke_green_required")
        if budget.require_recovery_floor and not plan.recovery_floor_passed:
            failures.append("recovery_floor_required")
        if plan.scenario_count < budget.minimum_scenario_count:
            failures.append("minimum_scenario_count_not_met")
        if plan.compared_family_count < budget.minimum_compared_family_count:
            failures.append("minimum_compared_family_count_not_met")
        expires_at = _parse_time(plan.expires_at)
        now = _parse_time(generated_at)
        if expires_at is not None and now is not None and now >= expires_at:
            failures.append("rollout_plan_expired")
        if plan.traffic_fraction == 0.0:
            warnings.append("rollout_traffic_zero")
    passed = not failures
    reason_codes = _dedupe(
        (
            "release_gate_source:operator_ack_budget",
            "operator_budget:passed" if passed else "operator_budget:blocked",
            *failures,
            *warnings,
        )
    )
    return (
        BrainReleaseGateCheck(
            check_id="operator_ack_budget",
            passed=passed,
            summary="Operator acknowledgement and rollout budget passed."
            if passed
            else "Operator acknowledgement or rollout budget blocked.",
            reason_codes=reason_codes,
        ),
        tuple(evidence_refs),
    )


def _report_id(
    *,
    plan: AdapterRoutingPlan | None,
    budget: RolloutBudget | None,
    checks: Iterable[BrainReleaseGateCheck],
    evidence_refs: Iterable[BrainReleaseGateEvidenceRef],
) -> str:
    return _stable_id(
        "release_gate",
        plan.plan_id if plan is not None else "no-plan",
        budget.budget_id if budget is not None else "no-budget",
        tuple((check.check_id, check.passed, check.reason_codes) for check in checks),
        tuple((ref.evidence_kind, ref.evidence_id, ref.passed) for ref in evidence_refs),
    )


def _report_outcome(
    *,
    plan: AdapterRoutingPlan | None,
    checks: Iterable[BrainReleaseGateCheck],
) -> str:
    if all(check.passed for check in checks):
        return _PASSING_OUTCOME
    return _BLOCKED_OUTCOME if plan is not None else _INSPECTION_OUTCOME


def build_release_gate_report(
    *,
    frontier_behavior_report: Any | None,
    autonomy_benchmark_report: Any | None,
    sim_to_real_digest: dict[str, Any] | None = None,
    adapter_governance: Any | None = None,
    rollout_plan: AdapterRoutingPlan | dict[str, Any] | None = None,
    rollout_budget: RolloutBudget | dict[str, Any] | None = None,
    tts_backend: str | None = None,
    voice_capabilities: BrainVoiceBackendCapabilities | dict[str, Any] | None = None,
    voice_registry: BrainVoiceBackendCapabilityRegistry | None = None,
    generated_at: str = "",
    artifact_links: dict[str, str] | None = None,
) -> BrainReleaseGateReport:
    """Build one deterministic release-gate report without mutating rollout state."""
    plan = _hydrate_plan(rollout_plan)
    budget = _hydrate_budget(rollout_budget)
    digest = _sim_digest(sim_to_real_digest=sim_to_real_digest, adapter_governance=adapter_governance)
    frontier_check, frontier_ref = _frontier_behavior_check(frontier_behavior_report)
    autonomy_check, autonomy_ref = _autonomy_benchmark_check(autonomy_benchmark_report)
    sim_check, sim_ref = _sim_to_real_check(digest=digest, plan=plan)
    voice_check, voice_ref = _voice_check(
        tts_backend=tts_backend,
        voice_capabilities=voice_capabilities,
        voice_registry=voice_registry,
    )
    budget_check, budget_refs = _budget_check(
        plan=plan,
        budget=budget,
        generated_at=generated_at,
    )
    checks = (frontier_check, autonomy_check, sim_check, voice_check, budget_check)
    evidence_refs = tuple(
        ref
        for ref in (
            frontier_ref,
            autonomy_ref,
            sim_ref,
            voice_ref,
            *budget_refs,
        )
        if ref is not None
    )
    blocking_reason_codes = _dedupe(
        code for check in checks if not check.passed for code in check.reason_codes
    )
    warning_reason_codes = _dedupe(
        code
        for check in checks
        for code in check.reason_codes
        if code.endswith("_missing") or code == "rollout_traffic_zero"
    )
    outcome = _report_outcome(plan=plan, checks=checks)
    report_id = _report_id(plan=plan, budget=budget, checks=checks, evidence_refs=evidence_refs)
    reason_codes = _dedupe(
        (
            "release_gate:v1",
            f"release_gate_outcome:{outcome}",
            f"release_gate_report_id:{report_id}",
            *(f"release_gate_check:{check.check_id}:{'pass' if check.passed else 'block'}" for check in checks),
            *blocking_reason_codes[:16],
        )
    )
    report = BrainReleaseGateReport(
        schema_version=RELEASE_GATE_SCHEMA_VERSION,
        suite_id=RELEASE_GATE_SUITE_ID,
        report_id=report_id,
        generated_at=_text(generated_at),
        outcome=outcome,
        rollout_plan_id=plan.plan_id if plan is not None else None,
        adapter_family=plan.adapter_family if plan is not None else None,
        candidate_backend_id=plan.candidate_backend_id if plan is not None else None,
        candidate_backend_version=plan.candidate_backend_version if plan is not None else None,
        budget_id=budget.budget_id if budget is not None else None,
        checks=checks,
        evidence_refs=evidence_refs,
        blocking_reason_codes=blocking_reason_codes,
        warning_reason_codes=warning_reason_codes,
        artifact_links=dict(artifact_links or {}),
        reason_codes=reason_codes,
    )
    leak_codes = _report_leak_codes(report.as_dict())
    if not leak_codes:
        return report
    leak_check = BrainReleaseGateCheck(
        check_id="release_gate_payload_safety",
        passed=False,
        summary="Release-gate payload contained blocked internal tokens.",
        reason_codes=leak_codes,
    )
    checks = (*checks, leak_check)
    blocking_reason_codes = _dedupe((*blocking_reason_codes, *leak_codes))
    outcome = _BLOCKED_OUTCOME if plan is not None else _INSPECTION_OUTCOME
    report_id = _report_id(plan=plan, budget=budget, checks=checks, evidence_refs=evidence_refs)
    return BrainReleaseGateReport(
        schema_version=RELEASE_GATE_SCHEMA_VERSION,
        suite_id=RELEASE_GATE_SUITE_ID,
        report_id=report_id,
        generated_at=_text(generated_at),
        outcome=outcome,
        rollout_plan_id=plan.plan_id if plan is not None else None,
        adapter_family=plan.adapter_family if plan is not None else None,
        candidate_backend_id=plan.candidate_backend_id if plan is not None else None,
        candidate_backend_version=plan.candidate_backend_version if plan is not None else None,
        budget_id=budget.budget_id if budget is not None else None,
        checks=checks,
        evidence_refs=evidence_refs,
        blocking_reason_codes=blocking_reason_codes,
        warning_reason_codes=warning_reason_codes,
        artifact_links=dict(artifact_links or {}),
        reason_codes=_dedupe(
            (
                "release_gate:v1",
                f"release_gate_outcome:{outcome}",
                f"release_gate_report_id:{report_id}",
                *(f"release_gate_check:{check.check_id}:{'pass' if check.passed else 'block'}" for check in checks),
                *blocking_reason_codes[:16],
            )
        ),
    )


def render_release_gate_markdown(report: BrainReleaseGateReport) -> str:
    """Render a compact Markdown release-gate summary."""
    lines = [
        "# Release Gate Report",
        "",
        f"- report_id: `{report.report_id}`",
        f"- suite: `{report.suite_id}`",
        f"- outcome: `{report.outcome}`",
        f"- passed: `{str(report.passed).lower()}`",
        f"- rollout_plan_id: `{report.rollout_plan_id or 'none'}`",
        f"- candidate: `{report.adapter_family or 'none'} / {report.candidate_backend_id or 'none'}@{report.candidate_backend_version or 'none'}`",
        "",
        "## Blocking Reason Codes",
        "",
    ]
    if report.blocking_reason_codes:
        lines.extend(f"- `{code}`" for code in report.blocking_reason_codes)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | passed | reason codes |",
            "| --- | --- | --- |",
        ]
    )
    for check in sorted(report.checks, key=lambda item: item.check_id):
        codes = ", ".join(check.reason_codes[:8]) or "none"
        lines.append(f"| `{check.check_id}` | `{str(check.passed).lower()}` | `{codes}` |")
    lines.extend(
        [
            "",
            "## Evidence References",
            "",
            "| kind | id | passed | summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for ref in sorted(report.evidence_refs, key=lambda item: (item.evidence_kind, item.evidence_id)):
        lines.append(
            f"| `{ref.evidence_kind}` | `{ref.evidence_id}` | "
            f"`{str(ref.passed).lower() if ref.passed is not None else 'n/a'}` | "
            f"{ref.summary or 'unavailable'} |"
        )
    if report.artifact_links:
        lines.extend(["", "## Artifacts", "", "| artifact | path |", "| --- | --- |"])
        for name, path in sorted(report.artifact_links.items()):
            lines.append(f"| `{name}` | `{path}` |")
    return "\n".join(lines)


def write_release_gate_artifacts(
    report: BrainReleaseGateReport,
    *,
    output_dir: str | Path = RELEASE_GATE_ARTIFACT_DIR,
) -> dict[str, str]:
    """Write deterministic JSON and Markdown release-gate artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "latest.json"
    markdown_path = output_path / "latest.md"
    json_payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(f"{json_payload}\n", encoding="utf-8")
    markdown_path.write_text(f"{render_release_gate_markdown(report)}\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "RELEASE_GATE_ARTIFACT_DIR",
    "RELEASE_GATE_SCHEMA_VERSION",
    "RELEASE_GATE_SUITE_ID",
    "BrainReleaseGateCheck",
    "BrainReleaseGateEvidenceRef",
    "BrainReleaseGateReport",
    "build_release_gate_report",
    "render_release_gate_markdown",
    "write_release_gate_artifacts",
]

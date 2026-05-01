"""Phase 21A embodied eval arena over Blink's landed runtime and simulation stack."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blink.brain.evals.adapter_benchmarks import (
    BrainEmbodiedAdapterBenchmarkReport,
    build_adapter_benchmark_report,
)
from blink.brain.evals.embodied_metrics import (
    BrainEmbodiedEvalMetricRow,
    build_embodied_eval_metric_row,
)
from blink.brain.evals.embodied_scenarios import (
    BrainEmbodiedEvalExpectation,
    BrainEmbodiedEvalMatrixEntry,
    BrainEmbodiedEvalScenario,
    BrainEmbodiedEvalSuite,
)
from blink.brain.identity import base_brain_system_prompt
from blink.brain.projections import BrainGoalStatus
from blink.brain.runtime import BrainRuntime
from blink.brain.session import BrainSessionIds, resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver, PreviewDriver
from blink.embodiment.robot_head.simulation import RobotHeadSimulationConfig, SimulationDriver
from blink.transcriptions.language import Language

_TERMINAL_GOAL_STATUSES = {
    BrainGoalStatus.COMPLETED.value,
    BrainGoalStatus.CANCELLED.value,
    BrainGoalStatus.FAILED.value,
    BrainGoalStatus.BLOCKED.value,
}


class _ArenaLLM:
    """Minimal LLM registration sink used by the eval arena runtime."""

    def register_function(self, function_name, handler):
        return None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def _event_row(event: Any) -> dict[str, Any]:
    return {
        "id": int(getattr(event, "id")),
        "event_id": str(getattr(event, "event_id")),
        "event_type": str(getattr(event, "event_type")),
        "ts": str(getattr(event, "ts")),
        "source": str(getattr(event, "source")),
        "correlation_id": getattr(event, "correlation_id"),
        "causal_parent_id": getattr(event, "causal_parent_id"),
        "payload": dict(getattr(event, "payload", {}) or {}),
        "tags": list(getattr(event, "tags", []) or []),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class BrainEmbodiedEvalRun:
    """One bounded scenario/profile run plus compact artifacts."""

    run_id: str
    scenario_id: str
    scenario_family: str
    profile_id: str
    matrix_index: int
    expectation_passed: bool
    expectation_failures: tuple[str, ...]
    metrics: BrainEmbodiedEvalMetricRow
    event_slice: tuple[dict[str, Any], ...]
    shell_snapshot: dict[str, Any]
    shell_digest: dict[str, Any]
    planning_outcome: str | None
    goal_status: str
    artifact_paths: dict[str, str | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable run payload."""
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "profile_id": self.profile_id,
            "matrix_index": self.matrix_index,
            "expectation_passed": self.expectation_passed,
            "expectation_failures": list(self.expectation_failures),
            "metrics": self.metrics.as_dict(),
            "event_slice": list(self.event_slice),
            "shell_snapshot": dict(self.shell_snapshot),
            "shell_digest": dict(self.shell_digest),
            "planning_outcome": self.planning_outcome,
            "goal_status": self.goal_status,
            "artifact_paths": dict(self.artifact_paths),
        }

    def render_markdown(self) -> str:
        """Render one compact run markdown artifact."""
        lines = [
            f"# Embodied Eval Run — {self.run_id}",
            "",
            f"- scenario: {self.scenario_id}",
            f"- family: {self.scenario_family}",
            f"- profile: {self.profile_id}",
            f"- expectation_passed: {self.expectation_passed}",
            f"- planning_outcome: {self.planning_outcome or 'none'}",
            f"- goal_status: {self.goal_status}",
            f"- task_success: {self.metrics.task_success}",
            f"- safety_success: {self.metrics.safety_success}",
            f"- preview_only: {self.metrics.preview_only}",
            f"- recovery_count: {self.metrics.recovery_count}",
            f"- review_floor_count: {self.metrics.review_floor_count}",
            f"- trace_status: {self.metrics.trace_status or 'none'}",
            "",
        ]
        if self.expectation_failures:
            lines.extend(["## Expectation Failures", ""])
            lines.extend(f"- {failure}" for failure in self.expectation_failures)
            lines.append("")
        lines.extend(
            [
                "## Artifacts",
                "",
                *(
                    f"- {key}: {value}"
                    for key, value in sorted(self.artifact_paths.items())
                    if value is not None
                ),
                "",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class BrainEmbodiedEvalReport:
    """One scenario report across its backend matrix entries."""

    suite_id: str
    scenario: BrainEmbodiedEvalScenario
    runs: tuple[BrainEmbodiedEvalRun, ...]
    benchmark_report: BrainEmbodiedAdapterBenchmarkReport
    report_json_path: Path | None = None
    report_markdown_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable scenario-report payload."""
        return {
            "suite_id": self.suite_id,
            "scenario": self.scenario.as_dict(),
            "runs": [run.as_dict() for run in self.runs],
            "benchmark_report": self.benchmark_report.as_dict(),
            "report_json_path": str(self.report_json_path) if self.report_json_path is not None else None,
            "report_markdown_path": (
                str(self.report_markdown_path) if self.report_markdown_path is not None else None
            ),
        }

    def render_markdown(self) -> str:
        """Render one compact scenario markdown report."""
        lines = [
            f"# Embodied Eval Report — {self.scenario.scenario_id}",
            "",
            f"- suite: {self.suite_id}",
            f"- family: {self.scenario.family}",
            f"- version: {self.scenario.version}",
            f"- runs: {len(self.runs)}",
            "",
            "## Run Summary",
            "",
            "| profile | backend | task_success | safety_success | preview_only | recovery_count | review_floor_count | expectation_passed |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for run in self.runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        run.profile_id,
                        run.metrics.execution_backend,
                        str(run.metrics.task_success),
                        str(run.metrics.safety_success),
                        str(run.metrics.preview_only),
                        str(run.metrics.recovery_count),
                        str(run.metrics.review_floor_count),
                        str(run.expectation_passed),
                    ]
                )
                + " |"
            )
        lines.extend(["", self.benchmark_report.render_markdown(), ""])
        return "\n".join(lines)


@dataclass(frozen=True)
class BrainEmbodiedEvalSuiteResult:
    """One aggregate suite result over one or more scenario reports."""

    suite: BrainEmbodiedEvalSuite
    reports: tuple[BrainEmbodiedEvalReport, ...]
    benchmark_report: BrainEmbodiedAdapterBenchmarkReport
    report_json_path: Path | None = None
    report_markdown_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable suite payload."""
        return {
            "suite": self.suite.as_dict(),
            "reports": [report.as_dict() for report in self.reports],
            "benchmark_report": self.benchmark_report.as_dict(),
            "report_json_path": str(self.report_json_path) if self.report_json_path is not None else None,
            "report_markdown_path": (
                str(self.report_markdown_path) if self.report_markdown_path is not None else None
            ),
        }

    def render_markdown(self) -> str:
        """Render one compact suite markdown artifact."""
        lines = [
            f"# Embodied Eval Suite — {self.suite.suite_id}",
            "",
            f"- scenarios: {len(self.reports)}",
            f"- comparison rows: {len(self.benchmark_report.comparisons)}",
            "",
        ]
        for report in self.reports:
            lines.append(f"- {report.scenario.scenario_id}: {len(report.runs)} runs")
        lines.extend(["", self.benchmark_report.render_markdown(), ""])
        return "\n".join(lines)


def _evaluate_expectation(
    expectation: BrainEmbodiedEvalExpectation,
    metrics: BrainEmbodiedEvalMetricRow,
) -> tuple[str, ...]:
    failures: list[str] = []
    if expectation.task_success is not None and metrics.task_success != expectation.task_success:
        failures.append(
            f"task_success expected {expectation.task_success} but saw {metrics.task_success}"
        )
    if expectation.safety_success is not None and metrics.safety_success != expectation.safety_success:
        failures.append(
            f"safety_success expected {expectation.safety_success} but saw {metrics.safety_success}"
        )
    if expectation.preview_only is not None and metrics.preview_only != expectation.preview_only:
        failures.append(
            f"preview_only expected {expectation.preview_only} but saw {metrics.preview_only}"
        )
    if expectation.require_recovery is not None and (
        metrics.recovery_count > 0
    ) != expectation.require_recovery:
        failures.append(
            f"require_recovery expected {expectation.require_recovery} but saw {metrics.recovery_count > 0}"
        )
    if expectation.required_trace_status is not None and (
        metrics.trace_status != expectation.required_trace_status
    ):
        failures.append(
            f"trace_status expected {expectation.required_trace_status} but saw {metrics.trace_status}"
        )
    required_codes = set(expectation.required_mismatch_codes)
    if required_codes and not required_codes.issubset(set(metrics.mismatch_codes)):
        failures.append(
            "required_mismatch_codes missing "
            + ",".join(sorted(required_codes.difference(set(metrics.mismatch_codes))))
        )
    if (
        expectation.min_review_floor_count is not None
        and metrics.review_floor_count < expectation.min_review_floor_count
    ):
        failures.append(
            f"review_floor_count expected >= {expectation.min_review_floor_count} but saw {metrics.review_floor_count}"
        )
    if (
        expectation.max_review_floor_count is not None
        and metrics.review_floor_count > expectation.max_review_floor_count
    ):
        failures.append(
            f"review_floor_count expected <= {expectation.max_review_floor_count} but saw {metrics.review_floor_count}"
        )
    if expectation.min_step_count is not None and metrics.step_count < expectation.min_step_count:
        failures.append(
            f"step_count expected >= {expectation.min_step_count} but saw {metrics.step_count}"
        )
    return tuple(failures)


class EmbodiedEvalArena:
    """Run typed embodied scenarios over Blink's landed runtime and driver seams."""

    def __init__(self, *, language: Language = Language.EN, runtime_kind: str = "browser"):
        """Initialize the arena with one bounded runtime profile."""
        self._language = language
        self._runtime_kind = runtime_kind

    async def run_scenario(
        self,
        *,
        suite_id: str,
        scenario: BrainEmbodiedEvalScenario,
        output_dir: Path | None = None,
    ) -> BrainEmbodiedEvalReport:
        """Run one scenario across its backend matrix and emit bounded artifacts."""
        resolved_output_dir = output_dir or (
            Path("artifacts") / "brain_evals" / _safe_name(suite_id) / _safe_name(scenario.scenario_id)
        )
        shutil.rmtree(resolved_output_dir, ignore_errors=True)
        runs: list[BrainEmbodiedEvalRun] = []
        for matrix_index, entry in enumerate(scenario.adapter_matrix.entries):
            run_dir = resolved_output_dir / _safe_name(entry.profile.profile_id)
            runs.append(
                await self._run_matrix_entry(
                    suite_id=suite_id,
                    scenario=scenario,
                    matrix_index=matrix_index,
                    entry=entry,
                    run_dir=run_dir,
                )
            )
        benchmark_report = build_adapter_benchmark_report([run.metrics for run in runs])
        report = BrainEmbodiedEvalReport(
            suite_id=suite_id,
            scenario=scenario,
            runs=tuple(runs),
            benchmark_report=benchmark_report,
            report_json_path=resolved_output_dir / "scenario_report.json",
            report_markdown_path=resolved_output_dir / "scenario_report.md",
        )
        _write_json(report.report_json_path, report.as_dict())
        report.report_markdown_path.write_text(report.render_markdown(), encoding="utf-8")
        return report

    async def run_suite(
        self,
        *,
        suite: BrainEmbodiedEvalSuite,
        output_dir: Path | None = None,
        scenario_id: str | None = None,
    ) -> BrainEmbodiedEvalSuiteResult:
        """Run one named suite and emit a bounded aggregate report."""
        resolved_output_dir = output_dir or Path("artifacts") / "brain_evals" / _safe_name(suite.suite_id)
        scenarios = (
            (suite.scenario(scenario_id),) if scenario_id is not None else suite.scenarios
        )
        selected = tuple(scenario for scenario in scenarios if scenario is not None)
        if not selected:
            raise KeyError(f"Unknown scenario '{scenario_id}' for suite '{suite.suite_id}'.")
        reports: list[BrainEmbodiedEvalReport] = []
        for scenario in selected:
            reports.append(
                await self.run_scenario(
                    suite_id=suite.suite_id,
                    scenario=scenario,
                    output_dir=resolved_output_dir / _safe_name(scenario.scenario_id),
                )
            )
        suite_benchmark = build_adapter_benchmark_report(
            [run.metrics for report in reports for run in report.runs]
        )
        result = BrainEmbodiedEvalSuiteResult(
            suite=suite,
            reports=tuple(reports),
            benchmark_report=suite_benchmark,
            report_json_path=resolved_output_dir / "suite_report.json",
            report_markdown_path=resolved_output_dir / "suite_report.md",
        )
        _write_json(result.report_json_path, result.as_dict())
        result.report_markdown_path.write_text(result.render_markdown(), encoding="utf-8")
        return result

    async def _run_matrix_entry(
        self,
        *,
        suite_id: str,
        scenario: BrainEmbodiedEvalScenario,
        matrix_index: int,
        entry: BrainEmbodiedEvalMatrixEntry,
        run_dir: Path,
    ) -> BrainEmbodiedEvalRun:
        run_dir.mkdir(parents=True, exist_ok=True)
        session_ids = resolve_brain_session_ids(
            runtime_kind=self._runtime_kind,
            client_id=f"eval-{scenario.scenario_id}-{entry.profile.profile_id}",
        )
        controller = RobotHeadController(
            catalog=build_default_robot_head_catalog(),
            driver=self._build_driver(entry=entry, run_dir=run_dir),
        )
        runtime = BrainRuntime(
            base_prompt=base_brain_system_prompt(self._language),
            language=self._language,
            runtime_kind=self._runtime_kind,
            session_resolver=lambda: session_ids,
            llm=_ArenaLLM(),
            robot_head_controller=controller,
            brain_db_path=run_dir / "brain.db",
        )
        try:
            baseline_event_ids = {
                str(event.event_id)
                for event in runtime.store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=64,
                )
            }
            goal_id = runtime.executive.create_commitment_goal(
                title=scenario.task.title,
                intent=scenario.task.intent,
                source=scenario.task.source,
                goal_family=scenario.task.goal_family,
                goal_status=BrainGoalStatus.OPEN.value,
                details={
                    "survive_restart": True,
                    "capabilities": [step.as_dict() for step in scenario.task.capabilities],
                    "eval_scenario_id": scenario.scenario_id,
                    "eval_suite_id": suite_id,
                    "eval_profile_id": entry.profile.profile_id,
                },
            )
            planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
            goal_status = BrainGoalStatus.OPEN.value
            for _ in range(scenario.stop_conditions.max_cycles):
                cycle_result = await runtime.executive.run_once()
                goal_status = self._goal_status(
                    store=runtime.store,
                    session_ids=session_ids,
                    goal_id=goal_id,
                )
                if (
                    scenario.stop_conditions.stop_on_terminal_goal
                    and goal_status in _TERMINAL_GOAL_STATUSES
                ):
                    break
                if not cycle_result.progressed:
                    break

            recent_events = list(
                reversed(
                    runtime.store.recent_brain_events(
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        limit=max(192, scenario.artifact_policy.keep_event_slice_limit * 2),
                    )
                )
            )
            event_slice = tuple(
                _event_row(event)
                for event in recent_events
                if str(event.event_id) not in baseline_event_ids
            )[-scenario.artifact_policy.keep_event_slice_limit :]
            shell_snapshot = runtime.shell.snapshot().as_dict()
            shell_digest = runtime.shell.runtime_shell_digest()
            audit_report = (
                runtime.shell.export_audit(output_dir=run_dir / "audit")
                if scenario.artifact_policy.export_audit
                else None
            )
            recent_action_events = list(
                reversed(
                    runtime.store.recent_action_events(
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        limit=24,
                    )
                )
            )
            embodied_projection = runtime.store.build_embodied_executive_projection(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                agent_id=session_ids.agent_id,
                presence_scope_key=runtime.presence_scope_key,
            )
            run_id = (
                f"{scenario.scenario_id}:{scenario.version}:{matrix_index}:{entry.profile.profile_id}"
            )
            artifact_paths = {
                "brain_db": str(run_dir / "brain.db"),
                "audit_json": str(audit_report.json_path) if audit_report and audit_report.json_path else None,
                "audit_markdown": (
                    str(audit_report.markdown_path)
                    if audit_report and audit_report.markdown_path
                    else None
                ),
                "run_json": str(run_dir / "run.json"),
                "run_markdown": str(run_dir / "run.md"),
            }
            metrics = build_embodied_eval_metric_row(
                run_id=run_id,
                scenario_id=scenario.scenario_id,
                scenario_family=scenario.family,
                scenario_version=scenario.version,
                profile_id=entry.profile.profile_id,
                matrix_index=matrix_index,
                execution_backend=entry.profile.execution_backend,
                perception_backend_id=entry.profile.perception_backend_id,
                world_model_backend_id=runtime.store.world_model_adapter.descriptor.backend_id,
                embodied_policy_backend_id=(
                    runtime.action_engine.policy_adapter.descriptor.backend_id
                    if runtime.action_engine is not None
                    else entry.profile.embodied_policy_backend_id
                ),
                goal_status=goal_status,
                planning_outcome=getattr(planning_result, "outcome", None),
                step_count=len(scenario.task.capabilities),
                recent_action_events=[
                    {
                        "action_id": record.action_id,
                        "source": record.source,
                        "accepted": record.accepted,
                        "preview_only": record.preview_only,
                        "metadata": dict(record.metadata),
                    }
                    for record in recent_action_events
                ],
                predictive_inspection=dict(shell_digest.get("predictive_inspection", {})),
                rehearsal_inspection=dict(shell_digest.get("rehearsal_inspection", {})),
                recent_execution_trace=(
                    dict(shell_snapshot["recent_embodied_execution_traces"][0])
                    if shell_snapshot.get("recent_embodied_execution_traces")
                    else None
                ),
                recent_recoveries=list(shell_snapshot.get("recent_embodied_recoveries", [])),
                artifact_paths=artifact_paths,
            )
            expectation_failures = _evaluate_expectation(entry.expectation, metrics)
            run = BrainEmbodiedEvalRun(
                run_id=run_id,
                scenario_id=scenario.scenario_id,
                scenario_family=scenario.family,
                profile_id=entry.profile.profile_id,
                matrix_index=matrix_index,
                expectation_passed=not expectation_failures,
                expectation_failures=expectation_failures,
                metrics=metrics,
                event_slice=event_slice,
                shell_snapshot=shell_snapshot,
                shell_digest=shell_digest,
                planning_outcome=getattr(planning_result, "outcome", None),
                goal_status=goal_status,
                artifact_paths=artifact_paths,
            )
            _write_json(run_dir / "run.json", run.as_dict())
            if scenario.artifact_policy.export_markdown:
                (run_dir / "run.md").write_text(run.render_markdown(), encoding="utf-8")
            runtime.store.record_memory_export(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                export_kind="embodied_eval_run",
                path=run_dir / "run.json",
                payload=run.as_dict(),
                metadata={
                    "suite_id": suite_id,
                    "scenario_id": scenario.scenario_id,
                    "profile_id": entry.profile.profile_id,
                },
            )
            return run
        finally:
            await controller.close()
            runtime.close()

    def _build_driver(self, *, entry: BrainEmbodiedEvalMatrixEntry, run_dir: Path):
        simulation_driver = SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=run_dir / "simulation")
        )
        if entry.profile.execution_backend == "simulation":
            return simulation_driver
        if entry.profile.execution_backend == "preview":
            return PreviewDriver(trace_dir=run_dir / "preview")
        if entry.profile.execution_backend == "fault":
            return FaultInjectionDriver(
                busy=entry.profile.fault_profile.busy,
                missing_arm=entry.profile.fault_profile.missing_arm,
                degraded=entry.profile.fault_profile.degraded,
                wrapped=simulation_driver,
            )
        raise ValueError(
            f"Unsupported embodied eval backend '{entry.profile.execution_backend}'."
        )

    def _goal_status(
        self,
        *,
        store: BrainStore,
        session_ids: BrainSessionIds,
        goal_id: str,
    ) -> str:
        agenda = store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = agenda.goal(goal_id)
        if goal is None:
            return BrainGoalStatus.CANCELLED.value
        return str(goal.status)


__all__ = [
    "BrainEmbodiedEvalReport",
    "BrainEmbodiedEvalRun",
    "BrainEmbodiedEvalSuiteResult",
    "EmbodiedEvalArena",
]

"""Deterministic backend-comparison summaries for Phase 21A embodied evals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blink.brain.evals.embodied_metrics import BrainEmbodiedEvalMetricRow


@dataclass(frozen=True)
class BrainEmbodiedAdapterBenchmarkComparison:
    """One incumbent-vs-candidate comparison row for a shared scenario family."""

    scenario_id: str
    scenario_family: str
    incumbent_profile_id: str
    candidate_profile_id: str
    incumbent_backend: str
    candidate_backend: str
    task_success_delta: int
    safety_success_delta: int
    recovery_count_delta: int
    review_floor_delta: int
    preview_only_delta: int
    candidate_improved: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable comparison payload."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "incumbent_profile_id": self.incumbent_profile_id,
            "candidate_profile_id": self.candidate_profile_id,
            "incumbent_backend": self.incumbent_backend,
            "candidate_backend": self.candidate_backend,
            "task_success_delta": self.task_success_delta,
            "safety_success_delta": self.safety_success_delta,
            "recovery_count_delta": self.recovery_count_delta,
            "review_floor_delta": self.review_floor_delta,
            "preview_only_delta": self.preview_only_delta,
            "candidate_improved": self.candidate_improved,
        }


@dataclass(frozen=True)
class BrainEmbodiedAdapterBenchmarkReport:
    """Compact benchmark report across one or more embodied eval scenarios."""

    scenario_count: int
    compared_family_count: int
    comparisons: tuple[BrainEmbodiedAdapterBenchmarkComparison, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable benchmark-report payload."""
        return {
            "scenario_count": self.scenario_count,
            "compared_family_count": self.compared_family_count,
            "comparisons": [comparison.as_dict() for comparison in self.comparisons],
        }

    def render_markdown(self) -> str:
        """Render a compact markdown summary for operator review."""
        lines = [
            "# Embodied Adapter Benchmark Report",
            "",
            f"- scenarios: {self.scenario_count}",
            f"- compared families: {self.compared_family_count}",
            f"- comparison rows: {len(self.comparisons)}",
            "",
        ]
        if not self.comparisons:
            lines.append("No shared-family incumbent/candidate comparisons were produced.")
            return "\n".join(lines)
        lines.append("| scenario | family | incumbent | candidate | task Δ | safety Δ | recovery Δ | review Δ |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in self.comparisons:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.scenario_id,
                        row.scenario_family,
                        row.incumbent_profile_id,
                        row.candidate_profile_id,
                        str(row.task_success_delta),
                        str(row.safety_success_delta),
                        str(row.recovery_count_delta),
                        str(row.review_floor_delta),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


def build_adapter_benchmark_report(
    metric_rows: tuple[BrainEmbodiedEvalMetricRow, ...] | list[BrainEmbodiedEvalMetricRow],
) -> BrainEmbodiedAdapterBenchmarkReport:
    """Build a deterministic incumbent-vs-candidate report from eval metric rows."""
    rows = sorted(
        list(metric_rows),
        key=lambda row: (
            row.scenario_id,
            row.matrix_index,
            row.profile_id,
            row.execution_backend,
        ),
    )
    grouped: dict[str, list[BrainEmbodiedEvalMetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.scenario_id, []).append(row)
    comparisons: list[BrainEmbodiedAdapterBenchmarkComparison] = []
    compared_families: set[str] = set()
    for scenario_id, scenario_rows in sorted(grouped.items()):
        if len(scenario_rows) < 2:
            continue
        incumbent = min(
            scenario_rows,
            key=lambda row: (row.matrix_index, row.profile_id, row.execution_backend),
        )
        for candidate in sorted(
            (row for row in scenario_rows if row.run_id != incumbent.run_id),
            key=lambda row: (row.matrix_index, row.profile_id, row.execution_backend),
        ):
            comparisons.append(
                BrainEmbodiedAdapterBenchmarkComparison(
                    scenario_id=scenario_id,
                    scenario_family=incumbent.scenario_family,
                    incumbent_profile_id=incumbent.profile_id,
                    candidate_profile_id=candidate.profile_id,
                    incumbent_backend=incumbent.execution_backend,
                    candidate_backend=candidate.execution_backend,
                    task_success_delta=int(candidate.task_success) - int(incumbent.task_success),
                    safety_success_delta=int(candidate.safety_success)
                    - int(incumbent.safety_success),
                    recovery_count_delta=candidate.recovery_count - incumbent.recovery_count,
                    review_floor_delta=candidate.review_floor_count
                    - incumbent.review_floor_count,
                    preview_only_delta=int(candidate.preview_only) - int(incumbent.preview_only),
                    candidate_improved=(
                        int(candidate.task_success) > int(incumbent.task_success)
                        or (
                            candidate.task_success == incumbent.task_success
                            and candidate.review_floor_count < incumbent.review_floor_count
                        )
                    ),
                )
            )
            compared_families.add(incumbent.scenario_family)
    return BrainEmbodiedAdapterBenchmarkReport(
        scenario_count=len(grouped),
        compared_family_count=len(compared_families),
        comparisons=tuple(comparisons),
    )


__all__ = [
    "BrainEmbodiedAdapterBenchmarkComparison",
    "BrainEmbodiedAdapterBenchmarkReport",
    "build_adapter_benchmark_report",
]

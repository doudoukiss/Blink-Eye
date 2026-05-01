"""Typed embodied-eval scenarios and built-in Phase 21A suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from blink.brain.projections import BrainGoalFamily


def _sorted_unique_texts(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True)
class BrainEmbodiedEvalStep:
    """One bounded capability step inside an embodied eval task."""

    capability_id: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the prompt-safe step payload."""
        return {
            "capability_id": self.capability_id,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalTask:
    """One bounded goal-backed embodied task for the eval arena."""

    title: str
    description: str
    capabilities: tuple[BrainEmbodiedEvalStep, ...]
    intent: str = "robot_head.sequence"
    goal_family: str = BrainGoalFamily.ENVIRONMENT.value
    source: str = "operator"

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable task payload."""
        return {
            "title": self.title,
            "description": self.description,
            "capabilities": [step.as_dict() for step in self.capabilities],
            "intent": self.intent,
            "goal_family": self.goal_family,
            "source": self.source,
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalStopConditions:
    """Bounded stopping rules for one eval scenario."""

    max_cycles: int = 2
    stop_on_terminal_goal: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the stop-condition payload."""
        return {
            "max_cycles": int(self.max_cycles),
            "stop_on_terminal_goal": bool(self.stop_on_terminal_goal),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalArtifactPolicy:
    """Retention and boundedness rules for one eval run."""

    keep_event_slice_limit: int = 96
    export_audit: bool = True
    export_markdown: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return the artifact-policy payload."""
        return {
            "keep_event_slice_limit": int(self.keep_event_slice_limit),
            "export_audit": bool(self.export_audit),
            "export_markdown": bool(self.export_markdown),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalExpectation:
    """Expected bounded outcome for one backend profile on one scenario."""

    task_success: bool | None = None
    safety_success: bool | None = None
    preview_only: bool | None = None
    require_recovery: bool | None = None
    required_trace_status: str | None = None
    required_mismatch_codes: tuple[str, ...] = ()
    min_review_floor_count: int | None = None
    max_review_floor_count: int | None = None
    min_step_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable expectation payload."""
        return {
            "task_success": self.task_success,
            "safety_success": self.safety_success,
            "preview_only": self.preview_only,
            "require_recovery": self.require_recovery,
            "required_trace_status": self.required_trace_status,
            "required_mismatch_codes": list(self.required_mismatch_codes),
            "min_review_floor_count": self.min_review_floor_count,
            "max_review_floor_count": self.max_review_floor_count,
            "min_step_count": self.min_step_count,
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalFaultProfile:
    """Deterministic lightweight fault profile for Phase 21A."""

    busy: bool = False
    missing_arm: bool = False
    degraded: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable fault-profile payload."""
        return {
            "busy": bool(self.busy),
            "missing_arm": bool(self.missing_arm),
            "degraded": bool(self.degraded),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalBackendProfile:
    """One bounded backend/runtime profile exercised by the eval arena."""

    profile_id: str
    label: str
    execution_backend: str
    description: str = ""
    fault_profile: BrainEmbodiedEvalFaultProfile = field(default_factory=BrainEmbodiedEvalFaultProfile)
    perception_backend_id: str | None = None
    world_model_backend_id: str = "local_world_model"
    embodied_policy_backend_id: str = "local_robot_head_policy"
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable backend-profile payload."""
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "execution_backend": self.execution_backend,
            "description": self.description,
            "fault_profile": self.fault_profile.as_dict(),
            "perception_backend_id": self.perception_backend_id,
            "world_model_backend_id": self.world_model_backend_id,
            "embodied_policy_backend_id": self.embodied_policy_backend_id,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalMatrixEntry:
    """One backend-profile plus expectation entry inside a scenario matrix."""

    profile: BrainEmbodiedEvalBackendProfile
    expectation: BrainEmbodiedEvalExpectation = field(default_factory=BrainEmbodiedEvalExpectation)

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable matrix-entry payload."""
        return {
            "profile": self.profile.as_dict(),
            "expectation": self.expectation.as_dict(),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalAdapterMatrix:
    """Bounded backend matrix for one shared embodied scenario."""

    entries: tuple[BrainEmbodiedEvalMatrixEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable adapter-matrix payload."""
        return {"entries": [entry.as_dict() for entry in self.entries]}


@dataclass(frozen=True)
class BrainEmbodiedEvalScenario:
    """One typed embodied eval scenario."""

    scenario_id: str
    family: str
    version: str
    description: str
    task: BrainEmbodiedEvalTask
    adapter_matrix: BrainEmbodiedEvalAdapterMatrix
    stop_conditions: BrainEmbodiedEvalStopConditions = field(
        default_factory=BrainEmbodiedEvalStopConditions
    )
    intervention_policy: str = "bounded_operator_visible"
    artifact_policy: BrainEmbodiedEvalArtifactPolicy = field(
        default_factory=BrainEmbodiedEvalArtifactPolicy
    )

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable scenario payload."""
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "version": self.version,
            "description": self.description,
            "task": self.task.as_dict(),
            "adapter_matrix": self.adapter_matrix.as_dict(),
            "stop_conditions": self.stop_conditions.as_dict(),
            "intervention_policy": self.intervention_policy,
            "artifact_policy": self.artifact_policy.as_dict(),
        }


@dataclass(frozen=True)
class BrainEmbodiedEvalSuite:
    """One named suite of embodied eval scenarios."""

    suite_id: str
    description: str
    scenarios: tuple[BrainEmbodiedEvalScenario, ...]

    def scenario(self, scenario_id: str) -> BrainEmbodiedEvalScenario | None:
        """Return one scenario by id, if present."""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return the inspectable suite payload."""
        return {
            "suite_id": self.suite_id,
            "description": self.description,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
        }


def _step(capability_id: str, **arguments: Any) -> BrainEmbodiedEvalStep:
    return BrainEmbodiedEvalStep(capability_id=capability_id, arguments=dict(arguments))


def _matrix_entry(
    *,
    profile_id: str,
    label: str,
    execution_backend: str,
    description: str = "",
    busy: bool = False,
    missing_arm: bool = False,
    degraded: bool = False,
    perception_backend_id: str | None = None,
    expectation: BrainEmbodiedEvalExpectation | None = None,
    tags: tuple[str, ...] = (),
) -> BrainEmbodiedEvalMatrixEntry:
    return BrainEmbodiedEvalMatrixEntry(
        profile=BrainEmbodiedEvalBackendProfile(
            profile_id=profile_id,
            label=label,
            execution_backend=execution_backend,
            description=description,
            fault_profile=BrainEmbodiedEvalFaultProfile(
                busy=busy,
                missing_arm=missing_arm,
                degraded=degraded,
            ),
            perception_backend_id=perception_backend_id,
            tags=_sorted_unique_texts(tags),
        ),
        expectation=expectation or BrainEmbodiedEvalExpectation(),
    )


def build_smoke_embodied_eval_suite() -> BrainEmbodiedEvalSuite:
    """Return the bounded CI-suitable Phase 21A smoke suite."""
    look_left_compare = BrainEmbodiedEvalScenario(
        scenario_id="robot_head_look_left_compare",
        family="robot_head_single_step",
        version="v1",
        description=(
            "Compare one healthy single-step robot-head action across incumbent "
            "simulation and preview candidate backends."
        ),
        task=BrainEmbodiedEvalTask(
            title="Look left through the eval arena",
            description="One bounded robot-head look-left action.",
            capabilities=(_step("robot_head.look_left"),),
        ),
        adapter_matrix=BrainEmbodiedEvalAdapterMatrix(
            entries=(
                _matrix_entry(
                    profile_id="incumbent_simulation",
                    label="Incumbent simulation",
                    execution_backend="simulation",
                    description="Deterministic execution through the simulation driver.",
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=True,
                        safety_success=True,
                        preview_only=False,
                        require_recovery=False,
                        required_trace_status="succeeded",
                        max_review_floor_count=0,
                        min_step_count=1,
                    ),
                    tags=("incumbent", "smoke"),
                ),
                _matrix_entry(
                    profile_id="candidate_preview",
                    label="Candidate preview",
                    execution_backend="preview",
                    description="Deterministic preview-only execution for backend comparison.",
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=True,
                        safety_success=True,
                        preview_only=True,
                        require_recovery=False,
                        required_trace_status="succeeded",
                        max_review_floor_count=0,
                        min_step_count=1,
                    ),
                    tags=("candidate", "smoke"),
                ),
            )
        ),
        stop_conditions=BrainEmbodiedEvalStopConditions(max_cycles=2),
    )
    busy_fault = BrainEmbodiedEvalScenario(
        scenario_id="robot_head_busy_fault",
        family="robot_head_busy_fault",
        version="v1",
        description=(
            "Exercise the bounded operator-review floor when the robot-head "
            "backend is busy before dispatch."
        ),
        task=BrainEmbodiedEvalTask(
            title="Attempt a robot-head action while ownership is busy",
            description="One bounded busy-fault planning scenario.",
            capabilities=(_step("robot_head.look_left"),),
        ),
        adapter_matrix=BrainEmbodiedEvalAdapterMatrix(
            entries=(
                _matrix_entry(
                    profile_id="fault_busy",
                    label="Busy fault",
                    execution_backend="fault",
                    description="Deterministic busy ownership rejection over the fault driver.",
                    busy=True,
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=False,
                        safety_success=True,
                        require_recovery=False,
                        min_review_floor_count=1,
                        min_step_count=1,
                    ),
                    tags=("fault", "smoke"),
                ),
            )
        ),
        stop_conditions=BrainEmbodiedEvalStopConditions(max_cycles=2),
    )
    multi_step = BrainEmbodiedEvalScenario(
        scenario_id="robot_head_multi_step_sequence",
        family="robot_head_multi_step",
        version="v1",
        description="Run a deterministic multi-step robot-head sequence through the eval arena.",
        task=BrainEmbodiedEvalTask(
            title="Look left and return neutral",
            description="A bounded two-step robot-head sequence.",
            capabilities=(
                _step("robot_head.look_left"),
                _step("robot_head.return_neutral"),
            ),
        ),
        adapter_matrix=BrainEmbodiedEvalAdapterMatrix(
            entries=(
                _matrix_entry(
                    profile_id="simulation_multi_step",
                    label="Simulation multi-step",
                    execution_backend="simulation",
                    description="Two-step deterministic simulation sequence.",
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=True,
                        safety_success=True,
                        preview_only=False,
                        require_recovery=False,
                        required_trace_status="succeeded",
                        max_review_floor_count=0,
                        min_step_count=2,
                    ),
                    tags=("smoke", "multi_step"),
                ),
            )
        ),
        stop_conditions=BrainEmbodiedEvalStopConditions(max_cycles=4),
    )
    return BrainEmbodiedEvalSuite(
        suite_id="smoke",
        description="Minimal deterministic embodied eval suite for CI-suitable runs.",
        scenarios=(look_left_compare, busy_fault, multi_step),
    )


def build_benchmark_embodied_eval_suite() -> BrainEmbodiedEvalSuite:
    """Return the bounded Phase 21A benchmark-suite skeleton."""
    degraded_compare = BrainEmbodiedEvalScenario(
        scenario_id="robot_head_degraded_comparison",
        family="robot_head_degraded_backend_comparison",
        version="v1",
        description=(
            "Benchmark skeleton that compares healthy simulation against a degraded "
            "fault-injection path on the same bounded action family."
        ),
        task=BrainEmbodiedEvalTask(
            title="Compare degraded backend behavior for one robot-head action",
            description="A shared family for benchmark-style comparison.",
            capabilities=(_step("robot_head.look_right"),),
        ),
        adapter_matrix=BrainEmbodiedEvalAdapterMatrix(
            entries=(
                _matrix_entry(
                    profile_id="benchmark_incumbent_simulation",
                    label="Benchmark incumbent simulation",
                    execution_backend="simulation",
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=True,
                        safety_success=True,
                        preview_only=False,
                        require_recovery=False,
                        required_trace_status="succeeded",
                        max_review_floor_count=0,
                        min_step_count=1,
                    ),
                    tags=("benchmark", "incumbent"),
                ),
                _matrix_entry(
                    profile_id="benchmark_candidate_degraded",
                    label="Benchmark candidate degraded fault",
                    execution_backend="fault",
                    description="Degraded deterministic candidate used for comparison skeletons.",
                    degraded=True,
                    expectation=BrainEmbodiedEvalExpectation(
                        task_success=True,
                        safety_success=True,
                        preview_only=False,
                        require_recovery=False,
                        required_trace_status="succeeded",
                        min_step_count=1,
                    ),
                    tags=("benchmark", "candidate"),
                ),
            )
        ),
        stop_conditions=BrainEmbodiedEvalStopConditions(max_cycles=2),
    )
    smoke = build_smoke_embodied_eval_suite()
    return BrainEmbodiedEvalSuite(
        suite_id="benchmark",
        description=(
            "Deterministic benchmark-suite skeleton built on the same bounded "
            "simulation and fault-driver seams as smoke."
        ),
        scenarios=(*smoke.scenarios, degraded_compare),
    )


def load_builtin_embodied_eval_suite(name: str) -> BrainEmbodiedEvalSuite:
    """Return one built-in eval suite by name."""
    normalized = str(name).strip().lower()
    if normalized == "smoke":
        return build_smoke_embodied_eval_suite()
    if normalized == "benchmark":
        return build_benchmark_embodied_eval_suite()
    raise KeyError(f"Unknown embodied eval suite: {name}")


__all__ = [
    "BrainEmbodiedEvalAdapterMatrix",
    "BrainEmbodiedEvalArtifactPolicy",
    "BrainEmbodiedEvalBackendProfile",
    "BrainEmbodiedEvalExpectation",
    "BrainEmbodiedEvalFaultProfile",
    "BrainEmbodiedEvalMatrixEntry",
    "BrainEmbodiedEvalScenario",
    "BrainEmbodiedEvalStep",
    "BrainEmbodiedEvalStopConditions",
    "BrainEmbodiedEvalSuite",
    "BrainEmbodiedEvalTask",
    "build_benchmark_embodied_eval_suite",
    "build_smoke_embodied_eval_suite",
    "load_builtin_embodied_eval_suite",
]

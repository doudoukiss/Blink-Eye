"""Deterministic offline simulation backend for Blink robot-head embodiment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from blink.embodiment.robot_head.driver import RobotHeadDriver
from blink.embodiment.robot_head.live_hardware import (
    RobotHeadLiveHardwareProfile,
    load_robot_head_live_hardware_profile,
)
from blink.embodiment.robot_head.models import (
    RobotHeadDriverStatus,
    RobotHeadExecutionPlan,
    RobotHeadExecutionResult,
)

if TYPE_CHECKING:
    from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog


SIMULATION_SCHEMA_VERSION = "blink_robot_head_simulation/v1"
DEFAULT_TRACE_DIR = Path.cwd() / "artifacts" / "robot_head_simulation"
DEFAULT_VOLTAGE = 120
DEFAULT_TEMPERATURE = 25
DEFAULT_CURRENT = 0
LOW_VOLTAGE_THRESHOLD = 110
HIGH_TEMPERATURE_THRESHOLD = 60


@dataclass(frozen=True)
class RobotHeadSimulationConfig:
    """Configuration for the robot-head simulation backend.

    Args:
        hardware_profile_path: Optional override path to the live hardware profile JSON.
        scenario_path: Optional override path to a simulation scenario JSON file.
        realtime: Whether simulated motion should wait in wall-clock time.
        trace_dir: Optional directory for simulation trace artifacts.
    """

    hardware_profile_path: str | Path | None = None
    scenario_path: str | Path | None = None
    realtime: bool = False
    trace_dir: Path | None = None


class RobotHeadFaultProfile(BaseModel):
    """Deterministic fault profile applied by the robot-head simulator."""

    model_config = ConfigDict(frozen=True)

    busy: bool = False
    missing_arm: bool = False
    degraded: bool = False
    missing_servo_ids: list[int] = Field(default_factory=list)
    stalled_servo_ids: list[int] = Field(default_factory=list)
    slow_servo_ids: list[int] = Field(default_factory=list)
    settle_extra_ms: int = 0
    voltage_by_servo: dict[int, int] = Field(default_factory=dict)
    temperature_by_servo: dict[int, int] = Field(default_factory=dict)
    current_by_servo: dict[int, int] = Field(default_factory=dict)
    status_byte_by_servo: dict[int, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RobotHeadSimulationScenario(BaseModel):
    """Replayable scenario for deterministic robot-head simulation."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SIMULATION_SCHEMA_VERSION
    name: str = "default"
    description: str = "Default healthy hardware-free Blink robot-head simulation."
    owner: str | None = None
    initial_positions: dict[int, int] = Field(default_factory=dict)
    faults: RobotHeadFaultProfile = Field(default_factory=RobotHeadFaultProfile)


def load_robot_head_simulation_scenario(
    path: str | Path | None = None,
) -> RobotHeadSimulationScenario:
    """Load a robot-head simulation scenario from JSON or return the default scenario."""
    if path in (None, ""):
        return RobotHeadSimulationScenario()
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return RobotHeadSimulationScenario.model_validate(payload)


@dataclass(frozen=True)
class RobotHeadMotionPresetKinetics:
    """Resolved transition lane for one validated robot-head preset."""

    speed: int
    acceleration: int
    default_transition_ms: int
    minimum_transition_ms: int
    neutral_recovery_ms: int


@dataclass(frozen=True)
class RobotHeadSimulationOutcome:
    """Internal simulation outcome produced for one validated plan."""

    preview_only: bool
    warnings: list[str]
    status: RobotHeadDriverStatus
    summary: str
    metadata: dict[str, Any]


class RobotHeadSimulator:
    """Pure in-memory robot-head simulator used by the simulation driver."""

    def __init__(
        self,
        *,
        hardware_profile: RobotHeadLiveHardwareProfile,
        scenario: RobotHeadSimulationScenario,
    ):
        """Initialize the simulator state.

        Args:
            hardware_profile: Hardware profile used for servo mapping.
            scenario: Deterministic scenario that configures faults and telemetry.
        """
        self._hardware_profile = hardware_profile
        self._scenario = scenario
        self._positions = hardware_profile.neutral_positions()
        self._positions.update({int(key): int(value) for key, value in scenario.initial_positions.items()})
        self._moving_flags = {servo_id: 0 for servo_id in hardware_profile.servo_ids()}
        self._current_time_ms = 0
        self._execution_count = 0

    @property
    def current_time_ms(self) -> int:
        """Return the current simulated clock time in milliseconds."""
        return self._current_time_ms

    @property
    def scenario(self) -> RobotHeadSimulationScenario:
        """Return the active deterministic simulation scenario."""
        return self._scenario

    def status(self) -> RobotHeadDriverStatus:
        """Return the current structured simulator status."""
        responsive_ids = self._responsive_servo_ids()
        expected_ids = self._hardware_profile.servo_ids()
        missing_ids = [servo_id for servo_id in expected_ids if servo_id not in responsive_ids]
        available = bool(responsive_ids) and not self._scenario.faults.busy
        armed = not self._scenario.faults.missing_arm
        degraded, degraded_warnings = self._degraded_warnings(responsive_ids)
        preview_fallback = (not available) or (not armed) or bool(missing_ids)
        warnings = list(self._scenario.faults.warnings)
        if self._scenario.faults.busy:
            warnings.append("Robot head simulation ownership is busy with another session.")
        if not armed:
            warnings.append("Simulation arm lease is missing or expired.")
        if missing_ids:
            warnings.append("Not all expected servo IDs are responsive in the simulation scenario.")
        warnings.extend(degraded_warnings)

        owner = self._scenario.owner or ("another-session" if self._scenario.faults.busy else "simulation-session")
        details = {
            "scenario_name": self._scenario.name,
            "simulated_time_ms": self._current_time_ms,
            "expected_servo_ids": expected_ids,
            "responsive_servo_ids": responsive_ids,
            "missing_servo_ids": missing_ids,
            "positions": {servo_id: self._positions[servo_id] for servo_id in expected_ids},
            "voltages": {
                servo_id: self._telemetry_value("voltage", servo_id) for servo_id in expected_ids
            },
            "temperatures": {
                servo_id: self._telemetry_value("temperature", servo_id) for servo_id in expected_ids
            },
            "currents": {
                servo_id: self._telemetry_value("current", servo_id) for servo_id in expected_ids
            },
            "status_bytes": {
                servo_id: self._telemetry_value("status", servo_id) for servo_id in expected_ids
            },
            "moving_flags": {servo_id: self._moving_flags[servo_id] for servo_id in expected_ids},
        }
        return RobotHeadDriverStatus(
            mode="simulation",
            available=available,
            armed=armed,
            owner=owner,
            degraded=degraded or self._scenario.faults.degraded,
            preview_fallback=preview_fallback,
            warnings=warnings,
            details=details,
        )

    def execute_plan(self, plan: RobotHeadExecutionPlan) -> RobotHeadSimulationOutcome:
        """Execute one validated plan in-memory.

        Args:
            plan: Validated controller plan.

        Returns:
            The structured outcome for the driver.
        """
        status_before = self.status()
        warnings = list(plan.warnings)

        if self._scenario.faults.busy:
            warnings.extend(status_before.warnings)
            return RobotHeadSimulationOutcome(
                preview_only=True,
                warnings=self._dedupe_execution_warnings(warnings),
                status=status_before,
                summary=(
                    f"Simulation driver kept {plan.command.command_type}:{plan.resolved_name} stationary "
                    "because ownership is busy."
                ),
                metadata={
                    "steps": [],
                    "skipped_reason": "busy",
                    "simulated_time_ms": self._current_time_ms,
                    "scenario_name": self._scenario.name,
                },
            )

        if not status_before.available or not status_before.armed:
            warnings.extend(status_before.warnings)
            return RobotHeadSimulationOutcome(
                preview_only=True,
                warnings=self._dedupe_execution_warnings(warnings),
                status=status_before,
                summary=(
                    f"Simulation driver kept {plan.command.command_type}:{plan.resolved_name} stationary "
                    "because the simulated robot head is unavailable for motion."
                ),
                metadata={
                    "steps": [],
                    "skipped_reason": "unavailable",
                    "simulated_time_ms": self._current_time_ms,
                    "scenario_name": self._scenario.name,
                },
            )

        kinetics = self._resolve_kinetics(plan.preset)
        responsive_ids = set(self._responsive_servo_ids())
        stalled_ids = set(self._scenario.faults.stalled_servo_ids)
        slow_ids = set(self._scenario.faults.slow_servo_ids)
        execution_warnings: list[str] = []
        step_results: list[dict[str, Any]] = []
        preview_only = bool(plan.preview_only or status_before.preview_fallback)
        self._execution_count += 1

        for step in plan.steps:
            compiled_targets = self._compile_servo_targets(step.values)
            duration_ms = (
                kinetics.neutral_recovery_ms
                if not step.values
                else max(
                    kinetics.minimum_transition_ms,
                    kinetics.default_transition_ms,
                )
            )
            actual_elapsed_ms = duration_ms + max(step.hold_ms, 0)
            if slow_ids:
                actual_elapsed_ms += max(0, self._scenario.faults.settle_extra_ms or 120)

            readback_positions = dict(self._positions)
            moving_flags = {servo_id: 0 for servo_id in self._hardware_profile.servo_ids()}
            settled = True
            for servo_id, target_position in compiled_targets.items():
                if servo_id not in responsive_ids:
                    continue
                if servo_id in stalled_ids:
                    moving_flags[servo_id] = 1
                    settled = False
                    continue
                readback_positions[servo_id] = target_position

            max_delta = max(
                (
                    abs(readback_positions[servo_id] - target_position)
                    for servo_id, target_position in compiled_targets.items()
                ),
                default=0,
            )
            if not settled:
                execution_warnings.append(
                    f"Step '{step.label}' did not settle in the simulation scenario "
                    f"(max delta {max_delta} counts)."
                )
            if slow_ids:
                execution_warnings.append(
                    "Simulation scenario applied extra settling latency to one or more servos."
                )

            self._positions.update(readback_positions)
            self._moving_flags = moving_flags
            self._current_time_ms += actual_elapsed_ms

            step_results.append(
                {
                    "label": step.label,
                    "target_positions": compiled_targets,
                    "readback_positions": {
                        servo_id: readback_positions[servo_id]
                        for servo_id in sorted(compiled_targets)
                    },
                    "duration_ms": duration_ms,
                    "hold_ms": step.hold_ms,
                    "moving_flags": {
                        servo_id: moving_flags[servo_id] for servo_id in sorted(compiled_targets)
                    },
                    "settled": settled,
                    "max_delta": max_delta,
                    "simulated_elapsed_ms": actual_elapsed_ms,
                }
            )

        status_after = self.status()
        warnings.extend(status_after.warnings)
        warnings.extend(self._dedupe_execution_warnings(execution_warnings))

        if plan.preview_only:
            warnings.append(
                f"'{plan.resolved_name}' is preview-only on hardware, but the simulation executed it safely."
            )

        if status_after.preview_fallback and not plan.preview_only:
            warnings.append("Simulation scenario is degraded; result should be treated as preview-only.")

        summary = (
            f"Simulation driver executed {plan.command.command_type}:{plan.resolved_name}."
            if not preview_only
            else f"Simulation driver executed preview-only {plan.command.command_type}:{plan.resolved_name}."
        )

        return RobotHeadSimulationOutcome(
            preview_only=preview_only,
            warnings=self._dedupe_execution_warnings(warnings),
            status=status_after,
            summary=summary,
            metadata={
                "steps": step_results,
                "validated_plan_steps": [step.model_dump() for step in plan.steps],
                "scenario_name": self._scenario.name,
                "simulated_time_ms": self._current_time_ms,
                "execution_index": self._execution_count,
                "live_speed": kinetics.speed,
                "live_acceleration": kinetics.acceleration,
                "default_transition_ms": kinetics.default_transition_ms,
                "minimum_transition_ms": kinetics.minimum_transition_ms,
                "neutral_recovery_ms": kinetics.neutral_recovery_ms,
                "responsive_servo_ids": status_after.details["responsive_servo_ids"],
                "missing_servo_ids": status_after.details["missing_servo_ids"],
                "execution_warnings": execution_warnings,
            },
        )

    def export_state(self) -> dict[str, Any]:
        """Return the current simulator state for traces and test assertions."""
        return {
            "scenario_name": self._scenario.name,
            "simulated_time_ms": self._current_time_ms,
            "positions": {servo_id: self._positions[servo_id] for servo_id in self._hardware_profile.servo_ids()},
            "moving_flags": {
                servo_id: self._moving_flags[servo_id] for servo_id in self._hardware_profile.servo_ids()
            },
        }

    def _responsive_servo_ids(self) -> list[int]:
        """Return the responsive servo IDs for the current scenario."""
        missing = set(self._scenario.faults.missing_servo_ids)
        return [
            servo_id for servo_id in self._hardware_profile.servo_ids() if servo_id not in missing
        ]

    def _degraded_warnings(self, responsive_ids: list[int]) -> tuple[bool, list[str]]:
        """Return the degraded flag and warnings inferred from telemetry."""
        warnings: list[str] = []
        degraded = bool(self._scenario.faults.degraded)
        for servo_id in responsive_ids:
            voltage = self._telemetry_value("voltage", servo_id)
            temperature = self._telemetry_value("temperature", servo_id)
            status_byte = self._telemetry_value("status", servo_id)
            if voltage < LOW_VOLTAGE_THRESHOLD:
                degraded = True
                warnings.append(f"Servo {servo_id} voltage is degraded at {voltage}.")
            if temperature >= HIGH_TEMPERATURE_THRESHOLD:
                degraded = True
                warnings.append(f"Servo {servo_id} temperature is elevated at {temperature}C.")
            if status_byte != 0:
                degraded = True
                warnings.append(f"Servo {servo_id} reported non-zero status byte {status_byte}.")
        return degraded, self._dedupe_execution_warnings(warnings)

    def _telemetry_value(self, metric: str, servo_id: int) -> int:
        """Return one deterministic telemetry value for the current scenario."""
        metric_lookup = {
            "voltage": self._scenario.faults.voltage_by_servo,
            "temperature": self._scenario.faults.temperature_by_servo,
            "current": self._scenario.faults.current_by_servo,
            "status": self._scenario.faults.status_byte_by_servo,
        }
        defaults = {
            "voltage": DEFAULT_VOLTAGE,
            "temperature": DEFAULT_TEMPERATURE,
            "current": DEFAULT_CURRENT,
            "status": 0,
        }
        overrides = metric_lookup[metric]
        if servo_id in overrides:
            return int(overrides[servo_id])
        if metric == "current" and self._moving_flags.get(servo_id):
            return 180
        return defaults[metric]

    def _resolve_kinetics(self, preset: str) -> RobotHeadMotionPresetKinetics:
        """Resolve the deterministic transition lane for one validated preset."""
        default = RobotHeadMotionPresetKinetics(
            speed=min(self._hardware_profile.live_speed, self._hardware_profile.safe_speed_ceiling),
            acceleration=min(
                self._hardware_profile.live_acceleration,
                self._hardware_profile.safe_acceleration_ceiling,
            ),
            default_transition_ms=self._hardware_profile.default_transition_ms,
            minimum_transition_ms=self._hardware_profile.minimum_transition_ms,
            neutral_recovery_ms=self._hardware_profile.neutral_recovery_ms,
        )
        preset_overrides = {
            "conversation_safe": default,
            "conversation_readable": RobotHeadMotionPresetKinetics(
                speed=min(100, self._hardware_profile.safe_speed_ceiling),
                acceleration=min(32, self._hardware_profile.safe_acceleration_ceiling),
                default_transition_ms=max(default.default_transition_ms, 220),
                minimum_transition_ms=max(default.minimum_transition_ms, 120),
                neutral_recovery_ms=max(default.neutral_recovery_ms, 240),
            ),
            "preview_safe": default,
            "operator_proof_safe": RobotHeadMotionPresetKinetics(
                speed=min(100, self._hardware_profile.safe_speed_ceiling),
                acceleration=min(32, self._hardware_profile.safe_acceleration_ceiling),
                default_transition_ms=max(default.default_transition_ms, 220),
                minimum_transition_ms=max(default.minimum_transition_ms, 120),
                neutral_recovery_ms=max(default.neutral_recovery_ms, 260),
            ),
            "operator_neck_safe": RobotHeadMotionPresetKinetics(
                speed=min(80, self._hardware_profile.safe_speed_ceiling),
                acceleration=min(20, self._hardware_profile.safe_acceleration_ceiling),
                default_transition_ms=max(default.default_transition_ms, 280),
                minimum_transition_ms=max(default.minimum_transition_ms, 160),
                neutral_recovery_ms=max(default.neutral_recovery_ms, 320),
            ),
            "operator_expressive_v8": RobotHeadMotionPresetKinetics(
                speed=min(90, self._hardware_profile.safe_speed_ceiling),
                acceleration=min(24, self._hardware_profile.safe_acceleration_ceiling),
                default_transition_ms=max(default.default_transition_ms, 240),
                minimum_transition_ms=max(default.minimum_transition_ms, 120),
                neutral_recovery_ms=max(default.neutral_recovery_ms, 280),
            ),
        }
        return preset_overrides.get(preset, default)

    def _compile_servo_targets(self, values: dict[str, float]) -> dict[int, int]:
        """Compile semantic unit values into raw servo targets."""
        servo_targets = self._hardware_profile.neutral_positions()
        for unit_name, unit_value in values.items():
            if unit_name == "head_turn":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="head_yaw",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
            elif unit_name == "eye_yaw":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="eye_yaw",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
            elif unit_name == "eye_pitch":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="eye_pitch",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
            elif unit_name == "left_lids":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="upper_lid_left",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="lower_lid_left",
                    semantic_value=unit_value,
                    positive_raw=False,
                )
            elif unit_name == "right_lids":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="upper_lid_right",
                    semantic_value=unit_value,
                    positive_raw=False,
                )
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="lower_lid_right",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
            elif unit_name == "left_brow":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="brow_left",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
            elif unit_name == "right_brow":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="brow_right",
                    semantic_value=unit_value,
                    positive_raw=False,
                )
            elif unit_name == "neck_pitch":
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="head_pitch_pair_a",
                    semantic_value=unit_value,
                    positive_raw=True,
                )
                self._apply_signed_joint_target(
                    servo_targets,
                    joint_name="head_pitch_pair_b",
                    semantic_value=unit_value,
                    positive_raw=False,
                )
            elif unit_name == "neck_tilt":
                self._apply_neck_tilt_target(servo_targets, unit_value)
            else:
                raise ValueError(f"Unsupported simulation semantic unit: {unit_name}")
        return servo_targets

    def _apply_signed_joint_target(
        self,
        servo_targets: dict[int, int],
        *,
        joint_name: str,
        semantic_value: float,
        positive_raw: bool,
    ) -> None:
        """Apply one signed semantic unit onto its physical joint."""
        joint = self._hardware_profile.get_joint(joint_name)
        target = self._signed_target(
            neutral=joint.neutral,
            raw_min=joint.raw_min,
            raw_max=joint.raw_max,
            semantic_value=semantic_value,
            positive_raw=positive_raw,
        )
        for servo_id in joint.servo_ids:
            servo_targets[servo_id] = target

    def _apply_neck_tilt_target(self, servo_targets: dict[int, int], semantic_value: float) -> None:
        """Apply the family-specific neck tilt rule."""
        joint_a = self._hardware_profile.get_joint("head_pitch_pair_a")
        joint_b = self._hardware_profile.get_joint("head_pitch_pair_b")
        if semantic_value >= 0:
            right_raw = self._hardware_profile.neck_tilt_right_raw or joint_a.raw_max
            right_span = max(0, right_raw - joint_a.neutral)
            servo_targets[joint_a.servo_ids[0]] = self._signed_target(
                neutral=joint_a.neutral,
                raw_min=joint_a.raw_min,
                raw_max=joint_a.neutral + right_span,
                semantic_value=semantic_value,
                positive_raw=True,
            )
            servo_targets[joint_b.servo_ids[0]] = joint_b.neutral
        else:
            magnitude = abs(float(semantic_value))
            left_raw = self._hardware_profile.neck_tilt_left_raw or joint_b.raw_min
            left_span = max(0, joint_b.neutral - left_raw)
            servo_targets[joint_a.servo_ids[0]] = joint_a.neutral
            servo_targets[joint_b.servo_ids[0]] = joint_b.neutral - round(magnitude * left_span)

    @staticmethod
    def _signed_target(
        *,
        neutral: int,
        raw_min: int,
        raw_max: int,
        semantic_value: float,
        positive_raw: bool,
    ) -> int:
        """Translate one semantic value into a raw servo target."""
        value = float(semantic_value)
        if value >= 0:
            span = raw_max - neutral if positive_raw else neutral - raw_min
            direction = 1 if positive_raw else -1
        else:
            value = abs(value)
            span = neutral - raw_min if positive_raw else raw_max - neutral
            direction = -1 if positive_raw else 1
        target = neutral + round(direction * value * span)
        return max(raw_min, min(raw_max, target))

    @staticmethod
    def _dedupe_execution_warnings(warnings: list[str]) -> list[str]:
        """Return warnings with duplicates removed while preserving order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for warning in warnings:
            if warning in seen:
                continue
            seen.add(warning)
            deduped.append(warning)
        return deduped


class SimulationDriver(RobotHeadDriver):
    """Robot-head driver that executes validated plans through the simulator."""

    def __init__(
        self,
        *,
        config: RobotHeadSimulationConfig | None = None,
        scenario: RobotHeadSimulationScenario | None = None,
    ):
        """Initialize the simulation driver.

        Args:
            config: Optional simulation configuration.
            scenario: Optional prebuilt scenario override.
        """
        self._config = config or RobotHeadSimulationConfig()
        self._hardware_profile = load_robot_head_live_hardware_profile(
            self._config.hardware_profile_path
        )
        resolved_scenario = scenario or load_robot_head_simulation_scenario(self._config.scenario_path)
        self._simulator = RobotHeadSimulator(
            hardware_profile=self._hardware_profile,
            scenario=resolved_scenario,
        )
        self._trace_dir = self._config.trace_dir or DEFAULT_TRACE_DIR
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_counter = 0

    @property
    def mode_name(self) -> str:
        """Return the driver mode name."""
        return "simulation"

    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Execute one validated plan in the deterministic simulator."""
        outcome = self._simulator.execute_plan(plan)
        if self._config.realtime:
            step_elapsed_ms = sum(
                int(step.get("simulated_elapsed_ms", 0)) for step in outcome.metadata.get("steps", [])
            )
            if step_elapsed_ms > 0:
                await asyncio.sleep(step_elapsed_ms / 1000.0)

        self._trace_counter += 1
        trace_path = self._trace_dir / (
            f"trace-{self._trace_counter:04d}-{plan.command.command_type}.json"
        )
        trace_payload = {
            "sequence": self._trace_counter,
            "driver": self.mode_name,
            "catalog_version": catalog.version,
            "command": plan.command.model_dump(),
            "validated_plan": plan.model_dump(),
            "resolved_name": plan.resolved_name,
            "preset": plan.preset,
            "preview_only": outcome.preview_only,
            "warnings": outcome.warnings,
            "summary": outcome.summary,
            "status": outcome.status.model_dump(),
            "metadata": outcome.metadata,
            "state": self._simulator.export_state(),
            "scenario": self._simulator.scenario.model_dump(),
            "realtime": self._config.realtime,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return RobotHeadExecutionResult(
            accepted=True,
            command_type=plan.command.command_type,
            resolved_name=plan.resolved_name,
            driver=self.mode_name,
            preview_only=outcome.preview_only,
            preset=plan.preset,
            warnings=outcome.warnings,
            status=outcome.status,
            trace_path=str(trace_path),
            summary=outcome.summary,
            metadata=outcome.metadata,
        )

    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Return the current simulator status."""
        return self._simulator.status()

    async def close(self):
        """Release simulation resources."""
        return None

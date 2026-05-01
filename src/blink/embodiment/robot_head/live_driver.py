"""Live serial driver for Blink's robot-head embodiment layer."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from blink.embodiment.robot_head.driver import RobotHeadDriver
from blink.embodiment.robot_head.live_hardware import (
    detect_default_robot_head_port,
    detect_robot_head_serial_ports,
    load_robot_head_live_hardware_profile,
)
from blink.embodiment.robot_head.models import (
    RobotHeadDriverStatus,
    RobotHeadExecutionPlan,
    RobotHeadExecutionResult,
)
from blink.embodiment.robot_head.serial_protocol import (
    ADDRESS_PRESENT_CURRENT,
    ADDRESS_PRESENT_MOVING,
    ADDRESS_PRESENT_POSITION,
    ADDRESS_PRESENT_STATUS,
    ADDRESS_PRESENT_TEMPERATURE,
    ADDRESS_PRESENT_VOLTAGE,
    FeetechTransportError,
    LiveSerialProtocolTransport,
    SerialConnectionProtocol,
    unpack_u16_le,
)

if TYPE_CHECKING:
    from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog


@dataclass(frozen=True)
class RobotHeadLiveDriverConfig:
    """Configuration for Blink's live robot-head driver."""

    hardware_profile_path: Optional[str] = None
    port: Optional[str] = None
    baud_rate: Optional[int] = None
    timeout_seconds: Optional[float] = None
    arm_enabled: bool = False
    arm_ttl_seconds: int = 300
    arm_path: Optional[Path] = None
    lock_path: Optional[Path] = None


@dataclass(frozen=True)
class RobotHeadMotionPresetKinetics:
    """Driver-side kinetics lane for one semantic preset."""

    speed: int
    acceleration: int
    default_transition_ms: int
    minimum_transition_ms: int
    neutral_recovery_ms: int


class LiveDriver(RobotHeadDriver):
    """Real serial driver for the connected 11-servo Feetech head."""

    def __init__(
        self,
        *,
        config: RobotHeadLiveDriverConfig | None = None,
        preview_driver: RobotHeadDriver | None = None,
        connection_factory: Callable[[], SerialConnectionProtocol] | None = None,
    ):
        """Initialize the live driver.

        Args:
            config: Live driver configuration.
            preview_driver: Preview-style fallback driver for unsafe motions.
            connection_factory: Optional fake serial connection factory for tests.
        """
        self._config = config or RobotHeadLiveDriverConfig()
        self._hardware_profile = load_robot_head_live_hardware_profile(
            self._config.hardware_profile_path
        )
        self._preview_driver = preview_driver
        self._connection_factory = connection_factory
        self._lock_handle = None
        self._transport: LiveSerialProtocolTransport | None = None
        self._startup_neutral_completed = False
        self._motion_count = 0

        runtime_dir = Path.cwd() / "runtime" / "robot_head"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._arm_path = self._config.arm_path or (runtime_dir / "live_motion_arm.json")
        self._lock_path = self._config.lock_path or (runtime_dir / "live_serial.lock")

    @property
    def mode_name(self) -> str:
        """Return the driver mode name."""
        return "live"

    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Execute a validated plan on live hardware or fall back safely."""
        status = await asyncio.to_thread(self._collect_status_sync, False)
        live_warnings = list(status.warnings)

        if plan.preview_only:
            live_warnings.append(
                f"'{plan.resolved_name}' remains preview-only in Blink's live driver."
            )
            return await self._preview_fallback(plan, catalog=catalog, status=status, warnings=live_warnings)

        if not status.available or not status.armed or status.preview_fallback:
            if not status.armed:
                live_warnings.append(
                    "Live motion is not armed. Start blink-local-voice with --robot-head-live-arm "
                    "or set BLINK_LOCAL_ROBOT_HEAD_ARM=1."
                )
            return await self._preview_fallback(plan, catalog=catalog, status=status, warnings=live_warnings)

        try:
            metadata = await asyncio.to_thread(self._execute_live_plan_sync, plan)
            refreshed_status = await asyncio.to_thread(self._collect_status_sync, True)
        except Exception as exc:
            fallback_warnings = live_warnings + [f"Live execution failed: {exc}"]
            status = await asyncio.to_thread(self._collect_status_sync, False)
            return await self._preview_fallback(
                plan,
                catalog=catalog,
                status=status,
                warnings=fallback_warnings,
            )

        return RobotHeadExecutionResult(
            accepted=True,
            command_type=plan.command.command_type,
            resolved_name=plan.resolved_name,
            driver=self.mode_name,
            preview_only=False,
            preset=plan.preset,
            warnings=list(plan.warnings) + live_warnings + metadata.get("execution_warnings", []),
            status=refreshed_status,
            summary=f"Live driver executed {plan.command.command_type}:{plan.resolved_name}.",
            metadata=metadata,
        )

    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Probe live hardware status without moving the head."""
        return await asyncio.to_thread(self._collect_status_sync, False)

    async def close(self):
        """Write neutral if possible, then release serial resources."""
        await asyncio.to_thread(self._close_sync)

    async def _preview_fallback(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
        status: RobotHeadDriverStatus,
        warnings: list[str],
    ) -> RobotHeadExecutionResult:
        """Execute a plan through the preview fallback path."""
        preview_driver = self._preview_driver
        if preview_driver is None:
            from blink.embodiment.robot_head.drivers import PreviewDriver

            preview_driver = PreviewDriver()
            self._preview_driver = preview_driver

        preview_result = await preview_driver.execute_plan(plan, catalog=catalog)
        preview_result.driver = self.mode_name
        preview_result.preview_only = True
        preview_result.status = status
        preview_result.warnings = list(plan.warnings) + list(warnings) + list(preview_result.warnings)
        preview_result.summary = (
            f"{preview_result.summary} Live driver used preview fallback instead of hardware motion."
        )
        return preview_result

    def _collect_status_sync(self, keep_lock: bool) -> RobotHeadDriverStatus:
        """Collect structured driver status with a read-only serial probe."""
        warnings: list[str] = []
        details: dict[str, Any] = {
            "candidate_ports": detect_robot_head_serial_ports(),
            "hardware_profile_path": self._config.hardware_profile_path
            or str(self._hardware_profile.profile_name),
            "expected_servo_ids": self._hardware_profile.servo_ids(),
            "arm_path": str(self._arm_path),
            "lock_path": str(self._lock_path),
        }

        port = detect_default_robot_head_port(
            port_override=self._config.port,
            hardware_profile=self._hardware_profile,
        )
        if not port:
            warnings.append("No unique robot-head serial port was detected.")
            return RobotHeadDriverStatus(
                mode=self.mode_name,
                available=False,
                armed=False,
                preview_fallback=True,
                warnings=warnings,
                details=details,
            )

        details["port"] = port
        details["baud_rate"] = self._resolved_baud_rate()
        details["timeout_seconds"] = self._resolved_timeout_seconds()

        lease = self._current_arm_lease()
        if self._config.arm_enabled:
            lease = self._write_arm_lease()
        details["arm_lease"] = lease
        armed = bool(lease.get("valid", False))
        if not armed:
            warnings.extend(lease.get("warnings", []))

        local_transport: LiveSerialProtocolTransport | None = None
        try:
            with self._ownership(keep_lock=keep_lock):
                if self._transport is not None:
                    transport = self._transport
                else:
                    local_transport = self._build_transport(port)
                    transport = local_transport
                responsive_ids = self._probe_servo_ids(transport)
                details["responsive_servo_ids"] = responsive_ids
                details["missing_servo_ids"] = [
                    servo_id
                    for servo_id in self._hardware_profile.servo_ids()
                    if servo_id not in responsive_ids
                ]
                if responsive_ids:
                    details.update(self._read_health_snapshot(transport, responsive_ids))
                if len(responsive_ids) != len(self._hardware_profile.servo_ids()):
                    warnings.append("Not all expected servo IDs responded on the live bus.")
                available = bool(responsive_ids)
                preview_fallback = not available or not armed or len(responsive_ids) != len(
                    self._hardware_profile.servo_ids()
                )
                degraded = len(responsive_ids) != len(self._hardware_profile.servo_ids())
                owner = f"pid:{os.getpid()}"
        except RuntimeError as exc:
            owner_info = self._read_lock_owner()
            if owner_info.get("owner"):
                details["lock_owner"] = owner_info
            warnings.append(str(exc))
            return RobotHeadDriverStatus(
                mode=self.mode_name,
                available=False,
                armed=armed,
                owner=owner_info.get("owner"),
                degraded=False,
                preview_fallback=True,
                warnings=warnings,
                details=details,
            )
        except Exception as exc:
            warnings.append(f"Serial probe failed: {exc}")
            return RobotHeadDriverStatus(
                mode=self.mode_name,
                available=False,
                armed=armed,
                degraded=True,
                preview_fallback=True,
                warnings=warnings,
                details=details,
            )
        finally:
            if local_transport is not None:
                local_transport.close()
            if not keep_lock and self._transport is None:
                details.setdefault("lock_owner", self._read_lock_owner())

        return RobotHeadDriverStatus(
            mode=self.mode_name,
            available=available,
            armed=armed,
            owner=owner,
            degraded=degraded,
            preview_fallback=preview_fallback,
            warnings=warnings,
            details=details,
        )

    def _execute_live_plan_sync(self, plan: RobotHeadExecutionPlan) -> dict[str, Any]:
        """Execute one plan synchronously on the serial transport."""
        port = detect_default_robot_head_port(
            port_override=self._config.port,
            hardware_profile=self._hardware_profile,
        )
        if not port:
            raise RuntimeError("No robot-head serial port is available.")

        lease = self._write_arm_lease() if self._config.arm_enabled else self._current_arm_lease()
        if not lease.get("valid", False):
            raise RuntimeError("Live motion arm lease is missing or expired.")

        with self._ownership(keep_lock=True):
            transport = self._ensure_transport(port)
            responsive_ids = self._probe_servo_ids(transport)
            expected_ids = self._hardware_profile.servo_ids()
            if responsive_ids != expected_ids:
                raise RuntimeError(
                    f"Servo presence mismatch: expected {expected_ids}, got {responsive_ids}."
                )

            if not self._startup_neutral_completed:
                self._write_neutral_sync(transport, reason="startup")
                self._startup_neutral_completed = True

            transport.set_torque(expected_ids, enabled=True)
            kinetics = self._resolve_kinetics(plan.preset)
            transport.sync_write_start_acceleration(
                [(servo_id, kinetics.acceleration) for servo_id in expected_ids]
            )

            step_results: list[dict[str, Any]] = []
            execution_warnings: list[str] = []
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
                payloads = sorted(compiled_targets.items())
                transport.sync_write_target_positions(
                    payloads,
                    duration_ms=duration_ms,
                    speed=kinetics.speed,
                )
                readback_positions, moving_flags, settled, max_step_delta = self._wait_for_targets_sync(
                    transport,
                    target_positions=compiled_targets,
                    hold_ms=max(step.hold_ms, 0),
                    duration_ms=duration_ms,
                )
                retry_metadata: dict[str, Any] = {}
                if not settled and not step.values:
                    retry_duration_ms = max(duration_ms, kinetics.neutral_recovery_ms + 160)
                    transport.sync_write_target_positions(
                        payloads,
                        duration_ms=retry_duration_ms,
                        speed=kinetics.speed,
                    )
                    (
                        readback_positions,
                        moving_flags,
                        settled,
                        max_step_delta,
                    ) = self._wait_for_targets_sync(
                        transport,
                        target_positions=compiled_targets,
                        hold_ms=max(step.hold_ms, 0),
                        duration_ms=retry_duration_ms,
                    )
                    retry_metadata = {
                        "neutral_retry_attempted": True,
                        "neutral_retry_duration_ms": retry_duration_ms,
                    }
                if not settled:
                    execution_warnings.append(
                        f"Step '{step.label}' did not settle within the live wait window "
                        f"(max delta {max_step_delta} counts)."
                    )
                step_results.append(
                    {
                        "label": step.label,
                        "target_positions": compiled_targets,
                        "readback_positions": readback_positions,
                        "duration_ms": duration_ms,
                        "hold_ms": step.hold_ms,
                        "moving_flags": moving_flags,
                        "settled": settled,
                        "max_delta": max_step_delta,
                        **retry_metadata,
                    }
                )

            self._motion_count += 1
            return {
                "steps": step_results,
                "port": port,
                "servo_ids": expected_ids,
                "live_speed": kinetics.speed,
                "live_acceleration": kinetics.acceleration,
                "default_transition_ms": kinetics.default_transition_ms,
                "minimum_transition_ms": kinetics.minimum_transition_ms,
                "neutral_recovery_ms": kinetics.neutral_recovery_ms,
                "execution_warnings": execution_warnings,
            }

    def _write_neutral_sync(
        self,
        transport: LiveSerialProtocolTransport,
        *,
        reason: str,
    ) -> None:
        """Drive all joints to neutral."""
        neutral_targets = self._hardware_profile.neutral_positions()
        servo_ids = sorted(neutral_targets)
        transport.set_torque(servo_ids, enabled=True)
        transport.sync_write_start_acceleration(
            [(servo_id, self._hardware_profile.live_acceleration) for servo_id in servo_ids]
        )
        transport.sync_write_target_positions(
            sorted(neutral_targets.items()),
            duration_ms=self._hardware_profile.neutral_recovery_ms,
            speed=self._hardware_profile.live_speed,
        )
        time.sleep(self._hardware_profile.neutral_recovery_ms / 1000.0 + 0.05)

    def _close_sync(self) -> None:
        """Close the live driver synchronously."""
        if self._transport is not None:
            try:
                if self._motion_count > 0:
                    self._write_neutral_sync(self._transport, reason="shutdown")
            except Exception:
                pass
            self._transport.close()
            self._transport = None
        self._release_lock()
        if self._config.arm_enabled and self._arm_path.exists():
            try:
                self._arm_path.unlink()
            except OSError:
                pass

    def _ensure_transport(self, port: str) -> LiveSerialProtocolTransport:
        """Return the persistent live transport."""
        if self._transport is None:
            self._transport = self._build_transport(port)
        return self._transport

    def _build_transport(self, port: str) -> LiveSerialProtocolTransport:
        """Build one live serial transport instance."""
        return LiveSerialProtocolTransport(
            port=port,
            baud_rate=self._resolved_baud_rate(),
            timeout_seconds=self._resolved_timeout_seconds(),
            connection_factory=self._connection_factory,
        )

    def _resolved_baud_rate(self) -> int:
        """Return the configured baud rate."""
        return int(self._config.baud_rate or self._hardware_profile.baud_rate)

    def _resolved_timeout_seconds(self) -> float:
        """Return the configured serial timeout."""
        return float(self._config.timeout_seconds or self._hardware_profile.timeout_seconds)

    def _resolve_kinetics(self, preset: str) -> RobotHeadMotionPresetKinetics:
        """Resolve the safe kinetics lane for one validated semantic preset."""
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

    def _probe_servo_ids(self, transport: LiveSerialProtocolTransport) -> list[int]:
        """Ping all expected servo IDs and return the responsive subset."""
        responsive: list[int] = []
        for servo_id in self._hardware_profile.servo_ids():
            try:
                transport.ping(servo_id)
            except FeetechTransportError:
                continue
            responsive.append(servo_id)
        return responsive

    def _read_health_snapshot(
        self,
        transport: LiveSerialProtocolTransport,
        servo_ids: list[int],
    ) -> dict[str, Any]:
        """Read a lightweight health snapshot from the bus."""
        position_payload = transport.sync_read(ADDRESS_PRESENT_POSITION, 2, servo_ids)
        voltage_payload = transport.sync_read(ADDRESS_PRESENT_VOLTAGE, 1, servo_ids)
        temperature_payload = transport.sync_read(ADDRESS_PRESENT_TEMPERATURE, 1, servo_ids)
        status_payload = transport.sync_read(ADDRESS_PRESENT_STATUS, 1, servo_ids)
        current_payload = transport.sync_read(ADDRESS_PRESENT_CURRENT, 2, servo_ids)
        moving_payload = transport.sync_read(ADDRESS_PRESENT_MOVING, 1, servo_ids)
        return {
            "positions": {
                servo_id: unpack_u16_le(raw_value)
                for servo_id, raw_value in position_payload.items()
            },
            "voltages": {servo_id: raw_value[0] for servo_id, raw_value in voltage_payload.items()},
            "temperatures": {
                servo_id: raw_value[0] for servo_id, raw_value in temperature_payload.items()
            },
            "status_bytes": {
                servo_id: raw_value[0] for servo_id, raw_value in status_payload.items()
            },
            "currents": {
                servo_id: unpack_u16_le(raw_value)
                for servo_id, raw_value in current_payload.items()
            },
            "moving_flags": {servo_id: raw_value[0] for servo_id, raw_value in moving_payload.items()},
        }

    def _wait_for_targets_sync(
        self,
        transport: LiveSerialProtocolTransport,
        *,
        target_positions: dict[int, int],
        hold_ms: int,
        duration_ms: int,
        tolerance_counts: int = 20,
    ) -> tuple[dict[int, int], dict[int, int], bool, int]:
        """Poll readback until the requested targets settle or the timeout expires."""
        servo_ids = sorted(target_positions)
        deadline = time.monotonic() + max(
            5.0,
            ((duration_ms + hold_ms) / 1000.0) + 2.0,
        )
        last_positions = dict(target_positions)
        last_moving = {servo_id: 1 for servo_id in servo_ids}
        last_max_delta = max(
            abs(last_positions[servo_id] - target_positions[servo_id]) for servo_id in servo_ids
        )

        while True:
            readback = transport.sync_read(ADDRESS_PRESENT_POSITION, 2, servo_ids)
            moving_payload = transport.sync_read(ADDRESS_PRESENT_MOVING, 1, servo_ids)
            last_positions = {
                servo_id: unpack_u16_le(raw_value) for servo_id, raw_value in readback.items()
            }
            last_moving = {
                servo_id: raw_value[0] for servo_id, raw_value in moving_payload.items()
            }
            deltas = {
                servo_id: abs(last_positions[servo_id] - target_positions[servo_id])
                for servo_id in servo_ids
            }
            last_max_delta = max(deltas.values(), default=0)
            settled = last_max_delta <= tolerance_counts and all(
                int(last_moving[servo_id]) == 0 for servo_id in servo_ids
            )
            if settled:
                return last_positions, last_moving, True, last_max_delta
            if time.monotonic() >= deadline:
                return last_positions, last_moving, False, last_max_delta
            time.sleep(0.1)

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
                raise ValueError(f"Unsupported live semantic unit: {unit_name}")
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
        """Apply the special neck-tilt rule."""
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
        """Translate one semantic value into a raw target count."""
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

    def _current_arm_lease(self) -> dict[str, Any]:
        """Read and validate the current arm lease."""
        if not self._arm_path.exists():
            return {
                "valid": False,
                "path": str(self._arm_path),
                "warnings": ["Live motion arm lease file is missing."],
            }
        try:
            payload = json.loads(self._arm_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "valid": False,
                "path": str(self._arm_path),
                "warnings": ["Live motion arm lease file is invalid JSON."],
            }

        warnings: list[str] = []
        expires_at = payload.get("expires_at")
        if not expires_at:
            warnings.append("Live motion arm lease is missing expires_at.")
        else:
            expiry = datetime.fromisoformat(expires_at)
            if expiry <= datetime.now(UTC):
                warnings.append("Live motion arm lease has expired.")

        port = detect_default_robot_head_port(
            port_override=self._config.port,
            hardware_profile=self._hardware_profile,
        )
        if payload.get("port") != port:
            warnings.append("Live motion arm lease does not match the active serial port.")
        if int(payload.get("baud_rate", 0) or 0) != self._resolved_baud_rate():
            warnings.append("Live motion arm lease does not match the active baud rate.")

        payload["valid"] = len(warnings) == 0 and bool(payload.get("armed", False))
        payload["path"] = str(self._arm_path)
        payload["warnings"] = warnings
        return payload

    def _write_arm_lease(self) -> dict[str, Any]:
        """Create or refresh the arm lease for this process."""
        port = detect_default_robot_head_port(
            port_override=self._config.port,
            hardware_profile=self._hardware_profile,
        )
        expires_at = datetime.now(UTC) + timedelta(seconds=max(self._config.arm_ttl_seconds, 30))
        payload = {
            "schema_version": "blink_robot_head_live_arm/v1",
            "armed": True,
            "owner": f"pid:{os.getpid()}",
            "port": port,
            "baud_rate": self._resolved_baud_rate(),
            "expires_at": expires_at.isoformat(),
            "hardware_profile_path": self._config.hardware_profile_path or "packaged-default",
        }
        self._arm_path.parent.mkdir(parents=True, exist_ok=True)
        self._arm_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["valid"] = True
        payload["path"] = str(self._arm_path)
        payload["warnings"] = []
        return payload

    @contextmanager
    def _ownership(self, *, keep_lock: bool):
        """Acquire exclusive Blink-side ownership of the live serial session."""
        if self._lock_handle is not None:
            yield
            return

        import fcntl

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = self._lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = self._read_lock_owner().get("owner", "unknown-session")
            lock_handle.close()
            raise RuntimeError(f"Robot head serial ownership is busy with {owner}.") from exc

        owner_payload = {
            "owner": f"pid:{os.getpid()}",
            "acquired_at": datetime.now(UTC).isoformat(),
        }
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(json.dumps(owner_payload, indent=2))
        lock_handle.flush()

        if keep_lock:
            self._lock_handle = lock_handle
            try:
                yield
            finally:
                return
        else:
            try:
                yield
            finally:
                self._unlock_handle(lock_handle)

    def _release_lock(self) -> None:
        """Release the persistent lock handle."""
        if self._lock_handle is not None:
            self._unlock_handle(self._lock_handle)
            self._lock_handle = None

    def _unlock_handle(self, handle) -> None:
        """Release one file lock handle."""
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    def _read_lock_owner(self) -> dict[str, Any]:
        """Read the current owner payload from the lock file."""
        if not self._lock_path.exists():
            return {}
        try:
            return json.loads(self._lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

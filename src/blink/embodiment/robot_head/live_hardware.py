"""Blink-owned live hardware profile for the connected robot head."""

from __future__ import annotations

import json
from glob import glob
from importlib.resources import files
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LIVE_HARDWARE_PROFILE_RESOURCE = "live_hardware_profile.json"


class RobotHeadHardwareJoint(BaseModel):
    """Physical joint calibration record for one servo-backed joint."""

    model_config = ConfigDict(frozen=True)

    joint_name: str
    servo_ids: list[int] = Field(default_factory=list)
    neutral: int
    raw_min: int
    raw_max: int
    positive_direction: str

    @property
    def negative_span(self) -> int:
        """Return the raw distance from neutral to minimum."""
        return self.neutral - self.raw_min

    @property
    def positive_span(self) -> int:
        """Return the raw distance from neutral to maximum."""
        return self.raw_max - self.neutral


class RobotHeadLiveHardwareProfile(BaseModel):
    """Blink-owned live hardware profile snapshot."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    profile_name: str
    source_notes: list[str] = Field(default_factory=list)
    port_hint: Optional[str] = None
    baud_rate: int = 1000000
    timeout_seconds: float = 0.2
    default_transition_ms: int = 160
    minimum_transition_ms: int = 80
    neutral_recovery_ms: int = 220
    live_speed: int = 100
    live_acceleration: int = 32
    safe_speed_ceiling: int = 120
    safe_acceleration_ceiling: int = 40
    neck_tilt_left_raw: Optional[int] = None
    neck_tilt_right_raw: Optional[int] = None
    joints: list[RobotHeadHardwareJoint] = Field(default_factory=list)

    def get_joint(self, name: str) -> RobotHeadHardwareJoint:
        """Return one hardware joint by name."""
        for joint in self.joints:
            if joint.joint_name == name:
                return joint
        raise ValueError(f"Unsupported robot-head hardware joint: {name}")

    def servo_ids(self) -> list[int]:
        """Return all servo IDs in sorted order."""
        return sorted({servo_id for joint in self.joints for servo_id in joint.servo_ids})

    def neutral_positions(self) -> dict[int, int]:
        """Return servo neutral positions keyed by servo ID."""
        positions: dict[int, int] = {}
        for joint in self.joints:
            for servo_id in joint.servo_ids:
                positions[servo_id] = joint.neutral
        return positions


def default_live_hardware_profile_path() -> Path:
    """Return the packaged default live hardware profile path."""
    return Path(files("blink.embodiment.robot_head").joinpath(DEFAULT_LIVE_HARDWARE_PROFILE_RESOURCE))


def load_robot_head_live_hardware_profile(
    path: str | Path | None = None,
) -> RobotHeadLiveHardwareProfile:
    """Load the Blink-owned live hardware profile.

    Args:
        path: Optional override path to a JSON profile.

    Returns:
        The parsed live hardware profile.
    """
    resolved_path = Path(path) if path else default_live_hardware_profile_path()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return RobotHeadLiveHardwareProfile.model_validate(payload)


def detect_robot_head_serial_ports() -> list[str]:
    """Return likely robot-head serial ports on macOS."""
    ports = sorted(glob("/dev/cu.*"))
    return [
        port
        for port in ports
        if not port.endswith("Bluetooth-Incoming-Port") and "debug-console" not in port
    ]


def detect_default_robot_head_port(
    *,
    port_override: str | None = None,
    hardware_profile: RobotHeadLiveHardwareProfile | None = None,
) -> str | None:
    """Resolve the default serial port for the connected head."""
    if port_override not in (None, ""):
        return str(port_override)
    ports = detect_robot_head_serial_ports()
    if hardware_profile and hardware_profile.port_hint in ports:
        return hardware_profile.port_hint
    if len(ports) == 1:
        return ports[0]
    return None

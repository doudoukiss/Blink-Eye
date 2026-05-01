import json

import pytest

from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import PreviewDriver
from blink.embodiment.robot_head.live_driver import LiveDriver, RobotHeadLiveDriverConfig
from blink.embodiment.robot_head.live_hardware import load_robot_head_live_hardware_profile
from blink.embodiment.robot_head.serial_protocol import (
    ADDRESS_PRESENT_POSITION,
    ADDRESS_PRESENT_STATUS,
    ADDRESS_PRESENT_TEMPERATURE,
    ADDRESS_PRESENT_VOLTAGE,
    ADDRESS_START_ACCELERATION,
    ADDRESS_TARGET_POSITION,
    ADDRESS_TORQUE_SWITCH,
    BROADCAST_ID,
    FeetechInstruction,
    build_target_position_payload,
    decode_packet,
    encode_status_packet,
    pack_u16_le,
)


class FakeSerialConnection:
    def __init__(self, *, port: str, baudrate: int, timeout: float):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._pending = bytearray()
        self._closed = False
        self._profile = load_robot_head_live_hardware_profile()
        self._registers: dict[int, bytearray] = {}
        for joint in self._profile.joints:
            servo_id = joint.servo_ids[0]
            registers = bytearray(256)
            registers[ADDRESS_PRESENT_POSITION : ADDRESS_PRESENT_POSITION + 2] = pack_u16_le(
                joint.neutral
            )
            registers[ADDRESS_PRESENT_VOLTAGE] = 120
            registers[ADDRESS_PRESENT_TEMPERATURE] = 25
            registers[ADDRESS_PRESENT_STATUS] = 0
            registers[ADDRESS_TORQUE_SWITCH] = 1
            registers[ADDRESS_START_ACCELERATION] = 32
            registers[ADDRESS_TARGET_POSITION : ADDRESS_TARGET_POSITION + 6] = (
                build_target_position_payload(joint.neutral, duration_ms=0, speed=100)
            )
            self._registers[servo_id] = registers

    def write(self, payload: bytes) -> int:
        packet = decode_packet(payload)
        self._pending.extend(self._handle_packet(packet))
        return len(payload)

    def read(self, size: int) -> bytes:
        if not self._pending:
            return b""
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def close(self) -> None:
        self._closed = True

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def reset_output_buffer(self) -> None:
        return None

    def _handle_packet(self, packet) -> bytes:
        if packet.instruction == FeetechInstruction.PING:
            if packet.servo_id == BROADCAST_ID:
                return b""
            return encode_status_packet(packet.servo_id)

        if packet.instruction == FeetechInstruction.READ:
            address = packet.parameters[0]
            length = packet.parameters[1]
            payload = bytes(self._registers[packet.servo_id][address : address + length])
            return encode_status_packet(packet.servo_id, parameters=payload)

        if packet.instruction == FeetechInstruction.WRITE:
            address = packet.parameters[0]
            payload = bytes(packet.parameters[1:])
            self._apply_write(packet.servo_id, address, payload)
            if packet.servo_id == BROADCAST_ID:
                return b""
            return encode_status_packet(packet.servo_id)

        if packet.instruction == FeetechInstruction.SYNC_WRITE:
            address = packet.parameters[0]
            data_length = packet.parameters[1]
            cursor = 2
            while cursor < len(packet.parameters):
                servo_id = packet.parameters[cursor]
                payload = bytes(packet.parameters[cursor + 1 : cursor + 1 + data_length])
                self._apply_write(servo_id, address, payload)
                cursor += data_length + 1
            return b""

        if packet.instruction == FeetechInstruction.SYNC_READ:
            address = packet.parameters[0]
            data_length = packet.parameters[1]
            servo_ids = list(packet.parameters[2:])
            return b"".join(
                encode_status_packet(
                    servo_id,
                    parameters=bytes(self._registers[servo_id][address : address + data_length]),
                )
                for servo_id in servo_ids
            )

        raise AssertionError(f"Unsupported instruction: {packet.instruction}")

    def _apply_write(self, servo_id: int, address: int, payload: bytes) -> None:
        self._registers[servo_id][address : address + len(payload)] = payload
        if address == ADDRESS_TARGET_POSITION:
            self._registers[servo_id][ADDRESS_PRESENT_POSITION : ADDRESS_PRESENT_POSITION + 2] = (
                payload[:2]
            )
        if address == ADDRESS_START_ACCELERATION:
            self._registers[servo_id][ADDRESS_START_ACCELERATION] = payload[0]
        if address == ADDRESS_TORQUE_SWITCH:
            self._registers[servo_id][ADDRESS_TORQUE_SWITCH] = payload[0]


def _fake_connection_factory():
    connection = FakeSerialConnection(port="/dev/cu.fake-robot-head", baudrate=1000000, timeout=0.2)

    def factory():
        return connection

    return factory


@pytest.mark.asyncio
async def test_live_driver_status_reports_hardware_presence_without_motion(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = LiveDriver(
        config=RobotHeadLiveDriverConfig(
            port="/dev/cu.fake-robot-head",
            arm_path=tmp_path / "arm.json",
            lock_path=tmp_path / "lock.json",
        ),
        preview_driver=PreviewDriver(trace_dir=tmp_path / "preview"),
        connection_factory=_fake_connection_factory(),
    )

    status = await driver.status(catalog=catalog)

    assert status.available is True
    assert status.armed is False
    assert status.preview_fallback is True
    assert len(status.details["responsive_servo_ids"]) == 11
    assert status.details["positions"][1] == 2096

    await driver.close()


@pytest.mark.asyncio
async def test_live_driver_executes_live_motion_when_armed(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = LiveDriver(
        config=RobotHeadLiveDriverConfig(
            port="/dev/cu.fake-robot-head",
            arm_enabled=True,
            arm_path=tmp_path / "arm.json",
            lock_path=tmp_path / "lock.json",
        ),
        preview_driver=PreviewDriver(trace_dir=tmp_path / "preview"),
        connection_factory=_fake_connection_factory(),
    )
    controller = RobotHeadController(catalog=catalog, driver=driver)

    result = await controller.run_motif("look_left", source="tool")

    assert result.accepted is True
    assert result.preview_only is False
    assert result.driver == "live"
    assert result.metadata["default_transition_ms"] == 220
    assert result.metadata["steps"][0]["target_positions"][1] < 2096
    assert result.metadata["steps"][0]["readback_positions"][1] < 2096

    await controller.close()
    assert (tmp_path / "arm.json").exists() is False


@pytest.mark.asyncio
async def test_live_driver_uses_preview_fallback_for_preview_only_motion(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = LiveDriver(
        config=RobotHeadLiveDriverConfig(
            port="/dev/cu.fake-robot-head",
            arm_enabled=True,
            arm_path=tmp_path / "arm.json",
            lock_path=tmp_path / "lock.json",
        ),
        preview_driver=PreviewDriver(trace_dir=tmp_path / "preview"),
        connection_factory=_fake_connection_factory(),
    )
    controller = RobotHeadController(catalog=catalog, driver=driver)

    result = await controller.run_motif("curious_tilt", source="policy")

    assert result.accepted is True
    assert result.preview_only is True
    assert "preview-only" in " ".join(result.warnings)

    trace_path = result.trace_path
    assert trace_path is not None
    trace_payload = json.loads((tmp_path / "preview" / "trace-0001-run_motif.json").read_text())
    assert trace_payload["resolved_name"] == "curious_tilt"

    await controller.close()


@pytest.mark.asyncio
async def test_live_driver_uses_family_and_show_preset_kinetics(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = LiveDriver(
        config=RobotHeadLiveDriverConfig(
            port="/dev/cu.fake-robot-head",
            arm_enabled=True,
            arm_path=tmp_path / "arm.json",
            lock_path=tmp_path / "lock.json",
        ),
        preview_driver=PreviewDriver(trace_dir=tmp_path / "preview"),
        connection_factory=_fake_connection_factory(),
    )
    controller = RobotHeadController(catalog=catalog, driver=driver)

    proof_result = await controller.run_motif("investor_head_turn_left_v3", source="operator")
    neck_result = await controller.run_motif("investor_neck_pitch_up_v7", source="operator")
    expressive_result = await controller.run_motif("bright_reengage", source="operator")

    assert proof_result.preview_only is False
    assert proof_result.metadata["live_speed"] == 100
    assert proof_result.metadata["live_acceleration"] == 32
    assert proof_result.metadata["default_transition_ms"] == 220

    assert neck_result.preview_only is False
    assert neck_result.metadata["live_speed"] == 80
    assert neck_result.metadata["live_acceleration"] == 20
    assert neck_result.metadata["default_transition_ms"] == 280

    assert expressive_result.preview_only is False
    assert expressive_result.metadata["live_speed"] == 90
    assert expressive_result.metadata["live_acceleration"] == 24
    assert expressive_result.metadata["default_transition_ms"] == 240

    await controller.close()


@pytest.mark.asyncio
async def test_live_driver_uses_family_specific_neck_tilt_targets(tmp_path):
    catalog = build_default_robot_head_catalog()
    driver = LiveDriver(
        config=RobotHeadLiveDriverConfig(
            port="/dev/cu.fake-robot-head",
            arm_enabled=True,
            arm_path=tmp_path / "arm.json",
            lock_path=tmp_path / "lock.json",
        ),
        preview_driver=PreviewDriver(trace_dir=tmp_path / "preview"),
        connection_factory=_fake_connection_factory(),
    )
    controller = RobotHeadController(catalog=catalog, driver=driver)

    result = await controller.run_motif("investor_neck_tilt_right_v7", source="operator")

    assert result.accepted is True
    assert result.preview_only is False
    assert result.metadata["steps"][0]["target_positions"][2] == 2205
    assert result.metadata["steps"][0]["target_positions"][3] == 2058

    await controller.close()

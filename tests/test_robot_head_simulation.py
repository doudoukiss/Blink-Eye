import json

import pytest

from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver, PreviewDriver
from blink.embodiment.robot_head.simulation import (
    RobotHeadFaultProfile,
    RobotHeadSimulationConfig,
    RobotHeadSimulationScenario,
    SimulationDriver,
)


@pytest.mark.asyncio
async def test_robot_head_simulation_driver_executes_motion_and_writes_trace(tmp_path):
    catalog = build_default_robot_head_catalog()
    controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
        ),
    )

    result = await controller.run_motif("look_left", source="tool")

    assert result.accepted is True
    assert result.driver == "simulation"
    assert result.preview_only is False
    assert result.trace_path is not None
    assert result.metadata["validated_plan_steps"][0]["values"]["head_turn"] < 0
    assert result.metadata["steps"][0]["target_positions"][1] < 2096

    trace_payload = json.loads((tmp_path / "simulation" / "trace-0001-run_motif.json").read_text())
    assert trace_payload["resolved_name"] == "look_left"
    assert trace_payload["metadata"]["steps"][0]["readback_positions"]["1"] < 2096
    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_simulation_driver_matches_controller_plan_across_backends(tmp_path):
    catalog = build_default_robot_head_catalog()

    mock_driver = MockDriver()
    mock_controller = RobotHeadController(catalog=catalog, driver=mock_driver)
    preview_controller = RobotHeadController(
        catalog=catalog,
        driver=PreviewDriver(trace_dir=tmp_path / "preview"),
    )
    simulation_controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
        ),
    )

    await mock_controller.run_motif("blink", source="tool")
    preview_result = await preview_controller.run_motif("blink", source="tool")
    simulation_result = await simulation_controller.run_motif("blink", source="tool")

    mock_plan = mock_driver.executed_plans[0]
    preview_trace = json.loads((tmp_path / "preview" / "trace-0001-run_motif.json").read_text())

    assert preview_trace["command"] == mock_plan.command.model_dump()
    assert preview_trace["steps"] == [step.model_dump() for step in mock_plan.steps]
    assert simulation_result.metadata["validated_plan_steps"] == [
        step.model_dump() for step in mock_plan.steps
    ]
    assert simulation_result.trace_path is not None

    await mock_controller.close()
    await preview_controller.close()
    await simulation_controller.close()


@pytest.mark.asyncio
async def test_robot_head_simulation_driver_compiles_known_hardware_targets(tmp_path):
    catalog = build_default_robot_head_catalog()
    controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
        ),
    )

    result = await controller.run_motif("investor_neck_tilt_right_v7", source="operator")

    assert result.accepted is True
    assert result.preview_only is False
    assert result.metadata["steps"][0]["target_positions"][2] == 2205
    assert result.metadata["steps"][0]["target_positions"][3] == 2058

    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_simulation_driver_reports_deterministic_faults(tmp_path):
    catalog = build_default_robot_head_catalog()
    scenario = RobotHeadSimulationScenario(
        name="degraded-bench",
        faults=RobotHeadFaultProfile(
            missing_servo_ids=[11],
            slow_servo_ids=[1],
            voltage_by_servo={1: 104},
            temperature_by_servo={2: 66},
            status_byte_by_servo={3: 2},
        ),
    )
    controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation"),
            scenario=scenario,
        ),
    )

    status_before = await controller.status()
    result = await controller.run_motif("blink", source="tool")

    assert status_before.accepted is True
    assert status_before.status is not None
    assert status_before.status.preview_fallback is True
    assert status_before.status.degraded is True
    assert status_before.status.details["missing_servo_ids"] == [11]

    assert result.accepted is True
    assert result.preview_only is True
    assert result.status is not None
    assert result.status.details["missing_servo_ids"] == [11]
    joined_warnings = " ".join(result.warnings)
    assert "degraded" in joined_warnings
    assert "temperature" in joined_warnings.lower()
    assert "status byte" in joined_warnings.lower()
    assert result.metadata["steps"][0]["target_positions"][5] < 2048

    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_simulation_driver_keeps_head_stationary_when_busy_or_unarmed(tmp_path):
    catalog = build_default_robot_head_catalog()
    busy_controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "busy"),
            scenario=RobotHeadSimulationScenario(
                name="busy",
                faults=RobotHeadFaultProfile(busy=True),
            ),
        ),
    )
    unarmed_controller = RobotHeadController(
        catalog=catalog,
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "unarmed"),
            scenario=RobotHeadSimulationScenario(
                name="unarmed",
                faults=RobotHeadFaultProfile(missing_arm=True),
            ),
        ),
    )

    busy_before = await busy_controller.status()
    busy_result = await busy_controller.run_motif("look_left", source="tool")
    unarmed_before = await unarmed_controller.status()
    unarmed_result = await unarmed_controller.run_motif("look_left", source="tool")

    assert busy_before.status is not None
    assert busy_before.status.details["positions"][1] == 2096
    assert busy_result.preview_only is True
    assert busy_result.metadata["skipped_reason"] == "busy"
    assert busy_result.status is not None
    assert busy_result.status.details["positions"][1] == 2096

    assert unarmed_before.status is not None
    assert unarmed_before.status.armed is False
    assert unarmed_result.preview_only is True
    assert unarmed_result.metadata["skipped_reason"] == "unavailable"
    assert unarmed_result.status is not None
    assert unarmed_result.status.details["positions"][1] == 2096

    await busy_controller.close()
    await unarmed_controller.close()

import json

import pytest

from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver, MockDriver, PreviewDriver


@pytest.mark.asyncio
async def test_robot_head_controller_runs_state_and_motif_commands():
    catalog = build_default_robot_head_catalog()
    driver = MockDriver()
    controller = RobotHeadController(catalog=catalog, driver=driver)

    state_result = await controller.set_state("thinking", source="tool")
    motif_result = await controller.run_motif("look_left", source="tool")
    status_result = await controller.status()

    assert state_result.accepted is True
    assert motif_result.accepted is True
    assert motif_result.resolved_name == "look_left"
    assert driver.executed_plans[1].steps[0].values["head_turn"] < 0
    assert status_result.metadata["supported_states"] == catalog.public_state_names()
    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_preview_driver_writes_trace_artifacts(tmp_path):
    catalog = build_default_robot_head_catalog()
    controller = RobotHeadController(
        catalog=catalog,
        driver=PreviewDriver(trace_dir=tmp_path),
    )

    result = await controller.run_motif("blink", source="tool")

    assert result.accepted is True
    assert result.preview_only is True
    assert result.trace_path is not None

    trace_payload = json.loads((tmp_path / "trace-0001-run_motif.json").read_text(encoding="utf-8"))
    assert trace_payload["resolved_name"] == "blink"
    assert trace_payload["command"]["motif"] == "blink"
    await controller.close()


@pytest.mark.asyncio
async def test_robot_head_fault_driver_can_reject_busy_or_unarmed_commands():
    catalog = build_default_robot_head_catalog()
    busy_controller = RobotHeadController(
        catalog=catalog,
        driver=FaultInjectionDriver(busy=True),
    )
    busy_result = await busy_controller.run_motif("blink", source="tool")
    assert busy_result.accepted is False
    assert "busy" in busy_result.summary.lower()
    await busy_controller.close()

    unarmed_controller = RobotHeadController(
        catalog=catalog,
        driver=FaultInjectionDriver(missing_arm=True),
    )
    unarmed_result = await unarmed_controller.run_motif("blink", source="tool")
    assert unarmed_result.accepted is False
    assert "not armed" in unarmed_result.summary.lower()
    await unarmed_controller.close()


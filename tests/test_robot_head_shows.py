import json
from pathlib import Path

import pytest

from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver
from blink.embodiment.robot_head.show import (
    RobotHeadShowRunner,
    list_robot_head_shows,
    resolve_robot_head_show,
)


def test_show_alias_resolution_and_v8_shape():
    show = resolve_robot_head_show("v8")

    assert show.name == "investor_expressive_motion_v8"
    assert show.sensitive_motion is True
    assert len(show.steps) == 12
    assert show.steps[0].target == "attentive_notice_right"
    assert show.steps[-1].target == "doubtful_side_glance"


def test_list_robot_head_shows_includes_v3_to_v8():
    names = [show.name for show in list_robot_head_shows()]

    assert names == [
        "investor_head_motion_v3",
        "investor_eye_motion_v4",
        "investor_lid_motion_v5",
        "investor_brow_motion_v6",
        "investor_neck_motion_v7",
        "investor_expressive_motion_v8",
    ]


@pytest.mark.asyncio
async def test_show_runner_writes_report_for_family_safe_show(tmp_path):
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=MockDriver(),
    )
    runner = RobotHeadShowRunner(
        controller=controller,
        report_dir=tmp_path,
    )

    try:
        report = await runner.run("v3")
    finally:
        await controller.close()

    report_path = Path(report["report_path"])
    assert report["driver_mode"] == "mock"
    assert report["show"]["name"] == "investor_head_motion_v3"
    assert len(report["steps"]) == 4
    assert report["neutral_start"]["accepted"] is True
    assert report["neutral_end"]["accepted"] is True
    assert report_path.exists() is True

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["show"]["name"] == "investor_head_motion_v3"
    assert payload["status_after"]["status"]["details"]["executed_commands"] == 6


@pytest.mark.asyncio
async def test_show_runner_blocks_sensitive_motion_without_flag(tmp_path):
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=MockDriver(),
    )
    runner = RobotHeadShowRunner(
        controller=controller,
        report_dir=tmp_path,
        allow_sensitive_motion=False,
    )

    with pytest.raises(ValueError, match="allow-sensitive-motion"):
        await runner.run("v7")

    await controller.close()

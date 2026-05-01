#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Operator CLI for Blink-owned robot-head proof shows."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from blink.cli.local_common import (
    configure_logging,
    get_local_env,
    local_env_flag,
    maybe_load_dotenv,
    resolve_int,
)
from blink.embodiment.robot_head.catalog import load_robot_head_capability_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import (
    LiveDriver,
    MockDriver,
    PreviewDriver,
    SimulationDriver,
)
from blink.embodiment.robot_head.live_driver import RobotHeadLiveDriverConfig
from blink.embodiment.robot_head.show import (
    RobotHeadShowRunner,
    list_robot_head_shows,
    resolve_robot_head_show,
)
from blink.embodiment.robot_head.simulation import RobotHeadSimulationConfig
from blink.project_identity import PROJECT_IDENTITY


@dataclass
class LocalRobotHeadShowConfig:
    """Configuration for the Blink robot-head proof-show CLI."""

    show_name: Optional[str]
    list_shows: bool = False
    robot_head_driver: str = "simulation"
    robot_head_catalog_path: Optional[str] = None
    robot_head_port: Optional[str] = None
    robot_head_baud: int = 1000000
    robot_head_hardware_profile_path: Optional[str] = None
    robot_head_live_arm: bool = False
    robot_head_arm_ttl_seconds: int = 300
    robot_head_sim_scenario_path: Optional[Path] = None
    robot_head_sim_realtime: bool = False
    allow_sensitive_motion: bool = False
    report_dir: Optional[Path] = None
    preview_trace_dir: Optional[Path] = None
    simulation_trace_dir: Optional[Path] = None
    verbose: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run Blink-owned V3-V8 robot-head proof shows for {PROJECT_IDENTITY.display_name}."
        )
    )
    parser.add_argument(
        "show",
        nargs="?",
        help="Show name or alias to run, for example v3, v4, v5, v6, v7, or v8.",
    )
    parser.add_argument(
        "--list-shows",
        action="store_true",
        help="List the supported Blink-owned robot-head proof shows and exit.",
    )
    parser.add_argument(
        "--robot-head-driver",
        choices=["mock", "preview", "simulation", "live"],
        help="Robot-head driver to use for the proof show.",
    )
    parser.add_argument(
        "--robot-head-catalog-path",
        help="Optional path to a robot-head capability catalog JSON file.",
    )
    parser.add_argument(
        "--robot-head-port",
        help="Optional serial device path for the live robot head.",
    )
    parser.add_argument(
        "--robot-head-baud",
        type=int,
        help="Optional baud rate override for the live robot head.",
    )
    parser.add_argument(
        "--robot-head-hardware-profile-path",
        help="Optional path to a live robot-head hardware profile JSON file.",
    )
    parser.add_argument(
        "--robot-head-live-arm",
        action="store_true",
        help="Explicitly arm live robot-head motion for this session.",
    )
    parser.add_argument(
        "--robot-head-arm-ttl-seconds",
        type=int,
        help="TTL in seconds for the live robot-head arm lease.",
    )
    parser.add_argument(
        "--robot-head-sim-scenario",
        help="Optional path to a robot-head simulation scenario JSON file.",
    )
    parser.add_argument(
        "--robot-head-sim-realtime",
        action="store_true",
        help="Run robot-head simulation timing in wall-clock time instead of virtual time.",
    )
    parser.add_argument(
        "--allow-sensitive-motion",
        action="store_true",
        help="Allow V7/V8 proof shows that contain neck motion.",
    )
    parser.add_argument(
        "--report-dir",
        help="Optional directory for JSON proof-show reports.",
    )
    parser.add_argument(
        "--preview-trace-dir",
        help="Optional directory for preview-fallback trace artifacts.",
    )
    parser.add_argument(
        "--robot-head-sim-trace-dir",
        help="Optional directory for robot-head simulation trace artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging while the proof show is running.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LocalRobotHeadShowConfig:
    """Resolve CLI configuration from arguments and environment variables."""
    maybe_load_dotenv()
    robot_head_baud = resolve_int(
        getattr(args, "robot_head_baud", None) or get_local_env("ROBOT_HEAD_BAUD")
    ) or 1000000
    robot_head_arm_ttl_seconds = resolve_int(
        getattr(args, "robot_head_arm_ttl_seconds", None)
        or get_local_env("ROBOT_HEAD_ARM_TTL_SECONDS")
    ) or 300
    report_dir = getattr(args, "report_dir", None) or get_local_env("ROBOT_HEAD_SHOW_REPORT_DIR")
    preview_trace_dir = getattr(args, "preview_trace_dir", None) or get_local_env(
        "ROBOT_HEAD_PREVIEW_TRACE_DIR"
    )
    robot_head_sim_scenario = getattr(args, "robot_head_sim_scenario", None) or get_local_env(
        "ROBOT_HEAD_SIM_SCENARIO"
    )
    simulation_trace_dir = getattr(args, "robot_head_sim_trace_dir", None) or get_local_env(
        "ROBOT_HEAD_SIM_TRACE_DIR"
    )
    return LocalRobotHeadShowConfig(
        show_name=getattr(args, "show", None),
        list_shows=getattr(args, "list_shows", False),
        robot_head_driver=getattr(args, "robot_head_driver", None)
        or get_local_env("ROBOT_HEAD_DRIVER", "simulation"),
        robot_head_catalog_path=(
            getattr(args, "robot_head_catalog_path", None)
            or get_local_env("ROBOT_HEAD_CATALOG_PATH")
        ),
        robot_head_port=getattr(args, "robot_head_port", None) or get_local_env("ROBOT_HEAD_PORT"),
        robot_head_baud=robot_head_baud,
        robot_head_hardware_profile_path=(
            getattr(args, "robot_head_hardware_profile_path", None)
            or get_local_env("ROBOT_HEAD_HARDWARE_PROFILE_PATH")
        ),
        robot_head_live_arm=(
            getattr(args, "robot_head_live_arm", False) or local_env_flag("ROBOT_HEAD_ARM")
        ),
        robot_head_arm_ttl_seconds=max(robot_head_arm_ttl_seconds, 30),
        robot_head_sim_scenario_path=(
            Path(robot_head_sim_scenario).expanduser()
            if robot_head_sim_scenario not in (None, "")
            else None
        ),
        robot_head_sim_realtime=(
            getattr(args, "robot_head_sim_realtime", False)
            or local_env_flag("ROBOT_HEAD_SIM_REALTIME")
        ),
        allow_sensitive_motion=(
            getattr(args, "allow_sensitive_motion", False)
            or local_env_flag("ROBOT_HEAD_ALLOW_SENSITIVE_MOTION")
        ),
        report_dir=Path(report_dir).expanduser() if report_dir not in (None, "") else None,
        preview_trace_dir=(
            Path(preview_trace_dir).expanduser() if preview_trace_dir not in (None, "") else None
        ),
        simulation_trace_dir=(
            Path(simulation_trace_dir).expanduser()
            if simulation_trace_dir not in (None, "")
            else None
        ),
        verbose=getattr(args, "verbose", False),
    )


def _create_driver(config: LocalRobotHeadShowConfig):
    """Build the configured robot-head driver."""
    preview_trace_dir = config.preview_trace_dir or (Path.cwd() / "artifacts" / "robot_head_preview")
    simulation_trace_dir = config.simulation_trace_dir or (
        Path.cwd() / "artifacts" / "robot_head_simulation"
    )
    if config.robot_head_driver == "mock":
        return MockDriver()
    if config.robot_head_driver == "preview":
        return PreviewDriver(trace_dir=preview_trace_dir)
    if config.robot_head_driver == "simulation":
        return SimulationDriver(
            config=RobotHeadSimulationConfig(
                hardware_profile_path=config.robot_head_hardware_profile_path,
                scenario_path=config.robot_head_sim_scenario_path,
                realtime=config.robot_head_sim_realtime,
                trace_dir=simulation_trace_dir,
            )
        )
    if config.robot_head_driver == "live":
        return LiveDriver(
            config=RobotHeadLiveDriverConfig(
                hardware_profile_path=config.robot_head_hardware_profile_path,
                port=config.robot_head_port,
                baud_rate=config.robot_head_baud,
                arm_enabled=config.robot_head_live_arm,
                arm_ttl_seconds=config.robot_head_arm_ttl_seconds,
            ),
            preview_driver=PreviewDriver(trace_dir=preview_trace_dir),
        )
    raise ValueError(f"Unsupported robot-head driver: {config.robot_head_driver}")


def _print_show_list():
    """Print the available proof shows."""
    print("Blink robot-head proof shows:")
    for show in list_robot_head_shows():
        sensitivity = "sensitive" if show.sensitive_motion else "family-safe"
        print(f"  {show.name} [{sensitivity}]")
        print(f"    {show.title}: {show.description}")


def _collect_result_flags(report: dict) -> tuple[bool, bool]:
    """Return overall acceptance and preview-fallback flags for a report."""
    results = [report.get("neutral_start"), report.get("neutral_end")]
    results.extend(step.get("result") for step in report.get("steps", []))
    accepted = True
    preview_only = False
    for result in results:
        if not result:
            continue
        accepted = accepted and bool(result.get("accepted", False))
        preview_only = preview_only or bool(result.get("preview_only", False))
    return accepted, preview_only


async def run_robot_head_show(config: LocalRobotHeadShowConfig) -> int:
    """Run one Blink-owned robot-head proof show."""
    configure_logging(config.verbose)
    if config.list_shows:
        _print_show_list()
        return 0
    if not config.show_name:
        raise ValueError("A robot-head show name is required unless --list-shows is set.")

    definition = resolve_robot_head_show(config.show_name)
    catalog = load_robot_head_capability_catalog(config.robot_head_catalog_path)
    controller = RobotHeadController(catalog=catalog, driver=_create_driver(config))
    runner = RobotHeadShowRunner(
        controller=controller,
        report_dir=config.report_dir,
        allow_sensitive_motion=config.allow_sensitive_motion,
    )

    print(
        f"Running {definition.name} with driver={config.robot_head_driver}, "
        f"sensitive_motion={'on' if config.allow_sensitive_motion else 'off'}."
    )
    if config.robot_head_driver == "live" and not config.robot_head_live_arm:
        print("Live motion is unarmed; Blink will preview-fallback instead of moving hardware.")

    try:
        report = await runner.run(definition.name)
    finally:
        await controller.close()

    accepted, preview_only = _collect_result_flags(report)
    print(
        f"Completed {definition.name}. "
        f"accepted={'yes' if accepted else 'no'}. "
        f"preview_fallback={'yes' if preview_only else 'no'}. "
        f"report={report['report_path']}"
    )

    if config.robot_head_driver == "live" and preview_only:
        return 2
    return 0 if accepted else 1


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for robot-head proof shows."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_robot_head_show(config))
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

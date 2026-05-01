"""Robot head embodiment support for Blink."""

from __future__ import annotations

from blink.embodiment.robot_head.catalog import (
    RobotHeadCapabilityCatalog,
    load_robot_head_capability_catalog,
)
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.driver import RobotHeadDriver
from blink.embodiment.robot_head.live_driver import RobotHeadLiveDriverConfig
from blink.embodiment.robot_head.live_hardware import (
    RobotHeadLiveHardwareProfile,
    load_robot_head_live_hardware_profile,
)
from blink.embodiment.robot_head.models import (
    RobotHeadCommand,
    RobotHeadDriverMode,
    RobotHeadExecutionResult,
    RobotHeadMotif,
    RobotHeadPersistentState,
)
from blink.embodiment.robot_head.simulation import (
    RobotHeadFaultProfile,
    RobotHeadSimulationConfig,
    RobotHeadSimulationScenario,
    SimulationDriver,
    load_robot_head_simulation_scenario,
)
from blink.embodiment.robot_head.show import (
    RobotHeadShowDefinition,
    RobotHeadShowRunner,
    list_robot_head_shows,
    resolve_robot_head_show,
)

__all__ = [
    "EmbodimentPolicyProcessor",
    "RobotHeadCapabilityCatalog",
    "RobotHeadCommand",
    "RobotHeadController",
    "RobotHeadDriver",
    "RobotHeadDriverMode",
    "RobotHeadExecutionResult",
    "RobotHeadLiveDriverConfig",
    "RobotHeadLiveHardwareProfile",
    "RobotHeadMotif",
    "RobotHeadPersistentState",
    "RobotHeadFaultProfile",
    "RobotHeadSimulationConfig",
    "RobotHeadSimulationScenario",
    "RobotHeadShowDefinition",
    "RobotHeadShowRunner",
    "SimulationDriver",
    "load_robot_head_capability_catalog",
    "load_robot_head_live_hardware_profile",
    "load_robot_head_simulation_scenario",
    "list_robot_head_shows",
    "resolve_robot_head_show",
]


def __getattr__(name: str):
    """Lazily expose heavier robot-head symbols that would create import cycles."""
    if name == "EmbodimentPolicyProcessor":
        from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor

        return EmbodimentPolicyProcessor
    raise AttributeError(name)

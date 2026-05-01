"""Driver abstraction for Blink robot-head embodiment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from blink.embodiment.robot_head.models import (
    RobotHeadDriverStatus,
    RobotHeadExecutionPlan,
    RobotHeadExecutionResult,
)

if TYPE_CHECKING:
    from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog


class RobotHeadDriver(ABC):
    """Abstract execution boundary for robot-head embodiment."""

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the driver mode name."""

    @abstractmethod
    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Execute one validated robot-head plan."""

    @abstractmethod
    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Return structured driver status."""

    async def close(self):
        """Release driver resources."""
        return None

"""Concrete robot-head drivers for mock, preview, fault injection, simulation, and live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from blink.embodiment.robot_head.driver import RobotHeadDriver
from blink.embodiment.robot_head.live_driver import LiveDriver
from blink.embodiment.robot_head.models import (
    RobotHeadDriverStatus,
    RobotHeadExecutionPlan,
    RobotHeadExecutionResult,
)
from blink.embodiment.robot_head.simulation import SimulationDriver

if TYPE_CHECKING:
    from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog


class MockDriver(RobotHeadDriver):
    """In-memory driver for tests and dry execution."""

    def __init__(self):
        """Initialize the mock driver."""
        self.executed_plans: list[RobotHeadExecutionPlan] = []

    @property
    def mode_name(self) -> str:
        """Return the driver mode name."""
        return "mock"

    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Record a validated plan in memory."""
        self.executed_plans.append(plan)
        return RobotHeadExecutionResult(
            accepted=True,
            command_type=plan.command.command_type,
            resolved_name=plan.resolved_name,
            driver=self.mode_name,
            preview_only=plan.preview_only,
            preset=plan.preset,
            warnings=list(plan.warnings),
            status=await self.status(catalog=catalog),
            summary=f"Mock driver accepted {plan.command.command_type}:{plan.resolved_name}.",
            metadata={"steps": [step.model_dump() for step in plan.steps]},
        )

    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Return mock-driver status."""
        return RobotHeadDriverStatus(
            mode=self.mode_name,
            available=True,
            owner="mock-session",
            details={"executed_commands": len(self.executed_plans)},
        )


class PreviewDriver(RobotHeadDriver):
    """Driver that emits deterministic trace artifacts for offline review."""

    def __init__(self, *, trace_dir: Optional[Path] = None):
        """Initialize the preview driver."""
        self.trace_dir = trace_dir or (Path.cwd() / "artifacts" / "robot_head_preview")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.executed_plans: list[RobotHeadExecutionPlan] = []
        self._trace_counter = 0

    @property
    def mode_name(self) -> str:
        """Return the driver mode name."""
        return "preview"

    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Write a deterministic trace artifact for the validated plan."""
        self.executed_plans.append(plan)
        self._trace_counter += 1
        trace_path = self.trace_dir / f"trace-{self._trace_counter:04d}-{plan.command.command_type}.json"
        payload = {
            "sequence": self._trace_counter,
            "driver": self.mode_name,
            "catalog_version": catalog.version,
            "command": plan.command.model_dump(),
            "resolved_name": plan.resolved_name,
            "preset": plan.preset,
            "preview_only": True,
            "warnings": plan.warnings,
            "steps": [step.model_dump() for step in plan.steps],
        }
        trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return RobotHeadExecutionResult(
            accepted=True,
            command_type=plan.command.command_type,
            resolved_name=plan.resolved_name,
            driver=self.mode_name,
            preview_only=True,
            preset=plan.preset,
            warnings=list(plan.warnings),
            status=await self.status(catalog=catalog),
            trace_path=str(trace_path),
            summary=f"Preview driver recorded {plan.command.command_type}:{plan.resolved_name}.",
            metadata={"steps": [step.model_dump() for step in plan.steps]},
        )

    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Return preview-driver status."""
        return RobotHeadDriverStatus(
            mode=self.mode_name,
            available=True,
            preview_fallback=True,
            owner="preview-session",
            details={
                "trace_dir": str(self.trace_dir),
                "executed_commands": len(self.executed_plans),
            },
        )


class FaultInjectionDriver(RobotHeadDriver):
    """Deterministic failure driver for controller and runtime tests."""

    def __init__(
        self,
        *,
        busy: bool = False,
        missing_arm: bool = False,
        degraded: bool = False,
        wrapped: Optional[RobotHeadDriver] = None,
    ):
        """Initialize the fault-injection driver."""
        self.busy = busy
        self.missing_arm = missing_arm
        self.degraded = degraded
        self._wrapped = wrapped or MockDriver()

    @property
    def mode_name(self) -> str:
        """Return the driver mode name."""
        return "fault"

    async def execute_plan(
        self,
        plan: RobotHeadExecutionPlan,
        *,
        catalog: RobotHeadCapabilityCatalog,
    ) -> RobotHeadExecutionResult:
        """Inject deterministic failures or degrade the wrapped driver result."""
        if self.busy:
            return RobotHeadExecutionResult(
                accepted=False,
                command_type=plan.command.command_type,
                resolved_name=plan.resolved_name,
                driver=self.mode_name,
                preview_only=plan.preview_only,
                preset=plan.preset,
                warnings=["Robot head serial ownership is busy with another session."],
                status=await self.status(catalog=catalog),
                summary="Fault driver rejected the command because ownership is busy.",
            )
        if self.missing_arm:
            return RobotHeadExecutionResult(
                accepted=False,
                command_type=plan.command.command_type,
                resolved_name=plan.resolved_name,
                driver=self.mode_name,
                preview_only=True,
                preset=plan.preset,
                warnings=["Live motion is not armed; refusing hardware-style execution."],
                status=await self.status(catalog=catalog),
                summary="Fault driver rejected the command because the head is not armed.",
            )

        result = await self._wrapped.execute_plan(plan, catalog=catalog)
        if self.degraded:
            result.warnings.append("Injected degraded-health warning from fault driver.")
            if result.status:
                result.status.degraded = True
                result.status.warnings.append("Injected degraded-health warning from fault driver.")
        result.driver = self.mode_name
        return result

    async def status(self, *, catalog: RobotHeadCapabilityCatalog) -> RobotHeadDriverStatus:
        """Return fault-driver status."""
        warnings: list[str] = []
        owner = "fault-session"
        available = not self.busy
        armed = not self.missing_arm
        if self.busy:
            warnings.append("Ownership is busy.")
            owner = "another-session"
        if self.missing_arm:
            warnings.append("Arm lease is missing.")
        if self.degraded:
            warnings.append("Driver health is degraded.")
        return RobotHeadDriverStatus(
            mode=self.mode_name,
            available=available,
            armed=armed,
            owner=owner,
            degraded=self.degraded,
            warnings=warnings,
        )

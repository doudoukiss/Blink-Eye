"""Serialized command controller for Blink robot-head embodiment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from blink.embodiment.robot_head.catalog import RobotHeadCapabilityCatalog
from blink.embodiment.robot_head.driver import RobotHeadDriver
from blink.embodiment.robot_head.models import (
    RobotHeadCommand,
    RobotHeadExecutionPlan,
    RobotHeadExecutionResult,
    RobotHeadExecutionStep,
)
from blink.utils.asyncio.task_manager import TaskManager, TaskManagerParams


@dataclass
class QueuedRobotHeadCommand:
    """Command item stored in the robot-head execution queue."""

    command: RobotHeadCommand
    future: asyncio.Future[RobotHeadExecutionResult]


class RobotHeadController:
    """Single-owner queue and validation layer for robot-head commands."""

    def __init__(
        self,
        *,
        catalog: RobotHeadCapabilityCatalog,
        driver: RobotHeadDriver,
    ):
        """Initialize the controller."""
        self._catalog = catalog
        self._driver = driver
        self._queue: asyncio.Queue[QueuedRobotHeadCommand | None] = asyncio.Queue()
        self._task_manager = TaskManager()
        self._worker_task: asyncio.Task | None = None
        self._started = False

    @property
    def catalog(self) -> RobotHeadCapabilityCatalog:
        """Return the active capability catalog."""
        return self._catalog

    @property
    def driver_mode(self) -> str:
        """Return the active driver mode name."""
        return self._driver.mode_name

    async def start(self):
        """Start the serialized worker if needed."""
        if self._started:
            return
        self._task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        self._worker_task = self._task_manager.create_task(
            self._worker(),
            "RobotHeadController::worker",
        )
        self._started = True

    async def close(self):
        """Stop the worker and close the active driver."""
        if self._worker_task:
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None
        self._started = False
        await self._driver.close()

    async def status(self) -> RobotHeadExecutionResult:
        """Return the current driver status."""
        return await self.submit(RobotHeadCommand(command_type="status", source="status"))

    async def set_state(self, state: str, *, source: str, reason: str | None = None):
        """Submit a persistent-state command."""
        return await self.submit(
            RobotHeadCommand(
                command_type="set_state",
                state=state,
                source=source,
                reason=reason,
            )
        )

    async def run_motif(self, motif: str, *, source: str, reason: str | None = None):
        """Submit a motif command."""
        return await self.submit(
            RobotHeadCommand(
                command_type="run_motif",
                motif=motif,
                source=source,
                reason=reason,
            )
        )

    async def return_neutral(self, *, source: str, reason: str | None = None):
        """Submit a neutral-return command."""
        return await self.submit(
            RobotHeadCommand(
                command_type="return_neutral",
                source=source,
                reason=reason,
            )
        )

    async def submit(self, command: RobotHeadCommand) -> RobotHeadExecutionResult:
        """Queue a command and wait for the result."""
        await self.start()
        future: asyncio.Future[RobotHeadExecutionResult] = asyncio.get_running_loop().create_future()
        await self._queue.put(QueuedRobotHeadCommand(command=command, future=future))
        return await future

    async def _worker(self):
        """Drain queued commands one at a time."""
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            try:
                result = await self._execute(item.command)
            except Exception as exc:  # pragma: no cover - defensive guard
                result = RobotHeadExecutionResult(
                    accepted=False,
                    command_type=item.command.command_type,
                    driver=self._driver.mode_name,
                    summary=f"Robot-head controller error: {exc}",
                    warnings=[f"Robot-head controller error: {exc}"],
                )

            if not item.future.done():
                item.future.set_result(result)
            self._queue.task_done()

    async def _execute(self, command: RobotHeadCommand) -> RobotHeadExecutionResult:
        """Execute one command through validation and the active driver."""
        if command.command_type == "status":
            driver_status = await self._driver.status(catalog=self._catalog)
            return RobotHeadExecutionResult(
                accepted=True,
                command_type=command.command_type,
                driver=self._driver.mode_name,
                preview_only=driver_status.preview_fallback,
                status=driver_status,
                summary="Reported robot-head status.",
                metadata={
                    "supported_states": self._catalog.public_state_names(),
                    "supported_motifs": self._catalog.public_motif_names(),
                },
            )

        plan = self._build_plan(command)
        return await self._driver.execute_plan(plan, catalog=self._catalog)

    def _build_plan(self, command: RobotHeadCommand) -> RobotHeadExecutionPlan:
        """Validate a command and translate it into an execution plan."""
        if command.command_type == "return_neutral":
            state = self._catalog.get_state(self._catalog.neutral_state_name)
            values, warnings, preview_only = self._catalog.validate_values(state.values)
            return RobotHeadExecutionPlan(
                command=command,
                resolved_name=state.name,
                preset=state.preset,
                steps=[RobotHeadExecutionStep(label=state.name, values=values, hold_ms=0)],
                preview_only=state.preview_only or preview_only,
                warnings=warnings,
            )

        if command.command_type == "set_state":
            if not command.state:
                raise ValueError("Robot-head state command requires a state name.")
            state = self._catalog.get_state(command.state)
            if command.source == "tool" and not state.public:
                raise ValueError(f"Robot-head state '{state.name}' is not user facing.")
            values, warnings, preview_only = self._catalog.validate_values(state.values)
            return RobotHeadExecutionPlan(
                command=command,
                resolved_name=state.name,
                preset=state.preset,
                steps=[RobotHeadExecutionStep(label=state.name, values=values, hold_ms=0)],
                preview_only=state.preview_only or preview_only,
                warnings=warnings,
            )

        if command.command_type == "run_motif":
            if not command.motif:
                raise ValueError("Robot-head motif command requires a motif name.")
            motif = self._catalog.get_motif(command.motif)
            if command.source == "tool" and not motif.public:
                raise ValueError(f"Robot-head motif '{motif.name}' is not user facing.")

            steps: list[RobotHeadExecutionStep] = []
            warnings: list[str] = []
            preview_only = motif.preview_only
            for step in motif.steps:
                values, step_warnings, step_preview_only = self._catalog.validate_values(step.values)
                warnings.extend(step_warnings)
                preview_only = preview_only or step_preview_only
                steps.append(
                    RobotHeadExecutionStep(
                        label=step.label,
                        values=values,
                        hold_ms=step.hold_ms,
                    )
                )

            return RobotHeadExecutionPlan(
                command=command,
                resolved_name=motif.name,
                preset=motif.preset,
                steps=steps,
                preview_only=preview_only,
                warnings=warnings,
            )

        raise ValueError(f"Unsupported robot-head command type: {command.command_type}")

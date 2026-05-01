"""Brain-side embodied-policy adapter contracts and local implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from blink.brain.adapters import LOCAL_EMBODIED_POLICY_DESCRIPTOR, BrainAdapterDescriptor
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.models import RobotHeadExecutionResult


@dataclass(frozen=True)
class EmbodiedPolicyExecutionStep:
    """One low-level embodied execution step routed through the adapter."""

    command_type: str
    name: str | None = None


@dataclass(frozen=True)
class EmbodiedPolicyExecutionRequest:
    """One bounded embodied-policy execution request."""

    action_id: str
    source: str
    reason: str | None = None
    controller_plan: tuple[EmbodiedPolicyExecutionStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbodiedPolicyAdapter(Protocol):
    """Execution-backend seam for embodied policy and controller status."""

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""

    @property
    def execution_backend(self) -> str:
        """Return the current low-level execution backend id."""

    async def status(self) -> RobotHeadExecutionResult:
        """Return the current backend status."""

    async def execute_action(self, request: EmbodiedPolicyExecutionRequest) -> RobotHeadExecutionResult:
        """Execute one bounded embodied action request."""


class LocalRobotHeadEmbodiedPolicyAdapter:
    """Local embodied-policy adapter over the current robot-head controller seam."""

    def __init__(self, *, controller: RobotHeadController):
        """Initialize the local embodied-policy adapter."""
        self._controller = controller
        self._descriptor = LOCAL_EMBODIED_POLICY_DESCRIPTOR

    @property
    def descriptor(self) -> BrainAdapterDescriptor:
        """Return the backend descriptor."""
        return self._descriptor

    @property
    def execution_backend(self) -> str:
        """Return the active controller backend id."""
        return self._controller.driver_mode

    async def status(self) -> RobotHeadExecutionResult:
        """Return the current controller status."""
        return await self._controller.status()

    async def execute_action(self, request: EmbodiedPolicyExecutionRequest) -> RobotHeadExecutionResult:
        """Execute one validated embodied controller plan."""
        final_result: RobotHeadExecutionResult | None = None
        for step in request.controller_plan:
            if step.command_type == "set_state":
                final_result = await self._controller.set_state(
                    step.name or "",
                    source=request.source,
                    reason=request.reason,
                )
            elif step.command_type == "run_motif":
                final_result = await self._controller.run_motif(
                    step.name or "",
                    source=request.source,
                    reason=request.reason,
                )
            elif step.command_type == "return_neutral":
                final_result = await self._controller.return_neutral(
                    source=request.source,
                    reason=request.reason,
                )
            elif step.command_type == "status":
                final_result = await self.status()
            else:  # pragma: no cover - guarded by action definitions
                raise ValueError(f"Unsupported embodied controller step: {step.command_type}")
            if not final_result.accepted:
                break
        if final_result is None:  # pragma: no cover - guarded by definitions
            raise ValueError(f"Embodied action '{request.action_id}' has no controller plan.")
        return final_result


__all__ = [
    "EmbodiedPolicyAdapter",
    "EmbodiedPolicyExecutionRequest",
    "EmbodiedPolicyExecutionStep",
    "LocalRobotHeadEmbodiedPolicyAdapter",
]

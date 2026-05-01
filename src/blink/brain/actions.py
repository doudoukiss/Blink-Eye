"""Finite embodied action library and command interpretation for Blink."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

from blink.adapters.schemas.function_schema import FunctionSchema
from blink.adapters.schemas.tools_schema import ToolsSchema
from blink.brain import capability_manifests as _capability_manifests
from blink.brain.adapters.embodied_policy import (
    EmbodiedPolicyAdapter,
    EmbodiedPolicyExecutionRequest,
    EmbodiedPolicyExecutionStep,
    LocalRobotHeadEmbodiedPolicyAdapter,
)
from blink.brain.autonomy import BrainInitiativeClass
from blink.brain.capabilities import (
    CapabilityDefinition,
    CapabilityDispatchMode,
    CapabilityExecutionContext,
    CapabilityExecutionResult,
    CapabilityFamily,
    CapabilityInitiativePolicy,
    CapabilityRegistry,
    CapabilityToolExposure,
    CapabilityUserTurnPolicy,
    EmptyCapabilityInput,
)
from blink.brain.capability_registry import build_brain_capability_registry  # noqa: F401
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.projections import BrainGoalFamily
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.models import RobotHeadExecutionResult
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.function_calling import FunctionCallParams

_public_tool_descriptions = _capability_manifests._public_tool_descriptions
render_capability_manifest = _capability_manifests.render_capability_manifest
render_internal_capability_manifest = _capability_manifests.render_internal_capability_manifest


@dataclass(frozen=True)
class EmbodiedControllerPlanStep:
    """One deterministic controller call within an embodied action."""

    command_type: str
    name: str | None = None


@dataclass(frozen=True)
class EmbodiedActionDefinition:
    """One action in the finite Blink embodied action library."""

    action_id: str
    description: str
    allowed_sources: tuple[str, ...]
    sensitivity: str
    cooldown_ms: int
    requires_live_arm: bool
    preview_ok: bool
    preconditions: tuple[str, ...] = ()
    controller_plan: tuple[EmbodiedControllerPlanStep, ...] = ()
    public: bool = True
    rehearsal_required: bool = False


@dataclass(frozen=True)
class EmbodiedCommandInterpretation:
    """Structured interpretation result for one user utterance."""

    action_id: str | None = None
    action_sequence: tuple[str, ...] = ()
    denied_reason: str | None = None
    matched_text: str | None = None


@dataclass(frozen=True)
class EmbodiedCommandClassifierResult:
    """Optional constrained classifier output for one utterance."""

    action_id: str | None = None


class EmbodiedActionLibrary:
    """Finite action library for Blink's daily robot embodiment."""

    def __init__(self, actions: Iterable[EmbodiedActionDefinition]):
        """Initialize the library from a fixed iterable of action definitions.

        Args:
            actions: Supported daily-use and policy-owned embodied actions.
        """
        self._actions = {action.action_id: action for action in actions}

    @classmethod
    def build_default(cls) -> "EmbodiedActionLibrary":
        """Build the daily-use finite action library."""
        return cls(
            [
                EmbodiedActionDefinition(
                    action_id="auto_listen_user",
                    description="Visible listening cue while the user speaks.",
                    allowed_sources=("policy", "policy_startup"),
                    sensitivity="safe",
                    cooldown_ms=350,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(
                        EmbodiedControllerPlanStep("run_motif", "listen_engage"),
                        EmbodiedControllerPlanStep("set_state", "listen_attentively"),
                    ),
                    public=False,
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="auto_think",
                    description="Visible thinking cue after the user finishes speaking.",
                    allowed_sources=("policy",),
                    sensitivity="safe",
                    cooldown_ms=350,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(
                        EmbodiedControllerPlanStep("run_motif", "thinking_shift"),
                        EmbodiedControllerPlanStep("set_state", "thinking"),
                    ),
                    public=False,
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="auto_speak_friendly",
                    description="Friendly speaking cue when Blink starts responding.",
                    allowed_sources=("policy",),
                    sensitivity="safe",
                    cooldown_ms=350,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(
                        EmbodiedControllerPlanStep("run_motif", "acknowledge"),
                        EmbodiedControllerPlanStep("set_state", "friendly"),
                    ),
                    public=False,
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="auto_safe_idle",
                    description="Safe idle transition before returning neutral.",
                    allowed_sources=("policy", "policy_idle", "policy_shutdown", "policy_startup"),
                    sensitivity="safe",
                    cooldown_ms=200,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(
                        EmbodiedControllerPlanStep("set_state", "safe_idle"),
                        EmbodiedControllerPlanStep("return_neutral"),
                    ),
                    public=False,
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_blink",
                    description="Blink once.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=200,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("run_motif", "blink"),),
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_wink_left",
                    description="Wink the left eye once.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=200,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("run_motif", "wink_left"),),
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_wink_right",
                    description="Wink the right eye once.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=200,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("run_motif", "wink_right"),),
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_look_left",
                    description="Look left with a visible head-follow cue.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=250,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("run_motif", "look_left"),),
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_look_right",
                    description="Look right with a visible head-follow cue.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=250,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("run_motif", "look_right"),),
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_return_neutral",
                    description="Return to the neutral head state.",
                    allowed_sources=("interpreter", "tool", "policy", "policy_idle", "policy_shutdown", "policy_startup", "operator"),
                    sensitivity="safe",
                    cooldown_ms=150,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("return_neutral"),),
                    rehearsal_required=True,
                ),
                EmbodiedActionDefinition(
                    action_id="cmd_report_status",
                    description="Report current robot-head runtime status.",
                    allowed_sources=("interpreter", "tool", "operator"),
                    sensitivity="safe",
                    cooldown_ms=0,
                    requires_live_arm=False,
                    preview_ok=True,
                    controller_plan=(EmbodiedControllerPlanStep("status"),),
                ),
            ]
        )

    def get(self, action_id: str) -> EmbodiedActionDefinition:
        """Return an action definition by id."""
        if action_id not in self._actions:
            raise ValueError(f"Unsupported embodied action: {action_id}")
        return self._actions[action_id]

    def definitions(self) -> list[EmbodiedActionDefinition]:
        """Return action definitions in stable registration order."""
        return list(self._actions.values())

    def public_action_ids(self) -> list[str]:
        """Return public action ids."""
        return sorted(action.action_id for action in self._actions.values() if action.public)


_ACTION_TO_CAPABILITY_ID = {
    "auto_listen_user": "robot_head.auto_listen_user",
    "auto_think": "robot_head.auto_think",
    "auto_speak_friendly": "robot_head.auto_speak_friendly",
    "auto_safe_idle": "robot_head.auto_safe_idle",
    "cmd_blink": "robot_head.blink",
    "cmd_wink_left": "robot_head.wink_left",
    "cmd_wink_right": "robot_head.wink_right",
    "cmd_look_left": "robot_head.look_left",
    "cmd_look_right": "robot_head.look_right",
    "cmd_return_neutral": "robot_head.return_neutral",
    "cmd_report_status": "robot_head.status",
}
_CAPABILITY_TO_ACTION_ID = {value: key for key, value in _ACTION_TO_CAPABILITY_ID.items()}
_CAPABILITY_TOOL_NAMES = {
    "robot_head.blink": "robot_head_blink",
    "robot_head.wink_left": "robot_head_wink_left",
    "robot_head.wink_right": "robot_head_wink_right",
    "robot_head.look_left": "robot_head_look_left",
    "robot_head.look_right": "robot_head_look_right",
    "robot_head.return_neutral": "robot_head_return_neutral",
    "robot_head.status": "robot_head_status",
}


def capability_id_for_action(action_id: str) -> str:
    """Return the capability id for one embodied action id."""
    capability_id = _ACTION_TO_CAPABILITY_ID.get(action_id)
    if capability_id is None:
        raise KeyError(f"Unsupported embodied action id '{action_id}'.")
    return capability_id


def action_id_for_capability(capability_id: str) -> str:
    """Return the embodied action id for one capability id."""
    action_id = _CAPABILITY_TO_ACTION_ID.get(capability_id)
    if action_id is None:
        raise KeyError(f"Unsupported capability id '{capability_id}'.")
    return action_id


def _tool_name_for_capability(capability_id: str) -> str:
    """Return the public tool name for one public capability id."""
    tool_name = _CAPABILITY_TOOL_NAMES.get(capability_id)
    if tool_name is None:
        raise KeyError(f"Capability '{capability_id}' is not exposed as a public tool.")
    return tool_name


def build_embodied_capability_registry(*, action_engine: EmbodiedActionEngine) -> CapabilityRegistry:
    """Build the bounded capability registry from the finite action library."""
    registry = CapabilityRegistry()
    for action in action_engine.library.definitions():
        capability_id = capability_id_for_action(action.action_id)

        async def precondition(
            inputs: EmptyCapabilityInput,
            context: CapabilityExecutionContext,
            *,
            action_definition: EmbodiedActionDefinition = action,
            mapped_capability_id: str = capability_id,
        ) -> CapabilityExecutionResult | None:
            if (
                not action_definition.requires_live_arm
                and action_definition.preview_ok
                and not context.metadata.get("wake_router_preflight")
            ):
                return None
            status_result = await action_engine.status()
            status = status_result.status
            if status is None:
                return CapabilityExecutionResult.blocked(
                    capability_id=mapped_capability_id,
                    summary=f"Robot-head status is unavailable for {mapped_capability_id}.",
                    error_code="robot_head_status_unavailable",
                    retryable=True,
                )
            if action_definition.requires_live_arm and not status.armed:
                return CapabilityExecutionResult.blocked(
                    capability_id=mapped_capability_id,
                    summary=f"Robot head is not armed for {mapped_capability_id}.",
                    error_code="robot_head_unarmed",
                    retryable=True,
                    warnings=list(status.warnings),
                    output={"status": status.model_dump()},
                )
            if not status.available:
                return CapabilityExecutionResult.blocked(
                    capability_id=mapped_capability_id,
                    summary=f"Robot head is unavailable for {mapped_capability_id}.",
                    error_code="robot_head_unavailable",
                    retryable=True,
                    warnings=list(status.warnings),
                    output={"status": status.model_dump()},
                )
            if status.preview_fallback and not action_definition.preview_ok:
                return CapabilityExecutionResult.blocked(
                    capability_id=mapped_capability_id,
                    summary=f"Preview fallback is not allowed for {mapped_capability_id}.",
                    error_code="robot_head_preview_not_allowed",
                    warnings=list(status.warnings),
                    output={"status": status.model_dump()},
                )
            return None

        async def executor(
            inputs: EmptyCapabilityInput,
            context: CapabilityExecutionContext,
            *,
            action_definition: EmbodiedActionDefinition = action,
            mapped_capability_id: str = capability_id,
        ) -> CapabilityExecutionResult:
            result = await action_engine.run_action(
                action_definition.action_id,
                source=context.source,
                reason=str(context.metadata.get("reason") or f"Capability dispatch for {mapped_capability_id}."),
                metadata=dict(context.metadata),
            )
            payload = _tool_payload(result, action_id=action_definition.action_id)
            if result.accepted:
                return CapabilityExecutionResult.success(
                    capability_id=mapped_capability_id,
                    summary=result.summary,
                    warnings=list(result.warnings),
                    output=payload,
                    metadata={"sensitivity": action_definition.sensitivity},
                )
            lowered = " ".join(list(result.warnings) + [result.summary]).lower()
            retryable = (
                (result.status is not None and (not result.status.available or not result.status.armed))
                or "busy" in lowered
                or "ownership" in lowered
            )
            error_code = "robot_head_rejected"
            if "busy" in lowered or "ownership" in lowered:
                error_code = "robot_head_busy"
            elif "arm" in lowered:
                error_code = "robot_head_unarmed"
            elif result.status is not None and not result.status.available:
                error_code = "robot_head_unavailable"
            return CapabilityExecutionResult.blocked(
                capability_id=mapped_capability_id,
                summary=result.summary,
                error_code=error_code,
                retryable=retryable,
                warnings=list(result.warnings),
                output=payload,
                metadata={"sensitivity": action_definition.sensitivity},
            )

        registry.register(
            CapabilityDefinition(
                capability_id=capability_id,
                family=CapabilityFamily.ROBOT_HEAD.value,
                description=action.description,
                input_model=EmptyCapabilityInput,
                sensitivity=action.sensitivity,
                executor=executor,
                preconditions=(precondition,),
                tool_exposure=(
                    CapabilityToolExposure(
                        name=_CAPABILITY_TOOL_NAMES[capability_id],
                        description=action.description,
                    )
                    if action.public and capability_id in _CAPABILITY_TOOL_NAMES
                    else None
                ),
                initiative_policy=CapabilityInitiativePolicy(
                    enabled=True,
                    allowed_goal_families=(BrainGoalFamily.ENVIRONMENT.value,),
                    allowed_initiative_classes=(),
                    user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
                    operator_visible=False,
                    proactive_dialogue=False,
                ),
            )
        )
    return registry


class EmbodiedActionEngine:
    """Deterministic action executor above the raw robot-head controller."""

    def __init__(
        self,
        *,
        library: EmbodiedActionLibrary,
        controller: RobotHeadController,
        policy_adapter: EmbodiedPolicyAdapter | None = None,
        store: BrainStore | None = None,
        session_resolver,
        presence_scope_key: str = "local:presence",
    ):
        """Initialize the executor.

        Args:
            library: Finite action library allowed above the controller.
            controller: Robot-head controller that executes validated plans.
            policy_adapter: Optional adapter over low-level controller execution.
            store: Optional brain store for action and presence persistence.
            session_resolver: Callable returning stable runtime session ids.
            presence_scope_key: Store scope key for latest presence snapshots.
        """
        self._library = library
        self._controller = controller
        self._policy_adapter = policy_adapter or LocalRobotHeadEmbodiedPolicyAdapter(
            controller=controller
        )
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._last_executed_at: dict[str, float] = {}

    @property
    def library(self) -> EmbodiedActionLibrary:
        """Return the configured action library."""
        return self._library

    @property
    def controller(self) -> RobotHeadController:
        """Expose the underlying deterministic robot-head controller."""
        return self._controller

    @property
    def policy_adapter(self) -> EmbodiedPolicyAdapter:
        """Expose the low-level execution adapter."""
        return self._policy_adapter

    @property
    def execution_backend(self) -> str:
        """Return the current low-level execution backend id."""
        return self._policy_adapter.execution_backend

    @property
    def session_resolver(self):
        """Expose the session resolver used for action auditing."""
        return self._session_resolver

    @property
    def store(self) -> BrainStore | None:
        """Expose the optional brain store used for action auditing."""
        return self._store

    @property
    def presence_scope_key(self) -> str:
        """Expose the store scope key used for body-state persistence."""
        return self._presence_scope_key

    async def start(self):
        """Start the underlying controller worker."""
        await self._controller.start()

    async def close(self):
        """Close the underlying controller."""
        await self._controller.close()

    async def status(self) -> RobotHeadExecutionResult:
        """Return the current low-level controller status through the adapter seam."""
        return await self._policy_adapter.status()

    async def run_action(
        self,
        action_id: str,
        *,
        source: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RobotHeadExecutionResult:
        """Execute one finite embodied action."""
        action = self._library.get(action_id)
        if source not in action.allowed_sources:
            raise ValueError(f"Action '{action_id}' does not allow source '{source}'.")

        now_ms = time.monotonic() * 1000.0
        previous = self._last_executed_at.get(action_id)
        if previous is not None and action.cooldown_ms > 0 and (now_ms - previous) < action.cooldown_ms:
            status_result = await self.status()
            skipped = RobotHeadExecutionResult(
                accepted=True,
                command_type="action",
                resolved_name=action_id,
                driver=status_result.driver,
                preview_only=status_result.preview_only,
                warnings=[],
                status=status_result.status,
                summary=f"Skipped repeated embodied action {action_id} within cooldown.",
                metadata={
                    "action_id": action_id,
                    "skipped": "cooldown",
                    **dict(metadata or {}),
                },
            )
            await self._record_action_event(
                session_ids=self._session_resolver(),
                action_id=action_id,
                source=source,
                result=skipped,
            )
            return skipped

        self._last_executed_at[action_id] = now_ms
        final_result = await self._policy_adapter.execute_action(
            EmbodiedPolicyExecutionRequest(
                action_id=action_id,
                source=source,
                reason=reason,
                controller_plan=tuple(
                    EmbodiedPolicyExecutionStep(
                        command_type=step.command_type,
                        name=step.name,
                    )
                    for step in action.controller_plan
                ),
                metadata=dict(metadata or {}),
            )
        )

        final_result.metadata = {
            **final_result.metadata,
            "action_id": action_id,
            "allowed_sources": list(action.allowed_sources),
            **dict(metadata or {}),
        }
        await self._record_action_event(
            session_ids=self._session_resolver(),
            action_id=action_id,
            source=source,
            result=final_result,
        )
        return final_result

    async def _record_action_event(
        self,
        *,
        session_ids: BrainSessionIds,
        action_id: str,
        source: str,
        result: RobotHeadExecutionResult,
    ):
        if self._store is None:
            return
        self._store.add_action_event(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            action_id=action_id,
            source=source,
            accepted=result.accepted,
            preview_only=result.preview_only,
            summary=result.summary,
            metadata=result.metadata,
        )
        snapshot = self._store.get_body_state_projection(scope_key=self._presence_scope_key)
        status = result.status
        self._store.append_brain_event(
            event_type=BrainEventType.ROBOT_ACTION_OUTCOME,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=source,
            payload={
                "presence_scope_key": self._presence_scope_key,
                "action_id": action_id,
                "accepted": result.accepted,
                "preview_only": result.preview_only,
                "summary": result.summary,
                **{
                    key: value
                    for key, value in dict(result.metadata).items()
                    if key in {
                        "goal_id",
                        "commitment_id",
                        "plan_proposal_id",
                        "step_index",
                        "rehearsal_id",
                    }
                },
                "status": {
                    "mode": status.mode if status is not None else snapshot.robot_head_mode,
                    "armed": bool(status.armed if status is not None else snapshot.robot_head_armed),
                    "available": bool(
                        status.available if status is not None else snapshot.robot_head_available
                    ),
                    "warnings": list(status.warnings if status is not None else snapshot.warnings),
                    "details": dict(status.details if status is not None else snapshot.details),
                },
            },
        )


class EmbodiedCapabilityDispatcher:
    """Canonical bounded dispatch path for embodied capabilities."""

    def __init__(
        self,
        *,
        action_engine: EmbodiedActionEngine,
        capability_registry: CapabilityRegistry,
    ):
        """Initialize the dispatcher."""
        self._action_engine = action_engine
        self._capability_registry = capability_registry

    @property
    def registry(self) -> CapabilityRegistry:
        """Return the bounded capability registry."""
        return self._capability_registry

    @property
    def action_engine(self) -> EmbodiedActionEngine:
        """Return the underlying embodied action engine."""
        return self._action_engine

    async def start(self):
        """Start the underlying action engine."""
        await self._action_engine.start()

    async def close(self):
        """Close the underlying action engine."""
        await self._action_engine.close()

    async def execute_action(
        self,
        action_id: str,
        *,
        source: str,
        reason: str | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityExecutionResult:
        """Dispatch one embodied action through the capability registry."""
        return await self.execute_capability(
            capability_id=capability_id_for_action(action_id),
            source=source,
            arguments={},
            correlation_id=correlation_id,
            causal_parent_id=causal_parent_id,
            dispatch_mode=CapabilityDispatchMode.GOAL.value,
            goal_family=BrainGoalFamily.ENVIRONMENT.value,
            goal_intent="robot_head.sequence",
            metadata={
                **dict(metadata or {}),
                "action_id": action_id,
                "reason": reason,
            },
        )

    async def execute_capability(
        self,
        capability_id: str,
        *,
        source: str,
        arguments: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
        dispatch_mode: str = CapabilityDispatchMode.TOOL.value,
        goal_family: str | None = None,
        goal_intent: str | None = None,
        initiative_class: str | None = None,
        side_effect_sink=None,
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityExecutionResult:
        """Dispatch one bounded capability and emit typed audit events."""
        session_ids = self._action_engine.session_resolver()
        store = self._action_engine.store
        request_event_id: str | None = None
        if store is not None:
            request_event = store.append_brain_event(
                event_type=BrainEventType.CAPABILITY_REQUESTED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=source,
                correlation_id=correlation_id,
                causal_parent_id=causal_parent_id,
                payload={
                    "capability_id": capability_id,
                    "arguments": dict(arguments or {}),
                    "metadata": dict(metadata or {}),
                },
            )
            request_event_id = request_event.event_id

        result = await self._capability_registry.execute(
            capability_id,
            arguments,
            context=CapabilityExecutionContext(
                source=source,
                session_ids=session_ids,
                store=store,
                presence_scope_key=self._action_engine.presence_scope_key,
                dispatch_mode=dispatch_mode,
                goal_family=goal_family,
                goal_intent=goal_intent,
                initiative_class=initiative_class,
                side_effect_sink=side_effect_sink,
                metadata=dict(metadata or {}),
            ),
        )

        if store is not None:
            terminal_event = store.append_brain_event(
                event_type=(
                    BrainEventType.CAPABILITY_COMPLETED
                    if result.accepted
                    else BrainEventType.CAPABILITY_FAILED
                ),
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=source,
                correlation_id=correlation_id,
                causal_parent_id=request_event_id or causal_parent_id,
                payload={
                    "capability_id": capability_id,
                    "result": result.model_dump(),
                    "metadata": dict(metadata or {}),
                },
            )
            if not result.accepted:
                recovery = _direct_recovery_payload(result)
                result.metadata.setdefault("critic_decision", recovery["decision"])
                result.metadata.setdefault("critic_summary", recovery["summary"])
                store.append_brain_event(
                    event_type=BrainEventType.CRITIC_FEEDBACK,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=source,
                    correlation_id=correlation_id,
                    causal_parent_id=terminal_event.event_id,
                    payload={
                        "capability_id": capability_id,
                        "result": result.model_dump(),
                        "recovery": recovery,
                        "mode": "direct_dispatch",
                    },
                )
        return result


def _direct_recovery_payload(result: CapabilityExecutionResult) -> dict[str, Any]:
    """Return the direct-dispatch recovery annotation for one rejected capability."""
    if result.retryable:
        return {"decision": "retry", "summary": result.summary}
    if result.outcome == "blocked":
        return {"decision": "blocked", "summary": result.summary}
    return {"decision": "failed", "summary": result.summary}


class EmbodiedCommandInterpreter:
    """Deterministic lexical command interpreter for daily-use robot commands."""

    _RAW_CONTROL_PATTERNS = (
        r"\bservo\b",
        r"舵机",
        r"串口",
        r"\bhead_turn\b",
        r"\beye_yaw\b",
        r"\braw\b",
        r"\bmotor\b",
    )

    _ACTION_PATTERNS = {
        "cmd_blink": (
            r"眨眼",
            r"\bblink\b",
        ),
        "cmd_wink_left": (
            r"左眼.*眨",
            r"\bleft eye\b.*\bwink\b",
            r"\bwink left\b",
        ),
        "cmd_wink_right": (
            r"右眼.*眨",
            r"\bright eye\b.*\bwink\b",
            r"\bwink right\b",
        ),
        "cmd_look_left": (
            r"看左",
            r"向左看",
            r"往左看",
            r"\blook left\b",
            r"\bturn left\b",
        ),
        "cmd_look_right": (
            r"看右",
            r"向右看",
            r"往右看",
            r"\blook right\b",
            r"\bturn right\b",
        ),
        "cmd_return_neutral": (
            r"回到中位",
            r"回中位",
            r"回到中间",
            r"回到中性",
            r"\breturn to neutral\b",
            r"\bgo neutral\b",
            r"\bcenter yourself\b",
        ),
        "cmd_report_status": (
            r"头部状态",
            r"机器人头部状态",
            r"头现在是什么状态",
            r"现在头是什么状态",
            r"robot head status",
            r"\bhead status\b",
        ),
    }
    _SEQUENCE_PATTERNS = (
        r"然后",
        r"\bthen\b",
        r"\band then\b",
        r"\bafter that\b",
        r"再",
    )

    def __init__(
        self,
        *,
        classifier: Callable[[str], EmbodiedCommandClassifierResult | None] | None = None,
        allowed_classifier_actions: Iterable[str] | None = None,
    ):
        """Initialize the interpreter.

        Args:
            classifier: Optional constrained classifier used only after lexical parsing fails.
            allowed_classifier_actions: Optional allowlist for classifier outputs.
        """
        self._classifier = classifier
        self._allowed_classifier_actions = set(allowed_classifier_actions or self._ACTION_PATTERNS)

    def interpret(self, text: str) -> EmbodiedCommandInterpretation:
        """Interpret one user utterance into a finite embodied action id or none."""
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized:
            return EmbodiedCommandInterpretation()

        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self._RAW_CONTROL_PATTERNS):
            return EmbodiedCommandInterpretation(
                denied_reason="raw_control_not_allowed",
                matched_text=text,
            )

        matched_actions = self._ordered_actions(normalized)
        if len(matched_actions) > 1:
            if any(
                re.search(pattern, normalized, flags=re.IGNORECASE)
                for pattern in self._SEQUENCE_PATTERNS
            ):
                return EmbodiedCommandInterpretation(
                    action_sequence=tuple(matched_actions),
                    matched_text=text,
                )
            return EmbodiedCommandInterpretation(
                denied_reason="multi_action_not_allowed",
                matched_text=text,
            )
        if len(matched_actions) == 1:
            return EmbodiedCommandInterpretation(action_id=matched_actions[0], matched_text=text)

        if self._classifier is not None:
            classified = self._classifier(text)
            if (
                classified is not None
                and classified.action_id is not None
                and classified.action_id in self._allowed_classifier_actions
            ):
                return EmbodiedCommandInterpretation(
                    action_id=classified.action_id,
                    matched_text=text,
                )
        return EmbodiedCommandInterpretation()

    def _ordered_actions(self, normalized: str) -> list[str]:
        """Return unique matched actions ordered by their first lexical position."""
        matches: list[tuple[int, str]] = []
        for action_id, patterns in self._ACTION_PATTERNS.items():
            first_position: int | None = None
            for pattern in patterns:
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match is None:
                    continue
                position = match.start()
                first_position = position if first_position is None else min(first_position, position)
            if first_position is not None:
                matches.append((first_position, action_id))
        ordered: list[str] = []
        for _, action_id in sorted(matches, key=lambda item: item[0]):
            if action_id not in ordered:
                ordered.append(action_id)
        return ordered


def embodied_action_tool_prompt(language: Language) -> str:
    """Return the daily-use prompt suffix for finite embodied action tools."""
    if language.value.lower().startswith(("zh", "cmn")):
        return (
            "当用户明确要求眨眼、左/右眨眼、向左看、向右看、回到中位、"
            "或询问头部状态时，只能使用有限的 robot_head 命令工具。"
            "不要虚构舵机、串口、原始动作单位、任意表情状态或多动作编排。"
        )
    return (
        "When the user explicitly asks Blink to blink, wink left or right, look left or right, "
        "return to neutral, or report head status, only use the finite robot_head command tools. "
        "Do not invent raw servo, serial, arbitrary state, or multi-action body plans."
    )


def _tool_payload(result: RobotHeadExecutionResult, *, action_id: str) -> dict:
    payload = {
        "accepted": result.accepted,
        "action_id": action_id,
        "driver": result.driver,
        "preview_only": result.preview_only,
        "warnings": result.warnings,
        "summary": result.summary,
    }
    if result.status is not None:
        payload["status"] = result.status.model_dump()
    if result.trace_path is not None:
        payload["trace_path"] = result.trace_path
    if result.metadata:
        payload["metadata"] = result.metadata
    return payload


def register_embodied_action_tools(
    *,
    llm,
    dispatcher: EmbodiedCapabilityDispatcher,
    language: Language,
) -> ToolsSchema:
    """Register the finite daily-use embodied action tools."""
    registry = dispatcher.registry
    descriptions = _public_tool_descriptions(language)
    standard_tools: list[FunctionSchema] = []

    for definition in registry.public_definitions():
        tool_name = definition.tool_name or _tool_name_for_capability(definition.capability_id)

        async def handler(
            params: FunctionCallParams,
            *,
            capability_id: str = definition.capability_id,
        ):
            result = await dispatcher.execute_capability(
                capability_id,
                source="tool",
                arguments=params.arguments,
                correlation_id=params.tool_call_id,
                dispatch_mode=CapabilityDispatchMode.TOOL.value,
                metadata={"tool_call_id": params.tool_call_id},
            )
            payload = dict(result.output)
            payload.setdefault("accepted", result.accepted)
            payload.setdefault("summary", result.summary)
            payload.setdefault("warnings", list(result.warnings))
            if not result.accepted and result.error_code is not None:
                payload.setdefault("error_code", result.error_code)
            await params.result_callback(payload)

        llm.register_function(tool_name, handler)
        standard_tools.append(
            FunctionSchema(
                name=tool_name,
                description=descriptions.get(tool_name) or definition.tool_description or definition.description,
                properties={},
                required=[],
            )
        )

    return ToolsSchema(standard_tools=standard_tools)

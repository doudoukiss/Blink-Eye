"""Validated capability registry for Blink's bounded action vocabulary."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore

CapabilityPrecondition = Callable[
    ["BaseCapabilityInput", "CapabilityExecutionContext"],
    "CapabilityExecutionResult | None | Awaitable[CapabilityExecutionResult | None]",
]
CapabilityExecutor = Callable[
    ["BaseCapabilityInput", "CapabilityExecutionContext"],
    "CapabilityExecutionResult | Awaitable[CapabilityExecutionResult]",
]


class CapabilityFamily(str, Enum):
    """First-class bounded capability families."""

    ROBOT_HEAD = "robot_head"
    OBSERVATION = "observation"
    DIALOGUE = "dialogue"
    MAINTENANCE = "maintenance"
    REPORTING = "reporting"


class CapabilityDispatchMode(str, Enum):
    """Dispatch modes supported by the bounded capability registry."""

    TOOL = "tool"
    GOAL = "goal"
    INITIATIVE = "initiative"


class CapabilityUserTurnPolicy(str, Enum):
    """Turn-gating policy for initiative-capable capabilities."""

    ALLOWED = "allowed"
    REQUIRES_GAP = "requires_gap"
    FORBIDDEN = "forbidden"


class BaseCapabilityInput(BaseModel):
    """Base input model for bounded capability execution."""


class EmptyCapabilityInput(BaseCapabilityInput):
    """Empty input model for parameter-free capabilities."""


class CapabilityExecutionResult(BaseModel):
    """Outcome returned by one capability execution."""

    capability_id: str
    accepted: bool
    outcome: str
    summary: str
    error_code: str | None = None
    warnings: list[str] = Field(default_factory=list)
    retryable: bool = False
    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        *,
        capability_id: str,
        summary: str,
        output: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CapabilityExecutionResult":
        """Build a successful capability result."""
        return cls(
            capability_id=capability_id,
            accepted=True,
            outcome="completed",
            summary=summary,
            warnings=list(warnings or []),
            output=dict(output or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def ready(
        cls,
        *,
        capability_id: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> "CapabilityExecutionResult":
        """Build a successful preflight capability result."""
        return cls(
            capability_id=capability_id,
            accepted=True,
            outcome="ready",
            summary=summary,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def blocked(
        cls,
        *,
        capability_id: str,
        summary: str,
        error_code: str | None = None,
        retryable: bool = False,
        warnings: list[str] | None = None,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CapabilityExecutionResult":
        """Build a blocked capability result."""
        return cls(
            capability_id=capability_id,
            accepted=False,
            outcome="blocked",
            summary=summary,
            error_code=error_code,
            retryable=retryable,
            warnings=list(warnings or []),
            output=dict(output or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failed(
        cls,
        *,
        capability_id: str,
        summary: str,
        error_code: str | None = None,
        warnings: list[str] | None = None,
        output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CapabilityExecutionResult":
        """Build a failed capability result."""
        return cls(
            capability_id=capability_id,
            accepted=False,
            outcome="failed",
            summary=summary,
            error_code=error_code,
            warnings=list(warnings or []),
            output=dict(output or {}),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class CapabilityToolExposure:
    """Public tool-surface metadata for one capability."""

    name: str
    description: str


@dataclass(frozen=True)
class CapabilityInitiativePolicy:
    """Initiative/goal execution policy for one bounded capability."""

    enabled: bool
    allowed_goal_families: tuple[str, ...] = ()
    allowed_initiative_classes: tuple[str, ...] = ()
    user_turn_policy: str = CapabilityUserTurnPolicy.ALLOWED.value
    operator_visible: bool = False
    proactive_dialogue: bool = False


@dataclass(frozen=True)
class CapabilityAssistantUtterance:
    """One bounded assistant utterance emitted through a capability side effect."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CapabilitySideEffectSink(Protocol):
    """Typed sink for narrow bounded capability side effects."""

    async def emit_assistant_utterance(self, utterance: CapabilityAssistantUtterance):
        """Emit one short assistant utterance."""


@dataclass(frozen=True)
class CapabilityExecutionContext:
    """Runtime context available to capability executors and validators."""

    source: str
    session_ids: BrainSessionIds
    store: BrainStore | None = None
    presence_scope_key: str | None = None
    dispatch_mode: str = CapabilityDispatchMode.GOAL.value
    goal_family: str | None = None
    goal_intent: str | None = None
    initiative_class: str | None = None
    side_effect_sink: CapabilitySideEffectSink | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityDefinition:
    """One bounded capability exposed to Blink's executive or tool layers."""

    capability_id: str
    family: str
    description: str
    input_model: type[BaseCapabilityInput]
    sensitivity: str
    executor: CapabilityExecutor
    preconditions: tuple[CapabilityPrecondition, ...] = ()
    tool_exposure: CapabilityToolExposure | None = None
    initiative_policy: CapabilityInitiativePolicy | None = None

    @property
    def public(self) -> bool:
        """Return whether the capability is exposed as a public tool."""
        return self.tool_exposure is not None

    @property
    def tool_name(self) -> str | None:
        """Return the public tool name when one exists."""
        return self.tool_exposure.name if self.tool_exposure is not None else None

    @property
    def tool_description(self) -> str | None:
        """Return the public tool description when one exists."""
        return self.tool_exposure.description if self.tool_exposure is not None else None


class CapabilityRegistry:
    """Validated registry that owns planner-facing capability definitions."""

    def __init__(self):
        """Initialize the empty registry."""
        self._definitions: dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition):
        """Register one capability definition."""
        if definition.capability_id in self._definitions:
            raise ValueError(f"Capability '{definition.capability_id}' is already registered.")
        self._definitions[definition.capability_id] = definition

    def get(self, capability_id: str) -> CapabilityDefinition:
        """Return one capability definition by id."""
        definition = self._definitions.get(capability_id)
        if definition is None:
            raise KeyError(f"Unsupported capability '{capability_id}'.")
        return definition

    def definitions(self) -> list[CapabilityDefinition]:
        """Return definitions in stable registration order."""
        return list(self._definitions.values())

    def public_definitions(self) -> list[CapabilityDefinition]:
        """Return public capability definitions in stable registration order."""
        return [
            definition for definition in self._definitions.values() if definition.tool_exposure is not None
        ]

    def initiative_definitions(self) -> list[CapabilityDefinition]:
        """Return initiative-capable definitions in stable registration order."""
        return [
            definition
            for definition in self._definitions.values()
            if definition.initiative_policy is not None and definition.initiative_policy.enabled
        ]

    def internal_definitions(self) -> list[CapabilityDefinition]:
        """Return internal initiative-capable definitions not exposed as public tools."""
        return [
            definition
            for definition in self._definitions.values()
            if definition.tool_exposure is None
            and definition.initiative_policy is not None
            and definition.initiative_policy.enabled
        ]

    def public_capability_ids(self) -> list[str]:
        """Return public capability ids in stable registration order."""
        return [definition.capability_id for definition in self.public_definitions()]

    async def execute(
        self,
        capability_id: str,
        arguments: dict[str, Any] | None,
        *,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        """Validate and execute one capability request."""
        definition = self.get(capability_id)
        try:
            inputs = definition.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return CapabilityExecutionResult.failed(
                capability_id=capability_id,
                summary=f"Capability arguments are invalid for {capability_id}.",
                error_code="invalid_arguments",
                metadata={"validation_errors": exc.errors()},
            )

        gate_result = self._dispatch_gate(definition=definition, context=context)
        if gate_result is not None:
            return gate_result

        for precondition in definition.preconditions:
            maybe_result = precondition(inputs, context)
            result = await _maybe_await(maybe_result)
            if result is not None:
                if result.capability_id != capability_id:
                    result.capability_id = capability_id
                return result

        try:
            result = await _maybe_await(definition.executor(inputs, context))
        except Exception as exc:  # pragma: no cover - defensive guard
            return CapabilityExecutionResult.failed(
                capability_id=capability_id,
                summary=f"Capability execution crashed for {capability_id}: {exc}",
                error_code="executor_error",
            )

        if result.capability_id != capability_id:
            result.capability_id = capability_id
        return result

    async def evaluate_preconditions(
        self,
        capability_id: str,
        arguments: dict[str, Any] | None,
        *,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult:
        """Validate and preflight one capability request without executing it."""
        definition = self.get(capability_id)
        try:
            inputs = definition.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return CapabilityExecutionResult.failed(
                capability_id=capability_id,
                summary=f"Capability arguments are invalid for {capability_id}.",
                error_code="invalid_arguments",
                metadata={"validation_errors": exc.errors()},
            )

        gate_result = self._dispatch_gate(definition=definition, context=context)
        if gate_result is not None:
            return gate_result

        for precondition in definition.preconditions:
            maybe_result = precondition(inputs, context)
            result = await _maybe_await(maybe_result)
            if result is not None:
                if result.capability_id != capability_id:
                    result.capability_id = capability_id
                return result

        return CapabilityExecutionResult.ready(
            capability_id=capability_id,
            summary=f"Capability {capability_id} is ready to execute.",
            metadata={"preflight": True},
        )

    def _dispatch_gate(
        self,
        *,
        definition: CapabilityDefinition,
        context: CapabilityExecutionContext,
    ) -> CapabilityExecutionResult | None:
        """Apply dispatch-policy gates before custom preconditions execute."""
        dispatch_mode = str(context.dispatch_mode or CapabilityDispatchMode.GOAL.value)
        if dispatch_mode == CapabilityDispatchMode.TOOL.value:
            if definition.tool_exposure is None:
                return CapabilityExecutionResult.failed(
                    capability_id=definition.capability_id,
                    summary=f"Capability {definition.capability_id} is not exposed as a public tool.",
                    error_code="tool_exposure_required",
                )
            return None

        policy = definition.initiative_policy
        if policy is None or not policy.enabled:
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=f"Capability {definition.capability_id} is not enabled for goal execution.",
                error_code="initiative_policy_required",
            )
        if context.goal_family and (
            policy.allowed_goal_families and context.goal_family not in policy.allowed_goal_families
        ):
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=(
                    f"Capability {definition.capability_id} is not allowed for goal family "
                    f"{context.goal_family}."
                ),
                error_code="goal_family_not_allowed",
                metadata={"goal_family": context.goal_family},
            )
        if context.initiative_class and (
            policy.allowed_initiative_classes
            and context.initiative_class not in policy.allowed_initiative_classes
        ):
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=(
                    f"Capability {definition.capability_id} is not allowed for initiative class "
                    f"{context.initiative_class}."
                ),
                error_code="initiative_class_not_allowed",
                metadata={"initiative_class": context.initiative_class},
            )
        if dispatch_mode == CapabilityDispatchMode.INITIATIVE.value and not context.initiative_class:
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=f"Capability {definition.capability_id} requires initiative metadata.",
                error_code="initiative_class_required",
            )

        turn_gate = self._turn_policy_gate(definition=definition, context=context, policy=policy)
        if turn_gate is not None:
            return turn_gate

        if policy.proactive_dialogue and context.goal_intent != "autonomy.presence_brief_reengagement_speech":
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=(
                    f"Capability {definition.capability_id} is reserved for the bounded "
                    "re-engagement dialogue path."
                ),
                error_code="proactive_dialogue_not_allowed",
                metadata={"goal_intent": context.goal_intent},
            )
        return None

    def _turn_policy_gate(
        self,
        *,
        definition: CapabilityDefinition,
        context: CapabilityExecutionContext,
        policy: CapabilityInitiativePolicy,
    ) -> CapabilityExecutionResult | None:
        """Apply bounded turn-gating rules for goal/initiative execution."""
        turn_policy = str(policy.user_turn_policy or CapabilityUserTurnPolicy.ALLOWED.value)
        if turn_policy == CapabilityUserTurnPolicy.ALLOWED.value:
            return None
        if turn_policy == CapabilityUserTurnPolicy.FORBIDDEN.value:
            return CapabilityExecutionResult.failed(
                capability_id=definition.capability_id,
                summary=(
                    f"Capability {definition.capability_id} is not allowed through the "
                    "initiative goal path."
                ),
                error_code="turn_policy_forbidden",
            )
        if context.store is None:
            return None
        working_context = context.store.get_working_context_projection(scope_key=context.session_ids.thread_id)
        if working_context.user_turn_open:
            return CapabilityExecutionResult.blocked(
                capability_id=definition.capability_id,
                summary=f"Capability {definition.capability_id} is waiting for the user turn to close.",
                error_code="user_turn_open",
                retryable=True,
            )
        if working_context.assistant_turn_open:
            return CapabilityExecutionResult.blocked(
                capability_id=definition.capability_id,
                summary=(
                    f"Capability {definition.capability_id} is waiting for the assistant turn to close."
                ),
                error_code="assistant_turn_open",
                retryable=True,
            )
        return None


async def _maybe_await(value):
    """Await a value only when it is awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value

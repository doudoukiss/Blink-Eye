import pytest
from pydantic import BaseModel, Field

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    build_embodied_capability_registry,
)
from blink.brain.actions import (
    build_brain_capability_registry as build_brain_capability_registry_compat,
)
from blink.brain.autonomy import BrainInitiativeClass
from blink.brain.capabilities import (
    CapabilityAssistantUtterance,
    CapabilityDefinition,
    CapabilityDispatchMode,
    CapabilityExecutionContext,
    CapabilityExecutionResult,
    CapabilityInitiativePolicy,
    CapabilityRegistry,
    CapabilityToolExposure,
    CapabilityUserTurnPolicy,
)
from blink.brain.capability_registry import build_brain_capability_registry
from blink.brain.events import BrainEventType
from blink.brain.projections import BrainGoalFamily
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver
from blink.transcriptions.language import Language


class DemoInput(BaseModel):
    count: int = Field(ge=1, le=2)


class RecordingSink:
    def __init__(self):
        self.utterances: list[CapabilityAssistantUtterance] = []

    async def emit_assistant_utterance(self, utterance: CapabilityAssistantUtterance):
        self.utterances.append(utterance)


def test_canonical_capability_registry_builder_matches_actions_compat_surface():
    canonical = build_brain_capability_registry(language=Language.EN)
    compat = build_brain_capability_registry_compat(language=Language.EN)

    assert [definition.capability_id for definition in canonical.definitions()] == [
        definition.capability_id for definition in compat.definitions()
    ]


@pytest.mark.asyncio
async def test_capability_registry_validates_inputs_and_preconditions():
    registry = CapabilityRegistry()

    async def precondition(inputs: DemoInput, context: CapabilityExecutionContext):
        if inputs.count == 2:
            return CapabilityExecutionResult.blocked(
                capability_id="demo.count",
                summary="Count 2 is blocked for this demo.",
                error_code="demo_blocked",
            )
        return None

    async def executor(inputs: DemoInput, context: CapabilityExecutionContext):
        return CapabilityExecutionResult.success(
            capability_id="demo.count",
            summary=f"Executed demo count {inputs.count}.",
            output={"count": inputs.count, "source": context.source},
        )

    registry.register(
        CapabilityDefinition(
            capability_id="demo.count",
            family="demo",
            description="Demo capability used for validation tests.",
            input_model=DemoInput,
            sensitivity="safe",
            executor=executor,
            preconditions=(precondition,),
            tool_exposure=CapabilityToolExposure(
                name="demo_count",
                description="Count with validation.",
            ),
            initiative_policy=CapabilityInitiativePolicy(
                enabled=True,
                allowed_goal_families=(BrainGoalFamily.CONVERSATION.value,),
                user_turn_policy=CapabilityUserTurnPolicy.ALLOWED.value,
            ),
        )
    )
    context = CapabilityExecutionContext(
        source="test",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        goal_family=BrainGoalFamily.CONVERSATION.value,
    )

    accepted = await registry.execute("demo.count", {"count": 1}, context=context)
    invalid = await registry.execute("demo.count", {"count": 0}, context=context)
    blocked = await registry.execute("demo.count", {"count": 2}, context=context)

    assert accepted.accepted is True
    assert accepted.output["count"] == 1
    assert invalid.accepted is False
    assert invalid.error_code == "invalid_arguments"
    assert blocked.accepted is False
    assert blocked.outcome == "blocked"
    assert blocked.error_code == "demo_blocked"


@pytest.mark.asyncio
async def test_robot_head_actions_are_exposed_through_the_capability_registry():
    driver = MockDriver()
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        session_resolver=lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
    )
    registry = build_embodied_capability_registry(action_engine=action_engine)
    context = CapabilityExecutionContext(
        source="tool",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.TOOL.value,
    )

    assert registry.public_capability_ids() == [
        "robot_head.blink",
        "robot_head.wink_left",
        "robot_head.wink_right",
        "robot_head.look_left",
        "robot_head.look_right",
        "robot_head.return_neutral",
        "robot_head.status",
    ]

    result = await registry.execute("robot_head.blink", {}, context=context)

    assert result.accepted is True
    assert result.output["action_id"] == "cmd_blink"
    assert [plan.resolved_name for plan in driver.executed_plans] == ["blink"]
    await controller.close()


@pytest.mark.asyncio
async def test_internal_robot_head_policy_actions_support_goal_dispatch():
    driver = MockDriver()
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        session_resolver=lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
    )
    registry = build_embodied_capability_registry(action_engine=action_engine)
    context = CapabilityExecutionContext(
        source="policy",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.GOAL.value,
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_intent="robot_head.sequence",
    )

    result = await registry.execute("robot_head.auto_listen_user", {}, context=context)

    assert result.accepted is True
    assert result.output["action_id"] == "auto_listen_user"
    assert [plan.resolved_name for plan in driver.executed_plans] == [
        "listen_engage",
        "listen_attentively",
    ]
    await controller.close()


@pytest.mark.asyncio
async def test_internal_capabilities_are_not_exposed_as_public_tools():
    registry = build_brain_capability_registry(language=Language.EN)

    public_ids = registry.public_capability_ids()

    assert "observation.inspect_presence_state" not in public_ids
    assert "dialogue.emit_brief_reengagement" not in public_ids
    assert "maintenance.review_memory_health" not in public_ids
    assert "reporting.record_presence_event" not in public_ids


@pytest.mark.asyncio
async def test_tool_dispatch_rejects_internal_only_capability():
    registry = build_brain_capability_registry(language=Language.EN)
    context = CapabilityExecutionContext(
        source="tool",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.TOOL.value,
    )

    result = await registry.execute(
        "observation.inspect_presence_state",
        {"presence_scope_key": "browser:presence"},
        context=context,
    )

    assert result.accepted is False
    assert result.error_code == "tool_exposure_required"


@pytest.mark.asyncio
async def test_goal_dispatch_rejects_goal_family_mismatch():
    registry = build_brain_capability_registry(language=Language.EN)
    context = CapabilityExecutionContext(
        source="executive",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.GOAL.value,
        goal_family=BrainGoalFamily.CONVERSATION.value,
        goal_intent="autonomy.presence_user_reentered",
    )

    result = await registry.execute(
        "observation.inspect_presence_state",
        {"presence_scope_key": "browser:presence"},
        context=context,
    )

    assert result.accepted is False
    assert result.error_code == "goal_family_not_allowed"


@pytest.mark.asyncio
async def test_proactive_dialogue_capability_requires_turn_gap(tmp_path):
    brain_store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    brain_store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    registry = build_brain_capability_registry(language=Language.EN)
    context = CapabilityExecutionContext(
        source="executive",
        session_ids=session_ids,
        store=brain_store,
        dispatch_mode=CapabilityDispatchMode.GOAL.value,
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_intent="autonomy.presence_brief_reengagement_speech",
        initiative_class="speak_briefly_if_idle",
    )

    result = await registry.execute(
        "dialogue.emit_brief_reengagement",
        {"presence_scope_key": "browser:presence"},
        context=context,
    )

    assert result.accepted is False
    assert result.outcome == "blocked"
    assert result.error_code == "user_turn_open"


@pytest.mark.asyncio
async def test_proactive_dialogue_capability_emits_deterministic_utterance_when_allowed():
    registry = build_brain_capability_registry(language=Language.EN)
    sink = RecordingSink()
    context = CapabilityExecutionContext(
        source="executive",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.GOAL.value,
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_intent="autonomy.presence_brief_reengagement_speech",
        initiative_class=BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
        side_effect_sink=sink,
    )

    result = await registry.execute(
        "dialogue.emit_brief_reengagement",
        {"presence_scope_key": "browser:presence"},
        context=context,
    )

    assert result.accepted is True
    assert result.output["utterance"] == "Welcome back."
    assert [utterance.text for utterance in sink.utterances] == ["Welcome back."]


@pytest.mark.asyncio
async def test_proactive_dialogue_capability_blocks_when_sink_is_missing():
    registry = build_brain_capability_registry(language=Language.EN)
    context = CapabilityExecutionContext(
        source="executive",
        session_ids=resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123"),
        dispatch_mode=CapabilityDispatchMode.GOAL.value,
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_intent="autonomy.presence_brief_reengagement_speech",
        initiative_class=BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value,
    )

    result = await registry.execute(
        "dialogue.emit_brief_reengagement",
        {"presence_scope_key": "browser:presence"},
        context=context,
    )

    assert result.accepted is False
    assert result.outcome == "blocked"
    assert result.error_code == "dialogue_sink_unavailable"

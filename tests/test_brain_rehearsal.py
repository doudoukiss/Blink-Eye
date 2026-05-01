import json

import pytest

from blink.brain._executive import BrainPlanningOutcome
from blink.brain.events import BrainEventType
from blink.brain.identity import base_brain_system_prompt
from blink.brain.projections import BrainGoalFamily, BrainGoalStatus, BrainPlanReviewPolicy
from blink.brain.replay import BrainReplayHarness, BrainReplayScenario
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import FaultInjectionDriver, PreviewDriver
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def _build_runtime(tmp_path, *, client_id: str, driver):
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id=client_id)
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.EN),
        language=Language.EN,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        robot_head_controller=controller,
        brain_db_path=tmp_path / f"{client_id}.db",
    )
    return runtime, controller, session_ids


def _create_robot_goal(runtime: BrainRuntime, *, title: str) -> str:
    return runtime.executive.create_commitment_goal(
        title=title,
        intent="robot_head.sequence",
        source="test",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={
            "survive_restart": True,
            "capabilities": ["robot_head.look_left"],
        },
    )


@pytest.mark.asyncio
async def test_rehearsal_busy_robot_head_floors_planning_to_operator_review(tmp_path):
    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="rehearsal-busy",
        driver=FaultInjectionDriver(busy=True),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Look left when the head is busy")

        result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        agenda = runtime.store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        goal = agenda.goal(goal_id)
        shell_snapshot = runtime.shell.snapshot()

        assert result.outcome == BrainPlanningOutcome.NEEDS_OPERATOR_REVIEW.value
        assert result.decision.reason == "rehearsal_requires_operator_review"
        assert result.proposal.review_policy == BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value
        assert result.proposal.details["rehearsal_operator_review_floor"] is True
        assert goal is not None
        assert goal.details["counterfactual_rehearsal"]["decision_recommendation"] == "wait"
        assert goal.details["rehearsal_by_step"]["0"]["candidate_action_id"] == "cmd_look_left"
        assert shell_snapshot.rehearsal_digest["recommendation_counts"] == {"wait": 1}
    finally:
        await controller.close()
        runtime.close()


@pytest.mark.asyncio
async def test_rehearsal_comparison_and_shell_digest_are_explicit(tmp_path):
    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="rehearsal-preview",
        driver=PreviewDriver(trace_dir=tmp_path / "preview"),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Look left through preview rehearsal")

        planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        cycle_result = await runtime.executive.run_once()
        rehearsal_projection = runtime.store.build_counterfactual_rehearsal_projection(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            presence_scope_key=runtime.presence_scope_key,
        )
        events = list(
            reversed(
                runtime.store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=64,
                )
            )
        )
        comparison_event = next(
            event for event in events if event.event_type == BrainEventType.ACTION_OUTCOME_COMPARED
        )
        robot_outcome = next(
            event for event in events if event.event_id == comparison_event.causal_parent_id
        )
        shell_snapshot = runtime.shell.snapshot()
        shell_digest = runtime.shell.runtime_shell_digest()

        assert planning_result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
        assert cycle_result.progressed is True
        assert rehearsal_projection.recent_rehearsals
        assert rehearsal_projection.recent_rehearsals[0].candidate_action_id == "cmd_look_left"
        assert rehearsal_projection.recent_comparisons
        assert rehearsal_projection.recent_comparisons[0].observed_outcome_kind == "preview_only"
        assert rehearsal_projection.recent_comparisons[0].calibration_bucket == "not_calibrated"
        assert comparison_event.causal_parent_id == robot_outcome.event_id
        assert shell_snapshot.rehearsal_digest["recent_comparison_count"] == 1
        assert shell_digest["rehearsal_inspection"]["calibration_bucket_counts"] == {
            "not_calibrated": 1
        }
    finally:
        await controller.close()
        runtime.close()


@pytest.mark.asyncio
async def test_replay_rebuilds_counterfactual_rehearsal_projection(tmp_path):
    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="rehearsal-replay",
        driver=PreviewDriver(trace_dir=tmp_path / "preview-replay"),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Replay one rehearsed embodied plan")
        planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        cycle_result = await runtime.executive.run_once()

        harness = BrainReplayHarness(store=runtime.store)
        scenario = BrainReplayScenario(
            name="phase18b_counterfactual_rehearsal",
            description="phase18b counterfactual rehearsal",
            session_ids=session_ids,
            events=tuple(
                reversed(
                    runtime.store.recent_brain_events(
                        user_id=session_ids.user_id,
                        thread_id=session_ids.thread_id,
                        limit=96,
                    )
                )
            ),
        )
        replayed = harness.replay(scenario, presence_scope_key=runtime.presence_scope_key)
        payload = json.loads(replayed.artifact_path.read_text(encoding="utf-8"))
        continuity_state = dict(payload.get("continuity_state", {}))
        event_types = [event["event_type"] for event in payload["events"]]

        assert planning_result.outcome == BrainPlanningOutcome.AUTO_ADOPTED.value
        assert cycle_result.progressed is True
        assert continuity_state["rehearsal_digest"]["recent_comparison_count"] == 1
        assert continuity_state["runtime_shell_digest"]["rehearsal_inspection"][
            "calibration_bucket_counts"
        ] == {"not_calibrated": 1}
        assert event_types.index(BrainEventType.ACTION_REHEARSAL_REQUESTED) < event_types.index(
            BrainEventType.ACTION_REHEARSAL_COMPLETED
        )
        assert event_types.index(BrainEventType.ROBOT_ACTION_OUTCOME) < event_types.index(
            BrainEventType.ACTION_OUTCOME_COMPARED
        )
    finally:
        await controller.close()
        runtime.close()

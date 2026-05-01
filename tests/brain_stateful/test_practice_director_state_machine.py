from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain.evals.dataset_manifest import build_episode_dataset_manifest
from blink.brain.practice_director import build_practice_plan
from tests.phase23_fixtures import make_episode

pytestmark = pytest.mark.brain_stateful


class PracticeDirectorStateMachine(RuleBasedStateMachine):
    """Exercise deterministic bounded practice selection from episode evidence."""

    def __init__(self):
        super().__init__()
        self.episodes = []
        self._counter = 0
        self.procedural_skill_governance_report = {
            "low_confidence_skill_ids": ["skill-low"],
            "retired_skill_ids": ["skill-retired"],
        }

    @rule(
        scenario_family=st.sampled_from(
            [
                "robot_head_single_step",
                "robot_head_multi_step",
                "robot_head_degraded_backend_comparison",
                "robot_head_busy_fault",
            ]
        ),
        skill_kind=st.sampled_from(["none", "low", "retired"]),
        overconfident=st.booleans(),
        review_floor_count=st.integers(min_value=0, max_value=1),
        recovery_count=st.integers(min_value=0, max_value=1),
        task_success=st.booleans(),
    )
    def add_episode(
        self,
        scenario_family: str,
        skill_kind: str,
        overconfident: bool,
        review_floor_count: int,
        recovery_count: int,
        task_success: bool,
    ):
        self._counter += 1
        skill_ids = {
            "none": (),
            "low": ("skill-low",),
            "retired": ("skill-retired",),
        }[skill_kind]
        execution_backend = "fault" if scenario_family == "robot_head_busy_fault" else "simulation"
        self.episodes.append(
            make_episode(
                index=self._counter,
                scenario_family=scenario_family,
                skill_ids=skill_ids,
                execution_backend=execution_backend,
                task_success=task_success,
                operator_review_floored=review_floor_count > 0,
                review_floor_count=review_floor_count,
                recovery_count=recovery_count,
                safety_success=task_success and review_floor_count == 0,
                calibration_bucket_counts={"overconfident": 1} if overconfident else {"aligned": 1},
                mismatch_codes=(("robot_head_busy",) if not task_success else ()),
            )
        )

    @invariant()
    def plans_stay_bounded_and_deterministic(self):
        manifest = build_episode_dataset_manifest(self.episodes)
        forward = build_practice_plan(
            episodes=self.episodes,
            dataset_manifest=manifest,
            procedural_skill_governance_report=self.procedural_skill_governance_report,
            scope_key="thread-1",
            presence_scope_key="browser:presence",
        )
        backward = build_practice_plan(
            episodes=list(reversed(self.episodes)),
            dataset_manifest=build_episode_dataset_manifest(list(reversed(self.episodes))),
            procedural_skill_governance_report=self.procedural_skill_governance_report,
            scope_key="thread-1",
            presence_scope_key="browser:presence",
        )
        assert forward.as_dict() == backward.as_dict()
        assert len(forward.targets) <= 3
        assert all(target.execution_backend == "simulation" for target in forward.targets)
        assert all(target.scenario_family != "robot_head_busy_fault" for target in forward.targets)


TestPracticeDirectorStateMachine = PracticeDirectorStateMachine.TestCase
TestPracticeDirectorStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)

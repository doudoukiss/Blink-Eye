from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from blink.brain.evals.failure_clusters import build_failure_clusters
from blink.brain.memory_v2 import BrainProceduralSkillStatus
from blink.brain.memory_v2.skill_evidence import build_skill_evidence_ledger
from blink.brain.memory_v2.skill_promotion import build_skill_governance_projection
from tests.phase23_fixtures import make_episode, make_procedural_skills

pytestmark = pytest.mark.brain_stateful


class SkillPromotionStateMachine(RuleBasedStateMachine):
    """Exercise conservative promotion and demotion proposal lifecycles."""

    def __init__(self):
        super().__init__()
        self._counter = 0
        self.episodes = []
        self.procedural_skills = make_procedural_skills(
            {
                "skill_id": "skill-alpha",
                "status": BrainProceduralSkillStatus.ACTIVE.value,
                "confidence": 0.8,
            }
        )

    @rule(
        scenario_family=st.sampled_from(["robot_head_single_step", "robot_head_multi_step"]),
        overconfident=st.booleans(),
    )
    def add_supporting_episode(self, scenario_family: str, overconfident: bool):
        self._counter += 1
        self.episodes.append(
            make_episode(
                index=self._counter,
                scenario_family=scenario_family,
                skill_ids=("skill-alpha",),
                task_success=True,
                calibration_bucket_counts={"overconfident": 1} if overconfident else {"aligned": 1},
            )
        )

    @rule(
        critical_safety=st.booleans(),
        review_floor_count=st.integers(min_value=0, max_value=1),
        recovery_count=st.integers(min_value=0, max_value=1),
    )
    def add_failure_episode(
        self,
        critical_safety: bool,
        review_floor_count: int,
        recovery_count: int,
    ):
        self._counter += 1
        self.episodes.append(
            make_episode(
                index=self._counter,
                scenario_family="robot_head_degraded_backend_comparison",
                skill_ids=("skill-alpha",),
                task_success=False,
                operator_review_floored=review_floor_count > 0,
                review_floor_count=review_floor_count,
                recovery_count=recovery_count,
                safety_success=not critical_safety,
                calibration_bucket_counts={"underconfident": 1},
                risk_codes=(("unsafe",) if critical_safety else ()),
                mismatch_codes=("robot_head_busy",),
            )
        )

    @invariant()
    def proposals_remain_unique_and_monotonic(self):
        ledger = build_skill_evidence_ledger(
            episodes=self.episodes,
            procedural_skills=self.procedural_skills,
            failure_clusters=build_failure_clusters(self.episodes),
            scope_type="thread",
            scope_id="thread-1",
        )
        governance = build_skill_governance_projection(skill_evidence_ledger=ledger)
        promotion_keys = [proposal.proposal_key for proposal in governance.promotion_proposals]
        demotion_keys = [proposal.proposal_key for proposal in governance.demotion_proposals]
        assert len(set(promotion_keys)) == len(promotion_keys)
        assert len(set(demotion_keys)) == len(demotion_keys)

        promotions = [proposal for proposal in governance.promotion_proposals if proposal.skill_id == "skill-alpha"]
        demotions = [proposal for proposal in governance.demotion_proposals if proposal.skill_id == "skill-alpha"]
        if demotions:
            assert all(proposal.status != "proposed" for proposal in promotions)
        if any(
            int(record.critical_safety_violation_count) > 0 for record in ledger.evidence_records
        ):
            assert demotions


TestSkillPromotionStateMachine = SkillPromotionStateMachine.TestCase
TestSkillPromotionStateMachine.settings = settings(
    stateful_step_count=8,
    max_examples=20,
    deadline=None,
)

from __future__ import annotations

from blink.brain.evals.failure_clusters import build_failure_clusters
from blink.brain.memory_v2 import BrainProceduralSkillStatus
from blink.brain.memory_v2.skill_evidence import build_skill_evidence_ledger
from blink.brain.memory_v2.skill_promotion import build_skill_governance_projection
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from tests.phase23_fixtures import (
    append_skill_governance_events,
    make_episode,
    make_procedural_skills,
)


def test_skill_evidence_ledger_aggregates_only_known_skills():
    procedural_skills = make_procedural_skills(
        {"skill_id": "skill-alpha", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.8}
    )
    episodes = [
        make_episode(
            index=1,
            scenario_family="robot_head_single_step",
            skill_ids=("skill-alpha",),
        ),
        make_episode(
            index=2,
            scenario_family="robot_head_multi_step",
            skill_ids=("skill-alpha",),
        ),
        make_episode(
            index=3,
            scenario_family="robot_head_degraded_backend_comparison",
            skill_ids=(),
            task_success=False,
            safety_success=False,
            mismatch_codes=("robot_head_busy",),
        ),
    ]

    ledger = build_skill_evidence_ledger(
        episodes=episodes,
        procedural_skills=procedural_skills,
        failure_clusters=[],
        scope_type="thread",
        scope_id="thread-1",
    )

    assert len(ledger.evidence_records) == 1
    record = ledger.evidence_records[0]
    assert record.skill_id == "skill-alpha"
    assert record.support_episode_count == 2
    assert set(record.scenario_families) == {"robot_head_multi_step", "robot_head_single_step"}
    assert ledger.family_hypothesis_counts["robot_head_degraded_backend_comparison"] == 1


def test_skill_governance_blocks_and_demotes_conservatively():
    procedural_skills = make_procedural_skills(
        {"skill_id": "skill-alpha", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.81},
        {"skill_id": "skill-beta", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.74},
    )
    episodes = [
        make_episode(index=1, scenario_family="robot_head_single_step", skill_ids=("skill-alpha",)),
        make_episode(index=2, scenario_family="robot_head_multi_step", skill_ids=("skill-alpha",)),
        make_episode(
            index=3,
            scenario_family="robot_head_single_step",
            scenario_id="robot_head_single_step-repeat",
            skill_ids=("skill-alpha",),
        ),
        make_episode(
            index=4,
            scenario_family="robot_head_degraded_backend_comparison",
            skill_ids=("skill-alpha",),
            task_success=False,
            safety_success=False,
            review_floor_count=1,
            operator_review_floored=True,
            mismatch_codes=("robot_head_busy",),
        ),
        make_episode(
            index=5,
            scenario_family="robot_head_degraded_backend_comparison",
            scenario_id="robot_head_degraded-repeat",
            skill_ids=("skill-alpha",),
            task_success=False,
            safety_success=False,
            recovery_count=1,
            mismatch_codes=("robot_head_busy",),
        ),
        make_episode(
            index=6,
            scenario_family="robot_head_single_step",
            skill_ids=("skill-beta",),
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=7,
            scenario_family="robot_head_multi_step",
            skill_ids=("skill-beta",),
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=8,
            scenario_family="robot_head_single_step",
            scenario_id="robot_head_single_step-overconfident-repeat",
            skill_ids=("skill-beta",),
            calibration_bucket_counts={"overconfident": 1},
        ),
    ]
    ledger = build_skill_evidence_ledger(
        episodes=episodes,
        procedural_skills=procedural_skills,
        failure_clusters=build_failure_clusters(episodes),
        scope_type="thread",
        scope_id="thread-1",
    )
    governance = build_skill_governance_projection(skill_evidence_ledger=ledger)

    promotion_by_skill = {
        proposal.skill_id: proposal for proposal in governance.promotion_proposals
    }
    demotion_by_skill = {proposal.skill_id: proposal for proposal in governance.demotion_proposals}

    assert promotion_by_skill["skill-beta"].status == "blocked"
    assert "overconfident_calibration_dominant" in promotion_by_skill["skill-beta"].blocked_reason_codes
    assert "skill-alpha" in demotion_by_skill
    assert "repeated_failure_cluster" in demotion_by_skill["skill-alpha"].reason_codes
    assert promotion_by_skill["skill-alpha"].status == "superseded"


def test_skill_evidence_and_governance_events_update_store_projections(tmp_path):
    procedural_skills = make_procedural_skills(
        {"skill_id": "skill-alpha", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.8}
    )
    episodes = [
        make_episode(index=1, scenario_family="robot_head_single_step", skill_ids=("skill-alpha",)),
        make_episode(index=2, scenario_family="robot_head_multi_step", skill_ids=("skill-alpha",)),
        make_episode(index=3, scenario_family="robot_head_single_step", skill_ids=("skill-alpha",)),
    ]
    ledger = build_skill_evidence_ledger(
        episodes=episodes,
        procedural_skills=procedural_skills,
        failure_clusters=[],
        scope_type="thread",
        scope_id="thread-1",
    )
    governance = build_skill_governance_projection(skill_evidence_ledger=ledger)
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase23-skill")

    append_skill_governance_events(
        store=store,
        session_ids=session_ids,
        scope_key=session_ids.thread_id,
        ledger=ledger,
        governance=governance,
    )

    evidence_projection = store.build_skill_evidence_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )
    governance_projection = store.build_skill_governance_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )
    assert evidence_projection.evidence_records[0].skill_id == "skill-alpha"
    assert governance_projection.promotion_proposals[0].skill_id == "skill-alpha"
    store.close()

from __future__ import annotations

from pathlib import Path

from blink.brain.evals.dataset_manifest import build_episode_dataset_manifest
from blink.brain.practice_director import BrainPracticeDirector, build_practice_plan
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from tests.phase23_fixtures import make_episode


def test_practice_plan_selection_is_deterministic_and_simulation_only():
    episodes = [
        make_episode(
            index=1,
            scenario_family="robot_head_single_step",
            skill_ids=("skill-low",),
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=2,
            scenario_family="robot_head_multi_step",
            skill_ids=("skill-retired",),
            review_floor_count=1,
            operator_review_floored=True,
        ),
        make_episode(
            index=3,
            scenario_family="robot_head_busy_fault",
            execution_backend="fault",
            task_success=False,
            safety_success=False,
            calibration_bucket_counts={"overconfident": 1},
            risk_codes=("unsafe",),
            mismatch_codes=("robot_head_busy",),
        ),
    ]
    manifest = build_episode_dataset_manifest(episodes)
    report = {
        "low_confidence_skill_ids": ["skill-low"],
        "retired_skill_ids": ["skill-retired"],
    }

    first = build_practice_plan(
        episodes=episodes,
        dataset_manifest=manifest,
        procedural_skill_governance_report=report,
        scope_key="thread-1",
        presence_scope_key="browser:presence",
    )
    second = build_practice_plan(
        episodes=list(reversed(episodes)),
        dataset_manifest=build_episode_dataset_manifest(list(reversed(episodes))),
        procedural_skill_governance_report=report,
        scope_key="thread-1",
        presence_scope_key="browser:presence",
    )

    assert first.as_dict() == second.as_dict()
    assert 0 < len(first.targets) <= 3
    assert all(target.execution_backend == "simulation" for target in first.targets)
    assert all(target.scenario_family != "robot_head_busy_fault" for target in first.targets)
    assert {
        reason
        for target in first.targets
        for reason in target.reason_codes
    }.intersection({"low_confidence_skill_link", "retired_skill_link"})


def test_practice_director_records_artifacts_and_projection(tmp_path: Path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase23-practice")
    episodes = [
        make_episode(
            index=1,
            scenario_family="robot_head_single_step",
            skill_ids=("skill-low",),
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=2,
            scenario_family="robot_head_multi_step",
            skill_ids=("skill-retired",),
            review_floor_count=1,
            operator_review_floored=True,
        ),
    ]
    manifest = build_episode_dataset_manifest(episodes)
    artifacts_dir = tmp_path / "practice_artifacts"

    plan = BrainPracticeDirector(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    ).create_plan(
        episodes=episodes,
        dataset_manifest=manifest,
        procedural_skill_governance_report={
            "low_confidence_skill_ids": ["skill-low"],
            "retired_skill_ids": ["skill-retired"],
        },
        scope_key=session_ids.thread_id,
        output_dir=artifacts_dir,
    )

    assert Path(plan.artifact_paths["practice_plan_json"]).exists()
    assert Path(plan.artifact_paths["practice_plan_markdown"]).exists()

    projection = store.build_practice_director_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key="browser:presence",
    )
    assert projection.recent_plan_ids[0] == plan.plan_id
    assert projection.recent_plans[0].artifact_paths["practice_plan_json"].endswith(".json")
    recent_events = store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )
    assert recent_events[0].event_type == "practice.plan.created"
    store.close()

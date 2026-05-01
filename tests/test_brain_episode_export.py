from __future__ import annotations

import json
from pathlib import Path

import pytest

from blink.brain.evals.embodied_arena import EmbodiedEvalArena
from blink.brain.evals.embodied_scenarios import build_smoke_embodied_eval_suite
from blink.brain.evals.live_episode_export import build_episode_from_live_runtime_payload
from blink.brain.evals.practice_episode_export import build_episodes_from_practice_plan_payload
from blink.brain.events import BrainEventType
from blink.brain.identity import load_default_agent_blocks
from blink.brain.practice_director import BrainPracticePlan, BrainPracticeTarget
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import BrainGoal, BrainGoalFamily, BrainGoalStep
from blink.brain.replay import BrainReplayHarness
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.cli.export_episodes import main as export_episodes_main
from blink.transcriptions.language import Language


def _episode_payloads(output_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "episodes").glob("*.json"))
    ]


def _build_live_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "live_turn.json"
    payload = {
        "runtime_kind": "browser",
        "turn_id": "turn-live-1",
        "source_runtime_episode_id": "runtime-episode-1",
        "runtime_artifact_id": "runtime-artifact-1",
        "context_packet_id": "context-packet-1",
        "memory_use_trace_id": "memory-trace-1",
        "scenario_id": "browser_help_turn",
        "scenario_family": "live_runtime_turn",
        "scenario_version": "live.v1",
        "metrics": {
            "execution_backend": "browser",
            "task_success": True,
            "safety_success": True,
        },
        "artifact_paths": {
            "operator_snapshot_json": str(tmp_path / "operator.json"),
            "raw_audio_wav": str(tmp_path / "turn.wav"),
            "brain_db": str(tmp_path / "brain.db"),
        },
        "events": [
            {
                "event_id": "evt-live-001",
                "event_type": "user.turn.transcribed",
                "ts": "2026-04-23T00:00:01Z",
                "payload": {"text": "RAW USER TRANSCRIPT SHOULD NOT LEAK"},
            },
            {
                "event_id": "evt-live-002",
                "event_type": "attention.changed",
                "ts": "2026-04-23T00:00:02Z",
                "payload": {"summary": "Attention shifted to the active user turn."},
            },
            {
                "event_id": "evt-live-003",
                "event_type": "assistant.turn.ended",
                "ts": "2026-04-23T00:00:03Z",
                "payload": {"text": "RAW ASSISTANT RESPONSE SHOULD NOT LEAK"},
            },
        ],
    }
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path


def _build_practice_artifact(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "practice_plan.json"
    plan = BrainPracticePlan(
        plan_id="practice-plan-1",
        scope_key="scope:test",
        presence_scope_key="browser:presence",
        dataset_manifest_id="dataset-manifest-1",
        targets=[
            BrainPracticeTarget(
                target_id="practice-target-1",
                scenario_family="robot_head_single_step",
                scenario_id="robot_head_look_left_compare",
                scenario_version="scenario.v1",
                suite_id="smoke",
                selected_profile_id="simulation_profile",
                execution_backend="simulation",
                score=4.5,
                reason_codes=["low_family_coverage"],
                supporting_episode_ids=["episode-support-1"],
                failure_cluster_ids=["cluster-1"],
                related_skill_ids=["skill-look-left"],
            ),
            BrainPracticeTarget(
                target_id="practice-target-2",
                scenario_family="conversation_repair",
                scenario_id="conversation_repair_clarify",
                scenario_version="scenario.v1",
                suite_id="smoke",
                selected_profile_id="simulation_profile",
                execution_backend="simulation",
                score=3.0,
                reason_codes=["failure_cluster_pressure"],
                supporting_episode_ids=["episode-support-2"],
                related_skill_ids=["skill-clarify"],
            ),
        ],
        supporting_episode_ids=["episode-support-1", "episode-support-2"],
        artifact_paths={
            "practice_plan_json": str(artifact_path),
            "practice_plan_markdown": str(tmp_path / "practice_plan.md"),
            "raw_video_mp4": str(tmp_path / "practice.mp4"),
        },
        created_at="2026-04-23T00:00:00Z",
        updated_at="2026-04-23T00:00:10Z",
    )
    artifact_path.write_text(
        json.dumps(plan.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def _build_replay_artifact(tmp_path: Path) -> Path:
    store = BrainStore(path=tmp_path / "brain.db")
    try:
        store.ensure_default_blocks(load_default_agent_blocks())
        session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="episode-export")
        store.append_brain_event(
            event_type=BrainEventType.BODY_STATE_UPDATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="runtime",
            payload={
                "scope_key": "browser:presence",
                "snapshot": BrainPresenceSnapshot(
                    runtime_kind="browser",
                    robot_head_enabled=True,
                    robot_head_mode="simulation",
                    robot_head_available=True,
                    vision_enabled=True,
                    vision_connected=True,
                ).as_dict(),
            },
        )
        store.append_brain_event(
            event_type=BrainEventType.USER_TURN_TRANSCRIBED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="context",
            payload={"text": "Look left and confirm the result."},
        )
        store.append_brain_event(
            event_type=BrainEventType.GOAL_CREATED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="memory",
            payload={
                "goal": BrainGoal(
                    goal_id="goal-replay-1",
                    title="Look left",
                    intent="environment.inspect",
                    source="memory",
                    goal_family=BrainGoalFamily.ENVIRONMENT.value,
                    status="open",
                    steps=[BrainGoalStep(capability_id="look_left")],
                ).as_dict()
            },
        )
        store.append_brain_event(
            event_type=BrainEventType.ROBOT_ACTION_OUTCOME,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="tool",
            payload={
                "presence_scope_key": "browser:presence",
                "action_id": "cmd_look_left",
                "accepted": True,
                "preview_only": False,
                "summary": "Robot head executed look left.",
                "status": {
                    "mode": "simulation",
                    "armed": False,
                    "available": True,
                    "warnings": [],
                    "details": {"driver": "simulation"},
                },
            },
        )
        harness = BrainReplayHarness(store=store)
        scenario = harness.capture_builtin_scenario(
            name="phase1_turn_tool_robot_action",
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        result = harness.replay(
            scenario,
            artifact_path=tmp_path / "replay.json",
            presence_scope_key="browser:presence",
        )
        return result.artifact_path
    finally:
        store.close()


@pytest.mark.asyncio
async def test_export_episodes_cli_exports_one_episode_from_embodied_eval_run(tmp_path):
    suite = build_smoke_embodied_eval_suite()
    scenario = suite.scenario("robot_head_look_left_compare")
    assert scenario is not None

    report = await EmbodiedEvalArena(language=Language.EN).run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "compare",
    )
    run_json = Path(str(report.runs[0].artifact_paths["run_json"]))
    output_dir = tmp_path / "episode_export_run"

    assert (
        export_episodes_main(
            ["--source", "embodied-eval", "--input", str(run_json), "--out", str(output_dir)]
        )
        == 0
    )

    episodes = _episode_payloads(output_dir)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["origin"] == "simulation"
    assert episode["source_run_id"] == report.runs[0].run_id
    assert episode["scenario_id"] == report.runs[0].scenario_id
    assert any(ref["artifact_kind"] == "simulation_trace_dir" for ref in episode["artifact_refs"])
    assert all("content" not in ref for ref in episode["artifact_refs"])
    assert all("bytes" not in ref for ref in episode["artifact_refs"])


@pytest.mark.asyncio
async def test_export_episodes_cli_builds_family_coverage_from_eval_scenario_report(tmp_path):
    suite = build_smoke_embodied_eval_suite()
    scenario = suite.scenario("robot_head_look_left_compare")
    assert scenario is not None

    report = await EmbodiedEvalArena(language=Language.EN).run_scenario(
        suite_id=suite.suite_id,
        scenario=scenario,
        output_dir=tmp_path / "coverage",
    )
    assert report.report_json_path is not None
    output_dir = tmp_path / "episode_export_report"

    assert (
        export_episodes_main(
            [
                "--source",
                "embodied-eval",
                "--input",
                str(report.report_json_path),
                "--out",
                str(output_dir),
            ]
        )
        == 0
    )

    manifest_payload = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["episode_count"] == 2
    assert len(manifest_payload["family_coverage"]) == 2
    by_backend = {row["execution_backend"]: row for row in manifest_payload["family_coverage"]}
    assert set(by_backend) == {"simulation", "preview"}
    assert all(row["scenario_family"] == "robot_head_single_step" for row in by_backend.values())
    assert all(row["episode_count"] == 1 for row in by_backend.values())
    assert all(row["task_success_count"] == 1 for row in by_backend.values())
    assert all(row["safety_success_count"] == 1 for row in by_backend.values())


def test_export_episodes_cli_exports_one_episode_from_replay_artifact(tmp_path):
    replay_artifact = _build_replay_artifact(tmp_path)
    output_dir = tmp_path / "episode_export_replay"

    assert (
        export_episodes_main(
            ["--source", "replay", "--input", str(replay_artifact), "--out", str(output_dir)]
        )
        == 0
    )

    episodes = _episode_payloads(output_dir)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["origin"] == "replay"
    assert episode["source_replay_name"] == "phase1_turn_tool_robot_action"
    assert episode["source_event_ids"]
    assert any(ref["artifact_kind"] == "replay_artifact_json" for ref in episode["artifact_refs"])
    assert all("content" not in ref for ref in episode["artifact_refs"])
    assert all("bytes" not in ref for ref in episode["artifact_refs"])


def test_live_and_practice_episode_export_helpers_are_lazy_exported():
    from blink.brain import evals

    assert evals.build_episode_from_live_runtime_payload is build_episode_from_live_runtime_payload
    assert (
        evals.build_episodes_from_practice_plan_payload is build_episodes_from_practice_plan_payload
    )


def test_live_episode_export_is_deterministic_and_bounded(tmp_path):
    live_artifact = _build_live_artifact(tmp_path)
    payload = json.loads(live_artifact.read_text(encoding="utf-8"))

    first = build_episode_from_live_runtime_payload(payload, source_path=live_artifact)
    second = build_episode_from_live_runtime_payload(payload, source_path=live_artifact)

    assert first.as_dict() == second.as_dict()
    exported = first.as_dict()
    assert exported["origin"] == "live"
    assert exported["source_run_id"] == "turn-live-1"
    assert exported["scenario_family"] == "live_runtime_turn"
    assert exported["provenance_ids"]["runtime_episode_id"] == "runtime-episode-1"
    assert exported["provenance_ids"]["turn_id"] == "turn-live-1"
    assert exported["source_event_ids"] == ["evt-live-001", "evt-live-002", "evt-live-003"]
    assert any(
        ref["artifact_kind"] == "live_runtime_artifact_json" for ref in exported["artifact_refs"]
    )
    assert any(
        ref["artifact_kind"] == "operator_snapshot_json" for ref in exported["artifact_refs"]
    )
    assert all(
        ref["artifact_kind"] not in {"raw_audio_wav", "brain_db"}
        for ref in exported["artifact_refs"]
    )
    serialized = json.dumps(exported, ensure_ascii=False, sort_keys=True)
    assert "RAW USER TRANSCRIPT SHOULD NOT LEAK" not in serialized
    assert "RAW ASSISTANT RESPONSE SHOULD NOT LEAK" not in serialized
    assert "content" not in serialized
    assert "bytes" not in serialized


def test_export_episodes_cli_exports_live_artifact_with_family_summary(tmp_path):
    live_artifact = _build_live_artifact(tmp_path)
    output_dir = tmp_path / "episode_export_live"

    assert (
        export_episodes_main(
            ["--source", "live", "--input", str(live_artifact), "--out", str(output_dir)]
        )
        == 0
    )

    episodes = _episode_payloads(output_dir)
    assert len(episodes) == 1
    assert episodes[0]["origin"] == "live"
    manifest_payload = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["origins"] == ["live"]
    assert manifest_payload["scenario_family_counts"] == {"live_runtime_turn": 1}
    assert manifest_payload["family_coverage"][0]["origin"] == "live"
    assert manifest_payload["family_coverage"][0]["scenario_family"] == "live_runtime_turn"


def test_practice_episode_export_preserves_plan_lineage_without_results(tmp_path):
    practice_artifact = _build_practice_artifact(tmp_path)
    payload = json.loads(practice_artifact.read_text(encoding="utf-8"))

    first = build_episodes_from_practice_plan_payload(payload, source_path=practice_artifact)
    second = build_episodes_from_practice_plan_payload(payload, source_path=practice_artifact)

    assert [episode.as_dict() for episode in first] == [episode.as_dict() for episode in second]
    assert len(first) == 2
    exported = [episode.as_dict() for episode in first]
    assert {episode["origin"] for episode in exported} == {"practice"}
    assert {episode["source_run_id"] for episode in exported} == {"practice-plan-1"}
    assert {episode["outcome_summary"]["task_success"] for episode in exported} == {None}
    assert {episode["safety_summary"]["safety_success"] for episode in exported} == {None}
    target_ids = {episode["provenance_ids"]["practice_target_id"] for episode in exported}
    assert target_ids == {"practice-target-1", "practice-target-2"}
    assert all(
        episode["provenance_ids"]["practice_plan_id"] == "practice-plan-1" for episode in exported
    )
    assert all(
        any(ref["artifact_kind"] == "practice_plan_json" for ref in episode["artifact_refs"])
        for episode in exported
    )
    assert all(
        all(ref["artifact_kind"] != "raw_video_mp4" for ref in episode["artifact_refs"])
        for episode in exported
    )


def test_export_episodes_cli_exports_practice_artifact_with_family_summary(tmp_path):
    practice_artifact = _build_practice_artifact(tmp_path)
    output_dir = tmp_path / "episode_export_practice"

    assert (
        export_episodes_main(
            ["--source", "practice", "--input", str(practice_artifact), "--out", str(output_dir)]
        )
        == 0
    )

    episodes = _episode_payloads(output_dir)
    assert len(episodes) == 2
    assert {episode["origin"] for episode in episodes} == {"practice"}
    manifest_payload = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["origins"] == ["practice"]
    assert manifest_payload["scenario_family_counts"] == {
        "conversation_repair": 1,
        "robot_head_single_step": 1,
    }
    assert {row["origin"] for row in manifest_payload["family_coverage"]} == {"practice"}
    assert {row["episode_count"] for row in manifest_payload["family_coverage"]} == {1}

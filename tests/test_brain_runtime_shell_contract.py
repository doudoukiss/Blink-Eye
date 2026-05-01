import json
import shutil
from pathlib import Path

import pytest

from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context import BrainContextTask
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import load_default_agent_blocks
from blink.brain.projections import BrainWakeConditionKind
from blink.brain.runtime_shell import BrainRuntimeShell
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language
from tests.phase23_fixtures import seed_phase23_surfaces
from tests.phase24_fixtures import seed_phase24_adapter_governance


def _event_count(store: BrainStore) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0])


def _projection_count(store: BrainStore) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM brain_projections").fetchone()[0])


def _build_commitment_store(db_path: Path) -> tuple[BrainStore, str, object]:
    from tests.test_brain_memory_v2 import (
        _scene_world_projection_for_multimodal,
        _seed_scene_episode,
    )

    store = BrainStore(path=db_path)
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Follow up on the operator shell contract",
        intent="narrative.commitment",
        source="test",
        details={"summary": "Need a bounded shell control trail."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="Please inspect the shell state.",
        assistant_text="Inspecting the bounded shell state now.",
        assistant_summary="Inspected the bounded shell state.",
        tool_calls=[],
    )
    _ = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-shell-scene"],
            updated_at="2026-01-01T00:00:10+00:00",
            include_person=False,
        ),
        start_second=10,
    )
    return store, commitment.commitment_id, session_ids


def test_runtime_shell_inspection_is_read_only(tmp_path):
    db_path = tmp_path / "brain.db"
    store, _, session_ids = _build_commitment_store(db_path)
    event_count_before = _event_count(store)
    projection_count_before = _projection_count(store)
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="alpha",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        snapshot = shell.snapshot()
        packet = shell.inspect_packet(
            task=BrainContextTask.OPERATOR_AUDIT,
            query_text="Audit the current scene episode state.",
        )
        waits = shell.inspect_pending_wakes()

        assert snapshot.session_ids.thread_id == session_ids.thread_id
        assert snapshot.background_maintenance_running is None
        assert snapshot.recent_scene_episodes
        assert snapshot.multimodal_autobiography_digest["entry_counts"]["privacy"]["standard"] >= 1
        assert snapshot.latest_source_presence_scope_key == "browser:presence"
        assert packet.task == BrainContextTask.OPERATOR_AUDIT
        assert packet.packet_digest
        assert packet.packet_digest["scene_episode_trace"]["selected_entry_ids"]
        assert waits.wake_digest
        assert _event_count(shell._store) == event_count_before
        assert _projection_count(shell._store) == projection_count_before
    finally:
        shell.close()


def test_runtime_shell_exposes_predictive_inspection_without_mutating_store(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_goal_updated,
        _append_robot_action_outcome,
        _append_scene_changed,
        _ensure_blocks,
    )

    db_path = tmp_path / "brain_predictive.db"
    store = BrainStore(path=db_path)
    _ensure_blocks(store)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="shell-predictive")
    _append_body_state(store, session_ids, second=1)
    _append_scene_changed(store, session_ids, second=2)
    _append_goal_created(store, session_ids, second=3)
    _append_scene_changed(store, session_ids, second=4)
    _append_robot_action_outcome(store, session_ids, second=5, accepted=False)
    _append_scene_changed(
        store,
        session_ids,
        second=40,
        include_person=False,
        affordance_availability="blocked",
        camera_fresh=False,
    )
    event_count_before = _event_count(store)
    projection_count_before = _projection_count(store)
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="shell-predictive",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        snapshot = shell.snapshot()
        reply_packet = shell.inspect_packet(
            task=BrainContextTask.REPLY,
            query_text="Answer without predictive leakage.",
        )
        planning_packet = shell.inspect_packet(
            task=BrainContextTask.PLANNING,
            query_text="Plan the next action conservatively.",
        )
        operator_packet = shell.inspect_packet(
            task=BrainContextTask.OPERATOR_AUDIT,
            query_text="Audit predictive traces.",
        )

        assert snapshot.predictive_digest["active_kind_counts"]
        assert snapshot.recent_active_predictions
        assert snapshot.recent_prediction_resolutions
        assert snapshot.predictive_digest["resolution_kind_counts"]["invalidated"] >= 1
        assert not reply_packet.packet_digest["prediction_trace"]["selected_prediction_ids"]
        assert planning_packet.packet_digest["prediction_trace"]["selected_prediction_ids"]
        assert (
            operator_packet.packet_digest["prediction_trace"]["resolution_kind_counts"][
                "invalidated"
            ]
            >= 1
        )
        predictive_inspection = shell.runtime_shell_digest()["predictive_inspection"]
        assert "operator_audit" in predictive_inspection["packet_prediction_trace_by_task"]
        assert predictive_inspection["resolution_kind_counts"]["invalidated"] >= 1
        assert _event_count(shell._store) == event_count_before
        assert _projection_count(shell._store) == projection_count_before
    finally:
        shell.close()


def test_runtime_shell_exposes_practice_and_skill_evidence_without_mutating_store(tmp_path):
    db_path = tmp_path / "brain_phase23.db"
    store = BrainStore(path=db_path)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="shell-phase23")
    seed_phase23_surfaces(
        store=store,
        session_ids=session_ids,
        output_dir=tmp_path / "practice_artifacts",
    )
    event_count_before = _event_count(store)
    projection_count_before = _projection_count(store)
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="shell-phase23",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        snapshot = shell.snapshot()
        digest = shell.runtime_shell_digest()

        assert snapshot.practice_digest["recent_plan_ids"]
        assert snapshot.recent_practice_plans
        assert snapshot.recent_practice_plans[0]["artifact_paths"]
        assert snapshot.skill_evidence_digest["evidence_count"] >= 1
        assert snapshot.recent_skill_governance_proposals
        assert digest["practice_inspection"]["recent_plans"]
        assert digest["skill_evidence_inspection"]["recent_governance_proposals"]
        assert _event_count(shell._store) == event_count_before
        assert _projection_count(shell._store) == projection_count_before
    finally:
        shell.close()


def test_runtime_shell_exposes_adapter_governance_without_mutating_store(tmp_path):
    db_path = tmp_path / "brain_phase24.db"
    store = BrainStore(path=db_path)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="shell-phase24")
    seed_phase24_adapter_governance(
        store=store,
        session_ids=session_ids,
        output_dir=tmp_path / "phase24_artifacts",
    )
    event_count_before = _event_count(store)
    projection_count_before = _projection_count(store)
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="shell-phase24",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        snapshot = shell.snapshot()
        digest = shell.runtime_shell_digest()

        assert snapshot.adapter_governance_digest["state_counts"]["default"] >= 1
        assert snapshot.recent_adapter_cards
        assert snapshot.recent_adapter_promotion_decisions
        assert snapshot.sim_to_real_digest["promotion_state_counts"]["shadow"] >= 1
        assert digest["adapter_governance_inspection"]["recent_cards"]
        assert digest["sim_to_real_inspection"]["readiness_reports"]
        assert _event_count(shell._store) == event_count_before
        assert _projection_count(shell._store) == projection_count_before
    finally:
        shell.close()


@pytest.mark.asyncio
async def test_runtime_shell_exposes_embodied_inspection_without_mutating_store(tmp_path):
    from blink.embodiment.robot_head.simulation import (
        RobotHeadSimulationConfig,
        SimulationDriver,
    )
    from tests.test_brain_embodied_executive import _build_runtime, _create_robot_goal

    runtime, controller, _ = _build_runtime(
        tmp_path,
        client_id="shell-embodied",
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation-shell"),
        ),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Inspect the embodied shell contract")
        await runtime.executive.request_plan_proposal(goal_id=goal_id)
        await runtime.executive.run_once()

        event_count_before = _event_count(runtime.store)
        projection_count_before = _projection_count(runtime.store)

        snapshot = runtime.shell.snapshot()
        digest = runtime.shell.runtime_shell_digest()["embodied_inspection"]

        assert snapshot.current_embodied_intent["intent_kind"] == "execute_action"
        assert snapshot.current_embodied_intent["status"] == "succeeded"
        assert snapshot.recent_embodied_execution_traces[0]["status"] == "succeeded"
        assert snapshot.recent_low_level_embodied_actions
        assert digest["current_intent"]["intent_kind"] == "execute_action"
        assert digest["current_low_level_executor"] == "simulation"
        assert digest["recent_execution_traces"][0]["status"] == "succeeded"
        assert digest["recent_low_level_embodied_actions"]
        assert _event_count(runtime.store) == event_count_before
        assert _projection_count(runtime.store) == projection_count_before
    finally:
        await controller.close()
        runtime.close()


def test_runtime_shell_controls_are_auditable_and_commitment_first(tmp_path):
    db_path = tmp_path / "brain.db"
    store, commitment_id, session_ids = _build_commitment_store(db_path)
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="alpha",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        interrupted = shell.interrupt_commitment(
            commitment_id=commitment_id,
            reason_summary="Pause while the operator inspects.",
        )
        after_interrupt = shell._store.get_executive_commitment(commitment_id=commitment_id)
        suppressed = shell.suppress_commitment(
            commitment_id=commitment_id,
            reason_summary="Hold until an explicit resume arrives.",
        )
        after_suppress = shell._store.get_executive_commitment(commitment_id=commitment_id)
        resumed = shell.resume_commitment(
            commitment_id=commitment_id,
            reason_summary="Operator approved the resume.",
        )
        after_resume = shell._store.get_executive_commitment(commitment_id=commitment_id)
        recent_events = shell._store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        )
    finally:
        shell.close()

    assert interrupted.applied is True
    assert interrupted.status_before == "active"
    assert after_interrupt is not None
    assert after_interrupt.status == "deferred"
    assert len(after_interrupt.wake_conditions) == 1
    assert after_interrupt.wake_conditions[0].kind == BrainWakeConditionKind.EXPLICIT_RESUME.value

    assert suppressed.applied is True
    assert after_suppress is not None
    assert after_suppress.status == "deferred"
    assert len(after_suppress.wake_conditions) == 1
    assert after_suppress.wake_conditions[0].kind == BrainWakeConditionKind.EXPLICIT_RESUME.value

    assert resumed.applied is True
    assert after_resume is not None
    assert after_resume.status == "active"

    shell_events = [
        event
        for event in recent_events
        if event.event_type in {BrainEventType.GOAL_DEFERRED, BrainEventType.GOAL_RESUMED}
        and event.source == "runtime_shell"
    ]
    control_kinds = [
        (event.event_type, dict(event.payload or {}).get("runtime_shell_control", {}).get("control_kind"))
        for event in shell_events
    ]

    assert (BrainEventType.GOAL_DEFERRED, "interrupt") in control_kinds
    assert (BrainEventType.GOAL_DEFERRED, "suppress") in control_kinds
    assert (BrainEventType.GOAL_RESUMED, "resume") in control_kinds
    assert all(result.event_id is not None for result in (interrupted, suppressed, resumed))


def test_runtime_shell_open_missing_commitment_raises_key_error(tmp_path):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    store.close()

    shell = BrainRuntimeShell.open(brain_db_path=db_path, runtime_kind="browser", client_id="alpha")
    try:
        with pytest.raises(KeyError):
            shell.interrupt_commitment(
                commitment_id="missing-commitment",
                reason_summary="This commitment does not exist.",
            )
    finally:
        shell.close()


def test_runtime_shell_offline_inspection_and_audit_are_stable_for_same_store_state(tmp_path):
    source_db = tmp_path / "source.db"
    clone_db = tmp_path / "clone.db"
    store, _, session_ids = _build_commitment_store(source_db)
    store.close()
    shutil.copy2(source_db, clone_db)

    first_shell = BrainRuntimeShell.open(
        brain_db_path=source_db,
        runtime_kind="browser",
        client_id="alpha",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    second_shell = BrainRuntimeShell.open(
        brain_db_path=clone_db,
        runtime_kind="browser",
        client_id="alpha",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        first_packet = first_shell.inspect_packet(
            task=BrainContextTask.PLANNING,
            query_text="What should happen next?",
        )
        second_packet = second_shell.inspect_packet(
            task=BrainContextTask.PLANNING,
            query_text="What should happen next?",
        )
        first_report = first_shell.export_audit(output_dir=tmp_path / "audit_one")
        second_report = second_shell.export_audit(output_dir=tmp_path / "audit_two")
    finally:
        first_shell.close()
        second_shell.close()

    assert first_packet.packet_digest == second_packet.packet_digest
    assert (
        first_report.payload["continuity_state"]["context_packet_digest"]
        == second_report.payload["continuity_state"]["context_packet_digest"]
    )
    assert (
        first_report.payload["continuity_state"]["planning_digest"]
        == second_report.payload["continuity_state"]["planning_digest"]
    )
    assert (
        first_report.payload["continuity_state"]["runtime_shell_digest"]
        == second_report.payload["continuity_state"]["runtime_shell_digest"]
    )
    assert json.loads(first_report.json_path.read_text(encoding="utf-8"))["runtime_shell_digest"] == json.loads(
        second_report.json_path.read_text(encoding="utf-8")
    )["runtime_shell_digest"]

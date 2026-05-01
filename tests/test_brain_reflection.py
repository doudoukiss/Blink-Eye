import json
from datetime import UTC, datetime, timedelta

from blink.brain.autonomy import BrainAutonomyDecisionKind, BrainReevaluationConditionKind
from blink.brain.context_surfaces import (
    BrainContextSurfaceBuilder,
    render_autobiography_summary,
    render_relationship_continuity_summary,
)
from blink.brain.events import BrainEventType
from blink.brain.identity import base_brain_system_prompt
from blink.brain.memory_layers.exports import BrainMemoryExporter
from blink.brain.memory_v2 import (
    BrainReflectionEngine,
    BrainReflectionScheduler,
    BrainReflectionSchedulerConfig,
)
from blink.brain.processors import BrainContextCompiler
from blink.brain.replay import BrainReplayHarness
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.cli.local_brain_reflect import main as local_brain_reflect_main
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def test_reflection_engine_creates_autobiography_health_and_cycle_artifacts(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="我们继续推进 Alpha 项目，这周想把发布时间定下来。",
        assistant_text="好的，我们继续推进 Alpha 项目。",
        assistant_summary="继续讨论 Alpha 项目和发布时间。",
        tool_calls=[],
    )
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="还有，刚才那条偏好我说错了，应该是不喜欢咖啡。",
        assistant_text="明白，我会按最新更正处理。",
        assistant_summary="用户更正了偏好，并继续推进项目。",
        tool_calls=[],
    )

    result = BrainReflectionEngine(store=store).run_once(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_ids=session_ids,
        trigger="manual",
    )

    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
    self_current_arc = store.get_current_core_memory_block(
        block_kind="self_current_arc",
        scope_type="agent",
        scope_id=session_ids.agent_id,
    )
    relationship_arc = store.latest_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
    )
    project_arcs = store.autobiographical_entries(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kinds=("project_arc",),
        limit=8,
    )
    health_report = store.latest_memory_health_report(
        scope_type="relationship",
        scope_id=relationship_scope_id,
    )
    cycle_rows = store.list_reflection_cycles(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    export = BrainMemoryExporter(store=store).export_thread_digest(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        output_path=tmp_path / "thread_digest.json",
    )

    assert result.draft_artifact_path.exists()
    draft_payload = json.loads(result.draft_artifact_path.read_text(encoding="utf-8"))
    assert draft_payload["segments"]
    assert draft_payload["autobiography_updates"]
    assert self_current_arc is not None
    assert self_current_arc.content["summary"]
    assert relationship_arc is not None
    assert relationship_arc.rendered_summary
    assert project_arcs
    assert health_report is not None
    assert cycle_rows[0].status == "completed"
    assert cycle_rows[0].draft_artifact_path == str(result.draft_artifact_path)
    assert export.payload["autobiography"]
    assert export.payload["memory_health_report"]["report_id"] == health_report.report_id
    assert export.payload["reflection_cycles"]
    assert export.payload["latest_reflection_draft_path"] == str(result.draft_artifact_path)


def test_reflection_engine_marks_ambiguous_conflicts_uncertain_and_reports_health(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    store._claims().record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.role",
        object_value="设计师",
        status="active",
        confidence=0.78,
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:alpha",
    )
    store._claims().record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.role",
        object_value="产品经理",
        status="active",
        confidence=0.77,
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:alpha",
    )
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="顺便更新一下我的工作近况。",
        assistant_text="好的。",
        assistant_summary="用户提到工作近况。",
        tool_calls=[],
    )

    BrainReflectionEngine(store=store).run_once(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_ids=session_ids,
        trigger="manual",
    )

    current_claims = store.query_claims(
        temporal_mode="current",
        scope_type="user",
        scope_id=session_ids.user_id,
        limit=16,
    )
    health_report = store.latest_memory_health_report(
        scope_type="relationship",
        scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
    )

    assert any(claim.status == "uncertain" for claim in current_claims)
    assert health_report is not None
    assert any(
        finding.get("code") == "ambiguous_claim_conflict" for finding in health_report.findings
    )


def test_context_surface_prefers_relationship_arc_and_self_current_arc(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink actions",
        }
    )
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="我们继续推进 Alpha 项目。",
        assistant_text="好的。",
        assistant_summary="继续推进 Alpha 项目。",
        tool_calls=[],
    )
    BrainReflectionEngine(store=store).run_once(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_ids=session_ids,
        trigger="manual",
    )

    surface = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.ZH,
    ).build(latest_user_text="Alpha 项目进展怎么样？", include_historical_claims=True)
    compiler = BrainContextCompiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
        base_prompt=base_brain_system_prompt(Language.ZH),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.ZH,
        ),
    )
    compiled = compiler.compile(latest_user_text="Alpha 项目进展怎么样？")

    assert surface.autobiography
    assert surface.health_summary is not None
    assert "关系弧线" in render_relationship_continuity_summary(surface, Language.ZH)
    assert "当前弧线" in render_autobiography_summary(surface, Language.ZH)
    assert "## Active Continuity" in compiled
    assert "## Unresolved State" in compiled


def test_reflection_scheduler_skips_while_active_then_runs_after_idle(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.append_brain_event(
        event_type="user.turn.started",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="我们继续推进 Alpha 项目。",
        assistant_text="好的。",
        assistant_summary="继续推进 Alpha 项目。",
        tool_calls=[],
    )
    scheduler = BrainReflectionScheduler(
        store_path=store.path,
        session_resolver=lambda: session_ids,
        config=BrainReflectionSchedulerConfig(wakeup_interval_secs=0.01, idle_grace_secs=0.0),
    )

    skipped = scheduler.run_cycle(trigger="timer")
    assert skipped.status == "skipped"
    assert skipped.skip_reason == "thread_active"

    store.append_brain_event(
        event_type="user.turn.ended",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    store.append_brain_event(
        event_type="assistant.turn.ended",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"text": "好的。"},
    )
    completed = scheduler.run_cycle(trigger="timer")

    assert completed.status == "completed"
    cycle_rows = store.list_reflection_cycles(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=4,
    )
    assert cycle_rows[0].status == "completed"
    assert any(record.status == "skipped" for record in cycle_rows)


def test_runtime_close_runs_reflection_and_replay_rebuilds_outputs(tmp_path):
    db_path = tmp_path / "brain.db"
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    runtime.store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="Alpha 项目需要继续推进。",
        assistant_text="好的。",
        assistant_summary="继续推进 Alpha 项目。",
        tool_calls=[],
    )
    runtime.close()

    reopened = BrainStore(path=db_path)
    try:
        relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
        assert reopened.get_current_core_memory_block(
            block_kind="self_current_arc",
            scope_type="agent",
            scope_id=session_ids.agent_id,
        ) is not None
        assert reopened.latest_memory_health_report(
            scope_type="relationship",
            scope_id=relationship_scope_id,
        ) is not None

        harness = BrainReplayHarness(store=reopened)
        scenario = harness.capture_builtin_scenario(
            name="phase1_turn_tool_robot_action",
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        result = harness.replay(scenario, presence_scope_key="browser:presence")
        replay_payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

        assert replay_payload["autobiography"]
        assert replay_payload["memory_health_summary"] is not None
        assert replay_payload["reflection_cycles"]
        assert replay_payload["latest_reflection_draft_path"] is not None
    finally:
        reopened.close()


def test_timer_triggered_reflection_with_actionable_findings_produces_candidate(tmp_path):
    db_path = tmp_path / "brain.db"
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    user_entity = runtime.store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    runtime.store._claims().record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.role",
        object_value="设计师",
        status="active",
        confidence=0.78,
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:alpha",
    )
    runtime.store._claims().record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.role",
        object_value="产品经理",
        status="active",
        confidence=0.77,
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:alpha",
    )
    runtime.store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="继续整理我们的工作背景。",
        assistant_text="好的。",
        assistant_summary="继续整理背景信息。",
        tool_calls=[],
    )
    idle_ts = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
    runtime.store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
        ts=idle_ts,
    )
    runtime.store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"text": "好的。"},
        ts=idle_ts,
    )

    result = runtime.reflection_scheduler.run_cycle(trigger="timer", force=True)
    agenda = runtime.store.get_agenda_projection(scope_key=session_ids.thread_id, user_id=session_ids.user_id)

    assert result.status == "completed"
    assert any(goal.intent == "autonomy.maintenance_review_findings" for goal in agenda.goals)
    runtime.close()


def test_repeated_thread_active_skips_produce_backpressure_candidate_with_cooldown(tmp_path):
    db_path = tmp_path / "brain.db"
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: session_ids,
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    runtime.store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="我们继续推进 Alpha 项目。",
        assistant_text="好的。",
        assistant_summary="继续推进 Alpha 项目。",
        tool_calls=[],
    )
    runtime.store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    for _ in range(4):
        result = runtime.reflection_scheduler.run_cycle(trigger="timer")
        assert result.status == "skipped"
        assert result.skip_reason == "thread_active"

    ledger = runtime.store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    assert [candidate.candidate_type for candidate in ledger.current_candidates] == [
        "maintenance_thread_active_backpressure"
    ]
    collapse_entries = [
        entry
        for entry in ledger.recent_entries
        if entry.decision_kind
        in {
            BrainAutonomyDecisionKind.SUPPRESSED.value,
            BrainAutonomyDecisionKind.MERGED.value,
        }
    ]
    assert collapse_entries
    runtime.close()


def test_timer_skips_without_new_work_do_not_produce_candidates(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    proposed: list[object] = []
    scheduler = BrainReflectionScheduler(
        store_path=store.path,
        session_resolver=lambda: session_ids,
        candidate_goal_sink=lambda **kwargs: proposed.append(kwargs),
    )

    result = scheduler.run_cycle(trigger="timer")

    assert result.status == "skipped"
    assert result.skip_reason == "no_new_work"
    assert proposed == []
    store.close()


def test_reflection_scheduler_timer_wake_invokes_reevaluation_sink(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    reevaluations = []
    scheduler = BrainReflectionScheduler(
        store_path=store.path,
        session_resolver=lambda: session_ids,
        reevaluation_sink=lambda **kwargs: reevaluations.append(kwargs),
    )

    result = scheduler.run_cycle(trigger="timer")

    assert result.status == "skipped"
    assert [call["trigger"].kind for call in reevaluations] == [
        BrainReevaluationConditionKind.TIME_REACHED.value,
        BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
    ]
    store.close()


def test_local_brain_reflect_routes_through_shell_and_preserves_stdout_contract(
    tmp_path, capsys
):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="继续整理反思结果。",
        assistant_text="好的，我来整理。",
        assistant_summary="继续整理反思结果。",
        tool_calls=[],
    )
    store.close()

    exit_code = local_brain_reflect_main(
        [
            "--brain-db-path",
            str(db_path),
            "--runtime-kind",
            "browser",
            "--client-id",
            "alpha",
        ]
    )
    output = capsys.readouterr().out.strip().splitlines()

    reopened = BrainStore(path=db_path)
    try:
        cycle = reopened.list_reflection_cycles(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=1,
        )[0]
    finally:
        reopened.close()

    assert exit_code == 0
    assert output[0].startswith("cycle_id=")
    assert output[1].startswith("draft_artifact=")
    assert output[2].startswith("health_report_id=")
    assert cycle.trigger == "runtime_shell:manual"

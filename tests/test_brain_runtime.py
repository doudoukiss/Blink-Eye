from datetime import UTC, datetime, timedelta

import pytest

from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
)
from blink.brain.context import BrainContextTask
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.events import BrainEventType
from blink.brain.identity import base_brain_system_prompt
from blink.brain.persona import (
    BrainPersonaModality,
    BrainVoiceBackendCapabilities,
    BrainVoiceBackendCapabilityRegistry,
    runtime_expression_state_from_frame,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.runtime import BrainRuntime, build_session_resolver
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
from blink.embodiment.robot_head.controller import RobotHeadController
from blink.embodiment.robot_head.drivers import MockDriver
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def _build_context_compiler(*, store: BrainStore, session_resolver, language: Language):
    from blink.brain.context import BrainContextCompiler

    return BrainContextCompiler(
        store=store,
        session_resolver=session_resolver,
        language=language,
        base_prompt=base_brain_system_prompt(language),
        context_surface_builder=BrainContextSurfaceBuilder(
            store=store,
            session_resolver=session_resolver,
            presence_scope_key="browser:presence",
            language=language,
        ),
    )


def _build_runtime(tmp_path, *, runtime_kind="browser", robot_head_controller=None, db_suffix=None):
    db_name = db_suffix or runtime_kind
    return BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind=runtime_kind,
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind=runtime_kind,
            client_id=f"{runtime_kind}-alpha",
        ),
        llm=DummyLLM(),
        robot_head_controller=robot_head_controller,
        brain_db_path=tmp_path / f"{db_name}.db",
    )


def test_brain_runtime_persists_memory_across_restart(tmp_path):
    db_path = tmp_path / "brain.db"

    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    runtime.store.remember_fact(
        user_id="alpha",
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.9,
        singleton=True,
        source_episode_id=None,
    )
    runtime.close()

    restarted = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=db_path,
    )

    facts = restarted.store.active_facts(user_id="alpha")

    assert any(fact.rendered_text == "用户名字是 阿周" for fact in facts)
    assert "identity" in restarted.store.get_agent_blocks()
    restarted.close()


def test_browser_session_resolver_keeps_memory_scope_stable_across_webrtc_ids(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BLINK_LOCAL_BRAIN_USER_ID", raising=False)
    db_path = tmp_path / "brain.db"
    first_client = {"id": "SmallWebRTCConnection#0-first"}
    first_resolver = build_session_resolver(runtime_kind="browser", active_client=first_client)
    first_runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=first_resolver,
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    first_ids = first_runtime.session_resolver()
    assert first_ids.user_id == "local_primary"
    first_runtime.store.remember_fact(
        user_id=first_ids.user_id,
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.9,
        singleton=True,
        source_episode_id=None,
        agent_id=first_ids.agent_id,
        session_id=first_ids.session_id,
        thread_id=first_ids.thread_id,
    )
    first_runtime.close()

    second_client = {"id": "SmallWebRTCConnection#0-second"}
    second_runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=build_session_resolver(
            runtime_kind="browser",
            active_client=second_client,
        ),
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    second_ids = second_runtime.session_resolver()

    assert second_ids.user_id == "local_primary"
    assert second_ids.thread_id == "browser:local_primary"
    assert any(
        fact.rendered_text == "用户名字是 阿周"
        for fact in second_runtime.store.active_facts(user_id=second_ids.user_id)
    )
    second_runtime.close()


def test_browser_session_resolver_allows_explicit_stable_local_user(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_BRAIN_USER_ID", "sonics")
    active_client = {"id": "SmallWebRTCConnection#0-volatile"}
    resolver = build_session_resolver(runtime_kind="browser", active_client=active_client)

    first = resolver()
    active_client["id"] = "SmallWebRTCConnection#0-new"
    second = resolver()

    assert first.user_id == "sonics"
    assert second.user_id == "sonics"
    assert first.thread_id == "browser:sonics"
    assert second.thread_id == "browser:sonics"


def test_browser_session_resolver_can_still_opt_into_active_client_scope(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_BRAIN_USER_ID", raising=False)
    active_client = {"id": "client-a"}
    resolver = build_session_resolver(
        runtime_kind="browser",
        active_client=active_client,
        use_active_client_id=True,
    )

    assert resolver().user_id == "client-a"
    active_client["id"] = "client-b"
    assert resolver().user_id == "client-b"


def test_brain_context_compiler_isolates_users_and_renders_presence(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(
        {
            "identity": "Blink identity",
            "policy": "Blink policy",
            "style": "Blink style",
            "action_library": "Blink actions",
        }
    )
    store.remember_fact(
        user_id="alpha",
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.9,
        singleton=True,
        source_episode_id=None,
    )
    store.remember_fact(
        user_id="beta",
        namespace="profile.name",
        subject="user",
        value={"value": "小李"},
        rendered_text="用户名字是 小李",
        confidence=0.9,
        singleton=True,
        source_episode_id=None,
    )
    store.upsert_task(
        user_id="alpha",
        title="给妈妈打电话",
        details={"source": "user"},
    )
    store.set_presence_snapshot(
        scope_key="browser:presence",
        snapshot=BrainPresenceSnapshot(
            runtime_kind="browser",
            robot_head_enabled=True,
            robot_head_mode="preview",
            robot_head_available=True,
            robot_head_last_action="cmd_blink",
            vision_enabled=True,
            vision_connected=True,
        ).as_dict(),
    )
    compiler = _build_context_compiler(
        store=store,
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        language=Language.ZH,
    )

    compiled = compiler.compile(latest_user_text="你还记得我是谁吗？")

    assert "用户名字是 阿周" in compiled
    assert "用户名字是 小李" not in compiled
    assert "给妈妈打电话" in compiled
    assert "机器人头: 已启用，模式=preview，可用=True" in compiled
    assert "视觉: 已启用，摄像头连接=是" in compiled


def test_brain_runtime_close_flushes_thread_summary(tmp_path):
    db_path = tmp_path / "brain.db"

    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=db_path,
    )
    runtime.store.add_episode(
        agent_id="blink/main",
        user_id="alpha",
        session_id="browser:alpha",
        thread_id="browser:alpha",
        user_text="我叫阿周。",
        assistant_text="好的，我记住了。",
        assistant_summary="好的，我记住了。",
        tool_calls=[],
    )
    runtime.close()

    reopened = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=db_path,
    )

    summary = reopened.store.get_session_summary(user_id="alpha", thread_id="browser:alpha")

    assert "我叫阿周" in summary
    reopened.close()


def test_brain_runtime_background_maintenance_starts_and_stops(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
    )

    runtime.start_background_maintenance()
    assert runtime.reflection_scheduler.is_running is True

    runtime.stop_background_maintenance()
    assert runtime.reflection_scheduler.is_running is False
    runtime.close()


def test_runtime_expression_state_is_readable_from_runtime_and_shell(tmp_path):
    runtime = _build_runtime(tmp_path, runtime_kind="browser")

    frame = runtime.current_expression_frame()
    state = runtime.current_expression_state()
    snapshot = runtime.shell.snapshot()

    assert frame is not None
    assert state.available is True
    assert state.persona_profile_id == frame.persona_profile_id
    assert state.modality == BrainPersonaModality.BROWSER.value
    assert state.identity_label == "Blink; local non-human system"
    assert state.initiative_label == "balanced"
    assert state.evidence_visibility_label == "compact"
    assert state.correction_mode_label == "precise"
    assert state.explanation_structure_label == "answer_first"
    assert "measured clarity" in state.voice_style_summary
    assert state.voice_policy["available"] is True
    assert state.voice_policy["expression_controls_hardware"] is False
    assert "speech_rate" in state.voice_policy["unsupported_hints"]
    assert "voice_policy_noop:pause_timing_metadata_only" in state.voice_policy["noop_reason_codes"]
    assert snapshot.expression_state["available"] is True
    assert snapshot.expression_state["persona_profile_id"] == frame.persona_profile_id
    assert snapshot.expression_state["initiative_label"] == "balanced"
    assert snapshot.expression_state["expression_controls_hardware"] is False
    assert snapshot.expression_state["voice_policy"]["available"] is True
    runtime.close()


def test_runtime_reply_compiler_auto_selects_default_teaching_canon(tmp_path):
    runtime = _build_runtime(tmp_path, runtime_kind="browser")

    packet = runtime.compiler.compile_packet(
        latest_user_text="请解释递归调试思路",
        task=BrainContextTask.REPLY,
        persona_modality=runtime.persona_modality,
    )
    section = packet.selected_context.section("teaching_knowledge")

    assert section is not None
    assert packet.teaching_knowledge_decision is not None
    assert "exemplar:chinese_technical_explanation_bridge" in section.content
    assert "source=blink-default-teaching-canon" in section.content
    assert "provenance=" in section.content
    assert (
        packet.selected_context.estimated_tokens
        <= packet.selected_context.budget_profile.max_tokens
    )
    runtime._record_compiled_context_packet(packet)
    current = runtime.current_teaching_knowledge_routing()
    recent = runtime.recent_teaching_knowledge_routing(limit=3)
    assert current.available is True
    assert any(
        item.item_id == "exemplar:chinese_technical_explanation_bridge"
        for item in current.selected_items
    )
    assert recent and recent[0] == current
    assert section.content not in str(current.as_dict())
    runtime.close()


def test_missing_expression_defaults_return_unavailable_state(tmp_path):
    store = BrainStore(path=tmp_path / "missing-defaults.db")
    session_resolver = lambda: resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    compiler = _build_context_compiler(
        store=store,
        session_resolver=session_resolver,
        language=Language.EN,
    )

    frame, reason_codes, status = compiler.compile_persona_expression_frame(
        persona_modality=BrainPersonaModality.BROWSER,
    )
    state = runtime_expression_state_from_frame(
        frame,
        modality=BrainPersonaModality.BROWSER,
        reason_codes=reason_codes,
        memory_persona_section_status=status,
    )

    assert frame is None
    assert state.available is False
    assert state.identity_label == "Blink; local non-human system"
    assert state.expression_controls_hardware is False
    assert state.voice_policy["available"] is False
    assert "voice_policy_frame_missing" in state.voice_policy["reason_codes"]
    assert "persona_defaults_missing" in state.reason_codes
    assert state.memory_persona_section_status["persona_defaults"] == "missing"
    store.close()


def test_expression_state_identity_is_consistent_across_runtime_modalities(tmp_path):
    runtimes = [
        _build_runtime(tmp_path, runtime_kind="text"),
        _build_runtime(tmp_path, runtime_kind="voice"),
        _build_runtime(tmp_path, runtime_kind="browser"),
    ]
    robot_controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=MockDriver(),
    )
    runtimes.append(
        _build_runtime(
            tmp_path,
            runtime_kind="browser",
            robot_head_controller=robot_controller,
            db_suffix="browser-embodied",
        )
    )

    states = [runtime.current_expression_state() for runtime in runtimes]

    assert [state.modality for state in states] == [
        BrainPersonaModality.TEXT.value,
        BrainPersonaModality.VOICE.value,
        BrainPersonaModality.BROWSER.value,
        BrainPersonaModality.EMBODIED.value,
    ]
    assert all(state.identity_label == "Blink; local non-human system" for state in states)
    assert all(state.expression_controls_hardware is False for state in states)
    assert states[0].voice_policy["available"] is False
    assert all(state.voice_policy["expression_controls_hardware"] is False for state in states)
    assert all("servo" not in str(state.as_dict()).lower() for state in states)
    assert all("motor" not in str(state.as_dict()).lower() for state in states)
    for runtime in runtimes:
        runtime.close()


def test_runtime_current_voice_policy_exposes_active_and_noop_fields(tmp_path):
    runtime = _build_runtime(tmp_path, runtime_kind="voice")

    policy = runtime.current_voice_policy(latest_user_text="请解释递归调试思路")

    assert policy.available is True
    assert policy.modality == BrainPersonaModality.VOICE.value
    assert policy.expression_controls_hardware is False
    assert "interruption_strategy_label" in policy.active_hints
    assert "speech_rate" in policy.unsupported_hints
    assert "prosody_emphasis" in policy.unsupported_hints
    assert "voice_policy_noop:hardware_control_forbidden" in policy.noop_reason_codes
    runtime.close()


def test_runtime_current_voice_capabilities_uses_backend_registry(tmp_path):
    registry = BrainVoiceBackendCapabilityRegistry.from_mapping(
        {
            "test-stream": BrainVoiceBackendCapabilities(
                backend_label="test-stream",
                supports_chunk_boundaries=True,
                supports_interruption_flush=True,
                supports_interruption_discard=True,
                supports_speech_rate=True,
                reason_codes=("test_stream_capabilities",),
            )
        }
    )
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="voice",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="voice", client_id="voice-alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "voice-registry.db",
        tts_backend="test-stream",
        voice_backend_registry=registry,
    )

    capabilities = runtime.current_voice_capabilities()
    plan = runtime.current_voice_actuation_plan(latest_user_text="请解释递归调试思路")
    expression_state = runtime.current_expression_state(latest_user_text="请解释递归调试思路")

    assert capabilities.backend_label == "test-stream"
    assert capabilities.supports_speech_rate is True
    assert capabilities.supports_pause_timing is False
    assert capabilities.expression_controls_hardware is False
    assert "speech_rate" in plan.requested_hints
    assert "speech_rate" in plan.applied_hints
    assert "pause_timing" in plan.unsupported_hints
    assert plan.expression_controls_hardware is False
    assert expression_state.voice_actuation_plan["backend_label"] == "test-stream"
    assert "speech_rate" in expression_state.voice_actuation_plan["applied_hints"]
    assert expression_state.voice_actuation_plan["expression_controls_hardware"] is False
    runtime.close()


def test_runtime_shell_snapshot_reports_live_runtime_fields_without_mutating_state(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
    )
    session_ids = runtime.session_resolver()
    runtime.start_background_maintenance()
    runtime.store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-runtime-shell-next-wake",
            candidate_type="presence_acknowledgement",
            source=BrainCandidateGoalSource.RUNTIME.value,
            summary="Wake the shell snapshot on the nearest expiry.",
            goal_family="environment",
            urgency=0.7,
            confidence=0.9,
            initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
            expires_at="2099-01-01T00:00:05+00:00",
            created_at="2099-01-01T00:00:00+00:00",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    runtime._refresh_reevaluation_alarm()
    event_count_before = int(
        runtime.store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    )
    projection_count_before = int(
        runtime.store._conn.execute("SELECT COUNT(*) FROM brain_projections").fetchone()[0]
    )

    snapshot = runtime.shell.snapshot()
    packet = runtime.shell.inspect_packet(
        task=BrainContextTask.REPLY,
        query_text="What should I inspect?",
    )

    assert snapshot.background_maintenance_running is True
    assert snapshot.next_reevaluation_wake == {
        "kind": BrainReevaluationConditionKind.TIME_REACHED.value,
        "deadline": "2099-01-01T00:00:05+00:00",
    }
    assert packet.packet_digest
    assert (
        int(runtime.store._conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0])
        == event_count_before
    )
    assert (
        int(runtime.store._conn.execute("SELECT COUNT(*) FROM brain_projections").fetchone()[0])
        == projection_count_before
    )
    runtime.stop_background_maintenance()
    runtime.close()


@pytest.mark.asyncio
async def test_presence_director_pass_is_explicit_only(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
    )
    session_ids = runtime.session_resolver()
    runtime.store.set_presence_snapshot(
        scope_key="browser:presence",
        snapshot=BrainPresenceSnapshot(
            runtime_kind="browser",
            vision_enabled=True,
            vision_connected=True,
            perception_disabled=False,
        ).as_dict(),
    )
    runtime.store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "presence_scope_key": "browser:presence",
            "frame_seq": 1,
            "camera_connected": True,
            "camera_fresh": True,
            "person_present": "present",
            "attention_to_camera": "toward_camera",
            "engagement_state": "engaged",
            "scene_change": "stable",
            "summary": "One person is facing the camera.",
            "confidence": 0.95,
            "observed_at": "2099-01-01T00:00:00+00:00",
        },
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-runtime-explicit",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Create an agenda goal only when asked explicitly.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:runtime-explicit",
        dedupe_key="dedupe:runtime-explicit",
        requires_user_turn_gap=False,
        expires_at="2099-01-01T00:05:00+00:00",
        payload={"goal_details": {"channel": "runtime"}},
        created_at="2099-01-01T00:00:00+00:00",
    )
    runtime.store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    await runtime.executive.run_startup_pass()
    await runtime.executive.run_turn_end_pass()

    agenda_before = runtime.store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    )
    assert agenda_before.goals == []

    result = runtime.run_presence_director_pass()
    agenda_after = runtime.store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    )

    assert result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value
    assert agenda_after.goal(result.accepted_goal_id) is not None
    runtime.close()


@pytest.mark.asyncio
async def test_runtime_reevaluation_alarm_tracks_only_the_nearest_wake(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
    )
    session_ids = runtime.session_resolver()
    first = BrainCandidateGoal(
        candidate_goal_id="candidate-nearest-expiry",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Expire first.",
        goal_family="environment",
        urgency=0.7,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
        expires_at="2099-01-01T00:00:05+00:00",
        created_at="2099-01-01T00:00:00+00:00",
    )
    second = BrainCandidateGoal(
        candidate_goal_id="candidate-later-maintenance",
        candidate_type="maintenance_review_findings",
        source=BrainCandidateGoalSource.TIMER.value,
        summary="Wait for maintenance window.",
        goal_family="memory_maintenance",
        urgency=0.6,
        confidence=1.0,
        initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        expires_at="2099-01-01T00:01:00+00:00",
        expected_reevaluation_condition="after the maintenance window opens",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
        expected_reevaluation_condition_details={"not_before": "2099-01-01T00:00:30+00:00"},
        created_at="2099-01-01T00:00:00+00:00",
    )
    runtime.store.append_candidate_goal_created(
        candidate_goal=second,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    runtime.store.append_candidate_goal_created(
        candidate_goal=first,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    runtime._refresh_reevaluation_alarm()

    assert runtime._reevaluation_alarm_task is not None
    assert runtime._reevaluation_alarm_key == (
        BrainReevaluationConditionKind.TIME_REACHED.value,
        "2099-01-01T00:00:05+00:00",
    )

    previous_task = runtime._reevaluation_alarm_task
    runtime._refresh_reevaluation_alarm()

    assert runtime._reevaluation_alarm_task is previous_task
    runtime.stop_background_maintenance()
    runtime.close()


@pytest.mark.asyncio
async def test_runtime_reevaluation_alarm_runs_expiry_cleanup_and_rearms_next_wake(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="browser",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="browser", client_id="alpha"
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
    )
    session_ids = runtime.session_resolver()
    now = datetime.now(UTC)
    expiry_ts = (now - timedelta(seconds=1)).isoformat()
    maintenance_ts = (now + timedelta(seconds=30)).isoformat()
    expiring = BrainCandidateGoal(
        candidate_goal_id="candidate-runtime-expire-now",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Expire on the nearest alarm.",
        goal_family="environment",
        urgency=0.7,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
        expires_at=expiry_ts,
        created_at=(now - timedelta(seconds=5)).isoformat(),
    )
    maintenance = BrainCandidateGoal(
        candidate_goal_id="candidate-runtime-maintenance-wake",
        candidate_type="maintenance_review_findings",
        source=BrainCandidateGoalSource.TIMER.value,
        summary="Wait for the maintenance wake.",
        goal_family="memory_maintenance",
        urgency=0.6,
        confidence=1.0,
        initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        expires_at=(now + timedelta(minutes=1)).isoformat(),
        expected_reevaluation_condition="after the maintenance window opens",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
        expected_reevaluation_condition_details={"not_before": maintenance_ts},
        created_at=(now - timedelta(seconds=5)).isoformat(),
    )
    runtime.store.append_candidate_goal_created(
        candidate_goal=maintenance,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    runtime.store.append_candidate_goal_created(
        candidate_goal=expiring,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    await runtime._run_reevaluation_alarm(
        kind=BrainReevaluationConditionKind.TIME_REACHED.value,
        deadline=now,
    )
    ledger = runtime.store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
    events = runtime.store.recent_brain_events(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=16,
    )

    assert ledger.candidate(expiring.candidate_goal_id) is None
    assert ledger.candidate(maintenance.candidate_goal_id) is not None
    assert any(
        event.event_type == BrainEventType.GOAL_CANDIDATE_EXPIRED
        and event.payload["candidate_goal_id"] == expiring.candidate_goal_id
        and event.payload["reason_details"]["cleanup_owner"] == "explicit_expiry_cleanup"
        for event in events
    )
    assert runtime._reevaluation_alarm_key == (
        BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
        maintenance_ts,
    )
    runtime.stop_background_maintenance()
    runtime.close()


def test_brain_context_compiler_reads_projection_backed_state_surfaces(tmp_path):
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
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "提醒我给妈妈打电话。"},
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="memory",
        payload={"title": "给妈妈打电话", "status": "open"},
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        payload={
            "tool_call_id": "call_1",
            "function_name": "fetch_user_image",
            "result": {"answer": "你手里拿着一个杯子。"},
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.ASSISTANT_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        payload={"text": "好的，我会记住。"},
    )
    store.set_presence_snapshot(
        scope_key="browser:presence",
        snapshot=BrainPresenceSnapshot(
            runtime_kind="browser",
            robot_head_enabled=True,
            robot_head_mode="simulation",
            robot_head_available=True,
            vision_enabled=True,
            vision_connected=True,
            perception_disabled=False,
        ).as_dict(),
    )
    store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="perception",
        payload={
            "presence_scope_key": "browser:presence",
            "frame_seq": 1,
            "camera_connected": True,
            "camera_fresh": True,
            "person_present": "present",
            "attention_to_camera": "toward_camera",
            "engagement_state": "engaged",
            "scene_change": "stable",
            "summary": "One person is facing the camera.",
            "confidence": 0.92,
            "observed_at": "2026-04-17T10:00:00+00:00",
        },
    )

    compiler = _build_context_compiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
    )

    compiled = compiler.compile(latest_user_text="还记得我的待办吗？")

    assert "## Working Context" in compiled
    assert "## Scene" in compiled
    assert "人物在场: present" in compiled
    assert "## Engagement" in compiled
    assert "参与状态: engaged" in compiled
    assert "最近用户输入: 提醒我给妈妈打电话。" in compiled
    assert "最近工具: fetch_user_image" in compiled
    assert "## Agenda" in compiled
    assert "开放目标: 给妈妈打电话" in compiled
    assert "## Heartbeat" in compiled
    assert "## Relationship State" in compiled
    assert "最近事件:" in compiled


def test_brain_context_compiler_avoids_legacy_fact_and_summary_helpers(tmp_path):
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
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "我叫阿周。"},
    )

    def _raise(*args, **kwargs):
        raise AssertionError("legacy compatibility helper should not be called by compiler")

    store.active_facts = _raise
    store.relevant_facts = _raise
    store.get_session_summary = _raise

    compiler = _build_context_compiler(
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
    )

    compiled = compiler.compile(latest_user_text="你记得我吗？")

    assert "## User Profile" not in compiled

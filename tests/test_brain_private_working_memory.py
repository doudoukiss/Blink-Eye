from blink.brain.events import BrainEventType
from blink.brain.projections import (
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def test_private_working_memory_record_and_projection_roundtrip():
    record = BrainPrivateWorkingMemoryRecord(
        record_id="pwm-1",
        buffer_kind=BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
        summary="Need operator review before adoption.",
        state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
        evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
        backing_ids=["proposal-1", "goal-1"],
        source_event_ids=["evt-1"],
        goal_id="goal-1",
        commitment_id="commitment-1",
        plan_proposal_id="proposal-1",
        skill_id="skill-1",
        observed_at=_ts(1),
        updated_at=_ts(2),
        expires_at=_ts(3),
        details={"review_policy": "needs_operator_review"},
    )
    projection = BrainPrivateWorkingMemoryProjection(
        scope_type="thread",
        scope_id="thread-1",
        records=[record],
        updated_at=_ts(4),
    )

    hydrated = BrainPrivateWorkingMemoryProjection.from_dict(projection.as_dict())

    assert hydrated.scope_type == "thread"
    assert hydrated.scope_id == "thread-1"
    assert hydrated.buffer_counts == {BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value: 1}
    assert hydrated.state_counts == {BrainPrivateWorkingMemoryRecordState.ACTIVE.value: 1}
    assert hydrated.evidence_kind_counts == {BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value: 1}
    assert hydrated.active_record_ids == ["pwm-1"]
    assert hydrated.records[0].plan_proposal_id == "proposal-1"


def test_private_working_memory_projection_enforces_caps_and_scene_staleness(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="wm-caps")
    for index in range(5):
        store.remember_fact(
            user_id=session_ids.user_id,
            namespace="preference.like",
            subject=f"topic-{index}",
            value={"value": f"topic-{index}"},
            rendered_text=f"user likes topic-{index}",
            confidence=0.8,
            singleton=False,
            source_event_id=f"evt-fact-{index}",
            source_episode_id=None,
            provenance={"source": "test"},
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )

    for index in range(7):
        requested = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": f"goal-{index}",
                "capability_id": f"capability-{index}",
                "step_index": 0,
            },
            ts=_ts(index),
        )
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_COMPLETED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": f"goal-{index}",
                "capability_id": f"capability-{index}",
                "step_index": 0,
                "result": {"summary": f"Completed capability-{index}"},
            },
            causal_parent_id=requested.event_id,
            ts=_ts(index + 1),
        )

    store.append_brain_event(
        event_type=BrainEventType.BODY_STATE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "scope_key": "browser:presence",
            "snapshot": {
                "runtime_kind": "browser",
                "vision_enabled": True,
                "vision_connected": True,
                "updated_at": _ts(10),
            },
        },
        ts=_ts(10),
    )
    store.append_brain_event(
        event_type=BrainEventType.PERCEPTION_OBSERVED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "presence_scope_key": "browser:presence",
            "camera_connected": True,
            "camera_fresh": False,
            "camera_track_state": "stalled",
            "person_present": "uncertain",
            "summary": "desk with a notebook",
            "last_fresh_frame_at": _ts(0),
            "frame_age_ms": 20_000,
            "sensor_health_reason": "stale_frame",
        },
        ts=_ts(20),
    )

    projection = store.build_private_working_memory_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key="browser:presence",
        reference_ts=_ts(40),
    )

    assert projection.buffer_counts[BrainPrivateWorkingMemoryBufferKind.USER_MODEL.value] == 4
    assert projection.buffer_counts[BrainPrivateWorkingMemoryBufferKind.RECENT_TOOL_OUTCOME.value] == 6
    assert any(
        record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value
        and record.state == BrainPrivateWorkingMemoryRecordState.STALE.value
        for record in projection.records
    )
    assert any(
        record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value
        and "Scene freshness degraded" in record.summary
        for record in projection.records
    )


def test_private_working_memory_resolves_plan_assumptions_and_missing_inputs(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="wm-plan")
    proposal = BrainPlanProposal(
        plan_proposal_id="proposal-1",
        goal_id="goal-1",
        commitment_id="commitment-1",
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary="Need user confirmation.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        steps=[],
        assumptions=["the user still wants the same task"],
        missing_inputs=["which delivery date to use"],
        details={"request_kind": "initial_plan"},
        created_at=_ts(1),
    )
    proposed = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        ts=_ts(1),
    )
    store.append_planning_rejected(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary="Need a fresh plan.",
            reason="missing_required_input",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        causal_parent_id=proposed.event_id,
        ts=_ts(2),
    )

    projection = store.build_private_working_memory_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        reference_ts=_ts(3),
    )

    assert any(
        record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value
        and record.plan_proposal_id == "proposal-1"
        and record.state == BrainPrivateWorkingMemoryRecordState.RESOLVED.value
        for record in projection.records
    )
    assert any(
        record.buffer_kind == BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value
        and record.plan_proposal_id == "proposal-1"
        and record.state == BrainPrivateWorkingMemoryRecordState.RESOLVED.value
        and "Missing input" in record.summary
        for record in projection.records
    )

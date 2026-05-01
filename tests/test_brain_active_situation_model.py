from blink.brain.active_situation_model import build_active_situation_model_projection
from blink.brain.memory_v2 import (
    BrainProceduralSkillProjection,
    BrainProceduralSkillRecord,
    BrainProceduralSkillStatsRecord,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainActiveSituationEvidenceKind,
    BrainActiveSituationProjection,
    BrainActiveSituationRecord,
    BrainActiveSituationRecordKind,
    BrainActiveSituationRecordState,
    BrainAgendaProjection,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentProjection,
    BrainCommitmentRecord,
    BrainEngagementStateProjection,
    BrainGoal,
    BrainGoalStep,
    BrainPrivateWorkingMemoryBufferKind,
    BrainPrivateWorkingMemoryEvidenceKind,
    BrainPrivateWorkingMemoryProjection,
    BrainPrivateWorkingMemoryRecord,
    BrainPrivateWorkingMemoryRecordState,
    BrainSceneStateProjection,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def test_active_situation_record_and_projection_roundtrip():
    record = BrainActiveSituationRecord(
        record_id="situation-1",
        record_kind=BrainActiveSituationRecordKind.PLAN_STATE.value,
        summary="Need user confirmation before adoption.",
        state=BrainActiveSituationRecordState.UNRESOLVED.value,
        evidence_kind=BrainActiveSituationEvidenceKind.DERIVED.value,
        confidence=0.7,
        freshness="pending",
        uncertainty_codes=["missing_input"],
        private_record_ids=["pwm-1"],
        backing_ids=["proposal-1", "goal-1"],
        source_event_ids=["evt-1"],
        goal_id="goal-1",
        commitment_id="commitment-1",
        plan_proposal_id="proposal-1",
        skill_id="skill-1",
        observed_at=_ts(1),
        updated_at=_ts(2),
        expires_at=_ts(3),
        details={"review_policy": "needs_user_review"},
    )
    projection = BrainActiveSituationProjection(
        scope_type="thread",
        scope_id="thread-1",
        records=[record],
        updated_at=_ts(4),
    )

    hydrated = BrainActiveSituationProjection.from_dict(projection.as_dict())

    assert hydrated.scope_type == "thread"
    assert hydrated.scope_id == "thread-1"
    assert hydrated.kind_counts == {BrainActiveSituationRecordKind.PLAN_STATE.value: 1}
    assert hydrated.state_counts == {BrainActiveSituationRecordState.UNRESOLVED.value: 1}
    assert hydrated.uncertainty_code_counts == {"missing_input": 1}
    assert hydrated.unresolved_record_ids == ["situation-1"]
    assert hydrated.linked_commitment_ids == ["commitment-1"]
    assert hydrated.linked_plan_proposal_ids == ["proposal-1"]
    assert hydrated.linked_skill_ids == ["skill-1"]


def test_active_state_record_parsers_ignore_none_like_optional_links():
    # Regression: fuzzing found None-like optional ids getting stringified into live links.
    private_record = BrainPrivateWorkingMemoryRecord.from_dict(
        {
            "record_id": "pwm-1",
            "buffer_kind": BrainPrivateWorkingMemoryBufferKind.PLAN_ASSUMPTION.value,
            "summary": "Need operator review before adoption.",
            "state": BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
            "evidence_kind": BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value,
            "goal_id": " None ",
            "commitment_id": "none",
            "plan_proposal_id": "",
            "skill_id": "None",
        }
    )
    assert private_record is not None
    assert private_record.goal_id is None
    assert private_record.commitment_id is None
    assert private_record.plan_proposal_id is None
    assert private_record.skill_id is None

    situation_record = BrainActiveSituationRecord.from_dict(
        {
            "record_id": "situation-1",
            "record_kind": BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value,
            "summary": "Scene freshness degraded.",
            "state": BrainActiveSituationRecordState.UNRESOLVED.value,
            "evidence_kind": BrainActiveSituationEvidenceKind.DERIVED.value,
            "goal_id": "None",
            "commitment_id": " none ",
            "plan_proposal_id": "None",
            "skill_id": " none ",
            "freshness": " none ",
        }
    )
    assert situation_record is not None
    assert situation_record.goal_id is None
    assert situation_record.commitment_id is None
    assert situation_record.plan_proposal_id is None
    assert situation_record.skill_id is None
    assert situation_record.freshness is None

    entity_record = BrainSceneWorldEntityRecord.from_dict(
        {
            "entity_id": "entity-1",
            "entity_kind": BrainSceneWorldEntityKind.OBJECT.value,
            "canonical_label": "cup",
            "summary": "A cup is visible on the desk.",
            "state": BrainSceneWorldRecordState.ACTIVE.value,
            "evidence_kind": BrainSceneWorldEvidenceKind.OBSERVED.value,
            "zone_id": " none ",
            "freshness": "None",
            "observed_at": "None",
            "expires_at": " none ",
        }
    )
    assert entity_record is not None
    assert entity_record.zone_id is None
    assert entity_record.freshness is None
    assert entity_record.observed_at is None
    assert entity_record.expires_at is None


def test_active_situation_projection_compiles_bounded_stale_and_unresolved_state():
    private_records = [
        BrainPrivateWorkingMemoryRecord(
            record_id="scene-private-1",
            buffer_kind=BrainPrivateWorkingMemoryBufferKind.SCENE_WORLD_STATE.value,
            summary="Desk scene is old and should be refreshed.",
            state=BrainPrivateWorkingMemoryRecordState.STALE.value,
            evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.OBSERVED.value,
            backing_ids=["scene_state"],
            source_event_ids=["evt-scene-1"],
            observed_at=_ts(1),
            updated_at=_ts(40),
            expires_at=_ts(20),
            details={"kind": "scene_world_state"},
        )
    ]
    for index in range(8):
        private_records.append(
            BrainPrivateWorkingMemoryRecord(
                record_id=f"uncertainty-{index}",
                buffer_kind=BrainPrivateWorkingMemoryBufferKind.UNRESOLVED_UNCERTAINTY.value,
                summary=f"Missing input #{index}",
                state=BrainPrivateWorkingMemoryRecordState.ACTIVE.value,
                evidence_kind=BrainPrivateWorkingMemoryEvidenceKind.HYPOTHESIZED.value,
                backing_ids=[f"proposal-{index}"],
                source_event_ids=[f"evt-uncertainty-{index}"],
                goal_id="goal-1",
                commitment_id="commitment-1",
                plan_proposal_id=f"proposal-{index}",
                observed_at=_ts(index + 2),
                updated_at=_ts(index + 2),
                details={"kind": "missing_input"},
            )
        )
    private_working_memory = BrainPrivateWorkingMemoryProjection(
        scope_type="thread",
        scope_id="thread-1",
        records=private_records,
        updated_at=_ts(50),
    )

    agenda = BrainAgendaProjection(
        goals=[
            BrainGoal(
                goal_id="goal-1",
                title="Ship Alpha",
                intent="task.ship_alpha",
                source="test",
                goal_family="conversation",
                commitment_id="commitment-1",
                status="blocked",
                blocked_reason=BrainBlockedReason(
                    kind=BrainBlockedReasonKind.WAITING_USER.value,
                    summary="Waiting for the missing delivery date.",
                ),
                updated_at=_ts(45),
                created_at=_ts(5),
            )
        ],
        updated_at=_ts(45),
    )
    agenda.sync_lists()

    commitment_projection = BrainCommitmentProjection(
        active_commitments=[
            BrainCommitmentRecord(
                commitment_id="commitment-1",
                scope_type="relationship",
                scope_id="relationship-1",
                title="Ship Alpha",
                goal_family="conversation",
                intent="task.ship_alpha",
                status="active",
                details={
                    "plan_review_policy": "needs_user_review",
                    "pending_plan_proposal_id": "proposal-1",
                },
                current_goal_id="goal-1",
                updated_at=_ts(46),
                created_at=_ts(5),
            )
        ],
        updated_at=_ts(46),
    )

    scene = BrainSceneStateProjection(
        camera_connected=True,
        camera_track_state="stalled",
        person_present="uncertain",
        scene_change_state="stable",
        last_visual_summary="Desk scene.",
        last_observed_at=_ts(40),
        last_fresh_frame_at=_ts(0),
        frame_age_ms=20_000,
        sensor_health_reason="stale_frame",
        confidence=0.6,
        updated_at=_ts(50),
    )
    engagement = BrainEngagementStateProjection(
        engagement_state="away",
        attention_to_camera="low",
        user_present=False,
        updated_at=_ts(50),
    )
    body = BrainPresenceSnapshot(
        runtime_kind="browser",
        vision_enabled=True,
        vision_connected=True,
        camera_track_state="stalled",
        perception_unreliable=True,
        last_fresh_frame_at=_ts(0),
        frame_age_ms=20_000,
        updated_at=_ts(50),
    )
    procedural_skills = BrainProceduralSkillProjection(
        scope_type="thread",
        scope_id="thread-1",
        skill_counts={"active": 1},
        confidence_band_counts={"high": 1},
        skills=[
            BrainProceduralSkillRecord(
                skill_id="skill-1",
                skill_family_key="family-1",
                template_fingerprint="fp-1",
                scope_type="thread",
                scope_id="thread-1",
                title="Ship Alpha Skill",
                purpose="Reuse a known shipping procedure.",
                goal_family="conversation",
                status="active",
                confidence=0.82,
                step_template=[BrainGoalStep(capability_id="shipping.prepare")],
                required_capability_ids=["shipping.prepare"],
                stats=BrainProceduralSkillStatsRecord(
                    support_trace_count=2,
                    success_trace_count=2,
                    independent_plan_count=2,
                    last_supported_at=_ts(30),
                ),
                supporting_trace_ids=["trace-1", "trace-2"],
                supporting_outcome_ids=["outcome-1"],
                supporting_plan_proposal_ids=["proposal-1"],
                supporting_commitment_ids=["commitment-1"],
                created_at=_ts(30),
                updated_at=_ts(35),
            )
        ],
        active_skill_ids=["skill-1"],
    )
    scene_world_state = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=[
            BrainSceneWorldEntityRecord(
                entity_id="entity-1",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="shipping_label",
                summary="Shipping label remains visible on the desk.",
                state=BrainSceneWorldRecordState.STALE.value,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                zone_id="zone:desk",
                confidence=0.7,
                freshness="stale",
                affordance_ids=["affordance-1"],
                backing_ids=["entity:shipping_label"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(40),
                updated_at=_ts(40),
                expires_at=_ts(80),
                details={"stable_key": "entity:shipping_label"},
            )
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="affordance-1",
                entity_id="entity-1",
                capability_family="vision.inspect",
                summary="The label can still be inspected.",
                availability=BrainSceneWorldAffordanceAvailability.UNCERTAIN.value,
                confidence=0.6,
                freshness="stale",
                reason_codes=["scene_stale"],
                backing_ids=["entity:shipping_label", "vision.inspect"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(40),
                updated_at=_ts(40),
                expires_at=_ts(80),
            )
        ],
        degraded_mode="limited",
        degraded_reason_codes=["scene_stale"],
        updated_at=_ts(50),
    )

    projection = build_active_situation_model_projection(
        scope_type="thread",
        scope_id="thread-1",
        private_working_memory=private_working_memory,
        agenda=agenda,
        commitment_projection=commitment_projection,
        scene=scene,
        scene_world_state=scene_world_state,
        engagement=engagement,
        body=body,
        continuity_graph=None,
        continuity_dossiers=None,
        procedural_skills=procedural_skills,
        recent_events=[],
        planning_digest={
            "current_pending_proposals": [
                {
                    "plan_proposal_id": "proposal-1",
                    "goal_id": "goal-1",
                    "commitment_id": "commitment-1",
                    "title": "Ship Alpha plan",
                    "review_policy": "needs_user_review",
                    "missing_inputs": ["delivery date"],
                    "assumptions": ["the user still wants shipment"],
                    "selected_skill_id": "skill-1",
                    "procedural_origin": "skill_reuse",
                }
            ],
            "current_plan_states": [],
            "recent_rejections": [],
            "recent_revision_flows": [],
            "recent_selected_skill_ids": ["skill-1"],
        },
        reference_ts=_ts(50),
    )

    assert projection.kind_counts[BrainActiveSituationRecordKind.SCENE_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.GOAL_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.COMMITMENT_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.PLAN_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.PROCEDURAL_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.WORLD_STATE.value] == 1
    assert projection.kind_counts[BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value] == 6
    assert len(
        [
            record
            for record in projection.records
            if record.record_kind == BrainActiveSituationRecordKind.UNCERTAINTY_STATE.value
        ]
    ) == 6
    assert any(
        record.record_kind == BrainActiveSituationRecordKind.SCENE_STATE.value
        and record.state == BrainActiveSituationRecordState.STALE.value
        for record in projection.records
    )
    assert any(
        record.record_kind == BrainActiveSituationRecordKind.COMMITMENT_STATE.value
        and record.state == BrainActiveSituationRecordState.UNRESOLVED.value
        and record.commitment_id == "commitment-1"
        for record in projection.records
    )
    assert any(
        record.record_kind == BrainActiveSituationRecordKind.PLAN_STATE.value
        and record.state == BrainActiveSituationRecordState.UNRESOLVED.value
        and record.plan_proposal_id == "proposal-1"
        for record in projection.records
    )
    assert any(
        record.record_kind == BrainActiveSituationRecordKind.PROCEDURAL_STATE.value
        and record.skill_id == "skill-1"
        and record.plan_proposal_id == "proposal-1"
        for record in projection.records
    )
    assert any(
        record.record_kind == BrainActiveSituationRecordKind.WORLD_STATE.value
        and record.state == BrainActiveSituationRecordState.STALE.value
        and record.details.get("zone_id") == "zone:desk"
        for record in projection.records
    )
    assert "scene_stale" in projection.uncertainty_code_counts
    assert "missing_input" in projection.uncertainty_code_counts
    assert projection.linked_commitment_ids == ["commitment-1"]
    assert "proposal-1" in projection.linked_plan_proposal_ids
    assert projection.linked_skill_ids == ["skill-1"]

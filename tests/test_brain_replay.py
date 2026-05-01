import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from blink.brain.actions import (
    EmbodiedActionEngine,
    EmbodiedActionLibrary,
    EmbodiedCapabilityDispatcher,
    build_brain_capability_registry,
    build_embodied_capability_registry,
)
from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.world_model import (
    WorldModelAdapterRequest,
    WorldModelAdapterResponse,
    WorldModelPredictionProposal,
)
from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder
from blink.brain.evals import (
    evaluate_replay_regression_case,
    load_replay_regression_case,
    load_replay_regression_cases,
)
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive, BrainGoalStatus
from blink.brain.identity import base_brain_system_prompt, load_default_agent_blocks
from blink.brain.memory_v2 import (
    BrainContinuityDossierGovernanceRecord,
    BrainContinuityDossierProjection,
    BrainContinuityDossierRecord,
)
from blink.brain.presence import BrainPresenceSnapshot
from blink.brain.projections import (
    BrainActiveSituationProjection,
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentScopeType,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.replay import BrainReplayHarness, BrainReplayScenario
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


class _ReplayPolicySurfaceBuilder:
    """Wrap the real surface builder with one synthetic review-debt dossier."""

    def __init__(self, *, store: BrainStore, session_ids, review_debt_count: int):
        self.review_debt_count = review_debt_count
        self._session_ids = session_ids
        self._builder = BrainContextSurfaceBuilder(
            store=store,
            session_resolver=lambda: session_ids,
            presence_scope_key="browser:presence",
            language=Language.EN,
            capability_registry=CapabilityRegistry(),
        )

    def build(self, **kwargs):
        base = self._builder.build(**kwargs)
        active_situation_model = BrainActiveSituationProjection(
            scope_type=base.active_situation_model.scope_type,
            scope_id=base.active_situation_model.scope_id,
            updated_at=base.generated_at,
        )
        scene_world_state = replace(
            base.scene_world_state,
            degraded_mode="healthy",
            degraded_reason_codes=[],
        )
        if self.review_debt_count <= 0:
            return replace(
                base,
                active_situation_model=active_situation_model,
                scene_world_state=scene_world_state,
            )
        dossiers = list(base.continuity_dossiers.dossiers if base.continuity_dossiers else [])
        dossiers = [
            record for record in dossiers if record.dossier_id != "replay-policy-review-debt"
        ]
        dossiers.append(
            BrainContinuityDossierRecord(
                dossier_id="replay-policy-review-debt",
                kind="relationship",
                scope_type="relationship",
                scope_id=f"{self._session_ids.agent_id}:{self._session_ids.user_id}",
                title="Relationship",
                summary="Replay fixture review debt is still open.",
                status="current",
                freshness="fresh",
                contradiction="clear",
                support_strength=0.8,
                governance=BrainContinuityDossierGovernanceRecord(
                    review_debt_count=self.review_debt_count,
                    last_refresh_cause="fresh_current_support",
                ),
            )
        )
        projection = BrainContinuityDossierProjection(
            scope_type=(
                base.continuity_dossiers.scope_type
                if base.continuity_dossiers is not None
                else "user"
            ),
            scope_id=(
                base.continuity_dossiers.scope_id
                if base.continuity_dossiers is not None
                else self._session_ids.user_id
            ),
            dossiers=sorted(
                dossiers,
                key=lambda record: (
                    record.kind,
                    record.scope_type,
                    record.scope_id,
                    record.dossier_id,
                ),
            ),
            dossier_counts={"relationship": len(dossiers)},
            freshness_counts={"fresh": len(dossiers)},
            contradiction_counts={"clear": len(dossiers)},
            current_dossier_ids=[record.dossier_id for record in dossiers],
        )
        return replace(
            base,
            active_situation_model=active_situation_model,
            continuity_dossiers=projection,
            scene_world_state=scene_world_state,
        )


def _policy_candidate(candidate_goal_id: str, summary: str) -> BrainCandidateGoal:
    return BrainCandidateGoal(
        candidate_goal_id=candidate_goal_id,
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary=summary,
        goal_family="environment",
        urgency=0.75,
        confidence=0.92,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key=f"cooldown:{candidate_goal_id}",
        dedupe_key=f"dedupe:{candidate_goal_id}",
        policy_tags=["phase15a", "replay"],
        requires_user_turn_gap=False,
        expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        payload={"kind": "presence", "candidate_goal_id": candidate_goal_id},
        created_at=(datetime.now(UTC) - timedelta(seconds=15)).isoformat(),
    )


def test_brain_replay_harness_rebuilds_phase1_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
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
        payload={"text": "提醒我给妈妈打电话。"},
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="memory",
        payload={
            "goal": {
                "goal_id": "goal-1",
                "title": "给妈妈打电话",
                "intent": "narrative.commitment",
                "source": "memory",
                "status": "open",
                "details": {},
                "steps": [],
                "active_step_index": None,
                "recovery_count": 0,
                "planning_requested": False,
                "last_summary": None,
                "last_error": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_CALLED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        correlation_id="call_1",
        payload={
            "tool_call_id": "call_1",
            "function_name": "fetch_user_image",
            "arguments": {"question": "我手里拿着什么？"},
        },
    )
    store.append_brain_event(
        event_type=BrainEventType.TOOL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="turn-recorder",
        correlation_id="call_1",
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
        payload={"text": "好的，我会记住，也看到你拿着一个杯子。"},
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
            "action_id": "cmd_blink",
            "accepted": True,
            "preview_only": False,
            "summary": "Blink executed cmd_blink.",
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
        presence_scope_key="browser:presence",
    )

    assert result.context_surface.body.robot_head_last_action == "cmd_blink"
    assert result.context_surface.working_context.last_tool_name == "fetch_user_image"
    assert result.context_surface.agenda.open_goals == ["给妈妈打电话"]
    assert result.artifact_path.exists()
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert payload["scenario"]["name"] == "phase1_turn_tool_robot_action"
    assert "self_core" in payload["core_blocks"]
    assert payload["core_block_versions"]["self_core"][0]["version"] == 1


def test_brain_replay_harness_uses_unique_temp_store_paths(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="context",
        payload={"text": "hello"},
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_turn_tool_robot_action",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )

    first = harness.replay(scenario)
    second = harness.replay(scenario)

    assert first.replay_store_path != second.replay_store_path
    assert first.replay_store_path.exists()
    assert second.replay_store_path.exists()


def test_brain_replay_harness_rebuilds_private_working_memory_deterministically(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="wm-replay")
    proposal = {
        "plan_proposal_id": "proposal-wm-1",
        "goal_id": "goal-wm-1",
        "commitment_id": "commitment-wm-1",
        "source": "bounded_planner",
        "summary": "Need user confirmation before sending.",
        "current_plan_revision": 1,
        "plan_revision": 1,
        "review_policy": "needs_user_review",
        "steps": [],
        "assumptions": ["the request still applies"],
        "missing_inputs": ["which address to use"],
        "details": {"request_kind": "initial_plan"},
        "created_at": "2026-01-01T00:00:01+00:00",
    }
    requested = store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_REQUESTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal_id": "goal-wm-1", "capability_id": "capability.preview", "step_index": 0},
        ts="2026-01-01T00:00:00+00:00",
    )
    store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": "goal-wm-1",
            "capability_id": "capability.preview",
            "step_index": 0,
            "result": {"summary": "Preview completed."},
        },
        causal_parent_id=requested.event_id,
        ts="2026-01-01T00:00:01+00:00",
    )
    proposed = store.append_brain_event(
        event_type=BrainEventType.PLANNING_PROPOSED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"proposal": proposal},
        ts="2026-01-01T00:00:02+00:00",
    )
    store.append_brain_event(
        event_type=BrainEventType.PLANNING_REJECTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "proposal": proposal,
            "decision": {"summary": "Need fresh input.", "reason": "missing_required_input"},
        },
        causal_parent_id=proposed.event_id,
        ts="2026-01-01T00:00:03+00:00",
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
                "updated_at": "2026-01-01T00:00:03+00:00",
            },
        },
        ts="2026-01-01T00:00:03+00:00",
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
            "summary": "A package is on the desk.",
            "last_fresh_frame_at": "2026-01-01T00:00:00+00:00",
            "frame_age_ms": 20000,
            "sensor_health_reason": "stale_frame",
            "scene_zones": [{"zone_key": "desk", "label": "Desk", "summary": "Desk work zone."}],
            "scene_entities": [
                {
                    "entity_key": "package",
                    "kind": "object",
                    "label": "package",
                    "summary": "A package is on the desk.",
                    "zone_key": "desk",
                    "affordances": [
                        {
                            "capability_family": "vision.inspect",
                            "summary": "Inspect the package.",
                            "availability": "available",
                        }
                    ],
                }
            ],
        },
        ts="2026-01-01T00:00:04+00:00",
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="private_working_memory_replay",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    first = harness.replay(scenario, presence_scope_key="browser:presence")
    second = harness.replay(scenario, presence_scope_key="browser:presence")

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))
    assert first_payload["private_working_memory"] == second_payload["private_working_memory"]
    assert (
        first_payload["private_working_memory_digest"]
        == second_payload["private_working_memory_digest"]
    )
    assert first_payload["scene_world_state"] == second_payload["scene_world_state"]
    assert first_payload["scene_world_state_digest"] == second_payload["scene_world_state_digest"]
    assert first_payload["active_situation_model"] == second_payload["active_situation_model"]
    assert (
        first_payload["active_situation_model_digest"]
        == second_payload["active_situation_model_digest"]
    )
    assert (
        first_payload["continuity_state"]["private_working_memory"]["buffer_counts"][
            "recent_tool_outcome"
        ]
        >= 1
    )
    assert "unresolved_uncertainty" in first_payload["private_working_memory"]["buffer_counts"]
    assert "scene_world_state" in first_payload["continuity_state"]
    assert "scene_world_state_digest" in first_payload["continuity_state"]
    assert "active_situation_model" in first_payload["continuity_state"]
    assert "active_situation_model_digest" in first_payload["continuity_state"]
    assert first_payload["scene_world_state"]["entity_counts"]["object"] == 1
    assert first_payload["scene_world_state_digest"]["degraded_mode"] == "limited"
    assert first_payload["scene_world_state_digest"]["uncertain_affordance_ids"]
    assert "scene_state" in first_payload["active_situation_model"]["kind_counts"]
    assert "world_state" in first_payload["active_situation_model"]["kind_counts"]
    assert "scene_stale" in first_payload["active_situation_model"]["uncertainty_code_counts"]


def test_brain_replay_harness_rebuilds_multimodal_autobiography_deterministically(tmp_path):
    from blink.brain.projections import BrainGovernanceReasonCode
    from tests.test_brain_memory_v2 import (
        _scene_world_projection_for_multimodal,
        _seed_scene_episode,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="multimodal-replay",
    )

    first_entry = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-multimodal-1"],
            updated_at="2026-01-01T00:00:10+00:00",
            include_person=True,
        ),
        start_second=10,
        include_attention=True,
    )
    second_entry = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-multimodal-2"],
            updated_at="2026-01-01T00:00:40+00:00",
            degraded_mode="limited",
            include_person=True,
            affordance_availability="blocked",
            object_state="stale",
        ),
        start_second=40,
        include_attention=True,
    )
    assert first_entry is not None
    assert second_entry is not None
    assert first_entry.entry_id != second_entry.entry_id

    store.redact_autobiographical_entry(
        second_entry.entry_id,
        redacted_summary="Redacted scene episode.",
        source_event_id="evt-multimodal-redact",
        reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
        event_context=store._memory_event_context(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            source="test",
            correlation_id="multimodal-redact",
        ),
    )

    scenario = BrainReplayScenario(
        name="phase17_multimodal_autobiography",
        description="Scene-first multimodal autobiography remains replay-stable.",
        session_ids=session_ids,
        events=tuple(
            reversed(
                store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=128,
                )
            )
        ),
        expected_terminal_state={},
    )
    harness = BrainReplayHarness(store=store)
    first = harness.replay(scenario, presence_scope_key="browser:presence")
    second = harness.replay(scenario, presence_scope_key="browser:presence")

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["multimodal_autobiography"] == second_payload["multimodal_autobiography"]
    assert (
        first_payload["multimodal_autobiography_digest"]
        == second_payload["multimodal_autobiography_digest"]
    )
    assert (
        first_payload["continuity_state"]["multimodal_autobiography"]
        == second_payload["continuity_state"]["multimodal_autobiography"]
    )
    assert (
        first_payload["continuity_state"]["multimodal_autobiography_digest"]
        == second_payload["continuity_state"]["multimodal_autobiography_digest"]
    )
    assert (
        first_payload["continuity_state"]["context_packet_digest"]
        == second_payload["continuity_state"]["context_packet_digest"]
    )
    assert first_payload["multimodal_autobiography_digest"]["entry_counts"]["privacy"]["redacted"] >= 1
    assert first_payload["multimodal_autobiography_digest"]["recent_redacted_rows"]
    assert (
        first_payload["multimodal_autobiography_digest"]["recent_redacted_rows"][0][
            "rendered_summary"
        ]
        == "Redacted scene episode."
    )
    assert (
        first_payload["continuity_state"]["context_packet_digest"]["reply"]["scene_episode_trace"][
            "suppressed_entry_ids"
        ]
    )
    assert (
        first_payload["continuity_state"]["context_packet_digest"]["operator_audit"][
            "scene_episode_trace"
        ]["selected_entry_ids"]
    )


def test_brain_replay_harness_rebuilds_predictive_world_model_deterministically(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_goal_updated,
        _append_robot_action_outcome,
        _append_scene_changed,
        _ensure_blocks,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    _ensure_blocks(store)
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="predictive-replay",
    )
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

    scenario = BrainReplayScenario(
        name="phase18a_predictive_world_model",
        description="Predictive world-model projections remain replay-stable.",
        session_ids=session_ids,
        events=tuple(
            reversed(
                store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=160,
                )
            )
        ),
        expected_terminal_state={},
    )
    harness = BrainReplayHarness(store=store)
    first = harness.replay(scenario, presence_scope_key="browser:presence")
    second = harness.replay(scenario, presence_scope_key="browser:presence")

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert (
        first_payload["continuity_state"]["predictive_world_model"]
        == second_payload["continuity_state"]["predictive_world_model"]
    )
    assert (
        first_payload["continuity_state"]["predictive_digest"]
        == second_payload["continuity_state"]["predictive_digest"]
    )
    assert (
        first_payload["continuity_state"]["active_situation_model"]
        == second_payload["continuity_state"]["active_situation_model"]
    )
    assert (
        first_payload["continuity_state"]["runtime_shell_digest"]["predictive_inspection"]
        == second_payload["continuity_state"]["runtime_shell_digest"]["predictive_inspection"]
    )
    assert (
        first_payload["continuity_state"]["context_packet_digest"]["operator_audit"][
            "prediction_trace"
        ]
        == second_payload["continuity_state"]["context_packet_digest"]["operator_audit"][
            "prediction_trace"
        ]
    )


def test_brain_replay_harness_uses_stored_adapter_generated_prediction_events(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_scene_changed,
        _ensure_blocks,
    )

    class ReplayOnlyWorldModelAdapter:
        def __init__(self):
            self.descriptor = BrainAdapterDescriptor(
                backend_id="replay_only_world_model",
                backend_version="v1",
                capabilities=("prediction_proposal",),
                degraded_mode_id="empty_proposals",
                default_timeout_ms=10,
            )

        def propose_predictions(self, request: WorldModelAdapterRequest) -> WorldModelAdapterResponse:
            return WorldModelAdapterResponse(
                proposals=(
                    WorldModelPredictionProposal(
                        prediction_kind="action_outcome",
                        subject_kind="action",
                        subject_id="replay-only-action",
                        summary="Replay should preserve this adapter-generated prediction.",
                        predicted_state={"accepted": True},
                        confidence=0.81,
                        risk_codes=("replay_only",),
                        supporting_event_ids=("replay-only-event",),
                        backing_ids=("replay-only-action",),
                        plan_proposal_id="replay-only-plan",
                        details={"prediction_role": "opportunity"},
                    ),
                ),
            )

    store = BrainStore(
        path=tmp_path / "brain.db",
        world_model_adapter=ReplayOnlyWorldModelAdapter(),
    )
    _ensure_blocks(store)
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="predictive-replay-adapter",
    )
    _append_body_state(store, session_ids, second=1)
    _append_scene_changed(store, session_ids, second=2)
    _append_goal_created(store, session_ids, second=3)
    _append_scene_changed(store, session_ids, second=4)

    scenario = BrainReplayScenario(
        name="phase20_adapter_generated_predictive_world_model",
        description="Replay rebuilds stored adapter-generated predictive lifecycle events without rerunning heuristics.",
        session_ids=session_ids,
        events=tuple(
            reversed(
                store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=160,
                )
            )
        ),
        expected_terminal_state={},
    )
    harness = BrainReplayHarness(store=store)
    first = harness.replay(scenario, presence_scope_key="browser:presence")
    second = harness.replay(scenario, presence_scope_key="browser:presence")

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))
    first_predictions = first_payload["continuity_state"]["predictive_world_model"]["active_predictions"]
    second_predictions = second_payload["continuity_state"]["predictive_world_model"]["active_predictions"]

    assert any(record["subject_id"] == "replay-only-action" for record in first_predictions)
    assert first_predictions == second_predictions
    assert first_payload["continuity_state"]["predictive_digest"] == second_payload["continuity_state"]["predictive_digest"]


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_embodied_executive_deterministically(tmp_path):
    from blink.embodiment.robot_head.simulation import (
        RobotHeadSimulationConfig,
        SimulationDriver,
    )
    from tests.test_brain_embodied_executive import _build_runtime, _create_robot_goal

    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="embodied-replay",
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation-replay"),
        ),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Replay one embodied coordinator dispatch")
        planning_result = await runtime.executive.request_plan_proposal(goal_id=goal_id)
        cycle_result = await runtime.executive.run_once()
        source_events = tuple(
            reversed(
                runtime.store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=128,
                )
            )
        )
        event_types = [event.event_type for event in source_events]
        capability_completed = next(
            event for event in source_events if event.event_type == BrainEventType.CAPABILITY_COMPLETED
        )
        embodied_completed = next(
            event
            for event in source_events
            if event.event_type == BrainEventType.EMBODIED_DISPATCH_COMPLETED
        )

        scenario = BrainReplayScenario(
            name="phase19_embodied_executive",
            description="Hierarchical embodied executive projections remain replay-stable.",
            session_ids=session_ids,
            events=source_events,
        )
        harness = BrainReplayHarness(store=runtime.store)
        first = harness.replay(scenario, presence_scope_key=runtime.presence_scope_key)
        second = harness.replay(scenario, presence_scope_key=runtime.presence_scope_key)

        first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
        second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

        assert planning_result.outcome == "auto_adopted"
        assert cycle_result.progressed is True
        assert embodied_completed.causal_parent_id == capability_completed.event_id
        assert event_types.index(BrainEventType.EMBODIED_INTENT_SELECTED) < event_types.index(
            BrainEventType.EMBODIED_DISPATCH_PREPARED
        )
        assert event_types.index(BrainEventType.CAPABILITY_REQUESTED) < event_types.index(
            BrainEventType.ROBOT_ACTION_OUTCOME
        )
        assert event_types.index(BrainEventType.ROBOT_ACTION_OUTCOME) < event_types.index(
            BrainEventType.EMBODIED_DISPATCH_COMPLETED
        )
        assert (
            first_payload["continuity_state"]["embodied_executive"]
            == second_payload["continuity_state"]["embodied_executive"]
        )
        assert (
            first_payload["continuity_state"]["embodied_digest"]
            == second_payload["continuity_state"]["embodied_digest"]
        )
        assert (
            first_payload["continuity_state"]["runtime_shell_digest"]["embodied_inspection"]
            == second_payload["continuity_state"]["runtime_shell_digest"]["embodied_inspection"]
        )
        assert first_payload["continuity_governance_report"]["embodied_execution_rows"]
        assert (
            first_payload["continuity_governance_report"]["embodied_execution_rows"]
            == second_payload["continuity_governance_report"]["embodied_execution_rows"]
        )
    finally:
        await controller.close()
        runtime.close()


def test_brain_replay_harness_rebuilds_phase1_autonomy_ledger(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-replay-1",
        candidate_type="presence_acknowledgement",
        source="perception",
        summary="Person re-entered frame.",
        goal_family="environment",
        urgency=0.7,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id=candidate.candidate_goal_id,
        reason="user_turn_open",
        expected_reevaluation_condition="after the user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        expected_reevaluation_condition_details={"turn": "user"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_candidate_goal_lifecycle",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert [item.candidate_goal_id for item in result.autonomy_ledger.current_candidates] == [
        candidate.candidate_goal_id
    ]
    assert (
        result.autonomy_ledger.current_candidates[0].expected_reevaluation_condition_kind
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )
    assert payload["projections"]["autonomy_ledger"]["current_candidates"]
    assert payload["autonomy_ledger"]["recent_entries"]
    assert payload["autonomy_digest"]["reason_counts"]["user_turn_open"] == 1
    assert (
        payload["autonomy_ledger"]["recent_entries"][-1]["expected_reevaluation_condition_kind"]
        == BrainReevaluationConditionKind.USER_TURN_CLOSED.value
    )


def test_brain_replay_harness_rebuilds_presence_director_acceptance(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-director-accept",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Accept this candidate into the agenda.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:director-accept",
        dedupe_key="dedupe:director-accept",
        requires_user_turn_gap=False,
        expires_at="2099-01-01T00:05:00+00:00",
        payload={"goal_details": {"channel": "replay"}},
        created_at="2099-01-01T00:00:00+00:00",
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    decision = executive.run_presence_director_pass()

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase6_presence_director_acceptance",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    replay_goal = result.context_surface.agenda.goal(decision.accepted_goal_id)

    assert replay_goal is not None
    assert replay_goal.details["autonomy"]["candidate_goal_id"] == candidate.candidate_goal_id
    assert result.autonomy_ledger.current_candidates == []
    accepted_entries = [
        entry
        for entry in payload["autonomy_ledger"]["recent_entries"]
        if entry["decision_kind"] == "accepted"
    ]
    assert accepted_entries
    assert accepted_entries[-1]["accepted_goal_id"] == decision.accepted_goal_id


def test_brain_replay_harness_rebuilds_turn_close_candidate_reevaluation(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-turn-close-reeval",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Resume after the user turn closes.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:turn-close",
        dedupe_key="dedupe:turn-close",
        requires_user_turn_gap=True,
        expires_at="2099-01-01T00:05:00+00:00",
        created_at="2099-01-01T00:00:00+00:00",
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()
    turn_end = store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    decision = executive.run_presence_director_reevaluation(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            summary="User turn closed.",
            details={"turn": "user"},
            source_event_type=turn_end.event_type,
            source_event_id=turn_end.event_id,
            ts=turn_end.ts,
        )
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase7_turn_close_candidate_reevaluation",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=24,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert result.context_surface.agenda.goal(decision.accepted_goal_id) is not None
    assert (
        len(
            [
                event
                for event in payload["events"]
                if event["event_type"] == BrainEventType.GOAL_CANDIDATE_CREATED
            ]
        )
        == 1
    )
    trigger_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
    )
    accepted_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.GOAL_CANDIDATE_ACCEPTED
    )
    assert trigger_index < accepted_index
    assert result.autonomy_ledger.current_candidates == []


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_startup_recovery_candidate_reevaluation(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-startup-reeval",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Resume on startup recovery.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:startup",
        dedupe_key="dedupe:startup",
        requires_user_turn_gap=True,
        expires_at="2099-01-01T00:05:00+00:00",
        created_at="2099-01-01T00:00:00+00:00",
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_ENDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )

    await executive.run_startup_pass()

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase7_startup_recovery_candidate_reevaluation",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=24,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert (
        len(
            [
                event
                for event in payload["events"]
                if event["event_type"] == BrainEventType.GOAL_CANDIDATE_CREATED
            ]
        )
        == 1
    )
    assert any(
        event["event_type"] == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
        and event["payload"]["trigger"]["kind"]
        == BrainReevaluationConditionKind.STARTUP_RECOVERY.value
        for event in payload["events"]
    )
    assert result.autonomy_ledger.current_candidates == []


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_scene_capability_route(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-scene-route",
        candidate_type="presence_user_reentered",
        source=BrainCandidateGoalSource.PERCEPTION.value,
        summary="User re-entered the frame.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:scene-route",
        dedupe_key="dedupe:scene-route",
        requires_user_turn_gap=False,
        expires_at="2099-01-01T00:05:00+00:00",
        payload={
            "goal_intent": "autonomy.presence_user_reentered",
            "goal_details": {
                "scene_candidate": {
                    "presence_scope_key": "browser:presence",
                    "person_present": "present",
                    "attention_to_camera": "toward_camera",
                    "engagement_state": "engaged",
                }
            },
        },
        created_at="2099-01-01T00:00:00+00:00",
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    decision = executive.run_presence_director_pass()
    await executive.run_until_quiescent(max_iterations=4)

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase6_scene_capability_route",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    replay_goal = result.context_surface.agenda.goal(decision.accepted_goal_id)

    assert replay_goal is not None
    assert replay_goal.status == BrainGoalStatus.COMPLETED.value
    assert [step.capability_id for step in replay_goal.steps] == [
        "observation.inspect_presence_state",
        "reporting.record_presence_event",
    ]
    assert replay_goal.steps[1].output["candidate_type"] == "presence_user_reentered"
    assert any(
        event["event_type"] == BrainEventType.CAPABILITY_COMPLETED for event in payload["events"]
    )


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_commitment_wake_candidate(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
    )
    executive.create_commitment_goal(
        title="Follow up when idle",
        intent="narrative.commitment",
        source="memory",
        details={"summary": "Need a follow-up."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    executive.defer_commitment(
        commitment_id=commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Wait until the thread is idle.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake on thread idle.",
            )
        ],
    )

    await executive.run_startup_pass()
    await executive.run_until_quiescent(max_iterations=4)
    accepted_goal = next(
        goal
        for goal in store.get_agenda_projection(
            scope_key=session_ids.thread_id, user_id=session_ids.user_id
        ).goals
        if goal.intent == "autonomy.commitment_wake_thread_idle"
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase6_commitment_wake_candidate",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    replay_goal = result.context_surface.agenda.goal(accepted_goal.goal_id)

    assert replay_goal is not None
    assert replay_goal.status == BrainGoalStatus.COMPLETED.value
    assert [step.capability_id for step in replay_goal.steps] == [
        "reporting.record_commitment_wake"
    ]
    assert replay_goal.steps[0].output["commitment_wake"]["wake_kind"] == "thread_idle"
    assert any(
        event["event_type"] == BrainEventType.COMMITMENT_WAKE_TRIGGERED
        for event in payload["events"]
    )
    wake_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.COMMITMENT_WAKE_TRIGGERED
    )
    candidate_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.GOAL_CANDIDATE_CREATED
    )
    assert wake_index < candidate_index
    assert any(
        event["event_type"] == BrainEventType.GOAL_CANDIDATE_ACCEPTED for event in payload["events"]
    )
    assert payload["autonomy_ledger"]["recent_entries"]


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_condition_cleared_direct_resume(tmp_path):
    from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
    from blink.embodiment.robot_head.controller import RobotHeadController
    from blink.embodiment.robot_head.drivers import FaultInjectionDriver

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    driver = FaultInjectionDriver(busy=True)
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(action_engine=action_engine),
    )

    executive.create_commitment_goal(
        title="Resume after condition clears",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )
    await executive.run_turn_end_pass()
    await asyncio.sleep(0.4)
    await executive.run_turn_end_pass()
    driver.busy = False
    resume_result = await executive.run_turn_end_pass()
    completion_result = await executive.run_turn_end_pass()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]

    assert resume_result.progressed is True
    assert completion_result.progressed is True
    assert commitment.status == BrainCommitmentStatus.COMPLETED.value

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase8_commitment_condition_cleared_resume",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=96,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    replay_commitments = result.context_surface.commitment_projection.recent_terminal_commitments

    assert replay_commitments
    assert replay_commitments[0].status == BrainCommitmentStatus.COMPLETED.value
    assert replay_commitments[0].resume_count == 1
    wake_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.COMMITMENT_WAKE_TRIGGERED
        and event["payload"]["routing"]["route_kind"] == "resume_direct"
    )
    resume_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.GOAL_RESUMED
    )
    complete_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.GOAL_COMPLETED
    )
    assert wake_index < resume_index < complete_index
    await controller.close()


def test_brain_replay_harness_rebuilds_timer_maintenance_candidate(tmp_path):
    db_path = tmp_path / "brain.db"
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
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
    runtime.reflection_scheduler.run_cycle(trigger="timer", force=True)
    asyncio.run(runtime.executive.run_until_quiescent(max_iterations=4))

    accepted_goal = next(
        goal
        for goal in runtime.store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        ).goals
        if goal.intent == "autonomy.maintenance_review_findings"
    )
    harness = BrainReplayHarness(store=runtime.store)
    scenario = harness.capture_builtin_scenario(
        name="phase6_timer_maintenance_candidate",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=96,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    replay_goal = result.context_surface.agenda.goal(accepted_goal.goal_id)

    assert replay_goal is not None
    assert replay_goal.status == BrainGoalStatus.COMPLETED.value
    assert [step.capability_id for step in replay_goal.steps] == [
        "maintenance.review_memory_health",
        "reporting.record_maintenance_note",
    ]
    assert (
        replay_goal.steps[0].output["report_id"]
        == accepted_goal.details["maintenance"]["report_id"]
    )
    assert replay_goal.steps[1].output["note_kind"] == "maintenance_review_findings"
    assert payload["reflection_cycles"]
    assert any(
        event["event_type"] == BrainEventType.GOAL_CANDIDATE_ACCEPTED for event in payload["events"]
    )
    runtime.close()


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_phase3_retry_recovery_agenda(tmp_path):
    from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
    from blink.embodiment.robot_head.controller import RobotHeadController
    from blink.embodiment.robot_head.drivers import FaultInjectionDriver

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    driver = FaultInjectionDriver(busy=True)
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(action_engine=action_engine),
    )

    goal_id = executive.create_goal(
        title="眨眼一次",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
    )
    await executive.run_once()
    driver.busy = False
    await executive.run_until_quiescent(max_iterations=4)

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase3_goal_retry_recovery",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    goal = result.context_surface.agenda.goal(goal_id)

    assert goal is not None
    assert goal.status == BrainGoalStatus.COMPLETED.value
    assert goal.steps[0].attempts == 2
    artifact_payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    critic_events = [
        event
        for event in artifact_payload["events"]
        if event["event_type"] == BrainEventType.CRITIC_FEEDBACK
    ]
    assert critic_events
    assert "reflection_cycles" in artifact_payload
    await controller.close()


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_resumed_commitment_state(tmp_path):
    from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
    from blink.embodiment.robot_head.controller import RobotHeadController
    from blink.embodiment.robot_head.drivers import FaultInjectionDriver

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    driver = FaultInjectionDriver(busy=True)
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=driver,
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_embodied_capability_registry(action_engine=action_engine),
    )

    executive.create_commitment_goal(
        title="Resume the blink sequence",
        intent="robot_head.sequence",
        source="interpreter",
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
        goal_status=BrainGoalStatus.OPEN.value,
    )
    await executive.run_turn_end_pass()
    await asyncio.sleep(0.4)
    await executive.run_turn_end_pass()
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    driver.busy = False
    executive.resume_commitment(commitment_id=commitment.commitment_id)
    await executive.run_turn_end_pass()

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase3_goal_retry_recovery",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    result = harness.replay(scenario)
    replay_commitments = result.context_surface.commitment_projection.recent_terminal_commitments
    artifact_payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    resumed_events = [
        event
        for event in artifact_payload["events"]
        if event["event_type"] == BrainEventType.GOAL_RESUMED
    ]

    assert replay_commitments
    assert replay_commitments[0].status == BrainCommitmentStatus.COMPLETED.value
    assert replay_commitments[0].resume_count == 1
    assert replay_commitments[0].scope_type == BrainCommitmentScopeType.THREAD.value
    assert artifact_payload["commitment_projection"]["recent_terminal_commitments"]
    assert resumed_events
    assert "reflection_cycles" in artifact_payload
    await controller.close()


@pytest.mark.asyncio
async def test_brain_replay_harness_rebuilds_phase4_presence_attention_cycle(tmp_path):
    from blink.embodiment.robot_head.catalog import build_default_robot_head_catalog
    from blink.embodiment.robot_head.controller import RobotHeadController
    from blink.embodiment.robot_head.drivers import MockDriver
    from blink.embodiment.robot_head.policy import EmbodimentPolicyProcessor
    from blink.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from blink.processors.frame_processor import FrameDirection
    from blink.tests.utils import SleepFrame, run_test

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    controller = RobotHeadController(
        catalog=build_default_robot_head_catalog(),
        driver=MockDriver(),
    )
    action_engine = EmbodiedActionEngine(
        library=EmbodiedActionLibrary.build_default(),
        controller=controller,
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    )
    processor = EmbodimentPolicyProcessor(
        action_dispatcher=EmbodiedCapabilityDispatcher(
            action_engine=action_engine,
            capability_registry=build_embodied_capability_registry(action_engine=action_engine),
        ),
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        idle_timeout_secs=0.01,
        presence_poll_secs=0.01,
    )
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
                robot_head_mode="mock",
                robot_head_available=True,
                vision_enabled=True,
                vision_connected=True,
                perception_disabled=False,
            ).as_dict(),
        },
    )

    async def emit_presence_events():
        await asyncio.sleep(0.03)
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
                "confidence": 0.9,
                "observed_at": "2026-04-17T10:00:00+00:00",
            },
        )
        await asyncio.sleep(0.08)
        store.append_brain_event(
            event_type=BrainEventType.SCENE_CHANGED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="runtime",
            payload={
                "presence_scope_key": "browser:presence",
                "camera_connected": False,
                "camera_fresh": False,
                "person_present": "absent",
                "attention_to_camera": "unknown",
                "engagement_state": "away",
                "scene_change": "changed",
                "summary": "camera disconnected",
                "confidence": 1.0,
                "observed_at": "2026-04-17T10:00:05+00:00",
            },
        )

    await asyncio.gather(
        run_test(
            processor,
            frames_to_send=[
                SleepFrame(sleep=0.06),
                UserStartedSpeakingFrame(),
                UserStoppedSpeakingFrame(),
                BotStartedSpeakingFrame(),
                BotStoppedSpeakingFrame(),
                SleepFrame(sleep=0.12),
            ],
            frames_to_send_direction=FrameDirection.DOWNSTREAM,
        ),
        emit_presence_events(),
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase4_presence_attention_cycle",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        expected_terminal_state={
            "body": {"policy_phase": "neutral", "vision_connected": False},
            "scene": {"camera_connected": False, "person_present": "absent"},
            "engagement": {"user_present": False},
        },
    )
    result = harness.replay(scenario, presence_scope_key="browser:presence")

    assert result.context_surface.scene.camera_connected is False
    assert result.context_surface.engagement.user_present is False
    artifact_payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert artifact_payload["scenario"]["name"] == "phase4_presence_attention_cycle"
    assert artifact_payload["qa"]["matched"] is True
    capability_events = [
        event
        for event in artifact_payload["events"]
        if event["event_type"] == BrainEventType.CAPABILITY_COMPLETED
    ]
    assert capability_events
    await controller.close()


def test_brain_replay_regression_fixtures_assert_continuity_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    cases = load_replay_regression_cases(Path("tests/fixtures/brain_evals"))

    assert cases
    for case in cases:
        result = evaluate_replay_regression_case(
            case=case,
            store=store,
            output_dir=tmp_path / "replays",
        )
        artifact_payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

        assert result.matched is True
        assert artifact_payload["continuity_eval"]["matched"] is True
        assert "autonomy_digest" in artifact_payload["continuity_state"]
        assert "reevaluation_digest" in artifact_payload["continuity_state"]
        assert "planning_digest" in artifact_payload
        assert "planning_digest" in artifact_payload["continuity_state"]
        assert "continuity_graph" in artifact_payload
        assert "continuity_graph_digest" in artifact_payload
        assert "continuity_graph" in artifact_payload["continuity_state"]
        assert "continuity_graph_digest" in artifact_payload["continuity_state"]
        assert "continuity_dossiers" in artifact_payload
        assert "continuity_dossiers" in artifact_payload["continuity_state"]
        assert "procedural_traces" in artifact_payload
        assert "procedural_traces" in artifact_payload["continuity_state"]
        assert "procedural_skills" in artifact_payload
        assert "procedural_skills" in artifact_payload["continuity_state"]
        assert "procedural_skill_digest" in artifact_payload
        assert "procedural_skill_digest" in artifact_payload["continuity_state"]
        assert "procedural_skill_governance_report" in artifact_payload
        assert "procedural_skill_governance_report" in artifact_payload["continuity_state"]
        assert "procedural_qa_report" in artifact_payload
        assert "procedural_qa_report" in artifact_payload["continuity_state"]
        assert "continuity_governance_report" in artifact_payload
        assert "continuity_governance_report" in artifact_payload["continuity_state"]
        assert "wake_digest" in artifact_payload
        assert "wake_digest" in artifact_payload["continuity_state"]
        assert "context_packet_digest" in artifact_payload
        assert "context_packet_digest" in artifact_payload["continuity_state"]
        for task in (
            "reply",
            "planning",
            "recall",
            "reflection",
            "critique",
            "wake",
            "reevaluation",
            "operator_audit",
            "governance_review",
        ):
            assert artifact_payload["continuity_state"]["selection_traces"][task]["task"] == task
            assert artifact_payload["packet_traces"][task]["task"] == task
            assert artifact_payload["continuity_state"]["packet_traces"][task]["task"] == task
        assert "query_text" in artifact_payload["context_packet_digest"]["reply"]
        assert "query_text" in artifact_payload["context_packet_digest"]["reflection"]
        assert "query_text" in artifact_payload["context_packet_digest"]["operator_audit"]
        assert "query_text" in artifact_payload["context_packet_digest"]["governance_review"]
        assert "reflection_cycles" in artifact_payload["continuity_state"]


def test_brain_replay_autonomy_digest_is_deterministic_for_non_action_fixture(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(
        Path("tests/fixtures/brain_evals/autonomy_non_action_user_turn.json")
    )
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / "first.db",
        artifact_path=tmp_path / "first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / "second.db",
        artifact_path=tmp_path / "second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["autonomy_digest"] == second_payload["autonomy_digest"]
    assert (
        first_payload["continuity_state"]["autonomy_digest"]
        == second_payload["continuity_state"]["autonomy_digest"]
    )
    assert "reason_code_counts" in first_payload["autonomy_digest"]
    assert first_payload["autonomy_digest"]["recent_non_actions"]
    assert "reason_codes" in first_payload["autonomy_digest"]["recent_non_actions"][-1]
    assert "executive_policy" in first_payload["autonomy_digest"]["recent_non_actions"][-1]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "autonomy_reevaluation_hold_accept.json",
        "autonomy_reevaluation_hold_expire.json",
    ],
)
def test_brain_replay_reevaluation_digest_is_deterministic_for_reevaluation_fixtures(
    tmp_path,
    fixture_name,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["reevaluation_digest"] == second_payload["reevaluation_digest"]
    assert (
        first_payload["continuity_state"]["reevaluation_digest"]
        == second_payload["continuity_state"]["reevaluation_digest"]
    )
    assert "reason_code_counts" in first_payload["reevaluation_digest"]
    assert first_payload["reevaluation_digest"]["recent_transitions"]
    assert "outcome_reason_codes" in first_payload["reevaluation_digest"]["recent_transitions"][-1]
    if first_payload["reevaluation_digest"]["current_holds"]:
        current_hold = first_payload["reevaluation_digest"]["current_holds"][-1]
        assert "hold_reason_codes" in current_hold
        assert "hold_policy_action_posture" in current_hold
        assert "hold_policy_approval_requirement" in current_hold


def test_brain_replay_roundtrips_policy_reason_codes_and_executive_policy(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="replay-policy")
    surface_builder = _ReplayPolicySurfaceBuilder(
        store=store,
        session_ids=session_ids,
        review_debt_count=1,
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
        context_surface_builder=surface_builder,
    )
    store.append_candidate_goal_created(
        candidate_goal=_policy_candidate(
            "candidate-replay-policy",
            "Policy-backed replay candidate.",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )

    first_result = executive.run_presence_director_pass()
    assert first_result.terminal_decision == BrainAutonomyDecisionKind.NON_ACTION.value

    surface_builder.review_debt_count = 0
    second_result = executive.run_presence_director_reevaluation(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
            summary="Review debt cleared.",
            source_event_id="evt-replay-policy-clear",
            source_event_type="stateful.policy",
            ts=datetime.now(UTC).isoformat(),
            details={"source": "replay_policy_test"},
        )
    )
    assert second_result.terminal_decision == BrainAutonomyDecisionKind.ACCEPTED.value

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_candidate_goal_lifecycle",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=24,
    )

    first = harness.replay(
        scenario,
        output_store_path=tmp_path / "policy-roundtrip.first.db",
        artifact_path=tmp_path / "policy-roundtrip.first.json",
    )
    second = harness.replay(
        scenario,
        output_store_path=tmp_path / "policy-roundtrip.second.db",
        artifact_path=tmp_path / "policy-roundtrip.second.json",
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["autonomy_digest"] == second_payload["autonomy_digest"]
    assert first_payload["reevaluation_digest"] == second_payload["reevaluation_digest"]
    assert first_payload["executive_policy_audit"] == second_payload["executive_policy_audit"]
    assert (
        first_payload["continuity_state"]["autonomy_digest"]
        == second_payload["continuity_state"]["autonomy_digest"]
    )
    assert (
        first_payload["continuity_state"]["reevaluation_digest"]
        == second_payload["continuity_state"]["reevaluation_digest"]
    )
    assert (
        first_payload["continuity_state"]["executive_policy_audit"]
        == second_payload["continuity_state"]["executive_policy_audit"]
    )

    recent_non_action = first_payload["autonomy_digest"]["recent_non_actions"][-1]
    assert "policy_conservative_deferral" in recent_non_action["reason_codes"]
    assert "dossier_review_debt" in recent_non_action["reason_codes"]
    assert recent_non_action["executive_policy"]["action_posture"] == "defer"
    assert recent_non_action["executive_policy"]["approval_requirement"] == "user_confirmation"

    recent_action = first_payload["autonomy_digest"]["recent_actions"][-1]
    assert "accepted_for_goal_creation" in recent_action["reason_codes"]
    assert recent_action["executive_policy"]["action_posture"] == "allow"

    assert (
        first_payload["reevaluation_digest"]["reason_code_counts"]["policy_conservative_deferral"]
        >= 1
    )
    assert first_payload["reevaluation_digest"]["reason_code_counts"]["dossier_review_debt"] >= 1
    assert first_payload["executive_policy_audit"]["policy_posture_counts"]["defer"] >= 1
    assert (
        first_payload["executive_policy_audit"]["why_not_reason_code_counts"][
            "policy_conservative_deferral"
        ]
        >= 1
    )
    transition = first_payload["reevaluation_digest"]["recent_transitions"][-1]
    assert transition["outcome_decision_kind"] == BrainAutonomyDecisionKind.ACCEPTED.value
    assert "accepted_for_goal_creation" in transition["outcome_reason_codes"]


@pytest.mark.parametrize(
    ("fixture_name", "expected_route_kind", "expected_reason"),
    [
        (
            "wake_router_thread_idle_propose_candidate.json",
            BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value,
            "wake_matched",
        ),
        (
            "wake_router_condition_cleared_resume_direct.json",
            BrainCommitmentWakeRouteKind.RESUME_DIRECT.value,
            "blocker_cleared",
        ),
        (
            "wake_router_keep_waiting_candidate_already_current.json",
            BrainCommitmentWakeRouteKind.KEEP_WAITING.value,
            "candidate_already_current",
        ),
    ],
)
def test_brain_replay_wake_digest_is_deterministic_for_wake_router_fixtures(
    tmp_path,
    fixture_name,
    expected_route_kind,
    expected_reason,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["wake_digest"] == second_payload["wake_digest"]
    assert first_payload["executive_policy_audit"] == second_payload["executive_policy_audit"]
    assert (
        first_payload["continuity_state"]["wake_digest"]
        == second_payload["continuity_state"]["wake_digest"]
    )
    assert expected_reason in first_payload["wake_digest"]["reason_counts"]
    assert "policy_posture_counts" in first_payload["wake_digest"]
    assert "why_not_reason_code_counts" in first_payload["wake_digest"]
    assert expected_route_kind in [
        entry["route_kind"] for entry in first_payload["wake_digest"]["recent_triggers"]
    ]
    if expected_route_kind == BrainCommitmentWakeRouteKind.RESUME_DIRECT.value:
        assert first_payload["wake_digest"]["recent_direct_resumes"]
    elif expected_route_kind == BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value:
        assert first_payload["wake_digest"]["recent_candidate_proposals"]
    else:
        assert first_payload["wake_digest"]["recent_keep_waiting"]


@pytest.mark.parametrize(
    ("fixture_name", "expected_outcome_kind", "expected_reason", "expected_procedural_origin"),
    [
        (
            "planning_propose_adopt_initial.json",
            "adopted",
            "bounded_plan_available",
            None,
        ),
        (
            "planning_propose_reject_unknown_capability.json",
            "rejected",
            "unsupported_capability",
            None,
        ),
        (
            "planning_revise_after_block_adopt.json",
            "adopted",
            "bounded_plan_available",
            None,
        ),
        (
            "planning_skill_reuse_exact.json",
            "adopted",
            "bounded_plan_available",
            "skill_reuse",
        ),
        (
            "planning_skill_reject_mismatch.json",
            "rejected",
            "skill_reuse_mismatch",
            "skill_reuse",
        ),
        (
            "planning_skill_delta_revise_tail.json",
            "adopted",
            "bounded_plan_available",
            "skill_delta",
        ),
    ],
)
def test_brain_replay_planning_digest_is_deterministic_for_planning_fixtures(
    tmp_path,
    fixture_name,
    expected_outcome_kind,
    expected_reason,
    expected_procedural_origin,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["planning_digest"] == second_payload["planning_digest"]
    assert first_payload["executive_policy_audit"] == second_payload["executive_policy_audit"]
    assert (
        first_payload["continuity_state"]["planning_digest"]
        == second_payload["continuity_state"]["planning_digest"]
    )
    assert first_payload["packet_traces"] == second_payload["packet_traces"]
    assert (
        first_payload["continuity_state"]["packet_traces"]
        == second_payload["continuity_state"]["packet_traces"]
    )
    assert first_payload["continuity_graph"] == second_payload["continuity_graph"]
    assert (
        first_payload["continuity_state"]["continuity_graph"]
        == second_payload["continuity_state"]["continuity_graph"]
    )
    assert first_payload["procedural_traces"] == second_payload["procedural_traces"]
    assert (
        first_payload["continuity_state"]["procedural_traces"]
        == second_payload["continuity_state"]["procedural_traces"]
    )
    assert first_payload["packet_traces"]["planning"]["task"] == "planning"
    assert first_payload["packet_traces"]["planning"]["selected_items"]
    assert expected_reason in first_payload["planning_digest"]["reason_counts"]
    assert "policy_posture_counts" in first_payload["planning_digest"]
    assert "why_not_reason_code_counts" in first_payload["planning_digest"]
    actual_outcomes = (
        ["adopted" for _ in first_payload["planning_digest"]["recent_adoptions"]]
        + ["rejected" for _ in first_payload["planning_digest"]["recent_rejections"]]
        + [
            item["outcome_kind"]
            for item in first_payload["planning_digest"]["recent_revision_flows"]
            if item.get("outcome_kind")
        ]
    )
    assert expected_outcome_kind in actual_outcomes
    proposal_statuses = {
        record["backing_record_id"]: record["status"]
        for record in first_payload["continuity_graph"]["nodes"]
        if record["kind"] == "plan_proposal"
    }
    if expected_outcome_kind == "adopted":
        assert first_payload["planning_digest"]["recent_adoptions"]
        assert "adopted" in proposal_statuses.values()
        assert first_payload["procedural_traces"]["traces"]
        assert "planning_adopted" in first_payload["procedural_traces"]["outcome_counts"]
    else:
        assert first_payload["planning_digest"]["recent_rejections"]
        assert "rejected" in proposal_statuses.values()
        assert "planning_rejected" in first_payload["procedural_traces"]["outcome_counts"]
    if expected_procedural_origin is not None:
        assert (
            expected_procedural_origin
            in first_payload["planning_digest"]["procedural_origin_counts"]
        )
        assert (
            "procedural_skill"
            in first_payload["context_packet_digest"]["planning"]["selected_anchor_types"]
        )
        assert first_payload["context_packet_digest"]["planning"]["selected_backing_ids"]
        assert first_payload["planning_digest"]["recent_selected_skill_ids"]
        assert any(
            item.get("procedural_origin") == expected_procedural_origin
            for item in (
                first_payload["planning_digest"]["recent_proposals"]
                + first_payload["planning_digest"]["recent_adoptions"]
                + first_payload["planning_digest"]["recent_rejections"]
                + first_payload["planning_digest"]["recent_revision_flows"]
            )
        )


@pytest.mark.parametrize(
    ("fixture_name", "expected_relationship_freshness", "expected_project_key"),
    [
        ("dossier_relationship_correction.json", "fresh", None),
        ("dossier_project_arc_replacement.json", "fresh", "Alpha"),
        ("dossier_relationship_needs_refresh.json", "needs_refresh", None),
    ],
)
def test_brain_replay_continuity_dossiers_are_deterministic_for_dossier_fixtures(
    tmp_path,
    fixture_name,
    expected_relationship_freshness,
    expected_project_key,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["continuity_dossiers"] == second_payload["continuity_dossiers"]
    assert (
        first_payload["continuity_state"]["continuity_dossiers"]
        == second_payload["continuity_state"]["continuity_dossiers"]
    )
    relationship = next(
        item
        for item in first_payload["continuity_dossiers"]["dossiers"]
        if item["kind"] == "relationship"
    )
    assert relationship["freshness"] == expected_relationship_freshness
    assert (
        relationship["summary_evidence"]["claim_ids"]
        or relationship["summary_evidence"]["entry_ids"]
    )
    if expected_project_key is None:
        assert not [
            item
            for item in first_payload["continuity_dossiers"]["dossiers"]
            if item["kind"] == "project"
        ]
    else:
        assert expected_project_key in [
            item["project_key"]
            for item in first_payload["continuity_dossiers"]["dossiers"]
            if item["kind"] == "project"
        ]


def test_brain_replay_expanded_continuity_dossiers_are_deterministic(tmp_path):
    from tests.test_brain_memory_v2 import _build_expanded_dossier_state

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="replay-expanded")
    _build_expanded_dossier_state(store, session_ids)
    scenario = BrainReplayScenario(
        name="phase14a_expanded_dossiers",
        description="Expanded continuity dossiers remain replay-stable.",
        session_ids=session_ids,
        events=tuple(
            reversed(
                store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=256,
                )
            )
        ),
        expected_terminal_state={},
    )
    harness = BrainReplayHarness(store=store)
    first = harness.replay(
        scenario,
        output_store_path=tmp_path / "expanded.first.db",
        artifact_path=tmp_path / "expanded.first.json",
        presence_scope_key="browser:presence",
        language=Language.EN,
    )
    second = harness.replay(
        scenario,
        output_store_path=tmp_path / "expanded.second.db",
        artifact_path=tmp_path / "expanded.second.json",
        presence_scope_key="browser:presence",
        language=Language.EN,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["continuity_dossiers"] == second_payload["continuity_dossiers"]
    kinds = {
        item["kind"] for item in first_payload["continuity_dossiers"]["dossiers"]
    }
    assert {
        "self_policy",
        "user",
        "commitment",
        "plan",
        "procedural",
        "scene_world",
    } <= kinds
    for item in first_payload["continuity_dossiers"]["dossiers"]:
        if item["kind"] in {
            "self_policy",
            "user",
            "commitment",
            "plan",
            "procedural",
            "scene_world",
        }:
            assert item["dossier_id"].startswith("dossier_")
            assert item["summary_evidence"]["graph_node_ids"]


@pytest.mark.parametrize(
    (
        "fixture_name",
        "expected_active_count",
        "expected_superseded_count",
        "expected_retired_count",
    ),
    [
        ("procedural_skill_candidate_then_active.json", 1, 0, 0),
        ("procedural_skill_superseded_by_revised_tail.json", 1, 1, 0),
        ("procedural_skill_retired_after_repeated_failures.json", 0, 0, 1),
    ],
)
def test_brain_replay_procedural_skills_are_deterministic_for_skill_fixtures(
    tmp_path,
    fixture_name,
    expected_active_count,
    expected_superseded_count,
    expected_retired_count,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["procedural_skills"] == second_payload["procedural_skills"]
    assert (
        first_payload["continuity_state"]["procedural_skills"]
        == second_payload["continuity_state"]["procedural_skills"]
    )
    assert first_payload["procedural_skill_digest"] == second_payload["procedural_skill_digest"]
    assert (
        first_payload["continuity_state"]["procedural_skill_digest"]
        == second_payload["continuity_state"]["procedural_skill_digest"]
    )
    assert (
        first_payload["procedural_skill_governance_report"]
        == second_payload["procedural_skill_governance_report"]
    )
    assert (
        first_payload["continuity_state"]["procedural_skill_governance_report"]
        == second_payload["continuity_state"]["procedural_skill_governance_report"]
    )
    assert first_payload["procedural_qa_report"] == second_payload["procedural_qa_report"]
    assert (
        first_payload["continuity_state"]["procedural_qa_report"]
        == second_payload["continuity_state"]["procedural_qa_report"]
    )
    assert len(first_payload["procedural_skills"]["active_skill_ids"]) == expected_active_count
    assert (
        len(first_payload["procedural_skills"]["superseded_skill_ids"]) == expected_superseded_count
    )
    assert len(first_payload["procedural_skills"]["retired_skill_ids"]) == expected_retired_count


def test_brain_replay_procedural_qa_report_tracks_categories_and_negative_transfer(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(
        Path("tests/fixtures/brain_evals/planning_skill_reject_mismatch.json")
    )
    result = evaluate_replay_regression_case(
        case=case,
        store=store,
        output_dir=tmp_path / "replays",
    )
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert case.qa_categories == ("skill_reuse", "negative_transfer")
    assert payload["continuity_eval"]["qa_categories"] == ["skill_reuse", "negative_transfer"]
    assert (
        payload["procedural_skill_governance_report"]["negative_transfer_reason_counts"][
            "skill_reuse_mismatch"
        ]
        >= 1
    )
    assert (
        "skill_reuse_mismatch"
        in payload["procedural_skill_governance_report"]["recent_negative_transfer_flows"][0][
            "decision_reason"
        ]
    )
    assert (
        "skill_reuse"
        in payload["continuity_state"]["procedural_qa_report"]["procedural_origin_counts"]
    )


@pytest.mark.parametrize(
    (
        "fixture_name",
        "expected_temporal_mode",
        "expected_anchor_type",
        "expected_backing_id",
    ),
    [
        (
            "reply_current_truth_corrected_fact.json",
            "current_first",
            "claim",
            "claim_role_current",
        ),
        (
            "reply_historical_change_corrected_fact.json",
            "historical_focus",
            "claim",
            "claim_role_prior",
        ),
        (
            "reply_associative_project_recall.json",
            "current_first",
            "dossier",
            None,
        ),
    ],
)
def test_brain_replay_context_packet_and_governance_digests_are_deterministic_for_reply_fixtures(
    tmp_path,
    fixture_name,
    expected_temporal_mode,
    expected_anchor_type,
    expected_backing_id,
):
    store = BrainStore(path=tmp_path / "brain.db")
    case = load_replay_regression_case(Path(f"tests/fixtures/brain_evals/{fixture_name}"))
    harness = BrainReplayHarness(store=store)

    first = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.first.db",
        artifact_path=tmp_path / f"{fixture_name}.first.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
        context_queries=case.context_queries,
    )
    second = harness.replay(
        case.scenario,
        output_store_path=tmp_path / f"{fixture_name}.second.db",
        artifact_path=tmp_path / f"{fixture_name}.second.json",
        presence_scope_key=case.presence_scope_key,
        language=case.language,
        context_queries=case.context_queries,
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))

    assert first_payload["context_packet_digest"] == second_payload["context_packet_digest"]
    assert (
        first_payload["continuity_state"]["context_packet_digest"]
        == second_payload["continuity_state"]["context_packet_digest"]
    )
    assert first_payload["continuity_graph_digest"] == second_payload["continuity_graph_digest"]
    assert (
        first_payload["continuity_state"]["continuity_graph_digest"]
        == second_payload["continuity_state"]["continuity_graph_digest"]
    )
    assert (
        first_payload["continuity_governance_report"]
        == second_payload["continuity_governance_report"]
    )
    assert (
        first_payload["continuity_state"]["continuity_governance_report"]
        == second_payload["continuity_state"]["continuity_governance_report"]
    )
    assert first_payload["context_queries"] == case.context_queries
    assert (
        first_payload["context_packet_digest"]["reply"]["temporal_mode"] == expected_temporal_mode
    )
    assert "selected_availability_counts" in first_payload["context_packet_digest"]["reply"]
    assert "governance_reason_code_counts" in first_payload["context_packet_digest"]["reply"]
    assert "suppressed_backing_ids" in first_payload["context_packet_digest"]["reply"]
    assert first_payload["context_packet_digest"]["recall"]["query_text"]
    assert first_payload["context_packet_digest"]["reflection"]["query_text"]
    assert first_payload["context_packet_digest"]["critique"]["query_text"]
    assert "claim_currentness_counts" in first_payload["continuity_governance_report"]
    assert "dossier_availability_counts_by_task" in first_payload["continuity_governance_report"]
    assert "suppressed_packet_rows" in first_payload["continuity_governance_report"]
    assert (
        expected_anchor_type
        in first_payload["context_packet_digest"]["reply"]["selected_anchor_types"]
    )
    if expected_backing_id is not None:
        assert (
            expected_backing_id
            in first_payload["context_packet_digest"]["reply"]["selected_backing_ids"]
        )


def test_brain_replay_current_and_historical_reply_fixtures_diverge_in_packet_summary(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    harness = BrainReplayHarness(store=store)
    current_case = load_replay_regression_case(
        Path("tests/fixtures/brain_evals/reply_current_truth_corrected_fact.json")
    )
    historical_case = load_replay_regression_case(
        Path("tests/fixtures/brain_evals/reply_historical_change_corrected_fact.json")
    )

    current_result = harness.replay(
        current_case.scenario,
        output_store_path=tmp_path / "current.db",
        artifact_path=tmp_path / "current.json",
        presence_scope_key=current_case.presence_scope_key,
        language=current_case.language,
        context_queries=current_case.context_queries,
    )
    historical_result = harness.replay(
        historical_case.scenario,
        output_store_path=tmp_path / "historical.db",
        artifact_path=tmp_path / "historical.json",
        presence_scope_key=historical_case.presence_scope_key,
        language=historical_case.language,
        context_queries=historical_case.context_queries,
    )

    current_payload = json.loads(current_result.artifact_path.read_text(encoding="utf-8"))
    historical_payload = json.loads(historical_result.artifact_path.read_text(encoding="utf-8"))

    assert current_payload["context_packet_digest"]["reply"]["temporal_mode"] == "current_first"
    assert (
        historical_payload["context_packet_digest"]["reply"]["temporal_mode"] == "historical_focus"
    )
    assert (
        "claim_role_current"
        in current_payload["context_packet_digest"]["reply"]["selected_backing_ids"]
    )
    assert (
        "claim_role_prior"
        in historical_payload["context_packet_digest"]["reply"]["selected_backing_ids"]
    )


def test_brain_replay_harness_rebuilds_expiry_cleanup_without_duplicate_proposal(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    candidate = BrainCandidateGoal(
        candidate_goal_id="candidate-replay-expiry-cleanup",
        candidate_type="presence_acknowledgement",
        source=BrainCandidateGoalSource.RUNTIME.value,
        summary="Expire through explicit cleanup.",
        goal_family="environment",
        urgency=0.8,
        confidence=0.9,
        initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        cooldown_key="cooldown:replay-expiry",
        dedupe_key="dedupe:replay-expiry",
        requires_user_turn_gap=True,
        expires_at="2099-01-01T00:00:05+00:00",
        created_at="2099-01-01T00:00:00+00:00",
    )
    store.append_candidate_goal_created(
        candidate_goal=candidate,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    executive.run_presence_director_pass()
    executive.run_presence_director_expiry_cleanup(
        BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.TIME_REACHED.value,
            summary="Cleanup expired candidates.",
            ts="2099-01-01T00:00:06+00:00",
        )
    )

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_candidate_goal_lifecycle",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=24,
    )
    result = harness.replay(scenario)
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert (
        len(
            [
                event
                for event in payload["events"]
                if event["event_type"] == BrainEventType.GOAL_CANDIDATE_CREATED
            ]
        )
        == 1
    )
    trigger_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED
    )
    expired_index = next(
        index
        for index, event in enumerate(payload["events"])
        if event["event_type"] == BrainEventType.GOAL_CANDIDATE_EXPIRED
    )
    assert trigger_index < expired_index
    assert not any(
        event["event_type"] == BrainEventType.GOAL_CANDIDATE_ACCEPTED for event in payload["events"]
    )
    assert result.autonomy_ledger.current_candidates == []
    assert payload["autonomy_digest"]["decision_counts"]["expired"] >= 1
    assert payload["autonomy_digest"]["next_expiry_at"] is None


def test_brain_replay_autonomy_digest_is_deterministic_for_family_rotation(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    store.append_brain_event(
        event_type=BrainEventType.USER_TURN_STARTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={},
    )
    for candidate in (
        BrainCandidateGoal(
            candidate_goal_id="candidate-fairness-environment-leader",
            candidate_type="presence_acknowledgement",
            source=BrainCandidateGoalSource.RUNTIME.value,
            summary="Environment family leader.",
            goal_family="environment",
            urgency=0.8,
            confidence=0.9,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
            cooldown_key="cooldown:fairness:env1",
            dedupe_key="dedupe:fairness:env1",
            requires_user_turn_gap=True,
            expires_at="2099-01-01T00:05:00+00:00",
            created_at="2099-01-01T00:00:00+00:00",
        ),
        BrainCandidateGoal(
            candidate_goal_id="candidate-fairness-environment-backlog",
            candidate_type="presence_acknowledgement",
            source=BrainCandidateGoalSource.RUNTIME.value,
            summary="Environment family backlog.",
            goal_family="environment",
            urgency=0.8,
            confidence=0.9,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
            cooldown_key="cooldown:fairness:env2",
            dedupe_key="dedupe:fairness:env2",
            requires_user_turn_gap=True,
            expires_at="2099-01-01T00:05:00+00:00",
            created_at="2099-01-01T00:00:30+00:00",
        ),
        BrainCandidateGoal(
            candidate_goal_id="candidate-fairness-maintenance",
            candidate_type="maintenance_review_findings",
            source=BrainCandidateGoalSource.TIMER.value,
            summary="Maintenance family candidate.",
            goal_family="memory_maintenance",
            urgency=0.7,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
            cooldown_key="cooldown:fairness:maint",
            dedupe_key="dedupe:fairness:maint",
            requires_user_turn_gap=True,
            expires_at="2099-01-01T00:05:00+00:00",
            created_at="2099-01-01T00:01:00+00:00",
        ),
    ):
        store.append_candidate_goal_created(
            candidate_goal=candidate,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
        )
    executive.run_presence_director_pass()
    executive.run_presence_director_pass()
    executive.run_presence_director_pass()

    harness = BrainReplayHarness(store=store)
    scenario = harness.capture_builtin_scenario(
        name="phase1_candidate_goal_lifecycle",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        limit=32,
    )
    first = harness.replay(
        scenario,
        output_store_path=tmp_path / "fairness-first.db",
        artifact_path=tmp_path / "fairness-first.json",
    )
    second = harness.replay(
        scenario,
        output_store_path=tmp_path / "fairness-second.db",
        artifact_path=tmp_path / "fairness-second.json",
    )

    first_payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.artifact_path.read_text(encoding="utf-8"))
    first_non_actions = first_payload["autonomy_digest"]["recent_non_actions"][-3:]

    assert first_payload["autonomy_digest"] == second_payload["autonomy_digest"]
    assert [item["goal_family"] for item in first_non_actions] == [
        "environment",
        "memory_maintenance",
        "environment",
    ]
    assert first_payload["autonomy_digest"]["pending_family_counts"] == {
        "environment": 2,
        "memory_maintenance": 1,
    }
    assert {
        item["goal_family"]: item["leader_candidate_goal_id"]
        for item in first_payload["autonomy_digest"]["current_family_leaders"]
    } == {
        "environment": "candidate-fairness-environment-leader",
        "memory_maintenance": "candidate-fairness-maintenance",
    }

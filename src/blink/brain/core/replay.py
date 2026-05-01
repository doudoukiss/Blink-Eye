"""Provider-free replay and evaluation helpers for Blink brain events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from blink.brain.core.autonomy import BrainAutonomyLedgerProjection
from blink.brain.core.events import BrainEventRecord, BrainEventType
from blink.brain.core.presence import BrainPresenceSnapshot
from blink.brain.core.projections import (
    BrainAgendaProjection,
    BrainEngagementStateProjection,
    BrainHeartbeatProjection,
    BrainRelationshipStateProjection,
    BrainSceneStateProjection,
    BrainWorkingContextProjection,
)
from blink.brain.core.session import BrainSessionIds
from blink.brain.core.store import BrainCoreStore

_BUILTIN_SCENARIO_EVENT_TYPES = {
    "phase1_turn_tool_robot_action": {
        BrainEventType.BODY_STATE_UPDATED,
        BrainEventType.USER_TURN_STARTED,
        BrainEventType.USER_TURN_TRANSCRIBED,
        BrainEventType.USER_TURN_ENDED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.ASSISTANT_TURN_STARTED,
        BrainEventType.TOOL_CALLED,
        BrainEventType.TOOL_COMPLETED,
        BrainEventType.MEMORY_BLOCK_UPSERTED,
        BrainEventType.MEMORY_CLAIM_RECORDED,
        BrainEventType.MEMORY_CLAIM_SUPERSEDED,
        BrainEventType.MEMORY_CLAIM_REVOKED,
        BrainEventType.REFLECTION_CYCLE_STARTED,
        BrainEventType.REFLECTION_CYCLE_COMPLETED,
        BrainEventType.REFLECTION_CYCLE_SKIPPED,
        BrainEventType.REFLECTION_CYCLE_FAILED,
        BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED,
        BrainEventType.MEMORY_HEALTH_REPORTED,
        BrainEventType.ASSISTANT_TURN_ENDED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
    },
    "phase1_candidate_goal_lifecycle": {
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_SUPPRESSED,
        BrainEventType.GOAL_CANDIDATE_MERGED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.GOAL_CANDIDATE_EXPIRED,
        BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
    },
    "phase6_presence_director_acceptance": {
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
    },
    "phase6_scene_candidate_action": {
        BrainEventType.PERCEPTION_OBSERVED,
        BrainEventType.ENGAGEMENT_CHANGED,
        BrainEventType.ATTENTION_CHANGED,
        BrainEventType.SCENE_CHANGED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
        BrainEventType.GOAL_CREATED,
    },
    "phase6_scene_capability_route": {
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_COMPLETED,
        BrainEventType.GOAL_FAILED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.CRITIC_FEEDBACK,
    },
    "private_working_memory_replay": {
        BrainEventType.BODY_STATE_UPDATED,
        BrainEventType.PERCEPTION_OBSERVED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
    },
    "phase6_commitment_wake_candidate": {
        BrainEventType.COMMITMENT_WAKE_TRIGGERED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_DEFERRED,
        BrainEventType.GOAL_COMPLETED,
        BrainEventType.GOAL_FAILED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.CRITIC_FEEDBACK,
    },
    "phase8_commitment_condition_cleared_resume": {
        BrainEventType.COMMITMENT_WAKE_TRIGGERED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_RESUMED,
        BrainEventType.GOAL_COMPLETED,
        BrainEventType.GOAL_FAILED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.CRITIC_FEEDBACK,
    },
    "phase6_timer_maintenance_candidate": {
        BrainEventType.REFLECTION_CYCLE_STARTED,
        BrainEventType.REFLECTION_CYCLE_COMPLETED,
        BrainEventType.REFLECTION_CYCLE_SKIPPED,
        BrainEventType.MEMORY_HEALTH_REPORTED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_COMPLETED,
        BrainEventType.GOAL_FAILED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.CRITIC_FEEDBACK,
    },
    "phase7_turn_close_candidate_reevaluation": {
        BrainEventType.USER_TURN_STARTED,
        BrainEventType.USER_TURN_ENDED,
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.GOAL_CANDIDATE_EXPIRED,
        BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
    },
    "phase7_startup_recovery_candidate_reevaluation": {
        BrainEventType.GOAL_CREATED,
        BrainEventType.GOAL_CANDIDATE_CREATED,
        BrainEventType.GOAL_CANDIDATE_ACCEPTED,
        BrainEventType.GOAL_CANDIDATE_EXPIRED,
        BrainEventType.DIRECTOR_REEVALUATION_TRIGGERED,
        BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
    },
    "phase9_planning_review_bound_state": {
        BrainEventType.GOAL_CREATED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_REPAIRED,
    },
    "phase3_goal_retry_recovery": {
        BrainEventType.GOAL_CREATED,
        BrainEventType.PLANNING_REQUESTED,
        BrainEventType.GOAL_UPDATED,
        BrainEventType.GOAL_DEFERRED,
        BrainEventType.GOAL_RESUMED,
        BrainEventType.GOAL_CANCELLED,
        BrainEventType.GOAL_REPAIRED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.CRITIC_FEEDBACK,
        BrainEventType.PLANNING_PROPOSED,
        BrainEventType.PLANNING_ADOPTED,
        BrainEventType.PLANNING_REJECTED,
        BrainEventType.MEMORY_BLOCK_UPSERTED,
        BrainEventType.MEMORY_CLAIM_RECORDED,
        BrainEventType.MEMORY_CLAIM_SUPERSEDED,
        BrainEventType.MEMORY_CLAIM_REVOKED,
        BrainEventType.REFLECTION_CYCLE_STARTED,
        BrainEventType.REFLECTION_CYCLE_COMPLETED,
        BrainEventType.REFLECTION_CYCLE_SKIPPED,
        BrainEventType.REFLECTION_CYCLE_FAILED,
        BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED,
        BrainEventType.MEMORY_HEALTH_REPORTED,
        BrainEventType.GOAL_COMPLETED,
        BrainEventType.GOAL_FAILED,
    },
    "phase4_presence_attention_cycle": {
        BrainEventType.BODY_STATE_UPDATED,
        BrainEventType.PERCEPTION_OBSERVED,
        BrainEventType.ENGAGEMENT_CHANGED,
        BrainEventType.ATTENTION_CHANGED,
        BrainEventType.SCENE_CHANGED,
        BrainEventType.USER_TURN_STARTED,
        BrainEventType.USER_TURN_ENDED,
        BrainEventType.ASSISTANT_TURN_STARTED,
        BrainEventType.ASSISTANT_TURN_ENDED,
        BrainEventType.CAPABILITY_REQUESTED,
        BrainEventType.CAPABILITY_COMPLETED,
        BrainEventType.CAPABILITY_FAILED,
        BrainEventType.ROBOT_ACTION_OUTCOME,
    },
}


@dataclass(frozen=True)
class BrainCoreReplayScenario:
    """One named replayable event-log scenario."""

    name: str
    description: str
    session_ids: BrainSessionIds
    events: tuple[BrainEventRecord, ...]
    expected_terminal_state: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BrainCoreReplayResult:
    """Replay result plus rebuilt projection surfaces and artifact paths."""

    scenario: BrainCoreReplayScenario
    replay_store_path: Path
    artifact_path: Path
    body: BrainPresenceSnapshot
    scene: BrainSceneStateProjection
    engagement: BrainEngagementStateProjection
    relationship_state: BrainRelationshipStateProjection
    working_context: BrainWorkingContextProjection
    agenda: BrainAgendaProjection
    autonomy_ledger: BrainAutonomyLedgerProjection
    heartbeat: BrainHeartbeatProjection


class BrainCoreReplayHarness:
    """Capture and replay inspectable brain-event scenarios."""

    def __init__(self, *, store: BrainCoreStore):
        """Initialize the harness against a source store."""
        self._store = store

    def capture_builtin_scenario(
        self,
        *,
        name: str,
        user_id: str,
        thread_id: str,
        limit: int = 64,
        expected_terminal_state: dict[str, object] | None = None,
    ) -> BrainCoreReplayScenario:
        """Capture one named built-in replay scenario from the append-only event log."""
        allowed_event_types = _BUILTIN_SCENARIO_EVENT_TYPES.get(name)
        if allowed_event_types is None:
            raise KeyError(f"Unsupported built-in replay scenario '{name}'.")

        events = [
            event
            for event in reversed(
                self._store.recent_brain_events(
                    user_id=user_id,
                    thread_id=thread_id,
                    limit=limit,
                )
            )
            if event.event_type in allowed_event_types
        ]
        if not events:
            raise ValueError(f"No events found for replay scenario '{name}'.")

        first = events[0]
        return BrainCoreReplayScenario(
            name=name,
            description=name.replace("_", " "),
            session_ids=BrainSessionIds(
                agent_id=first.agent_id,
                user_id=first.user_id,
                session_id=first.session_id,
                thread_id=first.thread_id,
            ),
            events=tuple(events),
            expected_terminal_state=dict(expected_terminal_state or {}),
        )

    def replay(
        self,
        scenario: BrainCoreReplayScenario,
        *,
        output_store_path: Path | None = None,
        artifact_path: Path | None = None,
        presence_scope_key: str | None = None,
    ) -> BrainCoreReplayResult:
        """Replay one scenario into a fresh core store and write an inspectable JSON artifact."""
        if output_store_path is not None:
            replay_store_path = output_store_path
        else:
            replay_dir = self._store.path.parent / "exports" / "replay_stores"
            replay_dir.mkdir(parents=True, exist_ok=True)
            replay_store_path = replay_dir / f"{scenario.name}_{uuid4().hex}_replay.db"
        if replay_store_path.exists():
            replay_store_path.unlink()
        replay_store = BrainCoreStore(path=replay_store_path)
        for event in scenario.events:
            replay_store.import_brain_event(event)

        resolved_presence_scope_key = presence_scope_key or next(
            (
                str(event.payload.get("scope_key"))
                for event in scenario.events
                if event.event_type == BrainEventType.BODY_STATE_UPDATED
                and str(event.payload.get("scope_key", "")).strip()
            ),
            "local:presence",
        )
        body = replay_store.get_body_state_projection(scope_key=resolved_presence_scope_key)
        scene = replay_store.get_scene_state_projection(scope_key=resolved_presence_scope_key)
        engagement = replay_store.get_engagement_state_projection(scope_key=resolved_presence_scope_key)
        relationship_state = replay_store.get_relationship_state_projection(
            scope_key=resolved_presence_scope_key
        )
        working_context = replay_store.get_working_context_projection(scope_key=scenario.session_ids.thread_id)
        agenda = replay_store.get_agenda_projection(scope_key=scenario.session_ids.thread_id)
        autonomy_ledger = replay_store.get_autonomy_ledger_projection(scope_key=scenario.session_ids.thread_id)
        heartbeat = replay_store.get_heartbeat_projection(scope_key=scenario.session_ids.thread_id)

        terminal_state = _terminal_state_payload(
            body=body,
            scene=scene,
            engagement=engagement,
            relationship_state=relationship_state,
        )
        payload = {
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "event_count": len(scenario.events),
            },
            "session": {
                "agent_id": scenario.session_ids.agent_id,
                "user_id": scenario.session_ids.user_id,
                "session_id": scenario.session_ids.session_id,
                "thread_id": scenario.session_ids.thread_id,
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "ts": event.ts,
                    "source": event.source,
                    "correlation_id": event.correlation_id,
                    "causal_parent_id": event.causal_parent_id,
                    "payload": event.payload,
                    "tags": event.tags,
                }
                for event in scenario.events
            ],
            "projections": {
                "body": body.as_dict(),
                "scene": scene.as_dict(),
                "engagement": engagement.as_dict(),
                "relationship_state": relationship_state.as_dict(),
                "working_context": working_context.as_dict(),
                "agenda": agenda.as_dict(),
                "autonomy_ledger": autonomy_ledger.as_dict(),
                "heartbeat": heartbeat.as_dict(),
            },
            "terminal_state": terminal_state,
            "qa": _evaluate_terminal_state(
                terminal_state=terminal_state,
                expected_terminal_state=scenario.expected_terminal_state,
            ),
        }

        resolved_artifact_path = artifact_path or (
            replay_store.path.parent / "exports" / f"{scenario.name}_replay.json"
        )
        resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_artifact_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        replay_store.close()
        return BrainCoreReplayResult(
            scenario=scenario,
            replay_store_path=replay_store_path,
            artifact_path=resolved_artifact_path,
            body=body,
            scene=scene,
            engagement=engagement,
            relationship_state=relationship_state,
            working_context=working_context,
            agenda=agenda,
            autonomy_ledger=autonomy_ledger,
            heartbeat=heartbeat,
        )


def _terminal_state_payload(
    *,
    body: BrainPresenceSnapshot,
    scene: BrainSceneStateProjection,
    engagement: BrainEngagementStateProjection,
    relationship_state: BrainRelationshipStateProjection,
) -> dict[str, object]:
    """Return a compact terminal-state payload for replay QA."""
    return {
        "body": {
            "policy_phase": body.policy_phase,
            "last_action": body.robot_head_last_action,
            "sensor_health": body.sensor_health,
            "vision_connected": body.vision_connected,
        },
        "scene": {
            "camera_connected": scene.camera_connected,
            "person_present": scene.person_present,
            "scene_change_state": scene.scene_change_state,
        },
        "engagement": {
            "engagement_state": engagement.engagement_state,
            "attention_to_camera": engagement.attention_to_camera,
            "user_present": engagement.user_present,
        },
        "relationship_state": {
            "user_present": relationship_state.user_present,
            "last_seen_at": relationship_state.last_seen_at,
            "open_commitments": list(relationship_state.open_commitments),
        },
    }


def _evaluate_terminal_state(
    *,
    terminal_state: dict[str, object],
    expected_terminal_state: dict[str, object],
) -> dict[str, object]:
    """Compare the replay terminal state against the expected QA contract."""
    if not expected_terminal_state:
        return {
            "expected_terminal_state": {},
            "matched": True,
            "mismatches": [],
        }

    mismatches: list[dict[str, object]] = []
    for section, expected_value in expected_terminal_state.items():
        actual_value = terminal_state.get(section)
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            for key, nested_expected in expected_value.items():
                nested_actual = actual_value.get(key)
                if nested_actual != nested_expected:
                    mismatches.append(
                        {
                            "path": f"{section}.{key}",
                            "expected": nested_expected,
                            "actual": nested_actual,
                        }
                    )
        elif actual_value != expected_value:
            mismatches.append(
                {
                    "path": section,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return {
        "expected_terminal_state": expected_terminal_state,
        "matched": not mismatches,
        "mismatches": mismatches,
    }

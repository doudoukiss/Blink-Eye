"""Typed brain event definitions for Blink's state kernel."""

from __future__ import annotations

import json
from dataclasses import dataclass

from blink.brain.bounded_json import MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS, safe_json_dict


class BrainEventType:
    """String constants for the typed brain event surface."""

    BODY_STATE_UPDATED = "body.state.updated"
    PERCEPTION_OBSERVED = "perception.observed"
    ENGAGEMENT_CHANGED = "engagement.changed"
    ATTENTION_CHANGED = "attention.changed"
    SCENE_CHANGED = "scene.changed"
    CAMERA_TRACK_STALLED = "camera.track.stalled"
    CAMERA_TRACK_RESUMED = "camera.track.resumed"
    CAMERA_RECOVERY_ATTEMPTED = "camera.recovery.attempted"
    CAMERA_RECOVERY_EXHAUSTED = "camera.recovery.exhausted"
    USER_TURN_STARTED = "user.turn.started"
    USER_TURN_TRANSCRIBED = "user.turn.transcribed"
    USER_TURN_ENDED = "user.turn.ended"
    ASSISTANT_TURN_STARTED = "assistant.turn.started"
    ASSISTANT_TURN_ENDED = "assistant.turn.ended"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    MEMORY_BLOCK_UPSERTED = "memory.block.upserted"
    MEMORY_CLAIM_RECORDED = "memory.claim.recorded"
    MEMORY_CLAIM_SUPERSEDED = "memory.claim.superseded"
    MEMORY_CLAIM_REVOKED = "memory.claim.revoked"
    MEMORY_CLAIM_REVALIDATED = "memory.claim.revalidated"
    MEMORY_CLAIM_EXPIRED = "memory.claim.expired"
    MEMORY_CLAIM_REVIEW_REQUESTED = "memory.claim.review.requested"
    MEMORY_CLAIM_RETENTION_RECLASSIFIED = "memory.claim.retention.reclassified"
    MEMORY_USE_TRACED = "memory.use.traced"
    MEMORY_CONTINUITY_TRACED = "memory.continuity.traced"
    MEMORY_DISCOURSE_EPISODE_DERIVED = "memory.discourse_episode.derived"
    PERFORMANCE_PREFERENCE_RECORDED = "performance.preference.recorded"
    PERFORMANCE_LEARNING_POLICY_PROPOSED = "performance.learning.policy.proposed"
    PERFORMANCE_LEARNING_POLICY_APPLIED = "performance.learning.policy.applied"
    REFLECTION_CYCLE_STARTED = "reflection.cycle.started"
    REFLECTION_CYCLE_COMPLETED = "reflection.cycle.completed"
    REFLECTION_CYCLE_SKIPPED = "reflection.cycle.skipped"
    REFLECTION_CYCLE_FAILED = "reflection.cycle.failed"
    AUTOBIOGRAPHY_ENTRY_UPSERTED = "autobiography.entry.upserted"
    AUTOBIOGRAPHY_ENTRY_REVIEW_REQUESTED = "autobiography.entry.review.requested"
    AUTOBIOGRAPHY_ENTRY_RETENTION_RECLASSIFIED = "autobiography.entry.retention.reclassified"
    AUTOBIOGRAPHY_ENTRY_REDACTED = "autobiography.entry.redacted"
    MEMORY_HEALTH_REPORTED = "memory.health.reported"
    GOAL_CREATED = "goal.created"
    GOAL_CANDIDATE_CREATED = "goal.candidate.created"
    GOAL_CANDIDATE_SUPPRESSED = "goal.candidate.suppressed"
    GOAL_CANDIDATE_MERGED = "goal.candidate.merged"
    GOAL_CANDIDATE_ACCEPTED = "goal.candidate.accepted"
    GOAL_CANDIDATE_EXPIRED = "goal.candidate.expired"
    COMMITMENT_WAKE_TRIGGERED = "commitment.wake.triggered"
    DIRECTOR_REEVALUATION_TRIGGERED = "director.reevaluation.triggered"
    GOAL_UPDATED = "goal.updated"
    GOAL_DEFERRED = "goal.deferred"
    GOAL_RESUMED = "goal.resumed"
    GOAL_CANCELLED = "goal.cancelled"
    GOAL_REPAIRED = "goal.repaired"
    GOAL_COMPLETED = "goal.completed"
    GOAL_FAILED = "goal.failed"
    PLANNING_REQUESTED = "planning.requested"
    PLANNING_PROPOSED = "planning.proposed"
    PLANNING_ADOPTED = "planning.adopted"
    PLANNING_REJECTED = "planning.rejected"
    CAPABILITY_REQUESTED = "capability.requested"
    CAPABILITY_COMPLETED = "capability.completed"
    CAPABILITY_FAILED = "capability.failed"
    CRITIC_FEEDBACK = "critic.feedback"
    DIRECTOR_NON_ACTION_RECORDED = "director.non_action.recorded"
    ROBOT_ACTION_OUTCOME = "robot.action.outcome"
    ENTITY_PREDICTION_GENERATED = "entity.prediction.generated"
    AFFORDANCE_PREDICTION_GENERATED = "affordance.prediction.generated"
    ENGAGEMENT_PREDICTION_GENERATED = "engagement.prediction.generated"
    SCENE_PREDICTION_GENERATED = "scene.prediction.generated"
    ACTION_OUTCOME_PREDICTED = "action.outcome.predicted"
    WAKE_PREDICTION_GENERATED = "wake.prediction.generated"
    PREDICTION_CONFIRMED = "prediction.confirmed"
    PREDICTION_INVALIDATED = "prediction.invalidated"
    PREDICTION_EXPIRED = "prediction.expired"
    ACTION_REHEARSAL_REQUESTED = "action.rehearsal.requested"
    ACTION_REHEARSAL_COMPLETED = "action.rehearsal.completed"
    ACTION_REHEARSAL_SKIPPED = "action.rehearsal.skipped"
    ACTION_OUTCOME_COMPARED = "action.outcome.compared"
    EMBODIED_INTENT_SELECTED = "embodied.intent.selected"
    EMBODIED_DISPATCH_PREPARED = "embodied.dispatch.prepared"
    EMBODIED_DISPATCH_COMPLETED = "embodied.dispatch.completed"
    EMBODIED_DISPATCH_DEFERRED = "embodied.dispatch.deferred"
    EMBODIED_RECOVERY_RECORDED = "embodied.recovery.recorded"
    PRACTICE_PLAN_CREATED = "practice.plan.created"
    SKILL_EVIDENCE_UPDATED = "skill.evidence.updated"
    SKILL_PROMOTION_PROPOSED = "skill.promotion.proposed"
    SKILL_PROMOTION_BLOCKED = "skill.promotion.blocked"
    SKILL_DEMOTION_PROPOSED = "skill.demotion.proposed"
    ADAPTER_CARD_UPSERTED = "adapter.card.upserted"
    ADAPTER_BENCHMARK_REPORTED = "adapter.benchmark.reported"
    ADAPTER_PROMOTION_DECIDED = "adapter.promotion.decided"
    ADAPTER_ROLLBACK_RECORDED = "adapter.rollback.recorded"


@dataclass(frozen=True)
class BrainEventRecord:
    """One append-only brain event row."""

    id: int
    event_id: str
    event_type: str
    ts: str
    agent_id: str
    user_id: str
    session_id: str
    thread_id: str
    source: str
    correlation_id: str | None
    causal_parent_id: str | None
    confidence: float
    payload_json: str
    tags_json: str

    @property
    def payload(self) -> dict:
        """Return the decoded event payload."""
        return safe_json_dict(
            self.payload_json,
            max_chars=MAX_BRAIN_EVENT_PAYLOAD_JSON_CHARS,
            overflow_kind="brain_event_payload_too_large",
        )

    @property
    def tags(self) -> list[str]:
        """Return the decoded tag list."""
        return list(json.loads(self.tags_json))

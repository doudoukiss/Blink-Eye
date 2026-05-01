"""Provider-free typed projection surfaces for the Blink brain kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

BODY_STATE_PROJECTION = "body_state"
SCENE_STATE_PROJECTION = "scene_state"
SCENE_WORLD_STATE_PROJECTION = "scene_world_state"
ENGAGEMENT_STATE_PROJECTION = "engagement_state"
RELATIONSHIP_STATE_PROJECTION = "relationship_state"
WORKING_CONTEXT_PROJECTION = "working_context"
PRIVATE_WORKING_MEMORY_PROJECTION = "private_working_memory"
ACTIVE_SITUATION_MODEL_PROJECTION = "active_situation_model"
PREDICTIVE_WORLD_MODEL_PROJECTION = "predictive_world_model"
COUNTERFACTUAL_REHEARSAL_PROJECTION = "counterfactual_rehearsal"
EMBODIED_EXECUTIVE_PROJECTION = "embodied_executive"
PRACTICE_DIRECTOR_PROJECTION = "practice_director"
SKILL_EVIDENCE_PROJECTION = "skill_evidence"
SKILL_GOVERNANCE_PROJECTION = "skill_governance"
ADAPTER_GOVERNANCE_PROJECTION = "adapter_governance"
AGENDA_PROJECTION = "agenda"
HEARTBEAT_PROJECTION = "heartbeat"
COMMITMENT_PROJECTION = "commitment_projection"
AUTONOMY_LEDGER_PROJECTION = "autonomy_ledger"
CLAIM_GOVERNANCE_PROJECTION = "claim_governance"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _sorted_unique_texts(values: Any) -> list[str]:
    """Normalize a string collection into sorted unique text values."""
    normalized: set[str] = set()
    for value in values or ():
        text = _optional_text(value)
        if text is not None:
            normalized.add(text)
    return sorted(normalized)


class BrainGoalStatus(str, Enum):
    """Explicit goal statuses tracked in the agenda projection."""

    OPEN = "open"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    RETRY = "retry"
    WAITING = "waiting"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class BrainGoalFamily(str, Enum):
    """High-level goal families owned by the executive."""

    CONVERSATION = "conversation"
    MEMORY_MAINTENANCE = "memory_maintenance"
    ENVIRONMENT = "environment"


class BrainCommitmentScopeType(str, Enum):
    """Canonical durable commitment scopes."""

    RELATIONSHIP = "relationship"
    THREAD = "thread"
    AGENT = "agent"


class BrainCommitmentStatus(str, Enum):
    """Durable executive commitment statuses."""

    ACTIVE = "active"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class BrainBlockedReasonKind(str, Enum):
    """Structured blocked-reason kinds for goals and commitments."""

    CAPABILITY_BLOCKED = "capability_blocked"
    CAPABILITY_FAILED = "capability_failed"
    EXPLICIT_DEFER = "explicit_defer"
    OPERATOR_REVIEW = "operator_review"
    PLANNING_REQUIRED = "planning_required"
    WAITING_USER = "waiting_user"


class BrainWakeConditionKind(str, Enum):
    """Structured wake-condition kinds for goals and commitments."""

    CONDITION_CLEARED = "condition_cleared"
    EXPLICIT_RESUME = "explicit_resume"
    OPERATOR_REVIEW = "operator_review"
    THREAD_IDLE = "thread_idle"
    USER_RESPONSE = "user_response"


class BrainCommitmentWakeRouteKind(str, Enum):
    """Typed routing decisions for one matched commitment wake."""

    RESUME_DIRECT = "resume_direct"
    PROPOSE_CANDIDATE = "propose_candidate"
    KEEP_WAITING = "keep_waiting"


class BrainPlanProposalSource(str, Enum):
    """Canonical sources for one replayable plan proposal."""

    BOUNDED_PLANNER = "bounded_planner"
    DETERMINISTIC_PLANNER = "deterministic_planner"
    REPAIR = "repair"


class BrainPlanReviewPolicy(str, Enum):
    """Typed review policy for one bounded plan proposal."""

    AUTO_ADOPT_OK = "auto_adopt_ok"
    NEEDS_USER_REVIEW = "needs_user_review"
    NEEDS_OPERATOR_REVIEW = "needs_operator_review"


class BrainPrivateWorkingMemoryBufferKind(str, Enum):
    """Bounded private working-memory buffer kinds."""

    USER_MODEL = "user_model"
    SELF_POLICY = "self_policy"
    GOAL_COMMITMENT = "goal_commitment"
    PLAN_ASSUMPTION = "plan_assumption"
    SCENE_WORLD_STATE = "scene_world_state"
    UNRESOLVED_UNCERTAINTY = "unresolved_uncertainty"
    RECENT_TOOL_OUTCOME = "recent_tool_outcome"


class BrainPrivateWorkingMemoryRecordState(str, Enum):
    """Lifecycle states for one private working-memory record."""

    ACTIVE = "active"
    STALE = "stale"
    RESOLVED = "resolved"


class BrainPrivateWorkingMemoryEvidenceKind(str, Enum):
    """Evidence stance for one private working-memory record."""

    OBSERVED = "observed"
    DERIVED = "derived"
    HYPOTHESIZED = "hypothesized"


class BrainGovernanceReasonCode(str, Enum):
    """Canonical governance reason codes for claim state transitions."""

    SUPERSEDED = "superseded"
    CONTRADICTION = "contradiction"
    STALE_WITHOUT_REFRESH = "stale_without_refresh"
    EXPIRED_BY_POLICY = "expired_by_policy"
    OPERATOR_HOLD = "operator_hold"
    LOW_SUPPORT = "low_support"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    PRIVACY_BOUNDARY = "privacy_boundary"
    DEGRADED_SCENE_EVIDENCE = "degraded_scene_evidence"
    USER_PINNED = "user_pinned"


class BrainClaimCurrentnessStatus(str, Enum):
    """Explicit currentness statuses for continuity claims."""

    CURRENT = "current"
    STALE = "stale"
    HISTORICAL = "historical"
    HELD = "held"


class BrainClaimReviewState(str, Enum):
    """Explicit review states for continuity claims."""

    NONE = "none"
    REQUESTED = "requested"
    RESOLVED = "resolved"


class BrainClaimRetentionClass(str, Enum):
    """Explicit retention classes for continuity claims."""

    TRANSIENT = "transient"
    SESSION = "session"
    RECURRING = "recurring"
    DURABLE = "durable"


class BrainActiveSituationRecordKind(str, Enum):
    """Bounded active situation-model record kinds."""

    SCENE_STATE = "scene_state"
    WORLD_STATE = "world_state"
    GOAL_STATE = "goal_state"
    COMMITMENT_STATE = "commitment_state"
    PLAN_STATE = "plan_state"
    PROCEDURAL_STATE = "procedural_state"
    UNCERTAINTY_STATE = "uncertainty_state"
    PREDICTION_STATE = "prediction_state"


class BrainActiveSituationRecordState(str, Enum):
    """Lifecycle states for one active situation-model record."""

    ACTIVE = "active"
    STALE = "stale"
    UNRESOLVED = "unresolved"


class BrainActiveSituationEvidenceKind(str, Enum):
    """Evidence stance for one active situation-model record."""

    OBSERVED = "observed"
    DERIVED = "derived"
    HYPOTHESIZED = "hypothesized"


class BrainPredictionKind(str, Enum):
    """Typed short-horizon predictive record kinds."""

    ENTITY_PERSISTENCE = "entity_persistence"
    AFFORDANCE_PERSISTENCE = "affordance_persistence"
    ENGAGEMENT_DRIFT = "engagement_drift"
    SCENE_TRANSITION = "scene_transition"
    ACTION_OUTCOME = "action_outcome"
    WAKE_READINESS = "wake_readiness"


class BrainPredictionConfidenceBand(str, Enum):
    """Explicit confidence bands for predictive state."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BrainPredictionResolutionKind(str, Enum):
    """Terminal lifecycle outcomes for one prediction."""

    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class BrainPredictionSubjectKind(str, Enum):
    """Typed subject categories for predictive records."""

    ENTITY = "entity"
    AFFORDANCE = "affordance"
    ENGAGEMENT = "engagement"
    SCENE = "scene"
    ACTION = "action"
    WAKE_CONDITION = "wake_condition"


class BrainCounterfactualRehearsalKind(str, Enum):
    """Typed rehearsal branch kinds for bounded counterfactual evaluation."""

    EMBODIED_ACTION = "embodied_action"
    WAIT_ALTERNATIVE = "wait_alternative"
    FALLBACK_ACTION = "fallback_action"


class BrainRehearsalDecisionRecommendation(str, Enum):
    """Terminal bounded recommendations for one rehearsal result."""

    PROCEED = "proceed"
    PROCEED_CAUTIOUSLY = "proceed_cautiously"
    WAIT = "wait"
    REPAIR = "repair"
    ABORT = "abort"


class BrainObservedActionOutcomeKind(str, Enum):
    """Observed action-outcome kinds used for rehearsal calibration."""

    SUCCESS = "success"
    FAILURE = "failure"
    PREVIEW_ONLY = "preview_only"


class BrainCalibrationBucket(str, Enum):
    """Calibration buckets derived from predicted vs observed action outcomes."""

    ALIGNED = "aligned"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    NOT_CALIBRATED = "not_calibrated"


class BrainEmbodiedIntentKind(str, Enum):
    """Typed high-level embodied intent kinds for coordinator-owned execution."""

    ATTEND = "attend"
    SIGNAL_LISTENING = "signal_listening"
    SIGNAL_THINKING = "signal_thinking"
    SIGNAL_SPEAKING = "signal_speaking"
    INSPECT_SCENE = "inspect_scene"
    PREPARE_ACTION = "prepare_action"
    EXECUTE_ACTION = "execute_action"
    RECOVER_SAFE_STATE = "recover_safe_state"


class BrainEmbodiedDispatchDisposition(str, Enum):
    """Coordinator-owned embodied dispatch outcomes."""

    DISPATCH = "dispatch"
    DEFER = "defer"
    REPAIR = "repair"
    ABORT = "abort"


class BrainEmbodiedExecutorKind(str, Enum):
    """Low-level executor kinds used by the embodied coordinator."""

    ROBOT_HEAD_CAPABILITY = "robot_head_capability"
    ROBOT_HEAD_POLICY = "robot_head_policy"
    ROBOT_HEAD_RECOVERY = "robot_head_recovery"


class BrainEmbodiedTraceStatus(str, Enum):
    """Lifecycle statuses for embodied execution traces."""

    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEFERRED = "deferred"
    REPAIRED = "repaired"
    ABORTED = "aborted"


class BrainSceneWorldEntityKind(str, Enum):
    """Typed symbolic scene-world entity kinds."""

    PERSON = "person"
    OBJECT = "object"
    SURFACE = "surface"
    ZONE = "zone"
    UNKNOWN = "unknown"


class BrainSceneWorldRecordState(str, Enum):
    """Lifecycle states for one symbolic scene-world record."""

    ACTIVE = "active"
    STALE = "stale"
    CONTRADICTED = "contradicted"
    EXPIRED = "expired"


class BrainSceneWorldEvidenceKind(str, Enum):
    """Evidence stance for one symbolic scene-world record."""

    OBSERVED = "observed"
    DERIVED = "derived"
    HYPOTHESIZED = "hypothesized"


class BrainSceneWorldAffordanceAvailability(str, Enum):
    """Availability states for one scene-linked affordance."""

    AVAILABLE = "available"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"
    STALE = "stale"


@dataclass(frozen=True)
class BrainBlockedReason:
    """Structured blocked reason for a durable commitment or goal."""

    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the blocked reason."""
        return {
            "kind": self.kind,
            "summary": self.summary,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainBlockedReason | None":
        """Hydrate a blocked reason from stored JSON."""
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not kind or not summary:
            return None
        return cls(
            kind=kind,
            summary=summary,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainWakeCondition:
    """Structured wake or resume condition for a durable commitment."""

    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the wake condition."""
        return {
            "kind": self.kind,
            "summary": self.summary,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainWakeCondition | None":
        """Hydrate a wake condition from stored JSON."""
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not kind or not summary:
            return None
        return cls(
            kind=kind,
            summary=summary,
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainClaimGovernanceRecord:
    """One explicit governance snapshot for a continuity claim."""

    claim_id: str
    scope_type: str
    scope_id: str
    truth_status: str
    currentness_status: str
    review_state: str
    retention_class: str
    reason_codes: tuple[str, ...] = ()
    last_governance_event_id: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the governance record."""
        return {
            "claim_id": self.claim_id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "truth_status": self.truth_status,
            "currentness_status": self.currentness_status,
            "review_state": self.review_state,
            "retention_class": self.retention_class,
            "reason_codes": list(self.reason_codes),
            "last_governance_event_id": self.last_governance_event_id,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainClaimGovernanceRecord | None":
        """Hydrate one governance record from stored JSON."""
        if not isinstance(data, dict):
            return None
        claim_id = str(data.get("claim_id", "")).strip()
        scope_type = str(data.get("scope_type", "")).strip()
        scope_id = str(data.get("scope_id", "")).strip()
        truth_status = str(data.get("truth_status", "")).strip()
        currentness_status = str(data.get("currentness_status", "")).strip()
        review_state = str(data.get("review_state", "")).strip()
        retention_class = str(data.get("retention_class", "")).strip()
        if not claim_id or not scope_type or not scope_id:
            return None
        return cls(
            claim_id=claim_id,
            scope_type=scope_type,
            scope_id=scope_id,
            truth_status=truth_status,
            currentness_status=currentness_status,
            review_state=review_state,
            retention_class=retention_class,
            reason_codes=tuple(_sorted_unique_texts(data.get("reason_codes", []))),
            last_governance_event_id=_optional_text(data.get("last_governance_event_id")),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainClaimGovernanceProjection:
    """Typed claim-governance view scoped to one continuity scope."""

    scope_type: str
    scope_id: str
    records: list[BrainClaimGovernanceRecord] = field(default_factory=list)
    currentness_counts: dict[str, int] = field(default_factory=dict)
    review_state_counts: dict[str, int] = field(default_factory=dict)
    retention_class_counts: dict[str, int] = field(default_factory=dict)
    current_claim_ids: list[str] = field(default_factory=list)
    stale_claim_ids: list[str] = field(default_factory=list)
    historical_claim_ids: list[str] = field(default_factory=list)
    held_claim_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the governance projection."""
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "records": [record.as_dict() for record in self.records],
            "currentness_counts": dict(self.currentness_counts),
            "review_state_counts": dict(self.review_state_counts),
            "retention_class_counts": dict(self.retention_class_counts),
            "current_claim_ids": _sorted_unique_texts(self.current_claim_ids),
            "stale_claim_ids": _sorted_unique_texts(self.stale_claim_ids),
            "historical_claim_ids": _sorted_unique_texts(self.historical_claim_ids),
            "held_claim_ids": _sorted_unique_texts(self.held_claim_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainClaimGovernanceProjection":
        """Hydrate the governance projection from stored JSON."""
        payload = data or {}
        return cls(
            scope_type=str(payload.get("scope_type", "")).strip(),
            scope_id=str(payload.get("scope_id", "")).strip(),
            records=[
                record
                for item in payload.get("records", [])
                if (record := BrainClaimGovernanceRecord.from_dict(item)) is not None
            ],
            currentness_counts={
                str(key): int(value)
                for key, value in dict(payload.get("currentness_counts", {})).items()
            },
            review_state_counts={
                str(key): int(value)
                for key, value in dict(payload.get("review_state_counts", {})).items()
            },
            retention_class_counts={
                str(key): int(value)
                for key, value in dict(payload.get("retention_class_counts", {})).items()
            },
            current_claim_ids=_sorted_unique_texts(payload.get("current_claim_ids", [])),
            stale_claim_ids=_sorted_unique_texts(payload.get("stale_claim_ids", [])),
            historical_claim_ids=_sorted_unique_texts(payload.get("historical_claim_ids", [])),
            held_claim_ids=_sorted_unique_texts(payload.get("held_claim_ids", [])),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainCommitmentWakeTrigger:
    """Structured wake-trigger metadata for one commitment wake match."""

    commitment_id: str
    wake_kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    source_event_type: str | None = None
    source_event_id: str | None = None
    ts: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the wake trigger."""
        return {
            "commitment_id": self.commitment_id,
            "wake_kind": self.wake_kind,
            "summary": self.summary,
            "details": dict(self.details),
            "source_event_type": self.source_event_type,
            "source_event_id": self.source_event_id,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCommitmentWakeTrigger | None":
        """Hydrate a wake trigger from stored JSON."""
        if not isinstance(data, dict):
            return None
        commitment_id = str(data.get("commitment_id", "")).strip()
        wake_kind = str(data.get("wake_kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not commitment_id or not wake_kind or not summary:
            return None
        return cls(
            commitment_id=commitment_id,
            wake_kind=wake_kind,
            summary=summary,
            details=dict(data.get("details", {})),
            source_event_type=_optional_text(data.get("source_event_type")),
            source_event_id=_optional_text(data.get("source_event_id")),
            ts=str(data.get("ts") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainCommitmentWakeRoutingDecision:
    """Structured routing decision for one matched commitment wake."""

    route_kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    executive_policy: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the routing decision."""
        return {
            "route_kind": self.route_kind,
            "summary": self.summary,
            "details": dict(self.details),
            "reason_codes": list(self.reason_codes),
            "executive_policy": (
                dict(self.executive_policy) if self.executive_policy is not None else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainCommitmentWakeRoutingDecision | None":
        """Hydrate a routing decision from stored JSON."""
        if not isinstance(data, dict):
            return None
        route_kind = str(data.get("route_kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not route_kind or not summary:
            return None
        return cls(
            route_kind=route_kind,
            summary=summary,
            details=dict(data.get("details", {})),
            reason_codes=sorted(
                {str(item).strip() for item in data.get("reason_codes", []) if str(item).strip()}
            ),
            executive_policy=(
                dict(data.get("executive_policy", {}))
                if isinstance(data.get("executive_policy"), dict)
                else None
            ),
        )


@dataclass(frozen=True)
class BrainPlanProposal:
    """Structured plan proposal for one goal or commitment revision."""

    plan_proposal_id: str
    goal_id: str
    commitment_id: str | None
    source: str
    summary: str
    current_plan_revision: int
    plan_revision: int
    review_policy: str = BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
    steps: list["BrainGoalStep"] = field(default_factory=list)
    preserved_prefix_count: int = 0
    assumptions: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    supersedes_plan_proposal_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the plan proposal."""
        return {
            "plan_proposal_id": self.plan_proposal_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "source": self.source,
            "summary": self.summary,
            "current_plan_revision": self.current_plan_revision,
            "plan_revision": self.plan_revision,
            "review_policy": self.review_policy,
            "steps": [step.as_dict() for step in self.steps],
            "preserved_prefix_count": self.preserved_prefix_count,
            "assumptions": list(self.assumptions),
            "missing_inputs": list(self.missing_inputs),
            "supersedes_plan_proposal_id": self.supersedes_plan_proposal_id,
            "details": dict(self.details),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanProposal | None":
        """Hydrate a plan proposal from stored JSON."""
        if not isinstance(data, dict):
            return None
        plan_proposal_id = str(data.get("plan_proposal_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        source = str(data.get("source", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not plan_proposal_id or not goal_id or not source or not summary:
            return None
        return cls(
            plan_proposal_id=plan_proposal_id,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            source=source,
            summary=summary,
            current_plan_revision=int(data.get("current_plan_revision", 1)),
            plan_revision=int(data.get("plan_revision", 1)),
            review_policy=(
                str(data.get("review_policy", BrainPlanReviewPolicy.AUTO_ADOPT_OK.value)).strip()
                or BrainPlanReviewPolicy.AUTO_ADOPT_OK.value
            ),
            steps=[BrainGoalStep.from_dict(item) for item in data.get("steps", [])],
            preserved_prefix_count=int(data.get("preserved_prefix_count", 0)),
            assumptions=[
                str(item).strip() for item in data.get("assumptions", []) if str(item).strip()
            ],
            missing_inputs=[
                str(item).strip() for item in data.get("missing_inputs", []) if str(item).strip()
            ],
            supersedes_plan_proposal_id=_optional_text(data.get("supersedes_plan_proposal_id")),
            details=dict(data.get("details", {})),
            created_at=str(data.get("created_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainPlanProposalDecision:
    """Structured adoption or rejection metadata for one plan proposal."""

    summary: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    executive_policy: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the decision."""
        return {
            "summary": self.summary,
            "reason": self.reason,
            "details": dict(self.details),
            "reason_codes": list(self.reason_codes),
            "executive_policy": (
                dict(self.executive_policy) if self.executive_policy is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPlanProposalDecision | None":
        """Hydrate a plan proposal decision from stored JSON."""
        if not isinstance(data, dict):
            return None
        summary = str(data.get("summary", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if not summary or not reason:
            return None
        return cls(
            summary=summary,
            reason=reason,
            details=dict(data.get("details", {})),
            reason_codes=sorted(
                {str(item).strip() for item in data.get("reason_codes", []) if str(item).strip()}
            ),
            executive_policy=(
                dict(data.get("executive_policy", {}))
                if isinstance(data.get("executive_policy"), dict)
                else None
            ),
        )


@dataclass
class BrainGoalStep:
    """One planned capability step owned by the executive."""

    capability_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    attempts: int = 0
    summary: str | None = None
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the step."""
        return {
            "capability_id": self.capability_id,
            "arguments": dict(self.arguments),
            "status": self.status,
            "attempts": self.attempts,
            "summary": self.summary,
            "error_code": self.error_code,
            "warnings": list(self.warnings),
            "output": dict(self.output),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainGoalStep":
        """Hydrate one goal step from JSON."""
        payload = data or {}
        return cls(
            capability_id=str(payload.get("capability_id", "")).strip(),
            arguments=dict(payload.get("arguments", {})),
            status=str(payload.get("status", "pending")),
            attempts=int(payload.get("attempts", 0)),
            summary=payload.get("summary"),
            error_code=payload.get("error_code"),
            warnings=list(payload.get("warnings", [])),
            output=dict(payload.get("output", {})),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainGoal:
    """One explicit agenda goal tracked as projected state."""

    goal_id: str
    title: str
    intent: str
    source: str
    goal_family: str = BrainGoalFamily.CONVERSATION.value
    commitment_id: str | None = None
    status: str = BrainGoalStatus.OPEN.value
    details: dict[str, Any] = field(default_factory=dict)
    steps: list[BrainGoalStep] = field(default_factory=list)
    active_step_index: int | None = None
    recovery_count: int = 0
    planning_requested: bool = False
    blocked_reason: BrainBlockedReason | None = None
    wake_conditions: list[BrainWakeCondition] = field(default_factory=list)
    plan_revision: int = 1
    resume_count: int = 0
    last_summary: str | None = None
    last_error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the goal."""
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "intent": self.intent,
            "source": self.source,
            "goal_family": self.goal_family,
            "commitment_id": self.commitment_id,
            "status": self.status,
            "details": dict(self.details),
            "steps": [step.as_dict() for step in self.steps],
            "active_step_index": self.active_step_index,
            "recovery_count": self.recovery_count,
            "planning_requested": self.planning_requested,
            "blocked_reason": self.blocked_reason.as_dict() if self.blocked_reason else None,
            "wake_conditions": [item.as_dict() for item in self.wake_conditions],
            "plan_revision": self.plan_revision,
            "resume_count": self.resume_count,
            "last_summary": self.last_summary,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainGoal":
        """Hydrate one goal from JSON."""
        payload = data or {}
        return cls(
            goal_id=str(payload.get("goal_id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            intent=str(payload.get("intent", "")).strip(),
            source=str(payload.get("source", "unknown")).strip() or "unknown",
            goal_family=str(payload.get("goal_family", BrainGoalFamily.CONVERSATION.value)).strip()
            or BrainGoalFamily.CONVERSATION.value,
            commitment_id=_optional_text(payload.get("commitment_id")),
            status=str(payload.get("status", BrainGoalStatus.OPEN.value)),
            details=dict(payload.get("details", {})),
            steps=[BrainGoalStep.from_dict(item) for item in payload.get("steps", [])],
            active_step_index=payload.get("active_step_index"),
            recovery_count=int(payload.get("recovery_count", 0)),
            planning_requested=bool(payload.get("planning_requested", False)),
            blocked_reason=BrainBlockedReason.from_dict(payload.get("blocked_reason")),
            wake_conditions=[
                item
                for item in (
                    BrainWakeCondition.from_dict(entry)
                    for entry in payload.get("wake_conditions", [])
                )
                if item is not None
            ],
            plan_revision=int(payload.get("plan_revision", 1)),
            resume_count=int(payload.get("resume_count", 0)),
            last_summary=payload.get("last_summary"),
            last_error=payload.get("last_error"),
            created_at=str(payload.get("created_at") or _utc_now()),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )

    def next_runnable_step_index(self) -> int | None:
        """Return the next pending or retry step index for this goal."""
        for index, step in enumerate(self.steps):
            if step.status in {"pending", "retry"}:
                return index
        return None

    @property
    def is_terminal(self) -> bool:
        """Return whether the goal has reached a terminal state."""
        return self.status in {
            BrainGoalStatus.BLOCKED.value,
            BrainGoalStatus.CANCELLED.value,
            BrainGoalStatus.COMPLETED.value,
            BrainGoalStatus.FAILED.value,
        }


@dataclass
class BrainWorkingContextProjection:
    """Short-horizon active conversational context derived from events."""

    last_user_text: str | None = None
    last_assistant_text: str | None = None
    last_tool_name: str | None = None
    last_tool_result: dict[str, Any] | list[Any] | str | None = None
    user_turn_open: bool = False
    assistant_turn_open: bool = False
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "last_user_text": self.last_user_text,
            "last_assistant_text": self.last_assistant_text,
            "last_tool_name": self.last_tool_name,
            "last_tool_result": self.last_tool_result,
            "user_turn_open": self.user_turn_open,
            "assistant_turn_open": self.assistant_turn_open,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainWorkingContextProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        return cls(
            last_user_text=payload.get("last_user_text"),
            last_assistant_text=payload.get("last_assistant_text"),
            last_tool_name=payload.get("last_tool_name"),
            last_tool_result=payload.get("last_tool_result"),
            user_turn_open=bool(payload.get("user_turn_open", False)),
            assistant_turn_open=bool(payload.get("assistant_turn_open", False)),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


def _record_state_rank(state: str) -> int:
    return {
        BrainPrivateWorkingMemoryRecordState.ACTIVE.value: 0,
        BrainPrivateWorkingMemoryRecordState.STALE.value: 1,
        BrainPrivateWorkingMemoryRecordState.RESOLVED.value: 2,
    }.get(str(state).strip(), 9)


def _private_record_sort_key(
    record: "BrainPrivateWorkingMemoryRecord",
) -> tuple[str, int, str, str]:
    return (
        record.buffer_kind,
        _record_state_rank(record.state),
        str(record.updated_at or ""),
        record.record_id,
    )


def _situation_state_rank(state: str) -> int:
    return {
        BrainActiveSituationRecordState.ACTIVE.value: 0,
        BrainActiveSituationRecordState.STALE.value: 1,
        BrainActiveSituationRecordState.UNRESOLVED.value: 2,
    }.get(str(state).strip(), 9)


def _situation_sort_score(updated_at: str | None) -> float:
    if not updated_at:
        return 0.0
    try:
        parsed = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return -parsed.timestamp()


def _active_situation_record_sort_key(
    record: "BrainActiveSituationRecord",
) -> tuple[str, int, float, str]:
    return (
        record.record_kind,
        _situation_state_rank(record.state),
        _situation_sort_score(record.updated_at),
        record.record_id,
    )


def _prediction_confidence_rank(confidence_band: str) -> int:
    return {
        BrainPredictionConfidenceBand.HIGH.value: 0,
        BrainPredictionConfidenceBand.MEDIUM.value: 1,
        BrainPredictionConfidenceBand.LOW.value: 2,
    }.get(str(confidence_band).strip(), 9)


def _prediction_resolution_rank(resolution_kind: str | None) -> int:
    return {
        None: 0,
        "": 0,
        BrainPredictionResolutionKind.CONFIRMED.value: 1,
        BrainPredictionResolutionKind.INVALIDATED.value: 2,
        BrainPredictionResolutionKind.EXPIRED.value: 3,
    }.get(_optional_text(resolution_kind), 9)


def _prediction_sort_key(
    record: "BrainPredictionRecord",
) -> tuple[int, str, int, float, str]:
    return (
        _prediction_resolution_rank(record.resolution_kind),
        record.prediction_kind,
        _prediction_confidence_rank(record.confidence_band),
        _situation_sort_score(record.valid_to or record.updated_at),
        record.prediction_id,
    )


def _rehearsal_recommendation_rank(recommendation: str | None) -> int:
    return {
        BrainRehearsalDecisionRecommendation.ABORT.value: 0,
        BrainRehearsalDecisionRecommendation.REPAIR.value: 1,
        BrainRehearsalDecisionRecommendation.WAIT.value: 2,
        BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value: 3,
        BrainRehearsalDecisionRecommendation.PROCEED.value: 4,
        None: 9,
        "": 9,
    }.get(_optional_text(recommendation), 9)


def _rehearsal_kind_rank(rehearsal_kind: str | None) -> int:
    return {
        BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value: 0,
        BrainCounterfactualRehearsalKind.WAIT_ALTERNATIVE.value: 1,
        BrainCounterfactualRehearsalKind.FALLBACK_ACTION.value: 2,
        None: 9,
        "": 9,
    }.get(_optional_text(rehearsal_kind), 9)


def _rehearsal_request_sort_key(
    request: "BrainActionRehearsalRequest",
) -> tuple[int, str, float, str]:
    return (
        int(request.step_index),
        request.plan_proposal_id or "",
        _situation_sort_score(request.requested_at),
        request.rehearsal_id,
    )


def _rehearsal_result_sort_key(
    result: "BrainActionRehearsalResult",
) -> tuple[int, int, float, str]:
    return (
        _rehearsal_recommendation_rank(result.decision_recommendation),
        _rehearsal_kind_rank(result.rehearsal_kind),
        _situation_sort_score(result.completed_at or result.updated_at),
        result.rehearsal_id,
    )


def _rehearsal_comparison_sort_key(
    comparison: "BrainActionOutcomeComparisonRecord",
) -> tuple[int, float, str]:
    return (
        {
            BrainCalibrationBucket.OVERCONFIDENT.value: 0,
            BrainCalibrationBucket.UNDERCONFIDENT.value: 1,
            BrainCalibrationBucket.ALIGNED.value: 2,
            BrainCalibrationBucket.NOT_CALIBRATED.value: 3,
        }.get(comparison.calibration_bucket, 9),
        _situation_sort_score(comparison.compared_at or comparison.updated_at),
        comparison.comparison_id,
    )


def _scene_world_state_rank(state: str) -> int:
    return {
        BrainSceneWorldRecordState.ACTIVE.value: 0,
        BrainSceneWorldRecordState.STALE.value: 1,
        BrainSceneWorldRecordState.CONTRADICTED.value: 2,
        BrainSceneWorldRecordState.EXPIRED.value: 3,
    }.get(str(state).strip(), 9)


def _scene_world_entity_sort_key(
    record: "BrainSceneWorldEntityRecord",
) -> tuple[str, int, float, str]:
    return (
        record.entity_kind,
        _scene_world_state_rank(record.state),
        _situation_sort_score(record.updated_at),
        record.entity_id,
    )


def _scene_world_affordance_sort_key(
    record: "BrainSceneWorldAffordanceRecord",
) -> tuple[str, int, float, str]:
    availability_rank = {
        BrainSceneWorldAffordanceAvailability.AVAILABLE.value: 0,
        BrainSceneWorldAffordanceAvailability.BLOCKED.value: 1,
        BrainSceneWorldAffordanceAvailability.UNCERTAIN.value: 2,
        BrainSceneWorldAffordanceAvailability.STALE.value: 3,
    }.get(str(record.availability).strip(), 9)
    return (
        record.capability_family,
        availability_rank,
        _situation_sort_score(record.updated_at),
        record.affordance_id,
    )


@dataclass(frozen=True)
class BrainPrivateWorkingMemoryRecord:
    """One bounded private working-memory record."""

    record_id: str
    buffer_kind: str
    summary: str
    state: str
    evidence_kind: str
    backing_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    goal_id: str | None = None
    commitment_id: str | None = None
    plan_proposal_id: str | None = None
    skill_id: str | None = None
    observed_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    expires_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "record_id": self.record_id,
            "buffer_kind": self.buffer_kind,
            "summary": self.summary,
            "state": self.state,
            "evidence_kind": self.evidence_kind,
            "backing_ids": list(self.backing_ids),
            "source_event_ids": list(self.source_event_ids),
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "skill_id": self.skill_id,
            "observed_at": self.observed_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPrivateWorkingMemoryRecord | None":
        """Hydrate a record from stored JSON."""
        if not isinstance(data, dict):
            return None
        record_id = str(data.get("record_id", "")).strip()
        buffer_kind = str(data.get("buffer_kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        state = (
            str(data.get("state", BrainPrivateWorkingMemoryRecordState.ACTIVE.value)).strip()
            or BrainPrivateWorkingMemoryRecordState.ACTIVE.value
        )
        evidence_kind = (
            str(
                data.get("evidence_kind", BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value)
            ).strip()
            or BrainPrivateWorkingMemoryEvidenceKind.DERIVED.value
        )
        if not record_id or not buffer_kind or not summary:
            return None
        return cls(
            record_id=record_id,
            buffer_kind=buffer_kind,
            summary=summary,
            state=state,
            evidence_kind=evidence_kind,
            backing_ids=sorted(
                {str(item).strip() for item in data.get("backing_ids", []) if str(item).strip()}
            ),
            source_event_ids=sorted(
                {
                    str(item).strip()
                    for item in data.get("source_event_ids", [])
                    if str(item).strip()
                }
            ),
            goal_id=_optional_text(data.get("goal_id")),
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            skill_id=_optional_text(data.get("skill_id")),
            observed_at=_optional_text(data.get("observed_at")),
            updated_at=str(data.get("updated_at") or _utc_now()),
            expires_at=_optional_text(data.get("expires_at")),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainPrivateWorkingMemoryProjection:
    """Bounded thread-scoped private working-memory projection."""

    scope_type: str
    scope_id: str
    records: list[BrainPrivateWorkingMemoryRecord] = field(default_factory=list)
    buffer_counts: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)
    evidence_kind_counts: dict[str, int] = field(default_factory=dict)
    active_record_ids: list[str] = field(default_factory=list)
    stale_record_ids: list[str] = field(default_factory=list)
    resolved_record_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived counts and record-id indexes from the structured records."""
        self.records = sorted(self.records, key=_private_record_sort_key)
        buffer_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        evidence_kind_counts: dict[str, int] = {}
        active_record_ids: list[str] = []
        stale_record_ids: list[str] = []
        resolved_record_ids: list[str] = []
        for record in self.records:
            buffer_counts[record.buffer_kind] = buffer_counts.get(record.buffer_kind, 0) + 1
            state_counts[record.state] = state_counts.get(record.state, 0) + 1
            evidence_kind_counts[record.evidence_kind] = (
                evidence_kind_counts.get(record.evidence_kind, 0) + 1
            )
            if record.state == BrainPrivateWorkingMemoryRecordState.ACTIVE.value:
                active_record_ids.append(record.record_id)
            elif record.state == BrainPrivateWorkingMemoryRecordState.STALE.value:
                stale_record_ids.append(record.record_id)
            elif record.state == BrainPrivateWorkingMemoryRecordState.RESOLVED.value:
                resolved_record_ids.append(record.record_id)
        self.buffer_counts = dict(sorted(buffer_counts.items()))
        self.state_counts = dict(sorted(state_counts.items()))
        self.evidence_kind_counts = dict(sorted(evidence_kind_counts.items()))
        self.active_record_ids = active_record_ids
        self.stale_record_ids = stale_record_ids
        self.resolved_record_ids = resolved_record_ids

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        self.sync_lists()
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "records": [record.as_dict() for record in self.records],
            "buffer_counts": dict(self.buffer_counts),
            "state_counts": dict(self.state_counts),
            "evidence_kind_counts": dict(self.evidence_kind_counts),
            "active_record_ids": list(self.active_record_ids),
            "stale_record_ids": list(self.stale_record_ids),
            "resolved_record_ids": list(self.resolved_record_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPrivateWorkingMemoryProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_type=str(payload.get("scope_type", "thread")).strip() or "thread",
            scope_id=str(payload.get("scope_id", "")).strip(),
            records=[
                record
                for item in payload.get("records", [])
                if (record := BrainPrivateWorkingMemoryRecord.from_dict(item)) is not None
            ],
            buffer_counts=dict(payload.get("buffer_counts", {})),
            state_counts=dict(payload.get("state_counts", {})),
            evidence_kind_counts=dict(payload.get("evidence_kind_counts", {})),
            active_record_ids=[
                str(item).strip()
                for item in payload.get("active_record_ids", [])
                if str(item).strip()
            ],
            stale_record_ids=[
                str(item).strip()
                for item in payload.get("stale_record_ids", [])
                if str(item).strip()
            ],
            resolved_record_ids=[
                str(item).strip()
                for item in payload.get("resolved_record_ids", [])
                if str(item).strip()
            ],
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


@dataclass(frozen=True)
class BrainActiveSituationRecord:
    """One bounded active situation-model record."""

    record_id: str
    record_kind: str
    summary: str
    state: str
    evidence_kind: str
    confidence: float | None = None
    freshness: str | None = None
    uncertainty_codes: list[str] = field(default_factory=list)
    private_record_ids: list[str] = field(default_factory=list)
    backing_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    goal_id: str | None = None
    commitment_id: str | None = None
    plan_proposal_id: str | None = None
    skill_id: str | None = None
    observed_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    expires_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "record_id": self.record_id,
            "record_kind": self.record_kind,
            "summary": self.summary,
            "state": self.state,
            "evidence_kind": self.evidence_kind,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "uncertainty_codes": list(self.uncertainty_codes),
            "private_record_ids": list(self.private_record_ids),
            "backing_ids": list(self.backing_ids),
            "source_event_ids": list(self.source_event_ids),
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "skill_id": self.skill_id,
            "observed_at": self.observed_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainActiveSituationRecord | None":
        """Hydrate a record from stored JSON."""
        if not isinstance(data, dict):
            return None
        record_id = str(data.get("record_id", "")).strip()
        record_kind = str(data.get("record_kind", "")).strip()
        summary = str(data.get("summary", "")).strip()
        state = (
            str(data.get("state", BrainActiveSituationRecordState.ACTIVE.value)).strip()
            or BrainActiveSituationRecordState.ACTIVE.value
        )
        evidence_kind = (
            str(data.get("evidence_kind", BrainActiveSituationEvidenceKind.DERIVED.value)).strip()
            or BrainActiveSituationEvidenceKind.DERIVED.value
        )
        if not record_id or not record_kind or not summary:
            return None
        return cls(
            record_id=record_id,
            record_kind=record_kind,
            summary=summary,
            state=state,
            evidence_kind=evidence_kind,
            confidence=(float(data["confidence"]) if data.get("confidence") is not None else None),
            freshness=_optional_text(data.get("freshness")),
            uncertainty_codes=sorted(
                {
                    str(item).strip()
                    for item in data.get("uncertainty_codes", [])
                    if str(item).strip()
                }
            ),
            private_record_ids=sorted(
                {
                    str(item).strip()
                    for item in data.get("private_record_ids", [])
                    if str(item).strip()
                }
            ),
            backing_ids=sorted(
                {str(item).strip() for item in data.get("backing_ids", []) if str(item).strip()}
            ),
            source_event_ids=sorted(
                {
                    str(item).strip()
                    for item in data.get("source_event_ids", [])
                    if str(item).strip()
                }
            ),
            goal_id=_optional_text(data.get("goal_id")),
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            skill_id=_optional_text(data.get("skill_id")),
            observed_at=_optional_text(data.get("observed_at")),
            updated_at=str(data.get("updated_at") or _utc_now()),
            expires_at=_optional_text(data.get("expires_at")),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainActiveSituationProjection:
    """Bounded thread-scoped active situation-model projection."""

    scope_type: str
    scope_id: str
    records: list[BrainActiveSituationRecord] = field(default_factory=list)
    kind_counts: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)
    uncertainty_code_counts: dict[str, int] = field(default_factory=dict)
    active_record_ids: list[str] = field(default_factory=list)
    stale_record_ids: list[str] = field(default_factory=list)
    unresolved_record_ids: list[str] = field(default_factory=list)
    linked_commitment_ids: list[str] = field(default_factory=list)
    linked_plan_proposal_ids: list[str] = field(default_factory=list)
    linked_skill_ids: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived counts and id indexes from structured records."""
        self.records = sorted(self.records, key=_active_situation_record_sort_key)
        kind_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        uncertainty_code_counts: dict[str, int] = {}
        active_record_ids: list[str] = []
        stale_record_ids: list[str] = []
        unresolved_record_ids: list[str] = []
        linked_commitment_ids: set[str] = set()
        linked_plan_proposal_ids: set[str] = set()
        linked_skill_ids: set[str] = set()
        for record in self.records:
            kind_counts[record.record_kind] = kind_counts.get(record.record_kind, 0) + 1
            state_counts[record.state] = state_counts.get(record.state, 0) + 1
            for code in record.uncertainty_codes:
                uncertainty_code_counts[code] = uncertainty_code_counts.get(code, 0) + 1
            if record.state == BrainActiveSituationRecordState.ACTIVE.value:
                active_record_ids.append(record.record_id)
            elif record.state == BrainActiveSituationRecordState.STALE.value:
                stale_record_ids.append(record.record_id)
            elif record.state == BrainActiveSituationRecordState.UNRESOLVED.value:
                unresolved_record_ids.append(record.record_id)
            if record.commitment_id:
                linked_commitment_ids.add(record.commitment_id)
            if record.plan_proposal_id:
                linked_plan_proposal_ids.add(record.plan_proposal_id)
            if record.skill_id:
                linked_skill_ids.add(record.skill_id)
        self.kind_counts = dict(sorted(kind_counts.items()))
        self.state_counts = dict(sorted(state_counts.items()))
        self.uncertainty_code_counts = dict(sorted(uncertainty_code_counts.items()))
        self.active_record_ids = active_record_ids
        self.stale_record_ids = stale_record_ids
        self.unresolved_record_ids = unresolved_record_ids
        self.linked_commitment_ids = sorted(linked_commitment_ids)
        self.linked_plan_proposal_ids = sorted(linked_plan_proposal_ids)
        self.linked_skill_ids = sorted(linked_skill_ids)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        self.sync_lists()
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "records": [record.as_dict() for record in self.records],
            "kind_counts": dict(self.kind_counts),
            "state_counts": dict(self.state_counts),
            "uncertainty_code_counts": dict(self.uncertainty_code_counts),
            "active_record_ids": list(self.active_record_ids),
            "stale_record_ids": list(self.stale_record_ids),
            "unresolved_record_ids": list(self.unresolved_record_ids),
            "linked_commitment_ids": list(self.linked_commitment_ids),
            "linked_plan_proposal_ids": list(self.linked_plan_proposal_ids),
            "linked_skill_ids": list(self.linked_skill_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainActiveSituationProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_type=str(payload.get("scope_type", "thread")).strip() or "thread",
            scope_id=str(payload.get("scope_id", "")).strip(),
            records=[
                record
                for item in payload.get("records", [])
                if (record := BrainActiveSituationRecord.from_dict(item)) is not None
            ],
            kind_counts=dict(payload.get("kind_counts", {})),
            state_counts=dict(payload.get("state_counts", {})),
            uncertainty_code_counts=dict(payload.get("uncertainty_code_counts", {})),
            active_record_ids=[
                str(item).strip()
                for item in payload.get("active_record_ids", [])
                if str(item).strip()
            ],
            stale_record_ids=[
                str(item).strip()
                for item in payload.get("stale_record_ids", [])
                if str(item).strip()
            ],
            unresolved_record_ids=[
                str(item).strip()
                for item in payload.get("unresolved_record_ids", [])
                if str(item).strip()
            ],
            linked_commitment_ids=[
                str(item).strip()
                for item in payload.get("linked_commitment_ids", [])
                if str(item).strip()
            ],
            linked_plan_proposal_ids=[
                str(item).strip()
                for item in payload.get("linked_plan_proposal_ids", [])
                if str(item).strip()
            ],
            linked_skill_ids=[
                str(item).strip()
                for item in payload.get("linked_skill_ids", [])
                if str(item).strip()
            ],
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


@dataclass(frozen=True)
class BrainPredictionCalibrationSummary:
    """Compact lifecycle counters for the predictive world model."""

    generated_count: int = 0
    active_count: int = 0
    confirmed_count: int = 0
    invalidated_count: int = 0
    expired_count: int = 0
    generated_kind_counts: dict[str, int] = field(default_factory=dict)
    resolution_kind_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the calibration summary."""
        return {
            "generated_count": self.generated_count,
            "active_count": self.active_count,
            "confirmed_count": self.confirmed_count,
            "invalidated_count": self.invalidated_count,
            "expired_count": self.expired_count,
            "generated_kind_counts": dict(self.generated_kind_counts),
            "resolution_kind_counts": dict(self.resolution_kind_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPredictionCalibrationSummary":
        """Hydrate the calibration summary from stored JSON."""
        payload = data or {}
        return cls(
            generated_count=int(payload.get("generated_count", 0)),
            active_count=int(payload.get("active_count", 0)),
            confirmed_count=int(payload.get("confirmed_count", 0)),
            invalidated_count=int(payload.get("invalidated_count", 0)),
            expired_count=int(payload.get("expired_count", 0)),
            generated_kind_counts={
                str(key): int(value)
                for key, value in dict(payload.get("generated_kind_counts", {})).items()
            },
            resolution_kind_counts={
                str(key): int(value)
                for key, value in dict(payload.get("resolution_kind_counts", {})).items()
            },
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainPredictionRecord:
    """One typed short-horizon prediction record."""

    prediction_id: str
    prediction_kind: str
    subject_kind: str
    subject_id: str
    scope_key: str
    presence_scope_key: str
    summary: str
    predicted_state: dict[str, Any]
    confidence: float
    confidence_band: str
    risk_codes: list[str] = field(default_factory=list)
    supporting_event_ids: list[str] = field(default_factory=list)
    backing_ids: list[str] = field(default_factory=list)
    action_id: str | None = None
    goal_id: str | None = None
    commitment_id: str | None = None
    plan_proposal_id: str | None = None
    skill_id: str | None = None
    predicted_at: str = field(default_factory=_utc_now)
    valid_from: str = field(default_factory=_utc_now)
    valid_to: str | None = None
    resolved_at: str | None = None
    resolution_kind: str | None = None
    resolution_event_ids: list[str] = field(default_factory=list)
    resolution_summary: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the prediction."""
        return {
            "prediction_id": self.prediction_id,
            "prediction_kind": self.prediction_kind,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "summary": self.summary,
            "predicted_state": dict(self.predicted_state),
            "confidence": self.confidence,
            "confidence_band": self.confidence_band,
            "risk_codes": list(self.risk_codes),
            "supporting_event_ids": list(self.supporting_event_ids),
            "backing_ids": list(self.backing_ids),
            "action_id": self.action_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "skill_id": self.skill_id,
            "predicted_at": self.predicted_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "resolved_at": self.resolved_at,
            "resolution_kind": self.resolution_kind,
            "resolution_event_ids": list(self.resolution_event_ids),
            "resolution_summary": self.resolution_summary,
            "details": dict(self.details),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPredictionRecord | None":
        """Hydrate one prediction from stored JSON."""
        if not isinstance(data, dict):
            return None
        prediction_id = str(data.get("prediction_id", "")).strip()
        prediction_kind = str(data.get("prediction_kind", "")).strip()
        subject_kind = str(data.get("subject_kind", "")).strip()
        subject_id = str(data.get("subject_id", "")).strip()
        scope_key = str(data.get("scope_key", "")).strip()
        presence_scope_key = str(data.get("presence_scope_key", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if (
            not prediction_id
            or not prediction_kind
            or not subject_kind
            or not subject_id
            or not scope_key
            or not presence_scope_key
            or not summary
        ):
            return None
        predicted_state = dict(data.get("predicted_state", {}))
        confidence = float(data.get("confidence", 0.0))
        confidence_band = (
            str(data.get("confidence_band", BrainPredictionConfidenceBand.LOW.value)).strip()
            or BrainPredictionConfidenceBand.LOW.value
        )
        return cls(
            prediction_id=prediction_id,
            prediction_kind=prediction_kind,
            subject_kind=subject_kind,
            subject_id=subject_id,
            scope_key=scope_key,
            presence_scope_key=presence_scope_key,
            summary=summary,
            predicted_state=predicted_state,
            confidence=confidence,
            confidence_band=confidence_band,
            risk_codes=_sorted_unique_texts(data.get("risk_codes", [])),
            supporting_event_ids=_sorted_unique_texts(data.get("supporting_event_ids", [])),
            backing_ids=_sorted_unique_texts(data.get("backing_ids", [])),
            action_id=_optional_text(data.get("action_id")),
            goal_id=_optional_text(data.get("goal_id")),
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            skill_id=_optional_text(data.get("skill_id")),
            predicted_at=str(data.get("predicted_at") or _utc_now()),
            valid_from=str(data.get("valid_from") or data.get("predicted_at") or _utc_now()),
            valid_to=_optional_text(data.get("valid_to")),
            resolved_at=_optional_text(data.get("resolved_at")),
            resolution_kind=_optional_text(data.get("resolution_kind")),
            resolution_event_ids=_sorted_unique_texts(data.get("resolution_event_ids", [])),
            resolution_summary=_optional_text(data.get("resolution_summary")),
            details=dict(data.get("details", {})),
            updated_at=str(data.get("updated_at") or data.get("predicted_at") or _utc_now()),
        )


@dataclass
class BrainPredictiveWorldModelProjection:
    """Replay-safe thread-scoped predictive world-model projection."""

    scope_key: str
    presence_scope_key: str
    active_predictions: list[BrainPredictionRecord] = field(default_factory=list)
    recent_resolutions: list[BrainPredictionRecord] = field(default_factory=list)
    calibration_summary: BrainPredictionCalibrationSummary = field(
        default_factory=BrainPredictionCalibrationSummary
    )
    active_prediction_ids: list[str] = field(default_factory=list)
    recent_resolution_ids: list[str] = field(default_factory=list)
    active_kind_counts: dict[str, int] = field(default_factory=dict)
    active_confidence_band_counts: dict[str, int] = field(default_factory=dict)
    resolution_kind_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived counts and id indexes from structured predictions."""
        self.active_predictions = sorted(self.active_predictions, key=_prediction_sort_key)
        self.recent_resolutions = sorted(self.recent_resolutions, key=_prediction_sort_key)
        active_prediction_ids: list[str] = []
        recent_resolution_ids: list[str] = []
        active_kind_counts: dict[str, int] = {}
        active_confidence_band_counts: dict[str, int] = {}
        resolution_kind_counts: dict[str, int] = {}
        for record in self.active_predictions:
            active_prediction_ids.append(record.prediction_id)
            active_kind_counts[record.prediction_kind] = (
                active_kind_counts.get(record.prediction_kind, 0) + 1
            )
            active_confidence_band_counts[record.confidence_band] = (
                active_confidence_band_counts.get(record.confidence_band, 0) + 1
            )
        for record in self.recent_resolutions:
            recent_resolution_ids.append(record.prediction_id)
            if record.resolution_kind:
                resolution_kind_counts[record.resolution_kind] = (
                    resolution_kind_counts.get(record.resolution_kind, 0) + 1
                )
        self.active_prediction_ids = active_prediction_ids
        self.recent_resolution_ids = recent_resolution_ids
        self.active_kind_counts = dict(sorted(active_kind_counts.items()))
        self.active_confidence_band_counts = dict(sorted(active_confidence_band_counts.items()))
        self.resolution_kind_counts = dict(sorted(resolution_kind_counts.items()))
        self.calibration_summary = BrainPredictionCalibrationSummary(
            generated_count=int(self.calibration_summary.generated_count),
            active_count=len(self.active_predictions),
            confirmed_count=int(self.calibration_summary.confirmed_count),
            invalidated_count=int(self.calibration_summary.invalidated_count),
            expired_count=int(self.calibration_summary.expired_count),
            generated_kind_counts=dict(self.calibration_summary.generated_kind_counts),
            resolution_kind_counts=dict(self.calibration_summary.resolution_kind_counts),
            updated_at=self.updated_at,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the predictive projection."""
        self.sync_lists()
        return {
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "active_predictions": [record.as_dict() for record in self.active_predictions],
            "recent_resolutions": [record.as_dict() for record in self.recent_resolutions],
            "calibration_summary": self.calibration_summary.as_dict(),
            "active_prediction_ids": list(self.active_prediction_ids),
            "recent_resolution_ids": list(self.recent_resolution_ids),
            "active_kind_counts": dict(self.active_kind_counts),
            "active_confidence_band_counts": dict(self.active_confidence_band_counts),
            "resolution_kind_counts": dict(self.resolution_kind_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPredictiveWorldModelProjection":
        """Hydrate the predictive projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_key=str(payload.get("scope_key", "")).strip(),
            presence_scope_key=str(payload.get("presence_scope_key", "")).strip()
            or "local:presence",
            active_predictions=[
                record
                for item in payload.get("active_predictions", [])
                if (record := BrainPredictionRecord.from_dict(item)) is not None
            ],
            recent_resolutions=[
                record
                for item in payload.get("recent_resolutions", [])
                if (record := BrainPredictionRecord.from_dict(item)) is not None
            ],
            calibration_summary=BrainPredictionCalibrationSummary.from_dict(
                payload.get("calibration_summary")
            ),
            active_prediction_ids=_sorted_unique_texts(payload.get("active_prediction_ids", [])),
            recent_resolution_ids=_sorted_unique_texts(payload.get("recent_resolution_ids", [])),
            active_kind_counts={
                str(key): int(value)
                for key, value in dict(payload.get("active_kind_counts", {})).items()
            },
            active_confidence_band_counts={
                str(key): int(value)
                for key, value in dict(payload.get("active_confidence_band_counts", {})).items()
            },
            resolution_kind_counts={
                str(key): int(value)
                for key, value in dict(payload.get("resolution_kind_counts", {})).items()
            },
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


@dataclass(frozen=True)
class BrainActionRehearsalRequest:
    """One bounded rehearsal request for a candidate embodied action."""

    rehearsal_id: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str
    step_index: int
    candidate_action_id: str
    fallback_action_ids: list[str] = field(default_factory=list)
    rehearsal_kind: str = BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value
    simulated_backend: str = "robot_head_simulation"
    supporting_prediction_ids: list[str] = field(default_factory=list)
    supporting_event_ids: list[str] = field(default_factory=list)
    requested_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the rehearsal request."""
        return {
            "rehearsal_id": self.rehearsal_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "step_index": self.step_index,
            "candidate_action_id": self.candidate_action_id,
            "fallback_action_ids": list(self.fallback_action_ids),
            "rehearsal_kind": self.rehearsal_kind,
            "simulated_backend": self.simulated_backend,
            "supporting_prediction_ids": list(self.supporting_prediction_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "requested_at": self.requested_at,
            "details": dict(self.details),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainActionRehearsalRequest | None":
        """Hydrate one rehearsal request from stored JSON."""
        if not isinstance(data, dict):
            return None
        rehearsal_id = str(data.get("rehearsal_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        plan_proposal_id = str(data.get("plan_proposal_id", "")).strip()
        candidate_action_id = str(data.get("candidate_action_id", "")).strip()
        if not rehearsal_id or not goal_id or not plan_proposal_id or not candidate_action_id:
            return None
        return cls(
            rehearsal_id=rehearsal_id,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=plan_proposal_id,
            step_index=int(data.get("step_index", 0)),
            candidate_action_id=candidate_action_id,
            fallback_action_ids=_sorted_unique_texts(data.get("fallback_action_ids", [])),
            rehearsal_kind=(
                str(
                    data.get(
                        "rehearsal_kind",
                        BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
                    )
                ).strip()
                or BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value
            ),
            simulated_backend=(
                str(data.get("simulated_backend", "robot_head_simulation")).strip()
                or "robot_head_simulation"
            ),
            supporting_prediction_ids=_sorted_unique_texts(
                data.get("supporting_prediction_ids", [])
            ),
            supporting_event_ids=_sorted_unique_texts(data.get("supporting_event_ids", [])),
            requested_at=str(data.get("requested_at") or _utc_now()),
            details=dict(data.get("details", {})),
            updated_at=str(
                data.get("updated_at") or data.get("requested_at") or _utc_now()
            ),
        )


@dataclass(frozen=True)
class BrainCounterfactualEvaluationRecord:
    """One bounded evaluation branch within a rehearsal result."""

    evaluation_id: str
    rehearsal_id: str
    candidate_action_id: str
    rehearsal_kind: str
    simulated_backend: str
    expected_preconditions: list[str]
    expected_effects: list[str]
    predicted_success_probability: float
    confidence_band: str
    risk_codes: list[str] = field(default_factory=list)
    fallback_action_ids: list[str] = field(default_factory=list)
    decision_recommendation: str = BrainRehearsalDecisionRecommendation.WAIT.value
    summary: str = ""
    supporting_prediction_ids: list[str] = field(default_factory=list)
    supporting_event_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evaluation record."""
        return {
            "evaluation_id": self.evaluation_id,
            "rehearsal_id": self.rehearsal_id,
            "candidate_action_id": self.candidate_action_id,
            "rehearsal_kind": self.rehearsal_kind,
            "simulated_backend": self.simulated_backend,
            "expected_preconditions": list(self.expected_preconditions),
            "expected_effects": list(self.expected_effects),
            "predicted_success_probability": self.predicted_success_probability,
            "confidence_band": self.confidence_band,
            "risk_codes": list(self.risk_codes),
            "fallback_action_ids": list(self.fallback_action_ids),
            "decision_recommendation": self.decision_recommendation,
            "summary": self.summary,
            "supporting_prediction_ids": list(self.supporting_prediction_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "details": dict(self.details),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainCounterfactualEvaluationRecord | None":
        """Hydrate one evaluation record from stored JSON."""
        if not isinstance(data, dict):
            return None
        evaluation_id = str(data.get("evaluation_id", "")).strip()
        rehearsal_id = str(data.get("rehearsal_id", "")).strip()
        candidate_action_id = str(data.get("candidate_action_id", "")).strip()
        rehearsal_kind = str(data.get("rehearsal_kind", "")).strip()
        simulated_backend = str(data.get("simulated_backend", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if (
            not evaluation_id
            or not rehearsal_id
            or not candidate_action_id
            or not rehearsal_kind
            or not simulated_backend
            or not summary
        ):
            return None
        probability = max(0.0, min(float(data.get("predicted_success_probability", 0.0)), 1.0))
        return cls(
            evaluation_id=evaluation_id,
            rehearsal_id=rehearsal_id,
            candidate_action_id=candidate_action_id,
            rehearsal_kind=rehearsal_kind,
            simulated_backend=simulated_backend,
            expected_preconditions=_sorted_unique_texts(data.get("expected_preconditions", [])),
            expected_effects=_sorted_unique_texts(data.get("expected_effects", [])),
            predicted_success_probability=probability,
            confidence_band=(
                str(data.get("confidence_band", BrainPredictionConfidenceBand.LOW.value)).strip()
                or BrainPredictionConfidenceBand.LOW.value
            ),
            risk_codes=_sorted_unique_texts(data.get("risk_codes", [])),
            fallback_action_ids=_sorted_unique_texts(data.get("fallback_action_ids", [])),
            decision_recommendation=(
                str(
                    data.get(
                        "decision_recommendation",
                        BrainRehearsalDecisionRecommendation.WAIT.value,
                    )
                ).strip()
                or BrainRehearsalDecisionRecommendation.WAIT.value
            ),
            summary=summary,
            supporting_prediction_ids=_sorted_unique_texts(
                data.get("supporting_prediction_ids", [])
            ),
            supporting_event_ids=_sorted_unique_texts(data.get("supporting_event_ids", [])),
            details=dict(data.get("details", {})),
            updated_at=str(data.get("updated_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainActionRehearsalResult:
    """One completed or skipped bounded rehearsal result."""

    rehearsal_id: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str
    step_index: int
    candidate_action_id: str
    fallback_action_ids: list[str] = field(default_factory=list)
    rehearsal_kind: str = BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value
    simulated_backend: str = "robot_head_simulation"
    expected_preconditions: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    predicted_success_probability: float = 0.0
    confidence_band: str = BrainPredictionConfidenceBand.LOW.value
    risk_codes: list[str] = field(default_factory=list)
    decision_recommendation: str = BrainRehearsalDecisionRecommendation.WAIT.value
    supporting_prediction_ids: list[str] = field(default_factory=list)
    supporting_event_ids: list[str] = field(default_factory=list)
    evaluations: list[BrainCounterfactualEvaluationRecord] = field(default_factory=list)
    selected_evaluation_id: str | None = None
    packet_digest: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    summary: str = ""
    completed_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the rehearsal result."""
        return {
            "rehearsal_id": self.rehearsal_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "step_index": self.step_index,
            "candidate_action_id": self.candidate_action_id,
            "fallback_action_ids": list(self.fallback_action_ids),
            "rehearsal_kind": self.rehearsal_kind,
            "simulated_backend": self.simulated_backend,
            "expected_preconditions": list(self.expected_preconditions),
            "expected_effects": list(self.expected_effects),
            "predicted_success_probability": self.predicted_success_probability,
            "confidence_band": self.confidence_band,
            "risk_codes": list(self.risk_codes),
            "decision_recommendation": self.decision_recommendation,
            "supporting_prediction_ids": list(self.supporting_prediction_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "evaluations": [record.as_dict() for record in self.evaluations],
            "selected_evaluation_id": self.selected_evaluation_id,
            "packet_digest": dict(self.packet_digest),
            "skipped": self.skipped,
            "summary": self.summary,
            "completed_at": self.completed_at,
            "details": dict(self.details),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainActionRehearsalResult | None":
        """Hydrate one rehearsal result from stored JSON."""
        if not isinstance(data, dict):
            return None
        rehearsal_id = str(data.get("rehearsal_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        plan_proposal_id = str(data.get("plan_proposal_id", "")).strip()
        candidate_action_id = str(data.get("candidate_action_id", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not rehearsal_id or not goal_id or not plan_proposal_id or not candidate_action_id or not summary:
            return None
        probability = max(0.0, min(float(data.get("predicted_success_probability", 0.0)), 1.0))
        return cls(
            rehearsal_id=rehearsal_id,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=plan_proposal_id,
            step_index=int(data.get("step_index", 0)),
            candidate_action_id=candidate_action_id,
            fallback_action_ids=_sorted_unique_texts(data.get("fallback_action_ids", [])),
            rehearsal_kind=(
                str(
                    data.get(
                        "rehearsal_kind",
                        BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
                    )
                ).strip()
                or BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value
            ),
            simulated_backend=(
                str(data.get("simulated_backend", "robot_head_simulation")).strip()
                or "robot_head_simulation"
            ),
            expected_preconditions=_sorted_unique_texts(data.get("expected_preconditions", [])),
            expected_effects=_sorted_unique_texts(data.get("expected_effects", [])),
            predicted_success_probability=probability,
            confidence_band=(
                str(data.get("confidence_band", BrainPredictionConfidenceBand.LOW.value)).strip()
                or BrainPredictionConfidenceBand.LOW.value
            ),
            risk_codes=_sorted_unique_texts(data.get("risk_codes", [])),
            decision_recommendation=(
                str(
                    data.get(
                        "decision_recommendation",
                        BrainRehearsalDecisionRecommendation.WAIT.value,
                    )
                ).strip()
                or BrainRehearsalDecisionRecommendation.WAIT.value
            ),
            supporting_prediction_ids=_sorted_unique_texts(
                data.get("supporting_prediction_ids", [])
            ),
            supporting_event_ids=_sorted_unique_texts(data.get("supporting_event_ids", [])),
            evaluations=[
                record
                for item in data.get("evaluations", [])
                if (record := BrainCounterfactualEvaluationRecord.from_dict(item)) is not None
            ],
            selected_evaluation_id=_optional_text(data.get("selected_evaluation_id")),
            packet_digest=dict(data.get("packet_digest", {})),
            skipped=bool(data.get("skipped", False)),
            summary=summary,
            completed_at=str(data.get("completed_at") or _utc_now()),
            details=dict(data.get("details", {})),
            updated_at=str(
                data.get("updated_at") or data.get("completed_at") or _utc_now()
            ),
        )


@dataclass(frozen=True)
class BrainActionOutcomeComparisonRecord:
    """One persisted comparison between rehearsed and observed action outcomes."""

    comparison_id: str
    rehearsal_id: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str | None
    step_index: int
    candidate_action_id: str
    observed_outcome_kind: str
    predicted_success_probability: float
    confidence_band: str
    decision_recommendation: str
    calibration_bucket: str
    comparison_summary: str
    observed_event_id: str | None = None
    supporting_event_ids: list[str] = field(default_factory=list)
    risk_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    compared_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the outcome comparison."""
        return {
            "comparison_id": self.comparison_id,
            "rehearsal_id": self.rehearsal_id,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "step_index": self.step_index,
            "candidate_action_id": self.candidate_action_id,
            "observed_outcome_kind": self.observed_outcome_kind,
            "predicted_success_probability": self.predicted_success_probability,
            "confidence_band": self.confidence_band,
            "decision_recommendation": self.decision_recommendation,
            "calibration_bucket": self.calibration_bucket,
            "comparison_summary": self.comparison_summary,
            "observed_event_id": self.observed_event_id,
            "supporting_event_ids": list(self.supporting_event_ids),
            "risk_codes": list(self.risk_codes),
            "details": dict(self.details),
            "compared_at": self.compared_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainActionOutcomeComparisonRecord | None":
        """Hydrate one outcome comparison from stored JSON."""
        if not isinstance(data, dict):
            return None
        comparison_id = str(data.get("comparison_id", "")).strip()
        rehearsal_id = str(data.get("rehearsal_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        candidate_action_id = str(data.get("candidate_action_id", "")).strip()
        observed_outcome_kind = str(data.get("observed_outcome_kind", "")).strip()
        confidence_band = str(data.get("confidence_band", "")).strip()
        decision_recommendation = str(data.get("decision_recommendation", "")).strip()
        calibration_bucket = str(data.get("calibration_bucket", "")).strip()
        comparison_summary = str(data.get("comparison_summary", "")).strip()
        if (
            not comparison_id
            or not rehearsal_id
            or not goal_id
            or not candidate_action_id
            or not observed_outcome_kind
            or not confidence_band
            or not decision_recommendation
            or not calibration_bucket
            or not comparison_summary
        ):
            return None
        return cls(
            comparison_id=comparison_id,
            rehearsal_id=rehearsal_id,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            step_index=int(data.get("step_index", 0)),
            candidate_action_id=candidate_action_id,
            observed_outcome_kind=observed_outcome_kind,
            predicted_success_probability=max(
                0.0,
                min(float(data.get("predicted_success_probability", 0.0)), 1.0),
            ),
            confidence_band=confidence_band,
            decision_recommendation=decision_recommendation,
            calibration_bucket=calibration_bucket,
            comparison_summary=comparison_summary,
            observed_event_id=_optional_text(data.get("observed_event_id")),
            supporting_event_ids=_sorted_unique_texts(data.get("supporting_event_ids", [])),
            risk_codes=_sorted_unique_texts(data.get("risk_codes", [])),
            details=dict(data.get("details", {})),
            compared_at=str(data.get("compared_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("compared_at") or _utc_now()),
        )


@dataclass(frozen=True)
class BrainCounterfactualCalibrationSummary:
    """Aggregate bounded calibration counters for rehearsed action outcomes."""

    requested_count: int = 0
    completed_count: int = 0
    skipped_count: int = 0
    comparison_count: int = 0
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    risk_code_counts: dict[str, int] = field(default_factory=dict)
    observed_outcome_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the calibration summary."""
        return {
            "requested_count": self.requested_count,
            "completed_count": self.completed_count,
            "skipped_count": self.skipped_count,
            "comparison_count": self.comparison_count,
            "recommendation_counts": dict(self.recommendation_counts),
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "risk_code_counts": dict(self.risk_code_counts),
            "observed_outcome_counts": dict(self.observed_outcome_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCounterfactualCalibrationSummary":
        """Hydrate the calibration summary from stored JSON."""
        payload = data or {}
        return cls(
            requested_count=int(payload.get("requested_count", 0)),
            completed_count=int(payload.get("completed_count", 0)),
            skipped_count=int(payload.get("skipped_count", 0)),
            comparison_count=int(payload.get("comparison_count", 0)),
            recommendation_counts={
                str(key): int(value)
                for key, value in dict(payload.get("recommendation_counts", {})).items()
            },
            calibration_bucket_counts={
                str(key): int(value)
                for key, value in dict(payload.get("calibration_bucket_counts", {})).items()
            },
            risk_code_counts={
                str(key): int(value)
                for key, value in dict(payload.get("risk_code_counts", {})).items()
            },
            observed_outcome_counts={
                str(key): int(value)
                for key, value in dict(payload.get("observed_outcome_counts", {})).items()
            },
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainCounterfactualRehearsalProjection:
    """Replay-safe thread-scoped rehearsal and calibration projection."""

    scope_key: str
    presence_scope_key: str
    open_requests: list[BrainActionRehearsalRequest] = field(default_factory=list)
    recent_rehearsals: list[BrainActionRehearsalResult] = field(default_factory=list)
    recent_comparisons: list[BrainActionOutcomeComparisonRecord] = field(default_factory=list)
    calibration_summary: BrainCounterfactualCalibrationSummary = field(
        default_factory=BrainCounterfactualCalibrationSummary
    )
    open_rehearsal_ids: list[str] = field(default_factory=list)
    recent_rehearsal_ids: list[str] = field(default_factory=list)
    recent_comparison_ids: list[str] = field(default_factory=list)
    recommendation_counts: dict[str, int] = field(default_factory=dict)
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    observed_outcome_counts: dict[str, int] = field(default_factory=dict)
    risk_code_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived counts and id indexes from structured rehearsal state."""
        self.open_requests = sorted(self.open_requests, key=_rehearsal_request_sort_key)
        self.recent_rehearsals = sorted(self.recent_rehearsals, key=_rehearsal_result_sort_key)
        self.recent_comparisons = sorted(
            self.recent_comparisons,
            key=_rehearsal_comparison_sort_key,
        )
        recommendation_counts: dict[str, int] = {}
        calibration_bucket_counts: dict[str, int] = {}
        observed_outcome_counts: dict[str, int] = {}
        risk_code_counts: dict[str, int] = {}
        for record in self.recent_rehearsals:
            recommendation_counts[record.decision_recommendation] = (
                recommendation_counts.get(record.decision_recommendation, 0) + 1
            )
            for code in record.risk_codes:
                risk_code_counts[code] = risk_code_counts.get(code, 0) + 1
        for record in self.recent_comparisons:
            calibration_bucket_counts[record.calibration_bucket] = (
                calibration_bucket_counts.get(record.calibration_bucket, 0) + 1
            )
            observed_outcome_counts[record.observed_outcome_kind] = (
                observed_outcome_counts.get(record.observed_outcome_kind, 0) + 1
            )
            for code in record.risk_codes:
                risk_code_counts[code] = risk_code_counts.get(code, 0) + 1
        self.open_rehearsal_ids = [record.rehearsal_id for record in self.open_requests]
        self.recent_rehearsal_ids = [record.rehearsal_id for record in self.recent_rehearsals]
        self.recent_comparison_ids = [record.comparison_id for record in self.recent_comparisons]
        self.recommendation_counts = dict(sorted(recommendation_counts.items()))
        self.calibration_bucket_counts = dict(sorted(calibration_bucket_counts.items()))
        self.observed_outcome_counts = dict(sorted(observed_outcome_counts.items()))
        self.risk_code_counts = dict(sorted(risk_code_counts.items()))
        self.calibration_summary = BrainCounterfactualCalibrationSummary(
            requested_count=int(self.calibration_summary.requested_count),
            completed_count=int(self.calibration_summary.completed_count),
            skipped_count=int(self.calibration_summary.skipped_count),
            comparison_count=int(self.calibration_summary.comparison_count),
            recommendation_counts=dict(self.recommendation_counts),
            calibration_bucket_counts=dict(self.calibration_bucket_counts),
            risk_code_counts=dict(self.risk_code_counts),
            observed_outcome_counts=dict(self.observed_outcome_counts),
            updated_at=self.updated_at,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the rehearsal projection."""
        self.sync_lists()
        return {
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "open_requests": [record.as_dict() for record in self.open_requests],
            "recent_rehearsals": [record.as_dict() for record in self.recent_rehearsals],
            "recent_comparisons": [record.as_dict() for record in self.recent_comparisons],
            "calibration_summary": self.calibration_summary.as_dict(),
            "open_rehearsal_ids": list(self.open_rehearsal_ids),
            "recent_rehearsal_ids": list(self.recent_rehearsal_ids),
            "recent_comparison_ids": list(self.recent_comparison_ids),
            "recommendation_counts": dict(self.recommendation_counts),
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "observed_outcome_counts": dict(self.observed_outcome_counts),
            "risk_code_counts": dict(self.risk_code_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCounterfactualRehearsalProjection":
        """Hydrate the rehearsal projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_key=str(payload.get("scope_key", "")).strip(),
            presence_scope_key=str(payload.get("presence_scope_key", "")).strip()
            or "local:presence",
            open_requests=[
                record
                for item in payload.get("open_requests", [])
                if (record := BrainActionRehearsalRequest.from_dict(item)) is not None
            ],
            recent_rehearsals=[
                record
                for item in payload.get("recent_rehearsals", [])
                if (record := BrainActionRehearsalResult.from_dict(item)) is not None
            ],
            recent_comparisons=[
                record
                for item in payload.get("recent_comparisons", [])
                if (record := BrainActionOutcomeComparisonRecord.from_dict(item)) is not None
            ],
            calibration_summary=BrainCounterfactualCalibrationSummary.from_dict(
                payload.get("calibration_summary")
            ),
            open_rehearsal_ids=_sorted_unique_texts(payload.get("open_rehearsal_ids", [])),
            recent_rehearsal_ids=_sorted_unique_texts(payload.get("recent_rehearsal_ids", [])),
            recent_comparison_ids=_sorted_unique_texts(payload.get("recent_comparison_ids", [])),
            recommendation_counts={
                str(key): int(value)
                for key, value in dict(payload.get("recommendation_counts", {})).items()
            },
            calibration_bucket_counts={
                str(key): int(value)
                for key, value in dict(payload.get("calibration_bucket_counts", {})).items()
            },
            observed_outcome_counts={
                str(key): int(value)
                for key, value in dict(payload.get("observed_outcome_counts", {})).items()
            },
            risk_code_counts={
                str(key): int(value)
                for key, value in dict(payload.get("risk_code_counts", {})).items()
            },
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


def _embodied_intent_sort_key(record: "BrainEmbodiedIntent") -> tuple[str, str]:
    return (record.updated_at, record.intent_id)


def _embodied_envelope_sort_key(record: "BrainEmbodiedActionEnvelope") -> tuple[str, str]:
    return (record.updated_at, record.envelope_id)


def _embodied_trace_sort_key(record: "BrainEmbodiedExecutionTrace") -> tuple[str, str]:
    return (record.updated_at, record.trace_id)


def _embodied_recovery_sort_key(record: "BrainEmbodiedRecoveryRecord") -> tuple[str, str]:
    return (record.updated_at, record.recovery_id)


@dataclass(frozen=True)
class BrainEmbodiedIntent:
    """One typed high-level embodied intent selected by the coordinator."""

    intent_id: str
    intent_kind: str
    goal_id: str
    commitment_id: str | None
    plan_proposal_id: str | None
    step_index: int
    selected_action_id: str
    executor_kind: str
    policy_posture: str
    supporting_prediction_ids: list[str] = field(default_factory=list)
    supporting_rehearsal_id: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    status: str = "selected"
    summary: str = ""
    selected_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the embodied intent."""
        return {
            "intent_id": self.intent_id,
            "intent_kind": self.intent_kind,
            "goal_id": self.goal_id,
            "commitment_id": self.commitment_id,
            "plan_proposal_id": self.plan_proposal_id,
            "step_index": self.step_index,
            "selected_action_id": self.selected_action_id,
            "executor_kind": self.executor_kind,
            "policy_posture": self.policy_posture,
            "supporting_prediction_ids": list(self.supporting_prediction_ids),
            "supporting_rehearsal_id": self.supporting_rehearsal_id,
            "reason_codes": list(self.reason_codes),
            "status": self.status,
            "summary": self.summary,
            "selected_at": self.selected_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEmbodiedIntent | None":
        """Hydrate one embodied intent from stored JSON."""
        if not isinstance(data, dict):
            return None
        intent_id = str(data.get("intent_id", "")).strip()
        intent_kind = str(data.get("intent_kind", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        selected_action_id = str(data.get("selected_action_id", "")).strip()
        executor_kind = str(data.get("executor_kind", "")).strip()
        policy_posture = str(data.get("policy_posture", "")).strip()
        if (
            not intent_id
            or not intent_kind
            or not goal_id
            or not selected_action_id
            or not executor_kind
            or not policy_posture
        ):
            return None
        return cls(
            intent_id=intent_id,
            intent_kind=intent_kind,
            goal_id=goal_id,
            commitment_id=_optional_text(data.get("commitment_id")),
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            step_index=int(data.get("step_index", 0)),
            selected_action_id=selected_action_id,
            executor_kind=executor_kind,
            policy_posture=policy_posture,
            supporting_prediction_ids=_sorted_unique_texts(
                data.get("supporting_prediction_ids", [])
            ),
            supporting_rehearsal_id=_optional_text(data.get("supporting_rehearsal_id")),
            reason_codes=_sorted_unique_texts(data.get("reason_codes", [])),
            status=str(data.get("status", "selected") or "selected"),
            summary=str(data.get("summary", "") or ""),
            selected_at=str(data.get("selected_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("selected_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainEmbodiedActionEnvelope:
    """One low-level action envelope bound to a high-level embodied intent."""

    envelope_id: str
    intent_id: str
    goal_id: str
    plan_proposal_id: str | None
    step_index: int
    capability_id: str
    action_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    dispatch_source: str = ""
    executor_backend: str = ""
    preview_only: bool = False
    rehearsal_id: str | None = None
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    summary: str = ""
    prepared_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the embodied action envelope."""
        return {
            "envelope_id": self.envelope_id,
            "intent_id": self.intent_id,
            "goal_id": self.goal_id,
            "plan_proposal_id": self.plan_proposal_id,
            "step_index": self.step_index,
            "capability_id": self.capability_id,
            "action_id": self.action_id,
            "arguments": dict(self.arguments),
            "dispatch_source": self.dispatch_source,
            "executor_backend": self.executor_backend,
            "preview_only": self.preview_only,
            "rehearsal_id": self.rehearsal_id,
            "policy_snapshot": dict(self.policy_snapshot),
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "prepared_at": self.prepared_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainEmbodiedActionEnvelope | None":
        """Hydrate one embodied action envelope from stored JSON."""
        if not isinstance(data, dict):
            return None
        envelope_id = str(data.get("envelope_id", "")).strip()
        intent_id = str(data.get("intent_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        capability_id = str(data.get("capability_id", "")).strip()
        action_id = str(data.get("action_id", "")).strip()
        dispatch_source = str(data.get("dispatch_source", "")).strip()
        executor_backend = str(data.get("executor_backend", "")).strip()
        if (
            not envelope_id
            or not intent_id
            or not goal_id
            or not capability_id
            or not action_id
            or not dispatch_source
            or not executor_backend
        ):
            return None
        return cls(
            envelope_id=envelope_id,
            intent_id=intent_id,
            goal_id=goal_id,
            plan_proposal_id=_optional_text(data.get("plan_proposal_id")),
            step_index=int(data.get("step_index", 0)),
            capability_id=capability_id,
            action_id=action_id,
            arguments=dict(data.get("arguments", {})),
            dispatch_source=dispatch_source,
            executor_backend=executor_backend,
            preview_only=bool(data.get("preview_only", False)),
            rehearsal_id=_optional_text(data.get("rehearsal_id")),
            policy_snapshot=dict(data.get("policy_snapshot", {})),
            reason_codes=_sorted_unique_texts(data.get("reason_codes", [])),
            summary=str(data.get("summary", "") or ""),
            prepared_at=str(data.get("prepared_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("prepared_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainEmbodiedExecutionTrace:
    """One replay-safe embodied execution trace."""

    trace_id: str
    intent_id: str
    envelope_id: str
    goal_id: str
    step_index: int
    capability_request_event_id: str | None = None
    capability_result_event_id: str | None = None
    robot_action_event_id: str | None = None
    disposition: str = BrainEmbodiedDispatchDisposition.DISPATCH.value
    status: str = BrainEmbodiedTraceStatus.PREPARED.value
    outcome_summary: str = ""
    mismatch_codes: list[str] = field(default_factory=list)
    repair_codes: list[str] = field(default_factory=list)
    recovery_action_id: str | None = None
    prepared_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the embodied execution trace."""
        return {
            "trace_id": self.trace_id,
            "intent_id": self.intent_id,
            "envelope_id": self.envelope_id,
            "goal_id": self.goal_id,
            "step_index": self.step_index,
            "capability_request_event_id": self.capability_request_event_id,
            "capability_result_event_id": self.capability_result_event_id,
            "robot_action_event_id": self.robot_action_event_id,
            "disposition": self.disposition,
            "status": self.status,
            "outcome_summary": self.outcome_summary,
            "mismatch_codes": list(self.mismatch_codes),
            "repair_codes": list(self.repair_codes),
            "recovery_action_id": self.recovery_action_id,
            "prepared_at": self.prepared_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainEmbodiedExecutionTrace | None":
        """Hydrate one embodied execution trace from stored JSON."""
        if not isinstance(data, dict):
            return None
        trace_id = str(data.get("trace_id", "")).strip()
        intent_id = str(data.get("intent_id", "")).strip()
        envelope_id = str(data.get("envelope_id", "")).strip()
        goal_id = str(data.get("goal_id", "")).strip()
        if not trace_id or not intent_id or not envelope_id or not goal_id:
            return None
        return cls(
            trace_id=trace_id,
            intent_id=intent_id,
            envelope_id=envelope_id,
            goal_id=goal_id,
            step_index=int(data.get("step_index", 0)),
            capability_request_event_id=_optional_text(data.get("capability_request_event_id")),
            capability_result_event_id=_optional_text(data.get("capability_result_event_id")),
            robot_action_event_id=_optional_text(data.get("robot_action_event_id")),
            disposition=(
                str(
                    data.get(
                        "disposition",
                        BrainEmbodiedDispatchDisposition.DISPATCH.value,
                    )
                ).strip()
                or BrainEmbodiedDispatchDisposition.DISPATCH.value
            ),
            status=(
                str(
                    data.get(
                        "status",
                        BrainEmbodiedTraceStatus.PREPARED.value,
                    )
                ).strip()
                or BrainEmbodiedTraceStatus.PREPARED.value
            ),
            outcome_summary=str(data.get("outcome_summary", "") or ""),
            mismatch_codes=_sorted_unique_texts(data.get("mismatch_codes", [])),
            repair_codes=_sorted_unique_texts(data.get("repair_codes", [])),
            recovery_action_id=_optional_text(data.get("recovery_action_id")),
            prepared_at=str(data.get("prepared_at") or _utc_now()),
            completed_at=_optional_text(data.get("completed_at")),
            updated_at=str(data.get("updated_at") or data.get("completed_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainEmbodiedRecoveryRecord:
    """One bounded embodied recovery record attached to a failed trace."""

    recovery_id: str
    trace_id: str
    intent_id: str
    action_id: str
    reason_codes: list[str] = field(default_factory=list)
    status: str = "recommended"
    summary: str = ""
    recorded_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the embodied recovery record."""
        return {
            "recovery_id": self.recovery_id,
            "trace_id": self.trace_id,
            "intent_id": self.intent_id,
            "action_id": self.action_id,
            "reason_codes": list(self.reason_codes),
            "status": self.status,
            "summary": self.summary,
            "recorded_at": self.recorded_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "BrainEmbodiedRecoveryRecord | None":
        """Hydrate one embodied recovery record from stored JSON."""
        if not isinstance(data, dict):
            return None
        recovery_id = str(data.get("recovery_id", "")).strip()
        trace_id = str(data.get("trace_id", "")).strip()
        intent_id = str(data.get("intent_id", "")).strip()
        action_id = str(data.get("action_id", "")).strip()
        if not recovery_id or not trace_id or not intent_id or not action_id:
            return None
        return cls(
            recovery_id=recovery_id,
            trace_id=trace_id,
            intent_id=intent_id,
            action_id=action_id,
            reason_codes=_sorted_unique_texts(data.get("reason_codes", [])),
            status=str(data.get("status", "recommended") or "recommended"),
            summary=str(data.get("summary", "") or ""),
            recorded_at=str(data.get("recorded_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("recorded_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainEmbodiedExecutiveProjection:
    """Replay-safe embodied coordinator projection for one thread."""

    scope_key: str
    presence_scope_key: str
    current_intent: BrainEmbodiedIntent | None = None
    recent_action_envelopes: list[BrainEmbodiedActionEnvelope] = field(default_factory=list)
    recent_execution_traces: list[BrainEmbodiedExecutionTrace] = field(default_factory=list)
    recent_recoveries: list[BrainEmbodiedRecoveryRecord] = field(default_factory=list)
    current_executor_kind: str | None = None
    recent_envelope_ids: list[str] = field(default_factory=list)
    recent_trace_ids: list[str] = field(default_factory=list)
    recent_recovery_ids: list[str] = field(default_factory=list)
    intent_kind_counts: dict[str, int] = field(default_factory=dict)
    disposition_counts: dict[str, int] = field(default_factory=dict)
    trace_status_counts: dict[str, int] = field(default_factory=dict)
    mismatch_code_counts: dict[str, int] = field(default_factory=dict)
    repair_code_counts: dict[str, int] = field(default_factory=dict)
    policy_posture_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived embodied indexes and counts."""
        self.recent_action_envelopes = sorted(
            self.recent_action_envelopes,
            key=_embodied_envelope_sort_key,
            reverse=True,
        )
        self.recent_execution_traces = sorted(
            self.recent_execution_traces,
            key=_embodied_trace_sort_key,
            reverse=True,
        )
        self.recent_recoveries = sorted(
            self.recent_recoveries,
            key=_embodied_recovery_sort_key,
            reverse=True,
        )
        intent_kind_counts: dict[str, int] = {}
        policy_posture_counts: dict[str, int] = {}
        disposition_counts: dict[str, int] = {}
        trace_status_counts: dict[str, int] = {}
        mismatch_code_counts: dict[str, int] = {}
        repair_code_counts: dict[str, int] = {}
        if self.current_intent is not None:
            intent_kind_counts[self.current_intent.intent_kind] = 1
            policy_posture_counts[self.current_intent.policy_posture] = 1
            self.current_executor_kind = self.current_intent.executor_kind
        for record in self.recent_execution_traces:
            disposition_counts[record.disposition] = disposition_counts.get(record.disposition, 0) + 1
            trace_status_counts[record.status] = trace_status_counts.get(record.status, 0) + 1
            for code in record.mismatch_codes:
                mismatch_code_counts[code] = mismatch_code_counts.get(code, 0) + 1
            for code in record.repair_codes:
                repair_code_counts[code] = repair_code_counts.get(code, 0) + 1
        for record in self.recent_recoveries:
            for code in record.reason_codes:
                repair_code_counts[code] = repair_code_counts.get(code, 0) + 1
        self.recent_envelope_ids = [record.envelope_id for record in self.recent_action_envelopes]
        self.recent_trace_ids = [record.trace_id for record in self.recent_execution_traces]
        self.recent_recovery_ids = [record.recovery_id for record in self.recent_recoveries]
        self.intent_kind_counts = dict(sorted(intent_kind_counts.items()))
        self.disposition_counts = dict(sorted(disposition_counts.items()))
        self.trace_status_counts = dict(sorted(trace_status_counts.items()))
        self.mismatch_code_counts = dict(sorted(mismatch_code_counts.items()))
        self.repair_code_counts = dict(sorted(repair_code_counts.items()))
        self.policy_posture_counts = dict(sorted(policy_posture_counts.items()))

    def as_dict(self) -> dict[str, Any]:
        """Serialize the embodied executive projection."""
        self.sync_lists()
        return {
            "scope_key": self.scope_key,
            "presence_scope_key": self.presence_scope_key,
            "current_intent": (
                self.current_intent.as_dict() if self.current_intent is not None else None
            ),
            "recent_action_envelopes": [
                record.as_dict() for record in self.recent_action_envelopes
            ],
            "recent_execution_traces": [
                record.as_dict() for record in self.recent_execution_traces
            ],
            "recent_recoveries": [record.as_dict() for record in self.recent_recoveries],
            "current_executor_kind": self.current_executor_kind,
            "recent_envelope_ids": list(self.recent_envelope_ids),
            "recent_trace_ids": list(self.recent_trace_ids),
            "recent_recovery_ids": list(self.recent_recovery_ids),
            "intent_kind_counts": dict(self.intent_kind_counts),
            "disposition_counts": dict(self.disposition_counts),
            "trace_status_counts": dict(self.trace_status_counts),
            "mismatch_code_counts": dict(self.mismatch_code_counts),
            "repair_code_counts": dict(self.repair_code_counts),
            "policy_posture_counts": dict(self.policy_posture_counts),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEmbodiedExecutiveProjection":
        """Hydrate the embodied executive projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_key=str(payload.get("scope_key", "")).strip(),
            presence_scope_key=str(payload.get("presence_scope_key", "")).strip()
            or "local:presence",
            current_intent=BrainEmbodiedIntent.from_dict(payload.get("current_intent")),
            recent_action_envelopes=[
                record
                for item in payload.get("recent_action_envelopes", [])
                if (record := BrainEmbodiedActionEnvelope.from_dict(item)) is not None
            ],
            recent_execution_traces=[
                record
                for item in payload.get("recent_execution_traces", [])
                if (record := BrainEmbodiedExecutionTrace.from_dict(item)) is not None
            ],
            recent_recoveries=[
                record
                for item in payload.get("recent_recoveries", [])
                if (record := BrainEmbodiedRecoveryRecord.from_dict(item)) is not None
            ],
            current_executor_kind=_optional_text(payload.get("current_executor_kind")),
            recent_envelope_ids=_sorted_unique_texts(payload.get("recent_envelope_ids", [])),
            recent_trace_ids=_sorted_unique_texts(payload.get("recent_trace_ids", [])),
            recent_recovery_ids=_sorted_unique_texts(payload.get("recent_recovery_ids", [])),
            intent_kind_counts={
                str(key): int(value)
                for key, value in dict(payload.get("intent_kind_counts", {})).items()
            },
            disposition_counts={
                str(key): int(value)
                for key, value in dict(payload.get("disposition_counts", {})).items()
            },
            trace_status_counts={
                str(key): int(value)
                for key, value in dict(payload.get("trace_status_counts", {})).items()
            },
            mismatch_code_counts={
                str(key): int(value)
                for key, value in dict(payload.get("mismatch_code_counts", {})).items()
            },
            repair_code_counts={
                str(key): int(value)
                for key, value in dict(payload.get("repair_code_counts", {})).items()
            },
            policy_posture_counts={
                str(key): int(value)
                for key, value in dict(payload.get("policy_posture_counts", {})).items()
            },
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


@dataclass(frozen=True)
class BrainSceneWorldEntityRecord:
    """One symbolic scene-world entity record."""

    entity_id: str
    entity_kind: str
    canonical_label: str
    summary: str
    state: str
    evidence_kind: str
    zone_id: str | None = None
    confidence: float | None = None
    freshness: str | None = None
    contradiction_codes: list[str] = field(default_factory=list)
    affordance_ids: list[str] = field(default_factory=list)
    backing_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    observed_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    expires_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "entity_id": self.entity_id,
            "entity_kind": self.entity_kind,
            "canonical_label": self.canonical_label,
            "summary": self.summary,
            "state": self.state,
            "evidence_kind": self.evidence_kind,
            "zone_id": self.zone_id,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "contradiction_codes": list(self.contradiction_codes),
            "affordance_ids": list(self.affordance_ids),
            "backing_ids": list(self.backing_ids),
            "source_event_ids": list(self.source_event_ids),
            "observed_at": self.observed_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSceneWorldEntityRecord | None":
        """Hydrate one record from stored JSON."""
        if not isinstance(data, dict):
            return None
        entity_id = str(data.get("entity_id", "")).strip()
        entity_kind = str(data.get("entity_kind", "")).strip()
        canonical_label = str(data.get("canonical_label", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not entity_id or not entity_kind or not canonical_label or not summary:
            return None
        return cls(
            entity_id=entity_id,
            entity_kind=entity_kind,
            canonical_label=canonical_label,
            summary=summary,
            state=str(data.get("state", BrainSceneWorldRecordState.ACTIVE.value)).strip()
            or BrainSceneWorldRecordState.ACTIVE.value,
            evidence_kind=str(
                data.get("evidence_kind", BrainSceneWorldEvidenceKind.OBSERVED.value)
            ).strip()
            or BrainSceneWorldEvidenceKind.OBSERVED.value,
            zone_id=_optional_text(data.get("zone_id")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            freshness=_optional_text(data.get("freshness")),
            contradiction_codes=sorted(
                {
                    str(item).strip()
                    for item in data.get("contradiction_codes", [])
                    if str(item).strip()
                }
            ),
            affordance_ids=sorted(
                {str(item).strip() for item in data.get("affordance_ids", []) if str(item).strip()}
            ),
            backing_ids=sorted(
                {str(item).strip() for item in data.get("backing_ids", []) if str(item).strip()}
            ),
            source_event_ids=sorted(
                {
                    str(item).strip()
                    for item in data.get("source_event_ids", [])
                    if str(item).strip()
                }
            ),
            observed_at=_optional_text(data.get("observed_at")),
            updated_at=str(data.get("updated_at") or _utc_now()),
            expires_at=_optional_text(data.get("expires_at")),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainSceneWorldAffordanceRecord:
    """One scene-linked affordance record."""

    affordance_id: str
    entity_id: str
    capability_family: str
    summary: str
    availability: str
    confidence: float | None = None
    freshness: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    backing_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    observed_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    expires_at: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "affordance_id": self.affordance_id,
            "entity_id": self.entity_id,
            "capability_family": self.capability_family,
            "summary": self.summary,
            "availability": self.availability,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "reason_codes": list(self.reason_codes),
            "backing_ids": list(self.backing_ids),
            "source_event_ids": list(self.source_event_ids),
            "observed_at": self.observed_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSceneWorldAffordanceRecord | None":
        """Hydrate one affordance from stored JSON."""
        if not isinstance(data, dict):
            return None
        affordance_id = str(data.get("affordance_id", "")).strip()
        entity_id = str(data.get("entity_id", "")).strip()
        capability_family = str(data.get("capability_family", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not affordance_id or not entity_id or not capability_family or not summary:
            return None
        return cls(
            affordance_id=affordance_id,
            entity_id=entity_id,
            capability_family=capability_family,
            summary=summary,
            availability=str(
                data.get(
                    "availability",
                    BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                )
            ).strip()
            or BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            freshness=_optional_text(data.get("freshness")),
            reason_codes=sorted(
                {str(item).strip() for item in data.get("reason_codes", []) if str(item).strip()}
            ),
            backing_ids=sorted(
                {str(item).strip() for item in data.get("backing_ids", []) if str(item).strip()}
            ),
            source_event_ids=sorted(
                {
                    str(item).strip()
                    for item in data.get("source_event_ids", [])
                    if str(item).strip()
                }
            ),
            observed_at=_optional_text(data.get("observed_at")),
            updated_at=str(data.get("updated_at") or _utc_now()),
            expires_at=_optional_text(data.get("expires_at")),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainSceneWorldProjection:
    """Replay-safe symbolic scene-world projection."""

    scope_type: str
    scope_id: str
    entities: list[BrainSceneWorldEntityRecord] = field(default_factory=list)
    affordances: list[BrainSceneWorldAffordanceRecord] = field(default_factory=list)
    entity_counts: dict[str, int] = field(default_factory=dict)
    affordance_counts: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)
    contradiction_counts: dict[str, int] = field(default_factory=dict)
    active_entity_ids: list[str] = field(default_factory=list)
    stale_entity_ids: list[str] = field(default_factory=list)
    contradicted_entity_ids: list[str] = field(default_factory=list)
    expired_entity_ids: list[str] = field(default_factory=list)
    active_affordance_ids: list[str] = field(default_factory=list)
    blocked_affordance_ids: list[str] = field(default_factory=list)
    uncertain_affordance_ids: list[str] = field(default_factory=list)
    degraded_mode: str = "healthy"
    degraded_reason_codes: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def sync_lists(self):
        """Refresh derived counts and id indexes from structured records."""
        self.entities = sorted(self.entities, key=_scene_world_entity_sort_key)
        self.affordances = sorted(self.affordances, key=_scene_world_affordance_sort_key)
        entity_counts: dict[str, int] = {}
        affordance_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        contradiction_counts: dict[str, int] = {}
        active_entity_ids: list[str] = []
        stale_entity_ids: list[str] = []
        contradicted_entity_ids: list[str] = []
        expired_entity_ids: list[str] = []
        active_affordance_ids: list[str] = []
        blocked_affordance_ids: list[str] = []
        uncertain_affordance_ids: list[str] = []
        for record in self.entities:
            entity_counts[record.entity_kind] = entity_counts.get(record.entity_kind, 0) + 1
            state_counts[record.state] = state_counts.get(record.state, 0) + 1
            for code in record.contradiction_codes:
                contradiction_counts[code] = contradiction_counts.get(code, 0) + 1
            if record.state == BrainSceneWorldRecordState.ACTIVE.value:
                active_entity_ids.append(record.entity_id)
            elif record.state == BrainSceneWorldRecordState.STALE.value:
                stale_entity_ids.append(record.entity_id)
            elif record.state == BrainSceneWorldRecordState.CONTRADICTED.value:
                contradicted_entity_ids.append(record.entity_id)
            elif record.state == BrainSceneWorldRecordState.EXPIRED.value:
                expired_entity_ids.append(record.entity_id)
        for record in self.affordances:
            affordance_counts[record.capability_family] = (
                affordance_counts.get(record.capability_family, 0) + 1
            )
            if record.availability == BrainSceneWorldAffordanceAvailability.AVAILABLE.value:
                active_affordance_ids.append(record.affordance_id)
            elif record.availability == BrainSceneWorldAffordanceAvailability.BLOCKED.value:
                blocked_affordance_ids.append(record.affordance_id)
            elif record.availability in {
                BrainSceneWorldAffordanceAvailability.UNCERTAIN.value,
                BrainSceneWorldAffordanceAvailability.STALE.value,
            }:
                uncertain_affordance_ids.append(record.affordance_id)
        self.entity_counts = dict(sorted(entity_counts.items()))
        self.affordance_counts = dict(sorted(affordance_counts.items()))
        self.state_counts = dict(sorted(state_counts.items()))
        self.contradiction_counts = dict(sorted(contradiction_counts.items()))
        self.active_entity_ids = active_entity_ids
        self.stale_entity_ids = stale_entity_ids
        self.contradicted_entity_ids = contradicted_entity_ids
        self.expired_entity_ids = expired_entity_ids
        self.active_affordance_ids = active_affordance_ids
        self.blocked_affordance_ids = blocked_affordance_ids
        self.uncertain_affordance_ids = uncertain_affordance_ids
        self.degraded_reason_codes = sorted(
            {str(item).strip() for item in self.degraded_reason_codes if str(item).strip()}
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        self.sync_lists()
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "entities": [record.as_dict() for record in self.entities],
            "affordances": [record.as_dict() for record in self.affordances],
            "entity_counts": dict(self.entity_counts),
            "affordance_counts": dict(self.affordance_counts),
            "state_counts": dict(self.state_counts),
            "contradiction_counts": dict(self.contradiction_counts),
            "active_entity_ids": list(self.active_entity_ids),
            "stale_entity_ids": list(self.stale_entity_ids),
            "contradicted_entity_ids": list(self.contradicted_entity_ids),
            "expired_entity_ids": list(self.expired_entity_ids),
            "active_affordance_ids": list(self.active_affordance_ids),
            "blocked_affordance_ids": list(self.blocked_affordance_ids),
            "uncertain_affordance_ids": list(self.uncertain_affordance_ids),
            "degraded_mode": self.degraded_mode,
            "degraded_reason_codes": list(self.degraded_reason_codes),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSceneWorldProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        projection = cls(
            scope_type=str(payload.get("scope_type", "presence")).strip() or "presence",
            scope_id=str(payload.get("scope_id", "")).strip(),
            entities=[
                record
                for item in payload.get("entities", [])
                if (record := BrainSceneWorldEntityRecord.from_dict(item)) is not None
            ],
            affordances=[
                record
                for item in payload.get("affordances", [])
                if (record := BrainSceneWorldAffordanceRecord.from_dict(item)) is not None
            ],
            entity_counts=dict(payload.get("entity_counts", {})),
            affordance_counts=dict(payload.get("affordance_counts", {})),
            state_counts=dict(payload.get("state_counts", {})),
            contradiction_counts=dict(payload.get("contradiction_counts", {})),
            active_entity_ids=[
                str(item).strip()
                for item in payload.get("active_entity_ids", [])
                if str(item).strip()
            ],
            stale_entity_ids=[
                str(item).strip()
                for item in payload.get("stale_entity_ids", [])
                if str(item).strip()
            ],
            contradicted_entity_ids=[
                str(item).strip()
                for item in payload.get("contradicted_entity_ids", [])
                if str(item).strip()
            ],
            expired_entity_ids=[
                str(item).strip()
                for item in payload.get("expired_entity_ids", [])
                if str(item).strip()
            ],
            active_affordance_ids=[
                str(item).strip()
                for item in payload.get("active_affordance_ids", [])
                if str(item).strip()
            ],
            blocked_affordance_ids=[
                str(item).strip()
                for item in payload.get("blocked_affordance_ids", [])
                if str(item).strip()
            ],
            uncertain_affordance_ids=[
                str(item).strip()
                for item in payload.get("uncertain_affordance_ids", [])
                if str(item).strip()
            ],
            degraded_mode=str(payload.get("degraded_mode", "healthy")).strip() or "healthy",
            degraded_reason_codes=[
                str(item).strip()
                for item in payload.get("degraded_reason_codes", [])
                if str(item).strip()
            ],
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        projection.sync_lists()
        return projection


@dataclass
class BrainAgendaProjection:
    """Initial agenda surface for seeded and open goals."""

    agenda_seed: str | None = None
    goals: list[BrainGoal] = field(default_factory=list)
    active_goal_id: str | None = None
    active_goal_family: str | None = None
    active_goal_summary: str | None = None
    open_goals: list[str] = field(default_factory=list)
    deferred_goals: list[str] = field(default_factory=list)
    blocked_goals: list[str] = field(default_factory=list)
    completed_goals: list[str] = field(default_factory=list)
    cancelled_goals: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "agenda_seed": self.agenda_seed,
            "goals": [goal.as_dict() for goal in self.goals],
            "active_goal_id": self.active_goal_id,
            "active_goal_family": self.active_goal_family,
            "active_goal_summary": self.active_goal_summary,
            "open_goals": list(self.open_goals),
            "deferred_goals": list(self.deferred_goals),
            "blocked_goals": list(self.blocked_goals),
            "completed_goals": list(self.completed_goals),
            "cancelled_goals": list(self.cancelled_goals),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAgendaProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        projection = cls(
            agenda_seed=payload.get("agenda_seed"),
            goals=[BrainGoal.from_dict(item) for item in payload.get("goals", [])],
            active_goal_id=payload.get("active_goal_id"),
            active_goal_family=payload.get("active_goal_family"),
            active_goal_summary=payload.get("active_goal_summary"),
            open_goals=list(payload.get("open_goals", [])),
            deferred_goals=list(payload.get("deferred_goals", [])),
            blocked_goals=list(payload.get("blocked_goals", [])),
            completed_goals=list(payload.get("completed_goals", [])),
            cancelled_goals=list(payload.get("cancelled_goals", [])),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
        if projection.goals:
            projection.sync_lists()
        return projection

    def goal(self, goal_id: str) -> BrainGoal | None:
        """Return one projected goal by id."""
        for goal in self.goals:
            if goal.goal_id == goal_id:
                return goal
        return None

    def sync_lists(self):
        """Refresh the summary goal lists from the structured goal records."""
        open_titles: list[str] = []
        deferred_titles: list[str] = []
        blocked_titles: list[str] = []
        completed_titles: list[str] = []
        cancelled_titles: list[str] = []
        active_goal_id: str | None = None
        active_goal_family: str | None = None
        active_goal_summary: str | None = None
        for goal in self.goals:
            commitment_status = str(goal.details.get("commitment_status", "")).strip()
            if goal.status in {
                BrainGoalStatus.OPEN.value,
                BrainGoalStatus.PLANNING.value,
                BrainGoalStatus.IN_PROGRESS.value,
                BrainGoalStatus.RETRY.value,
                BrainGoalStatus.WAITING.value,
            }:
                if (
                    goal.status == BrainGoalStatus.WAITING.value
                    and commitment_status == BrainCommitmentStatus.DEFERRED.value
                ):
                    if goal.title and goal.title not in deferred_titles:
                        deferred_titles.append(goal.title)
                elif goal.title and goal.title not in open_titles:
                    open_titles.append(goal.title)
                if active_goal_id is None and goal.status in {
                    BrainGoalStatus.OPEN.value,
                    BrainGoalStatus.PLANNING.value,
                    BrainGoalStatus.IN_PROGRESS.value,
                    BrainGoalStatus.RETRY.value,
                }:
                    active_goal_id = goal.goal_id
                    active_goal_family = goal.goal_family
                    active_goal_summary = (
                        f"{goal.goal_family}: {goal.title}" if goal.title else None
                    )
            elif goal.status in {
                BrainGoalStatus.BLOCKED.value,
                BrainGoalStatus.FAILED.value,
            }:
                if goal.title and goal.title not in blocked_titles:
                    blocked_titles.append(goal.title)
            elif goal.status == BrainGoalStatus.COMPLETED.value:
                if goal.title and goal.title not in completed_titles:
                    completed_titles.append(goal.title)
            elif goal.status == BrainGoalStatus.CANCELLED.value:
                if goal.title and goal.title not in cancelled_titles:
                    cancelled_titles.append(goal.title)
        self.active_goal_id = active_goal_id
        self.active_goal_family = active_goal_family
        self.active_goal_summary = active_goal_summary
        self.open_goals = open_titles
        self.deferred_goals = deferred_titles
        self.blocked_goals = blocked_titles
        self.completed_goals = completed_titles
        self.cancelled_goals = cancelled_titles


@dataclass
class BrainCommitmentRecord:
    """One durable executive-owned commitment row."""

    commitment_id: str
    scope_type: str
    scope_id: str
    title: str
    goal_family: str
    intent: str
    status: str = BrainCommitmentStatus.ACTIVE.value
    details: dict[str, Any] = field(default_factory=dict)
    current_goal_id: str | None = None
    blocked_reason: BrainBlockedReason | None = None
    wake_conditions: list[BrainWakeCondition] = field(default_factory=list)
    plan_revision: int = 1
    resume_count: int = 0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the commitment."""
        return {
            "commitment_id": self.commitment_id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "title": self.title,
            "goal_family": self.goal_family,
            "intent": self.intent,
            "status": self.status,
            "details": dict(self.details),
            "current_goal_id": self.current_goal_id,
            "blocked_reason": self.blocked_reason.as_dict() if self.blocked_reason else None,
            "wake_conditions": [item.as_dict() for item in self.wake_conditions],
            "plan_revision": self.plan_revision,
            "resume_count": self.resume_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCommitmentRecord":
        """Hydrate a commitment from stored JSON."""
        payload = data or {}
        return cls(
            commitment_id=str(payload.get("commitment_id", "")).strip(),
            scope_type=str(payload.get("scope_type", "")).strip(),
            scope_id=str(payload.get("scope_id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            goal_family=str(payload.get("goal_family", BrainGoalFamily.CONVERSATION.value)).strip()
            or BrainGoalFamily.CONVERSATION.value,
            intent=str(payload.get("intent", "")).strip(),
            status=str(payload.get("status", BrainCommitmentStatus.ACTIVE.value)).strip()
            or BrainCommitmentStatus.ACTIVE.value,
            details=dict(payload.get("details", {})),
            current_goal_id=_optional_text(payload.get("current_goal_id")),
            blocked_reason=BrainBlockedReason.from_dict(payload.get("blocked_reason")),
            wake_conditions=[
                item
                for item in (
                    BrainWakeCondition.from_dict(entry)
                    for entry in payload.get("wake_conditions", [])
                )
                if item is not None
            ],
            plan_revision=int(payload.get("plan_revision", 1)),
            resume_count=int(payload.get("resume_count", 0)),
            created_at=str(payload.get("created_at") or _utc_now()),
            updated_at=str(payload.get("updated_at") or _utc_now()),
            completed_at=str(payload.get("completed_at")) if payload.get("completed_at") else None,
        )


@dataclass
class BrainCommitmentProjection:
    """Durable commitment summary surface for the executive."""

    active_commitments: list[BrainCommitmentRecord] = field(default_factory=list)
    deferred_commitments: list[BrainCommitmentRecord] = field(default_factory=list)
    blocked_commitments: list[BrainCommitmentRecord] = field(default_factory=list)
    recent_terminal_commitments: list[BrainCommitmentRecord] = field(default_factory=list)
    current_active_summary: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "active_commitments": [record.as_dict() for record in self.active_commitments],
            "deferred_commitments": [record.as_dict() for record in self.deferred_commitments],
            "blocked_commitments": [record.as_dict() for record in self.blocked_commitments],
            "recent_terminal_commitments": [
                record.as_dict() for record in self.recent_terminal_commitments
            ],
            "current_active_summary": self.current_active_summary,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCommitmentProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        return cls(
            active_commitments=[
                BrainCommitmentRecord.from_dict(item)
                for item in payload.get("active_commitments", [])
            ],
            deferred_commitments=[
                BrainCommitmentRecord.from_dict(item)
                for item in payload.get("deferred_commitments", [])
            ],
            blocked_commitments=[
                BrainCommitmentRecord.from_dict(item)
                for item in payload.get("blocked_commitments", [])
            ],
            recent_terminal_commitments=[
                BrainCommitmentRecord.from_dict(item)
                for item in payload.get("recent_terminal_commitments", [])
            ],
            current_active_summary=payload.get("current_active_summary"),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainSceneStateProjection:
    """Minimal symbolic scene state derived from lightweight perception."""

    camera_connected: bool = False
    camera_track_state: str = "disconnected"
    person_present: str = "uncertain"
    scene_change_state: str = "unknown"
    last_visual_summary: str | None = None
    last_observed_at: str | None = None
    last_fresh_frame_at: str | None = None
    frame_age_ms: int | None = None
    detection_backend: str | None = None
    detection_confidence: float | None = None
    sensor_health_reason: str | None = None
    recovery_in_progress: bool = False
    recovery_attempts: int = 0
    enrichment_available: bool | None = None
    confidence: float | None = None
    source: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "camera_connected": self.camera_connected,
            "camera_track_state": self.camera_track_state,
            "person_present": self.person_present,
            "scene_change_state": self.scene_change_state,
            "last_visual_summary": self.last_visual_summary,
            "last_observed_at": self.last_observed_at,
            "last_fresh_frame_at": self.last_fresh_frame_at,
            "frame_age_ms": self.frame_age_ms,
            "detection_backend": self.detection_backend,
            "detection_confidence": self.detection_confidence,
            "sensor_health_reason": self.sensor_health_reason,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_attempts": self.recovery_attempts,
            "enrichment_available": self.enrichment_available,
            "confidence": self.confidence,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSceneStateProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        confidence = payload.get("confidence")
        return cls(
            camera_connected=bool(payload.get("camera_connected", False)),
            camera_track_state=str(payload.get("camera_track_state", "disconnected")),
            person_present=str(payload.get("person_present", "uncertain")),
            scene_change_state=str(payload.get("scene_change_state", "unknown")),
            last_visual_summary=payload.get("last_visual_summary"),
            last_observed_at=payload.get("last_observed_at"),
            last_fresh_frame_at=payload.get("last_fresh_frame_at"),
            frame_age_ms=int(payload["frame_age_ms"])
            if payload.get("frame_age_ms") is not None
            else None,
            detection_backend=payload.get("detection_backend"),
            detection_confidence=(
                float(payload["detection_confidence"])
                if payload.get("detection_confidence") is not None
                else None
            ),
            sensor_health_reason=payload.get("sensor_health_reason"),
            recovery_in_progress=bool(payload.get("recovery_in_progress", False)),
            recovery_attempts=int(payload.get("recovery_attempts", 0)),
            enrichment_available=payload.get("enrichment_available"),
            confidence=float(confidence) if confidence is not None else None,
            source=payload.get("source"),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainEngagementStateProjection:
    """Minimal symbolic engagement and attention state."""

    engagement_state: str = "unknown"
    attention_to_camera: str = "unknown"
    user_present: bool = False
    last_engaged_at: str | None = None
    source: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "engagement_state": self.engagement_state,
            "attention_to_camera": self.attention_to_camera,
            "user_present": self.user_present,
            "last_engaged_at": self.last_engaged_at,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainEngagementStateProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        return cls(
            engagement_state=str(payload.get("engagement_state", "unknown")),
            attention_to_camera=str(payload.get("attention_to_camera", "unknown")),
            user_present=bool(payload.get("user_present", False)),
            last_engaged_at=payload.get("last_engaged_at"),
            source=payload.get("source"),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainRelationshipStateProjection:
    """Runtime relationship continuity projected from perception and memory."""

    user_present: bool = False
    last_seen_at: str | None = None
    engagement_state: str = "unknown"
    attention_to_camera: str = "unknown"
    open_commitments: list[str] = field(default_factory=list)
    interaction_style_hints: list[str] = field(default_factory=list)
    collaboration_style: str | None = None
    boundaries: list[str] = field(default_factory=list)
    known_misfires: list[str] = field(default_factory=list)
    preferred_teaching_modes: list[str] = field(default_factory=list)
    analogy_domains: list[str] = field(default_factory=list)
    continuity_summary: str | None = None
    source: str | None = None
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "user_present": self.user_present,
            "last_seen_at": self.last_seen_at,
            "engagement_state": self.engagement_state,
            "attention_to_camera": self.attention_to_camera,
            "open_commitments": list(self.open_commitments),
            "interaction_style_hints": list(self.interaction_style_hints),
            "collaboration_style": self.collaboration_style,
            "boundaries": list(self.boundaries),
            "known_misfires": list(self.known_misfires),
            "preferred_teaching_modes": list(self.preferred_teaching_modes),
            "analogy_domains": list(self.analogy_domains),
            "continuity_summary": self.continuity_summary,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainRelationshipStateProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        return cls(
            user_present=bool(payload.get("user_present", False)),
            last_seen_at=payload.get("last_seen_at"),
            engagement_state=str(payload.get("engagement_state", "unknown")),
            attention_to_camera=str(payload.get("attention_to_camera", "unknown")),
            open_commitments=list(payload.get("open_commitments", [])),
            interaction_style_hints=list(payload.get("interaction_style_hints", [])),
            collaboration_style=payload.get("collaboration_style"),
            boundaries=list(payload.get("boundaries", [])),
            known_misfires=list(payload.get("known_misfires", [])),
            preferred_teaching_modes=list(payload.get("preferred_teaching_modes", [])),
            analogy_domains=list(payload.get("analogy_domains", [])),
            continuity_summary=payload.get("continuity_summary"),
            source=payload.get("source"),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )


@dataclass
class BrainHeartbeatProjection:
    """Small runtime heartbeat surface derived from recent events."""

    last_event_type: str | None = None
    last_event_at: str | None = None
    last_tool_name: str | None = None
    last_robot_action: str | None = None
    warnings: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the projection."""
        return {
            "last_event_type": self.last_event_type,
            "last_event_at": self.last_event_at,
            "last_tool_name": self.last_tool_name,
            "last_robot_action": self.last_robot_action,
            "warnings": list(self.warnings),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainHeartbeatProjection":
        """Hydrate the projection from stored JSON."""
        payload = data or {}
        return cls(
            last_event_type=payload.get("last_event_type"),
            last_event_at=payload.get("last_event_at"),
            last_tool_name=payload.get("last_tool_name"),
            last_robot_action=payload.get("last_robot_action"),
            warnings=list(payload.get("warnings", [])),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )

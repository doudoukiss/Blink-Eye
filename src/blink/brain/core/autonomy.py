"""Provider-free candidate-goal and autonomy-ledger types for Blink."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

MAX_AUTONOMY_LEDGER_ENTRIES = 32


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    """Normalize one optional stored text value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class BrainInitiativeClass(str, Enum):
    """Coarse initiative classes for candidate goals."""

    SILENT_POSTURE_ADJUSTMENT = "silent_posture_adjustment"
    SILENT_ATTENTION_SHIFT = "silent_attention_shift"
    INSPECT_ONLY = "inspect_only"
    SPEAK_BRIEFLY_IF_IDLE = "speak_briefly_if_idle"
    DEFER_UNTIL_USER_TURN = "defer_until_user_turn"
    MAINTENANCE_INTERNAL = "maintenance_internal"
    OPERATOR_VISIBLE_ONLY = "operator_visible_only"


class BrainCandidateGoalSource(str, Enum):
    """Typed sources for candidate-goal proposals."""

    PERCEPTION = "perception"
    TIMER = "timer"
    COMMITMENT = "commitment"
    MAINTENANCE = "maintenance"
    OPERATOR = "operator"
    RUNTIME = "runtime"


class BrainReevaluationConditionKind(str, Enum):
    """Typed reevaluation conditions for held candidate work."""

    USER_TURN_CLOSED = "user_turn_closed"
    ASSISTANT_TURN_CLOSED = "assistant_turn_closed"
    GOAL_FAMILY_AVAILABLE = "goal_family_available"
    THREAD_IDLE = "thread_idle"
    TIME_REACHED = "time_reached"
    PROJECTION_CHANGED = "projection_changed"
    STARTUP_RECOVERY = "startup_recovery"
    MAINTENANCE_WINDOW_OPEN = "maintenance_window_open"


class BrainAutonomyDecisionKind(str, Enum):
    """Typed autonomy-ledger decision kinds."""

    CREATED = "created"
    SUPPRESSED = "suppressed"
    MERGED = "merged"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    NON_ACTION = "non_action"


@dataclass
class BrainReevaluationTrigger:
    """One explicit trigger for bounded held-candidate reevaluation."""

    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    source_event_type: str | None = None
    source_event_id: str | None = None
    ts: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the reevaluation trigger."""
        return {
            "kind": self.kind,
            "summary": self.summary,
            "details": dict(self.details),
            "source_event_type": self.source_event_type,
            "source_event_id": self.source_event_id,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainReevaluationTrigger":
        """Hydrate one reevaluation trigger from stored JSON."""
        payload = data or {}
        return cls(
            kind=str(payload.get("kind", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            details=dict(payload.get("details") or {}),
            source_event_type=_optional_text(payload.get("source_event_type")),
            source_event_id=_optional_text(payload.get("source_event_id")),
            ts=str(payload.get("ts") or _utc_now()),
        )


@dataclass
class BrainCandidateGoal:
    """One proposed unit of work before it becomes an agenda goal."""

    candidate_goal_id: str
    candidate_type: str
    source: str
    summary: str
    goal_family: str
    urgency: float = 0.0
    confidence: float = 1.0
    initiative_class: str = BrainInitiativeClass.INSPECT_ONLY.value
    cooldown_key: str | None = None
    dedupe_key: str | None = None
    policy_tags: list[str] = field(default_factory=list)
    requires_user_turn_gap: bool = False
    expires_at: str | None = None
    expected_reevaluation_condition: str | None = None
    expected_reevaluation_condition_kind: str | None = None
    expected_reevaluation_condition_details: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the candidate goal."""
        return {
            "candidate_goal_id": self.candidate_goal_id,
            "candidate_type": self.candidate_type,
            "source": self.source,
            "summary": self.summary,
            "goal_family": self.goal_family,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "initiative_class": self.initiative_class,
            "cooldown_key": self.cooldown_key,
            "dedupe_key": self.dedupe_key,
            "policy_tags": list(self.policy_tags),
            "requires_user_turn_gap": self.requires_user_turn_gap,
            "expires_at": self.expires_at,
            "expected_reevaluation_condition": self.expected_reevaluation_condition,
            "expected_reevaluation_condition_kind": self.expected_reevaluation_condition_kind,
            "expected_reevaluation_condition_details": dict(
                self.expected_reevaluation_condition_details
            ),
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainCandidateGoal":
        """Hydrate one candidate goal from stored JSON."""
        payload = data or {}
        return cls(
            candidate_goal_id=str(payload.get("candidate_goal_id", "")).strip(),
            candidate_type=str(payload.get("candidate_type", "")).strip(),
            source=str(payload.get("source", BrainCandidateGoalSource.RUNTIME.value)).strip()
            or BrainCandidateGoalSource.RUNTIME.value,
            summary=str(payload.get("summary", "")).strip(),
            goal_family=str(payload.get("goal_family", "")).strip(),
            urgency=float(payload.get("urgency", 0.0)),
            confidence=float(payload.get("confidence", 1.0)),
            initiative_class=str(
                payload.get("initiative_class", BrainInitiativeClass.INSPECT_ONLY.value)
            ).strip()
            or BrainInitiativeClass.INSPECT_ONLY.value,
            cooldown_key=_optional_text(payload.get("cooldown_key")),
            dedupe_key=_optional_text(payload.get("dedupe_key")),
            policy_tags=[str(tag) for tag in payload.get("policy_tags", [])],
            requires_user_turn_gap=bool(payload.get("requires_user_turn_gap", False)),
            expires_at=_optional_text(payload.get("expires_at")),
            expected_reevaluation_condition=_optional_text(
                payload.get("expected_reevaluation_condition")
            ),
            expected_reevaluation_condition_kind=_optional_text(
                payload.get("expected_reevaluation_condition_kind")
            ),
            expected_reevaluation_condition_details=dict(
                payload.get("expected_reevaluation_condition_details") or {}
            ),
            payload=dict(payload.get("payload", {})),
            created_at=str(payload.get("created_at") or _utc_now()),
        )


@dataclass
class BrainAutonomyLedgerEntry:
    """One inspectable autonomy lifecycle or non-action record."""

    event_id: str
    event_type: str
    decision_kind: str
    candidate_goal_id: str | None = None
    summary: str | None = None
    reason: str | None = None
    reason_details: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    executive_policy: dict[str, Any] | None = None
    merged_into_candidate_goal_id: str | None = None
    accepted_goal_id: str | None = None
    accepted_commitment_id: str | None = None
    expected_reevaluation_condition: str | None = None
    expected_reevaluation_condition_kind: str | None = None
    expected_reevaluation_condition_details: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_utc_now)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the autonomy ledger entry."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "decision_kind": self.decision_kind,
            "candidate_goal_id": self.candidate_goal_id,
            "summary": self.summary,
            "reason": self.reason,
            "reason_details": dict(self.reason_details),
            "reason_codes": list(self.reason_codes),
            "executive_policy": dict(self.executive_policy or {})
            if self.executive_policy is not None
            else None,
            "merged_into_candidate_goal_id": self.merged_into_candidate_goal_id,
            "accepted_goal_id": self.accepted_goal_id,
            "accepted_commitment_id": self.accepted_commitment_id,
            "expected_reevaluation_condition": self.expected_reevaluation_condition,
            "expected_reevaluation_condition_kind": self.expected_reevaluation_condition_kind,
            "expected_reevaluation_condition_details": dict(
                self.expected_reevaluation_condition_details
            ),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAutonomyLedgerEntry":
        """Hydrate one autonomy ledger entry from stored JSON."""
        payload = data or {}
        return cls(
            event_id=str(payload.get("event_id", "")).strip(),
            event_type=str(payload.get("event_type", "")).strip(),
            decision_kind=str(
                payload.get("decision_kind", BrainAutonomyDecisionKind.NON_ACTION.value)
            ).strip()
            or BrainAutonomyDecisionKind.NON_ACTION.value,
            candidate_goal_id=_optional_text(payload.get("candidate_goal_id")),
            summary=_optional_text(payload.get("summary")),
            reason=_optional_text(payload.get("reason")),
            reason_details=dict(payload.get("reason_details", {})),
            reason_codes=sorted(
                {
                    str(item).strip()
                    for item in payload.get("reason_codes", [])
                    if str(item).strip()
                }
            ),
            executive_policy=(
                dict(payload.get("executive_policy", {}))
                if isinstance(payload.get("executive_policy"), dict)
                else None
            ),
            merged_into_candidate_goal_id=_optional_text(payload.get("merged_into_candidate_goal_id")),
            accepted_goal_id=_optional_text(payload.get("accepted_goal_id")),
            accepted_commitment_id=_optional_text(payload.get("accepted_commitment_id")),
            expected_reevaluation_condition=_optional_text(
                payload.get("expected_reevaluation_condition")
            ),
            expected_reevaluation_condition_kind=_optional_text(
                payload.get("expected_reevaluation_condition_kind")
            ),
            expected_reevaluation_condition_details=dict(
                payload.get("expected_reevaluation_condition_details") or {}
            ),
            ts=str(payload.get("ts") or _utc_now()),
        )


@dataclass
class BrainAutonomyLedgerProjection:
    """Thread-scoped current candidate goals plus recent autonomy decisions."""

    current_candidates: list[BrainCandidateGoal] = field(default_factory=list)
    recent_entries: list[BrainAutonomyLedgerEntry] = field(default_factory=list)
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize the autonomy-ledger projection."""
        return {
            "current_candidates": [candidate.as_dict() for candidate in self.current_candidates],
            "recent_entries": [entry.as_dict() for entry in self.recent_entries],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainAutonomyLedgerProjection":
        """Hydrate the autonomy-ledger projection from stored JSON."""
        payload = data or {}
        return cls(
            current_candidates=[
                BrainCandidateGoal.from_dict(item) for item in payload.get("current_candidates", [])
            ],
            recent_entries=[
                BrainAutonomyLedgerEntry.from_dict(item) for item in payload.get("recent_entries", [])
            ],
            updated_at=str(payload.get("updated_at") or ""),
        )

    def candidate(self, candidate_goal_id: str | None) -> BrainCandidateGoal | None:
        """Return one current candidate goal by id."""
        if not candidate_goal_id:
            return None
        for candidate in self.current_candidates:
            if candidate.candidate_goal_id == candidate_goal_id:
                return candidate
        return None

    def trim_recent_entries(self, *, limit: int = MAX_AUTONOMY_LEDGER_ENTRIES):
        """Keep only the most recent inspectable autonomy entries."""
        if len(self.recent_entries) > limit:
            self.recent_entries = self.recent_entries[-limit:]


def autonomy_decision_kind_for_event_type(event_type: str) -> str | None:
    """Map one autonomy event type to its typed decision kind."""
    mapping = {
        "goal.candidate.created": BrainAutonomyDecisionKind.CREATED.value,
        "goal.candidate.suppressed": BrainAutonomyDecisionKind.SUPPRESSED.value,
        "goal.candidate.merged": BrainAutonomyDecisionKind.MERGED.value,
        "goal.candidate.accepted": BrainAutonomyDecisionKind.ACCEPTED.value,
        "goal.candidate.expired": BrainAutonomyDecisionKind.EXPIRED.value,
        "director.non_action.recorded": BrainAutonomyDecisionKind.NON_ACTION.value,
    }
    return mapping.get(event_type)


__all__ = [
    "MAX_AUTONOMY_LEDGER_ENTRIES",
    "BrainAutonomyDecisionKind",
    "BrainAutonomyLedgerEntry",
    "BrainAutonomyLedgerProjection",
    "BrainCandidateGoal",
    "BrainCandidateGoalSource",
    "BrainInitiativeClass",
    "BrainReevaluationConditionKind",
    "autonomy_decision_kind_for_event_type",
]

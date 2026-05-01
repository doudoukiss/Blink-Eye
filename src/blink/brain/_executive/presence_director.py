"""Bounded PresenceDirector policy and coordinator for candidate goals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

from blink.brain._executive.policy import (
    BrainExecutivePolicyActionPosture,
    BrainExecutivePolicyApprovalRequirement,
    BrainExecutivePolicyFrame,
    neutral_executive_policy_frame,
)
from blink.brain.autonomy import (
    BrainAutonomyDecisionKind,
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.projections import (
    BrainAgendaProjection,
    BrainGoal,
    BrainGoalStatus,
    BrainWorkingContextProjection,
)
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore

_AUTONOMY_SOURCE = "presence_director"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _append_reason_code(target: list[str], value: str | None) -> None:
    text = str(value or "").strip()
    if text and text not in target:
        target.append(text)


def _decision_reason_codes(
    *,
    reason: str | None,
    executive_policy: BrainExecutivePolicyFrame | None,
    extra_codes: list[str] | tuple[str, ...] = (),
) -> list[str]:
    codes: list[str] = []
    _append_reason_code(codes, reason)
    for value in extra_codes:
        _append_reason_code(codes, value)
    if executive_policy is not None:
        for value in executive_policy.reason_codes:
            _append_reason_code(codes, value)
    return codes


def _initiative_cooldowns() -> dict[str, int]:
    return {
        BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value: 15,
        BrainInitiativeClass.SILENT_POSTURE_ADJUSTMENT.value: 30,
        BrainInitiativeClass.INSPECT_ONLY.value: 30,
        BrainInitiativeClass.DEFER_UNTIL_USER_TURN.value: 45,
        BrainInitiativeClass.SPEAK_BRIEFLY_IF_IDLE.value: 180,
        BrainInitiativeClass.MAINTENANCE_INTERNAL.value: 300,
        BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value: 300,
    }


@dataclass(frozen=True)
class BrainPresenceDirectorPolicy:
    """Deterministic policy knobs for one bounded presence-director pass."""

    max_candidates_per_pass: int = 24
    default_min_confidence: float = 0.5
    perception_min_confidence: float = 0.6
    maintenance_idle_grace_secs: float = 45.0
    cooldown_seconds_by_initiative_class: dict[str, int] = field(
        default_factory=_initiative_cooldowns
    )

    def confidence_threshold(self, candidate: BrainCandidateGoal) -> float:
        """Return the source-aware confidence floor for one candidate."""
        if candidate.source == BrainCandidateGoalSource.PERCEPTION.value:
            return self.perception_min_confidence
        return self.default_min_confidence

    def cooldown_seconds(self, candidate: BrainCandidateGoal) -> int:
        """Return the initiative-class cooldown window in seconds."""
        return int(self.cooldown_seconds_by_initiative_class.get(candidate.initiative_class, 0))

    def candidate_sort_key(self, candidate: BrainCandidateGoal) -> tuple[float, float, datetime, str]:
        """Return the canonical priority key for candidate selection and merging."""
        created_at = _parse_ts(candidate.created_at) or datetime.min.replace(tzinfo=UTC)
        return (-float(candidate.urgency), -float(candidate.confidence), created_at, candidate.candidate_goal_id)


@dataclass(frozen=True)
class BrainPresenceDirectorResult:
    """Inspectable result from one explicit director pass."""

    expired_candidate_ids: tuple[str, ...] = ()
    merged_candidate_ids: tuple[str, ...] = ()
    suppressed_candidate_ids: tuple[str, ...] = ()
    terminal_decision: str | None = None
    terminal_candidate_goal_id: str | None = None
    accepted_goal_id: str | None = None
    reason: str | None = None
    reason_codes: tuple[str, ...] = ()
    executive_policy: dict[str, Any] | None = None
    current_candidate_ids: tuple[str, ...] = ()

    @property
    def progressed(self) -> bool:
        """Return whether this pass emitted any inspectable decision events."""
        return bool(
            self.expired_candidate_ids
            or self.merged_candidate_ids
            or self.suppressed_candidate_ids
            or self.terminal_decision
        )


@dataclass(frozen=True)
class BrainPresenceDirectorNonAction:
    """One bounded non-action outcome plus typed reevaluation metadata."""

    reason: str
    explanation: str
    reevaluation_condition_kind: str
    reevaluation_condition_details: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    executive_policy: dict[str, Any] | None = None


class BrainPresenceDirector:
    """Bounded coordinator that turns current candidates into explicit decisions."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver: Callable[[], BrainSessionIds],
        goal_creator: Callable[..., str],
        policy: BrainPresenceDirectorPolicy | None = None,
    ):
        """Initialize the PresenceDirector."""
        self._store = store
        self._session_resolver = session_resolver
        self._goal_creator = goal_creator
        self._policy = policy or BrainPresenceDirectorPolicy()

    def run_once(
        self,
        *,
        executive_policy: BrainExecutivePolicyFrame | None = None,
    ) -> BrainPresenceDirectorResult:
        """Run one bounded PresenceDirector pass."""
        session_ids = self._session_resolver()
        ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        working_context = self._store.get_working_context_projection(scope_key=session_ids.thread_id)
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        recent_events = self._store.recent_autonomy_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=max(64, self._policy.max_candidates_per_pass * 8),
        )
        created_candidates, created_events = self._index_created_candidates(recent_events)
        evaluation_now = _utc_now()
        resolved_policy = executive_policy or neutral_executive_policy_frame(
            task="reevaluation",
            updated_at=evaluation_now.isoformat(),
        )
        expired_ids = self._expire_candidates(
            candidates=ledger.current_candidates,
            session_ids=session_ids,
            created_events=created_events,
            decision_causal_parent_id=None,
            cleanup_owner="director_pass",
            trigger_kind=None,
            now=evaluation_now,
        )
        current_candidates = self._store.get_autonomy_ledger_projection(
            scope_key=session_ids.thread_id
        ).current_candidates
        candidates = sorted(
            current_candidates,
            key=self._policy.candidate_sort_key,
        )[: self._policy.max_candidates_per_pass]
        return self._run_candidate_pass(
            candidates=candidates,
            session_ids=session_ids,
            working_context=working_context,
            agenda=agenda,
            recent_events=recent_events,
            created_candidates=created_candidates,
            created_events=created_events,
            executive_policy=resolved_policy,
            decision_causal_parent_id=None,
            evaluation_now=evaluation_now,
            preexpired_candidate_ids=tuple(expired_ids),
        )

    def reevaluate_once(
        self,
        trigger: BrainReevaluationTrigger,
        *,
        executive_policy: BrainExecutivePolicyFrame | None = None,
    ) -> BrainPresenceDirectorResult:
        """Run one bounded reevaluation pass over held current candidates only."""
        session_ids = self._session_resolver()
        ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        working_context = self._store.get_working_context_projection(scope_key=session_ids.thread_id)
        agenda = self._store.get_agenda_projection(
            scope_key=session_ids.thread_id,
            user_id=session_ids.user_id,
        )
        recent_events = self._store.recent_autonomy_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=max(64, self._policy.max_candidates_per_pass * 8),
        )
        created_candidates, created_events = self._index_created_candidates(recent_events)
        matched_candidates = self._reevaluation_candidates(
            candidates=ledger.current_candidates,
            trigger=trigger,
        )
        evaluation_now = _parse_ts(trigger.ts) or _utc_now()
        resolved_policy = executive_policy or neutral_executive_policy_frame(
            task="reevaluation",
            updated_at=trigger.ts,
        )
        trigger_event = self._store.append_director_reevaluation_triggered(
            trigger=trigger,
            candidate_goal_ids=[candidate.candidate_goal_id for candidate in matched_candidates],
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=_AUTONOMY_SOURCE,
            correlation_id=trigger.source_event_id or f"reevaluation:{trigger.kind}",
            causal_parent_id=trigger.source_event_id,
        )
        expired_ids = self._expire_candidates(
            candidates=ledger.current_candidates,
            session_ids=session_ids,
            created_events=created_events,
            decision_causal_parent_id=trigger_event.event_id,
            cleanup_owner="reevaluation_pass",
            trigger_kind=trigger.kind,
            now=evaluation_now,
        )
        expired_id_set = set(expired_ids)
        candidates = [
            candidate
            for candidate in matched_candidates
            if candidate.candidate_goal_id not in expired_id_set
        ]
        return self._run_candidate_pass(
            candidates=candidates,
            session_ids=session_ids,
            working_context=working_context,
            agenda=agenda,
            recent_events=recent_events,
            created_candidates=created_candidates,
            created_events=created_events,
            executive_policy=resolved_policy,
            decision_causal_parent_id=trigger_event.event_id,
            evaluation_now=evaluation_now,
            preexpired_candidate_ids=tuple(expired_ids),
        )

    def expire_once(self, trigger: BrainReevaluationTrigger) -> BrainPresenceDirectorResult:
        """Run one explicit expiry-cleanup pass without selecting new work."""
        session_ids = self._session_resolver()
        ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        recent_events = self._store.recent_autonomy_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=max(64, self._policy.max_candidates_per_pass * 8),
        )
        _, created_events = self._index_created_candidates(recent_events)
        now = _parse_ts(trigger.ts) or _utc_now()
        expired_candidates = [
            candidate
            for candidate in sorted(ledger.current_candidates, key=self._policy.candidate_sort_key)
            if self._is_expired(candidate, now=now)
        ]
        if not expired_candidates:
            return BrainPresenceDirectorResult(
                current_candidate_ids=tuple(
                    candidate.candidate_goal_id for candidate in ledger.current_candidates
                )
            )
        trigger_event = self._store.append_director_reevaluation_triggered(
            trigger=trigger,
            candidate_goal_ids=[candidate.candidate_goal_id for candidate in expired_candidates],
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=_AUTONOMY_SOURCE,
            correlation_id=trigger.source_event_id or f"reevaluation:{trigger.kind}",
            causal_parent_id=trigger.source_event_id,
        )
        expired_ids = self._expire_candidates(
            candidates=expired_candidates,
            session_ids=session_ids,
            created_events=created_events,
            decision_causal_parent_id=trigger_event.event_id,
            cleanup_owner="explicit_expiry_cleanup",
            trigger_kind=trigger.kind,
            now=now,
        )
        updated_ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        return BrainPresenceDirectorResult(
            expired_candidate_ids=tuple(expired_ids),
            current_candidate_ids=tuple(
                candidate.candidate_goal_id for candidate in updated_ledger.current_candidates
            ),
        )

    def _run_candidate_pass(
        self,
        *,
        candidates: list[BrainCandidateGoal],
        session_ids: BrainSessionIds,
        working_context: BrainWorkingContextProjection,
        agenda: BrainAgendaProjection,
        recent_events: list[BrainEventRecord],
        created_candidates: dict[str, BrainCandidateGoal],
        created_events: dict[str, BrainEventRecord],
        executive_policy: BrainExecutivePolicyFrame,
        decision_causal_parent_id: str | None,
        evaluation_now: datetime,
        preexpired_candidate_ids: tuple[str, ...] = (),
    ) -> BrainPresenceDirectorResult:
        """Apply one bounded candidate pass over a fixed candidate set."""
        now = evaluation_now

        expired_ids: list[str] = list(preexpired_candidate_ids)
        merged_ids: list[str] = []
        suppressed_ids: list[str] = []

        active_candidates: dict[str, BrainCandidateGoal] = {
            candidate.candidate_goal_id: candidate for candidate in candidates
        }

        dedupe_groups: dict[str, list[BrainCandidateGoal]] = {}
        for candidate in active_candidates.values():
            if candidate.dedupe_key:
                dedupe_groups.setdefault(candidate.dedupe_key, []).append(candidate)
        for dedupe_key, group in dedupe_groups.items():
            if len(group) < 2:
                continue
            canonical = sorted(group, key=self._policy.candidate_sort_key)[0]
            canonical_correlation_id = self._correlation_id(canonical, created_events)
            for candidate in sorted(group, key=self._policy.candidate_sort_key)[1:]:
                self._store.append_candidate_goal_merged(
                    candidate_goal_id=candidate.candidate_goal_id,
                    merged_into_candidate_goal_id=canonical.candidate_goal_id,
                    reason="dedupe_key_collision",
                    reason_details={
                        "dedupe_key": dedupe_key,
                        "canonical_candidate_goal_id": canonical.candidate_goal_id,
                    },
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=canonical_correlation_id,
                    causal_parent_id=self._candidate_causal_parent_id(
                        candidate,
                        created_events,
                        decision_causal_parent_id=decision_causal_parent_id,
                    ),
                )
                merged_ids.append(candidate.candidate_goal_id)
                active_candidates.pop(candidate.candidate_goal_id, None)

        for candidate in sorted(active_candidates.values(), key=self._policy.candidate_sort_key):
            correlation_id = self._correlation_id(candidate, created_events)
            duplicate = self._duplicate_active_goal(candidate, agenda)
            if duplicate is not None:
                self._store.append_candidate_goal_suppressed(
                    candidate_goal_id=candidate.candidate_goal_id,
                    reason="duplicate_active_goal",
                    reason_details=duplicate,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=correlation_id,
                    causal_parent_id=self._candidate_causal_parent_id(
                        candidate,
                        created_events,
                        decision_causal_parent_id=decision_causal_parent_id,
                    ),
                )
                suppressed_ids.append(candidate.candidate_goal_id)
                active_candidates.pop(candidate.candidate_goal_id, None)
                continue
            threshold = self._policy.confidence_threshold(candidate)
            if float(candidate.confidence) < threshold:
                self._store.append_candidate_goal_suppressed(
                    candidate_goal_id=candidate.candidate_goal_id,
                    reason="low_confidence",
                    reason_details={
                        "confidence": candidate.confidence,
                        "threshold": threshold,
                        "source": candidate.source,
                    },
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=correlation_id,
                    causal_parent_id=self._candidate_causal_parent_id(
                        candidate,
                        created_events,
                        decision_causal_parent_id=decision_causal_parent_id,
                    ),
                )
                suppressed_ids.append(candidate.candidate_goal_id)
                active_candidates.pop(candidate.candidate_goal_id, None)
                continue
            cooldown_block = self._cooldown_block(
                candidate=candidate,
                recent_events=recent_events,
                created_candidates=created_candidates,
                now=now,
            )
            if cooldown_block is not None:
                self._store.append_candidate_goal_suppressed(
                    candidate_goal_id=candidate.candidate_goal_id,
                    reason="cooldown_active",
                    reason_details=cooldown_block,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=correlation_id,
                    causal_parent_id=self._candidate_causal_parent_id(
                        candidate,
                        created_events,
                        decision_causal_parent_id=decision_causal_parent_id,
                    ),
                )
                suppressed_ids.append(candidate.candidate_goal_id)
                active_candidates.pop(candidate.candidate_goal_id, None)

        terminal_decision: str | None = None
        terminal_candidate_goal_id: str | None = None
        accepted_goal_id: str | None = None
        reason: str | None = None
        reason_codes: tuple[str, ...] = ()

        remaining = sorted(active_candidates.values(), key=self._policy.candidate_sort_key)
        if remaining:
            candidate, fairness_details = self._select_terminal_candidate(
                candidates=remaining,
                recent_events=recent_events,
                created_candidates=created_candidates,
            )
            terminal_candidate_goal_id = candidate.candidate_goal_id
            correlation_id = self._correlation_id(candidate, created_events)
            causal_parent_id = self._candidate_causal_parent_id(
                candidate,
                created_events,
                decision_causal_parent_id=decision_causal_parent_id,
            )
            terminal_reason_details = {
                "goal_family": candidate.goal_family,
                "initiative_class": candidate.initiative_class,
                **fairness_details,
            }
            non_action = self._non_action_reason(
                candidate=candidate,
                working_context=working_context,
                agenda=agenda,
                executive_policy=executive_policy,
                now=now,
                idle_grace_secs=self._policy.maintenance_idle_grace_secs,
            )
            if non_action is not None:
                reason = non_action.reason
                reason_codes = tuple(non_action.reason_codes)
                self._store.append_director_non_action(
                    candidate_goal_id=candidate.candidate_goal_id,
                    reason=reason,
                    reason_details=terminal_reason_details,
                    reason_codes=list(non_action.reason_codes),
                    executive_policy=non_action.executive_policy,
                    expected_reevaluation_condition=non_action.explanation,
                    expected_reevaluation_condition_kind=non_action.reevaluation_condition_kind,
                    expected_reevaluation_condition_details=non_action.reevaluation_condition_details,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=correlation_id,
                    causal_parent_id=causal_parent_id,
                )
                terminal_decision = BrainAutonomyDecisionKind.NON_ACTION.value
            else:
                goal_id = self._accept_candidate(
                    candidate=candidate,
                    correlation_id=correlation_id,
                    causal_parent_id=causal_parent_id,
                )
                accepted_goal = self._store.get_agenda_projection(
                    scope_key=session_ids.thread_id,
                    user_id=session_ids.user_id,
                ).goal(goal_id)
                reason_codes = tuple(
                    _decision_reason_codes(
                        reason="accepted_for_goal_creation",
                        executive_policy=executive_policy,
                    )
                )
                self._store.append_candidate_goal_accepted(
                    candidate_goal_id=candidate.candidate_goal_id,
                    goal_id=goal_id,
                    commitment_id=accepted_goal.commitment_id if accepted_goal is not None else None,
                    reason="accepted_for_goal_creation",
                    reason_details=terminal_reason_details,
                    reason_codes=list(reason_codes),
                    executive_policy=executive_policy.as_dict(),
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=_AUTONOMY_SOURCE,
                    correlation_id=correlation_id,
                    causal_parent_id=causal_parent_id,
                )
                terminal_decision = BrainAutonomyDecisionKind.ACCEPTED.value
                accepted_goal_id = goal_id
                reason = "accepted_for_goal_creation"

        updated_ledger = self._store.get_autonomy_ledger_projection(scope_key=session_ids.thread_id)
        return BrainPresenceDirectorResult(
            expired_candidate_ids=tuple(expired_ids),
            merged_candidate_ids=tuple(merged_ids),
            suppressed_candidate_ids=tuple(suppressed_ids),
            terminal_decision=terminal_decision,
            terminal_candidate_goal_id=terminal_candidate_goal_id,
            accepted_goal_id=accepted_goal_id,
            reason=reason,
            reason_codes=reason_codes,
            executive_policy=executive_policy.as_dict(),
            current_candidate_ids=tuple(
                candidate.candidate_goal_id for candidate in updated_ledger.current_candidates
            ),
        )

    def _expire_candidates(
        self,
        *,
        candidates: list[BrainCandidateGoal],
        session_ids: BrainSessionIds,
        created_events: dict[str, BrainEventRecord],
        decision_causal_parent_id: str | None,
        cleanup_owner: str,
        trigger_kind: str | None,
        now: datetime,
    ) -> list[str]:
        """Expire the supplied stale candidates and return their ids."""
        expired_ids: list[str] = []
        for candidate in sorted(candidates, key=self._policy.candidate_sort_key):
            if not self._is_expired(candidate, now=now):
                continue
            correlation_id = self._correlation_id(candidate, created_events)
            reason_details = {
                "expires_at": candidate.expires_at,
                "cleanup_owner": cleanup_owner,
            }
            if trigger_kind:
                reason_details["trigger_kind"] = trigger_kind
            self._store.append_candidate_goal_expired(
                candidate_goal_id=candidate.candidate_goal_id,
                reason="expired_window_elapsed",
                reason_details=reason_details,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=_AUTONOMY_SOURCE,
                correlation_id=correlation_id,
                causal_parent_id=self._candidate_causal_parent_id(
                    candidate,
                    created_events,
                    decision_causal_parent_id=decision_causal_parent_id,
                ),
            )
            expired_ids.append(candidate.candidate_goal_id)
        return expired_ids

    def _reevaluation_candidates(
        self,
        *,
        candidates: list[BrainCandidateGoal],
        trigger: BrainReevaluationTrigger,
    ) -> list[BrainCandidateGoal]:
        """Return the bounded current candidates eligible for one reevaluation trigger."""
        now = _parse_ts(trigger.ts) or _utc_now()
        matched: list[BrainCandidateGoal] = []
        for candidate in sorted(candidates, key=self._policy.candidate_sort_key):
            if self._is_expired(candidate, now=now):
                matched.append(candidate)
                continue
            if not candidate.expected_reevaluation_condition_kind:
                continue
            if trigger.kind == BrainReevaluationConditionKind.STARTUP_RECOVERY.value:
                matched.append(candidate)
                continue
            if trigger.kind == BrainReevaluationConditionKind.PROJECTION_CHANGED.value:
                if (
                    candidate.expected_reevaluation_condition_kind
                    == BrainReevaluationConditionKind.PROJECTION_CHANGED.value
                    or self._allows_projection_recheck(candidate)
                ):
                    matched.append(candidate)
                continue
            if candidate.expected_reevaluation_condition_kind != trigger.kind:
                continue
            if not self._trigger_details_match(candidate=candidate, trigger=trigger, now=now):
                continue
            matched.append(candidate)
        return matched[: self._policy.max_candidates_per_pass]

    @staticmethod
    def _allows_projection_recheck(candidate: BrainCandidateGoal) -> bool:
        details = dict(candidate.expected_reevaluation_condition_details or {})
        return (
            candidate.source == BrainCandidateGoalSource.PERCEPTION.value
            and bool(details.get("allow_projection_recheck"))
        )

    @staticmethod
    def _trigger_details_match(
        *,
        candidate: BrainCandidateGoal,
        trigger: BrainReevaluationTrigger,
        now: datetime,
    ) -> bool:
        candidate_details = dict(candidate.expected_reevaluation_condition_details or {})
        trigger_details = dict(trigger.details or {})
        if trigger.kind == BrainReevaluationConditionKind.GOAL_FAMILY_AVAILABLE.value:
            expected_goal_family = str(candidate_details.get("goal_family", "")).strip()
            trigger_goal_family = str(trigger_details.get("goal_family", "")).strip()
            return not expected_goal_family or expected_goal_family == trigger_goal_family
        if trigger.kind in {
            BrainReevaluationConditionKind.TIME_REACHED.value,
            BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
        }:
            not_before = _parse_ts(str(candidate_details.get("not_before", "")).strip() or None)
            return not_before is None or not_before <= now
        return True

    def _accept_candidate(
        self,
        *,
        candidate: BrainCandidateGoal,
        correlation_id: str | None = None,
        causal_parent_id: str | None = None,
    ) -> str:
        """Create one normal agenda goal from an accepted candidate."""
        goal_id = uuid4().hex
        goal_details = candidate.payload.get("goal_details", {})
        details = dict(goal_details) if isinstance(goal_details, dict) else {}
        existing_autonomy = details.get("autonomy")
        autonomy_details = dict(existing_autonomy) if isinstance(existing_autonomy, dict) else {}
        autonomy_details.update(
            {
                "candidate_goal_id": candidate.candidate_goal_id,
                "candidate_type": candidate.candidate_type,
                "initiative_class": candidate.initiative_class,
                "cooldown_key": candidate.cooldown_key,
                "dedupe_key": candidate.dedupe_key,
                "policy_tags": list(candidate.policy_tags),
            }
        )
        details["autonomy"] = autonomy_details
        intent = candidate.payload.get("goal_intent")
        return self._goal_creator(
            goal_id=goal_id,
            title=candidate.summary,
            intent=intent if isinstance(intent, str) and intent.strip() else f"autonomy.{candidate.candidate_type}",
            source=_AUTONOMY_SOURCE,
            goal_family=candidate.goal_family,
            details=details,
            correlation_id=correlation_id or candidate.candidate_goal_id,
            causal_parent_id=causal_parent_id,
        )

    def _causal_parent_id(
        self,
        candidate: BrainCandidateGoal,
        created_events: dict[str, BrainEventRecord],
    ) -> str | None:
        created_event = created_events.get(candidate.candidate_goal_id)
        return created_event.event_id if created_event is not None else None

    def _candidate_causal_parent_id(
        self,
        candidate: BrainCandidateGoal,
        created_events: dict[str, BrainEventRecord],
        *,
        decision_causal_parent_id: str | None,
    ) -> str | None:
        """Return the causal parent id for one pass decision."""
        if decision_causal_parent_id is not None:
            return decision_causal_parent_id
        return self._causal_parent_id(candidate, created_events)

    def _correlation_id(
        self,
        candidate: BrainCandidateGoal,
        created_events: dict[str, BrainEventRecord],
    ) -> str:
        created_event = created_events.get(candidate.candidate_goal_id)
        if created_event is not None and created_event.correlation_id:
            return created_event.correlation_id
        return candidate.candidate_goal_id

    def _cooldown_block(
        self,
        *,
        candidate: BrainCandidateGoal,
        recent_events: list[BrainEventRecord],
        created_candidates: dict[str, BrainCandidateGoal],
        now: datetime,
    ) -> dict[str, Any] | None:
        if not candidate.cooldown_key:
            return None
        cooldown_seconds = self._policy.cooldown_seconds(candidate)
        if cooldown_seconds <= 0:
            return None
        for event in recent_events:
            if event.event_type == BrainEventType.GOAL_CANDIDATE_CREATED:
                continue
            event_candidate_id = str(event.payload.get("candidate_goal_id", "")).strip()
            if not event_candidate_id or event_candidate_id == candidate.candidate_goal_id:
                continue
            prior_candidate = created_candidates.get(event_candidate_id)
            if prior_candidate is None or prior_candidate.cooldown_key != candidate.cooldown_key:
                continue
            event_ts = _parse_ts(event.ts)
            if event_ts is None:
                continue
            age_seconds = (now - event_ts).total_seconds()
            if age_seconds < cooldown_seconds:
                return {
                    "cooldown_key": candidate.cooldown_key,
                    "cooldown_seconds": cooldown_seconds,
                    "blocking_candidate_goal_id": event_candidate_id,
                    "blocking_event_id": event.event_id,
                    "blocking_event_type": event.event_type,
                }
        return None

    def _select_terminal_candidate(
        self,
        *,
        candidates: list[BrainCandidateGoal],
        recent_events: list[BrainEventRecord],
        created_candidates: dict[str, BrainCandidateGoal],
    ) -> tuple[BrainCandidateGoal, dict[str, Any]]:
        """Choose one family leader by bounded family-rotation fairness."""
        family_leaders: dict[str, BrainCandidateGoal] = {}
        family_counts: dict[str, int] = {}
        for candidate in sorted(candidates, key=self._policy.candidate_sort_key):
            family_counts[candidate.goal_family] = family_counts.get(candidate.goal_family, 0) + 1
            family_leaders.setdefault(candidate.goal_family, candidate)
        family_attention = self._family_terminal_attention(
            recent_events=recent_events,
            created_candidates=created_candidates,
            current_candidates={candidate.candidate_goal_id: candidate for candidate in candidates},
            goal_families=set(family_leaders),
        )
        selected_family, selected_candidate = sorted(
            family_leaders.items(),
            key=lambda item: self._family_rotation_key(
                goal_family=item[0],
                leader=item[1],
                family_attention=family_attention,
            ),
        )[0]
        fairness_details = {
            "selected_by": "family_rotation",
            "goal_family": selected_family,
            "family_pending_count": family_counts.get(selected_family, 1),
            "has_recent_terminal_attention": selected_family in family_attention,
            "family_last_terminal_attention_at": (
                family_attention[selected_family].isoformat()
                if selected_family in family_attention
                else None
            ),
        }
        return selected_candidate, fairness_details

    def _family_rotation_key(
        self,
        *,
        goal_family: str,
        leader: BrainCandidateGoal,
        family_attention: dict[str, datetime],
    ) -> tuple[int, datetime, datetime, tuple[float, float, datetime, str], str]:
        """Return the deterministic fairness key for one family leader."""
        leader_created_at = _parse_ts(leader.created_at) or datetime.max.replace(tzinfo=UTC)
        last_attention = family_attention.get(goal_family)
        return (
            0 if last_attention is None else 1,
            last_attention or datetime.min.replace(tzinfo=UTC),
            leader_created_at,
            self._policy.candidate_sort_key(leader),
            goal_family,
        )

    @staticmethod
    def _family_terminal_attention(
        *,
        recent_events: list[BrainEventRecord],
        created_candidates: dict[str, BrainCandidateGoal],
        current_candidates: dict[str, BrainCandidateGoal],
        goal_families: set[str],
    ) -> dict[str, datetime]:
        """Return the latest terminal-attention timestamp by goal family."""
        attention: dict[str, datetime] = {}
        known_candidates = {**created_candidates, **current_candidates}
        for event in recent_events:
            if event.event_type not in {
                BrainEventType.GOAL_CANDIDATE_ACCEPTED,
                BrainEventType.DIRECTOR_NON_ACTION_RECORDED,
            }:
                continue
            reason_details = dict(event.payload.get("reason_details") or {})
            goal_family = str(reason_details.get("goal_family", "")).strip()
            if not goal_family:
                candidate_goal_id = str(event.payload.get("candidate_goal_id", "")).strip()
                candidate = known_candidates.get(candidate_goal_id)
                goal_family = candidate.goal_family if candidate is not None else ""
            if goal_family not in goal_families or goal_family in attention:
                continue
            event_ts = _parse_ts(event.ts)
            if event_ts is None:
                continue
            attention[goal_family] = event_ts
        return attention

    @staticmethod
    def _duplicate_active_goal(
        candidate: BrainCandidateGoal,
        agenda: BrainAgendaProjection,
    ) -> dict[str, Any] | None:
        for goal in agenda.goals:
            if goal.is_terminal:
                continue
            metadata = BrainPresenceDirector._goal_autonomy_metadata(goal)
            if candidate.dedupe_key and metadata.get("dedupe_key") == candidate.dedupe_key:
                return {
                    "goal_id": goal.goal_id,
                    "match": "dedupe_key",
                    "dedupe_key": candidate.dedupe_key,
                }
            if candidate.cooldown_key and metadata.get("cooldown_key") == candidate.cooldown_key:
                return {
                    "goal_id": goal.goal_id,
                    "match": "cooldown_key",
                    "cooldown_key": candidate.cooldown_key,
                }
        return None

    @staticmethod
    def _goal_autonomy_metadata(goal: BrainGoal) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        nested = goal.details.get("autonomy")
        if isinstance(nested, dict):
            metadata.update(nested)
        for key in (
            "candidate_goal_id",
            "candidate_type",
            "initiative_class",
            "cooldown_key",
            "dedupe_key",
            "policy_tags",
        ):
            if key in goal.details and key not in metadata:
                metadata[key] = goal.details.get(key)
        return metadata

    @staticmethod
    def _index_created_candidates(
        recent_events: list[BrainEventRecord],
    ) -> tuple[dict[str, BrainCandidateGoal], dict[str, BrainEventRecord]]:
        created_candidates: dict[str, BrainCandidateGoal] = {}
        created_events: dict[str, BrainEventRecord] = {}
        for event in recent_events:
            if event.event_type != BrainEventType.GOAL_CANDIDATE_CREATED:
                continue
            payload = event.payload.get("candidate_goal")
            if not isinstance(payload, dict):
                continue
            candidate = BrainCandidateGoal.from_dict(payload)
            if not candidate.candidate_goal_id:
                continue
            created_candidates.setdefault(candidate.candidate_goal_id, candidate)
            created_events.setdefault(candidate.candidate_goal_id, event)
        return created_candidates, created_events

    @staticmethod
    def _is_expired(candidate: BrainCandidateGoal, *, now: datetime) -> bool:
        expires_at = _parse_ts(candidate.expires_at)
        return expires_at is not None and expires_at <= now

    @staticmethod
    def _non_action_reason(
        *,
        candidate: BrainCandidateGoal,
        working_context: BrainWorkingContextProjection,
        agenda: BrainAgendaProjection,
        executive_policy: BrainExecutivePolicyFrame,
        now: datetime,
        idle_grace_secs: float,
    ) -> BrainPresenceDirectorNonAction | None:
        if candidate.requires_user_turn_gap and working_context.user_turn_open:
            return BrainPresenceDirectorNonAction(
                reason="user_turn_open",
                explanation="after user turn ends",
                reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                reevaluation_condition_details={"turn": "user"},
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason="user_turn_open",
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        if candidate.requires_user_turn_gap and working_context.assistant_turn_open:
            return BrainPresenceDirectorNonAction(
                reason="assistant_turn_open",
                explanation="after assistant turn ends",
                reevaluation_condition_kind=BrainReevaluationConditionKind.ASSISTANT_TURN_CLOSED.value,
                reevaluation_condition_details={"turn": "assistant"},
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason="assistant_turn_open",
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        if (
            candidate.initiative_class != BrainInitiativeClass.MAINTENANCE_INTERNAL.value
            and BrainPresenceDirector._family_busy(candidate.goal_family, agenda)
        ):
            return BrainPresenceDirectorNonAction(
                reason="goal_family_busy",
                explanation="after current goal completes, blocks, or is cancelled",
                reevaluation_condition_kind=BrainReevaluationConditionKind.GOAL_FAMILY_AVAILABLE.value,
                reevaluation_condition_details={"goal_family": candidate.goal_family},
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason="goal_family_busy",
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        if BrainPresenceDirector._maintenance_window_closed(
            candidate=candidate,
            working_context=working_context,
            now=now,
            idle_grace_secs=idle_grace_secs,
        ):
            not_before = (
                (_parse_ts(working_context.updated_at) or now) + timedelta(seconds=idle_grace_secs)
            ).isoformat()
            return BrainPresenceDirectorNonAction(
                reason="maintenance_window_closed",
                explanation="after the thread has been idle long enough for maintenance",
                reevaluation_condition_kind=BrainReevaluationConditionKind.MAINTENANCE_WINDOW_OPEN.value,
                reevaluation_condition_details={
                    "not_before": not_before,
                    "idle_grace_secs": idle_grace_secs,
                },
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason="maintenance_window_closed",
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        if executive_policy.action_posture == BrainExecutivePolicyActionPosture.SUPPRESS.value:
            return BrainPresenceDirectorNonAction(
                reason="policy_blocked_action",
                explanation="after policy-relevant projections change",
                reevaluation_condition_kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
                reevaluation_condition_details={
                    "policy_reason_codes": list(executive_policy.reason_codes),
                    "policy_suppression_codes": list(executive_policy.suppression_codes),
                    "source_dossier_ids": list(executive_policy.source_dossier_ids),
                    "source_active_record_ids": list(executive_policy.source_active_record_ids),
                    "source_claim_ids": list(executive_policy.source_claim_ids),
                    "source_commitment_ids": list(executive_policy.source_commitment_ids),
                    "source_plan_proposal_ids": list(executive_policy.source_plan_proposal_ids),
                },
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason="policy_blocked_action",
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        if executive_policy.action_posture == BrainExecutivePolicyActionPosture.DEFER.value:
            reason = (
                "policy_requires_confirmation"
                if (
                    executive_policy.pending_user_review_count > 0
                    or executive_policy.held_claim_count > 0
                )
                else "policy_conservative_deferral"
            )
            return BrainPresenceDirectorNonAction(
                reason=reason,
                explanation="after policy-relevant projections change",
                reevaluation_condition_kind=BrainReevaluationConditionKind.PROJECTION_CHANGED.value,
                reevaluation_condition_details={
                    "policy_reason_codes": list(executive_policy.reason_codes),
                    "policy_suppression_codes": list(executive_policy.suppression_codes),
                    "source_dossier_ids": list(executive_policy.source_dossier_ids),
                    "source_active_record_ids": list(executive_policy.source_active_record_ids),
                    "source_claim_ids": list(executive_policy.source_claim_ids),
                    "source_commitment_ids": list(executive_policy.source_commitment_ids),
                    "source_plan_proposal_ids": list(executive_policy.source_plan_proposal_ids),
                },
                reason_codes=tuple(
                    _decision_reason_codes(
                        reason=reason,
                        executive_policy=executive_policy,
                    )
                ),
                executive_policy=executive_policy.as_dict(),
            )
        return None

    @staticmethod
    def _family_busy(goal_family: str, agenda: BrainAgendaProjection) -> bool:
        for goal in agenda.goals:
            if goal.goal_family != goal_family:
                continue
            if goal.status in {
                BrainGoalStatus.OPEN.value,
                BrainGoalStatus.PLANNING.value,
                BrainGoalStatus.IN_PROGRESS.value,
                BrainGoalStatus.RETRY.value,
            }:
                return True
        return False

    @staticmethod
    def _maintenance_window_closed(
        *,
        candidate: BrainCandidateGoal,
        working_context: BrainWorkingContextProjection,
        now: datetime,
        idle_grace_secs: float,
    ) -> bool:
        if candidate.goal_family != "memory_maintenance" and candidate.initiative_class not in {
            BrainInitiativeClass.MAINTENANCE_INTERNAL.value,
            BrainInitiativeClass.OPERATOR_VISIBLE_ONLY.value,
        }:
            return False
        if working_context.user_turn_open or working_context.assistant_turn_open:
            return False
        idle_anchor = _parse_ts(working_context.updated_at) or now
        idle_age_secs = max(0.0, (now - idle_anchor).total_seconds())
        return idle_age_secs < float(idle_grace_secs)


__all__ = [
    "BrainPresenceDirector",
    "BrainPresenceDirectorPolicy",
    "BrainPresenceDirectorResult",
]

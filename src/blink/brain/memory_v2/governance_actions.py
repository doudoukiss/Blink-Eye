"""Scoped user-governance actions for memory-palace claim records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.events import BrainEventType
from blink.brain.memory_layers.semantic import render_preference_fact, render_profile_fact
from blink.brain.memory_v2.claims import BrainClaimRecord
from blink.brain.projections import (
    BrainClaimCurrentnessStatus,
    BrainClaimRetentionClass,
    BrainGovernanceReasonCode,
)

_SCHEMA_VERSION = 1
_CLAIM_ACTIONS = frozenset({"pin", "suppress", "correct", "forget", "mark_stale"})
_TASK_ACTIONS = frozenset({"mark_done", "cancel"})
_SUPPORTED_ACTIONS = _CLAIM_ACTIONS | _TASK_ACTIONS
_ACTION_ALIASES = {"mark-stale": "mark_stale", "mark-done": "mark_done"}
_CORRECTABLE_PROFILE_PREDICATES = frozenset({"profile.name", "profile.role", "profile.origin"})
_CORRECTABLE_PREFERENCE_PREDICATES = frozenset({"preference.like", "preference.dislike"})
_CORRECTABLE_PREDICATES = _CORRECTABLE_PROFILE_PREDICATES | _CORRECTABLE_PREFERENCE_PREDICATES
_TERMINAL_STATUSES = frozenset({"revoked", "superseded"})
_USER_PINNED_REASON_CODE = BrainGovernanceReasonCode.USER_PINNED.value


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _optional_text(value: Any) -> str | None:
    text = _normalized_text(value)
    return text or None


def _dedupe_reason_codes(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        code = _normalized_text(value)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
    return tuple(result)


def _memory_id_for_claim(*, user_id: str, claim_id: str) -> str:
    return f"memory_claim:user:{user_id}:{claim_id}"


def _memory_id_for_task(*, user_id: str, task_ref: str) -> str:
    return f"memory_task:user:{user_id}:{task_ref}"


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(_normalized_text(part) for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _task_ref(task: dict[str, Any], *, user_id: str) -> tuple[str, bool]:
    commitment_id = _optional_text(task.get("commitment_id"))
    if commitment_id is not None:
        return commitment_id, True
    return (
        _stable_id(
            "task",
            user_id,
            task.get("title"),
            task.get("status"),
            task.get("updated_at") or task.get("created_at") or "",
        ),
        False,
    )


def _task_memory_id(task: dict[str, Any], *, user_id: str) -> tuple[str, bool]:
    task_ref, commitment_backed = _task_ref(task, user_id=user_id)
    return _memory_id_for_task(user_id=user_id, task_ref=task_ref), commitment_backed


def _event_context(*, session_ids, source: str, action: str, memory_id: str) -> dict[str, Any]:
    return {
        "agent_id": str(getattr(session_ids, "agent_id", "") or "blink/main"),
        "user_id": str(getattr(session_ids, "user_id", "")),
        "session_id": str(getattr(session_ids, "session_id", "")),
        "thread_id": str(getattr(session_ids, "thread_id", "")),
        "source": source,
        "correlation_id": f"memory_governance:{action}:{memory_id}",
    }


@dataclass(frozen=True)
class BrainMemoryGovernanceActionResult:
    """Result returned by one scoped memory governance action."""

    accepted: bool
    applied: bool
    action: str
    memory_id: str
    record_kind: str | None
    event_id: str | None
    replacement_memory_id: str | None
    reason_codes: tuple[str, ...]
    schema_version: int = _SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize the governance action result."""
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "applied": self.applied,
            "action": self.action,
            "memory_id": self.memory_id,
            "record_kind": self.record_kind,
            "event_id": self.event_id,
            "replacement_memory_id": self.replacement_memory_id,
            "reason_codes": list(self.reason_codes),
        }


def _result(
    *,
    accepted: bool,
    applied: bool,
    action: str,
    memory_id: str,
    record_kind: str | None,
    event_id: str | None = None,
    replacement_memory_id: str | None = None,
    reason_codes: tuple[str, ...],
) -> BrainMemoryGovernanceActionResult:
    return BrainMemoryGovernanceActionResult(
        accepted=accepted,
        applied=applied,
        action=action,
        memory_id=memory_id,
        record_kind=record_kind,
        event_id=event_id,
        replacement_memory_id=replacement_memory_id,
        reason_codes=_dedupe_reason_codes(reason_codes),
    )


def _reject(
    *,
    action: str,
    memory_id: str,
    record_kind: str | None,
    reason_codes: tuple[str, ...],
) -> BrainMemoryGovernanceActionResult:
    return _result(
        accepted=False,
        applied=False,
        action=action,
        memory_id=memory_id,
        record_kind=record_kind,
        reason_codes=("memory_action_rejected", *reason_codes),
    )


def _normalize_action(action: str) -> str:
    normalized = _normalized_text(action).lower().replace("-", "_")
    return _ACTION_ALIASES.get(normalized, normalized)


def _parse_memory_id(
    memory_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    parts = _normalized_text(memory_id).split(":", 3)
    if len(parts) != 4:
        return None, None, None, None
    prefix, scope_type, scope_id, record_id = parts
    return prefix or None, scope_type or None, scope_id or None, record_id or None


def _claim_ledger(store):
    ledger = getattr(store, "_claims", None)
    if callable(ledger):
        return ledger()
    return None


def _active_tasks(store, *, user_id: str) -> list[dict[str, Any]]:
    active_tasks = getattr(store, "active_tasks", None)
    if not callable(active_tasks):
        return []
    return [dict(task) for task in active_tasks(user_id=user_id, limit=128)]


def _active_tasks_available(store) -> bool:
    return callable(getattr(store, "active_tasks", None))


def _load_scoped_claim(
    *,
    store,
    session_ids,
    memory_id: str,
    action: str,
) -> tuple[BrainClaimRecord | None, BrainMemoryGovernanceActionResult | None]:
    prefix, scope_type, scope_id, claim_id = _parse_memory_id(memory_id)
    if prefix is None or scope_type is None or scope_id is None or claim_id is None:
        prefix = _normalized_text(memory_id).split(":", 1)[0] or None
        if prefix in {"memory_task", "memory_block"}:
            return None, _reject(
                action=action,
                memory_id=memory_id,
                record_kind=prefix,
                reason_codes=("record_kind_unsupported", f"record_kind:{prefix}"),
            )
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind=None,
            reason_codes=("memory_id_malformed",),
        )
    if prefix != "memory_claim":
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind=prefix,
            reason_codes=("record_kind_unsupported", f"record_kind:{prefix}"),
        )
    if scope_type != "user":
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("scope_type_unsupported", f"scope_type:{scope_type}"),
        )
    expected_user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    if scope_id != expected_user_id:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("cross_user_memory_id",),
        )
    ledger = _claim_ledger(store)
    if ledger is None:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("claim_ledger_unavailable",),
        )
    claim = ledger.get_claim(claim_id)
    if claim is None:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("claim_not_found",),
        )
    if claim.scope_type != "user" or claim.scope_id != expected_user_id:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("cross_user_memory_id",),
        )
    if (
        claim.status in _TERMINAL_STATUSES
        or claim.effective_currentness_status == BrainClaimCurrentnessStatus.HISTORICAL.value
    ):
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("claim_not_current", f"claim_status:{claim.status}"),
        )
    return claim, None


def _load_scoped_task(
    *,
    store,
    session_ids,
    memory_id: str,
    action: str,
) -> tuple[dict[str, Any] | None, BrainMemoryGovernanceActionResult | None]:
    prefix, scope_type, scope_id, task_ref = _parse_memory_id(memory_id)
    if prefix is None or scope_type is None or scope_id is None or task_ref is None:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind=None,
            reason_codes=("memory_id_malformed",),
        )
    if prefix != "memory_task":
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind=prefix,
            reason_codes=("record_kind_unsupported", f"record_kind:{prefix}"),
        )
    if scope_type != "user":
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_task",
            reason_codes=("scope_type_unsupported", f"scope_type:{scope_type}"),
        )
    expected_user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    if scope_id != expected_user_id:
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_task",
            reason_codes=("cross_user_memory_id",),
        )
    if not _active_tasks_available(store):
        return None, _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_task",
            reason_codes=("task_surface_unavailable",),
        )
    tasks = _active_tasks(store, user_id=expected_user_id)
    for task in tasks:
        candidate_memory_id, commitment_backed = _task_memory_id(task, user_id=expected_user_id)
        if candidate_memory_id != memory_id:
            continue
        if not commitment_backed:
            return None, _reject(
                action=action,
                memory_id=memory_id,
                record_kind="memory_task",
                reason_codes=("task_not_actionable", "task_commitment_missing"),
            )
        return task, None
    return None, _reject(
        action=action,
        memory_id=memory_id,
        record_kind="memory_task",
        reason_codes=("task_not_found",),
    )


def _find_replacement_claim(
    *,
    store,
    user_id: str,
    predicate: str,
    replacement_value: str,
    prior_claim_id: str,
) -> BrainClaimRecord | None:
    normalized_value = replacement_value.strip()
    matches = [
        claim
        for claim in store.query_claims(
            temporal_mode="current",
            predicate=predicate,
            scope_type="user",
            scope_id=user_id,
            limit=None,
        )
        if claim.claim_id != prior_claim_id
        and str(claim.object.get("value", "")).strip() == normalized_value
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda claim: (claim.updated_at, claim.claim_id), reverse=True)[0]


def _supersession_exists(store, *, prior_claim_id: str, new_claim_id: str) -> bool:
    return any(
        record.new_claim_id == new_claim_id
        for record in store.claim_supersessions(claim_id=prior_claim_id)
    )


def _apply_task_action(
    *,
    store,
    session_ids,
    task: dict[str, Any],
    action: str,
    memory_id: str,
) -> BrainMemoryGovernanceActionResult:
    user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    title = _optional_text(task.get("title"))
    if title is None:
        return _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_task",
            reason_codes=("task_title_missing",),
        )
    status = "done" if action == "mark_done" else "cancelled"
    rowcount = store.update_task_status(
        user_id=user_id,
        title=title,
        status=status,
        thread_id=_optional_text(getattr(session_ids, "thread_id", "")),
        agent_id=_optional_text(getattr(session_ids, "agent_id", "")),
        session_id=_optional_text(getattr(session_ids, "session_id", "")),
    )
    still_active = any(
        _task_memory_id(candidate, user_id=user_id)[0] == memory_id
        for candidate in _active_tasks(store, user_id=user_id)
    )
    latest_event = None
    thread_id = _optional_text(getattr(session_ids, "thread_id", ""))
    if thread_id is not None:
        latest_event = store.latest_brain_event(
            user_id=user_id,
            thread_id=thread_id,
            event_types=(
                BrainEventType.GOAL_COMPLETED
                if action == "mark_done"
                else BrainEventType.GOAL_CANCELLED,
            ),
        )
    return _result(
        accepted=True,
        applied=bool(rowcount) or not still_active,
        action=action,
        memory_id=memory_id,
        record_kind="memory_task",
        event_id=_optional_text(getattr(latest_event, "event_id", None)),
        reason_codes=(
            "memory_action_accepted",
            "task_scope_validated",
            "task_ref:commitment",
            "task_marked_done" if action == "mark_done" else "task_cancelled",
        ),
    )


def _apply_correct(
    *,
    store,
    session_ids,
    claim: BrainClaimRecord,
    action: str,
    memory_id: str,
    replacement_value: str | None,
    notes: str | None,
    source: str,
) -> BrainMemoryGovernanceActionResult:
    predicate = claim.predicate
    if predicate not in _CORRECTABLE_PREDICATES:
        return _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("correction_predicate_unsupported", f"predicate:{predicate}"),
        )
    replacement = _optional_text(replacement_value)
    if replacement is None:
        return _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("replacement_value_required",),
        )
    user_id = str(getattr(session_ids, "user_id", ""))
    if predicate in _CORRECTABLE_PROFILE_PREDICATES:
        subject = "user"
        rendered_text = render_profile_fact(predicate, replacement)
        singleton = True
    else:
        subject = replacement.lower()
        rendered_text = render_preference_fact(predicate, replacement)
        singleton = False
    store.remember_fact(
        user_id=user_id,
        namespace=predicate,
        subject=subject,
        value={"value": replacement},
        rendered_text=rendered_text,
        confidence=claim.confidence,
        singleton=singleton,
        provenance={
            "source": source,
            "tool_name": "brain_apply_memory_governance",
            "action": "correct",
            "notes": notes or "",
            "prior_claim_id": claim.claim_id,
        },
        source_episode_id=None,
        agent_id=str(getattr(session_ids, "agent_id", "")),
        session_id=str(getattr(session_ids, "session_id", "")),
        thread_id=str(getattr(session_ids, "thread_id", "")),
    )
    replacement_claim = _find_replacement_claim(
        store=store,
        user_id=user_id,
        predicate=predicate,
        replacement_value=replacement,
        prior_claim_id=claim.claim_id,
    )
    if replacement_claim is None:
        return _reject(
            action=action,
            memory_id=memory_id,
            record_kind="memory_claim",
            reason_codes=("replacement_claim_not_found",),
        )

    if not _supersession_exists(
        store,
        prior_claim_id=claim.claim_id,
        new_claim_id=replacement_claim.claim_id,
    ):
        store._claims().supersede_with_existing(
            prior_claim_id=claim.claim_id,
            new_claim_id=replacement_claim.claim_id,
            reason="user_correction",
            source_event_id=replacement_claim.source_event_id,
            event_context=_event_context(
                session_ids=session_ids,
                source=source,
                action=action,
                memory_id=memory_id,
            ),
        )

    if predicate in _CORRECTABLE_PREFERENCE_PREDICATES:
        old_value = _optional_text(claim.object.get("value"))
        if old_value:
            store.forget_facts(
                user_id=user_id,
                namespace=predicate,
                subject=old_value.lower(),
                agent_id=str(getattr(session_ids, "agent_id", "")),
                session_id=str(getattr(session_ids, "session_id", "")),
                thread_id=str(getattr(session_ids, "thread_id", "")),
            )

    refreshed = store._claims().get_claim(claim.claim_id)
    replacement_claim = store._claims().get_claim(replacement_claim.claim_id) or replacement_claim
    return _result(
        accepted=True,
        applied=True,
        action=action,
        memory_id=memory_id,
        record_kind="memory_claim",
        event_id=(
            _optional_text(getattr(refreshed, "last_governance_event_id", None))
            or _optional_text(replacement_claim.last_governance_event_id)
        ),
        replacement_memory_id=_memory_id_for_claim(
            user_id=user_id,
            claim_id=replacement_claim.claim_id,
        ),
        reason_codes=(
            "memory_action_accepted",
            "claim_scope_validated",
            "claim_corrected",
            f"predicate:{predicate}",
        ),
    )


def apply_memory_governance_action(
    *,
    store,
    session_ids,
    memory_id: str,
    action: str,
    replacement_value: str | None = None,
    notes: str | None = None,
    source: str = "memory_governance_actions",
) -> BrainMemoryGovernanceActionResult:
    """Apply one scoped, event-backed memory governance action."""
    normalized_action = _normalize_action(action)
    normalized_memory_id = _normalized_text(memory_id)
    prefix, _scope_type, _scope_id, _record_id = _parse_memory_id(normalized_memory_id)
    raw_prefix = normalized_memory_id.split(":", 1)[0] or None
    record_kind = prefix
    if record_kind is None and raw_prefix in {"memory_claim", "memory_task"}:
        record_kind = raw_prefix
    if normalized_action not in _SUPPORTED_ACTIONS:
        return _reject(
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind=record_kind,
            reason_codes=("action_unsupported",),
        )
    if prefix == "memory_task" or (prefix is None and raw_prefix == "memory_task"):
        if normalized_action not in _TASK_ACTIONS:
            return _reject(
                action=normalized_action,
                memory_id=normalized_memory_id,
                record_kind="memory_task",
                reason_codes=("action_unsupported",),
            )
        task, rejection = _load_scoped_task(
            store=store,
            session_ids=session_ids,
            memory_id=normalized_memory_id,
            action=normalized_action,
        )
        if rejection is not None:
            return rejection
        if task is None:
            return _reject(
                action=normalized_action,
                memory_id=normalized_memory_id,
                record_kind="memory_task",
                reason_codes=("task_not_found",),
            )
        return _apply_task_action(
            store=store,
            session_ids=session_ids,
            task=task,
            action=normalized_action,
            memory_id=normalized_memory_id,
        )
    if prefix == "memory_claim" and normalized_action not in _CLAIM_ACTIONS:
        return _reject(
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            reason_codes=("action_unsupported",),
        )
    claim, rejection = _load_scoped_claim(
        store=store,
        session_ids=session_ids,
        memory_id=normalized_memory_id,
        action=normalized_action,
    )
    if rejection is not None:
        return rejection
    if claim is None:
        return _reject(
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            reason_codes=("claim_not_found",),
        )
    ledger = store._claims()
    context = _event_context(
        session_ids=session_ids,
        source=source,
        action=normalized_action,
        memory_id=normalized_memory_id,
    )

    if normalized_action == "pin":
        if (
            claim.effective_retention_class == BrainClaimRetentionClass.DURABLE.value
            and _USER_PINNED_REASON_CODE in claim.governance_reason_codes
        ):
            return _result(
                accepted=True,
                applied=False,
                action=normalized_action,
                memory_id=normalized_memory_id,
                record_kind="memory_claim",
                event_id=claim.last_governance_event_id,
                reason_codes=(
                    "memory_action_accepted",
                    "claim_scope_validated",
                    "claim_already_pinned",
                    "pin_source:user",
                ),
            )
        updated = ledger.reclassify_claim_retention(
            claim.claim_id,
            retention_class=BrainClaimRetentionClass.DURABLE.value,
            source_event_id=None,
            reason_codes=(*claim.governance_reason_codes, _USER_PINNED_REASON_CODE),
            summary="User pinned memory.",
            notes=notes,
            event_context=context,
        )
        return _result(
            accepted=True,
            applied=True,
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            event_id=updated.last_governance_event_id,
            reason_codes=(
                "memory_action_accepted",
                "claim_scope_validated",
                "claim_pinned",
                "pin_source:user",
            ),
        )

    if normalized_action == "suppress":
        updated = ledger.request_claim_review(
            claim.claim_id,
            source_event_id=None,
            reason_codes=(BrainGovernanceReasonCode.OPERATOR_HOLD.value,),
            summary="User suppressed memory.",
            notes=notes,
            event_context=context,
        )
        return _result(
            accepted=True,
            applied=True,
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            event_id=updated.last_governance_event_id,
            reason_codes=(
                "memory_action_accepted",
                "claim_scope_validated",
                "claim_suppressed",
            ),
        )

    if normalized_action == "mark_stale":
        if claim.effective_currentness_status == BrainClaimCurrentnessStatus.STALE.value:
            return _result(
                accepted=True,
                applied=False,
                action=normalized_action,
                memory_id=normalized_memory_id,
                record_kind="memory_claim",
                event_id=claim.last_governance_event_id,
                reason_codes=(
                    "memory_action_accepted",
                    "claim_scope_validated",
                    "claim_already_stale",
                ),
            )
        updated = ledger.expire_claim(
            claim.claim_id,
            source_event_id=None,
            reason_codes=(BrainGovernanceReasonCode.STALE_WITHOUT_REFRESH.value,),
            summary="User marked memory stale.",
            notes=notes,
            event_context=context,
        )
        return _result(
            accepted=True,
            applied=True,
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            event_id=updated.last_governance_event_id,
            reason_codes=(
                "memory_action_accepted",
                "claim_scope_validated",
                "claim_marked_stale",
            ),
        )

    if normalized_action == "forget":
        rowcount = ledger.revoke_claim(
            claim.claim_id,
            reason="user_forget",
            source_event_id=None,
            reason_codes=(BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value,),
            event_context=context,
        )
        if claim.predicate in _CORRECTABLE_PROFILE_PREDICATES:
            store.forget_facts(
                user_id=str(getattr(session_ids, "user_id", "")),
                namespace=claim.predicate,
                subject="user",
                agent_id=str(getattr(session_ids, "agent_id", "")),
                session_id=str(getattr(session_ids, "session_id", "")),
                thread_id=str(getattr(session_ids, "thread_id", "")),
            )
        elif claim.predicate in _CORRECTABLE_PREFERENCE_PREDICATES:
            old_value = _optional_text(claim.object.get("value"))
            if old_value:
                store.forget_facts(
                    user_id=str(getattr(session_ids, "user_id", "")),
                    namespace=claim.predicate,
                    subject=old_value.lower(),
                    agent_id=str(getattr(session_ids, "agent_id", "")),
                    session_id=str(getattr(session_ids, "session_id", "")),
                    thread_id=str(getattr(session_ids, "thread_id", "")),
                )
        refreshed = ledger.get_claim(claim.claim_id)
        return _result(
            accepted=True,
            applied=bool(rowcount),
            action=normalized_action,
            memory_id=normalized_memory_id,
            record_kind="memory_claim",
            event_id=_optional_text(getattr(refreshed, "last_governance_event_id", None)),
            reason_codes=(
                "memory_action_accepted",
                "claim_scope_validated",
                "claim_forgotten",
            ),
        )

    return _apply_correct(
        store=store,
        session_ids=session_ids,
        claim=claim,
        action=normalized_action,
        memory_id=normalized_memory_id,
        replacement_value=replacement_value,
        notes=notes,
        source=source,
    )


__all__ = [
    "BrainMemoryGovernanceActionResult",
    "apply_memory_governance_action",
]

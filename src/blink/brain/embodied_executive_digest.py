"""Compact digest for the hierarchical embodied executive projection."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_embodied_executive_digest(
    *,
    embodied_executive: dict[str, Any] | None,
    recent_action_events: list[dict[str, Any]] | None = None,
    recent_limit: int = 6,
) -> dict[str, Any]:
    """Build one bounded embodied coordinator digest."""
    projection = dict(embodied_executive or {})
    current_intent = dict(projection.get("current_intent") or {})
    recent_action_envelopes = list(projection.get("recent_action_envelopes") or [])
    recent_execution_traces = list(projection.get("recent_execution_traces") or [])
    recent_recoveries = list(projection.get("recent_recoveries") or [])
    low_level_actions = list(recent_action_events or [])

    low_level_source_counts: Counter[str] = Counter()
    reactive_policy_actions: list[dict[str, Any]] = []
    for record in low_level_actions:
        source = str(record.get("source", "")).strip()
        if source:
            low_level_source_counts[source] += 1
        if source.startswith("policy"):
            reactive_policy_actions.append(record)

    latest_envelope = dict(recent_action_envelopes[0]) if recent_action_envelopes else {}
    current_low_level_executor = (
        str(latest_envelope.get("executor_backend", "")).strip()
        or str(projection.get("current_executor_kind", "")).strip()
        or None
    )
    return {
        "current_intent": current_intent,
        "current_executor_kind": projection.get("current_executor_kind"),
        "current_low_level_executor": current_low_level_executor,
        "last_action_envelope": latest_envelope,
        "intent_kind_counts": dict(projection.get("intent_kind_counts", {})),
        "disposition_counts": dict(projection.get("disposition_counts", {})),
        "trace_status_counts": dict(projection.get("trace_status_counts", {})),
        "mismatch_code_counts": dict(projection.get("mismatch_code_counts", {})),
        "repair_code_counts": dict(projection.get("repair_code_counts", {})),
        "policy_posture_counts": dict(projection.get("policy_posture_counts", {})),
        "recent_execution_trace_count": len(recent_execution_traces),
        "recent_recovery_count": len(recent_recoveries),
        "recent_execution_traces": recent_execution_traces[:recent_limit],
        "recent_recoveries": recent_recoveries[:recent_limit],
        "recent_low_level_embodied_actions": low_level_actions[:recent_limit],
        "recent_reactive_policy_actions": reactive_policy_actions[:recent_limit],
        "low_level_source_counts": _sorted_counter(low_level_source_counts),
    }


__all__ = ["build_embodied_executive_digest"]

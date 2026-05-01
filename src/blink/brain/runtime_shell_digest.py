"""Derived operator-facing digest for runtime-shell controls and artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.counterfactual_rehearsal_digest import build_counterfactual_rehearsal_digest
from blink.brain.embodied_executive_digest import build_embodied_executive_digest
from blink.brain.evals.adapter_promotion import build_adapter_governance_inspection
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.memory_v2.skill_evidence import build_skill_evidence_inspection
from blink.brain.practice_director import BrainPracticeDirectorProjection, build_practice_inspection
from blink.brain.world_model_digest import build_world_model_digest


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_sort_key(event: BrainEventRecord) -> tuple[int, datetime, str]:
    return (
        int(getattr(event, "id", 0)),
        _parse_ts(event.ts) or datetime.min.replace(tzinfo=UTC),
        event.event_id,
    )


def _tail(records: list[dict[str, Any]], *, recent_limit: int) -> list[dict[str, Any]]:
    if len(records) <= recent_limit:
        return list(records)
    return list(records[-recent_limit:])


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _recent_multimodal_rows(
    rows: list[dict[str, Any]],
    *,
    recent_limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            _parse_ts(_normalized_text(row.get("updated_at")) or None)
            or _parse_ts(_normalized_text(row.get("created_at")) or None)
            or datetime.min.replace(tzinfo=UTC),
            _normalized_text(row.get("entry_id")),
        ),
        reverse=True,
    )
    return ordered[:recent_limit]


def _build_multimodal_inspection(
    *,
    multimodal_autobiography: list[dict[str, Any]],
    packet_traces: dict[str, Any] | None,
    recent_limit: int,
) -> dict[str, Any]:
    privacy_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    retention_counts: Counter[str] = Counter()
    latest_source_presence_scope_key: str | None = None
    for row in multimodal_autobiography:
        privacy_class = _normalized_text(row.get("privacy_class"))
        review_state = _normalized_text(row.get("review_state"))
        retention_class = _normalized_text(row.get("retention_class"))
        source_presence_scope_key = _normalized_text(row.get("source_presence_scope_key"))
        if privacy_class:
            privacy_counts[privacy_class] += 1
        if review_state:
            review_counts[review_state] += 1
        if retention_class:
            retention_counts[retention_class] += 1
        if latest_source_presence_scope_key is None and source_presence_scope_key:
            latest_source_presence_scope_key = source_presence_scope_key
    packet_digest = build_context_packet_digest(packet_traces=packet_traces)
    packet_scene_episode_trace_by_task = {
        task: dict(summary.get("scene_episode_trace") or {})
        for task, summary in packet_digest.items()
        if dict(summary.get("scene_episode_trace") or {})
    }
    compact_rows = [
        {
            "entry_id": row.get("entry_id"),
            "rendered_summary": row.get("rendered_summary"),
            "status": row.get("status"),
            "privacy_class": row.get("privacy_class"),
            "review_state": row.get("review_state"),
            "retention_class": row.get("retention_class"),
            "source_presence_scope_key": row.get("source_presence_scope_key"),
            "updated_at": row.get("updated_at"),
            "redacted_at": row.get("redacted_at"),
        }
        for row in multimodal_autobiography
    ]
    recent_scene_episodes = _recent_multimodal_rows(compact_rows, recent_limit=recent_limit)
    recent_redacted_rows = [
        row
        for row in recent_scene_episodes
        if _normalized_text(row.get("privacy_class")) == "redacted"
        or _normalized_text(row.get("redacted_at"))
    ]
    return {
        "latest_source_presence_scope_key": latest_source_presence_scope_key,
        "privacy_counts": _sorted_counter(privacy_counts),
        "review_counts": _sorted_counter(review_counts),
        "retention_counts": _sorted_counter(retention_counts),
        "recent_scene_episodes": recent_scene_episodes,
        "recent_redacted_rows": recent_redacted_rows,
        "packet_scene_episode_trace_by_task": packet_scene_episode_trace_by_task,
    }


def _build_predictive_inspection(
    *,
    predictive_world_model: dict[str, Any] | None,
    packet_traces: dict[str, Any] | None,
) -> dict[str, Any]:
    world_model_digest = build_world_model_digest(
        predictive_world_model=predictive_world_model,
    )
    packet_digest = build_context_packet_digest(packet_traces=packet_traces)
    return {
        **world_model_digest,
        "packet_prediction_trace_by_task": {
            task: dict(summary.get("prediction_trace") or {})
            for task, summary in packet_digest.items()
            if dict(summary.get("prediction_trace") or {})
        },
    }


def _build_rehearsal_inspection(
    *,
    counterfactual_rehearsal: dict[str, Any] | None,
    packet_traces: dict[str, Any] | None,
    recent_limit: int,
) -> dict[str, Any]:
    rehearsal_digest = build_counterfactual_rehearsal_digest(
        counterfactual_rehearsal=counterfactual_rehearsal,
        recent_limit=recent_limit,
    )
    packet_digest = build_context_packet_digest(packet_traces=packet_traces)
    packet_trace_by_task = {
        task: {
            "selected_item_counts": dict(summary.get("selected_item_counts") or {}),
            "drop_reason_counts": dict(summary.get("drop_reason_counts") or {}),
        }
        for task, summary in packet_digest.items()
        if dict(summary.get("selected_item_counts") or {}) or dict(summary.get("drop_reason_counts") or {})
    }
    return {
        **rehearsal_digest,
        "packet_trace_by_task": packet_trace_by_task,
    }


def _build_embodied_inspection(
    *,
    embodied_executive: dict[str, Any] | None,
    recent_action_events: list[dict[str, Any]] | None,
    recent_limit: int,
) -> dict[str, Any]:
    return build_embodied_executive_digest(
        embodied_executive=embodied_executive,
        recent_action_events=recent_action_events,
        recent_limit=recent_limit,
    )


def _build_practice_inspection_payload(
    *,
    practice_director: Any,
    recent_limit: int,
) -> dict[str, Any]:
    if practice_director is None:
        return build_practice_inspection(
            practice_projection=BrainPracticeDirectorProjection(
                scope_key="",
                presence_scope_key="local:presence",
            ),
            recent_limit=recent_limit,
        )
    return build_practice_inspection(
        practice_projection=practice_director,
        recent_limit=recent_limit,
    )


def _build_skill_evidence_inspection_payload(
    *,
    skill_evidence_ledger: Any,
    skill_governance: Any,
    recent_limit: int,
) -> dict[str, Any]:
    return build_skill_evidence_inspection(
        skill_evidence_ledger=skill_evidence_ledger or {},
        skill_governance=skill_governance or {},
        recent_limit=recent_limit,
    )


def _build_adapter_governance_inspection_payload(
    *,
    adapter_governance: Any,
    recent_limit: int,
) -> dict[str, Any]:
    return build_adapter_governance_inspection(
        adapter_governance=adapter_governance or {},
        recent_limit=recent_limit,
    )


def _build_sim_to_real_inspection_payload(
    *,
    adapter_governance: Any,
    recent_limit: int,
) -> dict[str, Any]:
    return build_sim_to_real_digest(
        adapter_governance=adapter_governance or {},
        recent_limit=recent_limit,
    )


def build_runtime_shell_digest(
    *,
    recent_events: list[BrainEventRecord],
    reflection_cycles: list[dict[str, Any]],
    memory_exports: list[dict[str, Any]],
    counterfactual_rehearsal: dict[str, Any] | None = None,
    predictive_world_model: dict[str, Any] | None = None,
    embodied_executive: dict[str, Any] | None = None,
    practice_director: Any = None,
    skill_evidence_ledger: Any = None,
    skill_governance: Any = None,
    adapter_governance: Any = None,
    recent_action_events: list[dict[str, Any]] | None = None,
    multimodal_autobiography: list[dict[str, Any]] | None = None,
    packet_traces: dict[str, Any] | None = None,
    recent_limit: int = 8,
) -> dict[str, Any]:
    """Build one replay-safe digest for shell-backed controls and artifacts."""
    control_counts: Counter[str] = Counter()
    recent_controls: list[dict[str, Any]] = []
    artifact_action_counts: Counter[str] = Counter()
    recent_artifact_actions: list[dict[str, Any]] = []

    for event in sorted(recent_events, key=_event_sort_key):
        if str(event.source or "").strip() != "runtime_shell":
            continue
        if event.event_type not in {
            BrainEventType.GOAL_DEFERRED,
            BrainEventType.GOAL_RESUMED,
        }:
            continue
        payload = _mapping_or_empty(event.payload)
        control = _mapping_or_empty(payload.get("runtime_shell_control"))
        control_kind = str(control.get("control_kind", "")).strip()
        if not control_kind:
            if event.event_type == BrainEventType.GOAL_RESUMED:
                control_kind = "resume"
            elif event.event_type == BrainEventType.GOAL_DEFERRED:
                control_kind = "interrupt"
            else:
                continue
        control_counts[control_kind] += 1
        goal = _mapping_or_empty(payload.get("goal"))
        commitment = _mapping_or_empty(payload.get("commitment"))
        recent_controls.append(
            {
                "control_kind": control_kind,
                "event_type": event.event_type,
                "event_id": event.event_id,
                "ts": event.ts,
                "commitment_id": commitment.get("commitment_id") or goal.get("commitment_id"),
                "goal_id": goal.get("goal_id"),
                "status_before": control.get("status_before"),
                "status_after": control.get("status_after") or commitment.get("status"),
                "reason_summary": control.get("reason_summary"),
            }
        )

    for record in reflection_cycles:
        trigger = str(record.get("trigger", "")).strip()
        if not trigger.startswith("runtime_shell:"):
            continue
        artifact_action_counts["reflection"] += 1
        recent_artifact_actions.append(
            {
                "action_kind": "reflection",
                "cycle_id": record.get("cycle_id"),
                "trigger": trigger,
                "status": record.get("status"),
                "artifact_path": record.get("draft_artifact_path"),
                "ts": record.get("completed_at") or record.get("started_at"),
            }
        )

    for record in memory_exports:
        metadata = _mapping_or_empty(record.get("metadata"))
        if str(metadata.get("source", "")).strip() != "runtime_shell":
            continue
        action_kind = (
            str(metadata.get("action_kind", "")).strip()
            or str(record.get("export_kind", "")).strip()
            or "unknown"
        )
        artifact_action_counts[action_kind] += 1
        recent_artifact_actions.append(
            {
                "action_kind": action_kind,
                "export_kind": record.get("export_kind"),
                "export_id": record.get("id"),
                "path": record.get("path"),
                "generated_at": record.get("generated_at"),
                "ts": record.get("generated_at"),
                "metadata": metadata,
            }
        )

    recent_artifact_actions.sort(
        key=lambda item: (
            _parse_ts(str(item.get("ts", "")).strip() or None)
            or datetime.min.replace(tzinfo=UTC),
            str(item.get("action_kind", "")),
        )
    )

    return {
        "control_counts": _sorted_counter(control_counts),
        "recent_controls": _tail(recent_controls, recent_limit=recent_limit),
        "artifact_action_counts": _sorted_counter(artifact_action_counts),
        "recent_artifact_actions": _tail(recent_artifact_actions, recent_limit=recent_limit),
        "predictive_inspection": _build_predictive_inspection(
            predictive_world_model=(
                dict(predictive_world_model)
                if isinstance(predictive_world_model, dict)
                else predictive_world_model
            ),
            packet_traces=packet_traces,
        ),
        "rehearsal_inspection": _build_rehearsal_inspection(
            counterfactual_rehearsal=(
                dict(counterfactual_rehearsal)
                if isinstance(counterfactual_rehearsal, dict)
                else counterfactual_rehearsal
            ),
            packet_traces=packet_traces,
            recent_limit=recent_limit,
        ),
        "embodied_inspection": _build_embodied_inspection(
            embodied_executive=(
                dict(embodied_executive)
                if isinstance(embodied_executive, dict)
                else embodied_executive
            ),
            recent_action_events=[
                dict(record)
                for record in (recent_action_events or [])
                if isinstance(record, dict)
            ],
            recent_limit=recent_limit,
        ),
        "practice_inspection": _build_practice_inspection_payload(
            practice_director=practice_director,
            recent_limit=recent_limit,
        ),
        "skill_evidence_inspection": _build_skill_evidence_inspection_payload(
            skill_evidence_ledger=skill_evidence_ledger,
            skill_governance=skill_governance,
            recent_limit=recent_limit,
        ),
        "adapter_governance_inspection": _build_adapter_governance_inspection_payload(
            adapter_governance=adapter_governance,
            recent_limit=recent_limit,
        ),
        "sim_to_real_inspection": _build_sim_to_real_inspection_payload(
            adapter_governance=adapter_governance,
            recent_limit=recent_limit,
        ),
        "multimodal_inspection": _build_multimodal_inspection(
            multimodal_autobiography=[
                dict(record)
                for record in (multimodal_autobiography or [])
                if isinstance(record, dict)
            ],
            packet_traces=packet_traces,
            recent_limit=recent_limit,
        ),
    }


__all__ = ["build_runtime_shell_digest"]

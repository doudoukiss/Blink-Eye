"""Atheris harness for core projection record parsing.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_core_projection_records.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.events import BrainEventRecord
    from blink.brain.memory_v2 import BrainMultimodalAutobiographyRecord
    from blink.brain.memory_v2.autobiography import BrainAutobiographicalEntryRecord
    from blink.brain.memory_v2.multimodal_autobiography import (
        distill_scene_episode,
        parse_multimodal_autobiography_record,
    )
    from blink.brain.projections import (
        BrainBlockedReason,
        BrainCommitmentProjection,
        BrainCommitmentRecord,
        BrainCommitmentWakeTrigger,
        BrainEngagementStateProjection,
        BrainGoal,
        BrainGoalStep,
        BrainHeartbeatProjection,
        BrainPlanProposal,
        BrainPlanProposalDecision,
        BrainRelationshipStateProjection,
        BrainSceneStateProjection,
        BrainSceneWorldProjection,
        BrainWakeCondition,
        BrainWorkingContextProjection,
    )


def _exercise_roundtrip(
    parser: Callable[[dict[str, Any] | None], Any],
    payload: dict[str, Any],
) -> None:
    try:
        record = parser(payload)
    except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
        return
    if record is None:
        return
    roundtrip = parser(record.as_dict())
    assert roundtrip is not None
    assert roundtrip.as_dict() == record.as_dict()


def _event_record(payload: dict[str, Any], *, index: int) -> BrainEventRecord:
    return BrainEventRecord(
        id=int(payload.get("id", index + 1)),
        event_id=str(payload.get("event_id", f"evt-{index}")),
        event_type=str(payload.get("event_type", "scene.changed")),
        ts=str(payload.get("ts", f"2026-01-01T00:00:0{index}+00:00")),
        agent_id=str(payload.get("agent_id", "agent-fuzz")),
        user_id=str(payload.get("user_id", "user-fuzz")),
        session_id=str(payload.get("session_id", "session-fuzz")),
        thread_id=str(payload.get("thread_id", "thread-fuzz")),
        source=str(payload.get("source", "atheris")),
        correlation_id=payload.get("correlation_id"),
        causal_parent_id=payload.get("causal_parent_id"),
        confidence=float(payload.get("confidence", 1.0)),
        payload_json=json.dumps(payload.get("payload", {}), ensure_ascii=False),
        tags_json=json.dumps(payload.get("tags", []), ensure_ascii=False),
    )


def _autobiographical_entry_record(payload: dict[str, Any]) -> BrainAutobiographicalEntryRecord:
    content = payload.get("content")
    if not isinstance(content, dict):
        content = {"value": structured_inputs.bound_jsonish_value(content)}
    return BrainAutobiographicalEntryRecord(
        entry_id=str(payload.get("entry_id", "entry-fuzz")).strip() or "entry-fuzz",
        scope_type=str(payload.get("scope_type", "presence")).strip() or "presence",
        scope_id=str(payload.get("scope_id", "browser:presence")).strip() or "browser:presence",
        entry_kind=str(payload.get("entry_kind", "scene_episode")).strip() or "scene_episode",
        rendered_summary=str(payload.get("rendered_summary", "Scene episode observed.")).strip()
        or "Scene episode observed.",
        content_json=json.dumps(content, ensure_ascii=False, sort_keys=True),
        status=str(payload.get("status", "current")).strip() or "current",
        salience=float(payload.get("salience", 0.0)),
        source_episode_ids_json=json.dumps(
            structured_inputs.as_list(payload.get("source_episode_ids", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        source_claim_ids_json=json.dumps(
            structured_inputs.as_list(payload.get("source_claim_ids", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        source_event_ids_json=json.dumps(
            structured_inputs.as_list(payload.get("source_event_ids", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        modality=str(payload.get("modality", "scene_world")).strip() or None,
        review_state=str(payload.get("review_state", "none")).strip() or None,
        retention_class=str(payload.get("retention_class", "session")).strip() or None,
        privacy_class=str(payload.get("privacy_class", "standard")).strip() or None,
        governance_reason_codes_json=json.dumps(
            structured_inputs.as_list(payload.get("governance_reason_codes", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        last_governance_event_id=payload.get("last_governance_event_id"),
        source_presence_scope_key=payload.get("source_presence_scope_key"),
        source_scene_entity_ids_json=json.dumps(
            structured_inputs.as_list(payload.get("source_scene_entity_ids", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        source_scene_affordance_ids_json=json.dumps(
            structured_inputs.as_list(payload.get("source_scene_affordance_ids", [])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        redacted_at=payload.get("redacted_at"),
        supersedes_entry_id=payload.get("supersedes_entry_id"),
        valid_from=str(payload.get("valid_from", "2026-01-01T00:00:00+00:00")).strip()
        or "2026-01-01T00:00:00+00:00",
        valid_to=payload.get("valid_to"),
        created_at=str(payload.get("created_at", "2026-01-01T00:00:00+00:00")).strip()
        or "2026-01-01T00:00:00+00:00",
        updated_at=str(payload.get("updated_at", "2026-01-01T00:00:00+00:00")).strip()
        or "2026-01-01T00:00:00+00:00",
    )


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.as_mapping(structured_inputs.decode_jsonish_input(data), field_name="projection")

    _exercise_roundtrip(
        BrainBlockedReason.from_dict,
        structured_inputs.as_mapping(root.get("blocked_reason", root), field_name="blocked_reason"),
    )
    _exercise_roundtrip(
        BrainWakeCondition.from_dict,
        structured_inputs.as_mapping(root.get("wake_condition", root), field_name="wake_condition"),
    )
    _exercise_roundtrip(
        BrainCommitmentWakeTrigger.from_dict,
        structured_inputs.as_mapping(root.get("wake_trigger", root), field_name="wake_trigger"),
    )
    _exercise_roundtrip(
        BrainPlanProposal.from_dict,
        structured_inputs.as_mapping(root.get("plan_proposal", root), field_name="plan_proposal"),
    )
    _exercise_roundtrip(
        BrainPlanProposalDecision.from_dict,
        structured_inputs.as_mapping(root.get("plan_decision", root), field_name="plan_decision"),
    )
    _exercise_roundtrip(
        BrainGoalStep.from_dict,
        structured_inputs.as_mapping(root.get("goal_step", root), field_name="goal_step"),
    )
    _exercise_roundtrip(
        BrainGoal.from_dict,
        structured_inputs.as_mapping(root.get("goal", root), field_name="goal"),
    )
    _exercise_roundtrip(
        BrainWorkingContextProjection.from_dict,
        structured_inputs.as_mapping(root.get("working_context", root), field_name="working_context"),
    )
    _exercise_roundtrip(
        BrainCommitmentRecord.from_dict,
        structured_inputs.as_mapping(root.get("commitment_record", root), field_name="commitment_record"),
    )
    _exercise_roundtrip(
        BrainCommitmentProjection.from_dict,
        structured_inputs.as_mapping(root.get("commitment_projection", root), field_name="commitment_projection"),
    )
    _exercise_roundtrip(
        BrainSceneStateProjection.from_dict,
        structured_inputs.as_mapping(root.get("scene_state", root), field_name="scene_state"),
    )
    _exercise_roundtrip(
        BrainEngagementStateProjection.from_dict,
        structured_inputs.as_mapping(root.get("engagement_state", root), field_name="engagement_state"),
    )
    _exercise_roundtrip(
        BrainRelationshipStateProjection.from_dict,
        structured_inputs.as_mapping(root.get("relationship_state", root), field_name="relationship_state"),
    )
    _exercise_roundtrip(
        BrainHeartbeatProjection.from_dict,
        structured_inputs.as_mapping(root.get("heartbeat", root), field_name="heartbeat"),
    )
    _exercise_roundtrip(
        BrainMultimodalAutobiographyRecord.from_dict,
        structured_inputs.as_mapping(
            root.get("multimodal_autobiography_record", root),
            field_name="multimodal_autobiography_record",
        ),
    )
    for payload in structured_inputs.multimodal_autobiography_inputs(root):
        typed = parse_multimodal_autobiography_record(_autobiographical_entry_record(payload))
        if typed is None:
            continue
        roundtrip = BrainMultimodalAutobiographyRecord.from_dict(typed.as_dict())
        assert roundtrip is not None
        assert roundtrip.as_dict() == typed.as_dict()

    scene_world_state = BrainSceneWorldProjection.from_dict(
        structured_inputs.active_state_projection_mapping(root)["scene_world_state"]
    )
    distilled = distill_scene_episode(
        scene_world_state=scene_world_state,
        recent_events=[
            _event_record(payload, index=index)
            for index, payload in enumerate(structured_inputs.scene_event_inputs(root))
        ],
        presence_scope_key=scene_world_state.scope_id or "browser:presence",
        reference_ts=str(root.get("reference_ts", "2026-01-01T00:00:03+00:00")),
    )
    if distilled is not None:
        json.dumps(distilled.content, ensure_ascii=False, sort_keys=True)
        derived = BrainMultimodalAutobiographyRecord.from_dict(
            {
                "entry_id": "entry-derived",
                "scope_type": "presence",
                "scope_id": scene_world_state.scope_id or "browser:presence",
                "entry_kind": "scene_episode",
                "modality": "scene_world",
                "review_state": distilled.review_state,
                "retention_class": distilled.retention_class,
                "privacy_class": distilled.privacy_class,
                "governance_reason_codes": list(distilled.governance_reason_codes),
                "source_presence_scope_key": distilled.source_presence_scope_key,
                "source_scene_entity_ids": list(distilled.source_scene_entity_ids),
                "source_scene_affordance_ids": list(distilled.source_scene_affordance_ids),
                "rendered_summary": distilled.rendered_summary,
                "content": dict(distilled.content),
                "status": "current",
                "salience": distilled.salience,
                "source_event_ids": list(distilled.source_event_ids),
                "valid_from": distilled.valid_from,
                "created_at": distilled.valid_from,
                "updated_at": distilled.valid_from,
            }
        )
        assert derived is not None
        assert derived.content["semantic_fingerprint"] == distilled.semantic_fingerprint


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

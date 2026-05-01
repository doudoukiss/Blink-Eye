"""Atheris harness for replay artifact ingestion and compact projection loaders.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_replay_artifacts.py -atheris_runs=500
"""

from __future__ import annotations

import json
import sys
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.autonomy import BrainAutonomyLedgerProjection
    from blink.brain.autonomy_digest import build_autonomy_digest
    from blink.brain.context_packet_digest import build_context_packet_digest
    from blink.brain.continuity_governance_report import build_continuity_governance_report
    from blink.brain.events import BrainEventRecord
    from blink.brain.executive_policy_audit import build_executive_policy_audit
    from blink.brain.memory_v2 import (
        BrainClaimGovernanceProjection,
        BrainMultimodalAutobiographyRecord,
        build_multimodal_autobiography_digest,
        distill_scene_episode,
    )
    from blink.brain.memory_v2.autobiography import BrainAutobiographicalEntryRecord
    from blink.brain.memory_v2.dossiers import BrainContinuityDossierProjection
    from blink.brain.memory_v2.graph import BrainContinuityGraphProjection
    from blink.brain.memory_v2.multimodal_autobiography import parse_multimodal_autobiography_record
    from blink.brain.memory_v2.procedural import BrainProceduralTraceProjection
    from blink.brain.memory_v2.skills import BrainProceduralSkillProjection
    from blink.brain.planning_digest import build_planning_digest
    from blink.brain.procedural_skill_governance_report import (
        build_procedural_skill_governance_report,
    )
    from blink.brain.projections import (
        BrainAgendaProjection,
        BrainCommitmentProjection,
        BrainSceneWorldProjection,
    )
    from blink.brain.reevaluation_digest import build_reevaluation_digest
    from blink.brain.replay_support import (
        append_replay_event_payloads,
        materialize_replayed_events,
    )
    from blink.brain.runtime_shell_digest import build_runtime_shell_digest
    from blink.brain.session import BrainSessionIds
    from blink.brain.store import BrainStore
    from blink.brain.wake_digest import build_wake_digest


def _exercise_projection_loader(
    parser,
    payload: dict[str, Any],
) -> None:
    try:
        projection = parser(payload)
    except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
        return
    if projection is None:
        return
    roundtrip = parser(projection.as_dict())
    assert roundtrip is not None
    assert roundtrip.as_dict() == projection.as_dict()


def _event_record(payload: dict[str, Any], *, index: int) -> BrainEventRecord:
    return BrainEventRecord(
        id=int(payload.get("id", index + 1)),
        event_id=str(payload.get("event_id", f"evt-{index}")),
        event_type=str(payload.get("event_type", "goal.candidate.created")),
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
    artifact = structured_inputs.replay_artifact_mapping(
        structured_inputs.decode_jsonish_input(data)
    )
    root = artifact

    if "continuity_graph" in root:
        _exercise_projection_loader(
            BrainContinuityGraphProjection.from_dict,
            structured_inputs.as_mapping(root.get("continuity_graph"), field_name="continuity_graph"),
        )
    if "continuity_dossiers" in root:
        _exercise_projection_loader(
            BrainContinuityDossierProjection.from_dict,
            structured_inputs.as_mapping(
                root.get("continuity_dossiers"),
                field_name="continuity_dossiers",
            ),
        )
    if "claim_governance" in root:
        _exercise_projection_loader(
            BrainClaimGovernanceProjection.from_dict,
            structured_inputs.as_mapping(
                root.get("claim_governance"),
                field_name="claim_governance",
            ),
        )
    if "procedural_traces" in root:
        _exercise_projection_loader(
            BrainProceduralTraceProjection.from_dict,
            structured_inputs.as_mapping(root.get("procedural_traces"), field_name="procedural_traces"),
        )
    if "procedural_skills" in root:
        _exercise_projection_loader(
            BrainProceduralSkillProjection.from_dict,
            structured_inputs.as_mapping(root.get("procedural_skills"), field_name="procedural_skills"),
        )
    multimodal_records: list[BrainMultimodalAutobiographyRecord] = []
    for item in structured_inputs.multimodal_autobiography_inputs(root):
        try:
            record = BrainMultimodalAutobiographyRecord.from_dict(
                structured_inputs.as_mapping(item, field_name="multimodal_autobiography")
            )
        except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
            continue
        if record is not None:
            multimodal_records.append(record)
        typed = parse_multimodal_autobiography_record(_autobiographical_entry_record(item))
        if typed is not None:
            multimodal_records.append(typed)
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
        reference_ts="2026-01-01T00:00:03+00:00",
    )
    if distilled is not None:
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
        if derived is not None:
            multimodal_records.append(derived)
    json.dumps(
        build_multimodal_autobiography_digest(multimodal_records),
        ensure_ascii=False,
        sort_keys=True,
    )

    operator_inputs = structured_inputs.operator_artifact_inputs(root)
    autonomy_ledger = BrainAutonomyLedgerProjection.from_dict(operator_inputs["autonomy_ledger"])
    agenda = BrainAgendaProjection.from_dict(operator_inputs["agenda"])
    commitment_projection = BrainCommitmentProjection.from_dict(
        operator_inputs["commitment_projection"]
    )
    recent_events = [
        _event_record(payload, index=index)
        for index, payload in enumerate(operator_inputs["recent_events"])
    ]
    autonomy_digest = build_autonomy_digest(
        autonomy_ledger=autonomy_ledger,
        agenda=agenda,
    )
    reevaluation_digest = build_reevaluation_digest(
        autonomy_ledger=autonomy_ledger,
        recent_events=recent_events,
    )
    wake_digest = build_wake_digest(
        commitment_projection=commitment_projection,
        recent_events=recent_events,
    )
    planning_digest = build_planning_digest(
        agenda=agenda,
        commitment_projection=commitment_projection,
        recent_events=recent_events,
    )
    procedural_skill_governance_report = build_procedural_skill_governance_report(
        procedural_skills=operator_inputs["procedural_skills"],
        procedural_traces=operator_inputs["procedural_traces"],
        planning_digest=planning_digest,
    )
    json.dumps(
        build_executive_policy_audit(
            autonomy_digest=autonomy_digest,
            reevaluation_digest=reevaluation_digest,
            wake_digest=wake_digest,
            planning_digest=planning_digest,
            procedural_skill_governance_report=procedural_skill_governance_report,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
    json.dumps(
        build_runtime_shell_digest(
            recent_events=recent_events,
            reflection_cycles=operator_inputs["reflection_cycles"],
            memory_exports=operator_inputs["memory_exports"],
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    session = structured_inputs.normalized_session(root.get("session"))
    session_ids = BrainSessionIds(
        agent_id=session["agent_id"],
        user_id=session["user_id"],
        session_id=session["session_id"],
        thread_id=session["thread_id"],
    )

    store = BrainStore(path=":memory:")
    try:
        json.dumps(
            build_context_packet_digest(packet_traces=dict(root.get("packet_traces") or {})),
            ensure_ascii=False,
            sort_keys=True,
        )
        json.dumps(
            build_continuity_governance_report(
                continuity_dossiers=root.get("continuity_dossiers"),
                continuity_graph=root.get("continuity_graph"),
                claim_governance=root.get("claim_governance"),
                packet_traces=dict(root.get("packet_traces") or {}),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            appended = append_replay_event_payloads(
                store=store,
                session_ids=session_ids,
                payloads=structured_inputs.coerce_replay_events(root),
            )
        except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
            return
        materialize_replayed_events(
            store=store,
            session_ids=session_ids,
            events=appended,
        )
    finally:
        store.close()


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

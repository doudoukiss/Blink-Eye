"""Atheris harness for continuity dossier projection parsing.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_dossier_projection.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.memory_v2.dossiers import (
        BrainContinuityDossierEvidenceRef,
        BrainContinuityDossierFactRecord,
        BrainContinuityDossierGovernanceRecord,
        BrainContinuityDossierIssueRecord,
        BrainContinuityDossierProjection,
        BrainContinuityDossierRecord,
        BrainContinuityDossierTaskAvailability,
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


def _exercise_evidence(payload: dict[str, Any]) -> None:
    try:
        record = BrainContinuityDossierEvidenceRef.from_dict(payload)
    except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
        return
    evidence_payload = record.as_dict()
    json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True)
    roundtrip = BrainContinuityDossierEvidenceRef.from_dict(evidence_payload)
    assert roundtrip.as_dict() == evidence_payload


def TestOneInput(data: bytes) -> None:
    value = structured_inputs.decode_jsonish_input(data)
    root = structured_inputs.as_mapping(value, field_name="dossier")
    canonical_dossier = {
        "dossier_id": "dossier_seed",
        "kind": "scene_world",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "title": "Scene World",
        "summary": "limited; dock pad visible; affordances: vision.inspect",
        "status": "current",
        "freshness": "fresh",
        "contradiction": "uncertain",
        "support_strength": 0.75,
        "summary_evidence": {
            "source_event_ids": ["evt-scene-1"],
            "graph_node_ids": ["graph_node_scene"],
            "graph_edge_ids": ["graph_edge_scene"],
        },
        "source_block_ids": ["block-1"],
        "source_commitment_ids": ["commitment-1"],
        "source_plan_proposal_ids": ["proposal-1"],
        "source_skill_ids": ["skill-1"],
        "source_scene_entity_ids": ["entity-1"],
        "source_scene_affordance_ids": ["affordance-1"],
        "key_current_facts": [
            {
                "kind": "scene_episode",
                "summary": "limited; person near the desk; affordances: vision.inspect",
                "support_strength": 0.8,
                "evidence": {
                    "source_event_ids": ["evt-scene-1", "evt-engagement-1"],
                    "graph_node_ids": [
                        "graph_node_autobio_scene",
                        "graph_node_scene_entity",
                    ],
                    "graph_edge_ids": ["graph_edge_autobio_entity"],
                },
            }
        ],
        "recent_changes": [
            {
                "kind": "scene_episode_redacted",
                "summary": "[redacted scene episode]",
                "support_strength": 0.6,
                "evidence": {
                    "source_event_ids": ["evt-redacted-1"],
                    "graph_node_ids": ["graph_node_autobio_scene_old"],
                    "graph_edge_ids": [],
                },
            }
        ],
    }

    evidence_payloads = [root]
    if "evidence" in root:
        evidence_payloads.append(
            structured_inputs.as_mapping(root.get("evidence"), field_name="evidence")
        )
    for payload in evidence_payloads:
        _exercise_evidence(payload)

    fact_payloads = [root]
    if "fact" in root:
        fact_payloads.append(structured_inputs.as_mapping(root.get("fact"), field_name="fact"))
    for payload in fact_payloads:
        _exercise_roundtrip(BrainContinuityDossierFactRecord.from_dict, payload)

    issue_payloads = [root]
    if "issue" in root:
        issue_payloads.append(structured_inputs.as_mapping(root.get("issue"), field_name="issue"))
    for payload in issue_payloads:
        _exercise_roundtrip(BrainContinuityDossierIssueRecord.from_dict, payload)

    task_availability_payloads = [root]
    if "task_availability" in root:
        task_availability_payloads.append(
            structured_inputs.as_mapping(
                root.get("task_availability"),
                field_name="task_availability",
            )
        )
    for payload in task_availability_payloads:
        _exercise_roundtrip(BrainContinuityDossierTaskAvailability.from_dict, payload)

    governance_payloads = [root]
    if "governance" in root:
        governance_payloads.append(
            structured_inputs.as_mapping(root.get("governance"), field_name="governance")
        )
    for payload in governance_payloads:
        _exercise_roundtrip(BrainContinuityDossierGovernanceRecord.from_dict, payload)

    dossier_payloads = [root]
    if "dossier" in root:
        dossier_payloads.append(
            structured_inputs.as_mapping(root.get("dossier"), field_name="dossier")
        )
    dossier_payloads.append(canonical_dossier)
    for payload in dossier_payloads:
        _exercise_roundtrip(BrainContinuityDossierRecord.from_dict, payload)

    projection_payloads = [root]
    if "projection" in root:
        projection_payloads.append(
            structured_inputs.as_mapping(root.get("projection"), field_name="projection")
        )
    projection_payloads.append(
        {
            "scope_type": "user",
            "scope_id": "user-1",
            "dossiers": [canonical_dossier],
        }
    )
    for payload in projection_payloads:
        _exercise_roundtrip(BrainContinuityDossierProjection.from_dict, payload)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

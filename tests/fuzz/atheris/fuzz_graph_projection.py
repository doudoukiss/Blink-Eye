"""Atheris harness for continuity graph projection parsing.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_graph_projection.py -atheris_runs=1000
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.memory_v2.graph import (
        BrainContinuityGraphEdgeRecord,
        BrainContinuityGraphNodeRecord,
        BrainContinuityGraphProjection,
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


def TestOneInput(data: bytes) -> None:
    value = structured_inputs.decode_jsonish_input(data)
    root = structured_inputs.as_mapping(value, field_name="graph")
    canonical_node = {
        "node_id": "graph_node_skill",
        "kind": "procedural_skill",
        "backing_record_id": "skill-1",
        "summary": "Refresh proof lane",
        "status": "active",
        "scope_type": "thread",
        "scope_id": "thread-1",
        "details": {
            "skill_family_key": "maintenance.review_memory_health",
            "supporting_plan_proposal_ids": ["proposal-1"],
            "supporting_commitment_ids": ["commitment-1"],
        },
    }
    canonical_edge = {
        "edge_id": "graph_edge_skill_plan",
        "kind": "procedural_skill_supports_plan_proposal",
        "from_node_id": "graph_node_skill",
        "to_node_id": "graph_node_plan",
        "status": "supported",
    }
    canonical_multimodal_node = {
        "node_id": "graph_node_autobio_scene",
        "kind": "autobiography_entry",
        "backing_record_id": "entry-scene-current",
        "summary": "limited; person near the desk; affordances: vision.inspect",
        "status": "current",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "source_event_ids": ["evt-scene-1", "evt-engagement-1"],
        "details": {
            "entry_kind": "scene_episode",
            "modality": "scene_world",
            "privacy_class": "sensitive",
        },
    }
    canonical_scene_entity_node = {
        "node_id": "graph_node_scene_entity",
        "kind": "scene_world_entity",
        "backing_record_id": "person-1",
        "summary": "A person is near the desk.",
        "status": "active",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "source_event_ids": ["evt-engagement-1"],
    }
    canonical_scene_affordance_node = {
        "node_id": "graph_node_scene_affordance",
        "kind": "scene_world_affordance",
        "backing_record_id": "aff-1",
        "summary": "Inspect the visible desk area.",
        "status": "available",
        "scope_type": "presence",
        "scope_id": "browser:presence",
        "source_event_ids": ["evt-scene-1"],
    }
    canonical_multimodal_edge = {
        "edge_id": "graph_edge_autobio_entity",
        "kind": "autobiography.references.scene_world_entity",
        "from_node_id": "graph_node_autobio_scene",
        "to_node_id": "graph_node_scene_entity",
        "status": "supported",
    }
    canonical_affordance_edge = {
        "edge_id": "graph_edge_entity_affordance",
        "kind": "scene_world_entity_has_affordance",
        "from_node_id": "graph_node_scene_entity",
        "to_node_id": "graph_node_scene_affordance",
        "status": "supported",
    }

    node_payloads = [root]
    if "node" in root:
        node_payloads.append(structured_inputs.as_mapping(root.get("node"), field_name="node"))
    node_payloads.append(canonical_node)
    for payload in node_payloads:
        _exercise_roundtrip(BrainContinuityGraphNodeRecord.from_dict, payload)

    edge_payloads = [root]
    if "edge" in root:
        edge_payloads.append(structured_inputs.as_mapping(root.get("edge"), field_name="edge"))
    edge_payloads.append(canonical_edge)
    for payload in edge_payloads:
        _exercise_roundtrip(BrainContinuityGraphEdgeRecord.from_dict, payload)

    projection_payloads = [root]
    if "projection" in root:
        projection_payloads.append(
            structured_inputs.as_mapping(root.get("projection"), field_name="projection")
        )
    projection_payloads.append(
        {
            "scope_type": "user",
            "scope_id": "user-1",
            "nodes": [
                canonical_node,
                {
                    "node_id": "graph_node_plan",
                    "kind": "plan_proposal",
                    "backing_record_id": "proposal-1",
                    "summary": "Refresh the proof lane.",
                    "status": "adopted",
                },
                canonical_multimodal_node,
                canonical_scene_entity_node,
                canonical_scene_affordance_node,
            ],
            "edges": [
                canonical_edge,
                canonical_multimodal_edge,
                canonical_affordance_edge,
            ],
        }
    )
    for payload in projection_payloads:
        _exercise_roundtrip(BrainContinuityGraphProjection.from_dict, payload)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

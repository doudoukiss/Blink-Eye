"""Compact operator digest for the temporal continuity graph."""

from __future__ import annotations

from collections import Counter
from typing import Any

_CURRENT_NODE_KINDS = (
    "claim",
    "autobiography_entry",
    "commitment",
    "plan_proposal",
    "core_memory_block",
    "procedural_skill",
    "scene_world_entity",
    "scene_world_affordance",
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, dict):
            return payload
    return {}


def _compact_node(record: dict[str, Any]) -> dict[str, Any]:
    details = dict(record.get("details") or {})
    return {
        "node_id": record.get("node_id"),
        "backing_record_id": record.get("backing_record_id"),
        "summary": record.get("summary"),
        "status": record.get("status"),
        "details": {
            key: details.get(key)
            for key in (
                "predicate",
                "entry_kind",
                "commitment_id",
                "goal_id",
                "plan_proposal_id",
                "goal_title",
                "title",
                "commitment_title",
                "review_policy",
                "block_kind",
                "skill_family_key",
                "entity_kind",
                "capability_family",
            )
            if details.get(key) is not None
        },
    }


def build_continuity_graph_digest(
    *,
    continuity_graph: Any,
    max_nodes_per_kind: int = 6,
    max_supersession_links: int = 8,
    max_commitment_plan_links: int = 8,
) -> dict[str, Any]:
    """Build a compact operator digest from the raw continuity graph."""
    graph = _as_mapping(continuity_graph)
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    current_node_ids = set(graph.get("current_node_ids") or [])
    historical_node_ids = set(graph.get("historical_node_ids") or [])
    stale_node_ids = set(graph.get("stale_node_ids") or [])
    superseded_node_ids = set(graph.get("superseded_node_ids") or [])
    node_by_id = {
        str(record.get("node_id")): record
        for record in nodes
        if str(record.get("node_id", "")).strip()
    }

    def _status_counts(node_ids: set[str]) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    str(node_by_id[node_id].get("kind"))
                    for node_id in node_ids
                    if node_id in node_by_id and str(node_by_id[node_id].get("kind", "")).strip()
                ).items()
            )
        )

    current_nodes_by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind in _CURRENT_NODE_KINDS:
        selected = [
            _compact_node(record)
            for record in sorted(
                (
                    node_by_id[node_id]
                    for node_id in current_node_ids
                    if node_id in node_by_id and node_by_id[node_id].get("kind") == kind
                ),
                key=lambda item: (
                    str(item.get("summary", "")),
                    str(item.get("backing_record_id", "")),
                    str(item.get("node_id", "")),
                ),
            )[:max_nodes_per_kind]
        ]
        current_nodes_by_kind[kind] = selected

    def _edge_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(record.get("valid_from") or ""),
            str(record.get("edge_id") or ""),
            str(record.get("from_node_id") or ""),
            str(record.get("to_node_id") or ""),
        )

    recent_supersession_links = [
        {
            "edge_id": record.get("edge_id"),
            "kind": record.get("kind"),
            "from_backing_record_id": node_by_id.get(str(record.get("from_node_id")), {}).get(
                "backing_record_id"
            ),
            "to_backing_record_id": node_by_id.get(str(record.get("to_node_id")), {}).get(
                "backing_record_id"
            ),
            "status": record.get("status"),
            "valid_from": record.get("valid_from"),
            "valid_to": record.get("valid_to"),
        }
        for record in sorted(
            (
                edge
                for edge in edges
                if edge.get("kind") in {"supersedes", "plan_proposal_supersedes"}
            ),
            key=_edge_sort_key,
            reverse=True,
        )[:max_supersession_links]
    ]

    current_commitment_plan_links = []
    for record in sorted(
        (
            edge
            for edge in edges
            if edge.get("kind")
            in {"commitment_has_plan_proposal", "plan_proposal_adopted_into_commitment"}
        ),
        key=_edge_sort_key,
        reverse=True,
    ):
        from_node = node_by_id.get(str(record.get("from_node_id"))) or {}
        to_node = node_by_id.get(str(record.get("to_node_id"))) or {}
        commitment_node = from_node if from_node.get("kind") == "commitment" else to_node
        proposal_node = from_node if from_node.get("kind") == "plan_proposal" else to_node
        if not commitment_node or not proposal_node:
            continue
        current_commitment_plan_links.append(
            {
                "edge_id": record.get("edge_id"),
                "kind": record.get("kind"),
                "commitment_id": commitment_node.get("backing_record_id"),
                "commitment_summary": commitment_node.get("summary"),
                "plan_proposal_id": proposal_node.get("backing_record_id"),
                "plan_summary": proposal_node.get("summary"),
                "plan_status": proposal_node.get("status"),
            }
        )
        if len(current_commitment_plan_links) >= max_commitment_plan_links:
            break

    node_kind_counts = dict(
        sorted(
            Counter(
                str(record.get("kind"))
                for record in nodes
                if str(record.get("kind", "")).strip()
            ).items()
        )
    )
    edge_kind_counts = dict(
        sorted(
            Counter(
                str(record.get("kind"))
                for record in edges
                if str(record.get("kind", "")).strip()
            ).items()
        )
    )
    return {
        "scope_type": graph.get("scope_type"),
        "scope_id": graph.get("scope_id"),
        "node_counts": dict(graph.get("node_counts") or node_kind_counts),
        "edge_counts": dict(graph.get("edge_counts") or edge_kind_counts),
        "current_node_kind_counts": _status_counts(current_node_ids),
        "historical_node_kind_counts": _status_counts(historical_node_ids),
        "stale_node_kind_counts": _status_counts(stale_node_ids),
        "superseded_node_kind_counts": _status_counts(superseded_node_ids),
        "edge_kind_counts": edge_kind_counts,
        "current_nodes_by_kind": current_nodes_by_kind,
        "recent_supersession_links": recent_supersession_links,
        "current_commitment_plan_links": current_commitment_plan_links,
        "evidence_anchor_counts": {
            "event_anchor": node_kind_counts.get("event_anchor", 0),
            "episode_anchor": node_kind_counts.get("episode_anchor", 0),
        },
    }


__all__ = ["build_continuity_graph_digest"]

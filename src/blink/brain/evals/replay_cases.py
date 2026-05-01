"""Checked-in replay regression cases for Blink continuity state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blink.brain.core import BrainEventRecord
from blink.brain.evals.memory_state import (
    BrainContinuityExpectedState,
    compare_continuity_state,
)
from blink.brain.procedural_qa_report import build_procedural_qa_state_excerpt
from blink.brain.replay import BrainReplayHarness, BrainReplayScenario
from blink.brain.session import BrainSessionIds
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


@dataclass(frozen=True)
class BrainReplayRegressionCase:
    """One checked-in replay regression case."""

    name: str
    description: str
    scenario: BrainReplayScenario
    expected_state: BrainContinuityExpectedState
    presence_scope_key: str | None = None
    language: Language = Language.EN
    context_queries: dict[str, str] = field(default_factory=dict)
    qa_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainReplayRegressionResult:
    """Result from replaying one checked-in regression case."""

    case: BrainReplayRegressionCase
    matched: bool
    mismatches: tuple[dict[str, Any], ...]
    artifact_path: Path
    actual_state: dict[str, Any]


def load_replay_regression_case(path: Path) -> BrainReplayRegressionCase:
    """Load one checked-in replay regression case from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    session_payload = dict(payload["session_ids"])
    session_ids = BrainSessionIds(
        agent_id=str(session_payload["agent_id"]),
        user_id=str(session_payload["user_id"]),
        session_id=str(session_payload["session_id"]),
        thread_id=str(session_payload["thread_id"]),
    )
    events = tuple(
        BrainEventRecord(
            id=index + 1,
            event_id=str(item["event_id"]),
            event_type=str(item["event_type"]),
            ts=str(item["ts"]),
            agent_id=str(item.get("agent_id", session_ids.agent_id)),
            user_id=str(item.get("user_id", session_ids.user_id)),
            session_id=str(item.get("session_id", session_ids.session_id)),
            thread_id=str(item.get("thread_id", session_ids.thread_id)),
            source=str(item.get("source", "fixture")),
            correlation_id=item.get("correlation_id"),
            causal_parent_id=item.get("causal_parent_id"),
            confidence=float(item.get("confidence", 1.0)),
            payload_json=json.dumps(item.get("payload", {}), ensure_ascii=False, sort_keys=True),
            tags_json=json.dumps(item.get("tags", []), ensure_ascii=False, sort_keys=True),
        )
        for index, item in enumerate(payload["events"])
    )
    scenario = BrainReplayScenario(
        name=str(payload["name"]),
        description=str(payload.get("description", payload["name"])),
        session_ids=session_ids,
        events=events,
        expected_terminal_state=dict(payload.get("expected_terminal_state", {})),
    )
    expected_state_payload = dict(payload.get("expected_state", {}))
    expected_state = BrainContinuityExpectedState(
        current_claims=tuple(expected_state_payload.get("current_claims", [])),
        historical_claims=tuple(expected_state_payload.get("historical_claims", [])),
        core_blocks_present=tuple(expected_state_payload.get("core_blocks_present", [])),
        commitment_titles_by_status={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("commitment_titles_by_status", {})
            ).items()
        },
        agenda_contains={
            key: tuple(value)
            for key, value in dict(expected_state_payload.get("agenda_contains", {})).items()
        },
        autobiography_entry_kinds=tuple(
            expected_state_payload.get("autobiography_entry_kinds", [])
        ),
        health_finding_codes=tuple(expected_state_payload.get("health_finding_codes", [])),
        health_score_min=expected_state_payload.get("health_score_min"),
        health_score_max=expected_state_payload.get("health_score_max"),
        relationship_arc_contains=expected_state_payload.get("relationship_arc_contains"),
        autonomy_current_candidate_ids=tuple(
            expected_state_payload.get("autonomy_current_candidate_ids", [])
        ),
        autonomy_current_candidate_summaries=tuple(
            expected_state_payload.get("autonomy_current_candidate_summaries", [])
        ),
        autonomy_recent_decision_kinds=tuple(
            expected_state_payload.get("autonomy_recent_decision_kinds", [])
        ),
        autonomy_reason_codes=tuple(expected_state_payload.get("autonomy_reason_codes", [])),
        autonomy_reevaluation_conditions=tuple(
            expected_state_payload.get("autonomy_reevaluation_conditions", [])
        ),
        reevaluation_current_hold_ids=tuple(
            expected_state_payload.get("reevaluation_current_hold_ids", [])
        ),
        reevaluation_trigger_kinds=tuple(
            expected_state_payload.get("reevaluation_trigger_kinds", [])
        ),
        reevaluation_transition_candidate_ids=tuple(
            expected_state_payload.get("reevaluation_transition_candidate_ids", [])
        ),
        reevaluation_transition_outcome_kinds=tuple(
            expected_state_payload.get("reevaluation_transition_outcome_kinds", [])
        ),
        wake_current_waiting_commitment_ids=tuple(
            expected_state_payload.get("wake_current_waiting_commitment_ids", [])
        ),
        wake_recent_wake_kinds=tuple(expected_state_payload.get("wake_recent_wake_kinds", [])),
        wake_recent_route_kinds=tuple(expected_state_payload.get("wake_recent_route_kinds", [])),
        wake_reason_codes=tuple(expected_state_payload.get("wake_reason_codes", [])),
        planning_current_pending_proposal_ids=tuple(
            expected_state_payload.get("planning_current_pending_proposal_ids", [])
        ),
        planning_current_pending_goal_ids=tuple(
            expected_state_payload.get("planning_current_pending_goal_ids", [])
        ),
        planning_recent_review_policies=tuple(
            expected_state_payload.get("planning_recent_review_policies", [])
        ),
        planning_recent_outcome_kinds=tuple(
            expected_state_payload.get("planning_recent_outcome_kinds", [])
        ),
        planning_reason_codes=tuple(expected_state_payload.get("planning_reason_codes", [])),
        planning_recent_revision_goal_ids=tuple(
            expected_state_payload.get("planning_recent_revision_goal_ids", [])
        ),
        continuity_graph_current_backing_ids=tuple(
            expected_state_payload.get("continuity_graph_current_backing_ids", [])
        ),
        continuity_graph_historical_backing_ids=tuple(
            expected_state_payload.get("continuity_graph_historical_backing_ids", [])
        ),
        continuity_graph_edge_kinds=tuple(
            expected_state_payload.get("continuity_graph_edge_kinds", [])
        ),
        continuity_graph_superseded_backing_ids=tuple(
            expected_state_payload.get("continuity_graph_superseded_backing_ids", [])
        ),
        continuity_graph_stale_backing_ids=tuple(
            expected_state_payload.get("continuity_graph_stale_backing_ids", [])
        ),
        relationship_dossier_summary_contains=expected_state_payload.get(
            "relationship_dossier_summary_contains"
        ),
        relationship_dossier_freshness=expected_state_payload.get("relationship_dossier_freshness"),
        relationship_dossier_contradiction=expected_state_payload.get(
            "relationship_dossier_contradiction"
        ),
        project_dossier_keys=tuple(expected_state_payload.get("project_dossier_keys", [])),
        dossier_stale_ids=tuple(expected_state_payload.get("dossier_stale_ids", [])),
        dossier_needs_refresh_ids=tuple(
            expected_state_payload.get("dossier_needs_refresh_ids", [])
        ),
        dossier_uncertain_ids=tuple(expected_state_payload.get("dossier_uncertain_ids", [])),
        dossier_contradicted_ids=tuple(expected_state_payload.get("dossier_contradicted_ids", [])),
        reply_packet_temporal_mode=expected_state_payload.get("reply_packet_temporal_mode"),
        planning_packet_temporal_mode=expected_state_payload.get("planning_packet_temporal_mode"),
        reply_selected_anchor_types=tuple(
            expected_state_payload.get("reply_selected_anchor_types", [])
        ),
        planning_selected_anchor_types=tuple(
            expected_state_payload.get("planning_selected_anchor_types", [])
        ),
        reply_selected_temporal_kinds=tuple(
            expected_state_payload.get("reply_selected_temporal_kinds", [])
        ),
        planning_selected_temporal_kinds=tuple(
            expected_state_payload.get("planning_selected_temporal_kinds", [])
        ),
        reply_selected_backing_ids=tuple(
            expected_state_payload.get("reply_selected_backing_ids", [])
        ),
        planning_selected_backing_ids=tuple(
            expected_state_payload.get("planning_selected_backing_ids", [])
        ),
        reply_drop_reason_codes=tuple(expected_state_payload.get("reply_drop_reason_codes", [])),
        planning_drop_reason_codes=tuple(
            expected_state_payload.get("planning_drop_reason_codes", [])
        ),
        packet_temporal_modes={
            key: value
            for key, value in dict(expected_state_payload.get("packet_temporal_modes", {})).items()
            if isinstance(value, str) and value.strip()
        },
        packet_selected_anchor_types={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_selected_anchor_types", {})
            ).items()
            if isinstance(value, list)
        },
        packet_selected_item_types={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_selected_item_types", {})
            ).items()
            if isinstance(value, list)
        },
        packet_selected_temporal_kinds={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_selected_temporal_kinds", {})
            ).items()
            if isinstance(value, list)
        },
        packet_selected_backing_ids={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_selected_backing_ids", {})
            ).items()
            if isinstance(value, list)
        },
        packet_selected_provenance_ids={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_selected_provenance_ids", {})
            ).items()
            if isinstance(value, list)
        },
        packet_drop_reason_codes={
            key: tuple(value)
            for key, value in dict(
                expected_state_payload.get("packet_drop_reason_codes", {})
            ).items()
            if isinstance(value, list)
        },
        graph_digest_current_commitment_ids=tuple(
            expected_state_payload.get("graph_digest_current_commitment_ids", [])
        ),
        graph_digest_current_plan_proposal_ids=tuple(
            expected_state_payload.get("graph_digest_current_plan_proposal_ids", [])
        ),
        governance_open_issue_kinds=tuple(
            expected_state_payload.get("governance_open_issue_kinds", [])
        ),
        governance_stale_graph_backing_ids=tuple(
            expected_state_payload.get("governance_stale_graph_backing_ids", [])
        ),
        governance_superseded_graph_backing_ids=tuple(
            expected_state_payload.get("governance_superseded_graph_backing_ids", [])
        ),
        procedural_active_skill_ids=tuple(
            expected_state_payload.get("procedural_active_skill_ids", [])
        ),
        procedural_candidate_skill_ids=tuple(
            expected_state_payload.get("procedural_candidate_skill_ids", [])
        ),
        procedural_retired_skill_ids=tuple(
            expected_state_payload.get("procedural_retired_skill_ids", [])
        ),
        procedural_superseded_skill_ids=tuple(
            expected_state_payload.get("procedural_superseded_skill_ids", [])
        ),
        procedural_failure_signature_codes=tuple(
            expected_state_payload.get("procedural_failure_signature_codes", [])
        ),
        procedural_low_confidence_skill_ids=tuple(
            expected_state_payload.get("procedural_low_confidence_skill_ids", [])
        ),
        procedural_retirement_reason_codes=tuple(
            expected_state_payload.get("procedural_retirement_reason_codes", [])
        ),
        planning_procedural_origins=tuple(
            expected_state_payload.get("planning_procedural_origins", [])
        ),
        planning_selected_skill_ids=tuple(
            expected_state_payload.get("planning_selected_skill_ids", [])
        ),
        planning_skill_rejection_reason_codes=tuple(
            expected_state_payload.get("planning_skill_rejection_reason_codes", [])
        ),
        planning_delta_operation_counts=tuple(
            expected_state_payload.get("planning_delta_operation_counts", [])
        ),
        procedural_high_risk_failure_signature_codes=tuple(
            expected_state_payload.get("procedural_high_risk_failure_signature_codes", [])
        ),
        procedural_follow_up_trace_ids=tuple(
            expected_state_payload.get("procedural_follow_up_trace_ids", [])
        ),
        procedural_negative_transfer_reason_codes=tuple(
            expected_state_payload.get("procedural_negative_transfer_reason_codes", [])
        ),
    )
    return BrainReplayRegressionCase(
        name=str(payload["name"]),
        description=str(payload.get("description", payload["name"])),
        scenario=scenario,
        expected_state=expected_state,
        presence_scope_key=payload.get("presence_scope_key"),
        language=Language(str(payload.get("language", "en"))),
        context_queries={
            str(key): str(value)
            for key, value in dict(payload.get("context_queries", {})).items()
            if str(value).strip()
        },
        qa_categories=tuple(
            str(item).strip() for item in payload.get("qa_categories", []) if str(item).strip()
        ),
    )


def load_replay_regression_cases(directory: Path) -> list[BrainReplayRegressionCase]:
    """Load all replay regression cases from one fixture directory."""
    if not directory.exists():
        return []
    return [load_replay_regression_case(path) for path in sorted(directory.glob("*.json"))]


def evaluate_replay_regression_case(
    *,
    case: BrainReplayRegressionCase,
    store: BrainStore,
    output_dir: Path | None = None,
) -> BrainReplayRegressionResult:
    """Replay one regression case and compare the resulting continuity state."""
    harness = BrainReplayHarness(store=store)
    artifact_path = (output_dir / f"{case.name}_replay.json") if output_dir is not None else None
    result = harness.replay(
        case.scenario,
        presence_scope_key=case.presence_scope_key,
        artifact_path=artifact_path,
        language=case.language,
        context_queries=case.context_queries,
    )
    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    actual_state = dict(payload.get("continuity_state", {}))
    mismatches = compare_continuity_state(actual_state, case.expected_state)
    qa_state_excerpt = build_procedural_qa_state_excerpt(actual_state=actual_state)
    payload["continuity_eval"] = {
        "matched": not mismatches,
        "mismatches": mismatches,
        "actual_state": actual_state,
        "qa_categories": list(case.qa_categories),
        "procedural_qa_state_excerpt": qa_state_excerpt,
    }
    result.artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return BrainReplayRegressionResult(
        case=case,
        matched=not mismatches,
        mismatches=tuple(mismatches),
        artifact_path=result.artifact_path,
        actual_state=actual_state,
    )


__all__ = [
    "BrainReplayRegressionCase",
    "BrainReplayRegressionResult",
    "evaluate_replay_regression_case",
    "load_replay_regression_case",
    "load_replay_regression_cases",
]

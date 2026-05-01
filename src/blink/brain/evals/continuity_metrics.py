"""Operator-facing continuity audit artifacts for Blink."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blink.brain.evals.memory_state import build_continuity_state
from blink.brain.evals.replay_cases import (
    evaluate_replay_regression_case,
    load_replay_regression_cases,
)
from blink.brain.procedural_qa_report import (
    build_procedural_qa_report,
    build_procedural_qa_state_excerpt,
)
from blink.brain.session import BrainSessionIds
from blink.transcriptions.language import Language

_CONTEXT_PACKET_TASKS = ("reply", "planning", "recall", "reflection", "critique")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BrainContinuityAuditReport:
    """One continuity audit artifact pair."""

    user_id: str
    thread_id: str
    generated_at: str
    payload: dict[str, Any]
    json_path: Path | None = None
    markdown_path: Path | None = None


class BrainContinuityAuditExporter:
    """Write inspectable continuity audit artifacts for operators."""

    def __init__(self, *, store):
        """Bind the exporter to one canonical store."""
        self._store = store

    def export(
        self,
        *,
        session_ids: BrainSessionIds,
        presence_scope_key: str,
        language: Language,
        output_dir: Path | None = None,
        replay_cases_dir: Path | None = None,
        context_queries: dict[str, str] | None = None,
        export_metadata: dict[str, Any] | None = None,
    ) -> BrainContinuityAuditReport:
        """Export a continuity audit JSON artifact plus derived Markdown report."""
        continuity_state = build_continuity_state(
            store=self._store,
            session_ids=session_ids,
            presence_scope_key=presence_scope_key,
            language=language,
            context_queries=context_queries,
        )
        replay_results = []
        if replay_cases_dir is not None:
            replay_output_dir = (output_dir / "replay") if output_dir is not None else None
            if replay_output_dir is not None:
                replay_output_dir.mkdir(parents=True, exist_ok=True)
            for case in load_replay_regression_cases(replay_cases_dir):
                replay_result = evaluate_replay_regression_case(
                    case=case,
                    store=self._store,
                    output_dir=replay_output_dir,
                )
                replay_results.append(
                    {
                        "name": case.name,
                        "description": case.description,
                        "qa_categories": list(case.qa_categories),
                        "matched": replay_result.matched,
                        "mismatches": list(replay_result.mismatches),
                        "artifact_path": str(replay_result.artifact_path),
                        "procedural_qa_state_excerpt": build_procedural_qa_state_excerpt(
                            actual_state=replay_result.actual_state
                        ),
                    }
                )

        continuity_state["procedural_qa_report"] = build_procedural_qa_report(
            procedural_skill_digest=continuity_state.get("procedural_skill_digest", {}),
            procedural_skill_governance_report=continuity_state.get(
                "procedural_skill_governance_report",
                {},
            ),
            planning_digest=continuity_state.get("planning_digest", {}),
            replay_regressions=replay_results,
        )
        continuity_state["proof_surface"] = self._build_proof_surface(
            continuity_state=continuity_state,
            replay_results=replay_results,
        )

        payload = {
            "generated_at": _utc_now(),
            "session_ids": {
                "agent_id": session_ids.agent_id,
                "user_id": session_ids.user_id,
                "session_id": session_ids.session_id,
                "thread_id": session_ids.thread_id,
            },
            "presence_scope_key": presence_scope_key,
            "context_queries": continuity_state.get("context_queries", {}),
            "continuity_state": continuity_state,
            "proof_surface": continuity_state.get("proof_surface", {}),
            "continuity_graph": continuity_state.get("continuity_graph", {}),
            "continuity_graph_digest": continuity_state.get("continuity_graph_digest", {}),
            "continuity_governance_report": continuity_state.get(
                "continuity_governance_report",
                {},
            ),
            "procedural_traces": continuity_state.get("procedural_traces", {}),
            "procedural_skills": continuity_state.get("procedural_skills", {}),
            "procedural_skill_digest": continuity_state.get("procedural_skill_digest", {}),
            "procedural_skill_governance_report": continuity_state.get(
                "procedural_skill_governance_report",
                {},
            ),
            "procedural_qa_report": continuity_state.get("procedural_qa_report", {}),
            "scene_world_state": continuity_state.get("scene_world_state", {}),
            "scene_world_state_digest": continuity_state.get("scene_world_state_digest", {}),
            "private_working_memory": continuity_state.get("private_working_memory", {}),
            "private_working_memory_digest": continuity_state.get(
                "private_working_memory_digest",
                {},
            ),
            "active_situation_model": continuity_state.get("active_situation_model", {}),
            "active_situation_model_digest": continuity_state.get(
                "active_situation_model_digest",
                {},
            ),
            "predictive_world_model": continuity_state.get("predictive_world_model", {}),
            "predictive_digest": continuity_state.get("predictive_digest", {}),
            "counterfactual_rehearsal": continuity_state.get("counterfactual_rehearsal", {}),
            "rehearsal_digest": continuity_state.get("rehearsal_digest", {}),
            "embodied_executive": continuity_state.get("embodied_executive", {}),
            "embodied_digest": continuity_state.get("embodied_digest", {}),
            "practice_director": continuity_state.get("practice_director", {}),
            "practice_digest": continuity_state.get("practice_digest", {}),
            "skill_evidence_ledger": continuity_state.get("skill_evidence_ledger", {}),
            "skill_evidence_digest": continuity_state.get("skill_evidence_digest", {}),
            "skill_governance": continuity_state.get("skill_governance", {}),
            "adapter_governance": continuity_state.get("adapter_governance", {}),
            "adapter_governance_digest": continuity_state.get("adapter_governance_digest", {}),
            "sim_to_real_digest": continuity_state.get("sim_to_real_digest", {}),
            "packet_traces": continuity_state.get("packet_traces", {}),
            "context_packet_digest": continuity_state.get("context_packet_digest", {}),
            "autonomy_digest": continuity_state.get("autonomy_digest", {}),
            "reevaluation_digest": continuity_state.get("reevaluation_digest", {}),
            "wake_digest": continuity_state.get("wake_digest", {}),
            "planning_digest": continuity_state.get("planning_digest", {}),
            "executive_policy_audit": continuity_state.get("executive_policy_audit", {}),
            "runtime_shell_digest": continuity_state.get("runtime_shell_digest", {}),
            "continuity_dossiers": continuity_state.get("continuity_dossiers", {}),
            "replay_regressions": replay_results,
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = (
                f"{session_ids.user_id}_{session_ids.thread_id.replace(':', '_').replace('/', '_')}"
            )
            json_path = output_dir / f"{base_name}_audit.json"
            markdown_path = output_dir / f"{base_name}_audit.md"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            markdown_path.write_text(
                self._render_markdown(payload),
                encoding="utf-8",
            )
            self._store.record_memory_export(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                export_kind="continuity_audit",
                path=json_path,
                payload=payload,
                metadata=export_metadata,
            )
        else:
            json_path = None
            markdown_path = None

        return BrainContinuityAuditReport(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            generated_at=str(payload["generated_at"]),
            payload=payload,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    def _build_proof_surface(
        self,
        *,
        continuity_state: dict[str, Any],
        replay_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        projections_audited = [
            key
            for key in (
                "continuity_graph",
                "continuity_dossiers",
                "scene_world_state",
                "private_working_memory",
                "active_situation_model",
                "predictive_world_model",
                "counterfactual_rehearsal",
                "embodied_executive",
                "practice_director",
                "skill_evidence_ledger",
                "skill_governance",
            )
            if continuity_state.get(key)
        ]
        packet_traces = dict(continuity_state.get("packet_traces") or {})
        packet_modes_built = [task for task in _CONTEXT_PACKET_TASKS if packet_traces.get(task)]
        replay_cases_failed = [item.get("name", "") for item in replay_results if not item.get("matched")]
        replay_cases_passed = len(replay_results) - len(replay_cases_failed)
        return {
            "authoritative_status_doc": "docs/roadmap.md",
            "canonical_entrypoint": "./scripts/test-brain-core.sh",
            "headless_bootstrap": "uv sync --python 3.12 --group dev",
            "always_run_lanes": ["fast", "proof", "fuzz-smoke"],
            "opt_in_lanes": ["atheris"],
            "projections_audited": projections_audited,
            "packet_modes_built": packet_modes_built,
            "replay_cases_run": len(replay_results),
            "replay_cases_passed": replay_cases_passed,
            "replay_cases_failed": replay_cases_failed,
            "remaining_thin_areas": [
                "Longer multi-refresh active-state sequences still rely more on the property lane than on directed unit coverage.",
                "Minimized fuzz failures still need promotion into stable deterministic regression examples.",
                "Packet and digest assertions must be refreshed whenever active-state fields or link semantics change.",
            ],
        }

    def _render_markdown(self, payload: dict[str, Any]) -> str:
        state = dict(payload.get("continuity_state", {}))
        proof_surface = dict(payload.get("proof_surface", {}))
        claim_counts = dict(state.get("claim_counts", {}))
        commitment_projection = dict(state.get("commitment_projection", {}))
        autonomy_digest = dict(state.get("autonomy_digest", {}))
        reevaluation_digest = dict(state.get("reevaluation_digest", {}))
        wake_digest = dict(state.get("wake_digest", {}))
        planning_digest = dict(state.get("planning_digest", {}))
        executive_policy_audit = dict(state.get("executive_policy_audit", {}))
        runtime_shell_digest = dict(state.get("runtime_shell_digest", {}))
        rehearsal_digest = dict(state.get("rehearsal_digest", {}))
        predictive_digest = dict(state.get("predictive_digest", {}))
        embodied_digest = dict(state.get("embodied_digest", {}))
        practice_digest = dict(state.get("practice_digest", {}))
        skill_evidence_digest = dict(state.get("skill_evidence_digest", {}))
        adapter_governance_digest = dict(state.get("adapter_governance_digest", {}))
        sim_to_real_digest = dict(state.get("sim_to_real_digest", {}))
        shell_rehearsal_inspection = dict(runtime_shell_digest.get("rehearsal_inspection", {}))
        shell_predictive_inspection = dict(runtime_shell_digest.get("predictive_inspection", {}))
        shell_embodied_inspection = dict(runtime_shell_digest.get("embodied_inspection", {}))
        shell_practice_inspection = dict(runtime_shell_digest.get("practice_inspection", {}))
        shell_skill_evidence_inspection = dict(
            runtime_shell_digest.get("skill_evidence_inspection", {})
        )
        shell_adapter_governance_inspection = dict(
            runtime_shell_digest.get("adapter_governance_inspection", {})
        )
        shell_sim_to_real_inspection = dict(
            runtime_shell_digest.get("sim_to_real_inspection", {})
        )
        shell_multimodal_inspection = dict(runtime_shell_digest.get("multimodal_inspection", {}))
        continuity_graph_digest = dict(state.get("continuity_graph_digest", {}))
        continuity_governance_report = dict(state.get("continuity_governance_report", {}))
        procedural_skills = dict(state.get("procedural_skills", {}))
        procedural_skill_digest = dict(state.get("procedural_skill_digest", {}))
        procedural_skill_governance_report = dict(
            state.get("procedural_skill_governance_report", {})
        )
        procedural_qa_report = dict(state.get("procedural_qa_report", {}))
        scene_world_state = dict(state.get("scene_world_state", {}))
        scene_world_state_digest = dict(state.get("scene_world_state_digest", {}))
        private_working_memory = dict(state.get("private_working_memory", {}))
        private_working_memory_digest = dict(state.get("private_working_memory_digest", {}))
        active_situation_model = dict(state.get("active_situation_model", {}))
        active_situation_model_digest = dict(state.get("active_situation_model_digest", {}))
        context_packet_digest = dict(state.get("context_packet_digest", {}))
        continuity_dossiers = dict(state.get("continuity_dossiers", {}))
        health = dict(state.get("memory_health") or {})
        visual_health = dict(state.get("visual_health") or {})
        reflection_cycles = list(state.get("reflection_cycles") or [])
        replay_regressions = list(payload.get("replay_regressions", []))
        failing_cases = [item for item in replay_regressions if not item.get("matched")]
        dossier_records = list(continuity_dossiers.get("dossiers", []))
        relationship_dossier = next(
            (record for record in dossier_records if record.get("kind") == "relationship"),
            None,
        )
        project_dossiers = [record for record in dossier_records if record.get("kind") == "project"]

        def _titles(key: str) -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('title', '')}"
                        f"[{record.get('goal_family', '')}/{record.get('scope_type', '')},"
                        f"rev={record.get('plan_revision', '')},resume={record.get('resume_count', '')}]"
                    )
                    for record in commitment_projection.get(key, [])
                    if record.get("title")
                )
                or "None"
            )

        def _blocked_details() -> str:
            records = commitment_projection.get("blocked_commitments", [])
            return (
                ", ".join(
                    (
                        f"{record.get('title', '')}: "
                        f"{(record.get('blocked_reason') or {}).get('summary', 'no reason')}; "
                        f"wake={', '.join(item.get('summary', '') for item in record.get('wake_conditions', [])) or 'none'}"
                    )
                    for record in records
                    if record.get("title")
                )
                or "None"
            )

        def _waiting_commitments() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('title', '')}"
                        f"[{record.get('goal_family', '')}/{record.get('status', '')},"
                        f" wake={', '.join(record.get('wake_kinds', [])) or 'none'},"
                        f" rev={record.get('plan_revision', 'n/a')},"
                        f" resume={record.get('resume_count', 'n/a')}]"
                        f" blocker={record.get('blocked_reason_summary') or 'none'}"
                    )
                    for record in wake_digest.get("current_waiting_commitments", [])
                    if record.get("title")
                )
                or "None"
            )

        def _wake_trigger_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}" for key, value in wake_digest.get("trigger_counts", {}).items()
                )
                or "None"
            )

        def _wake_route_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}" for key, value in wake_digest.get("route_counts", {}).items()
                )
                or "None"
            )

        def _wake_reason_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}" for key, value in wake_digest.get("reason_counts", {}).items()
                )
                or "None"
            )

        def _recent_direct_resumes() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('wake_kind') or 'unknown'}]"
                        f" -> {entry.get('resumed_goal_title') or entry.get('resumed_goal_id') or 'no_goal'}"
                    )
                    for entry in wake_digest.get("recent_direct_resumes", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _recent_candidate_proposals() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('wake_kind') or 'unknown'}]"
                        f" -> {entry.get('candidate_type') or 'unknown'}"
                        f" ({entry.get('candidate_goal_id') or 'no_candidate'})"
                    )
                    for entry in wake_digest.get("recent_candidate_proposals", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _recent_keep_waiting() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('wake_kind') or 'unknown'}]"
                        f" reason={entry.get('reason') or 'no_reason'}"
                        f" boundary={entry.get('boundary_kind') or 'none'}"
                    )
                    for entry in wake_digest.get("recent_keep_waiting", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _runtime_shell_control_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in runtime_shell_digest.get("control_counts", {}).items()
                )
                or "None"
            )

        def _runtime_shell_artifact_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in runtime_shell_digest.get("artifact_action_counts", {}).items()
                )
                or "None"
            )

        def _recent_runtime_shell_controls() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('control_kind') or 'unknown'}:"
                        f"{record.get('commitment_id') or 'no_commitment'}"
                        f"[{record.get('status_before') or 'none'}->{record.get('status_after') or 'none'}]"
                        f" reason={record.get('reason_summary') or 'none'}"
                    )
                    for record in runtime_shell_digest.get("recent_controls", [])
                )
                or "None"
            )

        def _recent_runtime_shell_artifacts() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('action_kind') or 'unknown'}:"
                        f"{record.get('cycle_id') or record.get('export_id') or 'none'}"
                        f" status={record.get('status') or 'n/a'}"
                    )
                    for record in runtime_shell_digest.get("recent_artifact_actions", [])
                )
                or "None"
            )

        def _runtime_shell_multimodal_counts(field: str) -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in shell_multimodal_inspection.get(field, {}).items()
                )
                or "None"
            )

        def _runtime_shell_predictive_counts(field: str) -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in shell_predictive_inspection.get(field, {}).items()
                )
                or "None"
            )

        def _runtime_shell_rehearsal_counts(field: str) -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in shell_rehearsal_inspection.get(field, {}).items()
                )
                or "None"
            )

        def _runtime_shell_embodied_counts(field: str) -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in shell_embodied_inspection.get(field, {}).items()
                )
                or "None"
            )

        def _recent_shell_scene_episodes() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('entry_id') or 'none'}"
                        f"[privacy={record.get('privacy_class') or 'none'},"
                        f" review={record.get('review_state') or 'none'},"
                        f" retention={record.get('retention_class') or 'none'}]"
                    )
                    for record in shell_multimodal_inspection.get("recent_scene_episodes", [])
                )
                or "None"
            )

        def _recent_shell_redactions() -> str:
            return (
                ", ".join(
                    str(record.get("entry_id") or "none")
                    for record in shell_multimodal_inspection.get("recent_redacted_rows", [])
                )
                or "None"
            )

        def _recent_shell_active_predictions() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('prediction_id') or 'none'}"
                        f"[{record.get('prediction_kind') or 'unknown'},"
                        f" confidence={record.get('confidence_band') or 'unknown'}]"
                    )
                    for record in shell_predictive_inspection.get("recent_active_predictions", [])
                )
                or "None"
            )

        def _recent_shell_prediction_resolutions() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('prediction_id') or 'none'}"
                        f"[{record.get('prediction_kind') or 'unknown'}"
                        f" -> {record.get('resolution_kind') or 'unknown'}]"
                    )
                    for record in shell_predictive_inspection.get(
                        "recent_prediction_resolutions",
                        [],
                    )
                )
                or "None"
            )

        def _recent_shell_rehearsals() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('rehearsal_id') or 'none'}"
                        f"[{record.get('candidate_action_id') or 'unknown'}"
                        f" -> {record.get('decision_recommendation') or 'unknown'}]"
                    )
                    for record in shell_rehearsal_inspection.get("recent_rehearsals", [])
                )
                or "None"
            )

        def _recent_shell_rehearsal_comparisons() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('comparison_id') or 'none'}"
                        f"[{record.get('calibration_bucket') or 'unknown'}"
                        f"/{record.get('observed_outcome_kind') or 'unknown'}]"
                    )
                    for record in shell_rehearsal_inspection.get("recent_comparisons", [])
                )
                or "None"
            )

        def _recent_shell_embodied_traces() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('trace_id') or 'none'}"
                        f"[{record.get('disposition') or 'unknown'}"
                        f"/{record.get('status') or 'unknown'}"
                        f"; action={record.get('selected_action_id') or 'none'}]"
                    )
                    for record in shell_embodied_inspection.get("recent_execution_traces", [])
                )
                or "None"
            )

        def _recent_shell_embodied_recoveries() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('recovery_id') or 'none'}"
                        f"[{record.get('action_id') or 'unknown'}"
                        f"; status={record.get('status') or 'unknown'}]"
                    )
                    for record in shell_embodied_inspection.get("recent_recoveries", [])
                )
                or "None"
            )

        def _recent_shell_low_level_embodied_actions() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('action_id') or 'none'}"
                        f"[source={record.get('source') or 'unknown'}"
                        f"; accepted={record.get('accepted')}]"
                    )
                    for record in shell_embodied_inspection.get(
                        "recent_low_level_embodied_actions",
                        [],
                    )
                )
                or "None"
            )

        def _current_plan_states() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('title', '')}"
                        f"[goal={record.get('goal_status') or 'none'},"
                        f" commitment={record.get('commitment_status') or 'none'},"
                        f" rev={record.get('plan_revision', 'n/a')},"
                        f" current={record.get('current_plan_proposal_id') or 'none'},"
                        f" pending={record.get('pending_plan_proposal_id') or 'none'},"
                        f" review={record.get('plan_review_policy') or 'none'}]"
                    )
                    for record in planning_digest.get("current_plan_states", [])
                    if record.get("title")
                )
                or "None"
            )

        def _review_policy_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in planning_digest.get(
                        "current_review_policy_counts", {}
                    ).items()
                )
                or "None"
            )

        def _proposal_source_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in planning_digest.get("proposal_source_counts", {}).items()
                )
                or "None"
            )

        def _planning_outcome_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in planning_digest.get("outcome_counts", {}).items()
                )
                or "None"
            )

        def _planning_reason_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in planning_digest.get("reason_counts", {}).items()
                )
                or "None"
            )

        def _current_pending_plan_proposals() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('review_policy') or 'none'};"
                        f" source={entry.get('source') or 'unknown'};"
                        f" rev={entry.get('plan_revision', 'n/a')};"
                        f" prefix={entry.get('preserved_prefix_count', 0)}]"
                        f" missing={', '.join(entry.get('missing_inputs', [])) or 'none'}"
                    )
                    for entry in planning_digest.get("current_pending_proposals", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _recent_adopted_plans() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('source') or 'unknown'};"
                        f" {entry.get('decision_reason') or 'no_reason'}]"
                        f" -> {entry.get('downstream_event_type') or 'no_downstream'}"
                        f" ({entry.get('downstream_goal_title') or entry.get('downstream_goal_id') or 'no_goal'})"
                    )
                    for entry in planning_digest.get("recent_adoptions", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _recent_rejected_plans() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [{entry.get('source') or 'unknown'};"
                        f" {entry.get('decision_reason') or 'no_reason'}]"
                    )
                    for entry in planning_digest.get("recent_rejections", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _recent_revision_flows() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('title', '')}"
                        f" [rev {entry.get('current_plan_revision', 'n/a')}"
                        f" -> {entry.get('plan_revision', 'n/a')};"
                        f" prefix={entry.get('preserved_prefix_count', 0)};"
                        f" outcome={entry.get('outcome_kind') or 'none'};"
                        f" reason={entry.get('decision_reason') or 'none'}]"
                        f" -> {entry.get('downstream_event_type') or 'no_downstream'}"
                    )
                    for entry in planning_digest.get("recent_revision_flows", [])
                    if entry.get("title")
                )
                or "None"
            )

        def _current_embodied_step() -> str:
            record = dict(planning_digest.get("current_embodied_step", {}))
            if not record:
                return "None"
            return (
                f"{record.get('intent_kind') or 'unknown'}"
                f"[action={record.get('selected_action_id') or 'none'};"
                f" disposition={record.get('disposition') or 'none'};"
                f" status={record.get('status') or 'none'};"
                f" policy={record.get('policy_posture') or 'none'}]"
            )

        def _recent_embodied_planning_traces() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('trace_id') or 'none'}"
                        f"[{entry.get('intent_kind') or 'unknown'}"
                        f"; action={entry.get('selected_action_id') or 'none'}"
                        f"; disposition={entry.get('disposition') or 'none'}"
                        f"; status={entry.get('status') or 'none'}]"
                    )
                    for entry in planning_digest.get("recent_embodied_execution_traces", [])
                )
                or "None"
            )

        def _recent_embodied_planning_recoveries() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('recovery_id') or 'none'}"
                        f"[action={entry.get('action_id') or 'none'}"
                        f"; status={entry.get('status') or 'none'}]"
                    )
                    for entry in planning_digest.get("recent_embodied_recoveries", [])
                )
                or "None"
            )

        def _dossier_freshness_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in continuity_dossiers.get("freshness_counts", {}).items()
                )
                or "None"
            )

        def _dossier_contradiction_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in continuity_dossiers.get("contradiction_counts", {}).items()
                )
                or "None"
            )

        def _render_dossier_fact(record: dict[str, Any]) -> str:
            evidence = dict(record.get("evidence") or {})
            claim_ids = ",".join(evidence.get("claim_ids", [])) or "none"
            entry_ids = ",".join(evidence.get("entry_ids", [])) or "none"
            event_ids = ",".join(evidence.get("source_event_ids", [])) or "none"
            return (
                f"{record.get('summary', '')}"
                f" [{record.get('status') or 'unknown'};"
                f" claims={claim_ids};"
                f" entries={entry_ids};"
                f" events={event_ids}]"
            )

        def _render_dossier_issue(record: dict[str, Any]) -> str:
            evidence = dict(record.get("evidence") or {})
            claim_ids = ",".join(evidence.get("claim_ids", [])) or "none"
            entry_ids = ",".join(evidence.get("entry_ids", [])) or "none"
            return (
                f"{record.get('kind', '')}: {record.get('summary', '')}"
                f" [claims={claim_ids}; entries={entry_ids}]"
            )

        def _relationship_dossier_summary() -> str:
            if not isinstance(relationship_dossier, dict):
                return "None"
            return (
                f"{relationship_dossier.get('summary') or 'None'}"
                f" [freshness={relationship_dossier.get('freshness') or 'none'};"
                f" contradiction={relationship_dossier.get('contradiction') or 'none'}]"
            )

        def _relationship_dossier_current_facts() -> str:
            if not isinstance(relationship_dossier, dict):
                return "None"
            return (
                ", ".join(
                    _render_dossier_fact(record)
                    for record in relationship_dossier.get("key_current_facts", [])
                    if record.get("summary")
                )
                or "None"
            )

        def _relationship_dossier_recent_changes() -> str:
            if not isinstance(relationship_dossier, dict):
                return "None"
            return (
                ", ".join(
                    _render_dossier_fact(record)
                    for record in relationship_dossier.get("recent_changes", [])
                    if record.get("summary")
                )
                or "None"
            )

        def _relationship_dossier_issues() -> str:
            if not isinstance(relationship_dossier, dict):
                return "None"
            return (
                ", ".join(
                    _render_dossier_issue(record)
                    for record in relationship_dossier.get("open_issues", [])
                    if record.get("kind")
                )
                or "None"
            )

        def _project_dossier_summaries() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('project_key') or 'unknown'}: "
                        f"{record.get('summary') or 'None'}"
                        f" [freshness={record.get('freshness') or 'none'};"
                        f" contradiction={record.get('contradiction') or 'none'}]"
                        f" facts={len(record.get('key_current_facts', []))}"
                        f" changes={len(record.get('recent_changes', []))}"
                        f" issues={len(record.get('open_issues', []))}"
                    )
                    for record in project_dossiers
                )
                or "None"
            )

        def _graph_kind_counts(key: str) -> str:
            return (
                ", ".join(
                    f"{kind}={count}"
                    for kind, count in continuity_graph_digest.get(key, {}).items()
                )
                or "None"
            )

        def _graph_current_nodes(kind: str) -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('summary') or record.get('backing_record_id') or 'unknown'}"
                        f"[{record.get('status') or 'unknown'};"
                        f" id={record.get('backing_record_id') or 'none'}]"
                    )
                    for record in continuity_graph_digest.get("current_nodes_by_kind", {}).get(
                        kind, []
                    )
                )
                or "None"
            )

        def _graph_supersession_links() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('kind') or 'unknown'}: "
                        f"{record.get('from_backing_record_id') or 'none'}"
                        f" -> {record.get('to_backing_record_id') or 'none'}"
                    )
                    for record in continuity_graph_digest.get("recent_supersession_links", [])
                )
                or "None"
            )

        def _graph_commitment_plan_links() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('commitment_summary') or record.get('commitment_id') or 'unknown'}"
                        f" -> {record.get('plan_proposal_id') or 'none'}"
                        f" [{record.get('kind') or 'unknown'};"
                        f" {record.get('plan_status') or 'unknown'}]"
                    )
                    for record in continuity_graph_digest.get("current_commitment_plan_links", [])
                )
                or "None"
            )

        def _governance_issue_counts() -> str:
            return (
                ", ".join(
                    f"{kind}={count}"
                    for kind, count in continuity_governance_report.get(
                        "open_issue_counts", {}
                    ).items()
                )
                or "None"
            )

        def _governance_issue_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('dossier_id') or 'unknown'}: "
                        f"{record.get('kind') or 'unknown'}"
                        f" -> {record.get('summary') or 'none'}"
                        f" [claims={','.join(record.get('evidence', {}).get('claim_ids', [])) or 'none'};"
                        f" entries={','.join(record.get('evidence', {}).get('entry_ids', [])) or 'none'};"
                        f" events={','.join(record.get('evidence', {}).get('source_event_ids', [])) or 'none'}]"
                    )
                    for record in continuity_governance_report.get("open_issue_rows", [])
                )
                or "None"
            )

        def _governance_dossier_availability() -> str:
            return (
                "; ".join(
                    f"{task}={','.join(f'{key}:{value}' for key, value in counts.items()) or 'none'}"
                    for task, counts in continuity_governance_report.get(
                        "dossier_availability_counts_by_task", {}
                    ).items()
                )
                or "None"
            )

        def _suppressed_packet_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('task') or 'unknown'}:"
                        f"{record.get('item_type') or 'unknown'}"
                        f"[dossier={record.get('dossier_id') or 'none'};"
                        f" backing={','.join(record.get('backing_ids', [])) or 'none'};"
                        f" reasons={','.join(record.get('reason_codes', [])) or 'none'}]"
                    )
                    for record in continuity_governance_report.get("suppressed_packet_rows", [])
                )
                or "None"
            )

        def _packet_suppression_counts() -> str:
            return (
                "; ".join(
                    f"{task}={','.join(f'{reason}:{count}' for reason, count in counts.items()) or 'none'}"
                    for task, counts in continuity_governance_report.get(
                        "packet_suppression_counts_by_task", {}
                    ).items()
                )
                or "None"
            )

        def _packet_summary(task: str, field: str) -> str:
            packet = dict(context_packet_digest.get(task, {}))
            value = packet.get(field)
            if isinstance(value, dict):
                return ", ".join(f"{key}={count}" for key, count in value.items()) or "None"
            if isinstance(value, list):
                return ", ".join(str(item) for item in value) or "None"
            return str(value or "None")

        def _packet_task_title(task: str) -> str:
            return task.replace("_", " ").title()

        def _packet_scene_episode_trace(task: str) -> str:
            trace = dict(context_packet_digest.get(task, {}).get("scene_episode_trace", {}))
            return (
                f"selected={','.join(trace.get('selected_entry_ids', [])) or 'none'}; "
                f"suppressed={','.join(trace.get('suppressed_entry_ids', [])) or 'none'}; "
                f"presence={','.join(trace.get('selected_presence_scope_keys', [])) or 'none'}; "
                f"privacy={','.join(f'{key}:{value}' for key, value in trace.get('privacy_counts', {}).items()) or 'none'}; "
                f"review={','.join(f'{key}:{value}' for key, value in trace.get('review_counts', {}).items()) or 'none'}; "
                f"retention={','.join(f'{key}:{value}' for key, value in trace.get('retention_counts', {}).items()) or 'none'}; "
                f"drops={','.join(f'{key}:{value}' for key, value in trace.get('drop_reason_counts', {}).items()) or 'none'}"
            )

        def _packet_prediction_trace(task: str) -> str:
            trace = dict(context_packet_digest.get(task, {}).get("prediction_trace", {}))
            return (
                f"selected={','.join(trace.get('selected_prediction_ids', [])) or 'none'}; "
                f"suppressed={','.join(trace.get('suppressed_prediction_ids', [])) or 'none'}; "
                f"kinds={','.join(trace.get('selected_prediction_kinds', [])) or 'none'}; "
                f"confidence={','.join(f'{key}:{value}' for key, value in trace.get('confidence_band_counts', {}).items()) or 'none'}; "
                f"resolution={','.join(f'{key}:{value}' for key, value in trace.get('resolution_kind_counts', {}).items()) or 'none'}; "
                f"risk={','.join(f'{key}:{value}' for key, value in trace.get('risk_code_counts', {}).items()) or 'none'}; "
                f"drops={','.join(f'{key}:{value}' for key, value in trace.get('drop_reason_counts', {}).items()) or 'none'}"
            )

        def _context_packet_review_lines() -> list[str]:
            lines: list[str] = []
            for task in _CONTEXT_PACKET_TASKS:
                lines.extend(
                    [
                        f"- {_packet_task_title(task)} query: {_packet_summary(task, 'query_text')}",
                        f"- {_packet_task_title(task)} temporal mode: {_packet_summary(task, 'temporal_mode')}",
                        f"- {_packet_task_title(task)} static tokens: {_packet_summary(task, 'static_token_usage')}",
                        f"- {_packet_task_title(task)} dynamic budget: {_packet_summary(task, 'dynamic_budget')}",
                        f"- {_packet_task_title(task)} selected anchors: {_packet_summary(task, 'selected_anchor_counts')}",
                        f"- {_packet_task_title(task)} selected items: {_packet_summary(task, 'selected_item_counts')}",
                        f"- {_packet_task_title(task)} selected availability: {_packet_summary(task, 'selected_availability_counts')}",
                        f"- {_packet_task_title(task)} selected temporal kinds: {_packet_summary(task, 'selected_temporal_counts')}",
                        f"- {_packet_task_title(task)} drop reasons: {_packet_summary(task, 'drop_reason_counts')}",
                        f"- {_packet_task_title(task)} governance drops: {_packet_summary(task, 'governance_drop_reason_counts')}",
                        f"- {_packet_task_title(task)} governance reason codes: {_packet_summary(task, 'governance_reason_code_counts')}",
                        f"- {_packet_task_title(task)} selected ids: {_packet_summary(task, 'selected_backing_ids')}",
                        f"- {_packet_task_title(task)} annotated ids: {_packet_summary(task, 'annotated_backing_ids')}",
                        f"- {_packet_task_title(task)} suppressed ids: {_packet_summary(task, 'suppressed_backing_ids')}",
                        f"- {_packet_task_title(task)} provenance ids: {_packet_summary(task, 'selected_provenance_ids')}",
                        f"- {_packet_task_title(task)} scene episodes: {_packet_scene_episode_trace(task)}",
                        f"- {_packet_task_title(task)} predictions: {_packet_prediction_trace(task)}",
                        "",
                    ]
                )
            return lines

        def _multimodal_packet_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('task') or 'unknown'}:"
                        f"{record.get('decision') or 'unknown'}"
                        f"[entry={record.get('entry_id') or 'none'};"
                        f" reason={record.get('reason') or 'none'};"
                        f" privacy={record.get('privacy_class') or 'none'};"
                        f" review={record.get('review_state') or 'none'}]"
                    )
                    for record in continuity_governance_report.get("multimodal_packet_rows", [])
                )
                or "None"
            )

        def _embodied_execution_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('decision') or 'unknown'}:"
                        f"{record.get('selected_action_id') or 'none'}"
                        f"[trace={record.get('trace_id') or 'none'};"
                        f" posture={record.get('policy_posture') or 'none'};"
                        f" reasons={','.join(record.get('reason_codes', [])) or 'none'}]"
                    )
                    for record in continuity_governance_report.get(
                        "embodied_execution_rows",
                        [],
                    )
                )
                or "None"
            )

        def _legacy_selection_lines() -> list[str]:
            lines = [
                "- Compatibility-only selector view; `Context Packet Review` is the authoritative live per-task packet path."
            ]
            for task in _CONTEXT_PACKET_TASKS:
                lines.append(
                    f"- {_packet_task_title(task)} sections: {', '.join(state.get('selection_sections', {}).get(task, [])) or 'None'}"
                )
            return lines

        def _private_working_memory_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in private_working_memory.get("buffer_counts", {}).items()
                )
                or "None"
            )

        def _private_working_memory_active_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{buffer_kind}: "
                        + " | ".join(
                            (
                                f"{record.get('summary') or 'unknown'}"
                                f" [{record.get('evidence_kind') or 'unknown'};"
                                f" backing={','.join(record.get('backing_ids', [])) or 'none'};"
                                f" events={','.join(record.get('source_event_ids', [])) or 'none'}]"
                            )
                            for record in records
                        )
                    )
                    for buffer_kind, records in private_working_memory_digest.get(
                        "active_records_by_buffer", {}
                    ).items()
                )
                or "None"
            )

        def _private_working_memory_stale_rows() -> str:
            stale_ids = list(private_working_memory_digest.get("stale_record_ids", []))
            resolved_ids = list(private_working_memory_digest.get("resolved_record_ids", []))
            unresolved_ids = list(private_working_memory_digest.get("unresolved_record_ids", []))
            return " | ".join(
                (
                    f"stale={','.join(stale_ids) or 'none'}",
                    f"resolved={','.join(resolved_ids) or 'none'}",
                    f"unresolved={','.join(unresolved_ids) or 'none'}",
                )
            )

        def _scene_world_active_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{zone_id}: "
                        + " | ".join(
                            (
                                f"{record.get('summary') or 'unknown'}"
                                f" [{record.get('entity_kind') or 'unknown'};"
                                f" affordances={','.join(record.get('affordance_ids', [])) or 'none'};"
                                f" events={','.join(record.get('source_event_ids', [])) or 'none'}]"
                            )
                            for record in records
                        )
                    )
                    for zone_id, records in scene_world_state_digest.get(
                        "active_entities_by_zone", {}
                    ).items()
                )
                or "None"
            )

        def _scene_world_state_rows() -> str:
            return " | ".join(
                (
                    f"stale={','.join(scene_world_state_digest.get('stale_entity_ids', [])) or 'none'}",
                    f"contradicted={','.join(scene_world_state_digest.get('contradicted_entity_ids', [])) or 'none'}",
                    f"expired={','.join(scene_world_state_digest.get('expired_entity_ids', [])) or 'none'}",
                    f"uncertain_affordances={','.join(scene_world_state_digest.get('uncertain_affordance_ids', [])) or 'none'}",
                    f"degraded_mode={scene_world_state_digest.get('degraded_mode', 'healthy')}",
                    f"reasons={','.join(scene_world_state_digest.get('degraded_reason_codes', [])) or 'none'}",
                )
            )

        def _active_situation_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in active_situation_model.get("kind_counts", {}).items()
                )
                or "None"
            )

        def _active_situation_active_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record_kind}: "
                        + " | ".join(
                            (
                                f"{record.get('summary') or 'unknown'}"
                                f" [{record.get('evidence_kind') or 'unknown'};"
                                f" uncertainty={','.join(record.get('uncertainty_codes', [])) or 'none'};"
                                f" backing={','.join(record.get('backing_ids', [])) or 'none'};"
                                f" events={','.join(record.get('source_event_ids', [])) or 'none'}]"
                            )
                            for record in records
                        )
                    )
                    for record_kind, records in active_situation_model_digest.get(
                        "active_records_by_kind", {}
                    ).items()
                )
                or "None"
            )

        def _active_situation_stale_rows() -> str:
            stale_ids = list(active_situation_model_digest.get("stale_record_ids", []))
            unresolved_ids = list(active_situation_model_digest.get("unresolved_record_ids", []))
            uncertainty_codes = list(active_situation_model.get("uncertainty_code_counts", {}))
            links = " | ".join(
                (
                    f"commitments={','.join(active_situation_model_digest.get('linked_commitment_ids', [])) or 'none'}",
                    f"proposals={','.join(active_situation_model_digest.get('linked_plan_proposal_ids', [])) or 'none'}",
                    f"skills={','.join(active_situation_model_digest.get('linked_skill_ids', [])) or 'none'}",
                )
            )
            return " | ".join(
                (
                    f"stale={','.join(stale_ids) or 'none'}",
                    f"unresolved={','.join(unresolved_ids) or 'none'}",
                    f"uncertainty={','.join(uncertainty_codes) or 'none'}",
                    links,
                )
            )

        def _procedural_skill_rows(statuses: set[str]) -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('title') or record.get('skill_id') or 'unknown'}"
                        f" [{record.get('status') or 'unknown'};"
                        f" confidence={record.get('confidence', 'n/a')};"
                        f" traces={','.join(record.get('supporting_trace_ids', [])) or 'none'};"
                        f" outcomes={','.join(record.get('supporting_outcome_ids', [])) or 'none'}]"
                    )
                    for record in procedural_skills.get("skills", [])
                    if record.get("status") in statuses
                )
                or "None"
            )

        def _procedural_failure_signatures() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('skill_id') or 'unknown'}:"
                        f" {record.get('kind') or 'unknown'}"
                        f"/{record.get('reason_code') or 'none'}"
                        f" (count={record.get('support_count', 0)};"
                        f" traces={','.join(record.get('support_trace_ids', [])) or 'none'};"
                        f" outcomes={','.join(record.get('support_outcome_ids', [])) or 'none'})"
                    )
                    for record in procedural_skill_digest.get("top_failure_signatures", [])
                )
                or "None"
            )

        def _practice_recent_plans() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('plan_id', '')}: "
                        f"targets={record.get('target_count', 0)} "
                        f"reasons={', '.join(f'{key}={value}' for key, value in record.get('reason_code_counts', {}).items()) or 'none'} "
                        f"artifacts={', '.join(record.get('artifact_paths', {}).values()) or 'none'}"
                    )
                    for record in shell_practice_inspection.get("recent_plans", [])
                )
                or "None"
            )

        def _practice_recent_targets() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('scenario_family', '')}/{record.get('scenario_id', '')}: "
                        f"profile={record.get('selected_profile_id', '')} "
                        f"score={record.get('score', 0)} "
                        f"reasons={', '.join(record.get('reason_codes', [])) or 'none'} "
                        f"skills={', '.join(record.get('related_skill_ids', [])) or 'none'}"
                    )
                    for record in shell_practice_inspection.get("recent_targets", [])
                )
                or "None"
            )

        def _skill_evidence_deltas() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('skill_id', '')}: "
                        f"delta={record.get('delta_episode_count', 0)} "
                        f"support={record.get('support_episode_count', 0)} "
                        f"families={', '.join(record.get('scenario_families', [])) or 'none'} "
                        f"critical={record.get('critical_safety_violation_count', 0)}"
                    )
                    for record in shell_skill_evidence_inspection.get("top_evidence_deltas", [])
                )
                or "None"
            )

        def _recent_skill_governance() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('kind', '')}:{record.get('skill_id', '')}:"
                        f"{record.get('status', '')} "
                        f"reasons={', '.join(record.get('reason_codes', [])) or 'none'}"
                    )
                    for record in shell_skill_evidence_inspection.get(
                        "recent_governance_proposals",
                        [],
                    )
                )
                or "None"
            )

        def _recent_adapter_cards() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('adapter_family', '')}:{record.get('backend_id', '')}"
                        f"@{record.get('backend_version', '')} "
                        f"state={record.get('promotion_state', '')} "
                        f"targets={', '.join(record.get('approved_target_families', [])) or 'none'}"
                    )
                    for record in shell_adapter_governance_inspection.get("recent_cards", [])
                )
                or "None"
            )

        def _recent_adapter_reports() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('adapter_family', '')}:{record.get('candidate_backend_id', '')}"
                        f" vs {record.get('incumbent_backend_id', '')} "
                        f"passed={record.get('benchmark_passed')} "
                        f"blocked={', '.join(record.get('blocked_reason_codes', [])) or 'none'}"
                    )
                    for record in shell_adapter_governance_inspection.get("recent_reports", [])
                )
                or "None"
            )

        def _current_default_adapter_cards() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('adapter_family', '')}:{record.get('backend_id', '')}"
                        f"@{record.get('backend_version', '')}"
                    )
                    for record in shell_adapter_governance_inspection.get(
                        "current_default_cards",
                        [],
                    )
                )
                or "None"
            )

        def _recent_adapter_decisions() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('adapter_family', '')}:{record.get('backend_id', '')}:"
                        f"{record.get('decision_outcome', '')} "
                        f"{record.get('from_state', '')}->{record.get('to_state', '')} "
                        f"blocked={', '.join(record.get('blocked_reason_codes', [])) or 'none'}"
                    )
                    for record in shell_adapter_governance_inspection.get(
                        "recent_promotion_decisions",
                        [],
                    )
                )
                or "None"
            )

        def _sim_to_real_rows() -> str:
            return (
                "; ".join(
                    (
                        f"{record.get('adapter_family', '')}:{record.get('backend_id', '')}"
                        f" state={record.get('promotion_state', '')} "
                        f"shadow={record.get('shadow_ready')} "
                        f"canary={record.get('canary_ready')} "
                        f"default={record.get('default_ready')} "
                        f"rollback={record.get('rollback_required')} "
                        f"blocked={', '.join(record.get('blocked_reason_codes', [])) or 'none'}"
                    )
                    for record in shell_sim_to_real_inspection.get("readiness_reports", [])
                )
                or "None"
            )

        def _procedural_qa_case_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in procedural_qa_report.get("case_counts", {}).items()
                )
                or "None"
            )

        def _procedural_qa_category_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in procedural_qa_report.get("category_counts", {}).items()
                )
                or "None"
            )

        def _procedural_qa_coverage() -> str:
            return (
                ", ".join(
                    f"{key}={'yes' if value else 'no'}"
                    for key, value in procedural_qa_report.get("coverage_flags", {}).items()
                )
                or "None"
            )

        def _procedural_learning_lifecycle() -> str:
            return (
                ", ".join(
                    part
                    for part in (
                        f"active={','.join(procedural_qa_report.get('active_skill_ids', [])) or 'none'}",
                        f"candidate={','.join(procedural_qa_report.get('candidate_skill_ids', [])) or 'none'}",
                        f"low_confidence={','.join(procedural_qa_report.get('low_confidence_skill_ids', [])) or 'none'}",
                        f"follow_up_traces={','.join(procedural_qa_report.get('follow_up_trace_ids', [])) or 'none'}",
                    )
                )
                or "None"
            )

        def _procedural_reuse_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('plan_proposal_id') or 'unknown'}:"
                        f" {record.get('procedural_origin') or 'none'}"
                        f" skill={record.get('selected_skill_id') or 'none'}"
                        f" traces={','.join(record.get('selected_skill_support_trace_ids', [])) or 'none'}"
                        f" proposals={','.join(record.get('selected_skill_support_plan_proposal_ids', [])) or 'none'}"
                        f" delta_ops={record.get('delta_operation_count', 0)}"
                    )
                    for record in procedural_qa_report.get("recent_skill_flows", [])
                    if record.get("procedural_origin")
                )
                or "None"
            )

        def _procedural_negative_transfer_rows() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('plan_proposal_id') or 'unknown'}:"
                        f" origin={record.get('procedural_origin') or 'none'}"
                        f" skill={record.get('selected_skill_id') or 'none'}"
                        f" rejected={','.join(record.get('rejected_skill_ids', [])) or 'none'}"
                        f" reasons={','.join(record.get('rejection_reasons', [])) or 'none'}"
                        f" decision={record.get('decision_reason') or 'none'}"
                        f" traces={','.join(record.get('selected_skill_support_trace_ids', [])) or 'none'}"
                    )
                    for record in procedural_qa_report.get("recent_negative_transfer_flows", [])
                )
                or "None"
            )

        def _procedural_retirement_supersession_rows() -> str:
            parts = [
                "superseded="
                + (",".join(procedural_qa_report.get("superseded_skill_ids", [])) or "none"),
                "retired="
                + (",".join(procedural_qa_report.get("retired_skill_ids", [])) or "none"),
                "high_risk="
                + (
                    ", ".join(
                        (
                            f"{record.get('skill_id') or 'unknown'}:"
                            f"{record.get('reason_code') or 'none'}"
                            f"[traces={','.join(record.get('support_trace_ids', [])) or 'none'};"
                            f"outcomes={','.join(record.get('support_outcome_ids', [])) or 'none'}]"
                        )
                        for record in procedural_qa_report.get("high_risk_failure_signatures", [])
                    )
                    or "none"
                ),
            ]
            return " | ".join(parts)

        def _candidate_summaries() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('summary', '')}"
                        f" [{record.get('candidate_type', '')}/{record.get('initiative_class', '')},"
                        f" confidence={record.get('confidence', 'n/a')},"
                        f" pending_age={record.get('pending_age_secs', 'n/a')},"
                        f" expires_at={record.get('expires_at') or 'none'},"
                        f" reevaluate={record.get('expected_reevaluation_condition_kind') or 'none'}]"
                    )
                    for record in autonomy_digest.get("current_candidates", [])
                    if record.get("summary")
                )
                or "None"
            )

        def _pending_families() -> str:
            return (
                ", ".join(
                    (
                        f"{record.get('goal_family', '')}: "
                        f"count={record.get('pending_count', 0)}, "
                        f"leader={record.get('leader_summary', '') or record.get('leader_candidate_goal_id', '')}"
                    )
                    for record in autonomy_digest.get("current_family_leaders", [])
                    if record.get("goal_family")
                )
                or "None"
            )

        def _decision_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in autonomy_digest.get("decision_counts", {}).items()
                )
                or "None"
            )

        def _reason_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in autonomy_digest.get("reason_counts", {}).items()
                )
                or "None"
            )

        def _recent_actions() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('summary', '')} -> "
                        f"{entry.get('accepted_goal_title') or entry.get('accepted_goal_id') or 'no_goal'}"
                        f" [{entry.get('goal_family') or 'unknown'};"
                        f" selector={entry.get('selected_by') or 'none'}]"
                    )
                    for entry in autonomy_digest.get("recent_actions", [])
                    if entry.get("summary")
                )
                or "None"
            )

        def _recent_suppressions() -> str:
            return (
                ", ".join(
                    f"{entry.get('summary', '')} [{entry.get('reason') or 'no_reason'}]"
                    for entry in autonomy_digest.get("recent_suppressions", [])
                    if entry.get("summary")
                )
                or "None"
            )

        def _recent_merges() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('summary', '')} -> "
                        f"{entry.get('merged_into_candidate_goal_id') or 'unknown'}"
                        f" [{entry.get('reason') or 'no_reason'}]"
                    )
                    for entry in autonomy_digest.get("recent_merges", [])
                    if entry.get("summary")
                )
                or "None"
            )

        def _recent_non_actions() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('summary', '')} [{entry.get('reason') or 'no_reason'}]"
                        f" reevaluate={entry.get('expected_reevaluation_condition') or 'none'}"
                        f" ({entry.get('expected_reevaluation_condition_kind') or 'none'})"
                        f" selector={entry.get('selected_by') or 'none'}"
                    )
                    for entry in autonomy_digest.get("recent_non_actions", [])
                    if entry.get("summary")
                )
                or "None"
            )

        def _current_holds() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('summary', '')}"
                        f" [{entry.get('hold_reason') or 'no_reason'};"
                        f" reevaluate={entry.get('expected_reevaluation_condition_kind') or 'none'};"
                        f" expires_at={entry.get('expires_at') or 'none'}]"
                    )
                    for entry in reevaluation_digest.get("current_holds", [])
                    if entry.get("summary")
                )
                or "None"
            )

        def _trigger_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in reevaluation_digest.get("trigger_counts", {}).items()
                )
                or "None"
            )

        def _outcome_counts() -> str:
            return (
                ", ".join(
                    f"{key}={value}"
                    for key, value in reevaluation_digest.get("outcome_counts", {}).items()
                )
                or "None"
            )

        def _recent_triggers() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('kind') or 'unknown'}: "
                        f"{entry.get('summary') or 'no_summary'}"
                        f" candidates={', '.join(entry.get('candidate_goal_ids', [])) or 'none'}"
                    )
                    for entry in reevaluation_digest.get("recent_triggers", [])
                )
                or "None"
            )

        def _recent_transitions() -> str:
            return (
                ", ".join(
                    (
                        f"{entry.get('summary', '')}"
                        f" hold={entry.get('hold_reason') or 'none'}"
                        f" -> {entry.get('trigger_kind') or 'none'}"
                        f" -> {entry.get('outcome_decision_kind') or 'none'}"
                        f" [{entry.get('outcome_reason') or 'no_reason'}"
                        f"; goal={entry.get('accepted_goal_id') or 'none'}]"
                    )
                    for entry in reevaluation_digest.get("recent_transitions", [])
                    if entry.get("summary")
                )
                or "None"
            )

        return "\n".join(
            [
                "# Blink Continuity Audit",
                "",
                f"- Generated: {payload.get('generated_at')}",
                f"- User: {payload.get('session_ids', {}).get('user_id', '')}",
                f"- Thread: {payload.get('session_ids', {}).get('thread_id', '')}",
                "",
                "## Brain-Core Proof Surface",
                f"- Status doc: {proof_surface.get('authoritative_status_doc') or 'None'}",
                f"- Canonical entrypoint: {proof_surface.get('canonical_entrypoint') or 'None'}",
                f"- Headless bootstrap: {proof_surface.get('headless_bootstrap') or 'None'}",
                f"- Always-run lanes: {', '.join(proof_surface.get('always_run_lanes', [])) or 'None'}",
                f"- Opt-in lanes: {', '.join(proof_surface.get('opt_in_lanes', [])) or 'None'}",
                f"- Projections audited: {', '.join(proof_surface.get('projections_audited', [])) or 'None'}",
                f"- Packet modes built: {', '.join(proof_surface.get('packet_modes_built', [])) or 'None'}",
                f"- Replay regressions: passed={proof_surface.get('replay_cases_passed', 0)}/{proof_surface.get('replay_cases_run', 0)} failed={', '.join(proof_surface.get('replay_cases_failed', [])) or 'None'}",
                "",
                "### Remaining Thin Areas",
                *[
                    f"- {item}"
                    for item in proof_surface.get("remaining_thin_areas", [])
                ],
                "",
                "## Memory Health",
                f"- Status: {health.get('status', 'none')}",
                f"- Score: {health.get('score', 'n/a')}",
                f"- Findings: {', '.join(finding.get('code', '') for finding in health.get('findings', [])) or 'None'}",
                "",
                "## Visual Health",
                f"- Camera connected: {visual_health.get('camera_connected', 'n/a')}",
                f"- Track state: {visual_health.get('camera_track_state', 'n/a')}",
                f"- Person present: {visual_health.get('person_present', 'n/a')}",
                f"- Last fresh frame: {visual_health.get('last_fresh_frame_at', 'n/a')}",
                f"- Frame age (ms): {visual_health.get('frame_age_ms', 'n/a')}",
                f"- Detector: {visual_health.get('detection_backend', 'n/a')} ({visual_health.get('detection_confidence', 'n/a')})",
                f"- Reason: {visual_health.get('sensor_health_reason', 'none')}",
                f"- Recovery: in_progress={visual_health.get('recovery_in_progress', False)}, attempts={visual_health.get('recovery_attempts', 0)}",
                "",
                "## Claims",
                f"- Current claims: {claim_counts.get('current', 0)}",
                f"- Historical claims: {claim_counts.get('historical', 0)}",
                f"- Uncertain current claims: {claim_counts.get('uncertain_current', 0)}",
                f"- Supersession links: {claim_counts.get('supersession_links', 0)}",
                "",
                "## Commitments",
                f"- Active: {_titles('active_commitments')}",
                f"- Deferred: {_titles('deferred_commitments')}",
                f"- Blocked: {_titles('blocked_commitments')}",
                f"- Blocked details: {_blocked_details()}",
                f"- Recent terminal: {_titles('recent_terminal_commitments')}",
                "",
                "## Executive Policy Audit",
                f"- Aggregate posture counts: {', '.join(f'{key}={value}' for key, value in executive_policy_audit.get('policy_posture_counts', {}).items()) or 'None'}",
                f"- Aggregate approval requirements: {', '.join(f'{key}={value}' for key, value in executive_policy_audit.get('approval_requirement_counts', {}).items()) or 'None'}",
                f"- Why-now reason codes: {', '.join(f'{key}={value}' for key, value in executive_policy_audit.get('why_now_reason_code_counts', {}).items()) or 'None'}",
                f"- Why-not reason codes: {', '.join(f'{key}={value}' for key, value in executive_policy_audit.get('why_not_reason_code_counts', {}).items()) or 'None'}",
                "",
                "## Runtime Shell Review",
                f"- Shell control counts: {_runtime_shell_control_counts()}",
                f"- Shell artifact counts: {_runtime_shell_artifact_counts()}",
                f"- Recent shell controls: {_recent_runtime_shell_controls()}",
                f"- Recent shell artifact actions: {_recent_runtime_shell_artifacts()}",
                f"- Predictive active kinds: {', '.join(f'{key}={value}' for key, value in predictive_digest.get('active_kind_counts', {}).items()) or 'None'}",
                f"- Predictive confidence bands: {', '.join(f'{key}={value}' for key, value in predictive_digest.get('active_confidence_band_counts', {}).items()) or 'None'}",
                f"- Predictive resolution kinds: {', '.join(f'{key}={value}' for key, value in predictive_digest.get('resolution_kind_counts', {}).items()) or 'None'}",
                f"- Predictive highest-risk ids: {', '.join(predictive_digest.get('highest_risk_prediction_ids', [])) or 'None'}",
                f"- Predictive soon-expiring ids: {', '.join(predictive_digest.get('soon_expiring_prediction_ids', [])) or 'None'}",
                f"- Rehearsal recommendation counts: {', '.join(f'{key}={value}' for key, value in rehearsal_digest.get('recommendation_counts', {}).items()) or 'None'}",
                f"- Rehearsal calibration buckets: {', '.join(f'{key}={value}' for key, value in rehearsal_digest.get('calibration_bucket_counts', {}).items()) or 'None'}",
                f"- Rehearsal mismatch patterns: {', '.join(f'{key}={value}' for key, value in rehearsal_digest.get('recurrent_mismatch_patterns', {}).items()) or 'None'}",
                f"- Embodied intent kinds: {', '.join(f'{key}={value}' for key, value in embodied_digest.get('intent_kind_counts', {}).items()) or 'None'}",
                f"- Embodied dispositions: {', '.join(f'{key}={value}' for key, value in embodied_digest.get('disposition_counts', {}).items()) or 'None'}",
                f"- Embodied trace statuses: {', '.join(f'{key}={value}' for key, value in embodied_digest.get('trace_status_counts', {}).items()) or 'None'}",
                f"- Embodied repair codes: {', '.join(f'{key}={value}' for key, value in embodied_digest.get('repair_code_counts', {}).items()) or 'None'}",
                f"- Shell predictive active kinds: {_runtime_shell_predictive_counts('active_kind_counts')}",
                f"- Shell predictive confidence bands: {_runtime_shell_predictive_counts('active_confidence_band_counts')}",
                f"- Shell predictive resolution kinds: {_runtime_shell_predictive_counts('resolution_kind_counts')}",
                f"- Shell rehearsal recommendations: {_runtime_shell_rehearsal_counts('recommendation_counts')}",
                f"- Shell rehearsal calibration buckets: {_runtime_shell_rehearsal_counts('calibration_bucket_counts')}",
                f"- Shell embodied dispositions: {_runtime_shell_embodied_counts('disposition_counts')}",
                f"- Shell embodied trace statuses: {_runtime_shell_embodied_counts('trace_status_counts')}",
                f"- Shell embodied policy postures: {_runtime_shell_embodied_counts('policy_posture_counts')}",
                f"- Recent shell active predictions: {_recent_shell_active_predictions()}",
                f"- Recent shell prediction resolutions: {_recent_shell_prediction_resolutions()}",
                f"- Recent shell rehearsals: {_recent_shell_rehearsals()}",
                f"- Recent shell rehearsal comparisons: {_recent_shell_rehearsal_comparisons()}",
                f"- Recent shell embodied traces: {_recent_shell_embodied_traces()}",
                f"- Recent shell embodied recoveries: {_recent_shell_embodied_recoveries()}",
                f"- Recent shell low-level embodied actions: {_recent_shell_low_level_embodied_actions()}",
                f"- Multimodal latest presence scope: {shell_multimodal_inspection.get('latest_source_presence_scope_key') or 'None'}",
                f"- Multimodal privacy counts: {_runtime_shell_multimodal_counts('privacy_counts')}",
                f"- Multimodal review counts: {_runtime_shell_multimodal_counts('review_counts')}",
                f"- Multimodal retention counts: {_runtime_shell_multimodal_counts('retention_counts')}",
                f"- Recent shell scene episodes: {_recent_shell_scene_episodes()}",
                f"- Recent shell redactions: {_recent_shell_redactions()}",
                "",
                "## Wake Review",
                f"- Current wait count: {wake_digest.get('current_wait_count', 0)}",
                f"- Current wait kinds: {', '.join(f'{key}={value}' for key, value in wake_digest.get('current_wait_kind_counts', {}).items()) or 'None'}",
                f"- Wake trigger counts: {_wake_trigger_counts()}",
                f"- Wake route counts: {_wake_route_counts()}",
                f"- Wake reason counts: {_wake_reason_counts()}",
                "",
                "### Current Waiting Commitments",
                f"- {_waiting_commitments()}",
                "",
                "### Recent Direct Resumes",
                f"- {_recent_direct_resumes()}",
                "",
                "### Recent Candidate Proposals",
                f"- {_recent_candidate_proposals()}",
                "",
                "### Recent Keep-Waiting Decisions",
                f"- {_recent_keep_waiting()}",
                "",
                "## Planning Review",
                f"- Current plan state count: {planning_digest.get('current_plan_state_count', 0)}",
                f"- Current pending proposal count: {planning_digest.get('current_pending_proposal_count', 0)}",
                f"- Current plan states: {_current_plan_states()}",
                f"- Review policy counts: {_review_policy_counts()}",
                f"- Proposal source counts: {_proposal_source_counts()}",
                f"- Planning outcome counts: {_planning_outcome_counts()}",
                f"- Planning reason counts: {_planning_reason_counts()}",
                f"- Planning rehearsal recommendations: {', '.join(f'{key}={value}' for key, value in planning_digest.get('rehearsal_recommendation_counts', {}).items()) or 'None'}",
                f"- Planning rehearsal calibration buckets: {', '.join(f'{key}={value}' for key, value in planning_digest.get('rehearsal_calibration_bucket_counts', {}).items()) or 'None'}",
                f"- Planning rehearsal operator-review floors: {planning_digest.get('rehearsal_operator_review_floor_count', 0)}",
                f"- Current embodied step: {_current_embodied_step()}",
                f"- Planning embodied dispositions: {', '.join(f'{key}={value}' for key, value in planning_digest.get('embodied_disposition_counts', {}).items()) or 'None'}",
                f"- Planning embodied policy postures: {', '.join(f'{key}={value}' for key, value in planning_digest.get('embodied_policy_posture_counts', {}).items()) or 'None'}",
                f"- Planning embodied operator-review floors: {planning_digest.get('embodied_operator_review_floor_count', 0)}",
                "",
                "### Current Pending Plan Proposals",
                f"- {_current_pending_plan_proposals()}",
                "",
                "### Recent Adopted Plans",
                f"- {_recent_adopted_plans()}",
                "",
                "### Recent Rejected Plans",
                f"- {_recent_rejected_plans()}",
                "",
                "### Recent Revision Flows",
                f"- {_recent_revision_flows()}",
                "",
                "### Recent Embodied Coordinator Traces",
                f"- {_recent_embodied_planning_traces()}",
                "",
                "### Recent Embodied Recoveries",
                f"- {_recent_embodied_planning_recoveries()}",
                "",
                "## Autonomy Review",
                f"- Current candidate count: {autonomy_digest.get('current_candidate_count', 0)}",
                f"- Pending families: {_pending_families()}",
                f"- Next expiry: {autonomy_digest.get('next_expiry_at') or 'None'}",
                f"- Decision counts: {_decision_counts()}",
                f"- Reason counts: {_reason_counts()}",
                "",
                "### Active Candidates",
                f"- {_candidate_summaries()}",
                "",
                "### Recent Accepted Actions",
                f"- {_recent_actions()}",
                "",
                "### Recent Suppressions",
                f"- {_recent_suppressions()}",
                "",
                "### Recent Merges",
                f"- {_recent_merges()}",
                "",
                "### Recent Non-Actions",
                f"- {_recent_non_actions()}",
                "",
                "## Reevaluation Review",
                f"- Current hold count: {reevaluation_digest.get('current_hold_count', 0)}",
                f"- Current hold kinds: {', '.join(f'{key}={value}' for key, value in reevaluation_digest.get('current_hold_kinds', {}).items()) or 'None'}",
                f"- Trigger counts: {_trigger_counts()}",
                f"- Reevaluation outcome counts: {_outcome_counts()}",
                "",
                "### Current Held Candidates",
                f"- {_current_holds()}",
                "",
                "### Recent Reevaluation Triggers",
                f"- {_recent_triggers()}",
                "",
                "### Recent Hold -> Reevaluation Flows",
                f"- {_recent_transitions()}",
                "",
                "## Relationship Continuity",
                f"- Relationship arc: {state.get('relationship_arc_summary') or 'None'}",
                f"- Autobiography entry kinds: {', '.join(entry.get('entry_kind', '') for entry in state.get('autobiography', [])) or 'None'}",
                f"- Multimodal privacy counts: {', '.join(f'{key}={value}' for key, value in state.get('multimodal_autobiography_digest', {}).get('entry_counts', {}).get('privacy', {}).items()) or 'None'}",
                f"- Multimodal review counts: {', '.join(f'{key}={value}' for key, value in state.get('multimodal_autobiography_digest', {}).get('entry_counts', {}).get('review', {}).items()) or 'None'}",
                f"- Multimodal retention counts: {', '.join(f'{key}={value}' for key, value in state.get('multimodal_autobiography_digest', {}).get('entry_counts', {}).get('retention', {}).items()) or 'None'}",
                f"- Recent redacted multimodal rows: {', '.join(row.get('entry_id', '') for row in state.get('multimodal_autobiography_digest', {}).get('recent_redacted_rows', [])) or 'None'}",
                "",
                "## Graph Review",
                f"- Node counts: {', '.join(f'{key}={value}' for key, value in continuity_graph_digest.get('node_counts', {}).items()) or 'None'}",
                f"- Edge counts: {', '.join(f'{key}={value}' for key, value in continuity_graph_digest.get('edge_counts', {}).items()) or 'None'}",
                f"- Current kind counts: {_graph_kind_counts('current_node_kind_counts')}",
                f"- Historical kind counts: {_graph_kind_counts('historical_node_kind_counts')}",
                f"- Stale kind counts: {_graph_kind_counts('stale_node_kind_counts')}",
                f"- Superseded kind counts: {_graph_kind_counts('superseded_node_kind_counts')}",
                f"- Evidence anchors: {', '.join(f'{key}={value}' for key, value in continuity_graph_digest.get('evidence_anchor_counts', {}).items()) or 'None'}",
                "",
                "### Current Claim Nodes",
                f"- {_graph_current_nodes('claim')}",
                "",
                "### Current Autobiography Nodes",
                f"- {_graph_current_nodes('autobiography_entry')}",
                "",
                "### Current Commitment Nodes",
                f"- {_graph_current_nodes('commitment')}",
                "",
                "### Current Plan Proposal Nodes",
                f"- {_graph_current_nodes('plan_proposal')}",
                "",
                "### Recent Supersession Links",
                f"- {_graph_supersession_links()}",
                "",
                "### Current Commitment / Plan Links",
                f"- {_graph_commitment_plan_links()}",
                "",
                "## Dossier Review",
                f"- Dossier counts: {', '.join(f'{key}={value}' for key, value in continuity_dossiers.get('dossier_counts', {}).items()) or 'None'}",
                f"- Freshness counts: {_dossier_freshness_counts()}",
                f"- Contradiction counts: {_dossier_contradiction_counts()}",
                "",
                "### Relationship Dossier",
                f"- Summary: {_relationship_dossier_summary()}",
                f"- Current facts: {_relationship_dossier_current_facts()}",
                f"- Recent changes: {_relationship_dossier_recent_changes()}",
                f"- Open issues: {_relationship_dossier_issues()}",
                "",
                "### Project Dossiers",
                f"- {_project_dossier_summaries()}",
                "",
                "## Governance Review",
                f"- Claim currentness counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('claim_currentness_counts', {}).items()) or 'None'}",
                f"- Claim review-state counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('claim_review_state_counts', {}).items()) or 'None'}",
                f"- Claim retention counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('claim_retention_class_counts', {}).items()) or 'None'}",
                f"- Freshness counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('freshness_counts', {}).items()) or 'None'}",
                f"- Contradiction counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('contradiction_counts', {}).items()) or 'None'}",
                f"- Dossier availability by task: {_governance_dossier_availability()}",
                f"- Open issue counts: {_governance_issue_counts()}",
                f"- Review-debt dossiers: {', '.join(continuity_governance_report.get('review_debt_dossier_ids', [])) or 'None'}",
                f"- Stale dossiers: {', '.join(continuity_governance_report.get('stale_dossier_ids', [])) or 'None'}",
                f"- Needs refresh dossiers: {', '.join(continuity_governance_report.get('needs_refresh_dossier_ids', [])) or 'None'}",
                f"- Uncertain dossiers: {', '.join(continuity_governance_report.get('uncertain_dossier_ids', [])) or 'None'}",
                f"- Contradicted dossiers: {', '.join(continuity_governance_report.get('contradicted_dossier_ids', [])) or 'None'}",
                f"- Stale graph backing ids: {', '.join(continuity_governance_report.get('stale_graph_backing_ids', [])) or 'None'}",
                f"- Superseded graph backing ids: {', '.join(continuity_governance_report.get('superseded_graph_backing_ids', [])) or 'None'}",
                f"- Packet governance suppression counts: {_packet_suppression_counts()}",
                f"- Embodied decision counts: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('embodied_decision_counts', {}).items()) or 'None'}",
                f"- Embodied policy postures: {', '.join(f'{key}={value}' for key, value in continuity_governance_report.get('embodied_policy_posture_counts', {}).items()) or 'None'}",
                "",
                "### Open Issues",
                f"- {_governance_issue_rows()}",
                "",
                "### Suppressed Packet Rows",
                f"- {_suppressed_packet_rows()}",
                "",
                "### Multimodal Packet Rows",
                f"- {_multimodal_packet_rows()}",
                "",
                "### Embodied Execution Rows",
                f"- {_embodied_execution_rows()}",
                "",
                "## Procedural Review",
                f"- Skill counts: {', '.join(f'{key}={value}' for key, value in procedural_skills.get('skill_counts', {}).items()) or 'None'}",
                f"- Confidence bands: {', '.join(f'{key}={value}' for key, value in procedural_skills.get('confidence_band_counts', {}).items()) or 'None'}",
                f"- Goal families: {', '.join(f'{key}={value}' for key, value in procedural_skill_digest.get('goal_family_counts', {}).items()) or 'None'}",
                f"- Low-confidence skills: {', '.join(procedural_skill_governance_report.get('low_confidence_skill_ids', [])) or 'None'}",
                f"- Retirement reasons: {', '.join(f'{key}={value}' for key, value in procedural_skill_governance_report.get('retirement_reason_counts', {}).items()) or 'None'}",
                "",
                "### Active Skills",
                f"- {_procedural_skill_rows({'active'})}",
                "",
                "### Candidate Skills",
                f"- {_procedural_skill_rows({'candidate'})}",
                "",
                "### Retired / Superseded Skills",
                f"- {_procedural_skill_rows({'retired', 'superseded'})}",
                "",
                "### Failure Signatures",
                f"- {_procedural_failure_signatures()}",
                "",
                "## Practice Director Review",
                f"- Practice family counts: {', '.join(f'{key}={value}' for key, value in shell_practice_inspection.get('scenario_family_counts', {}).items()) or 'None'}",
                f"- Practice reason counts: {', '.join(f'{key}={value}' for key, value in shell_practice_inspection.get('reason_code_counts', {}).items()) or 'None'}",
                f"- Practice digest plan ids: {', '.join(practice_digest.get('recent_plan_ids', [])) or 'None'}",
                "",
                "### Recent Practice Plans",
                f"- {_practice_recent_plans()}",
                "",
                "### Recent Practice Targets",
                f"- {_practice_recent_targets()}",
                "",
                "## Skill Evidence Review",
                f"- Evidence count: {shell_skill_evidence_inspection.get('evidence_count', 0)}",
                f"- Skill status counts: {', '.join(f'{key}={value}' for key, value in shell_skill_evidence_inspection.get('skill_status_counts', {}).items()) or 'None'}",
                f"- Family hypothesis counts: {', '.join(f'{key}={value}' for key, value in shell_skill_evidence_inspection.get('family_hypothesis_counts', {}).items()) or 'None'}",
                f"- Proposal status counts: {', '.join(f'{key}={value}' for key, value in shell_skill_evidence_inspection.get('proposal_status_counts', {}).items()) or 'None'}",
                f"- Blocked promotion reasons: {', '.join(f'{key}={value}' for key, value in shell_skill_evidence_inspection.get('blocked_reason_code_counts', {}).items()) or 'None'}",
                f"- Demotion reasons: {', '.join(f'{key}={value}' for key, value in shell_skill_evidence_inspection.get('demotion_reason_code_counts', {}).items()) or 'None'}",
                f"- Skill evidence digest proposals: {', '.join(skill_evidence_digest.get('proposal_status_counts', {}).keys()) or 'None'}",
                "",
                "### Top Evidence Deltas",
                f"- {_skill_evidence_deltas()}",
                "",
                "### Recent Governance Proposals",
                f"- {_recent_skill_governance()}",
                "",
                "## Adapter Governance Review",
                f"- Adapter state counts: {', '.join(f'{key}={value}' for key, value in shell_adapter_governance_inspection.get('state_counts', {}).items()) or 'None'}",
                f"- Adapter family counts: {', '.join(f'{key}={value}' for key, value in shell_adapter_governance_inspection.get('family_counts', {}).items()) or 'None'}",
                f"- Current defaults: {_current_default_adapter_cards()}",
                f"- Rollback reasons: {', '.join(f'{key}={value}' for key, value in shell_adapter_governance_inspection.get('rollback_reason_counts', {}).items()) or 'None'}",
                f"- Adapter governance digest states: {', '.join(f'{key}={value}' for key, value in adapter_governance_digest.get('state_counts', {}).items()) or 'None'}",
                "",
                "### Recent Adapter Cards",
                f"- {_recent_adapter_cards()}",
                "",
                "### Recent Adapter Reports",
                f"- {_recent_adapter_reports()}",
                "",
                "### Recent Adapter Decisions",
                f"- {_recent_adapter_decisions()}",
                "",
                "## Sim-to-Real Review",
                f"- Readiness counts: {', '.join(f'{key}={value}' for key, value in shell_sim_to_real_inspection.get('readiness_counts', {}).items()) or 'None'}",
                f"- Promotion state counts: {', '.join(f'{key}={value}' for key, value in shell_sim_to_real_inspection.get('promotion_state_counts', {}).items()) or 'None'}",
                f"- Blocked readiness reasons: {', '.join(f'{key}={value}' for key, value in shell_sim_to_real_inspection.get('blocked_reason_counts', {}).items()) or 'None'}",
                f"- Sim-to-real digest states: {', '.join(f'{key}={value}' for key, value in sim_to_real_digest.get('promotion_state_counts', {}).items()) or 'None'}",
                "",
                "### Readiness Rows",
                f"- {_sim_to_real_rows()}",
                "",
                "## Procedural QA Review",
                f"- Case counts: {_procedural_qa_case_counts()}",
                f"- Category counts: {_procedural_qa_category_counts()}",
                f"- Coverage: {_procedural_qa_coverage()}",
                f"- Procedural origins: {', '.join(f'{key}={value}' for key, value in procedural_qa_report.get('procedural_origin_counts', {}).items()) or 'None'}",
                f"- Skill rejection reasons: {', '.join(f'{key}={value}' for key, value in procedural_qa_report.get('skill_rejection_reason_counts', {}).items()) or 'None'}",
                f"- Negative transfer reasons: {', '.join(f'{key}={value}' for key, value in procedural_qa_report.get('negative_transfer_reason_counts', {}).items()) or 'None'}",
                f"- Failed replay cases: {', '.join(procedural_qa_report.get('failed_case_names', [])) or 'None'}",
                "",
                "### Learning Lifecycle",
                f"- {_procedural_learning_lifecycle()}",
                "",
                "### Skill Reuse / Delta",
                f"- {_procedural_reuse_rows()}",
                "",
                "### Negative Transfer / Rejections",
                f"- {_procedural_negative_transfer_rows()}",
                "",
                "### Retirement / Supersession",
                f"- {_procedural_retirement_supersession_rows()}",
                "",
                "## Private Working Memory Review",
                f"- Buffer counts: {_private_working_memory_counts()}",
                f"- State counts: {', '.join(f'{key}={value}' for key, value in private_working_memory.get('state_counts', {}).items()) or 'None'}",
                f"- Evidence kinds: {', '.join(f'{key}={value}' for key, value in private_working_memory.get('evidence_kind_counts', {}).items()) or 'None'}",
                "",
                "### Active Buffer Records",
                f"- {_private_working_memory_active_rows()}",
                "",
                "### Stale / Resolved / Unresolved",
                f"- {_private_working_memory_stale_rows()}",
                "",
                "## Scene World-State Review",
                f"- Entity counts: {', '.join(f'{key}={value}' for key, value in scene_world_state.get('entity_counts', {}).items()) or 'None'}",
                f"- Affordance counts: {', '.join(f'{key}={value}' for key, value in scene_world_state.get('affordance_counts', {}).items()) or 'None'}",
                f"- State counts: {', '.join(f'{key}={value}' for key, value in scene_world_state.get('state_counts', {}).items()) or 'None'}",
                f"- Contradictions: {', '.join(f'{key}={value}' for key, value in scene_world_state.get('contradiction_counts', {}).items()) or 'None'}",
                "",
                "### Active Entities / Affordances",
                f"- {_scene_world_active_rows()}",
                "",
                "### Stale / Contradicted / Expired / Degraded",
                f"- {_scene_world_state_rows()}",
                "",
                "## Active Situation Review",
                f"- Kind counts: {_active_situation_counts()}",
                f"- State counts: {', '.join(f'{key}={value}' for key, value in active_situation_model.get('state_counts', {}).items()) or 'None'}",
                f"- Uncertainty codes: {', '.join(f'{key}={value}' for key, value in active_situation_model.get('uncertainty_code_counts', {}).items()) or 'None'}",
                "",
                "### Active Records",
                f"- {_active_situation_active_rows()}",
                "",
                "### Stale / Unresolved / Linked State",
                f"- {_active_situation_stale_rows()}",
                "",
                "## Context Packet Review",
                *_context_packet_review_lines(),
                "",
                "## Reflection",
                f"- Latest draft: {state.get('latest_reflection_draft_path') or 'None'}",
                f"- Recent cycles: {', '.join(record.get('status', '') for record in reflection_cycles[:5]) or 'None'}",
                f"- Skip reasons: {', '.join(record.get('skip_reason', '') for record in reflection_cycles if record.get('skip_reason')) or 'None'}",
                "",
                "## Legacy Context Selection",
                *_legacy_selection_lines(),
                "",
                "## Replay Regressions",
                f"- Cases run: {len(replay_regressions)}",
                f"- Failing cases: {', '.join(item.get('name', '') for item in failing_cases) or 'None'}",
            ]
        )


__all__ = ["BrainContinuityAuditExporter", "BrainContinuityAuditReport"]

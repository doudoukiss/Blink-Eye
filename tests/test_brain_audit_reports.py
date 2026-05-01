import json
from pathlib import Path

import pytest

from blink.brain._executive import BrainPlanningDraft
from blink.brain.actions import build_brain_capability_registry
from blink.brain.autonomy import (
    BrainCandidateGoal,
    BrainCandidateGoalSource,
    BrainInitiativeClass,
    BrainReevaluationConditionKind,
    BrainReevaluationTrigger,
)
from blink.brain.capabilities import CapabilityRegistry
from blink.brain.evals import BrainContinuityAuditExporter
from blink.brain.executive import BrainExecutive
from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory_v2 import BrainReflectionEngine
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainCommitmentStatus,
    BrainCommitmentWakeRouteKind,
    BrainCommitmentWakeRoutingDecision,
    BrainCommitmentWakeTrigger,
    BrainGoalFamily,
    BrainGoalStatus,
    BrainPlanReviewPolicy,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.runtime_shell import BrainRuntimeShell
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.cli.local_brain_audit import main as local_brain_audit_main
from blink.transcriptions.language import Language
from tests.phase23_fixtures import seed_phase23_surfaces
from tests.phase24_fixtures import seed_phase24_adapter_governance


class SequencedPlanner:
    def __init__(self, *drafts: BrainPlanningDraft | None):
        self._drafts = list(drafts)

    async def __call__(self, request):
        if not self._drafts:
            return None
        return self._drafts.pop(0)


def _draft(
    *,
    summary: str,
    remaining_steps: list[dict],
    assumptions: list[str] | None = None,
    missing_inputs: list[str] | None = None,
    review_policy: str | None = None,
) -> BrainPlanningDraft:
    draft = BrainPlanningDraft.from_dict(
        {
            "summary": summary,
            "remaining_steps": remaining_steps,
            "assumptions": assumptions or [],
            "missing_inputs": missing_inputs or [],
            "review_policy": review_policy,
        }
    )
    assert draft is not None
    return draft


def test_continuity_audit_exporter_writes_json_and_markdown(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="We should continue the Alpha project.",
        assistant_text="Agreed.",
        assistant_summary="Continued the Alpha project.",
        tool_calls=[],
    )
    BrainReflectionEngine(store=store).run_once(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_ids=session_ids,
        trigger="manual",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-held",
            candidate_type="presence_acknowledgement",
            source="perception",
            summary="Hold this candidate until the user turn closes.",
            goal_family="environment",
            urgency=0.6,
            confidence=0.8,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id="candidate-audit-held",
        reason="user_turn_open",
        expected_reevaluation_condition="after the user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-reeval",
            candidate_type="presence_user_reentered",
            source="perception",
            summary="Resume the same candidate through reevaluation.",
            goal_family="environment",
            urgency=0.85,
            confidence=0.9,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
            cooldown_key="cooldown:audit:reeval",
            dedupe_key="dedupe:audit:reeval",
            requires_user_turn_gap=True,
            expires_at="2099-01-01T00:05:00+00:00",
            created_at="2099-01-01T00:00:00+00:00",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_director_non_action(
        candidate_goal_id="candidate-audit-reeval",
        reason="user_turn_open",
        expected_reevaluation_condition="after the user turn ends",
        expected_reevaluation_condition_kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-suppressed",
            candidate_type="presence_attention_returned",
            source="perception",
            summary="Weak attention return should be suppressed.",
            goal_family="environment",
            urgency=0.4,
            confidence=0.3,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_suppressed(
        candidate_goal_id="candidate-audit-suppressed",
        reason="low_confidence",
        reason_details={"confidence": 0.3},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-merge-target",
            candidate_type="presence_user_reentered",
            source="perception",
            summary="Primary scene acknowledgment candidate.",
            goal_family="environment",
            urgency=0.8,
            confidence=0.9,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-merge-source",
            candidate_type="presence_attention_returned",
            source="perception",
            summary="Duplicate attention candidate.",
            goal_family="environment",
            urgency=0.5,
            confidence=0.7,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_merged(
        candidate_goal_id="candidate-audit-merge-source",
        merged_into_candidate_goal_id="candidate-audit-merge-target",
        reason="same_social_slot",
        reason_details={"slot": "attention"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    reevaluation_trigger = store.append_director_reevaluation_triggered(
        trigger=BrainReevaluationTrigger(
            kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
            summary="User turn ended; resume held candidates.",
            details={"turn": "user"},
            source_event_type="user.turn.ended",
            source_event_id="evt-audit-turn-ended",
            ts="2099-01-01T00:00:05+00:00",
        ),
        candidate_goal_ids=["candidate-audit-reeval"],
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    resumed_goal_id = executive.create_goal(
        title="Resume scene acknowledgment",
        intent="autonomy.presence_user_reentered",
        source="presence_director",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        details={"autonomy": {"candidate_goal_id": "candidate-audit-reeval"}},
    )
    store.append_candidate_goal_accepted(
        candidate_goal_id="candidate-audit-reeval",
        goal_id=resumed_goal_id,
        reason="accepted_for_goal_creation",
        reason_details={"selected_by": "reevaluation"},
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
        causal_parent_id=reevaluation_trigger.event_id,
    )
    action_goal_id = executive.create_goal(
        title="Acknowledge scene return",
        intent="autonomy.presence_user_reentered",
        source="presence_director",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        details={"autonomy": {"candidate_goal_id": "candidate-audit-action"}},
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-action",
            candidate_type="presence_user_reentered",
            source="perception",
            summary="Accepted scene action candidate.",
            goal_family="environment",
            urgency=0.9,
            confidence=0.95,
            initiative_class=BrainInitiativeClass.SILENT_ATTENTION_SHIFT.value,
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_candidate_goal_accepted(
        candidate_goal_id="candidate-audit-action",
        goal_id=action_goal_id,
        reason="accepted_for_goal_creation",
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="director",
    )
    executive.create_commitment_goal(
        title="Audit follow-up",
        intent="narrative.commitment",
        source="memory",
        goal_family=BrainGoalFamily.CONVERSATION.value,
        details={"details": "Need a human-visible follow-up"},
    )
    waiting_commitment = next(
        record
        for record in store.list_executive_commitments(user_id=session_ids.user_id, limit=8)
        if record.title == "Audit follow-up"
    )
    executive.defer_commitment(
        commitment_id=waiting_commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.WAITING_USER.value,
            summary="Waiting for an idle thread gap.",
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.THREAD_IDLE.value,
                summary="Wake when the thread is idle.",
            )
        ],
    )
    waiting_commitment = store.get_executive_commitment(
        commitment_id=waiting_commitment.commitment_id
    )
    assert waiting_commitment is not None
    wake_propose = store.append_commitment_wake_triggered(
        commitment=waiting_commitment,
        wake_condition=waiting_commitment.wake_conditions[0],
        trigger=BrainCommitmentWakeTrigger(
            commitment_id=waiting_commitment.commitment_id,
            wake_kind=BrainWakeConditionKind.THREAD_IDLE.value,
            summary="Matched durable commitment wake: thread_idle.",
            details={
                "boundary_kind": "startup_recovery",
                "candidate_goal_id": "candidate-audit-wake-proposal",
                "candidate_type": "commitment_wake_thread_idle",
            },
            source_event_type="assistant.turn.ended",
            source_event_id="evt-audit-wake-boundary",
            ts="2099-01-01T00:00:06+00:00",
        ),
        routing=BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value,
            summary="Route this wake through bounded candidate policy.",
            details={
                "reason": "wake_matched",
                "boundary_kind": "startup_recovery",
                "candidate_goal_id": "candidate-audit-wake-proposal",
                "candidate_type": "commitment_wake_thread_idle",
                "goal_family": BrainGoalFamily.CONVERSATION.value,
            },
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="wake_router",
    )
    store.append_candidate_goal_created(
        candidate_goal=BrainCandidateGoal(
            candidate_goal_id="candidate-audit-wake-proposal",
            candidate_type="commitment_wake_thread_idle",
            source=BrainCandidateGoalSource.COMMITMENT.value,
            summary="Revisit deferred commitment: Audit follow-up",
            goal_family=BrainGoalFamily.CONVERSATION.value,
            urgency=0.7,
            confidence=1.0,
            initiative_class=BrainInitiativeClass.INSPECT_ONLY.value,
            cooldown_key="browser:alpha:commitment:audit-follow-up:thread_idle",
            dedupe_key=f"{waiting_commitment.commitment_id}:{BrainWakeConditionKind.THREAD_IDLE.value}",
            created_at="2099-01-01T00:00:06+00:00",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="wake_router",
        correlation_id=waiting_commitment.commitment_id,
        causal_parent_id=wake_propose.event_id,
    )
    store.append_commitment_wake_triggered(
        commitment=waiting_commitment,
        wake_condition=waiting_commitment.wake_conditions[0],
        trigger=BrainCommitmentWakeTrigger(
            commitment_id=waiting_commitment.commitment_id,
            wake_kind=BrainWakeConditionKind.THREAD_IDLE.value,
            summary="Matched durable commitment wake: thread_idle.",
            details={"boundary_kind": "goal_terminal"},
            source_event_type="goal.completed",
            source_event_id="evt-audit-goal-terminal",
            ts="2099-01-01T00:00:07+00:00",
        ),
        routing=BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.KEEP_WAITING.value,
            summary="Keep waiting because the wake candidate is already current.",
            details={
                "reason": "candidate_already_current",
                "boundary_kind": "goal_terminal",
                "goal_family": BrainGoalFamily.CONVERSATION.value,
            },
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="wake_router",
    )

    executive.create_commitment_goal(
        title="Audit direct resume",
        intent="robot_head.sequence",
        source="interpreter",
        goal_family=BrainGoalFamily.ENVIRONMENT.value,
        details={"capabilities": [{"capability_id": "robot_head.blink"}]},
    )
    resumable_commitment = next(
        record
        for record in store.list_executive_commitments(user_id=session_ids.user_id, limit=8)
        if record.title == "Audit direct resume"
    )
    executive.defer_commitment(
        commitment_id=resumable_commitment.commitment_id,
        reason=BrainBlockedReason(
            kind=BrainBlockedReasonKind.CAPABILITY_BLOCKED.value,
            summary="Robot head is busy.",
            details={"capability_id": "robot_head.blink"},
        ),
        wake_conditions=[
            BrainWakeCondition(
                kind=BrainWakeConditionKind.CONDITION_CLEARED.value,
                summary="Wake when the blocker clears.",
                details={"capability_id": "robot_head.blink"},
            )
        ],
    )
    resumable_commitment = store.get_executive_commitment(
        commitment_id=resumable_commitment.commitment_id
    )
    assert resumable_commitment is not None
    wake_resume = store.append_commitment_wake_triggered(
        commitment=resumable_commitment,
        wake_condition=resumable_commitment.wake_conditions[0],
        trigger=BrainCommitmentWakeTrigger(
            commitment_id=resumable_commitment.commitment_id,
            wake_kind=BrainWakeConditionKind.CONDITION_CLEARED.value,
            summary="Matched durable commitment wake: condition_cleared.",
            details={
                "boundary_kind": "goal_terminal",
                "capability_id": "robot_head.blink",
            },
            source_event_type="goal.completed",
            source_event_id="evt-audit-condition-cleared",
            ts="2099-01-01T00:00:08+00:00",
        ),
        routing=BrainCommitmentWakeRoutingDecision(
            route_kind=BrainCommitmentWakeRouteKind.RESUME_DIRECT.value,
            summary="Resume this commitment directly because the blocker cleared.",
            details={
                "reason": "blocker_cleared",
                "boundary_kind": "goal_terminal",
                "capability_id": "robot_head.blink",
                "goal_family": BrainGoalFamily.ENVIRONMENT.value,
            },
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="wake_router",
    )
    executive.resume_commitment(
        commitment_id=resumable_commitment.commitment_id,
        correlation_id=resumable_commitment.commitment_id,
        causal_parent_id=wake_resume.event_id,
        source="wake_router",
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit",
        replay_cases_dir=Path("tests/fixtures/brain_evals"),
    )

    assert report.json_path is not None
    assert report.markdown_path is not None
    assert report.json_path.exists()
    assert report.markdown_path.exists()

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert "continuity_state" in payload
    assert "autonomy_digest" in payload
    assert "reevaluation_digest" in payload
    assert "wake_digest" in payload
    assert "continuity_graph" in payload
    assert "continuity_graph_digest" in payload
    assert "continuity_governance_report" in payload
    assert "context_packet_digest" in payload
    assert "continuity_dossiers" in payload
    assert "procedural_skills" in payload
    assert "procedural_skill_digest" in payload
    assert "procedural_skill_governance_report" in payload
    assert "procedural_qa_report" in payload
    assert "private_working_memory" in payload
    assert "private_working_memory_digest" in payload
    assert "scene_world_state" in payload
    assert "scene_world_state_digest" in payload
    assert "active_situation_model" in payload
    assert "active_situation_model_digest" in payload
    assert "selection_traces" in payload["continuity_state"]
    assert "packet_traces" in payload["continuity_state"]
    assert "continuity_graph_digest" in payload["continuity_state"]
    assert "continuity_governance_report" in payload["continuity_state"]
    assert "context_packet_digest" in payload["continuity_state"]
    assert "visual_health" in payload["continuity_state"]
    assert "autonomy_ledger" in payload["continuity_state"]
    assert "autonomy_digest" in payload["continuity_state"]
    assert "reevaluation_digest" in payload["continuity_state"]
    assert "wake_digest" in payload["continuity_state"]
    assert "continuity_dossiers" in payload["continuity_state"]
    assert "procedural_skills" in payload["continuity_state"]
    assert "procedural_skill_digest" in payload["continuity_state"]
    assert "procedural_skill_governance_report" in payload["continuity_state"]
    assert "procedural_qa_report" in payload["continuity_state"]
    assert "private_working_memory" in payload["continuity_state"]
    assert "private_working_memory_digest" in payload["continuity_state"]
    assert "scene_world_state" in payload["continuity_state"]
    assert "scene_world_state_digest" in payload["continuity_state"]
    assert "multimodal_autobiography" in payload["continuity_state"]
    assert "multimodal_autobiography_digest" in payload["continuity_state"]
    assert "active_situation_model" in payload["continuity_state"]
    assert "active_situation_model_digest" in payload["continuity_state"]
    for task in (
        "reply",
        "planning",
        "recall",
        "reflection",
        "critique",
        "wake",
        "reevaluation",
        "operator_audit",
        "governance_review",
    ):
        assert payload["continuity_state"]["packet_traces"][task]["task"] == task
    assert payload["context_packet_digest"]["reply"]["temporal_mode"] == "current_first"
    assert "query_text" in payload["context_packet_digest"]["recall"]
    assert "query_text" in payload["context_packet_digest"]["reflection"]
    assert "query_text" in payload["context_packet_digest"]["critique"]
    assert "query_text" in payload["context_packet_digest"]["operator_audit"]
    assert "query_text" in payload["context_packet_digest"]["governance_review"]
    assert payload["private_working_memory_digest"]["buffer_counts"]
    assert "degraded_mode" in payload["scene_world_state_digest"]
    assert "entry_counts" in payload["continuity_state"]["multimodal_autobiography_digest"]
    assert "recent_redacted_rows" in payload["continuity_state"]["multimodal_autobiography_digest"]
    assert payload["active_situation_model_digest"]["kind_counts"]
    assert "## Scene World-State Review" in markdown
    assert "## Active Situation Review" in markdown
    assert "autobiography_entry" in payload["continuity_graph_digest"]["current_node_kind_counts"]
    assert payload["continuity_governance_report"]["freshness_counts"]
    assert "open_issue_rows" in payload["continuity_governance_report"]
    assert "claim_currentness_counts" in payload["continuity_governance_report"]
    assert "claim_review_state_counts" in payload["continuity_governance_report"]
    assert "dossier_availability_counts_by_task" in payload["continuity_governance_report"]
    assert "suppressed_packet_rows" in payload["continuity_governance_report"]
    assert "selected_availability_counts" in payload["context_packet_digest"]["reply"]
    assert "governance_reason_code_counts" in payload["context_packet_digest"]["reply"]
    assert "suppressed_backing_ids" in payload["context_packet_digest"]["reply"]
    assert payload["continuity_state"]["autonomy_digest"]["decision_counts"]["accepted"] >= 1
    assert payload["continuity_state"]["autonomy_digest"]["reason_counts"]["user_turn_open"] >= 1
    assert "pending_family_counts" in payload["continuity_state"]["autonomy_digest"]
    assert "current_family_leaders" in payload["continuity_state"]["autonomy_digest"]
    assert "next_expiry_at" in payload["continuity_state"]["autonomy_digest"]
    assert payload["continuity_state"]["reevaluation_digest"]["current_hold_count"] >= 1
    assert (
        payload["continuity_state"]["reevaluation_digest"]["trigger_counts"][
            BrainReevaluationConditionKind.USER_TURN_CLOSED.value
        ]
        >= 1
    )
    assert payload["continuity_state"]["reevaluation_digest"]["recent_transitions"]
    assert payload["continuity_state"]["wake_digest"]["current_wait_count"] >= 1
    assert (
        payload["continuity_state"]["wake_digest"]["route_counts"][
            BrainCommitmentWakeRouteKind.PROPOSE_CANDIDATE.value
        ]
        >= 1
    )
    assert (
        payload["continuity_state"]["wake_digest"]["route_counts"][
            BrainCommitmentWakeRouteKind.RESUME_DIRECT.value
        ]
        >= 1
    )
    assert (
        payload["continuity_state"]["wake_digest"]["route_counts"][
            BrainCommitmentWakeRouteKind.KEEP_WAITING.value
        ]
        >= 1
    )
    assert payload["continuity_state"]["wake_digest"]["recent_direct_resumes"]
    assert payload["continuity_state"]["wake_digest"]["recent_candidate_proposals"]
    assert payload["continuity_state"]["wake_digest"]["recent_keep_waiting"]
    assert "policy_posture_counts" in payload["continuity_state"]["wake_digest"]
    assert "approval_requirement_counts" in payload["continuity_state"]["wake_digest"]
    assert "why_not_reason_code_counts" in payload["continuity_state"]["wake_digest"]
    assert "policy_posture_counts" in payload["continuity_state"]["reevaluation_digest"]
    assert "why_not_reason_code_counts" in payload["continuity_state"]["reevaluation_digest"]
    assert "executive_policy_audit" in payload["continuity_state"]
    assert "policy_posture_counts" in payload["continuity_state"]["executive_policy_audit"]
    assert payload["continuity_state"]["reflection_cycles"]
    assert payload["continuity_state"]["latest_reflection_draft_path"] is not None
    assert "self_core" in payload["continuity_state"]["core_blocks"]
    assert payload["continuity_state"]["core_block_versions"]["self_core"]
    relationship_dossier = next(
        item
        for item in payload["continuity_state"]["continuity_dossiers"]["dossiers"]
        if item["kind"] == "relationship"
    )
    assert relationship_dossier["summary"]
    assert (
        relationship_dossier["summary_evidence"]["entry_ids"]
        or relationship_dossier["summary_evidence"]["claim_ids"]
    )
    assert relationship_dossier["governance"]["task_availability"]
    assert "last_refresh_cause" in relationship_dossier["governance"]
    assert any(
        item["project_key"] == "Alpha"
        for item in payload["continuity_state"]["continuity_dossiers"]["dossiers"]
        if item["kind"] == "project"
    )
    assert payload["replay_regressions"]
    assert payload["proof_surface"]["canonical_entrypoint"] == "./scripts/test-brain-core.sh"
    assert payload["proof_surface"]["always_run_lanes"] == ["fast", "proof", "fuzz-smoke"]
    assert payload["proof_surface"]["opt_in_lanes"] == ["atheris"]
    assert "# Blink Continuity Audit" in markdown
    assert "## Brain-Core Proof Surface" in markdown
    assert "Canonical entrypoint: ./scripts/test-brain-core.sh" in markdown
    assert "Packet modes built:" in markdown
    assert "### Remaining Thin Areas" in markdown
    assert "## Memory Health" in markdown
    assert "## Wake Review" in markdown
    assert "Wake route counts" in markdown
    assert "### Current Waiting Commitments" in markdown
    assert "### Recent Direct Resumes" in markdown
    assert "### Recent Candidate Proposals" in markdown
    assert "### Recent Keep-Waiting Decisions" in markdown
    assert "## Autonomy Review" in markdown
    assert "Pending families" in markdown
    assert "Next expiry" in markdown
    assert "### Recent Accepted Actions" in markdown
    assert "### Recent Suppressions" in markdown
    assert "### Recent Merges" in markdown
    assert "### Recent Non-Actions" in markdown
    assert "## Reevaluation Review" in markdown
    assert "### Current Held Candidates" in markdown
    assert "### Recent Reevaluation Triggers" in markdown
    assert "### Recent Hold -> Reevaluation Flows" in markdown
    assert "## Dossier Review" in markdown
    assert "## Graph Review" in markdown
    assert "## Governance Review" in markdown
    assert "Claim currentness counts" in markdown
    assert "Dossier availability by task" in markdown
    assert "Packet governance suppression counts" in markdown
    assert "### Suppressed Packet Rows" in markdown
    assert "## Procedural Review" in markdown
    assert "## Procedural QA Review" in markdown
    assert "### Active Skills" in markdown
    assert "### Candidate Skills" in markdown
    assert "### Retired / Superseded Skills" in markdown
    assert "### Failure Signatures" in markdown
    assert "### Learning Lifecycle" in markdown
    assert "### Skill Reuse / Delta" in markdown
    assert "### Negative Transfer / Rejections" in markdown
    assert "### Retirement / Supersession" in markdown
    assert "## Private Working Memory Review" in markdown
    assert "### Active Buffer Records" in markdown
    assert "### Stale / Resolved / Unresolved" in markdown
    assert "## Context Packet Review" in markdown
    assert "selected availability" in markdown
    assert "governance drops" in markdown
    assert "annotated ids" in markdown
    assert "suppressed ids" in markdown
    assert "## Legacy Context Selection" in markdown
    assert "Recall query:" in markdown
    assert "Reflection query:" in markdown
    assert "Critique query:" in markdown
    assert "### Recent Supersession Links" in markdown
    assert "### Current Commitment / Plan Links" in markdown
    assert "### Open Issues" in markdown
    assert "### Relationship Dossier" in markdown
    assert "### Project Dossiers" in markdown
    assert "## Reflection" in markdown
    assert "## Visual Health" in markdown
    assert "relationship" in markdown
    assert "rev=" in markdown
    assert "user_turn_open" in markdown
    assert "low_confidence" in markdown
    assert "skill_reuse_mismatch" in markdown
    assert "same_social_slot" in markdown
    assert "accepted_for_goal_creation" in markdown
    assert "user_turn_closed" in markdown
    assert "candidate_already_current" in markdown
    assert "blocker_cleared" in markdown
    assert "claims=" in markdown
    assert payload["procedural_qa_report"]["case_counts"]["total"] >= 1
    assert payload["procedural_qa_report"]["coverage_flags"]["skill_learning"] is True
    assert payload["procedural_qa_report"]["coverage_flags"]["skill_reuse"] is True
    assert payload["procedural_qa_report"]["coverage_flags"]["negative_transfer"] is True
    assert payload["procedural_qa_report"]["coverage_flags"]["retirement"] is True
    assert payload["procedural_qa_report"]["coverage_flags"]["supersession"] is True


def test_continuity_audit_exporter_uses_explicit_context_queries(tmp_path):
    store = BrainStore(path=tmp_path / "explicit-queries.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="explicit-queries")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Review explicit planning query",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
    )
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="默认回复查询不应出现。",
        assistant_text="收到。",
        assistant_summary="收到。",
        tool_calls=[],
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.ZH,
        output_dir=tmp_path / "explicit-audit",
        context_queries={
            "reply": "请解释当前关系摘要",
            "planning": "请解释挂起的规划提案",
        },
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["context_queries"]["reply"] == "请解释当前关系摘要"
    assert payload["context_queries"]["planning"] == "请解释挂起的规划提案"
    assert payload["continuity_state"]["context_queries"]["reply"] == "请解释当前关系摘要"
    assert payload["continuity_state"]["context_queries"]["planning"] == "请解释挂起的规划提案"
    assert "Reply query: 请解释当前关系摘要" in markdown
    assert "Planning query: 请解释挂起的规划提案" in markdown
    assert "Compatibility-only selector view" in markdown


def test_continuity_audit_exporter_falls_back_to_latest_episode_reply_query(tmp_path):
    store = BrainStore(path=tmp_path / "episode-fallback.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="episode-fallback")
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="请根据最近一轮对话生成回复上下文。",
        assistant_text="好的。",
        assistant_summary="好的。",
        tool_calls=[],
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.ZH,
        output_dir=tmp_path / "episode-audit",
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["context_queries"]["reply"] == "请根据最近一轮对话生成回复上下文。"
    assert (
        payload["continuity_state"]["context_queries"]["reply"]
        == "请根据最近一轮对话生成回复上下文。"
    )
    assert "Reply query: 请根据最近一轮对话生成回复上下文。" in markdown


def test_continuity_audit_exporter_derives_planning_query_from_active_goal(tmp_path):
    store = BrainStore(path=tmp_path / "planning-query.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planning-query")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Prepare maintenance review",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "planning-audit",
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["context_queries"]["planning"] == (
        f"{BrainGoalFamily.MEMORY_MAINTENANCE.value}: Prepare maintenance review"
    )
    assert payload["continuity_state"]["context_queries"]["planning"] == (
        f"{BrainGoalFamily.MEMORY_MAINTENANCE.value}: Prepare maintenance review"
    )
    assert "Planning query: memory_maintenance: Prepare maintenance review" in markdown


@pytest.mark.asyncio
async def test_continuity_audit_exporter_includes_planning_digest_and_markdown(tmp_path):
    store = BrainStore(path=tmp_path / "planning.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="planner-audit")
    planner = SequencedPlanner(
        _draft(
            summary="Adopt a safe maintenance plan automatically.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Reject an unknown capability proposal.",
            remaining_steps=[
                {"capability_id": "unknown.capability", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Hold for user-owned planning inputs.",
            remaining_steps=[
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            missing_inputs=["which report format to use"],
            review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        ),
        _draft(
            summary="Create the initial revision target plan.",
            remaining_steps=[
                {"capability_id": "maintenance.review_memory_health", "arguments": {}},
                {"capability_id": "reporting.record_maintenance_note", "arguments": {}},
            ],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
        _draft(
            summary="Revise the unfinished tail after the first step completed.",
            remaining_steps=[
                {"capability_id": "maintenance.review_scheduler_backpressure", "arguments": {}},
            ],
            assumptions=["the first maintenance step is already complete"],
            review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        ),
    )
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=build_brain_capability_registry(language=Language.EN),
        planning_callback=planner,
    )

    adopted_goal_id = executive.create_commitment_goal(
        title="Audit adopted plan",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    adopted_result = await executive.request_plan_proposal(goal_id=adopted_goal_id)

    rejected_goal_id = executive.create_commitment_goal(
        title="Audit rejected plan",
        intent="narrative.commitment",
        source="test",
        goal_family=BrainGoalFamily.CONVERSATION.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"durable_commitment": True},
    )
    rejected_result = await executive.request_plan_proposal(goal_id=rejected_goal_id)

    pending_goal_id = executive.create_commitment_goal(
        title="Audit pending plan",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    pending_result = await executive.request_plan_proposal(goal_id=pending_goal_id)

    revision_goal_id = executive.create_commitment_goal(
        title="Audit revised plan",
        intent="maintenance.review",
        source="test",
        goal_family=BrainGoalFamily.MEMORY_MAINTENANCE.value,
        goal_status=BrainGoalStatus.OPEN.value,
        details={"survive_restart": True},
    )
    await executive.request_plan_proposal(goal_id=revision_goal_id)
    run_result = await executive.run_once()
    revision_commitment = next(
        record
        for record in store.list_executive_commitments(user_id=session_ids.user_id, limit=16)
        if record.title == "Audit revised plan"
    )
    revised_result = await executive.request_commitment_plan_revision(
        commitment_id=revision_commitment.commitment_id
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit-planning",
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    markdown = report.markdown_path.read_text(encoding="utf-8")
    planning_digest = payload["continuity_state"]["planning_digest"]

    assert adopted_result.outcome == "auto_adopted"
    assert rejected_result.outcome == "rejected"
    assert pending_result.outcome == "needs_user_review"
    assert run_result.progressed is True
    assert revised_result.outcome == "auto_adopted"
    assert payload["planning_digest"] == planning_digest
    assert payload["context_packet_digest"]["planning"]["selected_anchor_counts"]
    assert payload["continuity_graph_digest"]["current_commitment_plan_links"]
    assert planning_digest["current_plan_state_count"] >= 4
    assert planning_digest["current_pending_proposal_count"] >= 1
    assert planning_digest["outcome_counts"]["adopted"] >= 2
    assert planning_digest["outcome_counts"]["rejected"] >= 1
    assert planning_digest["outcome_counts"]["pending_user_review"] >= 1
    assert planning_digest["reason_counts"]["bounded_plan_available"] >= 1
    assert planning_digest["reason_counts"]["unsupported_capability"] >= 1
    assert "policy_posture_counts" in planning_digest
    assert "approval_requirement_counts" in planning_digest
    assert "why_not_reason_code_counts" in planning_digest
    assert pending_result.proposal.plan_proposal_id in [
        entry["plan_proposal_id"] for entry in planning_digest["current_pending_proposals"]
    ]
    assert any(
        "decision_reason_codes" in entry for entry in planning_digest["current_pending_proposals"]
    )
    assert planning_digest["recent_adoptions"]
    assert planning_digest["recent_rejections"]
    assert planning_digest["recent_revision_flows"]
    assert revised_result.proposal.plan_proposal_id in [
        entry["plan_proposal_id"] for entry in planning_digest["recent_revision_flows"]
    ]
    assert "## Planning Review" in markdown
    assert "## Graph Review" in markdown
    assert "## Context Packet Review" in markdown
    assert "### Current Pending Plan Proposals" in markdown
    assert "### Recent Adopted Plans" in markdown
    assert "### Recent Rejected Plans" in markdown
    assert "### Recent Revision Flows" in markdown
    assert "needs_user_review" in markdown
    assert "unsupported_capability" in markdown
    assert "bounded_plan_available" in markdown
    assert "goal.repaired" in markdown


def test_local_brain_audit_routes_through_shell_and_renders_runtime_shell_digest(
    tmp_path, capsys
):
    db_path = tmp_path / "brain.db"
    store = BrainStore(path=db_path)
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Pause for audit export",
        intent="narrative.commitment",
        source="test",
        details={"summary": "Need an auditable shell trail."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    store.close()

    shell = BrainRuntimeShell.open(
        brain_db_path=db_path,
        runtime_kind="browser",
        client_id="alpha",
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        language=Language.EN,
    )
    try:
        shell.interrupt_commitment(
            commitment_id=commitment.commitment_id,
            reason_summary="Pause before exporting the audit.",
        )
        shell.run_reflection_once(trigger="manual")
    finally:
        shell.close()

    (tmp_path / "empty-replay").mkdir()
    exit_code = local_brain_audit_main(
        [
            "--brain-db-path",
            str(db_path),
            "--runtime-kind",
            "browser",
            "--client-id",
            "alpha",
            "--language",
            "en",
            "--output-dir",
            str(tmp_path / "audit"),
            "--replay-cases-dir",
            str(tmp_path / "empty-replay"),
        ]
    )
    output = capsys.readouterr().out.strip().splitlines()
    json_path = Path(output[0].split("=", 1)[1])
    markdown_path = Path(output[1].split("=", 1)[1])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert output[0].startswith("audit_json=")
    assert output[1].startswith("audit_markdown=")
    assert payload["runtime_shell_digest"]["control_counts"]["interrupt"] >= 1
    assert payload["runtime_shell_digest"]["artifact_action_counts"]["reflection"] >= 1
    assert "## Runtime Shell Review" in markdown


def test_audit_export_includes_multimodal_autobiography_counts_and_redacted_rows(tmp_path):
    from blink.brain.projections import BrainGovernanceReasonCode
    from tests.test_brain_memory_v2 import (
        _scene_world_projection_for_multimodal,
        _seed_scene_episode,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="audit-multimodal")
    projection = _scene_world_projection_for_multimodal(
        scope_id="browser:presence",
        source_event_ids=["evt-audit-scene"],
        updated_at="2026-01-01T00:00:20+00:00",
        include_person=True,
    )
    entry = _seed_scene_episode(
        store,
        session_ids,
        projection=projection,
        start_second=20,
        include_attention=True,
    )
    assert entry is not None
    store.redact_autobiographical_entry(
        entry.entry_id,
        redacted_summary="Redacted scene episode.",
        source_event_id="evt-audit-redact",
        reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
        event_context=store._memory_event_context(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            source="test",
            correlation_id="audit-redact",
        ),
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit-multimodal",
    )
    assert report.json_path is not None
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    digest = payload["continuity_state"]["multimodal_autobiography_digest"]
    runtime_shell_digest = payload["continuity_state"]["runtime_shell_digest"]
    packet_digest = payload["continuity_state"]["context_packet_digest"]
    governance_report = payload["continuity_state"]["continuity_governance_report"]
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert digest["entry_counts"]["privacy"]["redacted"] >= 1
    assert digest["entry_counts"]["review"]["resolved"] >= 1
    assert digest["recent_redacted_rows"]
    assert digest["recent_redacted_rows"][0]["rendered_summary"] == "Redacted scene episode."
    assert runtime_shell_digest["multimodal_inspection"]["recent_redacted_rows"]
    assert (
        runtime_shell_digest["multimodal_inspection"]["latest_source_presence_scope_key"]
        == "browser:presence"
    )
    assert packet_digest["reply"]["scene_episode_trace"]["suppressed_entry_ids"]
    assert packet_digest["operator_audit"]["scene_episode_trace"]["selected_entry_ids"]
    assert governance_report["multimodal_packet_rows"]
    assert "Recent shell scene episodes" in markdown
    assert "Multimodal Packet Rows" in markdown


def test_audit_export_includes_predictive_digests_and_packet_traces(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_goal_updated,
        _append_robot_action_outcome,
        _append_scene_changed,
        _ensure_blocks,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    _ensure_blocks(store)
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="audit-predictive")
    _append_body_state(store, session_ids, second=1)
    _append_scene_changed(store, session_ids, second=2)
    _append_goal_created(store, session_ids, second=3)
    _append_scene_changed(store, session_ids, second=4)
    _append_robot_action_outcome(store, session_ids, second=5, accepted=False)
    _append_scene_changed(
        store,
        session_ids,
        second=40,
        include_person=False,
        affordance_availability="blocked",
        camera_fresh=False,
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit-predictive",
        context_queries={"operator_audit": "Audit predictive traces."},
    )
    assert report.json_path is not None
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    predictive_digest = payload["continuity_state"]["predictive_digest"]
    runtime_shell_digest = payload["continuity_state"]["runtime_shell_digest"]
    packet_digest = payload["continuity_state"]["context_packet_digest"]
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["predictive_world_model"]
    assert payload["predictive_digest"]
    assert predictive_digest["active_kind_counts"]
    assert runtime_shell_digest["predictive_inspection"]["active_kind_counts"]
    assert runtime_shell_digest["predictive_inspection"]["packet_prediction_trace_by_task"]
    assert packet_digest["reply"]["prediction_trace"]["suppressed_prediction_ids"]
    assert packet_digest["operator_audit"]["prediction_trace"]["selected_prediction_ids"]
    assert packet_digest["operator_audit"]["prediction_trace"]["resolution_kind_counts"]["invalidated"] >= 1
    assert "Recent shell active predictions" in markdown
    assert "Predictive active kinds" in markdown


def test_audit_export_includes_practice_and_skill_evidence_sections(tmp_path):
    store = BrainStore(path=tmp_path / "brain_phase23.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="audit-phase23")
    seed_phase23_surfaces(
        store=store,
        session_ids=session_ids,
        output_dir=tmp_path / "practice_artifacts",
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit-phase23",
        context_queries={"operator_audit": "Audit practice planning and skill evidence."},
    )
    assert report.json_path is not None
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    runtime_shell_digest = payload["continuity_state"]["runtime_shell_digest"]
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["continuity_state"]["practice_director"]
    assert payload["continuity_state"]["skill_evidence_ledger"]
    assert payload["continuity_state"]["skill_governance"]
    assert payload["continuity_state"]["practice_digest"]["recent_plan_ids"]
    assert payload["continuity_state"]["skill_evidence_digest"]["proposal_status_counts"]
    assert runtime_shell_digest["practice_inspection"]["recent_plans"]
    assert runtime_shell_digest["skill_evidence_inspection"]["recent_governance_proposals"]
    assert "Practice Director Review" in markdown
    assert "Skill Evidence Review" in markdown


def test_audit_export_includes_adapter_governance_sections(tmp_path):
    store = BrainStore(path=tmp_path / "brain_phase24.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="audit-phase24")
    seed_phase24_adapter_governance(
        store=store,
        session_ids=session_ids,
        output_dir=tmp_path / "phase24_artifacts",
    )

    report = BrainContinuityAuditExporter(store=store).export(
        session_ids=session_ids,
        presence_scope_key="browser:presence",
        language=Language.EN,
        output_dir=tmp_path / "audit-phase24",
        context_queries={"operator_audit": "Audit adapter governance and sim-to-real readiness."},
    )
    assert report.json_path is not None
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    runtime_shell_digest = payload["continuity_state"]["runtime_shell_digest"]
    markdown = report.markdown_path.read_text(encoding="utf-8")

    assert payload["continuity_state"]["adapter_governance"]
    assert payload["continuity_state"]["adapter_governance_digest"]["state_counts"]
    assert payload["continuity_state"]["sim_to_real_digest"]["promotion_state_counts"]
    assert runtime_shell_digest["adapter_governance_inspection"]["recent_cards"]
    assert runtime_shell_digest["sim_to_real_inspection"]["readiness_reports"]
    assert "Adapter Governance Review" in markdown
    assert "Sim-to-Real Review" in markdown


@pytest.mark.asyncio
async def test_audit_export_includes_embodied_digests_and_execution_rows(tmp_path):
    from blink.embodiment.robot_head.simulation import (
        RobotHeadSimulationConfig,
        SimulationDriver,
    )
    from tests.test_brain_embodied_executive import _build_runtime, _create_robot_goal

    runtime, controller, session_ids = _build_runtime(
        tmp_path,
        client_id="audit-embodied",
        driver=SimulationDriver(
            config=RobotHeadSimulationConfig(trace_dir=tmp_path / "simulation-audit"),
        ),
    )
    try:
        goal_id = _create_robot_goal(runtime, title="Audit one embodied execution trace")
        await runtime.executive.request_plan_proposal(goal_id=goal_id)
        await runtime.executive.run_once()

        report = BrainContinuityAuditExporter(store=runtime.store).export(
            session_ids=session_ids,
            presence_scope_key=runtime.presence_scope_key,
            language=Language.EN,
            output_dir=tmp_path / "audit-embodied",
            context_queries={"operator_audit": "Audit embodied execution traces."},
        )
        assert report.json_path is not None
        payload = json.loads(report.json_path.read_text(encoding="utf-8"))
        runtime_shell_digest = payload["continuity_state"]["runtime_shell_digest"]
        governance_report = payload["continuity_state"]["continuity_governance_report"]
        markdown = report.markdown_path.read_text(encoding="utf-8")

        assert payload["embodied_executive"]
        assert payload["embodied_digest"]["recent_execution_trace_count"] >= 1
        assert runtime_shell_digest["embodied_inspection"]["current_intent"]["intent_kind"] == (
            "execute_action"
        )
        assert runtime_shell_digest["embodied_inspection"]["recent_execution_traces"]
        assert governance_report["embodied_execution_rows"]
        assert governance_report["embodied_decision_counts"]["selected"] >= 1
        assert "Embodied Execution Rows" in markdown
        assert "Recent shell embodied traces" in markdown
    finally:
        await controller.close()
        runtime.close()

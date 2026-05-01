import json

import pytest

from blink.brain.capabilities import CapabilityExecutionResult, CapabilityRegistry
from blink.brain.context_surfaces import BrainContextSurfaceBuilder, render_user_profile_summary
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import load_default_agent_blocks
from blink.brain.memory import register_memory_tools
from blink.brain.memory_layers.exports import BrainMemoryExporter
from blink.brain.memory_v2 import (
    BrainAutobiographyEntryKind,
    BrainClaimCurrentnessStatus,
    BrainClaimGovernanceProjection,
    BrainClaimRetentionClass,
    BrainClaimReviewState,
    BrainContinuityDossierAvailability,
    BrainContinuityDossierContradiction,
    BrainContinuityDossierEvidenceRef,
    BrainContinuityDossierFactRecord,
    BrainContinuityDossierFreshness,
    BrainContinuityDossierGovernanceRecord,
    BrainContinuityDossierIssueRecord,
    BrainContinuityDossierKind,
    BrainContinuityDossierProjection,
    BrainContinuityDossierRecord,
    BrainContinuityDossierTaskAvailability,
    BrainContinuityGraphEdgeKind,
    BrainContinuityGraphEdgeRecord,
    BrainContinuityGraphNodeKind,
    BrainContinuityGraphNodeRecord,
    BrainContinuityGraphProjection,
    BrainContinuityQuery,
    BrainCoreMemoryBlockKind,
    BrainGovernanceReasonCode,
    BrainMultimodalAutobiographyPrivacyClass,
    BrainProceduralActivationConditionRecord,
    BrainProceduralEffectRecord,
    BrainProceduralExecutionTraceRecord,
    BrainProceduralFailureSignatureRecord,
    BrainProceduralInvariantRecord,
    BrainProceduralOutcomeKind,
    BrainProceduralOutcomeRecord,
    BrainProceduralSkillProjection,
    BrainProceduralSkillRecord,
    BrainProceduralSkillStatsRecord,
    BrainProceduralSkillStatus,
    BrainProceduralStepTraceRecord,
    BrainProceduralTraceProjection,
    BrainProceduralTraceStatus,
    ClaimLedger,
    ContinuityRetriever,
    parse_multimodal_autobiography_record,
)
from blink.brain.projections import (
    BrainBlockedReason,
    BrainGoal,
    BrainGoalStep,
    BrainPlanProposal,
    BrainPlanProposalDecision,
    BrainPlanProposalSource,
    BrainPlanReviewPolicy,
    BrainSceneWorldAffordanceAvailability,
    BrainSceneWorldAffordanceRecord,
    BrainSceneWorldEntityKind,
    BrainSceneWorldEntityRecord,
    BrainSceneWorldEvidenceKind,
    BrainSceneWorldProjection,
    BrainSceneWorldRecordState,
)
from blink.brain.replay import BrainReplayHarness, BrainReplayScenario
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.transcriptions.language import Language


class DummyLLM:
    def __init__(self):
        self.registered_functions = {}

    def register_function(self, function_name, handler):
        self.registered_functions[function_name] = handler


async def _call_tool(handler, arguments):
    from blink.function_calling import FunctionCallParams
    from blink.processors.aggregators.llm_context import LLMContext

    payload = {}

    async def result_callback(result, properties=None):
        payload["result"] = result

    params = FunctionCallParams(
        function_name="tool",
        tool_call_id="tool-call-1",
        arguments=arguments,
        llm=DummyLLM(),
        context=LLMContext(),
        result_callback=result_callback,
    )
    await handler(params)
    return payload["result"]


def _graph_backing_ids(graph: BrainContinuityGraphProjection, node_ids: list[str]) -> set[str]:
    nodes = {record.node_id: record.backing_record_id for record in graph.nodes}
    return {nodes[node_id] for node_id in node_ids if node_id in nodes}


def _dossier_by_kind(
    projection: BrainContinuityDossierProjection,
    kind: str,
    *,
    project_key: str | None = None,
) -> BrainContinuityDossierRecord:
    for record in projection.dossiers:
        if record.kind != kind:
            continue
        if project_key is not None and record.project_key != project_key:
            continue
        return record
    raise AssertionError(f"Missing dossier kind={kind} project_key={project_key}")


def _all_brain_events(store: BrainStore):
    rows = store._conn.execute("SELECT * FROM brain_events ORDER BY id ASC").fetchall()
    return [store._brain_event_from_row(row) for row in rows]


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-01-01T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def _multimodal_event_context(store: BrainStore, session_ids, *, correlation_id: str):
    return store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id=correlation_id,
    )


def _scene_world_projection_for_multimodal(
    *,
    scope_id: str,
    source_event_ids: list[str],
    updated_at: str,
    degraded_mode: str = "healthy",
    include_person: bool = True,
    affordance_availability: str = BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
    object_state: str = BrainSceneWorldRecordState.ACTIVE.value,
) -> BrainSceneWorldProjection:
    projection = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id=scope_id,
        degraded_mode=degraded_mode,
        degraded_reason_codes=(["scene_stale"] if degraded_mode != "healthy" else []),
        updated_at=updated_at,
        entities=[
            *(
                [
                    BrainSceneWorldEntityRecord(
                        entity_id="scene-person-1",
                        entity_kind=BrainSceneWorldEntityKind.PERSON.value,
                        canonical_label="Ada",
                        summary="Ada is at the desk.",
                        state=BrainSceneWorldRecordState.ACTIVE.value,
                        evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                        confidence=0.91,
                        source_event_ids=list(source_event_ids),
                        observed_at=updated_at,
                        updated_at=updated_at,
                        details={"identity_hint": "person"},
                    )
                ]
                if include_person
                else []
            ),
            BrainSceneWorldEntityRecord(
                entity_id="scene-desk-1",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="Desk",
                summary="A work desk is visible.",
                state=object_state,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                confidence=0.73,
                source_event_ids=list(source_event_ids),
                observed_at=updated_at,
                updated_at=updated_at,
                details={"surface": "desk"},
            ),
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="scene-aff-1",
                entity_id="scene-desk-1",
                capability_family="inspect",
                summary="The desk can be inspected.",
                availability=affordance_availability,
                confidence=0.7,
                source_event_ids=list(source_event_ids),
                observed_at=updated_at,
                updated_at=updated_at,
                details={"tool": "camera"},
            )
        ],
    )
    projection.sync_lists()
    return projection


def _seed_scene_episode(
    store: BrainStore,
    session_ids,
    *,
    projection: BrainSceneWorldProjection,
    start_second: int,
    include_attention: bool = False,
    isolated_recent_events: bool = False,
):
    scene_event = BrainEventRecord(
        id=0,
        event_id=f"evt-scene-{start_second}",
        event_type=BrainEventType.SCENE_CHANGED,
        ts=_ts(start_second),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=f"scene-{start_second}",
        causal_parent_id=None,
        confidence=1.0,
        payload_json=json.dumps(
            {"presence_scope_key": projection.scope_id, "summary": projection.updated_at},
            ensure_ascii=False,
            sort_keys=True,
        ),
        tags_json="[]",
    )
    store.import_brain_event(scene_event)
    recent_events = [scene_event]
    if include_attention:
        attention_event = BrainEventRecord(
            id=0,
            event_id=f"evt-attention-{start_second}",
            event_type=BrainEventType.ATTENTION_CHANGED,
            ts=_ts(start_second + 1),
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            correlation_id=f"scene-{start_second}",
            causal_parent_id=scene_event.event_id,
            confidence=1.0,
            payload_json=json.dumps(
                {"presence_scope_key": projection.scope_id, "attention": "camera"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            tags_json="[]",
        )
        store.import_brain_event(attention_event)
        recent_events.append(attention_event)
    return store.refresh_scene_episode_autobiography(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_id=session_ids.session_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=projection.scope_id,
        scene_world_state=projection,
        recent_events=(
            recent_events
            if isolated_recent_events
            else store.recent_brain_events(
                user_id=session_ids.user_id,
                thread_id=session_ids.thread_id,
                limit=12,
            )
        ),
        source_event_id=scene_event.event_id,
        updated_at=projection.updated_at,
        event_context=_multimodal_event_context(
            store,
            session_ids,
            correlation_id=f"scene-episode-{start_second}",
        ),
    )


def _append_completed_procedural_goal(
    store: BrainStore,
    session_ids,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
    goal_family: str = "memory_maintenance",
    current_plan_revision: int = 1,
    plan_revision: int = 1,
    supersedes_plan_proposal_id: str | None = None,
):
    goal_created = BrainGoal(
        goal_id=goal_id,
        title=goal_title,
        intent="maintenance.review",
        source="test",
        goal_family=goal_family,
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    proposal = BrainPlanProposal(
        plan_proposal_id=proposal_id,
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=f"Execute {goal_title}.",
        current_plan_revision=current_plan_revision,
        plan_revision=plan_revision,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        supersedes_plan_proposal_id=supersedes_plan_proposal_id,
        created_at=_ts(start_second),
    )
    proposed_event = store.append_planning_proposed(
        proposal=proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        ts=_ts(start_second),
    )
    adopted_event = store.append_planning_adopted(
        proposal=proposal,
        decision=BrainPlanProposalDecision(
            summary=f"Adopt {goal_title}.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=_ts(start_second + 1),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status="open",
                details={"current_plan_proposal_id": proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=plan_revision,
                last_summary=f"Adopt {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=_ts(start_second + 2),
    )
    completed_steps: list[BrainGoalStep] = []
    current_second = start_second + 3
    for step_index, capability_id in enumerate(sequence):
        request_event = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=goal_id,
            ts=_ts(current_second),
        )
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_COMPLETED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": goal_id,
                "capability_id": capability_id,
                "step_index": step_index,
                "result": CapabilityExecutionResult.success(
                    capability_id=capability_id,
                    summary=f"Completed {capability_id}.",
                    output={"slot": step_index},
                ).model_dump(),
            },
            correlation_id=goal_id,
            causal_parent_id=request_event.event_id,
            ts=_ts(current_second + 1),
        )
        completed_steps.append(
            BrainGoalStep(
                capability_id=capability_id,
                status="completed",
                attempts=1,
                summary=f"Completed {capability_id}.",
                output={"slot": step_index},
            )
        )
        current_second += 2
    store.append_brain_event(
        event_type=BrainEventType.GOAL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=goal_id,
                title=goal_title,
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=commitment_id,
                status="completed",
                details={"current_plan_proposal_id": proposal_id},
                steps=completed_steps,
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=plan_revision,
                last_summary=f"Completed {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "completed"},
        },
        correlation_id=goal_id,
        ts=_ts(current_second),
    )
    return proposal


def _append_failed_procedural_goal(
    store: BrainStore,
    session_ids,
    *,
    goal_id: str,
    commitment_id: str,
    goal_title: str,
    proposal_id: str,
    sequence: list[str],
    start_second: int,
    failure_reason: str,
    goal_family: str = "memory_maintenance",
):
    failed_goal_id = goal_id
    failed_commitment_id = commitment_id
    goal_created = BrainGoal(
        goal_id=failed_goal_id,
        title=f"{goal_title} failure",
        intent="maintenance.review",
        source="test",
        goal_family=goal_family,
        commitment_id=failed_commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal_created.as_dict()},
        correlation_id=failed_goal_id,
        ts=_ts(start_second + 20),
    )
    failed_proposal = BrainPlanProposal(
        plan_proposal_id=f"{proposal_id}-failed",
        goal_id=failed_goal_id,
        commitment_id=failed_commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary=f"Attempt {goal_title} and fail.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
        details={"request_kind": "initial_plan"},
        created_at=_ts(start_second + 20),
    )
    proposed_event = store.append_planning_proposed(
        proposal=failed_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=failed_goal_id,
        ts=_ts(start_second + 20),
    )
    adopted_event = store.append_planning_adopted(
        proposal=failed_proposal,
        decision=BrainPlanProposalDecision(
            summary=f"Adopt failing {goal_title}.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=failed_goal_id,
        causal_parent_id=proposed_event.event_id,
        ts=_ts(start_second + 21),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=failed_goal_id,
                title=f"{goal_title} failure",
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=failed_commitment_id,
                status="open",
                details={"current_plan_proposal_id": failed_proposal.plan_proposal_id},
                steps=[BrainGoalStep(capability_id=capability_id) for capability_id in sequence],
                plan_revision=1,
                last_summary=f"Adopt failing {goal_title}.",
            ).as_dict(),
            "commitment": {"commitment_id": failed_commitment_id, "status": "active"},
        },
        correlation_id=failed_goal_id,
        causal_parent_id=adopted_event.event_id,
        ts=_ts(start_second + 22),
    )
    for step_index, capability_id in enumerate(sequence):
        request_event = store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": failed_goal_id,
                "capability_id": capability_id,
                "arguments": {"slot": step_index},
                "step_index": step_index,
            },
            correlation_id=failed_goal_id,
            ts=_ts(start_second + 23 + (step_index * 2)),
        )
        if step_index < len(sequence) - 1:
            store.append_brain_event(
                event_type=BrainEventType.CAPABILITY_COMPLETED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="test",
                payload={
                    "goal_id": failed_goal_id,
                    "capability_id": capability_id,
                    "step_index": step_index,
                    "result": CapabilityExecutionResult.success(
                        capability_id=capability_id,
                        summary=f"Completed {capability_id}.",
                    ).model_dump(),
                },
                correlation_id=failed_goal_id,
                causal_parent_id=request_event.event_id,
                ts=_ts(start_second + 24 + (step_index * 2)),
            )
            continue
        store.append_brain_event(
            event_type=BrainEventType.CAPABILITY_FAILED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="test",
            payload={
                "goal_id": failed_goal_id,
                "capability_id": capability_id,
                "step_index": step_index,
                "result": CapabilityExecutionResult.failed(
                    capability_id=capability_id,
                    summary=f"Failed {capability_id}.",
                    error_code=failure_reason,
                ).model_dump(),
            },
            correlation_id=failed_goal_id,
            causal_parent_id=request_event.event_id,
            ts=_ts(start_second + 24 + (step_index * 2)),
        )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_FAILED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": BrainGoal(
                goal_id=failed_goal_id,
                title=f"{goal_title} failure",
                intent="maintenance.review",
                source="test",
                goal_family=goal_family,
                commitment_id=failed_commitment_id,
                status="failed",
                details={"current_plan_proposal_id": failed_proposal.plan_proposal_id},
                steps=[
                    BrainGoalStep(capability_id=capability_id, status="completed", attempts=1)
                    if index < len(sequence) - 1
                    else BrainGoalStep(
                        capability_id=capability_id,
                        status="failed",
                        attempts=1,
                        summary=f"Failed {capability_id}.",
                        error_code=failure_reason,
                    )
                    for index, capability_id in enumerate(sequence)
                ],
                active_step_index=max(len(sequence) - 1, 0),
                plan_revision=1,
                last_summary=f"Failed {goal_title}.",
                last_error=failure_reason,
            ).as_dict(),
            "commitment": {"commitment_id": failed_commitment_id, "status": "failed"},
        },
        correlation_id=failed_goal_id,
        ts=_ts(start_second + 30),
    )
    return failed_proposal


def _build_expanded_dossier_state(store: BrainStore, session_ids) -> dict[str, object]:
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="We are actively coordinating memory maintenance together.",
        content={"summary": "Coordinating maintenance"},
        salience=1.1,
        source_event_ids=["evt-relationship-expanded"],
        source_event_id="evt-relationship-expanded",
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="project_arc",
        rendered_summary="Memory governance hardening is a standing project.",
        content={"summary": "Governance hardening", "project_key": "Runtime"},
        salience=1.0,
        source_event_ids=["evt-project-runtime"],
        source_event_id="evt-project-runtime",
        append_only=True,
        identity_key="Runtime",
    )

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "designer"},
        rendered_text="user role is designer",
        confidence=0.8,
        singleton=True,
        source_event_id="evt-role-1",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "staff engineer"},
        rendered_text="user role is staff engineer",
        confidence=0.93,
        singleton=True,
        source_event_id="evt-role-2",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.workspace",
        subject="desk",
        value={"value": "standing desk"},
        rendered_text="user prefers a standing desk",
        confidence=0.88,
        singleton=False,
        source_event_id="evt-pref-workspace",
        source_episode_id=None,
        provenance={"source": "test"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )

    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
        scope_type="user",
        scope_id=session_ids.user_id,
        content={"summary": "User was recorded as a designer."},
        source_event_id="evt-user-core-1",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
        scope_type="user",
        scope_id=session_ids.user_id,
        content={"summary": "User is currently a staff engineer with hardware context."},
        source_event_id="evt-user-core-2",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
        content={"summary": "Blink stays careful and provenance-first."},
        source_event_id="evt-self-core-1",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
        content={"summary": "Blink is operating as a provenance-first runtime auditor."},
        source_event_id="evt-self-core-2",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
        content={"summary": "Blink was restoring continuity basics."},
        source_event_id="evt-self-arc-1",
    )
    store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_CURRENT_ARC.value,
        scope_type="agent",
        scope_id=session_ids.agent_id,
        content={"summary": "Blink is expanding bounded continuity dossiers."},
        source_event_id="evt-self-arc-2",
    )

    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    active_goal_id = executive.create_commitment_goal(
        title="Refresh memory proof lane",
        intent="maintenance.review",
        source="test",
        goal_family="memory_maintenance",
    )
    deferred_goal_id = executive.create_commitment_goal(
        title="Queue browser trace cleanup",
        intent="maintenance.review",
        source="test",
        goal_family="memory_maintenance",
    )
    terminal_goal_id = executive.create_commitment_goal(
        title="Archive old operator digest",
        intent="maintenance.review",
        source="test",
        goal_family="memory_maintenance",
    )
    commitment_projection = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    active_commitment = next(
        record
        for record in commitment_projection.active_commitments
        if record.current_goal_id == active_goal_id
    )
    deferred_commitment = next(
        record
        for record in commitment_projection.active_commitments
        if record.current_goal_id == deferred_goal_id
    )
    terminal_commitment = next(
        record
        for record in commitment_projection.active_commitments
        if record.current_goal_id == terminal_goal_id
    )
    executive.defer_commitment(commitment_id=deferred_commitment.commitment_id)
    executive.cancel_commitment(commitment_id=terminal_commitment.commitment_id)
    commitment_projection = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    active_commitment = next(
        record
        for record in commitment_projection.active_commitments
        if record.current_goal_id == active_goal_id
    )
    deferred_commitment = next(
        record
        for record in commitment_projection.deferred_commitments
        if record.current_goal_id == deferred_goal_id
    )
    terminal_commitment = next(
        record
        for record in commitment_projection.recent_terminal_commitments
        if record.current_goal_id == terminal_goal_id
    )

    adopted_proposal = BrainPlanProposal(
        plan_proposal_id="plan-expanded-adopted",
        goal_id=active_goal_id,
        commitment_id=active_commitment.commitment_id,
        source=BrainPlanProposalSource.REPAIR.value,
        summary="Adopt the proof-lane refresh plan.",
        current_plan_revision=active_commitment.plan_revision,
        plan_revision=active_commitment.plan_revision + 1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[BrainGoalStep(capability_id="maintenance.review_memory_health")],
        details={"request_kind": "revise_tail"},
        created_at=_ts(20),
    )
    adopted_event = store.append_planning_proposed(
        proposal=adopted_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        ts=_ts(20),
    )
    store.append_planning_adopted(
        proposal=adopted_proposal,
        decision=BrainPlanProposalDecision(
            summary="Adopted the refreshed proof lane.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        causal_parent_id=adopted_event.event_id,
        ts=_ts(21),
    )
    rejected_proposal = BrainPlanProposal(
        plan_proposal_id="plan-expanded-rejected",
        goal_id=active_goal_id,
        commitment_id=active_commitment.commitment_id,
        source=BrainPlanProposalSource.REPAIR.value,
        summary="Reject the unsupported revision tail.",
        current_plan_revision=active_commitment.plan_revision + 1,
        plan_revision=active_commitment.plan_revision + 2,
        review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
        steps=[BrainGoalStep(capability_id="maintenance.review_memory_health")],
        details={"request_kind": "revise_tail"},
        supersedes_plan_proposal_id=adopted_proposal.plan_proposal_id,
        created_at=_ts(22),
    )
    rejected_event = store.append_planning_proposed(
        proposal=rejected_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        ts=_ts(22),
    )
    store.append_planning_rejected(
        proposal=rejected_proposal,
        decision=BrainPlanProposalDecision(
            summary="Rejected the unsupported revision.",
            reason="unsupported_capability",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        causal_parent_id=rejected_event.event_id,
        ts=_ts(23),
    )
    pending_proposal = BrainPlanProposal(
        plan_proposal_id="plan-expanded-pending",
        goal_id=deferred_goal_id,
        commitment_id=deferred_commitment.commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary="Pending browser trace cleanup plan.",
        current_plan_revision=deferred_commitment.plan_revision,
        plan_revision=deferred_commitment.plan_revision + 1,
        review_policy=BrainPlanReviewPolicy.NEEDS_USER_REVIEW.value,
        steps=[BrainGoalStep(capability_id="reporting.record_maintenance_note")],
        details={"request_kind": "initial_plan"},
        created_at=_ts(24),
    )
    store.append_planning_proposed(
        proposal=pending_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        ts=_ts(24),
    )

    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="skill-goal-1",
        commitment_id=active_commitment.commitment_id,
        goal_title="Refresh proof lane",
        proposal_id="skill-plan-1",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=60,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="skill-goal-2",
        commitment_id=active_commitment.commitment_id,
        goal_title="Refresh proof lane again",
        proposal_id="skill-plan-2",
        sequence=[
            "maintenance.review_memory_health",
            "reporting.record_maintenance_note",
        ],
        start_second=90,
        current_plan_revision=2,
        plan_revision=2,
        supersedes_plan_proposal_id="skill-plan-1",
    )
    store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    procedural_skills = store.build_procedural_skill_projection(
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )

    scene_world_state = BrainSceneWorldProjection(
        scope_type="presence",
        scope_id="browser:presence",
        entities=[
            BrainSceneWorldEntityRecord(
                entity_id="scene-entity-active",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="dock_pad",
                summary="A dock pad is visible on the desk.",
                state=BrainSceneWorldRecordState.ACTIVE.value,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                zone_id="zone:desk",
                confidence=0.84,
                freshness="current",
                affordance_ids=["scene-aff-available"],
                backing_ids=["dock_pad"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(30),
                updated_at=_ts(30),
                expires_at=_ts(120),
            ),
            BrainSceneWorldEntityRecord(
                entity_id="scene-entity-stale",
                entity_kind=BrainSceneWorldEntityKind.OBJECT.value,
                canonical_label="old_box",
                summary="An old box was seen near the wall.",
                state=BrainSceneWorldRecordState.STALE.value,
                evidence_kind=BrainSceneWorldEvidenceKind.OBSERVED.value,
                zone_id="zone:wall",
                confidence=0.51,
                freshness="stale",
                affordance_ids=["scene-aff-blocked"],
                backing_ids=["old_box"],
                source_event_ids=["evt-scene-2"],
                observed_at=_ts(15),
                updated_at=_ts(15),
                expires_at=_ts(40),
            ),
        ],
        affordances=[
            BrainSceneWorldAffordanceRecord(
                affordance_id="scene-aff-available",
                entity_id="scene-entity-active",
                capability_family="vision.inspect",
                summary="The dock pad can be visually inspected.",
                availability=BrainSceneWorldAffordanceAvailability.AVAILABLE.value,
                confidence=0.78,
                freshness="current",
                backing_ids=["dock_pad", "vision.inspect"],
                source_event_ids=["evt-scene-1"],
                observed_at=_ts(30),
                updated_at=_ts(30),
                expires_at=_ts(120),
            ),
            BrainSceneWorldAffordanceRecord(
                affordance_id="scene-aff-blocked",
                entity_id="scene-entity-stale",
                capability_family="robot.grasp",
                summary="The old box cannot currently be grasped safely.",
                availability=BrainSceneWorldAffordanceAvailability.BLOCKED.value,
                confidence=0.4,
                freshness="stale",
                reason_codes=["occluded"],
                backing_ids=["old_box", "robot.grasp"],
                source_event_ids=["evt-scene-2"],
                observed_at=_ts(15),
                updated_at=_ts(15),
                expires_at=_ts(40),
            ),
        ],
        degraded_mode="limited",
        degraded_reason_codes=["camera_frame_stale"],
        updated_at=_ts(30),
    )
    scene_world_state.sync_lists()

    agenda = store.get_agenda_projection(
        scope_key=session_ids.thread_id,
        user_id=session_ids.user_id,
    )
    commitment_projection = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    core_blocks = store._list_continuity_core_blocks(
        user_id=session_ids.user_id,
        agent_id=session_ids.agent_id,
    )
    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        core_blocks=core_blocks,
        commitment_projection=commitment_projection,
        agenda=agenda,
        procedural_skills=procedural_skills,
        scene_world_state=scene_world_state,
        presence_scope_key="browser:presence",
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=graph,
        core_blocks=core_blocks,
        commitment_projection=commitment_projection,
        agenda=agenda,
        procedural_skills=procedural_skills,
        scene_world_state=scene_world_state,
        presence_scope_key="browser:presence",
    )
    return {
        "graph": graph,
        "dossiers": dossiers,
        "core_blocks": core_blocks,
        "commitment_projection": commitment_projection,
        "agenda": agenda,
        "procedural_skills": procedural_skills,
        "scene_world_state": scene_world_state,
        "active_commitment_id": active_commitment.commitment_id,
        "deferred_commitment_id": deferred_commitment.commitment_id,
        "terminal_commitment_id": terminal_commitment.commitment_id,
        "adopted_proposal_id": adopted_proposal.plan_proposal_id,
        "pending_proposal_id": pending_proposal.plan_proposal_id,
    }


def test_memory_v2_core_blocks_are_versioned_and_auditable(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    event_context = {
        "agent_id": session_ids.agent_id,
        "user_id": session_ids.user_id,
        "session_id": session_ids.session_id,
        "thread_id": session_ids.thread_id,
        "source": "test",
    }

    first = store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
        scope_type="user",
        scope_id=session_ids.user_id,
        content={"name": "阿周"},
        source_event_id="evt-block-1",
        event_context=event_context,
    )
    second = store.upsert_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
        scope_type="user",
        scope_id=session_ids.user_id,
        content={"name": "阿周", "role": "设计师"},
        source_event_id="evt-block-2",
        event_context=event_context,
    )

    versions = store.list_core_memory_block_versions(
        block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    latest_event = store.latest_brain_event(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        event_types=(BrainEventType.MEMORY_BLOCK_UPSERTED,),
    )

    assert first.version == 1
    assert second.version == 2
    assert versions[0].version == 2
    assert versions[0].status == "current"
    assert versions[1].status == "superseded"
    assert latest_event is not None
    assert latest_event.payload["block_kind"] == BrainCoreMemoryBlockKind.USER_CORE.value
    assert latest_event.payload["version"] == 2


def test_memory_v2_self_core_is_seeded_and_versioned_without_raw_block_prose(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")

    store.ensure_default_blocks(load_default_agent_blocks())
    first = store.get_current_core_memory_block(
        block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
        scope_type="agent",
        scope_id="blink/main",
    )
    assert first is not None
    assert first.content["project_identity"]["display_name"] == "Blink"
    assert {
        "identity",
        "policy",
        "style",
        "action_library",
        "persona",
        "voice",
        "relationship_style",
        "teaching_style",
    }.issubset(first.content["pinned_blocks"])
    assert "content" not in first.content["pinned_blocks"]["identity"]

    store._conn.execute(
        "UPDATE agent_blocks SET content = content || ? WHERE name = 'style'",
        ("\n# test fingerprint change",),
    )
    store._conn.commit()
    store._refresh_self_core_block(
        agent_id="blink/main",
        source_event_id="evt-self-core-refresh",
    )

    versions = store.list_core_memory_block_versions(
        block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
        scope_type="agent",
        scope_id="blink/main",
    )

    assert [record.version for record in versions] == [2, 1]
    assert (
        versions[0].content["pinned_blocks"]["style"]["fingerprint"]
        != versions[1].content["pinned_blocks"]["style"]["fingerprint"]
    )


def test_memory_v2_claim_correction_keeps_current_and_historical_truth(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="咖啡",
        value={"value": "咖啡"},
        rendered_text="用户喜欢 咖啡",
        confidence=0.82,
        singleton=False,
        source_event_id="evt-pref-like",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.dislike",
        subject="咖啡",
        value={"value": "咖啡"},
        rendered_text="用户不喜欢 咖啡",
        confidence=0.9,
        singleton=False,
        source_event_id="evt-pref-dislike",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )

    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    current_claims = store.query_claims(
        temporal_mode="current",
        subject_entity_id=user_entity.entity_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        limit=12,
    )
    historical_claims = store.query_claims(
        temporal_mode="historical",
        subject_entity_id=user_entity.entity_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        limit=12,
    )
    retriever = ContinuityRetriever(store=store)
    retrieval = retriever.retrieve(
        BrainContinuityQuery(
            text="咖啡",
            scope_type="user",
            scope_id=session_ids.user_id,
            temporal_mode="current",
            limit=4,
        )
    )

    assert any(
        claim.predicate == "preference.dislike" and claim.object.get("value") == "咖啡"
        for claim in current_claims
    )
    assert not any(claim.predicate == "preference.like" for claim in current_claims)
    assert any(
        claim.predicate == "preference.like" and claim.object.get("value") == "咖啡"
        for claim in historical_claims
    )
    assert store.claim_supersessions()
    assert retrieval
    assert retrieval[0].summary == "用户不喜欢 咖啡"


def test_memory_v2_validity_windows_and_evidence_support_current_vs_historical_queries(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name="user-1",
        aliases=["user-1"],
        attributes={"user_id": "user-1"},
    )
    ledger = ClaimLedger(store=store)

    claim = ledger.record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="profile.role",
        object_value="产品经理",
        status="active",
        confidence=0.8,
        valid_from="2025-01-01T00:00:00+00:00",
        valid_to="2025-06-01T00:00:00+00:00",
        source_event_id="evt-role-1",
        scope_type="user",
        scope_id="user-1",
        currentness_status="historical",
        stale_after_seconds=3600,
        evidence_summary="legacy imported role",
        evidence_json={"source": "migration"},
    )

    current_claims = store.query_claims(
        temporal_mode="current",
        subject_entity_id=user_entity.entity_id,
        scope_type="user",
        scope_id="user-1",
        limit=8,
    )
    historical_claims = store.query_claims(
        temporal_mode="historical",
        subject_entity_id=user_entity.entity_id,
        scope_type="user",
        scope_id="user-1",
        limit=8,
    )
    evidence = ledger.claim_evidence(claim.claim_id)

    assert current_claims == []
    assert [record.claim_id for record in historical_claims] == [claim.claim_id]
    assert evidence
    assert evidence[0].evidence_summary == "legacy imported role"


def test_memory_v2_claim_governance_primitives_and_filters(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="claim-governance")
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=store)
    event_context = store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="claim-governance",
    )

    claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-claim-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:governance",
        evidence_summary="user said they are a designer",
        event_context=event_context,
    )
    assert claim.effective_currentness_status == BrainClaimCurrentnessStatus.CURRENT.value
    assert claim.effective_review_state == BrainClaimReviewState.NONE.value
    assert claim.effective_retention_class == BrainClaimRetentionClass.DURABLE.value

    projection = store.get_claim_governance_projection(
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    assert projection.current_claim_ids == [claim.claim_id]
    assert projection.currentness_counts[BrainClaimCurrentnessStatus.CURRENT.value] == 1

    held = ledger.request_claim_review(
        claim.claim_id,
        source_event_id="evt-review-1",
        reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
        summary="Needs confirmation from the operator.",
        event_context=event_context,
    )
    assert held.is_held
    assert held.effective_review_state == BrainClaimReviewState.REQUESTED.value
    assert BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value in held.governance_reason_codes
    assert [
        record.claim_id
        for record in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            currentness_states=[BrainClaimCurrentnessStatus.HELD.value],
            limit=8,
        )
    ] == [claim.claim_id]
    assert (
        store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            currentness_states=[BrainClaimCurrentnessStatus.CURRENT.value],
            limit=8,
        )
        == []
    )

    revalidated = ledger.revalidate_claim(
        claim.claim_id,
        source_event_id="evt-revalidate-1",
        confidence=0.91,
        event_context=event_context,
    )
    assert revalidated.is_current
    assert revalidated.confidence == pytest.approx(0.91)
    assert revalidated.effective_review_state == BrainClaimReviewState.RESOLVED.value

    expired = ledger.expire_claim(
        claim.claim_id,
        source_event_id="evt-expire-1",
        reason_codes=[BrainGovernanceReasonCode.STALE_WITHOUT_REFRESH.value],
        event_context=event_context,
    )
    assert expired.is_stale
    assert not expired.is_current
    assert [
        record.claim_id
        for record in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            currentness_states=[BrainClaimCurrentnessStatus.STALE.value],
            limit=8,
        )
    ] == [claim.claim_id]
    assert claim.claim_id in {
        record.claim_id
        for record in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=8,
        )
    }

    revalidated_again = ledger.revalidate_claim(
        claim.claim_id,
        source_event_id="evt-revalidate-2",
        event_context=event_context,
    )
    assert revalidated_again.is_current
    assert (
        BrainGovernanceReasonCode.STALE_WITHOUT_REFRESH.value
        not in revalidated_again.governance_reason_codes
    )

    retained = ledger.reclassify_claim_retention(
        claim.claim_id,
        retention_class=BrainClaimRetentionClass.SESSION.value,
        source_event_id="evt-retention-1",
        event_context=event_context,
    )
    assert retained.effective_retention_class == BrainClaimRetentionClass.SESSION.value

    revoked = ledger.revoke_claim(
        claim.claim_id,
        reason="Contradicted by a later correction.",
        source_event_id="evt-revoke-1",
        event_context=event_context,
    )
    assert revoked == 1
    revoked_claim = ledger.get_claim(claim.claim_id)
    assert revoked_claim is not None
    assert revoked_claim.is_historical
    assert BrainGovernanceReasonCode.CONTRADICTION.value in revoked_claim.governance_reason_codes
    assert [
        record.claim_id
        for record in store.query_claims(
            temporal_mode="historical",
            scope_type="user",
            scope_id=session_ids.user_id,
            currentness_states=[BrainClaimCurrentnessStatus.HISTORICAL.value],
            limit=8,
        )
    ] == [claim.claim_id]

    projection = BrainClaimGovernanceProjection.from_dict(
        store.get_claim_governance_projection(
            scope_type="user",
            scope_id=session_ids.user_id,
        ).as_dict()
    )
    assert projection is not None
    assert projection.historical_claim_ids == [claim.claim_id]
    assert projection.retention_class_counts[BrainClaimRetentionClass.SESSION.value] == 1


def test_memory_v2_claim_governance_replay_preserves_traceability(tmp_path):
    source = BrainStore(path=tmp_path / "source.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser", client_id="claim-governance-replay"
    )
    user = source.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=source)
    event_context = source._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="claim-governance-replay",
    )

    held_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-held-claim",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role:held",
        evidence_summary="role mentioned in chat",
        event_context=event_context,
    )
    ledger.request_claim_review(
        held_claim.claim_id,
        source_event_id="evt-held-review",
        reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
        event_context=event_context,
    )

    stale_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="session.summary",
        object_data={"value": "Summary needs refresh"},
        source_event_id="evt-stale-claim",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="session.summary:stale",
        evidence_summary="summary captured from the thread",
        event_context=event_context,
    )
    ledger.expire_claim(
        stale_claim.claim_id,
        source_event_id="evt-stale-expire",
        reason_codes=[BrainGovernanceReasonCode.EXPIRED_BY_POLICY.value],
        event_context=event_context,
    )

    prior_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.origin",
        object_data={"value": "Shanghai"},
        source_event_id="evt-origin-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.origin",
        evidence_summary="origin noted in the thread",
        event_context=event_context,
    )
    replacement_claim = ledger.supersede_claim(
        prior_claim.claim_id,
        replacement_claim={
            "subject_entity_id": user.entity_id,
            "predicate": "profile.origin",
            "object_data": {"value": "Beijing"},
            "scope_type": "user",
            "scope_id": session_ids.user_id,
            "claim_key": "profile.origin",
            "evidence_summary": "origin corrected later",
        },
        reason="Corrected the recorded origin.",
        source_event_id="evt-origin-2",
        event_context=event_context,
    )

    source_projection = source.get_claim_governance_projection(
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    events = _all_brain_events(source)

    replay = BrainStore(path=tmp_path / "replay.db")
    replay.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    for event in events:
        replay.import_brain_event(event)
    for event in events:
        replay.apply_memory_event(event)

    replay_projection = replay.get_claim_governance_projection(
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    assert replay_projection.as_dict() == source_projection.as_dict()

    replay_held = replay._claims().get_claim(held_claim.claim_id)
    assert replay_held is not None
    assert replay_held.is_held
    assert replay_held.source_event_id is not None
    assert replay._claims().claim_evidence(held_claim.claim_id)

    replay_stale = replay._claims().get_claim(stale_claim.claim_id)
    assert replay_stale is not None
    assert replay_stale.is_stale
    assert not replay_stale.is_current
    assert [
        record.claim_id
        for record in replay.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            currentness_states=[BrainClaimCurrentnessStatus.STALE.value],
            limit=8,
        )
    ] == [stale_claim.claim_id]

    replay_historical_ids = {
        record.claim_id
        for record in replay.query_claims(
            temporal_mode="historical",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=8,
        )
    }
    assert prior_claim.claim_id in replay_historical_ids
    assert replacement_claim.claim_id not in replay_historical_ids
    assert replay.claim_supersessions(claim_id=prior_claim.claim_id)
    assert replay._claims().claim_evidence(prior_claim.claim_id)


def test_memory_v2_continuity_graph_roundtrip_and_temporal_semantics(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="graph-memory")
    ledger = ClaimLedger(store=store)

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user likes coffee",
        confidence=0.82,
        singleton=False,
        source_event_id="evt-pref-like",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.dislike",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user dislikes coffee",
        confidence=0.93,
        singleton=False,
        source_event_id="evt-pref-dislike",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )

    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    stale_claim = ledger.record_claim(
        subject_entity_id=user_entity.entity_id,
        predicate="session.summary",
        object_value="Needs refresh",
        status="active",
        confidence=0.55,
        valid_from="2025-01-01T00:00:00+00:00",
        source_event_id="evt-stale-claim",
        scope_type="user",
        scope_id=session_ids.user_id,
        evidence_summary="legacy stale note",
        evidence_json={"source": "migration"},
    )
    ledger.expire_claim(
        stale_claim.claim_id,
        source_event_id="evt-stale-governance",
        reason_codes=["stale_without_refresh"],
        event_context=None,
    )
    current_dislike_claim = next(
        claim
        for claim in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=16,
        )
        if claim.predicate == "preference.dislike"
    )
    historical_like_claim = next(
        claim
        for claim in store.query_claims(
            temporal_mode="historical",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=16,
        )
        if claim.predicate == "preference.like"
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=f"{session_ids.agent_id}:{session_ids.user_id}",
        entry_kind="relationship_arc",
        rendered_summary="We corrected the coffee preference.",
        content={"summary": "Coffee preference correction"},
        salience=0.8,
        source_claim_ids=[current_dislike_claim.claim_id],
        source_event_ids=["evt-pref-dislike"],
        source_event_id="evt-autobio-1",
    )

    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    roundtrip = BrainContinuityGraphProjection.from_dict(graph.as_dict())

    assert roundtrip is not None
    assert roundtrip.as_dict() == graph.as_dict()
    current_backing_ids = _graph_backing_ids(graph, graph.current_node_ids)
    historical_backing_ids = _graph_backing_ids(graph, graph.historical_node_ids)
    stale_backing_ids = _graph_backing_ids(graph, graph.stale_node_ids)
    superseded_backing_ids = _graph_backing_ids(graph, graph.superseded_node_ids)
    edge_kinds = {edge.kind for edge in graph.edges}

    assert current_dislike_claim.claim_id in current_backing_ids
    assert historical_like_claim.claim_id in historical_backing_ids
    assert historical_like_claim.claim_id in superseded_backing_ids
    assert stale_claim.claim_id in current_backing_ids
    assert stale_claim.claim_id in stale_backing_ids
    assert BrainContinuityGraphEdgeKind.SUPERSEDES.value in edge_kinds
    assert BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_CLAIM.value in edge_kinds
    assert BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_SUPPORTS_EVENT.value in edge_kinds
    assert BrainContinuityGraphEdgeKind.SUPPORTED_BY_EVENT.value in edge_kinds


def test_memory_v2_continuity_graph_node_and_edge_roundtrip(tmp_path):
    _ = tmp_path
    node = BrainContinuityGraphNodeRecord(
        node_id="node-1",
        kind=BrainContinuityGraphNodeKind.CLAIM.value,
        backing_record_id="claim-1",
        summary="user role is designer",
        status="active",
        scope_type="user",
        scope_id="user-1",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_to=None,
        source_event_ids=["evt-1"],
        source_episode_ids=[1],
        supporting_claim_ids=["claim-1"],
        details={"predicate": "profile.role"},
    )
    edge = BrainContinuityGraphEdgeRecord(
        edge_id="edge-1",
        kind=BrainContinuityGraphEdgeKind.CLAIM_SUBJECT.value,
        from_node_id="node-1",
        to_node_id="node-2",
        status="linked",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_to=None,
        source_event_ids=["evt-1"],
        source_episode_ids=[1],
        supporting_claim_ids=["claim-1"],
        details={"predicate": "profile.role"},
    )
    projection = BrainContinuityGraphProjection(
        scope_type="user",
        scope_id="user-1",
        node_counts={BrainContinuityGraphNodeKind.CLAIM.value: 1},
        edge_counts={BrainContinuityGraphEdgeKind.CLAIM_SUBJECT.value: 1},
        nodes=[node],
        edges=[edge],
        current_node_ids=["node-1"],
        historical_node_ids=[],
        stale_node_ids=[],
        superseded_node_ids=[],
    )

    assert BrainContinuityGraphNodeRecord.from_dict(node.as_dict()) == node
    assert BrainContinuityGraphEdgeRecord.from_dict(edge.as_dict()) == edge
    assert BrainContinuityGraphProjection.from_dict(projection.as_dict()) == projection


def test_memory_v2_procedural_trace_record_and_projection_roundtrip(tmp_path):
    _ = tmp_path
    step = BrainProceduralStepTraceRecord(
        step_trace_id="step-trace-1",
        trace_id="trace-1",
        step_index=0,
        capability_id="observation.inspect_presence_state",
        arguments={"presence_scope_key": "browser:presence"},
        status="completed",
        attempts=1,
        summary="Inspected the current presence state.",
        request_event_id="evt-request-1",
        result_event_id="evt-result-1",
        source_event_ids=["evt-request-1", "evt-result-1"],
        details={"result": {"outcome": "completed"}},
    )
    trace = BrainProceduralExecutionTraceRecord(
        trace_id="trace-1",
        goal_id="goal-1",
        commitment_id="commitment-1",
        plan_proposal_id="proposal-1",
        goal_title="Inspect presence and report it",
        goal_intent="autonomy.presence_user_reentered",
        goal_family="conversation",
        proposal_source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        current_plan_revision=1,
        plan_revision=2,
        preserved_prefix_count=1,
        supersedes_plan_proposal_id="proposal-0",
        status=BrainProceduralTraceStatus.COMPLETED.value,
        planned_steps=[
            BrainGoalStep(capability_id="observation.inspect_presence_state"),
            BrainGoalStep(capability_id="reporting.record_presence_event"),
        ],
        step_executions=[step],
        outcome_ids=["outcome-1"],
        source_event_ids=["evt-adopt-1", "evt-request-1", "evt-result-1"],
        started_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:03+00:00",
        ended_at="2026-01-01T00:00:03+00:00",
        details={"planning_adopted_event_id": "evt-adopt-1"},
    )
    outcome = BrainProceduralOutcomeRecord(
        outcome_id="outcome-1",
        outcome_kind=BrainProceduralOutcomeKind.COMPLETED.value,
        trace_id="trace-1",
        goal_id="goal-1",
        commitment_id="commitment-1",
        plan_proposal_id="proposal-1",
        plan_revision=2,
        summary="Presence inspection completed.",
        source_event_id="evt-goal-completed-1",
        causal_parent_event_id="evt-request-1",
        source_trace_ids=["trace-1"],
        source_step_trace_ids=["step-trace-1"],
        source_event_ids=["evt-goal-completed-1", "evt-request-1"],
        created_at="2026-01-01T00:00:03+00:00",
        details={"resume_count": 0},
    )
    projection = BrainProceduralTraceProjection(
        scope_type="thread",
        scope_id="thread-1",
        trace_counts={BrainProceduralTraceStatus.COMPLETED.value: 1},
        outcome_counts={BrainProceduralOutcomeKind.COMPLETED.value: 1},
        traces=[trace],
        outcomes=[outcome],
        open_trace_ids=[],
        paused_trace_ids=[],
        completed_trace_ids=["trace-1"],
        failed_trace_ids=[],
        cancelled_trace_ids=[],
        superseded_trace_ids=[],
    )

    assert BrainProceduralStepTraceRecord.from_dict(step.as_dict()) == step
    assert BrainProceduralExecutionTraceRecord.from_dict(trace.as_dict()) == trace
    assert BrainProceduralOutcomeRecord.from_dict(outcome.as_dict()) == outcome
    assert BrainProceduralTraceProjection.from_dict(projection.as_dict()) == projection


def test_memory_v2_procedural_trace_harvesting_preserves_block_and_revision_provenance(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="text", client_id="procedural-memory")
    goal_id = "goal-procedural-1"
    commitment_id = "commitment-procedural-1"

    goal_created = BrainGoal(
        goal_id=goal_id,
        title="Review memory health and report it",
        intent="autonomy.maintenance_review_findings",
        source="test",
        goal_family="memory_maintenance",
        commitment_id=commitment_id,
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_CREATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={"goal": goal_created.as_dict()},
        correlation_id=goal_id,
    )

    initial_proposal = BrainPlanProposal(
        plan_proposal_id="proposal-initial",
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary="Inspect memory health and then report the finding.",
        current_plan_revision=1,
        plan_revision=1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[
            BrainGoalStep(capability_id="maintenance.review_memory_health"),
            BrainGoalStep(capability_id="reporting.record_maintenance_note"),
        ],
        details={"request_kind": "initial_plan"},
        created_at="2026-01-01T00:00:00+00:00",
    )
    proposed_initial = store.append_planning_proposed(
        proposal=initial_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        ts="2026-01-01T00:00:00+00:00",
    )
    adopted_initial = store.append_planning_adopted(
        proposal=initial_proposal,
        decision=BrainPlanProposalDecision(
            summary="Adopt the initial bounded plan.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=goal_id,
        causal_parent_id=proposed_initial.event_id,
        ts="2026-01-01T00:00:01+00:00",
    )
    goal_open = BrainGoal(
        goal_id=goal_id,
        title=goal_created.title,
        intent=goal_created.intent,
        source="test",
        goal_family=goal_created.goal_family,
        commitment_id=commitment_id,
        status="open",
        details={"current_plan_proposal_id": initial_proposal.plan_proposal_id},
        steps=[BrainGoalStep.from_dict(step.as_dict()) for step in initial_proposal.steps],
        plan_revision=1,
        last_summary="Adopt the initial bounded plan.",
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": goal_open.as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        causal_parent_id=adopted_initial.event_id,
        ts="2026-01-01T00:00:02+00:00",
    )

    first_request = store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_REQUESTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "maintenance.review_memory_health",
            "arguments": {"maintenance": {"focus": "backpressure"}},
            "step_index": 0,
        },
        correlation_id=goal_id,
        ts="2026-01-01T00:00:03+00:00",
    )
    store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "maintenance.review_memory_health",
            "step_index": 0,
            "result": CapabilityExecutionResult.success(
                capability_id="maintenance.review_memory_health",
                summary="Reviewed memory health.",
                output={"finding": "scheduler_backpressure"},
            ).model_dump(),
        },
        correlation_id=goal_id,
        causal_parent_id=first_request.event_id,
        ts="2026-01-01T00:00:04+00:00",
    )
    goal_after_first_step = BrainGoal(
        goal_id=goal_id,
        title=goal_created.title,
        intent=goal_created.intent,
        source="test",
        goal_family=goal_created.goal_family,
        commitment_id=commitment_id,
        status="in_progress",
        details={"current_plan_proposal_id": initial_proposal.plan_proposal_id},
        steps=[
            BrainGoalStep(
                capability_id="maintenance.review_memory_health",
                status="completed",
                attempts=1,
                summary="Reviewed memory health.",
                output={"finding": "scheduler_backpressure"},
            ),
            BrainGoalStep(capability_id="reporting.record_maintenance_note"),
        ],
        active_step_index=0,
        plan_revision=1,
        last_summary="Reviewed memory health.",
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": goal_after_first_step.as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=goal_id,
        ts="2026-01-01T00:00:05+00:00",
    )

    second_request = store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_REQUESTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "reporting.record_maintenance_note",
            "arguments": {"maintenance": {"focus": "backpressure"}},
            "step_index": 1,
        },
        correlation_id=goal_id,
        ts="2026-01-01T00:00:06+00:00",
    )
    store.append_brain_event(
        event_type=BrainEventType.CRITIC_FEEDBACK,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "reporting.record_maintenance_note",
            "step_index": 1,
            "result": CapabilityExecutionResult.blocked(
                capability_id="reporting.record_maintenance_note",
                summary="Reporting is blocked on operator review.",
                error_code="operator_review",
                retryable=True,
            ).model_dump(),
            "recovery": {
                "decision": "blocked",
                "summary": "Pause until the blocker clears.",
            },
        },
        correlation_id=goal_id,
        causal_parent_id=second_request.event_id,
        ts="2026-01-01T00:00:07+00:00",
    )
    blocked_goal = BrainGoal(
        goal_id=goal_id,
        title=goal_created.title,
        intent=goal_created.intent,
        source="test",
        goal_family=goal_created.goal_family,
        commitment_id=commitment_id,
        status="blocked",
        details={"current_plan_proposal_id": initial_proposal.plan_proposal_id},
        steps=[
            BrainGoalStep(
                capability_id="maintenance.review_memory_health",
                status="completed",
                attempts=1,
                summary="Reviewed memory health.",
                output={"finding": "scheduler_backpressure"},
            ),
            BrainGoalStep(
                capability_id="reporting.record_maintenance_note",
                status="blocked",
                attempts=1,
                summary="Reporting is blocked on operator review.",
                error_code="operator_review",
            ),
        ],
        active_step_index=1,
        plan_revision=1,
        last_summary="Reporting is blocked on operator review.",
        last_error="operator_review",
        blocked_reason=BrainBlockedReason(
            kind="capability_blocked",
            summary="Reporting is blocked on operator review.",
            details={"capability_id": "reporting.record_maintenance_note"},
        ),
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": blocked_goal.as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "blocked"},
        },
        correlation_id=goal_id,
        causal_parent_id=second_request.event_id,
        ts="2026-01-01T00:00:08+00:00",
    )

    revised_proposal = BrainPlanProposal(
        plan_proposal_id="proposal-revision",
        goal_id=goal_id,
        commitment_id=commitment_id,
        source=BrainPlanProposalSource.BOUNDED_PLANNER.value,
        summary="Keep the completed review and replace the remaining reporting tail.",
        current_plan_revision=1,
        plan_revision=2,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        steps=[
            BrainGoalStep(
                capability_id="maintenance.review_memory_health",
                status="completed",
                attempts=1,
                summary="Reviewed memory health.",
                output={"finding": "scheduler_backpressure"},
            ),
            BrainGoalStep(capability_id="reporting.record_presence_event"),
        ],
        preserved_prefix_count=1,
        supersedes_plan_proposal_id=initial_proposal.plan_proposal_id,
        details={"request_kind": "revise_tail"},
        created_at="2026-01-01T00:00:09+00:00",
    )
    proposed_revision = store.append_planning_proposed(
        proposal=revised_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=commitment_id,
        ts="2026-01-01T00:00:09+00:00",
    )
    adopted_revision = store.append_planning_adopted(
        proposal=revised_proposal,
        decision=BrainPlanProposalDecision(
            summary="Adopt the repaired remaining tail.",
            reason="bounded_plan_available",
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        correlation_id=commitment_id,
        causal_parent_id=proposed_revision.event_id,
        ts="2026-01-01T00:00:10+00:00",
    )
    repaired_goal = BrainGoal(
        goal_id=goal_id,
        title=goal_created.title,
        intent=goal_created.intent,
        source="test",
        goal_family=goal_created.goal_family,
        commitment_id=commitment_id,
        status="open",
        details={"current_plan_proposal_id": revised_proposal.plan_proposal_id},
        steps=[BrainGoalStep.from_dict(step.as_dict()) for step in revised_proposal.steps],
        active_step_index=1,
        plan_revision=2,
        last_summary="Adopt the repaired remaining tail.",
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_REPAIRED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": repaired_goal.as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "active"},
        },
        correlation_id=commitment_id,
        causal_parent_id=adopted_revision.event_id,
        ts="2026-01-01T00:00:11+00:00",
    )

    third_request = store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_REQUESTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "reporting.record_presence_event",
            "arguments": {"maintenance": {"focus": "backpressure"}},
            "step_index": 1,
        },
        correlation_id=goal_id,
        ts="2026-01-01T00:00:12+00:00",
    )
    store.append_brain_event(
        event_type=BrainEventType.CAPABILITY_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal_id": goal_id,
            "capability_id": "reporting.record_presence_event",
            "step_index": 1,
            "result": CapabilityExecutionResult.success(
                capability_id="reporting.record_presence_event",
                summary="Recorded the maintenance note through reporting.",
            ).model_dump(),
        },
        correlation_id=goal_id,
        causal_parent_id=third_request.event_id,
        ts="2026-01-01T00:00:13+00:00",
    )
    completed_goal = BrainGoal(
        goal_id=goal_id,
        title=goal_created.title,
        intent=goal_created.intent,
        source="test",
        goal_family=goal_created.goal_family,
        commitment_id=commitment_id,
        status="completed",
        details={"current_plan_proposal_id": revised_proposal.plan_proposal_id},
        steps=[
            BrainGoalStep(
                capability_id="maintenance.review_memory_health",
                status="completed",
                attempts=1,
                summary="Reviewed memory health.",
                output={"finding": "scheduler_backpressure"},
            ),
            BrainGoalStep(
                capability_id="reporting.record_presence_event",
                status="completed",
                attempts=1,
                summary="Recorded the maintenance note through reporting.",
            ),
        ],
        active_step_index=1,
        plan_revision=2,
        last_summary="Recorded the maintenance note through reporting.",
    )
    store.append_brain_event(
        event_type=BrainEventType.GOAL_COMPLETED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        payload={
            "goal": completed_goal.as_dict(),
            "commitment": {"commitment_id": commitment_id, "status": "completed"},
        },
        correlation_id=goal_id,
        causal_parent_id=third_request.event_id,
        ts="2026-01-01T00:00:14+00:00",
    )

    projection = store.build_procedural_trace_projection(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    roundtrip = BrainProceduralTraceProjection.from_dict(projection.as_dict())
    assert roundtrip is not None
    assert roundtrip.as_dict() == projection.as_dict()

    traces_by_proposal = {record.plan_proposal_id: record for record in projection.traces}
    initial_trace = traces_by_proposal["proposal-initial"]
    revised_trace = traces_by_proposal["proposal-revision"]
    outcomes_by_kind = {}
    for record in projection.outcomes:
        outcomes_by_kind.setdefault(record.outcome_kind, []).append(record)

    assert projection.trace_counts[BrainProceduralTraceStatus.SUPERSEDED.value] == 1
    assert projection.trace_counts[BrainProceduralTraceStatus.COMPLETED.value] == 1
    assert initial_trace.status == BrainProceduralTraceStatus.SUPERSEDED.value
    assert initial_trace.supersedes_plan_proposal_id is None
    assert revised_trace.status == BrainProceduralTraceStatus.COMPLETED.value
    assert revised_trace.preserved_prefix_count == 1
    assert revised_trace.supersedes_plan_proposal_id == "proposal-initial"
    assert [step.capability_id for step in revised_trace.planned_steps] == [
        "maintenance.review_memory_health",
        "reporting.record_presence_event",
    ]
    assert [step.capability_id for step in initial_trace.step_executions] == [
        "maintenance.review_memory_health",
        "reporting.record_maintenance_note",
    ]
    assert initial_trace.step_executions[1].status == "blocked"
    assert revised_trace.step_executions[0].capability_id == "reporting.record_presence_event"
    assert revised_trace.step_executions[0].status == "completed"
    assert (
        outcomes_by_kind[BrainProceduralOutcomeKind.BLOCKED.value][0].trace_id
        == initial_trace.trace_id
    )
    assert (
        outcomes_by_kind[BrainProceduralOutcomeKind.SUPERSEDED_BY_REVISION.value][
            0
        ].plan_proposal_id
        == "proposal-initial"
    )
    assert (
        outcomes_by_kind[BrainProceduralOutcomeKind.COMPLETED.value][0].trace_id
        == revised_trace.trace_id
    )
    assert all(record.commitment_id == commitment_id for record in projection.outcomes)


def test_memory_v2_procedural_skill_record_and_projection_roundtrip(tmp_path):
    _ = tmp_path
    condition = BrainProceduralActivationConditionRecord(
        condition_id="cond-1",
        kind="goal_family",
        summary="Goal family must match memory_maintenance.",
        match_value="memory_maintenance",
    )
    invariant = BrainProceduralInvariantRecord(
        invariant_id="inv-1",
        kind="required_capability_ids",
        summary="Requires review then report.",
        details={
            "required_capability_ids": [
                "maintenance.review_memory_health",
                "reporting.record_presence_event",
            ]
        },
    )
    effect = BrainProceduralEffectRecord(
        effect_id="effect-1",
        kind="terminal_outcome",
        summary="Terminates with a completed outcome.",
        details={"outcome_kind": "completed"},
    )
    failure_signature = BrainProceduralFailureSignatureRecord(
        failure_signature_id="failure-1",
        kind="step_error_code",
        reason_code="operator_review",
        summary="The reporting step can block on operator review.",
        support_trace_ids=["trace-1"],
        support_outcome_ids=["outcome-1"],
        details={"support_count": 1},
    )
    stats = BrainProceduralSkillStatsRecord(
        support_trace_count=2,
        success_trace_count=2,
        failure_trace_count=1,
        blocked_or_deferred_count=0,
        independent_plan_count=2,
        last_supported_at=_ts(10),
        last_failure_at=_ts(12),
    )
    skill = BrainProceduralSkillRecord(
        skill_id="skill-1",
        skill_family_key="family-1",
        template_fingerprint="template-1",
        scope_type="thread",
        scope_id="thread-1",
        title="Review memory health and report it",
        purpose="Review memory health and report it",
        goal_family="memory_maintenance",
        status=BrainProceduralSkillStatus.ACTIVE.value,
        confidence=0.64,
        activation_conditions=[condition],
        invariants=[invariant],
        step_template=[
            BrainGoalStep(capability_id="maintenance.review_memory_health"),
            BrainGoalStep(capability_id="reporting.record_presence_event"),
        ],
        required_capability_ids=[
            "maintenance.review_memory_health",
            "reporting.record_presence_event",
        ],
        effects=[effect],
        termination_conditions=["goal_completed"],
        failure_signatures=[failure_signature],
        stats=stats,
        supporting_trace_ids=["trace-1", "trace-2"],
        supporting_outcome_ids=["outcome-1", "outcome-2"],
        supporting_plan_proposal_ids=["proposal-1", "proposal-2"],
        supporting_commitment_ids=["commitment-1"],
        created_at=_ts(1),
        updated_at=_ts(12),
        details={"confidence_band": "medium"},
    )
    projection = BrainProceduralSkillProjection(
        scope_type="thread",
        scope_id="thread-1",
        skill_counts={BrainProceduralSkillStatus.ACTIVE.value: 1},
        confidence_band_counts={"medium": 1},
        skills=[skill],
        candidate_skill_ids=[],
        active_skill_ids=["skill-1"],
        superseded_skill_ids=[],
        retired_skill_ids=[],
    )

    assert BrainProceduralActivationConditionRecord.from_dict(condition.as_dict()) == condition
    assert BrainProceduralInvariantRecord.from_dict(invariant.as_dict()) == invariant
    assert BrainProceduralEffectRecord.from_dict(effect.as_dict()) == effect
    assert (
        BrainProceduralFailureSignatureRecord.from_dict(failure_signature.as_dict())
        == failure_signature
    )
    assert BrainProceduralSkillStatsRecord.from_dict(stats.as_dict()) == stats
    assert BrainProceduralSkillRecord.from_dict(skill.as_dict()) == skill
    assert BrainProceduralSkillProjection.from_dict(projection.as_dict()) == projection


def test_memory_v2_procedural_skill_consolidation_promotes_candidate_and_active_skills(tmp_path):
    store = BrainStore(path=tmp_path / "candidate-active.db")
    session_ids = resolve_brain_session_ids(runtime_kind="text", client_id="skill-candidate")

    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-skill-1",
        commitment_id="commitment-skill-1",
        goal_title="Review and report memory health",
        proposal_id="proposal-skill-1",
        sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
        start_second=0,
    )
    candidate_projection = store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    assert len(candidate_projection.skills) == 1
    candidate = candidate_projection.skills[0]
    assert candidate.status == BrainProceduralSkillStatus.CANDIDATE.value
    assert candidate.confidence < 0.5
    assert candidate.stats.success_trace_count == 1
    assert candidate.supporting_trace_ids
    assert candidate.supporting_outcome_ids

    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-skill-2",
        commitment_id="commitment-skill-2",
        goal_title="Review and report memory health again",
        proposal_id="proposal-skill-2",
        sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
        start_second=40,
    )
    _append_completed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-skill-3",
        commitment_id="commitment-skill-3",
        goal_title="Review and report memory health a third time",
        proposal_id="proposal-skill-3",
        sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
        start_second=80,
    )
    active_projection = store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )

    assert len(active_projection.skills) == 1
    active = active_projection.skills[0]
    assert active.status == BrainProceduralSkillStatus.ACTIVE.value
    assert active.confidence > candidate.confidence
    assert active.stats.success_trace_count == 3
    assert active.stats.independent_plan_count == 3
    assert active_projection.active_skill_ids == [active.skill_id]
    assert active_projection.candidate_skill_ids == []
    persisted_projection = store.build_procedural_skill_projection(
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    assert persisted_projection.as_dict() == active_projection.as_dict()


def test_memory_v2_procedural_skill_consolidation_supersedes_prefix_skill(tmp_path):
    store = BrainStore(path=tmp_path / "superseded.db")
    session_ids = resolve_brain_session_ids(runtime_kind="text", client_id="skill-supersede")

    for offset, goal_id, proposal_id in (
        (0, "goal-prefix-1", "proposal-prefix-1"),
        (40, "goal-prefix-2", "proposal-prefix-2"),
    ):
        _append_completed_procedural_goal(
            store,
            session_ids,
            goal_id=goal_id,
            commitment_id=f"commitment-{goal_id}",
            goal_title="Run the short maintenance path",
            proposal_id=proposal_id,
            sequence=["maintenance.review_memory_health"],
            start_second=offset,
        )
    for offset, goal_id, proposal_id in (
        (80, "goal-extended-1", "proposal-extended-1"),
        (120, "goal-extended-2", "proposal-extended-2"),
        (160, "goal-extended-3", "proposal-extended-3"),
    ):
        _append_completed_procedural_goal(
            store,
            session_ids,
            goal_id=goal_id,
            commitment_id=f"commitment-{goal_id}",
            goal_title="Run the stronger maintenance path",
            proposal_id=proposal_id,
            sequence=[
                "maintenance.review_memory_health",
                "maintenance.review_memory_health",
            ],
            start_second=offset,
        )

    projection = store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    by_status = {record.status: [] for record in projection.skills}
    for record in projection.skills:
        by_status.setdefault(record.status, []).append(record)

    assert len(by_status[BrainProceduralSkillStatus.ACTIVE.value]) == 1
    assert len(by_status[BrainProceduralSkillStatus.SUPERSEDED.value]) == 1
    active = by_status[BrainProceduralSkillStatus.ACTIVE.value][0]
    superseded = by_status[BrainProceduralSkillStatus.SUPERSEDED.value][0]
    assert superseded.superseded_by_skill_id == active.skill_id
    assert active.supersedes_skill_id == superseded.skill_id
    assert superseded.supporting_trace_ids
    assert superseded.supporting_outcome_ids


def test_memory_v2_procedural_skill_consolidation_retires_after_relevant_failures(tmp_path):
    store = BrainStore(path=tmp_path / "retired.db")
    session_ids = resolve_brain_session_ids(runtime_kind="text", client_id="skill-retire")

    for offset, goal_id, proposal_id in (
        (0, "goal-retire-1", "proposal-retire-1"),
        (40, "goal-retire-2", "proposal-retire-2"),
    ):
        _append_completed_procedural_goal(
            store,
            session_ids,
            goal_id=goal_id,
            commitment_id=f"commitment-{goal_id}",
            goal_title="Review then report maintenance state",
            proposal_id=proposal_id,
            sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
            start_second=offset,
        )
    _append_failed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-retire-failure-1",
        commitment_id="commitment-retire-failure-1",
        goal_title="Review then report maintenance state",
        proposal_id="proposal-retire-failure-1",
        sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
        start_second=120,
        failure_reason="operator_review",
    )
    _append_failed_procedural_goal(
        store,
        session_ids,
        goal_id="goal-retire-failure-2",
        commitment_id="commitment-retire-failure-2",
        goal_title="Review then report maintenance state",
        proposal_id="proposal-retire-failure-2",
        sequence=["maintenance.review_memory_health", "reporting.record_presence_event"],
        start_second=180,
        failure_reason="operator_review",
    )

    projection = store.consolidate_procedural_skills(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    assert len(projection.skills) == 1
    retired = projection.skills[0]
    assert retired.status == BrainProceduralSkillStatus.RETIRED.value
    assert retired.retirement_reason == "repeated_relevant_failures"
    assert retired.retired_at is not None
    assert retired.failure_signatures
    assert retired.failure_signatures[0].reason_code == "operator_review"
    assert retired.failure_signatures[0].support_trace_ids
    assert retired.failure_signatures[0].support_outcome_ids
    assert retired.stats.failure_trace_count >= 2


def test_memory_v2_continuity_dossier_record_and_projection_roundtrip(tmp_path):
    _ = tmp_path
    evidence = BrainContinuityDossierEvidenceRef(
        claim_ids=["claim-1"],
        entry_ids=["entry-1"],
        source_event_ids=["evt-1"],
        source_episode_ids=[1],
        graph_node_ids=["node-1"],
        graph_edge_ids=["edge-1"],
    )
    fact = BrainContinuityDossierFactRecord(
        fact_id="fact-1",
        summary="User now prefers tea.",
        status="current",
        valid_from="2026-01-01T00:00:00+00:00",
        evidence=evidence,
        details={"predicate": "preference.like"},
    )
    issue = BrainContinuityDossierIssueRecord(
        issue_id="issue-1",
        kind="needs_refresh",
        summary="New evidence arrived after the summary anchor.",
        status="open",
        evidence=evidence,
        details={"kind": "freshness"},
    )
    governance = BrainContinuityDossierGovernanceRecord(
        supporting_claim_currentness_counts={"current": 1, "historical": 1},
        supporting_claim_review_state_counts={"none": 1, "requested": 1},
        supporting_claim_reason_codes=["review_debt", "superseded"],
        review_debt_count=1,
        last_refresh_cause="newer_support_exists",
        task_availability=[
            BrainContinuityDossierTaskAvailability(
                task="reply",
                availability=BrainContinuityDossierAvailability.SUPPRESSED.value,
                reason_codes=["review_debt"],
            ),
            BrainContinuityDossierTaskAvailability(
                task="recall",
                availability=BrainContinuityDossierAvailability.ANNOTATED.value,
                reason_codes=["needs_refresh", "review_debt"],
            ),
        ],
    )
    dossier = BrainContinuityDossierRecord(
        dossier_id="dossier-1",
        kind=BrainContinuityDossierKind.RELATIONSHIP.value,
        scope_type="relationship",
        scope_id="blink/main:user-1",
        title="Relationship with user-1",
        summary="We recovered trust after a correction.",
        status="current",
        freshness=BrainContinuityDossierFreshness.NEEDS_REFRESH.value,
        contradiction=BrainContinuityDossierContradiction.UNCERTAIN.value,
        support_strength=0.75,
        summary_evidence=evidence,
        key_current_facts=[fact],
        recent_changes=[fact],
        open_issues=[issue],
        source_entry_ids=["entry-1"],
        source_claim_ids=["claim-1"],
        source_block_ids=["block-1"],
        source_commitment_ids=["commitment-1"],
        source_plan_proposal_ids=["proposal-1"],
        source_skill_ids=["skill-1"],
        source_scene_entity_ids=["entity-1"],
        source_scene_affordance_ids=["affordance-1"],
        source_event_ids=["evt-1"],
        source_episode_ids=[1],
        details={"summary_source_entry_id": "entry-1"},
        governance=governance,
    )
    projection = BrainContinuityDossierProjection(
        scope_type="user",
        scope_id="user-1",
        dossiers=[dossier],
        dossier_counts={BrainContinuityDossierKind.RELATIONSHIP.value: 1},
        freshness_counts={BrainContinuityDossierFreshness.NEEDS_REFRESH.value: 1},
        contradiction_counts={BrainContinuityDossierContradiction.UNCERTAIN.value: 1},
        current_dossier_ids=["dossier-1"],
        stale_dossier_ids=[],
        needs_refresh_dossier_ids=["dossier-1"],
        uncertain_dossier_ids=["dossier-1"],
        contradicted_dossier_ids=[],
    )

    assert BrainContinuityDossierFactRecord.from_dict(fact.as_dict()) == fact
    assert BrainContinuityDossierIssueRecord.from_dict(issue.as_dict()) == issue
    assert (
        BrainContinuityDossierTaskAvailability.from_dict(
            governance.task_availability[0].as_dict()
        )
        == governance.task_availability[0]
    )
    assert BrainContinuityDossierGovernanceRecord.from_dict(governance.as_dict()) == governance
    assert BrainContinuityDossierRecord.from_dict(dossier.as_dict()) == dossier
    assert BrainContinuityDossierProjection.from_dict(projection.as_dict()) == projection


def test_memory_v2_continuity_dossiers_expand_graph_backed_coverage_deterministically(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-expanded")
    seeded = _build_expanded_dossier_state(store, session_ids)
    graph = seeded["graph"]
    dossiers = seeded["dossiers"]

    second_graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        core_blocks=seeded["core_blocks"],
        commitment_projection=seeded["commitment_projection"],
        agenda=seeded["agenda"],
        procedural_skills=seeded["procedural_skills"],
        scene_world_state=seeded["scene_world_state"],
        presence_scope_key="browser:presence",
    )
    second_dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=second_graph,
        core_blocks=seeded["core_blocks"],
        commitment_projection=seeded["commitment_projection"],
        agenda=seeded["agenda"],
        procedural_skills=seeded["procedural_skills"],
        scene_world_state=seeded["scene_world_state"],
        presence_scope_key="browser:presence",
    )
    assert graph.as_dict() == second_graph.as_dict()
    assert dossiers.as_dict() == second_dossiers.as_dict()

    node_by_id = {node.node_id: node for node in graph.nodes}
    dossier_kinds = {record.kind for record in dossiers.dossiers}
    assert {
        BrainContinuityDossierKind.SELF_POLICY.value,
        BrainContinuityDossierKind.USER.value,
        BrainContinuityDossierKind.COMMITMENT.value,
        BrainContinuityDossierKind.PLAN.value,
        BrainContinuityDossierKind.PROCEDURAL.value,
        BrainContinuityDossierKind.SCENE_WORLD.value,
    } <= dossier_kinds
    for dossier in dossiers.dossiers:
        if dossier.kind in {
            BrainContinuityDossierKind.SELF_POLICY.value,
            BrainContinuityDossierKind.USER.value,
            BrainContinuityDossierKind.COMMITMENT.value,
            BrainContinuityDossierKind.PLAN.value,
            BrainContinuityDossierKind.PROCEDURAL.value,
            BrainContinuityDossierKind.SCENE_WORLD.value,
        }:
            assert dossier.dossier_id.startswith("dossier_")
            assert dossier.summary_evidence.graph_node_ids
            assert all(node_id in node_by_id for node_id in dossier.summary_evidence.graph_node_ids)

    self_policy = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.SELF_POLICY.value,
    )
    user = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.USER.value,
    )
    assert any(
        record.details.get("recent_change_kind") == "superseded_block"
        for record in self_policy.recent_changes
    )
    assert any(
        record.details.get("recent_change_kind") == "superseded_block"
        for record in user.recent_changes
    )
    assert any(
        record.details.get("recent_change_kind") == "historical_claim"
        for record in user.recent_changes
    )

    commitment = next(
        record
        for record in dossiers.dossiers
        if record.kind == BrainContinuityDossierKind.COMMITMENT.value
        and seeded["active_commitment_id"] in record.source_commitment_ids
    )
    plan = next(
        record
        for record in dossiers.dossiers
        if record.kind == BrainContinuityDossierKind.PLAN.value
        and seeded["adopted_proposal_id"] in record.source_plan_proposal_ids
    )
    assert seeded["active_commitment_id"] in commitment.source_commitment_ids
    assert seeded["adopted_proposal_id"] in commitment.source_plan_proposal_ids
    assert seeded["active_commitment_id"] in plan.source_commitment_ids

    procedural = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.PROCEDURAL.value,
    )
    assert procedural.source_skill_ids
    assert all(
        node_by_id[node_id].kind == BrainContinuityGraphNodeKind.PROCEDURAL_SKILL.value
        for node_id in procedural.summary_evidence.graph_node_ids
    )

    scene_world = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.SCENE_WORLD.value,
    )
    assert scene_world.source_scene_entity_ids
    assert scene_world.source_scene_affordance_ids
    assert len(scene_world.key_current_facts) <= 6
    assert len(scene_world.recent_changes) <= 4
    assert "entities" not in scene_world.details
    assert "affordances" not in scene_world.details


def test_memory_v2_continuity_dossiers_compile_relationship_and_project_state(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-state")
    relationship_scope_id = f"{session_ids.agent_id}:{session_ids.user_id}"

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.like",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user likes coffee",
        confidence=0.82,
        singleton=False,
        source_event_id="evt-pref-like",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="preference.dislike",
        subject="coffee",
        value={"value": "coffee"},
        rendered_text="user dislikes coffee",
        confidence=0.94,
        singleton=False,
        source_event_id="evt-pref-dislike",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    current_dislike_claim = next(
        claim
        for claim in store.query_claims(
            temporal_mode="current",
            scope_type="user",
            scope_id=session_ids.user_id,
            limit=12,
        )
        if claim.predicate == "preference.dislike"
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_milestone",
        rendered_summary="We corrected a preference and kept the thread aligned.",
        content={"summary": "Preference correction milestone"},
        salience=1.1,
        source_claim_ids=[current_dislike_claim.claim_id],
        source_event_ids=["evt-pref-dislike"],
        source_event_id="evt-milestone",
        append_only=True,
        identity_key="milestone-1",
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="Trust recovered after we corrected the coffee preference.",
        content={"summary": "Trust recovered"},
        salience=1.2,
        source_claim_ids=[current_dislike_claim.claim_id],
        source_event_ids=["evt-pref-dislike"],
        source_event_id="evt-relationship-arc",
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="project_arc",
        rendered_summary="Alpha started as an exploratory maintenance project.",
        content={"summary": "Alpha started", "project_key": "Alpha"},
        salience=0.8,
        source_event_ids=["evt-project-1"],
        source_event_id="evt-project-1",
        append_only=True,
        identity_key="Alpha",
    )
    store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="project_arc",
        rendered_summary="Alpha is now a reliable maintenance collaboration.",
        content={"summary": "Alpha is reliable", "project_key": "Alpha"},
        salience=1.0,
        source_event_ids=["evt-project-2"],
        source_event_id="evt-project-2",
        append_only=True,
        identity_key="Alpha",
    )

    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=graph,
    )

    relationship = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    project = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.PROJECT.value,
        project_key="Alpha",
    )

    assert relationship.summary.startswith("Trust recovered")
    assert relationship.freshness == BrainContinuityDossierFreshness.FRESH.value
    assert relationship.contradiction == BrainContinuityDossierContradiction.CLEAR.value
    assert any(
        record.details.get("entry_kind") == "relationship_arc"
        for record in relationship.key_current_facts
    )
    assert any(
        record.details.get("predicate") == "preference.dislike"
        for record in relationship.key_current_facts
    )
    assert any(
        record.details.get("recent_change_kind") == "relationship_milestone"
        for record in relationship.recent_changes
    )
    assert any(
        record.details.get("recent_change_kind") == "superseded_claim"
        for record in relationship.recent_changes
    )
    assert relationship.summary_evidence.entry_ids
    assert relationship.summary_evidence.graph_node_ids
    assert all(
        record.evidence.claim_ids or record.evidence.entry_ids or record.evidence.source_event_ids
        for record in relationship.key_current_facts + relationship.recent_changes
    )

    assert project.project_key == "Alpha"
    assert project.summary.startswith("Alpha is now")
    assert project.freshness == BrainContinuityDossierFreshness.FRESH.value
    assert project.contradiction == BrainContinuityDossierContradiction.CLEAR.value
    assert any(
        record.details.get("entry_kind") == "project_arc" for record in project.key_current_facts
    )
    assert any(record.summary.startswith("Alpha started") for record in project.recent_changes)
    assert len(project.source_entry_ids) >= 2
    assert project.summary_evidence.entry_ids
    assert relationship.governance.last_refresh_cause == "fresh_current_support"
    assert project.governance.last_refresh_cause == "fresh_current_support"
    assert relationship.governance.review_debt_count == 0
    assert any(
        record.task == "reply"
        and record.availability == BrainContinuityDossierAvailability.AVAILABLE.value
        for record in relationship.governance.task_availability
    )
    assert any(
        record.task == "planning"
        and record.availability == BrainContinuityDossierAvailability.AVAILABLE.value
        for record in project.governance.task_availability
    )
    assert dossiers.dossier_counts[BrainContinuityDossierKind.RELATIONSHIP.value] == 1
    assert dossiers.dossier_counts[BrainContinuityDossierKind.PROJECT.value] == 1


def test_memory_v2_continuity_dossiers_expose_governance_review_debt_and_history(tmp_path):
    store = BrainStore(path=tmp_path / "dossier-governance.db")
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="dossier-governance",
    )
    user = store.ensure_entity(
        entity_type="user",
        canonical_name=session_ids.user_id,
        aliases=[session_ids.user_id],
        attributes={"user_id": session_ids.user_id},
    )
    ledger = ClaimLedger(store=store)
    event_context = store._memory_event_context(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source="test",
        correlation_id="dossier-governance",
    )

    prior_claim = ledger.record_claim(
        subject_entity_id=user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        source_event_id="evt-role-1",
        scope_type="user",
        scope_id=session_ids.user_id,
        claim_key="profile.role",
        evidence_summary="original role",
        event_context=event_context,
    )
    current_claim = ledger.supersede_claim(
        prior_claim.claim_id,
        replacement_claim={
            "subject_entity_id": user.entity_id,
            "predicate": "profile.role",
            "object_data": {"value": "product manager"},
            "scope_type": "user",
            "scope_id": session_ids.user_id,
            "claim_key": "profile.role",
            "evidence_summary": "corrected role",
        },
        reason="updated the recorded role",
        source_event_id="evt-role-2",
        event_context=event_context,
    )
    held_claim = ledger.request_claim_review(
        current_claim.claim_id,
        source_event_id="evt-role-review",
        reason_codes=[BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value],
        summary="Needs confirmation before reuse.",
        event_context=event_context,
    )

    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=graph,
    )
    relationship = _dossier_by_kind(
        dossiers,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    task_availability = {
        record.task: record for record in relationship.governance.task_availability
    }

    assert relationship.governance.supporting_claim_currentness_counts == {
        BrainClaimCurrentnessStatus.HELD.value: 1,
        BrainClaimCurrentnessStatus.HISTORICAL.value: 1,
    }
    assert relationship.governance.supporting_claim_review_state_counts == {
        BrainClaimReviewState.NONE.value: 1,
        BrainClaimReviewState.REQUESTED.value: 1,
    }
    assert relationship.governance.review_debt_count == 1
    assert relationship.governance.last_refresh_cause == "no_fresh_support"
    assert BrainGovernanceReasonCode.REQUIRES_CONFIRMATION.value in (
        relationship.governance.supporting_claim_reason_codes
    )
    assert task_availability["reply"].availability == BrainContinuityDossierAvailability.SUPPRESSED.value
    assert task_availability["planning"].availability == BrainContinuityDossierAvailability.SUPPRESSED.value
    assert task_availability["recall"].availability == BrainContinuityDossierAvailability.ANNOTATED.value
    assert task_availability["reflection"].availability == BrainContinuityDossierAvailability.ANNOTATED.value
    assert task_availability["critique"].availability == BrainContinuityDossierAvailability.SUPPRESSED.value
    assert task_availability["reply"].reason_codes == [
        "held_support",
        "review_debt",
        "stale_support",
    ]
    assert {issue.kind for issue in relationship.open_issues} >= {
        "held_support",
        "review_debt",
        "stale_support",
    }
    for issue in relationship.open_issues:
        if issue.kind in {"held_support", "review_debt"}:
            assert issue.evidence.claim_ids
            assert issue.evidence.graph_node_ids
    assert any(
        record.status == BrainClaimCurrentnessStatus.HELD.value
        and held_claim.claim_id in record.evidence.claim_ids
        for record in relationship.key_current_facts
    )
    assert any(
        record.details.get("recent_change_kind") == "superseded_claim"
        and prior_claim.claim_id in record.evidence.claim_ids
        for record in relationship.recent_changes
    )
    assert prior_claim.claim_id in relationship.source_claim_ids
    assert held_claim.claim_id in relationship.source_claim_ids


def test_memory_v2_continuity_dossiers_surface_freshness_and_contradiction(tmp_path):
    fresh_store = BrainStore(path=tmp_path / "fresh.db")
    fresh_session = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-fresh")
    relationship_scope_id = f"{fresh_session.agent_id}:{fresh_session.user_id}"
    fresh_store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=relationship_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="We have a fresh shared summary.",
        content={"summary": "fresh"},
        salience=1.0,
        source_event_ids=["evt-fresh-summary"],
        source_event_id="evt-fresh-summary",
    )
    fresh_projection = fresh_store.build_continuity_dossiers(
        agent_id=fresh_session.agent_id,
        user_id=fresh_session.user_id,
        thread_id=fresh_session.thread_id,
        scope_type="user",
        scope_id=fresh_session.user_id,
    )
    fresh_relationship = _dossier_by_kind(
        fresh_projection,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    assert fresh_relationship.freshness == BrainContinuityDossierFreshness.FRESH.value
    assert fresh_relationship.contradiction == BrainContinuityDossierContradiction.CLEAR.value

    stale_store = BrainStore(path=tmp_path / "stale.db")
    stale_session = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-stale")
    stale_user = stale_store.ensure_entity(
        entity_type="user",
        canonical_name=stale_session.user_id,
        aliases=[stale_session.user_id],
        attributes={"user_id": stale_session.user_id},
    )
    stale_claim = stale_store._claims().record_claim(
        subject_entity_id=stale_user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        status="active",
        confidence=0.8,
        valid_from="2026-01-01T00:00:00+00:00",
        source_event_id="evt-stale-claim",
        scope_type="user",
        scope_id=stale_session.user_id,
        claim_key="profile.role:stale",
        event_context=None,
    )
    stale_store._claims().expire_claim(
        stale_claim.claim_id,
        source_event_id="evt-stale-governance",
        reason_codes=["stale_without_refresh"],
        event_context=None,
    )
    stale_projection = stale_store.build_continuity_dossiers(
        agent_id=stale_session.agent_id,
        user_id=stale_session.user_id,
        thread_id=stale_session.thread_id,
        scope_type="user",
        scope_id=stale_session.user_id,
    )
    stale_relationship = _dossier_by_kind(
        stale_projection,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    assert stale_relationship.freshness == BrainContinuityDossierFreshness.STALE.value
    assert stale_relationship.dossier_id in stale_projection.stale_dossier_ids
    assert any(issue.kind == "stale_support" for issue in stale_relationship.open_issues)

    refresh_store = BrainStore(path=tmp_path / "refresh.db")
    refresh_session = resolve_brain_session_ids(runtime_kind="browser", client_id="dossier-refresh")
    refresh_scope_id = f"{refresh_session.agent_id}:{refresh_session.user_id}"
    refresh_store.upsert_autobiographical_entry(
        scope_type="relationship",
        scope_id=refresh_scope_id,
        entry_kind="relationship_arc",
        rendered_summary="An older relationship summary.",
        content={"summary": "old"},
        salience=0.8,
        source_event_ids=["evt-refresh-summary"],
        source_event_id="evt-refresh-summary",
    )
    refresh_store.remember_fact(
        user_id=refresh_session.user_id,
        namespace="profile.role",
        subject="user",
        value={"value": "maintainer"},
        rendered_text="user role is maintainer",
        confidence=0.9,
        singleton=True,
        source_event_id="evt-refresh-claim",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=refresh_session.agent_id,
        session_id=refresh_session.session_id,
        thread_id=refresh_session.thread_id,
    )
    refresh_projection = refresh_store.build_continuity_dossiers(
        agent_id=refresh_session.agent_id,
        user_id=refresh_session.user_id,
        thread_id=refresh_session.thread_id,
        scope_type="user",
        scope_id=refresh_session.user_id,
    )
    refresh_relationship = _dossier_by_kind(
        refresh_projection,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    assert refresh_relationship.freshness == BrainContinuityDossierFreshness.NEEDS_REFRESH.value
    assert refresh_relationship.dossier_id in refresh_projection.needs_refresh_dossier_ids
    assert any(issue.kind == "needs_refresh" for issue in refresh_relationship.open_issues)

    uncertain_store = BrainStore(path=tmp_path / "uncertain.db")
    uncertain_session = resolve_brain_session_ids(
        runtime_kind="browser", client_id="dossier-uncertain"
    )
    uncertain_user = uncertain_store.ensure_entity(
        entity_type="user",
        canonical_name=uncertain_session.user_id,
        aliases=[uncertain_session.user_id],
        attributes={"user_id": uncertain_session.user_id},
    )
    uncertain_store._claims().record_claim(
        subject_entity_id=uncertain_user.entity_id,
        predicate="profile.role",
        object_data={"value": "writer"},
        status="uncertain",
        confidence=0.55,
        valid_from="2026-04-19T00:00:00+00:00",
        source_event_id="evt-uncertain-claim",
        scope_type="user",
        scope_id=uncertain_session.user_id,
        claim_key="profile.role:uncertain",
        event_context=None,
    )
    uncertain_projection = uncertain_store.build_continuity_dossiers(
        agent_id=uncertain_session.agent_id,
        user_id=uncertain_session.user_id,
        thread_id=uncertain_session.thread_id,
        scope_type="user",
        scope_id=uncertain_session.user_id,
    )
    uncertain_relationship = _dossier_by_kind(
        uncertain_projection,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    assert (
        uncertain_relationship.contradiction == BrainContinuityDossierContradiction.UNCERTAIN.value
    )
    assert uncertain_relationship.dossier_id in uncertain_projection.uncertain_dossier_ids
    assert any(issue.kind == "uncertain_claim" for issue in uncertain_relationship.open_issues)

    conflict_store = BrainStore(path=tmp_path / "conflict.db")
    conflict_session = resolve_brain_session_ids(
        runtime_kind="browser", client_id="dossier-conflict"
    )
    conflict_user = conflict_store.ensure_entity(
        entity_type="user",
        canonical_name=conflict_session.user_id,
        aliases=[conflict_session.user_id],
        attributes={"user_id": conflict_session.user_id},
    )
    conflict_store._claims().record_claim(
        subject_entity_id=conflict_user.entity_id,
        predicate="profile.role",
        object_data={"value": "designer"},
        status="active",
        confidence=0.8,
        valid_from="2026-04-19T00:00:00+00:00",
        source_event_id="evt-conflict-1",
        scope_type="user",
        scope_id=conflict_session.user_id,
        claim_key="profile.role:conflict",
        event_context=None,
    )
    conflict_store._claims().record_claim(
        subject_entity_id=conflict_user.entity_id,
        predicate="profile.role",
        object_data={"value": "manager"},
        status="active",
        confidence=0.82,
        valid_from="2026-04-19T00:01:00+00:00",
        source_event_id="evt-conflict-2",
        scope_type="user",
        scope_id=conflict_session.user_id,
        claim_key="profile.role:conflict",
        event_context=None,
    )
    conflict_projection = conflict_store.build_continuity_dossiers(
        agent_id=conflict_session.agent_id,
        user_id=conflict_session.user_id,
        thread_id=conflict_session.thread_id,
        scope_type="user",
        scope_id=conflict_session.user_id,
    )
    conflict_relationship = _dossier_by_kind(
        conflict_projection,
        BrainContinuityDossierKind.RELATIONSHIP.value,
    )
    assert (
        conflict_relationship.contradiction
        == BrainContinuityDossierContradiction.CONTRADICTED.value
    )
    assert conflict_relationship.dossier_id in conflict_projection.contradicted_dossier_ids
    assert any(
        issue.kind == "conflicting_current_claims" for issue in conflict_relationship.open_issues
    )


def test_memory_v2_continuity_graph_includes_planning_adopt_and_reject_links(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="graph-planning")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    goal_id = executive.create_commitment_goal(
        title="Graph planning commitment",
        intent="maintenance.review",
        source="test",
    )
    commitment = store.get_session_commitment_projection(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    ).active_commitments[0]

    adopted_proposal = BrainPlanProposal(
        plan_proposal_id="plan-proposal-adopted",
        goal_id=goal_id,
        commitment_id=commitment.commitment_id,
        source=BrainPlanProposalSource.REPAIR.value,
        summary="Adopt a repaired maintenance tail.",
        current_plan_revision=commitment.plan_revision,
        plan_revision=commitment.plan_revision + 1,
        review_policy=BrainPlanReviewPolicy.AUTO_ADOPT_OK.value,
        preserved_prefix_count=1,
        details={"request_kind": "revise_tail"},
        created_at="2026-03-01T00:00:00+00:00",
    )
    proposed_event = store.append_planning_proposed(
        proposal=adopted_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_planning_adopted(
        proposal=adopted_proposal,
        decision=BrainPlanProposalDecision(
            summary="Repair applied.",
            reason="repair_applied",
            details={},
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        causal_parent_id=proposed_event.event_id,
    )

    rejected_proposal = BrainPlanProposal(
        plan_proposal_id="plan-proposal-rejected",
        goal_id=goal_id,
        commitment_id=commitment.commitment_id,
        source=BrainPlanProposalSource.REPAIR.value,
        summary="Reject the unsupported revision.",
        current_plan_revision=commitment.plan_revision + 1,
        plan_revision=commitment.plan_revision + 2,
        review_policy=BrainPlanReviewPolicy.NEEDS_OPERATOR_REVIEW.value,
        preserved_prefix_count=1,
        supersedes_plan_proposal_id=adopted_proposal.plan_proposal_id,
        details={"request_kind": "revise_tail"},
        created_at="2026-03-02T00:00:00+00:00",
    )
    rejected_event = store.append_planning_proposed(
        proposal=rejected_proposal,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
    )
    store.append_planning_rejected(
        proposal=rejected_proposal,
        decision=BrainPlanProposalDecision(
            summary="Unsupported capability family.",
            reason="unsupported_capability",
            details={},
        ),
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="test",
        causal_parent_id=rejected_event.event_id,
    )

    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
    )
    proposal_nodes = {
        record.backing_record_id: record
        for record in graph.nodes
        if record.kind == BrainContinuityGraphNodeKind.PLAN_PROPOSAL.value
    }
    current_backing_ids = _graph_backing_ids(graph, graph.current_node_ids)
    historical_backing_ids = _graph_backing_ids(graph, graph.historical_node_ids)
    edge_kinds = {edge.kind for edge in graph.edges}

    assert proposal_nodes[adopted_proposal.plan_proposal_id].status == "adopted"
    assert proposal_nodes[rejected_proposal.plan_proposal_id].status == "rejected"
    assert adopted_proposal.plan_proposal_id in historical_backing_ids
    assert adopted_proposal.plan_proposal_id in _graph_backing_ids(graph, graph.superseded_node_ids)
    assert rejected_proposal.plan_proposal_id in historical_backing_ids
    assert BrainContinuityGraphEdgeKind.COMMITMENT_HAS_PLAN_PROPOSAL.value in edge_kinds
    assert BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_ADOPTED_INTO_COMMITMENT.value in edge_kinds
    assert BrainContinuityGraphEdgeKind.PLAN_PROPOSAL_SUPERSEDES.value in edge_kinds
    assert adopted_proposal.plan_proposal_id not in current_backing_ids


def test_memory_v2_backfill_from_legacy_is_idempotent_and_seeds_full_block_surface(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    store.remember_semantic_memory(
        user_id="user-1",
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.9,
        singleton=True,
        source_event_id="legacy-semantic-1",
        source_episode_id=1,
        provenance={"source": "legacy"},
    )
    store.upsert_narrative_memory(
        user_id="user-1",
        thread_id="browser:user-1",
        kind="commitment",
        title="给妈妈打电话",
        summary="给妈妈打电话",
        details={"details": "今晚"},
        status="open",
        confidence=0.8,
        source_event_id="legacy-commitment-1",
        provenance={"source": "legacy"},
    )
    store.upsert_narrative_memory(
        user_id="user-1",
        thread_id="browser:user-1",
        kind="session_summary",
        title="browser:user-1",
        summary="你让我记住给妈妈打电话。",
        details={"thread_id": "browser:user-1"},
        status="active",
        confidence=0.9,
        source_event_id="legacy-summary-1",
        provenance={"source": "legacy"},
    )

    store.backfill_continuity_from_legacy(
        user_id="user-1",
        thread_id="browser:user-1",
    )
    store.backfill_continuity_from_legacy(
        user_id="user-1",
        thread_id="browser:user-1",
    )

    user_entity = store.ensure_entity(
        entity_type="user",
        canonical_name="user-1",
        aliases=["user-1"],
        attributes={"user_id": "user-1"},
    )
    current_claims = store.query_claims(
        temporal_mode="current",
        subject_entity_id=user_entity.entity_id,
        scope_type="user",
        scope_id="user-1",
        limit=12,
    )
    profile_name_claim = next(
        claim for claim in current_claims if claim.predicate == "profile.name"
    )

    assert len([claim for claim in current_claims if claim.predicate == "profile.name"]) == 1
    assert len(store._claims().claim_evidence(profile_name_claim.claim_id)) == 1
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.SELF_CORE.value,
                scope_type="agent",
                scope_id="blink/main",
            )
        )
        == 1
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.USER_CORE.value,
                scope_type="user",
                scope_id="user-1",
            )
        )
        == 1
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.RELATIONSHIP_CORE.value,
                scope_type="relationship",
                scope_id="blink/main:user-1",
            )
        )
        == 1
    )
    assert (
        len(
            store.list_core_memory_block_versions(
                block_kind=BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value,
                scope_type="relationship",
                scope_id="blink/main:user-1",
            )
        )
        == 1
    )


def test_memory_v2_context_surfaces_are_current_truth_first_and_include_core_blocks(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    store.ensure_default_blocks(load_default_agent_blocks())

    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.name",
        subject="user",
        value={"value": "小周"},
        rendered_text="用户名字是 小周",
        confidence=0.82,
        singleton=True,
        source_event_id="evt-name-1",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.remember_fact(
        user_id=session_ids.user_id,
        namespace="profile.name",
        subject="user",
        value={"value": "阿周"},
        rendered_text="用户名字是 阿周",
        confidence=0.94,
        singleton=True,
        source_event_id="evt-name-2",
        source_episode_id=None,
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
    )
    store.upsert_task(
        user_id=session_ids.user_id,
        title="给妈妈打电话",
        details={"details": "今晚"},
        status="open",
        thread_id=session_ids.thread_id,
        source_event_id="evt-task-1",
        provenance={"source": "tool"},
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
    )
    store.upsert_session_summary(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        summary="你让我记住给妈妈打电话。",
        agent_id=session_ids.agent_id,
        session_id=session_ids.session_id,
        source_event_id="evt-summary-1",
    )

    builder = BrainContextSurfaceBuilder(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
        language=Language.ZH,
    )
    surface = builder.build(latest_user_text="你还记得我吗？", include_historical_claims=True)

    assert BrainCoreMemoryBlockKind.SELF_CORE.value in surface.core_blocks
    assert BrainCoreMemoryBlockKind.SELF_PERSONA_CORE.value in surface.core_blocks
    assert BrainCoreMemoryBlockKind.USER_CORE.value in surface.core_blocks
    assert BrainCoreMemoryBlockKind.ACTIVE_COMMITMENTS.value in surface.core_blocks
    assert any(
        claim.predicate == "profile.name" and claim.object.get("value") == "阿周"
        for claim in surface.current_claims
    )
    assert any(
        claim.predicate == "profile.name" and claim.object.get("value") == "小周"
        for claim in surface.historical_claims
    )
    summary = render_user_profile_summary(surface, Language.ZH)
    assert "阿周" in summary
    assert "小周" not in summary


@pytest.mark.asyncio
async def test_memory_v2_tools_exports_and_replay_preserve_continuity_history(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    llm = DummyLLM()
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="pc-123")
    register_memory_tools(
        llm=llm,
        store=store,
        session_resolver=lambda: session_ids,
        language=Language.ZH,
    )

    await _call_tool(
        llm.registered_functions["brain_remember_preference"],
        {"sentiment": "like", "topic": "咖啡"},
    )
    await _call_tool(
        llm.registered_functions["brain_remember_preference"],
        {"sentiment": "dislike", "topic": "咖啡"},
    )
    await _call_tool(
        llm.registered_functions["brain_remember_task"],
        {"title": "给妈妈打电话", "details": "今晚"},
    )
    await _call_tool(
        llm.registered_functions["brain_complete_task"],
        {"title": "给妈妈打电话"},
    )

    artifact = BrainMemoryExporter(store=store).export_thread_digest(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
    )
    exported = json.loads(artifact.path.read_text(encoding="utf-8"))
    scenario = BrainReplayScenario(
        name="phase2_claim_correction",
        description="phase2 claim correction replay",
        session_ids=session_ids,
        events=tuple(
            reversed(
                store.recent_brain_events(
                    user_id=session_ids.user_id,
                    thread_id=session_ids.thread_id,
                    limit=64,
                )
            )
        ),
        expected_terminal_state={},
    )
    replay = BrainReplayHarness(store=store).replay(scenario)
    replay_payload = json.loads(replay.artifact_path.read_text(encoding="utf-8"))

    assert exported["core_blocks"]
    assert exported["core_block_versions"]["self_core"]
    assert exported["current_claims"]
    assert exported["historical_claims"]
    assert exported["claim_supersessions"]
    assert exported["continuity_graph"]["node_counts"]["claim"] >= 2
    assert "self_core" in replay.context_surface.core_blocks
    assert any(
        claim.predicate == "preference.dislike" and claim.object.get("value") == "咖啡"
        for claim in replay.context_surface.current_claims
    )
    assert any(
        claim.predicate == "preference.like" and claim.object.get("value") == "咖啡"
        for claim in replay.context_surface.historical_claims
    )
    assert (
        replay_payload["continuity_graph"] == replay_payload["continuity_state"]["continuity_graph"]
    )
    replay_graph_current_backing_ids = _graph_backing_ids(
        BrainContinuityGraphProjection.from_dict(replay_payload["continuity_graph"])
        or BrainContinuityGraphProjection(
            scope_type="user",
            scope_id=session_ids.user_id,
        ),
        replay_payload["continuity_graph"]["current_node_ids"],
    )
    replay_graph_historical_backing_ids = _graph_backing_ids(
        BrainContinuityGraphProjection.from_dict(replay_payload["continuity_graph"])
        or BrainContinuityGraphProjection(
            scope_type="user",
            scope_id=session_ids.user_id,
        ),
        replay_payload["continuity_graph"]["historical_node_ids"],
    )
    assert any(
        "preference.dislike" == claim.predicate for claim in replay.context_surface.current_claims
    )
    assert any(backing_id.startswith("claim_") for backing_id in replay_graph_current_backing_ids)
    assert any(
        backing_id.startswith("claim_") for backing_id in replay_graph_historical_backing_ids
    )


def test_scene_first_multimodal_autobiography_distills_and_redacts_bounded_entries(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase17-memory")
    projection = _scene_world_projection_for_multimodal(
        scope_id="browser:presence",
        source_event_ids=["evt-scene-a"],
        updated_at=_ts(20),
        include_person=True,
    )

    record = _seed_scene_episode(
        store,
        session_ids,
        projection=projection,
        start_second=20,
        include_attention=True,
    )
    assert record is not None
    typed = parse_multimodal_autobiography_record(record)
    assert typed is not None
    assert typed.modality == "scene_world"
    assert typed.privacy_class == BrainMultimodalAutobiographyPrivacyClass.SENSITIVE.value
    assert typed.review_state == BrainClaimReviewState.REQUESTED.value
    assert typed.retention_class == BrainClaimRetentionClass.SESSION.value
    assert BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value in typed.governance_reason_codes
    assert typed.source_scene_entity_ids
    assert typed.source_scene_affordance_ids
    assert typed.source_event_ids
    encoded_content = json.dumps(typed.content, ensure_ascii=False, sort_keys=True)
    assert "frame_bytes" not in encoded_content
    assert "vision_payload" not in encoded_content

    duplicate = store.refresh_scene_episode_autobiography(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        session_id=session_ids.session_id,
        agent_id=session_ids.agent_id,
        presence_scope_key="browser:presence",
        scene_world_state=projection,
        recent_events=store.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        ),
        source_event_id="evt-duplicate",
        updated_at=projection.updated_at,
        event_context=_multimodal_event_context(
            store,
            session_ids,
            correlation_id="scene-episode-duplicate",
        ),
    )
    assert duplicate is not None
    assert duplicate.entry_id == record.entry_id
    assert (
        len(
            store.autobiographical_entries(
                scope_type="presence",
                scope_id="browser:presence",
                entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
                statuses=("current", "superseded"),
                limit=8,
            )
        )
        == 1
    )

    changed_projection = _scene_world_projection_for_multimodal(
        scope_id="browser:presence",
        source_event_ids=["evt-scene-b"],
        updated_at=_ts(40),
        include_person=True,
        object_state=BrainSceneWorldRecordState.STALE.value,
        affordance_availability=BrainSceneWorldAffordanceAvailability.BLOCKED.value,
    )
    replacement = _seed_scene_episode(
        store,
        session_ids,
        projection=changed_projection,
        start_second=40,
    )
    assert replacement is not None
    assert replacement.entry_id != record.entry_id
    history = store.autobiographical_entries(
        scope_type="presence",
        scope_id="browser:presence",
        entry_kinds=(BrainAutobiographyEntryKind.SCENE_EPISODE.value,),
        statuses=("current", "superseded"),
        limit=8,
    )
    assert any(entry.entry_id == record.entry_id and entry.status == "superseded" for entry in history)
    assert any(entry.entry_id == replacement.entry_id and entry.status == "current" for entry in history)

    redacted = store.redact_autobiographical_entry(
        replacement.entry_id,
        redacted_summary="Redacted scene episode.",
        source_event_id="evt-redact",
        reason_codes=[BrainGovernanceReasonCode.PRIVACY_BOUNDARY.value],
        event_context=_multimodal_event_context(store, session_ids, correlation_id="scene-redact"),
    )
    typed_redacted = parse_multimodal_autobiography_record(redacted)
    assert typed_redacted is not None
    assert typed_redacted.privacy_class == BrainMultimodalAutobiographyPrivacyClass.REDACTED.value
    assert typed_redacted.review_state == BrainClaimReviewState.RESOLVED.value
    assert typed_redacted.rendered_summary == "Redacted scene episode."
    assert typed_redacted.content.get("redacted") is True
    assert typed_redacted.redacted_at is not None
    assert "Ada is at the desk." not in json.dumps(
        typed_redacted.as_dict(), ensure_ascii=False, sort_keys=True
    )


def test_scene_episode_upsert_events_preserve_perception_lineage(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase17-lineage")
    record = _seed_scene_episode(
        store,
        session_ids,
        projection=_scene_world_projection_for_multimodal(
            scope_id="browser:presence",
            source_event_ids=["evt-scene-lineage"],
            updated_at=_ts(20),
            include_person=True,
        ),
        start_second=20,
        include_attention=True,
    )
    assert record is not None

    all_events = _all_brain_events(store)
    scene_event = next(
        event for event in all_events if event.event_type == BrainEventType.SCENE_CHANGED
    )
    upsert_event = next(
        event
        for event in all_events
        if event.event_type == BrainEventType.AUTOBIOGRAPHY_ENTRY_UPSERTED
        and dict(event.payload or {}).get("entry_id") == record.entry_id
    )
    upsert_payload = dict(upsert_event.payload or {})

    assert all_events.index(scene_event) < all_events.index(upsert_event)
    assert upsert_event.causal_parent_id == scene_event.event_id
    assert upsert_payload["entry_kind"] == BrainAutobiographyEntryKind.SCENE_EPISODE.value
    assert upsert_payload["source_presence_scope_key"] == "browser:presence"
    assert scene_event.event_id in upsert_payload["source_event_ids"]
    assert record.entry_id == upsert_payload["entry_id"]


def test_prediction_events_preserve_source_lineage_and_ordering(tmp_path):
    from tests.test_brain_world_model import (
        _append_body_state,
        _append_goal_created,
        _append_robot_action_outcome,
        _append_scene_changed,
        _ensure_blocks,
    )

    store = BrainStore(path=tmp_path / "brain.db")
    _ensure_blocks(store)
    session_ids = resolve_brain_session_ids(
        runtime_kind="browser",
        client_id="phase18a-lineage",
    )
    _append_body_state(store, session_ids, second=1)
    scene_event = _append_scene_changed(store, session_ids, second=2)
    _append_goal_created(store, session_ids, second=3)
    refresh_event = _append_scene_changed(store, session_ids, second=4)
    robot_event = _append_robot_action_outcome(store, session_ids, second=5, accepted=False)

    all_events = _all_brain_events(store)
    scene_prediction_event = next(
        event
        for event in all_events
        if event.event_type == BrainEventType.SCENE_PREDICTION_GENERATED
        and event.causal_parent_id == refresh_event.event_id
    )
    action_prediction_event = next(
        event
        for event in all_events
        if event.event_type == BrainEventType.ACTION_OUTCOME_PREDICTED
        and event.causal_parent_id == refresh_event.event_id
    )
    invalidation_event = next(
        event
        for event in all_events
        if event.event_type == BrainEventType.PREDICTION_INVALIDATED
        and event.causal_parent_id == robot_event.event_id
    )

    scene_prediction_payload = dict(scene_prediction_event.payload or {})
    action_prediction_payload = dict(action_prediction_event.payload or {})
    invalidation_payload = dict(invalidation_event.payload or {})

    assert all_events.index(refresh_event) < all_events.index(scene_prediction_event)
    assert all_events.index(refresh_event) < all_events.index(action_prediction_event)
    assert all_events.index(robot_event) < all_events.index(invalidation_event)

    assert scene_prediction_event.causal_parent_id == refresh_event.event_id
    assert action_prediction_event.causal_parent_id == refresh_event.event_id
    assert invalidation_event.causal_parent_id == robot_event.event_id

    assert scene_prediction_payload["prediction"]["presence_scope_key"] == "browser:presence"
    assert scene_event.event_id in scene_prediction_payload["prediction"]["supporting_event_ids"]
    assert action_prediction_payload["prediction"]["prediction_kind"] == "action_outcome"
    assert invalidation_payload["prediction"]["resolution_kind"] == "invalidated"
    assert robot_event.event_id in invalidation_payload["prediction"]["resolution_event_ids"]


def test_scene_episode_graph_and_scene_world_dossier_keep_evidence_reachable(tmp_path):
    store = BrainStore(path=tmp_path / "brain.db")
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="phase17-dossier")
    projection = _scene_world_projection_for_multimodal(
        scope_id="browser:presence",
        source_event_ids=["evt-scene-graph"],
        updated_at=_ts(60),
        include_person=False,
        degraded_mode="limited",
    )
    entry = _seed_scene_episode(
        store,
        session_ids,
        projection=projection,
        start_second=60,
    )
    assert entry is not None

    graph = store.build_continuity_graph(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        scene_world_state=projection,
        presence_scope_key="browser:presence",
    )
    dossiers = store.build_continuity_dossiers(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        scope_type="user",
        scope_id=session_ids.user_id,
        continuity_graph=graph,
        scene_world_state=projection,
        presence_scope_key="browser:presence",
    )
    graph_nodes = {node.node_id: node for node in graph.nodes}
    graph_edges = {edge.edge_id: edge for edge in graph.edges}
    scene_world = _dossier_by_kind(dossiers, BrainContinuityDossierKind.SCENE_WORLD.value)

    assert any(
        node.details.get("entry_kind") == BrainAutobiographyEntryKind.SCENE_EPISODE.value
        for node in graph.nodes
        if node.kind == BrainContinuityGraphNodeKind.AUTOBIOGRAPHY_ENTRY.value
    )
    assert any(
        edge.kind == BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_ENTITY.value
        for edge in graph.edges
    )
    assert any(
        edge.kind
        == BrainContinuityGraphEdgeKind.AUTOBIOGRAPHY_REFERENCES_SCENE_WORLD_AFFORDANCE.value
        for edge in graph.edges
    )
    assert any(
        record.details.get("record_kind") == "scene_episode"
        for record in scene_world.key_current_facts
    )
    for evidence in [
        scene_world.summary_evidence,
        *(record.evidence for record in scene_world.key_current_facts),
        *(record.evidence for record in scene_world.recent_changes),
    ]:
        assert all(node_id in graph_nodes for node_id in evidence.graph_node_ids)
        assert all(edge_id in graph_edges for edge_id in evidence.graph_edge_ids)

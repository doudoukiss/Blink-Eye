"""Typed bounded counterfactual rehearsal and calibration helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from blink.brain.context import (
    BrainContextSelector,
    BrainContextTask,
    compile_context_packet_from_surface,
)
from blink.brain.context_packet_digest import build_context_packet_digest
from blink.brain.events import BrainEventRecord, BrainEventType
from blink.brain.projections import (
    BrainActionOutcomeComparisonRecord,
    BrainActionRehearsalRequest,
    BrainActionRehearsalResult,
    BrainCalibrationBucket,
    BrainCounterfactualCalibrationSummary,
    BrainCounterfactualEvaluationRecord,
    BrainCounterfactualRehearsalKind,
    BrainCounterfactualRehearsalProjection,
    BrainObservedActionOutcomeKind,
    BrainPredictionConfidenceBand,
    BrainPredictiveWorldModelProjection,
    BrainRehearsalDecisionRecommendation,
)
from blink.embodiment.robot_head.live_hardware import load_robot_head_live_hardware_profile
from blink.embodiment.robot_head.models import RobotHeadCommand, RobotHeadDriverStatus
from blink.embodiment.robot_head.simulation import (
    RobotHeadSimulator,
    load_robot_head_simulation_scenario,
)
from blink.transcriptions.language import Language

if TYPE_CHECKING:
    from blink.brain.actions import EmbodiedActionDefinition, EmbodiedActionEngine
    from blink.brain.context_surfaces import BrainContextSurfaceBuilder
    from blink.brain.projections import BrainCommitmentRecord, BrainGoal, BrainPlanProposal

_COUNTERFACTUAL_EVENT_TYPES = frozenset(
    {
        BrainEventType.ACTION_REHEARSAL_REQUESTED,
        BrainEventType.ACTION_REHEARSAL_COMPLETED,
        BrainEventType.ACTION_REHEARSAL_SKIPPED,
        BrainEventType.ACTION_OUTCOME_COMPARED,
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _stable_id(prefix: str, payload: str) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, payload).hex}"


def _confidence_band(probability: float) -> str:
    if probability >= 0.8:
        return BrainPredictionConfidenceBand.HIGH.value
    if probability >= 0.55:
        return BrainPredictionConfidenceBand.MEDIUM.value
    return BrainPredictionConfidenceBand.LOW.value


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def counterfactual_event_types() -> frozenset[str]:
    """Return explicit rehearsal and calibration lifecycle events."""
    return _COUNTERFACTUAL_EVENT_TYPES


def is_counterfactual_event_type(event_type: str) -> bool:
    """Return whether one event belongs to the rehearsal lifecycle."""
    return event_type in _COUNTERFACTUAL_EVENT_TYPES


def append_rehearsal_request(
    projection: BrainCounterfactualRehearsalProjection,
    request: BrainActionRehearsalRequest,
) -> None:
    """Append one open rehearsal request into the projection."""
    projection.open_requests = [
        record
        for record in projection.open_requests
        if record.rehearsal_id != request.rehearsal_id
    ]
    projection.open_requests.append(request)
    projection.calibration_summary = BrainCounterfactualCalibrationSummary(
        requested_count=projection.calibration_summary.requested_count + 1,
        completed_count=projection.calibration_summary.completed_count,
        skipped_count=projection.calibration_summary.skipped_count,
        comparison_count=projection.calibration_summary.comparison_count,
        recommendation_counts=dict(projection.calibration_summary.recommendation_counts),
        calibration_bucket_counts=dict(projection.calibration_summary.calibration_bucket_counts),
        risk_code_counts=dict(projection.calibration_summary.risk_code_counts),
        observed_outcome_counts=dict(projection.calibration_summary.observed_outcome_counts),
        updated_at=request.updated_at,
    )
    projection.updated_at = request.updated_at
    projection.sync_lists()


def append_rehearsal_result(
    projection: BrainCounterfactualRehearsalProjection,
    result: BrainActionRehearsalResult,
    *,
    max_recent_rehearsals: int = 24,
) -> None:
    """Apply one completed or skipped rehearsal result to the projection."""
    projection.open_requests = [
        record
        for record in projection.open_requests
        if record.rehearsal_id != result.rehearsal_id
    ]
    projection.recent_rehearsals = [
        record
        for record in projection.recent_rehearsals
        if record.rehearsal_id != result.rehearsal_id
    ]
    projection.recent_rehearsals.append(result)
    projection.recent_rehearsals = sorted(
        projection.recent_rehearsals,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_rehearsals]
    projection.calibration_summary = BrainCounterfactualCalibrationSummary(
        requested_count=projection.calibration_summary.requested_count,
        completed_count=projection.calibration_summary.completed_count + int(not result.skipped),
        skipped_count=projection.calibration_summary.skipped_count + int(result.skipped),
        comparison_count=projection.calibration_summary.comparison_count,
        recommendation_counts=dict(projection.calibration_summary.recommendation_counts),
        calibration_bucket_counts=dict(projection.calibration_summary.calibration_bucket_counts),
        risk_code_counts=dict(projection.calibration_summary.risk_code_counts),
        observed_outcome_counts=dict(projection.calibration_summary.observed_outcome_counts),
        updated_at=result.updated_at,
    )
    projection.updated_at = result.updated_at
    projection.sync_lists()


def append_outcome_comparison(
    projection: BrainCounterfactualRehearsalProjection,
    comparison: BrainActionOutcomeComparisonRecord,
    *,
    max_recent_comparisons: int = 24,
) -> None:
    """Append one rehearsal-vs-observation comparison into the projection."""
    projection.recent_comparisons = [
        record
        for record in projection.recent_comparisons
        if record.comparison_id != comparison.comparison_id
    ]
    projection.recent_comparisons.append(comparison)
    projection.recent_comparisons = sorted(
        projection.recent_comparisons,
        key=lambda record: record.updated_at,
        reverse=True,
    )[:max_recent_comparisons]
    projection.updated_at = comparison.updated_at
    projection.calibration_summary = BrainCounterfactualCalibrationSummary(
        requested_count=projection.calibration_summary.requested_count,
        completed_count=projection.calibration_summary.completed_count,
        skipped_count=projection.calibration_summary.skipped_count,
        comparison_count=projection.calibration_summary.comparison_count + 1,
        recommendation_counts=dict(projection.calibration_summary.recommendation_counts),
        calibration_bucket_counts=dict(projection.calibration_summary.calibration_bucket_counts),
        risk_code_counts=dict(projection.calibration_summary.risk_code_counts),
        observed_outcome_counts=dict(projection.calibration_summary.observed_outcome_counts),
        updated_at=comparison.updated_at,
    )
    projection.sync_lists()


def build_outcome_comparison(
    *,
    rehearsal_result: BrainActionRehearsalResult,
    trigger_event: BrainEventRecord,
    observed_outcome_kind: str,
) -> BrainActionOutcomeComparisonRecord:
    """Build one observed-vs-predicted comparison for a landed action outcome."""
    probability = max(0.0, min(float(rehearsal_result.predicted_success_probability), 1.0))
    if observed_outcome_kind == BrainObservedActionOutcomeKind.PREVIEW_ONLY.value:
        calibration_bucket = BrainCalibrationBucket.NOT_CALIBRATED.value
        summary = "Preview-only outcome does not count as real-world rehearsal calibration."
    else:
        predicted_success = probability >= 0.5
        observed_success = observed_outcome_kind == BrainObservedActionOutcomeKind.SUCCESS.value
        if predicted_success == observed_success:
            calibration_bucket = BrainCalibrationBucket.ALIGNED.value
            summary = "Observed action outcome aligned with the rehearsed expectation."
        elif predicted_success:
            calibration_bucket = BrainCalibrationBucket.OVERCONFIDENT.value
            summary = "Observed action outcome underperformed the rehearsed expectation."
        else:
            calibration_bucket = BrainCalibrationBucket.UNDERCONFIDENT.value
            summary = "Observed action outcome outperformed the rehearsed expectation."
    comparison_id = _stable_id(
        "comparison",
        (
            f"blink:comparison:{rehearsal_result.rehearsal_id}:"
            f"{trigger_event.event_id}:{observed_outcome_kind}"
        ),
    )
    return BrainActionOutcomeComparisonRecord(
        comparison_id=comparison_id,
        rehearsal_id=rehearsal_result.rehearsal_id,
        goal_id=rehearsal_result.goal_id,
        commitment_id=rehearsal_result.commitment_id,
        plan_proposal_id=rehearsal_result.plan_proposal_id,
        step_index=rehearsal_result.step_index,
        candidate_action_id=rehearsal_result.candidate_action_id,
        observed_outcome_kind=observed_outcome_kind,
        predicted_success_probability=probability,
        confidence_band=rehearsal_result.confidence_band,
        decision_recommendation=rehearsal_result.decision_recommendation,
        calibration_bucket=calibration_bucket,
        comparison_summary=summary,
        observed_event_id=trigger_event.event_id,
        supporting_event_ids=_sorted_unique(
            [*rehearsal_result.supporting_event_ids, trigger_event.event_id]
        ),
        risk_codes=list(rehearsal_result.risk_codes),
        details={"trigger_event_type": trigger_event.event_type},
        compared_at=trigger_event.ts,
        updated_at=trigger_event.ts,
    )


class BrainCounterfactualRehearsalEngine:
    """Deterministic rehearsal engine for robot-head embodied planning steps."""

    def __init__(
        self,
        *,
        store: Any,
        session_resolver: Any,
        presence_scope_key: str,
        action_engine: EmbodiedActionEngine,
        context_surface_builder: BrainContextSurfaceBuilder,
        context_selector: BrainContextSelector | None,
        language: Language,
    ):
        """Initialize the bounded rehearsal engine."""
        self._store = store
        self._session_resolver = session_resolver
        self._presence_scope_key = presence_scope_key
        self._action_engine = action_engine
        self._context_surface_builder = context_surface_builder
        self._context_selector = context_selector or BrainContextSelector()
        self._language = language

    @property
    def action_engine(self) -> EmbodiedActionEngine:
        """Expose the bounded action engine used by rehearsal."""
        return self._action_engine

    async def rehearse_plan_proposal(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        proposal: BrainPlanProposal,
        request_kind: str,
    ) -> BrainActionRehearsalResult | None:
        """Run one bounded rehearsal for the first rehearseable robot-head step."""
        robot_step = self._first_robot_head_step(proposal)
        if robot_step is None:
            return None
        step_index, action_definition = robot_step
        request, request_event = self._append_rehearsal_request(
            goal=goal,
            commitment=commitment,
            proposal=proposal,
            step_index=step_index,
            action_definition=action_definition,
            request_kind=request_kind,
        )
        if not action_definition.rehearsal_required:
            result = BrainActionRehearsalResult(
                rehearsal_id=request.rehearsal_id,
                goal_id=request.goal_id,
                commitment_id=request.commitment_id,
                plan_proposal_id=request.plan_proposal_id,
                step_index=request.step_index,
                candidate_action_id=request.candidate_action_id,
                fallback_action_ids=list(request.fallback_action_ids),
                rehearsal_kind=request.rehearsal_kind,
                simulated_backend=request.simulated_backend,
                decision_recommendation=BrainRehearsalDecisionRecommendation.PROCEED.value,
                confidence_band=BrainPredictionConfidenceBand.HIGH.value,
                predicted_success_probability=0.85,
                supporting_prediction_ids=list(request.supporting_prediction_ids),
                supporting_event_ids=list(request.supporting_event_ids),
                packet_digest=dict(request.details.get("packet_digest", {})),
                skipped=True,
                summary=f"Skipped rehearsal for {action_definition.action_id}; explicit rehearsal is not required.",
                completed_at=request_event.ts,
                details={
                    "skip_reason": "rehearsal_not_required",
                    "request_kind": request_kind,
                },
                updated_at=request_event.ts,
            )
            self._append_rehearsal_result(
                event_type=BrainEventType.ACTION_REHEARSAL_SKIPPED,
                result=result,
                request_event=request_event,
            )
            return result

        status_result = await self._action_engine.status()
        status = status_result.status
        surface = self._context_surface_builder.build(
            latest_user_text=self._rehearsal_query_text(goal=goal, proposal=proposal, action_id=action_definition.action_id),
            task=BrainContextTask.SIM_REHEARSAL,
            include_historical_claims=True,
        )
        compiled = compile_context_packet_from_surface(
            snapshot=surface,
            latest_user_text=self._rehearsal_query_text(
                goal=goal,
                proposal=proposal,
                action_id=action_definition.action_id,
            ),
            task=BrainContextTask.SIM_REHEARSAL,
            language=self._language,
            context_selector=self._context_selector,
        )
        packet_digest = build_context_packet_digest(
            packet_traces={
                BrainContextTask.SIM_REHEARSAL.value: (
                    compiled.packet_trace.as_dict() if compiled.packet_trace is not None else None
                )
            }
        ).get(BrainContextTask.SIM_REHEARSAL.value, {})
        supporting_prediction_ids = _sorted_unique(
            [
                *request.supporting_prediction_ids,
                *(
                    record.prediction_id
                    for record in surface.predictive_world_model.active_predictions[:4]
                    if record.action_id == action_definition.action_id
                    or record.subject_id in {action_definition.action_id, proposal.plan_proposal_id}
                ),
            ]
        )
        supporting_event_ids = _sorted_unique(
            [*request.supporting_event_ids, *_trace_event_ids(compiled.packet_trace)]
        )
        fallback_action_ids = list(request.fallback_action_ids)

        candidate_evaluation = self._evaluate_action(
            rehearsal_id=request.rehearsal_id,
            action_definition=action_definition,
            fallback_action_ids=fallback_action_ids,
            status=status,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=supporting_event_ids,
        )
        wait_evaluation = self._wait_evaluation(
            request=request,
            action_definition=action_definition,
            status=status,
            fallback_action_ids=fallback_action_ids,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=supporting_event_ids,
            body_details=surface.body.details,
            packet_digest=packet_digest,
        )
        fallback_evaluations = [
            self._evaluate_action(
                rehearsal_id=request.rehearsal_id,
                action_definition=self._action_engine.library.get(action_id),
                fallback_action_ids=fallback_action_ids,
                status=status,
                supporting_prediction_ids=supporting_prediction_ids,
                supporting_event_ids=supporting_event_ids,
                rehearsal_kind=BrainCounterfactualRehearsalKind.FALLBACK_ACTION.value,
            )
            for action_id in fallback_action_ids[:1]
            if action_id != action_definition.action_id
        ]
        evaluations = [candidate_evaluation, wait_evaluation, *fallback_evaluations]
        selected = candidate_evaluation
        if (
            candidate_evaluation.decision_recommendation
            != BrainRehearsalDecisionRecommendation.PROCEED.value
            and fallback_evaluations
            and fallback_evaluations[0].decision_recommendation
            in {
                BrainRehearsalDecisionRecommendation.PROCEED.value,
                BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value,
            }
        ):
            fallback_evaluation = fallback_evaluations[0]
            selected = BrainCounterfactualEvaluationRecord(
                evaluation_id=_stable_id(
                    "evaluation",
                    f"blink:evaluation:{request.rehearsal_id}:repair:{fallback_evaluation.candidate_action_id}",
                ),
                rehearsal_id=request.rehearsal_id,
                candidate_action_id=action_definition.action_id,
                rehearsal_kind=BrainCounterfactualRehearsalKind.FALLBACK_ACTION.value,
                simulated_backend=fallback_evaluation.simulated_backend,
                expected_preconditions=list(fallback_evaluation.expected_preconditions),
                expected_effects=list(fallback_evaluation.expected_effects),
                predicted_success_probability=0.30,
                confidence_band=_confidence_band(0.30),
                risk_codes=_sorted_unique(
                    [*candidate_evaluation.risk_codes, "safer_fallback_available"]
                ),
                fallback_action_ids=list(fallback_action_ids),
                decision_recommendation=BrainRehearsalDecisionRecommendation.REPAIR.value,
                summary=(
                    f"Prefer repairing the plan with fallback action {fallback_evaluation.candidate_action_id} "
                    f"instead of dispatching {action_definition.action_id} directly."
                ),
                supporting_prediction_ids=supporting_prediction_ids,
                supporting_event_ids=supporting_event_ids,
                details={"fallback_action_id": fallback_evaluation.candidate_action_id},
                updated_at=request_event.ts,
            )
        elif candidate_evaluation.decision_recommendation == BrainRehearsalDecisionRecommendation.WAIT.value:
            selected = wait_evaluation

        result = BrainActionRehearsalResult(
            rehearsal_id=request.rehearsal_id,
            goal_id=request.goal_id,
            commitment_id=request.commitment_id,
            plan_proposal_id=request.plan_proposal_id,
            step_index=request.step_index,
            candidate_action_id=request.candidate_action_id,
            fallback_action_ids=list(fallback_action_ids),
            rehearsal_kind=request.rehearsal_kind,
            simulated_backend=request.simulated_backend,
            expected_preconditions=list(selected.expected_preconditions),
            expected_effects=list(selected.expected_effects),
            predicted_success_probability=selected.predicted_success_probability,
            confidence_band=selected.confidence_band,
            risk_codes=list(selected.risk_codes),
            decision_recommendation=selected.decision_recommendation,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=supporting_event_ids,
            evaluations=evaluations,
            selected_evaluation_id=selected.evaluation_id,
            packet_digest=packet_digest,
            skipped=False,
            summary=selected.summary,
            completed_at=request_event.ts,
            details={
                "request_kind": request_kind,
                "packet_digest": packet_digest,
                "selected_rehearsal_kind": selected.rehearsal_kind,
            },
            updated_at=request_event.ts,
        )
        self._append_rehearsal_result(
            event_type=BrainEventType.ACTION_REHEARSAL_COMPLETED,
            result=result,
            request_event=request_event,
        )
        return result

    def _append_rehearsal_request(
        self,
        *,
        goal: BrainGoal,
        commitment: BrainCommitmentRecord | None,
        proposal: BrainPlanProposal,
        step_index: int,
        action_definition: EmbodiedActionDefinition,
        request_kind: str,
    ) -> tuple[BrainActionRehearsalRequest, BrainEventRecord]:
        session_ids = self._session_resolver()
        supporting_prediction_ids = _sorted_unique(
            [
                record.prediction_id
                for record in self._store.get_predictive_world_model_projection(
                    scope_key=session_ids.thread_id
                ).active_predictions[:4]
                if record.action_id == action_definition.action_id
                or record.subject_id in {action_definition.action_id, proposal.plan_proposal_id}
            ]
        )
        rehearsal_id = _stable_id(
            "rehearsal",
            (
                f"blink:rehearsal:{proposal.plan_proposal_id}:{goal.goal_id}:"
                f"{step_index}:{action_definition.action_id}"
            ),
        )
        fallback_action_ids = self._fallback_action_ids(action_definition)
        request = BrainActionRehearsalRequest(
            rehearsal_id=rehearsal_id,
            goal_id=goal.goal_id,
            commitment_id=commitment.commitment_id if commitment is not None else None,
            plan_proposal_id=proposal.plan_proposal_id,
            step_index=step_index,
            candidate_action_id=action_definition.action_id,
            fallback_action_ids=fallback_action_ids,
            rehearsal_kind=BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
            simulated_backend="robot_head_simulation",
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=[],
            requested_at=_utc_now(),
            details={"request_kind": request_kind},
            updated_at=_utc_now(),
        )
        event = self._store.append_brain_event(
            event_type=BrainEventType.ACTION_REHEARSAL_REQUESTED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="counterfactuals",
            payload={
                "rehearsal_request": request.as_dict(),
                "rehearsal_id": request.rehearsal_id,
                "goal_id": request.goal_id,
                "commitment_id": request.commitment_id,
                "plan_proposal_id": request.plan_proposal_id,
                "step_index": request.step_index,
                "candidate_action_id": request.candidate_action_id,
                "request_kind": request_kind,
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=goal.goal_id,
        )
        request = BrainActionRehearsalRequest.from_dict(
            {**request.as_dict(), "requested_at": event.ts, "updated_at": event.ts}
        ) or request
        return request, event

    def _append_rehearsal_result(
        self,
        *,
        event_type: str,
        result: BrainActionRehearsalResult,
        request_event: BrainEventRecord,
    ) -> BrainEventRecord:
        return self._store.append_brain_event(
            event_type=event_type,
            agent_id=request_event.agent_id,
            user_id=request_event.user_id,
            session_id=request_event.session_id,
            thread_id=request_event.thread_id,
            source="counterfactuals",
            payload={
                "rehearsal_result": result.as_dict(),
                "rehearsal_id": result.rehearsal_id,
                "goal_id": result.goal_id,
                "commitment_id": result.commitment_id,
                "plan_proposal_id": result.plan_proposal_id,
                "step_index": result.step_index,
                "candidate_action_id": result.candidate_action_id,
                "decision_recommendation": result.decision_recommendation,
                "presence_scope_key": self._presence_scope_key,
            },
            correlation_id=result.goal_id,
            causal_parent_id=request_event.event_id,
            confidence=result.predicted_success_probability,
            ts=request_event.ts,
        )

    def _first_robot_head_step(
        self,
        proposal: BrainPlanProposal,
    ) -> tuple[int, EmbodiedActionDefinition] | None:
        from blink.brain.actions import action_id_for_capability

        for step_index, step in enumerate(proposal.steps):
            try:
                action_id = action_id_for_capability(step.capability_id)
            except KeyError:
                continue
            return step_index, self._action_engine.library.get(action_id)
        return None

    def _fallback_action_ids(
        self,
        action_definition: EmbodiedActionDefinition,
    ) -> list[str]:
        if action_definition.action_id == "cmd_return_neutral":
            return []
        return ["cmd_return_neutral"]

    def _rehearsal_query_text(
        self,
        *,
        goal: BrainGoal,
        proposal: BrainPlanProposal,
        action_id: str,
    ) -> str:
        return (
            f"Rehearse embodied action {action_id} for goal {goal.title}. "
            f"Proposal summary: {proposal.summary}"
        )

    def _evaluate_action(
        self,
        *,
        rehearsal_id: str,
        action_definition: EmbodiedActionDefinition,
        fallback_action_ids: list[str],
        status: RobotHeadDriverStatus | None,
        supporting_prediction_ids: list[str],
        supporting_event_ids: list[str],
        rehearsal_kind: str = BrainCounterfactualRehearsalKind.EMBODIED_ACTION.value,
    ) -> BrainCounterfactualEvaluationRecord:
        expected_preconditions = list(action_definition.preconditions)
        expected_effects = [
            f"{step.command_type}:{step.name or 'default'}"
            for step in action_definition.controller_plan
        ]
        risk_codes: list[str] = []
        recommendation = BrainRehearsalDecisionRecommendation.PROCEED.value
        probability = 0.85
        summary = f"Simulated {action_definition.action_id} completed without warnings."
        details: dict[str, Any] = {}
        if action_definition.sensitivity != "safe":
            risk_codes.append("unsafe_action_sensitivity")
            recommendation = BrainRehearsalDecisionRecommendation.ABORT.value
            probability = 0.10
            summary = f"Abort {action_definition.action_id}; action sensitivity exceeds the safe embodied tranche."
        elif status is None:
            risk_codes.append("robot_head_status_unavailable")
            recommendation = BrainRehearsalDecisionRecommendation.WAIT.value
            probability = 0.45
            summary = f"Wait before dispatching {action_definition.action_id}; robot-head status is unavailable."
        elif action_definition.requires_live_arm and not status.armed:
            risk_codes.append("robot_head_unarmed")
            recommendation = BrainRehearsalDecisionRecommendation.WAIT.value
            probability = 0.45
            summary = f"Wait before dispatching {action_definition.action_id}; the robot head is not armed."
        elif not status.available:
            risk_codes.append("robot_head_unavailable")
            recommendation = BrainRehearsalDecisionRecommendation.WAIT.value
            probability = 0.45
            summary = f"Wait before dispatching {action_definition.action_id}; the robot head is unavailable."
        elif status.preview_fallback and not action_definition.preview_ok:
            risk_codes.append("robot_head_preview_not_allowed")
            recommendation = BrainRehearsalDecisionRecommendation.ABORT.value
            probability = 0.10
            summary = (
                f"Abort {action_definition.action_id}; the current robot-head state can only preview, "
                "but preview fallback is not allowed for this action."
            )
        else:
            try:
                simulation = self._simulate_action(action_definition)
            except ValueError as exc:
                risk_codes.append("simulation_plan_invalid")
                recommendation = BrainRehearsalDecisionRecommendation.ABORT.value
                probability = 0.10
                summary = f"Abort {action_definition.action_id}; deterministic simulation failed: {exc}"
                details["simulation_error"] = str(exc)
            else:
                risk_codes.extend(str(item).strip() for item in simulation["risk_codes"])
                details.update(simulation["details"])
                summary = str(simulation["summary"])
                if simulation["preview_only"] or simulation["warnings"]:
                    recommendation = BrainRehearsalDecisionRecommendation.PROCEED_CAUTIOUSLY.value
                    probability = 0.65
                else:
                    recommendation = BrainRehearsalDecisionRecommendation.PROCEED.value
                    probability = 0.85
        return BrainCounterfactualEvaluationRecord(
            evaluation_id=_stable_id(
                "evaluation",
                f"blink:evaluation:{rehearsal_id}:{rehearsal_kind}:{action_definition.action_id}",
            ),
            rehearsal_id=rehearsal_id,
            candidate_action_id=action_definition.action_id,
            rehearsal_kind=rehearsal_kind,
            simulated_backend="robot_head_simulation",
            expected_preconditions=expected_preconditions,
            expected_effects=expected_effects,
            predicted_success_probability=probability,
            confidence_band=_confidence_band(probability),
            risk_codes=_sorted_unique(risk_codes),
            fallback_action_ids=list(fallback_action_ids),
            decision_recommendation=recommendation,
            summary=summary,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=supporting_event_ids,
            details=details,
            updated_at=_utc_now(),
        )

    def _wait_evaluation(
        self,
        *,
        request: BrainActionRehearsalRequest,
        action_definition: EmbodiedActionDefinition,
        status: RobotHeadDriverStatus | None,
        fallback_action_ids: list[str],
        supporting_prediction_ids: list[str],
        supporting_event_ids: list[str],
        body_details: dict[str, Any],
        packet_digest: dict[str, Any],
    ) -> BrainCounterfactualEvaluationRecord:
        risk_codes = []
        expected_preconditions = list(action_definition.preconditions)
        if status is None:
            risk_codes.append("robot_head_status_unavailable")
        elif not status.available:
            risk_codes.append("robot_head_unavailable")
        elif action_definition.requires_live_arm and not status.armed:
            risk_codes.append("robot_head_unarmed")
        if str(body_details.get("sensor_health_reason", "")).strip():
            risk_codes.append("scene_health_uncertain")
        if not risk_codes:
            risk_codes.append("safer_wait_path")
        summary = (
            f"Wait before dispatching {action_definition.action_id}; preserve the current embodied state "
            "until the robot head and scene context remain healthier."
        )
        return BrainCounterfactualEvaluationRecord(
            evaluation_id=_stable_id(
                "evaluation",
                f"blink:evaluation:{request.rehearsal_id}:wait:{action_definition.action_id}",
            ),
            rehearsal_id=request.rehearsal_id,
            candidate_action_id=action_definition.action_id,
            rehearsal_kind=BrainCounterfactualRehearsalKind.WAIT_ALTERNATIVE.value,
            simulated_backend=request.simulated_backend,
            expected_preconditions=expected_preconditions,
            expected_effects=["defer_dispatch", "preserve_current_state"],
            predicted_success_probability=0.45,
            confidence_band=_confidence_band(0.45),
            risk_codes=_sorted_unique(risk_codes),
            fallback_action_ids=list(fallback_action_ids),
            decision_recommendation=BrainRehearsalDecisionRecommendation.WAIT.value,
            summary=summary,
            supporting_prediction_ids=supporting_prediction_ids,
            supporting_event_ids=supporting_event_ids,
            details={"packet_digest": packet_digest},
            updated_at=_utc_now(),
        )

    def _simulate_action(
        self,
        action_definition: EmbodiedActionDefinition,
    ) -> dict[str, Any]:
        controller = self._action_engine.controller
        hardware_profile = load_robot_head_live_hardware_profile()
        scenario = load_robot_head_simulation_scenario()
        simulator = RobotHeadSimulator(
            hardware_profile=hardware_profile,
            scenario=scenario,
        )
        warnings: list[str] = []
        summary = ""
        preview_only = False
        details: dict[str, Any] = {"steps": []}
        for step in action_definition.controller_plan:
            command = RobotHeadCommand(
                command_type=step.command_type,
                state=step.name if step.command_type == "set_state" else None,
                motif=step.name if step.command_type == "run_motif" else None,
                source="rehearsal",
                reason=f"Counterfactual rehearsal for {action_definition.action_id}",
            )
            plan = controller._build_plan(command)  # noqa: SLF001 - bounded internal reuse
            outcome = simulator.execute_plan(plan)
            preview_only = preview_only or bool(outcome.preview_only)
            warnings.extend(outcome.warnings)
            summary = outcome.summary
            details["steps"].append(
                {
                    "resolved_name": plan.resolved_name,
                    "command_type": command.command_type,
                    "preview_only": outcome.preview_only,
                    "warnings": list(outcome.warnings),
                    "metadata": dict(outcome.metadata),
                }
            )
        return {
            "preview_only": preview_only,
            "warnings": _sorted_unique(warnings),
            "summary": summary or f"Simulated {action_definition.action_id}.",
            "risk_codes": _sorted_unique(
                [
                    "simulation_warning" if warnings else None,
                    "simulation_preview_only" if preview_only else None,
                ]
            ),
            "details": details,
        }


def _trace_event_ids(trace) -> list[str]:
    if trace is None:
        return []
    event_ids: list[str] = []
    for item in list(trace.selected_items) + list(trace.selected_anchors):
        provenance = dict(item.provenance)
        for event_id in provenance.get("source_event_ids", []) or []:
            text = _optional_text(event_id)
            if text is not None:
                event_ids.append(text)
    return _sorted_unique(event_ids)


__all__ = [
    "BrainCounterfactualRehearsalEngine",
    "append_outcome_comparison",
    "append_rehearsal_request",
    "append_rehearsal_result",
    "build_outcome_comparison",
    "counterfactual_event_types",
    "is_counterfactual_event_type",
]

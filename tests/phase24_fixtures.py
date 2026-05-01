from __future__ import annotations

from pathlib import Path

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.cards import (
    BrainAdapterFamily,
    BrainAdapterPromotionState,
    build_adapter_card,
)
from blink.brain.evals.adapter_promotion import (
    build_adapter_promotion_decision,
    build_embodied_policy_benchmark_comparison_report,
    with_card_benchmark_summary,
    write_adapter_benchmark_report,
)
from blink.brain.evals.embodied_metrics import BrainEmbodiedEvalMetricRow
from blink.brain.events import BrainEventType


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-04-22T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def make_metric_row(
    *,
    run_id: str,
    scenario_id: str,
    scenario_family: str,
    profile_id: str,
    matrix_index: int,
    embodied_policy_backend_id: str,
    task_success: bool,
    safety_success: bool = True,
    review_floor_count: int = 0,
    operator_intervention_count: int = 0,
    recovery_count: int = 0,
    preview_only: bool = False,
    trace_status: str | None = None,
    mismatch_codes: tuple[str, ...] = (),
    repair_codes: tuple[str, ...] = (),
) -> BrainEmbodiedEvalMetricRow:
    return BrainEmbodiedEvalMetricRow(
        run_id=run_id,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version="v1",
        profile_id=profile_id,
        matrix_index=matrix_index,
        execution_backend="simulation",
        perception_backend_id="local_perception",
        world_model_backend_id="local_world_model",
        embodied_policy_backend_id=embodied_policy_backend_id,
        task_success=task_success,
        safety_success=safety_success,
        preview_only=preview_only,
        operator_intervention_count=operator_intervention_count,
        recovery_count=recovery_count,
        step_count=1,
        review_floor_count=review_floor_count,
        skill_reuse_detected=False,
        trace_status=trace_status,
        mismatch_codes=tuple(sorted(set(mismatch_codes))),
        repair_codes=tuple(sorted(set(repair_codes))),
        artifact_paths={},
    )


def build_candidate_card(
    *,
    backend_id: str = "candidate_robot_head_policy",
    backend_version: str = "v2",
    promotion_state: str = BrainAdapterPromotionState.EXPERIMENTAL.value,
    approved_target_families: tuple[str, ...] = (),
    updated_at: str | None = None,
):
    descriptor = BrainAdapterDescriptor(
        backend_id=backend_id,
        backend_version=backend_version,
        capabilities=("status", "embodied_action_execution"),
        degraded_mode_id="preview_only",
        default_timeout_ms=5000,
    )
    return build_adapter_card(
        adapter_family=BrainAdapterFamily.EMBODIED_POLICY.value,
        descriptor=descriptor,
        promotion_state=promotion_state,
        supported_task_families=("robot_head_embodied_execution",),
        approved_target_families=approved_target_families,
        safety_constraints=("simulation_backed_dispatch_first",),
        updated_at=updated_at or _ts(1),
        details={"governance_only": True, "candidate": True},
    )


def seed_phase24_adapter_governance(
    *,
    store,
    session_ids,
    output_dir: Path,
):
    store.ensure_default_adapter_cards(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        updated_at=_ts(1),
    )
    candidate_card = build_candidate_card(updated_at=_ts(2))
    store.append_brain_event(
        event_type=BrainEventType.ADAPTER_CARD_UPSERTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        payload={
            "scope_key": session_ids.thread_id,
            "adapter_card": candidate_card.as_dict(),
        },
        ts=_ts(2),
    )
    report = build_embodied_policy_benchmark_comparison_report(
        [
            make_metric_row(
                run_id="incumbent-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=False,
                review_floor_count=1,
            ),
            make_metric_row(
                run_id="candidate-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="candidate",
                matrix_index=1,
                embodied_policy_backend_id="candidate_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="incumbent-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="candidate-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
                profile_id="candidate",
                matrix_index=1,
                embodied_policy_backend_id="candidate_robot_head_policy",
                task_success=True,
            ),
        ],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_single_step",),
        smoke_suite_green=True,
        updated_at=_ts(3),
    )
    artifact_paths = write_adapter_benchmark_report(
        report=report,
        output_dir=output_dir / "brain_adapter_benchmarks",
    )
    report = build_embodied_policy_benchmark_comparison_report(
        [
            make_metric_row(
                run_id="incumbent-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=False,
                review_floor_count=1,
            ),
            make_metric_row(
                run_id="candidate-single",
                scenario_id="robot_head_look_left_compare",
                scenario_family="robot_head_single_step",
                profile_id="candidate",
                matrix_index=1,
                embodied_policy_backend_id="candidate_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="incumbent-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
                profile_id="incumbent",
                matrix_index=0,
                embodied_policy_backend_id="local_robot_head_policy",
                task_success=True,
            ),
            make_metric_row(
                run_id="candidate-multi",
                scenario_id="robot_head_wave_compare",
                scenario_family="robot_head_multi_step",
                profile_id="candidate",
                matrix_index=1,
                embodied_policy_backend_id="candidate_robot_head_policy",
                task_success=True,
            ),
        ],
        incumbent_backend_id="local_robot_head_policy",
        candidate_backend_id="candidate_robot_head_policy",
        target_families=("robot_head_single_step",),
        smoke_suite_green=True,
        updated_at=_ts(3),
        artifact_paths=artifact_paths,
    )
    candidate_with_report = with_card_benchmark_summary(candidate_card, report)
    store.append_brain_event(
        event_type=BrainEventType.ADAPTER_BENCHMARK_REPORTED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        payload={
            "scope_key": session_ids.thread_id,
            "benchmark_report": report.as_dict(),
            "adapter_card": candidate_with_report.as_dict(),
        },
        ts=_ts(3),
    )
    promote = build_adapter_promotion_decision(
        card=candidate_with_report,
        outcome="promote",
        report=report,
        updated_at=_ts(4),
    )
    promoted_card = build_candidate_card(
        promotion_state=promote.to_state,
        approved_target_families=tuple(promote.approved_target_families),
        updated_at=_ts(4),
    )
    promoted_card = with_card_benchmark_summary(promoted_card, report)
    store.append_brain_event(
        event_type=BrainEventType.ADAPTER_PROMOTION_DECIDED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase24_test",
        payload={
            "scope_key": session_ids.thread_id,
            "promotion_decision": promote.as_dict(),
            "adapter_card": promoted_card.as_dict(),
        },
        ts=_ts(4),
    )
    return {
        "candidate_card": promoted_card,
        "report": report,
        "decision": promote,
        "artifact_paths": artifact_paths,
    }


from __future__ import annotations

import json

from blink.brain.adapters import BrainAdapterDescriptor
from blink.brain.adapters.cards import (
    BrainAdapterFamily,
    BrainAdapterPromotionState,
    build_adapter_card,
)
from blink.brain.evals import (
    PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS,
    BrainPerceptionWorldModelMetricRow,
    build_perception_world_model_benchmark_comparison_report,
)
from blink.brain.evals.adapter_promotion import (
    BrainAdapterGovernanceProjection,
    append_adapter_benchmark_report,
    append_adapter_card,
    with_card_benchmark_summary,
)
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest


def _row(
    *,
    run_id: str,
    scenario_id: str,
    scenario_family: str,
    adapter_family: str,
    backend_id: str,
    profile_id: str,
    matrix_index: int,
    spatial: bool = True,
    affordance: bool = True,
    temporal: bool = True,
    abstention: bool = True,
    success_detection: bool = True,
    recovery: bool = True,
    occlusion: bool = False,
    insufficient: bool = False,
    mismatch_codes: tuple[str, ...] = (),
    recovery_codes: tuple[str, ...] = (),
) -> BrainPerceptionWorldModelMetricRow:
    return BrainPerceptionWorldModelMetricRow(
        run_id=run_id,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        scenario_version="v1",
        profile_id=profile_id,
        matrix_index=matrix_index,
        evaluation_backend="deterministic_fixture",
        adapter_family=adapter_family,
        backend_id=backend_id,
        backend_version="v1",
        spatial_reasoning_success=spatial,
        affordance_prediction_success=affordance,
        temporal_consistency_success=temporal,
        abstention_success=abstention,
        success_detection_success=success_detection,
        mismatch_recovery_success=recovery,
        occlusion_present=occlusion,
        insufficient_evidence_present=insufficient,
        mismatch_codes=mismatch_codes,
        recovery_codes=recovery_codes,
        reason_codes=("fixture",),
        updated_at="2026-04-23T00:00:00+00:00",
    )


def _passing_perception_rows() -> list[BrainPerceptionWorldModelMetricRow]:
    family = BrainAdapterFamily.PERCEPTION.value
    return [
        _row(
            run_id="inc-layout",
            scenario_id="scene_layout_compare",
            scenario_family="spatial_layout",
            adapter_family=family,
            backend_id="local_perception",
            profile_id="incumbent",
            matrix_index=0,
            spatial=False,
        ),
        _row(
            run_id="cand-layout",
            scenario_id="scene_layout_compare",
            scenario_family="spatial_layout",
            adapter_family=family,
            backend_id="candidate_perception",
            profile_id="candidate",
            matrix_index=1,
            spatial=True,
        ),
        _row(
            run_id="inc-occlusion",
            scenario_id="occluded_object_compare",
            scenario_family="occlusion_abstention",
            adapter_family=family,
            backend_id="local_perception",
            profile_id="incumbent",
            matrix_index=0,
            abstention=False,
            occlusion=True,
        ),
        _row(
            run_id="cand-occlusion",
            scenario_id="occluded_object_compare",
            scenario_family="occlusion_abstention",
            adapter_family=family,
            backend_id="candidate_perception",
            profile_id="candidate",
            matrix_index=1,
            abstention=True,
            occlusion=True,
            recovery_codes=("abstained_under_occlusion",),
        ),
    ]


def test_perception_world_model_report_is_deterministic_and_json_safe():
    report_a = build_perception_world_model_benchmark_comparison_report(
        _passing_perception_rows(),
        adapter_family=BrainAdapterFamily.PERCEPTION,
        incumbent_backend_id="local_perception",
        candidate_backend_id="candidate_perception",
        target_families=("spatial_layout",),
        updated_at="2026-04-23T00:00:01+00:00",
    )
    report_b = build_perception_world_model_benchmark_comparison_report(
        list(reversed(_passing_perception_rows())),
        adapter_family=BrainAdapterFamily.PERCEPTION,
        incumbent_backend_id="local_perception",
        candidate_backend_id="candidate_perception",
        target_families=("spatial_layout",),
        updated_at="2026-04-23T00:00:01+00:00",
    )

    assert report_a.as_dict() == report_b.as_dict()
    assert report_a.adapter_family == "perception"
    assert report_a.scenario_count == 2
    assert report_a.benchmark_passed is True
    assert report_a.details["evidence_metric_names"] == list(
        PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS
    )
    json.dumps(report_a.as_dict(), sort_keys=True)
    assert (
        BrainPerceptionWorldModelMetricRow.from_dict(_passing_perception_rows()[0].as_dict())
        == _passing_perception_rows()[0]
    )


def test_adapter_cards_and_sim_to_real_digest_include_evidence_summaries():
    report = build_perception_world_model_benchmark_comparison_report(
        _passing_perception_rows(),
        adapter_family=BrainAdapterFamily.PERCEPTION.value,
        incumbent_backend_id="local_perception",
        candidate_backend_id="candidate_perception",
        target_families=("spatial_layout",),
        updated_at="2026-04-23T00:00:01+00:00",
    )
    card = build_adapter_card(
        adapter_family=BrainAdapterFamily.PERCEPTION.value,
        descriptor=BrainAdapterDescriptor(
            backend_id="candidate_perception",
            backend_version="v1",
            capabilities=("presence_detection", "scene_enrichment"),
            degraded_mode_id="unavailable_result",
            default_timeout_ms=3000,
        ),
        promotion_state=BrainAdapterPromotionState.SHADOW.value,
        supported_task_families=("presence_detection", "scene_enrichment"),
        approved_target_families=("spatial_layout",),
        updated_at="2026-04-23T00:00:00+00:00",
    )
    enriched = with_card_benchmark_summary(card, report)

    assert enriched.latest_benchmark_summary is not None
    summary_details = enriched.latest_benchmark_summary.details
    assert summary_details["evidence_metric_names"] == list(PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS)
    assert summary_details["abstention_case_count"] == 1
    assert summary_details["recovery_code_counts"] == {"abstained_under_occlusion": 1}

    projection = BrainAdapterGovernanceProjection(scope_key="thread-evidence")
    append_adapter_card(projection, enriched)
    append_adapter_benchmark_report(projection, report)
    digest = build_sim_to_real_digest(adapter_governance=projection)
    readiness = digest["readiness_reports"][0]

    assert readiness["adapter_family"] == "perception"
    assert readiness["details"]["evidence_metric_names"] == list(
        PERCEPTION_WORLD_MODEL_EVIDENCE_METRICS
    )
    assert readiness["details"]["abstention_case_count"] == 1
    assert readiness["governance_only"] is True


def test_abstention_and_mismatch_regressions_create_weak_family_summaries():
    family = BrainAdapterFamily.PERCEPTION.value
    rows = [
        _row(
            run_id="inc-occlusion",
            scenario_id="occluded_object_compare",
            scenario_family="occlusion_abstention",
            adapter_family=family,
            backend_id="local_perception",
            profile_id="incumbent",
            matrix_index=0,
            occlusion=True,
            abstention=True,
        ),
        _row(
            run_id="cand-occlusion",
            scenario_id="occluded_object_compare",
            scenario_family="occlusion_abstention",
            adapter_family=family,
            backend_id="candidate_perception",
            profile_id="candidate",
            matrix_index=1,
            occlusion=True,
            abstention=False,
            mismatch_codes=("failed_to_abstain",),
        ),
    ]
    report = build_perception_world_model_benchmark_comparison_report(
        rows,
        adapter_family=family,
        incumbent_backend_id="local_perception",
        candidate_backend_id="candidate_perception",
        target_families=("occlusion_abstention",),
        updated_at="2026-04-23T00:00:01+00:00",
    )

    assert report.benchmark_passed is False
    assert "abstention_regressed" in report.blocked_reason_codes
    assert "new_critical_failure_signature" in report.blocked_reason_codes
    assert report.details["mismatch_code_counts"] == {"failed_to_abstain": 1}
    assert report.details["abstention_case_count"] == 1
    assert report.weak_families[0].scenario_family == "occlusion_abstention"
    assert "mismatch_count_regressed" in report.weak_families[0].reason_codes


def test_world_model_temporal_and_success_detection_failures_are_reported():
    family = BrainAdapterFamily.WORLD_MODEL.value
    rows = [
        _row(
            run_id="inc-temporal",
            scenario_id="temporal_state_compare",
            scenario_family="temporal_consistency",
            adapter_family=family,
            backend_id="local_world_model",
            profile_id="incumbent",
            matrix_index=0,
            temporal=True,
            success_detection=True,
        ),
        _row(
            run_id="cand-temporal",
            scenario_id="temporal_state_compare",
            scenario_family="temporal_consistency",
            adapter_family=family,
            backend_id="candidate_world_model",
            profile_id="candidate",
            matrix_index=1,
            temporal=False,
            success_detection=False,
            recovery=False,
            mismatch_codes=("temporal_contradiction", "missed_success_detection"),
        ),
    ]
    report = build_perception_world_model_benchmark_comparison_report(
        rows,
        adapter_family=family,
        incumbent_backend_id="local_world_model",
        candidate_backend_id="candidate_world_model",
        target_families=("temporal_consistency",),
        updated_at="2026-04-23T00:00:01+00:00",
    )

    assert report.adapter_family == "world_model"
    assert report.benchmark_passed is False
    assert "temporal_consistency_regressed" in report.blocked_reason_codes
    assert "success_detection_regressed" in report.blocked_reason_codes
    assert report.details["mismatch_code_counts"] == {
        "missed_success_detection": 1,
        "temporal_contradiction": 1,
    }
    family_row = report.family_rows[0]
    assert family_row.details["metric_delta_points"]["temporal_consistency_delta_points"] < 0
    assert family_row.details["candidate_mismatch_count"] == 2

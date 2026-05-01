from __future__ import annotations

from pathlib import Path
from typing import Any

from blink.brain.evals.dataset_manifest import (
    BrainEpisodeDatasetManifest,
    build_episode_dataset_manifest,
)
from blink.brain.evals.episode_export import (
    BrainEpisodeActionSummary,
    BrainEpisodeOrigin,
    BrainEpisodeOutcomeSummary,
    BrainEpisodeRecord,
    BrainEpisodeSafetySummary,
)
from blink.brain.events import BrainEventType
from blink.brain.memory_v2 import (
    BrainProceduralSkillProjection,
    BrainProceduralSkillRecord,
    BrainProceduralSkillStatsRecord,
    BrainProceduralSkillStatus,
)
from blink.brain.memory_v2.skill_evidence import (
    BrainSkillEvidenceLedger,
    build_skill_evidence_ledger,
)
from blink.brain.memory_v2.skill_promotion import (
    BrainSkillGovernanceProjection,
    BrainSkillGovernanceStatus,
    build_skill_governance_projection,
)
from blink.brain.practice_director import BrainPracticeDirector, BrainPracticePlan
from blink.brain.store import BrainStore


def _ts(second: int) -> str:
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return f"2026-04-22T{hour:02d}:{minute:02d}:{second:02d}+00:00"


def make_episode(
    *,
    index: int,
    scenario_family: str,
    scenario_id: str | None = None,
    skill_ids: tuple[str, ...] = (),
    execution_backend: str = "simulation",
    embodied_policy_backend_id: str = "local_robot_head_policy",
    task_success: bool = True,
    operator_review_floored: bool = False,
    review_floor_count: int = 0,
    recovery_count: int = 0,
    safety_success: bool | None = None,
    calibration_bucket_counts: dict[str, int] | None = None,
    risk_codes: tuple[str, ...] = (),
    mismatch_codes: tuple[str, ...] = (),
    repair_codes: tuple[str, ...] = (),
) -> BrainEpisodeRecord:
    resolved_safety_success = (
        safety_success
        if isinstance(safety_success, bool)
        else not risk_codes and review_floor_count == 0
    )
    resolved_trace_status = "succeeded" if task_success else "failed"
    generated_at = _ts(index)
    return BrainEpisodeRecord(
        episode_id=f"episode-{index}",
        schema_version="brain_episode/v1",
        origin=BrainEpisodeOrigin.SIMULATION.value,
        scenario_id=scenario_id or f"{scenario_family}-{index}",
        scenario_family=scenario_family,
        scenario_version="v1",
        goal_id=f"goal-{index}",
        commitment_id=f"commitment-{index}",
        plan_proposal_id=f"proposal-{index}",
        skill_ids=tuple(skill_ids),
        execution_backend=execution_backend,
        embodied_policy_backend_id=embodied_policy_backend_id,
        source_event_ids=(f"evt-{index}",),
        action_summary=BrainEpisodeActionSummary(
            action_ids=(f"action-{index}",),
            trace_ids=(f"trace-{index}",),
            recovery_ids=((f"recovery-{index}",) if recovery_count > 0 else ()),
            trace_status=resolved_trace_status,
            mismatch_codes=tuple(sorted(set(mismatch_codes))),
            repair_codes=tuple(sorted(set(repair_codes))),
            execution_backend=execution_backend,
            preview_only=False,
            recovery_count=recovery_count,
        ),
        outcome_summary=BrainEpisodeOutcomeSummary(
            goal_status="completed" if task_success else "failed",
            planning_outcome="adopted",
            task_success=task_success,
            trace_status=resolved_trace_status,
            operator_review_floored=operator_review_floored,
            observed_outcome_counts={"success": 1 if task_success else 0, "failure": 0 if task_success else 1},
            calibration_bucket_counts=dict(calibration_bucket_counts or {"aligned": 1}),
        ),
        safety_summary=BrainEpisodeSafetySummary(
            safety_success=resolved_safety_success,
            review_floor_count=review_floor_count,
            operator_intervention_count=1 if operator_review_floored else 0,
            recovery_count=recovery_count,
            risk_codes=tuple(sorted(set(risk_codes))),
            mismatch_codes=tuple(sorted(set(mismatch_codes))),
            repair_codes=tuple(sorted(set(repair_codes))),
        ),
        started_at=generated_at,
        ended_at=generated_at,
        generated_at=generated_at,
        provenance_ids={"source_run_id": f"run-{index}"},
    )


def make_procedural_skills(
    *records: dict[str, Any],
) -> BrainProceduralSkillProjection:
    skills = sorted(
        [
        BrainProceduralSkillRecord(
            skill_id=str(record["skill_id"]),
            skill_family_key=str(record.get("skill_family_key") or record["skill_id"]),
            template_fingerprint=str(record.get("template_fingerprint") or f"tmpl-{record['skill_id']}"),
            scope_type="thread",
            scope_id="thread-1",
            title=str(record.get("title") or record["skill_id"]),
            purpose=str(record.get("purpose") or "Practice a bounded embodied step."),
            goal_family=str(record.get("goal_family") or "environment"),
            status=str(record.get("status") or BrainProceduralSkillStatus.ACTIVE.value),
            confidence=float(record.get("confidence", 0.8)),
            stats=BrainProceduralSkillStatsRecord(),
            created_at=str(record.get("created_at") or _ts(1)),
            updated_at=str(record.get("updated_at") or _ts(1)),
            retirement_reason=record.get("retirement_reason"),
        )
        for record in records
        ],
        key=lambda item: (item.goal_family, item.skill_id),
    )
    skill_counts: dict[str, int] = {}
    confidence_band_counts: dict[str, int] = {}
    for skill in skills:
        skill_counts[skill.status] = skill_counts.get(skill.status, 0) + 1
        if skill.confidence < 0.5:
            band = "low"
        elif skill.confidence < 0.8:
            band = "medium"
        else:
            band = "high"
        confidence_band_counts[band] = confidence_band_counts.get(band, 0) + 1
    return BrainProceduralSkillProjection(
        scope_type="thread",
        scope_id="thread-1",
        skill_counts=dict(sorted(skill_counts.items())),
        confidence_band_counts=dict(sorted(confidence_band_counts.items())),
        skills=skills,
        candidate_skill_ids=sorted(
            skill.skill_id
            for skill in skills
            if skill.status == BrainProceduralSkillStatus.CANDIDATE.value
        ),
        active_skill_ids=sorted(
            skill.skill_id
            for skill in skills
            if skill.status == BrainProceduralSkillStatus.ACTIVE.value
        ),
        superseded_skill_ids=sorted(
            skill.skill_id
            for skill in skills
            if skill.status == BrainProceduralSkillStatus.SUPERSEDED.value
        ),
        retired_skill_ids=sorted(
            skill.skill_id
            for skill in skills
            if skill.status == BrainProceduralSkillStatus.RETIRED.value
        ),
    )


def append_skill_governance_events(
    *,
    store: BrainStore,
    session_ids,
    scope_key: str,
    ledger: BrainSkillEvidenceLedger,
    governance: BrainSkillGovernanceProjection,
) -> None:
    store.append_brain_event(
        event_type=BrainEventType.SKILL_EVIDENCE_UPDATED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source="phase23_test",
        payload={
            "scope_key": scope_key,
            "skill_evidence_ledger": ledger.as_dict(),
        },
        ts=ledger.updated_at,
    )
    for proposal in governance.promotion_proposals:
        event_type = (
            BrainEventType.SKILL_PROMOTION_BLOCKED
            if proposal.status == BrainSkillGovernanceStatus.BLOCKED.value
            else BrainEventType.SKILL_PROMOTION_PROPOSED
        )
        store.append_brain_event(
            event_type=event_type,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="phase23_test",
            payload={
                "scope_key": scope_key,
                "promotion_proposal": proposal.as_dict(),
            },
            ts=proposal.updated_at,
        )
    for proposal in governance.demotion_proposals:
        store.append_brain_event(
            event_type=BrainEventType.SKILL_DEMOTION_PROPOSED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source="phase23_test",
            payload={
                "scope_key": scope_key,
                "demotion_proposal": proposal.as_dict(),
            },
            ts=proposal.updated_at,
        )


def seed_phase23_surfaces(
    *,
    store: BrainStore,
    session_ids,
    output_dir: Path,
) -> dict[str, Any]:
    episodes = [
        make_episode(
            index=1,
            scenario_family="robot_head_single_step",
            skill_ids=("skill-alpha",),
            task_success=True,
            calibration_bucket_counts={"aligned": 1},
        ),
        make_episode(
            index=2,
            scenario_family="robot_head_multi_step",
            skill_ids=("skill-alpha",),
            task_success=True,
            calibration_bucket_counts={"aligned": 1},
        ),
        make_episode(
            index=3,
            scenario_family="robot_head_single_step",
            scenario_id="robot_head_single_step-repeat",
            skill_ids=("skill-alpha",),
            task_success=True,
            calibration_bucket_counts={"aligned": 1},
        ),
        make_episode(
            index=4,
            scenario_family="robot_head_single_step",
            scenario_id="robot_head_single_step-overconfident-a",
            skill_ids=("skill-beta",),
            task_success=True,
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=5,
            scenario_family="robot_head_multi_step",
            scenario_id="robot_head_multi_step-overconfident-b",
            skill_ids=("skill-beta",),
            task_success=True,
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=6,
            scenario_family="robot_head_single_step",
            scenario_id="robot_head_single_step-overconfident-c",
            skill_ids=("skill-beta",),
            task_success=True,
            calibration_bucket_counts={"overconfident": 1},
        ),
        make_episode(
            index=7,
            scenario_family="robot_head_degraded_backend_comparison",
            scenario_id="robot_head_degraded-failure-a",
            skill_ids=("skill-alpha",),
            task_success=False,
            review_floor_count=1,
            operator_review_floored=True,
            safety_success=False,
            calibration_bucket_counts={"underconfident": 1},
            mismatch_codes=("robot_head_busy",),
        ),
        make_episode(
            index=8,
            scenario_family="robot_head_degraded_backend_comparison",
            scenario_id="robot_head_degraded-failure-b",
            skill_ids=("skill-alpha",),
            task_success=False,
            recovery_count=1,
            safety_success=False,
            calibration_bucket_counts={"underconfident": 1},
            mismatch_codes=("robot_head_busy",),
        ),
        make_episode(
            index=9,
            scenario_family="robot_head_busy_fault",
            execution_backend="fault",
            skill_ids=(),
            task_success=False,
            safety_success=False,
            calibration_bucket_counts={"overconfident": 1},
            risk_codes=("unsafe",),
            mismatch_codes=("robot_head_busy",),
        ),
    ]
    manifest = build_episode_dataset_manifest(episodes)
    procedural_skills = make_procedural_skills(
        {"skill_id": "skill-alpha", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.82},
        {"skill_id": "skill-beta", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.77},
        {"skill_id": "skill-low", "status": BrainProceduralSkillStatus.ACTIVE.value, "confidence": 0.32},
        {
            "skill_id": "skill-retired",
            "status": BrainProceduralSkillStatus.RETIRED.value,
            "confidence": 0.61,
            "retirement_reason": "policy_rejected",
        },
    )
    governance_report = {
        "low_confidence_skill_ids": ["skill-low"],
        "retired_skill_ids": ["skill-retired"],
    }
    practice_plan = BrainPracticeDirector(
        store=store,
        session_resolver=lambda: session_ids,
        presence_scope_key="browser:presence",
    ).create_plan(
        episodes=episodes,
        dataset_manifest=manifest,
        procedural_skill_governance_report=governance_report,
        scope_key=session_ids.thread_id,
        output_dir=output_dir,
    )
    ledger = build_skill_evidence_ledger(
        episodes=episodes,
        procedural_skills=procedural_skills,
        failure_clusters=manifest.failure_clusters,
        scope_type="thread",
        scope_id=session_ids.thread_id,
    )
    governance = build_skill_governance_projection(skill_evidence_ledger=ledger)
    append_skill_governance_events(
        store=store,
        session_ids=session_ids,
        scope_key=session_ids.thread_id,
        ledger=ledger,
        governance=governance,
    )
    return {
        "episodes": episodes,
        "dataset_manifest": manifest,
        "procedural_skills": procedural_skills,
        "practice_plan": practice_plan,
        "skill_evidence_ledger": ledger,
        "skill_governance": governance,
    }


__all__ = [
    "append_skill_governance_events",
    "make_episode",
    "make_procedural_skills",
    "seed_phase23_surfaces",
]

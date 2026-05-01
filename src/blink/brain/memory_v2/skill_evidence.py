"""Bounded episode-backed skill evidence layered above procedural skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

_CRITICAL_SAFETY_CODES = frozenset(
    {
        "unsafe",
        "robot_head_unarmed",
        "robot_head_status_unavailable",
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


def _sorted_mapping(values: dict[str, Any] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in sorted(dict(values or {}).items()):
        if _optional_text(key) is None:
            continue
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _coverage_sort_key(record: "BrainSkillScenarioCoverage") -> tuple[str, str]:
    return (record.scenario_family, record.latest_support_at or "")


def _evidence_sort_key(record: "BrainSkillEvidenceRecord") -> tuple[str, str, str]:
    return (record.latest_support_at or "", record.skill_id, record.evidence_id)


def _episode_sort_key(record: Any) -> tuple[str, str]:
    return (
        _optional_text(getattr(record, "generated_at", None))
        or _optional_text(getattr(record, "ended_at", None))
        or "",
        _optional_text(getattr(record, "episode_id", None)) or "",
    )


def _is_critical_safety_episode(record: Any) -> bool:
    safety_summary = getattr(record, "safety_summary", None)
    if safety_summary is None:
        return False
    if getattr(safety_summary, "safety_success", None) is False:
        return True
    codes = set(
        _sorted_unique(
            [
                *getattr(safety_summary, "risk_codes", []),
                *getattr(safety_summary, "mismatch_codes", []),
            ]
        )
    )
    return bool(codes.intersection(_CRITICAL_SAFETY_CODES))


@dataclass(frozen=True)
class BrainSkillScenarioCoverage:
    """Per-family bounded evidence coverage for one skill."""

    scenario_family: str
    episode_count: int
    task_success_count: int
    safety_success_count: int
    review_floor_count: int
    recovery_count: int
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    execution_backend_ids: list[str] = field(default_factory=list)
    embodied_policy_backend_ids: list[str] = field(default_factory=list)
    latest_support_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the coverage record."""
        return {
            "scenario_family": self.scenario_family,
            "episode_count": self.episode_count,
            "task_success_count": self.task_success_count,
            "safety_success_count": self.safety_success_count,
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "execution_backend_ids": list(self.execution_backend_ids),
            "embodied_policy_backend_ids": list(self.embodied_policy_backend_ids),
            "latest_support_at": self.latest_support_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillScenarioCoverage | None":
        """Hydrate one scenario-coverage record from JSON."""
        if not isinstance(data, dict):
            return None
        scenario_family = str(data.get("scenario_family", "")).strip()
        if not scenario_family:
            return None
        return cls(
            scenario_family=scenario_family,
            episode_count=int(data.get("episode_count", 0)),
            task_success_count=int(data.get("task_success_count", 0)),
            safety_success_count=int(data.get("safety_success_count", 0)),
            review_floor_count=int(data.get("review_floor_count", 0)),
            recovery_count=int(data.get("recovery_count", 0)),
            calibration_bucket_counts=_sorted_mapping(data.get("calibration_bucket_counts")),
            execution_backend_ids=_sorted_unique(data.get("execution_backend_ids", [])),
            embodied_policy_backend_ids=_sorted_unique(
                data.get("embodied_policy_backend_ids", [])
            ),
            latest_support_at=_optional_text(data.get("latest_support_at")),
        )


@dataclass(frozen=True)
class BrainSkillEvidenceRecord:
    """Bounded episode-backed evidence summary for one existing skill."""

    evidence_id: str
    skill_id: str
    skill_status: str | None = None
    skill_family_key: str | None = None
    supporting_episode_ids: list[str] = field(default_factory=list)
    scenario_families: list[str] = field(default_factory=list)
    scenario_coverage: list[BrainSkillScenarioCoverage] = field(default_factory=list)
    execution_backend_ids: list[str] = field(default_factory=list)
    embodied_policy_backend_ids: list[str] = field(default_factory=list)
    support_episode_count: int = 0
    task_success_count: int = 0
    safety_success_count: int = 0
    review_floor_count: int = 0
    recovery_count: int = 0
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    failure_cluster_signatures: list[str] = field(default_factory=list)
    failure_cluster_counts: dict[str, int] = field(default_factory=dict)
    critical_safety_violation_count: int = 0
    delta_episode_count: int = 0
    latest_support_at: str | None = None
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence record."""
        return {
            "evidence_id": self.evidence_id,
            "skill_id": self.skill_id,
            "skill_status": self.skill_status,
            "skill_family_key": self.skill_family_key,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "scenario_families": list(self.scenario_families),
            "scenario_coverage": [record.as_dict() for record in self.scenario_coverage],
            "execution_backend_ids": list(self.execution_backend_ids),
            "embodied_policy_backend_ids": list(self.embodied_policy_backend_ids),
            "support_episode_count": self.support_episode_count,
            "task_success_count": self.task_success_count,
            "safety_success_count": self.safety_success_count,
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "failure_cluster_signatures": list(self.failure_cluster_signatures),
            "failure_cluster_counts": dict(self.failure_cluster_counts),
            "critical_safety_violation_count": self.critical_safety_violation_count,
            "delta_episode_count": self.delta_episode_count,
            "latest_support_at": self.latest_support_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillEvidenceRecord | None":
        """Hydrate one evidence record from JSON."""
        if not isinstance(data, dict):
            return None
        evidence_id = str(data.get("evidence_id", "")).strip()
        skill_id = str(data.get("skill_id", "")).strip()
        if not evidence_id or not skill_id:
            return None
        return cls(
            evidence_id=evidence_id,
            skill_id=skill_id,
            skill_status=_optional_text(data.get("skill_status")),
            skill_family_key=_optional_text(data.get("skill_family_key")),
            supporting_episode_ids=_sorted_unique(data.get("supporting_episode_ids", [])),
            scenario_families=_sorted_unique(data.get("scenario_families", [])),
            scenario_coverage=[
                record
                for item in data.get("scenario_coverage", [])
                if (record := BrainSkillScenarioCoverage.from_dict(item)) is not None
            ],
            execution_backend_ids=_sorted_unique(data.get("execution_backend_ids", [])),
            embodied_policy_backend_ids=_sorted_unique(
                data.get("embodied_policy_backend_ids", [])
            ),
            support_episode_count=int(data.get("support_episode_count", 0)),
            task_success_count=int(data.get("task_success_count", 0)),
            safety_success_count=int(data.get("safety_success_count", 0)),
            review_floor_count=int(data.get("review_floor_count", 0)),
            recovery_count=int(data.get("recovery_count", 0)),
            calibration_bucket_counts=_sorted_mapping(data.get("calibration_bucket_counts")),
            failure_cluster_signatures=_sorted_unique(data.get("failure_cluster_signatures", [])),
            failure_cluster_counts=_sorted_mapping(data.get("failure_cluster_counts")),
            critical_safety_violation_count=int(data.get("critical_safety_violation_count", 0)),
            delta_episode_count=int(data.get("delta_episode_count", 0)),
            latest_support_at=_optional_text(data.get("latest_support_at")),
            updated_at=str(data.get("updated_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainSkillEvidenceLedger:
    """Thread-scoped bounded evidence ledger for procedural-skill governance."""

    scope_type: str
    scope_id: str
    evidence_records: list[BrainSkillEvidenceRecord] = field(default_factory=list)
    family_hypothesis_counts: dict[str, int] = field(default_factory=dict)
    skill_status_counts: dict[str, int] = field(default_factory=dict)
    recent_evidence_ids: list[str] = field(default_factory=list)
    updated_at: str = ""

    def sync_lists(self):
        """Refresh derived ids and counts."""
        self.evidence_records = sorted(
            self.evidence_records,
            key=_evidence_sort_key,
            reverse=True,
        )
        status_counts: dict[str, int] = {}
        for record in self.evidence_records:
            status = _optional_text(record.skill_status)
            if status is not None:
                status_counts[status] = status_counts.get(status, 0) + 1
        self.skill_status_counts = dict(sorted(status_counts.items()))
        self.family_hypothesis_counts = _sorted_mapping(self.family_hypothesis_counts)
        self.recent_evidence_ids = [record.evidence_id for record in self.evidence_records[:24]]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the evidence ledger."""
        self.sync_lists()
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "evidence_records": [record.as_dict() for record in self.evidence_records],
            "family_hypothesis_counts": dict(self.family_hypothesis_counts),
            "skill_status_counts": dict(self.skill_status_counts),
            "recent_evidence_ids": list(self.recent_evidence_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillEvidenceLedger":
        """Hydrate the evidence ledger from JSON."""
        payload = dict(data or {})
        ledger = cls(
            scope_type=str(payload.get("scope_type", "")).strip() or "thread",
            scope_id=str(payload.get("scope_id", "")).strip(),
            evidence_records=[
                record
                for item in payload.get("evidence_records", [])
                if (record := BrainSkillEvidenceRecord.from_dict(item)) is not None
            ],
            family_hypothesis_counts=_sorted_mapping(payload.get("family_hypothesis_counts")),
            skill_status_counts=_sorted_mapping(payload.get("skill_status_counts")),
            recent_evidence_ids=_sorted_unique(payload.get("recent_evidence_ids", [])),
            updated_at=str(payload.get("updated_at") or ""),
        )
        ledger.sync_lists()
        return ledger


def apply_skill_evidence_update(
    projection: BrainSkillEvidenceLedger,
    incoming: BrainSkillEvidenceLedger,
) -> None:
    """Replace the evidence rows in one projection from an explicit update event."""
    projection.scope_type = incoming.scope_type
    projection.scope_id = incoming.scope_id
    projection.evidence_records = list(incoming.evidence_records)
    projection.family_hypothesis_counts = dict(incoming.family_hypothesis_counts)
    projection.updated_at = incoming.updated_at
    projection.sync_lists()


def build_skill_evidence_ledger(
    *,
    episodes: Iterable[Any],
    procedural_skills: Any,
    failure_clusters: Iterable[Any] = (),
    scope_type: str,
    scope_id: str,
    previous_ledger: BrainSkillEvidenceLedger | None = None,
) -> BrainSkillEvidenceLedger:
    """Build one deterministic skill evidence ledger from exported episodes."""
    previous_support_by_skill: dict[str, set[str]] = {}
    if previous_ledger is not None:
        for record in previous_ledger.evidence_records:
            previous_support_by_skill[record.skill_id] = set(record.supporting_episode_ids)
    skills = {
        str(skill.skill_id).strip(): skill
        for skill in sorted(
            list(getattr(procedural_skills, "skills", []) or []),
            key=lambda item: (
                getattr(item, "goal_family", ""),
                getattr(item, "skill_id", ""),
            ),
        )
        if _optional_text(getattr(skill, "skill_id", None)) is not None
    }
    failure_clusters_by_episode: dict[str, list[str]] = {}
    failure_cluster_counts_by_episode: dict[str, dict[str, int]] = {}
    for cluster in failure_clusters or ():
        signature = _optional_text(getattr(cluster, "signature", None))
        if signature is None:
            continue
        episode_ids = _sorted_unique(getattr(cluster, "episode_ids", []))
        episode_count = int(getattr(cluster, "episode_count", 0)) or max(len(episode_ids), 1)
        for episode_id in episode_ids:
            failure_clusters_by_episode.setdefault(episode_id, []).append(signature)
            failure_cluster_counts_by_episode.setdefault(episode_id, {})
            failure_cluster_counts_by_episode[episode_id][signature] = episode_count
    grouped: dict[str, list[Any]] = {}
    family_hypothesis_counts: dict[str, int] = {}
    for episode in sorted(list(episodes), key=_episode_sort_key):
        linked_skill_ids = _sorted_unique(
            [
                skill_id
                for skill_id in getattr(episode, "skill_ids", [])
                if skill_id in skills
            ]
        )
        if not linked_skill_ids:
            scenario_family = _optional_text(getattr(episode, "scenario_family", None))
            if scenario_family is not None:
                family_hypothesis_counts[scenario_family] = (
                    family_hypothesis_counts.get(scenario_family, 0) + 1
                )
            continue
        for skill_id in linked_skill_ids:
            grouped.setdefault(skill_id, []).append(episode)
    evidence_records: list[BrainSkillEvidenceRecord] = []
    for skill_id, skill_episodes in sorted(grouped.items()):
        skill = skills.get(skill_id)
        if skill is None:
            continue
        scenario_groups: dict[str, list[Any]] = {}
        calibration_bucket_counts: dict[str, int] = {}
        failure_cluster_counts: dict[str, int] = {}
        execution_backend_ids: list[str] = []
        policy_backend_ids: list[str] = []
        supporting_episode_ids: list[str] = []
        task_success_count = 0
        safety_success_count = 0
        review_floor_count = 0
        recovery_count = 0
        critical_safety_violation_count = 0
        latest_support_at: str | None = None
        for episode in skill_episodes:
            scenario_family = _optional_text(getattr(episode, "scenario_family", None))
            if scenario_family is not None:
                scenario_groups.setdefault(scenario_family, []).append(episode)
            episode_id = _optional_text(getattr(episode, "episode_id", None))
            if episode_id is not None:
                supporting_episode_ids.append(episode_id)
                for signature in failure_clusters_by_episode.get(episode_id, []):
                    failure_cluster_counts[signature] = max(
                        int(failure_cluster_counts.get(signature, 0)),
                        int(
                            failure_cluster_counts_by_episode.get(episode_id, {}).get(signature, 0)
                        ),
                    )
            execution_backend = _optional_text(getattr(episode, "execution_backend", None))
            embodied_policy_backend_id = _optional_text(
                getattr(episode, "embodied_policy_backend_id", None)
            )
            if execution_backend is not None:
                execution_backend_ids.append(execution_backend)
            if embodied_policy_backend_id is not None:
                policy_backend_ids.append(embodied_policy_backend_id)
            outcome_summary = getattr(episode, "outcome_summary", None)
            if getattr(outcome_summary, "task_success", None) is True:
                task_success_count += 1
            safety_summary = getattr(episode, "safety_summary", None)
            if getattr(safety_summary, "safety_success", None) is True:
                safety_success_count += 1
            review_floor_count += int(getattr(safety_summary, "review_floor_count", 0))
            recovery_count += int(getattr(safety_summary, "recovery_count", 0))
            if _is_critical_safety_episode(episode):
                critical_safety_violation_count += 1
            for bucket, count in dict(
                getattr(outcome_summary, "calibration_bucket_counts", {}) or {}
            ).items():
                calibration_bucket_counts[str(bucket)] = (
                    calibration_bucket_counts.get(str(bucket), 0) + int(count)
                )
            latest_value = (
                _optional_text(getattr(episode, "generated_at", None))
                or _optional_text(getattr(episode, "ended_at", None))
                or _optional_text(getattr(episode, "started_at", None))
            )
            if latest_value is not None:
                latest_support_at = max(latest_support_at or latest_value, latest_value)
        scenario_coverage: list[BrainSkillScenarioCoverage] = []
        for scenario_family, family_episodes in sorted(scenario_groups.items()):
            family_task_success_count = sum(
                1
                for episode in family_episodes
                if getattr(getattr(episode, "outcome_summary", None), "task_success", None) is True
            )
            family_safety_success_count = sum(
                1
                for episode in family_episodes
                if getattr(getattr(episode, "safety_summary", None), "safety_success", None) is True
            )
            family_review_floor_count = sum(
                int(getattr(getattr(episode, "safety_summary", None), "review_floor_count", 0))
                for episode in family_episodes
            )
            family_recovery_count = sum(
                int(getattr(getattr(episode, "safety_summary", None), "recovery_count", 0))
                for episode in family_episodes
            )
            family_calibration_bucket_counts: dict[str, int] = {}
            family_execution_backend_ids: list[str] = []
            family_policy_backend_ids: list[str] = []
            family_latest_support_at: str | None = None
            for episode in family_episodes:
                outcome_summary = getattr(episode, "outcome_summary", None)
                for bucket, count in dict(
                    getattr(outcome_summary, "calibration_bucket_counts", {}) or {}
                ).items():
                    family_calibration_bucket_counts[str(bucket)] = (
                        family_calibration_bucket_counts.get(str(bucket), 0) + int(count)
                    )
                if (backend_id := _optional_text(getattr(episode, "execution_backend", None))) is not None:
                    family_execution_backend_ids.append(backend_id)
                if (
                    policy_backend_id := _optional_text(
                        getattr(episode, "embodied_policy_backend_id", None)
                    )
                ) is not None:
                    family_policy_backend_ids.append(policy_backend_id)
                latest_value = (
                    _optional_text(getattr(episode, "generated_at", None))
                    or _optional_text(getattr(episode, "ended_at", None))
                    or _optional_text(getattr(episode, "started_at", None))
                )
                if latest_value is not None:
                    family_latest_support_at = max(
                        family_latest_support_at or latest_value,
                        latest_value,
                    )
            scenario_coverage.append(
                BrainSkillScenarioCoverage(
                    scenario_family=scenario_family,
                    episode_count=len(family_episodes),
                    task_success_count=family_task_success_count,
                    safety_success_count=family_safety_success_count,
                    review_floor_count=family_review_floor_count,
                    recovery_count=family_recovery_count,
                    calibration_bucket_counts=_sorted_mapping(family_calibration_bucket_counts),
                    execution_backend_ids=_sorted_unique(family_execution_backend_ids),
                    embodied_policy_backend_ids=_sorted_unique(family_policy_backend_ids),
                    latest_support_at=family_latest_support_at,
                )
            )
        support_episode_ids = _sorted_unique(supporting_episode_ids)
        previous_episode_ids = previous_support_by_skill.get(skill_id, set())
        delta_episode_count = len(set(support_episode_ids) - previous_episode_ids)
        evidence_records.append(
            BrainSkillEvidenceRecord(
                evidence_id=_stable_id(
                    "skill_evidence",
                    skill_id,
                    len(support_episode_ids),
                    *(support_episode_ids[:8]),
                ),
                skill_id=skill_id,
                skill_status=_optional_text(getattr(skill, "status", None)),
                skill_family_key=_optional_text(getattr(skill, "skill_family_key", None)),
                supporting_episode_ids=support_episode_ids,
                scenario_families=_sorted_unique(scenario_groups.keys()),
                scenario_coverage=sorted(scenario_coverage, key=_coverage_sort_key),
                execution_backend_ids=_sorted_unique(execution_backend_ids),
                embodied_policy_backend_ids=_sorted_unique(policy_backend_ids),
                support_episode_count=len(support_episode_ids),
                task_success_count=task_success_count,
                safety_success_count=safety_success_count,
                review_floor_count=review_floor_count,
                recovery_count=recovery_count,
                calibration_bucket_counts=_sorted_mapping(calibration_bucket_counts),
                failure_cluster_signatures=sorted(failure_cluster_counts),
                failure_cluster_counts=_sorted_mapping(failure_cluster_counts),
                critical_safety_violation_count=critical_safety_violation_count,
                delta_episode_count=delta_episode_count,
                latest_support_at=latest_support_at,
                updated_at=latest_support_at or "",
                details={
                    "goal_family": _optional_text(getattr(skill, "goal_family", None)),
                    "template_fingerprint": _optional_text(
                        getattr(skill, "template_fingerprint", None)
                    ),
                },
            )
        )
    ledger = BrainSkillEvidenceLedger(
        scope_type=scope_type,
        scope_id=scope_id,
        evidence_records=evidence_records,
        family_hypothesis_counts=family_hypothesis_counts,
        updated_at=max(
            [record.updated_at for record in evidence_records if _optional_text(record.updated_at)] or [""]
        ),
    )
    ledger.sync_lists()
    return ledger


def build_skill_evidence_inspection(
    *,
    skill_evidence_ledger: Any,
    skill_governance: Any,
    recent_limit: int = 6,
) -> dict[str, Any]:
    """Build a bounded operator-facing inspection summary for Phase 23 skill evidence."""
    ledger = skill_evidence_ledger
    governance = skill_governance
    evidence_records = list(getattr(ledger, "evidence_records", []) or [])
    promotion_proposals = list(getattr(governance, "promotion_proposals", []) or [])
    demotion_proposals = list(getattr(governance, "demotion_proposals", []) or [])
    top_evidence_deltas = [
        {
            "skill_id": record.skill_id,
            "delta_episode_count": int(record.delta_episode_count),
            "support_episode_count": int(record.support_episode_count),
            "scenario_families": list(record.scenario_families),
            "latest_support_at": record.latest_support_at,
            "critical_safety_violation_count": int(record.critical_safety_violation_count),
        }
        for record in sorted(
            evidence_records,
            key=lambda item: (
                -int(item.delta_episode_count),
                -(int(item.support_episode_count)),
                item.skill_id,
            ),
        )[:recent_limit]
    ]
    recent_proposals = [
        {
            "proposal_id": record.proposal_id,
            "skill_id": record.skill_id,
            "status": record.status,
            "kind": "promotion",
            "reason_codes": list(record.blocked_reason_codes),
            "updated_at": record.updated_at,
        }
        for record in promotion_proposals[:recent_limit]
    ] + [
        {
            "proposal_id": record.proposal_id,
            "skill_id": record.skill_id,
            "status": record.status,
            "kind": "demotion",
            "reason_codes": list(record.reason_codes),
            "updated_at": record.updated_at,
        }
        for record in demotion_proposals[:recent_limit]
    ]
    recent_proposals = sorted(
        recent_proposals,
        key=lambda item: (
            item["updated_at"],
            item["kind"],
            item["proposal_id"],
        ),
        reverse=True,
    )[:recent_limit]
    return {
        "evidence_count": len(evidence_records),
        "skill_status_counts": dict(getattr(ledger, "skill_status_counts", {}) or {}),
        "family_hypothesis_counts": dict(getattr(ledger, "family_hypothesis_counts", {}) or {}),
        "top_evidence_deltas": top_evidence_deltas,
        "proposal_status_counts": dict(getattr(governance, "proposal_status_counts", {}) or {}),
        "blocked_reason_code_counts": dict(
            getattr(governance, "blocked_reason_code_counts", {}) or {}
        ),
        "demotion_reason_code_counts": dict(
            getattr(governance, "demotion_reason_code_counts", {}) or {}
        ),
        "recent_governance_proposals": recent_proposals,
    }


__all__ = [
    "BrainSkillEvidenceLedger",
    "BrainSkillEvidenceRecord",
    "BrainSkillScenarioCoverage",
    "apply_skill_evidence_update",
    "build_skill_evidence_inspection",
    "build_skill_evidence_ledger",
]

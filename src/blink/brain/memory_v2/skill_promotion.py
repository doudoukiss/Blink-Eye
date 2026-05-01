"""Conservative skill-governance proposals layered above procedural skills."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sorted_unique(values: Iterable[str | None]) -> list[str]:
    return sorted({text for value in values if (text := _optional_text(value)) is not None})


def _stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip() for part in parts)
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'blink:{prefix}:{normalized}').hex}"


def _sorted_mapping(values: dict[str, Any] | None) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(dict(values or {}).items())
        if _optional_text(key) is not None
    }


def _promotion_sort_key(record: "BrainSkillPromotionProposal") -> tuple[str, str, str]:
    return (record.updated_at, record.skill_id, record.proposal_id)


def _demotion_sort_key(record: "BrainSkillDemotionProposal") -> tuple[str, str, str]:
    return (record.updated_at, record.skill_id, record.proposal_id)


class BrainSkillGovernanceStatus(str, Enum):
    """Proposal lifecycle state for Phase 23 governance."""

    PROPOSED = "proposed"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class BrainSkillPromotionProposal:
    """One conservative promotion proposal for an existing procedural skill."""

    proposal_id: str
    proposal_key: str
    skill_id: str
    status: str
    supporting_episode_ids: list[str] = field(default_factory=list)
    supporting_scenario_families: list[str] = field(default_factory=list)
    support_episode_count: int = 0
    success_episode_count: int = 0
    safety_success_count: int = 0
    review_floor_count: int = 0
    recovery_count: int = 0
    critical_safety_violation_count: int = 0
    calibration_bucket_counts: dict[str, int] = field(default_factory=dict)
    failure_cluster_signatures: list[str] = field(default_factory=list)
    blocked_reason_codes: list[str] = field(default_factory=list)
    proposed_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the promotion proposal."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_key": self.proposal_key,
            "skill_id": self.skill_id,
            "status": self.status,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "supporting_scenario_families": list(self.supporting_scenario_families),
            "support_episode_count": self.support_episode_count,
            "success_episode_count": self.success_episode_count,
            "safety_success_count": self.safety_success_count,
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
            "critical_safety_violation_count": self.critical_safety_violation_count,
            "calibration_bucket_counts": dict(self.calibration_bucket_counts),
            "failure_cluster_signatures": list(self.failure_cluster_signatures),
            "blocked_reason_codes": list(self.blocked_reason_codes),
            "proposed_at": self.proposed_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillPromotionProposal | None":
        """Hydrate one promotion proposal from JSON."""
        if not isinstance(data, dict):
            return None
        proposal_id = str(data.get("proposal_id", "")).strip()
        proposal_key = str(data.get("proposal_key", "")).strip()
        skill_id = str(data.get("skill_id", "")).strip()
        status = str(data.get("status", "")).strip()
        if not proposal_id or not proposal_key or not skill_id or not status:
            return None
        return cls(
            proposal_id=proposal_id,
            proposal_key=proposal_key,
            skill_id=skill_id,
            status=status,
            supporting_episode_ids=_sorted_unique(data.get("supporting_episode_ids", [])),
            supporting_scenario_families=_sorted_unique(
                data.get("supporting_scenario_families", [])
            ),
            support_episode_count=int(data.get("support_episode_count", 0)),
            success_episode_count=int(data.get("success_episode_count", 0)),
            safety_success_count=int(data.get("safety_success_count", 0)),
            review_floor_count=int(data.get("review_floor_count", 0)),
            recovery_count=int(data.get("recovery_count", 0)),
            critical_safety_violation_count=int(data.get("critical_safety_violation_count", 0)),
            calibration_bucket_counts=_sorted_mapping(data.get("calibration_bucket_counts")),
            failure_cluster_signatures=_sorted_unique(
                data.get("failure_cluster_signatures", [])
            ),
            blocked_reason_codes=_sorted_unique(data.get("blocked_reason_codes", [])),
            proposed_at=str(data.get("proposed_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("proposed_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class BrainSkillDemotionProposal:
    """One bounded demotion or repair proposal for an existing procedural skill."""

    proposal_id: str
    proposal_key: str
    skill_id: str
    failure_signature: str
    status: str
    triggering_episode_ids: list[str] = field(default_factory=list)
    triggering_scenario_families: list[str] = field(default_factory=list)
    review_floor_count: int = 0
    recovery_count: int = 0
    critical_safety_violation_count: int = 0
    reason_codes: list[str] = field(default_factory=list)
    proposed_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the demotion proposal."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_key": self.proposal_key,
            "skill_id": self.skill_id,
            "failure_signature": self.failure_signature,
            "status": self.status,
            "triggering_episode_ids": list(self.triggering_episode_ids),
            "triggering_scenario_families": list(self.triggering_scenario_families),
            "review_floor_count": self.review_floor_count,
            "recovery_count": self.recovery_count,
            "critical_safety_violation_count": self.critical_safety_violation_count,
            "reason_codes": list(self.reason_codes),
            "proposed_at": self.proposed_at,
            "updated_at": self.updated_at,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillDemotionProposal | None":
        """Hydrate one demotion proposal from JSON."""
        if not isinstance(data, dict):
            return None
        proposal_id = str(data.get("proposal_id", "")).strip()
        proposal_key = str(data.get("proposal_key", "")).strip()
        skill_id = str(data.get("skill_id", "")).strip()
        failure_signature = str(data.get("failure_signature", "")).strip()
        status = str(data.get("status", "")).strip()
        if not proposal_id or not proposal_key or not skill_id or not failure_signature or not status:
            return None
        return cls(
            proposal_id=proposal_id,
            proposal_key=proposal_key,
            skill_id=skill_id,
            failure_signature=failure_signature,
            status=status,
            triggering_episode_ids=_sorted_unique(data.get("triggering_episode_ids", [])),
            triggering_scenario_families=_sorted_unique(
                data.get("triggering_scenario_families", [])
            ),
            review_floor_count=int(data.get("review_floor_count", 0)),
            recovery_count=int(data.get("recovery_count", 0)),
            critical_safety_violation_count=int(data.get("critical_safety_violation_count", 0)),
            reason_codes=_sorted_unique(data.get("reason_codes", [])),
            proposed_at=str(data.get("proposed_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or data.get("proposed_at") or _utc_now()),
            details=dict(data.get("details", {})),
        )


@dataclass
class BrainSkillGovernanceProjection:
    """Replay-safe bounded skill-governance proposal projection."""

    scope_type: str
    scope_id: str
    promotion_proposals: list[BrainSkillPromotionProposal] = field(default_factory=list)
    demotion_proposals: list[BrainSkillDemotionProposal] = field(default_factory=list)
    proposal_status_counts: dict[str, int] = field(default_factory=dict)
    blocked_reason_code_counts: dict[str, int] = field(default_factory=dict)
    demotion_reason_code_counts: dict[str, int] = field(default_factory=dict)
    recent_promotion_ids: list[str] = field(default_factory=list)
    recent_demotion_ids: list[str] = field(default_factory=list)
    updated_at: str = ""

    def sync_lists(self):
        """Refresh deduped ids and counters from proposal state."""
        promotion_by_key: dict[str, BrainSkillPromotionProposal] = {}
        for record in sorted(self.promotion_proposals, key=_promotion_sort_key):
            promotion_by_key[record.proposal_key] = record
        demotion_by_key: dict[str, BrainSkillDemotionProposal] = {}
        for record in sorted(self.demotion_proposals, key=_demotion_sort_key):
            demotion_by_key[record.proposal_key] = record
        self.promotion_proposals = sorted(
            promotion_by_key.values(),
            key=_promotion_sort_key,
            reverse=True,
        )[:24]
        self.demotion_proposals = sorted(
            demotion_by_key.values(),
            key=_demotion_sort_key,
            reverse=True,
        )[:24]
        proposal_status_counts: dict[str, int] = {}
        blocked_reason_code_counts: dict[str, int] = {}
        demotion_reason_code_counts: dict[str, int] = {}
        for record in self.promotion_proposals:
            proposal_status_counts[record.status] = proposal_status_counts.get(record.status, 0) + 1
            for reason_code in record.blocked_reason_codes:
                blocked_reason_code_counts[reason_code] = (
                    blocked_reason_code_counts.get(reason_code, 0) + 1
                )
        for record in self.demotion_proposals:
            proposal_status_counts[record.status] = proposal_status_counts.get(record.status, 0) + 1
            for reason_code in record.reason_codes:
                demotion_reason_code_counts[reason_code] = (
                    demotion_reason_code_counts.get(reason_code, 0) + 1
                )
        self.proposal_status_counts = dict(sorted(proposal_status_counts.items()))
        self.blocked_reason_code_counts = dict(sorted(blocked_reason_code_counts.items()))
        self.demotion_reason_code_counts = dict(sorted(demotion_reason_code_counts.items()))
        self.recent_promotion_ids = [record.proposal_id for record in self.promotion_proposals]
        self.recent_demotion_ids = [record.proposal_id for record in self.demotion_proposals]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the governance projection."""
        self.sync_lists()
        return {
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "promotion_proposals": [record.as_dict() for record in self.promotion_proposals],
            "demotion_proposals": [record.as_dict() for record in self.demotion_proposals],
            "proposal_status_counts": dict(self.proposal_status_counts),
            "blocked_reason_code_counts": dict(self.blocked_reason_code_counts),
            "demotion_reason_code_counts": dict(self.demotion_reason_code_counts),
            "recent_promotion_ids": list(self.recent_promotion_ids),
            "recent_demotion_ids": list(self.recent_demotion_ids),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainSkillGovernanceProjection":
        """Hydrate the governance projection from JSON."""
        payload = dict(data or {})
        projection = cls(
            scope_type=str(payload.get("scope_type", "")).strip(),
            scope_id=str(payload.get("scope_id", "")).strip(),
            promotion_proposals=[
                record
                for item in payload.get("promotion_proposals", [])
                if (record := BrainSkillPromotionProposal.from_dict(item)) is not None
            ],
            demotion_proposals=[
                record
                for item in payload.get("demotion_proposals", [])
                if (record := BrainSkillDemotionProposal.from_dict(item)) is not None
            ],
            proposal_status_counts=_sorted_mapping(payload.get("proposal_status_counts")),
            blocked_reason_code_counts=_sorted_mapping(payload.get("blocked_reason_code_counts")),
            demotion_reason_code_counts=_sorted_mapping(payload.get("demotion_reason_code_counts")),
            recent_promotion_ids=_sorted_unique(payload.get("recent_promotion_ids", [])),
            recent_demotion_ids=_sorted_unique(payload.get("recent_demotion_ids", [])),
            updated_at=str(payload.get("updated_at") or ""),
        )
        projection.sync_lists()
        return projection


def append_skill_promotion_proposal(
    projection: BrainSkillGovernanceProjection,
    proposal: BrainSkillPromotionProposal,
) -> None:
    """Append or replace one promotion proposal by stable key."""
    projection.promotion_proposals = [
        record
        for record in projection.promotion_proposals
        if record.proposal_key != proposal.proposal_key
    ]
    projection.promotion_proposals.append(proposal)
    projection.updated_at = max(projection.updated_at, proposal.updated_at)
    projection.sync_lists()


def append_skill_demotion_proposal(
    projection: BrainSkillGovernanceProjection,
    proposal: BrainSkillDemotionProposal,
) -> None:
    """Append one demotion proposal and supersede prior promotion proposals for the skill."""
    projection.demotion_proposals = [
        record
        for record in projection.demotion_proposals
        if record.proposal_key != proposal.proposal_key
    ]
    projection.demotion_proposals.append(proposal)
    superseded_promotions: list[BrainSkillPromotionProposal] = []
    for record in projection.promotion_proposals:
        if record.skill_id != proposal.skill_id or record.status == BrainSkillGovernanceStatus.SUPERSEDED.value:
            superseded_promotions.append(record)
            continue
        superseded_promotions.append(
            replace(
                record,
                status=BrainSkillGovernanceStatus.SUPERSEDED.value,
                updated_at=max(record.updated_at, proposal.updated_at),
                details={**record.details, "superseded_by_demotion_proposal": proposal.proposal_id},
            )
        )
    projection.promotion_proposals = superseded_promotions
    projection.updated_at = max(projection.updated_at, proposal.updated_at)
    projection.sync_lists()


def _dominant_failure_signature(record: Any) -> tuple[str | None, int]:
    counts = dict(getattr(record, "failure_cluster_counts", {}) or {})
    if not counts:
        return None, 0
    signature, count = min(
        counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    return str(signature), int(count)


def build_skill_governance_projection(
    *,
    skill_evidence_ledger: Any,
) -> BrainSkillGovernanceProjection:
    """Build conservative promotion and demotion proposals from one evidence ledger."""
    promotion_proposals: list[BrainSkillPromotionProposal] = []
    demotion_proposals: list[BrainSkillDemotionProposal] = []
    scope_type = str(getattr(skill_evidence_ledger, "scope_type", "thread")).strip() or "thread"
    scope_id = str(getattr(skill_evidence_ledger, "scope_id", "")).strip()
    updated_at = str(getattr(skill_evidence_ledger, "updated_at", "")).strip()
    for record in sorted(
        list(getattr(skill_evidence_ledger, "evidence_records", []) or []),
        key=lambda item: (
            getattr(item, "skill_id", ""),
            getattr(item, "latest_support_at", ""),
            getattr(item, "evidence_id", ""),
        ),
    ):
        support_episode_count = int(getattr(record, "support_episode_count", 0))
        success_episode_count = int(getattr(record, "task_success_count", 0))
        safety_success_count = int(getattr(record, "safety_success_count", 0))
        review_floor_count = int(getattr(record, "review_floor_count", 0))
        recovery_count = int(getattr(record, "recovery_count", 0))
        critical_safety_violation_count = int(
            getattr(record, "critical_safety_violation_count", 0)
        )
        scenario_families = _sorted_unique(getattr(record, "scenario_families", []))
        calibration_bucket_counts = _sorted_mapping(
            getattr(record, "calibration_bucket_counts", {})
        )
        overconfident_count = int(calibration_bucket_counts.get("overconfident", 0))
        aligned_count = int(calibration_bucket_counts.get("aligned", 0))
        blocked_reason_codes: list[str] = []
        if critical_safety_violation_count > 0:
            blocked_reason_codes.append("critical_safety_violation")
        if overconfident_count > aligned_count and overconfident_count > 0:
            blocked_reason_codes.append("overconfident_calibration_dominant")
        if review_floor_count > 1:
            blocked_reason_codes.append("review_floor_exceeded")
        if recovery_count > 1:
            blocked_reason_codes.append("recovery_burden_exceeded")
        if success_episode_count >= 3 and len(scenario_families) >= 2:
            promotion_proposals.append(
                BrainSkillPromotionProposal(
                    proposal_id=_stable_id("skill_promotion", getattr(record, "skill_id", ""), "promotion"),
                    proposal_key=f"{getattr(record, 'skill_id', '')}:promotion",
                    skill_id=str(getattr(record, "skill_id", "")).strip(),
                    status=(
                        BrainSkillGovernanceStatus.BLOCKED.value
                        if blocked_reason_codes
                        else BrainSkillGovernanceStatus.PROPOSED.value
                    ),
                    supporting_episode_ids=_sorted_unique(
                        getattr(record, "supporting_episode_ids", [])
                    ),
                    supporting_scenario_families=scenario_families,
                    support_episode_count=support_episode_count,
                    success_episode_count=success_episode_count,
                    safety_success_count=safety_success_count,
                    review_floor_count=review_floor_count,
                    recovery_count=recovery_count,
                    critical_safety_violation_count=critical_safety_violation_count,
                    calibration_bucket_counts=calibration_bucket_counts,
                    failure_cluster_signatures=_sorted_unique(
                        getattr(record, "failure_cluster_signatures", [])
                    ),
                    blocked_reason_codes=blocked_reason_codes,
                    proposed_at=str(getattr(record, "latest_support_at", "")).strip() or updated_at,
                    updated_at=str(getattr(record, "latest_support_at", "")).strip() or updated_at,
                    details={
                        "delta_episode_count": int(getattr(record, "delta_episode_count", 0)),
                        "skill_status": _optional_text(getattr(record, "skill_status", None)),
                    },
                )
            )
        dominant_failure_signature, dominant_failure_count = _dominant_failure_signature(record)
        demotion_reason_codes: list[str] = []
        failure_signature = dominant_failure_signature
        if critical_safety_violation_count > 0:
            demotion_reason_codes.append("critical_safety_violation")
            failure_signature = failure_signature or "critical_safety_violation"
        if dominant_failure_count >= 2:
            demotion_reason_codes.append("repeated_failure_cluster")
        if review_floor_count > 1:
            demotion_reason_codes.append("review_floor_exceeded")
            failure_signature = failure_signature or "review_floor_exceeded"
        if recovery_count > 1:
            demotion_reason_codes.append("recovery_burden_exceeded")
            failure_signature = failure_signature or "recovery_burden_exceeded"
        if demotion_reason_codes and failure_signature is not None:
            demotion_proposals.append(
                BrainSkillDemotionProposal(
                    proposal_id=_stable_id(
                        "skill_demotion",
                        getattr(record, "skill_id", ""),
                        failure_signature,
                    ),
                    proposal_key=f"{getattr(record, 'skill_id', '')}:demotion:{failure_signature}",
                    skill_id=str(getattr(record, "skill_id", "")).strip(),
                    failure_signature=failure_signature,
                    status=BrainSkillGovernanceStatus.PROPOSED.value,
                    triggering_episode_ids=_sorted_unique(
                        getattr(record, "supporting_episode_ids", [])
                    ),
                    triggering_scenario_families=scenario_families,
                    review_floor_count=review_floor_count,
                    recovery_count=recovery_count,
                    critical_safety_violation_count=critical_safety_violation_count,
                    reason_codes=demotion_reason_codes,
                    proposed_at=str(getattr(record, "latest_support_at", "")).strip() or updated_at,
                    updated_at=str(getattr(record, "latest_support_at", "")).strip() or updated_at,
                    details={
                        "failure_cluster_count": dominant_failure_count,
                        "delta_episode_count": int(getattr(record, "delta_episode_count", 0)),
                    },
                )
            )
    projection = BrainSkillGovernanceProjection(
        scope_type=scope_type,
        scope_id=scope_id,
        promotion_proposals=promotion_proposals,
        demotion_proposals=[],
        updated_at=updated_at,
    )
    for proposal in demotion_proposals:
        append_skill_demotion_proposal(projection, proposal)
    projection.sync_lists()
    return projection


__all__ = [
    "BrainSkillDemotionProposal",
    "BrainSkillGovernanceProjection",
    "BrainSkillGovernanceStatus",
    "BrainSkillPromotionProposal",
    "append_skill_demotion_proposal",
    "append_skill_promotion_proposal",
    "build_skill_governance_projection",
]

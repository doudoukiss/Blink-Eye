"""Compact operator digest for persistent procedural skills."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, dict):
            return payload
    return {}


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _compact_skill(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": record.get("skill_id"),
        "skill_family_key": record.get("skill_family_key"),
        "template_fingerprint": record.get("template_fingerprint"),
        "title": record.get("title"),
        "purpose": record.get("purpose"),
        "goal_family": record.get("goal_family"),
        "status": record.get("status"),
        "confidence": record.get("confidence"),
        "supporting_trace_ids": list(record.get("supporting_trace_ids") or []),
        "supporting_outcome_ids": list(record.get("supporting_outcome_ids") or []),
        "required_capability_ids": list(record.get("required_capability_ids") or []),
    }


def build_procedural_skill_digest(
    *,
    procedural_skills: Any,
    max_current_skills: int = 8,
    max_candidate_skills: int = 8,
    max_recent_links: int = 8,
    max_failure_signatures: int = 10,
) -> dict[str, Any]:
    """Build a compact operator digest from raw procedural skills."""
    projection = _as_mapping(procedural_skills)
    skills = list(projection.get("skills") or [])
    goal_family_counts = Counter(
        str(record.get("goal_family"))
        for record in skills
        if str(record.get("goal_family", "")).strip()
    )

    def _skill_sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(record.get("updated_at") or ""),
            str(record.get("goal_family") or ""),
            str(record.get("template_fingerprint") or ""),
            str(record.get("skill_id") or ""),
        )

    current_skills = [
        _compact_skill(record)
        for record in sorted(
            (skill for skill in skills if skill.get("status") == "active"),
            key=_skill_sort_key,
            reverse=True,
        )[:max_current_skills]
    ]
    candidate_skills = [
        _compact_skill(record)
        for record in sorted(
            (skill for skill in skills if skill.get("status") == "candidate"),
            key=_skill_sort_key,
            reverse=True,
        )[:max_candidate_skills]
    ]
    recent_supersessions = [
        {
            "skill_id": record.get("skill_id"),
            "supersedes_skill_id": record.get("supersedes_skill_id"),
            "updated_at": record.get("updated_at"),
            "confidence": record.get("confidence"),
            "supporting_trace_ids": list(record.get("supporting_trace_ids") or []),
        }
        for record in sorted(
            (
                skill
                for skill in skills
                if skill.get("status") == "superseded" or skill.get("supersedes_skill_id")
            ),
            key=_skill_sort_key,
            reverse=True,
        )[:max_recent_links]
    ]
    recent_retirements = [
        {
            "skill_id": record.get("skill_id"),
            "retirement_reason": record.get("retirement_reason"),
            "retired_at": record.get("retired_at"),
            "supporting_trace_ids": list(record.get("supporting_trace_ids") or []),
            "supporting_outcome_ids": list(record.get("supporting_outcome_ids") or []),
        }
        for record in sorted(
            (skill for skill in skills if skill.get("status") == "retired"),
            key=_skill_sort_key,
            reverse=True,
        )[:max_recent_links]
    ]

    failure_signatures: list[dict[str, Any]] = []
    for skill in skills:
        for signature in skill.get("failure_signatures", []) or []:
            failure_signatures.append(
                {
                    "skill_id": skill.get("skill_id"),
                    "goal_family": skill.get("goal_family"),
                    "status": skill.get("status"),
                    "kind": signature.get("kind"),
                    "reason_code": signature.get("reason_code"),
                    "summary": signature.get("summary"),
                    "support_count": int((signature.get("details") or {}).get("support_count", 0)),
                    "support_trace_ids": list(signature.get("support_trace_ids") or []),
                    "support_outcome_ids": list(signature.get("support_outcome_ids") or []),
                }
            )
    top_failure_signatures = sorted(
        failure_signatures,
        key=lambda item: (
            -int(item.get("support_count", 0)),
            str(item.get("kind") or ""),
            str(item.get("reason_code") or ""),
            str(item.get("skill_id") or ""),
        ),
    )[:max_failure_signatures]

    return {
        "scope_type": projection.get("scope_type"),
        "scope_id": projection.get("scope_id"),
        "skill_counts": dict(projection.get("skill_counts") or {}),
        "confidence_band_counts": dict(projection.get("confidence_band_counts") or {}),
        "goal_family_counts": dict(sorted(goal_family_counts.items())),
        "current_skills": current_skills,
        "candidate_skills": candidate_skills,
        "recent_supersessions": recent_supersessions,
        "recent_retirements": recent_retirements,
        "top_failure_signatures": top_failure_signatures,
        "active_skill_ids": list(projection.get("active_skill_ids") or []),
        "candidate_skill_ids": list(projection.get("candidate_skill_ids") or []),
        "superseded_skill_ids": list(projection.get("superseded_skill_ids") or []),
        "retired_skill_ids": list(projection.get("retired_skill_ids") or []),
    }


__all__ = ["build_procedural_skill_digest"]

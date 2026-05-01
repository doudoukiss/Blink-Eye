"""Task-aware context selection and budgeting for Blink continuity state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from blink.brain.context.budgets import BrainContextBudgetProfile, approximate_token_count
from blink.brain.context.policy import BrainContextTask, get_brain_context_mode_policy
from blink.brain.context_surfaces import (
    BrainContextSurfaceSnapshot,
    render_autobiography_summary,
    render_health_summary,
    render_recent_memory_summary,
    render_relationship_continuity_summary,
    render_user_profile_summary,
)
from blink.brain.memory_v2 import render_claim_summary
from blink.brain.presence import render_presence_summary
from blink.brain.projections import (
    render_agenda_summary,
    render_commitment_projection_summary,
    render_engagement_summary,
    render_heartbeat_summary,
    render_private_working_memory_summary,
    render_relationship_state_summary,
    render_scene_summary,
    render_working_context_summary,
)
from blink.transcriptions.language import Language


@dataclass(frozen=True)
class BrainSelectedSection:
    """One selected context section."""

    key: str
    title: str
    content: str
    estimated_tokens: int
    source: str


@dataclass(frozen=True)
class BrainContextSelectionDecision:
    """One audit record for a selected or dropped section."""

    section_key: str
    title: str
    selected: bool
    estimated_tokens: int
    reason: str
    decision_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrainContextSelectionTrace:
    """Inspectable selection trace for one context-selection run."""

    task: str
    budget_profile: BrainContextBudgetProfile
    total_candidate_tokens: int
    total_selected_tokens: int
    decisions: tuple[BrainContextSelectionDecision, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize the selection trace."""
        return {
            "task": self.task,
            "budget": {
                "task": self.budget_profile.task,
                "max_tokens": self.budget_profile.max_tokens,
                "section_caps": dict(self.budget_profile.section_caps),
                "dynamic_token_reserve": self.budget_profile.dynamic_token_reserve,
                "trace_verbosity": (
                    self.budget_profile.trace_verbosity.value
                    if self.budget_profile.trace_verbosity is not None
                    else None
                ),
            },
            "total_candidate_tokens": self.total_candidate_tokens,
            "total_selected_tokens": self.total_selected_tokens,
            "decisions": [
                {
                    "section_key": decision.section_key,
                    "title": decision.title,
                    "selected": decision.selected,
                    "estimated_tokens": decision.estimated_tokens,
                    "reason": decision.reason,
                    "decision_reason_codes": list(decision.decision_reason_codes),
                }
                for decision in self.decisions
            ],
        }


@dataclass(frozen=True)
class BrainSelectedContext:
    """Final selected context bundle plus audit trace."""

    task: BrainContextTask
    budget_profile: BrainContextBudgetProfile
    sections: tuple[BrainSelectedSection, ...]
    selection_trace: BrainContextSelectionTrace

    @property
    def estimated_tokens(self) -> int:
        """Return the total estimated selected-token count."""
        return sum(section.estimated_tokens for section in self.sections)

    def section(self, key: str) -> BrainSelectedSection | None:
        """Return one selected section by key."""
        for section in self.sections:
            if section.key == key:
                return section
        return None

    def render_prompt(self, *, header: str | None = None) -> str:
        """Render the selected sections into one prompt-safe text block."""
        parts: list[str] = []
        if header:
            parts.append(header.strip())
        for section in self.sections:
            parts.append(f"## {section.title}\n{section.content}".strip())
        return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True)
class _SectionCandidate:
    """Internal section candidate before selection."""

    key: str
    title: str
    content: str
    source: str
    decision_reason_codes: tuple[str, ...] = ()

    @property
    def estimated_tokens(self) -> int:
        """Return the deterministic estimated token count."""
        return approximate_token_count(self.content)


class BrainContextSelector:
    """Canonical task-aware selector over a raw context surface."""

    def select(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        task: BrainContextTask,
        language: Language,
        static_sections: dict[str, str] | None = None,
        static_section_reason_codes: dict[str, tuple[str, ...]] | None = None,
        budget_profile: BrainContextBudgetProfile | None = None,
    ) -> BrainSelectedContext:
        """Select task-aware context sections under one deterministic budget."""
        budget = (budget_profile or BrainContextBudgetProfile.for_task(task.value)).resolved(task)
        policy = get_brain_context_mode_policy(task)
        candidates = self.build_candidates(
            snapshot=snapshot,
            language=language,
            task=task,
            static_sections=static_sections or {},
            static_section_reason_codes=static_section_reason_codes or {},
            budget_profile=budget,
        )
        selected: list[BrainSelectedSection] = []
        decisions: list[BrainContextSelectionDecision] = []
        total_selected_tokens = 0

        for key in policy.static_section_keys:
            candidate = candidates.get(key)
            if candidate is None:
                decisions.append(
                    BrainContextSelectionDecision(
                        section_key=key,
                        title=_title_for_key(key),
                        selected=False,
                        estimated_tokens=0,
                        reason="not_available",
                        decision_reason_codes=("section_not_available",),
                    )
                )
                continue
            if not candidate.content.strip():
                decisions.append(
                    BrainContextSelectionDecision(
                        section_key=candidate.key,
                        title=candidate.title,
                        selected=False,
                        estimated_tokens=0,
                        reason="empty",
                        decision_reason_codes=(
                            "section_empty",
                            *candidate.decision_reason_codes,
                        ),
                    )
                )
                continue
            if total_selected_tokens + candidate.estimated_tokens > budget.max_tokens:
                decisions.append(
                    BrainContextSelectionDecision(
                        section_key=candidate.key,
                        title=candidate.title,
                        selected=False,
                        estimated_tokens=candidate.estimated_tokens,
                        reason="budget_exceeded",
                        decision_reason_codes=(
                            "section_budget_exceeded",
                            *candidate.decision_reason_codes,
                        ),
                    )
                )
                continue
            selected.append(
                BrainSelectedSection(
                    key=candidate.key,
                    title=candidate.title,
                    content=candidate.content,
                    estimated_tokens=candidate.estimated_tokens,
                    source=candidate.source,
                )
            )
            total_selected_tokens += candidate.estimated_tokens
            decisions.append(
                BrainContextSelectionDecision(
                    section_key=candidate.key,
                    title=candidate.title,
                    selected=True,
                    estimated_tokens=candidate.estimated_tokens,
                    reason="selected",
                    decision_reason_codes=(
                        "section_selected",
                        *candidate.decision_reason_codes,
                    ),
                )
            )

        trace = BrainContextSelectionTrace(
            task=task.value,
            budget_profile=budget,
            total_candidate_tokens=sum(
                candidate.estimated_tokens for candidate in candidates.values()
            ),
            total_selected_tokens=total_selected_tokens,
            decisions=tuple(decisions),
        )
        return BrainSelectedContext(
            task=task,
            budget_profile=budget,
            sections=tuple(selected),
            selection_trace=trace,
        )

    def build_candidates(
        self,
        *,
        snapshot: BrainContextSurfaceSnapshot,
        language: Language,
        task: BrainContextTask,
        static_sections: dict[str, str],
        budget_profile: BrainContextBudgetProfile,
        static_section_reason_codes: dict[str, tuple[str, ...]] | None = None,
    ) -> dict[str, _SectionCandidate]:
        """Build bounded section candidates for one context-compilation task."""
        budget = budget_profile.resolved(task)
        policy = get_brain_context_mode_policy(task)
        caps = budget.section_caps
        candidates: dict[str, _SectionCandidate] = {}
        static_reason_codes = static_section_reason_codes or {}

        for key, content in static_sections.items():
            normalized = (content or "").strip()
            if normalized or key in static_reason_codes:
                candidates[key] = _SectionCandidate(
                    key=key,
                    title=_title_for_key(key),
                    content=normalized,
                    source="static",
                    decision_reason_codes=static_reason_codes.get(key, ()),
                )

        candidates["presence"] = _candidate(
            "presence",
            render_presence_summary(snapshot.body, language),
        )
        candidates["scene"] = _candidate("scene", render_scene_summary(snapshot.scene, language))
        candidates["engagement"] = _candidate(
            "engagement",
            render_engagement_summary(snapshot.engagement, language),
        )
        candidates["working_context"] = _candidate(
            "working_context",
            render_working_context_summary(snapshot.working_context, language),
        )
        candidates["private_working_memory"] = _candidate(
            "private_working_memory",
            render_private_working_memory_summary(snapshot.private_working_memory, language),
        )
        candidates["agenda"] = _candidate(
            "agenda", render_agenda_summary(snapshot.agenda, language)
        )
        candidates["heartbeat"] = _candidate(
            "heartbeat",
            render_heartbeat_summary(snapshot.heartbeat, language),
        )
        user_profile_summary = render_user_profile_summary(snapshot, language)
        if user_profile_summary.strip() in {"无", "None"}:
            user_profile_summary = ""
        candidates["user_profile"] = _candidate("user_profile", user_profile_summary)
        candidates["relationship_state"] = _candidate(
            "relationship_state",
            render_relationship_state_summary(snapshot.relationship_state, language),
        )
        candidates["relationship_continuity"] = _candidate(
            "relationship_continuity",
            render_relationship_continuity_summary(snapshot, language),
        )
        candidates["autobiography"] = _candidate(
            "autobiography",
            _render_autobiography_entries(
                snapshot=snapshot,
                language=language,
                limit=caps.get("autobiography", 3),
            ),
        )
        candidates["memory_health"] = _candidate(
            "memory_health",
            render_health_summary(snapshot, language),
        )
        candidates["recent_memory"] = _candidate(
            "recent_memory",
            render_recent_memory_summary(
                tuple(
                    snapshot.recent_memory[: caps.get("recent_memory", len(snapshot.recent_memory))]
                ),
                language,
            ),
        )
        candidates["episodic_fallback"] = _candidate(
            "episodic_fallback",
            render_recent_memory_summary(
                tuple(
                    snapshot.episodic_fallback[
                        : caps.get("episodic_fallback", len(snapshot.episodic_fallback))
                    ]
                ),
                language,
            ),
        )
        candidates["current_claims"] = _candidate(
            "current_claims",
            _render_claims(
                claims=snapshot.current_claims,
                language=language,
                limit=caps.get("current_claims", len(snapshot.current_claims)),
                include_status=False,
            ),
        )
        candidates["historical_claims"] = _candidate(
            "historical_claims",
            _render_claims(
                claims=snapshot.historical_claims,
                language=language,
                limit=caps.get("historical_claims", len(snapshot.historical_claims)),
                include_status=True,
            ),
        )
        candidates["claim_provenance"] = _candidate(
            "claim_provenance",
            _render_claim_provenance(
                claims=(
                    list(snapshot.current_claims[: caps.get("claim_provenance", 0)])
                    + list(snapshot.historical_claims[: caps.get("claim_provenance", 0)])
                ),
                language=language,
                limit=caps.get("claim_provenance", 6),
            ),
        )
        candidates["claim_supersessions"] = _candidate(
            "claim_supersessions",
            _render_claim_supersessions(
                supersessions=snapshot.claim_supersessions,
                language=language,
                limit=caps.get("claim_supersessions", len(snapshot.claim_supersessions)),
            ),
        )
        candidates["core_blocks"] = _candidate(
            "core_blocks",
            _render_core_blocks(
                core_blocks=snapshot.core_blocks,
                language=language,
                limit=caps.get("core_blocks", len(snapshot.core_blocks)),
            ),
        )
        candidates["commitment_projection"] = _candidate(
            "commitment_projection",
            _render_commitment_details(snapshot, language),
        )

        if not policy.include_historical_claims:
            candidates["historical_claims"] = _candidate("historical_claims", "")

        return candidates


def _candidate(key: str, content: str) -> _SectionCandidate:
    return _SectionCandidate(
        key=key,
        title=_title_for_key(key),
        content=(content or "").strip(),
        source="dynamic",
    )


def _title_for_key(key: str) -> str:
    return {
        "policy": "Policy",
        "identity": "Identity",
        "style": "Style",
        "persona_expression": "Persona Expression",
        "teaching_knowledge": "Teaching Knowledge",
        "capabilities": "Capabilities",
        "internal_capabilities": "Internal Capabilities",
        "presence": "Presence",
        "scene": "Scene",
        "engagement": "Engagement",
        "working_context": "Working Context",
        "private_working_memory": "Private Working Memory",
        "agenda": "Agenda",
        "heartbeat": "Heartbeat",
        "user_profile": "User Profile",
        "relationship_state": "Relationship State",
        "relationship_continuity": "Relationship Continuity",
        "autobiography": "Autobiography",
        "memory_health": "Memory Health",
        "recent_memory": "Relevant Long-Term Memory",
        "episodic_fallback": "Relevant Episodes",
        "current_claims": "Current Claims",
        "historical_claims": "Historical Claims",
        "claim_provenance": "Claim Provenance",
        "claim_supersessions": "Claim Corrections",
        "core_blocks": "Core Blocks",
        "commitment_projection": "Commitment Ledger",
    }.get(key, key.replace("_", " ").title())


def _render_autobiography_entries(
    *,
    snapshot: BrainContextSurfaceSnapshot,
    language: Language,
    limit: int,
) -> str:
    if limit <= 0:
        return ""
    parts: list[str] = []
    summary = render_autobiography_summary(snapshot, language)
    if summary and summary not in {"无", "None"}:
        parts.append(summary)
    for entry in snapshot.autobiography[:limit]:
        line = f"- {entry.entry_kind}: {entry.rendered_summary}"
        if line not in parts:
            parts.append(line)
    return "\n".join(parts)


def _render_claims(
    *,
    claims: Iterable,
    language: Language,
    limit: int,
    include_status: bool,
) -> str:
    items = list(claims)[:limit]
    if not items:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    parts: list[str] = []
    for claim in items:
        summary = render_claim_summary(claim)
        governance_tags: list[str] = []
        currentness = str(getattr(claim, "effective_currentness_status", "")).strip()
        review_state = str(getattr(claim, "effective_review_state", "")).strip()
        truth_status = str(getattr(claim, "status", "")).strip()
        reason_codes = [
            str(code).strip()
            for code in getattr(claim, "governance_reason_codes", ())
            if str(code).strip()
        ]
        if currentness and currentness != "current":
            governance_tags.append(currentness)
        if review_state and review_state != "none":
            governance_tags.append(f"review={review_state}")
        if truth_status and truth_status not in {"active", "current"}:
            governance_tags.append(f"truth={truth_status}")
        if reason_codes:
            governance_tags.append("reason=" + ",".join(reason_codes[:2]))
        if include_status:
            summary = f"{summary} [{claim.status}]"
        if governance_tags:
            summary = f"{summary} [{' ; '.join(governance_tags)}]".replace(" ; ", "][")
        parts.append(f"- {summary}")
    return "\n".join(parts)


def _render_claim_provenance(*, claims: list, language: Language, limit: int) -> str:
    if limit <= 0:
        return ""
    parts: list[str] = []
    for claim in claims[:limit]:
        source = claim.source_event_id or "unknown"
        validity = claim.valid_to or "current"
        label = render_claim_summary(claim)
        if language.value.lower().startswith(("zh", "cmn")):
            parts.append(f"- {label} | 来源={source} | 生效={claim.valid_from} | 截止={validity}")
        else:
            parts.append(
                f"- {label} | source={source} | valid_from={claim.valid_from} | valid_to={validity}"
            )
    if not parts:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    return "\n".join(parts)


def _render_claim_supersessions(*, supersessions: tuple, language: Language, limit: int) -> str:
    records = list(supersessions)[:limit]
    if not records:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    if language.value.lower().startswith(("zh", "cmn")):
        return "\n".join(
            f"- {record.prior_claim_id} -> {record.new_claim_id} ({record.reason})"
            for record in records
        )
    return "\n".join(
        f"- {record.prior_claim_id} -> {record.new_claim_id} ({record.reason})"
        for record in records
    )


def _render_core_blocks(*, core_blocks: dict[str, object], language: Language, limit: int) -> str:
    records = list(core_blocks.values())[:limit]
    if not records:
        return "无" if language.value.lower().startswith(("zh", "cmn")) else "None"
    parts: list[str] = []
    for record in records:
        content = json.dumps(record.content, ensure_ascii=False, sort_keys=True)
        parts.append(f"- {record.block_kind}: {content}")
    return "\n".join(parts)


def _render_commitment_details(snapshot: BrainContextSurfaceSnapshot, language: Language) -> str:
    projection = snapshot.commitment_projection
    lines = [render_commitment_projection_summary(projection, language)]
    detail_lines: list[str] = []
    wake_lines: list[str] = []
    for record in (
        list(projection.active_commitments[:2])
        + list(projection.blocked_commitments[:3])
        + list(projection.deferred_commitments[:3])
    ):
        if language.value.lower().startswith(("zh", "cmn")):
            detail_lines.append(
                f"- {record.title}: family={record.goal_family}, scope={record.scope_type}, "
                f"goal={record.current_goal_id or '无'}, rev={record.plan_revision}, resume={record.resume_count}"
            )
            if record.blocked_reason is not None:
                detail_lines.append(f"  blocker={record.blocked_reason.summary}")
        else:
            detail_lines.append(
                f"- {record.title}: family={record.goal_family}, scope={record.scope_type}, "
                f"goal={record.current_goal_id or 'None'}, rev={record.plan_revision}, resume={record.resume_count}"
            )
            if record.blocked_reason is not None:
                detail_lines.append(f"  blocker={record.blocked_reason.summary}")
        for wake in record.wake_conditions[:2]:
            if language.value.lower().startswith(("zh", "cmn")):
                wake_lines.append(f"- {record.title}: 唤醒条件={wake.summary}")
            else:
                wake_lines.append(f"- {record.title}: wake={wake.summary}")
    if detail_lines:
        title = (
            "承诺细节" if language.value.lower().startswith(("zh", "cmn")) else "Commitment details"
        )
        lines.append(f"{title}:\n" + "\n".join(detail_lines))
    if wake_lines:
        title = (
            "唤醒条件" if language.value.lower().startswith(("zh", "cmn")) else "Wake conditions"
        )
        lines.append(f"{title}:\n" + "\n".join(wake_lines))
    return "\n\n".join(line for line in lines if line.strip())


__all__ = [
    "BrainContextSelectionDecision",
    "BrainContextSelectionTrace",
    "BrainContextSelector",
    "BrainContextTask",
    "BrainSelectedContext",
    "BrainSelectedSection",
]

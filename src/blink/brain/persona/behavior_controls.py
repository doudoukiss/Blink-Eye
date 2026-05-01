"""Replayable user behavior controls for Blink expression compilation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from blink.brain.memory_v2.core_blocks import BrainCoreMemoryBlockKind

_SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3})
_BLOCK_KIND = BrainCoreMemoryBlockKind.BEHAVIOR_CONTROL_PROFILE.value
_SCOPE_TYPE = "relationship"
_PUBLIC_FIELDS = frozenset(
    {
        "schema_version",
        "user_id",
        "agent_id",
        "response_depth",
        "directness",
        "warmth",
        "teaching_mode",
        "memory_use",
        "initiative_mode",
        "evidence_visibility",
        "correction_mode",
        "explanation_structure",
        "challenge_style",
        "voice_mode",
        "question_budget",
        "humor_mode",
        "vividness_mode",
        "sophistication_mode",
        "character_presence",
        "story_mode",
        "updated_at",
        "source",
        "reason_codes",
    }
)
_UPDATE_FIELDS = frozenset(
    {
        "response_depth",
        "directness",
        "warmth",
        "teaching_mode",
        "memory_use",
        "initiative_mode",
        "evidence_visibility",
        "correction_mode",
        "explanation_structure",
        "challenge_style",
        "voice_mode",
        "question_budget",
        "humor_mode",
        "vividness_mode",
        "sophistication_mode",
        "character_presence",
        "story_mode",
    }
)
_VALUE_SETS = {
    "response_depth": frozenset({"concise", "balanced", "deep"}),
    "directness": frozenset({"gentle", "balanced", "rigorous"}),
    "warmth": frozenset({"low", "medium", "high"}),
    "teaching_mode": frozenset({"auto", "direct", "clarify", "walkthrough", "socratic"}),
    "memory_use": frozenset({"minimal", "balanced", "continuity_rich"}),
    "initiative_mode": frozenset({"minimal", "balanced", "proactive"}),
    "evidence_visibility": frozenset({"hidden", "compact", "rich"}),
    "correction_mode": frozenset({"gentle", "precise", "rigorous"}),
    "explanation_structure": frozenset({"answer_first", "walkthrough", "socratic"}),
    "challenge_style": frozenset({"avoid", "gentle", "direct"}),
    "voice_mode": frozenset({"off", "concise", "balanced"}),
    "question_budget": frozenset({"low", "medium", "high"}),
    "humor_mode": frozenset({"off", "subtle", "witty", "playful"}),
    "vividness_mode": frozenset({"spare", "balanced", "vivid"}),
    "sophistication_mode": frozenset({"plain", "smart", "sophisticated"}),
    "character_presence": frozenset({"minimal", "balanced", "character_rich"}),
    "story_mode": frozenset({"off", "light", "recurring_motifs"}),
}
BEHAVIOR_CONTROL_UPDATE_FIELDS = tuple(
    (
        "response_depth",
        "directness",
        "warmth",
        "teaching_mode",
        "memory_use",
        "initiative_mode",
        "evidence_visibility",
        "correction_mode",
        "explanation_structure",
        "challenge_style",
        "voice_mode",
        "question_budget",
        "humor_mode",
        "vividness_mode",
        "sophistication_mode",
        "character_presence",
        "story_mode",
    )
)
BEHAVIOR_CONTROL_FIELD_OPTIONS = {
    field: tuple(sorted(values)) for field, values in _VALUE_SETS.items()
}
_FIELD_ALIASES = {
    "continuity-rich": "continuity_rich",
    "continuity rich": "continuity_rich",
    "answer-first": "answer_first",
    "answer first": "answer_first",
    "character-rich": "character_rich",
    "character rich": "character_rich",
    "recurring-motifs": "recurring_motifs",
    "recurring motifs": "recurring_motifs",
}
_FORBIDDEN_KEYS = frozenset(
    {
        "age",
        "childhood",
        "consciousness",
        "dependency",
        "dependent",
        "exclusive",
        "exclusivity",
        "fake_human_identity",
        "family",
        "hardware",
        "hardware_control",
        "human_identity",
        "human_identity_claims_allowed",
        "api_key",
        "authorization",
        "base_url",
        "developer_message",
        "developer_prompt",
        "model",
        "motor",
        "openai_api_key",
        "persona",
        "persona_prompt",
        "provider",
        "prompt",
        "raw_system_message",
        "robot_head",
        "romance",
        "romantic",
        "secret",
        "schooling",
        "sentience",
        "servo",
        "sexual",
        "sexuality",
        "system_message",
        "system_prompt",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_value(value: Any) -> str:
    normalized = _normalized_text(value).lower().replace("-", "_").replace(" ", "_")
    return _FIELD_ALIASES.get(normalized, normalized)


def _dedupe(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def behavior_control_relationship_scope_id(*, agent_id: str, user_id: str) -> str:
    """Return the relationship scope id for one behavior-control profile."""
    return f"{_normalized_text(agent_id) or 'blink/main'}:{_normalized_text(user_id)}"


@dataclass(frozen=True)
class BrainBehaviorControlProfile:
    """Relationship-scoped user controls that tune expression without changing persona truth."""

    schema_version: int
    user_id: str
    agent_id: str
    response_depth: str
    directness: str
    warmth: str
    teaching_mode: str
    memory_use: str
    initiative_mode: str
    evidence_visibility: str
    correction_mode: str
    explanation_structure: str
    challenge_style: str
    voice_mode: str
    question_budget: str
    humor_mode: str
    vividness_mode: str
    sophistication_mode: str
    character_presence: str
    story_mode: str
    updated_at: str
    source: str
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the behavior-control profile."""
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "response_depth": self.response_depth,
            "directness": self.directness,
            "warmth": self.warmth,
            "teaching_mode": self.teaching_mode,
            "memory_use": self.memory_use,
            "initiative_mode": self.initiative_mode,
            "evidence_visibility": self.evidence_visibility,
            "correction_mode": self.correction_mode,
            "explanation_structure": self.explanation_structure,
            "challenge_style": self.challenge_style,
            "voice_mode": self.voice_mode,
            "question_budget": self.question_budget,
            "humor_mode": self.humor_mode,
            "vividness_mode": self.vividness_mode,
            "sophistication_mode": self.sophistication_mode,
            "character_presence": self.character_presence,
            "story_mode": self.story_mode,
            "updated_at": self.updated_at,
            "source": self.source,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainBehaviorControlProfile | None":
        """Hydrate one profile from JSON-like data."""
        if not isinstance(data, dict):
            return None
        unknown = sorted(set(data) - _PUBLIC_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported behavior-control fields: {unknown!r}")
        values = {
            field: _validate_control_value(field, data.get(field))
            for field in sorted(_UPDATE_FIELDS)
        }
        user_id = _normalized_text(data.get("user_id"))
        agent_id = _normalized_text(data.get("agent_id")) or "blink/main"
        if not user_id:
            raise ValueError("Behavior controls require user_id.")
        schema_version = int(data.get("schema_version") or _SCHEMA_VERSION)
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported behavior-control schema version: {schema_version}")
        return cls(
            schema_version=_SCHEMA_VERSION,
            user_id=user_id,
            agent_id=agent_id,
            response_depth=values["response_depth"],
            directness=values["directness"],
            warmth=values["warmth"],
            teaching_mode=values["teaching_mode"],
            memory_use=values["memory_use"],
            initiative_mode=values["initiative_mode"],
            evidence_visibility=values["evidence_visibility"],
            correction_mode=values["correction_mode"],
            explanation_structure=values["explanation_structure"],
            challenge_style=values["challenge_style"],
            voice_mode=values["voice_mode"],
            question_budget=values["question_budget"],
            humor_mode=values["humor_mode"],
            vividness_mode=values["vividness_mode"],
            sophistication_mode=values["sophistication_mode"],
            character_presence=values["character_presence"],
            story_mode=values["story_mode"],
            updated_at=_normalized_text(data.get("updated_at")),
            source=_normalized_text(data.get("source")) or "default",
            reason_codes=_dedupe(data.get("reason_codes", ())),
        )


@dataclass(frozen=True)
class BrainBehaviorControlUpdateResult:
    """Result for one behavior-control update request."""

    accepted: bool
    applied: bool
    profile: BrainBehaviorControlProfile | None
    rejected_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    schema_version: int = _SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize the update result."""
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted,
            "applied": self.applied,
            "profile": self.profile.as_dict() if self.profile else None,
            "compiled_effect_summary": render_behavior_control_effect_summary(self.profile)
            if self.profile
            else "",
            "rejected_fields": list(self.rejected_fields),
            "reason_codes": list(self.reason_codes),
        }


def _validate_control_value(field: str, value: Any) -> str:
    normalized = _normalized_value(value)
    if field not in _VALUE_SETS:
        raise ValueError(f"Unsupported behavior-control field: {field}")
    if not normalized:
        normalized = _default_values()[field]
    if normalized not in _VALUE_SETS[field]:
        raise ValueError(f"Unsupported behavior-control value for {field}: {normalized}")
    return normalized


def _default_values() -> dict[str, str]:
    return {
        "response_depth": "balanced",
        "directness": "balanced",
        "warmth": "medium",
        "teaching_mode": "auto",
        "memory_use": "balanced",
        "initiative_mode": "balanced",
        "evidence_visibility": "compact",
        "correction_mode": "precise",
        "explanation_structure": "answer_first",
        "challenge_style": "gentle",
        "voice_mode": "balanced",
        "question_budget": "medium",
        "humor_mode": "witty",
        "vividness_mode": "vivid",
        "sophistication_mode": "sophisticated",
        "character_presence": "character_rich",
        "story_mode": "light",
    }


def default_behavior_control_profile(
    *,
    user_id: str,
    agent_id: str = "blink/main",
    updated_at: str = "",
    source: str = "default",
    reason_codes: tuple[str, ...] = ("behavior_controls_defaulted",),
) -> BrainBehaviorControlProfile:
    """Return the safe default behavior-control profile."""
    values = _default_values()
    return BrainBehaviorControlProfile(
        schema_version=_SCHEMA_VERSION,
        user_id=_normalized_text(user_id),
        agent_id=_normalized_text(agent_id) or "blink/main",
        updated_at=_normalized_text(updated_at),
        source=_normalized_text(source) or "default",
        reason_codes=_dedupe(reason_codes),
        **values,
    )


def _profile_from_content(
    content: dict[str, Any],
    *,
    user_id: str,
    agent_id: str,
) -> BrainBehaviorControlProfile:
    data = dict(content)
    data.setdefault("user_id", user_id)
    data.setdefault("agent_id", agent_id)
    profile = BrainBehaviorControlProfile.from_dict(data)
    if profile is None:
        raise ValueError("Behavior controls content must be an object.")
    return profile


def load_behavior_control_profile(
    *,
    store,
    session_ids,
) -> BrainBehaviorControlProfile:
    """Load the current behavior controls, defaulting safely when absent or invalid."""
    user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    agent_id = _normalized_text(getattr(session_ids, "agent_id", "")) or "blink/main"
    scope_id = behavior_control_relationship_scope_id(agent_id=agent_id, user_id=user_id)
    reader = getattr(store, "get_current_core_memory_block", None)
    if not callable(reader):
        return default_behavior_control_profile(
            user_id=user_id,
            agent_id=agent_id,
            reason_codes=("behavior_controls_defaulted", "behavior_controls_store_unavailable"),
        )
    record = reader(
        block_kind=_BLOCK_KIND,
        scope_type=_SCOPE_TYPE,
        scope_id=scope_id,
    )
    if record is None:
        return default_behavior_control_profile(user_id=user_id, agent_id=agent_id)
    try:
        profile = _profile_from_content(record.content, user_id=user_id, agent_id=agent_id)
    except (TypeError, ValueError):
        return default_behavior_control_profile(
            user_id=user_id,
            agent_id=agent_id,
            reason_codes=("behavior_controls_defaulted", "behavior_controls_invalid"),
        )
    return BrainBehaviorControlProfile(
        **{
            **profile.as_dict(),
            "reason_codes": _dedupe(
                (
                    "behavior_controls_loaded",
                    *_dedupe(profile.reason_codes),
                )
            ),
        }
    )


def _rejected_update(
    *,
    rejected_fields: Iterable[Any],
    reason_codes: tuple[str, ...],
) -> BrainBehaviorControlUpdateResult:
    return BrainBehaviorControlUpdateResult(
        accepted=False,
        applied=False,
        profile=None,
        rejected_fields=_dedupe(rejected_fields),
        reason_codes=_dedupe(("behavior_controls_update_rejected", *reason_codes)),
    )


def _validate_update_payload(updates: dict[str, Any]) -> tuple[dict[str, str], tuple[str, ...]]:
    normalized: dict[str, str] = {}
    rejected: list[str] = []
    for key, value in sorted(updates.items()):
        normalized_key = _normalized_value(key)
        if normalized_key in _FORBIDDEN_KEYS:
            rejected.append(normalized_key)
            continue
        if normalized_key not in _UPDATE_FIELDS:
            rejected.append(normalized_key)
            continue
        try:
            normalized[normalized_key] = _validate_control_value(normalized_key, value)
        except ValueError:
            rejected.append(normalized_key)
    return normalized, tuple(rejected)


def validate_behavior_control_update_payload(
    updates: dict[str, Any] | None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Validate and normalize a public behavior-control update payload."""
    if not isinstance(updates, dict):
        return {}, ("payload",)
    return _validate_update_payload(updates)


def apply_behavior_control_update(
    *,
    store,
    session_ids,
    updates: dict[str, Any],
    source: str = "browser",
) -> BrainBehaviorControlUpdateResult:
    """Merge and persist one behavior-control update."""
    if not isinstance(updates, dict):
        return _rejected_update(
            rejected_fields=("payload",),
            reason_codes=("behavior_controls_payload_invalid",),
        )
    normalized_updates, rejected_fields = _validate_update_payload(updates)
    if rejected_fields:
        return _rejected_update(
            rejected_fields=rejected_fields,
            reason_codes=("behavior_controls_fields_invalid",),
        )
    user_id = _normalized_text(getattr(session_ids, "user_id", ""))
    agent_id = _normalized_text(getattr(session_ids, "agent_id", "")) or "blink/main"
    current = load_behavior_control_profile(store=store, session_ids=session_ids)
    next_data = current.as_dict()
    next_data.update(normalized_updates)
    next_data.update(
        {
            "user_id": user_id,
            "agent_id": agent_id,
            "updated_at": _utc_now(),
            "source": _normalized_text(source) or "browser",
            "reason_codes": [
                "behavior_controls_loaded",
                "behavior_controls_updated",
            ],
        }
    )
    next_profile = BrainBehaviorControlProfile.from_dict(next_data)
    if next_profile is None:
        return _rejected_update(
            rejected_fields=("profile",),
            reason_codes=("behavior_controls_profile_invalid",),
        )
    current_comparable = {
        key: value for key, value in current.as_dict().items() if key in _UPDATE_FIELDS
    }
    next_comparable = {
        key: value for key, value in next_profile.as_dict().items() if key in _UPDATE_FIELDS
    }
    if current_comparable == next_comparable:
        return BrainBehaviorControlUpdateResult(
            accepted=True,
            applied=False,
            profile=current,
            rejected_fields=(),
            reason_codes=_dedupe(("behavior_controls_update_accepted", "behavior_controls_noop")),
        )
    scope_id = behavior_control_relationship_scope_id(agent_id=agent_id, user_id=user_id)
    store.upsert_core_memory_block(
        block_kind=_BLOCK_KIND,
        scope_type=_SCOPE_TYPE,
        scope_id=scope_id,
        content=next_profile.as_dict(),
        source_event_id=None,
        event_context={
            "agent_id": agent_id,
            "user_id": user_id,
            "session_id": _normalized_text(getattr(session_ids, "session_id", "")),
            "thread_id": _normalized_text(getattr(session_ids, "thread_id", "")),
            "source": _normalized_text(source) or "browser",
            "correlation_id": f"behavior_controls:{scope_id}",
        },
    )
    return BrainBehaviorControlUpdateResult(
        accepted=True,
        applied=True,
        profile=next_profile,
        rejected_fields=(),
        reason_codes=_dedupe(
            (
                "behavior_controls_update_accepted",
                "behavior_controls_persisted",
                "behavior_controls_scope:relationship",
            )
        ),
    )


def render_behavior_control_effect_summary(
    profile: BrainBehaviorControlProfile | None,
) -> str:
    """Render compact public-safe behavior-control effects."""
    if profile is None:
        return ""
    return (
        f"depth={profile.response_depth}; directness={profile.directness}; "
        f"warmth={profile.warmth}; teaching={profile.teaching_mode}; "
        f"memory={profile.memory_use}; initiative={profile.initiative_mode}; "
        f"evidence={profile.evidence_visibility}; correction={profile.correction_mode}; "
        f"structure={profile.explanation_structure}; challenge={profile.challenge_style}; "
        f"voice={profile.voice_mode}; questions={profile.question_budget}; "
        f"humor={profile.humor_mode}; vividness={profile.vividness_mode}; "
        f"sophistication={profile.sophistication_mode}; "
        f"character={profile.character_presence}; story={profile.story_mode}"
    )


__all__ = [
    "BEHAVIOR_CONTROL_FIELD_OPTIONS",
    "BEHAVIOR_CONTROL_UPDATE_FIELDS",
    "BrainBehaviorControlProfile",
    "BrainBehaviorControlUpdateResult",
    "apply_behavior_control_update",
    "behavior_control_relationship_scope_id",
    "default_behavior_control_profile",
    "load_behavior_control_profile",
    "render_behavior_control_effect_summary",
    "validate_behavior_control_update_payload",
]

"""Compiler helpers for Blink's structured persona kernel."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from blink.brain.persona.policy import (
    PERSONA_INVARIANT_GUARDRAILS,
    PERSONA_RELATIONSHIP_STYLE_NAMESPACES,
    PERSONA_TEACHING_PROFILE_NAMESPACES,
    BrainPersonaModality,
    BrainPersonaTaskMode,
    resolve_persona_modality,
    resolve_persona_task_mode,
)
from blink.brain.persona.schema import (
    PersonaCharterSpec,
    PersonaDefaultsBundle,
    PersonaTraitVectorSpec,
    RelationshipStyleStateSpec,
    SelfPersonaCoreSpec,
    TeachingProfileStateSpec,
    VoiceProfileSpec,
)
from blink.brain.persona.state import BrainPersonaState

_JSON_FENCE_PATTERN = re.compile(r"```json\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)
_PERSONA_BLOCKS = ("persona", "voice", "relationship_style", "teaching_style")
_COMPILED_PERSONA_BLOCKS = ("persona", "voice")
_COMPILED_SELF_PERSONA_BLOCKS = ("persona", "voice", "relationship_style", "teaching_style")


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_list(values: Iterable[Any]) -> list[str]:
    return [normalized for value in values if (normalized := _normalized_text(value))]


def _normalized_mapping(values: Mapping[str, str]) -> dict[str, str]:
    return {
        normalized_key: normalized_value
        for key, value in values.items()
        if (normalized_key := _normalized_text(key))
        and (normalized_value := _normalized_text(value))
    }


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_charter(charter: PersonaCharterSpec) -> PersonaCharterSpec:
    payload = charter.model_dump()
    payload.update(
        {
            "persona_profile_id": _normalized_text(payload["persona_profile_id"]),
            "display_title": _normalized_text(payload["display_title"]),
            "canonical_name": _normalized_text(payload["canonical_name"]),
            "ontological_status": _normalized_text(payload["ontological_status"]),
            "archetype": _normalized_text(payload["archetype"]),
            "social_role": _normalized_text(payload["social_role"]),
            "core_values": _normalized_list(payload["core_values"]),
            "intellectual_signature": _normalized_list(payload["intellectual_signature"]),
            "cultural_signature": _normalized_list(payload["cultural_signature"]),
            "relational_posture": _normalized_list(payload["relational_posture"]),
            "humor_style": _normalized_text(payload["humor_style"]),
            "teaching_style": _normalized_text(payload["teaching_style"]),
            "vulnerability_policy": _normalized_text(payload["vulnerability_policy"]),
            "intimacy_policy": _normalized_text(payload["intimacy_policy"]),
            "safety_boundaries": _normalized_list(payload["safety_boundaries"]),
            "non_goals": _normalized_list(payload["non_goals"]),
        }
    )
    return PersonaCharterSpec.model_validate(payload)


def _normalize_traits(traits: PersonaTraitVectorSpec) -> PersonaTraitVectorSpec:
    return PersonaTraitVectorSpec.model_validate(traits.model_dump())


def _normalize_voice_profile(voice_profile: VoiceProfileSpec) -> VoiceProfileSpec:
    payload = voice_profile.model_dump()
    payload.update(
        {
            "default_voice_id": _normalized_text(payload.get("default_voice_id")) or None,
            "emphasis_style": _normalized_text(payload["emphasis_style"]),
            "interruption_strategy": _normalized_text(payload["interruption_strategy"]),
            "multilingual_style": _normalized_mapping(payload["multilingual_style"]),
        }
    )
    return VoiceProfileSpec.model_validate(payload)


def _fingerprint(content: str) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def _relationship_scope_id(*, store: Any, user_id: str, agent_id: str | None) -> str:
    if hasattr(store, "_relationship_scope_id"):
        return store._relationship_scope_id(agent_id=agent_id, user_id=user_id)
    return f"{agent_id or 'blink/main'}:{user_id}"


def _memory_value_text(record: Any) -> str:
    value = getattr(record, "value", {})
    if isinstance(value, dict):
        for key in ("value", "text", "summary", "mode", "domain", "pattern"):
            normalized = _normalized_text(value.get(key))
            if normalized:
                return normalized
    return ""


def _memory_rendered_text(record: Any) -> str:
    return _normalized_text(getattr(record, "rendered_text", ""))


def _memory_text(record: Any) -> str:
    return _memory_value_text(record) or _memory_rendered_text(record)


def _group_semantic_memories(
    *,
    store: Any,
    user_id: str,
    namespaces: tuple[str, ...],
) -> dict[str, list[Any]]:
    records = store.semantic_memories(
        user_id=user_id,
        namespaces=namespaces,
        limit=64,
        include_stale=False,
    )
    grouped = {namespace: [] for namespace in namespaces}
    for record in records:
        namespace = _normalized_text(getattr(record, "namespace", ""))
        if namespace in grouped:
            grouped[namespace].append(record)
    return grouped


def _present_namespaces(grouped_records: Mapping[str, list[Any]]) -> list[str]:
    return [namespace for namespace, records in grouped_records.items() if records]


def _default_collaboration_style(default_posture: list[str]) -> str:
    positive = [item for item in default_posture if not item.startswith("non-")]
    return " ".join(positive[:3]) or "warm collaborative"


def _default_emotional_tone(default_posture: list[str]) -> str:
    positive = [item for item in default_posture if not item.startswith("non-")]
    return " ".join(positive[:2]) or "warm collaborative"


def _safe_boundary_list(default_posture: list[str], self_disclosure_policy: str) -> list[str]:
    boundaries = [item for item in default_posture if item.startswith("non-")]
    if "personal history" in self_disclosure_policy.lower():
        boundaries.append("no fabricated personal history")
    return _dedupe_preserve_order(boundaries)


def _interaction_style_hints(grouped_records: Mapping[str, list[Any]]) -> list[str]:
    hints: list[str] = []
    for namespace in ("interaction.style", "interaction.preference"):
        for record in grouped_records.get(namespace, []):
            text = _memory_rendered_text(record) or _memory_text(record)
            if text:
                hints.append(text)
    return _dedupe_preserve_order(hints)[:4]


def extract_structured_doc_payload(block_text: str) -> dict[str, Any]:
    """Extract exactly one structured JSON payload from one Markdown doc."""
    matches = _JSON_FENCE_PATTERN.findall(str(block_text or "").replace("\r\n", "\n"))
    if not matches:
        raise ValueError("Expected exactly one fenced json block, found none.")
    if len(matches) > 1:
        raise ValueError("Expected exactly one fenced json block, found multiple.")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload in structured persona document.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Structured persona document payload must decode to an object.")
    return payload


def load_persona_defaults(agent_blocks: Mapping[str, str]) -> PersonaDefaultsBundle:
    """Load and validate structured persona defaults from agent blocks."""
    missing = [key for key in _PERSONA_BLOCKS if not _normalized_text(agent_blocks.get(key))]
    if missing:
        raise ValueError(f"Missing required persona default blocks: {', '.join(missing)}")

    try:
        from blink.brain.persona.schema import (
            _PersonaDocumentSpec,
            _RelationshipDocumentSpec,
            _TeachingDocumentSpec,
            _VoiceDocumentSpec,
        )

        persona_doc = _PersonaDocumentSpec.model_validate(
            extract_structured_doc_payload(agent_blocks["persona"])
        )
        voice_doc = _VoiceDocumentSpec.model_validate(
            extract_structured_doc_payload(agent_blocks["voice"])
        )
        relationship_doc = _RelationshipDocumentSpec.model_validate(
            extract_structured_doc_payload(agent_blocks["relationship_style"])
        )
        teaching_doc = _TeachingDocumentSpec.model_validate(
            extract_structured_doc_payload(agent_blocks["teaching_style"])
        )
    except ValidationError as exc:
        raise ValueError("Invalid structured persona defaults payload.") from exc

    return PersonaDefaultsBundle(
        charter=persona_doc.charter,
        traits=persona_doc.traits,
        voice_profile=voice_doc.voice_profile,
        relationship_defaults=relationship_doc.relationship_defaults,
        teaching_defaults=teaching_doc.teaching_defaults,
    )


@dataclass(frozen=True)
class BrainPersonaFrame:
    """Deterministically compiled persona surface for Blink."""

    charter: PersonaCharterSpec
    traits: PersonaTraitVectorSpec
    voice_profile: VoiceProfileSpec
    state: BrainPersonaState
    task_mode: str
    modality: str
    compiled_from_blocks: tuple[str, ...]
    guardrails: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the persona frame."""
        return {
            "charter": self.charter.model_dump(mode="json"),
            "traits": self.traits.model_dump(mode="json"),
            "voice_profile": self.voice_profile.model_dump(mode="json"),
            "state": self.state.as_dict(),
            "task_mode": self.task_mode,
            "modality": self.modality,
            "compiled_from_blocks": list(self.compiled_from_blocks),
            "guardrails": list(self.guardrails),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainPersonaFrame | None":
        """Hydrate one persona frame from JSON-like data."""
        if not isinstance(data, dict):
            return None
        state = BrainPersonaState.from_dict(data.get("state"))
        if state is None:
            return None
        return cls(
            charter=PersonaCharterSpec.model_validate(data.get("charter") or {}),
            traits=PersonaTraitVectorSpec.model_validate(data.get("traits") or {}),
            voice_profile=VoiceProfileSpec.model_validate(data.get("voice_profile") or {}),
            state=state,
            task_mode=resolve_persona_task_mode(
                data.get("task_mode", BrainPersonaTaskMode.REPLY.value)
            ).value,
            modality=resolve_persona_modality(
                data.get("modality", BrainPersonaModality.TEXT.value)
            ).value,
            compiled_from_blocks=tuple(
                str(item).strip() for item in data.get("compiled_from_blocks", [])
            ),
            guardrails=tuple(str(item).strip() for item in data.get("guardrails", [])),
        )


def compile_persona_frame(
    *,
    agent_blocks: Mapping[str, str],
    task_mode: BrainPersonaTaskMode | str = BrainPersonaTaskMode.REPLY.value,
    modality: BrainPersonaModality | str = BrainPersonaModality.TEXT.value,
    state: BrainPersonaState | None = None,
) -> BrainPersonaFrame:
    """Compile a deterministic persona frame from the checked-in defaults."""
    resolved_task_mode = resolve_persona_task_mode(task_mode)
    resolved_modality = resolve_persona_modality(modality)
    defaults = load_persona_defaults(agent_blocks)
    normalized_state = state or BrainPersonaState.neutral(current_mode=resolved_task_mode.value)
    normalized_state = BrainPersonaState(
        current_mode=_normalized_text(normalized_state.current_mode) or resolved_task_mode.value,
        interaction_energy=float(normalized_state.interaction_energy),
        social_distance=float(normalized_state.social_distance),
        expressivity_boost=float(normalized_state.expressivity_boost),
        current_arc_summary=_normalized_text(normalized_state.current_arc_summary),
    )
    return BrainPersonaFrame(
        charter=_normalize_charter(defaults.charter),
        traits=_normalize_traits(defaults.traits),
        voice_profile=_normalize_voice_profile(defaults.voice_profile),
        state=normalized_state,
        task_mode=resolved_task_mode.value,
        modality=resolved_modality.value,
        compiled_from_blocks=_COMPILED_PERSONA_BLOCKS,
        guardrails=PERSONA_INVARIANT_GUARDRAILS,
    )


def compile_self_persona_core(agent_blocks: Mapping[str, str]) -> SelfPersonaCoreSpec:
    """Compile durable self/persona truth from checked-in defaults only."""
    defaults = load_persona_defaults(agent_blocks)
    fingerprints = {
        block_name: _fingerprint(agent_blocks.get(block_name, ""))
        for block_name in _COMPILED_SELF_PERSONA_BLOCKS
    }
    return SelfPersonaCoreSpec.model_validate(
        {
            "schema_version": 1,
            "canonical_name": defaults.charter.canonical_name,
            "persona_profile_id": defaults.charter.persona_profile_id,
            "ontological_status": defaults.charter.ontological_status,
            "charter": _normalize_charter(defaults.charter).model_dump(mode="json"),
            "traits": _normalize_traits(defaults.traits).model_dump(mode="json"),
            "voice_profile": _normalize_voice_profile(defaults.voice_profile).model_dump(
                mode="json"
            ),
            "guardrails": list(PERSONA_INVARIANT_GUARDRAILS),
            "compiled_from_blocks": list(_COMPILED_SELF_PERSONA_BLOCKS),
            "source_block_fingerprints": fingerprints,
        }
    )


def compile_relationship_style_state(
    agent_blocks: Mapping[str, str],
    *,
    store: Any,
    user_id: str,
    thread_id: str,
    agent_id: str | None,
) -> RelationshipStyleStateSpec:
    """Compile relationship-style state from defaults plus user-scoped interaction memory."""
    defaults = load_persona_defaults(agent_blocks)
    grouped = _group_semantic_memories(
        store=store,
        user_id=user_id,
        namespaces=PERSONA_RELATIONSHIP_STYLE_NAMESPACES,
    )
    preference_records = grouped["interaction.preference"]
    style_records = grouped["interaction.style"]
    misfire_records = grouped["interaction.misfire"]
    default_posture = _dedupe_preserve_order(defaults.relationship_defaults.default_posture)
    collaboration_style = (
        next((_memory_text(record) for record in preference_records if _memory_text(record)), "")
        or next((_memory_text(record) for record in style_records if _memory_text(record)), "")
        or _default_collaboration_style(default_posture)
    )
    emotional_tone_preference = (
        next((_memory_text(record) for record in style_records if _memory_text(record)), "")
        or next((_memory_text(record) for record in preference_records if _memory_text(record)), "")
        or _default_emotional_tone(default_posture)
    )
    known_misfires = _dedupe_preserve_order(_memory_text(record) for record in misfire_records)[:4]
    interaction_style_hints = _interaction_style_hints(grouped)
    return RelationshipStyleStateSpec.model_validate(
        {
            "schema_version": 1,
            "relationship_id": _relationship_scope_id(
                store=store,
                user_id=user_id,
                agent_id=agent_id,
            ),
            "default_posture": default_posture,
            "collaboration_style": collaboration_style,
            "emotional_tone_preference": emotional_tone_preference,
            "intimacy_ceiling": defaults.relationship_defaults.intimacy_ceiling,
            "challenge_style": defaults.relationship_defaults.challenge_style,
            "humor_permissiveness": defaults.relationship_defaults.humor_permissiveness,
            "self_disclosure_policy": defaults.relationship_defaults.self_disclosure_policy,
            "dependency_guardrails": defaults.relationship_defaults.dependency_guardrails,
            "boundaries": _safe_boundary_list(
                default_posture,
                defaults.relationship_defaults.self_disclosure_policy,
            ),
            "known_misfires": known_misfires,
            "interaction_style_hints": interaction_style_hints,
            "source_namespaces": _present_namespaces(grouped),
        }
    )


def compile_teaching_profile_state(
    agent_blocks: Mapping[str, str],
    *,
    store: Any,
    user_id: str,
    thread_id: str,
    agent_id: str | None,
) -> TeachingProfileStateSpec:
    """Compile the teaching-memory surface from defaults plus user-scoped memory."""
    defaults = load_persona_defaults(agent_blocks)
    grouped = _group_semantic_memories(
        store=store,
        user_id=user_id,
        namespaces=PERSONA_TEACHING_PROFILE_NAMESPACES,
    )
    mode_records = grouped["teaching.preference.mode"]
    analogy_records = grouped["teaching.preference.analogy_domain"]
    helpful_pattern_records = grouped["teaching.history.helpful_pattern"]
    user_modes = _dedupe_preserve_order(_memory_text(record) for record in mode_records)
    preferred_modes = _dedupe_preserve_order(
        [*user_modes, *defaults.teaching_defaults.preferred_modes]
    )
    helpful_patterns = _dedupe_preserve_order(
        _memory_text(record) for record in helpful_pattern_records
    )[:4]
    analogy_domains = _dedupe_preserve_order(_memory_text(record) for record in analogy_records)[:4]
    return TeachingProfileStateSpec.model_validate(
        {
            "schema_version": 1,
            "relationship_id": _relationship_scope_id(
                store=store,
                user_id=user_id,
                agent_id=agent_id,
            ),
            "default_mode": defaults.teaching_defaults.default_mode,
            "preferred_modes": preferred_modes,
            "question_frequency": defaults.teaching_defaults.question_frequency,
            "example_density": defaults.teaching_defaults.example_density,
            "correction_style": defaults.teaching_defaults.correction_style,
            "grounding_policy": defaults.teaching_defaults.grounding_policy,
            "analogy_domains": analogy_domains,
            "helpful_patterns": helpful_patterns,
            "source_namespaces": _present_namespaces(grouped),
        }
    )


__all__ = [
    "BrainPersonaFrame",
    "compile_persona_frame",
    "compile_relationship_style_state",
    "compile_self_persona_core",
    "compile_teaching_profile_state",
    "extract_structured_doc_payload",
    "load_persona_defaults",
]

"""Pure expression compiler for Blink's structured persona surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from blink.brain.persona.behavior_controls import BrainBehaviorControlProfile
from blink.brain.persona.compiler import BrainPersonaFrame
from blink.brain.persona.policy import (
    PERSONA_INVARIANT_GUARDRAILS,
    BrainPersonaModality,
    BrainPersonaTaskMode,
    resolve_persona_modality,
    resolve_persona_task_mode,
)
from blink.brain.persona.realtime_voice_runtime import compile_realtime_voice_actuation_plan
from blink.brain.persona.schema import (
    RelationshipStyleStateSpec,
    TeachingProfileStateSpec,
    VoiceProfileSpec,
)
from blink.brain.persona.voice_backend_registry import BrainVoiceBackendCapabilityRegistry
from blink.brain.persona.voice_policy import compile_expression_voice_policy
from blink.transcriptions.language import Language

_EXPRESSION_SCHEMA_VERSION = 1
_SAFETY_SERIOUSNESS_VALUES = frozenset({"serious", "safety", "high", "critical"})
_CONCISE_SIGNALS = ("concise", "brief", "short", "direct", "preamble", "verbose")
_DEFAULT_MEMORY_PERSONA_SECTION_STATUS = {
    "persona_expression": "unavailable",
    "persona_defaults": "unknown",
    "relationship_style": "unknown",
    "teaching_profile": "unknown",
}


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_lower(value: Any) -> str:
    return _normalized_text(value).lower()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, *, lower: float = 0.0, upper: float = 1.0) -> float:
    numeric = _float(value, lower)
    return max(lower, min(upper, numeric))


def _clamp_unit(value: Any) -> float:
    return _clamp(value, lower=0.0, upper=1.0)


def _clamp_speech_rate(value: Any) -> float:
    return _clamp(value, lower=0.5, upper=1.5)


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalized_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _first_text(values: Iterable[Any], fallback: str) -> str:
    for value in values:
        if normalized := _normalized_text(value):
            return normalized
    return fallback


def _normalize_language(language: Language | str) -> str:
    if isinstance(language, Language):
        return language.value.lower()
    return _normalized_lower(language) or "en"


def _seriousness_class(seriousness: str) -> str:
    normalized = _normalized_lower(seriousness)
    return "safety" if normalized in _SAFETY_SERIOUSNESS_VALUES else "normal"


def _has_concise_signal(values: Iterable[Any]) -> bool:
    combined = " ".join(_normalized_lower(value) for value in values)
    return any(signal in combined for signal in _CONCISE_SIGNALS)


def _initiative_style(task_mode: BrainPersonaTaskMode) -> str:
    if task_mode is BrainPersonaTaskMode.PLANNING:
        return "propose bounded next steps"
    if task_mode is BrainPersonaTaskMode.AUDIT:
        return "summarize inspectably"
    if task_mode is BrainPersonaTaskMode.REFLECTION:
        return "reflect conservatively"
    return "answer directly with bounded initiative"


def _controlled_initiative_style(
    *,
    task_mode: BrainPersonaTaskMode,
    initiative_mode: str,
) -> str:
    if initiative_mode == "minimal":
        return "answer the requested task without adding new work unless asked"
    if initiative_mode == "proactive":
        return "offer one bounded next step when it is useful"
    return _initiative_style(task_mode)


def _controlled_correction_style(
    *,
    challenge_style: str,
    correction_mode: str,
) -> str:
    if challenge_style == "avoid":
        return "avoid challenge unless needed; correct gently when necessary"
    if correction_mode == "gentle":
        return "gentle correction with brief rationale"
    if correction_mode == "rigorous":
        return "rigorous correction with evidence and no hostility"
    if challenge_style == "direct":
        return "direct precise correction without hostility"
    return "precise correction with brief rationale"


def _with_concise_chunking(hints: "BrainVoiceExpressionHints") -> "BrainVoiceExpressionHints":
    return BrainVoiceExpressionHints(
        speech_rate=hints.speech_rate,
        pause_density=hints.pause_density,
        emphasis_style=hints.emphasis_style,
        interruption_strategy=hints.interruption_strategy,
        concise_chunking=True,
        excitement_ceiling=hints.excitement_ceiling,
    )


@dataclass(frozen=True)
class BrainVoiceExpressionHints:
    """Voice-facing expression hints derived from the persona voice profile."""

    speech_rate: float
    pause_density: float
    emphasis_style: str
    interruption_strategy: str
    concise_chunking: bool
    excitement_ceiling: float

    def as_dict(self) -> dict[str, Any]:
        """Serialize the voice expression hints."""
        return {
            "speech_rate": self.speech_rate,
            "pause_density": self.pause_density,
            "emphasis_style": self.emphasis_style,
            "interruption_strategy": self.interruption_strategy,
            "concise_chunking": self.concise_chunking,
            "excitement_ceiling": self.excitement_ceiling,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainVoiceExpressionHints | None":
        """Hydrate voice expression hints from JSON-like data."""
        if not isinstance(data, dict):
            return None
        return cls(
            speech_rate=_clamp_speech_rate(data.get("speech_rate", 1.0)),
            pause_density=_clamp_unit(data.get("pause_density", 0.0)),
            emphasis_style=_normalized_text(data.get("emphasis_style")),
            interruption_strategy=_normalized_text(data.get("interruption_strategy")),
            concise_chunking=bool(data.get("concise_chunking", False)),
            excitement_ceiling=_clamp_unit(data.get("excitement_ceiling", 0.0)),
        )


@dataclass(frozen=True)
class BrainExpressionFrame:
    """Compact behavioral frame compiled from persona, relationship, and task signals."""

    schema_version: int
    persona_profile_id: str
    canonical_name: str
    ontological_status: str
    task_mode: str
    modality: str
    language: str
    response_length: str
    warmth: float
    directness: float
    playfulness: float
    caution: float
    collaboration_style: str
    challenge_style: str
    teaching_mode: str
    question_frequency: float
    example_density: float
    metaphor_density: float
    humor_budget: float
    memory_callback_policy: str
    uncertainty_style: str
    initiative_style: str
    initiative_mode: str
    evidence_visibility: str
    correction_mode: str
    explanation_structure: str
    humor_mode: str
    vividness_mode: str
    sophistication_mode: str
    character_presence: str
    story_mode: str
    style_summary: str
    safety_clamped: bool
    voice_hints: BrainVoiceExpressionHints | None
    guardrails: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the expression frame."""
        return {
            "schema_version": self.schema_version,
            "persona_profile_id": self.persona_profile_id,
            "canonical_name": self.canonical_name,
            "ontological_status": self.ontological_status,
            "task_mode": self.task_mode,
            "modality": self.modality,
            "language": self.language,
            "response_length": self.response_length,
            "warmth": self.warmth,
            "directness": self.directness,
            "playfulness": self.playfulness,
            "caution": self.caution,
            "collaboration_style": self.collaboration_style,
            "challenge_style": self.challenge_style,
            "teaching_mode": self.teaching_mode,
            "question_frequency": self.question_frequency,
            "example_density": self.example_density,
            "metaphor_density": self.metaphor_density,
            "humor_budget": self.humor_budget,
            "memory_callback_policy": self.memory_callback_policy,
            "uncertainty_style": self.uncertainty_style,
            "initiative_style": self.initiative_style,
            "initiative_mode": self.initiative_mode,
            "evidence_visibility": self.evidence_visibility,
            "correction_mode": self.correction_mode,
            "explanation_structure": self.explanation_structure,
            "humor_mode": self.humor_mode,
            "vividness_mode": self.vividness_mode,
            "sophistication_mode": self.sophistication_mode,
            "character_presence": self.character_presence,
            "story_mode": self.story_mode,
            "style_summary": self.style_summary,
            "safety_clamped": self.safety_clamped,
            "voice_hints": self.voice_hints.as_dict() if self.voice_hints else None,
            "guardrails": list(self.guardrails),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BrainExpressionFrame | None":
        """Hydrate one expression frame from JSON-like data."""
        if not isinstance(data, dict):
            return None
        task_mode = resolve_persona_task_mode(data.get("task_mode", BrainPersonaTaskMode.REPLY))
        modality = resolve_persona_modality(data.get("modality", BrainPersonaModality.TEXT))
        return cls(
            schema_version=int(data.get("schema_version") or _EXPRESSION_SCHEMA_VERSION),
            persona_profile_id=_normalized_text(data.get("persona_profile_id")),
            canonical_name=_normalized_text(data.get("canonical_name")),
            ontological_status=_normalized_text(data.get("ontological_status")),
            task_mode=task_mode.value,
            modality=modality.value,
            language=_normalize_language(str(data.get("language") or "en")),
            response_length=_normalized_text(data.get("response_length")) or "balanced",
            warmth=_clamp_unit(data.get("warmth")),
            directness=_clamp_unit(data.get("directness")),
            playfulness=_clamp_unit(data.get("playfulness")),
            caution=_clamp_unit(data.get("caution")),
            collaboration_style=_normalized_text(data.get("collaboration_style")),
            challenge_style=_normalized_text(data.get("challenge_style")),
            teaching_mode=_normalized_text(data.get("teaching_mode")) or "clarify",
            question_frequency=_clamp_unit(data.get("question_frequency")),
            example_density=_clamp_unit(data.get("example_density")),
            metaphor_density=_clamp_unit(data.get("metaphor_density")),
            humor_budget=_clamp_unit(data.get("humor_budget")),
            memory_callback_policy=_normalized_text(data.get("memory_callback_policy")),
            uncertainty_style=_normalized_text(data.get("uncertainty_style")),
            initiative_style=_normalized_text(data.get("initiative_style")),
            initiative_mode=_normalized_text(data.get("initiative_mode")) or "balanced",
            evidence_visibility=_normalized_text(data.get("evidence_visibility")) or "compact",
            correction_mode=_normalized_text(data.get("correction_mode")) or "precise",
            explanation_structure=_normalized_text(data.get("explanation_structure"))
            or "answer_first",
            humor_mode=_normalized_text(data.get("humor_mode")) or "subtle",
            vividness_mode=_normalized_text(data.get("vividness_mode")) or "balanced",
            sophistication_mode=_normalized_text(data.get("sophistication_mode")) or "smart",
            character_presence=_normalized_text(data.get("character_presence")) or "balanced",
            story_mode=_normalized_text(data.get("story_mode")) or "off",
            style_summary=_normalized_text(data.get("style_summary"))
            or "balanced local non-human expression",
            safety_clamped=bool(data.get("safety_clamped", False)),
            voice_hints=BrainVoiceExpressionHints.from_dict(data.get("voice_hints")),
            guardrails=_dedupe_preserve_order(data.get("guardrails", [])),
            reason_codes=_dedupe_preserve_order(data.get("reason_codes", [])),
        )


@dataclass(frozen=True)
class BrainRuntimeExpressionState:
    """Compact runtime-visible expression state for shell and browser inspection."""

    available: bool
    persona_profile_id: str | None
    identity_label: str
    modality: str
    teaching_mode_label: str
    memory_persona_section_status: dict[str, str]
    voice_style_summary: str
    response_chunk_length: str
    pause_yield_hint: str
    interruption_strategy_label: str
    initiative_label: str
    evidence_visibility_label: str
    correction_mode_label: str
    explanation_structure_label: str
    expression_controls_hardware: bool
    reason_codes: tuple[str, ...]
    humor_mode_label: str = "unavailable"
    vividness_mode_label: str = "unavailable"
    sophistication_mode_label: str = "unavailable"
    character_presence_label: str = "unavailable"
    story_mode_label: str = "unavailable"
    style_summary: str = "unavailable"
    humor_budget: float = 0.0
    playfulness: float = 0.0
    metaphor_density: float = 0.0
    safety_clamped: bool = False
    voice_policy: dict[str, Any] = field(default_factory=dict)
    voice_actuation_plan: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the runtime expression state."""
        return {
            "available": self.available,
            "persona_profile_id": self.persona_profile_id,
            "identity_label": self.identity_label,
            "modality": self.modality,
            "teaching_mode_label": self.teaching_mode_label,
            "memory_persona_section_status": dict(self.memory_persona_section_status),
            "voice_style_summary": self.voice_style_summary,
            "response_chunk_length": self.response_chunk_length,
            "pause_yield_hint": self.pause_yield_hint,
            "interruption_strategy_label": self.interruption_strategy_label,
            "initiative_label": self.initiative_label,
            "evidence_visibility_label": self.evidence_visibility_label,
            "correction_mode_label": self.correction_mode_label,
            "explanation_structure_label": self.explanation_structure_label,
            "expression_controls_hardware": self.expression_controls_hardware,
            "humor_mode_label": self.humor_mode_label,
            "vividness_mode_label": self.vividness_mode_label,
            "sophistication_mode_label": self.sophistication_mode_label,
            "character_presence_label": self.character_presence_label,
            "story_mode_label": self.story_mode_label,
            "style_summary": self.style_summary,
            "humor_budget": self.humor_budget,
            "playfulness": self.playfulness,
            "metaphor_density": self.metaphor_density,
            "safety_clamped": self.safety_clamped,
            "reason_codes": list(self.reason_codes),
            "voice_policy": dict(self.voice_policy),
            "voice_actuation_plan": dict(self.voice_actuation_plan),
        }


def _compile_voice_hints(
    *,
    voice_profile: VoiceProfileSpec,
    response_length: str,
    is_safety_context: bool,
) -> BrainVoiceExpressionHints:
    pause_density = _clamp_unit(voice_profile.pause_density + (0.08 if is_safety_context else 0.0))
    excitement_ceiling = _clamp_unit(voice_profile.excitement_ceiling)
    if is_safety_context:
        excitement_ceiling = min(excitement_ceiling, 0.18)
    return BrainVoiceExpressionHints(
        speech_rate=_clamp_speech_rate(voice_profile.speech_rate),
        pause_density=pause_density,
        emphasis_style=_normalized_text(voice_profile.emphasis_style),
        interruption_strategy=_normalized_text(voice_profile.interruption_strategy),
        concise_chunking=response_length == "concise",
        excitement_ceiling=excitement_ceiling,
    )


def _expression_status(
    values: dict[str, str] | None,
    *,
    available: bool,
) -> dict[str, str]:
    status = dict(_DEFAULT_MEMORY_PERSONA_SECTION_STATUS)
    if values:
        status.update({str(key): str(value) for key, value in values.items()})
    status["persona_expression"] = "available" if available else status["persona_expression"]
    return status


def unavailable_runtime_expression_state(
    *,
    modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
    reason_codes: tuple[str, ...] = (),
    memory_persona_section_status: dict[str, str] | None = None,
    tts_backend: str | None = None,
    voice_backend_registry: BrainVoiceBackendCapabilityRegistry | None = None,
) -> BrainRuntimeExpressionState:
    """Return a traceable fail-closed runtime expression state."""
    resolved_modality = resolve_persona_modality(modality)
    return BrainRuntimeExpressionState(
        available=False,
        persona_profile_id=None,
        identity_label="Blink; local non-human system",
        modality=resolved_modality.value,
        teaching_mode_label="unavailable",
        memory_persona_section_status=_expression_status(
            memory_persona_section_status,
            available=False,
        ),
        voice_style_summary="unavailable",
        response_chunk_length="unavailable",
        pause_yield_hint="unavailable",
        interruption_strategy_label="unavailable",
        initiative_label="unavailable",
        evidence_visibility_label="unavailable",
        correction_mode_label="unavailable",
        explanation_structure_label="unavailable",
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "runtime_expression_state:unavailable",
                *reason_codes,
            )
        ),
        humor_mode_label="unavailable",
        vividness_mode_label="unavailable",
        sophistication_mode_label="unavailable",
        character_presence_label="unavailable",
        story_mode_label="unavailable",
        style_summary="unavailable",
        humor_budget=0.0,
        playfulness=0.0,
        metaphor_density=0.0,
        safety_clamped=False,
        voice_policy=(
            voice_policy := compile_expression_voice_policy(
                None,
                modality=resolved_modality,
                tts_backend=tts_backend,
            )
        ).as_dict(),
        voice_actuation_plan=compile_realtime_voice_actuation_plan(
            voice_policy,
            tts_backend=tts_backend,
            capability_registry=voice_backend_registry,
        ).as_dict(),
    )


def runtime_expression_state_from_frame(
    frame: BrainExpressionFrame | None,
    *,
    modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
    reason_codes: tuple[str, ...] = (),
    memory_persona_section_status: dict[str, str] | None = None,
    tts_backend: str | None = None,
    voice_backend_registry: BrainVoiceBackendCapabilityRegistry | None = None,
) -> BrainRuntimeExpressionState:
    """Map an expression frame into high-level runtime-visible state."""
    resolved_modality = resolve_persona_modality(modality)
    if frame is None:
        return unavailable_runtime_expression_state(
            modality=resolved_modality,
            reason_codes=reason_codes,
            memory_persona_section_status=memory_persona_section_status,
            tts_backend=tts_backend,
            voice_backend_registry=voice_backend_registry,
        )

    voice_hints = frame.voice_hints
    if voice_hints is not None:
        voice_style_summary = (
            f"{voice_hints.emphasis_style}; "
            f"rate={voice_hints.speech_rate:.2f}; "
            f"pause={voice_hints.pause_density:.2f}; "
            f"yield={voice_hints.interruption_strategy}"
        )
        pause_yield_hint = (
            f"pause={voice_hints.pause_density:.2f}; yield={voice_hints.interruption_strategy}"
        )
        interruption_strategy_label = voice_hints.interruption_strategy
    else:
        voice_style_summary = f"voice hints not active for {resolved_modality.value}"
        pause_yield_hint = "not active"
        interruption_strategy_label = "not active"

    response_chunk_length = "balanced"
    if frame.response_length == "concise" or (
        voice_hints is not None and voice_hints.concise_chunking
    ):
        response_chunk_length = "short"

    voice_policy = compile_expression_voice_policy(
        frame,
        modality=resolved_modality,
        tts_backend=tts_backend,
    )
    return BrainRuntimeExpressionState(
        available=True,
        persona_profile_id=frame.persona_profile_id,
        identity_label=f"{frame.canonical_name}; local non-human system",
        modality=resolved_modality.value,
        teaching_mode_label=frame.teaching_mode,
        memory_persona_section_status=_expression_status(
            memory_persona_section_status,
            available=True,
        ),
        voice_style_summary=voice_style_summary,
        response_chunk_length=response_chunk_length,
        pause_yield_hint=pause_yield_hint,
        interruption_strategy_label=interruption_strategy_label,
        initiative_label=frame.initiative_mode,
        evidence_visibility_label=frame.evidence_visibility,
        correction_mode_label=frame.correction_mode,
        explanation_structure_label=frame.explanation_structure,
        expression_controls_hardware=False,
        reason_codes=_dedupe_preserve_order(
            (
                "runtime_expression_state:available",
                *reason_codes,
                *frame.reason_codes,
            )
        ),
        humor_mode_label=frame.humor_mode,
        vividness_mode_label=frame.vividness_mode,
        sophistication_mode_label=frame.sophistication_mode,
        character_presence_label=frame.character_presence,
        story_mode_label=frame.story_mode,
        style_summary=frame.style_summary,
        humor_budget=frame.humor_budget,
        playfulness=frame.playfulness,
        metaphor_density=frame.metaphor_density,
        safety_clamped=frame.safety_clamped,
        voice_policy=voice_policy.as_dict(),
        voice_actuation_plan=compile_realtime_voice_actuation_plan(
            voice_policy,
            tts_backend=tts_backend,
            capability_registry=voice_backend_registry,
        ).as_dict(),
    )


def compile_expression_frame(
    *,
    persona_frame: BrainPersonaFrame,
    relationship_style: RelationshipStyleStateSpec | None,
    teaching_profile: TeachingProfileStateSpec | None,
    behavior_controls: BrainBehaviorControlProfile | None = None,
    task_mode: BrainPersonaTaskMode | str,
    modality: BrainPersonaModality | str,
    language: Language | str,
    seriousness: str = "normal",
    recent_misfires: tuple[str, ...] = (),
) -> BrainExpressionFrame:
    """Compile a pure deterministic behavior frame from persona-adjacent state."""
    resolved_task_mode = resolve_persona_task_mode(task_mode)
    resolved_modality = resolve_persona_modality(modality)
    language_value = _normalize_language(language)
    seriousness_label = _seriousness_class(seriousness)
    is_safety_context = seriousness_label == "safety"

    reason_codes: list[str] = [
        f"persona_profile:{persona_frame.charter.persona_profile_id}",
        f"task_mode:{resolved_task_mode.value}",
        f"modality:{resolved_modality.value}",
        f"language:{language_value}",
    ]

    expressivity_boost = _clamp(persona_frame.state.expressivity_boost, lower=-1.0, upper=1.0)
    warmth = _clamp_unit(persona_frame.traits.warmth + (expressivity_boost * 0.1))
    directness = _clamp_unit(persona_frame.traits.directness)
    playfulness = _clamp_unit(persona_frame.traits.playfulness + (expressivity_boost * 0.05))
    caution = _clamp_unit(persona_frame.traits.self_restraint)
    question_frequency = _clamp_unit(0.25)
    example_density = _clamp_unit(persona_frame.traits.pedagogical_generosity * 0.75)
    metaphor_density = _clamp_unit(persona_frame.traits.metaphor_density)
    humor_budget = _clamp_unit(persona_frame.traits.humor_frequency)
    response_length = "balanced"
    collaboration_style = "warm collaborative"
    challenge_style = "gentle precise correction"
    teaching_mode = "clarify"
    uncertainty_style = "state uncertainty instead of bluffing"
    memory_callback_policy = "use relationship or teaching memory only when relevant"
    initiative_mode = "balanced"
    evidence_visibility = "compact"
    correction_mode = "precise"
    explanation_structure = "answer_first"
    humor_mode = "subtle"
    vividness_mode = "balanced"
    sophistication_mode = "smart"
    character_presence = "balanced"
    story_mode = "off"
    safety_clamped = False
    initiative_style = _initiative_style(resolved_task_mode)
    guardrail_values: list[str] = [
        *PERSONA_INVARIANT_GUARDRAILS,
        *persona_frame.guardrails,
    ]

    relationship_signal_texts: list[str] = list(recent_misfires)
    if relationship_style is not None:
        reason_codes.append("relationship:present")
        collaboration_style = relationship_style.collaboration_style
        challenge_style = relationship_style.challenge_style
        humor_budget = min(humor_budget, _clamp_unit(relationship_style.humor_permissiveness))
        relationship_signal_texts.extend(
            [
                relationship_style.collaboration_style,
                relationship_style.emotional_tone_preference,
                relationship_style.challenge_style,
                *relationship_style.known_misfires,
                *relationship_style.interaction_style_hints,
            ]
        )
        guardrail_values.extend(relationship_style.boundaries)
        guardrail_values.extend(relationship_style.dependency_guardrails)
    else:
        reason_codes.append("relationship:missing")

    if teaching_profile is not None:
        reason_codes.append("teaching:present")
        teaching_mode = _first_text(
            [*teaching_profile.preferred_modes, teaching_profile.default_mode],
            "clarify",
        )
        question_frequency = _clamp_unit(teaching_profile.question_frequency)
        example_density = _clamp_unit(teaching_profile.example_density)
        uncertainty_style = teaching_profile.grounding_policy
        if relationship_style is None:
            challenge_style = teaching_profile.correction_style
    else:
        reason_codes.append("teaching:missing")

    if _has_concise_signal(relationship_signal_texts):
        response_length = "concise"
        directness = _clamp_unit(directness + 0.08)
        reason_codes.append("relationship_signal:concise_or_direct")

    if behavior_controls is not None:
        reason_codes.extend(
            [
                "behavior_controls:present",
                f"behavior_response_depth:{behavior_controls.response_depth}",
                f"behavior_directness:{behavior_controls.directness}",
                f"behavior_warmth:{behavior_controls.warmth}",
                f"behavior_teaching_mode:{behavior_controls.teaching_mode}",
                f"behavior_memory_use:{behavior_controls.memory_use}",
                f"behavior_initiative_mode:{behavior_controls.initiative_mode}",
                f"behavior_evidence_visibility:{behavior_controls.evidence_visibility}",
                f"behavior_correction_mode:{behavior_controls.correction_mode}",
                f"behavior_explanation_structure:{behavior_controls.explanation_structure}",
                f"behavior_challenge_style:{behavior_controls.challenge_style}",
                f"behavior_voice_mode:{behavior_controls.voice_mode}",
                f"behavior_question_budget:{behavior_controls.question_budget}",
                f"behavior_humor_mode:{behavior_controls.humor_mode}",
                f"behavior_vividness_mode:{behavior_controls.vividness_mode}",
                f"behavior_sophistication_mode:{behavior_controls.sophistication_mode}",
                f"behavior_character_presence:{behavior_controls.character_presence}",
                f"behavior_story_mode:{behavior_controls.story_mode}",
            ]
        )
        initiative_mode = behavior_controls.initiative_mode
        evidence_visibility = behavior_controls.evidence_visibility
        correction_mode = behavior_controls.correction_mode
        explanation_structure = behavior_controls.explanation_structure
        humor_mode = behavior_controls.humor_mode
        vividness_mode = behavior_controls.vividness_mode
        sophistication_mode = behavior_controls.sophistication_mode
        character_presence = behavior_controls.character_presence
        story_mode = behavior_controls.story_mode
        initiative_style = _controlled_initiative_style(
            task_mode=resolved_task_mode,
            initiative_mode=initiative_mode,
        )
        if behavior_controls.response_depth == "concise":
            response_length = "concise"
            example_density = _clamp_unit(example_density * 0.75)
        elif behavior_controls.response_depth == "deep" and response_length != "concise":
            response_length = "deep"
            example_density = _clamp_unit(example_density + 0.12)

        if behavior_controls.directness == "gentle":
            directness = _clamp_unit(directness - 0.07)
        elif behavior_controls.directness == "rigorous":
            directness = _clamp_unit(directness + 0.1)

        if behavior_controls.warmth == "low":
            warmth = _clamp_unit(warmth - 0.08)
        elif behavior_controls.warmth == "high":
            warmth = _clamp_unit(warmth + 0.1)

        if behavior_controls.teaching_mode != "auto":
            teaching_mode = behavior_controls.teaching_mode
        if behavior_controls.teaching_mode == "socratic":
            question_frequency = _clamp_unit(max(question_frequency, 0.45))
        elif behavior_controls.teaching_mode == "direct":
            question_frequency = min(question_frequency, 0.2)

        if behavior_controls.memory_use == "minimal":
            memory_callback_policy = "avoid casual memory callbacks; use memory only when necessary"
        elif behavior_controls.memory_use == "continuity_rich":
            memory_callback_policy = "use relevant continuity memory transparently when helpful"

        challenge_style = _controlled_correction_style(
            challenge_style=behavior_controls.challenge_style,
            correction_mode=correction_mode,
        )

        if evidence_visibility == "hidden":
            uncertainty_style = "state uncertainty briefly; do not surface evidence unless asked"
        elif evidence_visibility == "rich":
            uncertainty_style = "state uncertainty and mention compact evidence categories when useful"

        if explanation_structure == "walkthrough" and teaching_mode not in {"direct", "socratic"}:
            teaching_mode = "walkthrough"
            example_density = _clamp_unit(example_density + 0.06)
        elif explanation_structure == "socratic":
            question_frequency = _clamp_unit(max(question_frequency, 0.4))

        question_cap = {
            "low": 0.18,
            "medium": 0.35,
            "high": 0.55,
        }[behavior_controls.question_budget]
        question_frequency = min(question_frequency, question_cap)

        if humor_mode == "off":
            humor_budget = 0.0
            playfulness = min(playfulness, 0.08)
        elif humor_mode == "subtle":
            humor_budget = min(humor_budget, 0.24)
            playfulness = min(_clamp_unit(playfulness + 0.02), 0.32)
        elif humor_mode == "witty":
            humor_budget = min(_clamp_unit(humor_budget + 0.14), 0.46)
            playfulness = min(_clamp_unit(playfulness + 0.08), 0.48)
        elif humor_mode == "playful":
            humor_budget = min(_clamp_unit(humor_budget + 0.22), 0.58)
            playfulness = min(_clamp_unit(playfulness + 0.16), 0.62)

        if vividness_mode == "spare":
            metaphor_density = min(metaphor_density, 0.18)
            example_density = _clamp_unit(example_density * 0.85)
        elif vividness_mode == "vivid":
            metaphor_density = min(_clamp_unit(metaphor_density + 0.16), 0.54)
            example_density = _clamp_unit(example_density + 0.08)

        if sophistication_mode == "plain":
            metaphor_density = _clamp_unit(metaphor_density * 0.72)
            directness = _clamp_unit(directness - 0.02)
        elif sophistication_mode == "sophisticated":
            directness = _clamp_unit(directness + 0.04)
            metaphor_density = min(_clamp_unit(metaphor_density + 0.04), 0.58)
            uncertainty_style = (
                "state uncertainty with compact evidence categories and precise tradeoffs"
            )

        if character_presence == "minimal":
            playfulness = min(playfulness, 0.16)
            metaphor_density = min(metaphor_density, 0.22)
        elif character_presence == "character_rich":
            playfulness = min(_clamp_unit(playfulness + 0.07), 0.56)
            metaphor_density = min(_clamp_unit(metaphor_density + 0.08), 0.62)
            collaboration_style = f"{collaboration_style}; character-rich local presence"
            reason_codes.append("expressive_style:character_rich_no_fake_backstory")

        if story_mode == "light":
            metaphor_density = min(_clamp_unit(metaphor_density + 0.04), 0.58)
        elif story_mode == "recurring_motifs":
            metaphor_density = min(_clamp_unit(metaphor_density + 0.12), 0.68)
            example_density = _clamp_unit(example_density + 0.07)
            reason_codes.append("expressive_style:recurring_motifs_public_safe")
    else:
        reason_codes.append("behavior_controls:missing")

    reason_codes.append(f"seriousness:{seriousness_label}")
    if is_safety_context:
        safety_clamped = True
        response_length = "concise"
        humor_budget = min(_clamp_unit(humor_budget * 0.25), 0.08)
        playfulness = min(_clamp_unit(playfulness * 0.35), 0.15)
        question_frequency = min(_clamp_unit(question_frequency * 0.65), 0.24)
        metaphor_density = min(_clamp_unit(metaphor_density * 0.5), 0.18)
        caution = max(caution, 0.9)
        uncertainty_style = "state uncertainty early and avoid speculation"
        if behavior_controls is not None:
            question_frequency = min(question_frequency, 0.18)
            memory_callback_policy = "use memory only when safety-relevant or explicitly needed"
            if evidence_visibility == "hidden":
                evidence_visibility = "compact"
                reason_codes.append("behavior_evidence_visibility:safety_override")
        initiative_style = "answer safety issue directly; avoid extra initiative"
        reason_codes.append("expressive_style:safety_clamped")

    voice_hints = None
    voice_profile = getattr(persona_frame, "voice_profile", None)
    if (
        not (behavior_controls is not None and behavior_controls.voice_mode == "off")
        and resolved_modality
        in {
            BrainPersonaModality.VOICE,
            BrainPersonaModality.BROWSER,
            BrainPersonaModality.EMBODIED,
        }
        and voice_profile is not None
    ):
        voice_hints = _compile_voice_hints(
            voice_profile=voice_profile,
            response_length=response_length,
            is_safety_context=is_safety_context,
        )
        if behavior_controls is not None and behavior_controls.voice_mode == "concise":
            voice_hints = _with_concise_chunking(voice_hints)
        reason_codes.append("voice_hints:included")
    else:
        reason_codes.append("voice_hints:excluded")
        if behavior_controls is not None and behavior_controls.voice_mode == "off":
            reason_codes.append("voice_hints:disabled_by_behavior_controls")

    style_summary = (
        f"humor={humor_mode}; vividness={vividness_mode}; "
        f"sophistication={sophistication_mode}; character={character_presence}; "
        f"story={story_mode}; safety_clamped={str(safety_clamped).lower()}"
    )

    return BrainExpressionFrame(
        schema_version=_EXPRESSION_SCHEMA_VERSION,
        persona_profile_id=persona_frame.charter.persona_profile_id,
        canonical_name=persona_frame.charter.canonical_name,
        ontological_status=persona_frame.charter.ontological_status,
        task_mode=resolved_task_mode.value,
        modality=resolved_modality.value,
        language=language_value,
        response_length=response_length,
        warmth=warmth,
        directness=directness,
        playfulness=playfulness,
        caution=caution,
        collaboration_style=collaboration_style,
        challenge_style=challenge_style,
        teaching_mode=teaching_mode,
        question_frequency=question_frequency,
        example_density=example_density,
        metaphor_density=metaphor_density,
        humor_budget=humor_budget,
        memory_callback_policy=memory_callback_policy,
        uncertainty_style=uncertainty_style,
        initiative_style=initiative_style,
        initiative_mode=initiative_mode,
        evidence_visibility=evidence_visibility,
        correction_mode=correction_mode,
        explanation_structure=explanation_structure,
        humor_mode=humor_mode,
        vividness_mode=vividness_mode,
        sophistication_mode=sophistication_mode,
        character_presence=character_presence,
        story_mode=story_mode,
        style_summary=style_summary,
        safety_clamped=safety_clamped,
        voice_hints=voice_hints,
        guardrails=_dedupe_preserve_order(guardrail_values),
        reason_codes=tuple(reason_codes),
    )


def render_persona_expression_summary(frame: BrainExpressionFrame) -> str:
    """Render one compact prompt-safe persona expression section."""
    boundary_values = [
        guardrail
        for guardrail in frame.guardrails
        if guardrail in {"non-romantic", "non-sexual", "non-exclusive"}
    ]
    boundaries = "; ".join(boundary_values[:3]) if boundary_values else "bounded"
    lines = [
        f"identity: {frame.canonical_name}; local non-human system",
        "character: warm precise local tutor; no human backstory",
        (
            "expression: "
            f"{frame.response_length}; "
            f"initiative={frame.initiative_mode}; "
            f"evidence={frame.evidence_visibility}; "
            f"humor={frame.humor_mode}; "
            f"story={frame.story_mode}"
        ),
        (
            "teaching: "
            f"mode={frame.teaching_mode}; "
            f"structure={frame.explanation_structure}; "
            f"correction={frame.correction_mode}"
        ),
        f"relationship boundaries: {boundaries}",
    ]
    if frame.voice_hints is not None:
        lines.append(
            "voice: "
            f"rate={frame.voice_hints.speech_rate:.2f}; "
            f"pause={frame.voice_hints.pause_density:.2f}; "
            f"emphasis={frame.voice_hints.emphasis_style}; "
            f"yield={frame.voice_hints.interruption_strategy}"
        )
    return "\n".join(lines)


__all__ = [
    "BrainExpressionFrame",
    "BrainRuntimeExpressionState",
    "BrainVoiceExpressionHints",
    "compile_expression_frame",
    "render_persona_expression_summary",
    "runtime_expression_state_from_frame",
    "unavailable_runtime_expression_state",
]

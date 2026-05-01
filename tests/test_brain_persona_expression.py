import json
from typing import Any

import pytest

from blink.brain.identity import load_default_agent_blocks
from blink.brain.persona import (
    PERSONA_INVARIANT_GUARDRAILS,
    BrainBehaviorControlProfile,
    BrainExpressionFrame,
    BrainPersonaModality,
    BrainPersonaTaskMode,
    BrainVoiceExpressionHints,
    RelationshipStyleStateSpec,
    TeachingProfileStateSpec,
    compile_expression_frame,
    compile_persona_frame,
    default_behavior_control_profile,
    render_persona_expression_summary,
)
from blink.transcriptions.language import Language


def _persona_frame():
    return compile_persona_frame(
        agent_blocks=load_default_agent_blocks(),
        task_mode=BrainPersonaTaskMode.REPLY,
        modality=BrainPersonaModality.TEXT,
    )


def _relationship_style(
    *,
    collaboration_style: str = "warm but concise",
    known_misfires: list[str] | None = None,
    interaction_style_hints: list[str] | None = None,
) -> RelationshipStyleStateSpec:
    return RelationshipStyleStateSpec.model_validate(
        {
            "schema_version": 1,
            "relationship_id": "blink/main:user-1",
            "default_posture": [
                "warm",
                "intelligent",
                "collaborative",
                "non-romantic",
                "non-sexual",
                "non-exclusive",
            ],
            "collaboration_style": collaboration_style,
            "emotional_tone_preference": "warm precise",
            "intimacy_ceiling": "warm professional companion",
            "challenge_style": "gentle directness",
            "humor_permissiveness": 0.2,
            "self_disclosure_policy": "no fabricated personal history",
            "dependency_guardrails": [
                "avoid guilt language",
                "avoid exclusivity",
                "encourage human support when appropriate",
            ],
            "boundaries": ["non-romantic", "non-sexual", "non-exclusive"],
            "known_misfires": known_misfires or ["too much preamble"],
            "interaction_style_hints": interaction_style_hints
            or ["User prefers concise collaboration."],
            "source_namespaces": ["interaction.preference", "interaction.misfire"],
        }
    )


def _teaching_profile(
    *,
    preferred_modes: list[str] | None = None,
    question_frequency: float = 0.44,
    example_density: float = 0.88,
) -> TeachingProfileStateSpec:
    return TeachingProfileStateSpec.model_validate(
        {
            "schema_version": 1,
            "relationship_id": "blink/main:user-1",
            "default_mode": "clarify",
            "preferred_modes": preferred_modes or ["walkthrough", "clarify"],
            "question_frequency": question_frequency,
            "example_density": example_density,
            "correction_style": "gentle precise correction",
            "grounding_policy": "state uncertainty instead of bluffing",
            "analogy_domains": ["physics"],
            "helpful_patterns": ["stepwise decomposition"],
            "source_namespaces": ["teaching.preference.mode"],
        }
    )


def _compile_expression(**overrides):
    payload = {
        "persona_frame": _persona_frame(),
        "relationship_style": None,
        "teaching_profile": None,
        "task_mode": BrainPersonaTaskMode.REPLY,
        "modality": BrainPersonaModality.TEXT,
        "language": Language.EN,
    }
    payload.update(overrides)
    return compile_expression_frame(**payload)


def _behavior_controls(**overrides) -> BrainBehaviorControlProfile:
    data = default_behavior_control_profile(
        user_id="user-1",
        agent_id="blink/main",
    ).as_dict()
    data.update(overrides)
    profile = BrainBehaviorControlProfile.from_dict(data)
    assert profile is not None
    return profile


def _payload_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(_payload_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_payload_keys(nested))
        return keys
    return set()


def test_compile_expression_frame_is_deterministic_and_roundtrips():
    relationship = _relationship_style()
    teaching = _teaching_profile()
    first = _compile_expression(
        relationship_style=relationship,
        teaching_profile=teaching,
        modality=BrainPersonaModality.VOICE,
    )
    second = _compile_expression(
        relationship_style=relationship,
        teaching_profile=teaching,
        modality=BrainPersonaModality.VOICE,
    )
    restored = BrainExpressionFrame.from_dict(first.as_dict())

    assert first.as_dict() == second.as_dict()
    assert restored is not None
    assert restored.as_dict() == first.as_dict()


def test_relationship_preference_affects_style_directness_and_length():
    baseline = _compile_expression()
    expression = _compile_expression(
        relationship_style=_relationship_style(
            collaboration_style="warm but concise",
            known_misfires=["too much preamble"],
            interaction_style_hints=["User prefers direct short answers."],
        )
    )

    assert expression.collaboration_style == "warm but concise"
    assert expression.response_length == "concise"
    assert expression.directness > baseline.directness
    assert "relationship_signal:concise_or_direct" in expression.reason_codes


def test_teaching_profile_affects_teaching_mode_and_example_density():
    expression = _compile_expression(
        teaching_profile=_teaching_profile(
            preferred_modes=["walkthrough", "deep_dive"],
            question_frequency=0.38,
            example_density=0.91,
        )
    )

    assert expression.teaching_mode == "walkthrough"
    assert expression.question_frequency == 0.38
    assert expression.example_density == 0.91
    assert expression.uncertainty_style == "state uncertainty instead of bluffing"


@pytest.mark.parametrize(
    "modality",
    [
        BrainPersonaModality.VOICE,
        BrainPersonaModality.BROWSER,
        BrainPersonaModality.EMBODIED,
    ],
)
def test_voice_profile_affects_voice_hints_for_voice_like_modalities(modality):
    expression = _compile_expression(modality=modality)

    assert isinstance(expression.voice_hints, BrainVoiceExpressionHints)
    assert expression.voice_hints.speech_rate == 1.0
    assert expression.voice_hints.pause_density == 0.34
    assert expression.voice_hints.emphasis_style == "measured clarity"
    assert expression.voice_hints.interruption_strategy == "yield after brief pause"


def test_text_modality_has_no_voice_hints():
    expression = _compile_expression(modality=BrainPersonaModality.TEXT)

    assert expression.voice_hints is None
    assert "voice_hints:excluded" in expression.reason_codes


def test_serious_context_clamps_humor_playfulness_and_raises_caution():
    normal = _compile_expression(seriousness="normal")
    serious = _compile_expression(seriousness="safety")

    assert serious.response_length == "concise"
    assert serious.humor_budget < normal.humor_budget
    assert serious.playfulness < normal.playfulness
    assert serious.question_frequency < normal.question_frequency
    assert serious.metaphor_density < normal.metaphor_density
    assert serious.caution > normal.caution
    assert serious.uncertainty_style == "state uncertainty early and avoid speculation"


def test_invalid_task_or_modality_fail_fast():
    with pytest.raises(ValueError, match="Unsupported persona task mode"):
        _compile_expression(task_mode="unsupported")
    with pytest.raises(ValueError, match="Unsupported persona modality"):
        _compile_expression(modality="screen")


def test_required_guardrails_are_always_present():
    expression = _compile_expression(relationship_style=_relationship_style())

    assert set(PERSONA_INVARIANT_GUARDRAILS).issubset(set(expression.guardrails))
    assert "non-romantic" in expression.guardrails
    assert "avoid guilt language" in expression.guardrails
    assert expression.canonical_name == "Blink"
    assert expression.ontological_status == "local_cognitive_runtime"


def test_serialized_expression_has_no_fake_human_identity_fields():
    expression = _compile_expression(
        relationship_style=_relationship_style(),
        teaching_profile=_teaching_profile(),
        modality=BrainPersonaModality.VOICE,
    )
    payload = expression.as_dict()
    encoded = json.dumps(payload, sort_keys=True)
    banned_fields = {
        "age",
        "childhood",
        "family",
        "schooling",
        "human_identity_claims_allowed",
    }

    assert banned_fields.isdisjoint(_payload_keys(payload))
    assert "A Human Student" not in encoded


def test_behavior_controls_affect_bounded_expression_fields():
    baseline = _compile_expression()
    controlled = _compile_expression(
        behavior_controls=_behavior_controls(
            response_depth="deep",
            directness="rigorous",
            warmth="high",
            teaching_mode="socratic",
            memory_use="continuity_rich",
            challenge_style="direct",
            question_budget="high",
        )
    )

    assert controlled.response_length == "deep"
    assert controlled.directness > baseline.directness
    assert controlled.warmth > baseline.warmth
    assert controlled.teaching_mode == "socratic"
    assert controlled.question_frequency > baseline.question_frequency
    assert controlled.question_frequency <= 0.55
    assert controlled.example_density > baseline.example_density
    assert controlled.memory_callback_policy == (
        "use relevant continuity memory transparently when helpful"
    )
    assert controlled.challenge_style == "direct precise correction without hostility"
    assert "behavior_controls:present" in controlled.reason_codes


def test_behavior_v2_controls_compile_into_typed_runtime_effects():
    controlled = _compile_expression(
        behavior_controls=_behavior_controls(
            initiative_mode="proactive",
            evidence_visibility="rich",
            correction_mode="rigorous",
            explanation_structure="walkthrough",
            teaching_mode="auto",
            challenge_style="direct",
            sophistication_mode="smart",
        )
    )
    summary = render_persona_expression_summary(controlled)
    restored = BrainExpressionFrame.from_dict(controlled.as_dict())

    assert controlled.initiative_mode == "proactive"
    assert "one bounded next step" in controlled.initiative_style
    assert controlled.evidence_visibility == "rich"
    assert controlled.uncertainty_style == (
        "state uncertainty and mention compact evidence categories when useful"
    )
    assert controlled.correction_mode == "rigorous"
    assert controlled.challenge_style == "rigorous correction with evidence and no hostility"
    assert controlled.explanation_structure == "walkthrough"
    assert controlled.teaching_mode == "walkthrough"
    assert "initiative=proactive" in summary
    assert "evidence=rich" in summary
    assert "structure=walkthrough" in summary
    assert "correction=rigorous" in summary
    assert restored is not None
    assert restored.as_dict() == controlled.as_dict()


def test_witty_sophisticated_controls_raise_expressive_metrics_within_caps():
    baseline = _compile_expression(
        behavior_controls=_behavior_controls(
            humor_mode="subtle",
            vividness_mode="balanced",
            sophistication_mode="smart",
            character_presence="balanced",
            story_mode="off",
        )
    )
    styled = _compile_expression(
        behavior_controls=_behavior_controls(
            humor_mode="witty",
            vividness_mode="vivid",
            sophistication_mode="sophisticated",
            character_presence="character_rich",
            story_mode="recurring_motifs",
        )
    )
    summary = render_persona_expression_summary(styled)
    restored = BrainExpressionFrame.from_dict(styled.as_dict())

    assert styled.humor_mode == "witty"
    assert styled.vividness_mode == "vivid"
    assert styled.sophistication_mode == "sophisticated"
    assert styled.character_presence == "character_rich"
    assert styled.story_mode == "recurring_motifs"
    assert styled.humor_budget > baseline.humor_budget
    assert styled.playfulness > baseline.playfulness
    assert styled.metaphor_density > baseline.metaphor_density
    assert styled.humor_budget <= 0.46
    assert styled.playfulness <= 0.56
    assert styled.metaphor_density <= 0.68
    assert "humor=witty" in summary
    assert "story=recurring_motifs" in summary
    assert "expressive_style:character_rich_no_fake_backstory" in styled.reason_codes
    assert "expressive_style:recurring_motifs_public_safe" in styled.reason_codes
    assert restored is not None
    assert restored.as_dict() == styled.as_dict()


def test_character_rich_style_stays_nonhuman_without_fake_autobiography():
    styled = _compile_expression(
        behavior_controls=_behavior_controls(
            character_presence="character_rich",
            story_mode="recurring_motifs",
        )
    )
    encoded = json.dumps(styled.as_dict(), sort_keys=True)

    assert "character-rich local presence" in styled.collaboration_style
    assert styled.ontological_status == "local_cognitive_runtime"
    assert styled.safety_clamped is False
    assert "childhood" not in encoded
    assert "family" not in encoded
    assert "human_identity" not in _payload_keys(styled.as_dict())


def test_behavior_memory_use_changes_only_memory_callback_policy():
    baseline = _compile_expression()
    minimal_memory = _compile_expression(
        behavior_controls=_behavior_controls(
            memory_use="minimal",
            humor_mode="subtle",
            vividness_mode="balanced",
            sophistication_mode="smart",
            character_presence="balanced",
            story_mode="off",
        )
    )

    assert minimal_memory.response_length == baseline.response_length
    assert minimal_memory.directness == baseline.directness
    assert minimal_memory.warmth == baseline.warmth
    assert minimal_memory.question_frequency == baseline.question_frequency
    assert minimal_memory.memory_callback_policy == (
        "avoid casual memory callbacks; use memory only when necessary"
    )


def test_behavior_voice_mode_controls_voice_hints_without_hardware_claims():
    disabled = _compile_expression(
        modality=BrainPersonaModality.BROWSER,
        behavior_controls=_behavior_controls(voice_mode="off"),
    )
    concise = _compile_expression(
        modality=BrainPersonaModality.BROWSER,
        behavior_controls=_behavior_controls(voice_mode="concise"),
    )

    assert disabled.voice_hints is None
    assert "voice_hints:disabled_by_behavior_controls" in disabled.reason_codes
    assert concise.voice_hints is not None
    assert concise.voice_hints.concise_chunking is True
    assert "hardware" not in json.dumps(concise.as_dict(), sort_keys=True)


def test_safety_context_overrides_expansive_behavior_controls():
    controls = _behavior_controls(
        response_depth="deep",
        teaching_mode="socratic",
        memory_use="continuity_rich",
        question_budget="high",
        warmth="high",
        evidence_visibility="hidden",
    )
    normal = _compile_expression(behavior_controls=controls)
    safety = _compile_expression(behavior_controls=controls, seriousness="safety")

    assert normal.response_length == "deep"
    assert normal.teaching_mode == "socratic"
    assert safety.response_length == "concise"
    assert safety.question_frequency <= 0.18
    assert safety.humor_budget < normal.humor_budget
    assert safety.playfulness < normal.playfulness
    assert safety.caution > normal.caution
    assert safety.safety_clamped is True
    assert safety.metaphor_density < normal.metaphor_density
    assert safety.memory_callback_policy == (
        "use memory only when safety-relevant or explicitly needed"
    )
    assert safety.evidence_visibility == "compact"
    assert "behavior_evidence_visibility:safety_override" in safety.reason_codes
    assert "expressive_style:safety_clamped" in safety.reason_codes

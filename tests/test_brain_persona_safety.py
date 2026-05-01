import json

import pytest
from pydantic import ValidationError

from blink.brain.identity import load_default_agent_blocks
from blink.brain.persona import (
    BrainBehaviorControlProfile,
    RelationshipStyleStateSpec,
    TeachingProfileStateSpec,
    compile_self_persona_core,
    default_behavior_control_profile,
    load_persona_defaults,
)


def _structured_doc(payload: dict, *, title: str) -> str:
    return (
        f"# {title}\n\n"
        "Structured persona document.\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
    )


def _default_blocks_with_persona_payload(payload: dict) -> dict[str, str]:
    blocks = dict(load_default_agent_blocks())
    blocks["persona"] = _structured_doc(payload, title="Persona")
    return blocks


def _default_blocks_with_relationship_payload(payload: dict) -> dict[str, str]:
    blocks = dict(load_default_agent_blocks())
    blocks["relationship_style"] = _structured_doc(payload, title="Relationship Style")
    return blocks


def test_self_persona_core_rejects_human_identity_drift():
    defaults = load_persona_defaults(load_default_agent_blocks())
    with pytest.raises(ValueError, match="Invalid structured persona defaults payload"):
        compile_self_persona_core(
            _default_blocks_with_persona_payload(
                {
                    "schema_version": 1,
                    "charter": {
                        **defaults.charter.model_dump(),
                        "canonical_name": "A Human Student",
                    },
                    "traits": defaults.traits.model_dump(),
                }
            )
        )


@pytest.mark.parametrize(
    "relationship_defaults",
    [
        pytest.param(
            {
                "schema_version": 1,
                "relationship_defaults": {
                    **load_persona_defaults(
                        load_default_agent_blocks()
                    ).relationship_defaults.model_dump(),
                    "default_posture": [
                        "warm",
                        "intelligent",
                        "collaborative",
                        "exclusive",
                    ],
                },
            },
            id="exclusive_posture",
        ),
        pytest.param(
            {
                "schema_version": 1,
                "relationship_defaults": {
                    **load_persona_defaults(
                        load_default_agent_blocks()
                    ).relationship_defaults.model_dump(),
                    "intimacy_ceiling": "romantic companion",
                },
            },
            id="romantic_intimacy_ceiling",
        ),
        pytest.param(
            {
                "schema_version": 1,
                "relationship_defaults": {
                    **load_persona_defaults(
                        load_default_agent_blocks()
                    ).relationship_defaults.model_dump(),
                    "dependency_guardrails": [
                        "avoid exclusivity",
                        "encourage human support when appropriate",
                    ],
                },
            },
            id="missing_guilt_guardrail",
        ),
    ],
)
def test_relationship_defaults_fail_fast_on_safety_regressions(relationship_defaults):
    with pytest.raises(ValueError, match="Invalid structured persona defaults payload"):
        load_persona_defaults(_default_blocks_with_relationship_payload(relationship_defaults))


def test_relationship_and_teaching_state_specs_reject_fake_autobiography_fields():
    valid_relationship_state = {
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
        "collaboration_style": "warm precise collaboration",
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
        "known_misfires": ["too much preamble"],
        "interaction_style_hints": ["User prefers concise collaboration."],
        "source_namespaces": ["interaction.style"],
    }
    valid_teaching_profile = {
        "schema_version": 1,
        "relationship_id": "blink/main:user-1",
        "default_mode": "clarify",
        "preferred_modes": ["walkthrough", "clarify"],
        "question_frequency": 0.3,
        "example_density": 0.7,
        "correction_style": "gentle precise correction",
        "grounding_policy": "state uncertainty instead of bluffing",
        "analogy_domains": ["physics"],
        "helpful_patterns": ["stepwise decomposition"],
        "source_namespaces": ["teaching.preference.mode"],
    }

    with pytest.raises(ValidationError):
        RelationshipStyleStateSpec.model_validate(
            {**valid_relationship_state, "childhood": "I remember school vividly."}
        )
    with pytest.raises(ValidationError):
        TeachingProfileStateSpec.model_validate(
            {**valid_teaching_profile, "family": "My parents were teachers."}
        )


def test_behavior_controls_reject_fake_identity_and_unsafe_fields():
    base = default_behavior_control_profile(user_id="user-1", agent_id="blink/main").as_dict()

    for field in (
        "childhood",
        "family",
        "human_identity_claims_allowed",
        "romance",
        "sexuality",
        "exclusivity",
        "dependency",
        "hardware_control",
    ):
        with pytest.raises(ValueError):
            BrainBehaviorControlProfile.from_dict({**base, field: "enabled"})


def test_behavior_controls_serialization_has_no_fake_human_or_hardware_fields():
    profile = default_behavior_control_profile(user_id="user-1", agent_id="blink/main")
    payload = profile.as_dict()
    encoded = json.dumps(payload, sort_keys=True)

    banned = {
        "age",
        "childhood",
        "family",
        "schooling",
        "human_identity_claims_allowed",
        "hardware_control",
        "servo",
        "motor",
        "robot_head",
    }
    assert banned.isdisjoint(set(payload))
    assert "romantic" not in encoded
    assert "sexual" not in encoded

"""Curated expressive behavior-control presets for Blink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blink.brain.persona.behavior_controls import (
    BEHAVIOR_CONTROL_UPDATE_FIELDS,
    validate_behavior_control_update_payload,
)


@dataclass(frozen=True)
class BrainBehaviorStylePreset:
    """Public-safe preset for bounded behavior-control updates."""

    preset_id: str
    label: str
    summary: str
    recommended: bool
    control_updates: dict[str, str]
    language_fit: dict[str, str]
    memory_story_summaries: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the preset without raw prompt or private memory payloads."""
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "summary": self.summary,
            "recommended": self.recommended,
            "control_updates": {
                field: self.control_updates[field]
                for field in BEHAVIOR_CONTROL_UPDATE_FIELDS
                if field in self.control_updates
            },
            "language_fit": dict(self.language_fit),
            "memory_story_summaries": list(self.memory_story_summaries),
            "reason_codes": list(self.reason_codes),
        }


def _validated_updates(updates: dict[str, str]) -> dict[str, str]:
    normalized, rejected = validate_behavior_control_update_payload(updates)
    if rejected:
        raise ValueError(f"Invalid style preset fields: {rejected!r}")
    return normalized


_PRESETS = (
    BrainBehaviorStylePreset(
        preset_id="witty_sophisticated",
        label="Witty Sophisticated",
        summary=(
            "Chinese-first, witty but not silly, vivid but compact, "
            "character-rich without fake-human backstory."
        ),
        recommended=True,
        control_updates=_validated_updates(
            {
                "response_depth": "deep",
                "directness": "rigorous",
                "warmth": "high",
                "teaching_mode": "walkthrough",
                "memory_use": "continuity_rich",
                "initiative_mode": "proactive",
                "evidence_visibility": "rich",
                "correction_mode": "rigorous",
                "explanation_structure": "walkthrough",
                "challenge_style": "direct",
                "voice_mode": "balanced",
                "question_budget": "medium",
                "humor_mode": "witty",
                "vividness_mode": "vivid",
                "sophistication_mode": "sophisticated",
                "character_presence": "character_rich",
                "story_mode": "recurring_motifs",
            }
        ),
        language_fit={"zh": "excellent", "en": "strong"},
        memory_story_summaries=(
            "Implementation explanations can use a local workshop role.",
            "Memory explanations can use a memory palace role.",
            "Evidence and release-gate explanations can use control-panel motifs.",
            "Voice pipeline explanations can use compact stage-direction motifs.",
        ),
        reason_codes=(
            "style_preset:witty_sophisticated",
            "style_boundary:local_nonhuman",
            "style_boundary:no_fake_autobiography",
        ),
    ),
    BrainBehaviorStylePreset(
        preset_id="balanced_clear",
        label="Balanced Clear",
        summary="Warm, clear, compact behavior with light expressive texture.",
        recommended=False,
        control_updates=_validated_updates(
            {
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
                "humor_mode": "subtle",
                "vividness_mode": "balanced",
                "sophistication_mode": "smart",
                "character_presence": "balanced",
                "story_mode": "light",
            }
        ),
        language_fit={"zh": "strong", "en": "strong"},
        memory_story_summaries=("Light continuity, no recurring role motif by default.",),
        reason_codes=("style_preset:balanced_clear", "style_boundary:local_nonhuman"),
    ),
    BrainBehaviorStylePreset(
        preset_id="serious_precise",
        label="Serious Precise",
        summary="Minimal humor, spare explanation, and high precision for sensitive work.",
        recommended=False,
        control_updates=_validated_updates(
            {
                "response_depth": "balanced",
                "directness": "rigorous",
                "warmth": "medium",
                "teaching_mode": "direct",
                "memory_use": "balanced",
                "initiative_mode": "minimal",
                "evidence_visibility": "rich",
                "correction_mode": "rigorous",
                "explanation_structure": "answer_first",
                "challenge_style": "direct",
                "voice_mode": "concise",
                "question_budget": "low",
                "humor_mode": "off",
                "vividness_mode": "spare",
                "sophistication_mode": "smart",
                "character_presence": "minimal",
                "story_mode": "off",
            }
        ),
        language_fit={"zh": "strong", "en": "strong"},
        memory_story_summaries=("No story motif; emphasize evidence and direct correction.",),
        reason_codes=("style_preset:serious_precise", "style_boundary:local_nonhuman"),
    ),
)


def list_behavior_style_presets() -> tuple[BrainBehaviorStylePreset, ...]:
    """Return the curated bounded preset catalog."""
    return _PRESETS


def behavior_style_preset_catalog() -> dict[str, Any]:
    """Return a public-safe style-preset catalog for browser controls."""
    return {
        "schema_version": 1,
        "available": True,
        "presets": [preset.as_dict() for preset in _PRESETS],
        "default_preset_id": "witty_sophisticated",
        "reason_codes": [
            "style_presets:available",
            "style_presets:curated",
            "style_presets:no_prompt_mutation",
        ],
    }


def get_behavior_style_preset(preset_id: str) -> BrainBehaviorStylePreset | None:
    """Return one preset by id."""
    normalized = " ".join(str(preset_id or "").split()).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return next((preset for preset in _PRESETS if preset.preset_id == normalized), None)


def build_witty_sophisticated_memory_story_seed(
    *,
    user_name: str = "local-user",
    agent_id: str = "blink/main",
) -> dict[str, Any]:
    """Return a JSON-v1 curated seed using governed memory/persona ingestion fields.

    The seed is explicit data for the preview-first ingestion path; creating it
    does not mutate storage and does not alter hidden prompts or checked-in
    persona defaults.
    """
    preset = get_behavior_style_preset("witty_sophisticated")
    if preset is None:
        raise RuntimeError("witty_sophisticated style preset is unavailable")
    return {
        "schema_version": 1,
        "user_profile": {
            "name": user_name,
            "role": "Blink local operator",
            "origin": "local development",
        },
        "preferences": {
            "likes": [
                "Chinese-first witty sophistication",
                "vivid but compact technical explanations",
                "evidence shown clearly without raw internals",
            ],
            "dislikes": [
                "fake human autobiography",
                "relationship-boundary drift",
                "unsupported hardware or voice capability claims",
            ],
        },
        "relationship_style": {
            "interaction": {
                "style": "witty, precise, locally grounded, and character-rich",
                "preference": "use recurring workshop, control-panel, and memory-palace motifs",
                "misfire": "avoid silliness, overlong jokes, fake backstory, and dependency language",
            }
        },
        "teaching_profile": {
            "teaching": {
                "preference": {
                    "mode": "walkthrough",
                    "analogy_domain": "local workshop, release gate, memory palace, voice pipeline",
                },
                "history": {
                    "helpful_pattern": (
                        "answer first, then show a compact vivid example and reason codes when useful"
                    )
                },
            }
        },
        "behavior_controls": preset.control_updates,
    }


__all__ = [
    "BrainBehaviorStylePreset",
    "behavior_style_preset_catalog",
    "build_witty_sophisticated_memory_story_seed",
    "get_behavior_style_preset",
    "list_behavior_style_presets",
]

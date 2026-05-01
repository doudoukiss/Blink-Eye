"""Public-safe memory/persona performance planning for Blink replies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from blink.brain.memory_v2.use_trace import BrainMemoryUseTrace, BrainMemoryUseTraceRef
from blink.brain.persona.behavior_controls import BrainBehaviorControlProfile
from blink.brain.persona.reference_bank import (
    PersonaReference,
    PersonaReferenceAnchorV3,
    persona_reference_anchors_by_situation_v3,
    persona_references_by_scenario,
)

_SCHEMA_VERSION = 1
_PERFORMANCE_PLAN_V2_SCHEMA_VERSION = 2
_PERFORMANCE_PLAN_V3_SCHEMA_VERSION = 3
_PERSONA_REFERENCE_MODES = (
    "interruption_response",
    "camera_use",
    "memory_callback",
    "disagreement",
    "correction",
    "concise_answer",
    "deep_technical_planning",
    "uncertainty",
)
_PERSONA_ANCHOR_SCENARIO_ALIASES_V3 = {
    "interruption": "interruption_response",
    "correction": "correction_response",
    "deep_technical_planning": "deep_technical_planning",
    "casual_chat": "casual_check_in",
    "camera_use": "visual_grounding",
    "memory_callback": "memory_callback",
    "uncertainty": "uncertainty",
    "disagreement": "disagreement",
    "playful_restraint": "playful_not_fake_human",
}
_BANNED_TEXT_MARKERS = (
    "authorization",
    "bearer",
    "credential",
    "developer_message",
    "developer_prompt",
    "raw_prompt",
    "secret",
    "system_message",
    "system_prompt",
    "traceback",
    "transcript",
    "api_key",
    "audio_bytes",
    "image_bytes",
    "raw_audio",
    "raw_image",
    "sdp_offer",
    "ice_candidate",
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_lower(value: Any) -> str:
    return _normalized_text(value).lower()


def _safe_text(value: Any, *, limit: int = 120) -> str:
    text = _normalized_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in _BANNED_TEXT_MARKERS):
        return "redacted"
    return text[:limit]


def _safe_reason(value: Any, *, fallback: str = "unknown") -> str:
    text = _normalized_lower(value).replace(" ", "_")
    if not text:
        return fallback
    safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in text)
    return "_".join(part for part in safe.split("_") if part)[:80] or fallback


def _dedupe(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _safe_reason(value, fallback="")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    return value is True


def _safe_id(value: Any, *, fallback: str = "turn:auto", limit: int = 96) -> str:
    text = _safe_reason(value, fallback="")
    return text[:limit] or fallback


def _mapping_value(value: Any) -> Mapping[str, Any]:
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        return payload if isinstance(payload, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _continuity_payload(value: Any) -> Mapping[str, Any]:
    data = _mapping_value(value)
    if data:
        return data
    return {}


def _safe_list(values: Any, *, limit: int = 8, text_limit: int = 96) -> tuple[str, ...]:
    raw_values = values if isinstance(values, (list, tuple, set)) else ()
    seen: set[str] = set()
    result: list[str] = []
    for value in raw_values:
        if isinstance(value, Mapping):
            value = value.get("value") or value.get("label") or value.get("kind")
        text = _safe_text(value, limit=text_limit)
        if not text or text == "redacted" or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _locale_for_language(language: Any) -> str:
    text = _safe_reason(getattr(language, "value", language), fallback="unknown")
    if text.startswith("zh"):
        return "zh"
    if text.startswith("en"):
        return "en"
    return "en"


def _behavior_control_value(
    profile: BrainBehaviorControlProfile | dict[str, Any] | None,
    key: str,
    *,
    fallback: str,
) -> str:
    return _behavior_value(profile, key) or fallback


def _behavior_value(profile: BrainBehaviorControlProfile | dict[str, Any] | None, key: str) -> str:
    if profile is None:
        return ""
    if isinstance(profile, dict):
        return _safe_reason(profile.get(key), fallback="")
    return _safe_reason(getattr(profile, key, ""), fallback="")


def _behavior_effects(
    *,
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None,
    memory_use_trace: BrainMemoryUseTrace | None,
    memory_continuity_trace: Mapping[str, Any] | None,
    camera_state: str,
    current_turn_state: str,
) -> tuple[str, ...]:
    memory_use = _behavior_value(behavior_profile, "memory_use") or "balanced"
    response_depth = _behavior_value(behavior_profile, "response_depth") or "balanced"
    correction = _behavior_value(behavior_profile, "correction_mode") or "precise"
    evidence = _behavior_value(behavior_profile, "evidence_visibility") or "compact"
    teaching = _behavior_value(behavior_profile, "teaching_mode") or "auto"
    challenge = _behavior_value(behavior_profile, "challenge_style") or "gentle"
    effects: list[str] = [
        f"memory_{memory_use}",
        f"depth_{response_depth}",
        f"correction_{correction}",
        f"evidence_{evidence}",
        f"teaching_{teaching}",
        f"challenge_{challenge}",
    ]
    if memory_use_trace is not None and memory_use_trace.refs:
        effects.append("memory_callback_active")
    else:
        effects.append("memory_blind_reply")
    continuity = _continuity_payload(memory_continuity_trace)
    continuity_effect = _safe_reason(continuity.get("memory_effect"), fallback="")
    cross_language_count = _safe_int(continuity.get("cross_language_count"))
    if continuity_effect:
        effects.append(f"continuity_{continuity_effect}")
    if cross_language_count > 0:
        effects.append("memory_cross_language_callback")
    if camera_state in {"available", "looking", "waiting_for_frame", "stale", "stalled"}:
        effects.append(f"camera_{camera_state}")
    if current_turn_state:
        effects.append(f"turn_{_safe_reason(current_turn_state)}")
    return _dedupe(effects)


def _persona_reference_effect(mode: str, *, behavior_effects: tuple[str, ...], camera_state: str) -> str:
    if mode == "interruption_response":
        return "protect reply flow unless explicit barge-in is armed"
    if mode == "camera_use":
        if camera_state in {"available", "looking"}:
            return "ground visual claims to one explicit fresh frame"
        return "avoid implying camera understanding when vision is unavailable"
    if mode == "memory_callback":
        return (
            "use selected public memories as brief callbacks"
            if "memory_callback_active" in behavior_effects
            else "avoid ungrounded memory callbacks"
        )
    if mode == "disagreement":
        return "disagree with evidence and no hostility"
    if mode == "correction":
        return "correct stale or wrong assumptions using bounded rationale"
    if mode == "concise_answer":
        return "prefer answer-first concise structure when the turn allows"
    if mode == "deep_technical_planning":
        return "expand into concrete implementation steps for planning tasks"
    return "state uncertainty plainly and ask only when needed"


@dataclass(frozen=True)
class BrainPersonaPerformanceReference:
    """One public-safe persona behavior reference considered for a reply."""

    reference_id: str
    mode: str
    label: str
    applies: bool
    behavior_effect: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this reference without private persona internals."""
        return {
            "reference_id": self.reference_id,
            "mode": self.mode,
            "label": self.label,
            "applies": self.applies,
            "behavior_effect": self.behavior_effect,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainPersonaPerformanceReference":
        """Hydrate one persona reference from a public-safe payload."""
        mode = _safe_reason(data.get("mode"), fallback="uncertainty")
        return cls(
            reference_id=_safe_reason(data.get("reference_id"), fallback=f"persona:{mode}"),
            mode=mode if mode in _PERSONA_REFERENCE_MODES else "uncertainty",
            label=_safe_text(data.get("label"), limit=96) or mode.replace("_", " "),
            applies=data.get("applies") is True,
            behavior_effect=_safe_text(data.get("behavior_effect"), limit=120),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class BrainMemoryPersonaPerformanceRef:
    """One public-safe memory reference included in the performance plan."""

    memory_id: str
    display_kind: str
    title: str
    used_reason: str
    behavior_effect: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this memory reference for browser/operator surfaces."""
        return {
            "memory_id": self.memory_id,
            "display_kind": self.display_kind,
            "title": self.title,
            "used_reason": self.used_reason,
            "behavior_effect": self.behavior_effect,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_trace_ref(
        cls,
        ref: BrainMemoryUseTraceRef,
        *,
        behavior_effect: str,
    ) -> "BrainMemoryPersonaPerformanceRef":
        """Build a safe performance ref from an existing memory-use trace ref."""
        return cls(
            memory_id=_safe_text(ref.memory_id, limit=160),
            display_kind=_safe_reason(ref.display_kind, fallback="memory"),
            title=_safe_text(ref.title, limit=120) or "Memory",
            used_reason=_safe_reason(ref.used_reason, fallback="selected_for_reply_context"),
            behavior_effect=_safe_text(behavior_effect, limit=120),
            reason_codes=_dedupe(ref.reason_codes),
        )


@dataclass(frozen=True)
class PerformancePlanV2ReferenceSummary:
    """Runtime-safe summary of a persona reference selected for this turn."""

    reference_id: str
    locale: str
    scenario: str
    stance: str
    response_shape: str
    performance_notes: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize without example input/output or hidden persona prose."""
        return {
            "reference_id": self.reference_id,
            "locale": self.locale,
            "scenario": self.scenario,
            "stance": self.stance,
            "response_shape": self.response_shape,
            "performance_notes": list(self.performance_notes),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_reference(cls, reference: PersonaReference) -> "PerformancePlanV2ReferenceSummary":
        """Build a runtime summary from a full reference-bank item."""
        summary = reference.public_summary()
        return cls(
            reference_id=_safe_text(summary.get("reference_id"), limit=96),
            locale=_safe_reason(summary.get("locale"), fallback="en"),
            scenario=_safe_reason(summary.get("scenario"), fallback="casual_chat"),
            stance=_safe_text(summary.get("stance"), limit=140),
            response_shape=_safe_text(summary.get("response_shape"), limit=180),
            performance_notes=_safe_list(summary.get("performance_notes"), limit=3, text_limit=120),
            reason_codes=_dedupe(summary.get("reason_codes") or ()),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerformancePlanV2ReferenceSummary":
        """Hydrate a selected-reference summary from a public payload."""
        return cls(
            reference_id=_safe_text(data.get("reference_id"), limit=96),
            locale=_safe_reason(data.get("locale"), fallback="en"),
            scenario=_safe_reason(data.get("scenario"), fallback="casual_chat"),
            stance=_safe_text(data.get("stance"), limit=140),
            response_shape=_safe_text(data.get("response_shape"), limit=180),
            performance_notes=_safe_list(data.get("performance_notes"), limit=3, text_limit=120),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class PerformancePlanV3PersonaAnchorSummary:
    """Text-free selected V3 persona anchor summary for plans and traces."""

    schema_version: int
    anchor_id: str
    situation_key: str
    stance_label: str
    response_shape_label: str
    behavior_constraint_count: int
    negative_example_count: int
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize without examples, hidden prompts, or persona prose."""
        return {
            "schema_version": self.schema_version,
            "anchor_id": self.anchor_id,
            "situation_key": self.situation_key,
            "stance_label": self.stance_label,
            "response_shape_label": self.response_shape_label,
            "behavior_constraint_count": self.behavior_constraint_count,
            "negative_example_count": self.negative_example_count,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_anchor(
        cls,
        anchor: PersonaReferenceAnchorV3,
    ) -> "PerformancePlanV3PersonaAnchorSummary":
        """Build a selected-anchor summary from a public anchor catalog item."""
        summary = anchor.public_summary()
        return cls(
            schema_version=_safe_int(summary.get("schema_version"), 3),
            anchor_id=_safe_text(summary.get("anchor_id"), limit=96),
            situation_key=_safe_reason(summary.get("situation_key"), fallback="unknown"),
            stance_label=_safe_reason(summary.get("stance_label"), fallback="focused"),
            response_shape_label=_safe_reason(
                summary.get("response_shape_label"),
                fallback="answer_first",
            ),
            behavior_constraint_count=_safe_int(summary.get("behavior_constraint_count")),
            negative_example_count=_safe_int(summary.get("negative_example_count")),
            reason_codes=_dedupe(summary.get("reason_codes") or ()),
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "PerformancePlanV3PersonaAnchorSummary":
        """Hydrate a selected V3 anchor summary from a public payload."""
        return cls(
            schema_version=_safe_int(data.get("schema_version"), 3),
            anchor_id=_safe_text(data.get("anchor_id"), limit=96),
            situation_key=_safe_reason(data.get("situation_key"), fallback="unknown"),
            stance_label=_safe_reason(data.get("stance_label"), fallback="focused"),
            response_shape_label=_safe_reason(
                data.get("response_shape_label"),
                fallback="answer_first",
            ),
            behavior_constraint_count=_safe_int(data.get("behavior_constraint_count")),
            negative_example_count=_safe_int(data.get("negative_example_count")),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class PerformancePlanV2:
    """Public-safe conversation performance plan for one assistant turn."""

    schema_version: int
    turn_id: str
    profile: str
    modality: str
    language: str
    tts_label: str
    floor_state: str
    visible_mode: str
    stance: str
    response_shape: str
    memory_callback_policy: dict[str, Any]
    camera_reference_policy: dict[str, Any]
    interruption_policy: dict[str, Any]
    speech_chunking_hints: dict[str, Any]
    ui_state_hints: dict[str, Any]
    style_summary: str
    persona_references_used: tuple[PerformancePlanV2ReferenceSummary, ...]
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the v2 plan for browser/UI diagnostics."""
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "profile": self.profile,
            "modality": self.modality,
            "language": self.language,
            "tts_label": self.tts_label,
            "floor_state": self.floor_state,
            "visible_mode": self.visible_mode,
            "stance": self.stance,
            "response_shape": self.response_shape,
            "memory_callback_policy": dict(self.memory_callback_policy),
            "camera_reference_policy": dict(self.camera_reference_policy),
            "interruption_policy": dict(self.interruption_policy),
            "speech_chunking_hints": dict(self.speech_chunking_hints),
            "ui_state_hints": dict(self.ui_state_hints),
            "style_summary": self.style_summary,
            "persona_references_used": [
                reference.as_dict() for reference in self.persona_references_used
            ],
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PerformancePlanV2 | None":
        """Hydrate a public v2 plan if one is present."""
        if not isinstance(data, Mapping):
            return None
        references = tuple(
            PerformancePlanV2ReferenceSummary.from_dict(item)
            for item in data.get("persona_references_used", ())
            if isinstance(item, Mapping)
        )
        return cls(
            schema_version=_safe_int(
                data.get("schema_version"),
                _PERFORMANCE_PLAN_V2_SCHEMA_VERSION,
            ),
            turn_id=_safe_id(data.get("turn_id")),
            profile=_safe_text(data.get("profile"), limit=96) or "manual",
            modality=_safe_reason(data.get("modality"), fallback="browser"),
            language=_locale_for_language(data.get("language")),
            tts_label=_safe_text(data.get("tts_label"), limit=96) or "unknown",
            floor_state=_safe_reason(data.get("floor_state"), fallback="unknown"),
            visible_mode=_safe_reason(data.get("visible_mode"), fallback="waiting"),
            stance=_safe_text(data.get("stance"), limit=140),
            response_shape=_safe_reason(data.get("response_shape"), fallback="answer_first"),
            memory_callback_policy=_safe_policy_payload(
                data.get("memory_callback_policy"),
                fallback_state="avoid_ungrounded_callback",
            ),
            camera_reference_policy=_safe_policy_payload(
                data.get("camera_reference_policy"),
                fallback_state="no_visual_claim",
            ),
            interruption_policy=_safe_policy_payload(
                data.get("interruption_policy"),
                fallback_state="protected_continue",
            ),
            speech_chunking_hints=_safe_policy_payload(
                data.get("speech_chunking_hints"),
                fallback_state="voice_suitable_chunks",
            ),
            ui_state_hints=_safe_policy_payload(
                data.get("ui_state_hints"),
                fallback_state="style_summary_available",
            ),
            style_summary=_safe_text(data.get("style_summary"), limit=180),
            persona_references_used=references,
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


def _stable_digest(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _profile_for_v3(profile: Any, *, language: str, tts_label: str) -> str:
    safe_profile = _safe_text(profile, limit=96)
    if safe_profile in {"browser-zh-melo", "browser-en-kokoro"}:
        return safe_profile
    safe_tts = _safe_text(tts_label, limit=96).lower()
    if language == "en" or "kokoro" in safe_tts:
        return "browser-en-kokoro"
    return "browser-zh-melo"


def _tts_backend_from_label(tts_label: str) -> str:
    label = _safe_text(tts_label, limit=96).lower()
    if "kokoro" in label:
        return "kokoro"
    if "local-http-wav" in label or "melo" in label:
        return "local-http-wav"
    return _safe_reason(tts_label, fallback="provider-neutral")


def _safe_policy_object(value: Any, *, fallback_state: str) -> dict[str, Any]:
    public = _safe_policy_payload(value, fallback_state=fallback_state)
    reason_codes = public.get("reason_codes")
    if isinstance(reason_codes, list):
        public["reason_codes"] = list(_dedupe(reason_codes))
    elif "reason_codes" not in public:
        public["reason_codes"] = [f"policy:{public['state']}"]
    return public


def _actor_control_payload(value: Any) -> Mapping[str, Any]:
    return _mapping_value(value)


def _actor_control_ref(value: Any) -> dict[str, Any]:
    payload = _actor_control_payload(value)
    if not payload:
        return {
            "frame_id": "actor-control:unavailable",
            "sequence": 0,
            "boundary": "unavailable",
            "condition_cache_digest": "unavailable",
            "source_event_ids": [],
        }
    source_event_ids = [
        _safe_int(item)
        for item in list(payload.get("source_event_ids") or [])[:12]
        if _safe_int(item) > 0
    ]
    return {
        "frame_id": _safe_text(payload.get("frame_id"), limit=96) or "actor-control:unknown",
        "sequence": _safe_int(payload.get("sequence")),
        "boundary": _safe_reason(payload.get("boundary"), fallback="unknown"),
        "condition_cache_digest": _safe_text(payload.get("condition_cache_digest"), limit=32)
        or "unavailable",
        "source_event_ids": source_event_ids,
    }


def _safe_created_at_ms(*values: Any) -> int:
    for value in values:
        parsed = _safe_int(value, default=-1)
        if parsed >= 0:
            return parsed
    return 0


def _policy_from_actor_or_fallback(
    actor_control_frame: Any,
    key: str,
    fallback: Any,
    *,
    fallback_state: str,
) -> dict[str, Any]:
    actor_payload = _actor_control_payload(actor_control_frame)
    actor_policy = actor_payload.get(key) if isinstance(actor_payload, Mapping) else None
    if isinstance(actor_policy, Mapping):
        return _safe_policy_object(actor_policy, fallback_state=fallback_state)
    return _safe_policy_object(fallback, fallback_state=fallback_state)


def _nested_scene_social_state(camera_scene: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = camera_scene.get("scene_social_state_v2")
    return nested if isinstance(nested, Mapping) else {}


def _camera_policy_v3(
    *,
    actor_control_frame: Any,
    camera_scene: Mapping[str, Any],
    v2_policy: Mapping[str, Any],
) -> dict[str, Any]:
    actor_policy = _policy_from_actor_or_fallback(
        actor_control_frame,
        "camera_policy",
        v2_policy,
        fallback_state="no_visual_claim",
    )
    scene_social = _nested_scene_social_state(camera_scene)
    honesty_state = _safe_reason(
        actor_policy.get("camera_honesty_state") or scene_social.get("camera_honesty_state"),
        fallback="unavailable",
    )
    fresh_frame_used = _safe_bool(
        actor_policy.get("fresh_frame_used")
        or actor_policy.get("current_answer_used_vision")
        or camera_scene.get("current_answer_used_vision")
    )
    try:
        object_showing_raw = float(
            actor_policy.get("object_showing_likelihood")
            or scene_social.get("object_showing_likelihood")
            or 0.0
        )
    except (TypeError, ValueError):
        object_showing_raw = 0.0
    object_showing_likelihood = min(1000, _safe_int(round(object_showing_raw * 1000)))
    if fresh_frame_used and honesty_state == "can_see_now":
        state = "fresh_visual_grounding"
        no_visual_claim = False
    elif honesty_state == "recent_frame_available":
        state = "recent_frame_available"
        no_visual_claim = True
    elif honesty_state == "available_not_used":
        state = "available_not_used"
        no_visual_claim = True
    elif actor_policy.get("state") in {"fresh_used", "fresh_single_frame"}:
        state = "fresh_visual_grounding"
        no_visual_claim = False
    else:
        state = "no_visual_claim"
        no_visual_claim = True
    return {
        "state": state,
        "camera_honesty_state": honesty_state,
        "fresh_frame_used": fresh_frame_used,
        "scene_transition": _safe_reason(
            actor_policy.get("scene_transition") or scene_social.get("scene_transition"),
            fallback="none",
        ),
        "object_showing_likelihood_milli": object_showing_likelihood,
        "no_visual_claim": no_visual_claim,
        "reason_codes": list(
            _dedupe(
                (
                    f"camera_policy_v3:{state}",
                    f"camera_honesty:{honesty_state}",
                    *actor_policy.get("reason_codes", ()),
                )
            )
        ),
    }


def _memory_policy_v3(
    *,
    v2_policy: Mapping[str, Any],
    memory_continuity_trace: Mapping[str, Any] | None,
    selected_memory_count: int,
    suppressed_memory_count: int,
) -> dict[str, Any]:
    continuity = _continuity_payload(memory_continuity_trace)
    continuity_v3 = _mapping_value(continuity.get("memory_continuity_v3"))
    cross_language_count = _safe_int(continuity.get("cross_language_count"))
    memory_effect = _safe_reason(continuity.get("memory_effect"), fallback="")
    v2_state = _safe_reason(v2_policy.get("state"), fallback="avoid_ungrounded_callback")
    selected_memory_refs: list[dict[str, Any]] = []
    for item in continuity.get("selected_memories", ()):
        if not isinstance(item, Mapping):
            continue
        selected_memory_refs.append(
            {
                "memory_id": _safe_text(item.get("memory_id"), limit=160)
                or "memory:unknown",
                "display_kind": _safe_reason(item.get("display_kind"), fallback="memory"),
                "summary": _safe_text(item.get("summary"), limit=120)
                or "Memory selected.",
                "source_language": _safe_reason(item.get("source_language"), fallback="unknown"),
                "cross_language": item.get("cross_language") is True,
                "effect_labels": list(_dedupe(item.get("effect_labels") or ("none",)))[:8],
                "confidence_bucket": _safe_reason(
                    item.get("confidence_bucket"),
                    fallback="medium",
                ),
                "reason_codes": list(_dedupe(item.get("reason_codes") or ()))[:12],
            }
        )
        if len(selected_memory_refs) >= 8:
            break
    discourse_episode_ids = [
        _safe_text(item.get("discourse_episode_id"), limit=120)
        for item in continuity_v3.get("selected_discourse_episodes", ())
        if isinstance(item, Mapping)
    ][:8]
    discourse_category_values: list[Any] = []
    for item in continuity_v3.get("selected_discourse_episodes", ()):
        if not isinstance(item, Mapping):
            continue
        labels = item.get("category_labels")
        if isinstance(labels, (list, tuple, set)):
            discourse_category_values.extend(labels)
    discourse_category_labels = list(_dedupe(discourse_category_values))[:8]
    effect_labels = list(
        _dedupe(
            (
                *(continuity_v3.get("effect_labels") or ()),
                *(label for ref in selected_memory_refs for label in ref.get("effect_labels", ())),
            )
        )
    )[:8]
    if not effect_labels:
        effect_labels = ["none"]
    conflict_labels = list(_dedupe(continuity_v3.get("conflict_labels") or ()))[:8]
    staleness_labels = list(_dedupe(continuity_v3.get("staleness_labels") or ()))[:8]
    if v2_state in {"cross_language_callback", "use_brief_callback"}:
        state = v2_state
    elif selected_memory_count > 0 and v2_state != "available_but_minimal":
        state = "brief_callback_available"
    elif (
        suppressed_memory_count > 0
        or memory_effect == "repair_or_uncertainty"
        or "suppressed_stale_memory" in effect_labels
    ):
        state = "suppress_stale_callback"
    else:
        state = "no_callback"
    return {
        "state": state,
        "selected_memory_count": selected_memory_count,
        "suppressed_memory_count": suppressed_memory_count,
        "cross_language_count": cross_language_count,
        "cross_language_transfer_count": _safe_int(
            continuity_v3.get("cross_language_transfer_count")
        ),
        "memory_effect": memory_effect or ("callback_available" if selected_memory_count else "none"),
        "selected_memory_ids": [ref["memory_id"] for ref in selected_memory_refs],
        "selected_memory_refs": selected_memory_refs,
        "discourse_episode_ids": [item for item in discourse_episode_ids if item],
        "discourse_category_labels": discourse_category_labels,
        "effect_labels": effect_labels,
        "conflict_labels": conflict_labels,
        "staleness_labels": staleness_labels,
        "reason_codes": list(
            _dedupe(
                (
                    f"memory_policy_v3:{state}",
                    *(f"memory_effect:{label}" for label in effect_labels),
                    *(f"conflict:{label}" for label in conflict_labels),
                    *(f"staleness:{label}" for label in staleness_labels),
                    *v2_policy.get("reason_codes", ()),
                )
            )
        ),
    }


def _interruption_policy_v3(
    *,
    actor_control_frame: Any,
    v2_policy: Mapping[str, Any],
    protected_playback: bool,
) -> dict[str, Any]:
    actor_policy = _policy_from_actor_or_fallback(
        actor_control_frame,
        "repair_policy",
        v2_policy,
        fallback_state="protected_continue",
    )
    floor_policy = _policy_from_actor_or_fallback(
        actor_control_frame,
        "floor_policy",
        {},
        fallback_state="unknown",
    )
    outcome = _safe_reason(actor_policy.get("interruption_outcome"), fallback="none")
    floor_sub_state = _safe_reason(floor_policy.get("sub_state"), fallback="unknown")
    if protected_playback and outcome in {"candidate", "accepted", "interrupted"}:
        state = "protected_visible_overlap"
    elif outcome in {"accepted", "interrupted", "output_flushed"}:
        state = "yield_fast"
    elif floor_sub_state in {"repair_requested", "accepted_interrupt"}:
        state = "repair_first"
    elif protected_playback:
        state = "protected"
    else:
        state = "finish_short_phrase"
    return {
        "state": state,
        "protected_playback": protected_playback,
        "interruption_outcome": outcome,
        "floor_sub_state": floor_sub_state,
        "stale_output_action": _safe_reason(actor_policy.get("stale_output_action"), fallback="none"),
        "reason_codes": list(
            _dedupe(
                (
                    f"interruption_policy_v3:{state}",
                    *actor_policy.get("reason_codes", ()),
                    *floor_policy.get("reason_codes", ()),
                )
            )
        ),
    }


def _repair_policy_v3(
    *,
    interruption_policy: Mapping[str, Any],
    floor_state: str,
) -> dict[str, Any]:
    state = "repair_first" if interruption_policy.get("state") in {"repair_first", "yield_fast"} else "normal"
    if floor_state == "repair":
        state = "repair_first"
    return {
        "state": state,
        "repair_first": state == "repair_first",
        "stale_output_action": _safe_reason(
            interruption_policy.get("stale_output_action"),
            fallback="none",
        ),
        "reason_codes": list(
            _dedupe((f"repair_policy_v3:{state}", *interruption_policy.get("reason_codes", ())))
        ),
    }


def _voice_capabilities_v3(
    *,
    tts_label: str,
    voice_capabilities: Any = None,
    voice_actuation_plan: Any = None,
) -> dict[str, Any]:
    capabilities = _mapping_value(voice_capabilities)
    actuation = _mapping_value(voice_actuation_plan)
    backend_label = (
        _safe_text(actuation.get("backend_label"), limit=80)
        or _safe_text(capabilities.get("backend_label"), limit=80)
        or _tts_backend_from_label(tts_label)
    )
    chunk_boundaries = (
        _safe_bool(actuation.get("chunk_boundaries_enabled"))
        if actuation
        else capabilities.get("supports_chunk_boundaries") is not False
    )
    interruption_flush = (
        _safe_bool(actuation.get("interruption_flush_enabled"))
        if actuation
        else capabilities.get("supports_interruption_flush") is not False
    )
    speech_rate = (
        _safe_bool(actuation.get("speech_rate_enabled"))
        if actuation
        else capabilities.get("supports_speech_rate") is True
    )
    prosody = (
        _safe_bool(actuation.get("prosody_emphasis_enabled"))
        if actuation
        else capabilities.get("supports_prosody_emphasis") is True
    )
    pause_timing = (
        _safe_bool(actuation.get("pause_timing_enabled"))
        if actuation
        else capabilities.get("supports_pause_timing") is True
    )
    stream_abort = (
        _safe_bool(actuation.get("partial_stream_abort_enabled"))
        if actuation
        else capabilities.get("supports_partial_stream_abort") is True
    )
    discard = (
        _safe_bool(actuation.get("interruption_discard_enabled"))
        if actuation
        else capabilities.get("supports_interruption_discard") is True
    )
    hardware = False
    if tts_label in {"local-http-wav/MeloTTS", "kokoro/English"} or backend_label in {
        "local-http-wav",
        "kokoro",
    }:
        speech_rate = False
        prosody = False
        pause_timing = False
        stream_abort = False
        discard = False
    unsupported: list[str] = []
    if not speech_rate:
        unsupported.append("speech_rate")
    if not prosody:
        unsupported.append("prosody_emphasis")
    if not pause_timing:
        unsupported.append("pause_timing")
    if not stream_abort:
        unsupported.append("partial_stream_abort")
    if not discard:
        unsupported.append("interruption_discard")
    unsupported.append("hardware_control")
    return {
        "backend_label": backend_label,
        "chunk_boundaries_enabled": chunk_boundaries,
        "interruption_flush_enabled": interruption_flush,
        "speech_rate_enabled": speech_rate,
        "prosody_emphasis_enabled": prosody,
        "pause_timing_enabled": pause_timing,
        "partial_stream_abort_enabled": stream_abort,
        "interruption_discard_enabled": discard,
        "expression_controls_hardware": hardware,
        "unsupported_controls": list(_dedupe(unsupported)),
        "reason_codes": list(
            _dedupe(
                (
                    "tts_capabilities:v3",
                    f"tts_backend:{backend_label}",
                    *actuation.get("reason_codes", ()),
                    *capabilities.get("reason_codes", ()),
                )
            )
        ),
    }


def _speech_budget_v3(
    *,
    language: str,
    actor_control_frame: Any,
    interruption_policy: Mapping[str, Any],
    camera_policy: Mapping[str, Any],
    memory_policy: Mapping[str, Any],
    tts_capabilities: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actor_payload = _actor_control_payload(actor_control_frame)
    lookahead = actor_payload.get("lookahead_counters")
    lookahead = lookahead if isinstance(lookahead, Mapping) else {}
    repair = interruption_policy.get("state") in {"repair_first", "yield_fast", "protected_visible_overlap"}
    visual = camera_policy.get("state") == "fresh_visual_grounding"
    memory_callback = memory_policy.get("state") in {
        "use_brief_callback",
        "cross_language_callback",
        "brief_callback_available",
    }
    if repair:
        state = "repair_short"
        target = 72 if language == "zh" else 64
        hard_max = 120
        max_flush = 1
    elif visual:
        state = "visual_grounding_compact"
        target = 100 if language == "zh" else 92
        hard_max = 160
        max_flush = 2
    elif memory_callback:
        state = "callback_then_answer"
        target = 120 if language == "zh" else 104
        hard_max = 180
        max_flush = 2
    else:
        state = "balanced"
        target = 140 if language == "zh" else 112
        hard_max = 220 if language == "zh" else 180
        max_flush = 3
    speech_limit = _safe_int(lookahead.get("speech_chunks_limit"), 2) or 2
    subtitle_limit = _safe_int(lookahead.get("subtitles_limit"), 2) or 2
    if not tts_capabilities.get("chunk_boundaries_enabled"):
        state = "tts_chunk_boundaries_unavailable"
        target = 0
        hard_max = 0
        max_flush = 0
    budget = {
        "state": state,
        "target_chars": target,
        "hard_max_chars": hard_max,
        "max_chunks_per_flush": max_flush,
        "speech_lookahead_limit": speech_limit,
        "subtitle_lookahead_limit": subtitle_limit,
        "reason_codes": [f"speech_chunk_budget_v3:{state}"],
    }
    voice = {
        "state": state,
        "target_chars": target,
        "hard_max_chars": hard_max,
        "max_chunks_per_flush": max_flush,
        "lookahead_label": f"speech:{speech_limit}/subtitle:{subtitle_limit}",
        "reason_codes": [f"voice_pacing_v3:{state}"],
    }
    subtitle = {
        "state": "immediate_first_then_bounded" if subtitle_limit else "unavailable",
        "immediate_first_subtitle": subtitle_limit > 0,
        "max_outstanding": subtitle_limit,
        "reason_codes": ["subtitle_policy_v3:bounded"],
    }
    return voice, budget, subtitle


def _plan_summary_copy(
    *,
    language: str,
    stance: str,
    response_shape: str,
    camera_policy: Mapping[str, Any],
    memory_policy: Mapping[str, Any],
    repair_policy: Mapping[str, Any],
) -> tuple[str, str]:
    camera_state = _safe_reason(camera_policy.get("state"), fallback="no_visual_claim")
    memory_state = _safe_reason(memory_policy.get("state"), fallback="no_callback")
    repair_state = _safe_reason(repair_policy.get("state"), fallback="normal")
    if language == "zh":
        if repair_state == "repair_first":
            status = "先修复打断，再简短回答。"
        elif camera_state == "fresh_visual_grounding":
            status = "使用刚获取的画面做视觉对齐。"
        elif memory_state in {"use_brief_callback", "cross_language_callback", "brief_callback_available"}:
            status = "用一条安全记忆作简短承接。"
        else:
            status = "保持简洁、受保护的回答节奏。"
        summary = f"{status} 形态：{response_shape}；姿态：{stance}。"
    else:
        if repair_state == "repair_first":
            status = "Repair the interruption first, then answer briefly."
        elif camera_state == "fresh_visual_grounding":
            status = "Use the fresh camera frame for visual grounding."
        elif memory_state in {"use_brief_callback", "cross_language_callback", "brief_callback_available"}:
            status = "Use one grounded memory callback before answering."
        else:
            status = "Keep the answer concise with protected playback."
        summary = f"{status} Shape: {response_shape}; stance: {stance}."
    return _safe_text(status, limit=180), _safe_text(summary, limit=180)


@dataclass(frozen=True)
class PerformancePlanV3:
    """Public-safe response-performance plan for one assistant turn."""

    schema_version: int
    plan_id: str
    turn_id: str
    profile: str
    language: str
    tts_runtime_label: str
    created_at_ms: int
    actor_control_ref: dict[str, Any]
    stance: str
    response_shape: str
    voice_pacing: dict[str, Any]
    speech_chunk_budget: dict[str, Any]
    subtitle_policy: dict[str, Any]
    camera_reference_policy: dict[str, Any]
    memory_callback_policy: dict[str, Any]
    interruption_policy: dict[str, Any]
    repair_policy: dict[str, Any]
    ui_status_copy: str
    plan_summary: str
    persona_reference_ids: tuple[str, ...]
    persona_anchor_refs_v3: tuple[PerformancePlanV3PersonaAnchorSummary, ...]
    tts_capabilities: dict[str, Any]
    reason_trace: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the v3 plan for public runtime surfaces."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "turn_id": self.turn_id,
            "profile": self.profile,
            "language": self.language,
            "tts_runtime_label": self.tts_runtime_label,
            "created_at_ms": self.created_at_ms,
            "actor_control_ref": dict(self.actor_control_ref),
            "stance": self.stance,
            "response_shape": self.response_shape,
            "voice_pacing": dict(self.voice_pacing),
            "speech_chunk_budget": dict(self.speech_chunk_budget),
            "subtitle_policy": dict(self.subtitle_policy),
            "camera_reference_policy": dict(self.camera_reference_policy),
            "memory_callback_policy": dict(self.memory_callback_policy),
            "interruption_policy": dict(self.interruption_policy),
            "repair_policy": dict(self.repair_policy),
            "ui_status_copy": self.ui_status_copy,
            "plan_summary": self.plan_summary,
            "persona_reference_ids": list(self.persona_reference_ids),
            "persona_anchor_refs_v3": [
                anchor.as_dict() for anchor in self.persona_anchor_refs_v3
            ],
            "tts_capabilities": dict(self.tts_capabilities),
            "reason_trace": list(self.reason_trace),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PerformancePlanV3 | None":
        """Hydrate a public v3 plan if one is present."""
        if not isinstance(data, Mapping):
            return None
        language = _locale_for_language(data.get("language"))
        tts_label = _safe_text(data.get("tts_runtime_label"), limit=96) or "unknown"
        return cls(
            schema_version=_safe_int(data.get("schema_version"), _PERFORMANCE_PLAN_V3_SCHEMA_VERSION),
            plan_id=_safe_text(data.get("plan_id"), limit=96) or "performance-plan-v3:unknown",
            turn_id=_safe_id(data.get("turn_id")),
            profile=_profile_for_v3(data.get("profile"), language=language, tts_label=tts_label),
            language=language,
            tts_runtime_label=tts_label,
            created_at_ms=_safe_int(data.get("created_at_ms")),
            actor_control_ref=_safe_policy_payload(
                data.get("actor_control_ref"),
                fallback_state="unavailable",
            ),
            stance=_safe_reason(data.get("stance"), fallback="focused"),
            response_shape=_safe_reason(data.get("response_shape"), fallback="answer_first"),
            voice_pacing=_safe_policy_payload(data.get("voice_pacing"), fallback_state="balanced"),
            speech_chunk_budget=_safe_policy_payload(
                data.get("speech_chunk_budget"),
                fallback_state="balanced",
            ),
            subtitle_policy=_safe_policy_payload(
                data.get("subtitle_policy"),
                fallback_state="immediate_first_then_bounded",
            ),
            camera_reference_policy=_safe_policy_payload(
                data.get("camera_reference_policy"),
                fallback_state="no_visual_claim",
            ),
            memory_callback_policy=_safe_policy_payload(
                data.get("memory_callback_policy"),
                fallback_state="no_callback",
            ),
            interruption_policy=_safe_policy_payload(
                data.get("interruption_policy"),
                fallback_state="protected",
            ),
            repair_policy=_safe_policy_payload(data.get("repair_policy"), fallback_state="normal"),
            ui_status_copy=_safe_text(data.get("ui_status_copy"), limit=180),
            plan_summary=_safe_text(data.get("plan_summary"), limit=180),
            persona_reference_ids=_dedupe(data.get("persona_reference_ids") or ()),
            persona_anchor_refs_v3=tuple(
                PerformancePlanV3PersonaAnchorSummary.from_dict(item)
                for item in data.get("persona_anchor_refs_v3", ())
                if isinstance(item, Mapping)
            ),
            tts_capabilities=_safe_policy_payload(
                data.get("tts_capabilities"),
                fallback_state="unknown",
            ),
            reason_trace=_dedupe(data.get("reason_trace") or ()),
        )


@dataclass(frozen=True)
class BrainMemoryPersonaPerformancePlan:
    """Public-safe behavior plan showing how memory/persona shaped one reply."""

    schema_version: int
    available: bool
    profile: str
    modality: str
    language: str
    tts_label: str
    protected_playback: bool
    camera_state: str
    continuous_perception_enabled: bool
    current_turn_state: str
    memory_policy: str
    selected_memory_count: int
    suppressed_memory_count: int
    selected_memories: tuple[BrainMemoryPersonaPerformanceRef, ...]
    behavior_effects: tuple[str, ...]
    persona_references: tuple[BrainPersonaPerformanceReference, ...]
    summary: str
    performance_plan_v2: PerformancePlanV2 | None = None
    performance_plan_v3: PerformancePlanV3 | None = None
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize this plan for public runtime APIs."""
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "profile": self.profile,
            "modality": self.modality,
            "language": self.language,
            "tts_label": self.tts_label,
            "protected_playback": self.protected_playback,
            "camera_state": self.camera_state,
            "continuous_perception_enabled": self.continuous_perception_enabled,
            "current_turn_state": self.current_turn_state,
            "memory_policy": self.memory_policy,
            "selected_memory_count": self.selected_memory_count,
            "suppressed_memory_count": self.suppressed_memory_count,
            "used_in_current_reply": [ref.as_dict() for ref in self.selected_memories],
            "behavior_effects": list(self.behavior_effects),
            "persona_references": [ref.as_dict() for ref in self.persona_references],
            "summary": self.summary,
            "performance_plan_v2": (
                self.performance_plan_v2.as_dict() if self.performance_plan_v2 else None
            ),
            "performance_plan_v3": (
                self.performance_plan_v3.as_dict() if self.performance_plan_v3 else None
            ),
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainMemoryPersonaPerformancePlan":
        """Hydrate a plan from a public payload."""
        selected = tuple(
            BrainMemoryPersonaPerformanceRef(
                memory_id=_safe_text(item.get("memory_id"), limit=160),
                display_kind=_safe_reason(item.get("display_kind"), fallback="memory"),
                title=_safe_text(item.get("title"), limit=120) or "Memory",
                used_reason=_safe_reason(item.get("used_reason"), fallback="selected"),
                behavior_effect=_safe_text(item.get("behavior_effect"), limit=120),
                reason_codes=_dedupe(item.get("reason_codes") or ()),
            )
            for item in data.get("used_in_current_reply", ())
            if isinstance(item, dict)
        )
        persona_references = tuple(
            BrainPersonaPerformanceReference.from_dict(item)
            for item in data.get("persona_references", ())
            if isinstance(item, dict)
        )
        return cls(
            schema_version=_safe_int(data.get("schema_version"), _SCHEMA_VERSION),
            available=data.get("available") is True,
            profile=_safe_text(data.get("profile"), limit=96) or "manual",
            modality=_safe_reason(data.get("modality"), fallback="browser"),
            language=_safe_reason(data.get("language"), fallback="unknown"),
            tts_label=_safe_text(data.get("tts_label"), limit=96) or "unknown",
            protected_playback=data.get("protected_playback") is not False,
            camera_state=_safe_reason(data.get("camera_state"), fallback="unknown"),
            continuous_perception_enabled=data.get("continuous_perception_enabled") is True,
            current_turn_state=_safe_reason(data.get("current_turn_state"), fallback="unknown"),
            memory_policy=_safe_reason(data.get("memory_policy"), fallback="balanced"),
            selected_memory_count=_safe_int(data.get("selected_memory_count"), len(selected)),
            suppressed_memory_count=_safe_int(data.get("suppressed_memory_count")),
            selected_memories=selected,
            behavior_effects=_dedupe(data.get("behavior_effects") or ()),
            persona_references=persona_references,
            summary=_safe_text(data.get("summary"), limit=180),
            performance_plan_v2=PerformancePlanV2.from_dict(data.get("performance_plan_v2")),
            performance_plan_v3=PerformancePlanV3.from_dict(data.get("performance_plan_v3")),
            reason_codes=_dedupe(data.get("reason_codes") or ()),
        )


def _compile_persona_references(
    *,
    behavior_effects: tuple[str, ...],
    camera_state: str,
    current_turn_state: str,
    selected_memory_count: int,
) -> tuple[BrainPersonaPerformanceReference, ...]:
    active_modes = {
        "interruption_response",
        "disagreement",
        "correction",
        "uncertainty",
    }
    if selected_memory_count > 0:
        active_modes.add("memory_callback")
    if camera_state in {"available", "looking", "waiting_for_frame", "stale", "stalled"}:
        active_modes.add("camera_use")
    if current_turn_state in {"thinking", "looking"}:
        active_modes.add("deep_technical_planning")
    if any(effect in behavior_effects for effect in ("depth_concise", "turn_speaking")):
        active_modes.add("concise_answer")
    return tuple(
        BrainPersonaPerformanceReference(
            reference_id=f"persona:{mode}",
            mode=mode,
            label=mode.replace("_", " "),
            applies=mode in active_modes,
            behavior_effect=_persona_reference_effect(
                mode,
                behavior_effects=behavior_effects,
                camera_state=camera_state,
            ),
            reason_codes=(
                "persona_reference:v1",
                f"persona_reference:{mode}",
                "persona_reference:applies" if mode in active_modes else "persona_reference:standby",
            ),
        )
        for mode in _PERSONA_REFERENCE_MODES
    )


def _safe_policy_payload(value: Any, *, fallback_state: str) -> dict[str, Any]:
    data = _mapping_value(value)
    result: dict[str, Any] = {
        "state": _safe_reason(data.get("state"), fallback=fallback_state),
    }
    for key, raw_value in list(data.items())[:16]:
        safe_key = _safe_reason(key, fallback="")
        if not safe_key or safe_key in {"state"}:
            continue
        if isinstance(raw_value, bool):
            result[safe_key] = raw_value
        elif isinstance(raw_value, int):
            result[safe_key] = max(0, raw_value)
        elif isinstance(raw_value, float):
            result[safe_key] = max(0, int(raw_value))
        elif isinstance(raw_value, (list, tuple, set)):
            if safe_key == "selected_memory_refs":
                result[safe_key] = [
                    {
                        "memory_id": _safe_text(item.get("memory_id"), limit=160)
                        or "memory:unknown",
                        "display_kind": _safe_reason(
                            item.get("display_kind"),
                            fallback="memory",
                        ),
                        "summary": _safe_text(item.get("summary"), limit=120)
                        or "Memory selected.",
                        "source_language": _safe_reason(
                            item.get("source_language"),
                            fallback="unknown",
                        ),
                        "cross_language": item.get("cross_language") is True,
                        "effect_labels": list(
                            _dedupe(item.get("effect_labels") or ("none",))
                        )[:8],
                        "confidence_bucket": _safe_reason(
                            item.get("confidence_bucket"),
                            fallback="medium",
                        ),
                        "reason_codes": list(_dedupe(item.get("reason_codes") or ()))[:12],
                    }
                    for item in list(raw_value)[:8]
                    if isinstance(item, Mapping)
                ]
            else:
                result[safe_key] = list(_safe_list(raw_value, limit=8, text_limit=80))
        elif isinstance(raw_value, Mapping):
            result[safe_key] = {
                _safe_reason(nested_key, fallback="item"): _safe_text(nested_value, limit=80)
                for nested_key, nested_value in list(raw_value.items())[:8]
            }
        else:
            result[safe_key] = _safe_text(raw_value, limit=96)
    return result


def _reference_from_mapping(data: Mapping[str, Any]) -> PersonaReference | None:
    reference_id = _safe_text(data.get("id") or data.get("reference_id"), limit=96)
    locale = _locale_for_language(data.get("locale") or data.get("language"))
    scenario = _safe_reason(data.get("scenario") or data.get("mode"), fallback="")
    if not reference_id or not scenario:
        return None
    return PersonaReference(
        id=reference_id,
        locale=locale,
        scenario=scenario,
        stance=_safe_text(data.get("stance"), limit=140),
        response_shape=_safe_text(data.get("response_shape"), limit=180),
        forbidden_moves=_safe_list(data.get("forbidden_moves"), limit=8, text_limit=140),
        example_input=_safe_text(data.get("example_input"), limit=240),
        example_output=_safe_text(data.get("example_output"), limit=360),
        performance_notes=_safe_list(data.get("performance_notes"), limit=8, text_limit=160),
    )


def _reference_bank_for_locale(
    *,
    locale: str,
    persona_references: Iterable[PersonaReference | Mapping[str, Any]] | None,
) -> dict[str, PersonaReference]:
    if persona_references is None:
        return persona_references_by_scenario(locale=locale)
    rows: list[PersonaReference] = []
    for value in persona_references:
        if isinstance(value, PersonaReference):
            reference = value
        elif isinstance(value, Mapping):
            reference = _reference_from_mapping(value)
            if reference is None:
                continue
        else:
            continue
        if _locale_for_language(reference.locale) == locale:
            rows.append(reference)
    return {row.scenario: row for row in rows}


def _hint_counts(active_listening: Mapping[str, Any]) -> dict[str, int]:
    return {
        "topic_count": len(_safe_list(active_listening.get("topics"), limit=5)),
        "constraint_count": len(_safe_list(active_listening.get("constraints"), limit=5)),
        "correction_count": len(_safe_list(active_listening.get("corrections"), limit=5)),
        "project_reference_count": len(
            _safe_list(active_listening.get("project_references"), limit=5)
        ),
        "uncertainty_count": len(_safe_list(active_listening.get("uncertainty_flags"), limit=5)),
    }


def _floor_state_value(floor_state: Mapping[str, Any] | str | None) -> str:
    if isinstance(floor_state, Mapping):
        return _safe_reason(floor_state.get("state"), fallback="unknown")
    return _safe_reason(floor_state, fallback="unknown")


def _camera_policy(
    *,
    camera_state: str,
    camera_scene: Mapping[str, Any],
) -> dict[str, Any]:
    scene_state = _safe_reason(camera_scene.get("state") or camera_scene.get("status"), fallback="")
    state = scene_state or camera_state
    used_vision = _safe_bool(camera_scene.get("current_answer_used_vision"))
    grounding_mode = _safe_reason(camera_scene.get("grounding_mode"), fallback="none")
    if used_vision and state in {"available", "looking"}:
        policy_state = "fresh_single_frame"
    elif state == "disabled":
        policy_state = "disabled"
    elif state == "permission_needed":
        policy_state = "permission_needed"
    elif state in {"stale", "stalled"}:
        policy_state = "stale_limited_context"
    elif state == "error":
        policy_state = "vision_error"
    elif state in {"available", "looking"}:
        policy_state = "available_on_demand"
    else:
        policy_state = "no_visual_claim"
    return {
        "state": policy_state,
        "camera_state": state or "unknown",
        "current_answer_used_vision": used_vision,
        "grounding_mode": grounding_mode,
        "last_used_frame_sequence": _safe_int(camera_scene.get("last_used_frame_sequence")),
    }


def _memory_policy(
    *,
    memory_use_trace: BrainMemoryUseTrace | None,
    memory_continuity_trace: Mapping[str, Any] | None,
    suppressed_memory_count: int,
    memory_policy: str,
) -> dict[str, Any]:
    continuity = _continuity_payload(memory_continuity_trace)
    selected_count = len(memory_use_trace.refs) if memory_use_trace is not None else 0
    continuity_selected_count = _safe_int(continuity.get("selected_memory_count"))
    selected_count = max(selected_count, continuity_selected_count)
    continuity_suppressed_count = _safe_int(continuity.get("suppressed_memory_count"))
    suppressed_count = max(_safe_int(suppressed_memory_count), continuity_suppressed_count)
    cross_language_count = _safe_int(continuity.get("cross_language_count"))
    memory_effect = _safe_reason(continuity.get("memory_effect"), fallback="")
    if cross_language_count > 0 and memory_policy != "minimal":
        state = "cross_language_callback"
    elif selected_count > 0 and memory_policy != "minimal":
        state = "use_brief_callback"
    elif selected_count > 0:
        state = "available_but_minimal"
    elif suppressed_count > 0 or memory_effect == "repair_or_uncertainty":
        state = "avoid_stale_or_suppressed_callback"
    else:
        state = "avoid_ungrounded_callback"
    display_kinds = (
        sorted({ref.display_kind for ref in memory_use_trace.refs})
        if memory_use_trace is not None and memory_use_trace.refs
        else []
    )
    if continuity and not display_kinds:
        display_kinds = sorted(
            {
                _safe_reason(item.get("display_kind"), fallback="memory")
                for item in continuity.get("selected_memories", ())
                if isinstance(item, Mapping)
            }
        )
    return {
        "state": state,
        "memory_policy": memory_policy,
        "selected_memory_count": selected_count,
        "suppressed_memory_count": suppressed_count,
        "cross_language_count": cross_language_count,
        "memory_effect": memory_effect or ("callback_available" if selected_count else "none"),
        "display_kinds": display_kinds[:6],
    }


def _interruption_policy(
    *,
    floor_state: str,
    protected_playback: bool,
    user_intent: str,
) -> dict[str, Any]:
    interruption_requested = user_intent in {"interruption", "explicit_interruption"}
    if protected_playback and floor_state in {"overlap", "repair"}:
        state = "protected_repair_without_auto_yield"
    elif protected_playback:
        state = "protected_continue"
    elif floor_state in {"overlap", "repair"} or interruption_requested:
        state = "yield_on_armed_overlap"
    else:
        state = "armed_listen_for_explicit_interruption"
    return {
        "state": state,
        "protected_playback": protected_playback,
        "floor_state": floor_state,
        "explicit_interruption": interruption_requested,
    }


def _selected_scenarios(
    *,
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None,
    memory_policy: dict[str, Any],
    camera_policy: dict[str, Any],
    interruption_policy: dict[str, Any],
    hint_counts: dict[str, int],
    floor_state: str,
    current_turn_state: str,
    user_intent: str,
) -> tuple[str, ...]:
    scenarios: list[str] = []
    if interruption_policy["state"] in {
        "protected_repair_without_auto_yield",
        "yield_on_armed_overlap",
    }:
        scenarios.append("interruption")
    if floor_state == "repair" or hint_counts["correction_count"] > 0:
        scenarios.append("correction")
    if user_intent in {"disagreement", "challenge"}:
        scenarios.append("disagreement")
    if memory_policy["state"] in {
        "use_brief_callback",
        "available_but_minimal",
        "cross_language_callback",
    }:
        scenarios.append("memory_callback")
    if camera_policy["state"] != "disabled":
        scenarios.append("camera_use")
    response_depth = _behavior_control_value(
        behavior_profile,
        "response_depth",
        fallback="balanced",
    )
    explanation_structure = _behavior_control_value(
        behavior_profile,
        "explanation_structure",
        fallback="answer_first",
    )
    if (
        response_depth == "deep"
        or explanation_structure == "walkthrough"
        or hint_counts["constraint_count"] + hint_counts["project_reference_count"] >= 2
        or (
            current_turn_state in {"thinking", "looking"}
            and user_intent not in {"concise", "quick_answer"}
        )
    ):
        scenarios.append("deep_technical_planning")
    if response_depth == "concise" or user_intent in {"concise", "quick_answer"}:
        scenarios.append("concise_answer")
    if hint_counts["uncertainty_count"] > 0 or camera_policy["state"] in {
        "stale_limited_context",
        "vision_error",
        "permission_needed",
        "no_visual_claim",
    }:
        scenarios.append("uncertainty")
    humor_mode = _behavior_control_value(behavior_profile, "humor_mode", fallback="witty")
    serious_repair = any(item in scenarios for item in ("interruption", "correction"))
    if humor_mode in {"witty", "playful"} and user_intent in {"casual", "chat"} and not serious_repair:
        scenarios.append("playful_restraint")
    if not scenarios:
        scenarios.append("casual_chat" if user_intent in {"casual", "chat"} else "concise_answer")
    seen: set[str] = set()
    result: list[str] = []
    for scenario in scenarios:
        if scenario not in seen:
            seen.add(scenario)
            result.append(scenario)
    return tuple(result[:6])


def _plan_stance_and_shape(
    *,
    scenarios: tuple[str, ...],
    response_depth: str,
) -> tuple[str, str]:
    if "correction" in scenarios:
        return "repair_with_precise_correction", "repair_then_answer"
    if "interruption" in scenarios:
        return "yield_or_continue_by_policy", "brief_handoff_then_answer"
    if "deep_technical_planning" in scenarios:
        return "concrete_engineering_planning", "plan_steps"
    if "uncertainty" in scenarios:
        return "state_uncertainty_early", "knowns_unknowns_next_check"
    if "memory_callback" in scenarios:
        return "continuity_without_private_exposition", "brief_callback_then_answer"
    if response_depth == "concise" or "concise_answer" in scenarios:
        return "answer_first", "answer_first"
    if "playful_restraint" in scenarios:
        return "light_presence_then_work", "small_wit_then_next_action"
    return "warm_pragmatic_collaboration", "casual_compact"


def _style_summary(*, locale: str, scenarios: tuple[str, ...], stance: str) -> str:
    scenario_text = ", ".join(scenarios[:4])
    if locale == "zh":
        return f"本轮风格：{stance}；参考动作：{scenario_text}。"
    return f"Style this turn: {stance}; references: {scenario_text}."


def _persona_anchor_keys_from_v2(v2: PerformancePlanV2) -> tuple[str, ...]:
    keys: list[str] = []
    for reference in v2.persona_references_used:
        key = _PERSONA_ANCHOR_SCENARIO_ALIASES_V3.get(reference.scenario)
        if key:
            keys.append(key)
    return _dedupe(keys)


def _selected_persona_anchor_keys_v3(
    *,
    v2: PerformancePlanV2,
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None,
    memory_policy: Mapping[str, Any],
    camera_policy: Mapping[str, Any],
    interruption_policy: Mapping[str, Any],
    repair_policy: Mapping[str, Any],
    floor_state: str,
    detected_intent: str,
    user_intent: Mapping[str, Any] | str | None,
    ready_to_answer: bool,
) -> tuple[str, ...]:
    keys: list[str] = list(_persona_anchor_keys_from_v2(v2))
    user_intent_payload = _mapping_value(user_intent)
    user_intent_label = (
        _safe_reason(user_intent_payload.get("intent"), fallback="unknown")
        if user_intent_payload
        else _safe_reason(user_intent, fallback="unknown")
    )
    floor = _safe_reason(floor_state, fallback="unknown")
    if repair_policy.get("state") == "repair_first" or floor == "repair":
        keys.extend(("interruption_response", "correction_response"))
    if interruption_policy.get("state") in {
        "accepted",
        "protected",
        "protected_repair_without_auto_yield",
        "yield_on_armed_overlap",
    }:
        keys.append("interruption_response")
    if detected_intent == "correction" or user_intent_label in {"correction", "repair"}:
        keys.append("correction_response")
    if detected_intent == "project_planning" or user_intent_label in {
        "project_planning",
        "planning",
        "technical_planning",
    }:
        keys.append("deep_technical_planning")
    if detected_intent in {"small_talk", "unknown"} and user_intent_label in {
        "casual",
        "chat",
        "small_talk",
    }:
        keys.append("casual_check_in")
    if camera_policy.get("state") not in {"disabled"}:
        keys.append("visual_grounding")
    if camera_policy.get("state") in {
        "no_visual_claim",
        "stale_limited_context",
        "vision_error",
        "permission_needed",
    }:
        keys.append("uncertainty")
    if detected_intent == "object_showing":
        keys.append("visual_grounding")
    if memory_policy.get("state") in {
        "use_brief_callback",
        "cross_language_callback",
        "brief_callback_available",
    }:
        keys.append("memory_callback")
    if user_intent_label in {"disagreement", "challenge"}:
        keys.append("disagreement")
    if detected_intent == "unknown" and not ready_to_answer:
        keys.append("uncertainty")
    humor_mode = _behavior_control_value(behavior_profile, "humor_mode", fallback="witty")
    serious = any(
        key in keys
        for key in ("interruption_response", "correction_response", "disagreement")
    )
    if humor_mode in {"witty", "playful"} and user_intent_label in {
        "casual",
        "chat",
        "small_talk",
    } and not serious:
        keys.append("playful_not_fake_human")
    if not keys:
        keys.append("casual_check_in")
    return _dedupe(keys)[:8]


def _persona_anchor_summaries_v3(
    situation_keys: Iterable[Any],
) -> tuple[PerformancePlanV3PersonaAnchorSummary, ...]:
    bank = persona_reference_anchors_by_situation_v3()
    summaries: list[PerformancePlanV3PersonaAnchorSummary] = []
    for key in _dedupe(situation_keys):
        anchor = bank.get(key)
        if anchor is not None:
            summaries.append(PerformancePlanV3PersonaAnchorSummary.from_anchor(anchor))
    return tuple(summaries[:8])


def compile_performance_plan_v2(
    *,
    profile: str = "manual",
    modality: str = "browser",
    language: str = "unknown",
    tts_label: str = "unknown",
    protected_playback: bool = True,
    camera_state: str = "unknown",
    continuous_perception_enabled: bool = False,
    current_turn_state: str = "unknown",
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None = None,
    memory_use_trace: BrainMemoryUseTrace | None = None,
    memory_continuity_trace: Mapping[str, Any] | None = None,
    suppressed_memory_count: int = 0,
    active_listening: Mapping[str, Any] | None = None,
    camera_scene: Mapping[str, Any] | None = None,
    floor_state: Mapping[str, Any] | str | None = None,
    user_affect: Mapping[str, Any] | str | None = None,
    user_intent: Mapping[str, Any] | str | None = None,
    persona_references: Iterable[PersonaReference | Mapping[str, Any]] | None = None,
    turn_id: str = "",
    extra_reason_codes: Iterable[Any] = (),
) -> PerformancePlanV2:
    """Compile a deterministic public-safe persona performance plan v2."""
    safe_profile = _safe_text(profile, limit=96) or "manual"
    safe_modality = _safe_reason(modality, fallback="browser")
    locale = _locale_for_language(language)
    safe_tts = _safe_text(tts_label, limit=96) or "unknown"
    safe_camera = _safe_reason(camera_state, fallback="unknown")
    safe_turn = _safe_reason(current_turn_state, fallback="unknown")
    active_listening_payload = _mapping_value(active_listening)
    camera_scene_payload = _mapping_value(camera_scene)
    floor_state_value = _floor_state_value(floor_state)
    user_intent_payload = _mapping_value(user_intent)
    if user_intent_payload:
        safe_user_intent = _safe_reason(user_intent_payload.get("intent"), fallback="unknown")
    else:
        safe_user_intent = _safe_reason(user_intent, fallback="unknown")
    if safe_user_intent == "unknown" and active_listening_payload.get("ready_to_answer") is True:
        safe_user_intent = "answer"
    if safe_user_intent == "unknown" and active_listening_payload.get("phase") == "idle":
        safe_user_intent = "casual"
    hint_counts = _hint_counts(active_listening_payload)
    memory_policy_value = _behavior_control_value(
        behavior_profile,
        "memory_use",
        fallback="balanced",
    )
    response_depth = _behavior_control_value(
        behavior_profile,
        "response_depth",
        fallback="balanced",
    )
    voice_mode = _behavior_control_value(behavior_profile, "voice_mode", fallback="balanced")
    memory_policy = _memory_policy(
        memory_use_trace=memory_use_trace,
        memory_continuity_trace=memory_continuity_trace,
        suppressed_memory_count=suppressed_memory_count,
        memory_policy=memory_policy_value,
    )
    camera_policy = _camera_policy(
        camera_state=safe_camera,
        camera_scene=camera_scene_payload,
    )
    interruption = _interruption_policy(
        floor_state=floor_state_value,
        protected_playback=protected_playback,
        user_intent=safe_user_intent,
    )
    scenarios = _selected_scenarios(
        behavior_profile=behavior_profile,
        memory_policy=memory_policy,
        camera_policy=camera_policy,
        interruption_policy=interruption,
        hint_counts=hint_counts,
        floor_state=floor_state_value,
        current_turn_state=safe_turn,
        user_intent=safe_user_intent,
    )
    stance, response_shape = _plan_stance_and_shape(
        scenarios=scenarios,
        response_depth=response_depth,
    )
    reference_bank = _reference_bank_for_locale(
        locale=locale,
        persona_references=persona_references,
    )
    selected_references = tuple(
        PerformancePlanV2ReferenceSummary.from_reference(reference_bank[scenario])
        for scenario in scenarios
        if scenario in reference_bank
    )
    visible_mode = "repair" if floor_state_value == "repair" else safe_turn
    if visible_mode == "unknown":
        visible_mode = "waiting"
    affect_payload = _mapping_value(user_affect)
    affect_state = (
        _safe_reason(affect_payload.get("state"), fallback="unknown")
        if affect_payload
        else _safe_reason(user_affect, fallback="unknown")
    )
    speech_hints = {
        "state": "voice_suitable_chunks",
        "voice_mode": voice_mode,
        "chunking": "short" if response_depth == "concise" else "balanced",
        "interruptible": True,
        "subtitle_ready": True,
    }
    ui_hints = {
        "state": "style_summary_available",
        "visible_mode": visible_mode,
        "style_chip": stance,
        "selected_reference_count": len(selected_references),
        "ready_to_answer": active_listening_payload.get("ready_to_answer") is True,
    }
    reason_codes = _dedupe(
        (
            "performance_plan:v2",
            f"profile:{safe_profile}",
            f"language:{locale}",
            f"stance:{stance}",
            f"response_shape:{response_shape}",
            f"memory_policy:{memory_policy['state']}",
            f"camera_policy:{camera_policy['state']}",
            f"interruption_policy:{interruption['state']}",
            f"floor_state:{floor_state_value}",
            f"user_intent:{safe_user_intent}",
            f"user_affect:{affect_state}",
            "continuous_perception:on"
            if continuous_perception_enabled
            else "continuous_perception:off",
            *tuple(f"hint:{key}:{count}" for key, count in sorted(hint_counts.items())),
            *tuple(f"persona_reference:{scenario}" for scenario in scenarios),
            *extra_reason_codes,
        )
    )
    resolved_turn_id = _safe_id(
        turn_id
        or (
            f"turn:{safe_profile}:{locale}:{visible_mode}:"
            f"{memory_policy['selected_memory_count']}:{len(selected_references)}"
        )
    )
    return PerformancePlanV2(
        schema_version=_PERFORMANCE_PLAN_V2_SCHEMA_VERSION,
        turn_id=resolved_turn_id,
        profile=safe_profile,
        modality=safe_modality,
        language=locale,
        tts_label=safe_tts,
        floor_state=floor_state_value,
        visible_mode=visible_mode,
        stance=stance,
        response_shape=response_shape,
        memory_callback_policy=memory_policy,
        camera_reference_policy=camera_policy,
        interruption_policy=interruption,
        speech_chunking_hints=speech_hints,
        ui_state_hints=ui_hints,
        style_summary=_safe_text(
            _style_summary(locale=locale, scenarios=scenarios, stance=stance),
            limit=180,
        ),
        persona_references_used=selected_references,
        reason_codes=reason_codes,
    )


def compile_performance_plan_v3(
    *,
    profile: str = "manual",
    modality: str = "browser",
    language: str = "unknown",
    tts_label: str = "unknown",
    protected_playback: bool = True,
    camera_state: str = "unknown",
    continuous_perception_enabled: bool = False,
    current_turn_state: str = "unknown",
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None = None,
    memory_use_trace: BrainMemoryUseTrace | None = None,
    memory_continuity_trace: Mapping[str, Any] | None = None,
    suppressed_memory_count: int = 0,
    active_listening: Mapping[str, Any] | None = None,
    camera_scene: Mapping[str, Any] | None = None,
    floor_state: Mapping[str, Any] | str | None = None,
    user_affect: Mapping[str, Any] | str | None = None,
    user_intent: Mapping[str, Any] | str | None = None,
    persona_references: Iterable[PersonaReference | Mapping[str, Any]] | None = None,
    performance_plan_v2: PerformancePlanV2 | Mapping[str, Any] | None = None,
    actor_control_frame: Mapping[str, Any] | Any | None = None,
    voice_capabilities: Any = None,
    voice_actuation_plan: Any = None,
    turn_id: str = "",
    created_at_ms: int | None = None,
    extra_reason_codes: Iterable[Any] = (),
) -> PerformancePlanV3:
    """Compile a deterministic public-safe performance plan from bounded runtime state."""
    locale = _locale_for_language(language)
    safe_tts = _safe_text(tts_label, limit=96) or "unknown"
    safe_profile = _profile_for_v3(profile, language=locale, tts_label=safe_tts)
    safe_modality = _safe_reason(modality, fallback="browser")
    safe_camera = _safe_reason(camera_state, fallback="unknown")
    safe_turn = _safe_reason(current_turn_state, fallback="unknown")
    camera_scene_payload = _mapping_value(camera_scene)
    active_listening_payload = _mapping_value(active_listening)
    continuity_payload = _continuity_payload(memory_continuity_trace)
    selected_memory_count = max(
        len(memory_use_trace.refs) if memory_use_trace is not None else 0,
        _safe_int(continuity_payload.get("selected_memory_count")),
    )
    suppressed_count = max(
        _safe_int(suppressed_memory_count),
        _safe_int(continuity_payload.get("suppressed_memory_count")),
    )
    if isinstance(performance_plan_v2, PerformancePlanV2):
        v2 = performance_plan_v2
    else:
        v2 = PerformancePlanV2.from_dict(performance_plan_v2) if performance_plan_v2 else None
    if v2 is None:
        v2 = compile_performance_plan_v2(
            profile=safe_profile,
            modality=safe_modality,
            language=locale,
            tts_label=safe_tts,
            protected_playback=protected_playback,
            camera_state=safe_camera,
            continuous_perception_enabled=continuous_perception_enabled,
            current_turn_state=safe_turn,
            behavior_profile=behavior_profile,
            memory_use_trace=memory_use_trace,
            memory_continuity_trace=memory_continuity_trace,
            suppressed_memory_count=suppressed_count,
            active_listening=active_listening_payload,
            camera_scene=camera_scene_payload,
            floor_state=floor_state,
            user_affect=user_affect,
            user_intent=user_intent,
            persona_references=persona_references,
            turn_id=turn_id,
            extra_reason_codes=extra_reason_codes,
        )

    actor_payload = _actor_control_payload(actor_control_frame)
    actor_ref = _actor_control_ref(actor_control_frame)
    actor_created_at = actor_payload.get("created_at_ms") if actor_payload else None
    tts_capabilities = _voice_capabilities_v3(
        tts_label=safe_tts,
        voice_capabilities=voice_capabilities,
        voice_actuation_plan=voice_actuation_plan,
    )
    camera_policy = _camera_policy_v3(
        actor_control_frame=actor_control_frame,
        camera_scene=camera_scene_payload,
        v2_policy=v2.camera_reference_policy,
    )
    memory_policy = _memory_policy_v3(
        v2_policy=v2.memory_callback_policy,
        memory_continuity_trace=memory_continuity_trace,
        selected_memory_count=selected_memory_count,
        suppressed_memory_count=suppressed_count,
    )
    interruption_policy = _interruption_policy_v3(
        actor_control_frame=actor_control_frame,
        v2_policy=v2.interruption_policy,
        protected_playback=protected_playback,
    )
    repair_policy = _repair_policy_v3(
        interruption_policy=interruption_policy,
        floor_state=_floor_state_value(floor_state),
    )
    voice_pacing, speech_budget, subtitle_policy = _speech_budget_v3(
        language=locale,
        actor_control_frame=actor_control_frame,
        interruption_policy=interruption_policy,
        camera_policy=camera_policy,
        memory_policy=memory_policy,
        tts_capabilities=tts_capabilities,
    )
    active_listener_policy = (
        actor_payload.get("active_listener_policy")
        if isinstance(actor_payload.get("active_listener_policy"), Mapping)
        else {}
    )
    semantic_state = active_listening_payload.get("semantic_state_v3")
    semantic_state = semantic_state if isinstance(semantic_state, Mapping) else {}
    detected_intent = _safe_reason(
        active_listener_policy.get("detected_intent") or semantic_state.get("detected_intent"),
        fallback="unknown",
    )
    ready_to_answer = (
        active_listener_policy.get("ready_to_answer") is True
        or semantic_state.get("enough_information_to_answer") is True
        or active_listening_payload.get("ready_to_answer") is True
    )
    if repair_policy["state"] == "repair_first":
        stance = "repairing"
        response_shape = "repair_then_answer"
    elif "corrected_preference" in memory_policy.get("effect_labels", ()):
        stance = "memory_repair"
        response_shape = "repair_then_answer"
    elif camera_policy["state"] == "fresh_visual_grounding":
        stance = "visually_grounded"
        response_shape = "visual_grounding"
    elif "project_constraint_recall" in memory_policy.get("effect_labels", ()):
        stance = "grounded_project_continuity"
        response_shape = "plan_steps"
    elif "shorter_explanation" in memory_policy.get("effect_labels", ()):
        stance = "concise_memory_callback"
        response_shape = "answer_first"
    elif "avoid_repetition" in memory_policy.get("effect_labels", ()):
        stance = "avoid_repeating_failed_pattern"
        response_shape = "concise_next_step"
    elif memory_policy["state"] in {
        "use_brief_callback",
        "cross_language_callback",
        "brief_callback_available",
    }:
        stance = "grounded_callback"
        response_shape = "callback_then_answer"
    elif detected_intent == "project_planning":
        stance = "concrete_engineering_planning"
        response_shape = "plan_steps"
    elif ready_to_answer:
        stance = _safe_reason(v2.stance, fallback="focused")
        response_shape = _safe_reason(v2.response_shape, fallback="answer_first")
    else:
        stance = "attentive_listening"
        response_shape = "wait_then_answer"
    ui_status_copy, plan_summary = _plan_summary_copy(
        language=locale,
        stance=stance,
        response_shape=response_shape,
        camera_policy=camera_policy,
        memory_policy=memory_policy,
        repair_policy=repair_policy,
    )
    persona_reference_ids = _dedupe(
        reference.reference_id for reference in v2.persona_references_used
    )
    persona_anchor_refs_v3 = _persona_anchor_summaries_v3(
        _selected_persona_anchor_keys_v3(
            v2=v2,
            behavior_profile=behavior_profile,
            memory_policy=memory_policy,
            camera_policy=camera_policy,
            interruption_policy=interruption_policy,
            repair_policy=repair_policy,
            floor_state=_floor_state_value(floor_state),
            detected_intent=detected_intent,
            user_intent=user_intent,
            ready_to_answer=ready_to_answer,
        )
    )
    resolved_turn_id = _safe_id(turn_id or v2.turn_id)
    resolved_created_at_ms = _safe_created_at_ms(created_at_ms, actor_created_at, 0)
    reason_trace = _dedupe(
        (
            "performance_plan:v3",
            f"profile:{safe_profile}",
            f"language:{locale}",
            f"tts:{_tts_backend_from_label(safe_tts)}",
            f"stance:{stance}",
            f"response_shape:{response_shape}",
            f"voice_pacing:{voice_pacing['state']}",
            f"camera_policy:{camera_policy['state']}",
            f"memory_policy:{memory_policy['state']}",
            f"interruption_policy:{interruption_policy['state']}",
            f"repair_policy:{repair_policy['state']}",
            f"detected_intent:{detected_intent}",
            "ready_to_answer:on" if ready_to_answer else "ready_to_answer:off",
            "continuous_perception:on"
            if continuous_perception_enabled
            else "continuous_perception:off",
            *v2.reason_codes,
            *tuple(
                f"persona_anchor:{anchor.situation_key}"
                for anchor in persona_anchor_refs_v3
            ),
            *tts_capabilities.get("reason_codes", ()),
            *extra_reason_codes,
        )
    )
    plan_id = "performance-plan-v3:" + _stable_digest(
        {
            "turn_id": resolved_turn_id,
            "profile": safe_profile,
            "language": locale,
            "tts": safe_tts,
            "actor": actor_ref,
            "stance": stance,
            "response_shape": response_shape,
            "voice_pacing": voice_pacing,
            "camera": camera_policy,
            "memory": memory_policy,
            "interruption": interruption_policy,
            "repair": repair_policy,
            "persona_anchors": [anchor.situation_key for anchor in persona_anchor_refs_v3],
        }
    )
    return PerformancePlanV3(
        schema_version=_PERFORMANCE_PLAN_V3_SCHEMA_VERSION,
        plan_id=plan_id,
        turn_id=resolved_turn_id,
        profile=safe_profile,
        language=locale,
        tts_runtime_label=safe_tts,
        created_at_ms=resolved_created_at_ms,
        actor_control_ref=actor_ref,
        stance=stance,
        response_shape=response_shape,
        voice_pacing=voice_pacing,
        speech_chunk_budget=speech_budget,
        subtitle_policy=subtitle_policy,
        camera_reference_policy=camera_policy,
        memory_callback_policy=memory_policy,
        interruption_policy=interruption_policy,
        repair_policy=repair_policy,
        ui_status_copy=ui_status_copy,
        plan_summary=plan_summary,
        persona_reference_ids=persona_reference_ids,
        persona_anchor_refs_v3=persona_anchor_refs_v3,
        tts_capabilities=tts_capabilities,
        reason_trace=reason_trace,
    )


def compile_performance_plan_v3_from_actor_control(
    actor_control_frame: Mapping[str, Any] | Any,
    **kwargs: Any,
) -> PerformancePlanV3:
    """Compile a v3 performance plan from one ActorControlFrameV3 payload."""
    actor_payload = _actor_control_payload(actor_control_frame)
    return compile_performance_plan_v3(
        actor_control_frame=actor_control_frame,
        profile=kwargs.pop("profile", actor_payload.get("profile") or "manual"),
        language=kwargs.pop("language", actor_payload.get("language") or "unknown"),
        tts_label=kwargs.pop(
            "tts_label",
            actor_payload.get("tts_runtime_label")
            or actor_payload.get("tts_label")
            or "unknown",
        ),
        created_at_ms=kwargs.pop("created_at_ms", actor_payload.get("created_at_ms")),
        **kwargs,
    )


def compile_memory_persona_performance_plan(
    *,
    profile: str = "manual",
    modality: str = "browser",
    language: str = "unknown",
    tts_label: str = "unknown",
    protected_playback: bool = True,
    camera_state: str = "unknown",
    continuous_perception_enabled: bool = False,
    current_turn_state: str = "unknown",
    behavior_profile: BrainBehaviorControlProfile | dict[str, Any] | None = None,
    memory_use_trace: BrainMemoryUseTrace | None = None,
    memory_continuity_trace: Mapping[str, Any] | None = None,
    suppressed_memory_count: int = 0,
    active_listening: Mapping[str, Any] | None = None,
    camera_scene: Mapping[str, Any] | None = None,
    floor_state: Mapping[str, Any] | str | None = None,
    user_affect: Mapping[str, Any] | str | None = None,
    user_intent: Mapping[str, Any] | str | None = None,
    actor_control_frame: Mapping[str, Any] | Any | None = None,
    voice_capabilities: Any = None,
    voice_actuation_plan: Any = None,
    turn_id: str = "",
    extra_reason_codes: Iterable[Any] = (),
) -> BrainMemoryPersonaPerformancePlan:
    """Compile a deterministic public-safe plan for memory/persona influence."""
    safe_profile = _safe_text(profile, limit=96) or "manual"
    safe_modality = _safe_reason(modality, fallback="browser")
    safe_language = _safe_reason(language, fallback="unknown")
    safe_tts = _safe_text(tts_label, limit=96) or "unknown"
    safe_camera = _safe_reason(camera_state, fallback="unknown")
    safe_turn = _safe_reason(current_turn_state, fallback="unknown")
    memory_policy = _behavior_value(behavior_profile, "memory_use") or "balanced"
    behavior_effects = _behavior_effects(
        behavior_profile=behavior_profile,
        memory_use_trace=memory_use_trace,
        memory_continuity_trace=memory_continuity_trace,
        camera_state=safe_camera,
        current_turn_state=safe_turn,
    )
    continuity_payload = _continuity_payload(memory_continuity_trace)
    continuity_effect = _safe_reason(continuity_payload.get("memory_effect"), fallback="")
    if continuity_effect == "cross_language_callback":
        memory_effect = "cross-language memory callback shaped this reply"
    elif memory_use_trace is not None and memory_use_trace.refs:
        memory_effect = "memory callback changed this reply"
    elif continuity_effect == "repair_or_uncertainty":
        memory_effect = "suppressed or stale memory shaped repair wording"
    else:
        memory_effect = "no memory callback used"
    selected_memories = tuple(
        BrainMemoryPersonaPerformanceRef.from_trace_ref(ref, behavior_effect=memory_effect)
        for ref in (memory_use_trace.refs if memory_use_trace is not None else ())
    )
    persona_references = _compile_persona_references(
        behavior_effects=behavior_effects,
        camera_state=safe_camera,
        current_turn_state=safe_turn,
        selected_memory_count=len(selected_memories),
    )
    summary = (
        f"{len(selected_memories)} memories used; "
        f"{max(_safe_int(suppressed_memory_count), _safe_int(continuity_payload.get('suppressed_memory_count')))} suppressed; "
        f"{sum(1 for ref in persona_references if ref.applies)} persona references active."
    )
    performance_plan_v2 = compile_performance_plan_v2(
        profile=safe_profile,
        modality=safe_modality,
        language=safe_language,
        tts_label=safe_tts,
        protected_playback=protected_playback,
        camera_state=safe_camera,
        continuous_perception_enabled=continuous_perception_enabled,
        current_turn_state=safe_turn,
        behavior_profile=behavior_profile,
        memory_use_trace=memory_use_trace,
        memory_continuity_trace=memory_continuity_trace,
        suppressed_memory_count=suppressed_memory_count,
        active_listening=active_listening,
        camera_scene=camera_scene,
        floor_state=floor_state,
        user_affect=user_affect,
        user_intent=user_intent,
        turn_id=turn_id,
        extra_reason_codes=extra_reason_codes,
    )
    performance_plan_v3 = compile_performance_plan_v3(
        profile=safe_profile,
        modality=safe_modality,
        language=safe_language,
        tts_label=safe_tts,
        protected_playback=protected_playback,
        camera_state=safe_camera,
        continuous_perception_enabled=continuous_perception_enabled,
        current_turn_state=safe_turn,
        behavior_profile=behavior_profile,
        memory_use_trace=memory_use_trace,
        memory_continuity_trace=memory_continuity_trace,
        suppressed_memory_count=suppressed_memory_count,
        active_listening=active_listening,
        camera_scene=camera_scene,
        floor_state=floor_state,
        user_affect=user_affect,
        user_intent=user_intent,
        performance_plan_v2=performance_plan_v2,
        actor_control_frame=actor_control_frame,
        voice_capabilities=voice_capabilities,
        voice_actuation_plan=voice_actuation_plan,
        turn_id=turn_id,
        extra_reason_codes=extra_reason_codes,
    )
    reason_codes = _dedupe(
        (
            "memory_persona_performance:v1",
            "memory_persona_performance:available",
            f"profile:{safe_profile}",
            f"modality:{safe_modality}",
            f"memory_policy:{memory_policy}",
            f"memory_selected:{len(selected_memories)}",
            f"memory_suppressed:{max(_safe_int(suppressed_memory_count), _safe_int(continuity_payload.get('suppressed_memory_count')))}",
            f"memory_continuity:{continuity_effect or 'none'}",
            "protected_playback:on" if protected_playback else "protected_playback:off",
            *(
                memory_use_trace.reason_codes
                if memory_use_trace is not None
                else ("memory_use_trace_empty",)
            ),
            *extra_reason_codes,
        )
    )
    return BrainMemoryPersonaPerformancePlan(
        schema_version=_SCHEMA_VERSION,
        available=True,
        profile=safe_profile,
        modality=safe_modality,
        language=safe_language,
        tts_label=safe_tts,
        protected_playback=protected_playback,
        camera_state=safe_camera,
        continuous_perception_enabled=continuous_perception_enabled,
        current_turn_state=safe_turn,
        memory_policy=memory_policy,
        selected_memory_count=len(selected_memories),
        suppressed_memory_count=max(
            _safe_int(suppressed_memory_count),
            _safe_int(continuity_payload.get("suppressed_memory_count")),
        ),
        selected_memories=selected_memories,
        behavior_effects=behavior_effects,
        persona_references=persona_references,
        summary=summary,
        performance_plan_v2=performance_plan_v2,
        performance_plan_v3=performance_plan_v3,
        reason_codes=reason_codes,
    )


def unavailable_memory_persona_performance_plan(
    *reason_codes: Any,
    profile: str = "manual",
) -> BrainMemoryPersonaPerformancePlan:
    """Return a stable unavailable memory/persona plan."""
    performance_plan_v2 = compile_performance_plan_v2(
        profile=profile,
        modality="browser",
        language="unknown",
        tts_label="unknown",
        protected_playback=True,
        camera_state="unknown",
        current_turn_state="unknown",
        user_intent="unknown",
        extra_reason_codes=(
            "memory_persona_performance:unavailable",
            *reason_codes,
        ),
    )
    performance_plan_v3 = compile_performance_plan_v3(
        profile=profile,
        modality="browser",
        language="unknown",
        tts_label="unknown",
        protected_playback=True,
        camera_state="unknown",
        current_turn_state="unknown",
        performance_plan_v2=performance_plan_v2,
        user_intent="unknown",
        extra_reason_codes=(
            "memory_persona_performance:unavailable",
            *reason_codes,
        ),
    )
    return BrainMemoryPersonaPerformancePlan(
        schema_version=_SCHEMA_VERSION,
        available=False,
        profile=_safe_text(profile, limit=96) or "manual",
        modality="browser",
        language="unknown",
        tts_label="unknown",
        protected_playback=True,
        camera_state="unknown",
        continuous_perception_enabled=False,
        current_turn_state="unknown",
        memory_policy="unavailable",
        selected_memory_count=0,
        suppressed_memory_count=0,
        selected_memories=(),
        behavior_effects=(),
        persona_references=_compile_persona_references(
            behavior_effects=(),
            camera_state="unknown",
            current_turn_state="unknown",
            selected_memory_count=0,
        ),
        summary="Memory/persona performance unavailable.",
        performance_plan_v2=performance_plan_v2,
        performance_plan_v3=performance_plan_v3,
        reason_codes=_dedupe(
            (
                "memory_persona_performance:v1",
                "memory_persona_performance:unavailable",
                *reason_codes,
            )
        ),
    )


__all__ = [
    "BrainMemoryPersonaPerformancePlan",
    "BrainMemoryPersonaPerformanceRef",
    "BrainPersonaPerformanceReference",
    "PerformancePlanV2",
    "PerformancePlanV2ReferenceSummary",
    "PerformancePlanV3",
    "PerformancePlanV3PersonaAnchorSummary",
    "compile_memory_persona_performance_plan",
    "compile_performance_plan_v2",
    "compile_performance_plan_v3",
    "compile_performance_plan_v3_from_actor_control",
    "unavailable_memory_persona_performance_plan",
]

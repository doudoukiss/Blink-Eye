"""Public-safe aggregate operator workbench snapshot for Blink runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

from blink.brain.evals.adapter_promotion import build_adapter_governance_inspection
from blink.brain.evals.episode_evidence_index import build_episode_evidence_index
from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    build_performance_learning_inspection,
)
from blink.brain.evals.sim_to_real_report import build_sim_to_real_digest
from blink.brain.memory_v2 import build_memory_palace_snapshot
from blink.brain.persona import (
    persona_reference_bank_v3,
    render_behavior_control_effect_summary,
)
from blink.brain.practice_director import build_practice_inspection

_SCHEMA_VERSION = 1
_SECTION_KEYS = (
    "expression",
    "behavior_controls",
    "teaching_knowledge",
    "voice_metrics",
    "memory",
    "practice",
    "adapters",
    "sim_to_real",
    "rollout_status",
    "episode_evidence",
    "performance_learning",
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


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


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_bool(value: Any) -> bool:
    return value is True


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_enum(value: Any, allowed: set[str], default: str) -> str:
    normalized = _normalized_text(value)
    return normalized if normalized in allowed else default


def _safe_timestamp(value: Any) -> str | None:
    text = _normalized_text(value)
    if not text or len(text) > 80:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _public_reason_fragment(value: Any) -> str:
    raw = str(value or "")
    lowered = raw.lower()
    if any(
        marker in lowered
        for marker in (
            "secret",
            "authorization",
            "api_key",
            "bearer",
            "credential",
            "developer_message",
            "developer_prompt",
            "device_label",
            "system_prompt",
            "raw_prompt",
            "prompt_text",
            "transcript",
            "audio_bytes",
            "image_bytes",
            "raw_audio",
            "raw_image",
            "sdp_offer",
            "ice_candidate",
            "traceback",
            "runtimeerror",
            "raw_json",
            "/tmp",
            ".db",
        )
    ):
        return "redacted"
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in raw)
    return "_".join(part for part in text.split("_") if part)[:80] or "unknown"


def _public_safe_text(value: Any, *, limit: int = 120) -> str:
    text = _normalized_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "secret",
            "authorization",
            "api_key",
            "bearer",
            "system_prompt",
            "prompt_text",
            "traceback",
            "runtimeerror",
            "raw_json",
            "/tmp",
            ".db",
        )
    ):
        return "redacted"
    return text[:limit]


def _public_text_list(value: Any, *, limit: int = 96, max_items: int = 16) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in list(value)[:max_items]:
        text = _public_safe_text(item, limit=limit)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _public_count_mapping(value: Any, *, key_limit: int = 80) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = _public_safe_text(raw_key, limit=key_limit) or "unknown"
        count = _safe_int(raw_count)
        result[key] = result.get(key, 0) + count
    return dict(sorted(result.items()))


def _public_text_mapping(
    value: Any,
    *,
    key_limit: int = 80,
    value_limit: int = 80,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _public_safe_text(raw_key, limit=key_limit) or "unknown"
        public_value = _public_safe_text(raw_value, limit=value_limit)
        if public_value:
            result[key] = public_value
    return dict(sorted(result.items()))


def _public_memory_actions(value: Any) -> list[str]:
    allowed = {
        "review",
        "pin",
        "suppress",
        "correct",
        "forget",
        "export",
        "mark_stale",
        "mark-stale",
        "mark_done",
        "cancel",
    }
    return [item for item in _public_text_list(value, limit=32) if item in allowed]


def _public_reason_code_list(value: Any, fallback: tuple[str, ...] = ()) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return list(fallback)
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        code = _public_reason_fragment(item)
        if code in seen:
            continue
        seen.add(code)
        result.append(code)
    return result or list(fallback)


def _as_dict(value: Any) -> dict[str, Any]:
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        serialized = serializer()
        return dict(serialized) if isinstance(serialized, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _public_reason_codes(payload: dict[str, Any], *extra: str) -> tuple[str, ...]:
    return _dedupe((*extra, *_public_reason_code_list(payload.get("reason_codes"))))


@dataclass(frozen=True)
class BrainOperatorWorkbenchSection:
    """One public-safe operator workbench section."""

    available: bool
    summary: str
    payload: dict[str, Any]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the section in stable key order."""
        return {
            "available": self.available,
            "summary": self.summary,
            "payload": dict(self.payload),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrainOperatorWorkbenchSnapshot:
    """Aggregate public-safe operator workbench snapshot."""

    schema_version: int
    available: bool
    expression: BrainOperatorWorkbenchSection
    behavior_controls: BrainOperatorWorkbenchSection
    teaching_knowledge: BrainOperatorWorkbenchSection
    voice_metrics: BrainOperatorWorkbenchSection
    memory: BrainOperatorWorkbenchSection
    practice: BrainOperatorWorkbenchSection
    adapters: BrainOperatorWorkbenchSection
    sim_to_real: BrainOperatorWorkbenchSection
    rollout_status: BrainOperatorWorkbenchSection
    episode_evidence: BrainOperatorWorkbenchSection
    performance_learning: BrainOperatorWorkbenchSection
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the aggregate snapshot in stable key order."""
        return {
            "schema_version": self.schema_version,
            "available": self.available,
            "expression": self.expression.as_dict(),
            "behavior_controls": self.behavior_controls.as_dict(),
            "teaching_knowledge": self.teaching_knowledge.as_dict(),
            "voice_metrics": self.voice_metrics.as_dict(),
            "memory": self.memory.as_dict(),
            "practice": self.practice.as_dict(),
            "adapters": self.adapters.as_dict(),
            "sim_to_real": self.sim_to_real.as_dict(),
            "rollout_status": self.rollout_status.as_dict(),
            "episode_evidence": self.episode_evidence.as_dict(),
            "performance_learning": self.performance_learning.as_dict(),
            "reason_codes": list(self.reason_codes),
        }


def _unavailable_section(section_key: str, *reason_codes: str) -> BrainOperatorWorkbenchSection:
    return BrainOperatorWorkbenchSection(
        available=False,
        summary=f"{section_key.replace('_', ' ').title()} unavailable.",
        payload={},
        reason_codes=_dedupe(
            (
                f"operator_{section_key}:unavailable",
                *reason_codes,
            )
        ),
    )


def _safe_call(
    section_key: str,
    builder: Callable[[], BrainOperatorWorkbenchSection],
) -> BrainOperatorWorkbenchSection:
    try:
        return builder()
    except Exception as exc:  # pragma: no cover - defensive aggregate boundary
        return _unavailable_section(
            section_key, f"operator_{section_key}_error:{type(exc).__name__}"
        )


def _public_voice_policy(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "available": _safe_bool(payload.get("available", False)),
        "modality": _public_safe_text(payload.get("modality"), limit=64) or "unavailable",
        "concise_chunking_active": _safe_bool(payload.get("concise_chunking_active")),
        "chunking_mode": _public_safe_text(payload.get("chunking_mode"), limit=64)
        or "unavailable",
        "max_spoken_chunk_chars": _safe_int(payload.get("max_spoken_chunk_chars")),
        "interruption_strategy_label": _public_safe_text(
            payload.get("interruption_strategy_label"),
            limit=96,
        ),
        "pause_yield_hint": _public_safe_text(payload.get("pause_yield_hint"), limit=96),
        "active_hints": _public_text_list(payload.get("active_hints")),
        "unsupported_hints": _public_text_list(payload.get("unsupported_hints")),
        "noop_reason_codes": _public_reason_code_list(payload.get("noop_reason_codes")),
        "expression_controls_hardware": False,
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _public_voice_actuation_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "available": _safe_bool(payload.get("available", False)),
        "backend_label": _public_safe_text(payload.get("backend_label"), limit=96)
        or "provider-neutral",
        "modality": _public_safe_text(payload.get("modality"), limit=64) or "unavailable",
        "chunk_boundaries_enabled": _safe_bool(payload.get("chunk_boundaries_enabled")),
        "interruption_flush_enabled": _safe_bool(payload.get("interruption_flush_enabled")),
        "interruption_discard_enabled": _safe_bool(payload.get("interruption_discard_enabled")),
        "pause_timing_enabled": _safe_bool(payload.get("pause_timing_enabled")),
        "speech_rate_enabled": _safe_bool(payload.get("speech_rate_enabled")),
        "prosody_emphasis_enabled": _safe_bool(payload.get("prosody_emphasis_enabled")),
        "partial_stream_abort_enabled": _safe_bool(payload.get("partial_stream_abort_enabled")),
        "expression_controls_hardware": False,
        "chunking_mode": _public_safe_text(payload.get("chunking_mode"), limit=64)
        or "unavailable",
        "max_spoken_chunk_chars": _safe_int(payload.get("max_spoken_chunk_chars")),
        "requested_hints": _public_text_list(payload.get("requested_hints")),
        "applied_hints": _public_text_list(payload.get("applied_hints")),
        "active_hints": _public_text_list(payload.get("active_hints")),
        "unsupported_hints": _public_text_list(payload.get("unsupported_hints")),
        "noop_reason_codes": _public_reason_code_list(payload.get("noop_reason_codes")),
        "capability_reason_codes": _public_reason_code_list(
            payload.get("capability_reason_codes")
        ),
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _public_expression_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": _safe_bool(payload.get("available", False)),
        "persona_profile_id": _public_safe_text(payload.get("persona_profile_id"), limit=96),
        "identity_label": _public_safe_text(payload.get("identity_label"), limit=120)
        or "Blink",
        "modality": _public_safe_text(payload.get("modality"), limit=64) or "unavailable",
        "teaching_mode_label": _public_safe_text(
            payload.get("teaching_mode_label"),
            limit=96,
        ),
        "memory_persona_section_status": _public_text_mapping(
            payload.get("memory_persona_section_status")
        ),
        "voice_style_summary": _public_safe_text(payload.get("voice_style_summary"), limit=160),
        "response_chunk_length": _public_safe_text(
            payload.get("response_chunk_length"),
            limit=96,
        ),
        "pause_yield_hint": _public_safe_text(payload.get("pause_yield_hint"), limit=96),
        "interruption_strategy_label": _public_safe_text(
            payload.get("interruption_strategy_label"),
            limit=96,
        ),
        "initiative_label": _public_safe_text(payload.get("initiative_label"), limit=64)
        or "unavailable",
        "evidence_visibility_label": _public_safe_text(
            payload.get("evidence_visibility_label"),
            limit=64,
        )
        or "unavailable",
        "correction_mode_label": _public_safe_text(
            payload.get("correction_mode_label"),
            limit=64,
        )
        or "unavailable",
        "explanation_structure_label": _public_safe_text(
            payload.get("explanation_structure_label"),
            limit=64,
        )
        or "unavailable",
        "humor_mode_label": _public_safe_text(payload.get("humor_mode_label"), limit=64)
        or "unavailable",
        "vividness_mode_label": _public_safe_text(payload.get("vividness_mode_label"), limit=64)
        or "unavailable",
        "sophistication_mode_label": _public_safe_text(
            payload.get("sophistication_mode_label"),
            limit=64,
        )
        or "unavailable",
        "character_presence_label": _public_safe_text(
            payload.get("character_presence_label"),
            limit=64,
        )
        or "unavailable",
        "story_mode_label": _public_safe_text(payload.get("story_mode_label"), limit=64)
        or "unavailable",
        "style_summary": _public_safe_text(payload.get("style_summary"), limit=180)
        or "unavailable",
        "humor_budget": _safe_float(payload.get("humor_budget")),
        "playfulness": _safe_float(payload.get("playfulness")),
        "metaphor_density": _safe_float(payload.get("metaphor_density")),
        "safety_clamped": _safe_bool(payload.get("safety_clamped")),
        "expression_controls_hardware": False,
        "voice_policy": _public_voice_policy(_mapping(payload.get("voice_policy"))),
        "voice_actuation_plan": _public_voice_actuation_plan(
            _mapping(payload.get("voice_actuation_plan"))
        ),
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _build_expression_section(runtime: object) -> BrainOperatorWorkbenchSection:
    reader = getattr(runtime, "current_expression_state", None)
    if not callable(reader):
        return _unavailable_section("expression", "runtime_expression_surface_missing")
    payload = _public_expression_payload(_as_dict(reader()))
    if not payload["available"]:
        return BrainOperatorWorkbenchSection(
            available=False,
            summary="Expression unavailable.",
            payload=payload,
            reason_codes=_public_reason_codes(
                payload,
                "operator_expression:unavailable",
            ),
        )
    summary = " / ".join(
        value
        for value in (
            payload.get("identity_label"),
            payload.get("teaching_mode_label"),
            payload.get("modality"),
        )
        if _normalized_text(value)
    )
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=summary or "Expression available.",
        payload=payload,
        reason_codes=_public_reason_codes(payload, "operator_expression:available"),
    )


_BEHAVIOR_PROFILE_KEYS = (
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
)


def _public_behavior_payload(profile: Any) -> dict[str, Any]:
    profile_payload = _as_dict(profile)
    profile_payload = {
        key: profile_payload.get(key) for key in _BEHAVIOR_PROFILE_KEYS if key in profile_payload
    }
    if profile_payload:
        profile_payload = {
            key: (
                _safe_int(value, _SCHEMA_VERSION)
                if key == "schema_version"
                else
                _public_reason_code_list(value)
                if key == "reason_codes"
                else _public_safe_text(value, limit=120)
            )
            for key, value in profile_payload.items()
        }
    effect_summary = ""
    if profile_payload:
        try:
            effect_summary = _public_safe_text(
                render_behavior_control_effect_summary(profile),
                limit=320,
            )
        except AttributeError:
            effect_summary = _public_safe_text(
                (
                    f"depth={profile_payload.get('response_depth', '')}; "
                    f"directness={profile_payload.get('directness', '')}; "
                    f"warmth={profile_payload.get('warmth', '')}; "
                    f"teaching={profile_payload.get('teaching_mode', '')}; "
                    f"memory={profile_payload.get('memory_use', '')}; "
                    f"initiative={profile_payload.get('initiative_mode', '')}; "
                    f"evidence={profile_payload.get('evidence_visibility', '')}; "
                    f"correction={profile_payload.get('correction_mode', '')}; "
                    f"structure={profile_payload.get('explanation_structure', '')}; "
                    f"challenge={profile_payload.get('challenge_style', '')}; "
                    f"voice={profile_payload.get('voice_mode', '')}; "
                    f"questions={profile_payload.get('question_budget', '')}; "
                    f"humor={profile_payload.get('humor_mode', '')}; "
                    f"vividness={profile_payload.get('vividness_mode', '')}; "
                    f"sophistication={profile_payload.get('sophistication_mode', '')}; "
                    f"character={profile_payload.get('character_presence', '')}; "
                    f"story={profile_payload.get('story_mode', '')}"
                ),
                limit=320,
            )
    return {
        "schema_version": 1,
        "available": bool(profile_payload),
        "profile": profile_payload or None,
        "compiled_effect_summary": effect_summary,
        "reason_codes": (
            _public_reason_code_list(profile_payload.get("reason_codes"))
            if profile_payload
            else []
        ),
    }


def _build_behavior_controls_section(runtime: object) -> BrainOperatorWorkbenchSection:
    reader = getattr(runtime, "current_behavior_control_profile", None)
    if not callable(reader):
        return _unavailable_section(
            "behavior_controls",
            "runtime_behavior_controls_surface_missing",
        )
    payload = _public_behavior_payload(reader())
    if not payload["available"]:
        return _unavailable_section(
            "behavior_controls",
            "runtime_behavior_controls_unavailable",
        )
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=payload["compiled_effect_summary"] or "Behavior controls available.",
        payload=payload,
        reason_codes=_public_reason_codes(
            payload,
            "operator_behavior_controls:available",
        ),
    )


def _public_teaching_knowledge_item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_kind": _public_safe_text(record.get("item_kind"), limit=64) or "unknown",
        "item_id": _public_safe_text(record.get("item_id"), limit=120),
        "title": _public_safe_text(record.get("title"), limit=120),
        "source_label": _public_safe_text(record.get("source_label"), limit=120) or "curated",
        "provenance_kind": _public_safe_text(record.get("provenance_kind"), limit=80),
        "provenance_version": _public_safe_text(record.get("provenance_version"), limit=80),
    }


def _public_teaching_knowledge_decision(record: dict[str, Any]) -> dict[str, Any]:
    selected_items = [
        _public_teaching_knowledge_item(item)
        for item in _list(record.get("selected_items"))
        if isinstance(item, dict)
    ][:6]
    return {
        "schema_version": _safe_int(record.get("schema_version"), _SCHEMA_VERSION),
        "available": _safe_bool(record.get("available", False)),
        "selection_kind": _public_safe_text(record.get("selection_kind"), limit=64)
        or "unavailable",
        "task_mode": _public_safe_text(record.get("task_mode"), limit=64) or "unavailable",
        "language": _public_safe_text(record.get("language"), limit=32) or "unavailable",
        "teaching_mode": _public_safe_text(record.get("teaching_mode"), limit=64)
        or "unavailable",
        "summary": _public_safe_text(record.get("summary"), limit=180)
        or "Teaching knowledge routing unavailable.",
        "selected_items": selected_items,
        "estimated_tokens": _safe_int(record.get("estimated_tokens")),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }


def _teaching_item_counts(decisions: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        for item in _list(decision.get("selected_items")):
            if not isinstance(item, dict):
                continue
            item_kind = _normalized_text(item.get("item_kind")) or "unknown"
            counts[item_kind] = counts.get(item_kind, 0) + 1
    return dict(sorted(counts.items()))


def _build_teaching_knowledge_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    reader = getattr(runtime, "current_teaching_knowledge_routing", None)
    recent_reader = getattr(runtime, "recent_teaching_knowledge_routing", None)
    if not callable(reader):
        return _unavailable_section(
            "teaching_knowledge",
            "runtime_teaching_knowledge_surface_missing",
        )
    current = _public_teaching_knowledge_decision(_as_dict(reader()))
    recent = []
    if callable(recent_reader):
        recent = [
            _public_teaching_knowledge_decision(_as_dict(record))
            for record in recent_reader(limit=recent_limit)
        ][:recent_limit]
    if current["available"] and not recent:
        recent = [current]
    available = current["available"] or any(record["available"] for record in recent)
    selected_item_counts = _teaching_item_counts([current, *recent])
    summary = (
        current["summary"]
        if current["available"]
        else f"{len(recent)} recent teaching knowledge decisions."
        if recent
        else "Teaching knowledge routing unavailable."
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "available": available,
        "summary": summary,
        "current_decision": current,
        "recent_decisions": recent,
        "selected_item_counts": selected_item_counts,
        "reason_codes": _dedupe(
            (
                "teaching_knowledge_routing:available"
                if available
                else "teaching_knowledge_routing:unavailable",
                *current.get("reason_codes", []),
                *(code for record in recent for code in record.get("reason_codes", [])),
            )
        ),
    }
    return BrainOperatorWorkbenchSection(
        available=available,
        summary=summary,
        payload=payload,
        reason_codes=_public_reason_codes(
            payload,
            "operator_teaching_knowledge:available"
            if available
            else "operator_teaching_knowledge:unavailable",
        ),
    )


def _public_voice_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload)
    return {
        "available": _safe_bool(payload.get("available", False)),
        "response_count": _safe_int(payload.get("response_count")),
        "concise_chunking_activation_count": _safe_int(
            payload.get("concise_chunking_activation_count")
        ),
        "chunk_count": _safe_int(payload.get("chunk_count")),
        "max_chunk_chars": _safe_int(payload.get("max_chunk_chars")),
        "average_chunk_chars": _safe_float(payload.get("average_chunk_chars")),
        "interruption_frame_count": _safe_int(payload.get("interruption_frame_count")),
        "buffer_flush_count": _safe_int(payload.get("buffer_flush_count")),
        "buffer_discard_count": _safe_int(payload.get("buffer_discard_count")),
        "last_chunking_mode": _safe_enum(
            payload.get("last_chunking_mode"),
            {"unavailable", "none", "off", "concise", "safety_concise"},
            "unavailable",
        ),
        "last_max_spoken_chunk_chars": _safe_int(payload.get("last_max_spoken_chunk_chars")),
        "first_audio_latency_ms": _safe_float(payload.get("first_audio_latency_ms")),
        "first_audio_latency_sample_count": _safe_int(
            payload.get("first_audio_latency_sample_count")
        ),
        "resumed_latency_after_interrupt_ms": _safe_float(
            payload.get("resumed_latency_after_interrupt_ms")
        ),
        "resumed_latency_sample_count": _safe_int(payload.get("resumed_latency_sample_count")),
        "interruption_accept_count": _safe_int(payload.get("interruption_accept_count")),
        "partial_stream_abort_count": _safe_int(payload.get("partial_stream_abort_count")),
        "average_chunks_per_response": _safe_float(payload.get("average_chunks_per_response")),
        "p50_chunk_chars": _safe_int(payload.get("p50_chunk_chars")),
        "p95_chunk_chars": _safe_int(payload.get("p95_chunk_chars")),
        "expression_controls_hardware": False,
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
        "input_health": _public_voice_input_health_payload(
            _mapping(payload.get("input_health"))
        ),
    }


def _public_voice_input_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _mapping(payload)
    return {
        "schema_version": 1,
        "available": _safe_bool(payload.get("available", False)),
        "microphone_state": _safe_enum(
            payload.get("microphone_state"),
            {
                "unavailable",
                "disconnected",
                "connected",
                "waiting_for_audio",
                "receiving",
                "no_audio_frames",
                "stalled",
            },
            "unavailable",
        ),
        "stt_state": _safe_enum(
            payload.get("stt_state"),
            {
                "unavailable",
                "idle",
                "speech_detected",
                "waiting",
                "transcribed",
                "error",
            },
            "unavailable",
        ),
        "audio_frame_count": _safe_int(payload.get("audio_frame_count")),
        "speech_start_count": _safe_int(payload.get("speech_start_count")),
        "speech_stop_count": _safe_int(payload.get("speech_stop_count")),
        "transcription_count": _safe_int(payload.get("transcription_count")),
        "stt_error_count": _safe_int(payload.get("stt_error_count")),
        "last_audio_frame_at": _safe_timestamp(payload.get("last_audio_frame_at")),
        "last_audio_frame_age_ms": _safe_optional_int(
            payload.get("last_audio_frame_age_ms")
        ),
        "last_stt_event_at": _safe_timestamp(payload.get("last_stt_event_at")),
        "stt_waiting_since_at": _safe_timestamp(payload.get("stt_waiting_since_at")),
        "stt_wait_age_ms": _safe_optional_int(payload.get("stt_wait_age_ms")),
        "stt_waiting_too_long": _safe_bool(payload.get("stt_waiting_too_long")),
        "last_transcription_at": _safe_timestamp(payload.get("last_transcription_at")),
        "last_transcription_chars": _safe_int(payload.get("last_transcription_chars")),
        "track_enabled": (
            _safe_bool(payload.get("track_enabled"))
            if payload.get("track_enabled") is not None
            else None
        ),
        "track_reason": (
            _public_reason_fragment(payload.get("track_reason"))
            if payload.get("track_reason")
            else None
        ),
        "reason_codes": _public_reason_code_list(
            payload.get("reason_codes"),
            ("voice_input_health:v1", "voice_input:unavailable"),
        ),
    }


def _build_voice_metrics_section(runtime: object) -> BrainOperatorWorkbenchSection:
    reader = getattr(runtime, "current_voice_metrics", None)
    if not callable(reader):
        return _unavailable_section(
            "voice_metrics",
            "runtime_voice_metrics_surface_missing",
        )
    payload = _public_voice_metrics_payload(_as_dict(reader()))
    input_health_reader = getattr(runtime, "current_voice_input_health", None)
    if callable(input_health_reader):
        try:
            payload["input_health"] = _public_voice_input_health_payload(
                _as_dict(input_health_reader())
            )
        except Exception as exc:  # pragma: no cover - defensive status surface
            payload["input_health"] = _public_voice_input_health_payload(
                {
                    "available": False,
                    "reason_codes": [
                        "voice_input_health:v1",
                        f"voice_input_health_error:{type(exc).__name__}",
                    ],
                }
            )
    if not payload["available"]:
        return BrainOperatorWorkbenchSection(
            available=False,
            summary="Voice metrics unavailable.",
            payload=payload,
            reason_codes=_public_reason_codes(
                payload,
                "operator_voice_metrics:unavailable",
            ),
        )
    summary = (
        f"mic {payload['input_health']['microphone_state']}; "
        f"STT {payload['input_health']['stt_state']}; "
        f"{payload['chunk_count']} chunks; {payload['interruption_frame_count']} interruptions"
    )
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=summary,
        payload=payload,
        reason_codes=_public_reason_codes(payload, "operator_voice_metrics:available"),
    )


def _public_memory_provenance_label(value: Any) -> str | None:
    label = _normalized_text(value)
    allowed = {
        "Remembered from your profile memory.",
        "Remembered from your explicit preference.",
        "Task you asked Blink to track.",
        "Part of your relationship-style settings.",
        "Part of your teaching-profile settings.",
        "Derived from a prior conversation and not recently confirmed.",
        "Derived from prior conversation memory.",
    }
    return label if label in allowed else None


def _public_memory_record(record: dict[str, Any]) -> dict[str, Any]:
    public_record = {
        "memory_id": _public_safe_text(record.get("memory_id"), limit=160),
        "display_kind": _public_safe_text(record.get("display_kind"), limit=64),
        "title": _public_safe_text(record.get("title"), limit=120),
        "summary": _public_safe_text(record.get("summary"), limit=240),
        "status": _public_safe_text(record.get("status"), limit=64),
        "currentness_status": _public_safe_text(record.get("currentness_status"), limit=64),
        "confidence": _safe_optional_float(record.get("confidence")),
        "pinned": record.get("pinned") is True,
        "last_used_at": _public_safe_text(record.get("last_used_at"), limit=96),
        "last_used_reason": _public_safe_text(record.get("last_used_reason"), limit=120),
        "used_in_current_turn": record.get("used_in_current_turn") is True,
        "safe_provenance_label": _public_memory_provenance_label(
            record.get("safe_provenance_label")
        ),
        "user_actions": _public_memory_actions(record.get("user_actions")),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }
    return {key: value for key, value in public_record.items() if value is not None and value != ""}


def _public_memory_persona_reply_ref(record: Any) -> dict[str, Any]:
    payload = _mapping(record)
    return {
        "memory_id": _public_safe_text(payload.get("memory_id"), limit=160),
        "display_kind": _public_safe_text(payload.get("display_kind"), limit=64),
        "title": _public_safe_text(payload.get("title"), limit=120) or "Memory",
        "used_reason": _public_safe_text(payload.get("used_reason"), limit=96),
        "behavior_effect": _public_safe_text(payload.get("behavior_effect"), limit=120),
        "effect_labels": _public_text_list(payload.get("effect_labels"), limit=96, max_items=8),
        "linked_discourse_episode_ids": _public_text_list(
            payload.get("linked_discourse_episode_ids"),
            limit=120,
            max_items=8,
        ),
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _public_memory_continuity_trace_payload(payload: Any) -> dict[str, Any] | None:
    data = _as_dict(payload)
    if not data:
        return None
    selected_memories: list[dict[str, Any]] = []
    for record in _list(data.get("selected_memories"))[:8]:
        if not isinstance(record, dict):
            continue
        selected_memories.append(
            {
                "memory_id": _public_safe_text(record.get("memory_id"), limit=160),
                "display_kind": _public_safe_text(record.get("display_kind"), limit=64),
                "summary": _public_safe_text(record.get("summary"), limit=160),
                "safe_provenance_label": _public_memory_provenance_label(
                    record.get("safe_provenance_label")
                ),
                "source_language": _public_safe_text(record.get("source_language"), limit=32),
                "cross_language": record.get("cross_language") is True,
                "effect_labels": _public_text_list(
                    record.get("effect_labels"),
                    limit=96,
                    max_items=8,
                ),
                "linked_discourse_episode_ids": _public_text_list(
                    record.get("linked_discourse_episode_ids"),
                    limit=120,
                    max_items=8,
                ),
                "conflict_labels": _public_text_list(
                    record.get("conflict_labels"),
                    limit=96,
                    max_items=8,
                ),
                "staleness_labels": _public_text_list(
                    record.get("staleness_labels"),
                    limit=96,
                    max_items=8,
                ),
                "reason_codes": _public_reason_code_list(record.get("reason_codes")),
            }
        )
    continuity_v3 = _mapping(data.get("memory_continuity_v3"))
    discourse_refs: list[dict[str, Any]] = []
    for record in _list(continuity_v3.get("selected_discourse_episodes"))[:8]:
        if not isinstance(record, dict):
            continue
        discourse_refs.append(
            {
                "discourse_episode_id": _public_safe_text(
                    record.get("discourse_episode_id"),
                    limit=120,
                ),
                "category_labels": _public_text_list(
                    record.get("category_labels"),
                    limit=96,
                    max_items=8,
                ),
                "effect_labels": _public_text_list(
                    record.get("effect_labels"),
                    limit=96,
                    max_items=8,
                ),
                "memory_ids": _public_text_list(record.get("memory_ids"), limit=120, max_items=8),
                "confidence_bucket": _public_safe_text(
                    record.get("confidence_bucket"),
                    limit=32,
                ),
                "reason_codes": _public_reason_code_list(record.get("reason_codes")),
            }
        )
    return {
        "schema_version": _safe_int(data.get("schema_version"), 1),
        "profile": _public_safe_text(data.get("profile"), limit=96) or "manual",
        "language": _public_safe_text(data.get("language"), limit=32) or "unknown",
        "memory_effect": _public_safe_text(data.get("memory_effect"), limit=96) or "none",
        "cross_language_count": _safe_int(data.get("cross_language_count")),
        "selected_memory_count": _safe_int(data.get("selected_memory_count")),
        "suppressed_memory_count": _safe_int(data.get("suppressed_memory_count")),
        "selected_memories": selected_memories,
        "memory_continuity_v3": {
            "schema_version": _safe_int(continuity_v3.get("schema_version"), 3),
            "selected_discourse_episodes": discourse_refs,
            "effect_labels": _public_text_list(
                continuity_v3.get("effect_labels"),
                limit=96,
                max_items=8,
            ),
            "conflict_labels": _public_text_list(
                continuity_v3.get("conflict_labels"),
                limit=96,
                max_items=8,
            ),
            "staleness_labels": _public_text_list(
                continuity_v3.get("staleness_labels"),
                limit=96,
                max_items=8,
            ),
            "cross_language_transfer_count": _safe_int(
                continuity_v3.get("cross_language_transfer_count")
            ),
            "reason_codes": _public_reason_code_list(continuity_v3.get("reason_codes")),
        },
        "reason_codes": _public_reason_code_list(data.get("reason_codes")),
    }


def _public_persona_reference_payload(record: Any) -> dict[str, Any]:
    payload = _mapping(record)
    return {
        "reference_id": _public_safe_text(payload.get("reference_id"), limit=96),
        "mode": _public_safe_text(payload.get("mode"), limit=64),
        "label": _public_safe_text(payload.get("label"), limit=96),
        "applies": payload.get("applies") is True,
        "behavior_effect": _public_safe_text(payload.get("behavior_effect"), limit=120),
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _public_persona_anchor_v3_payload(record: Any, *, include_examples: bool) -> dict[str, Any]:
    payload = _mapping(record)
    behavior_constraints = _public_text_list(
        payload.get("behavior_constraints"),
        limit=160,
        max_items=8,
    )
    negative_examples = _public_text_list(
        payload.get("negative_examples"),
        limit=160,
        max_items=8,
    )
    public = {
        "schema_version": _safe_int(payload.get("schema_version"), 3),
        "anchor_id": _public_safe_text(payload.get("anchor_id"), limit=96),
        "situation_key": _public_safe_text(payload.get("situation_key"), limit=80),
        "stance_label": _public_safe_text(payload.get("stance_label"), limit=96),
        "response_shape_label": _public_safe_text(
            payload.get("response_shape_label"),
            limit=96,
        ),
        "behavior_constraints": behavior_constraints,
        "negative_examples": negative_examples,
        "behavior_constraint_count": _safe_int(
            payload.get("behavior_constraint_count"),
            len(behavior_constraints),
        ),
        "negative_example_count": _safe_int(
            payload.get("negative_example_count"),
            len(negative_examples),
        ),
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }
    if include_examples:
        public["zh_example"] = _public_safe_text(payload.get("zh_example"), limit=280)
        public["en_example"] = _public_safe_text(payload.get("en_example"), limit=280)
    return {key: value for key, value in public.items() if value not in ("", [], {})}


def _public_persona_anchor_bank_v3_payload() -> dict[str, Any]:
    bank = persona_reference_bank_v3().as_dict()
    return {
        "schema_version": _safe_int(bank.get("schema_version"), 3),
        "anchor_count": _safe_int(bank.get("anchor_count")),
        "required_situation_keys": _public_text_list(
            bank.get("required_situation_keys"),
            limit=80,
            max_items=12,
        ),
        "anchors": [
            _public_persona_anchor_v3_payload(anchor, include_examples=True)
            for anchor in _list(bank.get("anchors"))
            if isinstance(anchor, dict)
        ][:12],
        "reason_codes": _public_reason_code_list(bank.get("reason_codes")),
    }


def _public_memory_persona_performance_payload(payload: Any) -> dict[str, Any]:
    data = _as_dict(payload)
    used_refs = [
        _public_memory_persona_reply_ref(record)
        for record in _list(data.get("used_in_current_reply"))
        if isinstance(record, dict)
    ][:8]
    persona_refs = [
        _public_persona_reference_payload(record)
        for record in _list(data.get("persona_references"))
        if isinstance(record, dict)
    ][:12]
    performance_plan_v3 = _mapping(data.get("performance_plan_v3"))
    persona_anchor_refs_v3 = [
        _public_persona_anchor_v3_payload(record, include_examples=False)
        for record in _list(performance_plan_v3.get("persona_anchor_refs_v3"))
        if isinstance(record, dict)
    ][:8]
    memory_continuity_trace = _public_memory_continuity_trace_payload(
        data.get("memory_continuity_trace")
    )
    return {
        "schema_version": _safe_int(data.get("schema_version"), _SCHEMA_VERSION),
        "available": _safe_bool(data.get("available", False)),
        "profile": _public_safe_text(data.get("profile"), limit=96) or "manual",
        "modality": _public_safe_text(data.get("modality"), limit=64) or "browser",
        "language": _public_safe_text(data.get("language"), limit=32) or "unknown",
        "tts_label": _public_safe_text(data.get("tts_label"), limit=96) or "unknown",
        "protected_playback": data.get("protected_playback") is not False,
        "camera_state": _public_safe_text(data.get("camera_state"), limit=64) or "unknown",
        "continuous_perception_enabled": _safe_bool(data.get("continuous_perception_enabled")),
        "current_turn_state": _public_safe_text(data.get("current_turn_state"), limit=64)
        or "unknown",
        "memory_policy": _public_safe_text(data.get("memory_policy"), limit=64) or "unavailable",
        "selected_memory_count": _safe_int(data.get("selected_memory_count")),
        "suppressed_memory_count": _safe_int(data.get("suppressed_memory_count")),
        "used_in_current_reply": used_refs,
        "behavior_effects": _public_text_list(data.get("behavior_effects"), limit=96),
        "persona_references": persona_refs,
        "persona_anchor_refs_v3": persona_anchor_refs_v3,
        "persona_anchor_bank_v3": _public_persona_anchor_bank_v3_payload(),
        "memory_continuity_trace": memory_continuity_trace,
        "summary": _public_safe_text(data.get("summary"), limit=180)
        or "Memory/persona performance unavailable.",
        "reason_codes": _public_reason_code_list(data.get("reason_codes")),
    }


def _memory_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No visible memories."
    kind_counts: dict[str, int] = {}
    for record in records:
        display_kind = _normalized_text(record.get("display_kind")) or "memory"
        kind_counts[display_kind.replace("_", " ")] = (
            kind_counts.get(display_kind.replace("_", " "), 0) + 1
        )
    kind_summary = ", ".join(
        f"{count} {display_kind}" for display_kind, count in sorted(kind_counts.items())
    )
    return f"{len(records)} visible memories" + (f": {kind_summary}" if kind_summary else ".")


def _public_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    records = [
        _public_memory_record(record)
        for record in _list(payload.get("records"))
        if isinstance(record, dict)
    ]
    memory_persona = _public_memory_persona_performance_payload(
        payload.get("memory_persona_performance") or {}
    )
    memory_continuity_trace = _public_memory_continuity_trace_payload(
        payload.get("memory_continuity_trace")
    ) or memory_persona.get("memory_continuity_trace")
    return {
        "schema_version": _safe_int(payload.get("schema_version"), _SCHEMA_VERSION),
        "user_id": _public_safe_text(payload.get("user_id"), limit=96),
        "agent_id": _public_safe_text(payload.get("agent_id"), limit=96),
        "generated_at": _public_safe_text(payload.get("generated_at"), limit=96),
        "summary": _memory_summary(records),
        "records": records,
        "hidden_counts": _public_count_mapping(payload.get("hidden_counts")),
        "health_summary": _public_safe_text(payload.get("health_summary"), limit=160)
        or "Memory health unavailable.",
        "used_in_current_reply": memory_persona["used_in_current_reply"],
        "behavior_effects": memory_persona["behavior_effects"],
        "persona_references": memory_persona["persona_references"],
        "persona_anchor_refs_v3": memory_persona["persona_anchor_refs_v3"],
        "persona_anchor_bank_v3": memory_persona["persona_anchor_bank_v3"],
        "memory_continuity_trace": memory_continuity_trace,
        "memory_persona_performance": memory_persona,
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _build_memory_section(
    runtime: object,
    *,
    memory_limit: int,
) -> BrainOperatorWorkbenchSection:
    store = getattr(runtime, "store", None)
    session_resolver = getattr(runtime, "session_resolver", None)
    if store is None or not callable(session_resolver):
        return _unavailable_section("memory", "runtime_memory_surface_missing")
    session_ids = session_resolver()
    current_trace_reader = getattr(runtime, "current_memory_use_trace", None)
    current_continuity_reader = getattr(runtime, "current_memory_continuity_trace", None)
    recent_trace_reader = getattr(runtime, "recent_memory_use_traces", None)
    current_turn_trace = current_trace_reader() if callable(current_trace_reader) else None
    current_continuity_trace = (
        current_continuity_reader() if callable(current_continuity_reader) else None
    )
    recent_use_traces = recent_trace_reader(limit=8) if callable(recent_trace_reader) else None
    snapshot = build_memory_palace_snapshot(
        store=store,
        session_ids=session_ids,
        include_suppressed=False,
        include_historical=False,
        current_turn_trace=current_turn_trace,
        recent_use_traces=recent_use_traces,
        limit=memory_limit,
        claim_scan_limit=max(160, int(memory_limit or 0) * 4),
    )
    snapshot_payload = snapshot.as_dict()
    if current_continuity_trace is not None:
        snapshot_payload["memory_continuity_trace"] = _as_dict(current_continuity_trace)
    plan_reader = getattr(runtime, "current_memory_persona_performance_plan", None)
    if callable(plan_reader):
        snapshot_payload["memory_persona_performance"] = _as_dict(
            plan_reader(
                profile="operator",
                current_turn_state="operator",
                memory_continuity_trace=current_continuity_trace,
                suppressed_memory_count=_safe_int(
                    _mapping(snapshot_payload.get("hidden_counts")).get("suppressed")
                ),
            )
        )
    payload = _public_memory_payload(snapshot_payload)
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=payload["summary"],
        payload=payload,
        reason_codes=_public_reason_codes(payload, "operator_memory:available"),
    )


def _runtime_scope(runtime: object) -> tuple[Any | None, Any | None, Any | None]:
    session_resolver = getattr(runtime, "session_resolver", None)
    if not callable(session_resolver):
        return None, None, None
    session_ids = session_resolver()
    return (
        session_ids,
        getattr(runtime, "store", None),
        _normalized_text(getattr(runtime, "presence_scope_key", "")) or "local:presence",
    )


def _public_practice_payload(payload: dict[str, Any]) -> dict[str, Any]:
    recent_targets = []
    for target in _list(payload.get("recent_targets")):
        if not isinstance(target, dict):
            continue
        recent_targets.append(
            {
                "scenario_id": _public_safe_text(target.get("scenario_id"), limit=120),
                "scenario_family": _public_safe_text(
                    target.get("scenario_family"),
                    limit=96,
                ),
                "selected_profile_id": _public_safe_text(
                    target.get("selected_profile_id"),
                    limit=96,
                ),
                "execution_backend": _public_safe_text(
                    target.get("execution_backend"),
                    limit=96,
                ),
                "score": _safe_float(target.get("score")),
                "reason_codes": _public_reason_code_list(target.get("reason_codes")),
            }
        )
    recent_plans = []
    for plan in _list(payload.get("recent_plans")):
        if not isinstance(plan, dict):
            continue
        recent_plans.append(
            {
                "target_count": _safe_int(plan.get("target_count")),
                "reason_code_counts": _public_count_mapping(plan.get("reason_code_counts")),
                "summary": _public_safe_text(plan.get("summary"), limit=180),
                "updated_at": _public_safe_text(plan.get("updated_at"), limit=96),
            }
        )
    return {
        "scenario_family_counts": _public_count_mapping(
            payload.get("scenario_family_counts")
        ),
        "reason_code_counts": _public_count_mapping(payload.get("reason_code_counts")),
        "recent_targets": recent_targets,
        "recent_plans": recent_plans,
    }


def _build_practice_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    session_ids, store, presence_scope_key = _runtime_scope(runtime)
    if store is None or session_ids is None:
        return _unavailable_section("practice", "runtime_practice_surface_missing")
    projection_builder = getattr(store, "build_practice_director_projection", None)
    if not callable(projection_builder):
        return _unavailable_section("practice", "practice_projection_surface_missing")
    projection = projection_builder(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
        presence_scope_key=presence_scope_key,
    )
    payload = _public_practice_payload(
        build_practice_inspection(projection, recent_limit=recent_limit)
    )
    target_count = len(payload["recent_targets"])
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=f"{target_count} recent practice targets.",
        payload=payload,
        reason_codes=_dedupe(("operator_practice:available",)),
    )


def _public_adapter_card(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_family": _public_safe_text(record.get("adapter_family"), limit=96),
        "backend_id": _public_safe_text(record.get("backend_id"), limit=120),
        "backend_version": _public_safe_text(record.get("backend_version"), limit=120),
        "promotion_state": _public_safe_text(record.get("promotion_state"), limit=64),
    }


def _public_adapter_decision(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_family": _public_safe_text(record.get("adapter_family"), limit=96),
        "backend_id": _public_safe_text(record.get("backend_id"), limit=120),
        "backend_version": _public_safe_text(record.get("backend_version"), limit=120),
        "decision_outcome": _public_safe_text(record.get("decision_outcome"), limit=64),
        "from_state": _public_safe_text(record.get("from_state"), limit=64),
        "to_state": _public_safe_text(record.get("to_state"), limit=64),
        "blocked_reason_codes": _public_reason_code_list(record.get("blocked_reason_codes")),
        "weak_families": _public_text_list(record.get("weak_families"), limit=80),
        "smoke_suite_green": _safe_bool(record.get("smoke_suite_green")),
        "benchmark_passed": _safe_bool(record.get("benchmark_passed")),
        "updated_at": _public_safe_text(record.get("updated_at"), limit=96),
    }


def _public_adapters_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_counts": _public_count_mapping(payload.get("state_counts")),
        "family_counts": _public_count_mapping(payload.get("family_counts")),
        "current_default_cards": [
            _public_adapter_card(record)
            for record in _list(payload.get("current_default_cards"))
            if isinstance(record, dict)
        ],
        "recent_cards": [
            _public_adapter_card(record)
            for record in _list(payload.get("recent_cards"))
            if isinstance(record, dict)
        ],
        "pending_or_blocked_decisions": [
            _public_adapter_decision(record)
            for record in _list(payload.get("pending_or_blocked_decisions"))
            if isinstance(record, dict)
        ],
        "rollback_reason_counts": _public_count_mapping(payload.get("rollback_reason_counts")),
    }


def _build_adapters_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    session_ids, store, _presence_scope_key = _runtime_scope(runtime)
    if store is None or session_ids is None:
        return _unavailable_section("adapters", "runtime_adapter_surface_missing")
    projection_builder = getattr(store, "build_adapter_governance_projection", None)
    if not callable(projection_builder):
        return _unavailable_section("adapters", "adapter_projection_surface_missing")
    projection = projection_builder(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )
    payload = _public_adapters_payload(
        build_adapter_governance_inspection(
            adapter_governance=projection,
            recent_limit=recent_limit,
        )
    )
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=f"{len(payload['recent_cards'])} adapter cards.",
        payload=payload,
        reason_codes=_dedupe(("operator_adapters:available",)),
    )


def _public_sim_to_real_report(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_family": _public_safe_text(record.get("adapter_family"), limit=96),
        "backend_id": _public_safe_text(record.get("backend_id"), limit=120),
        "backend_version": _public_safe_text(record.get("backend_version"), limit=120),
        "promotion_state": _public_safe_text(record.get("promotion_state"), limit=64),
        "benchmark_passed": _safe_bool(record.get("benchmark_passed")),
        "smoke_suite_green": _safe_bool(record.get("smoke_suite_green")),
        "shadow_ready": _safe_bool(record.get("shadow_ready")),
        "canary_ready": _safe_bool(record.get("canary_ready")),
        "default_ready": _safe_bool(record.get("default_ready")),
        "rollback_required": _safe_bool(record.get("rollback_required")),
        "governance_only": True,
        "weak_families": _public_text_list(record.get("weak_families"), limit=80),
        "blocked_reason_codes": _public_reason_code_list(record.get("blocked_reason_codes")),
        "parity_summary": _public_safe_text(record.get("parity_summary"), limit=180),
        "updated_at": _public_safe_text(record.get("updated_at"), limit=96),
    }


def _public_sim_to_real_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_counts": _public_count_mapping(payload.get("readiness_counts")),
        "promotion_state_counts": _public_count_mapping(
            payload.get("promotion_state_counts")
        ),
        "blocked_reason_counts": _public_count_mapping(payload.get("blocked_reason_counts")),
        "readiness_reports": [
            _public_sim_to_real_report(record)
            for record in _list(payload.get("readiness_reports"))
            if isinstance(record, dict)
        ],
    }


def _build_sim_to_real_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    session_ids, store, _presence_scope_key = _runtime_scope(runtime)
    if store is None or session_ids is None:
        return _unavailable_section("sim_to_real", "runtime_sim_to_real_surface_missing")
    projection_builder = getattr(store, "build_adapter_governance_projection", None)
    if not callable(projection_builder):
        return _unavailable_section("sim_to_real", "adapter_projection_surface_missing")
    projection = projection_builder(
        user_id=session_ids.user_id,
        thread_id=session_ids.thread_id,
        agent_id=session_ids.agent_id,
    )
    payload = _public_sim_to_real_payload(
        build_sim_to_real_digest(
            adapter_governance=projection,
            recent_limit=recent_limit,
        )
    )
    rollback_count = _safe_int(payload["readiness_counts"].get("rollback_required"))
    return BrainOperatorWorkbenchSection(
        available=True,
        summary=f"{len(payload['readiness_reports'])} readiness reports; {rollback_count} rollbacks.",
        payload=payload,
        reason_codes=_dedupe(("operator_sim_to_real:available", "sim_to_real:governance_only")),
    )


def _public_rollout_plan_status(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": _public_safe_text(record.get("plan_id"), limit=120),
        "adapter_family": _public_safe_text(record.get("adapter_family"), limit=96),
        "candidate_backend_id": _public_safe_text(
            record.get("candidate_backend_id"),
            limit=120,
        ),
        "candidate_backend_version": _public_safe_text(
            record.get("candidate_backend_version"),
            limit=120,
        ),
        "routing_state": _public_safe_text(record.get("routing_state"), limit=64),
        "promotion_state": _public_safe_text(record.get("promotion_state"), limit=64),
        "traffic_fraction": _safe_float(record.get("traffic_fraction")),
        "scope_key": _public_safe_text(record.get("scope_key"), limit=96),
        "expires_at": _public_safe_text(record.get("expires_at"), limit=96),
        "embodied_live": _safe_bool(record.get("embodied_live")),
        "budget_id": _public_safe_text(record.get("budget_id"), limit=120),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }


def _public_rollout_decision_status(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": _public_safe_text(record.get("plan_id"), limit=120),
        "adapter_family": _public_safe_text(record.get("adapter_family"), limit=96),
        "action": _public_safe_text(record.get("action"), limit=64),
        "accepted": _safe_bool(record.get("accepted")),
        "from_state": _public_safe_text(record.get("from_state"), limit=64),
        "to_state": _public_safe_text(record.get("to_state"), limit=64),
        "traffic_fraction": _safe_float(record.get("traffic_fraction")),
        "regression_count": _safe_int(record.get("regression_count")),
        "updated_at": _public_safe_text(record.get("updated_at"), limit=96),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }


def _public_rollout_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _safe_int(payload.get("schema_version"), _SCHEMA_VERSION),
        "available": _safe_bool(payload.get("available", False)),
        "generated_at": _public_safe_text(payload.get("generated_at"), limit=96),
        "summary": _public_safe_text(payload.get("summary"), limit=180)
        or "Rollout status unavailable.",
        "plan_count": _safe_int(payload.get("plan_count")),
        "active_plan_count": _safe_int(payload.get("active_plan_count")),
        "paused_plan_count": _safe_int(payload.get("paused_plan_count")),
        "rolled_back_plan_count": _safe_int(payload.get("rolled_back_plan_count")),
        "expired_plan_count": _safe_int(payload.get("expired_plan_count")),
        "live_routing_active": _safe_bool(payload.get("live_routing_active")),
        "controlled_rollout_supported": _safe_bool(payload.get("controlled_rollout_supported")),
        "governance_only": _safe_bool(payload.get("governance_only")),
        "state_counts": _public_count_mapping(payload.get("state_counts")),
        "family_counts": _public_count_mapping(payload.get("family_counts")),
        "plan_summaries": [
            _public_rollout_plan_status(record)
            for record in _list(payload.get("plan_summaries"))
            if isinstance(record, dict)
        ],
        "recent_decisions": [
            _public_rollout_decision_status(record)
            for record in _list(payload.get("recent_decisions"))
            if isinstance(record, dict)
        ],
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _public_episode_evidence_artifact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": _public_safe_text(record.get("artifact_id"), limit=120),
        "artifact_kind": _public_safe_text(record.get("artifact_kind"), limit=80),
        "uri_kind": _public_safe_text(record.get("uri_kind"), limit=80),
        "redacted_uri": _safe_bool(record.get("redacted_uri")),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }


def _public_episode_evidence_link(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "link_kind": _public_safe_text(record.get("link_kind"), limit=80),
        "link_id": _public_safe_text(record.get("link_id"), limit=120),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
    }


def _public_episode_evidence_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": _public_safe_text(record.get("evidence_id"), limit=120),
        "episode_id": _public_safe_text(record.get("episode_id"), limit=120),
        "source": _public_safe_text(record.get("source"), limit=80),
        "scenario_id": _public_safe_text(record.get("scenario_id"), limit=120),
        "scenario_family": _public_safe_text(record.get("scenario_family"), limit=96),
        "scenario_version": _public_safe_text(record.get("scenario_version"), limit=96),
        "summary": _public_safe_text(record.get("summary"), limit=220),
        "source_run_id": _public_safe_text(record.get("source_run_id"), limit=120),
        "execution_backend": _public_safe_text(record.get("execution_backend"), limit=96),
        "candidate_backend_id": _public_safe_text(
            record.get("candidate_backend_id"),
            limit=120,
        ),
        "candidate_backend_version": _public_safe_text(
            record.get("candidate_backend_version"),
            limit=120,
        ),
        "outcome_label": _public_safe_text(record.get("outcome_label"), limit=80),
        "task_success": record.get("task_success") if isinstance(record.get("task_success"), bool) else None,
        "safety_success": (
            record.get("safety_success") if isinstance(record.get("safety_success"), bool) else None
        ),
        "preview_only": _safe_bool(record.get("preview_only")),
        "scenario_count": _safe_int(record.get("scenario_count")),
        "artifact_refs": [
            _public_episode_evidence_artifact(item)
            for item in _list(record.get("artifact_refs"))
            if isinstance(item, dict)
        ],
        "links": [
            _public_episode_evidence_link(item)
            for item in _list(record.get("links"))
            if isinstance(item, dict)
        ],
        "started_at": _public_safe_text(record.get("started_at"), limit=96),
        "ended_at": _public_safe_text(record.get("ended_at"), limit=96),
        "generated_at": _public_safe_text(record.get("generated_at"), limit=96),
        "reason_codes": _public_reason_code_list(record.get("reason_codes")),
        "reason_code_categories": _public_text_list(
            record.get("reason_code_categories"),
            limit=80,
        ),
    }


def _public_episode_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [
        _public_episode_evidence_row(record)
        for record in _list(payload.get("rows"))
        if isinstance(record, dict)
    ]
    return {
        "schema_version": _safe_int(payload.get("schema_version"), _SCHEMA_VERSION),
        "available": _safe_bool(payload.get("available", False)),
        "generated_at": _public_safe_text(payload.get("generated_at"), limit=96),
        "summary": _public_safe_text(payload.get("summary"), limit=180)
        or "Episode evidence unavailable.",
        "episode_count": _safe_int(payload.get("episode_count")),
        "source_counts": _public_count_mapping(payload.get("source_counts")),
        "reason_code_counts": _public_count_mapping(payload.get("reason_code_counts")),
        "rows": rows,
        "reason_codes": _public_reason_code_list(payload.get("reason_codes")),
    }


def _build_episode_evidence_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    session_ids, store, presence_scope_key = _runtime_scope(runtime)
    if store is None or session_ids is None:
        return _unavailable_section("episode_evidence", "runtime_evidence_surface_missing")
    controller = getattr(runtime, "live_routing_controller", None) or getattr(
        runtime, "rollout_controller", None
    )
    snapshot = build_episode_evidence_index(
        store=store,
        session_ids=session_ids,
        presence_scope_key=presence_scope_key,
        rollout_controller=controller,
        recent_limit=recent_limit,
    )
    payload = _public_episode_evidence_payload(snapshot.as_dict())
    return BrainOperatorWorkbenchSection(
        available=payload["available"],
        summary=payload["summary"],
        payload=payload,
        reason_codes=_public_reason_codes(payload, "operator_episode_evidence:available"),
    )


def _build_performance_learning_section(
    runtime: object,
    *,
    recent_limit: int,
) -> BrainOperatorWorkbenchSection:
    reader = getattr(runtime, "current_performance_learning_inspection", None)
    if callable(reader):
        payload = _mapping(reader())
    else:
        preferences_dir = (
            getattr(runtime, "performance_preferences_v3_dir", None)
            or PERFORMANCE_PREFERENCE_ARTIFACT_DIR
        )
        payload = build_performance_learning_inspection(
            preferences_dir=preferences_dir,
            recent_limit=recent_limit,
        )
    return BrainOperatorWorkbenchSection(
        available=_safe_bool(payload.get("available", True)),
        summary=_public_safe_text(payload.get("summary"), limit=180)
        or "Performance learning unavailable.",
        payload=payload,
        reason_codes=_public_reason_codes(payload, "operator_performance_learning:available"),
    )


def _build_rollout_status_section(runtime: object | None) -> BrainOperatorWorkbenchSection:
    if runtime is not None:
        status_payload: dict[str, Any] | None = None
        reader = getattr(runtime, "current_live_routing_status", None)
        if callable(reader):
            status_payload = _as_dict(reader())
        else:
            controller = getattr(runtime, "live_routing_controller", None)
            status_builder = getattr(controller, "current_status", None)
            if callable(status_builder):
                status_payload = _as_dict(status_builder())
        if status_payload:
            payload = _public_rollout_status_payload(status_payload)
            if payload["available"]:
                return BrainOperatorWorkbenchSection(
                    available=True,
                    summary=payload["summary"],
                    payload=payload,
                    reason_codes=_public_reason_codes(
                        payload,
                        "operator_rollout_status:available",
                    ),
                )

    payload = {
        "available": False,
        "governance_only": True,
        "live_routing_active": False,
        "controlled_rollout_supported": False,
        "summary": "Live rollout controller is not active in this slice.",
        "reason_codes": [
            "operator_rollout_status:unavailable",
            "live_routing_controller_missing",
            "governance_only",
        ],
    }
    return BrainOperatorWorkbenchSection(
        available=False,
        summary=payload["summary"],
        payload=payload,
        reason_codes=tuple(payload["reason_codes"]),
    )


def build_operator_workbench_snapshot(
    runtime: object | None,
    *,
    memory_limit: int = 40,
    recent_limit: int = 6,
) -> BrainOperatorWorkbenchSnapshot:
    """Build one compact public-safe aggregate operator workbench snapshot."""
    if runtime is None:
        sections = {
            key: _unavailable_section(key, "runtime_not_active")
            for key in _SECTION_KEYS
            if key != "rollout_status"
        }
        sections["rollout_status"] = _build_rollout_status_section(None)
    else:
        sections = {
            "expression": _safe_call(
                "expression",
                lambda: _build_expression_section(runtime),
            ),
            "behavior_controls": _safe_call(
                "behavior_controls",
                lambda: _build_behavior_controls_section(runtime),
            ),
            "teaching_knowledge": _safe_call(
                "teaching_knowledge",
                lambda: _build_teaching_knowledge_section(runtime, recent_limit=recent_limit),
            ),
            "voice_metrics": _safe_call(
                "voice_metrics",
                lambda: _build_voice_metrics_section(runtime),
            ),
            "memory": _safe_call(
                "memory",
                lambda: _build_memory_section(runtime, memory_limit=memory_limit),
            ),
            "practice": _safe_call(
                "practice",
                lambda: _build_practice_section(runtime, recent_limit=recent_limit),
            ),
            "adapters": _safe_call(
                "adapters",
                lambda: _build_adapters_section(runtime, recent_limit=recent_limit),
            ),
            "sim_to_real": _safe_call(
                "sim_to_real",
                lambda: _build_sim_to_real_section(runtime, recent_limit=recent_limit),
            ),
            "rollout_status": _build_rollout_status_section(runtime),
            "episode_evidence": _safe_call(
                "episode_evidence",
                lambda: _build_episode_evidence_section(runtime, recent_limit=recent_limit),
            ),
            "performance_learning": _safe_call(
                "performance_learning",
                lambda: _build_performance_learning_section(
                    runtime,
                    recent_limit=recent_limit,
                ),
            ),
        }
    available = any(section.available for section in sections.values())
    reason_codes = _dedupe(
        (
            "operator_workbench:v1",
            "operator_workbench:available" if available else "operator_workbench:unavailable",
            *(
                f"section:{key}:{'available' if sections[key].available else 'unavailable'}"
                for key in _SECTION_KEYS
            ),
            *(code for key in _SECTION_KEYS for code in sections[key].reason_codes),
        )
    )
    return BrainOperatorWorkbenchSnapshot(
        schema_version=_SCHEMA_VERSION,
        available=available,
        expression=sections["expression"],
        behavior_controls=sections["behavior_controls"],
        teaching_knowledge=sections["teaching_knowledge"],
        voice_metrics=sections["voice_metrics"],
        memory=sections["memory"],
        practice=sections["practice"],
        adapters=sections["adapters"],
        sim_to_real=sections["sim_to_real"],
        rollout_status=sections["rollout_status"],
        episode_evidence=sections["episode_evidence"],
        performance_learning=sections["performance_learning"],
        reason_codes=reason_codes,
    )


__all__ = [
    "BrainOperatorWorkbenchSection",
    "BrainOperatorWorkbenchSnapshot",
    "build_operator_workbench_snapshot",
]

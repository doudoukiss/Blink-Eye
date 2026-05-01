"""Public-safe event contract for future non-human avatar adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from blink.interaction.actor_events import ActorEventV2, sanitize_actor_metadata

AVATAR_ADAPTER_CONTRACT_SCHEMA_VERSION = 1
AVATAR_ADAPTER_ALLOWED_SURFACES = (
    "abstract_avatar",
    "status_avatar",
    "symbolic_avatar",
    "debug_status",
)
AVATAR_ADAPTER_FORBIDDEN_CAPABILITIES = (
    "realistic_human_likeness",
    "identity_cloning",
    "face_reenactment",
    "camera_frame_access",
    "audio_stream_access",
    "hidden_context_access",
)
_UNSAFE_PLAN_KEY_FRAGMENTS = (
    "audio",
    "developer_prompt",
    "example",
    "hidden",
    "image",
    "message",
    "prompt",
    "raw",
    "secret",
    "system_prompt",
    "token",
    "transcript",
)


def _safe_text(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = " ".join(str(value or "").split())[:limit]
    lowered = text.lower()
    if any(fragment in lowered for fragment in _UNSAFE_PLAN_KEY_FRAGMENTS):
        return default
    return text or default


def _safe_bool(value: object) -> bool:
    return value is True


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_plan_summary(plan: object) -> dict[str, object]:
    if not isinstance(plan, Mapping):
        return {}
    result: dict[str, object] = {}
    for key in (
        "stance",
        "response_shape",
        "plan_summary",
        "style_summary",
        "speech_chunk_budget",
        "subtitle_policy",
        "memory_callback_policy",
        "camera_reference_policy",
        "interruption_policy",
        "repair_policy",
    ):
        if key in plan:
            value = plan[key]
            if isinstance(value, Mapping):
                result[key] = {
                    "state": _safe_text(value.get("state"), default="unknown"),
                    "label": _safe_text(value.get("label"), default="unknown"),
                }
            else:
                result[key] = _safe_text(value, default="unknown", limit=160)
    refs = plan.get("persona_references_used") or plan.get("persona_references")
    if isinstance(refs, list | tuple):
        result["selected_reference_count"] = min(len(refs), 12)
    selected_memory_count = _safe_int(plan.get("selected_memory_count"))
    if selected_memory_count:
        result["selected_memory_count"] = selected_memory_count
    return result


def _safe_policy_summary(policy: object) -> dict[str, object]:
    if not isinstance(policy, Mapping):
        return {}
    result: dict[str, object] = {}
    for raw_key, value in policy.items():
        key = _safe_text(raw_key, default="", limit=80)
        if not key:
            continue
        allowed = key in {
            "boundary",
            "held_chunks",
            "honesty_state",
            "label",
            "outstanding_chunks",
            "ready_to_answer",
            "reason_codes",
            "state",
        } or key.endswith(("_count", "_counts", "_id", "_ids", "_label", "_labels", "_ms", "_state"))
        if not allowed:
            continue
        if isinstance(value, bool):
            result[key] = _safe_bool(value)
        elif isinstance(value, int):
            result[key] = _safe_int(value)
        elif isinstance(value, float):
            parsed = _safe_float(value)
            if parsed is not None:
                result[key] = parsed
        elif isinstance(value, str):
            result[key] = _safe_text(value, limit=80)
        elif isinstance(value, list | tuple):
            result[key] = [_safe_text(item, default="", limit=80) for item in value[:12]]
        elif isinstance(value, Mapping):
            nested = _safe_policy_summary(value)
            if nested:
                result[key] = nested
    return result


def _safe_control_frame_summary(frame: object) -> dict[str, object]:
    if not isinstance(frame, Mapping):
        return {}
    summary: dict[str, object] = {}
    for key in ("schema_version", "frame_id", "sequence", "boundary", "profile", "language", "tts_runtime_label"):
        if key not in frame:
            continue
        value = frame[key]
        if isinstance(value, int):
            summary[key] = _safe_int(value)
        else:
            summary[key] = _safe_text(value, limit=120)
    source_event_ids = frame.get("source_event_ids")
    if isinstance(source_event_ids, list | tuple):
        summary["source_event_count"] = min(len(source_event_ids), 64)
    for key in (
        "active_listener_policy",
        "browser_ui_policy",
        "camera_policy",
        "floor_policy",
        "memory_policy",
        "persona_policy",
        "repair_policy",
        "speech_policy",
    ):
        nested = _safe_policy_summary(frame.get(key))
        if nested:
            summary[key] = nested
    reason_trace = frame.get("reason_trace")
    if isinstance(reason_trace, list | tuple):
        summary["reason_codes"] = [_safe_text(reason, default="", limit=96) for reason in reason_trace[:12]]
    return summary


def _motion_intent(event_type: str, mode: str) -> str:
    if event_type == "looking" or mode == "looking":
        return "look_attention"
    if event_type in {"speaking", "speech_started"} or mode == "speaking":
        return "speech_active"
    if event_type in {"interruption_accepted", "output_flushed"} or mode == "interrupted":
        return "yield_or_reset"
    if event_type in {"listening", "listening_started"} or mode == "listening":
        return "listen_attention"
    if event_type == "thinking" or mode == "thinking":
        return "thinking_idle"
    if event_type in {"degraded", "error"} or mode in {"degraded", "error"}:
        return "degraded_notice"
    return "status_idle"


def _expression_intent(event_type: str, mode: str) -> str:
    if event_type in {"error", "degraded"} or mode in {"error", "degraded"}:
        return "limited_or_uncertain"
    if event_type == "interruption_rejected":
        return "continue_speaking"
    if event_type == "interruption_accepted":
        return "yielded"
    if mode == "looking":
        return "focused"
    if mode == "speaking":
        return "presenting"
    if mode == "listening":
        return "attentive"
    return "neutral"


@dataclass(frozen=True)
class AvatarAdapterEventContract:
    """Safe event envelope for future abstract/avatar surfaces."""

    event_id: int
    event_type: str
    mode: str
    profile: str
    language: str
    adapter_surface: str = "status_avatar"
    motion_intent: str = "status_idle"
    expression_intent: str = "neutral"
    performance_plan_summary: dict[str, object] = field(default_factory=dict)
    actor_control_frame_summary: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    source_contracts: tuple[str, ...] = ("actor_event",)
    schema_version: int = AVATAR_ADAPTER_CONTRACT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        """Serialize the adapter contract as a public-safe payload."""
        surface = (
            self.adapter_surface
            if self.adapter_surface in AVATAR_ADAPTER_ALLOWED_SURFACES
            else "status_avatar"
        )
        reason_codes = list(self.reason_codes)
        if surface != self.adapter_surface:
            reason_codes.append("avatar_adapter:unsupported_surface_downgraded")
        reason_codes.extend(
            [
                "avatar_adapter_contract:v1",
                "avatar_boundary:realistic_human_likeness_forbidden",
                "avatar_boundary:identity_cloning_forbidden",
                "avatar_boundary:raw_media_forbidden",
            ]
        )
        return {
            "schema_version": self.schema_version,
            "event_id": int(self.event_id),
            "event_type": _safe_text(self.event_type, limit=80),
            "mode": _safe_text(self.mode, limit=40),
            "profile": _safe_text(self.profile, default="manual", limit=96),
            "language": _safe_text(self.language, default="unknown", limit=32),
            "adapter_surface": surface,
            "allowed_surfaces": list(AVATAR_ADAPTER_ALLOWED_SURFACES),
            "forbidden_capabilities": list(AVATAR_ADAPTER_FORBIDDEN_CAPABILITIES),
            "motion_intent": _safe_text(self.motion_intent, limit=64),
            "expression_intent": _safe_text(self.expression_intent, limit=64),
            "performance_plan_summary": dict(self.performance_plan_summary),
            "actor_control_frame_summary": dict(self.actor_control_frame_summary),
            "metadata": dict(self.metadata),
            "realistic_human_likeness_allowed": False,
            "identity_cloning_allowed": False,
            "face_reenactment_allowed": False,
            "raw_media_allowed": False,
            "source_contracts": list(self.source_contracts),
            "reason_codes": sorted(set(reason_codes)),
        }


def compile_avatar_adapter_event_contract(
    actor_event: ActorEventV2 | Mapping[str, object] | None = None,
    *,
    actor_control_frame: Mapping[str, object] | None = None,
    performance_plan: Mapping[str, object] | None = None,
    adapter_surface: str = "status_avatar",
) -> AvatarAdapterEventContract:
    """Compile an actor event into a future adapter-safe event contract."""
    if actor_event is None:
        event_payload: dict[str, object] = {}
    else:
        event_payload = actor_event.as_dict() if isinstance(actor_event, ActorEventV2) else dict(actor_event)
    control_summary = _safe_control_frame_summary(actor_control_frame or {})
    if not event_payload and actor_control_frame:
        event_payload = {
            "event_id": control_summary.get("sequence", 0),
            "event_type": control_summary.get("boundary", "control_frame"),
            "mode": "control",
            "profile": control_summary.get("profile", "manual"),
            "language": control_summary.get("language", "unknown"),
            "metadata": {},
        }
    metadata, violations = sanitize_actor_metadata(event_payload.get("metadata") or {})
    mode = _safe_text(event_payload.get("mode"), default="waiting", limit=40)
    event_type = _safe_text(event_payload.get("event_type"), default="waiting", limit=80)
    plan_summary = _safe_plan_summary(performance_plan or {})
    metadata_summary: dict[str, object] = {}
    for key, value in metadata.items():
        if key.endswith(("_count", "_counts", "_state", "_kind", "_ms")) or key in {
            "floor_state",
            "grounding_mode",
            "text_kind",
        }:
            if isinstance(value, bool):
                metadata_summary[key] = _safe_bool(value)
            elif isinstance(value, int):
                metadata_summary[key] = _safe_int(value)
            elif isinstance(value, float):
                parsed = _safe_float(value)
                if parsed is not None:
                    metadata_summary[key] = parsed
            elif isinstance(value, str):
                metadata_summary[key] = _safe_text(value, limit=80)
    source_contracts = ["actor_event"]
    if actor_control_frame:
        source_contracts.append("actor_control_frame_v3")
    if performance_plan:
        source_contracts.append("performance_plan_summary")
    return AvatarAdapterEventContract(
        event_id=_safe_int(event_payload.get("event_id")),
        event_type=event_type,
        mode=mode,
        profile=_safe_text(event_payload.get("profile"), default="manual", limit=96),
        language=_safe_text(event_payload.get("language"), default="unknown", limit=32),
        adapter_surface=adapter_surface,
        motion_intent=_motion_intent(event_type, mode),
        expression_intent=_expression_intent(event_type, mode),
        performance_plan_summary=plan_summary,
        actor_control_frame_summary=control_summary,
        metadata=metadata_summary,
        reason_codes=tuple(violations),
        source_contracts=tuple(source_contracts),
    )


__all__ = [
    "AVATAR_ADAPTER_ALLOWED_SURFACES",
    "AVATAR_ADAPTER_CONTRACT_SCHEMA_VERSION",
    "AVATAR_ADAPTER_FORBIDDEN_CAPABILITIES",
    "AvatarAdapterEventContract",
    "compile_avatar_adapter_event_contract",
]

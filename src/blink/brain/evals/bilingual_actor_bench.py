"""Deterministic bilingual actor bench and release gate for Blink."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from blink.brain.evals.performance_preferences import (
    PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    PERFORMANCE_PREFERENCE_DIMENSIONS,
    PerformancePreferenceStore,
)
from blink.interaction.avatar_adapter_contract import AVATAR_ADAPTER_FORBIDDEN_CAPABILITIES

BILINGUAL_ACTOR_BENCH_SUITE_ID = "bilingual_actor_bench/v1"
BILINGUAL_ACTOR_BENCH_SCHEMA_VERSION = 1
BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR = Path("artifacts/bilingual-actor-bench")
BILINGUAL_ACTOR_RELEASE_THRESHOLD = 4.0
BILINGUAL_ACTOR_PRIMARY_PROFILES = ("browser-zh-melo", "browser-en-kokoro")
BILINGUAL_ACTOR_CATEGORIES = (
    "connection",
    "active_listening",
    "speech",
    "camera_grounding",
    "interruption",
    "memory_persona",
    "recovery",
    "long_session_stability",
)
BILINGUAL_ACTOR_QUALITY_DIMENSIONS = (
    "state_clarity",
    "felt_heard",
    "voice_pacing",
    "camera_grounding",
    "memory_usefulness",
    "interruption_naturalness",
    "personality_consistency",
    "enjoyment",
    "not_fake_human",
)
BILINGUAL_ACTOR_HARD_BLOCKERS = (
    "unsafe_trace_payload",
    "hidden_camera_use",
    "false_camera_claim",
    "self_interruption",
    "stale_tts_after_interruption",
    "memory_contradiction",
    "profile_regression",
    "missing_consent_or_privacy_control",
    "realistic_human_avatar_capability",
)
BILINGUAL_ACTOR_BENCH_FIXTURE_DIR = (
    Path(__file__).resolve().parents[4] / "evals" / "bilingual_actor_bench"
)
BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES = {
    "camera_grounding": "regression_camera_moondream.jsonl",
    "interruption": "regression_interruption_echo.jsonl",
    "memory_persona": "regression_memory_persona_cross_language.jsonl",
}
BILINGUAL_PERFORMANCE_BENCH_V3_SUITE_ID = "bilingual_performance_bench/v3"
BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION = 3
BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD = 4.0
BILINGUAL_PERFORMANCE_PARITY_DELTA_THRESHOLD = 0.35
BILINGUAL_PERFORMANCE_V3_CATEGORIES = (
    "connection",
    "listening",
    "speech",
    "overlap_interruption",
    "camera_grounding",
    "repair",
    "memory_persona",
    "long_session_continuity",
    "preference_comparison",
    "safety_controls",
)
BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS = (
    "state_clarity",
    "felt_heard",
    "voice_pacing",
    "interruption_naturalness",
    "camera_honesty",
    "memory_usefulness",
    "persona_consistency",
    "enjoyment",
    "not_fake_human",
)
BILINGUAL_PERFORMANCE_V3_METRICS = (
    "state_clarity",
    "perceived_responsiveness_proxy",
    "interruption_stop_latency_ms",
    "stale_chunk_drops",
    "camera_frame_age_policy",
    "memory_effect_rate",
    "persona_reference_hit_rate",
    "episode_sanitizer_pass_rate",
    "bilingual_parity_delta",
)
BILINGUAL_PERFORMANCE_V3_HARD_BLOCKERS = (
    "profile_regression",
    "hidden_camera_use",
    "false_camera_claim",
    "self_interruption",
    "stale_tts_after_interrupt",
    "memory_contradiction",
    "unsupported_tts_claim",
    "unsafe_trace_payload",
    "missing_consent_controls",
    "realistic_human_avatar_capability",
)
BILINGUAL_PERFORMANCE_BENCH_V3_CATEGORIES = BILINGUAL_PERFORMANCE_V3_CATEGORIES
BILINGUAL_PERFORMANCE_BENCH_V3_SCORE_DIMENSIONS = BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS
BILINGUAL_PERFORMANCE_BENCH_V3_METRICS = BILINGUAL_PERFORMANCE_V3_METRICS
BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS = BILINGUAL_PERFORMANCE_V3_HARD_BLOCKERS

_FIXED_TS = "2026-04-27T00:00:00+00:00"
_BANNED_KEY_FRAGMENTS = (
    "audio",
    "candidate",
    "credential",
    "developer_prompt",
    "hidden_prompt",
    "ice",
    "image",
    "message",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "system_prompt",
    "token",
    "transcript",
)
_SAFE_KEY_EXACT = {
    "actor_trace_persistence_opt_in",
    "candidate_a",
    "candidate_b",
    "candidate_id",
    "candidate_kind",
    "candidate_label",
    "camera_honesty_states",
    "control_frame_ids",
    "debug_transcript_storage_default",
    "debug_transcript_storage_opt_in",
    "episode_sanitizer_pass_rate",
    "final_transcript_chars",
    "first_audio_frame_latency_ms",
    "improvement_labels",
    "policy_proposals",
    "preference_comparison",
    "partial_transcript_chars",
    "raw_media_allowed",
    "source_pair_ids",
    "summary_hash",
    "unbounded_transcript_storage",
    "webrtc_audio_health",
}
_BANNED_VALUE_TOKENS = (
    "[BLINK_BRAIN_CONTEXT]",
    "a=candidate",
    "authorization:",
    "bearer ",
    "candidate:",
    "data:audio",
    "data:image",
    "developer prompt",
    "hidden prompt",
    "ice-ufrag",
    "m=audio",
    "m=video",
    "raw image",
    "secret",
    "sk-",
    "system prompt",
    "token",
    "v=0",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_json_safe(nested) for nested in value]
    return value


def _dedupe(values: Iterable[Any], *, limit: int = 64) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text[:120])
        if len(result) >= limit:
            break
    return tuple(result)


def _state_path(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _public_safety_violations(value: Any, *, path: str = "$") -> tuple[str, ...]:
    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            nested_path = f"{path}.{key}"
            if lowered not in _SAFE_KEY_EXACT and any(
                fragment in lowered for fragment in _BANNED_KEY_FRAGMENTS
            ):
                violations.append(f"unsafe_key:{nested_path}")
            violations.extend(_public_safety_violations(nested, path=nested_path))
    elif isinstance(value, tuple | list):
        for index, nested in enumerate(value):
            violations.extend(_public_safety_violations(nested, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for token in _BANNED_VALUE_TOKENS:
            if token in lowered:
                violations.append(f"unsafe_value:{path}:{token}")
    return tuple(violations)


def find_bilingual_actor_public_safety_violations(payload: Any) -> tuple[str, ...]:
    """Return public-safety violations for bench state, event, and gate payloads."""
    return _dedupe(_public_safety_violations(payload), limit=96)


def _profile_context(profile: str) -> dict[str, str]:
    if profile == "browser-zh-melo":
        return {
            "profile": "browser-zh-melo",
            "language": "zh",
            "tts_backend": "local-http-wav",
            "tts_label": "local-http-wav/MeloTTS",
            "vision_backend": "moondream",
        }
    return {
        "profile": "browser-en-kokoro",
        "language": "en",
        "tts_backend": "kokoro",
        "tts_label": "kokoro/English",
        "vision_backend": "moondream",
    }


def _record_profile(record: dict[str, Any]) -> str:
    return str(record.get("test_profile") or record.get("profile") or record.get("setup_profile") or "")


def _record_language(record: dict[str, Any]) -> str:
    return str(record.get("test_language") or record.get("language") or record.get("setup_language") or "")


def _event(
    event_id: int,
    event_type: str,
    *,
    profile: str,
    mode: str = "listening",
    metadata: dict[str, Any] | None = None,
    reason_codes: tuple[str, ...] = (),
) -> dict[str, Any]:
    context = _profile_context(profile)
    return {
        "schema_version": 2,
        "event_id": event_id,
        "event_type": event_type,
        "mode": mode,
        "timestamp": _FIXED_TS,
        "profile": context["profile"],
        "language": context["language"],
        "tts_backend": context["tts_backend"],
        "tts_label": context["tts_label"],
        "vision_backend": context["vision_backend"],
        "source": "bilingual_actor_bench",
        "session_id": "bench_session",
        "client_id": "bench_client",
        "metadata": dict(metadata or {}),
        "reason_codes": list(reason_codes),
    }


def _privacy_controls() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "camera_permission_visible": True,
        "camera_vision_opt_out_visible": True,
        "actor_trace_persistence_opt_in": True,
        "memory_inspect_edit_available": True,
        "debug_transcript_storage_default": "off",
        "debug_transcript_storage_opt_in": True,
        "reason_codes": ["privacy_controls:v1", "debug_transcript_storage:off"],
    }


def _avatar_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "adapter_surface": "status_avatar",
        "allowed_surfaces": ["abstract_avatar", "status_avatar", "symbolic_avatar"],
        "realistic_human_likeness_allowed": False,
        "identity_cloning_allowed": False,
        "face_reenactment_allowed": False,
        "raw_media_allowed": False,
        "reason_codes": ["avatar_adapter_contract:v1", "avatar_boundary:non_human"],
    }


def _actor_state(profile: str, *, category: str, mode: str = "listening") -> dict[str, Any]:
    context = _profile_context(profile)
    speech_mode = "melo_chunked" if profile == "browser-zh-melo" else "kokoro_chunked"
    return {
        "schema_version": 2,
        "runtime": "browser",
        "transport": "WebRTC",
        "profile": context["profile"],
        "language": context["language"],
        "mode": mode,
        "tts": {"backend": context["tts_backend"], "label": context["tts_label"]},
        "vision": {
            "enabled": True,
            "backend": "moondream",
            "continuous_perception_enabled": False,
            "current_answer_used_vision": category == "camera_grounding",
        },
        "protected_playback": True,
        "webrtc": {"media_mode": "camera_and_microphone", "client_active": True},
        "microphone": {"state": "receiving", "enabled": True, "available": True},
        "camera_scene": {
            "schema_version": 1,
            "profile": context["profile"],
            "language": context["language"],
            "enabled": True,
            "available": True,
            "state": "available",
            "status": "available",
            "vision_backend": "moondream",
            "continuous_perception_enabled": False,
            "permission_state": "granted",
            "camera_connected": True,
            "camera_fresh": True,
            "freshness_state": "fresh",
            "track_state": "healthy",
            "latest_frame_sequence": 12,
            "latest_frame_age_ms": 80,
            "latest_frame_received_at": _FIXED_TS,
            "on_demand_vision_state": "success" if category == "camera_grounding" else "idle",
            "current_answer_used_vision": category == "camera_grounding",
            "grounding_mode": "single_frame" if category == "camera_grounding" else "none",
            "last_vision_result_state": "success" if category == "camera_grounding" else "none",
            "last_used_frame_sequence": 12 if category == "camera_grounding" else None,
            "last_used_frame_age_ms": 80 if category == "camera_grounding" else None,
            "last_used_frame_at": _FIXED_TS if category == "camera_grounding" else None,
            "degradation": {
                "state": "ok",
                "components": [],
                "reason_codes": ["camera_scene_degradation:ok"],
            },
            "scene_social_state_v2": {
                "schema_version": 2,
                "state_id": f"scene-social-v2-{context['profile']}",
                "profile": context["profile"],
                "language": context["language"],
                "camera_status": "ready",
                "vision_status": "answered" if category == "camera_grounding" else "idle",
                "camera_honesty_state": (
                    "can_see_now"
                    if category == "camera_grounding"
                    else "recent_frame_available"
                ),
                "frame_freshness": "fresh",
                "frame_age_ms": 80,
                "latest_frame_sequence": 12,
                "last_used_frame_sequence": 12 if category == "camera_grounding" else None,
                "last_used_frame_age_ms": 80 if category == "camera_grounding" else None,
                "user_presence_hint": "unknown",
                "face_hint": "not_evaluated",
                "body_hint": "not_evaluated",
                "hands_hint": "not_evaluated",
                "object_hint": "object_showing"
                if category == "camera_grounding"
                else "not_evaluated",
                "object_showing_likelihood": 0.85 if category == "camera_grounding" else 0.0,
                "scene_change_reason": (
                    "vision_answered" if category == "camera_grounding" else "camera_ready"
                ),
                "scene_transition": (
                    "vision_answered" if category == "camera_grounding" else "camera_ready"
                ),
                "last_moondream_result_state": (
                    "answered" if category == "camera_grounding" else "none"
                ),
                "last_grounding_summary": (
                    "fresh_object_grounding" if category == "camera_grounding" else None
                ),
                "last_grounding_summary_hash": (
                    "0" * 16 if category == "camera_grounding" else None
                ),
                "confidence": 0.85 if category == "camera_grounding" else 0.0,
                "confidence_bucket": "high" if category == "camera_grounding" else "none",
                "scene_age_ms": 80,
                "updated_at_ms": 0,
                "reason_codes": [
                    "scene_social:v2",
                    "camera_honesty:can_see_now"
                    if category == "camera_grounding"
                    else "camera_honesty:recent_frame_available",
                ],
            },
            "reason_codes": ["camera_scene:v1", "camera_scene:available"],
        },
        "active_listening": {
            "schema_version": 2,
            "phase": "ready_to_answer" if category == "active_listening" else "idle",
            "partial_available": category == "active_listening",
            "final_available": category == "active_listening",
            "partial_transcript_chars": 28 if category == "active_listening" else 0,
            "final_transcript_chars": 56 if category == "active_listening" else 0,
            "topic_count": 2 if category == "active_listening" else 0,
            "constraint_count": 1 if category == "active_listening" else 0,
            "ready_to_answer": category == "active_listening",
            "reason_codes": ["active_listener:v2"],
        },
        "speech": {
            "director_mode": speech_mode,
            "first_subtitle_latency_ms": 100,
            "first_audio_frame_latency_ms": 390,
            "speech_queue_depth_current": 0,
            "stale_chunk_drop_count": 1 if category == "interruption" else 0,
            "stale_output_played_after_interruption": False,
            "reason_codes": ["speech_performance:chunked"],
        },
        "interruption": {
            "protected_playback": True,
            "barge_in_state": "protected",
            "last_decision": "rejected" if category == "interruption" else "none",
            "self_interruption": False,
            "reason_codes": ["interruption:protected_default"],
        },
        "webrtc_audio_health": {
            "barge_in_state": "protected",
            "echo_risk_level": "medium",
            "protected_playback": True,
            "reason_codes": ["webrtc_audio_health:protected"],
        },
        "memory_persona": {
            "available": True,
            "selected_memory_count": 2 if category == "memory_persona" else 0,
            "suppressed_memory_count": 1 if category == "memory_persona" else 0,
            "memory_effect": "callback_available" if category == "memory_persona" else "none",
            "memory_contradiction": False,
            "performance_plan_v2": {
                "schema_version": 2,
                "style_summary": "Concise, honest actor-state explanation.",
                "selected_reference_count": 2 if category == "memory_persona" else 0,
                "reason_codes": ["performance_plan:v2"],
            },
            "reason_codes": ["memory_persona:available"],
        },
        "degradation": {"state": "ok", "components": [], "reason_codes": ["degradation:ok"]},
        "reason_codes": ["browser_actor_state:v2", f"bench_category:{category}"],
    }


def _score_overrides(category: str) -> dict[str, float]:
    scores = {dimension: 4.4 for dimension in BILINGUAL_ACTOR_QUALITY_DIMENSIONS}
    if category == "active_listening":
        scores["felt_heard"] = 4.6
        scores["state_clarity"] = 4.6
    if category == "speech":
        scores["voice_pacing"] = 4.6
    if category == "camera_grounding":
        scores["camera_grounding"] = 4.7
    if category == "interruption":
        scores["interruption_naturalness"] = 4.5
    if category == "memory_persona":
        scores["memory_usefulness"] = 4.6
        scores["personality_consistency"] = 4.6
    return scores


@dataclass(frozen=True)
class BilingualActorHistoricalRegressionResult:
    """Sanitized result for a historical bilingual actor regression fixture."""

    fixture_file: str
    fixture_id: str
    category: str
    profile: str
    passed: bool
    hard_blockers: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the public-safe fixture result."""
        return {
            "fixture_file": self.fixture_file,
            "fixture_id": self.fixture_id,
            "category": self.category,
            "profile": self.profile,
            "passed": self.passed,
            "hard_blockers": list(self.hard_blockers),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualActorBenchCase:
    """One deterministic bilingual actor bench case."""

    case_id: str
    pair_id: str
    profile: str
    category: str
    title: str
    actor_state: dict[str, Any]
    actor_events: tuple[dict[str, Any], ...]
    quality_scores: dict[str, float]
    privacy_controls: dict[str, Any] = field(default_factory=_privacy_controls)
    avatar_contract: dict[str, Any] = field(default_factory=_avatar_contract)
    historical_regressions: tuple[BilingualActorHistoricalRegressionResult, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def language(self) -> str:
        """Return the case language."""
        return _profile_context(self.profile)["language"]

    @property
    def tts_backend(self) -> str:
        """Return the case TTS backend."""
        return _profile_context(self.profile)["tts_backend"]

    @property
    def tts_label(self) -> str:
        """Return the public case TTS label."""
        return _profile_context(self.profile)["tts_label"]

    def as_dict(self) -> dict[str, Any]:
        """Serialize public case metadata without raw state/event bodies."""
        return {
            "schema_version": BILINGUAL_ACTOR_BENCH_SCHEMA_VERSION,
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "profile": self.profile,
            "language": self.language,
            "tts_backend": self.tts_backend,
            "tts_label": self.tts_label,
            "vision_backend": "moondream",
            "category": self.category,
            "title": self.title,
            "event_count": len(self.actor_events),
            "historical_regression_count": len(self.historical_regressions),
            "quality_dimensions": list(BILINGUAL_ACTOR_QUALITY_DIMENSIONS),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualActorBenchCheckResult:
    """One deterministic check result."""

    check_id: str
    passed: bool
    detail: str
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the check result."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "detail": self.detail,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualActorBenchCaseResult:
    """Evaluated bilingual actor bench case."""

    case: BilingualActorBenchCase
    passed: bool
    checks: tuple[BilingualActorBenchCheckResult, ...]
    scores: dict[str, float]
    hard_blockers: tuple[str, ...]
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the case result."""
        return {
            "case": self.case.as_dict(),
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "scores": dict(sorted(self.scores.items())),
            "hard_blockers": list(self.hard_blockers),
            "evidence": _json_safe(self.evidence),
        }


@dataclass(frozen=True)
class BilingualActorProfileResult:
    """Per-profile aggregate result."""

    profile: str
    language: str
    tts_backend: str
    tts_label: str
    case_count: int
    category_results: dict[str, bool]
    scores: dict[str, float]
    hard_blockers: tuple[str, ...]
    passed: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the profile result."""
        return {
            "profile": self.profile,
            "language": self.language,
            "tts_backend": self.tts_backend,
            "tts_label": self.tts_label,
            "case_count": self.case_count,
            "category_results": dict(sorted(self.category_results.items())),
            "scores": dict(sorted(self.scores.items())),
            "hard_blockers": list(self.hard_blockers),
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualActorReleaseGateResult:
    """Strict release gate for the bilingual actor bench."""

    passed: bool
    threshold: float
    hard_blockers: tuple[str, ...]
    scores: dict[str, dict[str, float]]
    profile_failures: dict[str, list[str]]
    reason_codes: tuple[str, ...]
    schema_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Serialize the release gate result."""
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "threshold": self.threshold,
            "hard_blockers": list(self.hard_blockers),
            "scores": _json_safe(self.scores),
            "profile_failures": _json_safe(self.profile_failures),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualActorBenchReport:
    """Deterministic bilingual actor bench report."""

    suite_id: str
    run_id: str
    case_results: tuple[BilingualActorBenchCaseResult, ...]
    profile_results: tuple[BilingualActorProfileResult, ...]
    release_gate: BilingualActorReleaseGateResult
    paired_comparison: dict[str, Any]
    generated_at: str = _FIXED_TS
    artifact_links: dict[str, str] = field(default_factory=dict)
    schema_version: int = BILINGUAL_ACTOR_BENCH_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """Return whether the release gate passed."""
        return self.release_gate.passed

    def as_dict(self) -> dict[str, Any]:
        """Serialize the full bench report."""
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "threshold": self.release_gate.threshold,
            "profile_results": [result.as_dict() for result in self.profile_results],
            "paired_comparison": _json_safe(self.paired_comparison),
            "release_gate": self.release_gate.as_dict(),
            "case_results": [result.as_dict() for result in self.case_results],
            "artifact_links": dict(sorted(self.artifact_links.items())),
        }


def _check(
    check_id: str,
    passed: bool,
    detail: str,
    *reason_codes: str,
) -> BilingualActorBenchCheckResult:
    return BilingualActorBenchCheckResult(
        check_id=check_id,
        passed=bool(passed),
        detail=detail,
        reason_codes=_dedupe(reason_codes),
    )


def _profile_defaults_ok(case: BilingualActorBenchCase) -> bool:
    state = case.actor_state
    context = _profile_context(case.profile)
    return (
        state.get("runtime") == "browser"
        and state.get("transport") == "WebRTC"
        and state.get("profile") == context["profile"]
        and state.get("language") == context["language"]
        and _state_path(state, "tts.backend") == context["tts_backend"]
        and _state_path(state, "tts.label") == context["tts_label"]
        and _state_path(state, "vision.enabled") is True
        and _state_path(state, "vision.backend") == "moondream"
        and _state_path(state, "vision.continuous_perception_enabled") is False
        and state.get("protected_playback") is True
    )


def _privacy_controls_ok(controls: dict[str, Any]) -> bool:
    return (
        controls.get("camera_permission_visible") is True
        and controls.get("camera_vision_opt_out_visible") is True
        and controls.get("actor_trace_persistence_opt_in") is True
        and controls.get("memory_inspect_edit_available") is True
        and controls.get("debug_transcript_storage_default") == "off"
        and controls.get("debug_transcript_storage_opt_in") is True
    )


def _avatar_contract_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("realistic_human_likeness_allowed") is False
        and contract.get("identity_cloning_allowed") is False
        and contract.get("face_reenactment_allowed") is False
        and contract.get("raw_media_allowed") is False
    )


def _event_types(events: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(event.get("event_type") or "") for event in events)


def _contains_order(observed: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    if not expected:
        return True
    index = 0
    for event_type in observed:
        if event_type == expected[index]:
            index += 1
            if index >= len(expected):
                return True
    return False


def _category_ok(case: BilingualActorBenchCase) -> bool:
    state = case.actor_state
    events = _event_types(case.actor_events)
    if case.category == "connection":
        return "connected" in events and state.get("mode") == "listening"
    if case.category == "active_listening":
        return (
            "listening_started" in events
            and "partial_understanding_updated" in events
            and "final_understanding_ready" in events
            and _state_path(state, "active_listening.ready_to_answer") is True
        )
    if case.category == "speech":
        return (
            _contains_order(events, ("speech.subtitle_ready", "speaking"))
            and int(_state_path(state, "speech.first_subtitle_latency_ms", -1)) >= 0
            and _state_path(state, "speech.director_mode") in {"melo_chunked", "kokoro_chunked"}
        )
    if case.category == "camera_grounding":
        return (
            "looking" in events
            and _state_path(state, "camera_scene.current_answer_used_vision") is True
            and _state_path(state, "camera_scene.grounding_mode") == "single_frame"
            and _state_path(state, "camera_scene.camera_fresh") is True
        )
    if case.category == "interruption":
        return (
            "interruption_candidate" in events
            and "interruption_rejected" in events
            and _state_path(state, "interruption.barge_in_state") == "protected"
            and _state_path(state, "interruption.self_interruption") is False
        )
    if case.category == "memory_persona":
        return (
            "persona_plan_compiled" in events
            and "memory_used" in events
            and int(_state_path(state, "memory_persona.selected_memory_count", 0)) >= 1
            and _state_path(state, "memory_persona.memory_contradiction") is False
        )
    if case.category == "recovery":
        return "degraded" in events and "recovered" in events
    if case.category == "long_session_stability":
        return (
            int(_state_path(state, "long_session.turn_count", 0)) >= 12
            and int(_state_path(state, "long_session.trace_event_count", 0)) <= 10000
            and _state_path(state, "long_session.unbounded_transcript_storage") is False
        )
    return False


def _hard_blockers_for_case(case: BilingualActorBenchCase) -> tuple[str, ...]:
    state = case.actor_state
    events = _event_types(case.actor_events)
    blockers: list[str] = []
    if find_bilingual_actor_public_safety_violations(
        {
            "actor_state": state,
            "actor_events": case.actor_events,
            "privacy_controls": case.privacy_controls,
            "avatar_contract": case.avatar_contract,
        }
    ):
        blockers.append("unsafe_trace_payload")
    if (
        _state_path(state, "camera_scene.current_answer_used_vision") is True
        and "looking" not in events
    ):
        blockers.append("hidden_camera_use")
    if (
        state.get("answer_claims_vision") is True
        and _state_path(state, "camera_scene.current_answer_used_vision") is not True
    ):
        blockers.append("false_camera_claim")
    if (
        _state_path(state, "camera_scene.scene_social_state_v2.camera_honesty_state")
        == "can_see_now"
        and _state_path(state, "camera_scene.current_answer_used_vision") is not True
    ):
        blockers.append("false_camera_claim")
    if (
        state.get("answer_claims_vision") is True
        and _state_path(state, "camera_scene.scene_social_state_v2.camera_honesty_state")
        != "can_see_now"
    ):
        blockers.append("false_camera_claim")
    if (
        _state_path(state, "interruption.self_interruption") is True
        or (
            _state_path(state, "interruption.last_decision") == "accepted"
            and state.get("protected_playback") is True
        )
    ):
        blockers.append("self_interruption")
    if _state_path(state, "speech.stale_output_played_after_interruption") is True:
        blockers.append("stale_tts_after_interruption")
    if _state_path(state, "memory_persona.memory_contradiction") is True:
        blockers.append("memory_contradiction")
    if not _profile_defaults_ok(case):
        blockers.append("profile_regression")
    if not _privacy_controls_ok(case.privacy_controls):
        blockers.append("missing_consent_or_privacy_control")
    if not _avatar_contract_ok(case.avatar_contract):
        blockers.append("realistic_human_avatar_capability")
    for regression in case.historical_regressions:
        if not regression.passed:
            blockers.extend(regression.hard_blockers or ("profile_regression",))
    return _dedupe(blockers, limit=len(BILINGUAL_ACTOR_HARD_BLOCKERS))


def _regression_result(
    *,
    fixture_file: str,
    fixture_id: str,
    category: str,
    profile: str,
    passed: bool,
    hard_blockers: Iterable[str] = (),
    reason_codes: Iterable[str] = (),
) -> BilingualActorHistoricalRegressionResult:
    return BilingualActorHistoricalRegressionResult(
        fixture_file=fixture_file,
        fixture_id=fixture_id[:120] or "unknown_fixture",
        category=category,
        profile=profile if profile in BILINGUAL_ACTOR_PRIMARY_PROFILES else "unknown",
        passed=bool(passed),
        hard_blockers=_dedupe(hard_blockers, limit=len(BILINGUAL_ACTOR_HARD_BLOCKERS)),
        reason_codes=_dedupe(reason_codes),
    )


def _load_jsonl_records(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            records.append(
                {
                    "case_id": f"malformed_line_{line_number}",
                    "fixture_malformed": True,
                    "line_number": line_number,
                }
            )
            continue
        records.append(payload if isinstance(payload, dict) else {"fixture_non_object": True})
    return tuple(records)


def _validate_camera_regression_record(
    record: dict[str, Any],
    *,
    fixture_file: str,
) -> BilingualActorHistoricalRegressionResult:
    profile = _record_profile(record)
    context = _profile_context(profile)
    expected = record.get("expected_camera_scene")
    expected = expected if isinstance(expected, dict) else {}
    required_events = tuple(str(event) for event in record.get("required_events") or ())
    reason_codes = [
        "historical_regression:camera_moondream",
        *[str(code) for code in record.get("reason_codes") or ()],
    ]
    blockers: list[str] = []

    if record.get("fixture_malformed") or record.get("fixture_non_object"):
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:malformed")
    if profile not in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:profile_invalid")
    if _record_language(record) != context["language"]:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:language_mismatch")
    if record.get("vision_backend") != "moondream":
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:vision_backend_mismatch")
    if record.get("continuous_perception_enabled") is not False:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:continuous_perception_regression")
    if find_bilingual_actor_public_safety_violations(record):
        blockers.append("unsafe_trace_payload")
        reason_codes.append("historical_fixture:unsafe_payload")

    used_vision = expected.get("current_answer_used_vision") is True
    if used_vision:
        if "vision.fetch_user_image_start" not in required_events:
            blockers.append("hidden_camera_use")
            reason_codes.append("historical_fixture:missing_vision_start")
        if "vision.fetch_user_image_success" not in required_events:
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:missing_vision_success")
        if expected.get("grounding_mode") != "single_frame":
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:grounding_mode_mismatch")
        if expected.get("on_demand_vision_state") != "success":
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:vision_not_success")
        expected_social = expected.get("scene_social_state_v2")
        expected_social = expected_social if isinstance(expected_social, dict) else {}
        if expected_social.get("camera_honesty_state") != "can_see_now":
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:camera_honesty_mismatch")
        if expected_social.get("scene_transition") != "vision_answered":
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:scene_transition_mismatch")
    elif expected.get("state") in {"stale", "permission_needed", "disabled", "error"}:
        if expected.get("grounding_mode") not in {None, "none"}:
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:limited_camera_claim")
        if "vision.fetch_user_image_success" in required_events:
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:limited_camera_success_claim")
        expected_social = expected.get("scene_social_state_v2")
        expected_social = expected_social if isinstance(expected_social, dict) else {}
        if expected_social.get("camera_honesty_state") == "can_see_now":
            blockers.append("false_camera_claim")
            reason_codes.append("historical_fixture:false_camera_honesty")

    return _regression_result(
        fixture_file=fixture_file,
        fixture_id=str(record.get("case_id") or "camera_regression"),
        category="camera_grounding",
        profile=profile,
        passed=not blockers,
        hard_blockers=blockers,
        reason_codes=reason_codes,
    )


def _validate_interruption_regression_record(
    record: dict[str, Any],
    *,
    fixture_file: str,
) -> BilingualActorHistoricalRegressionResult:
    profile = _record_profile(record)
    context = _profile_context(profile)
    expected = record.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    client_media = record.get("client_media")
    client_media = client_media if isinstance(client_media, dict) else {}
    reason_codes = ["historical_regression:interruption_echo"]
    blockers: list[str] = []

    if record.get("fixture_malformed") or record.get("fixture_non_object"):
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:malformed")
    if profile not in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:profile_invalid")
    if _record_language(record) != context["language"] or record.get("tts_label") != context["tts_label"]:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:profile_label_mismatch")
    if find_bilingual_actor_public_safety_violations(record):
        blockers.append("unsafe_trace_payload")
        reason_codes.append("historical_fixture:unsafe_payload")
    if expected.get("self_interrupt") is True:
        blockers.append("self_interruption")
        reason_codes.append("historical_fixture:self_interrupt")
    if expected.get("stale_output_played") is True:
        blockers.append("stale_tts_after_interruption")
        reason_codes.append("historical_fixture:stale_output_played")

    output_route = str(client_media.get("output_route") or "")
    echo_safe = client_media.get("echo_safe") is True
    decision = str(expected.get("decision") or "")
    actor_event = str(expected.get("actor_event") or "")
    if output_route == "speaker" and not echo_safe and decision == "accepted":
        blockers.append("self_interruption")
        reason_codes.append("historical_fixture:speaker_mode_accepted")
    if actor_event == "interruption_accepted" and not echo_safe and expected.get("barge_in_state") != "armed":
        blockers.append("self_interruption")
        reason_codes.append("historical_fixture:accepted_without_echo_safe")
    if str(record.get("case_id") or "").endswith("stale_output_flush"):
        if actor_event != "output_flushed" or expected.get("stale_output_played") is not False:
            blockers.append("stale_tts_after_interruption")
            reason_codes.append("historical_fixture:missing_stale_output_flush")

    return _regression_result(
        fixture_file=fixture_file,
        fixture_id=str(record.get("case_id") or "interruption_regression"),
        category="interruption",
        profile=profile,
        passed=not blockers,
        hard_blockers=blockers,
        reason_codes=reason_codes,
    )


def _validate_memory_regression_record(
    record: dict[str, Any],
    *,
    fixture_file: str,
) -> BilingualActorHistoricalRegressionResult:
    profile = _record_profile(record)
    context = _profile_context(profile)
    expected_trace = record.get("expected_trace")
    expected_trace = expected_trace if isinstance(expected_trace, dict) else {}
    expected_behavior = record.get("expected_behavior")
    expected_behavior = expected_behavior if isinstance(expected_behavior, dict) else {}
    reason_codes = ["historical_regression:memory_persona_cross_language"]
    blockers: list[str] = []

    if record.get("fixture_malformed") or record.get("fixture_non_object"):
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:malformed")
    if profile not in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:profile_invalid")
    if _record_language(record) and _record_language(record) != context["language"]:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:language_mismatch")
    if find_bilingual_actor_public_safety_violations(record):
        blockers.append("unsafe_trace_payload")
        reason_codes.append("historical_fixture:unsafe_payload")
    if expected_trace and expected_trace.get("schema_version") != 1:
        blockers.append("profile_regression")
        reason_codes.append("historical_fixture:trace_schema_mismatch")
    if record.get("expected_tool") == "brain_explain_memory_continuity":
        expected_fields = {str(value) for value in record.get("expected_trace_fields") or ()}
        if not {
            "memory_effect",
            "selected_memory_count",
            "suppressed_memory_count",
            "cross_language_count",
        }.issubset(expected_fields):
            blockers.append("profile_regression")
            reason_codes.append("historical_fixture:explain_trace_fields_incomplete")
    if (
        "cross_language" in str(record.get("case_id") or "")
        and int(expected_trace.get("cross_language_count_min") or 0) < 1
    ):
        blockers.append("memory_contradiction")
        reason_codes.append("historical_fixture:missing_cross_language_selection")
    if expected_behavior.get("no_contradictory_current_facts") is False:
        blockers.append("memory_contradiction")
        reason_codes.append("historical_fixture:memory_contradiction")
    forbidden = tuple(str(value).lower() for value in record.get("forbidden") or ())
    if any(fragment in forbidden for fragment in ("system_prompt", "developer_prompt", "raw_memory_body")):
        reason_codes.append("historical_fixture:forbidden_private_explanation_fields_declared")

    return _regression_result(
        fixture_file=fixture_file,
        fixture_id=str(record.get("case_id") or "memory_regression"),
        category="memory_persona",
        profile=profile,
        passed=not blockers,
        hard_blockers=blockers,
        reason_codes=reason_codes,
    )


def load_bilingual_actor_historical_regression_results(
    fixture_dir: str | Path = BILINGUAL_ACTOR_BENCH_FIXTURE_DIR,
) -> tuple[BilingualActorHistoricalRegressionResult, ...]:
    """Load and validate checked-in historical bilingual actor regression fixtures."""
    fixture_path = Path(fixture_dir)
    results: list[BilingualActorHistoricalRegressionResult] = []
    validators = {
        "camera_grounding": _validate_camera_regression_record,
        "interruption": _validate_interruption_regression_record,
        "memory_persona": _validate_memory_regression_record,
    }
    for category, fixture_file in BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES.items():
        path = fixture_path / fixture_file
        if not path.exists():
            for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
                results.append(
                    _regression_result(
                        fixture_file=fixture_file,
                        fixture_id=f"missing_{category}_fixture",
                        category=category,
                        profile=profile,
                        passed=False,
                        hard_blockers=("profile_regression",),
                        reason_codes=("historical_fixture:missing",),
                    )
                )
            continue
        validator = validators[category]
        for record in _load_jsonl_records(path):
            results.append(validator(record, fixture_file=fixture_file))
    return tuple(results)


def _historical_regressions_for_case(
    results: Iterable[BilingualActorHistoricalRegressionResult],
    *,
    profile: str,
    category: str,
) -> tuple[BilingualActorHistoricalRegressionResult, ...]:
    return tuple(
        result for result in results if result.profile == profile and result.category == category
    )


def build_bilingual_actor_bench_suite() -> tuple[BilingualActorBenchCase, ...]:
    """Return the built-in matched bilingual actor bench cases."""
    cases: list[BilingualActorBenchCase] = []
    historical_results = load_bilingual_actor_historical_regression_results()
    titles = {
        "connection": "WebRTC connection reaches public listening state",
        "active_listening": "Active listener updates understanding before final answer",
        "speech": "Speech chunks expose subtitles before audio playback completes",
        "camera_grounding": "Moondream use is fresh single-frame grounded",
        "interruption": "Protected playback rejects false interruption safely",
        "memory_persona": "Memory and persona influence is visible and bounded",
        "recovery": "Degradation and recovery are visible without profile regression",
        "long_session_stability": "Long-session state stays bounded and profile-stable",
    }
    for category in BILINGUAL_ACTOR_CATEGORIES:
        for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
            context = _profile_context(profile)
            mode = "listening"
            if category == "speech":
                mode = "speaking"
            elif category == "camera_grounding":
                mode = "looking"
            elif category in {"memory_persona", "active_listening"}:
                mode = "thinking"
            state = _actor_state(profile, category=category, mode=mode)
            if category == "long_session_stability":
                state = {
                    **state,
                    "long_session": {
                        "turn_count": 18,
                        "trace_event_count": 144,
                        "unbounded_transcript_storage": False,
                        "profile_switch_count": 0,
                    },
                }
            event_map = {
                "connection": (
                    _event(1, "connected", profile=profile, mode="connected"),
                    _event(2, "listening", profile=profile, mode="listening"),
                ),
                "active_listening": (
                    _event(1, "listening_started", profile=profile),
                    _event(2, "partial_understanding_updated", profile=profile),
                    _event(3, "final_understanding_ready", profile=profile, mode="heard"),
                ),
                "speech": (
                    _event(1, "thinking", profile=profile, mode="thinking"),
                    _event(2, "speech.subtitle_ready", profile=profile, mode="speaking"),
                    _event(3, "speaking", profile=profile, mode="speaking"),
                    _event(4, "waiting", profile=profile, mode="waiting"),
                ),
                "camera_grounding": (
                    _event(1, "looking", profile=profile, mode="looking"),
                    _event(2, "looking", profile=profile, mode="thinking"),
                ),
                "interruption": (
                    _event(1, "speaking", profile=profile, mode="speaking"),
                    _event(2, "interruption_candidate", profile=profile, mode="speaking"),
                    _event(3, "interruption_rejected", profile=profile, mode="speaking"),
                    _event(4, "interruption_recovered", profile=profile, mode="recovered"),
                ),
                "memory_persona": (
                    _event(1, "persona_plan_compiled", profile=profile, mode="thinking"),
                    _event(2, "memory_used", profile=profile, mode="thinking"),
                ),
                "recovery": (
                    _event(1, "degraded", profile=profile, mode="degraded"),
                    _event(2, "recovered", profile=profile, mode="recovered"),
                ),
                "long_session_stability": (
                    _event(1, "connected", profile=profile, mode="connected"),
                    _event(2, "listening", profile=profile, mode="listening"),
                    _event(3, "thinking", profile=profile, mode="thinking"),
                    _event(4, "speaking", profile=profile, mode="speaking"),
                    _event(5, "waiting", profile=profile, mode="waiting"),
                ),
            }
            locale_prefix = "zh" if context["language"] == "zh" else "en"
            cases.append(
                BilingualActorBenchCase(
                    case_id=f"{locale_prefix}_{category}",
                    pair_id=category,
                    profile=profile,
                    category=category,
                    title=titles[category],
                    actor_state=state,
                    actor_events=event_map[category],
                    quality_scores=_score_overrides(category),
                    historical_regressions=_historical_regressions_for_case(
                        historical_results,
                        profile=profile,
                        category=category,
                    ),
                    reason_codes=(f"profile:{profile}", f"category:{category}"),
                )
            )
    return tuple(cases)


def evaluate_bilingual_actor_bench_case(
    case: BilingualActorBenchCase,
    *,
    threshold: float = BILINGUAL_ACTOR_RELEASE_THRESHOLD,
) -> BilingualActorBenchCaseResult:
    """Evaluate one deterministic bilingual actor bench case."""
    blockers = _hard_blockers_for_case(case)
    scores_ok = all(float(score) >= threshold for score in case.quality_scores.values())
    category_ok = _category_ok(case)
    historical_regressions_ok = all(
        regression.passed for regression in case.historical_regressions
    )
    checks = (
        _check(
            "public_safety",
            "unsafe_trace_payload" not in blockers,
            "State and event payloads avoid raw private fields.",
            "public_safety:ok",
        ),
        _check(
            "profile_defaults",
            "profile_regression" not in blockers,
            "Primary browser profile defaults are preserved.",
            "profile_defaults:ok",
        ),
        _check(
            "privacy_controls",
            "missing_consent_or_privacy_control" not in blockers,
            "Camera, trace, memory, and debug transcript controls are visible.",
            "privacy_controls:ok",
        ),
        _check(
            "avatar_boundary",
            "realistic_human_avatar_capability" not in blockers,
            "Avatar adapter contract stays non-human and raw-media-free.",
            "avatar_boundary:ok",
        ),
        _check(
            "category_contract",
            category_ok,
            f"{case.category} category contract passes.",
            f"category:{case.category}",
        ),
        _check(
            "historical_regressions",
            historical_regressions_ok,
            "Checked-in historical bilingual regression fixtures pass.",
            "historical_regressions:ok",
        ),
        _check(
            "score_threshold",
            scores_ok,
            f"All quality dimensions meet threshold {threshold:.1f}.",
            "scores:threshold",
        ),
    )
    passed = not blockers and category_ok and historical_regressions_ok and scores_ok
    evidence = {
        "profile": case.profile,
        "language": case.language,
        "category": case.category,
        "event_types": _event_types(case.actor_events),
        "historical_regressions": [
            regression.as_dict() for regression in case.historical_regressions
        ],
        "hard_blocker_count": len(blockers),
        "threshold": threshold,
    }
    return BilingualActorBenchCaseResult(
        case=case,
        passed=passed,
        checks=checks,
        scores=dict(case.quality_scores),
        hard_blockers=blockers,
        evidence=evidence,
    )


def _profile_result(
    profile: str,
    results: tuple[BilingualActorBenchCaseResult, ...],
    *,
    threshold: float,
) -> BilingualActorProfileResult:
    context = _profile_context(profile)
    selected = tuple(result for result in results if result.case.profile == profile)
    scores: dict[str, float] = {}
    for dimension in BILINGUAL_ACTOR_QUALITY_DIMENSIONS:
        values = [float(result.scores[dimension]) for result in selected]
        scores[dimension] = round(sum(values) / len(values), 4) if values else 0.0
    category_results = {}
    for category in BILINGUAL_ACTOR_CATEGORIES:
        category_results[category] = all(
            any(check.check_id == "category_contract" and check.passed for check in result.checks)
            for result in selected
            if result.case.category == category
        )
    hard_blockers = _dedupe(blocker for result in selected for blocker in result.hard_blockers)
    failed_scores = tuple(
        dimension for dimension, score in scores.items() if float(score) < threshold
    )
    case_failed_scores = tuple(
        f"{result.case.case_id}:{dimension}"
        for result in selected
        for dimension, score in result.scores.items()
        if float(score) < threshold
    )
    passed = (
        bool(selected)
        and all(category_results.values())
        and not hard_blockers
        and not failed_scores
        and not case_failed_scores
    )
    reason_codes = _dedupe(
        [
            f"profile:{profile}",
            "profile_result:passed" if passed else "profile_result:failed",
            *(f"score_below_threshold:{dimension}" for dimension in failed_scores),
            *(
                f"score_below_threshold:{case_dimension.split(':', 1)[1]}"
                for case_dimension in case_failed_scores
            ),
            *(
                f"case_score_below_threshold:{case_dimension}"
                for case_dimension in case_failed_scores
            ),
            *(f"hard_blocker:{blocker}" for blocker in hard_blockers),
        ]
    )
    return BilingualActorProfileResult(
        profile=profile,
        language=context["language"],
        tts_backend=context["tts_backend"],
        tts_label=context["tts_label"],
        case_count=len(selected),
        category_results=category_results,
        scores=scores,
        hard_blockers=hard_blockers,
        passed=passed,
        reason_codes=reason_codes,
    )


def build_bilingual_actor_release_gate(
    profile_results: Iterable[BilingualActorProfileResult],
    *,
    threshold: float = BILINGUAL_ACTOR_RELEASE_THRESHOLD,
) -> BilingualActorReleaseGateResult:
    """Build the strict bilingual actor release gate from profile results."""
    profiles = tuple(profile_results)
    scores = {profile.profile: dict(profile.scores) for profile in profiles}
    hard_blockers = _dedupe(blocker for profile in profiles for blocker in profile.hard_blockers)
    profile_failures: dict[str, list[str]] = {}
    for profile in profiles:
        failures = [
            f"score_below_threshold:{dimension}"
            for dimension, score in profile.scores.items()
            if float(score) < threshold
        ]
        failures.extend(
            reason_code
            for reason_code in profile.reason_codes
            if reason_code.startswith(("score_below_threshold:", "case_score_below_threshold:"))
        )
        failures.extend(
            f"category_failed:{category}"
            for category, ok in profile.category_results.items()
            if not ok
        )
        failures.extend(f"hard_blocker:{blocker}" for blocker in profile.hard_blockers)
        if profile.profile not in BILINGUAL_ACTOR_PRIMARY_PROFILES:
            failures.append("unsupported_profile")
        if failures:
            profile_failures[profile.profile] = sorted(set(failures))
    coverage = {profile.profile for profile in profiles}
    for expected_profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        if expected_profile not in coverage:
            profile_failures.setdefault(expected_profile, []).append("missing_primary_profile")
    passed = not hard_blockers and not profile_failures
    reason_codes = _dedupe(
        [
            "bilingual_actor_release_gate:v1",
            f"release_gate:{'passed' if passed else 'failed'}",
            *(f"hard_blocker:{blocker}" for blocker in hard_blockers),
            *(f"profile_failure:{profile}" for profile in sorted(profile_failures)),
        ]
    )
    return BilingualActorReleaseGateResult(
        passed=passed,
        threshold=threshold,
        hard_blockers=hard_blockers,
        scores=scores,
        profile_failures=profile_failures,
        reason_codes=reason_codes,
    )


def evaluate_bilingual_actor_bench_suite(
    cases: Iterable[BilingualActorBenchCase] | None = None,
    *,
    threshold: float = BILINGUAL_ACTOR_RELEASE_THRESHOLD,
    run_id: str = "bilingual_actor_bench_fixed",
) -> BilingualActorBenchReport:
    """Evaluate the deterministic bilingual actor bench suite."""
    selected_cases = tuple(cases) if cases is not None else build_bilingual_actor_bench_suite()
    case_results = tuple(
        evaluate_bilingual_actor_bench_case(case, threshold=threshold) for case in selected_cases
    )
    profile_results = tuple(
        _profile_result(profile, case_results, threshold=threshold)
        for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES
    )
    release_gate = build_bilingual_actor_release_gate(profile_results, threshold=threshold)
    paired_comparison = {
        "matched_pair_count": len(BILINGUAL_ACTOR_CATEGORIES),
        "categories": list(BILINGUAL_ACTOR_CATEGORIES),
        "profiles": list(BILINGUAL_ACTOR_PRIMARY_PROFILES),
        "structural_parity": all(
            {
                result.case.category
                for result in case_results
                if result.case.profile == profile
            }
            == set(BILINGUAL_ACTOR_CATEGORIES)
            for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES
        ),
    }
    return BilingualActorBenchReport(
        suite_id=BILINGUAL_ACTOR_BENCH_SUITE_ID,
        run_id=run_id,
        case_results=case_results,
        profile_results=profile_results,
        release_gate=release_gate,
        paired_comparison=paired_comparison,
    )


def render_bilingual_actor_bench_metrics_rows(
    report: BilingualActorBenchReport,
) -> tuple[dict[str, Any], ...]:
    """Return stable profile-level metric rows."""
    return tuple(result.as_dict() for result in report.profile_results)


def render_bilingual_actor_bench_human_rating_form(report: BilingualActorBenchReport) -> str:
    """Render a stable dogfooding rating form."""
    labels = {
        "state_clarity": "State clarity",
        "felt_heard": "Felt-heard",
        "voice_pacing": "Voice pacing",
        "camera_grounding": "Camera grounding",
        "memory_usefulness": "Memory usefulness",
        "interruption_naturalness": "Interruption naturalness",
        "personality_consistency": "Personality consistency",
        "enjoyment": "Enjoyment",
        "not_fake_human": "Not fake-human",
    }
    lines = [
        "# Blink Bilingual Actor Bench human rating form",
        "",
        f"- suite: `{report.suite_id}`",
        f"- threshold: `{report.release_gate.threshold:.1f}`",
        "",
        "Rate each dimension from 1 (poor) to 5 (excellent).",
        "",
    ]
    for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        lines.extend([f"## {profile}", "", "| Dimension | Score | Notes |", "| --- | ---: | --- |"])
        for dimension in BILINGUAL_ACTOR_QUALITY_DIMENSIONS:
            lines.append(f"| {labels[dimension]} |  |  |")
        lines.extend(["", "Hard failures observed:", ""])
        for blocker in BILINGUAL_ACTOR_HARD_BLOCKERS:
            lines.append(f"- [ ] {blocker.replace('_', ' ')}")
        lines.append("")
    return "\n".join(lines)


def render_bilingual_actor_bench_pairwise_form(report: BilingualActorBenchReport) -> str:
    """Render a stable zh/en pairwise comparison form."""
    lines = [
        "# Blink Bilingual Actor Bench pairwise form",
        "",
        "| category | zh/Melo/Moondream | en/Kokoro/Moondream | notes |",
        "| --- | --- | --- | --- |",
    ]
    for category in BILINGUAL_ACTOR_CATEGORIES:
        lines.append(f"| `{category}` | 1 2 3 4 5 | 1 2 3 4 5 |  |")
    lines.append("")
    lines.append(f"Release gate passed: `{str(report.release_gate.passed).lower()}`")
    return "\n".join(lines)


def render_bilingual_actor_bench_markdown(report: BilingualActorBenchReport) -> str:
    """Render a compact Markdown report."""
    lines = [
        "# Blink Bilingual Actor Bench Report",
        "",
        f"- suite: `{report.suite_id}`",
        f"- schema: `{report.schema_version}`",
        f"- passed: `{str(report.passed).lower()}`",
        f"- threshold: `{report.release_gate.threshold:.1f}`",
        "",
        "## Release Gate",
        "",
        f"- passed: `{str(report.release_gate.passed).lower()}`",
        f"- hard blockers: `{len(report.release_gate.hard_blockers)}`",
        "",
        "## Profiles",
        "",
        "| profile | passed | cases | minimum score | hard blockers |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in report.profile_results:
        minimum_score = min(result.scores.values()) if result.scores else 0.0
        lines.append(
            f"| `{result.profile}` | `{str(result.passed).lower()}` | {result.case_count} | "
            f"{minimum_score:.2f} | {len(result.hard_blockers)} |"
        )
    lines.extend(["", "## Cases", "", "| case | profile | category | passed |", "| --- | --- | --- | --- |"])
    for result in report.case_results:
        lines.append(
            f"| `{result.case.case_id}` | `{result.case.profile}` | "
            f"`{result.case.category}` | `{str(result.passed).lower()}` |"
        )
    return "\n".join(lines)


def write_bilingual_actor_bench_artifacts(
    report: BilingualActorBenchReport,
    *,
    output_dir: str | Path = BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR,
) -> dict[str, str]:
    """Write deterministic JSON, JSONL, Markdown, and rating-form artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "latest.json"
    jsonl_path = output_path / "latest.jsonl"
    markdown_path = output_path / "latest.md"
    human_form_path = output_path / "human_rating_form.md"
    pairwise_form_path = output_path / "pairwise_form.md"

    json_path.write_text(
        f"{json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    jsonl_lines = [
        json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)
        for result in report.case_results
    ]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    markdown_path.write_text(f"{render_bilingual_actor_bench_markdown(report)}\n", encoding="utf-8")
    human_form_path.write_text(
        f"{render_bilingual_actor_bench_human_rating_form(report)}\n",
        encoding="utf-8",
    )
    pairwise_form_path.write_text(
        f"{render_bilingual_actor_bench_pairwise_form(report)}\n",
        encoding="utf-8",
    )
    return {
        "human_rating_form": str(human_form_path),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
        "pairwise_form": str(pairwise_form_path),
    }


def _safe_rate(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _safe_score(value: float) -> float:
    return round(max(0.0, min(5.0, float(value))), 4)


def _v3_base_category(category: str) -> str:
    return {
        "listening": "active_listening",
        "overlap_interruption": "interruption",
        "repair": "interruption",
        "long_session_continuity": "long_session_stability",
        "preference_comparison": "memory_persona",
        "safety_controls": "connection",
    }.get(category, category)


def _v3_scores(category: str) -> dict[str, float]:
    scores = {dimension: 4.45 for dimension in BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS}
    if category == "listening":
        scores.update({"felt_heard": 4.65, "state_clarity": 4.65})
    if category == "speech":
        scores["voice_pacing"] = 4.65
    if category == "overlap_interruption":
        scores["interruption_naturalness"] = 4.55
    if category == "camera_grounding":
        scores["camera_honesty"] = 4.7
    if category == "repair":
        scores.update({"interruption_naturalness": 4.6, "state_clarity": 4.55})
    if category == "memory_persona":
        scores.update({"memory_usefulness": 4.65, "persona_consistency": 4.65})
    if category == "preference_comparison":
        scores.update({"state_clarity": 4.55, "felt_heard": 4.55})
    if category == "safety_controls":
        scores["not_fake_human"] = 4.7
    return scores


def _v3_metrics(category: str) -> dict[str, float]:
    latency = 0.0
    stale_drops = 0.0
    memory_rate = 0.0
    persona_rate = 0.0
    if category == "overlap_interruption":
        latency = 280.0
    if category == "repair":
        latency = 180.0
        stale_drops = 2.0
    if category == "memory_persona":
        memory_rate = 1.0
        persona_rate = 1.0
    if category == "preference_comparison":
        memory_rate = 0.5
        persona_rate = 0.5
    return {
        "state_clarity": 4.55 if category in {"listening", "repair"} else 4.45,
        "perceived_responsiveness_proxy": 4.5,
        "interruption_stop_latency_ms": latency,
        "stale_chunk_drops": stale_drops,
        "camera_frame_age_policy": 1.0,
        "memory_effect_rate": memory_rate,
        "persona_reference_hit_rate": persona_rate,
        "episode_sanitizer_pass_rate": 1.0,
        "bilingual_parity_delta": 0.0,
    }


def _preference_review_evidence(
    preferences_dir: str | Path = PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
) -> dict[str, Any]:
    store = PerformancePreferenceStore(preferences_dir)
    pairs = store.load_pairs(limit=12)
    proposals = store.load_proposals(limit=12)
    profile_counts: dict[str, int] = {}
    for pair in pairs:
        profile_counts[pair.profile] = profile_counts.get(pair.profile, 0) + 1
    return {
        "schema_version": 3,
        "source": "synthetic_with_optional_jsonl",
        "synthetic_pair_count": len(BILINGUAL_ACTOR_PRIMARY_PROFILES),
        "real_pair_count": len(pairs),
        "real_proposal_count": len(proposals),
        "profile_counts": dict(sorted(profile_counts.items())),
        "dimensions": list(PERFORMANCE_PREFERENCE_DIMENSIONS),
        "synthetic_evidence_refs": [
            {
                "evidence_kind": "synthetic_preference_pair",
                "evidence_id": f"synthetic-preference:{profile}",
                "summary": "Provider-free preference comparison evidence.",
                "reason_codes": ["preference_review:synthetic"],
            }
            for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES
        ],
        "real_pair_ids": [pair.pair_id for pair in pairs],
        "policy_proposal_ids": [proposal.proposal_id for proposal in proposals],
        "sanitizer_pass": True,
        "reason_codes": [
            "performance_preference_review:v3",
            "performance_preference_review:synthetic_available",
            "performance_preference_review:jsonl_optional",
        ],
    }


def _v3_actor_state(profile: str, *, category: str) -> dict[str, Any]:
    state = _actor_state(profile, category=_v3_base_category(category), mode="listening")
    context = _profile_context(profile)
    state = {
        **state,
        "performance_bench_v3": {
            "schema_version": 3,
            "category": category,
            "profile": profile,
            "reason_codes": ["bilingual_performance_bench:v3"],
        },
        "answer_claims_vision": category == "camera_grounding",
    }
    if category == "speech":
        state["mode"] = "speaking"
        state["speech"] = {
            **state["speech"],
            "speech_director_version": 3,
            "subtitle_timing": "before_or_at_playback_start",
            "chunk_budget_label": "bounded_dual_tts",
            "backend_capabilities": {
                "chunk_boundaries": True,
                "interruption_flush": True,
                "speech_rate_control": False,
                "prosody_control": False,
                "emotional_prosody": False,
                "pause_timing_control": False,
                "partial_abort": False,
                "discard": False,
                "hardware_controls": False,
            },
        }
    if category == "listening":
        state["mode"] = "listening"
        state["active_listening"] = {
            **state["active_listening"],
            "ready_to_answer": True,
            "semantic_state_v3": {
                "schema_version": 3,
                "detected_intent": "project_planning",
                "listener_chips": ["still_listening", "constraint_detected", "ready_to_answer"],
                "enough_information_to_answer": True,
                "reason_codes": ["semantic_listener:v3"],
            },
        }
    if category == "overlap_interruption":
        state["mode"] = "speaking"
        state["conversation_floor"] = {
            "schema_version": 1,
            "floor_model_version": 3,
            "state": "assistant_has_floor",
            "floor_sub_state": "ignored_backchannel",
            "yield_decision": "continue",
            "reason_codes": ["short_backchannel", "protected_playback"],
        }
        state["interruption"] = {
            **state["interruption"],
            "last_decision": "rejected",
            "barge_in_state": "protected",
            "self_interruption": False,
        }
    if category == "camera_grounding":
        state["mode"] = "looking"
        state["camera_scene"] = {
            **state["camera_scene"],
            "current_answer_used_vision": True,
            "grounding_mode": "single_frame",
            "last_used_frame_age_ms": 80,
            "scene_social_state_v2": {
                **state["camera_scene"]["scene_social_state_v2"],
                "camera_honesty_state": "can_see_now",
                "scene_transition": "vision_answered",
                "last_moondream_result_state": "answered",
                "object_showing_likelihood": 0.85,
            },
        }
    if category == "repair":
        state["mode"] = "speaking"
        state["conversation_floor"] = {
            "schema_version": 1,
            "floor_model_version": 3,
            "state": "repair",
            "floor_sub_state": "repair_requested",
            "yield_decision": "yield",
            "reason_codes": ["explicit_interrupt", "repair_requested"],
        }
        state["interruption"] = {
            **state["interruption"],
            "last_decision": "accepted",
            "barge_in_state": "adaptive",
            "echo_safe": True,
            "self_interruption": False,
        }
        state["speech"] = {
            **state["speech"],
            "stale_chunk_drop_count": 2,
            "stale_output_played_after_interruption": False,
        }
    if category == "memory_persona":
        state["memory_persona"] = {
            **state["memory_persona"],
            "selected_memory_count": 2,
            "suppressed_memory_count": 1,
            "memory_effect": "cross_language_callback",
            "performance_plan_v3": {
                "schema_version": 3,
                "plan_id": f"performance-plan-v3:{profile}",
                "memory_callback_policy": {"state": "cross_language_callback"},
                "persona_anchor_refs_v3": [
                    {"anchor_id": "persona-anchor-v3:memory_callback", "situation_key": "memory_callback"}
                ],
                "tts_capabilities": {
                    "chunk_boundaries": True,
                    "interruption_flush": True,
                    "speech_rate_control": False,
                    "prosody_control": False,
                    "emotional_prosody": False,
                    "hardware_controls": False,
                },
            },
        }
    if category == "long_session_continuity":
        state["long_session"] = {
            "turn_count": 24,
            "trace_event_count": 192,
            "unbounded_transcript_storage": False,
            "profile_switch_count": 0,
        }
    if category == "preference_comparison":
        state["preference_review"] = {
            "schema_version": 3,
            "synthetic_pair_available": True,
            "dimension_count": len(PERFORMANCE_PREFERENCE_DIMENSIONS),
            "reason_codes": ["performance_preference_review:v3"],
        }
    if category == "safety_controls":
        state["privacy_controls"] = _privacy_controls()
        state["avatar_contract"] = _avatar_contract()
    state["tts"] = {"backend": context["tts_backend"], "label": context["tts_label"]}
    return state


def _v3_events(profile: str, *, category: str) -> tuple[dict[str, Any], ...]:
    events = {
        "connection": (
            _event(1, "connected", profile=profile, mode="connected"),
            _event(2, "listening", profile=profile, mode="listening"),
        ),
        "listening": (
            _event(1, "listening_started", profile=profile),
            _event(2, "partial_understanding_updated", profile=profile),
            _event(3, "final_understanding_ready", profile=profile, mode="heard"),
        ),
        "speech": (
            _event(1, "thinking", profile=profile, mode="thinking"),
            _event(2, "speech.subtitle_ready", profile=profile, mode="speaking"),
            _event(3, "speech.tts_request_start", profile=profile, mode="speaking"),
            _event(4, "speech.audio_start", profile=profile, mode="speaking"),
        ),
        "overlap_interruption": (
            _event(1, "speaking", profile=profile, mode="speaking"),
            _event(2, "floor.overlap_candidate", profile=profile, mode="speaking"),
            _event(3, "interruption_candidate", profile=profile, mode="speaking"),
            _event(4, "interruption_rejected", profile=profile, mode="speaking"),
        ),
        "camera_grounding": (
            _event(1, "vision.fetch_user_image_start", profile=profile, mode="looking"),
            _event(2, "vision.fetch_user_image_success", profile=profile, mode="looking"),
            _event(3, "looking", profile=profile, mode="looking"),
        ),
        "repair": (
            _event(1, "speaking", profile=profile, mode="speaking"),
            _event(2, "interruption_candidate", profile=profile, mode="speaking"),
            _event(3, "interruption_accepted", profile=profile, mode="interrupted"),
            _event(4, "output_flushed", profile=profile, mode="interrupted"),
            _event(5, "interruption_recovered", profile=profile, mode="recovered"),
        ),
        "memory_persona": (
            _event(1, "memory_used", profile=profile, mode="thinking"),
            _event(2, "persona_plan_compiled", profile=profile, mode="thinking"),
        ),
        "long_session_continuity": (
            _event(1, "connected", profile=profile, mode="connected"),
            _event(2, "listening", profile=profile, mode="listening"),
            _event(3, "thinking", profile=profile, mode="thinking"),
            _event(4, "speech.audio_start", profile=profile, mode="speaking"),
        ),
        "preference_comparison": (
            _event(1, "performance.preference.recorded", profile=profile, mode="thinking"),
            _event(2, "performance.learning.policy.proposed", profile=profile, mode="thinking"),
        ),
        "safety_controls": (
            _event(1, "privacy.controls_visible", profile=profile, mode="waiting"),
            _event(2, "avatar.contract_checked", profile=profile, mode="waiting"),
        ),
    }
    return events[category]


def _v3_control_frame(profile: str, *, category: str) -> dict[str, Any]:
    context = _profile_context(profile)
    boundary = "stt_final_boundary"
    if category == "speech":
        boundary = "speech_chunk_boundary"
    elif category == "camera_grounding":
        boundary = "camera_frame_boundary"
    elif category in {"overlap_interruption", "repair"}:
        boundary = "interruption_boundary"
    return {
        "schema_version": 3,
        "frame_id": f"control-v3:{profile}:{category}",
        "sequence": 1,
        "profile": profile,
        "language": context["language"],
        "tts_runtime_label": context["tts_label"],
        "boundary": boundary,
        "source_event_ids": [1, 2],
        "floor_policy": {"state": "repair" if category == "repair" else "stable"},
        "speech_policy": {
            "state": "bounded",
            "outstanding_chunks": 0,
            "held_chunks": 0,
        },
        "camera_policy": {
            "honesty_state": "can_see_now" if category == "camera_grounding" else "available_not_used"
        },
        "active_listener_policy": {"ready_to_answer": category == "listening"},
        "reason_trace": ["actor_control_frame_v3:bench"],
    }


def _v3_plan_summary(profile: str, *, category: str) -> dict[str, Any]:
    context = _profile_context(profile)
    return {
        "schema_version": 3,
        "plan_id": f"performance-plan-v3:{profile}:{category}",
        "profile": profile,
        "language": context["language"],
        "tts_runtime_label": context["tts_label"],
        "stance": "repair_first" if category == "repair" else "grounded_and_concise",
        "response_shape": "repair_then_answer" if category == "repair" else "answer_first",
        "plan_summary": "Public-safe performance plan evidence.",
        "camera_reference_policy": {
            "state": "fresh_visual_grounding" if category == "camera_grounding" else "no_visual_claim"
        },
        "memory_callback_policy": {
            "state": "cross_language_callback" if category == "memory_persona" else "none"
        },
        "interruption_policy": {
            "state": "repair" if category == "repair" else "protected_playback_default"
        },
        "persona_anchor_refs_v3": [
            {
                "anchor_id": f"persona-anchor-v3:{category}",
                "situation_key": "memory_callback" if category == "memory_persona" else "uncertainty",
            }
        ],
        "tts_capabilities": {
            "chunk_boundaries": True,
            "interruption_flush": True,
            "speech_rate_control": False,
            "prosody_control": False,
            "emotional_prosody": False,
            "pause_timing_control": False,
            "partial_abort": False,
            "discard": False,
            "hardware_controls": False,
        },
        "reason_trace": ["performance_plan_v3:bench"],
    }


@dataclass(frozen=True)
class BilingualPerformanceBenchV3Case:
    """One deterministic Phase 12 performance bench case."""

    case_id: str
    pair_id: str
    profile: str
    category: str
    title: str
    actor_state: dict[str, Any]
    actor_events: tuple[dict[str, Any], ...]
    control_frames: tuple[dict[str, Any], ...]
    plan_summaries: tuple[dict[str, Any], ...]
    quality_scores: dict[str, float]
    metrics: dict[str, float]
    preference_evidence: dict[str, Any] = field(default_factory=dict)
    privacy_controls: dict[str, Any] = field(default_factory=_privacy_controls)
    avatar_contract: dict[str, Any] = field(default_factory=_avatar_contract)
    reason_codes: tuple[str, ...] = ()

    @property
    def language(self) -> str:
        """Return the case language."""
        return _profile_context(self.profile)["language"]

    @property
    def tts_label(self) -> str:
        """Return the public TTS label."""
        return _profile_context(self.profile)["tts_label"]

    def as_dict(self) -> dict[str, Any]:
        """Serialize public case metadata."""
        return {
            "schema_version": BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION,
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "profile": self.profile,
            "language": self.language,
            "tts_label": self.tts_label,
            "category": self.category,
            "title": self.title,
            "event_count": len(self.actor_events),
            "control_frame_count": len(self.control_frames),
            "plan_summary_count": len(self.plan_summaries),
            "score_dimensions": list(BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS),
            "metric_names": list(BILINGUAL_PERFORMANCE_V3_METRICS),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualPerformanceBenchV3CaseResult:
    """Evaluated Phase 12 performance bench case."""

    case: BilingualPerformanceBenchV3Case
    passed: bool
    checks: tuple[BilingualActorBenchCheckResult, ...]
    scores: dict[str, float]
    metrics: dict[str, float]
    hard_blockers: tuple[str, ...]
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the V3 case result."""
        return {
            "case": self.case.as_dict(),
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "scores": dict(sorted(self.scores.items())),
            "metrics": dict(sorted(self.metrics.items())),
            "hard_blockers": list(self.hard_blockers),
            "evidence": _json_safe(self.evidence),
        }


@dataclass(frozen=True)
class BilingualPerformanceProfileResultV3:
    """Per-profile V3 aggregate result."""

    profile: str
    language: str
    tts_label: str
    case_count: int
    category_results: dict[str, bool]
    scores: dict[str, float]
    metrics: dict[str, float]
    hard_blockers: tuple[str, ...]
    passed: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the V3 profile result."""
        return {
            "profile": self.profile,
            "language": self.language,
            "tts_label": self.tts_label,
            "case_count": self.case_count,
            "category_results": dict(sorted(self.category_results.items())),
            "scores": dict(sorted(self.scores.items())),
            "metrics": dict(sorted(self.metrics.items())),
            "hard_blockers": list(self.hard_blockers),
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualPerformanceReleaseGateV3:
    """Strict Phase 12 release gate."""

    passed: bool
    threshold: float
    parity_delta_threshold: float
    hard_blockers: tuple[str, ...]
    profile_failures: dict[str, list[str]]
    parity_deltas: dict[str, float]
    reason_codes: tuple[str, ...]
    schema_version: int = BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Serialize the V3 release gate."""
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "threshold": self.threshold,
            "parity_delta_threshold": self.parity_delta_threshold,
            "hard_blockers": list(self.hard_blockers),
            "profile_failures": _json_safe(self.profile_failures),
            "parity_deltas": dict(sorted(self.parity_deltas.items())),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BilingualPerformanceBenchV3Report:
    """Full Phase 12 performance bench report."""

    suite_id: str
    run_id: str
    case_results: tuple[BilingualPerformanceBenchV3CaseResult, ...]
    profile_results: tuple[BilingualPerformanceProfileResultV3, ...]
    release_gate: BilingualPerformanceReleaseGateV3
    preference_review: dict[str, Any]
    avatar_contract_evidence: dict[str, Any]
    paired_comparison: dict[str, Any]
    aggregate_metrics: dict[str, float]
    alive_performance_summary: str
    generated_at: str = _FIXED_TS
    artifact_links: dict[str, str] = field(default_factory=dict)
    schema_version: int = BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """Return whether the V3 release gate passed."""
        return self.release_gate.passed

    def as_dict(self) -> dict[str, Any]:
        """Serialize the V3 report."""
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "profiles": list(BILINGUAL_ACTOR_PRIMARY_PROFILES),
            "passed": self.passed,
            "threshold": self.release_gate.threshold,
            "case_results": [result.as_dict() for result in self.case_results],
            "profile_results": [result.as_dict() for result in self.profile_results],
            "release_gate": self.release_gate.as_dict(),
            "hard_blockers": list(self.release_gate.hard_blockers),
            "scores": {
                result.profile: dict(sorted(result.scores.items()))
                for result in self.profile_results
            },
            "aggregate_metrics": dict(sorted(self.aggregate_metrics.items())),
            "paired_comparison": _json_safe(self.paired_comparison),
            "preference_review": _json_safe(self.preference_review),
            "avatar_contract_evidence": _json_safe(self.avatar_contract_evidence),
            "alive_performance_summary": self.alive_performance_summary,
            "artifact_links": dict(sorted(self.artifact_links.items())),
        }


def build_bilingual_performance_bench_v3_suite(
    *,
    preferences_dir: str | Path = PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
) -> tuple[BilingualPerformanceBenchV3Case, ...]:
    """Build matched provider-free Phase 12 cases for both primary browser paths."""
    preference_evidence = _preference_review_evidence(preferences_dir)
    titles = {
        "connection": "Primary browser path connects with state visible",
        "listening": "Semantic listener makes long turns feel heard",
        "speech": "Dual TTS speech is chunked, subtitled, and interruptible",
        "overlap_interruption": "Backchannels do not kill protected playback",
        "camera_grounding": "Moondream grounding is fresh and honest",
        "repair": "Accepted interruption drops stale output and repairs",
        "memory_persona": "Memory/persona anchors visibly affect behavior",
        "long_session_continuity": "Long session continuity remains bounded",
        "preference_comparison": "Preference review creates structured evidence",
        "safety_controls": "Privacy and avatar safety controls are visible",
    }
    cases: list[BilingualPerformanceBenchV3Case] = []
    for category in BILINGUAL_PERFORMANCE_V3_CATEGORIES:
        for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
            locale = "zh" if profile == "browser-zh-melo" else "en"
            cases.append(
                BilingualPerformanceBenchV3Case(
                    case_id=f"{locale}_{category}_v3",
                    pair_id=category,
                    profile=profile,
                    category=category,
                    title=titles[category],
                    actor_state=_v3_actor_state(profile, category=category),
                    actor_events=_v3_events(profile, category=category),
                    control_frames=(_v3_control_frame(profile, category=category),),
                    plan_summaries=(_v3_plan_summary(profile, category=category),),
                    quality_scores=_v3_scores(category),
                    metrics=_v3_metrics(category),
                    preference_evidence=preference_evidence if category == "preference_comparison" else {},
                    reason_codes=(f"profile:{profile}", f"performance_category:{category}"),
                )
            )
    return tuple(cases)


_UNSUPPORTED_TTS_CAPABILITY_KEYS = {
    "speech_rate",
    "speech_rate_control",
    "prosody",
    "prosody_control",
    "emotional_prosody",
    "emphasis_control",
    "pause_timing",
    "pause_timing_control",
    "partial_abort",
    "stream_abort",
    "discard",
    "hardware_controls",
}


def _unsupported_tts_claims(value: Any) -> tuple[str, ...]:
    claims: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            if key in _UNSUPPORTED_TTS_CAPABILITY_KEYS and nested is True:
                claims.append(key)
            claims.extend(_unsupported_tts_claims(nested))
    elif isinstance(value, tuple | list):
        for nested in value:
            claims.extend(_unsupported_tts_claims(nested))
    return _dedupe(claims, limit=24)


def _v3_category_ok(case: BilingualPerformanceBenchV3Case) -> bool:
    state = case.actor_state
    events = _event_types(case.actor_events)
    if case.category == "connection":
        return "connected" in events and "listening" in events and _profile_defaults_ok(
            BilingualActorBenchCase(
                case_id=case.case_id,
                pair_id=case.pair_id,
                profile=case.profile,
                category="connection",
                title=case.title,
                actor_state=state,
                actor_events=case.actor_events,
                quality_scores={dimension: 5.0 for dimension in BILINGUAL_ACTOR_QUALITY_DIMENSIONS},
            )
        )
    if case.category == "listening":
        return (
            "partial_understanding_updated" in events
            and _state_path(state, "active_listening.semantic_state_v3.schema_version") == 3
            and _state_path(state, "active_listening.ready_to_answer") is True
        )
    if case.category == "speech":
        return (
            _contains_order(events, ("speech.subtitle_ready", "speech.tts_request_start", "speech.audio_start"))
            and _state_path(state, "speech.subtitle_timing") == "before_or_at_playback_start"
            and _state_path(state, "speech.backend_capabilities.chunk_boundaries") is True
        )
    if case.category == "overlap_interruption":
        return (
            "interruption_candidate" in events
            and "interruption_rejected" in events
            and _state_path(state, "conversation_floor.floor_sub_state") == "ignored_backchannel"
            and _state_path(state, "interruption.self_interruption") is False
        )
    if case.category == "camera_grounding":
        return (
            "vision.fetch_user_image_success" in events
            and _state_path(state, "camera_scene.current_answer_used_vision") is True
            and _state_path(state, "camera_scene.scene_social_state_v2.camera_honesty_state")
            == "can_see_now"
            and float(_state_path(state, "camera_scene.last_used_frame_age_ms", 9999)) <= 500
        )
    if case.category == "repair":
        return (
            _contains_order(events, ("interruption_accepted", "output_flushed", "interruption_recovered"))
            and int(_state_path(state, "speech.stale_chunk_drop_count", 0)) >= 1
            and _state_path(state, "speech.stale_output_played_after_interruption") is False
        )
    if case.category == "memory_persona":
        return (
            "memory_used" in events
            and "persona_plan_compiled" in events
            and int(_state_path(state, "memory_persona.selected_memory_count", 0)) >= 1
            and _state_path(state, "memory_persona.performance_plan_v3.schema_version") == 3
        )
    if case.category == "long_session_continuity":
        return (
            int(_state_path(state, "long_session.turn_count", 0)) >= 12
            and int(_state_path(state, "long_session.trace_event_count", 0)) <= 10000
            and _state_path(state, "long_session.unbounded_transcript_storage") is False
        )
    if case.category == "preference_comparison":
        return (
            case.preference_evidence.get("synthetic_pair_count") >= 2
            and case.preference_evidence.get("sanitizer_pass") is True
            and set(PERFORMANCE_PREFERENCE_DIMENSIONS).issubset(
                set(case.preference_evidence.get("dimensions") or ())
            )
        )
    if case.category == "safety_controls":
        return _privacy_controls_ok(case.privacy_controls) and _avatar_contract_ok(
            case.avatar_contract
        )
    return False


def _v3_hard_blockers_for_case(case: BilingualPerformanceBenchV3Case) -> tuple[str, ...]:
    state = case.actor_state
    events = _event_types(case.actor_events)
    blockers: list[str] = []
    if find_bilingual_actor_public_safety_violations(
        {
            "actor_state": state,
            "actor_events": case.actor_events,
            "control_frames": case.control_frames,
            "plan_summaries": case.plan_summaries,
            "preference_evidence": case.preference_evidence,
            "privacy_controls": case.privacy_controls,
            "avatar_contract": case.avatar_contract,
        }
    ):
        blockers.append("unsafe_trace_payload")
    compat_case = BilingualActorBenchCase(
        case_id=case.case_id,
        pair_id=case.pair_id,
        profile=case.profile,
        category=_v3_base_category(case.category),
        title=case.title,
        actor_state=state,
        actor_events=case.actor_events,
        quality_scores={dimension: 5.0 for dimension in BILINGUAL_ACTOR_QUALITY_DIMENSIONS},
        privacy_controls=case.privacy_controls,
        avatar_contract=case.avatar_contract,
    )
    if not _profile_defaults_ok(compat_case):
        blockers.append("profile_regression")
    if (
        _state_path(state, "camera_scene.current_answer_used_vision") is True
        and "vision.fetch_user_image_success" not in events
    ):
        blockers.append("hidden_camera_use")
    if (
        state.get("answer_claims_vision") is True
        and _state_path(state, "camera_scene.current_answer_used_vision") is not True
    ):
        blockers.append("false_camera_claim")
    if (
        _state_path(state, "camera_scene.scene_social_state_v2.camera_honesty_state")
        == "can_see_now"
        and (
            _state_path(state, "camera_scene.current_answer_used_vision") is not True
            or "vision.fetch_user_image_success" not in events
        )
    ):
        blockers.append("false_camera_claim")
    if (
        _state_path(state, "interruption.self_interruption") is True
        or (
            _state_path(state, "interruption.last_decision") == "accepted"
            and _state_path(state, "interruption.echo_safe") is not True
            and state.get("protected_playback") is True
        )
    ):
        blockers.append("self_interruption")
    if _state_path(state, "speech.stale_output_played_after_interruption") is True:
        blockers.append("stale_tts_after_interrupt")
    if _state_path(state, "memory_persona.memory_contradiction") is True:
        blockers.append("memory_contradiction")
    if _unsupported_tts_claims({"state": state, "plans": case.plan_summaries}):
        blockers.append("unsupported_tts_claim")
    if not _privacy_controls_ok(case.privacy_controls):
        blockers.append("missing_consent_controls")
    if not _avatar_contract_ok(case.avatar_contract):
        blockers.append("realistic_human_avatar_capability")
    return _dedupe(blockers, limit=len(BILINGUAL_PERFORMANCE_V3_HARD_BLOCKERS))


def evaluate_bilingual_performance_bench_v3_case(
    case: BilingualPerformanceBenchV3Case,
    *,
    threshold: float = BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD,
) -> BilingualPerformanceBenchV3CaseResult:
    """Evaluate one Phase 12 case."""
    blockers = _v3_hard_blockers_for_case(case)
    category_ok = _v3_category_ok(case)
    scores_ok = all(float(score) >= threshold for score in case.quality_scores.values())
    metrics_ok = (
        float(case.metrics.get("episode_sanitizer_pass_rate", 0.0)) >= 1.0
        and float(case.metrics.get("camera_frame_age_policy", 0.0)) >= 1.0
        and float(case.metrics.get("interruption_stop_latency_ms", 0.0)) <= 500.0
    )
    checks = (
        _check("public_safety", "unsafe_trace_payload" not in blockers, "No private payloads.", "public_safety:ok"),
        _check("profile_defaults", "profile_regression" not in blockers, "Primary browser defaults held.", "profile_defaults:ok"),
        _check("category_contract", category_ok, f"{case.category} V3 contract passes.", f"category:{case.category}"),
        _check("metric_bounds", metrics_ok, "Performance metrics are bounded and release-safe.", "metrics:bounded"),
        _check("score_threshold", scores_ok, f"Scores meet threshold {threshold:.1f}.", "scores:threshold"),
        _check("avatar_boundary", "realistic_human_avatar_capability" not in blockers, "Avatar contract remains non-human.", "avatar_boundary:ok"),
    )
    passed = not blockers and category_ok and scores_ok and metrics_ok
    evidence = {
        "profile": case.profile,
        "language": case.language,
        "category": case.category,
        "event_types": _event_types(case.actor_events),
        "control_frame_count": len(case.control_frames),
        "plan_summary_count": len(case.plan_summaries),
        "preference_real_pair_count": int(case.preference_evidence.get("real_pair_count", 0) or 0),
        "hard_blocker_count": len(blockers),
    }
    return BilingualPerformanceBenchV3CaseResult(
        case=case,
        passed=passed,
        checks=checks,
        scores={key: _safe_score(value) for key, value in case.quality_scores.items()},
        metrics={key: round(float(value), 4) for key, value in case.metrics.items()},
        hard_blockers=blockers,
        evidence=evidence,
    )


def _v3_profile_result(
    profile: str,
    results: tuple[BilingualPerformanceBenchV3CaseResult, ...],
    *,
    threshold: float,
) -> BilingualPerformanceProfileResultV3:
    context = _profile_context(profile)
    selected = tuple(result for result in results if result.case.profile == profile)
    scores = {
        dimension: _safe_score(
            sum(float(result.scores[dimension]) for result in selected) / len(selected)
        )
        if selected
        else 0.0
        for dimension in BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS
    }
    metrics = {
        metric: round(sum(float(result.metrics.get(metric, 0.0)) for result in selected) / len(selected), 4)
        if selected
        else 0.0
        for metric in BILINGUAL_PERFORMANCE_V3_METRICS
    }
    category_results = {
        category: all(
            any(check.check_id == "category_contract" and check.passed for check in result.checks)
            for result in selected
            if result.case.category == category
        )
        for category in BILINGUAL_PERFORMANCE_V3_CATEGORIES
    }
    hard_blockers = _dedupe(blocker for result in selected for blocker in result.hard_blockers)
    failed_scores = tuple(
        dimension for dimension, score in scores.items() if float(score) < threshold
    )
    passed = bool(selected) and all(category_results.values()) and not hard_blockers and not failed_scores
    reason_codes = _dedupe(
        (
            f"profile:{profile}",
            "performance_profile_result:passed" if passed else "performance_profile_result:failed",
            *(f"score_below_threshold:{dimension}" for dimension in failed_scores),
            *(f"hard_blocker:{blocker}" for blocker in hard_blockers),
        )
    )
    return BilingualPerformanceProfileResultV3(
        profile=profile,
        language=context["language"],
        tts_label=context["tts_label"],
        case_count=len(selected),
        category_results=category_results,
        scores=scores,
        metrics=metrics,
        hard_blockers=hard_blockers,
        passed=passed,
        reason_codes=reason_codes,
    )


def _v3_parity_deltas(
    profile_results: tuple[BilingualPerformanceProfileResultV3, ...],
) -> dict[str, float]:
    by_profile = {result.profile: result for result in profile_results}
    zh = by_profile.get("browser-zh-melo")
    en = by_profile.get("browser-en-kokoro")
    if zh is None or en is None:
        return {dimension: 5.0 for dimension in BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS}
    deltas = {
        dimension: round(abs(float(zh.scores[dimension]) - float(en.scores[dimension])), 4)
        for dimension in BILINGUAL_PERFORMANCE_V3_SCORE_DIMENSIONS
    }
    deltas["bilingual_parity_delta"] = max(deltas.values()) if deltas else 0.0
    return deltas


def build_bilingual_performance_release_gate_v3(
    profile_results: Iterable[BilingualPerformanceProfileResultV3],
    *,
    threshold: float = BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD,
    parity_delta_threshold: float = BILINGUAL_PERFORMANCE_PARITY_DELTA_THRESHOLD,
) -> BilingualPerformanceReleaseGateV3:
    """Build the strict V3 release gate from profile results."""
    profiles = tuple(profile_results)
    hard_blockers = _dedupe(blocker for profile in profiles for blocker in profile.hard_blockers)
    parity_deltas = _v3_parity_deltas(profiles)
    parity_failed = parity_deltas.get("bilingual_parity_delta", 5.0) > parity_delta_threshold
    profile_failures: dict[str, list[str]] = {}
    for profile in profiles:
        failures = [
            f"score_below_threshold:{dimension}"
            for dimension, score in profile.scores.items()
            if float(score) < threshold
        ]
        failures.extend(
            f"category_failed:{category}"
            for category, ok in profile.category_results.items()
            if not ok
        )
        failures.extend(f"hard_blocker:{blocker}" for blocker in profile.hard_blockers)
        if parity_failed:
            failures.append("bilingual_parity_delta_exceeded")
        if failures:
            profile_failures[profile.profile] = sorted(set(failures))
    coverage = {profile.profile for profile in profiles}
    for expected_profile in BILINGUAL_ACTOR_PRIMARY_PROFILES:
        if expected_profile not in coverage:
            profile_failures.setdefault(expected_profile, []).append("missing_primary_profile")
    passed = not hard_blockers and not profile_failures and not parity_failed
    reason_codes = _dedupe(
        (
            "bilingual_performance_release_gate:v3",
            f"release_gate:{'passed' if passed else 'failed'}",
            *(f"hard_blocker:{blocker}" for blocker in hard_blockers),
            "bilingual_parity:failed" if parity_failed else "bilingual_parity:passed",
        )
    )
    return BilingualPerformanceReleaseGateV3(
        passed=passed,
        threshold=threshold,
        parity_delta_threshold=parity_delta_threshold,
        hard_blockers=hard_blockers,
        profile_failures=profile_failures,
        parity_deltas=parity_deltas,
        reason_codes=reason_codes,
    )


def evaluate_bilingual_performance_bench_v3(
    cases: Iterable[BilingualPerformanceBenchV3Case] | None = None,
    *,
    threshold: float = BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD,
    preferences_dir: str | Path = PERFORMANCE_PREFERENCE_ARTIFACT_DIR,
    run_id: str = "bilingual_performance_bench_v3_fixed",
) -> BilingualPerformanceBenchV3Report:
    """Evaluate the provider-free Phase 12 bilingual performance bench."""
    selected_cases = (
        tuple(cases)
        if cases is not None
        else build_bilingual_performance_bench_v3_suite(preferences_dir=preferences_dir)
    )
    case_results = tuple(
        evaluate_bilingual_performance_bench_v3_case(case, threshold=threshold)
        for case in selected_cases
    )
    profile_results = tuple(
        _v3_profile_result(profile, case_results, threshold=threshold)
        for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES
    )
    release_gate = build_bilingual_performance_release_gate_v3(
        profile_results,
        threshold=threshold,
    )
    paired_comparison = {
        "matched_pair_count": len(BILINGUAL_PERFORMANCE_V3_CATEGORIES),
        "categories": list(BILINGUAL_PERFORMANCE_V3_CATEGORIES),
        "profiles": list(BILINGUAL_ACTOR_PRIMARY_PROFILES),
        "structural_parity": all(
            {
                result.case.category
                for result in case_results
                if result.case.profile == profile
            }
            == set(BILINGUAL_PERFORMANCE_V3_CATEGORIES)
            for profile in BILINGUAL_ACTOR_PRIMARY_PROFILES
        ),
        "bilingual_parity_delta": release_gate.parity_deltas.get("bilingual_parity_delta", 0.0),
    }
    aggregate_metrics: dict[str, float] = {}
    for metric in BILINGUAL_PERFORMANCE_V3_METRICS:
        values = [float(result.metrics.get(metric, 0.0)) for result in profile_results]
        aggregate_metrics[metric] = round(sum(values) / len(values), 4) if values else 0.0
    aggregate_metrics["bilingual_parity_delta"] = paired_comparison["bilingual_parity_delta"]
    preference_review = _preference_review_evidence(preferences_dir)
    avatar_contract_evidence = {
        "schema_version": 3,
        "allowed_surfaces": ["abstract_avatar", "status_avatar", "symbolic_avatar", "debug_status"],
        "forbidden_capabilities": list(AVATAR_ADAPTER_FORBIDDEN_CAPABILITIES),
        "raw_media_allowed": False,
        "reason_codes": ["avatar_adapter_contract:v3", "avatar_boundary:non_human"],
    }
    alive_summary = (
        "Both primary browser paths pass perceived performance gates with bounded listening, "
        "speech, camera honesty, repair, memory/persona continuity, preference review, and "
        "avatar-safe evidence."
        if release_gate.passed
        else "One or more primary browser paths failed perceived performance release gates."
    )
    return BilingualPerformanceBenchV3Report(
        suite_id=BILINGUAL_PERFORMANCE_BENCH_V3_SUITE_ID,
        run_id=run_id,
        case_results=case_results,
        profile_results=profile_results,
        release_gate=release_gate,
        preference_review=preference_review,
        avatar_contract_evidence=avatar_contract_evidence,
        paired_comparison=paired_comparison,
        aggregate_metrics=aggregate_metrics,
        alive_performance_summary=alive_summary,
    )


def render_bilingual_performance_bench_v3_markdown(
    report: BilingualPerformanceBenchV3Report,
) -> str:
    """Render a compact Phase 12 release report."""
    lines = [
        "# Blink Bilingual Performance Bench V3",
        "",
        f"- suite: `{report.suite_id}`",
        f"- schema: `{report.schema_version}`",
        f"- passed: `{str(report.passed).lower()}`",
        f"- alive summary: {report.alive_performance_summary}",
        "",
        "## Release Gate",
        "",
        f"- hard blockers: `{len(report.release_gate.hard_blockers)}`",
        f"- bilingual parity delta: `{report.paired_comparison['bilingual_parity_delta']:.3f}`",
        "",
        "## Profiles",
        "",
        "| profile | passed | cases | min score | hard blockers |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in report.profile_results:
        minimum_score = min(result.scores.values()) if result.scores else 0.0
        lines.append(
            f"| `{result.profile}` | `{str(result.passed).lower()}` | {result.case_count} | "
            f"{minimum_score:.2f} | {len(result.hard_blockers)} |"
        )
    lines.extend(["", "## Cases", "", "| case | profile | category | passed |", "| --- | --- | --- | --- |"])
    for result in report.case_results:
        lines.append(
            f"| `{result.case.case_id}` | `{result.case.profile}` | "
            f"`{result.case.category}` | `{str(result.passed).lower()}` |"
        )
    return "\n".join(lines)


def render_bilingual_performance_release_checklist_v3(
    report: BilingualPerformanceBenchV3Report,
) -> str:
    """Render the Phase 12 manual merge checklist."""
    return "\n".join(
        [
            "# Bilingual Performance Release Checklist V3",
            "",
            f"- [x] Provider-free V3 bench passed: `{str(report.passed).lower()}`",
            "- [ ] Run browser workflow tests with runner/webrtc extras",
            "- [ ] Review latest PerformanceEpisodeV3 replay and sanitizer output",
            "- [ ] Review local PerformancePreferencePair JSONL when available",
            "- [ ] Confirm avatar adapter contract remains abstract/status/symbolic only",
            "- [ ] Complete five-minute zh/Melo dogfooding conversation",
            "- [ ] Complete five-minute en/Kokoro dogfooding conversation",
            "",
            f"Alive performance summary: {report.alive_performance_summary}",
        ]
    )


def write_bilingual_performance_bench_v3_artifacts(
    report: BilingualPerformanceBenchV3Report,
    *,
    output_dir: str | Path = BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR,
) -> dict[str, str]:
    """Write V3 JSON, JSONL, Markdown, and release-checklist artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "latest_v3.json"
    jsonl_path = output_path / "latest_v3.jsonl"
    markdown_path = output_path / "latest_v3.md"
    checklist_path = output_path / "release_checklist_v3.md"
    json_path.write_text(
        f"{json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    jsonl_lines = [
        json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)
        for result in report.case_results
    ]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    markdown_path.write_text(
        f"{render_bilingual_performance_bench_v3_markdown(report)}\n",
        encoding="utf-8",
    )
    checklist_path.write_text(
        f"{render_bilingual_performance_release_checklist_v3(report)}\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
        "release_checklist": str(checklist_path),
    }


__all__ = [
    "BILINGUAL_ACTOR_BENCH_ARTIFACT_DIR",
    "BILINGUAL_ACTOR_BENCH_FIXTURE_DIR",
    "BILINGUAL_ACTOR_BENCH_SCHEMA_VERSION",
    "BILINGUAL_ACTOR_BENCH_SUITE_ID",
    "BILINGUAL_ACTOR_CATEGORIES",
    "BILINGUAL_ACTOR_HARD_BLOCKERS",
    "BILINGUAL_ACTOR_PRIMARY_PROFILES",
    "BILINGUAL_ACTOR_QUALITY_DIMENSIONS",
    "BILINGUAL_ACTOR_REGRESSION_FIXTURE_FILES",
    "BILINGUAL_ACTOR_RELEASE_THRESHOLD",
    "BILINGUAL_PERFORMANCE_BENCH_V3_CATEGORIES",
    "BILINGUAL_PERFORMANCE_BENCH_V3_HARD_BLOCKERS",
    "BILINGUAL_PERFORMANCE_BENCH_V3_METRICS",
    "BILINGUAL_PERFORMANCE_BENCH_V3_SCHEMA_VERSION",
    "BILINGUAL_PERFORMANCE_BENCH_V3_SCORE_DIMENSIONS",
    "BILINGUAL_PERFORMANCE_BENCH_V3_SUITE_ID",
    "BILINGUAL_PERFORMANCE_PARITY_DELTA_THRESHOLD",
    "BILINGUAL_PERFORMANCE_RELEASE_THRESHOLD",
    "BilingualActorBenchCase",
    "BilingualActorBenchCaseResult",
    "BilingualActorBenchCheckResult",
    "BilingualActorBenchReport",
    "BilingualActorHistoricalRegressionResult",
    "BilingualActorProfileResult",
    "BilingualActorReleaseGateResult",
    "BilingualPerformanceBenchV3Case",
    "BilingualPerformanceBenchV3CaseResult",
    "BilingualPerformanceBenchV3Report",
    "BilingualPerformanceProfileResultV3",
    "BilingualPerformanceReleaseGateV3",
    "build_bilingual_actor_release_gate",
    "build_bilingual_actor_bench_suite",
    "build_bilingual_performance_bench_v3_suite",
    "build_bilingual_performance_release_gate_v3",
    "evaluate_bilingual_actor_bench_case",
    "evaluate_bilingual_actor_bench_suite",
    "evaluate_bilingual_performance_bench_v3",
    "evaluate_bilingual_performance_bench_v3_case",
    "find_bilingual_actor_public_safety_violations",
    "load_bilingual_actor_historical_regression_results",
    "render_bilingual_actor_bench_human_rating_form",
    "render_bilingual_actor_bench_markdown",
    "render_bilingual_actor_bench_metrics_rows",
    "render_bilingual_actor_bench_pairwise_form",
    "render_bilingual_performance_bench_v3_markdown",
    "render_bilingual_performance_release_checklist_v3",
    "write_bilingual_actor_bench_artifacts",
    "write_bilingual_performance_bench_v3_artifacts",
]

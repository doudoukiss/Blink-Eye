"""Deterministic browser/WebRTC performance bench for Blink."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

BROWSER_PERF_BENCH_SUITE_ID = "browser_perf_bench/v1"
BROWSER_PERF_BENCH_SCHEMA_VERSION = 1
BROWSER_PERF_BENCH_ARTIFACT_DIR = Path("artifacts/browser-perf-bench")
BROWSER_PERF_BENCH_PROFILES = ("browser-zh-melo", "browser-en-kokoro")
BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES = (
    "native-en-kokoro",
    "native-en-kokoro-macos-camera",
)
BROWSER_PERF_BENCH_CATEGORIES = (
    "connection",
    "active_listening",
    "speech",
    "camera",
    "interruption",
    "memory_persona",
    "recovery",
    "profile_parity",
    "native_guardrail",
)
BROWSER_PERF_BENCH_METRICS = (
    "event_order_score",
    "state_freshness_score",
    "subtitle_score",
    "tts_latency_score",
    "camera_grounding_score",
    "barge_in_safety_score",
    "memory_persona_score",
    "recovery_score",
    "holistic_signal_score",
)

_FIXED_TS = "2026-04-27T00:00:00+00:00"
_BANNED_KEY_FRAGMENTS = (
    "audio_bytes",
    "candidate",
    "credential",
    "device_label",
    "exception",
    "ice",
    "image",
    "memory_body",
    "messages",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "token",
    "transcript",
)
_SAFE_KEY_EXCEPTIONS = {
    "active_listening",
    "active_session_id",
    "active_client_id",
    "client_id",
    "event_id",
    "event_type",
    "final_transcript_chars",
    "last_event_id",
    "latest_event_id",
    "partial_transcript_available",
    "partial_transcript_chars",
    "session_id",
}
_BANNED_VALUE_TOKENS = (
    "[BLINK_BRAIN_CONTEXT]",
    "Traceback",
    "RuntimeError:",
    "data:image",
    "a=candidate",
    "v=0\\r\\n",
    "/tmp/brain.db",
    "sk-",
)
_HUMAN_RATING_LABELS = (
    "state clarity",
    "felt-heard",
    "voice pacing",
    "memory usefulness",
    "camera grounding",
    "interruption naturalness",
    "enjoyment",
)
_CASE_METRICS = {
    "connection": ("event_order_score", "state_freshness_score"),
    "active_listening": ("event_order_score", "state_freshness_score"),
    "speech": ("event_order_score", "subtitle_score", "tts_latency_score"),
    "camera": ("event_order_score", "camera_grounding_score"),
    "interruption": ("event_order_score", "barge_in_safety_score"),
    "memory_persona": ("memory_persona_score",),
    "recovery": ("event_order_score", "recovery_score"),
    "profile_parity": ("state_freshness_score",),
    "native_guardrail": ("barge_in_safety_score", "state_freshness_score"),
}
_CHECK_METRICS = {
    "event_order": ("event_order_score",),
    "state_freshness": ("state_freshness_score",),
    "profile_defaults": ("state_freshness_score",),
    "subtitle_readiness": ("subtitle_score",),
    "tts_latency": ("tts_latency_score",),
    "camera_grounding": ("camera_grounding_score",),
    "barge_in_safety": ("barge_in_safety_score",),
    "memory_persona": ("memory_persona_score",),
    "native_guardrail": ("barge_in_safety_score", "state_freshness_score"),
    "recovery": ("recovery_score",),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_json_safe(nested) for nested in value]
    return value


def _dedupe(values: Iterable[Any], *, limit: int = 48) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text[:96])
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


def _public_safety_violations(value: Any, *, path: str = "$") -> tuple[str, ...]:
    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            lowered = key.lower()
            nested_path = f"{path}.{key}"
            if lowered not in _SAFE_KEY_EXCEPTIONS and any(
                fragment in lowered for fragment in _BANNED_KEY_FRAGMENTS
            ):
                violations.append(f"unsafe_key:{nested_path}")
            violations.extend(_public_safety_violations(nested, path=nested_path))
    elif isinstance(value, tuple | list):
        for index, nested in enumerate(value):
            violations.extend(_public_safety_violations(nested, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for token in _BANNED_VALUE_TOKENS:
            if token in value:
                violations.append(f"unsafe_value:{path}:{token}")
    return tuple(violations)


def find_browser_perf_public_safety_violations(payload: Any) -> tuple[str, ...]:
    """Return public-safety violations in a trace payload."""
    return _dedupe(_public_safety_violations(payload), limit=64)


def _event(
    event_id: int,
    event_type: str,
    *,
    mode: str = "listening",
    source: str = "browser_runtime",
    metadata: dict[str, Any] | None = None,
    reason_codes: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": event_id,
        "event_type": event_type,
        "source": source,
        "mode": mode,
        "timestamp": _FIXED_TS,
        "session_id": "bench_session",
        "client_id": "bench_client",
        "metadata": dict(metadata or {}),
        "reason_codes": list(reason_codes),
    }


def _state(
    *,
    profile: str,
    mode: str = "listening",
    last_event_id: int = 0,
    speech: dict[str, Any] | None = None,
    active_listening: dict[str, Any] | None = None,
    camera_presence: dict[str, Any] | None = None,
    camera_scene: dict[str, Any] | None = None,
    memory_persona: dict[str, Any] | None = None,
    interruption: dict[str, Any] | None = None,
    browser_media: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_zh_melo = profile == "browser-zh-melo"
    default_camera_presence = {
        "state": "available",
        "enabled": True,
        "connected": True,
        "fresh": True,
        "track_state": "live",
        "latest_frame_sequence": 7,
        "latest_frame_age_ms": 90,
        "last_vision_result_state": "idle",
        "current_answer_used_vision": False,
        "grounding_mode": "none",
        "reason_codes": ["camera_presence:available"],
    }
    default_camera_scene = {
        "schema_version": 1,
        "profile": profile,
        "language": "zh" if is_zh_melo else "en",
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
        "latest_frame_sequence": 7,
        "latest_frame_age_ms": 90,
        "latest_frame_received_at": _FIXED_TS,
        "on_demand_vision_state": "idle",
        "current_answer_used_vision": False,
        "grounding_mode": "none",
        "last_vision_result_state": "idle",
        "last_used_frame_sequence": None,
        "last_used_frame_age_ms": None,
        "last_used_frame_at": None,
        "degradation": {
            "state": "ok",
            "components": [],
            "reason_codes": ["camera_scene_degradation:ok"],
        },
        "scene_social_state_v2": {
            "schema_version": 2,
            "state_id": f"scene-social-v2-{profile}",
            "profile": profile,
            "language": "zh" if is_zh_melo else "en",
            "camera_status": "ready",
            "vision_status": "idle",
            "camera_honesty_state": "recent_frame_available",
            "frame_freshness": "fresh",
            "frame_age_ms": 90,
            "latest_frame_sequence": 7,
            "last_used_frame_sequence": None,
            "last_used_frame_age_ms": None,
            "user_presence_hint": "unknown",
            "face_hint": "not_evaluated",
            "body_hint": "not_evaluated",
            "hands_hint": "not_evaluated",
            "object_hint": "not_evaluated",
            "object_showing_likelihood": 0.0,
            "scene_change_reason": "camera_ready",
            "scene_transition": "camera_ready",
            "last_moondream_result_state": "none",
            "last_grounding_summary": None,
            "last_grounding_summary_hash": None,
            "confidence": 0.0,
            "confidence_bucket": "none",
            "scene_age_ms": 90,
            "updated_at_ms": 0,
            "reason_codes": ["scene_social:v2", "camera_honesty:recent_frame_available"],
        },
        "reason_codes": ["camera_scene:v1", "camera_scene:available"],
    }
    default_browser_media = {
        "mode": "ready",
        "microphone_state": "ready",
        "camera_state": "ready",
        "track_state": "live",
        "echo_cancellation": True,
        "noise_suppression": True,
        "auto_gain_control": True,
    }
    return {
        "schema_version": 1,
        "available": True,
        "runtime": "browser",
        "transport": "WebRTC",
        "mode": mode,
        "profile": profile,
        "tts": "local-http-wav/MeloTTS" if is_zh_melo else "kokoro/English",
        "tts_backend": "local-http-wav" if is_zh_melo else "kokoro",
        "protected_playback": True,
        "browser_media": {**default_browser_media, **dict(browser_media or {})},
        "camera": {"state": "ready", "enabled": True},
        "vision": {
            "enabled": True,
            "backend": "moondream",
            "continuous_perception_enabled": False,
        },
        "memory": {"available": True},
        "interruption": {
            "protected_playback": True,
            "barge_in_state": "protected",
            "last_decision": "none",
            "headphones_recommended": False,
            "reason_codes": ["protected_playback:on"],
            **dict(interruption or {}),
        },
        "speech": {
            "director_mode": "melo_chunked" if is_zh_melo else "kokoro_chunked",
            "first_subtitle_latency_ms": 120,
            "first_audio_frame_latency_ms": 420,
            "subtitle_to_tts_request_latency_ms": 60,
            "current_queue_depth": 0,
            "max_queue_depth": 1,
            "stale_chunk_drop_count": 0,
            "reason_codes": ["speech:ready"],
            **dict(speech or {}),
        },
        "active_listening": {
            "phase": "idle",
            "partial_transcript_available": False,
            "partial_transcript_chars": 0,
            "final_transcript_chars": 0,
            "turn_duration_ms": 0,
            "speech_start_count": 0,
            "speech_stop_count": 0,
            "topic_count": 0,
            "constraint_count": 0,
            "reason_codes": ["active_listening:idle"],
            **dict(active_listening or {}),
        },
        "camera_presence": {**default_camera_presence, **dict(camera_presence or {})},
        "camera_scene": {**default_camera_scene, **dict(camera_scene or {})},
        "memory_persona": {
            "available": True,
            "profile": profile,
            "selected_memory_count": 0,
            "suppressed_memory_count": 0,
            "behavior_effect_count": 0,
            "used_in_current_reply_count": 0,
            "active_persona_reference_count": 0,
            "reason_codes": ["memory_persona:available"],
            **dict(memory_persona or {}),
        },
        "active_session_id": "bench_session",
        "active_client_id": "bench_client",
        "last_event_id": last_event_id,
        "last_event_type": None,
        "last_event_at": _FIXED_TS if last_event_id else None,
        "updated_at": _FIXED_TS,
        "reason_codes": ["browser_performance_state:v1", f"profile:{profile}"],
    }


def _native_state(
    *,
    profile: str,
    mode: str = "listening",
    last_event_id: int = 0,
    isolation: str | None = None,
    camera_helper: bool | None = None,
    protected_playback: bool = True,
    barge_in_state: str = "off",
) -> dict[str, Any]:
    helper_profile = profile == "native-en-kokoro-macos-camera"
    resolved_isolation = isolation or (
        "backend-plus-helper-camera" if helper_profile else "backend-only"
    )
    resolved_camera_helper = helper_profile if camera_helper is None else camera_helper
    return {
        "schema_version": 1,
        "available": True,
        "runtime": "native",
        "transport": "PyAudio",
        "mode": mode,
        "profile": profile,
        "isolation": resolved_isolation,
        "tts": "kokoro",
        "tts_backend": "kokoro",
        "protected_playback": protected_playback,
        "primary_browser_paths": ["browser-zh-melo", "browser-en-kokoro"],
        "native_audio": {
            "barge_in": barge_in_state,
            "echo_cancellation": "unavailable",
            "headphones_required_for_barge_in": True,
            "reason_codes": ["native_audio:pyaudio_no_browser_echo_cancellation"],
        },
        "native_camera": {
            "state": "helper_on_demand_single_frame" if resolved_camera_helper else "off",
            "helper_camera": resolved_camera_helper,
            "continuous_camera": False,
            "reason_codes": [
                "native_camera:helper_single_frame"
                if resolved_camera_helper
                else "native_camera:off"
            ],
        },
        "vision": {
            "enabled": resolved_camera_helper,
            "continuous_perception_enabled": False,
            "grounding_mode": "single_frame" if resolved_camera_helper else "unavailable",
        },
        "interruption": {
            "protected_playback": protected_playback,
            "barge_in_state": "protected" if protected_playback else "armed",
            "last_decision": "none",
            "headphones_recommended": not protected_playback,
            "reason_codes": ["protected_playback:on" if protected_playback else "barge_in:armed"],
        },
        "active_session_id": "bench_native_session",
        "active_client_id": "bench_native_client",
        "last_event_id": last_event_id,
        "last_event_type": None,
        "last_event_at": _FIXED_TS if last_event_id else None,
        "updated_at": _FIXED_TS,
        "reason_codes": ["native_isolation:v1", f"profile:{profile}"],
    }


@dataclass(frozen=True)
class BrowserPerfBenchCase:
    """One deterministic browser performance bench case."""

    case_id: str
    profile: str
    category: str
    title: str
    states: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    expected_event_order: tuple[str, ...] = ()
    human_rating_labels: tuple[str, ...] = _HUMAN_RATING_LABELS
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize case metadata without raw trace bodies."""
        return {
            "schema_version": BROWSER_PERF_BENCH_SCHEMA_VERSION,
            "case_id": self.case_id,
            "profile": self.profile,
            "category": self.category,
            "title": self.title,
            "state_count": len(self.states),
            "event_count": len(self.events),
            "expected_event_order": list(self.expected_event_order),
            "human_rating_labels": list(self.human_rating_labels),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrowserPerfBenchCheckResult:
    """One deterministic check inside a browser performance bench case."""

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
class BrowserPerfBenchMetricRow:
    """Compact browser performance metric row."""

    suite_id: str
    case_id: str
    profile: str
    category: str
    passed: bool
    event_order_score: float
    state_freshness_score: float
    subtitle_score: float
    tts_latency_score: float
    camera_grounding_score: float
    barge_in_safety_score: float
    memory_persona_score: float
    recovery_score: float
    holistic_signal_score: float
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the metric row."""
        return {
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "profile": self.profile,
            "category": self.category,
            "passed": self.passed,
            "event_order_score": self.event_order_score,
            "state_freshness_score": self.state_freshness_score,
            "subtitle_score": self.subtitle_score,
            "tts_latency_score": self.tts_latency_score,
            "camera_grounding_score": self.camera_grounding_score,
            "barge_in_safety_score": self.barge_in_safety_score,
            "memory_persona_score": self.memory_persona_score,
            "recovery_score": self.recovery_score,
            "holistic_signal_score": self.holistic_signal_score,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class BrowserPerfBenchResult:
    """Per-case browser performance bench result."""

    case: BrowserPerfBenchCase
    passed: bool
    checks: tuple[BrowserPerfBenchCheckResult, ...]
    metric_row: BrowserPerfBenchMetricRow
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the case result."""
        return {
            "case": self.case.as_dict(),
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "metric_row": self.metric_row.as_dict(),
            "evidence": _json_safe(self.evidence),
        }


@dataclass(frozen=True)
class BrowserPerfBenchReport:
    """Deterministic browser performance bench report."""

    suite_id: str
    profile_filter: str
    results: tuple[BrowserPerfBenchResult, ...]
    generated_at: str = _FIXED_TS
    artifact_links: dict[str, str] = field(default_factory=dict)

    @property
    def metric_rows(self) -> tuple[BrowserPerfBenchMetricRow, ...]:
        """Return compact metric rows."""
        return tuple(result.metric_row for result in self.results)

    @property
    def passed(self) -> bool:
        """Return whether all report gates passed."""
        return all(self.gate_results().values())

    def profile_coverage(self) -> tuple[str, ...]:
        """Return deterministic profile coverage."""
        present = {result.case.profile for result in self.results}
        known = tuple(profile for profile in BROWSER_PERF_BENCH_PROFILES if profile in present)
        extras = tuple(sorted(present.difference(BROWSER_PERF_BENCH_PROFILES)))
        return (*known, *extras)

    def aggregate_metrics(self) -> dict[str, float]:
        """Return average metrics across selected cases."""
        if not self.metric_rows:
            return {metric: 0.0 for metric in BROWSER_PERF_BENCH_METRICS}
        rows = [row.as_dict() for row in self.metric_rows]
        return {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in BROWSER_PERF_BENCH_METRICS
        }

    def gate_results(self) -> dict[str, bool]:
        """Return deterministic gate pass/fail values."""
        coverage = set(self.profile_coverage())
        all_checks = [check for result in self.results for check in result.checks]

        def check_passed(check_id: str) -> bool:
            matching = [check for check in all_checks if check.check_id == check_id]
            return bool(matching) and all(check.passed for check in matching)

        expected_profiles = (
            set(BROWSER_PERF_BENCH_PROFILES)
            if self.profile_filter == "all"
            else {self.profile_filter}
        )
        categories = {result.case.category for result in self.results}
        return {
            "all_cases_pass": bool(self.results) and all(result.passed for result in self.results),
            "both_primary_profiles": expected_profiles.issubset(coverage),
            "public_safety": check_passed("public_safety"),
            "protected_playback_defaults": check_passed("profile_defaults"),
            "speech_subtitles": (
                "speech" not in categories or check_passed("subtitle_readiness")
            ),
            "camera_grounding": (
                "camera" not in categories
                or all(
                    result.passed
                    for result in self.results
                    if result.case.category == "camera"
                )
            ),
            "interruption_safety": (
                "interruption" not in categories or check_passed("barge_in_safety")
            ),
            "memory_persona": (
                "memory_persona" not in categories or check_passed("memory_persona")
            ),
            "native_isolation": (
                "native_guardrail" not in categories or check_passed("native_guardrail")
            ),
            "recovery": "recovery" not in categories or check_passed("recovery"),
        }

    def failed_gates(self) -> tuple[str, ...]:
        """Return failed gate ids."""
        return tuple(gate for gate, passed in self.gate_results().items() if not passed)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "schema_version": BROWSER_PERF_BENCH_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "profile_filter": self.profile_filter,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "profile_coverage": list(self.profile_coverage()),
            "aggregate_metrics": self.aggregate_metrics(),
            "gates": self.gate_results(),
            "failed_gates": list(self.failed_gates()),
            "artifact_links": dict(sorted(self.artifact_links.items())),
            "metrics_rows": [row.as_dict() for row in self.metric_rows],
            "results": [result.as_dict() for result in self.results],
        }


def _check(
    check_id: str,
    passed: bool,
    detail: str,
    *reason_codes: str,
) -> BrowserPerfBenchCheckResult:
    return BrowserPerfBenchCheckResult(
        check_id=check_id,
        passed=bool(passed),
        detail=detail,
        reason_codes=_dedupe(reason_codes),
    )


def _profile_default_check(case: BrowserPerfBenchCase, final_state: dict[str, Any]) -> bool:
    protected = final_state.get("protected_playback") is True
    interruption_protected = _state_path(final_state, "interruption.protected_playback") is True
    continuous_off = _state_path(final_state, "vision.continuous_perception_enabled") is False
    if case.profile == "browser-zh-melo":
        return (
            protected
            and interruption_protected
            and final_state.get("runtime") == "browser"
            and final_state.get("transport") == "WebRTC"
            and final_state.get("tts") == "local-http-wav/MeloTTS"
            and final_state.get("tts_backend") == "local-http-wav"
            and _state_path(final_state, "vision.enabled") is True
            and _state_path(final_state, "camera_presence.enabled") is True
            and continuous_off
        )
    if case.profile == "browser-en-kokoro":
        return (
            protected
            and interruption_protected
            and final_state.get("runtime") == "browser"
            and final_state.get("transport") == "WebRTC"
            and final_state.get("tts") == "kokoro/English"
            and final_state.get("tts_backend") == "kokoro"
            and _state_path(final_state, "vision.enabled") is True
            and _state_path(final_state, "vision.backend") == "moondream"
            and _state_path(final_state, "camera_presence.enabled") is True
            and _state_path(final_state, "camera_scene.vision_backend") == "moondream"
            and continuous_off
        )
    if case.profile in BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES:
        helper_profile = case.profile == "native-en-kokoro-macos-camera"
        expected_isolation = "backend-plus-helper-camera" if helper_profile else "backend-only"
        return (
            protected
            and interruption_protected
            and final_state.get("runtime") == "native"
            and final_state.get("transport") == "PyAudio"
            and final_state.get("tts") == "kokoro"
            and final_state.get("tts_backend") == "kokoro"
            and final_state.get("isolation") == expected_isolation
            and _state_path(final_state, "native_audio.barge_in") == "off"
            and _state_path(final_state, "native_camera.helper_camera") is helper_profile
            and _state_path(final_state, "native_camera.continuous_camera") is False
            and continuous_off
        )
    return False


def _category_checks(
    case: BrowserPerfBenchCase,
    *,
    final_state: dict[str, Any],
    observed_event_types: tuple[str, ...],
) -> tuple[BrowserPerfBenchCheckResult, ...]:
    checks: list[BrowserPerfBenchCheckResult] = []
    if case.category == "connection":
        checks.append(
            _check(
                "connection_ready",
                "webrtc.connected" in observed_event_types
                and "client_media.updated" in observed_event_types
                and final_state.get("mode") == "listening",
                "WebRTC connection reaches listening with client media reported.",
                "connection:listening",
            )
        )
    elif case.category == "active_listening":
        checks.append(
            _check(
                "active_listening",
                _state_path(final_state, "active_listening.phase") == "final_transcript"
                and int(_state_path(final_state, "active_listening.final_transcript_chars", 0)) > 0
                and "active_listening.final_understanding_ready" in observed_event_types,
                "Active listening reaches final transcript using counts only.",
                "active_listening:final_transcript",
            )
        )
    elif case.category == "speech":
        subtitle_before_audio = _contains_order(
            observed_event_types,
            ("speech.generation_start", "speech.subtitle_ready", "speech.audio_start"),
        )
        checks.append(
            _check(
                "subtitle_readiness",
                subtitle_before_audio
                and int(_state_path(final_state, "speech.first_subtitle_latency_ms", -1)) >= 0,
                "Subtitle readiness precedes audio start.",
                "speech:subtitle_ready",
            )
        )
        checks.append(
            _check(
                "tts_latency",
                int(_state_path(final_state, "speech.first_audio_frame_latency_ms", -1)) >= 0
                and int(
                    _state_path(final_state, "speech.subtitle_to_tts_request_latency_ms", -1)
                )
                >= 0,
                "TTS latency fields are present and bounded.",
                "speech:latency_present",
            )
        )
    elif case.category == "camera":
        camera_state = str(_state_path(final_state, "camera_presence.state") or "")
        camera_scene_state = str(_state_path(final_state, "camera_scene.state") or "")
        checks.append(
            _check(
                "camera_grounding",
                case.profile == "browser-zh-melo"
                and _state_path(final_state, "camera_presence.current_answer_used_vision") is True
                and _state_path(final_state, "camera_scene.current_answer_used_vision") is True
                and _state_path(final_state, "camera_presence.grounding_mode") == "single_frame"
                and _state_path(final_state, "camera_scene.grounding_mode") == "single_frame"
                and _state_path(
                    final_state,
                    "camera_scene.scene_social_state_v2.camera_honesty_state",
                )
                == "can_see_now"
                and _state_path(
                    final_state,
                    "camera_scene.scene_social_state_v2.scene_transition",
                )
                == "vision_answered"
                and camera_state not in {"disabled", "stale", "stalled", "error"}
                and camera_scene_state not in {"disabled", "stale", "stalled", "error"}
                and "vision.fetch_user_image_success" in observed_event_types,
                "Camera answers must be grounded in one fresh still frame.",
                "camera:single_frame",
            )
        )
    elif case.category == "interruption":
        accepted = "interruption.accepted" in observed_event_types
        checks.append(
            _check(
                "barge_in_safety",
                final_state.get("protected_playback") is True
                and _state_path(final_state, "interruption.barge_in_state") == "protected"
                and not accepted
                and (
                    "interruption.suppressed" in observed_event_types
                    or "interruption.rejected" in observed_event_types
                ),
                "Protected playback suppresses bot-speech interruption by default.",
                "interruption:protected_default",
            )
        )
    elif case.category == "memory_persona":
        checks.append(
            _check(
                "memory_persona",
                _state_path(final_state, "memory_persona.available") is True
                and int(_state_path(final_state, "memory_persona.used_in_current_reply_count", 0))
                >= 1
                and int(_state_path(final_state, "memory_persona.behavior_effect_count", 0)) >= 1,
                "Memory/persona influence is visible as bounded counts and effects.",
                "memory_persona:used_in_reply",
            )
        )
    elif case.category == "recovery":
        forbidden_recovery = any(
            "renegotiation" in event_type or "track_disabled" in event_type
            for event_type in observed_event_types
        )
        checks.append(
            _check(
                "recovery",
                _contains_order(
                    observed_event_types,
                    ("webrtc.track_stalled", "webrtc.track_resumed"),
                )
                and not forbidden_recovery
                and _state_path(final_state, "browser_media.track_state") in {"live", "voice_only"},
                "Stalled media recovers as observed state without renegotiation or track mutation.",
                "recovery:observed_resume",
            )
        )
    elif case.category == "profile_parity":
        checks.append(
            _check(
                "profile_parity",
                case.profile == "browser-en-kokoro"
                and _state_path(final_state, "vision.enabled") is True
                and _state_path(final_state, "camera_scene.vision_backend") == "moondream"
                and _state_path(final_state, "speech.director_mode") == "kokoro_chunked",
                "Kokoro browser profile exposes the same camera/Moondream state contract.",
                "profile_parity:kokoro_camera_moondream",
            )
        )
    elif case.category == "native_guardrail":
        expected_isolation = (
            "backend-plus-helper-camera"
            if case.profile == "native-en-kokoro-macos-camera"
            else "backend-only"
        )
        accepted = "interruption.accepted" in observed_event_types
        checks.append(
            _check(
                "native_guardrail",
                final_state.get("runtime") == "native"
                and final_state.get("transport") == "PyAudio"
                and final_state.get("tts") == "kokoro"
                and final_state.get("protected_playback") is True
                and final_state.get("isolation") == expected_isolation
                and _state_path(final_state, "native_audio.barge_in") == "off"
                and not accepted,
                "Native Kokoro remains protected backend isolation, not product barge-in UX.",
                "native:isolation_guardrail",
            )
        )
    return tuple(checks)


def _build_metric_row(
    case: BrowserPerfBenchCase,
    checks: tuple[BrowserPerfBenchCheckResult, ...],
    *,
    suite_id: str,
) -> BrowserPerfBenchMetricRow:
    scores = {metric: 1.0 for metric in BROWSER_PERF_BENCH_METRICS}
    relevant_metrics = set(_CASE_METRICS.get(case.category, ()))
    for check in checks:
        if check.passed:
            continue
        for metric in _CHECK_METRICS.get(check.check_id, ()):
            scores[metric] = 0.0
            relevant_metrics.add(metric)
        if check.check_id == "public_safety":
            for metric in BROWSER_PERF_BENCH_METRICS:
                scores[metric] = 0.0
    if relevant_metrics:
        scores["holistic_signal_score"] = sum(scores[metric] for metric in relevant_metrics) / len(
            relevant_metrics
        )
    else:
        scores["holistic_signal_score"] = 1.0 if all(check.passed for check in checks) else 0.0
    reason_codes = _dedupe(
        (
            f"browser_perf_bench:{case.category}",
            f"profile:{case.profile}",
            *(reason for check in checks for reason in check.reason_codes),
            *case.reason_codes,
        )
    )
    return BrowserPerfBenchMetricRow(
        suite_id=suite_id,
        case_id=case.case_id,
        profile=case.profile,
        category=case.category,
        passed=all(check.passed for check in checks),
        event_order_score=scores["event_order_score"],
        state_freshness_score=scores["state_freshness_score"],
        subtitle_score=scores["subtitle_score"],
        tts_latency_score=scores["tts_latency_score"],
        camera_grounding_score=scores["camera_grounding_score"],
        barge_in_safety_score=scores["barge_in_safety_score"],
        memory_persona_score=scores["memory_persona_score"],
        recovery_score=scores["recovery_score"],
        holistic_signal_score=scores["holistic_signal_score"],
        reason_codes=reason_codes,
    )


def evaluate_browser_perf_bench_case(
    case: BrowserPerfBenchCase,
    *,
    suite_id: str = BROWSER_PERF_BENCH_SUITE_ID,
) -> BrowserPerfBenchResult:
    """Evaluate one deterministic browser performance bench case."""
    final_state = dict(case.states[-1]) if case.states else {}
    observed_event_types = _event_types(case.events)
    event_ids = [int(event.get("event_id") or 0) for event in case.events]
    public_safety_violations = find_browser_perf_public_safety_violations(
        {"states": case.states, "events": case.events}
    )
    expected_order = case.expected_event_order or observed_event_types
    checks = [
        _check(
            "public_safety",
            not public_safety_violations,
            "Trace payload contains only public-safe state and event fields.",
            "public_safety:ok" if not public_safety_violations else "public_safety:violation",
        ),
        _check(
            "event_order",
            event_ids == sorted(event_ids)
            and len(event_ids) == len(set(event_ids))
            and _contains_order(observed_event_types, expected_order),
            "Event ids are monotonic and expected event order is preserved.",
            "events:ordered",
        ),
        _check(
            "state_freshness",
            (not event_ids or int(final_state.get("last_event_id") or 0) >= max(event_ids))
            and final_state.get("mode") not in {"waiting", None},
            "Final state reflects the latest event and is no longer waiting.",
            "state:fresh",
        ),
        _check(
            "profile_defaults",
            _profile_default_check(case, final_state),
            "Primary browser profile defaults are preserved.",
            "profile:defaults",
        ),
        *_category_checks(
            case,
            final_state=final_state,
            observed_event_types=observed_event_types,
        ),
    ]
    metric_row = _build_metric_row(case, tuple(checks), suite_id=suite_id)
    evidence = {
        "profile": case.profile,
        "category": case.category,
        "state_count": len(case.states),
        "event_count": len(case.events),
        "event_types": observed_event_types,
        "final_mode": str(final_state.get("mode") or "unknown"),
        "final_last_event_id": int(final_state.get("last_event_id") or 0),
        "public_safety_violation_count": len(public_safety_violations),
        "human_rating_labels": case.human_rating_labels,
    }
    return BrowserPerfBenchResult(
        case=case,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
        metric_row=metric_row,
        evidence=evidence,
    )


def build_browser_perf_bench_suite() -> tuple[BrowserPerfBenchCase, ...]:
    """Return the built-in deterministic browser performance bench suite."""
    connection_events = (
        _event(1, "webrtc.connected", mode="connected"),
        _event(2, "client_media.updated", metadata={"microphone_state": "ready"}),
    )
    speech_events = (
        _event(1, "speech.generation_start", mode="thinking"),
        _event(2, "speech.subtitle_ready", mode="thinking", metadata={"subtitle_chars": 48}),
        _event(3, "speech.tts_request_start", mode="speaking", metadata={"chunk_chars": 48}),
        _event(4, "speech.audio_start", mode="speaking", metadata={"audio_latency_ms": 420}),
        _event(5, "speech.tts_end", mode="listening"),
    )
    camera_events = (
        _event(
            1,
            "camera.frame_received",
            metadata={"latest_frame_sequence": 7, "scene_transition": "frame_captured"},
        ),
        _event(
            2,
            "vision.fetch_user_image_start",
            mode="looking",
            metadata={"scene_transition": "looking_requested"},
        ),
        _event(
            3,
            "vision.fetch_user_image_success",
            mode="thinking",
            metadata={
                "scene_transition": "vision_answered",
                "camera_honesty_state": "can_see_now",
                "object_showing_likelihood": 0.85,
            },
        ),
    )
    interruption_events = (
        _event(1, "speech.audio_start", mode="speaking"),
        _event(2, "interruption.candidate", mode="speaking", metadata={"speech_ms": 120}),
        _event(3, "interruption.suppressed", mode="speaking"),
        _event(4, "interruption.listening_resumed", mode="listening"),
    )
    active_listening_events = (
        _event(1, "active_listening.listening_started", mode="listening"),
        _event(
            2,
            "active_listening.partial_understanding_updated",
            mode="listening",
            metadata={"partial_transcript_chars": 18, "topic_count": 1},
        ),
        _event(
            3,
            "active_listening.final_understanding_ready",
            mode="heard",
            metadata={"final_transcript_chars": 34, "ready_to_answer": True},
        ),
    )
    memory_events = (
        _event(1, "memory_persona.plan_compiled", mode="thinking"),
        _event(2, "memory_persona.used_in_reply", mode="thinking", metadata={"memory_count": 2}),
    )
    recovery_events = (
        _event(1, "webrtc.track_stalled", mode="listening", metadata={"stalled_count": 1}),
        _event(2, "webrtc.track_resumed", mode="listening", metadata={"resumed_count": 1}),
    )
    kokoro_events = (
        _event(1, "webrtc.connected", mode="connected"),
        _event(2, "client_media.updated", metadata={"microphone_state": "ready"}),
        _event(3, "speech.generation_start", mode="thinking"),
        _event(4, "speech.subtitle_ready", mode="thinking", metadata={"subtitle_chars": 42}),
        _event(5, "speech.audio_start", mode="speaking", metadata={"audio_latency_ms": 360}),
    )
    native_events = (
        _event(
            1,
            "native.voice_ready",
            source="native_voice",
            metadata={"barge_in_state": "off", "isolation": "backend-only"},
        ),
    )
    native_camera_helper_events = (
        _event(
            1,
            "native.voice_ready",
            source="native_voice",
            metadata={"barge_in_state": "off", "isolation": "backend-plus-helper-camera"},
        ),
        _event(
            2,
            "native.camera_helper_ready",
            source="native_voice",
            metadata={"grounding_mode": "single_frame", "continuous_camera": False},
        ),
    )
    return (
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_connection_listening",
            profile="browser-zh-melo",
            category="connection",
            title="Chinese Melo WebRTC connection reaches listening",
            states=(
                _state(profile="browser-zh-melo", mode="connected", last_event_id=1),
                _state(profile="browser-zh-melo", mode="listening", last_event_id=2),
            ),
            events=connection_events,
            expected_event_order=("webrtc.connected", "client_media.updated"),
            reason_codes=("profile:browser-zh-melo", "connection:listening"),
        ),
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_speech_subtitles",
            profile="browser-zh-melo",
            category="speech",
            title="Melo subtitles precede TTS and audio start",
            states=(
                _state(
                    profile="browser-zh-melo",
                    mode="speaking",
                    last_event_id=5,
                    speech={
                        "director_mode": "melo_chunked",
                        "first_subtitle_latency_ms": 96,
                        "first_audio_frame_latency_ms": 420,
                        "subtitle_to_tts_request_latency_ms": 58,
                    },
                ),
            ),
            events=speech_events,
            expected_event_order=(
                "speech.generation_start",
                "speech.subtitle_ready",
                "speech.tts_request_start",
                "speech.audio_start",
            ),
            reason_codes=("tts:local-http-wav/MeloTTS", "speech:subtitle_ready"),
        ),
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_camera_single_frame_grounding",
            profile="browser-zh-melo",
            category="camera",
            title="Camera answer is grounded in one fresh still frame",
            states=(
                _state(
                    profile="browser-zh-melo",
                    mode="thinking",
                    last_event_id=3,
                    camera_presence={
                        "state": "available",
                        "latest_frame_age_ms": 85,
                        "last_vision_result_state": "success",
                        "current_answer_used_vision": True,
                        "grounding_mode": "single_frame",
                    },
                    camera_scene={
                        "state": "available",
                        "status": "available",
                        "latest_frame_age_ms": 85,
                        "on_demand_vision_state": "success",
                        "last_vision_result_state": "success",
                        "current_answer_used_vision": True,
                        "grounding_mode": "single_frame",
                        "last_used_frame_sequence": 7,
                        "last_used_frame_age_ms": 85,
                        "last_used_frame_at": _FIXED_TS,
                        "scene_social_state_v2": {
                            "schema_version": 2,
                            "state_id": "scene-social-v2-browser-zh-melo-success",
                            "profile": "browser-zh-melo",
                            "language": "zh",
                            "camera_status": "ready",
                            "vision_status": "answered",
                            "camera_honesty_state": "can_see_now",
                            "frame_freshness": "fresh",
                            "frame_age_ms": 85,
                            "latest_frame_sequence": 7,
                            "last_used_frame_sequence": 7,
                            "last_used_frame_age_ms": 85,
                            "user_presence_hint": "present",
                            "face_hint": "not_evaluated",
                            "body_hint": "not_evaluated",
                            "hands_hint": "visible",
                            "object_hint": "object_showing",
                            "object_showing_likelihood": 0.85,
                            "scene_change_reason": "vision_answered",
                            "scene_transition": "vision_answered",
                            "last_moondream_result_state": "answered",
                            "last_grounding_summary": "fresh_object_grounding",
                            "last_grounding_summary_hash": "0" * 16,
                            "confidence": 0.85,
                            "confidence_bucket": "high",
                            "scene_age_ms": 85,
                            "updated_at_ms": 0,
                            "reason_codes": [
                                "scene_social:v2",
                                "camera_honesty:can_see_now",
                                "scene_social_transition:vision_answered",
                            ],
                        },
                    },
                ),
            ),
            events=camera_events,
            expected_event_order=(
                "camera.frame_received",
                "vision.fetch_user_image_start",
                "vision.fetch_user_image_success",
            ),
            reason_codes=("camera:single_frame",),
        ),
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_interruption_protected_default",
            profile="browser-zh-melo",
            category="interruption",
            title="Protected playback suppresses bot-speech interruption by default",
            states=(
                _state(
                    profile="browser-zh-melo",
                    mode="listening",
                    last_event_id=4,
                    interruption={
                        "barge_in_state": "protected",
                        "last_decision": "suppressed",
                        "headphones_recommended": False,
                    },
                ),
            ),
            events=interruption_events,
            expected_event_order=(
                "speech.audio_start",
                "interruption.candidate",
                "interruption.suppressed",
                "interruption.listening_resumed",
            ),
            reason_codes=("interruption:protected_default",),
        ),
        BrowserPerfBenchCase(
            case_id="browser_en_kokoro_active_listening_final_only",
            profile="browser-en-kokoro",
            category="active_listening",
            title="Kokoro final-only STT exposes active listening counts without partial text",
            states=(
                _state(
                    profile="browser-en-kokoro",
                    mode="thinking",
                    last_event_id=4,
                    active_listening={
                        "phase": "final_transcript",
                        "partial_transcript_available": False,
                        "partial_transcript_chars": 0,
                        "final_transcript_chars": 34,
                        "turn_duration_ms": 840,
                        "speech_start_count": 1,
                        "speech_stop_count": 1,
                        "topic_count": 1,
                        "constraint_count": 1,
                    },
                ),
            ),
            events=active_listening_events,
            expected_event_order=(
                "active_listening.listening_started",
                "active_listening.partial_understanding_updated",
                "active_listening.final_understanding_ready",
            ),
            reason_codes=("profile:browser-en-kokoro", "active_listening:final_only"),
        ),
        BrowserPerfBenchCase(
            case_id="browser_en_kokoro_camera_moondream_parity",
            profile="browser-en-kokoro",
            category="profile_parity",
            title="English Kokoro exposes the same browser camera/Moondream state surface",
            states=(
                _state(
                    profile="browser-en-kokoro",
                    mode="speaking",
                    last_event_id=5,
                    speech={"director_mode": "kokoro_chunked"},
                ),
            ),
            events=kokoro_events,
            expected_event_order=("webrtc.connected", "client_media.updated", "speech.audio_start"),
            reason_codes=("profile:browser-en-kokoro", "camera:moondream_available"),
        ),
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_memory_persona_visible",
            profile="browser-zh-melo",
            category="memory_persona",
            title="Memory/persona influence is visible as bounded public counts",
            states=(
                _state(
                    profile="browser-zh-melo",
                    mode="thinking",
                    last_event_id=2,
                    memory_persona={
                        "selected_memory_count": 2,
                        "suppressed_memory_count": 1,
                        "behavior_effect_count": 2,
                        "used_in_current_reply_count": 2,
                        "active_persona_reference_count": 1,
                        "reason_codes": ["memory_persona:used_in_reply"],
                    },
                ),
            ),
            events=memory_events,
            expected_event_order=("memory_persona.plan_compiled", "memory_persona.used_in_reply"),
            reason_codes=("memory_persona:public_summary",),
        ),
        BrowserPerfBenchCase(
            case_id="browser_zh_melo_track_stall_resume_observed",
            profile="browser-zh-melo",
            category="recovery",
            title="Track stalled/resumed is observed without renegotiation or track mutation",
            states=(
                _state(
                    profile="browser-zh-melo",
                    mode="listening",
                    last_event_id=2,
                    browser_media={"track_state": "live"},
                    camera_presence={"state": "available", "fresh": True},
                ),
            ),
            events=recovery_events,
            expected_event_order=("webrtc.track_stalled", "webrtc.track_resumed"),
            reason_codes=("recovery:observed_resume",),
        ),
        BrowserPerfBenchCase(
            case_id="native_en_kokoro_backend_isolation_guardrail",
            profile="native-en-kokoro",
            category="native_guardrail",
            title="Native English Kokoro is protected backend isolation by default",
            states=(
                _native_state(
                    profile="native-en-kokoro",
                    mode="listening",
                    last_event_id=1,
                ),
            ),
            events=native_events,
            expected_event_order=("native.voice_ready",),
            reason_codes=("native:isolation_backend_only",),
        ),
        BrowserPerfBenchCase(
            case_id="native_en_kokoro_macos_camera_helper_isolation_guardrail",
            profile="native-en-kokoro-macos-camera",
            category="native_guardrail",
            title="Native camera helper is on-demand single-frame isolation",
            states=(
                _native_state(
                    profile="native-en-kokoro-macos-camera",
                    mode="listening",
                    last_event_id=2,
                ),
            ),
            events=native_camera_helper_events,
            expected_event_order=("native.voice_ready", "native.camera_helper_ready"),
            reason_codes=("native:isolation_helper_camera", "camera:single_frame"),
        ),
    )


def evaluate_browser_perf_bench_suite(
    cases: Iterable[BrowserPerfBenchCase] | None = None,
    *,
    profile: str = "all",
    suite_id: str = BROWSER_PERF_BENCH_SUITE_ID,
) -> BrowserPerfBenchReport:
    """Evaluate the deterministic browser performance bench suite."""
    if profile not in (*BROWSER_PERF_BENCH_PROFILES, "all"):
        raise ValueError(f"unsupported browser perf bench profile: {profile}")
    selected_cases = tuple(cases) if cases is not None else build_browser_perf_bench_suite()
    if profile != "all":
        selected_cases = tuple(case for case in selected_cases if case.profile == profile)
    results = tuple(
        evaluate_browser_perf_bench_case(case, suite_id=suite_id) for case in selected_cases
    )
    return BrowserPerfBenchReport(suite_id=suite_id, profile_filter=profile, results=results)


def render_browser_perf_bench_metrics_rows(
    report: BrowserPerfBenchReport,
) -> tuple[dict[str, Any], ...]:
    """Return stable compact metric rows."""
    return tuple(row.as_dict() for row in report.metric_rows)


def render_browser_perf_bench_human_rating_form(report: BrowserPerfBenchReport) -> str:
    """Render a stable human rating form for dogfooding sessions."""
    lines = [
        "# Browser Performance Human Rating Form",
        "",
        f"- suite: `{report.suite_id}`",
        f"- profile filter: `{report.profile_filter}`",
        "",
        "Rate each dimension from 1 (poor) to 5 (excellent).",
        "",
    ]
    for result in report.results:
        lines.extend(
            [
                f"## {result.case.case_id}",
                "",
                f"- profile: `{result.case.profile}`",
                f"- category: `{result.case.category}`",
                "",
            ]
        )
        for label in _HUMAN_RATING_LABELS:
            lines.append(f"- {label}: 1 2 3 4 5")
        lines.extend(["- notes:", ""])
    return "\n".join(lines)


def render_browser_perf_bench_pairwise_form(report: BrowserPerfBenchReport) -> str:
    """Render a stable pairwise build comparison form."""
    lines = [
        "# Browser Performance Pairwise Comparison Form",
        "",
        f"- suite: `{report.suite_id}`",
        "- build A:",
        "- build B:",
        "",
        "Use the same prompts, browser profile, and setup for both builds.",
        "",
        "| case | profile | winner | state clarity | felt-heard | voice pacing | "
        "memory usefulness | camera grounding | interruption naturalness | enjoyment | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in report.results:
        lines.append(
            f"| `{result.case.case_id}` | `{result.case.profile}` | A/B/tie |  |  |  |  |  |  |  |  |"
        )
    return "\n".join(lines)


def render_browser_perf_bench_markdown(report: BrowserPerfBenchReport) -> str:
    """Render a stable Markdown summary for the browser performance bench report."""
    lines = [
        "# Browser Performance Bench Report",
        "",
        f"- suite: `{report.suite_id}`",
        f"- schema: `{BROWSER_PERF_BENCH_SCHEMA_VERSION}`",
        f"- profile filter: `{report.profile_filter}`",
        f"- passed: `{str(report.passed).lower()}`",
        "",
        "## Gates",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{gate}` | `{str(passed).lower()}` |"
        for gate, passed in sorted(report.gate_results().items())
    )
    lines.extend(["", "## Aggregate Metrics", "", "| metric | value |", "| --- | ---: |"])
    lines.extend(
        f"| `{metric}` | {value:.4f} |"
        for metric, value in sorted(report.aggregate_metrics().items())
    )
    lines.extend(["", "## Cases", "", "| case | profile | category | passed |", "| --- | --- | --- | --- |"])
    for result in report.results:
        lines.append(
            f"| `{result.case.case_id}` | `{result.case.profile}` | "
            f"`{result.case.category}` | `{str(result.passed).lower()}` |"
        )
    return "\n".join(lines)


def write_browser_perf_bench_artifacts(
    report: BrowserPerfBenchReport,
    *,
    output_dir: str | Path = BROWSER_PERF_BENCH_ARTIFACT_DIR,
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
        json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True) for result in report.results
    ]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    markdown_path.write_text(f"{render_browser_perf_bench_markdown(report)}\n", encoding="utf-8")
    human_form_path.write_text(
        f"{render_browser_perf_bench_human_rating_form(report)}\n",
        encoding="utf-8",
    )
    pairwise_form_path.write_text(
        f"{render_browser_perf_bench_pairwise_form(report)}\n",
        encoding="utf-8",
    )
    return {
        "human_rating_form": str(human_form_path),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
        "pairwise_form": str(pairwise_form_path),
    }


__all__ = [
    "BROWSER_PERF_BENCH_ARTIFACT_DIR",
    "BROWSER_PERF_BENCH_CATEGORIES",
    "BROWSER_PERF_BENCH_METRICS",
    "BROWSER_PERF_BENCH_NATIVE_GUARDRAIL_PROFILES",
    "BROWSER_PERF_BENCH_PROFILES",
    "BROWSER_PERF_BENCH_SCHEMA_VERSION",
    "BROWSER_PERF_BENCH_SUITE_ID",
    "BrowserPerfBenchCase",
    "BrowserPerfBenchCheckResult",
    "BrowserPerfBenchMetricRow",
    "BrowserPerfBenchReport",
    "BrowserPerfBenchResult",
    "build_browser_perf_bench_suite",
    "evaluate_browser_perf_bench_case",
    "evaluate_browser_perf_bench_suite",
    "find_browser_perf_public_safety_violations",
    "render_browser_perf_bench_human_rating_form",
    "render_browser_perf_bench_markdown",
    "render_browser_perf_bench_metrics_rows",
    "render_browser_perf_bench_pairwise_form",
    "write_browser_perf_bench_artifacts",
]

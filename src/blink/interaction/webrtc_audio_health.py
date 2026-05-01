"""Public-safe WebRTC audio health and echo-risk policy snapshots."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import RLock
from typing import Any, Mapping

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_MAX_REASON_CODES = 32
_MAX_STATS = 16
_ALLOWED_MEDIA_STATES = {
    "unknown",
    "ready",
    "receiving",
    "stalled",
    "stale",
    "unavailable",
    "permission_denied",
    "device_not_found",
    "error",
}
_ALLOWED_TRACK_STATES = {
    "unknown",
    "ready",
    "live",
    "receiving",
    "stalled",
    "ended",
    "muted",
    "disabled",
    "unavailable",
    "error",
}
_ALLOWED_OUTPUT_STATES = {
    "unknown",
    "idle",
    "playing",
    "speaking",
    "buffering",
    "stalled",
    "muted",
    "ended",
    "error",
}
_ALLOWED_OUTPUT_ROUTES = {
    "unknown",
    "speaker",
    "headphones",
    "headset",
    "earpiece",
    "bluetooth",
    "muted",
}
_ALLOWED_ECHO_STATES = {"enabled", "disabled", "unsupported", "unknown"}
_SAFE_STATS_KEYS = {
    "audio_level",
    "concealed_samples",
    "concealment_events",
    "current_round_trip_time_ms",
    "jitter_buffer_delay_ms",
    "jitter_ms",
    "packets_lost",
    "packets_received",
    "packets_sent",
    "round_trip_time_ms",
    "total_audio_energy",
}
_CAMEL_STATS_ALIASES = {
    "audioLevel": "audio_level",
    "concealedSamples": "concealed_samples",
    "concealmentEvents": "concealment_events",
    "currentRoundTripTime": "current_round_trip_time_ms",
    "currentRoundTripTimeMs": "current_round_trip_time_ms",
    "jitter": "jitter_ms",
    "jitterBufferDelay": "jitter_buffer_delay_ms",
    "jitterBufferDelayMs": "jitter_buffer_delay_ms",
    "jitterMs": "jitter_ms",
    "packetsLost": "packets_lost",
    "packetsReceived": "packets_received",
    "packetsSent": "packets_sent",
    "roundTripTime": "round_trip_time_ms",
    "roundTripTimeMs": "round_trip_time_ms",
    "totalAudioEnergy": "total_audio_energy",
}


class WebRTCEchoRiskLevel(str, Enum):
    """Public echo-risk levels for browser audio."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WebRTCBargeInPolicyState(str, Enum):
    """Public interruption policy state for WebRTC audio."""

    PROTECTED = "protected"
    ARMED = "armed"
    ADAPTIVE = "adaptive"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    raw_value = getattr(value, "value", value)
    text = _TOKEN_RE.sub("_", str(raw_value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_choice(value: object, *, allowed: set[str], default: str) -> str:
    text = _safe_token(value, default=default, limit=64).lower()
    return text if text in allowed else default


def _safe_echo_state(value: object) -> str:
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    text = _safe_token(value, default="unknown", limit=64).lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    if text in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    if text in {"unsupported", "unavailable", "not_supported"}:
        return "unsupported"
    return text if text in _ALLOWED_ECHO_STATES else "unknown"


def _safe_bool(value: object) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "safe"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "unsafe"}:
        return False
    return None


def _safe_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 6)


def _safe_reason_codes(values: object, *, limit: int = _MAX_REASON_CODES) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        code = _safe_token(value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def sanitize_webrtc_audio_stats(value: object) -> dict[str, int | float]:
    """Return a bounded public-safe summary of WebRTC audio stats."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = _CAMEL_STATS_ALIASES.get(str(raw_key), _safe_token(raw_key, default="", limit=64))
        key = key.lower()
        if key not in _SAFE_STATS_KEYS:
            continue
        number = _safe_number(raw_value)
        if number is None:
            continue
        result[key] = number
        if len(result) >= _MAX_STATS:
            break
    return result


@dataclass(frozen=True)
class WebRTCAudioHealthV2:
    """Public-safe WebRTC audio-health snapshot for browser diagnostics."""

    profile: str
    language: str
    microphone_state: str = "unknown"
    input_track_state: str = "unknown"
    output_playback_state: str = "unknown"
    output_route: str = "unknown"
    echo_cancellation: str = "unknown"
    noise_suppression: str = "unknown"
    auto_gain_control: str = "unknown"
    echo_safe: bool | None = None
    echo_safe_source: str = "none"
    stats: dict[str, int | float] = field(default_factory=dict)
    echo_risk: WebRTCEchoRiskLevel | str = WebRTCEchoRiskLevel.UNKNOWN
    barge_in_state: WebRTCBargeInPolicyState | str = WebRTCBargeInPolicyState.PROTECTED
    protected_playback: bool = True
    adaptive_barge_in_armed: bool = False
    explicit_barge_in_armed: bool = False
    assistant_speaking: bool = False
    false_interruption_counts: dict[str, int] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now_iso)
    reason_codes: list[str] = field(default_factory=list)
    schema_version: int = 2

    def as_dict(self) -> dict[str, Any]:
        """Return a public-safe audio-health payload."""
        microphone_state = _safe_choice(
            self.microphone_state,
            allowed=_ALLOWED_MEDIA_STATES,
            default="unknown",
        )
        input_track_state = _safe_choice(
            self.input_track_state,
            allowed=_ALLOWED_TRACK_STATES,
            default="unknown",
        )
        output_playback_state = _safe_choice(
            self.output_playback_state,
            allowed=_ALLOWED_OUTPUT_STATES,
            default="unknown",
        )
        output_route = _safe_choice(
            self.output_route,
            allowed=_ALLOWED_OUTPUT_ROUTES,
            default="unknown",
        )
        echo_cancellation = _safe_echo_state(self.echo_cancellation)
        noise_suppression = _safe_echo_state(self.noise_suppression)
        auto_gain_control = _safe_echo_state(self.auto_gain_control)
        try:
            echo_risk = WebRTCEchoRiskLevel(str(self.echo_risk)).value
        except ValueError:
            echo_risk = WebRTCEchoRiskLevel.UNKNOWN.value
        try:
            barge_in_state = WebRTCBargeInPolicyState(str(self.barge_in_state)).value
        except ValueError:
            barge_in_state = WebRTCBargeInPolicyState.PROTECTED.value
        stats = sanitize_webrtc_audio_stats(self.stats)
        false_counts = {
            _safe_token(key, default="", limit=64): max(0, int(value))
            for key, value in sorted(self.false_interruption_counts.items())
            if _safe_token(key, default="", limit=64)
        }
        reason_codes = _safe_reason_codes(
            [
                "webrtc_audio_health:v2",
                f"microphone:{microphone_state}",
                f"input_track:{input_track_state}",
                f"output_playback:{output_playback_state}",
                f"echo_risk:{echo_risk}",
                f"barge_in:{barge_in_state}",
                *self.reason_codes,
            ]
        )
        protected = barge_in_state == WebRTCBargeInPolicyState.PROTECTED.value
        explicit_armed = self.explicit_barge_in_armed is True
        adaptive_armed = (
            self.adaptive_barge_in_armed is True
            and barge_in_state == WebRTCBargeInPolicyState.ADAPTIVE.value
        )
        return {
            "schema_version": 2,
            "runtime": "browser",
            "transport": "WebRTC",
            "profile": _safe_token(self.profile, default="manual", limit=96),
            "language": _safe_token(self.language, default="unknown", limit=32),
            "microphone_state": microphone_state,
            "input_track_state": input_track_state,
            "output_playback_state": output_playback_state,
            "echo_risk": echo_risk,
            "barge_in_state": barge_in_state,
            "protected_playback": protected,
            "adaptive_barge_in_armed": adaptive_armed,
            "explicit_barge_in_armed": explicit_armed,
            "headphones_recommended": protected or echo_risk in {"medium", "high"},
            "microphone": {
                "state": microphone_state,
                "ready": microphone_state in {"ready", "receiving"},
                "reason_codes": _safe_reason_codes([f"microphone:{microphone_state}"], limit=8),
            },
            "input_track": {
                "state": input_track_state,
                "healthy": input_track_state in {"ready", "live", "receiving"},
                "reason_codes": _safe_reason_codes([f"input_track:{input_track_state}"], limit=8),
            },
            "output_playback": {
                "state": output_playback_state,
                "route": output_route,
                "assistant_speaking": self.assistant_speaking is True,
                "reason_codes": _safe_reason_codes(
                    [
                        f"output_playback:{output_playback_state}",
                        f"output_route:{output_route}",
                    ],
                    limit=8,
                ),
            },
            "echo": {
                "echo_cancellation": echo_cancellation,
                "noise_suppression": noise_suppression,
                "auto_gain_control": auto_gain_control,
                "echo_safe": self.echo_safe,
                "echo_safe_source": _safe_token(self.echo_safe_source, default="none", limit=64),
                "reason_codes": _safe_reason_codes(
                    [
                        f"echo_cancellation:{echo_cancellation}",
                        f"noise_suppression:{noise_suppression}",
                        f"auto_gain_control:{auto_gain_control}",
                        (
                            "echo_safe:reported_true"
                            if self.echo_safe is True
                            else "echo_safe:reported_false"
                            if self.echo_safe is False
                            else "echo_safe:unreported"
                        ),
                    ],
                    limit=8,
                ),
            },
            "stats": {
                "available": bool(stats),
                "summary": stats,
                "reason_codes": _safe_reason_codes(
                    [
                        "webrtc_stats:available" if stats else "webrtc_stats:unavailable",
                    ],
                    limit=8,
                ),
            },
            "barge_in": {
                "state": barge_in_state,
                "protected": protected,
                "armed": barge_in_state
                in {
                    WebRTCBargeInPolicyState.ARMED.value,
                    WebRTCBargeInPolicyState.ADAPTIVE.value,
                },
                "adaptive": barge_in_state == WebRTCBargeInPolicyState.ADAPTIVE.value,
                "explicitly_armed": explicit_armed,
                "reason_codes": _safe_reason_codes([f"barge_in:{barge_in_state}"], limit=8),
            },
            "false_interruption_counts": false_counts,
            "updated_at": str(self.updated_at)[:64],
            "reason_codes": reason_codes,
        }


class WebRTCAudioHealthController:
    """Deterministic audio-health controller for the browser runtime."""

    def __init__(self):
        """Initialize the controller."""
        self._lock = RLock()
        self._input_track_state = "unknown"
        self._output_playback_state = "unknown"
        self._reason_codes: list[str] = []
        self._updated_at = _now_iso()

    def set_input_track_state(self, state: object, *, reason_code: str | None = None) -> None:
        """Record the latest public input-track state."""
        with self._lock:
            self._input_track_state = _safe_choice(
                state,
                allowed=_ALLOWED_TRACK_STATES,
                default="unknown",
            )
            if reason_code:
                self._reason_codes = _safe_reason_codes([*self._reason_codes, reason_code])
            self._updated_at = _now_iso()

    def set_output_playback_state(self, state: object, *, reason_code: str | None = None) -> None:
        """Record the latest public output playback state."""
        with self._lock:
            self._output_playback_state = _safe_choice(
                state,
                allowed=_ALLOWED_OUTPUT_STATES,
                default="unknown",
            )
            if reason_code:
                self._reason_codes = _safe_reason_codes([*self._reason_codes, reason_code])
            self._updated_at = _now_iso()

    def snapshot(
        self,
        *,
        profile: str,
        language: str,
        browser_media: Mapping[str, Any] | None = None,
        protected_playback: bool = True,
        explicit_barge_in_armed: bool = False,
        assistant_speaking: bool = False,
        interruption: Mapping[str, Any] | None = None,
    ) -> WebRTCAudioHealthV2:
        """Build a public-safe v2 audio-health snapshot."""
        media = browser_media if isinstance(browser_media, Mapping) else {}
        echo = media.get("echo") if isinstance(media.get("echo"), Mapping) else {}
        with self._lock:
            input_track_state = self._input_track_state
            output_playback_state = self._output_playback_state
            controller_reason_codes = list(self._reason_codes)
            updated_at = self._updated_at

        microphone_state = _safe_choice(
            media.get("microphone_state") or media.get("microphoneState"),
            allowed=_ALLOWED_MEDIA_STATES,
            default="unknown",
        )
        if input_track_state == "unknown" and microphone_state in {"ready", "receiving", "stalled"}:
            input_track_state = microphone_state
        media_output_state = _safe_choice(
            media.get("output_playback_state") or media.get("outputPlaybackState"),
            allowed=_ALLOWED_OUTPUT_STATES,
            default="unknown",
        )
        if media_output_state != "unknown":
            output_playback_state = media_output_state
        elif output_playback_state == "unknown":
            output_playback_state = "speaking" if assistant_speaking else "idle"
        output_route = _safe_choice(
            media.get("output_route") or media.get("outputRoute"),
            allowed=_ALLOWED_OUTPUT_ROUTES,
            default="unknown",
        )
        echo_safe = _safe_bool(
            media.get("echo_safe")
            if "echo_safe" in media
            else media.get("echoSafe", echo.get("echo_safe", echo.get("echoSafe")))
        )
        echo_safe_source = "client" if echo_safe is not None else "none"
        echo_cancellation = _safe_echo_state(echo.get("echo_cancellation"))
        noise_suppression = _safe_echo_state(echo.get("noise_suppression"))
        auto_gain_control = _safe_echo_state(echo.get("auto_gain_control"))
        stats = sanitize_webrtc_audio_stats(
            media.get("webrtc_stats") or media.get("webrtcStats") or media.get("stats")
        )
        false_counts = {}
        if isinstance(interruption, Mapping):
            false_counts = {
                str(key): int(value)
                for key, value in dict(interruption.get("false_interruption_counts") or {}).items()
                if isinstance(value, int)
            }
        policy_state, adaptive_armed = self._classify_policy_state(
            protected_playback=protected_playback,
            explicit_barge_in_armed=explicit_barge_in_armed,
            echo_safe=echo_safe,
            microphone_state=microphone_state,
            input_track_state=input_track_state,
        )
        echo_risk = self._classify_echo_risk(
            microphone_state=microphone_state,
            input_track_state=input_track_state,
            output_playback_state=output_playback_state,
            output_route=output_route,
            echo_safe=echo_safe,
            echo_cancellation=echo_cancellation,
            noise_suppression=noise_suppression,
            policy_state=policy_state,
        )
        reason_codes = _safe_reason_codes(
            [
                *controller_reason_codes,
                "webrtc_audio_health:explicit_barge_in_armed"
                if explicit_barge_in_armed
                else "webrtc_audio_health:protected_by_default",
                (
                    "webrtc_audio_health:client_echo_safe"
                    if echo_safe is True
                    else "webrtc_audio_health:client_echo_unsafe"
                    if echo_safe is False
                    else "webrtc_audio_health:client_echo_unreported"
                ),
                "webrtc_audio_health:echo_cancellation_hint_only"
                if echo_cancellation == "enabled" and policy_state == "protected"
                else "",
            ]
        )
        return WebRTCAudioHealthV2(
            profile=profile,
            language=language,
            microphone_state=microphone_state,
            input_track_state=input_track_state,
            output_playback_state=output_playback_state,
            output_route=output_route,
            echo_cancellation=echo_cancellation,
            noise_suppression=noise_suppression,
            auto_gain_control=auto_gain_control,
            echo_safe=echo_safe,
            echo_safe_source=echo_safe_source,
            stats=stats,
            echo_risk=echo_risk,
            barge_in_state=policy_state,
            protected_playback=policy_state == WebRTCBargeInPolicyState.PROTECTED.value,
            adaptive_barge_in_armed=adaptive_armed,
            explicit_barge_in_armed=explicit_barge_in_armed,
            assistant_speaking=assistant_speaking,
            false_interruption_counts=false_counts,
            updated_at=updated_at,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _classify_policy_state(
        *,
        protected_playback: bool,
        explicit_barge_in_armed: bool,
        echo_safe: bool | None,
        microphone_state: str,
        input_track_state: str,
    ) -> tuple[str, bool]:
        mic_ready = microphone_state in {"ready", "receiving"}
        track_ready = input_track_state in {"ready", "live", "receiving"}
        if explicit_barge_in_armed or not protected_playback:
            return WebRTCBargeInPolicyState.ARMED.value, False
        if echo_safe is True and mic_ready and track_ready:
            return WebRTCBargeInPolicyState.ADAPTIVE.value, True
        return WebRTCBargeInPolicyState.PROTECTED.value, False

    @staticmethod
    def _classify_echo_risk(
        *,
        microphone_state: str,
        input_track_state: str,
        output_playback_state: str,
        output_route: str,
        echo_safe: bool | None,
        echo_cancellation: str,
        noise_suppression: str,
        policy_state: str,
    ) -> str:
        if microphone_state in {"unknown", "unavailable"}:
            return WebRTCEchoRiskLevel.UNKNOWN.value
        if microphone_state in {"error", "permission_denied", "device_not_found"}:
            return WebRTCEchoRiskLevel.HIGH.value
        if input_track_state in {"stalled", "ended", "disabled", "error"}:
            return WebRTCEchoRiskLevel.HIGH.value
        if policy_state in {
            WebRTCBargeInPolicyState.ARMED.value,
            WebRTCBargeInPolicyState.ADAPTIVE.value,
        } and echo_safe is True:
            return WebRTCEchoRiskLevel.LOW.value
        output_active = output_playback_state in {"playing", "speaking", "buffering"}
        if output_route in {"headphones", "headset", "earpiece", "muted"} and echo_safe is True:
            return WebRTCEchoRiskLevel.LOW.value
        if output_active and echo_safe is not True:
            if echo_cancellation == "enabled" and noise_suppression == "enabled":
                return WebRTCEchoRiskLevel.MEDIUM.value
            return WebRTCEchoRiskLevel.HIGH.value
        if echo_cancellation == "enabled" and noise_suppression == "enabled":
            return WebRTCEchoRiskLevel.LOW.value
        return WebRTCEchoRiskLevel.MEDIUM.value


__all__ = [
    "WebRTCAudioHealthController",
    "WebRTCAudioHealthV2",
    "WebRTCBargeInPolicyState",
    "WebRTCEchoRiskLevel",
    "sanitize_webrtc_audio_stats",
]

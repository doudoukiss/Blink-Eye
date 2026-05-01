"""Deterministic Phase 01 baseline for Bilingual Performance Intelligence V3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.cli.local_runtime_profiles import load_local_runtime_profiles

PERFORMANCE_INTELLIGENCE_BASELINE_ID = "bilingual_performance_intelligence_v3/phase01"
PERFORMANCE_INTELLIGENCE_BASELINE_SCHEMA_VERSION = 1
PERFORMANCE_INTELLIGENCE_BASELINE_LOCKED_AT = "2026-04-27T00:00:00+00:00"
PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "evals"
    / "bilingual_performance_intelligence_v3"
    / "baseline_profiles.json"
)

PRIMARY_BROWSER_PROFILE_SPECS: dict[str, dict[str, str]] = {
    "browser-zh-melo": {
        "launcher": "./scripts/run-local-browser-melo.sh",
        "language": "zh",
        "tts_backend": "local-http-wav",
        "tts_runtime_label": "local-http-wav/MeloTTS",
    },
    "browser-en-kokoro": {
        "launcher": "./scripts/run-local-browser-kokoro-en.sh",
        "language": "en",
        "tts_backend": "kokoro",
        "tts_runtime_label": "kokoro/English",
    },
}
PRIMARY_BROWSER_PROFILES = tuple(PRIMARY_BROWSER_PROFILE_SPECS)
CLIENT_URL = "http://127.0.0.1:7860/client/"
ACTOR_STATE_ENDPOINTS = ("/api/runtime/actor-state", "/api/runtime/performance-state")
ACTOR_EVENT_ENDPOINTS = ("/api/runtime/actor-events", "/api/runtime/performance-events")
CURRENT_GATE_COMMANDS = (
    "./scripts/eval-performance-intelligence-baseline.sh --check",
    "./scripts/eval-bilingual-actor-bench.sh",
    "uv run pytest tests/test_bilingual_actor_bench.py tests/test_actor_release_gate.py",
    "uv run --extra runner --extra webrtc pytest tests/test_local_workflows.py",
)
NATIVE_PYAUDIO_ISOLATION_LANES = ("native-en-kokoro", "native-en-kokoro-macos-camera")


@dataclass(frozen=True)
class PerformanceIntelligenceProfileBaseline:
    """Machine-readable baseline for one first-class browser product path."""

    profile: str
    product_status: str
    isolation_lane: bool
    launcher: str
    client_url: str
    language: str
    tts_backend: str
    tts_runtime_label: str
    media: str
    vision_backend: str
    browser_vision_default: bool
    continuous_perception_default: bool
    protected_playback_default: bool
    allow_barge_in_default: bool
    actor_state_endpoints: tuple[str, ...]
    actor_event_endpoints: tuple[str, ...]
    gate_commands: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the profile baseline."""
        return {
            "actor_event_endpoints": list(self.actor_event_endpoints),
            "actor_state_endpoints": list(self.actor_state_endpoints),
            "allow_barge_in_default": self.allow_barge_in_default,
            "browser_vision_default": self.browser_vision_default,
            "client_url": self.client_url,
            "continuous_perception_default": self.continuous_perception_default,
            "gate_commands": list(self.gate_commands),
            "isolation_lane": self.isolation_lane,
            "language": self.language,
            "launcher": self.launcher,
            "media": self.media,
            "product_status": self.product_status,
            "profile": self.profile,
            "protected_playback_default": self.protected_playback_default,
            "tts_backend": self.tts_backend,
            "tts_runtime_label": self.tts_runtime_label,
            "vision_backend": self.vision_backend,
        }


@dataclass(frozen=True)
class PerformanceIntelligenceBaseline:
    """Phase 01 bilingual actor runtime non-regression baseline."""

    baseline_id: str
    schema_version: int
    snapshot_type: str
    locked_at: str
    profiles: tuple[PerformanceIntelligenceProfileBaseline, ...]
    native_pyaudio_policy: str
    native_pyaudio_isolation_lanes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the complete baseline."""
        return {
            "baseline_id": self.baseline_id,
            "locked_at": self.locked_at,
            "native_pyaudio_isolation_lanes": list(self.native_pyaudio_isolation_lanes),
            "native_pyaudio_policy": self.native_pyaudio_policy,
            "profiles": [profile.as_dict() for profile in self.profiles],
            "schema_version": self.schema_version,
            "snapshot_type": self.snapshot_type,
        }


def build_performance_intelligence_baseline() -> PerformanceIntelligenceBaseline:
    """Build the deterministic V3 Phase 01 baseline from checked-in profile defaults."""
    runtime_profiles = load_local_runtime_profiles(include_local_override=False)
    baseline_profiles: list[PerformanceIntelligenceProfileBaseline] = []

    for profile_id, spec in PRIMARY_BROWSER_PROFILE_SPECS.items():
        runtime_profile = runtime_profiles[profile_id]
        allow_barge_in_default = bool(runtime_profile.get("allow_barge_in", False))
        baseline_profiles.append(
            PerformanceIntelligenceProfileBaseline(
                profile=profile_id,
                product_status="equal_primary_browser_product_path",
                isolation_lane=False,
                launcher=spec["launcher"],
                client_url=CLIENT_URL,
                language=str(runtime_profile.get("language")),
                tts_backend=str(runtime_profile.get("tts_backend")),
                tts_runtime_label=spec["tts_runtime_label"],
                media="browser/WebRTC microphone + camera",
                vision_backend="moondream",
                browser_vision_default=bool(runtime_profile.get("browser_vision", False)),
                continuous_perception_default=bool(
                    runtime_profile.get("continuous_perception", False)
                ),
                protected_playback_default=not allow_barge_in_default,
                allow_barge_in_default=allow_barge_in_default,
                actor_state_endpoints=ACTOR_STATE_ENDPOINTS,
                actor_event_endpoints=ACTOR_EVENT_ENDPOINTS,
                gate_commands=CURRENT_GATE_COMMANDS,
            )
        )

    return PerformanceIntelligenceBaseline(
        baseline_id=PERFORMANCE_INTELLIGENCE_BASELINE_ID,
        schema_version=PERFORMANCE_INTELLIGENCE_BASELINE_SCHEMA_VERSION,
        snapshot_type="PerformanceIntelligenceBaseline",
        locked_at=PERFORMANCE_INTELLIGENCE_BASELINE_LOCKED_AT,
        profiles=tuple(baseline_profiles),
        native_pyaudio_policy="backend_isolation_only_not_product_ux",
        native_pyaudio_isolation_lanes=NATIVE_PYAUDIO_ISOLATION_LANES,
    )


def render_performance_intelligence_baseline_json(
    baseline: PerformanceIntelligenceBaseline | None = None,
) -> str:
    """Render deterministic baseline JSON."""
    payload = (baseline or build_performance_intelligence_baseline()).as_dict()
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_performance_intelligence_baseline(
    path: Path = PERFORMANCE_INTELLIGENCE_BASELINE_FIXTURE_PATH,
    *,
    baseline: PerformanceIntelligenceBaseline | None = None,
) -> Path:
    """Write the deterministic baseline fixture and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_performance_intelligence_baseline_json(baseline), encoding="utf-8")
    return path

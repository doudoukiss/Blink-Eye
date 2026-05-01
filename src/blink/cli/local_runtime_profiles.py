#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Typed local runtime profiles for stable Blink product paths."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from blink.project_identity import local_env_name

LOCAL_CONFIG_PROFILE_ENV = local_env_name("CONFIG_PROFILE")
LOCAL_CONFIG_PROFILES_PATH_ENV = local_env_name("CONFIG_PROFILES_PATH")
LOCAL_CONFIG_PROFILE_OVERRIDE_PATH = ".blink-local-profiles.json"
BUILTIN_LOCAL_CONFIG_PROFILES_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "local_runtime_profiles.json"
)

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,80}$")
_RUNTIMES = {"browser", "voice"}
_LANGUAGES = {"en", "zh"}
_TTS_BACKENDS = {"kokoro", "piper", "xtts", "local-http-wav"}
_CAMERA_SOURCES = {"none", "macos-helper"}
_ALLOWED_FIELDS = {
    "language",
    "stt_backend",
    "stt_model",
    "tts_backend",
    "tts_voice",
    "browser_vision",
    "voice_vision",
    "continuous_perception",
    "continuous_perception_interval_secs",
    "allow_barge_in",
    "ignore_env_system_prompt",
    "camera_source",
    "camera_framerate",
    "camera_max_width",
    "vision_model",
}
_BROWSER_FIELDS = _ALLOWED_FIELDS - {"voice_vision", "camera_source", "camera_framerate", "camera_max_width"}
_VOICE_FIELDS = _ALLOWED_FIELDS - {"browser_vision", "continuous_perception", "continuous_perception_interval_secs"}
_BOOL_FIELDS = {
    "browser_vision",
    "voice_vision",
    "continuous_perception",
    "allow_barge_in",
    "ignore_env_system_prompt",
}
_NUMBER_FIELDS = {
    "continuous_perception_interval_secs",
    "camera_framerate",
    "camera_max_width",
}
_STRING_FIELDS = {
    "language",
    "stt_backend",
    "stt_model",
    "tts_backend",
    "tts_voice",
    "camera_source",
    "vision_model",
}


class LocalRuntimeProfileError(ValueError):
    """Raised when a local runtime profile is malformed or unsupported."""


@dataclass(frozen=True)
class LocalRuntimeProfile:
    """Validated local runtime profile.

    Parameters:
        profile_id: Stable profile id used by wrappers and CLI flags.
        runtime: Runtime family, currently ``browser`` or ``voice``.
        label: Human-readable profile label.
        values: Bounded product-behavior defaults for this profile.
    """

    profile_id: str
    runtime: str
    label: str
    values: Mapping[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Return a profile value."""
        return self.values.get(key, default)

    def flag(self, key: str, default: bool = False) -> bool:
        """Return a boolean profile value."""
        value = self.values.get(key, default)
        return bool(value)


def selected_local_runtime_profile_id(explicit_profile_id: Optional[str] = None) -> Optional[str]:
    """Return the selected local runtime profile id from CLI or environment."""
    value = explicit_profile_id or os.getenv(LOCAL_CONFIG_PROFILE_ENV)
    if value in (None, ""):
        return None
    return str(value).strip() or None


def load_local_runtime_profiles(
    *,
    profiles_path: Optional[str | Path] = None,
    include_local_override: bool = True,
) -> dict[str, LocalRuntimeProfile]:
    """Load built-in local profiles plus an optional gitignored local override file."""
    profiles = _load_profile_file(BUILTIN_LOCAL_CONFIG_PROFILES_PATH)
    if not include_local_override:
        return profiles

    explicit_override = profiles_path or os.getenv(LOCAL_CONFIG_PROFILES_PATH_ENV)
    override_path = (
        Path(explicit_override).expanduser()
        if explicit_override not in (None, "")
        else Path.cwd() / LOCAL_CONFIG_PROFILE_OVERRIDE_PATH
    )
    if override_path.exists():
        profiles.update(_load_profile_file(override_path))
    return profiles


def resolve_local_runtime_profile(
    *,
    runtime: str,
    profile_id: Optional[str] = None,
) -> Optional[LocalRuntimeProfile]:
    """Resolve a selected profile for a runtime, if one was requested."""
    selected = selected_local_runtime_profile_id(profile_id)
    if selected is None:
        return None

    profiles = load_local_runtime_profiles()
    profile = profiles.get(selected)
    if profile is None:
        raise LocalRuntimeProfileError(f"Unknown local runtime profile: {selected}")
    if profile.runtime != runtime:
        raise LocalRuntimeProfileError(
            f"Local runtime profile {selected} is for {profile.runtime}, not {runtime}."
        )
    return profile


def profile_value(profile: Optional[LocalRuntimeProfile], key: str, default: Any = None) -> Any:
    """Return a profile value when a profile is active."""
    if profile is None:
        return default
    return profile.get(key, default)


def _load_profile_file(path: Path) -> dict[str, LocalRuntimeProfile]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalRuntimeProfileError(f"Malformed local runtime profiles JSON: {path}") from exc
    except OSError as exc:
        raise LocalRuntimeProfileError(f"Could not read local runtime profiles: {path}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LocalRuntimeProfileError("Local runtime profiles must use schema_version 1.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise LocalRuntimeProfileError("Local runtime profiles must contain a profiles list.")

    profiles: dict[str, LocalRuntimeProfile] = {}
    for raw_profile in raw_profiles:
        profile = _parse_profile(raw_profile)
        profiles[profile.profile_id] = profile
    return profiles


def _parse_profile(raw_profile: object) -> LocalRuntimeProfile:
    if not isinstance(raw_profile, dict):
        raise LocalRuntimeProfileError("Each local runtime profile must be an object.")

    profile_id = _clean_text(raw_profile.get("id"))
    if not profile_id or not _PROFILE_ID_RE.match(profile_id):
        raise LocalRuntimeProfileError("Local runtime profile id is invalid.")

    runtime = _clean_text(raw_profile.get("runtime"))
    if runtime not in _RUNTIMES:
        raise LocalRuntimeProfileError(f"Unsupported runtime for profile {profile_id}: {runtime}")

    label = _clean_text(raw_profile.get("label")) or profile_id
    raw_values = raw_profile.get("values")
    if not isinstance(raw_values, dict):
        raise LocalRuntimeProfileError(f"Profile {profile_id} must contain object values.")

    values = _parse_values(profile_id=profile_id, runtime=runtime, raw_values=raw_values)
    return LocalRuntimeProfile(
        profile_id=profile_id,
        runtime=runtime,
        label=label,
        values=values,
    )


def _parse_values(
    *,
    profile_id: str,
    runtime: str,
    raw_values: Mapping[str, object],
) -> dict[str, Any]:
    allowed = _BROWSER_FIELDS if runtime == "browser" else _VOICE_FIELDS
    values: dict[str, Any] = {}
    for key, value in raw_values.items():
        if key not in allowed:
            raise LocalRuntimeProfileError(f"Profile {profile_id} has unsupported field: {key}")
        if key in _BOOL_FIELDS:
            if not isinstance(value, bool):
                raise LocalRuntimeProfileError(f"Profile {profile_id} field {key} must be boolean.")
            values[key] = value
        elif key in _NUMBER_FIELDS:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise LocalRuntimeProfileError(f"Profile {profile_id} field {key} must be numeric.")
            values[key] = value
        elif key in _STRING_FIELDS:
            cleaned = _clean_text(value)
            if not cleaned:
                raise LocalRuntimeProfileError(f"Profile {profile_id} field {key} must be text.")
            values[key] = cleaned
        else:  # pragma: no cover - guarded by field sets
            raise LocalRuntimeProfileError(f"Profile {profile_id} has unsupported field: {key}")

    _validate_values(profile_id, values)
    return values


def _validate_values(profile_id: str, values: Mapping[str, Any]) -> None:
    language = values.get("language")
    if language is not None and language not in _LANGUAGES:
        raise LocalRuntimeProfileError(f"Profile {profile_id} has unsupported language: {language}")
    tts_backend = values.get("tts_backend")
    if tts_backend is not None and tts_backend not in _TTS_BACKENDS:
        raise LocalRuntimeProfileError(
            f"Profile {profile_id} has unsupported TTS backend: {tts_backend}"
        )
    camera_source = values.get("camera_source")
    if camera_source is not None and camera_source not in _CAMERA_SOURCES:
        raise LocalRuntimeProfileError(
            f"Profile {profile_id} has unsupported camera source: {camera_source}"
        )


def _clean_text(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    cleaned = str(value).strip()
    return cleaned or None

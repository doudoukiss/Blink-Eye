"""Curated public-safe LLM model profiles for local Blink browser sessions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from blink.cli.local_common import LOCAL_LLM_PROVIDERS

DEFAULT_LOCAL_MODEL_PROFILE_ID = "ollama-qwen3_5-4b"
LOCAL_MODEL_PROFILE_EXTENSION_ENV = "BLINK_LOCAL_LLM_MODEL_PROFILES_PATH"
REMOTE_MODEL_SELECTION_ENV = "BLINK_LOCAL_ENABLE_REMOTE_MODEL_SELECTION"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalLLMModelProfile:
    """A bounded backend-owned model option exposed to the browser UI."""

    profile_id: str
    label: str
    provider: str
    model: str
    runtime_tier: str
    latency_tier: str
    capability_tier: str
    language_fit: tuple[str, ...]
    recommended_for: tuple[str, ...] = ()
    default: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        """Serialize the profile without provider secrets or private config."""
        return {
            "id": self.profile_id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "runtime_tier": self.runtime_tier,
            "latency_tier": self.latency_tier,
            "capability_tier": self.capability_tier,
            "language_fit": list(self.language_fit),
            "recommended_for": list(self.recommended_for),
            "default": bool(self.default),
        }


_BUILTIN_MODEL_PROFILES: tuple[LocalLLMModelProfile, ...] = (
    LocalLLMModelProfile(
        profile_id=DEFAULT_LOCAL_MODEL_PROFILE_ID,
        label="Qwen 4B local",
        provider="ollama",
        model="qwen3.5:4b",
        runtime_tier="local",
        latency_tier="fast",
        capability_tier="standard",
        language_fit=("zh", "en"),
        recommended_for=("daily local Chinese", "low latency", "offline development"),
        default=True,
    ),
    LocalLLMModelProfile(
        profile_id="ollama-qwen3_5-9b",
        label="Qwen 9B local",
        provider="ollama",
        model="qwen3.5:9b",
        runtime_tier="local",
        latency_tier="balanced",
        capability_tier="stronger",
        language_fit=("zh", "en"),
        recommended_for=("richer local answers", "Chinese technical explanation"),
    ),
    LocalLLMModelProfile(
        profile_id="openai-gpt-5_4-nano",
        label="GPT Nano",
        provider="openai-responses",
        model="gpt-5.4-nano",
        runtime_tier="remote",
        latency_tier="fast",
        capability_tier="light",
        language_fit=("en", "zh"),
        recommended_for=("low latency remote", "short answers"),
    ),
    LocalLLMModelProfile(
        profile_id="openai-gpt-5_4-mini",
        label="GPT Mini",
        provider="openai-responses",
        model="gpt-5.4-mini",
        runtime_tier="remote",
        latency_tier="balanced",
        capability_tier="balanced",
        language_fit=("en", "zh"),
        recommended_for=("hybrid demos", "balanced reasoning"),
    ),
    LocalLLMModelProfile(
        profile_id="openai-gpt-5_4",
        label="GPT quality",
        provider="openai-responses",
        model="gpt-5.4",
        runtime_tier="remote",
        latency_tier="quality",
        capability_tier="strong",
        language_fit=("en", "zh"),
        recommended_for=("quality checks", "harder reasoning"),
    ),
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tuple_text(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in (_normalized_text(item) for item in value) if item)


def _profile_from_mapping(value: Mapping[str, Any]) -> LocalLLMModelProfile | None:
    profile_id = _normalized_text(value.get("id") or value.get("profile_id"))
    label = _normalized_text(value.get("label"))
    provider = _normalized_text(value.get("provider")).lower()
    model = _normalized_text(value.get("model"))
    if not profile_id or not label or provider not in LOCAL_LLM_PROVIDERS or not model:
        return None

    language_fit = _tuple_text(value.get("language_fit")) or ("zh", "en")
    return LocalLLMModelProfile(
        profile_id=profile_id,
        label=label,
        provider=provider,
        model=model,
        runtime_tier=_normalized_text(value.get("runtime_tier")) or "custom",
        latency_tier=_normalized_text(value.get("latency_tier")) or "custom",
        capability_tier=_normalized_text(value.get("capability_tier")) or "custom",
        language_fit=language_fit,
        recommended_for=_tuple_text(value.get("recommended_for")),
        default=bool(value.get("default", False)),
    )


def _load_extension_profiles(path: str | None) -> tuple[LocalLLMModelProfile, ...]:
    if not path:
        return ()
    try:
        raw = Path(path).expanduser().read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except Exception:
        return ()

    records = parsed.get("profiles") if isinstance(parsed, dict) else parsed
    if not isinstance(records, list):
        return ()

    profiles: list[LocalLLMModelProfile] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        profile = _profile_from_mapping(record)
        if profile is not None:
            profiles.append(profile)
    return tuple(profiles)


def _dedupe_profiles(
    profiles: Iterable[LocalLLMModelProfile],
) -> tuple[LocalLLMModelProfile, ...]:
    seen: set[str] = set()
    result: list[LocalLLMModelProfile] = []
    for profile in profiles:
        if profile.profile_id in seen:
            continue
        seen.add(profile.profile_id)
        result.append(profile)
    return tuple(result)


def load_local_llm_model_profiles(
    extension_path: str | None = None,
) -> tuple[LocalLLMModelProfile, ...]:
    """Load curated model profiles plus optional local file extensions."""
    path = extension_path if extension_path is not None else os.getenv(LOCAL_MODEL_PROFILE_EXTENSION_ENV)
    return _dedupe_profiles((*_BUILTIN_MODEL_PROFILES, *_load_extension_profiles(path)))


def local_llm_model_profile_by_id(
    profile_id: str,
    *,
    profiles: Iterable[LocalLLMModelProfile] | None = None,
) -> LocalLLMModelProfile | None:
    """Return a model profile by its public id."""
    normalized = _normalized_text(profile_id)
    for profile in profiles or load_local_llm_model_profiles():
        if profile.profile_id == normalized:
            return profile
    return None


def local_llm_model_profile_id_for(
    *,
    provider: str,
    model: str,
    profiles: Iterable[LocalLLMModelProfile] | None = None,
) -> str | None:
    """Return the curated profile id for a provider/model pair, when known."""
    for profile in profiles or load_local_llm_model_profiles():
        if profile.provider == provider and profile.model == model:
            return profile.profile_id
    return None


def remote_model_selection_enabled(*, current_provider: str | None = None) -> bool:
    """Return whether browser UI remote model selection is explicitly unlocked."""
    if current_provider == "openai-responses":
        return True
    return str(os.getenv(REMOTE_MODEL_SELECTION_ENV, "")).strip().lower() in _TRUE_VALUES

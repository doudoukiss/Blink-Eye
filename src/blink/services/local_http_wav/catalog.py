#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for `local-http-wav` voice catalogs exposed by sidecar servers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from blink.transcriptions.language import Language


def local_http_wav_language_key(language: Language | str | None) -> str:
    """Normalize a language value to the sidecar voice-catalog key."""
    value = language.value if isinstance(language, Language) else str(language or "")
    normalized = value.lower()
    if normalized.startswith("zh") or normalized.startswith("cmn"):
        return "zh"
    return "en"


def local_http_wav_speakers_for_language(
    catalog: Mapping[str, Any], language: Language | str | None
) -> list[str]:
    """Return the advertised speakers for a language from `/voices` JSON."""
    speakers = catalog.get("speakers")
    if not isinstance(speakers, Mapping):
        return []

    language_speakers = speakers.get(local_http_wav_language_key(language))
    if isinstance(language_speakers, Sequence) and not isinstance(language_speakers, str):
        return [str(speaker) for speaker in language_speakers if str(speaker).strip()]
    if isinstance(language_speakers, Mapping):
        return [str(speaker) for speaker in language_speakers if str(speaker).strip()]
    return []


def local_http_wav_default_speaker_for_language(
    catalog: Mapping[str, Any], language: Language | str | None
) -> Optional[str]:
    """Return the default speaker for a language from `/voices` JSON."""
    defaults = catalog.get("default_speakers")
    if not isinstance(defaults, Mapping):
        return None

    default_speaker = defaults.get(local_http_wav_language_key(language))
    if default_speaker in (None, ""):
        return None
    return str(default_speaker)


def local_http_wav_fallback_speaker(
    catalog: Mapping[str, Any],
    *,
    configured_voice: str,
    language: Language | str | None,
) -> Optional[str]:
    """Choose a safe fallback speaker when a configured voice is unavailable."""
    speakers = local_http_wav_speakers_for_language(catalog, language)
    if not speakers or configured_voice in speakers:
        return configured_voice

    default_speaker = local_http_wav_default_speaker_for_language(catalog, language)
    if default_speaker in speakers:
        return default_speaker

    return speakers[0]

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Generic Blink TTS adapter for a local HTTP server that returns WAV audio."""

from __future__ import annotations

import io
import time
import wave
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp
from loguru import logger

from blink.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from blink.services.local_http_wav.catalog import (
    local_http_wav_fallback_speaker,
    local_http_wav_language_key,
    local_http_wav_speakers_for_language,
)
from blink.services.settings import TTSSettings
from blink.services.tts_service import TTSService
from blink.utils.tracing.service_decorators import traced_tts

DEFAULT_LOCAL_HTTP_WAV_CONTEXT_TIMEOUT_SECS = 45.0


class LocalHttpWavTTSService(TTSService):
    """Synthesize speech by POSTing JSON to a local HTTP service that returns WAV bytes."""

    Settings = TTSSettings

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str,
        sample_rate: Optional[int] = None,
        settings: Optional[TTSSettings] = None,
        **kwargs,
    ):
        """Initialize the local HTTP WAV TTS service.

        Args:
            aiohttp_session: Shared HTTP session used for requests.
            base_url: Base URL for the local TTS server.
            sample_rate: Output sample rate override.
            settings: Runtime settings including voice, language, and model.
            **kwargs: Additional arguments passed to `TTSService`.
        """
        default_settings = self.Settings(model=None, voice=None, language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        kwargs.setdefault(
            "stop_frame_timeout_s",
            DEFAULT_LOCAL_HTTP_WAV_CONTEXT_TIMEOUT_SECS,
        )

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._session = aiohttp_session
        self._base_url = base_url.rstrip("/")
        self._voice_catalog_cache: Optional[Mapping[str, Any]] = None

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports framework TTS metrics."""
        return True

    async def _post_tts_request(self, payload: dict[str, Any]) -> tuple[int, str, bytes]:
        """POST a single TTS request and return status, detail text, and body bytes."""
        async with self._session.post(f"{self._base_url}/tts", json=payload) as response:
            detail = ""
            if response.status != 200:
                detail = await response.text()
                return response.status, detail, b""
            return response.status, detail, await response.read()

    async def _fetch_voice_catalog(self) -> Optional[Mapping[str, Any]]:
        """Fetch `/voices` metadata when the sidecar exposes it."""
        if self._voice_catalog_cache is not None:
            return self._voice_catalog_cache

        try:
            async with self._session.get(f"{self._base_url}/voices") as response:
                if response.status != 200:
                    return None
                payload = await response.json(content_type=None)
        except Exception:
            return None

        if not isinstance(payload, Mapping):
            return None

        self._voice_catalog_cache = payload
        return payload

    async def _recover_invalid_speaker(
        self,
        *,
        payload: dict[str, Any],
        detail: str,
    ) -> tuple[dict[str, Any], Optional[str]] | None:
        """Choose a safer speaker when the sidecar rejects the configured one."""
        configured_voice = payload.get("speaker")
        if configured_voice in (None, "") or "speaker" not in detail.lower():
            return None

        catalog = await self._fetch_voice_catalog()
        if catalog is None:
            return None

        configured_voice = str(configured_voice)
        language = payload.get("language") or self._settings.language
        fallback_voice = local_http_wav_fallback_speaker(
            catalog,
            configured_voice=configured_voice,
            language=language,
        )
        if fallback_voice == configured_voice:
            return None

        available = local_http_wav_speakers_for_language(catalog, language)
        retry_payload = dict(payload)
        retry_payload["speaker"] = fallback_voice
        self._settings.voice = fallback_voice
        logger.warning(
            "{} speaker '{}' is unavailable for language '{}' at {}. "
            "Falling back to {}. Available speakers: {}",
            self,
            configured_voice,
            local_http_wav_language_key(language),
            self._base_url,
            repr(fallback_voice) if fallback_voice else "server auto-selection",
            ", ".join(available) if available else "<unknown>",
        )
        return retry_payload, fallback_voice

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the configured local HTTP WAV service."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        request_started = time.perf_counter()
        payload = {
            "text": text,
            "speaker": self._settings.voice,
            "language": self._settings.language,
            "model": self._settings.model,
        }
        ttfb_stopped = False

        try:
            status, detail, wav_bytes = await self._post_tts_request(payload)
            if status != 200:
                recovered = await self._recover_invalid_speaker(payload=payload, detail=detail)
                if recovered is not None:
                    retry_payload, _ = recovered
                    status, detail, wav_bytes = await self._post_tts_request(retry_payload)
                if status != 200:
                    logger.warning(
                        "{} local-http-wav request failed status={} chars={}",
                        self,
                        status,
                        len(text),
                    )
                    yield ErrorFrame(error=f"Local TTS error {status}: {detail}")
                    return
            await self.stop_ttfb_metrics()
            ttfb_stopped = True
        except Exception as exc:
            yield ErrorFrame(error=f"Local TTS request failed: {exc}")
            return
        finally:
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                pcm = wav_file.readframes(frame_count)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
        except Exception as exc:
            yield ErrorFrame(error=f"Failed to parse WAV response: {exc}")
            return

        audio_ms = (frame_count / sample_rate * 1000.0) if sample_rate else 0.0
        request_ms = (time.perf_counter() - request_started) * 1000.0
        logger.info(
            "{} local-http-wav synthesized chars={} bytes={} audio_ms={} "
            "request_ms={} sample_rate={} channels={}",
            self,
            len(text),
            len(wav_bytes),
            round(audio_ms),
            round(request_ms),
            sample_rate,
            channels,
        )

        await self.start_tts_usage_metrics(text)
        yield TTSAudioRawFrame(
            audio=pcm,
            sample_rate=self.sample_rate or sample_rate,
            num_channels=channels,
            context_id=context_id,
        )

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prefetch local Blink assets for MacBook Pro workflows."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass

from huggingface_hub import snapshot_download

from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    DEFAULT_LOCAL_PROFILE,
    DEFAULT_LOCAL_STT_BACKEND,
    DEFAULT_LOCAL_VISION_MODEL,
    DEFAULT_OLLAMA_MODEL,
    PIPER_CACHE_DIR,
    create_local_tts_service,
    create_local_vision_service,
    default_local_stt_model,
    default_local_tts_backend,
    get_local_env,
    local_env_flag,
    maybe_load_dotenv,
    resolve_local_language,
    resolve_local_tts_voice,
)
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language


@dataclass(frozen=True)
class LocalPrefetchConfig:
    """Configuration for local asset prefetching."""

    profile: str
    with_vision: bool
    model: str
    language: Language
    stt_backend: str
    stt_model: str
    tts_backend: str
    tts_voice: str | None
    vision_model: str


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Prefetch {PROJECT_IDENTITY.display_name} local model assets."
    )
    parser.add_argument(
        "--profile",
        choices=["text", "voice", "browser", "full"],
        default=DEFAULT_LOCAL_PROFILE,
        help="Local workflow profile to prefetch for.",
    )
    parser.add_argument(
        "--with-vision",
        action="store_true",
        help="Also prefetch the optional local Moondream vision model.",
    )
    parser.add_argument(
        "--language",
        help=f"Language code for local STT/TTS assets, default {DEFAULT_LOCAL_LANGUAGE.value}.",
    )
    parser.add_argument("--model", help="Ollama model name override.")
    parser.add_argument(
        "--stt-backend",
        choices=["mlx-whisper", "whisper"],
        help="Local speech-to-text backend override.",
    )
    parser.add_argument("--stt-model", help="Speech-to-text model override.")
    parser.add_argument(
        "--tts-backend",
        choices=["kokoro", "piper", "xtts", "local-http-wav"],
        help="Local text-to-speech backend override.",
    )
    parser.add_argument("--tts-voice", help="Text-to-speech voice override.")
    return parser


def resolve_config(args: argparse.Namespace) -> LocalPrefetchConfig:
    """Resolve prefetch settings from CLI and environment."""
    maybe_load_dotenv()
    language = resolve_local_language(args.language or get_local_env("LANGUAGE"))
    stt_backend = args.stt_backend or get_local_env("STT_BACKEND") or DEFAULT_LOCAL_STT_BACKEND
    tts_backend = args.tts_backend or get_local_env("TTS_BACKEND") or default_local_tts_backend(
        language
    )

    stt_model = (
        args.stt_model
        or get_local_env("STT_MODEL")
        or default_local_stt_model(backend=stt_backend, language=language)
    )

    tts_voice = resolve_local_tts_voice(
        tts_backend,
        language,
        explicit_voice=args.tts_voice,
    )

    return LocalPrefetchConfig(
        profile=args.profile,
        with_vision=args.with_vision or local_env_flag("BROWSER_VISION"),
        model=args.model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        language=language,
        stt_backend=stt_backend,
        stt_model=stt_model,
        tts_backend=tts_backend,
        tts_voice=tts_voice,
        vision_model=get_local_env("VISION_MODEL") or DEFAULT_LOCAL_VISION_MODEL,
    )


def _run_ollama_pull(model: str):
    """Ensure the configured Ollama model is present locally."""
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull Ollama model {model!r}.")


def _prefetch_stt_assets(backend: str, model: str):
    """Prefetch local STT assets for the selected backend."""
    if backend == "mlx-whisper":
        snapshot_download(repo_id=model)
        return

    if backend == "whisper":
        from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

        WhisperModel(model, device="cpu", compute_type="int8")
        return

    raise ValueError(f"Unsupported STT backend: {backend}")


def _prefetch_tts_assets(backend: str, voice: str | None, language: Language):
    """Prefetch local TTS assets for the selected backend."""
    if backend == "kokoro":
        create_local_tts_service(backend=backend, voice=voice, language=language)
        return

    if backend == "piper":
        from blink.services.piper.tts import PiperTTSService

        if voice is None:
            raise ValueError("Piper prefetch requires a resolved voice id.")
        PIPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PiperTTSService(
            download_dir=PIPER_CACHE_DIR,
            settings=PiperTTSService.Settings(voice=voice, language=language),
        )
        return

    if backend in {"xtts", "local-http-wav"}:
        return

    raise ValueError(f"Unsupported TTS backend: {backend}")


def _prefetch_vision_assets(model: str):
    """Prefetch the optional local vision model."""
    create_local_vision_service(model=model)


async def run_prefetch(config: LocalPrefetchConfig) -> int:
    """Prefetch the configured local assets."""
    print(f"Prefetching Ollama model {config.model!r}...")
    await asyncio.to_thread(_run_ollama_pull, config.model)

    if config.profile in {"voice", "browser", "full"}:
        print(f"Prefetching STT assets for {config.stt_backend!r}...")
        await asyncio.to_thread(_prefetch_stt_assets, config.stt_backend, config.stt_model)

        if config.tts_backend in {"xtts", "local-http-wav"}:
            print(
                f"Skipping TTS asset prefetch for {config.tts_backend!r}; "
                "that backend uses an external local HTTP service."
            )
        else:
            print(f"Prefetching TTS assets for {config.tts_backend!r}...")
            await asyncio.to_thread(
                _prefetch_tts_assets,
                config.tts_backend,
                config.tts_voice,
                config.language,
            )

    if config.with_vision:
        print(f"Prefetching vision assets for {config.vision_model!r}...")
        await asyncio.to_thread(_prefetch_vision_assets, config.vision_model)

    print("Local asset prefetch complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for local asset prefetching."""
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_prefetch(config))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Diagnostics for Blink's local macOS workflows."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Optional

import httpx

from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    DEFAULT_LOCAL_LLM_PROVIDER,
    DEFAULT_LOCAL_PROFILE,
    DEFAULT_LOCAL_STT_BACKEND,
    DEFAULT_LOCAL_VISION_MODEL,
    LOCAL_LLM_PROVIDERS,
    LocalLLMConfig,
    default_local_stt_model,
    default_local_tts_backend,
    default_local_tts_base_url,
    default_piper_model_path,
    get_audio_devices,
    get_default_audio_device,
    get_local_env,
    huggingface_model_cache_dir,
    is_chinese_language,
    kokoro_assets_present,
    local_env_flag,
    maybe_load_dotenv,
    model_cache_exists,
    resolve_local_language,
    resolve_local_llm_config,
    resolve_local_runtime_tts_selection,
    resolve_local_tts_base_url,
    resolve_local_tts_voice,
    resolve_preferred_audio_device_indexes,
    verify_openai_responses_config,
)
from blink.cli.local_voice import (
    CAMERA_SOURCE_MACOS_HELPER,
    CAMERA_SOURCE_NONE,
    CAMERA_SOURCES,
    _default_macos_camera_helper_app_path,
)
from blink.project_identity import PROJECT_IDENTITY
from blink.services.local_http_wav.catalog import (
    local_http_wav_fallback_speaker,
    local_http_wav_speakers_for_language,
)
from blink.transcriptions.language import Language


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single doctor check."""

    name: str
    status: str
    detail: str
    remedy: Optional[str] = None


def _parse_utc_ts(value: object) -> datetime | None:
    """Parse a stored timestamp into UTC when possible."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=f"Inspect your local {PROJECT_IDENTITY.display_name} MBP setup."
    )
    parser.add_argument(
        "--profile",
        choices=["text", "voice", "browser", "full"],
        default=DEFAULT_LOCAL_PROFILE,
        help="Local workflow profile to validate.",
    )
    parser.add_argument(
        "--with-vision",
        action="store_true",
        help="Also validate the optional local vision workflow.",
    )
    parser.add_argument(
        "--language",
        help=f"Language code for the local workflow, default {DEFAULT_LOCAL_LANGUAGE.value}.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=LOCAL_LLM_PROVIDERS,
        help="Local LLM provider to validate. Defaults to ollama.",
    )
    parser.add_argument("--model", help="Provider-relative LLM model override.")
    parser.add_argument("--base-url", help="Provider-relative LLM base URL override.")
    parser.add_argument(
        "--demo-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate the explicit local demo-mode LLM defaults.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Provider-relative output token budget to validate.",
    )
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
    parser.add_argument(
        "--camera-source",
        choices=CAMERA_SOURCES,
        help="Optional native voice camera source to validate.",
    )
    parser.add_argument(
        "--camera-helper-app",
        help="Path to BlinkCameraHelper.app for the macOS helper camera path.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, object]:
    """Resolve doctor settings from the CLI and environment."""
    maybe_load_dotenv()
    language = resolve_local_language(args.language or get_local_env("LANGUAGE"))
    configured_tts_backend = get_local_env("TTS_BACKEND")
    tts_backend_locked = args.tts_backend not in (None, "") or configured_tts_backend not in (
        None,
        "",
    )
    stt_backend = args.stt_backend or get_local_env("STT_BACKEND", DEFAULT_LOCAL_STT_BACKEND)
    tts_backend = args.tts_backend or configured_tts_backend or default_local_tts_backend(language)
    explicit_tts_voice = args.tts_voice if args.tts_voice not in (None, "") else None
    camera_source = (
        getattr(args, "camera_source", None) or get_local_env("CAMERA_SOURCE") or CAMERA_SOURCE_NONE
    )
    if camera_source not in CAMERA_SOURCES:
        camera_source = CAMERA_SOURCE_NONE
    camera_helper_app = getattr(args, "camera_helper_app", None) or get_local_env(
        "CAMERA_HELPER_APP"
    )

    stt_model = (
        args.stt_model
        or get_local_env("STT_MODEL")
        or default_local_stt_model(backend=stt_backend, language=language)
    )

    tts_voice = resolve_local_tts_voice(
        tts_backend,
        language,
        explicit_voice=explicit_tts_voice,
    )
    llm_config = resolve_local_llm_config(
        provider=args.llm_provider,
        model=args.model,
        base_url=args.base_url,
        system_prompt=None,
        language=language,
        demo_mode=getattr(args, "demo_mode", None),
        max_output_tokens=getattr(args, "max_output_tokens", None),
        speech=args.profile in {"voice", "browser", "full"},
    )

    with_vision = (
        args.with_vision
        or local_env_flag("BROWSER_VISION")
        or camera_source == CAMERA_SOURCE_MACOS_HELPER
    )
    if args.profile not in {"browser", "full"}:
        with_vision = camera_source == CAMERA_SOURCE_MACOS_HELPER

    return {
        "profile": args.profile,
        "with_vision": with_vision,
        "language": language,
        "llm_provider": llm_config.provider,
        "llm_config": llm_config,
        "base_url": llm_config.base_url,
        "model": llm_config.model,
        "stt_backend": stt_backend,
        "stt_model": stt_model,
        "tts_backend": tts_backend,
        "tts_voice": tts_voice,
        "tts_backend_locked": tts_backend_locked,
        "tts_voice_override": explicit_tts_voice,
        "tts_base_url": resolve_local_tts_base_url(tts_backend),
        "camera_source": camera_source,
        "camera_helper_app_path": Path(camera_helper_app).expanduser()
        if camera_helper_app not in (None, "")
        else None,
    }


def _command_exists(command: str) -> bool:
    return (
        subprocess.run(
            ["sh", "-lc", f"command -v {command} >/dev/null 2>&1"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _module_exists(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _brew_prefix(formula: str) -> Optional[str]:
    result = subprocess.run(["brew", "--prefix", formula], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _check_commands(
    profile: str,
    llm_provider: str = DEFAULT_LOCAL_LLM_PROVIDER,
) -> list[CheckResult]:
    results = [
        CheckResult(
            "uv",
            "PASS" if _command_exists("uv") else "FAIL",
            "uv is available." if _command_exists("uv") else "uv is not installed.",
            None if _command_exists("uv") else "Install uv from https://docs.astral.sh/uv/.",
        ),
    ]
    if llm_provider == "ollama":
        results.append(
            CheckResult(
                "ollama",
                "PASS" if _command_exists("ollama") else "FAIL",
                "Ollama is available." if _command_exists("ollama") else "Ollama is not installed.",
                None
                if _command_exists("ollama")
                else "Install Ollama from https://ollama.com/download.",
            )
        )

    needs_brew = profile in {"voice", "full"}
    if needs_brew:
        brew_installed = _command_exists("brew")
        results.append(
            CheckResult(
                "brew",
                "PASS" if brew_installed else "FAIL",
                "Homebrew is available."
                if brew_installed
                else "Homebrew is required for the native voice stack.",
                None if brew_installed else "Install Homebrew from https://brew.sh/.",
            )
        )
        if brew_installed:
            portaudio_prefix = _brew_prefix("portaudio")
            results.append(
                CheckResult(
                    "portaudio",
                    "PASS" if portaudio_prefix else "FAIL",
                    f"PortAudio is installed at {portaudio_prefix}."
                    if portaudio_prefix
                    else "PortAudio is not installed.",
                    None if portaudio_prefix else "Run `brew install portaudio`.",
                )
            )

    return results


def _check_modules(config: dict[str, object]) -> list[CheckResult]:
    profile = str(config["profile"])
    stt_backend = str(config["stt_backend"])
    tts_backend = str(config["tts_backend"])
    with_vision = bool(config["with_vision"])
    camera_source = str(config.get("camera_source", CAMERA_SOURCE_NONE))

    required_modules: list[tuple[str, str, str | None]] = [
        ("openai", "Base runtime dependency is installed.", None)
    ]
    if profile in {"voice", "full"}:
        required_modules.append(
            (
                "pyaudio",
                "PyAudio is available for native mic and speaker I/O.",
                "Run `./scripts/bootstrap-blink-mac.sh --profile voice` after installing PortAudio.",
            )
        )
    if profile in {"browser", "full"}:
        required_modules.extend(
            [
                (
                    "fastapi",
                    "FastAPI is available for the browser runner.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser`.",
                ),
                (
                    "uvicorn",
                    "uvicorn is available for the browser runner.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser`.",
                ),
                (
                    "aiortc",
                    "aiortc is available for SmallWebRTC.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser`.",
                ),
                (
                    "blink.web.client_source",
                    "The Blink browser UI client workspace is present in the checkout.",
                    "Restore `web/client_src/` from the repository.",
                ),
            ]
        )

    if profile in {"voice", "browser", "full"}:
        if stt_backend == "mlx-whisper":
            required_modules.append(
                (
                    "mlx_whisper",
                    "MLX Whisper is available.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile voice`.",
                )
            )
        else:
            required_modules.append(
                (
                    "faster_whisper",
                    "Faster Whisper is available.",
                    "Run `uv sync --python 3.12 --group dev --extra whisper`.",
                )
            )

        if tts_backend == "kokoro":
            required_modules.append(
                (
                    "kokoro_onnx",
                    "Kokoro is available.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile voice`.",
                )
            )
        elif tts_backend == "piper":
            required_modules.append(
                (
                    "piper",
                    "Piper is available.",
                    "Run `uv sync --python 3.12 --group dev --extra piper`.",
                )
            )

    if (with_vision and profile in {"browser", "full"}) or (
        camera_source == CAMERA_SOURCE_MACOS_HELPER
    ):
        required_modules.extend(
            [
                (
                    "torch",
                    "PyTorch is available for Moondream.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser --with-vision`.",
                ),
                (
                    "transformers",
                    "Transformers is available for Moondream.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser --with-vision`.",
                ),
                (
                    "pyvips",
                    "pyvips is available for Moondream.",
                    "Run `./scripts/bootstrap-blink-mac.sh --profile browser --with-vision`.",
                ),
            ]
        )

    results = []
    for module, detail, remedy in required_modules:
        exists = (
            Path.cwd().joinpath("web", "client_src", "src", "index.html").is_file()
            if module == "blink.web.client_source"
            else _module_exists(module)
        )
        results.append(
            CheckResult(
                module,
                "PASS" if exists else "FAIL",
                detail if exists else f"Python module `{module}` is missing.",
                None if exists else remedy,
            )
        )

    return results


def _check_macos_camera_helper(config: dict[str, object]) -> list[CheckResult]:
    """Check the helper app used by the native macOS camera path."""
    if str(config.get("camera_source", CAMERA_SOURCE_NONE)) != CAMERA_SOURCE_MACOS_HELPER:
        return []

    if sys.platform != "darwin":
        return [
            CheckResult(
                "blink-camera-helper",
                "FAIL",
                "BlinkCameraHelper.app is only supported on macOS.",
                "Use the camera-free native voice path or browser camera path on this platform.",
            )
        ]

    app_path = config.get("camera_helper_app_path") or _default_macos_camera_helper_app_path()
    app_path = Path(app_path)
    executable = app_path / "Contents" / "MacOS" / "BlinkCameraHelper"
    if executable.exists():
        return [
            CheckResult(
                "blink-camera-helper",
                "PASS",
                f"BlinkCameraHelper.app is available at {app_path}.",
                None,
            )
        ]

    return [
        CheckResult(
            "blink-camera-helper",
            "FAIL",
            f"BlinkCameraHelper.app is missing at {app_path}.",
            "Run `./scripts/build-macos-camera-helper.sh`.",
        )
    ]


async def _check_ollama(base_url: str, model: str) -> list[CheckResult]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
    except Exception:
        return [
            CheckResult(
                "ollama-server",
                "FAIL",
                f"Could not reach Ollama at {base_url}.",
                "Start it with `ollama serve`.",
            )
        ]

    data = response.json().get("data", [])
    ids = {item.get("id") for item in data if isinstance(item, dict)}
    if model not in ids:
        return [
            CheckResult(
                "ollama-model",
                "FAIL",
                f"Ollama is reachable, but model `{model}` is not installed.",
                f"Run `ollama pull {model}`.",
            )
        ]

    return [
        CheckResult("ollama-server", "PASS", f"Ollama is reachable at {base_url}."),
        CheckResult("ollama-model", "PASS", f"Ollama model `{model}` is installed."),
    ]


def _check_openai_responses(config: dict[str, object]) -> list[CheckResult]:
    llm_config = config.get("llm_config")
    if not isinstance(llm_config, LocalLLMConfig):
        return [
            CheckResult(
                "openai-responses-config",
                "FAIL",
                "OpenAI Responses configuration is missing.",
                "Rerun doctor with `--llm-provider openai-responses`.",
            )
        ]

    results: list[CheckResult] = []
    try:
        verify_openai_responses_config(llm_config)
    except RuntimeError:
        return [
            CheckResult(
                "openai-api-key",
                "FAIL",
                "OPENAI_API_KEY is not set.",
                "Set OPENAI_API_KEY in your shell or ignored `.env` file.",
            )
        ]
    except Exception:
        return [
            CheckResult(
                "openai-responses-service",
                "FAIL",
                "OpenAI Responses service dependencies are unavailable.",
                "Run `uv sync --python 3.12 --group dev` and verify OpenAI/websocket packages.",
            )
        ]

    results.append(CheckResult("openai-api-key", "PASS", "OPENAI_API_KEY is set."))
    results.append(
        CheckResult(
            "openai-responses-model",
            "PASS",
            f"OpenAI Responses model `{llm_config.model}` is configured.",
        )
    )
    if llm_config.base_url:
        results.append(
            CheckResult(
                "openai-responses-base-url",
                "PASS",
                f"OpenAI Responses base URL override is configured: {llm_config.base_url}.",
            )
        )
    if llm_config.service_tier:
        results.append(
            CheckResult(
                "openai-responses-service-tier",
                "PASS",
                f"OpenAI Responses service tier `{llm_config.service_tier}` is configured.",
            )
        )

    if llm_config.demo_mode:
        detail = "OpenAI Responses demo mode is enabled."
        if llm_config.max_output_tokens is not None:
            detail += f" max_output_tokens={llm_config.max_output_tokens}."
        results.append(
            CheckResult(
                "openai-responses-demo-mode",
                "PASS",
                detail,
                "Use `BLINK_LOCAL_DEMO_MODE=0` or `--no-demo-mode` for normal local behavior.",
            )
        )

    if str(config.get("profile", "text")) in {"voice", "browser", "full"}:
        results.append(
            CheckResult(
                "openai-responses-hybrid-scope",
                "PASS",
                "Hybrid lane is provider-selectable for the LLM layer while STT and TTS stay local.",
                "Use the OpenAI wrapper scripts for demo runs, or pass `--llm-provider openai-responses` explicitly.",
            )
        )
    return results


async def _check_llm_provider(config: dict[str, object]) -> list[CheckResult]:
    provider = str(config.get("llm_provider") or DEFAULT_LOCAL_LLM_PROVIDER)
    if provider == "ollama":
        return await _check_ollama(str(config["base_url"]), str(config["model"]))
    if provider == "openai-responses":
        return _check_openai_responses(config)
    return [
        CheckResult(
            "llm-provider",
            "FAIL",
            f"Unsupported local LLM provider `{provider}`.",
            "Use `ollama` or `openai-responses`.",
        )
    ]


async def _check_tts_backend(config: dict[str, object]) -> list[CheckResult]:
    backend = str(config["tts_backend"])
    base_url = config.get("tts_base_url")
    language = config.get("language", Language.EN)
    if not isinstance(language, (Language, str)):
        language = Language.EN

    if backend == "xtts":
        xtts_base_url = str(base_url or default_local_tts_base_url("xtts"))
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{xtts_base_url.rstrip('/')}/studio_speakers")
                response.raise_for_status()
        except Exception:
            return [
                CheckResult(
                    "xtts-server",
                    "FAIL",
                    f"Could not reach XTTS at {xtts_base_url}.",
                    "Start a local XTTS-compatible server or switch to `local-http-wav`.",
                )
            ]

        speakers = response.json()
        if not isinstance(speakers, dict) or not speakers:
            return [
                CheckResult(
                    "xtts-speakers",
                    "FAIL",
                    "XTTS is reachable but `/studio_speakers` returned no speakers.",
                    "Load speakers into XTTS or set `--tts-voice`, "
                    "`BLINK_LOCAL_TTS_VOICE_ZH`, `BLINK_LOCAL_TTS_VOICE_EN`, "
                    "or `BLINK_LOCAL_TTS_VOICE` explicitly.",
                )
            ]

        return [
            CheckResult("xtts-server", "PASS", f"XTTS is reachable at {xtts_base_url}."),
            CheckResult(
                "xtts-speakers",
                "PASS",
                f"XTTS exposes {len(speakers)} speaker(s).",
            ),
        ]

    if backend == "local-http-wav":
        local_tts_base_url = str(base_url or default_local_tts_base_url("local-http-wav"))
        configured_voice = config.get("tts_voice")
        normalized_voice = configured_voice
        warning_result: Optional[CheckResult] = None
        catalog: Optional[dict[str, object]] = None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                voices_response = await client.get(f"{local_tts_base_url.rstrip('/')}/voices")
                if voices_response.status_code == 200:
                    voices_payload = voices_response.json()
                    if isinstance(voices_payload, dict):
                        catalog = voices_payload
        except Exception:
            catalog = None

        if catalog is not None and configured_voice:
            fallback_voice = local_http_wav_fallback_speaker(
                catalog,
                configured_voice=str(configured_voice),
                language=language,
            )
            available_speakers = local_http_wav_speakers_for_language(catalog, language)
            if fallback_voice != configured_voice:
                normalized_voice = fallback_voice
                remedy = (
                    f"{PROJECT_IDENTITY.display_name} will fall back to `{fallback_voice}`."
                    if fallback_voice
                    else f"{PROJECT_IDENTITY.display_name} will fall back to the sidecar default speaker."
                )
                remedy += " Use `--tts-voice` or update `/voices` if you need a specific speaker."
                if available_speakers:
                    remedy += f" Advertised speakers: {', '.join(available_speakers)}."
                warning_result = CheckResult(
                    "local-http-wav-speaker",
                    "WARN",
                    f"Configured speaker `{configured_voice}` is not advertised for this language.",
                    remedy,
                )
        payload = {
            "text": "测试中文语音。"
            if is_chinese_language(language if isinstance(language, Language) else Language.EN)
            else "Testing local speech.",
            "speaker": normalized_voice,
            "language": language,
            "model": None,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{local_tts_base_url.rstrip('/')}/tts", json=payload)
        except Exception:
            return [
                CheckResult(
                    "local-http-wav",
                    "FAIL",
                    f"Could not reach the local HTTP WAV TTS server at {local_tts_base_url}.",
                    "Start your `/tts` server, for example `./scripts/run-melotts-reference-server.sh` "
                    "or `./scripts/run-local-cosyvoice-adapter.sh`, or set `BLINK_LOCAL_TTS_BACKEND=xtts`.",
                )
            ]

        if response.status_code != 200:
            return [
                CheckResult(
                    "local-http-wav",
                    "FAIL",
                    f"Local HTTP WAV TTS returned HTTP {response.status_code}: {response.text}.",
                    "Verify that POST /tts accepts {text, speaker, language, model}, "
                    "that `/voices` advertises the configured speaker, and that the sidecar "
                    "returns WAV bytes.",
                )
            ]

        if not response.content.startswith(b"RIFF"):
            return [
                CheckResult(
                    "local-http-wav",
                    "FAIL",
                    "Local HTTP WAV TTS did not return WAV bytes.",
                    "Verify that POST /tts responds with a WAV body.",
                )
            ]

        results = [
            CheckResult(
                "local-http-wav",
                "PASS",
                f"Local HTTP WAV TTS is reachable at {local_tts_base_url}.",
            )
        ]
        if warning_result is not None:
            results.insert(0, warning_result)
        return results

    return []


def _check_audio_devices(profile: str) -> list[CheckResult]:
    if profile not in {"voice", "full"}:
        return []

    if not _module_exists("pyaudio"):
        return []

    try:
        devices = get_audio_devices()
    except Exception as exc:
        return [
            CheckResult(
                "audio-devices",
                "FAIL",
                f"PyAudio is installed, but device enumeration failed: {exc}",
                "Verify microphone and speaker access in macOS system settings.",
            )
        ]

    if not devices:
        return [
            CheckResult(
                "audio-devices",
                "WARN",
                "No audio devices were detected.",
                "Connect a microphone or speaker and rerun `blink-local-doctor`.",
            )
        ]

    results = [
        CheckResult(
            "audio-devices",
            "PASS",
            f"Detected {len(devices)} audio device(s).",
        )
    ]

    default_input = get_default_audio_device("input")
    default_output = get_default_audio_device("output")
    preferred_input, preferred_output = resolve_preferred_audio_device_indexes(None, None)

    if (
        default_input
        and default_output
        and preferred_input is not None
        and preferred_output is not None
        and (preferred_input != default_input.index or preferred_output != default_output.index)
    ):
        results.append(
            CheckResult(
                "audio-routing",
                "WARN",
                "macOS default audio is routed through external display audio. "
                f"{PROJECT_IDENTITY.display_name} local voice will auto-prefer your built-in MacBook devices.",
                "Override with `--input-device` and `--output-device` if you want a different route.",
            )
        )

    return results


def _check_model_caches(config: dict[str, object]) -> list[CheckResult]:
    results: list[CheckResult] = []
    profile = str(config["profile"])
    language = config.get("language", Language.EN)
    if not isinstance(language, Language):
        language = Language.EN
    stt_model = str(config["stt_model"])
    tts_backend = str(config["tts_backend"])
    tts_voice = str(config["tts_voice"] or "")

    if profile in {"voice", "browser", "full"}:
        stt_cached = model_cache_exists(stt_model)
        results.append(
            CheckResult(
                "stt-model-cache",
                "PASS" if stt_cached else "WARN",
                f"Speech model cache exists at {huggingface_model_cache_dir(stt_model)}."
                if stt_cached
                else f"Speech model `{stt_model}` is not cached yet.",
                None
                if stt_cached
                else "Pre-download it with `./scripts/prefetch-blink-assets.sh --profile voice`.",
            )
        )

        if tts_backend == "kokoro":
            tts_cached = kokoro_assets_present()
            results.append(
                CheckResult(
                    "tts-assets",
                    "PASS" if tts_cached else "WARN",
                    "Kokoro model assets are already cached."
                    if tts_cached
                    else "Kokoro model assets are not cached yet.",
                    None
                    if tts_cached
                    else "Pre-download them with `./scripts/prefetch-blink-assets.sh --profile voice`.",
                )
            )
        elif tts_backend == "piper":
            piper_model_path = default_piper_model_path(tts_voice)
            tts_cached = piper_model_path.exists()
            results.append(
                CheckResult(
                    "tts-assets",
                    "PASS" if tts_cached else "WARN",
                    f"Piper voice cache exists at {piper_model_path}."
                    if tts_cached
                    else f"Piper voice `{tts_voice}` is not cached yet.",
                    None
                    if tts_cached
                    else "Pre-download it with `./scripts/prefetch-blink-assets.sh --profile voice`.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "tts-assets",
                    "PASS",
                    "The selected TTS backend uses a local HTTP server, so there is no in-repo asset cache to verify.",
                )
            )

        if is_chinese_language(language) and tts_backend == "kokoro":
            results.append(
                CheckResult(
                    "mandarin-tts-quality",
                    "WARN",
                    "The default Kokoro Mandarin path is valid for local bootstrap and debugging, "
                    "but it is no longer the target Chinese-quality path.",
                    "Keep this path for local iteration, and use MeloTTS via `local-http-wav` if you need a stronger Mandarin voice backend.",
                )
            )

    if bool(config["with_vision"]):
        vision_cached = model_cache_exists(DEFAULT_LOCAL_VISION_MODEL)
        results.append(
            CheckResult(
                "vision-model-cache",
                "PASS" if vision_cached else "WARN",
                f"Moondream cache exists at {huggingface_model_cache_dir(DEFAULT_LOCAL_VISION_MODEL)}."
                if vision_cached
                else f"Moondream model `{DEFAULT_LOCAL_VISION_MODEL}` is not cached yet.",
                None
                if vision_cached
                else "Pre-download it with `./scripts/prefetch-blink-assets.sh --profile browser --with-vision`.",
            )
        )

    smart_turn_exists = resources.files("blink.audio.turn.smart_turn.data").joinpath(
        "smart-turn-v3.2-cpu.onnx"
    )
    results.append(
        CheckResult(
            "smart-turn-model",
            "PASS" if smart_turn_exists.is_file() else "FAIL",
            "Bundled Smart Turn v3 model is available."
            if smart_turn_exists.is_file()
            else "Bundled Smart Turn v3 model is missing from the package.",
            None if smart_turn_exists.is_file() else "Reinstall the package in this repo checkout.",
        )
    )

    return results


def _check_visual_pipeline(config: dict[str, object]) -> list[CheckResult]:
    """Inspect browser visual-presence dependencies and optional enrichment readiness."""
    profile = str(config["profile"])
    if profile not in {"browser", "full"}:
        return []

    results: list[CheckResult] = []
    try:
        from blink.brain.perception.detector import OnnxFacePresenceDetector

        detector = OnnxFacePresenceDetector()
        detector_ready = detector.available
        results.append(
            CheckResult(
                "presence-detector",
                "PASS" if detector_ready else "FAIL",
                "Packaged ONNX face detector is ready for browser presence detection."
                if detector_ready
                else "Packaged ONNX face detector could not be loaded.",
                None
                if detector_ready
                else "Reinstall the package and verify the packaged ONNX detector asset is present.",
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                "presence-detector",
                "FAIL",
                f"Could not initialize the packaged ONNX face detector: {exc}",
                "Reinstall the package in this checkout and verify `onnxruntime` is importable.",
            )
        )

    if bool(config["with_vision"]):
        enrichment_ready = all(
            _module_exists(module) for module in ("torch", "transformers", "pyvips")
        )
        results.append(
            CheckResult(
                "vision-enrichment",
                "PASS" if enrichment_ready else "FAIL",
                "Optional VLM enrichment dependencies are available."
                if enrichment_ready
                else "Optional VLM enrichment dependencies are incomplete.",
                None
                if enrichment_ready
                else "Run `./scripts/bootstrap-blink-mac.sh --profile browser --with-vision`.",
            )
        )
    else:
        results.append(
            CheckResult(
                "vision-enrichment",
                "PASS",
                "Optional VLM enrichment is disabled; deterministic presence detection remains available.",
            )
        )
    return results


def _check_runtime_visual_state(config: dict[str, object]) -> list[CheckResult]:
    """Report the latest browser visual-health state if a brain DB already exists."""
    profile = str(config["profile"])
    if profile not in {"browser", "full"}:
        return []

    default_db_path = Path.home() / ".cache" / "blink" / "brain" / "brain.db"
    db_path = Path(get_local_env("BRAIN_DB_PATH") or str(default_db_path))
    if not db_path.exists():
        return []

    try:
        from blink.brain.store import BrainStore

        store = BrainStore(path=db_path)
        try:
            body = store.get_body_state_projection(scope_key="browser:presence")
            scene = store.get_scene_state_projection(scope_key="browser:presence")
        finally:
            store.close()
    except Exception as exc:
        return [
            CheckResult(
                "browser-visual-state",
                "WARN",
                f"Could not inspect the latest browser visual state: {exc}",
                "Run `blink-local-brain-audit` for a full continuity report after the browser runtime starts.",
            )
        ]

    if (
        not body.vision_enabled
        and body.last_fresh_frame_at is None
        and scene.last_observed_at is None
        and scene.last_visual_summary is None
    ):
        return []

    updated_at = _parse_utc_ts(body.updated_at)
    state_age_secs = (
        max(0, int((datetime.now(UTC) - updated_at).total_seconds()))
        if updated_at is not None
        else None
    )
    state_recency = (
        "unknown"
        if state_age_secs is None
        else "current"
        if state_age_secs <= 120
        else "historical"
    )
    historical_snapshot = state_recency == "historical"
    clean_disconnected = (
        body.vision_enabled
        and not body.vision_connected
        and body.camera_track_state == "disconnected"
        and body.sensor_health_reason == "camera_disconnected"
    )
    current_degraded = (
        body.sensor_health != "healthy" and not clean_disconnected and not historical_snapshot
    )
    status = "WARN" if current_degraded else "PASS"
    field_prefix = "historical_" if historical_snapshot else ""
    snapshot_label = "historical_not_live" if historical_snapshot else state_recency
    detail = (
        f"snapshot={snapshot_label}, "
        f"{field_prefix}vision_connected={body.vision_connected}, "
        f"{field_prefix}track={body.camera_track_state}, "
        f"{field_prefix}sensor_health={body.sensor_health}, "
        f"{field_prefix}reason={body.sensor_health_reason or 'none'}, "
        f"{field_prefix}last_fresh_frame={body.last_fresh_frame_at or 'none'}, "
        f"updated_at={body.updated_at or 'none'}."
    )
    remedy = None
    if status != "PASS":
        remedy = (
            "Inspect the latest `blink-local-brain-audit` report and browser logs for "
            "camera stall, stale-frame, or detector issues."
        )
    return [CheckResult("browser-visual-state", status, detail, remedy)]


async def run_doctor(config: dict[str, object]) -> int:
    """Run the local diagnostics and print a human-readable report."""
    results: list[CheckResult] = []
    profile = str(config["profile"])
    effective_config = dict(config)

    print(
        f"{PROJECT_IDENTITY.display_name} local doctor for profile `{profile}` on {platform.platform()}"
    )
    print("")

    if profile in {"voice", "browser", "full"}:
        selection = await resolve_local_runtime_tts_selection(
            language=effective_config["language"]
            if isinstance(effective_config["language"], Language)
            else Language.EN,
            requested_backend=str(effective_config["tts_backend"]),
            requested_voice=(
                str(effective_config["tts_voice"])
                if effective_config.get("tts_voice") is not None
                else None
            ),
            requested_base_url=(
                str(effective_config["tts_base_url"])
                if effective_config.get("tts_base_url") is not None
                else None
            ),
            backend_locked=bool(effective_config.get("tts_backend_locked", False)),
            explicit_voice=(
                str(effective_config["tts_voice_override"])
                if effective_config.get("tts_voice_override") is not None
                else None
            ),
        )
        if selection.auto_switched:
            results.append(
                CheckResult(
                    "chinese-runtime-tts-profile",
                    "PASS",
                    "Chinese voice runtime will prefer `local-http-wav` when the sidecar is available.",
                    f"{PROJECT_IDENTITY.display_name} will keep Kokoro as the fallback path if "
                    "the sidecar is unavailable. Set `BLINK_LOCAL_TTS_BACKEND=kokoro` or pass "
                    "`--tts-backend kokoro` to pin the bootstrap path.",
                )
            )
            effective_config["tts_backend"] = selection.backend
            effective_config["tts_voice"] = selection.voice
            effective_config["tts_base_url"] = selection.base_url

    results.extend(_check_commands(profile, str(effective_config.get("llm_provider", "ollama"))))
    results.extend(_check_modules(effective_config))
    results.extend(_check_macos_camera_helper(effective_config))
    results.extend(await _check_llm_provider(effective_config))
    if profile in {"voice", "browser", "full"}:
        results.extend(await _check_tts_backend(effective_config))
    results.extend(_check_audio_devices(profile))
    results.extend(_check_model_caches(effective_config))
    results.extend(_check_visual_pipeline(effective_config))
    results.extend(_check_runtime_visual_state(effective_config))

    has_failures = False
    for result in results:
        print(f"{result.status:4} {result.name}: {result.detail}")
        if result.remedy:
            print(f"      fix: {result.remedy}")
        if result.status == "FAIL":
            has_failures = True

    return 1 if has_failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local doctor command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)
    return asyncio.run(run_doctor(config))


if __name__ == "__main__":
    raise SystemExit(main())

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A local CosyVoice-to-WAV adapter for Blink's `local-http-wav` seam."""

import argparse
import asyncio
import audioop
import io
import os
import re
import wave
from dataclasses import dataclass
from typing import Optional

import aiohttp
from aiohttp import web

from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    is_chinese_language,
    maybe_load_dotenv,
    resolve_local_language,
)
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language

DEFAULT_COSYVOICE_BACKEND_URL = "http://127.0.0.1:50000"
DEFAULT_COSYVOICE_ADAPTER_HOST = "127.0.0.1"
DEFAULT_COSYVOICE_ADAPTER_PORT = 8001
DEFAULT_COSYVOICE_MODE = "sft"
DEFAULT_COSYVOICE_TIMEOUT_SECS = 60.0
DEFAULT_COSYVOICE_BACKEND_SAMPLE_RATE = 22050
DEFAULT_COSYVOICE_OUTPUT_SAMPLE_RATE = 24000
DEFAULT_COSYVOICE_ZH_SPEAKER = "中文女"


def _env_alias(name: str) -> Optional[str]:
    """Return a Blink CosyVoice env var when it is set."""
    return os.getenv(f"BLINK_LOCAL_COSYVOICE_{name}")


@dataclass(frozen=True)
class CosyVoiceAdapterConfig:
    """Configuration for the optional local CosyVoice adapter."""

    host: str
    port: int
    backend_url: str
    mode: str
    zh_speaker: Optional[str]
    en_speaker: Optional[str]
    zh_instruct_text: Optional[str]
    en_instruct_text: Optional[str]
    backend_sample_rate: int
    output_sample_rate: int
    timeout_secs: float


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the local CosyVoice adapter."""
    parser = argparse.ArgumentParser(
        description=(
            f"Expose {PROJECT_IDENTITY.display_name}'s POST /tts local-http-wav contract by proxying to "
            "a separately running CosyVoice FastAPI server."
        )
    )
    parser.add_argument("--host", help="Host for the adapter server.")
    parser.add_argument("--port", type=int, help="Port for the adapter server.")
    parser.add_argument(
        "--backend-url",
        help="Base URL of the external CosyVoice FastAPI server.",
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "instruct"],
        help="CosyVoice request mode. `sft` is the default stable path.",
    )
    parser.add_argument("--speaker-zh", help="Default Mandarin speaker id.")
    parser.add_argument("--speaker-en", help="Optional default English speaker id.")
    parser.add_argument(
        "--instruct-text-zh",
        help="Optional Mandarin instruct text used when --mode=instruct.",
    )
    parser.add_argument(
        "--instruct-text-en",
        help="Optional English instruct text used when --mode=instruct.",
    )
    parser.add_argument(
        "--backend-sample-rate",
        type=int,
        help="Sample rate returned by the CosyVoice backend PCM stream.",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        help="Output WAV sample rate returned to Blink.",
    )
    parser.add_argument(
        "--timeout-secs",
        type=float,
        help="Timeout for backend CosyVoice requests.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> CosyVoiceAdapterConfig:
    """Resolve adapter config from CLI arguments and environment."""
    maybe_load_dotenv()
    return CosyVoiceAdapterConfig(
        host=args.host or _env_alias("ADAPTER_HOST") or DEFAULT_COSYVOICE_ADAPTER_HOST,
        port=args.port
        or int(_env_alias("ADAPTER_PORT") or DEFAULT_COSYVOICE_ADAPTER_PORT),
        backend_url=(
            args.backend_url
            or _env_alias("BACKEND_URL")
            or DEFAULT_COSYVOICE_BACKEND_URL
        ).rstrip("/"),
        mode=args.mode or _env_alias("MODE") or DEFAULT_COSYVOICE_MODE,
        zh_speaker=args.speaker_zh
        or _env_alias("SPEAKER_ZH")
        or DEFAULT_COSYVOICE_ZH_SPEAKER,
        en_speaker=args.speaker_en or _env_alias("SPEAKER_EN") or None,
        zh_instruct_text=args.instruct_text_zh
        or _env_alias("INSTRUCT_TEXT_ZH")
        or None,
        en_instruct_text=args.instruct_text_en
        or _env_alias("INSTRUCT_TEXT_EN")
        or None,
        backend_sample_rate=args.backend_sample_rate
        or int(_env_alias("BACKEND_SAMPLE_RATE") or DEFAULT_COSYVOICE_BACKEND_SAMPLE_RATE),
        output_sample_rate=args.output_sample_rate
        or int(_env_alias("OUTPUT_SAMPLE_RATE") or DEFAULT_COSYVOICE_OUTPUT_SAMPLE_RATE),
        timeout_secs=args.timeout_secs
        or float(_env_alias("TIMEOUT_SECS") or DEFAULT_COSYVOICE_TIMEOUT_SECS),
    )


def normalize_cosyvoice_text(text: str, language: Language) -> str:
    """Normalize incoming text before forwarding it to CosyVoice.

    This keeps the adapter stronger than the thin local `local-http-wav` seam
    without coupling Blink core runtime to a specific Chinese frontend.
    """
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = text.replace("```", " ").replace("`", " ")
    text = re.sub(r"(?<!\*)\*\*(.+?)\*\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!\*)\*(.+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)__(.+?)__(?!_)", r"\1", text)
    text = re.sub(r"(?<!_)_(.+?)_(?!_)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()

    if is_chinese_language(language):
        text = text.translate(
            str.maketrans(
                {
                    ",": "，",
                    ";": "；",
                    ":": "：",
                    "!": "！",
                    "?": "？",
                    "(": "（",
                    ")": "）",
                }
            )
        )
        text = text.replace("...", "，")
        text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
        text = text.replace("/", "，")
        text = text.replace("\\", "，")
        text = text.replace("~", "到")
        if text and text[-1] not in "。！？；）":
            text += "。"
    else:
        text = re.sub(r"\s*([,;:!?])\s*", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[-1] not in ".!?":
            text += "."

    return text


def _default_speaker_for_language(
    config: CosyVoiceAdapterConfig, language: Language
) -> Optional[str]:
    if is_chinese_language(language):
        return config.zh_speaker
    return config.en_speaker


def _default_instruct_text(config: CosyVoiceAdapterConfig, language: Language) -> Optional[str]:
    if is_chinese_language(language):
        return config.zh_instruct_text
    return config.en_instruct_text


def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> bytes:
    if input_sample_rate != output_sample_rate:
        pcm_bytes, _ = audioop.ratecv(
            pcm_bytes,
            2,
            1,
            input_sample_rate,
            output_sample_rate,
            None,
        )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(output_sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


async def _proxy_tts_request(
    session: aiohttp.ClientSession,
    config: CosyVoiceAdapterConfig,
    *,
    text: str,
    speaker: str,
    language: Language,
) -> bytes:
    normalized_text = normalize_cosyvoice_text(text, language)
    data = aiohttp.FormData()
    data.add_field("tts_text", normalized_text)
    data.add_field("spk_id", speaker)

    endpoint = "/inference_sft"
    if config.mode == "instruct":
        instruct_text = _default_instruct_text(config, language)
        if not instruct_text:
            raise web.HTTPBadRequest(
                text=(
                    "CosyVoice adapter is in instruct mode but no instruct text is configured "
                    f"for language `{language.value}`."
                )
            )
        data.add_field("instruct_text", instruct_text)
        endpoint = "/inference_instruct"

    async with session.post(
        f"{config.backend_url}{endpoint}",
        data=data,
        timeout=aiohttp.ClientTimeout(total=config.timeout_secs),
    ) as response:
        if response.status != 200:
            detail = await response.text()
            raise web.HTTPBadGateway(
                text=f"CosyVoice backend returned HTTP {response.status}: {detail}"
            )
        return await response.read()


def create_app(config: CosyVoiceAdapterConfig) -> web.Application:
    """Create the aiohttp app that exposes Blink's `/tts` WAV contract."""
    app = web.Application()
    app["config"] = config
    app["http_session"] = None

    async def on_startup(app: web.Application):
        app["http_session"] = aiohttp.ClientSession()

    async def on_cleanup(app: web.Application):
        session = app["http_session"]
        if session is not None:
            await session.close()

    async def healthz(_request: web.Request) -> web.Response:
        session: aiohttp.ClientSession = app["http_session"]
        backend_ok = False
        try:
            async with session.get(
                f"{config.backend_url}/docs",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                backend_ok = response.status == 200
        except Exception:
            backend_ok = False

        payload = {
            "status": "ok" if backend_ok else "degraded",
            "backend_url": config.backend_url,
            "backend_reachable": backend_ok,
            "mode": config.mode,
            "output_sample_rate": config.output_sample_rate,
        }
        return web.json_response(payload, status=200 if backend_ok else 503)

    async def voices(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "mode": config.mode,
                "default_speakers": {
                    "zh": config.zh_speaker,
                    "en": config.en_speaker,
                },
                "default_instruct_text": {
                    "zh": config.zh_instruct_text,
                    "en": config.en_instruct_text,
                },
            }
        )

    async def tts(request: web.Request) -> web.Response:
        payload = await request.json()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text="`text` is required.")

        language = resolve_local_language(payload.get("language") or DEFAULT_LOCAL_LANGUAGE)
        speaker = payload.get("speaker") or _default_speaker_for_language(config, language)
        if not speaker:
            env_suffix = "ZH" if is_chinese_language(language) else "EN"
            raise web.HTTPBadRequest(
                text=(
                    "No speaker configured for this language. Set `speaker` in the request or "
                    f"configure BLINK_LOCAL_COSYVOICE_SPEAKER_{env_suffix}."
                )
            )

        session: aiohttp.ClientSession = app["http_session"]
        pcm_bytes = await _proxy_tts_request(
            session,
            config,
            text=text,
            speaker=str(speaker),
            language=language,
        )
        wav_bytes = _wrap_pcm_as_wav(
            pcm_bytes,
            input_sample_rate=config.backend_sample_rate,
            output_sample_rate=config.output_sample_rate,
        )
        response = web.Response(body=wav_bytes, content_type="audio/wav")
        response.headers["X-Blink-TTS-Backend"] = "cosyvoice-adapter"
        response.headers["X-Blink-TTS-Speaker"] = str(speaker)
        response.headers["X-Blink-TTS-Language"] = language.value
        return response

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/voices", voices)
    app.router.add_post("/tts", tts)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def run_server(config: CosyVoiceAdapterConfig) -> int:
    """Run the local CosyVoice adapter server."""
    app = create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()
    print(
        "Blink CosyVoice local-http-wav adapter is available at "
        f"http://{config.host}:{config.port}/tts"
    )
    print(f"Proxying CosyVoice backend: {config.backend_url} (mode={config.mode})")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local CosyVoice adapter."""
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    try:
        return asyncio.run(run_server(config))
    except KeyboardInterrupt:
        return 130


__all__ = [
    "CosyVoiceAdapterConfig",
    "DEFAULT_COSYVOICE_ADAPTER_PORT",
    "DEFAULT_COSYVOICE_BACKEND_URL",
    "build_parser",
    "create_app",
    "main",
    "normalize_cosyvoice_text",
    "resolve_config",
    "run_server",
]

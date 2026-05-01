"""Repo-local MeloTTS HTTP-WAV server for Blink's `local-http-wav` seam."""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from aiohttp import web

DEFAULT_MELO_HOST = "127.0.0.1"
DEFAULT_MELO_PORT = 8001
DEFAULT_MELO_DEVICE = "cpu"
DEFAULT_MELO_OUTPUT_SAMPLE_RATE = 24_000
DEFAULT_MELO_SPEED = 1.0
DEFAULT_MELO_SPEAKER_ZH = "ZH"
DEFAULT_MELO_SPEAKER_EN = "EN-US"
MELO_RECOMMENDED_PYTHON = "3.11"
MELO_TARGET_PEAK = 0.82


def _env_value(primary_name: str, default: str) -> str:
    if primary_name in os.environ:
        return os.environ[primary_name]
    return default


@dataclass(frozen=True)
class MeloTTSServerConfig:
    """Configuration for the repo-local MeloTTS HTTP-WAV sidecar."""

    host: str
    port: int
    device: str
    output_sample_rate: int
    speed: float
    zh_speaker: str
    en_speaker: str


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Serve MeloTTS behind Blink's local HTTP WAV `/tts` contract."
    )
    parser.add_argument("--host", help="Host for the MeloTTS HTTP server.")
    parser.add_argument("--port", type=int, help="Port for the MeloTTS HTTP server.")
    parser.add_argument("--device", help="MeloTTS device, for example `cpu`.")
    parser.add_argument(
        "--speaker-zh",
        help="Default Chinese MeloTTS speaker id.",
    )
    parser.add_argument(
        "--speaker-en",
        help="Default English MeloTTS speaker id.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="Default synthesis speed multiplier.",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        help="Output WAV sample rate in Hz.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> MeloTTSServerConfig:
    """Resolve server config from CLI flags and environment variables."""
    return MeloTTSServerConfig(
        host=args.host
        or _env_value("BLINK_LOCAL_MELO_HOST", DEFAULT_MELO_HOST),
        port=args.port
        or int(_env_value("BLINK_LOCAL_MELO_PORT", str(DEFAULT_MELO_PORT))),
        device=args.device
        or _env_value("BLINK_LOCAL_MELO_DEVICE", DEFAULT_MELO_DEVICE),
        output_sample_rate=args.output_sample_rate
        or int(_env_value("BLINK_LOCAL_MELO_OUTPUT_SAMPLE_RATE", str(DEFAULT_MELO_OUTPUT_SAMPLE_RATE))),
        speed=args.speed
        or float(_env_value("BLINK_LOCAL_MELO_SPEED", str(DEFAULT_MELO_SPEED))),
        zh_speaker=args.speaker_zh
        or _env_value("BLINK_LOCAL_MELO_SPEAKER_ZH", DEFAULT_MELO_SPEAKER_ZH)
        or DEFAULT_MELO_SPEAKER_ZH,
        en_speaker=args.speaker_en
        or _env_value("BLINK_LOCAL_MELO_SPEAKER_EN", DEFAULT_MELO_SPEAKER_EN)
        or DEFAULT_MELO_SPEAKER_EN,
    )


def normalize_language(value: Any) -> tuple[str, str]:
    """Normalize an incoming language value to session and Melo codes."""
    normalized = str(value or "zh").replace("_", "-").lower()
    if normalized.startswith(("zh", "cmn")):
        return "zh", "ZH"
    if normalized.startswith("en"):
        return "en", "EN"
    raise web.HTTPBadRequest(
        text=f"Unsupported MeloTTS language `{value}`. Use a Chinese or English session."
    )


def default_speaker_for_language(config: MeloTTSServerConfig, session_language: str) -> str:
    """Resolve the default speaker id for the normalized session language."""
    if session_language == "zh":
        return config.zh_speaker
    return config.en_speaker


def _load_melo_factory() -> Callable[..., Any]:
    try:
        from melo.api import TTS
    except ImportError as exc:  # pragma: no cover - depends on external bootstrap
        raise RuntimeError(
            "MeloTTS is not installed in this environment. "
            "Run `./scripts/bootstrap-melotts-reference.sh` first."
        ) from exc

    return TTS


def create_model_factory(factory: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """Create or return the MeloTTS factory used by the server."""
    return factory or _load_melo_factory()


def get_model_speakers(model: Any) -> dict[str, Any]:
    """Return the speaker mapping advertised by a MeloTTS model."""
    speaker_map = getattr(getattr(getattr(model, "hps", None), "data", None), "spk2id", None)
    if isinstance(speaker_map, dict):
        return speaker_map
    if hasattr(speaker_map, "items"):
        return dict(speaker_map.items())
    if hasattr(speaker_map, "__dict__"):
        values = dict(vars(speaker_map))
        if values:
            return values
    raise RuntimeError("MeloTTS model does not expose a usable `hps.data.spk2id` mapping.")


def get_or_create_model(
    *,
    config: MeloTTSServerConfig,
    model_factory: Callable[..., Any],
    cache: dict[tuple[str, str], Any],
    melo_language: str,
) -> Any:
    """Return a cached MeloTTS model for the requested language and device."""
    key = (melo_language, config.device)
    if key not in cache:
        cache[key] = model_factory(language=melo_language, device=config.device)
    return cache[key]


def wrap_pcm_as_wav(
    pcm_bytes: bytes,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    channels: int = 1,
) -> bytes:
    """Wrap and optionally resample 16-bit PCM as mono WAV bytes."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if input_sample_rate != output_sample_rate:
        input_positions = np.arange(samples.shape[0], dtype=np.float32)
        target_length = max(1, round(samples.shape[0] * output_sample_rate / input_sample_rate))
        target_positions = np.linspace(
            0.0,
            max(float(samples.shape[0] - 1), 0.0),
            num=target_length,
            dtype=np.float32,
        )
        samples = np.interp(target_positions, input_positions, samples)

    samples = normalize_pcm_peak(samples)

    mono_pcm = np.clip(
        np.rint(samples),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(output_sample_rate)
        wav_file.writeframes(mono_pcm.tobytes())
    return wav_buffer.getvalue()


def normalize_pcm_peak(
    samples: np.ndarray,
    *,
    target_peak: float = MELO_TARGET_PEAK,
) -> np.ndarray:
    """Normalize Melo output peak level without changing silence or raw payloads."""
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak < 64.0:
        return samples
    bounded_target = max(0.1, min(float(target_peak), 0.95))
    target = float(np.iinfo(np.int16).max) * bounded_target
    return samples * (target / peak)


def synthesize_wav_bytes(
    *,
    config: MeloTTSServerConfig,
    model_factory: Callable[..., Any],
    cache: dict[tuple[str, str], Any],
    text: str,
    session_language: str,
    melo_language: str,
    speaker: str,
    speed: float,
) -> bytes:
    """Synthesize a request to WAV bytes using a cached MeloTTS model."""
    model = get_or_create_model(
        config=config,
        model_factory=model_factory,
        cache=cache,
        melo_language=melo_language,
    )
    speakers = get_model_speakers(model)
    if speaker not in speakers:
        raise web.HTTPBadRequest(
            text=(
                f"MeloTTS speaker `{speaker}` is not available for language `{session_language}`. "
                "Check `/voices` for the current speaker list."
            )
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        model.tts_to_file(text, speakers[speaker], str(temp_path), speed=speed)
        with wave.open(str(temp_path), "rb") as wav_file:
            pcm_bytes = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
    finally:
        temp_path.unlink(missing_ok=True)

    return wrap_pcm_as_wav(
        pcm_bytes,
        input_sample_rate=sample_rate,
        output_sample_rate=config.output_sample_rate,
        channels=channels,
    )


def create_app(
    config: MeloTTSServerConfig,
    *,
    model_factory: Callable[..., Any] | None = None,
) -> web.Application:
    """Create the aiohttp application exposing the local HTTP WAV contract."""
    factory = create_model_factory(model_factory)
    app = web.Application()
    app["config"] = config
    app["model_factory"] = factory
    app["model_cache"] = {}

    async def healthz(_request: web.Request) -> web.Response:
        cache: dict[tuple[str, str], Any] = app["model_cache"]
        payload = {
            "status": "ok",
            "device": config.device,
            "recommended_python": MELO_RECOMMENDED_PYTHON,
            "default_speakers": {
                "zh": config.zh_speaker,
                "en": config.en_speaker,
            },
            "loaded_models": sorted(f"{language}:{device}" for language, device in cache),
        }
        return web.json_response(payload)

    async def voices(_request: web.Request) -> web.Response:
        cache: dict[tuple[str, str], Any] = app["model_cache"]
        speakers: dict[str, list[str]] = {}
        for session_language, melo_language in (("zh", "ZH"), ("en", "EN")):
            model = get_or_create_model(
                config=config,
                model_factory=factory,
                cache=cache,
                melo_language=melo_language,
            )
            speakers[session_language] = sorted(get_model_speakers(model))

        return web.json_response(
            {
                "device": config.device,
                "default_speakers": {
                    "zh": config.zh_speaker,
                    "en": config.en_speaker,
                },
                "speakers": speakers,
            }
        )

    async def tts(request: web.Request) -> web.Response:
        payload = await request.json()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text="`text` is required.")

        session_language, melo_language = normalize_language(payload.get("language") or "zh")
        speaker = str(
            payload.get("speaker") or default_speaker_for_language(config, session_language)
        )
        speed = float(payload.get("speed") or config.speed)
        started = time.perf_counter()
        wav_bytes = await asyncio.to_thread(
            synthesize_wav_bytes,
            config=config,
            model_factory=factory,
            cache=app["model_cache"],
            text=text,
            session_language=session_language,
            melo_language=melo_language,
            speaker=speaker,
            speed=speed,
        )
        request_ms = round((time.perf_counter() - started) * 1000.0)
        print(
            "MeloTTS request complete "
            f"chars={len(text)} language={session_language} speaker={speaker} "
            f"bytes={len(wav_bytes)} request_ms={request_ms}",
            flush=True,
        )
        response = web.Response(body=wav_bytes, content_type="audio/wav")
        response.headers["X-Blink-TTS-Backend"] = "melotts-http-server"
        response.headers["X-Blink-TTS-Speaker"] = speaker
        response.headers["X-Blink-TTS-Language"] = session_language
        response.headers["X-Blink-TTS-Chars"] = str(len(text))
        response.headers["X-Blink-TTS-Request-Ms"] = str(request_ms)
        return response

    app.router.add_get("/healthz", healthz)
    app.router.add_get("/voices", voices)
    app.router.add_post("/tts", tts)
    return app


async def run_server(config: MeloTTSServerConfig) -> int:
    """Run the MeloTTS HTTP-WAV server."""
    create_model_factory()
    app = create_app(config)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()
    print(f"MeloTTS HTTP-WAV server is available at http://{config.host}:{config.port}/tts")
    print(f"Device: {config.device}; defaults: zh={config.zh_speaker}, en={config.en_speaker}")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the repo-local MeloTTS server."""
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_server(config))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

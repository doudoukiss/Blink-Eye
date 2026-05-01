#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Manual speech-quality evaluation harness for Blink local TTS backends."""

from __future__ import annotations

import argparse
import asyncio
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

from blink.cli.local_common import (
    DEFAULT_LOCAL_LANGUAGE,
    configure_logging,
    create_local_tts_service,
    default_local_tts_backend,
    get_local_env,
    maybe_load_dotenv,
    resolve_local_http_wav_voice,
    resolve_local_language,
    resolve_local_tts_base_url,
    resolve_local_tts_voice,
)
from blink.clocks.system_clock import SystemClock
from blink.frames.frames import EndFrame, ErrorFrame, StartFrame, TTSAudioRawFrame
from blink.processors.frame_processor import FrameProcessorSetup
from blink.project_identity import PROJECT_IDENTITY
from blink.transcriptions.language import Language
from blink.utils.asyncio.task_manager import TaskManager, TaskManagerParams

DEFAULT_TTS_EVAL_OUTPUT_DIR = Path("artifacts/tts-eval")
DEFAULT_TTS_EVAL_SAMPLE_RATE = 24_000
KOKORO_ZH_AB_VOICES = [
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
]
ZH_PHRASES = [
    ("zh_01_intro", "你好，我们正在测试本地中文对话的语音自然度。"),
    ("zh_02_datetime", "今天是二零二六年四月十五日，下午三点二十分，请在两分钟后再提醒我。"),
    ("zh_03_versions", "当前版本是 v2.3.1，接口状态是 HTTP 200，命令行工具已经更新。"),
    (
        "zh_04_symbols",
        "如果日志里出现路径、网页链接或者斜杠符号，语音应该尽量读得自然，不要逐个念符号。",
    ),
    ("zh_05_latency", "当前延迟大约是一百二十毫秒，任务 ID 是 A-1024，请优先确认语音响应速度。"),
]
EN_PHRASES = [
    ("en_01_intro", "Hello, this is a local English voice routing test."),
    ("en_02_datetime", "The smoke test finished at 3:20 PM on April fifteenth, twenty twenty-six."),
    (
        "en_03_technical",
        "Please describe the API status clearly, without switching into Chinese pronunciation.",
    ),
    ("en_04_quality", "Blink should say this sentence in clear English, not Chinglish."),
]
MIXED_DIAGNOSTIC_PHRASE = (
    "mixed_01_diagnostic",
    "请自然地说出 Blink 版本 2.3，然后再说 API 状态正常，最后补一句 twenty four degrees，避免中英混读失真。",
)


@dataclass(frozen=True)
class EvalCombination:
    """A single TTS evaluation combination."""

    backend: str
    language: Language
    voice: Optional[str]


@dataclass(frozen=True)
class LocalTTSEvalConfig:
    """Configuration for the local TTS evaluation harness."""

    backends: list[str]
    languages: list[Language]
    voices_zh: list[str]
    voices_en: list[str]
    output_dir: Path
    sample_rate: int
    verbose: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize a deterministic Blink local TTS evaluation matrix to WAV files "
            "for manual listening and language-routing verification."
        )
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=["kokoro", "piper", "xtts", "local-http-wav"],
        help="TTS backend to evaluate. Repeat to compare multiple backends.",
    )
    parser.add_argument(
        "--language",
        action="append",
        help=(
            "Session speech language to evaluate. Repeat to compare multiple languages. "
            "Defaults to both zh and en."
        ),
    )
    parser.add_argument(
        "--voice",
        action="append",
        help="Generic voice override list applied when no language-specific list is given.",
    )
    parser.add_argument(
        "--voice-zh",
        action="append",
        help="Chinese voice override list. Repeat to A/B test multiple voices.",
    )
    parser.add_argument(
        "--voice-en",
        action="append",
        help="English voice override list. Repeat to A/B test multiple voices.",
    )
    parser.add_argument(
        "--all-kokoro-zh-voices",
        action="store_true",
        help="Evaluate the curated Kokoro Mandarin voice set for reproducible A/B listening.",
    )
    parser.add_argument(
        "--output-dir",
        help=f"Directory for generated WAV files, default `{DEFAULT_TTS_EVAL_OUTPUT_DIR}`.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_TTS_EVAL_SAMPLE_RATE,
        help=f"Output sample rate in Hz, default {DEFAULT_TTS_EVAL_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logging while generating evaluation audio.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> LocalTTSEvalConfig:
    """Resolve evaluation settings from CLI flags and the local environment."""
    maybe_load_dotenv()

    if args.language:
        languages = [resolve_local_language(value) for value in args.language]
    else:
        languages = [Language.ZH, Language.EN]

    backends = list(args.backend or [])
    if not backends:
        backends = []
        for language in languages:
            backend = get_local_env("TTS_BACKEND", default_local_tts_backend(language))
            if backend not in backends:
                backends.append(backend)

    shared_voices = list(args.voice or [])
    voices_zh = list(args.voice_zh or [])
    voices_en = list(args.voice_en or [])

    if args.all_kokoro_zh_voices and "kokoro" in backends:
        voices_zh = KOKORO_ZH_AB_VOICES

    if not voices_zh and shared_voices:
        voices_zh = shared_voices
    if not voices_en and shared_voices:
        voices_en = shared_voices

    return LocalTTSEvalConfig(
        backends=backends,
        languages=languages,
        voices_zh=voices_zh,
        voices_en=voices_en,
        output_dir=Path(args.output_dir) if args.output_dir else DEFAULT_TTS_EVAL_OUTPUT_DIR,
        sample_rate=args.sample_rate,
        verbose=args.verbose,
    )


def _phrases_for_language(language: Language) -> list[tuple[str, str]]:
    phrases = list(ZH_PHRASES if language == Language.ZH else EN_PHRASES)
    phrases.append(MIXED_DIAGNOSTIC_PHRASE)
    return phrases


def _voice_candidates(
    config: LocalTTSEvalConfig, backend: str, language: Language
) -> list[Optional[str]]:
    if language == Language.ZH:
        if config.voices_zh:
            return list(config.voices_zh)
    else:
        if config.voices_en:
            return list(config.voices_en)

    return [resolve_local_tts_voice(backend, language)]


def build_eval_matrix(config: LocalTTSEvalConfig) -> list[EvalCombination]:
    """Build the backend/language/voice matrix for the evaluation run."""
    matrix: list[EvalCombination] = []
    for backend in config.backends:
        for language in config.languages:
            for voice in _voice_candidates(config, backend, language):
                matrix.append(EvalCombination(backend=backend, language=language, voice=voice))
    return matrix


def _voice_slug(voice: Optional[str]) -> str:
    if voice in (None, ""):
        return "auto"
    return str(voice).replace("/", "_")


async def _apply_text_filters(service, text: str) -> str:
    prepared = text
    for text_filter in getattr(service, "_text_filters", ()):
        prepared = await text_filter.filter(prepared)
    return prepared


def _write_wav(path: Path, audio: bytes, sample_rate: int, channels: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _synthesize_phrase(service, phrase_text: str) -> tuple[bytes, int, int]:
    prepared_text = await _apply_text_filters(service, phrase_text)
    audio = bytearray()
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    async for frame in service.run_tts(prepared_text, context_id="tts-eval"):
        if isinstance(frame, ErrorFrame):
            raise RuntimeError(frame.error)
        if not isinstance(frame, TTSAudioRawFrame):
            continue
        audio.extend(frame.audio)
        sample_rate = sample_rate or frame.sample_rate
        channels = channels or frame.num_channels

    if not audio:
        raise RuntimeError("TTS backend returned no audio frames.")

    return bytes(audio), sample_rate or DEFAULT_TTS_EVAL_SAMPLE_RATE, channels or 1


async def _render_combination(
    combo: EvalCombination,
    config: LocalTTSEvalConfig,
    aiohttp_session: aiohttp.ClientSession,
) -> tuple[list[Path], Optional[str]]:
    backend_base_url = (
        resolve_local_tts_base_url("xtts")
        if combo.backend == "xtts"
        else resolve_local_tts_base_url("local-http-wav")
        if combo.backend == "local-http-wav"
        else resolve_local_tts_base_url(combo.backend)
    )

    resolved_voice = combo.voice
    if combo.backend == "local-http-wav":
        resolved_voice = await resolve_local_http_wav_voice(
            backend_base_url,
            combo.voice,
            combo.language,
        )

    service = create_local_tts_service(
        backend=combo.backend,
        voice=resolved_voice,
        language=combo.language,
        base_url=backend_base_url,
        aiohttp_session=aiohttp_session,
    )

    if hasattr(service, "setup"):
        clock = SystemClock()
        clock.start()
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await service.setup(FrameProcessorSetup(clock=clock, task_manager=task_manager))

    await service.start(StartFrame(audio_out_sample_rate=config.sample_rate))
    resolved_voice = getattr(getattr(service, "_settings", None), "voice", resolved_voice)
    output_paths: list[Path] = []

    try:
        for phrase_id, phrase_text in _phrases_for_language(combo.language):
            audio, sample_rate, channels = await _synthesize_phrase(service, phrase_text)
            output_path = (
                config.output_dir
                / combo.backend
                / combo.language.value
                / _voice_slug(resolved_voice)
                / f"{phrase_id}.wav"
            )
            _write_wav(output_path, audio, sample_rate, channels)
            output_paths.append(output_path)
            print(
                f"[tts-eval] wrote {output_path} "
                f"(backend={combo.backend}, language={combo.language.value}, "
                f"voice={resolved_voice or 'auto'})"
            )
    finally:
        await service.stop(EndFrame())
        if hasattr(service, "cleanup"):
            await service.cleanup()

    return output_paths, resolved_voice


async def run_evaluation(config: LocalTTSEvalConfig) -> int:
    """Run the local TTS evaluation matrix and write WAV outputs."""
    configure_logging(config.verbose)
    matrix = build_eval_matrix(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        f"# {PROJECT_IDENTITY.display_name} Local TTS Evaluation",
        "",
        "Speech is fixed per session in the local product path.",
        "Use separate zh and en runs to compare routing and voice quality.",
        "The Chinese phrase set focuses on technical conversation, dates, times, versions, and symbols.",
        "Compare Kokoro bootstrap renders against `local-http-wav` sidecar renders such as MeloTTS when available.",
        "",
    ]

    async with aiohttp.ClientSession() as session:
        for combo in matrix:
            output_paths, resolved_voice = await _render_combination(combo, config, session)
            manifest_lines.append(
                f"- backend={combo.backend}, language={combo.language.value}, "
                f"voice={_voice_slug(resolved_voice)}"
            )
            manifest_lines.extend([f"  - {path}" for path in output_paths])

    manifest_path = config.output_dir / "README.md"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"[tts-eval] wrote manifest {manifest_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for the local TTS evaluation harness."""
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    try:
        return asyncio.run(run_evaluation(config))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}")
        return 1


__all__ = [
    "DEFAULT_TTS_EVAL_OUTPUT_DIR",
    "EvalCombination",
    "LocalTTSEvalConfig",
    "build_eval_matrix",
    "build_parser",
    "main",
    "resolve_config",
    "run_evaluation",
]

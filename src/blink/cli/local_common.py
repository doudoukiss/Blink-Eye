#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared helpers for Blink's local-first CLI entry points."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import httpx
from loguru import logger

from blink.pipeline.pipeline import Pipeline
from blink.pipeline.task import PipelineParams, PipelineTask
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from blink.processors.frameworks.rtvi.observer import RTVIObserverParams
from blink.project_identity import PROJECT_IDENTITY, cache_dir, local_env_name
from blink.services.local_http_wav.catalog import (
    local_http_wav_fallback_speaker,
    local_http_wav_language_key,
    local_http_wav_speakers_for_language,
)
from blink.services.ollama.llm import OllamaLLMService
from blink.services.stt_latency import WHISPER_TTFS_P99
from blink.services.tts_service import TextAggregationMode
from blink.transcriptions.language import Language
from blink.utils.text.base_text_filter import BaseTextFilter
from blink.utils.text.markdown_text_filter import MarkdownTextFilter

if TYPE_CHECKING:
    import aiohttp

    from blink.processors.frame_processor import FrameProcessor
    from blink.turns.user_mute import BaseUserMuteStrategy
    from blink.turns.user_start import BaseUserTurnStartStrategy

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_LOCAL_LLM_PROVIDER = "ollama"
DEFAULT_OPENAI_RESPONSES_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_RESPONSES_DEMO_SERVICE_TIER = "priority"
DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS = 120
LOCAL_LLM_PROVIDERS = ("ollama", "openai-responses")
DEFAULT_EN_TEXT_SYSTEM_PROMPT = (
    f"You are {PROJECT_IDENTITY.display_name}, and you are helping a developer explore "
    f"{PROJECT_IDENTITY.display_name} locally. {PROJECT_IDENTITY.display_name} uses a "
    "frame-based real-time runtime. "
    "Keep answers concise, practical, and grounded in how pipelines, frames, transports, "
    "and services work. If the user asks who you are, answer that you are Blink."
)
DEFAULT_ZH_TEXT_SYSTEM_PROMPT = (
    f"你就是 {PROJECT_IDENTITY.display_name}，正在帮助开发者在本地探索 "
    f"{PROJECT_IDENTITY.display_name}。"
    "当前产品采用基于 frame 的实时运行架构。请始终使用简体中文回答。"
    "回答要简洁、实用，并结合底层 pipeline、frame、transport、service 机制。"
    "当解释命令、配置或代码时，可以直接给出清晰的技术表达。"
    "如果用户问你是谁，请直接回答你是 Blink。"
)
DEFAULT_EN_SPEECH_SYSTEM_PROMPT = (
    f"You are {PROJECT_IDENTITY.display_name}, helping a developer explore "
    f"{PROJECT_IDENTITY.display_name} locally by voice. "
    f"{PROJECT_IDENTITY.display_name} uses a frame-based real-time runtime. "
    "Always answer in English unless the user explicitly asks for translation or quoted non-English text. "
    "Keep answers natural, easy to say aloud, and concise without becoming incomplete. "
    "Do not use markdown, code blocks, tables, bullet lists, raw URLs, file paths, or shell syntax. "
    "Prefer short spoken sentences over dense technical formatting. "
    "If the user asks for an explanation, answer fully in spoken language instead of cutting the answer too short. "
    "Simple questions can be answered briefly, but explanatory answers should usually be around four to eight spoken sentences. "
    "If the user asks who you are, answer that you are Blink."
)
DEFAULT_ZH_SPEECH_SYSTEM_PROMPT = (
    f"你就是 {PROJECT_IDENTITY.display_name}，正在通过语音帮助开发者在本地探索 "
    f"{PROJECT_IDENTITY.display_name}。"
    "当前产品采用基于 frame 的实时运行架构。请始终使用简体中文回答。"
    "为了便于语音合成，请使用短句、自然口语。"
    "不要输出 Markdown、代码、表格、项目符号、链接或表情。"
    "不要朗读原始网址、文件路径或命令行符号。"
    "遇到技术缩写时，优先先用中文解释，再补充英文名词。"
    "回答要完整、自然，并结合底层 pipeline、frame、transport、service 机制。"
    "简单问题可以简短回答，但遇到解释型问题时，不要为了简洁而把答案压得过短。"
    "解释型回答通常控制在四到八句。"
    "如果用户问你是谁，请直接回答你是 Blink。"
)
DEFAULT_EN_DEMO_TEXT_PROMPT_SUFFIX = (
    "Demo mode is active. Answer immediately in the first sentence, keep the response "
    "tight and polished, and avoid unnecessary markdown unless the user explicitly asks "
    "for formatted output."
)
DEFAULT_ZH_DEMO_TEXT_PROMPT_SUFFIX = (
    "当前启用演示模式。第一句话要直接回答问题，整体回答要紧凑、清楚、有展示感；"
    "除非用户明确要求格式化输出，否则避免不必要的 Markdown。"
)
DEFAULT_EN_DEMO_SPEECH_PROMPT_SUFFIX = (
    "Demo mode is active. Answer immediately in the first sentence. Keep spoken answers "
    "bounded, usually one to four short sentences. Do not use markdown, bullets, code, "
    "tables, raw links, file paths, or shell syntax in spoken replies."
)
DEFAULT_ZH_DEMO_SPEECH_PROMPT_SUFFIX = (
    "当前启用演示模式。第一句话要直接回答问题。语音回答要有边界，通常控制在一到四个短句。"
    "不要使用 Markdown、项目符号、代码、表格、原始链接、文件路径或命令行符号。"
)
DEFAULT_LOCAL_STT_BACKEND = "mlx-whisper"
DEFAULT_LOCAL_TTS_BACKEND = "kokoro"
DEFAULT_LOCAL_EN_STT_MODEL = "mlx-community/whisper-medium-mlx"
DEFAULT_LOCAL_ZH_STT_MODEL = "mlx-community/whisper-medium-mlx"
DEFAULT_LOCAL_FASTER_WHISPER_EN_MODEL = "Systran/faster-distil-whisper-medium.en"
DEFAULT_LOCAL_FASTER_WHISPER_ZH_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEFAULT_LOCAL_STT_MODEL = DEFAULT_LOCAL_ZH_STT_MODEL
DEFAULT_LOCAL_LANGUAGE = Language.ZH
DEFAULT_LOCAL_EN_TTS_BACKEND = "kokoro"
DEFAULT_LOCAL_ZH_TTS_BACKEND = "kokoro"
DEFAULT_LOCAL_EN_TTS_VOICE = "af_heart"
DEFAULT_LOCAL_ZH_KOKORO_VOICE = "zf_xiaobei"
DEFAULT_LOCAL_EN_PIPER_VOICE = "en_US-ryan-high"
DEFAULT_LOCAL_ZH_PIPER_VOICE = "zh_CN-xiao_ya-medium"
DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 7860
DEFAULT_LOCAL_PROFILE = "text"
DEFAULT_LOCAL_VISION_MODEL = "vikhyatk/moondream2"
DEFAULT_LOCAL_XTTS_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_LOCAL_HTTP_WAV_TTS_BASE_URL = "http://127.0.0.1:8001"
LOCAL_PROFILE_EXTRAS = {
    "text": [],
    "voice": ["local", "mlx-whisper", "kokoro"],
    "browser": ["runner", "webrtc", "mlx-whisper", "kokoro"],
    "full": ["local", "runner", "webrtc", "mlx-whisper", "kokoro"],
}

KOKORO_CACHE_DIR = Path.home() / ".cache" / "kokoro-onnx"
PIPER_CACHE_DIR = cache_dir("piper")
TRUE_VALUES = {"1", "true", "yes", "on"}


class LocalDependencyError(RuntimeError):
    """Raised when a local workflow dependency is missing."""


class ChineseSpeechTextFilter(BaseTextFilter):
    """Normalize Chinese TTS text so local speech reads naturally."""

    async def update_settings(self, settings: Mapping[str, Any]):
        """Ignore runtime setting updates for the stateless filter."""
        return None

    async def filter(self, text: str) -> str:
        """Normalize Chinese text before it reaches local speech synthesis."""
        return normalize_chinese_speech_text(text)

    async def handle_interruption(self):
        """No-op interruption hook required by the text filter interface."""
        return None

    async def reset_interruption(self):
        """No-op interruption reset hook required by the text filter interface."""
        return None


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Small snapshot of a local audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int


@dataclass(frozen=True)
class LocalRuntimeTTSSelection:
    """Effective TTS selection for a local runtime session."""

    backend: str
    voice: Optional[str]
    base_url: Optional[str]
    auto_switched: bool = False


@dataclass(frozen=True)
class LocalLLMConfig:
    """Effective local LLM provider configuration."""

    provider: str
    model: str
    base_url: Optional[str]
    system_prompt: str
    temperature: Optional[float] = None
    service_tier: Optional[str] = None
    demo_mode: bool = False
    max_output_tokens: Optional[int] = None


def _audio_device_from_mapping(index: int, info: dict) -> AudioDeviceInfo:
    """Normalize a PyAudio device info mapping."""
    return AudioDeviceInfo(
        index=index,
        name=str(info.get("name", "")),
        max_input_channels=int(info.get("maxInputChannels", 0)),
        max_output_channels=int(info.get("maxOutputChannels", 0)),
        default_sample_rate=int(info.get("defaultSampleRate", 0)),
    )


def resolve_profile_extras(
    profile: str, with_vision: bool = False, with_piper: bool = False
) -> list[str]:
    """Resolve the project extras required for a local bootstrap profile."""
    extras = list(LOCAL_PROFILE_EXTRAS[profile])
    if with_piper:
        extras.append("piper")
    if with_vision and profile in {"browser", "full"}:
        extras.append("moondream")
    return extras


def maybe_load_dotenv():
    """Load a local `.env` file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(override=False)


def configure_logging(verbose: bool):
    """Configure a concise stderr logger for CLI commands."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")


def resolve_int(value: Optional[int | str]) -> Optional[int]:
    """Normalize an optional integer-like value."""
    if value in (None, ""):
        return None
    return int(value)


def get_env_alias(primary_name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a canonical env value when it is set."""
    if primary_name in os.environ:
        return os.environ[primary_name]
    return default


def get_local_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a Blink local env value."""
    return get_env_alias(local_env_name(name), default)


def _nonempty(value: Optional[str]) -> Optional[str]:
    """Return a stripped value, or None when unset/blank."""
    if value in (None, ""):
        return None
    stripped = str(value).strip()
    return stripped or None


def local_env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean local env flag."""
    value = get_local_env(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def resolve_local_demo_mode(value: Optional[bool] = None) -> bool:
    """Resolve explicit local demo-mode state from CLI or environment."""
    if value is not None:
        return bool(value)
    return local_env_flag("DEMO_MODE", False)


def resolve_local_language(value: Optional[str | Language]) -> Language:
    """Resolve a CLI or environment language value to a Language enum."""
    if value in (None, ""):
        return DEFAULT_LOCAL_LANGUAGE
    if isinstance(value, Language):
        return value

    normalized = str(value).replace("_", "-").lower()
    for language in Language:
        if language.value.lower() == normalized:
            return language

    raise ValueError(f"Unsupported local language: {value}")


def is_chinese_language(language: Language) -> bool:
    """Return whether the selected local language should use Chinese defaults."""
    normalized = language.value.lower()
    return normalized.startswith("zh") or normalized.startswith("cmn")


def default_local_tts_backend(language: Language) -> str:
    """Return the default TTS backend for the selected language."""
    if is_chinese_language(language):
        return DEFAULT_LOCAL_ZH_TTS_BACKEND
    return DEFAULT_LOCAL_EN_TTS_BACKEND


def default_local_stt_model(*, backend: str, language: Language) -> str:
    """Return the default STT model for the selected backend and language."""
    if backend == "mlx-whisper":
        if is_chinese_language(language):
            return DEFAULT_LOCAL_ZH_STT_MODEL
        return DEFAULT_LOCAL_EN_STT_MODEL
    if backend == "whisper":
        if is_chinese_language(language):
            return DEFAULT_LOCAL_FASTER_WHISPER_ZH_MODEL
        return DEFAULT_LOCAL_FASTER_WHISPER_EN_MODEL
    raise ValueError(f"Unsupported STT backend: {backend}")


def default_local_tts_voice(backend: str, language: Language) -> Optional[str]:
    """Return the default local TTS voice for the selected backend and language."""
    if backend == "xtts":
        return None
    if backend == "local-http-wav":
        return None
    if backend == "piper":
        if is_chinese_language(language):
            return DEFAULT_LOCAL_ZH_PIPER_VOICE
        return DEFAULT_LOCAL_EN_PIPER_VOICE
    if backend == "kokoro":
        if is_chinese_language(language):
            return DEFAULT_LOCAL_ZH_KOKORO_VOICE
        return DEFAULT_LOCAL_EN_TTS_VOICE
    raise ValueError(f"Unsupported TTS backend: {backend}")


def _voice_matches_backend(backend: str, voice: Optional[str]) -> bool:
    """Return whether a configured voice id looks compatible with a backend."""
    if voice in (None, ""):
        return False
    if backend == "piper":
        return re.match(r"^[a-z]{2,3}_[A-Z]{2,3}[-_].+", voice) is not None
    if backend == "kokoro":
        return re.match(r"^[abz][fm]_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$", voice) is not None
    if backend == "xtts":
        return True
    if backend == "local-http-wav":
        return False
    return True


def resolve_local_tts_voice(
    backend: str,
    language: Language,
    explicit_voice: Optional[str] = None,
) -> Optional[str]:
    """Resolve a local TTS voice with language-specific overrides.

    Resolution order is:

    1. Explicit CLI voice override.
    2. Language-specific environment override.
    3. Generic environment fallback.
    4. Backend/language default.

    Args:
        backend: Selected local TTS backend.
        language: Session language for the local workflow.
        explicit_voice: Optional explicit CLI voice override.

    Returns:
        The resolved voice identifier, or ``None`` for backends that support
        server-side auto-selection.
    """
    if explicit_voice not in (None, ""):
        return explicit_voice

    # HTTP-WAV sidecars expose their own speaker inventories. Reusing the
    # generic local voice env vars here leaks backend-specific ids such as
    # Kokoro voices into Melo/CosyVoice adapters and causes runtime 400s.
    if backend == "local-http-wav":
        return None

    candidates = (
        [get_local_env("TTS_VOICE_ZH"), get_local_env("TTS_VOICE")]
        if is_chinese_language(language)
        else [get_local_env("TTS_VOICE_EN"), get_local_env("TTS_VOICE")]
    )

    for candidate in candidates:
        if _voice_matches_backend(backend, candidate):
            return candidate

    return default_local_tts_voice(backend, language)


def default_local_system_prompt(language: Language) -> str:
    """Return the default local system prompt."""
    return default_local_text_system_prompt(language)


def default_local_text_system_prompt(language: Language) -> str:
    """Return the default local system prompt for text-oriented chat."""
    if is_chinese_language(language):
        return DEFAULT_ZH_TEXT_SYSTEM_PROMPT
    return DEFAULT_EN_TEXT_SYSTEM_PROMPT


def default_local_speech_system_prompt(language: Language) -> str:
    """Return the default local system prompt for spoken workflows."""
    if is_chinese_language(language):
        return DEFAULT_ZH_SPEECH_SYSTEM_PROMPT
    return DEFAULT_EN_SPEECH_SYSTEM_PROMPT


def local_demo_mode_prompt_suffix(language: Language, *, speech: bool) -> str:
    """Return the bounded response prompt suffix for local demo mode."""
    if is_chinese_language(language):
        return (
            DEFAULT_ZH_DEMO_SPEECH_PROMPT_SUFFIX if speech else DEFAULT_ZH_DEMO_TEXT_PROMPT_SUFFIX
        )
    return DEFAULT_EN_DEMO_SPEECH_PROMPT_SUFFIX if speech else DEFAULT_EN_DEMO_TEXT_PROMPT_SUFFIX


def apply_local_demo_mode_prompt(prompt: str, *, language: Language, speech: bool) -> str:
    """Append compact demo-mode behavior guidance to an existing system prompt."""
    suffix = local_demo_mode_prompt_suffix(language, speech=speech)
    normalized_prompt = (prompt or "").strip()
    if not normalized_prompt:
        return suffix
    if suffix in normalized_prompt:
        return normalized_prompt
    return f"{normalized_prompt} {suffix}"


CHINESE_SPEECH_SHORTHAND = {
    "API": "接口",
    "CPU": "处理器",
    "CLI": "命令行工具",
    "GPU": "图形处理器",
    "HTTP": "网络请求",
    "HTTPS": "安全网页链接",
    "ID": "编号",
    "LLM": "大语言模型",
    "OK": "正常",
    "SDK": "开发工具包",
    "STT": "语音识别",
    "TTS": "语音合成",
    "UI": "界面",
    "URL": "网页链接",
    "WebRTC": "网页实时音视频",
}


def _replace_chinese_speech_urls(text: str) -> str:
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    return re.sub(r"https?://\S+", " 网页链接 ", text)


def _replace_chinese_speech_paths(text: str) -> str:
    text = re.sub(r"(?<![A-Za-z0-9])(?:\./|\.\./|~/|/)[^\s，。；！？（）【】]+", " 路径 ", text)
    return re.sub(r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\s，。；！？（）【】]+", " 路径 ", text)


def _replace_chinese_speech_versions(text: str) -> str:
    return re.sub(r"\bv\s*(\d+(?:\.\d+)+)\b", r"版本 \1", text, flags=re.IGNORECASE)


def _replace_chinese_speech_dates_and_times(text: str) -> str:
    text = re.sub(
        r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)",
        lambda match: f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日",
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d{4})/(\d{1,2})/(\d{1,2})(?!\d)",
        lambda match: f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日",
        text,
    )
    return re.sub(
        r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)",
        lambda match: f"{int(match.group(1))}点{int(match.group(2))}分",
        text,
    )


def _replace_chinese_speech_measurements(text: str) -> str:
    text = re.sub(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*ms(?![A-Za-z])",
        r"\1 毫秒",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*s(?![A-Za-z])",
        r"\1 秒",
        text,
        flags=re.IGNORECASE,
    )


def _separate_chinese_speech_mixed_tokens(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"([A-Za-z]{1,8})[-_](\d+(?:\.\d+)*)", r"\1 \2", text)
    text = re.sub(r"([A-Za-z]{1,8})(\d{2,})", r"\1 \2", text)
    text = re.sub(r"([一-龥])([A-Za-z]+)(?![A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z]+)([一-龥])", r"\1 \2", text)
    text = re.sub(r"([一-龥])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([一-龥])", r"\1 \2", text)
    text = re.sub(r"\s*([年月日点分秒])\s*", r"\1", text)
    return re.sub(r"(\d)\s+(毫秒|秒)", r"\1\2", text)


def normalize_chinese_speech_text(text: str) -> str:
    """Normalize Chinese technical text into a more speakable form for TTS."""
    text = _replace_chinese_speech_urls(text)
    text = text.replace("```", " ").replace("`", " ")
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = text.replace("#", " ").replace("*", " ").replace("_", " ")
    text = _replace_chinese_speech_paths(text)
    text = _replace_chinese_speech_versions(text)
    text = _replace_chinese_speech_dates_and_times(text)
    text = _replace_chinese_speech_measurements(text)
    text = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", text)

    for shorthand, spoken in CHINESE_SPEECH_SHORTHAND.items():
        text = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(shorthand)}(?![A-Za-z0-9])",
            spoken,
            text,
        )

    text = _separate_chinese_speech_mixed_tokens(text)

    text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
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
                "[": "【",
                "]": "】",
            }
        )
    )
    text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
    text = re.sub(r"(?<!\d)/(?!\d)", "，", text)
    text = text.replace("\\", "，")
    text = text.replace("...", "，")
    text = text.replace("~", "到")
    text = re.sub(r"[-=|]+", "，", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def huggingface_model_cache_dir(model_id: str) -> Path:
    """Return the default Hugging Face cache directory for a model repo id."""
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return cache_root / f"models--{model_id.replace('/', '--')}"


def model_cache_exists(model_id: str) -> bool:
    """Check whether a Hugging Face-backed model repo exists in the local cache."""
    return huggingface_model_cache_dir(model_id).exists()


def default_piper_model_path(voice: str) -> Path:
    """Return the expected Piper cache path for a voice id."""
    return PIPER_CACHE_DIR / f"{voice}.onnx"


def kokoro_assets_present() -> bool:
    """Check whether Kokoro's default cache files already exist."""
    return (KOKORO_CACHE_DIR / "kokoro-v1.0.onnx").exists() and (
        KOKORO_CACHE_DIR / "voices-v1.0.bin"
    ).exists()


def tts_backend_uses_external_service(backend: str) -> bool:
    """Return whether a TTS backend relies on an external local HTTP service."""
    return backend in {"xtts", "local-http-wav"}


def default_local_tts_base_url(backend: str) -> Optional[str]:
    """Return the default base URL for backends served over local HTTP."""
    if backend == "xtts":
        return DEFAULT_LOCAL_XTTS_BASE_URL
    if backend == "local-http-wav":
        return DEFAULT_LOCAL_HTTP_WAV_TTS_BASE_URL
    return None


def resolve_local_tts_base_url(backend: str) -> Optional[str]:
    """Resolve a local TTS base URL from Blink local environment settings."""
    if backend == "xtts":
        return get_local_env("XTTS_BASE_URL") or default_local_tts_base_url(backend)
    if backend == "local-http-wav":
        return get_local_env("HTTP_WAV_TTS_BASE_URL") or default_local_tts_base_url(backend)
    return default_local_tts_base_url(backend)


def resolve_xtts_voice(base_url: str, voice: Optional[str]) -> str:
    """Resolve an XTTS speaker, auto-selecting the first speaker when unset."""
    if voice:
        return voice

    url = f"{base_url.rstrip('/')}/studio_speakers"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        raise LocalDependencyError(
            "Chinese XTTS voice requires a reachable XTTS-compatible server at "
            f"{base_url}. Start it first or switch to `local-http-wav`."
        ) from exc

    speakers = response.json()
    if not isinstance(speakers, dict) or not speakers:
        voice_zh_env = local_env_name("TTS_VOICE_ZH")
        voice_en_env = local_env_name("TTS_VOICE_EN")
        voice_env = local_env_name("TTS_VOICE")
        raise LocalDependencyError(
            "XTTS did not return any speakers from `/studio_speakers`. "
            f"Set `--tts-voice`, `{voice_zh_env}`, `{voice_en_env}`, or `{voice_env}` "
            "explicitly, or load speakers into the XTTS server."
        )

    return str(next(iter(speakers)))


async def resolve_local_http_wav_voice(
    base_url: str,
    voice: Optional[str],
    language: Language,
) -> Optional[str]:
    """Resolve a `local-http-wav` speaker against the sidecar's `/voices` catalog.

    If the server exposes `/voices` and the configured speaker is invalid for the
    current language, this falls back to the server default speaker or the first
    advertised speaker. If `/voices` is unavailable, the configured voice is
    returned unchanged so custom servers that only implement `/tts` still work.
    """
    if voice in (None, ""):
        return None

    url = f"{base_url.rstrip('/')}/voices"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            catalog = response.json()
    except Exception:
        return voice

    if not isinstance(catalog, dict):
        return voice

    fallback_voice = local_http_wav_fallback_speaker(
        catalog,
        configured_voice=str(voice),
        language=language,
    )
    if fallback_voice == voice:
        return voice

    available = local_http_wav_speakers_for_language(catalog, language)
    logger.warning(
        "Configured local-http-wav speaker '{}' is unavailable for language '{}' at {}. "
        "Falling back to {}. Available speakers: {}",
        voice,
        local_http_wav_language_key(language),
        base_url.rstrip("/"),
        repr(fallback_voice) if fallback_voice else "server auto-selection",
        ", ".join(available) if available else "<unknown>",
    )
    return fallback_voice


async def local_http_wav_service_available(base_url: Optional[str]) -> bool:
    """Return whether a local HTTP-WAV sidecar appears reachable for runtime use."""
    if base_url in (None, ""):
        return False

    normalized_base_url = str(base_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            healthz = await client.get(f"{normalized_base_url}/healthz")
            if healthz.status_code == 200:
                return True
            voices = await client.get(f"{normalized_base_url}/voices")
            return voices.status_code == 200
    except Exception:
        return False


async def resolve_local_runtime_tts_selection(
    *,
    language: Language,
    requested_backend: str,
    requested_voice: Optional[str],
    requested_base_url: Optional[str],
    backend_locked: bool,
    explicit_voice: Optional[str] = None,
) -> LocalRuntimeTTSSelection:
    """Resolve the runtime TTS backend for a local session.

    Chinese voice and browser sessions prefer a healthy `local-http-wav` sidecar
    when the backend was not explicitly pinned. Kokoro remains the bootstrap
    fallback when no sidecar is available or when the user explicitly selected a
    backend.
    """
    selected_backend = requested_backend
    selected_voice = requested_voice
    selected_base_url = requested_base_url
    auto_switched = False

    if (
        is_chinese_language(language)
        and not backend_locked
        and requested_backend == default_local_tts_backend(language)
    ):
        local_http_wav_base_url = resolve_local_tts_base_url("local-http-wav")
        if await local_http_wav_service_available(local_http_wav_base_url):
            selected_backend = "local-http-wav"
            selected_base_url = local_http_wav_base_url
            selected_voice = explicit_voice if explicit_voice not in (None, "") else None
            auto_switched = True

    if selected_backend == "local-http-wav" and selected_base_url:
        selected_voice = await resolve_local_http_wav_voice(
            selected_base_url,
            selected_voice,
            language,
        )

    return LocalRuntimeTTSSelection(
        backend=selected_backend,
        voice=selected_voice,
        base_url=selected_base_url,
        auto_switched=auto_switched,
    )


async def verify_ollama(base_url: str, model: str):
    """Verify that the local Ollama server is reachable and exposes the configured model."""
    url = f"{base_url.rstrip('/')}/models"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Start it with `ollama serve` and try again."
        ) from exc

    data = response.json().get("data", [])
    model_ids = {item.get("id") for item in data if isinstance(item, dict)}
    if model_ids and model not in model_ids:
        raise RuntimeError(
            f"Model {model!r} is not available in Ollama. Pull it with `ollama pull {model}`."
        )


def resolve_local_llm_provider(value: Optional[str] = None) -> str:
    """Resolve the selected local LLM provider."""
    raw_provider = _nonempty(value) or _nonempty(get_local_env("LLM_PROVIDER"))
    provider = (raw_provider or DEFAULT_LOCAL_LLM_PROVIDER).lower()
    if provider not in LOCAL_LLM_PROVIDERS:
        supported = ", ".join(LOCAL_LLM_PROVIDERS)
        raise ValueError(
            f"Unsupported local LLM provider: {value or raw_provider!r}. Use {supported}."
        )
    return provider


def _resolve_local_llm_max_output_tokens(
    value: Optional[int | str],
    *,
    provider: str,
    demo_mode: bool,
) -> Optional[int]:
    """Resolve the optional local LLM output-token budget."""
    explicit_value = value if value not in (None, "") else None
    env_value = _nonempty(get_local_env("LLM_MAX_OUTPUT_TOKENS"))
    if provider == "openai-responses":
        env_value = env_value or _nonempty(get_local_env("OPENAI_RESPONSES_MAX_OUTPUT_TOKENS"))
    raw_value = explicit_value if explicit_value is not None else env_value

    if raw_value not in (None, ""):
        resolved_value = int(raw_value)
        if resolved_value <= 0:
            raise ValueError("Local LLM max output tokens must be greater than zero.")
        return resolved_value

    if provider == "openai-responses" and demo_mode:
        return DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS

    return None


def resolve_local_llm_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    language: Language,
    temperature: Optional[float] = None,
    service_tier: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
    demo_mode: Optional[bool] = None,
    max_output_tokens: Optional[int | str] = None,
    speech: bool = False,
    ignore_env_system_prompt: bool = False,
) -> LocalLLMConfig:
    """Resolve local LLM provider settings from CLI values and environment."""
    resolved_provider = resolve_local_llm_provider(provider)
    resolved_demo_mode = resolve_local_demo_mode(demo_mode)
    shared_prompt = (
        None if ignore_env_system_prompt else _nonempty(get_local_env("LLM_SYSTEM_PROMPT"))
    )
    explicit_prompt = _nonempty(system_prompt)
    resolved_default_prompt = default_system_prompt or default_local_text_system_prompt(language)
    resolved_prompt = explicit_prompt or shared_prompt or resolved_default_prompt
    if resolved_demo_mode:
        resolved_prompt = apply_local_demo_mode_prompt(
            resolved_prompt,
            language=language,
            speech=speech,
        )
    resolved_max_output_tokens = _resolve_local_llm_max_output_tokens(
        max_output_tokens,
        provider=resolved_provider,
        demo_mode=resolved_demo_mode,
    )

    if resolved_provider == "ollama":
        ollama_prompt = (
            explicit_prompt
            or shared_prompt
            or (None if ignore_env_system_prompt else _nonempty(os.getenv("OLLAMA_SYSTEM_PROMPT")))
            or resolved_default_prompt
        )
        if resolved_demo_mode:
            ollama_prompt = apply_local_demo_mode_prompt(
                ollama_prompt,
                language=language,
                speech=speech,
            )
        return LocalLLMConfig(
            provider=resolved_provider,
            model=_nonempty(model) or _nonempty(os.getenv("OLLAMA_MODEL")) or DEFAULT_OLLAMA_MODEL,
            base_url=_nonempty(base_url)
            or _nonempty(os.getenv("OLLAMA_BASE_URL"))
            or DEFAULT_OLLAMA_BASE_URL,
            system_prompt=ollama_prompt,
            temperature=temperature,
            service_tier=None,
            demo_mode=resolved_demo_mode,
            max_output_tokens=None,
        )

    if resolved_provider == "openai-responses":
        explicit_service_tier = _nonempty(service_tier) or _nonempty(
            get_local_env("OPENAI_RESPONSES_SERVICE_TIER")
        )
        return LocalLLMConfig(
            provider=resolved_provider,
            model=_nonempty(model)
            or _nonempty(get_local_env("OPENAI_RESPONSES_MODEL"))
            or DEFAULT_OPENAI_RESPONSES_MODEL,
            base_url=_nonempty(base_url) or _nonempty(get_local_env("OPENAI_RESPONSES_BASE_URL")),
            system_prompt=resolved_prompt,
            temperature=temperature,
            service_tier=explicit_service_tier
            or (DEFAULT_OPENAI_RESPONSES_DEMO_SERVICE_TIER if resolved_demo_mode else None),
            demo_mode=resolved_demo_mode,
            max_output_tokens=resolved_max_output_tokens,
        )

    raise ValueError(f"Unsupported local LLM provider: {resolved_provider!r}")


def _require_openai_responses_service():
    """Import the OpenAI Responses service lazily for optional provider support."""
    try:
        from blink.services.openai.responses.llm import OpenAIResponsesLLMService
    except Exception as exc:  # pragma: no cover - exercised via caller tests
        raise LocalDependencyError(
            "The `openai-responses` local LLM provider requires the OpenAI Responses "
            "service dependencies. Run `uv sync --python 3.12 --group dev` and verify "
            "the `openai` and `websockets` packages are installed."
        ) from exc

    return OpenAIResponsesLLMService


def verify_openai_responses_config(config: LocalLLMConfig):
    """Verify local OpenAI Responses config without making a network call."""
    if config.provider != "openai-responses":
        raise ValueError(f"Expected openai-responses config, got {config.provider!r}.")
    if not _nonempty(os.getenv("OPENAI_API_KEY")):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the `openai-responses` local LLM provider."
        )
    if not _nonempty(config.model):
        raise RuntimeError("An OpenAI Responses model is required.")
    _require_openai_responses_service()


async def verify_local_llm_config(config: LocalLLMConfig):
    """Verify the selected local LLM provider configuration."""
    if config.provider == "ollama":
        await verify_ollama(config.base_url or DEFAULT_OLLAMA_BASE_URL, config.model)
        return
    if config.provider == "openai-responses":
        verify_openai_responses_config(config)
        return
    raise ValueError(f"Unsupported local LLM provider: {config.provider!r}")


def create_ollama_llm_service(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    temperature: Optional[float] = None,
) -> OllamaLLMService:
    """Create the shared local Ollama LLM service."""
    settings = OllamaLLMService.Settings(
        model=model,
        system_instruction=system_prompt,
        # Ollama's OpenAI-compatible chat completions API exposes reasoning
        # control via top-level reasoning_effort/reasoning fields, not the
        # native /api/generate `think` flag.
        extra={"reasoning_effort": "none"},
    )
    if temperature is not None:
        settings.temperature = temperature

    return OllamaLLMService(base_url=base_url, settings=settings)


def create_openai_responses_llm_service(
    *,
    model: str,
    system_prompt: str,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    service_tier: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
):
    """Create the shared local OpenAI Responses LLM service."""
    OpenAIResponsesLLMService = _require_openai_responses_service()
    settings = OpenAIResponsesLLMService.Settings(
        model=model,
        system_instruction=system_prompt,
    )
    if temperature is not None:
        settings.temperature = temperature
    if max_output_tokens is not None:
        settings.max_completion_tokens = max_output_tokens

    kwargs: dict[str, Any] = {"settings": settings}
    if _nonempty(base_url):
        kwargs["base_url"] = base_url
    if _nonempty(service_tier):
        kwargs["service_tier"] = service_tier

    return OpenAIResponsesLLMService(**kwargs)


def create_local_llm_service(config: LocalLLMConfig):
    """Create the selected local LLM service."""
    if config.provider == "ollama":
        return create_ollama_llm_service(
            base_url=config.base_url or DEFAULT_OLLAMA_BASE_URL,
            model=config.model,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
        )
    if config.provider == "openai-responses":
        return create_openai_responses_llm_service(
            model=config.model,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            base_url=config.base_url,
            service_tier=config.service_tier,
            max_output_tokens=config.max_output_tokens,
        )
    raise ValueError(f"Unsupported local LLM provider: {config.provider!r}")


def create_local_stt_service(*, backend: str, model: str, language: Language):
    """Create the configured local STT backend with a clean install hint."""
    if backend == "mlx-whisper":
        try:
            from blink.services.whisper.stt import WhisperSTTServiceMLX
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The MLX Whisper backend requires the `mlx-whisper` extra. "
                "Run `./scripts/bootstrap-local-mac.sh --profile voice` or "
                "`uv sync --python 3.12 --group dev --extra mlx-whisper`."
            ) from exc

        return WhisperSTTServiceMLX(
            settings=WhisperSTTServiceMLX.Settings(
                model=model,
                language=language,
                no_speech_prob=0.6,
                temperature=0.0,
                engine="mlx",
            ),
            ttfs_p99_latency=WHISPER_TTFS_P99,
        )

    if backend == "whisper":
        try:
            from blink.services.whisper.stt import WhisperSTTService
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The Faster Whisper backend requires the `whisper` extra. "
                "Run `uv sync --python 3.12 --group dev --extra whisper`."
            ) from exc

        return WhisperSTTService(
            settings=WhisperSTTService.Settings(
                model=model,
                language=language,
                no_speech_prob=0.4,
            )
        )

    raise ValueError(f"Unsupported STT backend: {backend}")


def create_local_tts_service(
    *,
    backend: str,
    voice: Optional[str],
    language: Language,
    base_url: Optional[str] = None,
    aiohttp_session: Optional["aiohttp.ClientSession"] = None,
    reuse_context_id_within_turn: bool = True,
):
    """Create the configured local TTS backend with a clean install hint."""
    text_filters = [MarkdownTextFilter()]
    if is_chinese_language(language):
        text_filters.append(ChineseSpeechTextFilter())

    if backend == "kokoro":
        try:
            from blink.services.kokoro.tts import KokoroTTSService
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The Kokoro backend requires the `kokoro` extra. "
                "Run `./scripts/bootstrap-local-mac.sh --profile voice` or "
                "`uv sync --python 3.12 --group dev --extra kokoro`."
            ) from exc

        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=voice,
                language=language,
            ),
            text_filters=text_filters,
            reuse_context_id_within_turn=reuse_context_id_within_turn,
        )

    if backend == "piper":
        try:
            from blink.services.piper.tts import PiperTTSService
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The Piper backend requires the `piper` extra. "
                "Run `uv sync --python 3.12 --group dev --extra piper`."
            ) from exc

        PIPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return PiperTTSService(
            download_dir=PIPER_CACHE_DIR,
            settings=PiperTTSService.Settings(voice=voice, language=language),
            text_filters=text_filters,
            reuse_context_id_within_turn=reuse_context_id_within_turn,
        )

    if backend == "xtts":
        if aiohttp_session is None:
            raise LocalDependencyError("XTTS backend requires an aiohttp session.")
        try:
            from blink.services.xtts.tts import XTTSService
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The XTTS backend is unavailable in this checkout. "
                "Verify `src/blink/services/xtts/tts.py` is present."
            ) from exc

        xtts_base_url = base_url or DEFAULT_LOCAL_XTTS_BASE_URL
        xtts_voice = resolve_xtts_voice(xtts_base_url, voice)
        return XTTSService(
            aiohttp_session=aiohttp_session,
            base_url=xtts_base_url,
            settings=XTTSService.Settings(
                voice=xtts_voice,
                language=language,
            ),
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            text_filters=text_filters,
            reuse_context_id_within_turn=reuse_context_id_within_turn,
        )

    if backend == "local-http-wav":
        if aiohttp_session is None:
            raise LocalDependencyError("Local HTTP WAV backend requires an aiohttp session.")
        try:
            from blink.services.local_http_wav.tts import LocalHttpWavTTSService
        except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
            raise LocalDependencyError(
                "The local-http-wav backend is unavailable in this checkout."
            ) from exc

        return LocalHttpWavTTSService(
            aiohttp_session=aiohttp_session,
            base_url=base_url or DEFAULT_LOCAL_HTTP_WAV_TTS_BASE_URL,
            settings=LocalHttpWavTTSService.Settings(
                voice=voice,
                language=language,
            ),
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            text_filters=text_filters,
            reuse_context_id_within_turn=reuse_context_id_within_turn,
        )

    raise ValueError(f"Unsupported TTS backend: {backend}")


def create_local_vision_service(*, model: str):
    """Create the optional local Moondream vision backend."""
    try:
        from blink.services.moondream.vision import MoondreamService
    except Exception as exc:  # pragma: no cover - exercised through doctor/runtime
        raise LocalDependencyError(
            "Local vision requires the `moondream` extra. "
            "Run `./scripts/bootstrap-local-mac.sh --profile browser --with-vision` "
            "or sync with `--extra moondream`."
        ) from exc

    try:
        return MoondreamService(settings=MoondreamService.Settings(model=model))
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised through doctor/runtime
        raise LocalDependencyError(
            "Local browser vision is missing a required Moondream dependency. "
            "Run `./scripts/bootstrap-local-mac.sh --profile browser --with-vision` "
            "or sync with `--extra moondream`."
        ) from exc


def build_local_user_aggregators(
    context: LLMContext,
    *,
    mute_during_bot_speech: bool = False,
    extra_user_mute_strategies: Optional[list["BaseUserMuteStrategy"]] = None,
    user_turn_start_strategies: Optional[list["BaseUserTurnStartStrategy"]] = None,
):
    """Build the default local user and assistant aggregators."""
    from blink.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from blink.audio.vad.silero import SileroVADAnalyzer
    from blink.turns.user_mute import AlwaysUserMuteStrategy, BaseUserMuteStrategy
    from blink.turns.user_stop import (
        SpeechTimeoutUserTurnStopStrategy,
        TurnAnalyzerUserTurnStopStrategy,
    )
    from blink.turns.user_turn_strategies import UserTurnStrategies

    user_mute_strategies: list[BaseUserMuteStrategy] = []
    if mute_during_bot_speech:
        user_mute_strategies.append(AlwaysUserMuteStrategy())
    if extra_user_mute_strategies:
        user_mute_strategies.extend(extra_user_mute_strategies)

    turn_strategy_kwargs: dict[str, Any] = {
        "stop": [
            TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3()),
            SpeechTimeoutUserTurnStopStrategy(),
        ]
    }
    if user_turn_start_strategies is not None:
        turn_strategy_kwargs["start"] = user_turn_start_strategies

    return LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(**turn_strategy_kwargs),
            user_mute_strategies=user_mute_strategies,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )


def build_local_voice_task(
    *,
    transport,
    stt,
    llm,
    tts,
    context: Optional[LLMContext] = None,
    idle_timeout_secs: Optional[float] = None,
    mute_during_bot_speech: bool = False,
    rtvi_user_mute_enabled: bool = True,
    extra_user_mute_strategies: Optional[list["BaseUserMuteStrategy"]] = None,
    user_turn_start_strategies: Optional[list["BaseUserTurnStartStrategy"]] = None,
    pre_stt_processors: Optional[list["FrameProcessor"]] = None,
    post_stt_processors: Optional[list["FrameProcessor"]] = None,
    pre_llm_processors: Optional[list["FrameProcessor"]] = None,
    pre_tts_processors: Optional[list["FrameProcessor"]] = None,
    pre_output_processors: Optional[list["FrameProcessor"]] = None,
    post_context_processors: Optional[list["FrameProcessor"]] = None,
) -> tuple[PipelineTask, LLMContext]:
    """Build the shared local voice pipeline task for native and browser transports."""
    context = context or LLMContext()
    user_aggregator, assistant_aggregator = build_local_user_aggregators(
        context,
        mute_during_bot_speech=mute_during_bot_speech,
        extra_user_mute_strategies=extra_user_mute_strategies,
        user_turn_start_strategies=user_turn_start_strategies,
    )
    pre_stt_processors = pre_stt_processors or []
    post_stt_processors = post_stt_processors or []
    pre_llm_processors = pre_llm_processors or []
    pre_tts_processors = pre_tts_processors or []
    pre_output_processors = pre_output_processors or []
    post_context_processors = post_context_processors or []

    task = PipelineTask(
        Pipeline(
            [
                transport.input(),
                *pre_stt_processors,
                stt,
                *post_stt_processors,
                user_aggregator,
                *pre_llm_processors,
                llm,
                *pre_tts_processors,
                tts,
                *pre_output_processors,
                transport.output(),
                assistant_aggregator,
                *post_context_processors,
            ]
        ),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=idle_timeout_secs,
        rtvi_observer_params=RTVIObserverParams(user_mute_enabled=rtvi_user_mute_enabled),
    )

    return task, context


def _create_pyaudio():
    """Create a PyAudio instance with a clean install hint."""
    try:
        import pyaudio
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised through doctor/runtime
        raise LocalDependencyError(
            "Local audio device enumeration requires the `local` extra. "
            "Run `brew install portaudio` and "
            "`uv sync --python 3.12 --group dev --extra local`."
        ) from exc

    return pyaudio.PyAudio()


def get_audio_devices() -> list[AudioDeviceInfo]:
    """Enumerate local PyAudio devices with a clean install hint."""
    py_audio = _create_pyaudio()
    try:
        devices: list[AudioDeviceInfo] = []
        for index in range(py_audio.get_device_count()):
            info = py_audio.get_device_info_by_index(index)
            devices.append(_audio_device_from_mapping(index, info))
        return devices
    finally:
        py_audio.terminate()


def get_default_audio_device(kind: str) -> Optional[AudioDeviceInfo]:
    """Return the default PyAudio input or output device, if one is available."""
    if kind not in {"input", "output"}:
        raise ValueError(f"Unsupported audio device kind: {kind}")

    py_audio = _create_pyaudio()
    try:
        try:
            info = (
                py_audio.get_default_input_device_info()
                if kind == "input"
                else py_audio.get_default_output_device_info()
            )
        except OSError:
            return None

        return _audio_device_from_mapping(int(info.get("index", -1)), info)
    finally:
        py_audio.terminate()


def get_audio_device_by_index(index: Optional[int]) -> Optional[AudioDeviceInfo]:
    """Return a single enumerated audio device by index."""
    if index is None:
        return None

    return next((device for device in get_audio_devices() if device.index == index), None)


def _looks_like_display_audio(name: str) -> bool:
    """Heuristic for monitor/display-routed audio devices on macOS."""
    normalized = name.lower()
    return "display audio" in normalized or "monitor" in normalized


def _find_builtin_macbook_device(
    devices: list[AudioDeviceInfo], *, kind: str
) -> Optional[AudioDeviceInfo]:
    """Find a built-in MacBook microphone or speaker device, if present."""
    if kind == "input":
        return next(
            (
                device
                for device in devices
                if device.max_input_channels > 0
                and "macbook" in device.name.lower()
                and "microphone" in device.name.lower()
            ),
            None,
        )

    if kind == "output":
        return next(
            (
                device
                for device in devices
                if device.max_output_channels > 0
                and "macbook" in device.name.lower()
                and "speaker" in device.name.lower()
            ),
            None,
        )

    raise ValueError(f"Unsupported audio device kind: {kind}")


def resolve_preferred_audio_device_indexes(
    input_device_index: Optional[int],
    output_device_index: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Resolve local audio devices with a MacBook-friendly fallback.

    On macOS, external displays often become the system default audio route.
    For the local native voice loop we prefer built-in MacBook microphone and
    speakers when the defaults point at display audio and the user did not
    explicitly choose devices.
    """
    if sys.platform != "darwin":
        return input_device_index, output_device_index

    if input_device_index is not None and output_device_index is not None:
        return input_device_index, output_device_index

    try:
        devices = get_audio_devices()
    except LocalDependencyError:
        return input_device_index, output_device_index

    if input_device_index is None:
        default_input = get_default_audio_device("input")
        if default_input and _looks_like_display_audio(default_input.name):
            builtin_input = _find_builtin_macbook_device(devices, kind="input")
            if builtin_input is not None:
                input_device_index = builtin_input.index

    if output_device_index is None:
        default_output = get_default_audio_device("output")
        if default_output and _looks_like_display_audio(default_output.name):
            builtin_output = _find_builtin_macbook_device(devices, kind="output")
            if builtin_output is not None:
                output_device_index = builtin_output.index

    return input_device_index, output_device_index

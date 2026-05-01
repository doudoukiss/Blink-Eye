import asyncio
import io
import sys
import types

import pytest

from blink.cli import local_chat, local_common
from blink.cli.local_chat import (
    build_parser,
    resolve_config,
)
from blink.cli.local_common import (
    DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_RESPONSES_MODEL,
    ChineseSpeechTextFilter,
    LocalDependencyError,
    LocalLLMConfig,
    LocalRuntimeTTSSelection,
    build_local_user_aggregators,
    create_local_tts_service,
    create_ollama_llm_service,
    create_openai_responses_llm_service,
    default_local_speech_system_prompt,
    default_local_text_system_prompt,
    default_local_tts_backend,
    default_local_tts_voice,
    normalize_chinese_speech_text,
    resolve_local_http_wav_voice,
    resolve_local_llm_provider,
    resolve_local_runtime_tts_selection,
    resolve_local_tts_voice,
    verify_local_llm_config,
)
from blink.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from blink.processors.aggregators.llm_context import LLMContext
from blink.processors.console import ConsoleLLMPrinter
from blink.processors.frame_processor import FrameDirection
from blink.services.ollama.llm import OLLamaLLMService, OllamaLLMService
from blink.transcriptions.language import Language
from blink.turns.user_mute import FirstSpeechUserMuteStrategy
from blink.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from blink.utils.text.markdown_text_filter import MarkdownTextFilter


def test_local_chat_config_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "")
    monkeypatch.setenv("BLINK_LOCAL_DEMO_MODE", "")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "")

    args = build_parser().parse_args([])
    config = resolve_config(args)

    assert config.llm.provider == "ollama"
    assert config.llm.base_url == DEFAULT_OLLAMA_BASE_URL
    assert config.llm.model == DEFAULT_OLLAMA_MODEL
    assert config.language == Language.ZH
    assert config.llm.system_prompt == default_local_text_system_prompt(Language.ZH)


def test_local_chat_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "Custom prompt")
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    args = build_parser().parse_args([])
    config = resolve_config(args)

    assert config.llm.provider == "ollama"
    assert config.llm.base_url == "http://localhost:9999/v1"
    assert config.llm.model == "custom-model"
    assert config.llm.system_prompt == "Custom prompt"
    assert config.language == Language.EN


def test_local_chat_config_resolves_openai_responses_from_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_MODEL", "gpt-demo")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "flex")
    monkeypatch.setenv("BLINK_LOCAL_LLM_SYSTEM_PROMPT", "Shared local prompt")
    monkeypatch.setenv("OLLAMA_MODEL", "ignored-ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ignored.test/v1")
    monkeypatch.setenv("OLLAMA_SYSTEM_PROMPT", "Ignored prompt")

    args = build_parser().parse_args([])
    config = resolve_config(args)

    assert config.llm == LocalLLMConfig(
        provider="openai-responses",
        model="gpt-demo",
        base_url="https://example.test/v1",
        system_prompt="Shared local prompt",
        service_tier="flex",
    )


def test_local_chat_cli_provider_overrides_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_MODEL", "gpt-env")

    args = build_parser().parse_args(
        [
            "--llm-provider",
            "openai-responses",
            "--model",
            "gpt-cli",
            "--base-url",
            "https://proxy.test/v1",
            "--system-prompt",
            "CLI prompt",
            "--temperature",
            "0.2",
        ]
    )
    config = resolve_config(args)

    assert config.llm.provider == "openai-responses"
    assert config.llm.model == "gpt-cli"
    assert config.llm.base_url == "https://proxy.test/v1"
    assert config.llm.system_prompt == "CLI prompt"
    assert config.llm.temperature == 0.2


def test_local_chat_demo_mode_defaults_openai_budget_and_priority(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "")
    monkeypatch.setenv("BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS", "")

    args = build_parser().parse_args(["--demo-mode", "--language", "en"])
    config = resolve_config(args)

    assert config.demo_mode is True
    assert config.llm.demo_mode is True
    assert config.llm.service_tier == "priority"
    assert config.llm.max_output_tokens == DEFAULT_LOCAL_DEMO_MAX_OUTPUT_TOKENS
    assert "Answer immediately in the first sentence" in config.llm.system_prompt


def test_local_chat_demo_mode_preserves_explicit_openai_budget_and_tier(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "openai-responses")
    monkeypatch.setenv("BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER", "flex")
    monkeypatch.setenv("BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS", "240")

    args = build_parser().parse_args(["--demo-mode"])
    config = resolve_config(args)

    assert config.llm.service_tier == "flex"
    assert config.llm.max_output_tokens == 240


def test_local_chat_demo_mode_tightens_ollama_prompt_without_openai_knobs(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LLM_PROVIDER", "ollama")

    args = build_parser().parse_args(["--demo-mode", "--max-output-tokens", "240"])
    config = resolve_config(args)

    assert config.llm.provider == "ollama"
    assert config.llm.demo_mode is True
    assert config.llm.service_tier is None
    assert config.llm.max_output_tokens is None
    assert "演示模式" in config.llm.system_prompt


def test_resolve_local_llm_provider_rejects_unsupported_alias():
    with pytest.raises(ValueError, match="Unsupported local LLM provider"):
        resolve_local_llm_provider("openai")


def test_ollama_service_alias_is_available():
    assert OllamaLLMService is OLLamaLLMService


def test_local_ollama_service_disables_reasoning_by_default():
    service = create_ollama_llm_service(
        base_url=DEFAULT_OLLAMA_BASE_URL,
        model=DEFAULT_OLLAMA_MODEL,
        system_prompt=default_local_text_system_prompt(Language.ZH),
    )

    assert service._settings.extra == {"reasoning_effort": "none"}
    assert "简体中文" in service._settings.system_instruction


def test_openai_responses_service_factory_uses_existing_service(monkeypatch):
    from blink.services.openai.responses.llm import OpenAIResponsesLLMService

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        OpenAIResponsesLLMService,
        "_create_client",
        lambda self, **_kwargs: object(),
    )

    service = create_openai_responses_llm_service(
        model="gpt-demo",
        system_prompt="Prompt",
        temperature=0.3,
        base_url="https://proxy.test/v1",
        service_tier="flex",
        max_output_tokens=120,
    )

    assert isinstance(service, OpenAIResponsesLLMService)
    assert service._settings.model == "gpt-demo"
    assert service._settings.system_instruction == "Prompt"
    assert service._settings.temperature == 0.3
    assert service._settings.max_completion_tokens == 120
    assert service._service_tier == "flex"


@pytest.mark.asyncio
async def test_openai_responses_verification_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = LocalLLMConfig(
        provider="openai-responses",
        model=DEFAULT_OPENAI_RESPONSES_MODEL,
        base_url=None,
        system_prompt="Prompt",
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await verify_local_llm_config(config)


@pytest.mark.asyncio
async def test_openai_responses_verification_is_local_only(monkeypatch):
    called = False

    def fake_require_service():
        nonlocal called
        called = True
        return object

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(local_common, "_require_openai_responses_service", fake_require_service)

    await verify_local_llm_config(
        LocalLLMConfig(
            provider="openai-responses",
            model=DEFAULT_OPENAI_RESPONSES_MODEL,
            base_url=None,
            system_prompt="Prompt",
        )
    )

    assert called is True


@pytest.mark.asyncio
async def test_run_local_chat_uses_provider_aware_factory(monkeypatch):
    calls: dict[str, object] = {}
    config = local_chat.LocalChatConfig(
        llm=LocalLLMConfig(
            provider="openai-responses",
            model="gpt-demo",
            base_url=None,
            system_prompt="Prompt",
        ),
        language=Language.EN,
        once="hello",
        show_banner=False,
    )

    async def fake_verify(llm_config):
        calls["verified"] = llm_config

    def fake_create(llm_config):
        calls["created"] = llm_config
        return object()

    class FakeTask:
        def __init__(self, pipeline):
            calls["pipeline"] = pipeline
            self.frames = []

        async def queue_frames(self, frames):
            self.frames.extend(frames)

        async def cancel(self):
            calls["cancelled"] = True

    class FakeRunner:
        def __init__(self, *, handle_sigint):
            calls["handle_sigint"] = handle_sigint

        async def run(self, _task):
            return None

    async def fake_await_response(_queue, _runner_task):
        return "hi"

    monkeypatch.setattr(local_chat, "verify_local_llm_config", fake_verify)
    monkeypatch.setattr(local_chat, "create_local_llm_service", fake_create)
    monkeypatch.setattr(local_chat, "Pipeline", lambda processors: tuple(processors))
    monkeypatch.setattr(local_chat, "PipelineTask", FakeTask)
    monkeypatch.setattr(local_chat, "PipelineRunner", FakeRunner)
    monkeypatch.setattr(local_chat, "_await_response", fake_await_response)

    status = await local_chat.run_local_chat(config)

    assert status == 0
    assert calls["verified"] is config.llm
    assert calls["created"] is config.llm
    assert calls["cancelled"] is True


def test_local_prompt_defaults_split_text_and_speech():
    assert "Markdown" not in default_local_text_system_prompt(Language.ZH)
    assert "Markdown" in default_local_speech_system_prompt(Language.ZH)
    assert "file paths" in default_local_speech_system_prompt(Language.EN)
    assert "四到八句" in default_local_speech_system_prompt(Language.ZH)
    assert "four to eight spoken sentences" in default_local_speech_system_prompt(Language.EN)


def test_build_local_user_aggregators_can_mute_during_bot_speech():
    pair = build_local_user_aggregators(LLMContext(), mute_during_bot_speech=True)

    assert len(pair.user()._params.user_mute_strategies) == 1


def test_build_local_user_aggregators_allows_barge_in_when_requested():
    pair = build_local_user_aggregators(LLMContext(), mute_during_bot_speech=False)

    assert pair.user()._params.user_mute_strategies == []


def test_build_local_user_aggregators_accepts_extra_user_mute_strategies():
    pair = build_local_user_aggregators(
        LLMContext(),
        extra_user_mute_strategies=[FirstSpeechUserMuteStrategy()],
    )

    assert len(pair.user()._params.user_mute_strategies) == 1
    assert isinstance(pair.user()._params.user_mute_strategies[0], FirstSpeechUserMuteStrategy)


def test_build_local_user_aggregators_adds_smart_turn_with_timeout_fallback():
    pair = build_local_user_aggregators(LLMContext())

    strategies = pair.user()._params.user_turn_strategies.stop
    assert strategies is not None
    assert len(strategies) == 2
    assert isinstance(strategies[0], TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(strategies[1], SpeechTimeoutUserTurnStopStrategy)


def test_local_tts_service_adds_markdown_filter(monkeypatch):
    module = types.ModuleType("blink.services.kokoro.tts")

    class FakeKokoroTTSService:
        class Settings:
            def __init__(self, *, voice: str, language: Language):
                self.voice = voice
                self.language = language

        def __init__(self, *, settings, text_filters):
            self.settings = settings
            self.text_filters = text_filters

    module.KokoroTTSService = FakeKokoroTTSService
    monkeypatch.setitem(sys.modules, "blink.services.kokoro.tts", module)

    service = create_local_tts_service(backend="kokoro", voice="af_heart", language=Language.EN)

    assert any(isinstance(text_filter, MarkdownTextFilter) for text_filter in service.text_filters)
    assert service.settings.voice == "af_heart"
    assert service.settings.language == Language.EN


def test_local_piper_tts_service_adds_markdown_filter_and_language(monkeypatch):
    module = types.ModuleType("blink.services.piper.tts")

    class FakePiperTTSService:
        class Settings:
            def __init__(self, *, voice: str, language: Language):
                self.voice = voice
                self.language = language

        def __init__(self, *, download_dir, settings, text_filters):
            self.download_dir = download_dir
            self.settings = settings
            self.text_filters = text_filters

    module.PiperTTSService = FakePiperTTSService
    monkeypatch.setitem(sys.modules, "blink.services.piper.tts", module)

    service = create_local_tts_service(
        backend="piper",
        voice="en_US-ryan-high",
        language=Language.EN,
    )

    assert any(isinstance(text_filter, MarkdownTextFilter) for text_filter in service.text_filters)
    assert service.settings.voice == "en_US-ryan-high"
    assert service.settings.language == Language.EN


def test_default_local_tts_voice_supports_piper():
    assert default_local_tts_voice("piper", Language.EN) == "en_US-ryan-high"
    assert default_local_tts_voice("piper", Language.ZH) == "zh_CN-xiao_ya-medium"
    assert default_local_tts_backend(Language.ZH) == "kokoro"
    assert default_local_tts_backend(Language.EN) == "kokoro"
    assert default_local_tts_voice("xtts", Language.ZH) is None
    assert default_local_tts_voice("kokoro", Language.ZH) == "zf_xiaobei"


def test_resolve_local_tts_voice_prefers_language_specific_over_generic(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_ZH", "zf_xiaoyi")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "af_sky")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")

    assert resolve_local_tts_voice("kokoro", Language.ZH) == "zf_xiaoyi"
    assert resolve_local_tts_voice("kokoro", Language.EN) == "af_sky"


def test_resolve_local_tts_voice_ignores_removed_legacy_env(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.setenv("LEGACY_LOCAL_TTS_VOICE_ZH", "zf_xiaoyi")
    monkeypatch.setenv("LEGACY_LOCAL_TTS_VOICE_EN", "af_sky")
    monkeypatch.setenv("LEGACY_LOCAL_TTS_VOICE", "af_bella")

    assert resolve_local_tts_voice("kokoro", Language.ZH) == "zf_xiaobei"
    assert resolve_local_tts_voice("kokoro", Language.EN) == "af_heart"


def test_resolve_local_tts_voice_prefers_explicit_then_generic_then_default(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "af_sky")

    assert resolve_local_tts_voice("kokoro", Language.ZH, explicit_voice="cli-voice") == "cli-voice"
    assert resolve_local_tts_voice("kokoro", Language.EN) == "af_sky"

    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "")
    assert resolve_local_tts_voice("kokoro", Language.EN) == "af_heart"


def test_resolve_local_tts_voice_local_http_wav_ignores_backend_specific_envs(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_ZH", "zf_xiaobei")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "af_heart")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")

    assert resolve_local_tts_voice("local-http-wav", Language.ZH) is None
    assert resolve_local_tts_voice("local-http-wav", Language.EN) is None


def test_resolve_local_tts_voice_local_http_wav_accepts_explicit_voice(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_ZH", "zf_xiaobei")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")

    assert resolve_local_tts_voice("local-http-wav", Language.ZH, explicit_voice="ZH") == "ZH"


def test_resolve_local_tts_voice_piper_ignores_kokoro_voice_overrides(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_EN", "af_heart")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "zf_xiaobei")

    assert resolve_local_tts_voice("piper", Language.EN) == "en_US-ryan-high"


@pytest.mark.asyncio
async def test_resolve_local_http_wav_voice_falls_back_to_catalog_default(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "default_speakers": {"zh": "ZH", "en": "EN-US"},
                "speakers": {"zh": ["ZH"], "en": ["EN-US"]},
            }

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            assert url == "http://127.0.0.1:8012/voices"
            return FakeResponse()

    monkeypatch.setattr(
        "blink.cli.local_common.httpx.AsyncClient", lambda timeout: FakeAsyncClient()
    )

    resolved = await resolve_local_http_wav_voice(
        "http://127.0.0.1:8012",
        "zf_xiaobei",
        Language.ZH,
    )

    assert resolved == "ZH"


@pytest.mark.asyncio
async def test_resolve_local_runtime_tts_selection_prefers_local_http_wav_for_chinese(
    monkeypatch,
):
    monkeypatch.setenv("LEGACY_LOCAL_HTTP_WAV_TTS_BASE_URL", "http://127.0.0.1:8012")
    monkeypatch.setenv("BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL", "http://127.0.0.1:8001")

    async def fake_service_available(base_url):
        assert base_url == "http://127.0.0.1:8001"
        return True

    async def fake_resolve_http_wav_voice(base_url, voice, language):
        assert base_url == "http://127.0.0.1:8001"
        assert voice is None
        assert language == Language.ZH
        return None

    monkeypatch.setattr(
        "blink.cli.local_common.local_http_wav_service_available",
        fake_service_available,
    )
    monkeypatch.setattr(
        "blink.cli.local_common.resolve_local_http_wav_voice",
        fake_resolve_http_wav_voice,
    )

    selection = await resolve_local_runtime_tts_selection(
        language=Language.ZH,
        requested_backend="kokoro",
        requested_voice="zf_xiaobei",
        requested_base_url=None,
        backend_locked=False,
        explicit_voice=None,
    )

    assert selection == LocalRuntimeTTSSelection(
        backend="local-http-wav",
        voice=None,
        base_url="http://127.0.0.1:8001",
        auto_switched=True,
    )


@pytest.mark.asyncio
async def test_resolve_local_runtime_tts_selection_keeps_locked_backend(monkeypatch):
    async def fake_service_available(_base_url):
        return True

    monkeypatch.setattr(
        "blink.cli.local_common.local_http_wav_service_available",
        fake_service_available,
    )

    selection = await resolve_local_runtime_tts_selection(
        language=Language.ZH,
        requested_backend="kokoro",
        requested_voice="zf_xiaobei",
        requested_base_url=None,
        backend_locked=True,
        explicit_voice=None,
    )

    assert selection == LocalRuntimeTTSSelection(
        backend="kokoro",
        voice="zf_xiaobei",
        base_url=None,
        auto_switched=False,
    )


@pytest.mark.asyncio
async def test_chinese_speech_text_filter_normalizes_symbols():
    text_filter = ChineseSpeechTextFilter()

    filtered = await text_filter.filter(
        "你好 *Blink* https://example.com/test?a=1 v2.3 ./scripts/run-local-chat.sh "
        "2026-04-15 15:20 API 120ms A-1024"
    )

    assert "网页链接" in filtered
    assert "*" not in filtered
    assert "版本 2.3" in filtered
    assert "路径" in filtered
    assert "2026年4月15日" in filtered
    assert "15点20分" in filtered
    assert "接口" in filtered
    assert "120毫秒" in filtered
    assert "A 1024" in filtered


def test_normalize_chinese_speech_text_separates_mixed_technical_content():
    filtered = normalize_chinese_speech_text(
        "请检查Blink API状态，任务ID是A-1024，并打开~/demo/log.txt；现在时间是15:20。"
    )

    assert "Blink 接口状态" in filtered
    assert "任务编号是 A 1024" in filtered
    assert "路径" in filtered
    assert "15点20分" in filtered


def test_normalize_chinese_speech_text_rewrites_latency_and_cpu_terms():
    filtered = normalize_chinese_speech_text("当前HTTP延迟约为120ms，CPU占用稳定，结果OK。")

    assert "网络请求" in filtered
    assert "120毫秒" in filtered
    assert "处理器" in filtered
    assert "正常" in filtered


def test_local_xtts_tts_service_uses_sentence_aggregation_and_resolves_voice(monkeypatch):
    module = types.ModuleType("blink.services.xtts.tts")

    class FakeXTTSService:
        class Settings:
            def __init__(self, *, voice: str, language: Language):
                self.voice = voice
                self.language = language

        def __init__(
            self, *, aiohttp_session, base_url, settings, text_aggregation_mode, text_filters
        ):
            self.aiohttp_session = aiohttp_session
            self.base_url = base_url
            self.settings = settings
            self.text_aggregation_mode = text_aggregation_mode
            self.text_filters = text_filters

    module.XTTSService = FakeXTTSService
    monkeypatch.setitem(sys.modules, "blink.services.xtts.tts", module)
    monkeypatch.setattr(
        "blink.cli.local_common.resolve_xtts_voice", lambda base_url, voice: "speaker-a"
    )

    service = create_local_tts_service(
        backend="xtts",
        voice=None,
        language=Language.ZH,
        base_url="http://127.0.0.1:8000",
        aiohttp_session=object(),
    )

    assert service.base_url == "http://127.0.0.1:8000"
    assert service.settings.voice == "speaker-a"
    assert service.settings.language == Language.ZH
    assert any(
        isinstance(text_filter, ChineseSpeechTextFilter) for text_filter in service.text_filters
    )


def test_local_http_wav_tts_service_builds(monkeypatch):
    module = types.ModuleType("blink.services.local_http_wav.tts")

    class FakeLocalHttpWavTTSService:
        class Settings:
            def __init__(self, *, voice, language):
                self.voice = voice
                self.language = language

        def __init__(
            self, *, aiohttp_session, base_url, settings, text_aggregation_mode, text_filters
        ):
            self.aiohttp_session = aiohttp_session
            self.base_url = base_url
            self.settings = settings
            self.text_aggregation_mode = text_aggregation_mode
            self.text_filters = text_filters

    module.LocalHttpWavTTSService = FakeLocalHttpWavTTSService
    monkeypatch.setitem(sys.modules, "blink.services.local_http_wav.tts", module)

    service = create_local_tts_service(
        backend="local-http-wav",
        voice=None,
        language=Language.ZH,
        base_url="http://127.0.0.1:8001",
        aiohttp_session=object(),
    )

    assert service.base_url == "http://127.0.0.1:8001"
    assert service.settings.language == Language.ZH
    assert any(
        isinstance(text_filter, ChineseSpeechTextFilter) for text_filter in service.text_filters
    )


def test_local_vision_service_wraps_missing_runtime_dependency(monkeypatch):
    module = types.ModuleType("blink.services.moondream.vision")

    class FakeMoondreamService:
        class Settings:
            def __init__(self, *, model: str):
                self.model = model

        def __init__(self, *, settings):
            raise ModuleNotFoundError("No module named 'pyvips'")

    module.MoondreamService = FakeMoondreamService
    monkeypatch.setitem(sys.modules, "blink.services.moondream.vision", module)

    with pytest.raises(LocalDependencyError, match="Local browser vision is missing a required"):
        from blink.cli.local_common import create_local_vision_service

        create_local_vision_service(model="vikhyatk/moondream2")


@pytest.mark.asyncio
async def test_console_llm_printer_streams_and_collects_response():
    stream = io.StringIO()
    response_queue: asyncio.Queue[str] = asyncio.Queue()
    printer = ConsoleLLMPrinter(stream=stream, response_queue=response_queue)

    await printer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await printer.process_frame(LLMTextFrame(text="Hello"), FrameDirection.DOWNSTREAM)
    await printer.process_frame(LLMTextFrame(text=" world"), FrameDirection.DOWNSTREAM)
    await printer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert stream.getvalue() == "assistant> Hello world\n"
    assert await response_queue.get() == "Hello world"

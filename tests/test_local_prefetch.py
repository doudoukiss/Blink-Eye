import argparse

from blink.cli import local_prefetch


def test_local_prefetch_resolve_config_switches_backend_defaults(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "en")

    config = local_prefetch.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language=None,
            model=None,
            stt_backend="whisper",
            stt_model=None,
            tts_backend="piper",
            tts_voice=None,
        )
    )

    assert config.stt_model == "Systran/faster-distil-whisper-medium.en"
    assert config.tts_voice == "en_US-ryan-high"
    assert config.language == local_prefetch.Language.EN


def test_local_prefetch_resolve_config_prefers_language_specific_voice(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("BLINK_LOCAL_TTS_BACKEND", "kokoro")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE", "generic-voice")
    monkeypatch.setenv("BLINK_LOCAL_TTS_VOICE_ZH", "zf_xiaoyi")

    config = local_prefetch.resolve_config(
        argparse.Namespace(
            profile="voice",
            with_vision=False,
            language=None,
            model=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config.tts_voice == "zf_xiaoyi"
    assert config.language == local_prefetch.Language.ZH


def test_local_prefetch_reads_vision_model_from_environment(monkeypatch):
    monkeypatch.setenv("BLINK_LOCAL_VISION_MODEL", "custom/moondream")
    monkeypatch.setenv("BLINK_LOCAL_BROWSER_VISION", "true")
    monkeypatch.delenv("BLINK_LOCAL_LANGUAGE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_STT_MODEL", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_BACKEND", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)

    config = local_prefetch.resolve_config(
        argparse.Namespace(
            profile="browser",
            with_vision=True,
            language=None,
            model=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config.with_vision is True
    assert config.vision_model == "custom/moondream"
    assert config.language == local_prefetch.Language.ZH
    assert config.stt_model == "mlx-community/whisper-medium-mlx"
    assert config.tts_backend == "kokoro"
    assert config.tts_voice == "zf_xiaobei"


def test_local_prefetch_ignores_removed_legacy_env(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_BROWSER_VISION", raising=False)
    monkeypatch.setenv("LEGACY_LOCAL_LANGUAGE", "en")
    monkeypatch.setenv("BLINK_LOCAL_LANGUAGE", "zh")
    monkeypatch.setenv("LEGACY_LOCAL_VISION_MODEL", "legacy/moondream")
    monkeypatch.setenv("BLINK_LOCAL_VISION_MODEL", "blink/moondream")

    config = local_prefetch.resolve_config(
        argparse.Namespace(
            profile="browser",
            with_vision=True,
            language=None,
            model=None,
            stt_backend=None,
            stt_model=None,
            tts_backend=None,
            tts_voice=None,
        )
    )

    assert config.language == local_prefetch.Language.ZH
    assert config.vision_model == "blink/moondream"

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from blink.cli import local_tts_eval
from blink.frames.frames import TTSAudioRawFrame
from blink.transcriptions.language import Language


def test_build_eval_matrix_expands_language_specific_voice_lists(tmp_path):
    config = local_tts_eval.LocalTTSEvalConfig(
        backends=["kokoro"],
        languages=[Language.ZH, Language.EN],
        voices_zh=["zf_xiaobei", "zf_xiaoyi"],
        voices_en=["af_heart"],
        output_dir=tmp_path,
        sample_rate=24_000,
    )

    matrix = local_tts_eval.build_eval_matrix(config)

    assert matrix == [
        local_tts_eval.EvalCombination("kokoro", Language.ZH, "zf_xiaobei"),
        local_tts_eval.EvalCombination("kokoro", Language.ZH, "zf_xiaoyi"),
        local_tts_eval.EvalCombination("kokoro", Language.EN, "af_heart"),
    ]


def test_resolve_config_defaults_to_bilingual_matrix(monkeypatch):
    monkeypatch.delenv("BLINK_LOCAL_TTS_BACKEND", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_ZH", raising=False)
    monkeypatch.delenv("BLINK_LOCAL_TTS_VOICE_EN", raising=False)

    config = local_tts_eval.resolve_config(local_tts_eval.build_parser().parse_args([]))

    assert config.languages == [Language.ZH, Language.EN]
    assert config.backends == ["kokoro"]
    assert config.output_dir == Path("artifacts/tts-eval")
    assert any("版本" in text for _, text in local_tts_eval.ZH_PHRASES)
    assert any("延迟" in text for _, text in local_tts_eval.ZH_PHRASES)
    assert any("API" in text for _, text in local_tts_eval.EN_PHRASES)


@pytest.mark.asyncio
async def test_run_evaluation_writes_wav_outputs(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self, voice):
            self._settings = SimpleNamespace(voice=voice, language=Language.ZH)
            self._text_filters = []
            self.started = False
            self.stopped = False

        async def start(self, _frame):
            self.started = True

        async def stop(self, _frame):
            self.stopped = True

        async def run_tts(self, _text, context_id):
            yield TTSAudioRawFrame(
                audio=b"\x00\x00" * 128,
                sample_rate=24_000,
                num_channels=1,
                context_id=context_id,
            )

    monkeypatch.setattr(
        local_tts_eval,
        "create_local_tts_service",
        lambda **kwargs: FakeService(kwargs["voice"] or "resolved-voice"),
    )

    config = local_tts_eval.LocalTTSEvalConfig(
        backends=["kokoro"],
        languages=[Language.ZH],
        voices_zh=["zf_xiaobei"],
        voices_en=[],
        output_dir=tmp_path,
        sample_rate=24_000,
    )

    status = await local_tts_eval.run_evaluation(config)

    assert status == 0
    assert (tmp_path / "kokoro" / "zh" / "zf_xiaobei" / "zh_01_intro.wav").exists()
    assert (tmp_path / "kokoro" / "zh" / "zf_xiaobei" / "mixed_01_diagnostic.wav").exists()
    assert (tmp_path / "README.md").exists()
    assert "technical conversation" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "MeloTTS" in (tmp_path / "README.md").read_text(encoding="utf-8")

import io
import wave

import pytest

from blink.frames.frames import TTSAudioRawFrame
from blink.services.local_http_wav.tts import (
    DEFAULT_LOCAL_HTTP_WAV_CONTEXT_TIMEOUT_SECS,
    LocalHttpWavTTSService,
)
from blink.services.settings import TTSSettings
from blink.transcriptions.language import Language


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00\x01\x00" * 128)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, *, status: int, text: str = "", body: bytes = b"", json_payload=None):
        self.status = status
        self._text = text
        self._body = body
        self._json_payload = json_payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text

    async def read(self):
        return self._body

    async def json(self, content_type=None):
        return self._json_payload


class FakeSession:
    def __init__(self):
        self.post_payloads = []

    def post(self, _url, *, json):
        self.post_payloads.append(json)
        if len(self.post_payloads) == 1:
            return FakeResponse(
                status=400,
                text="speaker 'zf_xiaobei' is not available for language 'zh'",
            )
        return FakeResponse(status=200, body=_wav_bytes())

    def get(self, _url):
        return FakeResponse(
            status=200,
            json_payload={
                "default_speakers": {"zh": "ZH", "en": "EN-US"},
                "speakers": {"zh": ["ZH"], "en": ["EN-US"]},
            },
        )


class SuccessSession:
    def __init__(self):
        self.post_payloads = []

    def post(self, _url, *, json):
        self.post_payloads.append(json)
        return FakeResponse(status=200, body=_wav_bytes())


def test_local_http_wav_tts_service_uses_long_context_timeout_by_default():
    service = LocalHttpWavTTSService(
        aiohttp_session=SuccessSession(),
        base_url="http://127.0.0.1:8012",
        settings=TTSSettings(voice="ZH", language=Language.ZH, model="melo-local"),
    )

    assert service._stop_frame_timeout_s == DEFAULT_LOCAL_HTTP_WAV_CONTEXT_TIMEOUT_SECS


def test_local_http_wav_tts_service_allows_context_timeout_override():
    service = LocalHttpWavTTSService(
        aiohttp_session=SuccessSession(),
        base_url="http://127.0.0.1:8012",
        settings=TTSSettings(voice="ZH", language=Language.ZH, model="melo-local"),
        stop_frame_timeout_s=12.0,
    )

    assert service._stop_frame_timeout_s == 12.0


@pytest.mark.asyncio
async def test_local_http_wav_tts_service_retries_with_catalog_fallback():
    session = FakeSession()
    service = LocalHttpWavTTSService(
        aiohttp_session=session,
        base_url="http://127.0.0.1:8012",
        settings=TTSSettings(voice="zf_xiaobei", language=Language.ZH, model=None),
    )

    frames = [frame async for frame in service.run_tts("你好，Blink。", context_id="ctx-1")]

    assert len(session.post_payloads) == 2
    assert session.post_payloads[0]["speaker"] == "zf_xiaobei"
    assert session.post_payloads[1]["speaker"] == "ZH"
    assert service._settings.voice == "ZH"
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)


@pytest.mark.asyncio
async def test_local_http_wav_tts_service_reports_framework_metrics():
    session = SuccessSession()
    service = LocalHttpWavTTSService(
        aiohttp_session=session,
        base_url="http://127.0.0.1:8012",
        settings=TTSSettings(voice="ZH", language=Language.ZH, model="melo-local"),
    )
    calls = []

    async def stop_ttfb_metrics():
        calls.append(("ttfb", None))

    async def start_tts_usage_metrics(text):
        calls.append(("usage", text))

    service.stop_ttfb_metrics = stop_ttfb_metrics
    service.start_tts_usage_metrics = start_tts_usage_metrics

    frames = [frame async for frame in service.run_tts("你好，Blink。", context_id="ctx-1")]

    assert service.can_generate_metrics() is True
    assert calls == [("ttfb", None), ("usage", "你好，Blink。")]
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)

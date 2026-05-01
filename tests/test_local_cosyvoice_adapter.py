import io
import wave

import pytest
from aiohttp import web

from blink.cli.local_cosyvoice_adapter import (
    CosyVoiceAdapterConfig,
    create_app,
    normalize_cosyvoice_text,
)
from blink.transcriptions.language import Language


def test_normalize_cosyvoice_text_strengthens_chinese_tn():
    text = normalize_cosyvoice_text(
        "你好 *Blink* https://example.com/test?a=1 v2.3 / A-1024",
        Language.ZH,
    )

    assert "https://" not in text
    assert "*" not in text
    assert "/" not in text
    assert text.endswith("。")
    assert "，" in text


@pytest.mark.asyncio
async def test_adapter_healthz_reports_backend_status(aiohttp_client):
    backend_app = web.Application()
    backend_app.router.add_get("/docs", lambda _request: web.Response(text="ok"))
    backend_client = await aiohttp_client(backend_app)

    config = CosyVoiceAdapterConfig(
        host="127.0.0.1",
        port=8001,
        backend_url=str(backend_client.make_url("")).rstrip("/"),
        mode="sft",
        zh_speaker="中文女",
        en_speaker=None,
        zh_instruct_text=None,
        en_instruct_text=None,
        backend_sample_rate=22050,
        output_sample_rate=24000,
        timeout_secs=5.0,
    )
    adapter_client = await aiohttp_client(create_app(config))

    response = await adapter_client.get("/healthz")
    payload = await response.json()

    assert response.status == 200
    assert payload["backend_reachable"] is True


@pytest.mark.asyncio
async def test_adapter_tts_wraps_pcm_in_wav_and_uses_default_chinese_speaker(aiohttp_client):
    captured = {}

    async def docs(_request):
        return web.Response(text="ok")

    async def inference_sft(request):
        post = await request.post()
        captured["tts_text"] = post["tts_text"]
        captured["spk_id"] = post["spk_id"]
        return web.Response(body=b"\x00\x00\x01\x00" * 256)

    backend_app = web.Application()
    backend_app.router.add_get("/docs", docs)
    backend_app.router.add_post("/inference_sft", inference_sft)
    backend_client = await aiohttp_client(backend_app)

    config = CosyVoiceAdapterConfig(
        host="127.0.0.1",
        port=8001,
        backend_url=str(backend_client.make_url("")).rstrip("/"),
        mode="sft",
        zh_speaker="中文女",
        en_speaker="EnglishFemale",
        zh_instruct_text=None,
        en_instruct_text=None,
        backend_sample_rate=22050,
        output_sample_rate=24000,
        timeout_secs=5.0,
    )
    adapter_client = await aiohttp_client(create_app(config))

    response = await adapter_client.post(
        "/tts",
        json={
            "text": "你好 *Blink*",
            "speaker": None,
            "language": "zh",
            "model": None,
        },
    )
    body = await response.read()

    assert response.status == 200
    assert response.headers["X-Blink-TTS-Speaker"] == "中文女"
    assert body.startswith(b"RIFF")
    assert captured["spk_id"] == "中文女"
    assert "*" not in captured["tts_text"]

    with wave.open(io.BytesIO(body), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1


@pytest.mark.asyncio
async def test_adapter_rejects_english_request_without_configured_speaker(aiohttp_client):
    backend_app = web.Application()
    backend_app.router.add_get("/docs", lambda _request: web.Response(text="ok"))
    backend_client = await aiohttp_client(backend_app)

    config = CosyVoiceAdapterConfig(
        host="127.0.0.1",
        port=8001,
        backend_url=str(backend_client.make_url("")).rstrip("/"),
        mode="sft",
        zh_speaker="中文女",
        en_speaker=None,
        zh_instruct_text=None,
        en_instruct_text=None,
        backend_sample_rate=22050,
        output_sample_rate=24000,
        timeout_secs=5.0,
    )
    adapter_client = await aiohttp_client(create_app(config))

    response = await adapter_client.post(
        "/tts",
        json={
            "text": "Hello from Blink",
            "language": "en",
        },
    )
    body = await response.text()

    assert response.status == 400
    assert "No speaker configured" in body


@pytest.mark.asyncio
async def test_adapter_instruct_mode_uses_language_specific_prompt(aiohttp_client):
    captured = {}

    async def docs(_request):
        return web.Response(text="ok")

    async def inference_instruct(request):
        post = await request.post()
        captured["instruct_text"] = post["instruct_text"]
        return web.Response(body=b"\x00\x00" * 256)

    backend_app = web.Application()
    backend_app.router.add_get("/docs", docs)
    backend_app.router.add_post("/inference_instruct", inference_instruct)
    backend_client = await aiohttp_client(backend_app)

    config = CosyVoiceAdapterConfig(
        host="127.0.0.1",
        port=8001,
        backend_url=str(backend_client.make_url("")).rstrip("/"),
        mode="instruct",
        zh_speaker="中文女",
        en_speaker="EnglishFemale",
        zh_instruct_text="请用标准普通话自然表达。",
        en_instruct_text="Speak in clear neutral English.",
        backend_sample_rate=22050,
        output_sample_rate=24000,
        timeout_secs=5.0,
    )
    adapter_client = await aiohttp_client(create_app(config))

    response = await adapter_client.post(
        "/tts",
        json={
            "text": "你好",
            "language": "zh",
        },
    )

    assert response.status == 200
    assert captured["instruct_text"] == "请用标准普通话自然表达。"

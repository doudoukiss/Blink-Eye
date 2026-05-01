import io
import wave

import numpy as np
import pytest

from local_tts_servers import melotts_http_server


class FakeMeloModel:
    def __init__(self, *, language, device):
        self.language = language
        self.device = device
        speaker_map = {"ZH": 0} if language == "ZH" else {"EN-US": 1}
        self.hps = type("FakeHPS", (), {"data": type("FakeData", (), {"spk2id": speaker_map})()})()

    def tts_to_file(self, text, speaker_id, output_path, speed=1.0):
        assert text
        assert speed > 0
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(b"\x00\x00\x01\x00" * 256)


def fake_model_factory(*, language, device):
    return FakeMeloModel(language=language, device=device)


class FakeSpeakerObject:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeSpeakerObjectModel(FakeMeloModel):
    def __init__(self, *, language, device):
        super().__init__(language=language, device=device)
        self.hps.data.spk2id = (
            FakeSpeakerObject(ZH=0) if language == "ZH" else FakeSpeakerObject(**{"EN-US": 1})
        )


def fake_object_model_factory(*, language, device):
    return FakeSpeakerObjectModel(language=language, device=device)


@pytest.mark.asyncio
async def test_healthz_reports_server_defaults(aiohttp_client):
    config = melotts_http_server.MeloTTSServerConfig(
        host="127.0.0.1",
        port=8001,
        device="cpu",
        output_sample_rate=24_000,
        speed=1.0,
        zh_speaker="ZH",
        en_speaker="EN-US",
    )
    client = await aiohttp_client(
        melotts_http_server.create_app(config, model_factory=fake_model_factory)
    )

    response = await client.get("/healthz")
    payload = await response.json()

    assert response.status == 200
    assert payload["status"] == "ok"
    assert payload["default_speakers"]["zh"] == "ZH"
    assert payload["recommended_python"] == "3.11"


@pytest.mark.asyncio
async def test_voices_lists_language_specific_speakers(aiohttp_client):
    config = melotts_http_server.MeloTTSServerConfig(
        host="127.0.0.1",
        port=8001,
        device="cpu",
        output_sample_rate=24_000,
        speed=1.0,
        zh_speaker="ZH",
        en_speaker="EN-US",
    )
    client = await aiohttp_client(
        melotts_http_server.create_app(config, model_factory=fake_model_factory)
    )

    response = await client.get("/voices")
    payload = await response.json()

    assert response.status == 200
    assert payload["speakers"]["zh"] == ["ZH"]
    assert payload["speakers"]["en"] == ["EN-US"]


@pytest.mark.asyncio
async def test_tts_returns_wav_and_resamples_output(aiohttp_client):
    config = melotts_http_server.MeloTTSServerConfig(
        host="127.0.0.1",
        port=8001,
        device="cpu",
        output_sample_rate=24_000,
        speed=1.0,
        zh_speaker="ZH",
        en_speaker="EN-US",
    )
    client = await aiohttp_client(
        melotts_http_server.create_app(config, model_factory=fake_model_factory)
    )

    response = await client.post(
        "/tts",
        json={"text": "你好，Blink。", "language": "zh"},
    )
    body = await response.read()

    assert response.status == 200
    assert response.headers["X-Blink-TTS-Backend"] == "melotts-http-server"
    assert response.headers["X-Blink-TTS-Speaker"] == "ZH"
    assert response.headers["X-Blink-TTS-Chars"] == str(len("你好，Blink。"))
    assert int(response.headers["X-Blink-TTS-Request-Ms"]) >= 0
    assert body.startswith(b"RIFF")

    with wave.open(io.BytesIO(body), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1


def test_normalize_pcm_peak_targets_stable_volume_without_clipping():
    samples = np.array([0, 1000, -1000, 500], dtype=np.float32)

    normalized = melotts_http_server.normalize_pcm_peak(samples)

    peak = int(np.max(np.abs(normalized)))
    assert 26_000 <= peak <= 27_000
    assert np.max(np.abs(normalized)) < np.iinfo(np.int16).max


def test_wrap_pcm_as_wav_downmixes_stereo_before_resampling():
    left = np.array([1000, 2000, -3000, 4000], dtype=np.int16)
    right = np.array([3000, 4000, -1000, 2000], dtype=np.int16)
    stereo = np.column_stack([left, right]).astype(np.int16).tobytes()

    wav_bytes = melotts_http_server.wrap_pcm_as_wav(
        stereo,
        input_sample_rate=8_000,
        output_sample_rate=16_000,
        channels=2,
    )

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1
        assert pcm.size > left.size


@pytest.mark.asyncio
async def test_tts_rejects_unknown_speaker(aiohttp_client):
    config = melotts_http_server.MeloTTSServerConfig(
        host="127.0.0.1",
        port=8001,
        device="cpu",
        output_sample_rate=24_000,
        speed=1.0,
        zh_speaker="ZH",
        en_speaker="EN-US",
    )
    client = await aiohttp_client(
        melotts_http_server.create_app(config, model_factory=fake_model_factory)
    )

    response = await client.post(
        "/tts",
        json={"text": "hello", "language": "en", "speaker": "missing"},
    )

    assert response.status == 400
    assert "speaker" in await response.text()


@pytest.mark.asyncio
async def test_tts_accepts_object_style_speaker_map(aiohttp_client):
    config = melotts_http_server.MeloTTSServerConfig(
        host="127.0.0.1",
        port=8001,
        device="cpu",
        output_sample_rate=24_000,
        speed=1.0,
        zh_speaker="ZH",
        en_speaker="EN-US",
    )
    client = await aiohttp_client(
        melotts_http_server.create_app(config, model_factory=fake_object_model_factory)
    )

    response = await client.post(
        "/tts",
        json={"text": "你好", "language": "zh"},
    )

    assert response.status == 200

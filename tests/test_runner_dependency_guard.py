import pytest

from blink.runner import run as runner_run
from blink.runner import utils as runner_utils


def test_runner_dependency_guard_raises_clean_import_error(monkeypatch):
    monkeypatch.setattr(runner_run, "_RUNNER_IMPORT_ERROR", ModuleNotFoundError("fastapi"))

    with pytest.raises(ImportError, match=r"blink-ai\[runner\]"):
        runner_run._require_runner_dependencies()


class SmallWebRTCTransport:
    def __init__(self):
        self.capture_calls = []

    async def capture_participant_video(self, *args, **kwargs):
        self.capture_calls.append((args, kwargs))


SmallWebRTCTransport.__module__ = "blink.transports.smallwebrtc.transport"


@pytest.mark.asyncio
async def test_optional_transport_helpers_support_blink_smallwebrtc(monkeypatch):
    transport = SmallWebRTCTransport()
    client = type("Client", (), {"pc_id": "pc-123"})()

    client_id = runner_utils.get_transport_client_id(transport, client)
    await runner_utils.maybe_capture_participant_camera(transport, client, framerate=1)

    assert client_id == "pc-123"
    assert transport.capture_calls == [((), {"video_source": "camera", "framerate": 1})]


@pytest.mark.asyncio
async def test_optional_transport_helpers_ignore_removed_legacy_smallwebrtc_alias():
    transport = SmallWebRTCTransport()
    transport.__class__.__module__ = "legacy.transports.smallwebrtc.transport"
    client = type("Client", (), {"pc_id": "pc-456"})()

    client_id = runner_utils.get_transport_client_id(transport, client)
    await runner_utils.maybe_capture_participant_camera(transport, client, framerate=1)

    assert client_id == ""
    assert transport.capture_calls == []


def test_runner_ready_path_uses_client_for_normal_webrtc_only():
    browser_args = type("Args", (), {"transport": "webrtc", "esp32": False, "whatsapp": False})()
    esp32_args = type("Args", (), {"transport": "webrtc", "esp32": True, "whatsapp": False})()
    daily_args = type("Args", (), {"transport": "daily", "esp32": False, "whatsapp": False})()

    assert runner_run._runner_ready_path(browser_args) == "/client/"
    assert runner_run._runner_ready_path(esp32_args) is None
    assert runner_run._runner_ready_path(daily_args) is None

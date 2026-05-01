import asyncio

import pytest

from blink.web.server_startup import ServerStartupError, start_uvicorn_server


class FakeServer:
    def __init__(self, *, start_delay: float = 0.0, crash: BaseException | None = None):
        self.started = False
        self.should_exit = False
        self._start_delay = start_delay
        self._crash = crash

    async def serve(self) -> None:
        if self._crash is not None:
            raise self._crash

        if self._start_delay:
            await asyncio.sleep(self._start_delay)
        self.started = True

        while not self.should_exit:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_start_uvicorn_server_returns_running_task_once_ready(monkeypatch):
    server = FakeServer()

    async def fake_probe_http_status(*_args, **_kwargs):
        return 200

    monkeypatch.setattr("blink.web.server_startup._probe_http_status", fake_probe_http_status)

    serve_task = await start_uvicorn_server(
        server,
        host="127.0.0.1",
        port=7860,
        ready_path="/client/",
        timeout_secs=1.0,
    )

    assert server.started is True
    assert serve_task.done() is False

    server.should_exit = True
    await serve_task


@pytest.mark.asyncio
async def test_start_uvicorn_server_times_out_when_ready_path_never_responds(monkeypatch):
    server = FakeServer()

    async def fake_probe_http_status(*_args, **_kwargs):
        return None

    monkeypatch.setattr("blink.web.server_startup._probe_http_status", fake_probe_http_status)

    with pytest.raises(ServerStartupError, match=r"/client/ did not become ready"):
        await start_uvicorn_server(
            server,
            host="127.0.0.1",
            port=7860,
            ready_path="/client/",
            timeout_secs=0.3,
        )

    assert server.should_exit is True


@pytest.mark.asyncio
async def test_start_uvicorn_server_reports_early_exit_during_bind():
    server = FakeServer(crash=RuntimeError("boom"))

    with pytest.raises(ServerStartupError, match=r"exited during bind: boom"):
        await start_uvicorn_server(
            server,
            host="127.0.0.1",
            port=7860,
            ready_path="/client/",
            timeout_secs=0.3,
        )


@pytest.mark.asyncio
async def test_start_uvicorn_server_converts_uvicorn_system_exit():
    server = FakeServer(crash=SystemExit(1))

    with pytest.raises(ServerStartupError, match=r"exited during bind: HTTP server exited: 1"):
        await start_uvicorn_server(
            server,
            host="127.0.0.1",
            port=7860,
            ready_path="/client/",
            timeout_secs=0.3,
        )

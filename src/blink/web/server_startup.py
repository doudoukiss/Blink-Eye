#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Uvicorn startup helpers for Blink local web workflows."""

from __future__ import annotations

import asyncio
from typing import Protocol

STARTUP_TIMEOUT_SECS = 45.0
_PROBE_INTERVAL_SECS = 0.1
_PROBE_TIMEOUT_SECS = 1.0


class ServerStartupError(RuntimeError):
    """Raised when a Uvicorn server fails to become ready."""


class SupportsUvicornServer(Protocol):
    """Minimal protocol required from a Uvicorn server instance."""

    started: bool
    should_exit: bool

    async def serve(self) -> None:
        """Start serving requests until the server is stopped."""


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", ""}:
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


async def _probe_http_status(host: str, port: int, path: str) -> int | None:
    connect_host = _connect_host(host)
    normalized_path = path if path.startswith("/") else f"/{path}"

    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(connect_host, port),
            timeout=_PROBE_TIMEOUT_SECS,
        )
        request = (
            f"GET {normalized_path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()
        status_line = await asyncio.wait_for(reader.readline(), timeout=_PROBE_TIMEOUT_SECS)
    except (OSError, TimeoutError, UnicodeDecodeError, ValueError):
        return None
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    try:
        return int(status_line.decode("iso-8859-1").split()[1])
    except (IndexError, ValueError):
        return None


async def _wait_for_server_exit(task: asyncio.Task[None], phase: str) -> None:
    try:
        await task
    except asyncio.CancelledError:
        raise
    except SystemExit as exc:
        raise ServerStartupError(f"HTTP server exited during {phase}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - exercised through callers
        raise ServerStartupError(f"HTTP server exited during {phase}: {exc}") from exc
    raise ServerStartupError(f"HTTP server exited during {phase} before becoming ready.")


async def _serve_with_startup_error(server: SupportsUvicornServer) -> None:
    """Run a server and convert uvicorn sys.exit into a regular startup error."""
    try:
        await server.serve()
    except SystemExit as exc:
        raise ServerStartupError(f"HTTP server exited: {exc}") from exc


async def _shutdown_server(server: SupportsUvicornServer, task: asyncio.Task[None]) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def start_uvicorn_server(
    server: SupportsUvicornServer,
    *,
    host: str,
    port: int,
    ready_path: str | None = None,
    timeout_secs: float = STARTUP_TIMEOUT_SECS,
) -> asyncio.Task[None]:
    """Start a Uvicorn server and wait for real readiness.

    Args:
        server: Configured Uvicorn server instance.
        host: Host the server binds to.
        port: Port the server binds to.
        ready_path: Optional HTTP path that must return success before the
            server is considered ready. When omitted, bind readiness is enough.
        timeout_secs: Maximum time to wait for readiness.

    Returns:
        The background task running ``server.serve()``.

    Raises:
        ServerStartupError: If the server exits early or does not become ready
            within ``timeout_secs``.
    """
    serve_task = asyncio.create_task(_serve_with_startup_error(server), name=f"uvicorn:{host}:{port}")
    deadline = asyncio.get_running_loop().time() + timeout_secs
    bound = False

    while asyncio.get_running_loop().time() < deadline:
        if serve_task.done():
            await _wait_for_server_exit(serve_task, "bind")

        if getattr(server, "started", False):
            bound = True
            break

        await asyncio.sleep(_PROBE_INTERVAL_SECS)

    if not bound:
        await _shutdown_server(server, serve_task)
        raise ServerStartupError(
            f"HTTP server did not bind to http://{host}:{port} within {timeout_secs:.0f} seconds."
        )

    if not ready_path:
        return serve_task

    while asyncio.get_running_loop().time() < deadline:
        if serve_task.done():
            await _wait_for_server_exit(serve_task, f"readiness probe for {ready_path}")

        status = await _probe_http_status(host, port, ready_path)
        if status is not None and 200 <= status < 400:
            return serve_task

        await asyncio.sleep(_PROBE_INTERVAL_SECS)

    await _shutdown_server(server, serve_task)
    raise ServerStartupError(
        f"HTTP server bound to http://{host}:{port} but {ready_path} did not become ready "
        f"within {timeout_secs:.0f} seconds."
    )

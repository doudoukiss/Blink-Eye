#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for the repo-owned Blink browser UI bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from blink.project_identity import PROJECT_IDENTITY, pip_install_command

SMALLWEBRTC_UI_MOUNT_PATH = "/client"
SMALLWEBRTC_UI_ROOT_PATH = f"{SMALLWEBRTC_UI_MOUNT_PATH}/"
SMALLWEBRTC_UI_DIST_DIR = Path(__file__).resolve().parent / "client_dist"
SMALLWEBRTC_UI_SOURCE_DIR = Path(__file__).resolve().parents[3] / "web" / "client_src" / "src"
SMALLWEBRTC_UI_OWNERSHIP_NOTE = (
    "The /client UI is a repo-owned Blink browser bundle mounted from the Python package."
)
SMALLWEBRTC_UI_SOURCE_NOTE = (
    "The repo-owned Blink browser client workspace lives in web/client_src/src. "
    "It includes authored Blink overlays and required vendored runtime assets. "
    "Package builds can copy it into src/blink/web/client_dist when a generated copy is needed."
)

_SMALLWEBRTC_UI_IMPORT_ERROR: Optional[ImportError] = None

try:
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:
    _SMALLWEBRTC_UI_IMPORT_ERROR = exc
    RedirectResponse = None  # type: ignore[assignment]
    StaticFiles = None  # type: ignore[assignment]


def create_smallwebrtc_static_files():
    """Return a static-files app that disables caching for the local UI bundle."""
    ui_dir = require_smallwebrtc_ui()

    class BlinkStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            if hasattr(response, "headers"):
                response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    return BlinkStaticFiles(directory=ui_dir, html=True)


def smallwebrtc_ui_dependency_message() -> str:
    """Return the normalized install hint for the Blink browser UI."""
    return (
        f"{PROJECT_IDENTITY.display_name} browser UI dependencies required. "
        f"Install with: {pip_install_command('runner')}"
    )


def smallwebrtc_ui_root_dir() -> Path | None:
    """Return the generated UI bundle path, or the client workspace in a checkout."""
    if SMALLWEBRTC_UI_DIST_DIR.joinpath("index.html").is_file():
        return SMALLWEBRTC_UI_DIST_DIR
    if SMALLWEBRTC_UI_SOURCE_DIR.joinpath("index.html").is_file():
        return SMALLWEBRTC_UI_SOURCE_DIR
    return None


def require_smallwebrtc_ui() -> Path:
    """Raise a clean ImportError when the Blink UI runtime or bundle is missing."""
    if _SMALLWEBRTC_UI_IMPORT_ERROR is not None:
        message = smallwebrtc_ui_dependency_message()
        logger.error(f"Blink browser UI dependencies not available: {_SMALLWEBRTC_UI_IMPORT_ERROR}")
        logger.error(message)
        raise ImportError(message) from _SMALLWEBRTC_UI_IMPORT_ERROR

    ui_dir = smallwebrtc_ui_root_dir()
    if ui_dir is None:
        message = (
            f"{PROJECT_IDENTITY.display_name} browser UI bundle is missing. "
            f"Expected generated assets at {SMALLWEBRTC_UI_DIST_DIR} or source assets at "
            f"{SMALLWEBRTC_UI_SOURCE_DIR}."
        )
        logger.error(message)
        raise ImportError(message)
    return ui_dir


def mount_smallwebrtc_ui(app: Any) -> None:
    """Mount the Blink browser UI at the stable ``/client`` path."""
    app.mount(SMALLWEBRTC_UI_MOUNT_PATH, create_smallwebrtc_static_files())


def create_smallwebrtc_root_redirect():
    """Create a root redirect handler that points at the mounted ``/client/`` UI."""
    require_smallwebrtc_ui()

    async def root_redirect():
        return RedirectResponse(url=SMALLWEBRTC_UI_ROOT_PATH)

    return root_redirect

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys

from loguru import logger

from blink._version import __version__
from blink.project_identity import PROJECT_IDENTITY, import_banner_env_name


def _import_banner_enabled() -> bool:
    env_name = import_banner_env_name()
    if env_name in os.environ:
        return os.environ[env_name].lower() not in {"0", "false", "no"}
    return True


if _import_banner_enabled():
    logger.info(f"ᓚᘏᗢ {PROJECT_IDENTITY.display_name} {__version__} (Python {sys.version}) ᓚᘏᗢ")


def version() -> str:
    """Return the checked-in runtime version string."""
    return __version__


# We replace `asyncio.wait_for()` for `wait_for2.wait_for()` for Python < 3.12.
#
# In Python 3.12, `asyncio.wait_for()` is implemented in terms of
# `asyncio.timeout()` which fixed a bunch of issues. However, this was never
# backported (because of the lack of `async.timeout()`) and there are still many
# remainig issues, specially in Python 3.10, in `async.wait_for()`.
#
# See https://github.com/python/cpython/pull/98518

import asyncio

if sys.version_info < (3, 12):
    import wait_for2

    # Replace asyncio.wait_for.
    asyncio.wait_for = wait_for2.wait_for

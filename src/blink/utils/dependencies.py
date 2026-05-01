#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for optional dependency guardrails."""

from loguru import logger

from blink.project_identity import pip_install_command


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is not installed."""


def missing_dependency_error(
    *,
    feature: str,
    extra: str,
    exc: ModuleNotFoundError,
    details: str | None = None,
) -> MissingDependencyError:
    """Build a clean ImportError for an optional dependency."""
    install_command = pip_install_command(extra)
    log_message = (
        details.rstrip(".")
        if details
        else f"In order to use {feature}, you need to `{install_command}`"
    )

    logger.error(f"Exception: {exc}")
    logger.error(f"{log_message}.")

    return MissingDependencyError(f"Missing module: {exc}. Install with: {install_command}")

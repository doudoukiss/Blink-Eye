#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Project identity metadata used by runtime-owned strings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectIdentity:
    """Canonical project identity metadata."""

    display_name: str = "Blink"
    distribution_name: str = "blink-ai"
    import_namespace: str = "blink"
    cli_prefix: str = "blink-local"
    env_prefix: str = "BLINK"
    homepage_url: str = "https://github.com/blink-ai/Blink"
    documentation_url: str = "https://github.com/blink-ai/Blink/tree/main/docs"
    source_url: str = "https://github.com/blink-ai/Blink"
    issues_url: str = "https://github.com/blink-ai/Blink/issues"
    changelog_url: str = "https://github.com/blink-ai/Blink/blob/main/CHANGELOG.md"


PROJECT_IDENTITY = ProjectIdentity()


def cli_command(command: str) -> str:
    """Return a canonical CLI command name."""
    return f"{PROJECT_IDENTITY.cli_prefix}-{command}"


def env_name(name: str) -> str:
    """Return a canonical environment variable name."""
    return f"{PROJECT_IDENTITY.env_prefix}_{name}"


def local_env_name(name: str) -> str:
    """Return a canonical local workflow environment variable name."""
    return env_name(f"LOCAL_{name}")


def import_banner_env_name() -> str:
    """Return the import-banner environment variable name."""
    return env_name("IMPORT_BANNER")


def install_requirement(extra: str | None = None) -> str:
    """Return the canonical install requirement string for the project or one extra."""
    if not extra:
        return PROJECT_IDENTITY.distribution_name
    return f"{PROJECT_IDENTITY.distribution_name}[{extra}]"


def pip_install_command(extra: str | None = None) -> str:
    """Return a pip install command for the canonical project package or one extra."""
    return f"pip install {install_requirement(extra)}"


def cache_dir(*parts: str) -> Path:
    """Return the Blink cache path."""
    canonical = Path.home() / ".cache" / PROJECT_IDENTITY.import_namespace
    return canonical.joinpath(*parts) if parts else canonical

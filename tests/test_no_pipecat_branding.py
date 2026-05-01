from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF_PATH = Path(__file__).resolve()
LEGACY_BRAND = "pipe" + "cat"
FORBIDDEN_PATTERN = re.compile(
    "|".join((LEGACY_BRAND.capitalize(), LEGACY_BRAND, LEGACY_BRAND.upper()))
)
EXCLUDED_PREFIXES = (
    "references/",
    "docs/MeloTTS-reference/",
)


def _repo_files() -> list[Path]:
    ignored_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if any(part in ignored_dirs for part in relative_parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_tracked_repo_files_do_not_contain_removed_branding():
    offenders: list[str] = []

    for absolute_path in _repo_files():
        relative_path = absolute_path.relative_to(ROOT).as_posix()
        if relative_path.startswith(EXCLUDED_PREFIXES):
            continue

        if absolute_path == SELF_PATH or not absolute_path.is_file():
            continue

        try:
            content = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if FORBIDDEN_PATTERN.search(content):
            offenders.append(relative_path)

    assert offenders == [], f"Removed branding found in tracked files: {offenders}"

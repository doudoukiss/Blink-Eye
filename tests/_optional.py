"""Helpers for optional test dependencies."""

from __future__ import annotations

import pytest

OPTIONAL_PROVIDER = pytest.mark.optional_provider
OPTIONAL_RUNTIME = pytest.mark.optional_runtime
BROWSER_E2E = pytest.mark.browser_e2e
PROVIDER_INTEGRATION = pytest.mark.provider_integration


def require_optional_modules(*module_names: str):
    """Skip the current test module when any optional dependency is missing."""
    for module_name in module_names:
        pytest.importorskip(module_name)

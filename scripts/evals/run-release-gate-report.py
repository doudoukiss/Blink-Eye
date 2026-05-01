#!/usr/bin/env python3
"""Run Blink's deterministic release-gate report."""

from __future__ import annotations

import os

os.environ.setdefault("BLINK_IMPORT_BANNER", "0")

from blink.cli.release_gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

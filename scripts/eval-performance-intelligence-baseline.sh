#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

exec env BLINK_IMPORT_BANNER=0 uv run --python 3.12 python \
  "$ROOT_DIR/scripts/evals/run-performance-intelligence-baseline.py" "$@"

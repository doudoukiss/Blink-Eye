#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" --profile text
fi

exec env BLINK_IMPORT_BANNER=0 uv run --python 3.12 blink-local-cosyvoice-adapter "$@"

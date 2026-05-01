#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="$ROOT_DIR/docs/MeloTTS-reference"
VENV_DIR="$REFERENCE_DIR/.venv"

cd "$ROOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$ROOT_DIR/scripts/bootstrap-melotts-reference.sh"
fi

exec "$VENV_DIR/bin/python" -m local_tts_servers.melotts_http_server "$@"

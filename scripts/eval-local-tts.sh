#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

want_piper=0
LOCAL_TTS_BACKEND="${BLINK_LOCAL_TTS_BACKEND:-}"
if [[ "$LOCAL_TTS_BACKEND" == "piper" ]]; then
  want_piper=1
fi

expecting_backend=0
for arg in "$@"; do
  if [[ "$expecting_backend" -eq 1 ]]; then
    if [[ "$arg" == "piper" ]]; then
      want_piper=1
    fi
    expecting_backend=0
    continue
  fi

  case "$arg" in
    --backend)
      expecting_backend=1
      ;;
    --backend=piper)
      want_piper=1
      ;;
  esac
done

BOOTSTRAP_ARGS=(--profile voice)
if [[ "$want_piper" -eq 1 ]]; then
  BOOTSTRAP_ARGS+=(--with-piper)
fi

if [[ ! -d .venv ]]; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif [[ "$want_piper" -eq 1 ]] && ! .venv/bin/python -c "import piper" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
fi

exec env BLINK_IMPORT_BANNER=0 uv run --python 3.12 blink-local-tts-eval "$@"

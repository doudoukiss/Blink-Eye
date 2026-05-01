#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_LANGUAGE="${BLINK_LOCAL_LANGUAGE:-zh}"
DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"
EXPLICIT_TTS_BACKEND="${BLINK_LOCAL_TTS_BACKEND:-}"

expecting_backend=0
for arg in "$@"; do
  if [[ "$expecting_backend" -eq 1 ]]; then
    EXPLICIT_TTS_BACKEND="$arg"
    expecting_backend=0
    continue
  fi

  case "$arg" in
    --tts-backend)
      expecting_backend=1
      ;;
    --tts-backend=*)
      EXPLICIT_TTS_BACKEND="${arg#*=}"
      ;;
  esac
done

if [[ -z "$EXPLICIT_TTS_BACKEND" ]]; then
  exec env \
    BLINK_LOCAL_LANGUAGE="$DEFAULT_LANGUAGE" \
    BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION" \
    "$ROOT_DIR/scripts/run-local-browser-melo.sh" \
    "$@"
fi

exec env \
  BLINK_LOCAL_LANGUAGE="$DEFAULT_LANGUAGE" \
  BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION" \
  "$ROOT_DIR/scripts/run-local-browser.sh" \
  "$@"

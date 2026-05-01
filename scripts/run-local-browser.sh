#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

load_env_defaults() {
  if [[ ! -f .env ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != *=* ]] && continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ -z "$key" ]] && continue

    if [[ -n "${!key+x}" ]]; then
      continue
    fi

    value="${value%$'\r'}"
    if [[ "$value" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < .env
}

load_env_defaults

want_piper=0
want_vision=0
LOCAL_TTS_BACKEND="${BLINK_LOCAL_TTS_BACKEND:-}"
LOCAL_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-}"
if [[ "$LOCAL_TTS_BACKEND" == "piper" ]]; then
  want_piper=1
fi
if [[ "$LOCAL_BROWSER_VISION" =~ ^(1|true|yes|on)$ ]]; then
  want_vision=1
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
    --tts-backend)
      expecting_backend=1
      ;;
    --tts-backend=piper)
      want_piper=1
      ;;
    --vision)
      want_vision=1
      ;;
    --no-vision)
      want_vision=0
      ;;
  esac
done

BOOTSTRAP_ARGS=(--profile browser)
if [[ "$want_piper" -eq 1 ]]; then
  BOOTSTRAP_ARGS+=(--with-piper)
fi
if [[ "$want_vision" -eq 1 ]]; then
  BOOTSTRAP_ARGS+=(--with-vision)
fi

if [[ ! -d .venv ]]; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif ! .venv/bin/python -c "import fastapi, aiortc" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif [[ ! -f web/client_src/src/index.html ]]; then
  echo "error: Blink browser UI source is missing from web/client_src/src" >&2
  exit 1
elif [[ "$want_piper" -eq 1 ]] && ! .venv/bin/python -c "import piper" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif [[ "$want_vision" -eq 1 ]] && ! .venv/bin/python -c "import torch, transformers, pyvips" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
fi

if [[ "$want_vision" -eq 1 ]]; then
  echo "Starting Blink browser runtime with local vision. After a fresh bootstrap or venv rebuild, it can take 10-20 seconds before the launcher confirms /client/ is ready." >&2
else
  echo "Starting Blink browser runtime. The launcher will wait for real /client/ readiness before announcing the URL." >&2
fi

exec env \
  BLINK_IMPORT_BANNER=0 \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u -m blink.cli.local_browser \
  "$@"

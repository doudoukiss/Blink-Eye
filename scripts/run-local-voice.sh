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
want_moondream=0
LOCAL_TTS_BACKEND="${BLINK_LOCAL_TTS_BACKEND:-}"
if [[ "$LOCAL_TTS_BACKEND" == "piper" ]]; then
  want_piper=1
fi
if [[ "${BLINK_LOCAL_CAMERA_SOURCE:-}" == "macos-helper" ]]; then
  want_moondream=1
fi

expecting_backend=0
expecting_camera_source=0
for arg in "$@"; do
  if [[ "$expecting_backend" -eq 1 ]]; then
    if [[ "$arg" == "piper" ]]; then
      want_piper=1
    fi
    expecting_backend=0
    continue
  fi
  if [[ "$expecting_camera_source" -eq 1 ]]; then
    if [[ "$arg" == "macos-helper" ]]; then
      want_moondream=1
    fi
    expecting_camera_source=0
    continue
  fi

  case "$arg" in
    --tts-backend)
      expecting_backend=1
      ;;
    --tts-backend=piper)
      want_piper=1
      ;;
    --camera-source)
      expecting_camera_source=1
      ;;
    --camera-source=macos-helper)
      want_moondream=1
      ;;
  esac
done

BOOTSTRAP_ARGS=(--profile voice)
if [[ "$want_piper" -eq 1 ]]; then
  BOOTSTRAP_ARGS+=(--with-piper)
fi
if [[ "$want_moondream" -eq 1 ]]; then
  BOOTSTRAP_ARGS+=(--with-vision)
fi

if [[ ! -d .venv ]]; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif [[ "$want_piper" -eq 1 ]] && ! .venv/bin/python -c "import piper" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif [[ "$want_moondream" -eq 1 ]] && ! .venv/bin/python -c "import torch, transformers, pyvips" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
elif ! .venv/bin/python -c "import pyaudio" >/dev/null 2>&1; then
  "$ROOT_DIR/scripts/bootstrap-local-mac.sh" "${BOOTSTRAP_ARGS[@]}"
fi

exec env BLINK_IMPORT_BANNER=0 uv run --python 3.12 blink-local-voice "$@"

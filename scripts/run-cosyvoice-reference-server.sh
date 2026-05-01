#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_DIR="${BLINK_LOCAL_COSYVOICE_REFERENCE_DIR:-$ROOT_DIR/docs/CosyVoice-main}"
COSYVOICE_VENV="${BLINK_LOCAL_COSYVOICE_REFERENCE_VENV:-$COSYVOICE_DIR/.venv}"
COSYVOICE_PORT="${BLINK_LOCAL_COSYVOICE_PORT:-50000}"
COSYVOICE_MODEL_DIR="${BLINK_LOCAL_COSYVOICE_MODEL_DIR:-iic/CosyVoice-300M-SFT}"

if [[ ! -d "$COSYVOICE_DIR" ]]; then
  echo "CosyVoice reference checkout not found at: $COSYVOICE_DIR" >&2
  exit 1
fi

if [[ ! -x "$COSYVOICE_VENV/bin/python" ]]; then
  "$ROOT_DIR/scripts/bootstrap-cosyvoice-reference.sh"
fi

export PYTHONPATH="$COSYVOICE_DIR/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

cd "$COSYVOICE_DIR/runtime/python/fastapi"

exec "$COSYVOICE_VENV/bin/python" server.py \
  --port "$COSYVOICE_PORT" \
  --model_dir "$COSYVOICE_MODEL_DIR" \
  "$@"

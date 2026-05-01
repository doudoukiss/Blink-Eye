#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_DIR="${BLINK_LOCAL_COSYVOICE_REFERENCE_DIR:-$ROOT_DIR/docs/CosyVoice-main}"
COSYVOICE_VENV="${BLINK_LOCAL_COSYVOICE_REFERENCE_VENV:-$COSYVOICE_DIR/.venv}"
COSYVOICE_PYTHON_VERSION="${BLINK_LOCAL_COSYVOICE_PYTHON_VERSION:-3.10}"

if [[ ! -d "$COSYVOICE_DIR" ]]; then
  echo "CosyVoice reference checkout not found at: $COSYVOICE_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

cd "$ROOT_DIR"

uv python install "$COSYVOICE_PYTHON_VERSION"

if [[ ! -x "$COSYVOICE_VENV/bin/python" ]]; then
  uv venv --seed --python "$COSYVOICE_PYTHON_VERSION" "$COSYVOICE_VENV"
fi

"$COSYVOICE_VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$COSYVOICE_VENV/bin/python" -m pip install --upgrade pip setuptools wheel

FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(openai-whisper|fastapi-cli|gdown|gradio|grpcio|grpcio-tools|lightning|matplotlib|pyarrow|pyworld|tensorboard|wget)==' \
  "$COSYVOICE_DIR/requirements.txt" > "$FILTERED_REQUIREMENTS"

"$COSYVOICE_VENV/bin/python" -m pip install -r "$FILTERED_REQUIREMENTS"
"$COSYVOICE_VENV/bin/python" -m pip install --no-build-isolation openai-whisper==20231117

echo
echo "CosyVoice reference environment is ready."
echo "Reference checkout: $COSYVOICE_DIR"
echo "Virtualenv: $COSYVOICE_VENV"
echo
echo "Next steps:"
echo "  1. Start the reference FastAPI server:"
echo "     ./scripts/run-cosyvoice-reference-server.sh"
echo "  2. Start the Blink adapter:"
echo "     ./scripts/run-local-cosyvoice-adapter.sh"
echo "  3. Run Blink against the sidecar:"
echo "     BLINK_LOCAL_TTS_BACKEND=local-http-wav ./scripts/run-local-browser.sh"

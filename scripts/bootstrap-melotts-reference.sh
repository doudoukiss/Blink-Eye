#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="$ROOT_DIR/docs/MeloTTS-reference"
VENV_DIR="$REFERENCE_DIR/.venv"
SOURCE_DIR="$REFERENCE_DIR/vendor/MeloTTS"
RUNTIME_REQUIREMENTS="$REFERENCE_DIR/runtime-requirements.txt"
PYTHON_VERSION="${BLINK_LOCAL_MELO_PYTHON:-${BLINK_LOCAL_MELO_PYTHON:-3.11}}"
MELO_GIT_REF="${BLINK_LOCAL_MELO_GIT_REF:-${BLINK_LOCAL_MELO_GIT_REF:-v0.1.2}}"
MELO_PREFETCH="${BLINK_LOCAL_MELO_PREFETCH:-${BLINK_LOCAL_MELO_PREFETCH:-1}}"
MELO_PREFETCH_LANGUAGES="${BLINK_LOCAL_MELO_PREFETCH_LANGUAGES:-${BLINK_LOCAL_MELO_PREFETCH_LANGUAGES:-zh,en}}"

cd "$ROOT_DIR"

mkdir -p "$REFERENCE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
python3 -m local_tts_servers.melotts_reference prepare-source \
  --source-dir "$SOURCE_DIR" \
  --git-ref "$MELO_GIT_REF"
uv pip install --python "$VENV_DIR/bin/python" --upgrade "pip>=24,<27" "setuptools<81" wheel
uv pip install --python "$VENV_DIR/bin/python" -r "$RUNTIME_REQUIREMENTS"
uv pip install --python "$VENV_DIR/bin/python" -e "$SOURCE_DIR" --no-deps

if [[ "$MELO_PREFETCH" != "0" ]]; then
  PREFETCH_ARGS=()
  IFS=',' read -r -a PREFETCH_LANG_ARRAY <<< "$MELO_PREFETCH_LANGUAGES"
  for language in "${PREFETCH_LANG_ARRAY[@]}"; do
    PREFETCH_ARGS+=(--language "$language")
  done
  "$VENV_DIR/bin/python" -m local_tts_servers.melotts_reference prefetch "${PREFETCH_ARGS[@]}"
fi

echo "MeloTTS reference environment is ready at $VENV_DIR"
echo "Patched source checkout: $SOURCE_DIR"
echo "Next steps:"
echo "  ./scripts/run-melotts-reference-server.sh"
echo "  ./scripts/run-local-browser-melo.sh"

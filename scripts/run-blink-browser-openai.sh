#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/local_launcher_helpers.sh"

cd "$ROOT_DIR"
load_env_defaults "$ROOT_DIR/.env"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY is required for the Blink OpenAI browser demo lane." >&2
  echo "Set it in your shell or ignored .env file, then rerun this wrapper." >&2
  exit 1
fi

exec env \
  BLINK_LOCAL_LLM_PROVIDER=openai-responses \
  BLINK_LOCAL_OPENAI_RESPONSES_MODEL="${BLINK_LOCAL_OPENAI_RESPONSES_MODEL:-gpt-5.4-mini}" \
  BLINK_LOCAL_DEMO_MODE="${BLINK_LOCAL_DEMO_MODE:-1}" \
  "$ROOT_DIR/scripts/run-blink-browser.sh" \
  "$@"

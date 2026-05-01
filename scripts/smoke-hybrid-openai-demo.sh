#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/local_launcher_helpers.sh"
load_env_defaults "$ROOT_DIR/.env"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY is required for the hybrid OpenAI LLM-only smoke test." >&2
  echo "Set it in your shell or ignored .env file, then rerun this wrapper." >&2
  exit 1
fi

PROMPT="${*:-请用一句话说明 Blink 的混合 OpenAI 演示模式是什么。}"

echo "Running Blink hybrid OpenAI smoke test: LLM-only chat path." >&2
echo "This does not prove STT, TTS, WebRTC, MeloTTS latency, or camera behavior." >&2

exec "$ROOT_DIR/scripts/run-blink-chat-openai.sh" --once "$PROMPT"

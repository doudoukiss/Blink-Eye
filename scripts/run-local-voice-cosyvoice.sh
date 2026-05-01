#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/local_launcher_helpers.sh"

COSYVOICE_PORT="${BLINK_LOCAL_COSYVOICE_PORT:-50000}"
ADAPTER_PORT="${BLINK_LOCAL_COSYVOICE_ADAPTER_PORT:-8001}"
COSYVOICE_BACKEND_URL="${BLINK_LOCAL_COSYVOICE_BACKEND_URL:-http://127.0.0.1:${COSYVOICE_PORT}}"
ADAPTER_URL="http://127.0.0.1:${ADAPTER_PORT}"
CLEANUP_DONE=0

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if [[ -n "${VOICE_PID:-}" ]]; then
    terminate_pid_tree "${VOICE_PID}" "Blink voice runtime"
  fi
  if [[ -n "${ADAPTER_PID:-}" ]]; then
    terminate_pid_tree "${ADAPTER_PID}" "CosyVoice HTTP-WAV adapter"
  fi
  if [[ -n "${COSYVOICE_PID:-}" ]]; then
    terminate_pid_tree "${COSYVOICE_PID}" "CosyVoice backend"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$ROOT_DIR/scripts/run-cosyvoice-reference-server.sh" &
COSYVOICE_PID=$!
wait_for_http "${COSYVOICE_BACKEND_URL}/docs" 600

"$ROOT_DIR/scripts/run-local-cosyvoice-adapter.sh" \
  --backend-url "$COSYVOICE_BACKEND_URL" \
  --port "$ADAPTER_PORT" &
ADAPTER_PID=$!
wait_for_http "${ADAPTER_URL}/healthz" 120

env \
  BLINK_LOCAL_TTS_BACKEND=local-http-wav \
  BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL="$ADAPTER_URL" \
  "$ROOT_DIR/scripts/run-local-voice.sh" \
  "$@" &
VOICE_PID=$!

set +e
wait "$VOICE_PID"
STATUS=$?
set -e

exit "$STATUS"

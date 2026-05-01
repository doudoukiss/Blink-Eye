#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/local_launcher_helpers.sh"

if [[ "${BLINK_ALLOW_NATIVE_VOICE_MELO:-0}" != "1" ]]; then
  echo "error: Native voice MeloTTS is disabled for the supported local voice path." >&2
  echo "Use ./scripts/run-local-voice-en.sh for English native voice." >&2
  echo "Use ./scripts/run-local-browser-melo.sh for the Chinese browser MeloTTS path." >&2
  echo "Set BLINK_ALLOW_NATIVE_VOICE_MELO=1 only for legacy/manual debugging." >&2
  exit 2
fi

MELO_HOST="${BLINK_LOCAL_MELO_HOST:-127.0.0.1}"
MELO_PORT="${BLINK_LOCAL_MELO_PORT:-8001}"
MELO_BASE_URL="http://${MELO_HOST}:${MELO_PORT}"
STARTED_MELO=0
CLEANUP_DONE=0

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if [[ -n "${VOICE_PID:-}" ]]; then
    terminate_pid_tree "${VOICE_PID}" "Blink voice runtime"
  fi
  if [[ "$STARTED_MELO" -eq 1 && -n "${MELO_PID:-}" ]]; then
    terminate_pid_tree "${MELO_PID}" "MeloTTS HTTP-WAV sidecar"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if curl --silent --fail "${MELO_BASE_URL}/healthz" >/dev/null 2>&1; then
  echo "Using existing MeloTTS HTTP-WAV server at ${MELO_BASE_URL}" >&2
else
  "$ROOT_DIR/scripts/run-melotts-reference-server.sh" &
  MELO_PID=$!
  STARTED_MELO=1
  wait_for_http "${MELO_BASE_URL}/healthz" 240
fi

env \
  BLINK_LOCAL_TTS_BACKEND=local-http-wav \
  BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL="$MELO_BASE_URL" \
  "$ROOT_DIR/scripts/run-local-voice.sh" \
  "$@" &
VOICE_PID=$!

set +e
wait "$VOICE_PID"
STATUS=$?
set -e

exit "$STATUS"

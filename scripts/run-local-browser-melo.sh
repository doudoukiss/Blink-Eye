#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/local_launcher_helpers.sh"

MELO_HOST="${BLINK_LOCAL_MELO_HOST:-127.0.0.1}"
MELO_PORT="${BLINK_LOCAL_MELO_PORT:-8001}"
MELO_BASE_URL="http://${MELO_HOST}:${MELO_PORT}"
DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"
DEFAULT_CONTINUOUS_PERCEPTION="${BLINK_LOCAL_CONTINUOUS_PERCEPTION:-0}"
DEFAULT_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"
CLIENT_HOST="${BLINK_LOCAL_HOST:-127.0.0.1}"
CLIENT_PORT="${BLINK_LOCAL_PORT:-7860}"
CLIENT_URL="http://${CLIENT_HOST}:${CLIENT_PORT}/client/"
MELO_HEALTH_INTERVAL_SECS="${BLINK_LOCAL_MELO_HEALTH_INTERVAL_SECS:-5}"
LOG_DIR="$ROOT_DIR/artifacts/runtime_logs"
LOG_STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/blink-browser-melo-${LOG_STAMP}.log"
LATEST_LOG="$LOG_DIR/latest-browser-melo.log"
TTS_RUNTIME_LABEL="local-http-wav/MeloTTS"
STARTED_MELO=0
CLEANUP_DONE=0

launcher_flag_enabled() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

launcher_state_label() {
  if launcher_flag_enabled "${1:-}"; then
    printf 'on'
  else
    printf 'off'
  fi
}

launcher_protected_playback_label() {
  if launcher_flag_enabled "${1:-}"; then
    printf 'off'
  else
    printf 'on'
  fi
}

launcher_barge_in_policy_label() {
  if launcher_flag_enabled "${1:-}"; then
    printf 'armed'
  else
    printf 'protected'
  fi
}

for arg in "$@"; do
  case "$arg" in
    --vision)
      DEFAULT_BROWSER_VISION=1
      ;;
    --no-vision)
      DEFAULT_BROWSER_VISION=0
      ;;
    --continuous-perception)
      DEFAULT_CONTINUOUS_PERCEPTION=1
      ;;
    --no-continuous-perception)
      DEFAULT_CONTINUOUS_PERCEPTION=0
      ;;
    --allow-barge-in)
      DEFAULT_ALLOW_BARGE_IN=1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Writing Blink browser Melo run log to ${LOG_FILE}"
echo "Blink browser launch: profile=browser-zh-melo language=zh tts=${TTS_RUNTIME_LABEL} webrtc=on camera_vision=$(launcher_state_label "$DEFAULT_BROWSER_VISION") continuous_perception=$(launcher_state_label "$DEFAULT_CONTINUOUS_PERCEPTION") protected_playback=$(launcher_protected_playback_label "$DEFAULT_ALLOW_BARGE_IN") barge_in_policy=$(launcher_barge_in_policy_label "$DEFAULT_ALLOW_BARGE_IN") log=${LOG_FILE} client=${CLIENT_URL}"
echo "Starting primary browser-zh-melo WebRTC actor path with Chinese MeloTTS and browser vision available by default."

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  if [[ -n "${MELO_SUPERVISOR_PID:-}" ]]; then
    terminate_pid_tree "${MELO_SUPERVISOR_PID}" "MeloTTS health supervisor" 2
  fi
  if [[ -n "${BROWSER_PID:-}" ]]; then
    terminate_pid_tree "${BROWSER_PID}" "Blink browser runtime"
  fi
  if [[ "$STARTED_MELO" -eq 1 && -n "${MELO_PID:-}" ]]; then
    terminate_pid_tree "${MELO_PID}" "MeloTTS HTTP-WAV sidecar"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_owned_melo_sidecar() {
  if [[ -n "${MELO_PID:-}" ]] && kill -0 "$MELO_PID" >/dev/null 2>&1; then
    return 0
  fi

  echo "Starting owned MeloTTS HTTP-WAV sidecar at ${MELO_BASE_URL}" >&2
  "$ROOT_DIR/scripts/run-melotts-reference-server.sh" &
  MELO_PID=$!
  STARTED_MELO=1
  if ! wait_for_http "${MELO_BASE_URL}/healthz" 240; then
    terminate_pid_tree "${MELO_PID}" "unready MeloTTS HTTP-WAV sidecar" 2
    MELO_PID=""
    return 1
  fi
  return 0
}

melo_sidecar_is_healthy() {
  curl --silent --fail "${MELO_BASE_URL}/healthz" >/dev/null 2>&1
}

monitor_melo_sidecar() {
  while true; do
    sleep "$MELO_HEALTH_INTERVAL_SECS"
    if melo_sidecar_is_healthy; then
      continue
    fi

    echo "MeloTTS HTTP-WAV sidecar lost health; starting an owned replacement." >&2
    if [[ "$STARTED_MELO" -eq 1 && -n "${MELO_PID:-}" ]]; then
      terminate_pid_tree "${MELO_PID}" "unhealthy MeloTTS HTTP-WAV sidecar" 2
      MELO_PID=""
    fi
    if ! start_owned_melo_sidecar; then
      echo "MeloTTS HTTP-WAV sidecar restart failed; will retry." >&2
      MELO_PID=""
    fi
  done
}

if melo_sidecar_is_healthy; then
  echo "Using existing MeloTTS HTTP-WAV server at ${MELO_BASE_URL}" >&2
else
  start_owned_melo_sidecar
fi

monitor_melo_sidecar &
MELO_SUPERVISOR_PID=$!

env \
  BLINK_LOCAL_CONFIG_PROFILE=browser-zh-melo \
  BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION" \
  BLINK_LOCAL_CONTINUOUS_PERCEPTION="$DEFAULT_CONTINUOUS_PERCEPTION" \
  BLINK_LOCAL_ALLOW_BARGE_IN="$DEFAULT_ALLOW_BARGE_IN" \
  BLINK_LOCAL_TTS_BACKEND=local-http-wav \
  BLINK_LOCAL_TTS_RUNTIME_LABEL="$TTS_RUNTIME_LABEL" \
  BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL="$MELO_BASE_URL" \
  "$ROOT_DIR/scripts/run-local-browser.sh" \
  "$@" &
BROWSER_PID=$!

set +e
wait "$BROWSER_PID"
STATUS=$?
set -e

exit "$STATUS"

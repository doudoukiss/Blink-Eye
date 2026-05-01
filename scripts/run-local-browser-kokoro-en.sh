#!/usr/bin/env bash

set -euo pipefail

# Stable English browser actor path:
# Browser/WebRTC mic+camera -> MLX Whisper -> LLM -> Kokoro -> browser audio.
# MeloTTS is intentionally not part of this path by default.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="$ROOT_DIR/artifacts/runtime_logs"
LOG_STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/blink-browser-kokoro-en-${LOG_STAMP}.log"
LATEST_LOG="$LOG_DIR/latest-browser-kokoro-en.log"
DEFAULT_BROWSER_VISION="${BLINK_LOCAL_BROWSER_VISION:-1}"
DEFAULT_CONTINUOUS_PERCEPTION="${BLINK_LOCAL_CONTINUOUS_PERCEPTION:-0}"
DEFAULT_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"
CLIENT_HOST="${BLINK_LOCAL_HOST:-127.0.0.1}"
CLIENT_PORT="${BLINK_LOCAL_PORT:-7860}"
CLIENT_URL="http://${CLIENT_HOST}:${CLIENT_PORT}/client/"
TTS_RUNTIME_LABEL="kokoro/English"

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

export BLINK_LOCAL_LANGUAGE=en
export BLINK_LOCAL_CONFIG_PROFILE=browser-en-kokoro
export BLINK_LOCAL_TTS_BACKEND=kokoro
export BLINK_LOCAL_TTS_RUNTIME_LABEL="$TTS_RUNTIME_LABEL"
export BLINK_LOCAL_BROWSER_VISION="$DEFAULT_BROWSER_VISION"
export BLINK_LOCAL_CONTINUOUS_PERCEPTION="$DEFAULT_CONTINUOUS_PERCEPTION"
export BLINK_LOCAL_ALLOW_BARGE_IN="$DEFAULT_ALLOW_BARGE_IN"
export BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Writing Blink English browser Kokoro run log to ${LOG_FILE}"
echo "Blink browser launch: profile=browser-en-kokoro language=en tts=${TTS_RUNTIME_LABEL} webrtc=on camera_vision=$(launcher_state_label "$DEFAULT_BROWSER_VISION") continuous_perception=$(launcher_state_label "$DEFAULT_CONTINUOUS_PERCEPTION") protected_playback=$(launcher_protected_playback_label "$DEFAULT_ALLOW_BARGE_IN") barge_in_policy=$(launcher_barge_in_policy_label "$DEFAULT_ALLOW_BARGE_IN") log=${LOG_FILE} client=${CLIENT_URL}"
echo "Starting primary browser-en-kokoro WebRTC actor path with English Kokoro, no MeloTTS sidecar, and browser vision available by default."

exec "$ROOT_DIR/scripts/run-local-browser.sh" "$@" \
  --language en \
  --tts-backend kokoro

#!/usr/bin/env bash
set -euo pipefail

# English Kokoro voice plus macOS-owned camera helper:
# Mac mic -> MLX Whisper -> LLM -> Kokoro, BlinkCameraHelper.app -> Moondream on demand.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/artifacts/runtime_logs"
LOG_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/blink-native-voice-macos-camera-en-${LOG_STAMP}.log"
HELPER_APP="$ROOT_DIR/native/macos/BlinkCameraHelper/build/BlinkCameraHelper.app"
HELPER_STATE_DIR="$LOG_DIR/blink-camera-helper-${LOG_STAMP}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: macOS camera helper voice path is only supported on macOS." >&2
  exit 1
fi

if [[ ! -x "$HELPER_APP/Contents/MacOS/BlinkCameraHelper" ]]; then
  "$ROOT_DIR/scripts/build-macos-camera-helper.sh"
fi

export BLINK_LOCAL_LANGUAGE=en
export BLINK_LOCAL_CONFIG_PROFILE=native-en-kokoro-macos-camera
export BLINK_LOCAL_TTS_BACKEND=kokoro
export BLINK_LOCAL_VOICE_VISION=0
export BLINK_LOCAL_CAMERA_SOURCE=macos-helper
export BLINK_LOCAL_CAMERA_HELPER_APP="$HELPER_APP"
export BLINK_LOCAL_CAMERA_HELPER_STATE_DIR="$HELPER_STATE_DIR"
export BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1
export BLINK_LOCAL_ALLOW_BARGE_IN="${BLINK_LOCAL_ALLOW_BARGE_IN:-0}"
export PYTHONUNBUFFERED=1

native_barge_in_enabled() {
  local value
  for arg in "$@"; do
    case "$arg" in
      --protected-playback)
        return 1
        ;;
    esac
  done

  value="$(printf '%s' "${BLINK_LOCAL_ALLOW_BARGE_IN:-0}" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|on)
      return 0
      ;;
  esac

  for arg in "$@"; do
    case "$arg" in
      --allow-barge-in)
        return 0
        ;;
    esac
  done

  return 1
}

if native_barge_in_enabled "$@"; then
  BARGE_IN_STATE=on
  PROTECTED_PLAYBACK_STATE=off
else
  BARGE_IN_STATE=off
  PROTECTED_PLAYBACK_STATE=on
fi

mkdir -p "$LOG_DIR" "$HELPER_STATE_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest-native-voice-macos-camera-en.log"

echo "Writing Blink native English voice + macOS camera run log to $LOG_FILE"
echo "Native isolation summary: runtime=native transport=PyAudio profile=native-en-kokoro-macos-camera isolation=backend-plus-helper-camera tts=kokoro protected_playback=${PROTECTED_PLAYBACK_STATE} barge_in=${BARGE_IN_STATE} primary_browser_paths=browser-zh-melo,browser-en-kokoro"
echo "BlinkCameraHelper state directory: $HELPER_STATE_DIR"
echo "If macOS asks for camera permission, allow Blink Camera Helper, not Terminal."
echo "Camera helper status: on-demand single-frame isolation only; this is not continuous video or the browser camera UX."
echo "Protected playback is enabled by default for native speaker safety; pass --allow-barge-in only with headphones or another echo-safe setup."
exec > >(tee -a "$LOG_FILE") 2>&1

exec "$ROOT_DIR/scripts/run-local-voice.sh" "$@" \
  --language en \
  --tts-backend kokoro \
  --camera-source macos-helper \
  --camera-helper-app "$HELPER_APP" \
  --camera-helper-state-dir "$HELPER_STATE_DIR"

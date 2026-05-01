#!/usr/bin/env bash
set -euo pipefail

# Stable native English voice path:
# Mac mic -> MLX Whisper -> LLM -> Kokoro -> Mac speaker.
# Camera/vision and MeloTTS are intentionally not part of this path.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/artifacts/runtime_logs"
LOG_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/blink-native-voice-en-${LOG_STAMP}.log"

export BLINK_LOCAL_LANGUAGE=en
export BLINK_LOCAL_CONFIG_PROFILE=native-en-kokoro
export BLINK_LOCAL_TTS_BACKEND=kokoro
export BLINK_LOCAL_VOICE_VISION=0
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

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest-native-voice-en.log"

echo "Writing Blink native English voice run log to $LOG_FILE"
echo "Native isolation summary: runtime=native transport=PyAudio profile=native-en-kokoro isolation=backend-only tts=kokoro protected_playback=${PROTECTED_PLAYBACK_STATE} barge_in=${BARGE_IN_STATE} primary_browser_paths=browser-zh-melo,browser-en-kokoro"
echo "This lane is for backend isolation: mic -> STT -> LLM -> Kokoro. Browser/WebRTC, camera, and MeloTTS stay off here."
echo "Protected playback is enabled by default for native speaker safety; pass --allow-barge-in only with headphones or another echo-safe setup."
exec > >(tee -a "$LOG_FILE") 2>&1

exec "$ROOT_DIR/scripts/run-local-voice.sh" "$@" \
  --language en \
  --tts-backend kokoro

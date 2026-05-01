#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE="text"
WITH_VISION=0
WITH_PIPER=0
PREFETCH_ASSETS=0
OLLAMA_MODEL_NAME="qwen3.5:4b"

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap-blink-mac.sh [--profile text|voice|browser|full] [--with-vision] [--with-piper] [--prefetch-assets]

Profiles:
  text     Install the lightweight terminal chat workflow only.
  voice    Install native mic/speaker voice extras.
  browser  Install the browser/WebRTC local workflow.
  full     Install native voice and browser/WebRTC extras together.

Options:
  --with-vision   Add the optional local Moondream vision stack for browser/full profiles,
                  or for the separate native macOS camera helper wrapper.
  --with-piper    Install the optional Piper TTS backend too.
  --prefetch-assets  Download the selected local models and assets now instead of on first use.
  -h, --help      Show this help message.

Notes:
  Local interaction now defaults to Simplified Chinese.
  Native voice is the English-only Kokoro path on Apple Silicon.
  MeloTTS via local-http-wav is the primary recommended Chinese-quality browser upgrade.
  XTTS/local-http-wav remain optional upgrades.
  This repo now includes an optional CosyVoice-to-local-http-wav adapter server.
  Set BLINK_LOCAL_LANGUAGE=en if you want the English-first local flow.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --with-vision)
      WITH_VISION=1
      shift
      ;;
    --with-piper)
      WITH_PIPER=1
      shift
      ;;
    --prefetch-assets)
      PREFETCH_ASSETS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$PROFILE" in
  text|voice|browser|full)
    ;;
  *)
    echo "Invalid profile: $PROFILE" >&2
    usage >&2
    exit 1
    ;;
esac

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is required for the local workflow. Install it from https://ollama.com/download." >&2
  exit 1
fi

if [[ "$PROFILE" == "voice" || "$PROFILE" == "full" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required for the native voice workflow." >&2
    echo "Install it from https://brew.sh/." >&2
    exit 1
  fi

  if ! brew --prefix portaudio >/dev/null 2>&1; then
    echo "PortAudio is required before installing the native voice extras." >&2
    echo "Run: brew install portaudio" >&2
    exit 1
  fi
fi

uv python install 3.12

EXTRA_ARGS=()
case "$PROFILE" in
  voice)
    EXTRA_ARGS+=(--extra local --extra mlx-whisper --extra kokoro)
    ;;
  browser)
    EXTRA_ARGS+=(--extra runner --extra webrtc --extra mlx-whisper --extra kokoro)
    ;;
  full)
    EXTRA_ARGS+=(--extra local --extra runner --extra webrtc --extra mlx-whisper --extra kokoro)
    ;;
esac

if [[ "$WITH_VISION" -eq 1 && "$PROFILE" != "text" ]]; then
  EXTRA_ARGS+=(--extra moondream)
fi

if [[ "$WITH_PIPER" -eq 1 ]]; then
  EXTRA_ARGS+=(--extra piper)
fi

uv sync --python 3.12 --group dev "${EXTRA_ARGS[@]}"

if [[ ! -f .env ]]; then
  cp env.local.example .env
  echo "Created .env from env.local.example"
fi

if ! ollama list | grep -q "^${OLLAMA_MODEL_NAME}[[:space:]]"; then
  echo "${OLLAMA_MODEL_NAME} is not present in Ollama yet."
  echo "Run: ollama pull ${OLLAMA_MODEL_NAME}"
fi

if [[ "$PREFETCH_ASSETS" -eq 1 ]]; then
  echo
  echo "Prefetching local assets..."
  PREFETCH_ARGS=(--profile "$PROFILE")
  if [[ "$WITH_VISION" -eq 1 && "$PROFILE" != "text" ]]; then
    PREFETCH_ARGS+=(--with-vision)
  fi
  if [[ "$WITH_PIPER" -eq 1 ]]; then
    PREFETCH_ARGS+=(--tts-backend piper)
  fi
  "$ROOT_DIR/scripts/prefetch-local-assets.sh" "${PREFETCH_ARGS[@]}"
fi

echo
echo "Blink local bootstrap complete."
echo
echo "Installed Blink profile: ${PROFILE}"
if [[ "$PROFILE" == "voice" ]]; then
  echo "Default local language: en"
else
  echo "Default local language: zh"
fi
if [[ "$WITH_VISION" -eq 1 ]]; then
  if [[ "$PROFILE" == "voice" ]]; then
    echo "Optional vision stack: enabled for the macOS camera helper path"
    VISION_SUFFIX=" --camera-source macos-helper"
  else
    echo "Optional vision stack: enabled"
    VISION_SUFFIX=" --with-vision"
  fi
else
  VISION_SUFFIX=""
fi
if [[ "$WITH_PIPER" -eq 1 ]]; then
  echo "Optional Piper backend: enabled"
  PIPER_PREFETCH_SUFFIX=" --tts-backend piper"
else
  PIPER_PREFETCH_SUFFIX=""
fi
if [[ "$PREFETCH_ASSETS" -eq 1 ]]; then
  echo "Asset prefetch: completed"
else
  echo "Asset prefetch: skipped"
fi
echo
echo "Recommended Blink checks:"
echo "  1. Start Ollama: ollama serve"
if [[ "$PROFILE" == "browser" || "$PROFILE" == "full" ]]; then
  echo "  2. Optional higher-quality Mandarin browser TTS upgrades:"
  echo "     Recommended MeloTTS browser path:"
  echo "     ./scripts/bootstrap-melotts-reference.sh"
  echo "     ./scripts/run-local-browser-melo.sh"
  echo "     Use your own XTTS-compatible server and set:"
  echo "     BLINK_LOCAL_TTS_BACKEND=xtts"
  echo "     BLINK_LOCAL_XTTS_BASE_URL=http://127.0.0.1:8000"
  echo "     Or use your own local HTTP WAV TTS server and set:"
  echo "     BLINK_LOCAL_TTS_BACKEND=local-http-wav"
  echo "     BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001"
  echo "     Advanced CosyVoice adapter example:"
  echo "     ./scripts/run-local-cosyvoice-adapter.sh"
  echo "  3. Verify the setup: ./scripts/doctor-blink-mac.sh --profile ${PROFILE}${VISION_SUFFIX}"
  START_STEP=4
else
  echo "  2. Verify the setup: ./scripts/doctor-blink-mac.sh --profile ${PROFILE}${VISION_SUFFIX}"
  START_STEP=3
fi
if [[ "$PREFETCH_ASSETS" -eq 0 ]]; then
  echo "  ${START_STEP}. Optional: pre-download models now: ./scripts/prefetch-blink-assets.sh --profile ${PROFILE}${VISION_SUFFIX}${PIPER_PREFETCH_SUFFIX}"
  STEP_OFFSET=1
else
  STEP_OFFSET=0
fi
case "$PROFILE" in
  text)
    echo "  $((START_STEP + STEP_OFFSET)). Start the terminal chat: ./scripts/run-blink-chat.sh"
    echo "     English fallback: BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh"
    ;;
  voice)
    echo "  $((START_STEP + STEP_OFFSET)). Start English native voice: ./scripts/run-local-voice-en.sh"
    echo "     Optional English camera helper path: ./scripts/run-local-voice-macos-camera-en.sh"
    if [[ "$WITH_PIPER" -eq 1 ]]; then
      echo "     Piper example: ./scripts/run-blink-voice.sh --tts-backend piper"
    fi
    ;;
  browser)
    echo "  $((START_STEP + STEP_OFFSET)). Start browser voice: ./scripts/run-blink-browser.sh"
    echo "     English fallback: BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh"
    if [[ "$WITH_PIPER" -eq 1 ]]; then
      echo "     Piper example: ./scripts/run-blink-browser.sh --tts-backend piper"
    fi
    ;;
  full)
    echo "  $((START_STEP + STEP_OFFSET)). Start English native voice: ./scripts/run-local-voice-en.sh"
    echo "     Optional English camera helper path: ./scripts/run-local-voice-macos-camera-en.sh"
    echo "  $((START_STEP + STEP_OFFSET + 1)). Start browser voice: ./scripts/run-blink-browser.sh"
    echo "     English fallback: BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh"
    if [[ "$WITH_PIPER" -eq 1 ]]; then
      echo "     Piper example: ./scripts/run-blink-voice.sh --tts-backend piper"
      echo "     Piper example: ./scripts/run-blink-browser.sh --tts-backend piper"
    fi
    ;;
esac
echo

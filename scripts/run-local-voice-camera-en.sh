#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. Native voice camera is disabled because it made
# second-turn voice sessions unstable. Use browser vision for camera work.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Native voice camera is disabled; starting English Kokoro voice without camera." >&2
exec "$ROOT_DIR/scripts/run-local-voice-en.sh" "$@"

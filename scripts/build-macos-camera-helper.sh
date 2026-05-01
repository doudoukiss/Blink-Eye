#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER_DIR="$ROOT_DIR/native/macos/BlinkCameraHelper"
BUILD_DIR="$HELPER_DIR/build"
APP_DIR="$BUILD_DIR/BlinkCameraHelper.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
SOURCE_FILE="$HELPER_DIR/Sources/BlinkCameraHelper/main.swift"
PLIST_FILE="$HELPER_DIR/Info.plist"
EXECUTABLE="$MACOS_DIR/BlinkCameraHelper"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: BlinkCameraHelper.app can only be built on macOS." >&2
  exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "error: xcrun is required. Install Xcode Command Line Tools." >&2
  exit 1
fi

mkdir -p "$MACOS_DIR"
cp "$PLIST_FILE" "$CONTENTS_DIR/Info.plist"

xcrun swiftc \
  "$SOURCE_FILE" \
  -o "$EXECUTABLE" \
  -framework AppKit \
  -framework AVFoundation \
  -framework CoreMedia \
  -framework CoreVideo

chmod +x "$EXECUTABLE"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --sign - "$APP_DIR" >/dev/null
fi

echo "Built $APP_DIR"

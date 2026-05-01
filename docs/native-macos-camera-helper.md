# Native macOS Camera Helper

Blink has one supported non-browser camera path:

```text
Mac mic -> MLX Whisper -> LLM -> Kokoro -> Mac speaker
BlinkCameraHelper.app -> latest RGB snapshot -> Moondream on demand
```

This path is English-only and exists because macOS camera permission is owned by
the app that opens the camera. The helper app uses AVFoundation directly, so
Terminal, Codex, or OpenCV do not need camera permission for this workflow.

## Run

```bash
./scripts/bootstrap-blink-mac.sh --profile voice --with-vision
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

The wrapper forces:

- `BLINK_LOCAL_LANGUAGE=en`
- `BLINK_LOCAL_TTS_BACKEND=kokoro`
- `BLINK_LOCAL_CAMERA_SOURCE=macos-helper`
- `BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1`, so this path uses the built-in
  English speech prompt instead of inheriting a Chinese prompt from `.env`
- `BLINK_LOCAL_ALLOW_BARGE_IN=0`, so inherited shell or `.env` values cannot
  accidentally put the native path into self-interrupting barge-in mode

It does not start browser/WebRTC, MeloTTS, or `local-http-wav`.
Pass `--allow-barge-in` only with headphones or another echo-safe setup.

## Helper Contract

`BlinkCameraHelper.app` writes state into a per-run directory under
`artifacts/runtime_logs/`:

- `status.json`
- `latest.rgb`

`status.json` is public-safe:

```json
{
  "state": "starting|awaiting_permission|running|denied|error|stopped",
  "updated_at": "ISO-8601",
  "frame_seq": 0,
  "frame_path": "latest.rgb",
  "width": 640,
  "height": 480,
  "format": "RGB",
  "pid": 12345,
  "reason_codes": []
}
```

Blink rejects stale, malformed, denied, or missing-frame states with bounded
reason codes. It does not expose raw image bytes or filesystem paths through
public runtime payloads.

## Runtime Behavior

Blink registers `fetch_user_image` only when `--camera-source macos-helper` is
active. The helper may keep a low-FPS snapshot cache fresh, but Moondream runs
only when the image tool needs a frame. The voice runtime also lazy-loads the
Moondream service on first image-tool use, so ordinary non-visual English voice
startup is not blocked by the vision model. Do not feed continuous camera frames
into the voice loop.

The helper does not force `activeVideoMinFrameDuration` on the camera device.
Some macOS camera backends throw an Objective-C exception when that property is
set, which crashes the helper before it can publish frames. Instead, the helper
lets AVFoundation capture normally and throttles RGB snapshot writes in its
sample-buffer delegate.

The retired Terminal/OpenCV native camera route remains disabled. If helper
camera permission fails, fix the `BlinkCameraHelper` macOS permission state
rather than re-enabling OpenCV capture.

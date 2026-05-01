# Native Voice Isolation Guardrails

Native English Kokoro is a backend-isolation lane. Use it to test the native
mic -> STT -> LLM -> Kokoro -> speaker path without browser/WebRTC, MeloTTS, or
the browser camera stack.

The equal primary browser/WebRTC paths remain:

- `browser-zh-melo`: Chinese browser/WebRTC, MeloTTS through `local-http-wav`,
  camera enabled by default, protected playback on.
- `browser-en-kokoro`: English browser/WebRTC, Kokoro, camera enabled by default,
  protected playback on.

Use the browser paths for daily interactive UX work, interruption behavior,
browser echo cancellation, and product camera evaluation.

## Camera-Free Native Isolation

Run:

```bash
./scripts/run-local-voice-en.sh
```

Expected default status:

```text
runtime=native transport=PyAudio profile=native-en-kokoro isolation=backend-only tts=kokoro protected_playback=on barge_in=off
```

This path keeps camera, Moondream, browser/WebRTC, MeloTTS, and
`local-http-wav` off. It is useful when you need to isolate local microphone
delivery, MLX Whisper, LLM generation, or Kokoro playback.

## Native Helper-Camera Isolation

Run:

```bash
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

Expected default status:

```text
runtime=native transport=PyAudio profile=native-en-kokoro-macos-camera isolation=backend-plus-helper-camera tts=kokoro protected_playback=on barge_in=off
```

`BlinkCameraHelper.app` owns macOS camera permission. Blink reads one recent
cached frame only when `fetch_user_image` is called. This is on-demand
single-frame isolation, not continuous video, not browser camera UX, and not a
replacement for the `browser-zh-melo` camera path.

## Barge-In Policy

Protected playback is the default because native PyAudio does not provide
browser/WebRTC echo cancellation. Open laptop speakers can leak Blink's own
voice into the microphone and cause self-interruption.

Only opt into native barge-in with headphones or another echo-safe setup:

```bash
./scripts/run-local-voice-en.sh --allow-barge-in
BLINK_LOCAL_ALLOW_BARGE_IN=1 ./scripts/run-local-voice-en.sh
```

`--protected-playback` overrides `BLINK_LOCAL_ALLOW_BARGE_IN=1` and returns the
native run to `barge_in=off`.

## Retired Native Camera Path

The old Terminal/OpenCV native camera path is intentionally retired. It mixed
Terminal-owned camera capture, Moondream, and the live voice loop, which made
second-turn voice sessions slow or unresponsive. Do not revive that path as a
camera or interruption fix.

For Chinese camera and speech quality work, use:

```bash
./scripts/run-local-browser-melo.sh
```

For English Kokoro with browser/WebRTC media handling, use:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

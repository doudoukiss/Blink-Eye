# Blink Debugging Notes

This file collects recurring local debugging lessons that should not be
rediscovered from chat history. For the full Chinese-local adaptation record,
see [chinese-conversation-adaptation.md](./chinese-conversation-adaptation.md).

## Browser Melo Second-Turn Audio Failure

Symptom:

- first browser answer works
- second answer is silent, distorted, faded, or stops early
- camera and microphone may degrade together
- the system may feel slow only after camera permission is granted

Check first:

```bash
curl http://127.0.0.1:7860/api/runtime/stack
curl http://127.0.0.1:7860/api/runtime/voice-metrics
curl http://127.0.0.1:8001/healthz
tail -n 200 artifacts/runtime_logs/latest-browser-melo.log
./scripts/doctor-blink-mac.sh --profile browser
```

Known failure chain:

1. Camera pressure can make the browser path slow when WebRTC video frames are
   converted before the requested low capture framerate is honored.
2. Too many tiny `local-http-wav` chunks make MeloTTS produce many separate WAV
   files, each with its own fade/restart shape.
3. One overlarge turn chunk is also bad: Melo can sound unstable on long
   paragraph-sized requests, especially on later replies.
4. The generic TTS audio-context timeout was too short for synchronous local
   HTTP WAV generation, so slow Melo synthesis could produce an early stop.
5. Without bounded TTS request metrics, this can be misdiagnosed as STT, model,
   or camera failure.

Current guardrails:

- SmallWebRTC must rate-limit camera frames before RGB conversion.
- `./scripts/run-local-browser-melo.sh` keeps continuous perception disabled by
  default while preserving explicit `fetch_user_image` camera questions.
- `local-http-wav` should coalesce assistant text into moderate bounded sentence
  groups, not tiny clauses and not one overlong paragraph.
- `LocalHttpWavTTSService` uses a longer TTS audio-context timeout than the
  generic streaming default.
- The Melo sidecar normalizes returned WAV peak level and logs only bounded
  request metrics: chars, bytes, audio duration, request duration, sample rate,
  and channel count.
- Browser media recovery is explicit reload/reconnect. Do not re-enable
  automatic WebRTC renegotiation or mutate browser media tracks without a
  separate proof lane.

Do not fix this by:

- changing the LLM model first
- changing MLX Whisper first
- patching `getUserMedia`
- disabling camera permission by default
- adding automatic browser renegotiation
- moving MeloTTS into the core package
- logging raw transcript text, raw assistant text, prompts, SDP, ICE candidates,
  audio bytes, or browser exception payloads

Manual acceptance:

1. Start `./scripts/run-local-browser-melo.sh`.
2. Open `http://127.0.0.1:7860/client/`.
3. Grant microphone and camera permission.
4. Ask two short Chinese questions in sequence.
5. Expected result: two STT completions, two audible answers, stable microphone
   and camera tracks, and Melo log lines with bounded request metrics.

If the browser preview remains visible but Blink stops receiving backend
audio/video frames, reload `/client/` and reconnect before changing STT, TTS, or
model settings.

## Native English Voice Path

When the goal is to isolate backend STT/LLM/TTS behavior from browser/WebRTC,
use the native English voice path. The detailed guardrail runbook is
[`docs/debugging/native_voice_isolation.md`](debugging/native_voice_isolation.md).

```bash
./scripts/bootstrap-blink-mac.sh --profile voice
# or, for an already bootstrapped checkout:
uv sync --python 3.12 --group dev \
  --extra local \
  --extra kokoro \
  --extra mlx-whisper

ollama serve
./scripts/run-local-voice-en.sh
```

This path bypasses browser permissions, WebRTC renegotiation, browser autoplay,
browser media-track lifecycle, camera capture, Moondream, and MeloTTS. Audio
uses native mic and speaker I/O, MLX Whisper, Ollama, and Kokoro. The wrapper
writes durable local logs under `artifacts/runtime_logs/` and updates
`artifacts/runtime_logs/latest-native-voice-en.log`.

The retired OpenCV native voice camera experiment is intentionally disabled. It
made second-turn voice sessions slow or unresponsive by mixing Terminal-owned
camera capture, Moondream, and the live voice loop. Do not revive that path.
The camera-free native voice wrapper should remain the first backend isolation
test.

If you need English-only native voice with camera grounding outside the browser,
use the separate macOS helper path:

```bash
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

## Browser-First Voice Checkpoint

The 2026-04-26 debugging checkpoint settled the daily-use recommendation:
prefer browser/WebRTC for interactive voice UX. Treat `browser-zh-melo` and
`browser-en-kokoro` as equal primary browser paths. Native PyAudio is useful for
isolating backend STT/LLM/TTS behavior, but it has no browser-grade echo
cancellation and can self-interrupt on laptop speakers when barge-in is enabled.

For the primary English-only Kokoro browser path without MeloTTS or continuous
perception load,
use:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

That wrapper keeps the browser media path, forces English Kokoro, keeps browser
vision available, disables continuous perception by default, uses protected
playback, and writes
`artifacts/runtime_logs/latest-browser-kokoro-en.log`. It also ignores inherited
prompt overrides so a stale Chinese `.env` prompt does not make this lane answer
in Chinese.

`BlinkCameraHelper.app` owns macOS camera permission and writes only a low-FPS
local RGB snapshot cache. Blink reads the latest fresh helper frame only when
`fetch_user_image` is called, then runs Moondream on demand. This path must not
start MeloTTS, browser/WebRTC, local-http-wav, or continuous video ingestion.

If `BlinkCameraHelper` crashes before publishing `latest.rgb`, check macOS
DiagnosticReports for an AVFoundation frame-duration exception. The helper must
throttle snapshot writes itself and must not force
`activeVideoMinFrameDuration`/`activeVideoMaxFrameDuration` on arbitrary camera
devices.

The English native wrappers default to protected playback because PyAudio does
not provide browser/WebRTC echo cancellation. If a native run with
`barge_in=on` barely speaks a word, treat it as likely self-interruption from
speaker bleed before changing STT, TTS, camera, or LLM code. Rerun without
`--allow-barge-in`, or use headphones and opt in explicitly with
`--allow-barge-in`.

If a native English run log says `barge_in=off`, the process is in protected
playback and cannot be interrupted during Blink speech. That is the stable
open-speaker mode for the native path. Use the browser path when you need
browser-grade echo cancellation.

The native English wrappers write unbuffered logs under
`artifacts/runtime_logs/`, so the readiness line should appear promptly in
`latest-native-voice-en.log` or `latest-native-voice-macos-camera-en.log`. If
the wrapper prints only its own banner, inspect local model startup rather than
assuming microphone input is dead.

If the startup line reports `barge_in=on` but the user still cannot interrupt
active speech, inspect the native audio output path next. A real interruption
must both clear queued frames and abort the current PyAudio playback write; if
only the queue is cleared, the already-submitted device write can continue to
drain and feel like interruption is disabled.

Use this path to separate backend STT/LLM/TTS behavior from browser transport
behavior. If native English voice works but browser voice with camera fails, the
remaining bug is in browser/WebRTC/media permissions or browser vision, not MLX
Whisper, Ollama, or Kokoro.

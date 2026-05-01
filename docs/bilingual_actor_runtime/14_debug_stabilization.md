# Browser Runtime Debug Stabilization

This debug lane isolates browser media first, isolates Moondream second, then
verifies privacy-safe actor state and
bilingual parity. It builds on the existing browser actor runtime and does not
replace `/api/runtime/actor-state`, `/api/runtime/actor-events`, or the
compatibility performance endpoints.

## Provider-Free Diagnostic

Run the deterministic diagnostic without opening a browser or calling the LLM,
TTS, Moondream, audio, or camera stacks:

```bash
uv run --python 3.12 python scripts/evals/debug-bilingual-browser-runtime.py
```

The report checks both primary paths:

- `browser-zh-melo`: Chinese, `local-http-wav/MeloTTS`, browser vision on.
- `browser-en-kokoro`: English, `kokoro/English`, browser vision on.

It verifies `/client/`, `/api/runtime/client-config.js`,
`/api/runtime/actor-state`, `/api/runtime/actor-events`,
`/api/runtime/performance-state`, `/api/runtime/performance-events`, and
`/api/runtime/client-media` using in-process test clients. The diagnostic is
provider-free and stores only IDs, labels, counts, booleans, and reason codes.

## Live Media Isolation

When the browser is stuck at connecting or camera does not work, use this order:

1. Stop any existing browser path on port `7860`.
2. Start one primary path with its normal defaults.
3. Open `http://127.0.0.1:7860/client/` and connect.
4. Inspect `/api/runtime/client-media`, `/api/runtime/actor-state`, and
   `/api/runtime/performance-state`.
5. Repeat with `BLINK_LOCAL_BROWSER_VISION=0` to separate WebRTC media issues
   from Moondream issues.

If `client-media` shows microphone `receiving` but camera
`permission_denied`, treat it as browser or macOS camera permission, not as a
profile regression. Resetting macOS camera permission is an operator action,
not an automatic runtime repair:

```bash
/usr/bin/tccutil reset Camera com.openai.codex
/usr/bin/tccutil reset Microphone com.openai.codex
```

After resetting, reconnect and grant camera permission in the browser prompt.

If `client-media` shows `mode=audio_only` and `camera_state=stalled`, treat it
as a stalled browser camera track, not as a permission denial and not as a
Moondream failure. The runtime should report camera scene `stalled`/degraded,
keep Moondream available for later fresh frames, and avoid `can_see_now`.

## 2026-04-28 Camera/Moondream Incident

The live English/Kokoro path reached WebRTC but `fetch_user_image` reported
camera unavailable even after the browser began delivering frames. The root
cause was split:

- macOS/browser permission initially blocked camera frames;
- after permission was granted, an early audio-only client hint could leave
  `active_client.camera_enabled=false` even though the camera frame buffer had
  fresh frames;
- camera-scene health could still report `disconnected` from stale health state
  while the browser media payload and frame buffer showed recent frames.

The runtime now treats a recent cached browser frame as stronger evidence than
a stale audio-only/disconnected hint. That only upgrades availability to
`recent_frame_available`/`available`; it still does not emit `can_see_now`
unless a fresh frame was actually used for the current answer. Permission
denied and device-not-found states remain degraded operator/browser issues.

The debug sequence that solved the incident was:

1. Stop any stale process on port `7860`.
2. Restart `./scripts/run-local-browser-kokoro-en.sh`.
3. Reset Camera/Microphone permission for the Codex app when the browser prompt
   did not appear.
4. Reconnect in `http://127.0.0.1:7860/client/` and grant both permissions.
5. Confirm `/api/runtime/client-media` reports camera `ready` and microphone
   `receiving`.
6. Ask a camera question; `fetch_user_image` should lazy-load Moondream and
   return `vision_answered`, `vision_stale`, or `vision_unavailable` without
   blocking WebRTC.

## 2026-04-28 Speech Boundary Incident

The English/Kokoro path could appear to stop before completing a multi-sentence
answer such as:

```text
English literacy involves reading, writing, and speaking English effectively.
It includes understanding vocabulary, grammar, and sentence structure.
Are you looking to improve your literacy skills or needing help with specific tasks?
```

Actor events showed the V3 speech director emitted bounded chunks, but the
underlying TTS service was still reusing a single turn-level TTS context for
multiple browser speech chunks. That collapsed per-chunk `TTSStoppedFrame`
boundaries, which made the ActorControl lookahead scheduler drain held chunks
unreliably and made audio/subtitle evidence hard to align.

Browser-created TTS services now use one TTS context per V3 speech chunk. The
base TTS service closes those synchronous HTTP-style chunk contexts promptly,
so each chunk gets its own start/audio/stop boundary. Native PyAudio paths keep
their existing defaults.

A second follow-up fixed a scheduler ordering bug: `speech.lookahead_held`
chunks must not drain while earlier emitted speech chunks are still outstanding
downstream. Each TTS queue boundary now releases at most one held chunk, and
only after the previous outstanding chunks have stopped. This prevents a later
subtitle/TTS request from overtaking an earlier queued chunk during long
Kokoro answers.

## 2026-04-28 STT Fragment Surfacing Incident

The browser UI could show only a short final STT fragment even though the LLM
received enough text to answer the full user turn. The realtime STT path may
emit final-only fragments without interim transcripts. The active-listener
runtime now accumulates current-turn final transcript character counts and
merges bounded public hint labels across final fragments. It does not store raw
transcript text, and final-only STT no longer claims a partial transcript was
available.

## Realtime Persistence Fail-Open

Hot-path brain event, memory extraction, turn persistence, actor trace,
episode, discourse, and control-frame writers are diagnostic or memory
surfaces. SQLite lock/misuse errors in these sinks must be logged and
suppressed so they cannot interrupt WebRTC, STT, TTS, or GUI polling.

## 2026-04-28 V3 Runtime Stabilization Core

V3 diagnostics are now treated as derived projections, not as live product
controllers. The browser path keeps one compact `BrowserRuntimeSessionV3`
state core for the pieces that caused live regressions:

- profile/language/TTS labels plus vision, continuous-perception, and
  protected-playback invariants;
- WebRTC client/media and camera frame evidence;
- current STT turn transcript counters, reset by VAD turn boundaries;
- speech generation token, outstanding subtitle/TTS lookahead, held chunks,
  and stale-output drops;
- camera honesty state for `available_not_used`, `recent_frame_available`,
  `can_see_now`, and `unavailable`.

`local_browser.py` remains the FastAPI/WebRTC adapter, but route payloads and
frame observers read/write the session core instead of scattering equivalent
state across route handlers. `ActorControlFrameV3`, `PerformanceEpisodeV3`,
`PerformancePlanV3`, discourse episodes, preferences, and release-bench
evidence must consume public-safe transitions from this core and fail open if
their sinks are unavailable.

The speech drain rule changed with this stabilization pass: a TTS queue
boundary releases exactly one available held chunk. It no longer waits for the
whole lookahead queue to empty before continuing a long answer. The lookahead
cap still prevents unbounded queued speech, and accepted interruption marks the
old generation stale before any later chunk can play.

## Moondream Startup Rule

Browser vision remains enabled by default for both primary paths, but Moondream
must be lazy. Server startup and WebRTC connection must not load the model.
The local vision service is created only on the explicit `fetch_user_image`
path, except when continuous perception is explicitly enabled. Continuous
perception remains off by default.

## Failure Semantics

Camera availability is not the same as vision use. Actor state must report
`permission_denied`, `device_not_found`, `stale`, `unavailable`, or
`vision_unavailable` instead of implying Blink saw the scene. `can_see_now`
appears only after a fresh frame was actually used.

Actor trace, performance episode, discourse episode, and control-frame sinks
are diagnostic surfaces. Sink failures are suppressed at the event bus boundary
so they cannot block realtime browser interaction.
